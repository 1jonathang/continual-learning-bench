"""Public-tool planner for Cohort Studies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import re
from typing import Any

from .cohort_runtime import CohortReadout, CohortStudyContext


@dataclass
class CohortToolPlan:
    tool: str
    tool_call: dict[str, Any]
    reason: str
    signature: str

    def to_artifact(self) -> dict[str, Any]:
        return asdict(self)


def parse_cohort_context(
    prompt: str,
    metadata: dict[str, Any],
) -> CohortStudyContext:
    step_raw = metadata.get("step", 0)
    step = int(step_raw) if isinstance(step_raw, int | float) else 0
    study_name = str(metadata.get("study_name") or _study_name_from_prompt(prompt))
    instance_idx = int(metadata.get("instance_idx", metadata.get("instance_index", 0)) or 0)
    stage_raw = metadata.get("stage_index")
    stage_index = int(stage_raw) if isinstance(stage_raw, int | float) else None
    return CohortStudyContext(
        instance_idx=instance_idx,
        study_name=study_name,
        variant_id=str(metadata.get("variant_id") or ""),
        stage_index=stage_index,
        region_slice=str(metadata.get("region_slice") or ""),
        step=step,
        budget=int(metadata.get("budget", 20) or 20),
        remaining=int(metadata["remaining"]) if isinstance(metadata.get("remaining"), int | float) else None,
        prompt=prompt,
    )


def choose_cohort_tool_action(
    context: CohortStudyContext,
    readout: CohortReadout,
    *,
    max_tool_steps: int = 20,
) -> CohortToolPlan:
    """Choose one legal Cohort Studies tool call using public memory only."""

    attempted = set(readout.attempted_tool_signatures)
    if not readout.metadata_seen:
        return _plan("get_database_metadata", {}, "Need public study metadata and column map.")
    if not readout.summary_seen:
        return _plan("get_data_summary", {}, "Need public univariate summary for visible columns.")

    if context.remaining is not None and context.remaining <= 1:
        return _plan("submit_cohort_report", {}, "Reserve final turn for report submission.")
    if context.step >= max_tool_steps - 1:
        return _plan("submit_cohort_report", {}, "Reached configured exploration limit.")

    for expr in _candidate_group_expressions(readout.visible_columns):
        signature = f"estimate_survival_by_group:{expr}"
        if signature not in attempted:
            return _plan(
                "estimate_survival_by_group",
                {"group_expression": expr},
                "Probe public group survival and cohort decompositions.",
            )

    for expr in _candidate_predict_expressions(readout.visible_columns):
        signature = f"predict_cohort_survival:{expr}"
        if signature not in attempted:
            return _plan(
                "predict_cohort_survival",
                {"group_expression": expr},
                "Use biased in-study fit only as a diagnostic coherence check.",
            )

    return _plan("submit_cohort_report", {}, "No remaining legal public probes.")


def _plan(tool: str, payload: dict[str, Any], reason: str) -> CohortToolPlan:
    tool_call = {"tool": tool, **payload}
    if tool in {"estimate_survival_by_group", "predict_cohort_survival"}:
        signature = f"{tool}:{payload.get('group_expression', '')}"
    elif tool == "query_sql":
        signature = f"{tool}:{payload.get('sql', '')}"
    else:
        signature = tool
    return CohortToolPlan(tool=tool, tool_call=tool_call, reason=reason, signature=signature)


def _study_name_from_prompt(prompt: str) -> str:
    match = re.search(r"## Study \d+/\d+:\s*(.+)$", prompt, re.MULTILINE)
    if match:
        return match.group(1).strip()
    return "unknown_study"


def _candidate_group_expressions(columns: list[str]) -> list[str]:
    cols = set(columns)
    out: list[str] = ["CASE WHEN 1=1 THEN 'all' END"]

    def has(col: str) -> bool:
        return col in cols

    if has("age"):
        out.append(
            "CASE WHEN age >= 60 THEN 'age_high' "
            "WHEN age >= 40 THEN 'age_mid' ELSE 'age_low' END"
        )
    if has("sex"):
        out.append("CASE WHEN sex IS NULL THEN 'sex_unknown' ELSE sex END")
    if has("region"):
        out.append("CASE WHEN region IS NULL THEN 'region_unknown' ELSE region END")
    if has("prb1_ratio"):
        out.append(
            "CASE WHEN prb1_ratio >= 28 THEN 'prb_high' "
            "WHEN prb1_ratio < 25 THEN 'prb_low' ELSE 'prb_mid' END"
        )
    if has("systolic_bp"):
        out.append(
            "CASE WHEN systolic_bp >= 140 THEN 'bp_high' "
            "WHEN systolic_bp < 120 THEN 'bp_low' ELSE 'bp_mid' END"
        )
    exposure_terms = []
    if has("ambient_voc_ppb"):
        exposure_terms.append("ambient_voc_ppb >= 15")
    if has("voc_mg_l"):
        exposure_terms.append("voc_mg_l >= 5")
    if exposure_terms:
        out.append(
            "CASE WHEN "
            + " OR ".join(exposure_terms)
            + " THEN 'exposure_high' ELSE 'exposure_low' END"
        )
    if has("hhbp_serum_level"):
        out.append(
            "CASE WHEN hhbp_serum_level >= 8 THEN 'hhbp_high' "
            "WHEN hhbp_serum_level < 6 THEN 'hhbp_low' ELSE 'hhbp_mid' END"
        )
    if has("mmse_score"):
        out.append(
            "CASE WHEN mmse_score < 23 THEN 'cognition_low' "
            "WHEN mmse_score >= 26 THEN 'cognition_high' ELSE 'cognition_mid' END"
        )
    if has("moca_score"):
        out.append(
            "CASE WHEN moca_score < 20 THEN 'cognition_low' "
            "WHEN moca_score >= 23 THEN 'cognition_high' ELSE 'cognition_mid' END"
        )
    if has("ad8_score"):
        out.append(
            "CASE WHEN ad8_score > 4 THEN 'cognition_low' "
            "WHEN ad8_score <= 2 THEN 'cognition_high' ELSE 'cognition_mid' END"
        )
    if has("genotype_group"):
        out.append(
            "CASE WHEN genotype_group = 'GG-1' THEN 'gg1' "
            "WHEN genotype_group = 'GG-3' THEN 'gg3' ELSE 'other_genotype' END"
        )
    if has("family_history_count"):
        out.append(
            "CASE WHEN family_history_count > 0 THEN 'family_history_yes' "
            "ELSE 'family_history_no' END"
        )
    elif has("family_history"):
        out.append(
            "CASE WHEN family_history IN (1, '1', 'yes', 'Yes', 'Y', 'true', 'True') "
            "THEN 'family_history_yes' ELSE 'family_history_no' END"
        )
    if has("acron_use_avg"):
        out.append(
            "CASE WHEN acron_use_avg >= 0.6 THEN 'acron_high' "
            "WHEN acron_use_avg <= 0.25 THEN 'acron_low' ELSE 'acron_mid' END"
        )
    if has("comorbidity_count"):
        out.append(
            "CASE WHEN comorbidity_count >= 3 THEN 'comorbidity_high' "
            "WHEN comorbidity_count = 0 THEN 'comorbidity_none' ELSE 'comorbidity_some' END"
        )
    if has("years_of_education"):
        out.append(
            "CASE WHEN years_of_education >= 16 THEN 'education_high' "
            "WHEN years_of_education < 12 THEN 'education_low' ELSE 'education_mid' END"
        )

    if has("age") and has("prb1_ratio"):
        out.append(
            "CASE WHEN age >= 55 AND prb1_ratio >= 28 THEN 'age_prb_high' "
            "WHEN age < 55 AND prb1_ratio < 25 THEN 'age_prb_low' ELSE 'age_prb_mixed' END"
        )
    if has("age") and exposure_terms:
        out.append(
            "CASE WHEN age >= 55 AND ("
            + " OR ".join(exposure_terms)
            + ") THEN 'older_exposed' WHEN age < 55 THEN 'younger' ELSE 'other' END"
        )
    if has("age") and has("genotype_group"):
        out.append(
            "CASE WHEN age >= 55 AND genotype_group = 'GG-1' THEN 'older_gg1' "
            "WHEN genotype_group = 'GG-3' THEN 'gg3' ELSE 'other' END"
        )
    if has("prb1_ratio") and has("systolic_bp"):
        out.append(
            "CASE WHEN prb1_ratio >= 28 AND systolic_bp >= 140 THEN 'metabolic_bp_high' "
            "WHEN prb1_ratio < 25 AND systolic_bp < 120 THEN 'metabolic_bp_low' ELSE 'mixed' END"
        )

    deduped: list[str] = []
    seen: set[str] = set()
    for expr in out:
        if expr not in seen:
            deduped.append(expr)
            seen.add(expr)
    return deduped


def _candidate_predict_expressions(columns: list[str]) -> list[str]:
    # Keep predict_cohort_survival sparse because it optimizes biased in-study fit.
    candidates = _candidate_group_expressions(columns)
    return candidates[:3]
