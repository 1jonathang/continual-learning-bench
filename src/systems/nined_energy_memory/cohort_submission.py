"""Build Cohort Studies structured submissions from public memory state."""

from __future__ import annotations

from collections import Counter
import json
import math
import re
from typing import Any

from pydantic import BaseModel

from .cohort_runtime import CohortMemoryRuntime, CohortReadout, cohort_layer


DEFAULT_SURVIVAL = (0.74, 0.56, 0.41)


def build_cohort_submission(
    *,
    schema: type[BaseModel],
    readout: CohortReadout,
    runtime: CohortMemoryRuntime,
) -> BaseModel:
    """Convert public memory artifacts into the task's flat submission model."""

    cohort_ids = _cohort_ids_from_schema(schema)
    data: dict[str, float] = {}
    values: dict[str, tuple[float, float, float]] = {}
    provenance: dict[str, dict[str, Any]] = {}
    fallback_counts: Counter[str] = Counter()
    violations = 0

    for cohort_id in cohort_ids:
        estimate, prov = _estimate_cohort(cohort_id, runtime, readout)
        fixed, had_violation = _monotone_clip(estimate)
        violations += int(had_violation)
        method = str(prov.get("method", "unknown"))
        fallback_counts[method] += 1
        values[cohort_id] = fixed
        provenance[cohort_id] = prov
        data[f"{cohort_id}__s12"] = fixed[0]
        data[f"{cohort_id}__s24"] = fixed[1]
        data[f"{cohort_id}__s36"] = fixed[2]

    runtime.record_submission(
        context=readout.context,
        n_fields=len(data),
        n_cohorts=len(cohort_ids),
        values=values,
        fallback_counts_by_method=dict(fallback_counts),
        provenance=provenance,
        monotonicity_violations_before_fix=violations,
    )
    return schema(**data)


def _cohort_ids_from_schema(schema: type[BaseModel]) -> list[str]:
    ids = set()
    fields = getattr(schema, "model_fields", {})
    for name in fields:
        if name.endswith("__s12") or name.endswith("__s24") or name.endswith("__s36"):
            ids.add(name.rsplit("__s", 1)[0])
    return sorted(ids, key=_cohort_sort_key)


def _cohort_sort_key(cohort_id: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", cohort_id)
    if match:
        return (int(match.group(1)), cohort_id)
    return (10_000, cohort_id)


def _estimate_cohort(
    cohort_id: str,
    runtime: CohortMemoryRuntime,
    global_readout: CohortReadout,
) -> tuple[tuple[float, float, float], dict[str, Any]]:
    cohort_readout = runtime.read_for_cohort(global_readout.context, cohort_id)
    candidates = []
    for row in cohort_readout.top_facts:
        if row.get("fact_type") != "COHORT_SURVIVAL_ESTIMATE":
            continue
        if cohort_id not in (row.get("cohort_ids") or []) and row.get("subject") != cohort_id:
            continue
        decoded = _decode_survival(row.get("object"))
        if decoded is None:
            continue
        support = float(row.get("support_count") or 1.0)
        confidence = float(row.get("confidence_logit") or 0.0)
        energy_weight = float(row.get("read_weight") or 0.0)
        source = str(row.get("source") or "")
        bias_penalty = 0.70 if source == "predict_cohort_survival" else 1.0
        n = float(decoded.get("n") or 1.0)
        weight = max(0.001, (1.0 + support) * math.log1p(n) * math.exp(confidence / 4.0))
        weight *= 1.0 + energy_weight
        weight *= bias_penalty
        candidates.append((weight, decoded, row))

    if candidates:
        total = sum(w for w, _, _ in candidates)
        s12 = sum(w * c["s12"] for w, c, _ in candidates) / total
        s24 = sum(w * c["s24"] for w, c, _ in candidates) / total
        s36 = sum(w * c["s36"] for w, c, _ in candidates) / total
        top_rows = sorted(candidates, key=lambda item: item[0], reverse=True)[:8]
        support_total = sum(float(row.get("support_count") or 0.0) for _, _, row in top_rows)
        return (
            (s12, s24, s36),
            {
                "method": "cross_study_cohort_estimate"
                if len({row.get("study_name") for _, _, row in candidates}) > 1
                else "cohort_estimate",
                "selected_fact_ids": [row.get("fact_id") for _, _, row in top_rows],
                "selected_basin_ids": [
                    row.get("basin_id") for row in cohort_readout.top_basins[:4]
                ],
                "energy_margin": cohort_readout.energy_margin,
                "support_count": support_total,
                "uncertainty": 1.0 / max(1.0, support_total),
                "layer": cohort_layer(cohort_id),
            },
        )

    nearest = _nearest_layer_estimates(cohort_id, runtime, cohort_readout)
    if nearest is not None:
        estimate, used = nearest
        return (
            estimate,
            {
                "method": "nearest_layer_estimate",
                "selected_fact_ids": used,
                "selected_basin_ids": [row.get("basin_id") for row in cohort_readout.top_basins[:4]],
                "energy_margin": cohort_readout.energy_margin,
                "support_count": len(used),
                "uncertainty": 1.0 / max(1, len(used)),
                "layer": cohort_layer(cohort_id),
            },
        )

    overall = _overall_estimate(runtime)
    if overall is not None:
        estimate, used = overall
        return (
            estimate,
            {
                "method": "current_or_cross_study_overall",
                "selected_fact_ids": used,
                "selected_basin_ids": [row.get("basin_id") for row in cohort_readout.top_basins[:4]],
                "energy_margin": cohort_readout.energy_margin,
                "support_count": len(used),
                "uncertainty": 1.0,
                "layer": cohort_layer(cohort_id),
            },
        )

    return (
        DEFAULT_SURVIVAL,
        {
            "method": "conservative_prior_fallback",
            "selected_fact_ids": [],
            "selected_basin_ids": [],
            "energy_margin": cohort_readout.energy_margin,
            "support_count": 0,
            "uncertainty": 1.0,
            "layer": cohort_layer(cohort_id),
        },
    )


def _decode_survival(raw: Any) -> dict[str, float] | None:
    if not isinstance(raw, str):
        return None
    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        return None
    required = {"s12", "s24", "s36"}
    if not required.issubset(data):
        return None
    out = {
        "s12": float(data["s12"]),
        "s24": float(data["s24"]),
        "s36": float(data["s36"]),
    }
    if "n" in data:
        out["n"] = float(data["n"])
    return out


def _nearest_layer_estimates(
    cohort_id: str,
    runtime: CohortMemoryRuntime,
    readout: CohortReadout,
) -> tuple[tuple[float, float, float], list[str]] | None:
    layer = cohort_layer(cohort_id)
    rows = []
    for fact in runtime.facts.values():
        if fact.fact_type != "COHORT_SURVIVAL_ESTIMATE":
            continue
        if layer not in {cohort_layer(cid) for cid in fact.cohort_ids}:
            continue
        decoded = _decode_survival(fact.object)
        if decoded is None:
            continue
        rows.append((fact, decoded))
    if not rows:
        return None
    top_ids = {str(row.get("fact_id")) for row in readout.top_facts[:32]}
    weighted = []
    for fact, decoded in rows:
        weight = 1.0 + fact.support_count
        if fact.fact_id in top_ids:
            weight *= 1.5
        weighted.append((weight, fact, decoded))
    total = sum(w for w, _, _ in weighted)
    s12 = sum(w * d["s12"] for w, _, d in weighted) / total
    s24 = sum(w * d["s24"] for w, _, d in weighted) / total
    s36 = sum(w * d["s36"] for w, _, d in weighted) / total
    used = [fact.fact_id for _, fact, _ in sorted(weighted, key=lambda x: x[0], reverse=True)[:8]]
    return (s12, s24, s36), used


def _overall_estimate(
    runtime: CohortMemoryRuntime,
) -> tuple[tuple[float, float, float], list[str]] | None:
    rows = []
    for fact in runtime.overall_survival_estimates():
        decoded = _decode_survival(fact.object)
        if decoded is not None:
            rows.append((fact, decoded))
    if not rows:
        return None
    weighted = []
    for fact, decoded in rows:
        n = float(decoded.get("n") or 1.0)
        weight = (1.0 + fact.support_count) * math.log1p(n)
        weighted.append((weight, fact, decoded))
    total = sum(w for w, _, _ in weighted)
    s12 = sum(w * d["s12"] for w, _, d in weighted) / total
    s24 = sum(w * d["s24"] for w, _, d in weighted) / total
    s36 = sum(w * d["s36"] for w, _, d in weighted) / total
    used = [fact.fact_id for _, fact, _ in sorted(weighted, key=lambda x: x[0], reverse=True)[:8]]
    return (s12, s24, s36), used


def _monotone_clip(values: tuple[float, float, float]) -> tuple[tuple[float, float, float], bool]:
    s12, s24, s36 = values
    before = (s12, s24, s36)
    s12 = max(s12, s24, s36)
    s24 = min(s12, max(s24, s36))
    s36 = min(s24, s36)
    fixed = (
        min(0.99, max(0.01, s12)),
        min(0.99, max(0.01, s24)),
        min(0.99, max(0.01, s36)),
    )
    return fixed, fixed != before
