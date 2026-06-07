"""Cohort Studies bounded online memory runtime.

This module consumes only public task evidence: prompts, query metadata, tool
outputs, and the runtime's own prior artifacts.  It deliberately does not import
the Cohort Studies task package, scorer, frozen database helpers, or ground
truth files.
"""

from __future__ import annotations

from collections import Counter, deque
from dataclasses import asdict, dataclass, field
import hashlib
import json
import math
import re
from typing import Any

import numpy as np


EPS = 1e-9


@dataclass
class CohortMemoryConfig:
    capacity: int = 96
    fact_capacity: int = 2048
    manifold_dim: int = 96
    read_temperature: float = 0.70
    sliding_window: int = 6
    write_lr: float = 0.30
    profile_count: int = 4


@dataclass
class CohortStudyContext:
    instance_idx: int
    study_name: str
    variant_id: str
    stage_index: int | None
    region_slice: str
    step: int
    budget: int
    remaining: int | None
    prompt: str


@dataclass
class CohortFact:
    fact_id: str
    fact_type: str
    scope: str
    subject: str
    predicate: str
    object: str
    confidence_logit: float
    support_count: int
    contradiction_count: int
    created_at: int
    last_seen: int
    source: str
    study_name: str
    stage_index: int | None
    cohort_ids: list[str] = field(default_factory=list)
    variables: list[str] = field(default_factory=list)
    embedding_key: list[float] = field(default_factory=list)
    embedding_value: list[float] = field(default_factory=list)


@dataclass
class CohortBasin:
    basin_id: str
    basin_kind: str
    scope: str
    subject: str
    fact_ids: list[str] = field(default_factory=list)
    key: list[float] = field(default_factory=list)
    value: list[float] = field(default_factory=list)
    confidence_logit: float = 0.0
    support_count: int = 0
    contradiction_count: int = 0
    created_at: int = 0
    last_seen: int = 0


@dataclass
class CohortReadout:
    context: CohortStudyContext
    mode: str
    read_policy: str
    target_kind: str
    target_id: str
    top_facts: list[dict[str, Any]]
    top_basins: list[dict[str, Any]]
    energy_decomposition: list[dict[str, Any]]
    read_entropy: float
    free_energy: float
    selected_energy: float | None
    runnerup_energy: float | None
    energy_margin: float | None
    landscape_metrics: dict[str, Any]
    visible_columns: list[str]
    metadata_seen: bool
    summary_seen: bool
    attempted_tool_signatures: list[str]
    facts_by_type: dict[str, int]
    facts_by_study: dict[str, int]
    memory_policy: str


_NUMERIC_SUMMARY_RE = re.compile(
    r"^\s*(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s+\(numeric,\s*n=(?P<n>\d+)\):\s*(?P<body>.*)$"
)
_CATEGORICAL_SUMMARY_RE = re.compile(
    r"^\s*(?P<col>[A-Za-z_][A-Za-z0-9_]*)\s+\(categorical,\s*(?P<k>\d+)\s+levels\):\s*(?P<body>.*)$"
)
_GROUP_SURVIVAL_RE = re.compile(
    r"^\s*(?P<group>[^:]+):\s*n=(?P<n>\d+).*?"
    r"S\(12m\)=(?P<s12>[0-9.]+)\s+"
    r"S\(24m\)=(?P<s24>[0-9.]+)\s+"
    r"S\(36m\)=(?P<s36>[0-9.]+)"
)
_COHORT_DECOMP_RE = re.compile(
    r"^\s*(?P<cohort>[A-Za-z0-9_]+)\s+\(n=(?P<n>\d+)\):\s*(?P<parts>.*)$"
)
_PREDICT_RE = re.compile(
    r"^\s*(?P<cohort>[A-Za-z0-9_]+)\s+\(n=(?P<n>\d+)\):\s+"
    r"model=(?P<ms12>[0-9.]+)/(?P<ms24>[0-9.]+)/(?P<ms36>[0-9.]+)\s+"
    r"KM=(?P<ks12>[0-9.]+)/(?P<ks24>[0-9.]+)/(?P<ks36>[0-9.]+)\s+"
    r"KL=(?P<kl>[0-9.]+)"
)


def sigmoid(x: float) -> float:
    if x >= 0:
        z = math.exp(-x)
        return 1.0 / (1.0 + z)
    z = math.exp(x)
    return z / (1.0 + z)


def cosine(a: list[float], b: list[float]) -> float:
    if not a or not b:
        return 0.0
    n = min(len(a), len(b))
    dot = sum(a[i] * b[i] for i in range(n))
    an = math.sqrt(sum(a[i] * a[i] for i in range(n)))
    bn = math.sqrt(sum(b[i] * b[i] for i in range(n)))
    if an <= EPS or bn <= EPS:
        return 0.0
    return dot / (an * bn)


def hash_vector(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[A-Za-z0-9_]+", text.lower())
    if not tokens:
        tokens = [text.lower() or "empty"]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=24).digest()
        for i, byte in enumerate(digest):
            idx = (byte + 31 * i) % dim
            sign = 1.0 if byte & 1 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= EPS:
        return vec
    return [v / norm for v in vec]


def stable_id(*parts: object, digest_size: int = 10) -> str:
    raw = "|".join(str(part) for part in parts)
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=digest_size).hexdigest()


def _clip_survival(value: float) -> float:
    return min(0.99, max(0.01, float(value)))


def _canonical_variables_for_column(column: str) -> list[str]:
    col = column.lower()
    out: list[str] = []
    if "age" in col:
        out.append("age")
    if col == "sex":
        out.append("sex")
    if "region" in col:
        out.append("region")
    if "prb" in col or "metabolic" in col:
        out.append("metabolic")
    if "bp" in col or "blood" in col or "hhbp" in col:
        out.append("cardiovascular")
    if "voc" in col or "ambient" in col or "exposure" in col:
        out.append("exposure")
    if "mmse" in col or "moca" in col or "ad8" in col or "cogn" in col:
        out.append("cognition")
    if "genotype" in col:
        out.append("genotype")
    if "family" in col:
        out.append("family_history")
    if "acron" in col:
        out.append("behavioral_exposure")
    if "education" in col:
        out.append("education")
    if "comorbidity" in col:
        out.append("comorbidity")
    return out or [col]


def cohort_layer(cohort_id: str) -> str:
    match = re.search(r"(\d+)$", cohort_id)
    if not match:
        return "layer_unknown"
    idx = int(match.group(1))
    if idx <= 4:
        return "layer_1"
    if idx <= 20:
        return "layer_2"
    return "layer_3"


class CohortMemoryRuntime:
    """Bounded public-evidence memory for Cohort Studies."""

    def __init__(
        self,
        cfg: CohortMemoryConfig | None = None,
        *,
        mode: str = "cohort_energy",
    ) -> None:
        self.cfg = cfg or CohortMemoryConfig()
        self.mode = mode.strip().lower()
        self.step_index = 0
        self.facts: dict[str, CohortFact] = {}
        self.basins: dict[str, CohortBasin] = {}
        self.current_study_key: str | None = None
        self.current_study_facts: list[str] = []
        self.recent_study_fact_ids: deque[list[str]] = deque(
            maxlen=max(1, self.cfg.sliding_window)
        )
        self.study_history: list[dict[str, Any]] = []
        self.tool_events: list[dict[str, Any]] = []
        self.read_events: list[dict[str, Any]] = []
        self.write_events: list[dict[str, Any]] = []
        self.submission_events: list[dict[str, Any]] = []
        self.last_readout: CohortReadout | None = None
        self.last_update: dict[str, Any] = {}
        self.facts_created = 0
        self.facts_evicted = 0
        self.facts_contradicted = 0

    def reset(self) -> None:
        self.__init__(self.cfg, mode=self.mode)

    @property
    def write_enabled(self) -> bool:
        return self.mode not in {"cohort_static_policy", "cohort_no_write"}

    @property
    def uses_energy(self) -> bool:
        return self.mode.startswith("cohort_energy")

    @property
    def memory_policy(self) -> str:
        if self.mode == "cohort_vanilla_online":
            return "support_recency_confidence"
        if self.mode == "cohort_hard_cache":
            return "hard_scope_key"
        if self.mode == "cohort_sliding_window":
            return "sliding_window"
        if self.mode == "cohort_current_study_only":
            return "current_study_scratch"
        if self.mode in {"cohort_static_policy", "cohort_no_write"}:
            return "no_persistent_write"
        return "energy_ranked"

    def begin_turn(
        self,
        context: CohortStudyContext,
        *,
        target_kind: str = "tool",
        target_id: str = "study",
        variables: list[str] | None = None,
        cohort_ids: list[str] | None = None,
    ) -> CohortReadout:
        self._start_context(context)
        readout = self.read(
            context,
            target_kind=target_kind,
            target_id=target_id,
            variables=variables or [],
            cohort_ids=cohort_ids or [],
        )
        self.last_readout = readout
        return readout

    def read_for_cohort(
        self,
        context: CohortStudyContext,
        cohort_id: str,
    ) -> CohortReadout:
        return self.read(
            context,
            target_kind="cohort_submission",
            target_id=cohort_id,
            variables=[cohort_layer(cohort_id)],
            cohort_ids=[cohort_id],
        )

    def read(
        self,
        context: CohortStudyContext,
        *,
        target_kind: str,
        target_id: str,
        variables: list[str],
        cohort_ids: list[str],
        limit: int = 32,
    ) -> CohortReadout:
        self._start_context(context)
        q_text = " ".join(
            [
                target_kind,
                target_id,
                context.study_name,
                context.region_slice,
                *variables,
                *cohort_ids,
            ]
        )
        q_key = hash_vector(q_text, self.cfg.manifold_dim)

        scored = []
        for fact in self._available_facts():
            components = self._energy_components(
                context,
                fact,
                q_key,
                target_kind=target_kind,
                target_id=target_id,
                variables=variables,
                cohort_ids=cohort_ids,
            )
            energy = sum(components.values())
            scored.append((energy, fact, components))

        non_energy_policy = self.mode in {"cohort_vanilla_online", "cohort_hard_cache"}
        if self.mode == "cohort_vanilla_online":
            scored.sort(key=lambda item: self._vanilla_sort_key(context, item[1]))
        elif self.mode == "cohort_hard_cache":
            scored.sort(
                key=lambda item: self._hard_cache_sort_key(
                    context,
                    item[1],
                    target_id=target_id,
                    variables=variables,
                    cohort_ids=cohort_ids,
                )
            )
        elif self.mode == "cohort_energy_scrambled_rank":
            scored.sort(key=lambda item: stable_id(item[1].fact_id, target_id))
        elif self.mode == "cohort_energy_flat":
            scored.sort(key=lambda item: self._vanilla_sort_key(context, item[1]))
        else:
            scored.sort(key=lambda item: item[0])

        top = scored[:limit]
        weights = self._rank_weights(len(top)) if non_energy_policy else self._read_weights([item[0] for item in top])
        top_facts = []
        decomposition = []
        for idx, ((energy, fact, components), weight) in enumerate(zip(top, weights)):
            row = asdict(fact)
            row["energy"] = energy
            row["read_weight"] = weight
            row["rank"] = idx
            top_facts.append(row)
            decomp = {
                "fact_id": fact.fact_id,
                "fact_type": fact.fact_type,
                "energy": energy,
                "weight": weight,
                **{f"component_{k}": v for k, v in components.items()},
            }
            decomposition.append(decomp)

        basins = self._rank_basins(context, q_key, target_id, variables, cohort_ids)
        energies = [item[0] for item in top]
        entropy = -sum(w * math.log(max(w, EPS)) for w in weights) if weights else 0.0
        free_energy = (
            -self.cfg.read_temperature
            * math.log(sum(math.exp(-(e - min(energies)) / self.cfg.read_temperature) for e in energies))
            + min(energies)
            if energies
            else 0.0
        )
        selected = energies[0] if energies else None
        runnerup = energies[1] if len(energies) > 1 else None
        margin = None if selected is None or runnerup is None else runnerup - selected
        readout = CohortReadout(
            context=context,
            mode=self.mode,
            read_policy=self.memory_policy,
            target_kind=target_kind,
            target_id=target_id,
            top_facts=top_facts,
            top_basins=basins[:16],
            energy_decomposition=decomposition,
            read_entropy=entropy,
            free_energy=free_energy,
            selected_energy=selected,
            runnerup_energy=runnerup,
            energy_margin=margin,
            landscape_metrics=self.landscape_metrics(max_rows=96),
            visible_columns=self.visible_columns(context.study_name),
            metadata_seen=self.metadata_seen(context.study_name),
            summary_seen=self.summary_seen(context.study_name),
            attempted_tool_signatures=self.attempted_tool_signatures(context),
            facts_by_type=dict(Counter(f.fact_type for f in self._available_facts())),
            facts_by_study=dict(Counter(f.study_name for f in self._available_facts())),
            memory_policy=self.memory_policy,
        )
        self._record_read(readout)
        return readout

    def observe_tool_result(
        self,
        *,
        context: CohortStudyContext,
        tool_name: str,
        tool_payload: dict[str, Any],
        result_text: str,
    ) -> None:
        self._start_context(context)
        signature = self.tool_signature(tool_name, tool_payload)
        event = {
            "instance_idx": context.instance_idx,
            "study_name": context.study_name,
            "variant_id": context.variant_id,
            "step": context.step,
            "tool": tool_name,
            "signature": signature,
            "payload": dict(tool_payload),
            "result_excerpt": result_text[:1200],
            "error": "ERROR" in result_text or "SQL ERROR" in result_text,
        }
        self.tool_events.append(event)
        if len(self.tool_events) > 512:
            self.tool_events = self.tool_events[-512:]
        if not self.write_enabled:
            self.last_update = {
                "mode": self.mode,
                "write_enabled": False,
                "tool": tool_name,
                "created": 0,
            }
            return

        before = len(self.facts)
        if tool_name == "get_database_metadata":
            self._parse_metadata(context, result_text)
        elif tool_name == "get_data_summary":
            self._parse_data_summary(context, result_text)
        elif tool_name == "estimate_survival_by_group":
            self._parse_estimate_survival(context, result_text, tool_payload)
        elif tool_name == "predict_cohort_survival":
            self._parse_predict_survival(context, result_text, tool_payload)
        elif tool_name == "query_sql":
            self._write_fact(
                fact_type="QUERY_RESULT",
                scope=f"study:{context.study_name}",
                subject=stable_id(tool_payload.get("sql", "")),
                predicate="result",
                obj=result_text[:800],
                confidence_logit=0.3 if "ERROR" not in result_text else -1.0,
                source="query_sql",
                context=context,
                variables=[],
            )
        elif tool_name == "submit_cohort_report":
            pass

        if self.mode == "cohort_sliding_window":
            self._evict_to_sliding_window()
        self._evict_to_capacity()
        self.last_update = {
            "mode": self.mode,
            "write_enabled": True,
            "tool": tool_name,
            "created_or_updated": len(self.facts) - before,
            "fact_count": len(self.facts),
            "basin_count": len(self.basins),
        }

    def record_submission(
        self,
        *,
        context: CohortStudyContext,
        n_fields: int,
        n_cohorts: int,
        values: dict[str, tuple[float, float, float]],
        fallback_counts_by_method: dict[str, int],
        provenance: dict[str, dict[str, Any]],
        monotonicity_violations_before_fix: int,
    ) -> None:
        s12 = [v[0] for v in values.values()]
        s24 = [v[1] for v in values.values()]
        s36 = [v[2] for v in values.values()]
        margins = [
            float(p.get("energy_margin", 0.0) or 0.0) for p in provenance.values()
        ]
        supports = [
            float(p.get("support_count", 0.0) or 0.0) for p in provenance.values()
        ]
        uncertainties = [
            float(p.get("uncertainty", 0.0) or 0.0) for p in provenance.values()
        ]

        def mean(xs: list[float]) -> float:
            return sum(xs) / len(xs) if xs else 0.0

        layer_method_counts: dict[str, Counter[str]] = {
            "layer_1": Counter(),
            "layer_2": Counter(),
            "layer_3": Counter(),
            "layer_unknown": Counter(),
        }
        for cid, prov in provenance.items():
            layer_method_counts.setdefault(cohort_layer(cid), Counter())[
                str(prov.get("method", "unknown"))
            ] += 1

        event = {
            "instance_idx": context.instance_idx,
            "study_name": context.study_name,
            "n_fields": n_fields,
            "n_cohorts": n_cohorts,
            "mean_survival_12": mean(s12),
            "mean_survival_24": mean(s24),
            "mean_survival_36": mean(s36),
            "monotonicity_violations_before_fix": monotonicity_violations_before_fix,
            "fallback_counts_by_method": dict(fallback_counts_by_method),
            "mean_energy_margin": mean(margins),
            "mean_support_count": mean(supports),
            "mean_uncertainty": mean(uncertainties),
            "layer_1_method_counts": dict(layer_method_counts["layer_1"]),
            "layer_2_method_counts": dict(layer_method_counts["layer_2"]),
            "layer_3_method_counts": dict(layer_method_counts["layer_3"]),
            "provenance_sample": {
                key: provenance[key] for key in sorted(provenance)[:8]
            },
        }
        self.submission_events.append(event)
        if len(self.submission_events) > 128:
            self.submission_events = self.submission_events[-128:]

    def metadata_seen(self, study_name: str) -> bool:
        return any(
            f.study_name == study_name and f.fact_type == "STUDY_METADATA"
            for f in self.facts.values()
        )

    def summary_seen(self, study_name: str) -> bool:
        return any(
            f.study_name == study_name and f.fact_type == "COLUMN_DISTRIBUTION"
            for f in self.facts.values()
        )

    def visible_columns(self, study_name: str) -> list[str]:
        cols = {
            f.subject
            for f in self.facts.values()
            if f.study_name == study_name and f.fact_type == "COLUMN_EXISTS"
        }
        return sorted(cols)

    def attempted_tool_signatures(self, context: CohortStudyContext) -> list[str]:
        return [
            str(event.get("signature"))
            for event in self.tool_events
            if event.get("instance_idx") == context.instance_idx
        ]

    @staticmethod
    def tool_signature(tool_name: str, payload: dict[str, Any]) -> str:
        if tool_name in {"estimate_survival_by_group", "predict_cohort_survival"}:
            return f"{tool_name}:{payload.get('group_expression', '')}"
        if tool_name == "query_sql":
            return f"{tool_name}:{payload.get('sql', '')}"
        return tool_name

    def survival_estimate_facts(self, cohort_id: str) -> list[CohortFact]:
        return [
            fact
            for fact in self._available_facts()
            if fact.fact_type == "COHORT_SURVIVAL_ESTIMATE"
            and cohort_id in fact.cohort_ids
        ]

    def overall_survival_estimates(self) -> list[CohortFact]:
        return [
            fact
            for fact in self._available_facts()
            if fact.fact_type == "SURVIVAL_GROUP_ESTIMATE" and fact.subject.endswith(":all")
        ]

    def artifacts(self) -> dict[str, Any]:
        facts = [asdict(fact) for fact in self.facts.values()]
        basins = [asdict(basin) for basin in self.basins.values()]
        return {
            "runtime": "cohort_memory",
            "mode": self.mode,
            "config": asdict(self.cfg),
            "step_index": self.step_index,
            "study_count": len(self.study_history),
            "fact_count": len(self.facts),
            "basin_count": len(self.basins),
            "facts_by_type": dict(Counter(f.fact_type for f in self.facts.values())),
            "facts_by_scope": dict(Counter(f.scope for f in self.facts.values())),
            "facts_by_study": dict(Counter(f.study_name for f in self.facts.values())),
            "facts_by_variable": dict(
                Counter(v for f in self.facts.values() for v in f.variables)
            ),
            "facts_by_cohort_layer": dict(
                Counter(
                    cohort_layer(cid)
                    for f in self.facts.values()
                    for cid in f.cohort_ids
                )
            ),
            "facts_created": self.facts_created,
            "facts_evicted": self.facts_evicted,
            "facts_contradicted": self.facts_contradicted,
            "read_count": len(self.read_events),
            "write_count": len(self.write_events),
            "tool_event_count": len(self.tool_events),
            "submission_count": len(self.submission_events),
            "last_update": dict(self.last_update),
            "last_readout": asdict(self.last_readout) if self.last_readout else None,
            "recent_reads": list(self.read_events[-64:]),
            "recent_writes": list(self.write_events[-128:]),
            "recent_tool_events": list(self.tool_events[-128:]),
            "recent_submissions": list(self.submission_events[-32:]),
            "study_history": list(self.study_history),
            "landscape_metrics": self.landscape_metrics(max_rows=384),
            "facts": facts[-512:],
            "basins": basins[-256:],
        }

    def landscape_metrics(self, *, max_rows: int = 384) -> dict[str, Any]:
        facts = self._available_facts()
        total_rows = len(facts)
        if total_rows > max_rows:
            step = max(1, total_rows // max_rows)
            sampled_facts = facts[::step][:max_rows]
        else:
            sampled_facts = facts
        rows = [fact.embedding_key for fact in sampled_facts]
        n = len(rows)
        dim = self.cfg.manifold_dim
        if not rows:
            return {
                "n_memory_rows": 0,
                "spectral_sample_rows": 0,
                "key_dim": dim,
                "g_trace": 0.0,
                "g_fro_norm": 0.0,
                "g_spectral_norm": 0.0,
                "g_effective_rank": 0.0,
                "g_spectral_entropy": 0.0,
                "g_condition_number": 0.0,
                "top_eigenvalues": [],
                "mean_pairwise_cosine": 0.0,
                "mean_in_block_similarity": 0.0,
                "mean_out_block_similarity": 0.0,
                "in_out_similarity_gap": 0.0,
                "block_modularity_by_study": 0.0,
                "block_modularity_by_variable": 0.0,
                "block_modularity_by_cohort_layer": 0.0,
                "block_modularity_by_profile_basin": 0.0,
            }
        k = np.array(rows, dtype=float)
        gram = k @ k.T
        vals = np.linalg.eigvalsh(gram)
        vals = np.clip(vals, 0.0, None)
        total = float(vals.sum())
        if total > EPS:
            p = vals / total
            entropy = float(-np.sum([x * math.log(max(float(x), EPS)) for x in p]))
            erank = float(math.exp(entropy))
        else:
            entropy = 0.0
            erank = 0.0
        nonzero = [float(v) for v in vals if v > 1e-8]
        condition = max(nonzero) / min(nonzero) if len(nonzero) >= 2 else 0.0
        off_diag = [
            float(gram[i, j])
            for i in range(n)
            for j in range(i + 1, n)
        ]
        in_out = self._block_similarity(gram, sampled_facts)
        return {
            "n_memory_rows": total_rows,
            "spectral_sample_rows": n,
            "key_dim": dim,
            "g_trace": float(np.trace(gram)),
            "g_fro_norm": float(np.linalg.norm(gram, ord="fro")),
            "g_spectral_norm": float(vals[-1]) if len(vals) else 0.0,
            "g_effective_rank": erank,
            "g_spectral_entropy": entropy,
            "g_condition_number": condition,
            "top_eigenvalues": [float(v) for v in vals[-8:][::-1]],
            "mean_pairwise_cosine": float(sum(off_diag) / len(off_diag))
            if off_diag
            else 0.0,
            **in_out,
        }

    def _start_context(self, context: CohortStudyContext) -> None:
        study_key = f"{context.instance_idx}:{context.study_name}:{context.variant_id}"
        if self.current_study_key == study_key:
            return
        if self.current_study_facts:
            self.recent_study_fact_ids.append(list(self.current_study_facts))
        self.current_study_key = study_key
        self.current_study_facts = []
        self.step_index += 1
        if self.mode == "cohort_current_study_only":
            self.facts.clear()
            self.basins.clear()
        self.study_history.append(
            {
                "instance_idx": context.instance_idx,
                "study_name": context.study_name,
                "variant_id": context.variant_id,
                "stage_index": context.stage_index,
                "region_slice": context.region_slice,
                "started_at_step": self.step_index,
            }
        )
        if len(self.study_history) > 128:
            self.study_history = self.study_history[-128:]

    def _available_facts(self) -> list[CohortFact]:
        if self.mode == "cohort_no_write":
            return []
        if self.mode == "cohort_sliding_window":
            allowed = set(fid for ids in self.recent_study_fact_ids for fid in ids)
            allowed.update(self.current_study_facts)
            return [fact for fid, fact in self.facts.items() if fid in allowed]
        return list(self.facts.values())

    def _write_fact(
        self,
        *,
        fact_type: str,
        scope: str,
        subject: str,
        predicate: str,
        obj: str,
        confidence_logit: float,
        source: str,
        context: CohortStudyContext,
        cohort_ids: list[str] | None = None,
        variables: list[str] | None = None,
    ) -> CohortFact | None:
        if not self.write_enabled:
            return None
        cohort_ids = sorted(set(cohort_ids or []))
        variables = sorted(set(variables or []))
        fid = stable_id(fact_type, scope, subject, predicate, obj)
        embedding_text = " ".join(
            [fact_type, scope, subject, predicate, obj, *cohort_ids, *variables]
        )
        key = hash_vector(embedding_text, self.cfg.manifold_dim)
        value = hash_vector("value " + embedding_text, self.cfg.manifold_dim)
        existing = self.facts.get(fid)
        if existing is not None:
            existing.support_count += 1
            existing.confidence_logit = (
                (1.0 - self.cfg.write_lr) * existing.confidence_logit
                + self.cfg.write_lr * confidence_logit
            )
            existing.last_seen = self.step_index
            existing.source = source
            for cid in cohort_ids:
                if cid not in existing.cohort_ids:
                    existing.cohort_ids.append(cid)
            for var in variables:
                if var not in existing.variables:
                    existing.variables.append(var)
            fact = existing
        else:
            fact = CohortFact(
                fact_id=fid,
                fact_type=fact_type,
                scope=scope,
                subject=subject,
                predicate=predicate,
                object=obj,
                confidence_logit=confidence_logit,
                support_count=1,
                contradiction_count=0,
                created_at=self.step_index,
                last_seen=self.step_index,
                source=source,
                study_name=context.study_name,
                stage_index=context.stage_index,
                cohort_ids=cohort_ids,
                variables=variables,
                embedding_key=key,
                embedding_value=value,
            )
            self.facts[fid] = fact
            self.current_study_facts.append(fid)
            self.facts_created += 1
        self._update_basin(fact)
        self.write_events.append(
            {
                "step_index": self.step_index,
                "instance_idx": context.instance_idx,
                "study_name": context.study_name,
                "fact_id": fact.fact_id,
                "fact_type": fact.fact_type,
                "scope": fact.scope,
                "subject": fact.subject,
                "support_count": fact.support_count,
            }
        )
        if len(self.write_events) > 512:
            self.write_events = self.write_events[-512:]
        return fact

    def _update_basin(self, fact: CohortFact) -> None:
        kind = self._basin_kind_for_fact(fact)
        basin_id = stable_id(kind, fact.scope, fact.subject)
        basin = self.basins.get(basin_id)
        if basin is None:
            basin = CohortBasin(
                basin_id=basin_id,
                basin_kind=kind,
                scope=fact.scope,
                subject=fact.subject,
                fact_ids=[],
                key=list(fact.embedding_key),
                value=list(fact.embedding_value),
                confidence_logit=fact.confidence_logit,
                support_count=0,
                contradiction_count=0,
                created_at=self.step_index,
                last_seen=self.step_index,
            )
            self.basins[basin_id] = basin
        if fact.fact_id not in basin.fact_ids:
            basin.fact_ids.append(fact.fact_id)
        basin.support_count += 1
        basin.contradiction_count += fact.contradiction_count
        basin.confidence_logit = (
            0.75 * basin.confidence_logit + 0.25 * fact.confidence_logit
        )
        basin.last_seen = self.step_index
        basin.key = self._average_vectors([basin.key, fact.embedding_key])
        basin.value = self._average_vectors([basin.value, fact.embedding_value])

    @staticmethod
    def _average_vectors(vectors: list[list[float]]) -> list[float]:
        valid = [v for v in vectors if v]
        if not valid:
            return []
        n = min(len(v) for v in valid)
        out = [sum(v[i] for v in valid) / len(valid) for i in range(n)]
        norm = math.sqrt(sum(x * x for x in out))
        if norm <= EPS:
            return out
        return [x / norm for x in out]

    @staticmethod
    def _basin_kind_for_fact(fact: CohortFact) -> str:
        if fact.fact_type in {"COLUMN_EXISTS", "COLUMN_CANONICAL_MAP", "CODING_CONVENTION"}:
            return "canonical_variable"
        if fact.fact_type in {"STUDY_METADATA", "STUDY_BIAS", "REGION_SLICE"}:
            return "study_bias"
        if fact.fact_type == "SURVIVAL_GROUP_ESTIMATE":
            return "survival_curve"
        if fact.fact_type in {"COHORT_GROUP_DECOMPOSITION", "COHORT_SURVIVAL_ESTIMATE"}:
            return "cohort_layer"
        if fact.fact_type in {"PROFILE_SURVIVAL_ESTIMATE", "PROFILE_MIXTURE_ESTIMATE"}:
            return "latent_profile"
        return "study_schema"

    def _parse_metadata(self, context: CohortStudyContext, text: str) -> None:
        section = ""
        info: dict[str, str] = {}
        columns: list[tuple[str, str]] = []
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            if line.startswith("==="):
                section = line.strip("= ").lower()
                continue
            if section == "study info":
                match = re.match(r"^\s*([A-Za-z_]+):\s*(.*)$", line)
                if match:
                    info[match.group(1)] = match.group(2).strip()
            elif section == "columns":
                match = re.match(r"^\s*([A-Za-z_][A-Za-z0-9_]*):\s*(.*)$", line)
                if match:
                    columns.append((match.group(1), match.group(2).strip()))

        if info:
            self._write_fact(
                fact_type="STUDY_METADATA",
                scope=f"study:{context.study_name}",
                subject=context.study_name,
                predicate="metadata",
                obj=json.dumps(info, sort_keys=True),
                confidence_logit=1.2,
                source="get_database_metadata",
                context=context,
                variables=["study_metadata"],
            )
            enrollment = info.get("enrollment_brief", "")
            if enrollment:
                self._write_fact(
                    fact_type="STUDY_BIAS",
                    scope=f"study:{context.study_name}",
                    subject=context.study_name,
                    predicate="enrollment_bias",
                    obj=enrollment,
                    confidence_logit=0.9,
                    source="get_database_metadata",
                    context=context,
                    variables=["study_bias"],
                )
        for col, desc in columns:
            variables = _canonical_variables_for_column(col)
            self._write_fact(
                fact_type="COLUMN_EXISTS",
                scope=f"study:{context.study_name}",
                subject=col,
                predicate="exists",
                obj=desc,
                confidence_logit=1.4,
                source="get_database_metadata",
                context=context,
                variables=variables,
            )
            for var in variables:
                self._write_fact(
                    fact_type="COLUMN_CANONICAL_MAP",
                    scope=f"variable:{var}",
                    subject=col,
                    predicate="maps_to",
                    obj=var,
                    confidence_logit=0.8,
                    source="get_database_metadata",
                    context=context,
                    variables=[var],
                )
            if "(" in desc or "coding" in desc.lower() or "level" in desc.lower():
                self._write_fact(
                    fact_type="CODING_CONVENTION",
                    scope=f"study:{context.study_name}",
                    subject=col,
                    predicate="coding",
                    obj=desc,
                    confidence_logit=0.6,
                    source="get_database_metadata",
                    context=context,
                    variables=variables,
                )

    def _parse_data_summary(self, context: CohortStudyContext, text: str) -> None:
        for line in text.splitlines():
            num = _NUMERIC_SUMMARY_RE.match(line)
            cat = _CATEGORICAL_SUMMARY_RE.match(line)
            if num:
                col = num.group("col")
                self._write_fact(
                    fact_type="COLUMN_DISTRIBUTION",
                    scope=f"study:{context.study_name}",
                    subject=col,
                    predicate="numeric_distribution",
                    obj=num.group("body"),
                    confidence_logit=0.7,
                    source="get_data_summary",
                    context=context,
                    variables=_canonical_variables_for_column(col),
                )
            elif cat:
                col = cat.group("col")
                self._write_fact(
                    fact_type="COLUMN_DISTRIBUTION",
                    scope=f"study:{context.study_name}",
                    subject=col,
                    predicate="categorical_distribution",
                    obj=cat.group("body"),
                    confidence_logit=0.65,
                    source="get_data_summary",
                    context=context,
                    variables=_canonical_variables_for_column(col),
                )

    def _parse_estimate_survival(
        self,
        context: CohortStudyContext,
        text: str,
        payload: dict[str, Any],
    ) -> None:
        expr = str(payload.get("group_expression", ""))
        expr_id = stable_id(expr, digest_size=6)
        variables = self._variables_from_expression(expr)
        groups: dict[str, tuple[int, tuple[float, float, float]]] = {}
        decomp_rows: list[tuple[str, int, dict[str, float]]] = []
        in_unobservable = False

        for line in text.splitlines():
            group_match = _GROUP_SURVIVAL_RE.match(line)
            if group_match:
                group = group_match.group("group").strip()
                groups[group] = (
                    int(group_match.group("n")),
                    (
                        float(group_match.group("s12")),
                        float(group_match.group("s24")),
                        float(group_match.group("s36")),
                    ),
                )
                continue
            if line.startswith("Unobservable cohorts"):
                in_unobservable = True
                continue
            if in_unobservable:
                cid = line.strip()
                if cid:
                    self._write_fact(
                        fact_type="COHORT_UNOBSERVABLE",
                        scope=f"study:{context.study_name}",
                        subject=cid,
                        predicate="unobservable",
                        obj=expr_id,
                        confidence_logit=0.4,
                        source="estimate_survival_by_group",
                        context=context,
                        cohort_ids=[cid],
                        variables=variables,
                    )
                continue
            decomp = _COHORT_DECOMP_RE.match(line)
            if decomp:
                cid = decomp.group("cohort")
                n = int(decomp.group("n"))
                parts = self._parse_decomposition_parts(decomp.group("parts"))
                decomp_rows.append((cid, n, parts))

        for group, (n, surv) in groups.items():
            obj = json.dumps(
                {"n": n, "s12": surv[0], "s24": surv[1], "s36": surv[2]},
                sort_keys=True,
            )
            self._write_fact(
                fact_type="SURVIVAL_GROUP_ESTIMATE",
                scope=f"study:{context.study_name}",
                subject=f"{expr_id}:{group}",
                predicate="km_survival",
                obj=obj,
                confidence_logit=0.5 + min(1.5, math.log1p(n) / 4.0),
                source="estimate_survival_by_group",
                context=context,
                variables=variables,
            )

        for cid, n, parts in decomp_rows:
            obj = json.dumps({"n": n, "parts": parts, "expr": expr_id}, sort_keys=True)
            self._write_fact(
                fact_type="COHORT_GROUP_DECOMPOSITION",
                scope=f"cohort:{cohort_layer(cid)}",
                subject=cid,
                predicate="group_decomposition",
                obj=obj,
                confidence_logit=0.5 + min(1.2, math.log1p(n) / 5.0),
                source="estimate_survival_by_group",
                context=context,
                cohort_ids=[cid],
                variables=variables,
            )
            estimate = self._mix_group_survival(parts, groups)
            if estimate is not None:
                s12, s24, s36 = estimate
                est_obj = json.dumps(
                    {
                        "n": n,
                        "s12": s12,
                        "s24": s24,
                        "s36": s36,
                        "expr": expr_id,
                        "source_study": context.study_name,
                    },
                    sort_keys=True,
                )
                self._write_fact(
                    fact_type="COHORT_SURVIVAL_ESTIMATE",
                    scope=f"cohort:{cohort_layer(cid)}",
                    subject=cid,
                    predicate="mixture_survival",
                    obj=est_obj,
                    confidence_logit=0.25 + min(1.2, math.log1p(n) / 5.0),
                    source="estimate_survival_by_group",
                    context=context,
                    cohort_ids=[cid],
                    variables=variables,
                )

    def _parse_predict_survival(
        self,
        context: CohortStudyContext,
        text: str,
        payload: dict[str, Any],
    ) -> None:
        expr = str(payload.get("group_expression", ""))
        expr_id = stable_id(expr, digest_size=6)
        variables = self._variables_from_expression(expr)
        for line in text.splitlines():
            match = _PREDICT_RE.match(line)
            if not match:
                continue
            cid = match.group("cohort")
            kl = float(match.group("kl"))
            obj = json.dumps(
                {
                    "n": int(match.group("n")),
                    "s12": float(match.group("ms12")),
                    "s24": float(match.group("ms24")),
                    "s36": float(match.group("ms36")),
                    "km_s12": float(match.group("ks12")),
                    "km_s24": float(match.group("ks24")),
                    "km_s36": float(match.group("ks36")),
                    "kl": kl,
                    "expr": expr_id,
                    "biased_diagnostic": True,
                },
                sort_keys=True,
            )
            self._write_fact(
                fact_type="COHORT_SURVIVAL_ESTIMATE",
                scope=f"cohort:{cohort_layer(cid)}",
                subject=cid,
                predicate="predict_diagnostic_survival",
                obj=obj,
                confidence_logit=max(-0.2, 0.8 - kl),
                source="predict_cohort_survival",
                context=context,
                cohort_ids=[cid],
                variables=variables,
            )
            self._write_fact(
                fact_type="QUERY_RESULT",
                scope=f"study:{context.study_name}",
                subject=cid,
                predicate="predict_kl",
                obj=f"{kl:.6f}",
                confidence_logit=0.2,
                source="predict_cohort_survival",
                context=context,
                cohort_ids=[cid],
                variables=variables,
            )

    @staticmethod
    def _parse_decomposition_parts(text: str) -> dict[str, float]:
        parts: dict[str, float] = {}
        for chunk in text.split(","):
            match = re.match(r"\s*(?P<group>[^=]+)=\d+\s+\((?P<pct>[0-9.]+)%\)", chunk)
            if match:
                parts[match.group("group").strip()] = float(match.group("pct")) / 100.0
        total = sum(parts.values())
        if total > EPS:
            parts = {key: val / total for key, val in parts.items()}
        return parts

    @staticmethod
    def _mix_group_survival(
        parts: dict[str, float],
        groups: dict[str, tuple[int, tuple[float, float, float]]],
    ) -> tuple[float, float, float] | None:
        if not parts or not groups:
            return None
        out = [0.0, 0.0, 0.0]
        total = 0.0
        for group, weight in parts.items():
            if group not in groups:
                continue
            total += weight
            surv = groups[group][1]
            out[0] += weight * surv[0]
            out[1] += weight * surv[1]
            out[2] += weight * surv[2]
        if total <= EPS:
            return None
        return (_clip_survival(out[0] / total), _clip_survival(out[1] / total), _clip_survival(out[2] / total))

    def _variables_from_expression(self, expr: str) -> list[str]:
        cols = set(self.visible_columns(self._current_study_name()))
        found: set[str] = set()
        for token in re.findall(r"[A-Za-z_][A-Za-z0-9_]*", expr):
            if token in cols:
                found.update(_canonical_variables_for_column(token))
        return sorted(found)

    def _current_study_name(self) -> str:
        if not self.study_history:
            return ""
        return str(self.study_history[-1].get("study_name", ""))

    def _energy_components(
        self,
        context: CohortStudyContext,
        fact: CohortFact,
        q_key: list[float],
        *,
        target_kind: str,
        target_id: str,
        variables: list[str],
        cohort_ids: list[str],
    ) -> dict[str, float]:
        age = max(0, self.step_index - fact.last_seen)
        key_sim = cosine(q_key, fact.embedding_key)
        scope_penalty = self._scope_penalty(context, fact, target_kind, target_id, variables, cohort_ids)
        coverage = self._composition_coverage(variables, fact.variables)
        if cohort_ids and any(cid in fact.cohort_ids for cid in cohort_ids):
            coverage += 1.0
        bias = self._bias_penalty(fact)
        components = {
            "E_key": -key_sim,
            "E_scope": scope_penalty,
            "E_support": -math.log1p(fact.support_count),
            "E_recency": 0.015 * math.log1p(age) / math.sqrt(max(1, fact.support_count)),
            "E_uncertainty": -fact.confidence_logit,
            "E_contradiction": 1.25 * fact.contradiction_count,
            "E_bias": bias,
            "E_composition": -coverage,
        }
        mode = self.mode
        if mode == "cohort_energy_flat":
            return {key: 0.0 for key in components}
        if mode == "cohort_energy_no_key":
            components["E_key"] = 0.0
        if mode == "cohort_energy_no_scope":
            components["E_scope"] = 0.0
        if mode == "cohort_energy_no_recency":
            components["E_recency"] = 0.0
        if mode == "cohort_energy_no_uncertainty":
            components["E_uncertainty"] = 0.0
        if mode == "cohort_energy_no_contradiction":
            components["E_contradiction"] = 0.0
        if mode == "cohort_energy_no_bias":
            components["E_bias"] = 0.0
        return components

    @staticmethod
    def _composition_coverage(target: list[str], fact_vars: list[str]) -> float:
        if not target:
            return 0.0
        overlap = len(set(target) & set(fact_vars))
        return overlap / max(1, len(set(target)))

    @staticmethod
    def _bias_penalty(fact: CohortFact) -> float:
        if fact.source == "predict_cohort_survival":
            return 0.35
        if fact.fact_type == "STUDY_BIAS":
            return 0.10
        if "biased" in fact.object.lower():
            return 0.20
        return 0.0

    @staticmethod
    def _scope_penalty(
        context: CohortStudyContext,
        fact: CohortFact,
        target_kind: str,
        target_id: str,
        variables: list[str],
        cohort_ids: list[str],
    ) -> float:
        penalty = 0.0
        if target_kind == "tool" and fact.study_name == context.study_name:
            penalty -= 0.25
        if target_kind == "cohort_submission":
            if target_id in fact.cohort_ids or fact.subject == target_id:
                penalty -= 1.0
            elif fact.cohort_ids:
                penalty += 0.3
            if fact.scope == f"cohort:{cohort_layer(target_id)}":
                penalty -= 0.25
        if variables and not (set(variables) & set(fact.variables)):
            penalty += 0.15
        if cohort_ids and fact.cohort_ids and not (set(cohort_ids) & set(fact.cohort_ids)):
            penalty += 0.4
        return penalty

    def _vanilla_sort_key(
        self,
        context: CohortStudyContext,
        fact: CohortFact,
    ) -> tuple[float, float, float, str]:
        age = max(0, self.step_index - fact.last_seen)
        same_study = 0 if fact.study_name == context.study_name else 1
        return (
            same_study,
            -float(fact.support_count),
            float(age),
            fact.fact_id,
        )

    def _hard_cache_sort_key(
        self,
        context: CohortStudyContext,
        fact: CohortFact,
        *,
        target_id: str,
        variables: list[str],
        cohort_ids: list[str],
    ) -> tuple[float, float, float, str]:
        exact_cohort = 0 if target_id in fact.cohort_ids or fact.subject == target_id else 1
        exact_study = 0 if fact.study_name == context.study_name else 1
        var_miss = 0 if not variables or set(variables) & set(fact.variables) else 1
        return (exact_cohort, var_miss, exact_study, fact.fact_id)

    def _read_weights(self, energies: list[float]) -> list[float]:
        if not energies:
            return []
        tau = max(0.05, self.cfg.read_temperature)
        mn = min(energies)
        vals = [math.exp(-(e - mn) / tau) for e in energies]
        total = sum(vals)
        if total <= EPS:
            return [1.0 / len(vals)] * len(vals)
        return [v / total for v in vals]

    @staticmethod
    def _rank_weights(n: int) -> list[float]:
        if n <= 0:
            return []
        vals = [1.0 / math.sqrt(idx + 1.0) for idx in range(n)]
        total = sum(vals)
        return [v / total for v in vals]

    def _rank_basins(
        self,
        context: CohortStudyContext,
        q_key: list[float],
        target_id: str,
        variables: list[str],
        cohort_ids: list[str],
    ) -> list[dict[str, Any]]:
        rows = []
        for basin in self.basins.values():
            energy = -cosine(q_key, basin.key)
            if basin.subject == target_id or target_id in basin.subject:
                energy -= 0.5
            if cohort_ids and basin.scope == f"cohort:{cohort_layer(cohort_ids[0])}":
                energy -= 0.2
            if variables and not any(v in basin.subject for v in variables):
                energy += 0.05
            row = asdict(basin)
            row["energy"] = energy
            rows.append((energy, row))
        rows.sort(key=lambda item: item[0])
        return [row for _, row in rows]

    def _record_read(self, readout: CohortReadout) -> None:
        event = {
            "instance_idx": readout.context.instance_idx,
            "study_name": readout.context.study_name,
            "step": readout.context.step,
            "mode": self.mode,
            "read_policy": readout.read_policy,
            "target_kind": readout.target_kind,
            "target_id": readout.target_id,
            "top_fact_ids": [row.get("fact_id") for row in readout.top_facts[:12]],
            "top_basin_ids": [row.get("basin_id") for row in readout.top_basins[:8]],
            "read_entropy": readout.read_entropy,
            "free_energy": readout.free_energy,
            "selected_energy": readout.selected_energy,
            "runnerup_energy": readout.runnerup_energy,
            "energy_margin": readout.energy_margin,
            "schema_or_variable_coverage": len(readout.visible_columns),
        }
        self.read_events.append(event)
        if len(self.read_events) > 512:
            self.read_events = self.read_events[-512:]

    def _block_similarity(
        self,
        gram: np.ndarray,
        facts: list[CohortFact],
    ) -> dict[str, float]:
        n = len(facts)
        if n < 2:
            return {
                "mean_in_block_similarity": 0.0,
                "mean_out_block_similarity": 0.0,
                "in_out_similarity_gap": 0.0,
                "block_modularity_by_study": 0.0,
                "block_modularity_by_variable": 0.0,
                "block_modularity_by_cohort_layer": 0.0,
                "block_modularity_by_profile_basin": 0.0,
            }
        in_vals = []
        out_vals = []
        for i in range(n):
            for j in range(i + 1, n):
                val = float(gram[i, j])
                same = facts[i].study_name == facts[j].study_name
                if same:
                    in_vals.append(val)
                else:
                    out_vals.append(val)
        mean_in = sum(in_vals) / len(in_vals) if in_vals else 0.0
        mean_out = sum(out_vals) / len(out_vals) if out_vals else 0.0
        return {
            "mean_in_block_similarity": mean_in,
            "mean_out_block_similarity": mean_out,
            "in_out_similarity_gap": mean_in - mean_out,
            "block_modularity_by_study": self._modularity_by(lambda f: f.study_name, gram, facts),
            "block_modularity_by_variable": self._modularity_by(lambda f: ",".join(sorted(f.variables)), gram, facts),
            "block_modularity_by_cohort_layer": self._modularity_by(
                lambda f: ",".join(sorted({cohort_layer(cid) for cid in f.cohort_ids})) or "none",
                gram,
                facts,
            ),
            "block_modularity_by_profile_basin": self._modularity_by(lambda f: self._basin_kind_for_fact(f), gram, facts),
        }

    @staticmethod
    def _modularity_by(
        key_fn: Any,
        gram: np.ndarray,
        facts: list[CohortFact],
    ) -> float:
        in_vals = []
        out_vals = []
        keys = [key_fn(f) for f in facts]
        for i in range(len(facts)):
            for j in range(i + 1, len(facts)):
                if keys[i] == keys[j]:
                    in_vals.append(float(gram[i, j]))
                else:
                    out_vals.append(float(gram[i, j]))
        mean_in = sum(in_vals) / len(in_vals) if in_vals else 0.0
        mean_out = sum(out_vals) / len(out_vals) if out_vals else 0.0
        return mean_in - mean_out

    def _evict_to_sliding_window(self) -> None:
        allowed = set(fid for ids in self.recent_study_fact_ids for fid in ids)
        allowed.update(self.current_study_facts)
        for fid in list(self.facts):
            if fid not in allowed:
                self.facts.pop(fid, None)
                self.facts_evicted += 1
        self._rebuild_basins()

    def _evict_to_capacity(self) -> None:
        if len(self.facts) <= self.cfg.fact_capacity:
            return
        ranked = sorted(
            self.facts.values(),
            key=lambda f: (f.support_count, f.last_seen, f.confidence_logit),
        )
        n_remove = len(self.facts) - self.cfg.fact_capacity
        for fact in ranked[:n_remove]:
            self.facts.pop(fact.fact_id, None)
            self.facts_evicted += 1
        self._rebuild_basins()

    def _rebuild_basins(self) -> None:
        self.basins = {}
        for fact in self.facts.values():
            self._update_basin(fact)
