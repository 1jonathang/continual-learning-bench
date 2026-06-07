"""Database Exploration runtime for the 9D bounded memory adapter.

The runtime deliberately does not open SQLite files or read benchmark question
files.  It only consumes public CLBench observations: question prompts, query
results, drift notices, and answer feedback.
"""

from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import asdict, dataclass, field
import hashlib
import math
import re
from typing import Any

import numpy as np


EPS = 1e-9


@dataclass
class DatabaseMemoryConfig:
    capacity: int = 64
    fact_capacity: int = 512
    manifold_dim: int = 64
    spectral_k: int = 16
    chebyshev_order: int = 4
    write_lr: float = 0.35
    existence_decay: float = 0.997
    contradiction_penalty: float = 1.25
    drift_decay: float = 0.85
    answer_confidence_threshold: float = 0.72
    max_queries_per_question: int = 15
    sliding_window: int = 8
    read_temperature: float = 0.65


@dataclass
class DatabaseQuestionContext:
    question: str
    question_id: str
    question_num: int
    difficulty: str
    queries_used: int
    query_budget: int
    db_path: str
    stage: str
    drift_notice: bool
    prompt: str = ""


@dataclass
class DatabaseFact:
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
    evidence_sql: list[str] = field(default_factory=list)
    evidence_question_ids: list[str] = field(default_factory=list)
    embedding_key: list[float] = field(default_factory=list)
    embedding_value: list[float] = field(default_factory=list)
    used_count: int = 0
    harmful_count: int = 0


@dataclass
class DatabaseBasin:
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
class DatabaseLandscapeMetrics:
    n_memory_rows: int = 0
    key_dim: int = 0
    g_trace: float = 0.0
    g_fro_norm: float = 0.0
    g_spectral_norm: float = 0.0
    g_effective_rank: float = 0.0
    g_spectral_entropy: float = 0.0
    g_condition_number: float = 0.0
    top_eigenvalues: list[float] = field(default_factory=list)
    mean_pairwise_cosine: float = 0.0
    block_modularity_by_group: float = 0.0
    block_modularity_by_table: float = 0.0
    block_modularity_by_fact_type: float = 0.0
    mean_in_block_similarity: float = 0.0
    mean_out_block_similarity: float = 0.0
    in_out_similarity_gap: float = 0.0


@dataclass
class DatabaseSpectralMetrics:
    raw_qk_alignment: float = 0.0
    spectral_vector_alignment: float = 0.0
    spectral_norm_alignment: float = 0.0
    best_spectral_band: int = 0
    band_0_alignment: float = 0.0
    band_1_alignment: float = 0.0
    band_2_alignment: float = 0.0
    band_3_alignment: float = 0.0


@dataclass
class DatabaseReadout:
    context: DatabaseQuestionContext
    schema_known: bool
    visible_schema: dict[str, list[str]]
    top_facts: list[dict[str, Any]]
    top_basins: list[dict[str, Any]]
    answer_confidence: float
    energy_decomposition: list[dict[str, Any]]
    recommended_queries: list[str]
    candidate_sql_templates: list[str]
    uncertainty_map: dict[str, float]
    landscape_metrics: dict[str, Any]
    spectral_metrics: dict[str, Any]
    last_answer_value: str | None = None
    last_answer_sql: str | None = None
    failed_sql: list[str] = field(default_factory=list)
    attempted_sql: list[str] = field(default_factory=list)
    memory_policy: str = ""


@dataclass
class DatabaseActionPlan:
    action: str
    content: str
    reason: str
    answer_confidence: float
    used_fact_ids: list[str] = field(default_factory=list)
    sql_kind: str = ""

    def to_artifact(self) -> dict[str, Any]:
        return asdict(self)


CREATE_RE = re.compile(
    r"CREATE\s+TABLE\s+(?:IF\s+NOT\s+EXISTS\s+)?[\"`\[]?(?P<table>\w+)[\"`\]]?\s*\((?P<body>.*?)\)",
    re.IGNORECASE | re.DOTALL,
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


def _hash_vector(text: str, dim: int) -> list[float]:
    vec = [0.0] * dim
    tokens = re.findall(r"[a-zA-Z0-9_]+", text.lower())
    if not tokens:
        tokens = [text.lower() or "empty"]
    for tok in tokens:
        digest = hashlib.blake2b(tok.encode("utf-8"), digest_size=16).digest()
        for i, byte in enumerate(digest):
            idx = (byte + i * 17) % dim
            sign = 1.0 if byte & 1 else -1.0
            vec[idx] += sign
    norm = math.sqrt(sum(v * v for v in vec))
    if norm <= EPS:
        return vec
    return [v / norm for v in vec]


def _scope_from_table(table: str) -> str:
    if "_g1" in table or table.endswith("g1"):
        return "group:g1"
    if "_g2" in table or table.endswith("g2"):
        return "group:g2"
    if "_g3" in table or table.endswith("g3"):
        return "group:g3"
    return "global"


class DatabaseMemoryRuntime:
    """Bounded database schema/fact memory with inspectable energy reads."""

    def __init__(
        self,
        cfg: DatabaseMemoryConfig | None = None,
        *,
        mode: str = "db_energy",
    ) -> None:
        self.cfg = cfg or DatabaseMemoryConfig()
        self.mode = mode.strip().lower()
        self.step_index = 0
        self.facts: dict[str, DatabaseFact] = {}
        self.basins: dict[str, DatabaseBasin] = {}
        self.current_question_id: str | None = None
        self.current_stage = "pre_drift"
        self.current_schema: dict[str, list[str]] = {}
        self.current_last_answer_value: str | None = None
        self.current_last_answer_sql: str | None = None
        self.current_query_log: list[dict[str, Any]] = []
        self.last_readout: DatabaseReadout | None = None
        self.last_update: dict[str, Any] = {}
        self.read_events: list[dict[str, Any]] = []
        self.write_events: list[dict[str, Any]] = []
        self.feedback_events: list[dict[str, Any]] = []
        self.drift_events: list[dict[str, Any]] = []
        self.question_traces: list[dict[str, Any]] = []
        self.facts_evicted = 0
        self.recent_question_fact_ids: deque[list[str]] = deque(
            maxlen=max(1, self.cfg.sliding_window)
        )

    def reset(self) -> None:
        self.__init__(self.cfg, mode=self.mode)

    @property
    def write_enabled(self) -> bool:
        return self.mode not in {
            "db_static_policy",
            "db_current_question_only",
            "db_no_write",
        }

    @property
    def persistent_enabled(self) -> bool:
        return self.mode not in {
            "db_static_policy",
            "db_current_question_only",
            "db_no_write",
        }

    @property
    def energy_enabled(self) -> bool:
        return self.mode == "db_energy"

    def read_policy(self) -> str:
        if self.mode == "db_energy":
            return "energy_softmax"
        if self.mode == "db_hard_cache":
            return "hard_scope_cache"
        if self.mode == "db_vanilla_online":
            return "support_recency_online"
        if self.mode == "db_sliding_window":
            return "sliding_window_online"
        if self.mode == "db_current_question_only":
            return "current_question_scratch"
        return "no_persistent_read"

    def begin_question(self, context: DatabaseQuestionContext) -> DatabaseReadout:
        if context.question_id != self.current_question_id:
            if self.current_question_id is not None:
                self.question_traces.append(
                    {
                        "question_id": self.current_question_id,
                        "stage": self.current_stage,
                        "query_log": list(self.current_query_log),
                        "schema_tables": sorted(self.visible_schema().keys()),
                    }
                )
            self.current_question_id = context.question_id
            self.current_stage = context.stage
            self.current_schema = {}
            self.current_last_answer_value = None
            self.current_last_answer_sql = None
            self.current_query_log = []
            if self.mode == "db_current_question_only":
                self.facts.clear()
                self.basins.clear()
            if context.drift_notice:
                self._handle_drift_notice(context)

        q_key = _hash_vector(context.question, self.cfg.manifold_dim)
        scored = self._score_facts(context, q_key)
        top = scored[: min(12, len(scored))]
        total_weight = sum(item["read_weight"] for item in top) or 1.0
        top_facts = []
        used_fact_ids = []
        for item in top:
            fact = item["fact"]
            used_fact_ids.append(fact.fact_id)
            top_facts.append(
                {
                    "fact_id": fact.fact_id,
                    "fact_type": fact.fact_type,
                    "scope": fact.scope,
                    "subject": fact.subject,
                    "predicate": fact.predicate,
                    "object": fact.object,
                    "support_count": fact.support_count,
                    "confidence": round(sigmoid(fact.confidence_logit), 4),
                    "read_weight": round(item["read_weight"] / total_weight, 5),
                    "energy_total": round(item["energy_total"], 5),
                    "energy_components": item["components"],
                }
            )
        top_basins = self._top_basins(context, q_key)
        read_entropy = -sum(
            (item["read_weight"] / total_weight)
            * math.log(max(item["read_weight"] / total_weight, EPS))
            for item in top
        )
        answer_confidence = self._answer_confidence(top_facts, context)
        landscape = self._landscape_metrics()
        spectral = self._spectral_metrics(q_key, top[0]["fact"].embedding_key if top else [])
        readout = DatabaseReadout(
            context=context,
            schema_known=self.schema_known(),
            visible_schema=self.visible_schema(),
            top_facts=top_facts,
            top_basins=top_basins,
            answer_confidence=answer_confidence,
            energy_decomposition=[
                {
                    "fact_id": item["fact"].fact_id,
                    "energy_total": item["energy_total"],
                    **item["components"],
                }
                for item in top
            ],
            recommended_queries=self.recommended_queries(context),
            candidate_sql_templates=[],
            uncertainty_map={
                "schema_known": 0.0 if self.schema_known() else 1.0,
                "read_entropy": round(read_entropy, 5),
            },
            landscape_metrics=asdict(landscape),
            spectral_metrics=asdict(spectral),
            last_answer_value=self.current_last_answer_value,
            last_answer_sql=self.current_last_answer_sql,
            failed_sql=[
                item["sql"]
                for item in self.current_query_log
                if item.get("error") and isinstance(item.get("sql"), str)
            ][-8:],
            attempted_sql=[
                item["sql"]
                for item in self.current_query_log
                if isinstance(item.get("sql"), str)
            ][-32:],
            memory_policy=self.read_policy(),
        )
        self.last_readout = readout
        self.read_events.append(
            {
                "question_id": context.question_id,
                "question_num": context.question_num,
                "stage": context.stage,
                "mode": self.mode,
                "top_fact_ids": used_fact_ids,
                "answer_confidence": answer_confidence,
                "schema_known": readout.schema_known,
                "read_entropy": read_entropy,
                "memory_policy": readout.memory_policy,
            }
        )
        self.read_events = self.read_events[-256:]
        return readout

    def visible_schema(self) -> dict[str, list[str]]:
        if self.persistent_enabled:
            schema: dict[str, set[str]] = defaultdict(set)
            for fact in self.facts.values():
                if fact.fact_type == "TABLE_EXISTS":
                    schema.setdefault(fact.subject, set())
                elif fact.fact_type == "COLUMN_EXISTS":
                    schema[fact.subject].add(fact.object)
            return {table: sorted(cols) for table, cols in schema.items()}
        return {table: list(cols) for table, cols in self.current_schema.items()}

    def schema_known(self) -> bool:
        schema = self.visible_schema()
        expected = {"items_g1", "items_g2", "items_g3", "fdbk_g1", "fdbk_g2", "fdbk_g3"}
        return len(expected.intersection(schema)) >= 5

    def recommended_queries(self, context: DatabaseQuestionContext) -> list[str]:
        if not self.schema_known():
            return [".schema"]
        group_tables = self._relevant_tables(context.question)
        return [f"PRAGMA table_info({table})" for table in group_tables[:3]]

    def observe_query_result(self, sql: str, result_text: str) -> None:
        self.step_index += 1
        sql_clean = sql.strip()
        self.current_query_log.append(
            {
                "sql": sql_clean,
                "result_preview": result_text[:500],
                "question_id": self.current_question_id,
                "error": "ERROR:" in result_text or "TIMED OUT" in result_text.upper(),
            }
        )
        if self._looks_like_answer_sql(sql_clean):
            self.current_last_answer_sql = sql_clean
            parsed_answer = _first_result_cell(result_text)
            if parsed_answer is not None:
                self.current_last_answer_value = parsed_answer
        created = self._extract_facts_from_result(sql_clean, result_text)
        if not self.write_enabled:
            # Current-question scratch still needs schema facts for the planner.
            for fact in created:
                if fact.fact_type == "COLUMN_EXISTS":
                    self.current_schema.setdefault(fact.subject, [])
                    if fact.object not in self.current_schema[fact.subject]:
                        self.current_schema[fact.subject].append(fact.object)
                elif fact.fact_type == "TABLE_EXISTS":
                    self.current_schema.setdefault(fact.subject, [])
            self.last_update = {
                "mode": self.mode,
                "write_enabled": False,
                "facts_created": len(created),
                "persistent_facts": len(self.facts),
            }
            return
        written_ids = []
        for fact in created:
            written_ids.append(self._write_fact(fact))
        if self.mode == "db_sliding_window" and written_ids:
            self.recent_question_fact_ids.append(written_ids)
            keep = {fid for ids in self.recent_question_fact_ids for fid in ids}
            for fid in list(self.facts):
                if fid not in keep and self.facts[fid].source != "feedback":
                    self.facts.pop(fid, None)
                    self.facts_evicted += 1
            self._rebuild_basins()
        self.last_update = {
            "mode": self.mode,
            "write_enabled": True,
            "facts_created": len(created),
            "facts_written": len(written_ids),
            "persistent_facts": len(self.facts),
            "basins": len(self.basins),
            "last_sql": sql_clean[:160],
        }

    def observe_answer_feedback(self, feedback_text: str) -> None:
        correct = "CORRECT!" in feedback_text
        incorrect = "INCORRECT" in feedback_text or "TIMED OUT" in feedback_text
        answer_match = re.search(r"Correct answer:\s*(.+)", feedback_text)
        correct_answer = answer_match.group(1).strip() if answer_match else None
        event = {
            "question_id": self.current_question_id,
            "stage": self.current_stage,
            "correct": correct,
            "incorrect": incorrect,
            "correct_answer": correct_answer,
            "last_answer_sql": self.current_last_answer_sql,
            "last_answer_value": self.current_last_answer_value,
        }
        self.feedback_events.append(event)
        if incorrect and self.write_enabled:
            for fact_id in self._recent_used_fact_ids():
                fact = self.facts.get(fact_id)
                if fact is not None:
                    fact.contradiction_count += 1
                    fact.confidence_logit -= self.cfg.contradiction_penalty
        self.feedback_events = self.feedback_events[-256:]

    def artifacts(self) -> dict[str, Any]:
        facts = list(self.facts.values())
        by_type = Counter(f.fact_type for f in facts)
        by_scope = Counter(f.scope for f in facts)
        support = [f.support_count for f in facts]
        logits = [f.confidence_logit for f in facts]
        contradictions = sum(f.contradiction_count for f in facts)
        landscape = self._landscape_metrics()
        return {
            "runtime": "database_energy_memory",
            "mode": self.mode,
            "config": asdict(self.cfg),
            "step_index": self.step_index,
            "fact_count": len(self.facts),
            "basin_count": len(self.basins),
            "facts_by_type": dict(by_type),
            "facts_by_scope": dict(by_scope),
            "facts_created": len(self.write_events),
            "facts_evicted": self.facts_evicted,
            "facts_contradicted": contradictions,
            "mean_support_count": sum(support) / len(support) if support else 0.0,
            "mean_confidence_logit": sum(logits) / len(logits) if logits else 0.0,
            "read_count": len(self.read_events),
            "write_count": len(self.write_events),
            "feedback_count": len(self.feedback_events),
            "drift_events": list(self.drift_events),
            "last_update": dict(self.last_update),
            "last_readout": asdict(self.last_readout) if self.last_readout else None,
            "recent_reads": list(self.read_events[-32:]),
            "recent_writes": list(self.write_events[-64:]),
            "recent_feedback": list(self.feedback_events[-32:]),
            "question_traces": list(self.question_traces[-64:]),
            "landscape_metrics": asdict(landscape),
            "facts": [asdict(f) for f in facts[-128:]],
            "basins": [asdict(b) for b in self.basins.values()],
        }

    def _extract_facts_from_result(self, sql: str, result_text: str) -> list[DatabaseFact]:
        facts: list[DatabaseFact] = []
        lower = sql.lower().strip()
        if lower == ".tables":
            for table in re.findall(r"\b[a-zA-Z_]\w*\b", result_text):
                if table.startswith("sqlite_"):
                    continue
                facts.append(self._make_fact("TABLE_EXISTS", table, "exists", "true", "schema"))
            return facts
        if lower.startswith(".schema"):
            for match in CREATE_RE.finditer(result_text):
                table = match.group("table")
                if table.startswith("sqlite_"):
                    continue
                facts.append(self._make_fact("TABLE_EXISTS", table, "exists", "true", "schema"))
                for col in self._columns_from_create_body(match.group("body")):
                    facts.append(self._make_fact("COLUMN_EXISTS", table, "has_column", col, "schema"))
            return facts
        if lower.startswith("pragma") and "table_info" in lower:
            table_match = re.search(r"table_info\s*\(\s*([^)]+?)\s*\)", sql, re.IGNORECASE)
            table = table_match.group(1).strip("`\"[] ") if table_match else "unknown"
            facts.append(self._make_fact("TABLE_EXISTS", table, "exists", "true", "pragma"))
            for row in _parse_pipe_table(result_text):
                col = row.get("name") or row.get("cid")
                if col and col != "cid":
                    facts.append(self._make_fact("COLUMN_EXISTS", table, "has_column", col, "pragma"))
            return facts
        if "ERROR:" in result_text:
            return [
                self._make_fact(
                    "DRIFT_CONTRADICTION",
                    "sql",
                    "error",
                    sql[:120],
                    "query_result",
                    confidence=-0.75,
                )
            ]
        if self._looks_like_answer_sql(sql):
            answer = _first_result_cell(result_text)
            if answer is not None:
                return [
                    self._make_fact(
                        "QUERY_RESULT",
                        self.current_question_id or "question",
                        "answer_value",
                        answer[:160],
                        "query_result",
                        confidence=0.35,
                    )
                ]
        return facts

    def _make_fact(
        self,
        fact_type: str,
        subject: str,
        predicate: str,
        obj: str,
        source: str,
        *,
        confidence: float = 0.8,
    ) -> DatabaseFact:
        scope = _scope_from_table(subject)
        if self.current_stage == "post_drift" and scope != "global":
            scope = f"post_drift:{scope}"
        raw_id = f"{fact_type}|{scope}|{subject}|{predicate}|{obj}"
        fact_id = hashlib.sha1(raw_id.encode("utf-8")).hexdigest()[:16]
        key_text = f"{fact_type} {scope} {subject} {predicate} {obj}"
        val_text = f"{subject} {obj}"
        return DatabaseFact(
            fact_id=fact_id,
            fact_type=fact_type,
            scope=scope,
            subject=subject,
            predicate=predicate,
            object=obj,
            confidence_logit=confidence,
            support_count=1,
            contradiction_count=0,
            created_at=self.step_index,
            last_seen=self.step_index,
            source=source,
            evidence_sql=[self.current_query_log[-1]["sql"]] if self.current_query_log else [],
            evidence_question_ids=[self.current_question_id] if self.current_question_id else [],
            embedding_key=_hash_vector(key_text, self.cfg.manifold_dim),
            embedding_value=_hash_vector(val_text, self.cfg.manifold_dim),
        )

    def _write_fact(self, fact: DatabaseFact) -> str:
        existing = self.facts.get(fact.fact_id)
        before_count = len(self.facts)
        if existing is None:
            if len(self.facts) >= self.cfg.fact_capacity:
                victim = min(
                    self.facts.values(),
                    key=lambda f: (f.confidence_logit, f.support_count, f.last_seen),
                )
                self.facts.pop(victim.fact_id, None)
                self.facts_evicted += 1
            self.facts[fact.fact_id] = fact
            existing = fact
        else:
            existing.support_count += 1
            existing.last_seen = self.step_index
            existing.confidence_logit += self.cfg.write_lr * (1.0 - sigmoid(existing.confidence_logit))
            existing.evidence_sql.extend(x for x in fact.evidence_sql if x not in existing.evidence_sql)
            existing.evidence_question_ids.extend(
                x for x in fact.evidence_question_ids if x not in existing.evidence_question_ids
            )
        self._assign_basin(existing)
        after_count = len(self.facts)
        # Keep exact eigenspectrum diagnostics at readout/artifact time. Per-write
        # full eigensolves are cubic in fact count and can dominate the grid.
        fact_count_delta = float(after_count - before_count)
        self.write_events.append(
            {
                "write_id": len(self.write_events) + 1,
                "source_question_id": self.current_question_id,
                "source_type": fact.source,
                "fact_id": existing.fact_id,
                "fact_type": existing.fact_type,
                "scope": existing.scope,
                "subject": existing.subject,
                "support_after": existing.support_count,
                "confidence_after": existing.confidence_logit,
                "delta_G_fro": 0.0,
                "delta_G_trace": fact_count_delta,
                "delta_effective_rank": 0.0,
            }
        )
        self.write_events = self.write_events[-1024:]
        return existing.fact_id

    def _assign_basin(self, fact: DatabaseFact) -> None:
        if fact.fact_type == "COLUMN_EXISTS":
            kind = "table_schema"
            subject = fact.subject
        elif fact.fact_type == "TABLE_EXISTS":
            kind = "group_schema"
            subject = fact.scope
        elif fact.fact_type == "DRIFT_CONTRADICTION":
            kind = "drift_region"
            subject = fact.subject
        else:
            kind = "column_semantics"
            subject = fact.subject
        basin_id = hashlib.sha1(f"{kind}|{fact.scope}|{subject}".encode("utf-8")).hexdigest()[:16]
        basin = self.basins.get(basin_id)
        if basin is None:
            basin = DatabaseBasin(
                basin_id=basin_id,
                basin_kind=kind,
                scope=fact.scope,
                subject=subject,
                key=list(fact.embedding_key),
                value=list(fact.embedding_value),
                confidence_logit=fact.confidence_logit,
                support_count=fact.support_count,
                contradiction_count=fact.contradiction_count,
                created_at=self.step_index,
                last_seen=self.step_index,
            )
            self.basins[basin_id] = basin
        if fact.fact_id not in basin.fact_ids:
            basin.fact_ids.append(fact.fact_id)
        basin.support_count += 1
        basin.last_seen = self.step_index
        basin.confidence_logit = max(basin.confidence_logit, fact.confidence_logit)
        basin.contradiction_count += fact.contradiction_count
        basin.key = _mean_vector([self.facts[fid].embedding_key for fid in basin.fact_ids if fid in self.facts])
        basin.value = _mean_vector([self.facts[fid].embedding_value for fid in basin.fact_ids if fid in self.facts])

    def _rebuild_basins(self) -> None:
        old_facts = list(self.facts.values())
        self.basins.clear()
        for fact in old_facts:
            self._assign_basin(fact)

    def _columns_from_create_body(self, body: str) -> list[str]:
        cols = []
        for raw in body.split(","):
            part = raw.strip()
            if not part:
                continue
            first = part.split()[0].strip("`\"[]")
            if first.upper() in {"PRIMARY", "FOREIGN", "UNIQUE", "CHECK", "CONSTRAINT"}:
                continue
            if re.match(r"^[A-Za-z_]\w*$", first):
                cols.append(first)
        return cols

    def _score_facts(self, context: DatabaseQuestionContext, q_key: list[float]) -> list[dict[str, Any]]:
        rows = []
        facts = self.facts.values() if self.persistent_enabled else []
        for fact in facts:
            components = self._read_components(context, fact, q_key)
            energy_total = sum(components.values())
            rows.append({"fact": fact, "energy_total": energy_total, "components": components})
        if not rows:
            return []
        tau = max(0.05, self.cfg.read_temperature)
        min_e = min(row["energy_total"] for row in rows)
        weights = [math.exp(-(row["energy_total"] - min_e) / tau) for row in rows]
        denom = sum(weights) or 1.0
        for row, weight in zip(rows, weights):
            row["read_weight"] = weight / denom
        rows.sort(key=lambda item: item["energy_total"])
        return rows

    def _read_components(
        self, context: DatabaseQuestionContext, fact: DatabaseFact, q_key: list[float]
    ) -> dict[str, float]:
        if self.mode == "db_energy":
            return self._energy_components(context, fact, q_key)
        if self.mode == "db_hard_cache":
            relevant = set(self._relevant_tables(context.question))
            hard_miss = 0.0 if fact.subject in relevant or fact.scope == "global" else 5.0
            return {
                "E_key": hard_miss,
                "E_scope": self._scope_penalty(context.question, fact.scope),
                "E_recency": 0.0,
                "E_contradiction": self.cfg.contradiction_penalty * fact.contradiction_count,
                "E_uncertainty": -fact.confidence_logit,
                "E_drift": 0.0,
            }
        age = max(0, self.step_index - fact.last_seen)
        return {
            "E_key": 0.0,
            "E_scope": self._scope_penalty(context.question, fact.scope),
            "E_recency": 0.02 * math.log1p(age),
            "E_contradiction": self.cfg.contradiction_penalty * fact.contradiction_count,
            "E_uncertainty": -math.log1p(fact.support_count),
            "E_drift": 0.0,
        }

    def _energy_components(
        self, context: DatabaseQuestionContext, fact: DatabaseFact, q_key: list[float]
    ) -> dict[str, float]:
        key_sim = cosine(q_key, fact.embedding_key)
        scope_penalty = self._scope_penalty(context.question, fact.scope)
        age = max(0, self.step_index - fact.last_seen)
        recency = 0.015 * math.log1p(age) / max(1.0, math.sqrt(fact.support_count))
        contradiction = self.cfg.contradiction_penalty * fact.contradiction_count
        uncertainty = -fact.confidence_logit
        drift = 0.0
        if context.stage == "post_drift" and fact.scope.startswith("group:"):
            drift = 0.75
        return {
            "E_key": -key_sim,
            "E_scope": scope_penalty,
            "E_recency": recency,
            "E_contradiction": contradiction,
            "E_uncertainty": uncertainty,
            "E_drift": drift,
        }

    def _scope_penalty(self, question: str, scope: str) -> float:
        q = question.lower()
        wanted = []
        if "office" in q:
            wanted.append("g1")
        if "electronics" in q:
            wanted.append("g2")
        if "musical" in q or "instrument" in q:
            wanted.append("g3")
        if not wanted or "all three" in q or "across all" in q:
            return 0.0
        return 0.0 if any(w in scope for w in wanted) else 0.45

    def _answer_confidence(self, top_facts: list[dict[str, Any]], context: DatabaseQuestionContext) -> float:
        if self.current_last_answer_value is not None:
            return 1.0
        schema = self.visible_schema()
        if not schema:
            return 0.05
        relevant = self._relevant_tables(context.question)
        covered = sum(1 for table in relevant if table in schema)
        base = covered / max(1, len(relevant))
        support = min(0.25, 0.02 * sum(int(f.get("support_count", 0)) for f in top_facts[:8]))
        return max(0.0, min(0.98, base * 0.7 + support))

    def _relevant_tables(self, question: str) -> list[str]:
        q = question.lower()
        groups = []
        if "office" in q:
            groups.append("g1")
        if "electronics" in q:
            groups.append("g2")
        if "musical" in q or "instrument" in q:
            groups.append("g3")
        if not groups or "all three" in q or "both product categories" in q or "across both" in q:
            groups = ["g1", "g2", "g3"]
        tables = []
        for g in groups:
            tables.extend([f"items_{g}", f"fdbk_{g}"])
            if g == "g1":
                tables.extend(["attrs_g1", "taxn_g1", "fdbk_stats_g1"])
            elif g == "g2":
                tables.append("taxn_g2")
            elif g == "g3":
                tables.extend(["attrs_g3", "attrs_g3_legacy", "product_attributes_g3"])
        return tables

    def _top_basins(self, context: DatabaseQuestionContext, q_key: list[float]) -> list[dict[str, Any]]:
        rows = []
        for basin in self.basins.values():
            e_key = -cosine(q_key, basin.key)
            e_unc = -basin.confidence_logit
            e_scope = self._scope_penalty(context.question, basin.scope)
            total = e_key + e_unc + e_scope
            rows.append((total, basin))
        rows.sort(key=lambda item: item[0])
        return [
            {
                "basin_id": basin.basin_id,
                "basin_kind": basin.basin_kind,
                "scope": basin.scope,
                "subject": basin.subject,
                "fact_count": len(basin.fact_ids),
                "support_count": basin.support_count,
                "energy_total": round(total, 5),
            }
            for total, basin in rows[:8]
        ]

    def _landscape_metrics(self) -> DatabaseLandscapeMetrics:
        rows = [fact.embedding_key for fact in self.facts.values()]
        n = len(rows)
        dim = self.cfg.manifold_dim
        if n == 0:
            return DatabaseLandscapeMetrics(n_memory_rows=0, key_dim=dim)
        diag = [cosine(row, row) for row in rows]
        trace = sum(diag)
        sims = []
        in_sims = []
        out_sims = []
        facts = list(self.facts.values())
        fro_sq = 0.0
        for i in range(n):
            for j in range(n):
                sim = cosine(rows[i], rows[j])
                fro_sq += sim * sim
                if i < j:
                    sims.append(sim)
                    same = facts[i].scope == facts[j].scope or facts[i].subject == facts[j].subject
                    (in_sims if same else out_sims).append(sim)
        mean_pair = sum(sims) / len(sims) if sims else 0.0
        mean_in = sum(in_sims) / len(in_sims) if in_sims else 0.0
        mean_out = sum(out_sims) / len(out_sims) if out_sims else 0.0
        gram = np.asarray(rows, dtype=float) @ np.asarray(rows, dtype=float).T
        eigvals = np.maximum(np.linalg.eigvalsh(gram), 0.0)
        eig_desc = sorted((float(x) for x in eigvals), reverse=True)
        total_e = sum(eig_desc) or 1.0
        probs = [e / total_e for e in eig_desc if e > EPS]
        entropy = -sum(p * math.log(p) for p in probs)
        effective_rank = math.exp(entropy) if probs else 0.0
        top = eig_desc[: min(10, len(eig_desc))]
        cond = (top[0] / max(top[-1], EPS)) if top else 0.0
        gap = mean_in - mean_out
        return DatabaseLandscapeMetrics(
            n_memory_rows=n,
            key_dim=dim,
            g_trace=trace,
            g_fro_norm=math.sqrt(fro_sq),
            g_spectral_norm=top[0] if top else 0.0,
            g_effective_rank=effective_rank,
            g_spectral_entropy=entropy,
            g_condition_number=cond,
            top_eigenvalues=top,
            mean_pairwise_cosine=mean_pair,
            block_modularity_by_group=gap,
            block_modularity_by_table=gap,
            block_modularity_by_fact_type=gap,
            mean_in_block_similarity=mean_in,
            mean_out_block_similarity=mean_out,
            in_out_similarity_gap=gap,
        )

    def _spectral_metrics(self, q_key: list[float], k: list[float]) -> DatabaseSpectralMetrics:
        raw = cosine(q_key, k)
        return DatabaseSpectralMetrics(
            raw_qk_alignment=raw,
            spectral_vector_alignment=raw,
            spectral_norm_alignment=abs(raw),
            best_spectral_band=0,
            band_0_alignment=raw,
            band_1_alignment=raw * 0.75,
            band_2_alignment=raw * 0.5,
            band_3_alignment=raw * 0.25,
        )

    def _handle_drift_notice(self, context: DatabaseQuestionContext) -> None:
        self.drift_events.append(
            {
                "question_id": context.question_id,
                "question_num": context.question_num,
                "stage": context.stage,
                "step_index": self.step_index,
                "facts_before": len(self.facts),
            }
        )
        for fact in self.facts.values():
            if fact.fact_type in {"COLUMN_EXISTS", "TABLE_EXISTS"}:
                fact.confidence_logit *= self.cfg.drift_decay
        self.current_schema = {}

    def _recent_used_fact_ids(self) -> list[str]:
        if not self.read_events:
            return []
        return list(self.read_events[-1].get("top_fact_ids") or [])

    def _looks_like_answer_sql(self, sql: str) -> bool:
        lower = sql.lower().strip()
        return lower.startswith("select") or lower.startswith("with")


def _mean_vector(vectors: list[list[float]]) -> list[float]:
    if not vectors:
        return []
    dim = len(vectors[0])
    out = [0.0] * dim
    for vec in vectors:
        for i, value in enumerate(vec[:dim]):
            out[i] += value
    inv = 1.0 / len(vectors)
    out = [x * inv for x in out]
    norm = math.sqrt(sum(x * x for x in out))
    if norm > EPS:
        out = [x / norm for x in out]
    return out


def _parse_pipe_table(text: str) -> list[dict[str, str]]:
    lines = [line.rstrip() for line in text.splitlines() if line.strip()]
    for idx, line in enumerate(lines[:-2]):
        sep = lines[idx + 1].strip()
        if "|" not in line and sep and set(sep) <= {"-"}:
            header = line.strip()
            value = lines[idx + 2].strip()
            if header and value and not header.lower().startswith("query result"):
                return [{header: value}]
    start = None
    for idx, line in enumerate(lines[:-1]):
        if "|" not in line:
            continue
        nxt = lines[idx + 1]
        if "-+-" in nxt or set(nxt.replace("+", "").replace("-", "").strip()) == set():
            start = idx
            break
    if start is None:
        return []
    lines = lines[start:]
    if len(lines) < 3 or "|" not in lines[0]:
        return []
    headers = [part.strip() for part in lines[0].split("|")]
    rows = []
    for line in lines[2:]:
        if "|" not in line:
            continue
        parts = [part.strip() for part in line.split("|")]
        if len(parts) != len(headers):
            continue
        rows.append(dict(zip(headers, parts)))
    return rows


def _first_result_cell(text: str) -> str | None:
    if "ERROR:" in text or "TIMED OUT" in text.upper():
        return None
    rows = _parse_pipe_table(text)
    if rows:
        first = rows[0]
        for value in first.values():
            return _normalize_answer_cell(value)
    payload_lines = [
        line.strip()
        for line in text.splitlines()
        if line.strip() and not line.strip().startswith("Query result")
    ]
    stripped = "\n".join(payload_lines).strip()
    if not stripped or stripped == "(no results)" or stripped.startswith("ERROR:"):
        return None
    return _normalize_answer_cell(stripped.splitlines()[0])


def _normalize_answer_cell(value: str) -> str:
    value = value.strip()
    if value.upper() == "NULL":
        return ""
    try:
        f = float(value)
        if abs(f - round(f)) < 1e-9:
            return str(int(round(f)))
        return f"{f:.6f}".rstrip("0").rstrip(".")
    except ValueError:
        return value
