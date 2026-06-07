#!/usr/bin/env python3
"""Extract Database Exploration memory diagnostics from trace directories."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
import statistics as stats
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _mean(values: list[float]) -> float:
    return stats.mean(values) if values else 0.0


def _action_from_interaction(interaction: dict[str, Any]) -> str | None:
    response = interaction.get("response") or {}
    action = response.get("action")
    if isinstance(action, dict):
        raw = action.get("action")
        return str(raw).upper() if raw is not None else None
    return None


def _content_from_interaction(interaction: dict[str, Any]) -> str:
    response = interaction.get("response") or {}
    action = response.get("action")
    if isinstance(action, dict):
        return str(action.get("content", ""))
    return ""


def _query_kind(sql: str) -> str:
    s = sql.strip().lower()
    if s == ".tables":
        return "TABLE_LIST"
    if s.startswith(".schema"):
        return "SCHEMA_DUMP"
    if s.startswith("pragma"):
        return "PRAGMA_TABLE_INFO"
    if "error" in s:
        return "ERROR_RECOVERY"
    if " join " in f" {s} ":
        return "JOIN"
    if any(fn in s for fn in ["count(", "avg(", "sum(", "round(", "max(", "min("]):
        return "AGGREGATE"
    if s.startswith("select") or s.startswith("with"):
        return "ANSWER_SQL"
    return "OTHER"


def _load_saved_artifacts(trace_path: Path) -> dict[str, Any]:
    artifact_path = trace_path.parent / "artifacts" / trace_path.stem / "artifacts.json"
    if not artifact_path.exists():
        return {}
    payload = _load_json(artifact_path)
    return payload if isinstance(payload, dict) else {}


def _db_artifact(trace: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    artifacts = trace.get("system_artifacts") or trace.get("artifacts") or {}
    if isinstance(artifacts, dict):
        db = artifacts.get("db_runtime")
        if isinstance(db, dict):
            return db
    saved = _load_saved_artifacts(trace_path)
    if isinstance(saved, dict):
        return saved.get("db_runtime") or {}
    return {}


def _run_summary(path: Path) -> dict[str, Any]:
    trace = _load_json(path)
    result = trace["result"]
    metrics = result.get("metrics", {})
    db = _db_artifact(trace, path)
    interactions = trace.get("interactions") or []
    query_rows = []
    for idx, interaction in enumerate(interactions):
        action = _action_from_interaction(interaction)
        if action != "QUERY":
            continue
        response_meta = (interaction.get("response") or {}).get("metadata") or {}
        observation = interaction.get("observation") or {}
        sql = _content_from_interaction(interaction)
        query_rows.append(
            {
                "run": path.stem,
                "interaction_index": idx,
                "question_id": response_meta.get("question_id"),
                "question_num": response_meta.get("question_num"),
                "stage": response_meta.get("stage"),
                "mode": response_meta.get("mode"),
                "sql_kind": response_meta.get("db_sql_kind") or _query_kind(sql),
                "sql": sql,
                "error": "ERROR:" in str(observation.get("content", "")),
            }
        )
    facts = db.get("facts") or []
    basins = db.get("basins") or []
    reads = db.get("recent_reads") or []
    writes = db.get("recent_writes") or []
    drift = db.get("drift_events") or []
    return {
        "path": str(path),
        "score": result.get("score"),
        "accuracy": metrics.get("accuracy"),
        "total_queries": metrics.get("total_queries"),
        "avg_queries_per_question": metrics.get("avg_queries_per_question"),
        "cumulative_regret": metrics.get("cumulative_regret"),
        "db": db,
        "facts": facts,
        "basins": basins,
        "reads": reads,
        "writes": writes,
        "drift": drift,
        "query_rows": query_rows,
    }


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = sorted({key for row in rows for key in row})
    with path.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--trace-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()

    trace_dir = Path(args.trace_dir)
    out_dir = Path(args.output_dir)
    run_paths = sorted(trace_dir.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json files found in {trace_dir}")

    runs = [_run_summary(path) for path in run_paths]
    dbs = [run["db"] for run in runs]
    facts = [fact for run in runs for fact in run["facts"]]
    basins = [basin for run in runs for basin in run["basins"]]
    reads = [read for run in runs for read in run["reads"]]
    writes = [write for run in runs for write in run["writes"]]
    drift = [event for run in runs for event in run["drift"]]
    query_rows = [row for run in runs for row in run["query_rows"]]

    memory_summary = {
        "trace_dir": str(trace_dir),
        "n_runs": len(runs),
        "score_mean": _mean([float(run["score"] or 0) for run in runs]),
        "accuracy_mean": _mean([float(run["accuracy"] or 0) for run in runs]),
        "avg_queries_mean": _mean([float(run["avg_queries_per_question"] or 0) for run in runs]),
        "cumulative_regret_mean": _mean([float(run["cumulative_regret"] or 0) for run in runs]),
        "fact_count_mean": _mean([float(db.get("fact_count") or 0) for db in dbs]),
        "basin_count_mean": _mean([float(db.get("basin_count") or 0) for db in dbs]),
        "write_count_mean": _mean([float(db.get("write_count") or 0) for db in dbs]),
        "read_count_mean": _mean([float(db.get("read_count") or 0) for db in dbs]),
        "drift_event_count_mean": _mean([float(len(db.get("drift_events") or [])) for db in dbs]),
        "facts_by_type": _merge_counts([db.get("facts_by_type") or {} for db in dbs]),
        "facts_by_scope": _merge_counts([db.get("facts_by_scope") or {} for db in dbs]),
    }

    landscape_metrics = {
        "per_run": [db.get("landscape_metrics") or {} for db in dbs],
        "mean_effective_rank": _mean(
            [float((db.get("landscape_metrics") or {}).get("g_effective_rank") or 0) for db in dbs]
        ),
        "mean_in_out_similarity_gap": _mean(
            [float((db.get("landscape_metrics") or {}).get("in_out_similarity_gap") or 0) for db in dbs]
        ),
    }

    spectral_alignment = {
        "recent_read_spectral_metrics": [
            (db.get("last_readout") or {}).get("spectral_metrics") or {} for db in dbs
        ],
        "note": "Current implementation reports a Chebyshev-compatible proxy; future regression diagnostics need downstream reuse labels.",
    }

    drift_metrics = {
        "events": drift,
        "event_count": len(drift),
        "runs_with_drift_notice": sum(1 for run in runs if run["drift"]),
    }

    fact_reuse_rows = []
    for fact in facts:
        fact_reuse_rows.append(
            {
                "fact_id": fact.get("fact_id"),
                "fact_type": fact.get("fact_type"),
                "scope": fact.get("scope"),
                "subject": fact.get("subject"),
                "object": fact.get("object"),
                "support_count": fact.get("support_count"),
                "contradiction_count": fact.get("contradiction_count"),
                "used_count": fact.get("used_count"),
                "harmful_count": fact.get("harmful_count"),
            }
        )

    _write_json(out_dir / "memory_summary.json", memory_summary)
    _write_json(out_dir / "per_question_memory_trace.json", runs)
    _write_json(out_dir / "landscape_metrics.json", landscape_metrics)
    _write_json(out_dir / "spectral_alignment_metrics.json", spectral_alignment)
    _write_json(out_dir / "drift_metrics.json", drift_metrics)
    _write_csv(out_dir / "fact_reuse_table.csv", fact_reuse_rows)
    _write_csv(out_dir / "query_classification_table.csv", query_rows)
    print(json.dumps(memory_summary, indent=2, sort_keys=True))


def _merge_counts(dicts: list[dict[str, Any]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for key, value in d.items():
            out[str(key)] = out.get(str(key), 0) + int(value)
    return out


if __name__ == "__main__":
    main()
