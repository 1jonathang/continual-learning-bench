#!/usr/bin/env python3
"""Compare Database Exploration CLBench runs.

Reports score, accuracy, query regret, query counts, drift-stage splits, SQL
errors, malformed outputs, and paired run-level differences.
"""

from __future__ import annotations

import argparse
import gzip
import json
import math
from pathlib import Path
import random
import statistics as stats
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with path.open() as fh:
        return json.load(fh)


def _load_saved_artifacts(trace_path: Path) -> dict[str, Any]:
    artifact_path = trace_path.parent / "artifacts" / trace_path.stem / "artifacts.json"
    if not artifact_path.exists():
        return {}
    payload = _load_json(artifact_path)
    return payload if isinstance(payload, dict) else {}


def _attach_saved_artifacts(trace: dict[str, Any], trace_path: Path) -> dict[str, Any]:
    saved = _load_saved_artifacts(trace_path)
    if saved:
        trace = dict(trace)
        trace["system_artifacts"] = saved
    return trace


def _mean(values: list[float]) -> float:
    return stats.mean(values) if values else float("nan")


def _sample_sd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _se(values: list[float]) -> float:
    return _sample_sd(values) / math.sqrt(len(values)) if values else float("nan")


def _percentile(values: list[float], pct: float) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    pos = (len(ordered) - 1) * pct
    lo = math.floor(pos)
    hi = math.ceil(pos)
    if lo == hi:
        return ordered[lo]
    frac = pos - lo
    return ordered[lo] * (1 - frac) + ordered[hi] * frac


def _bootstrap_mean_ci(values: list[float], *, seed: int, n: int) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    return {"low": _percentile(means, 0.025), "high": _percentile(means, 0.975)}


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


def _count_malformed_actions(trace: dict[str, Any]) -> int:
    count = 0
    for interaction in trace.get("interactions") or []:
        action = _action_from_interaction(interaction)
        if action is not None and action not in {"QUERY", "ANSWER"}:
            count += 1
    return count


def _count_invalid_sql(trace: dict[str, Any]) -> int:
    count = 0
    for interaction in trace.get("interactions") or []:
        action = _action_from_interaction(interaction)
        if action != "QUERY":
            continue
        observation = interaction.get("observation") or {}
        content = str(observation.get("content", ""))
        if "ERROR:" in content:
            count += 1
    return count


def _sql_categories(trace: dict[str, Any]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for interaction in trace.get("interactions") or []:
        if _action_from_interaction(interaction) != "QUERY":
            continue
        sql = _content_from_interaction(interaction).strip().lower()
        if sql == ".tables":
            key = "TABLE_LIST"
        elif sql.startswith(".schema"):
            key = "SCHEMA_DUMP"
        elif sql.startswith("pragma"):
            key = "PRAGMA_TABLE_INFO"
        elif " join " in f" {sql} ":
            key = "JOIN"
        elif any(fn in sql for fn in ["count(", "avg(", "sum(", "round(", "max(", "min("]):
            key = "AGGREGATE"
        elif sql.startswith("select"):
            key = "ANSWER_SQL"
        else:
            key = "OTHER"
        counts[key] = counts.get(key, 0) + 1
    return counts


def _stage_metrics(question_history: list[dict[str, Any]]) -> dict[str, Any]:
    pre = question_history[:20]
    post = question_history[20:]

    def summarize(rows: list[dict[str, Any]]) -> dict[str, Any]:
        if not rows:
            return {
                "n": 0,
                "accuracy": float("nan"),
                "avg_queries": float("nan"),
                "cumulative_regret": float("nan"),
            }
        return {
            "n": len(rows),
            "accuracy": sum(1 for r in rows if r.get("correct")) / len(rows),
            "avg_queries": _mean([float(r.get("num_queries", 0)) for r in rows]),
            "cumulative_regret": sum(float(r.get("regret", 0)) for r in rows),
        }

    out = {"pre_drift": summarize(pre), "post_drift": summarize(post)}
    if post:
        out["first_5_post_drift"] = summarize(post[:5])
        out["last_5_post_drift"] = summarize(post[-5:])
    return out


def _db_runtime_summary(trace: dict[str, Any]) -> dict[str, Any]:
    artifacts = trace.get("system_artifacts") or trace.get("artifacts") or {}
    db = artifacts.get("db_runtime") if isinstance(artifacts, dict) else None
    if not isinstance(db, dict):
        return {}
    return {
        "mode": db.get("mode"),
        "fact_count": db.get("fact_count"),
        "basin_count": db.get("basin_count"),
        "facts_created": db.get("facts_created"),
        "facts_contradicted": db.get("facts_contradicted"),
        "read_count": db.get("read_count"),
        "write_count": db.get("write_count"),
        "drift_event_count": len(db.get("drift_events") or []),
        "landscape_metrics": db.get("landscape_metrics") or {},
        "facts_by_type": db.get("facts_by_type") or {},
    }


def _extract_result(trace: dict[str, Any]) -> dict[str, Any]:
    result = trace["result"]
    metrics = result.get("metrics", {})
    eval_metrics = result.get("eval_metrics", {})
    outcomes = result.get("instance_outcomes", [])
    qhist = metrics.get("question_history") or []
    return {
        "score": float(result["score"]),
        "accuracy": float(metrics.get("accuracy", 0.0)),
        "correct": int(metrics.get("correct", 0)),
        "total_questions": int(metrics.get("total_questions", len(outcomes))),
        "total_queries": int(metrics.get("total_queries", 0)),
        "avg_queries_per_question": float(metrics.get("avg_queries_per_question", 0.0)),
        "first_half_avg_queries": float(metrics.get("first_half_avg_queries", 0.0)),
        "second_half_avg_queries": float(metrics.get("second_half_avg_queries", 0.0)),
        "query_reduction": float(metrics.get("query_reduction", 0.0)),
        "cumulative_regret": float(metrics.get("cumulative_regret", 0.0)),
        "normalized_regret": float(metrics.get("normalized_regret", 0.0)),
        "loss_curve": [float(x) for x in metrics.get("loss_curve", [])],
        "question_history": qhist,
        "stage_metrics": _stage_metrics(qhist),
        "actual_performance": float(eval_metrics.get("actual_performance", 0.0)),
        "instance_ids": [o.get("instance_id") for o in outcomes],
        "instance_rewards": [float(o.get("reward", 0.0)) for o in outcomes],
        "blocked_instances": trace.get("blocked_instances", []),
        "total_interactions": trace.get("execution", {}).get("total_interactions"),
        "malformed_action_count": _count_malformed_actions(trace),
        "invalid_sql_count": _count_invalid_sql(trace),
        "sql_categories": _sql_categories(trace),
        "db_runtime": _db_runtime_summary(trace),
    }


def _load_trace_dir(path: Path, label: str) -> dict[str, Any]:
    run_paths = sorted(path.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json files found in {path}")
    runs = [_extract_result(_attach_saved_artifacts(_load_json(p), p)) for p in run_paths]
    baseline = None
    baseline_path = path / "baseline.json"
    if baseline_path.exists():
        baseline = _extract_result(_attach_saved_artifacts(_load_json(baseline_path), baseline_path))
    return {"label": label, "source": str(path), "kind": "trace_dir", "runs": runs, "baseline": baseline}


def _load_archive(path: Path, label: str) -> dict[str, Any]:
    data = _load_json(path)
    if "run_traces" not in data:
        raise ValueError(f"{path} does not look like a CLBench final-results archive")
    runs = [_extract_result(item["trace"]) for item in data["run_traces"]]
    baseline = _extract_result(data["baseline_trace"]) if data.get("baseline_trace") else None
    return {
        "label": label,
        "source": str(path),
        "kind": "archive",
        "system": data.get("summary", {}).get("system", {}),
        "runs": runs,
        "baseline": baseline,
    }


def _load_source(path_text: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if path.is_dir():
        return _load_trace_dir(path, label)
    return _load_archive(path, label)


def _sum_dicts(dicts: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for d in dicts:
        for k, v in d.items():
            out[k] = out.get(k, 0) + int(v)
    return out


def _summarize_arm(arm: dict[str, Any]) -> dict[str, Any]:
    runs = arm["runs"]
    scores = [r["score"] for r in runs]
    accuracies = [r["accuracy"] for r in runs]
    avg_queries = [r["avg_queries_per_question"] for r in runs]
    total_queries = [r["total_queries"] for r in runs]
    regret = [r["cumulative_regret"] for r in runs]
    norm_regret = [r["normalized_regret"] for r in runs]
    query_reduction = [r["query_reduction"] for r in runs]
    invalid_sql = [r["invalid_sql_count"] for r in runs]
    malformed = [r["malformed_action_count"] for r in runs]
    blocked = sum(len(r.get("blocked_instances") or []) for r in runs)
    db_summaries = [r.get("db_runtime") or {} for r in runs]
    return {
        "label": arm["label"],
        "source": arm["source"],
        "n": len(runs),
        "scores": scores,
        "score_mean": _mean(scores),
        "score_sd": _sample_sd(scores),
        "score_se": _se(scores),
        "accuracy_mean": _mean(accuracies),
        "accuracy_sd": _sample_sd(accuracies),
        "avg_queries_mean": _mean(avg_queries),
        "avg_queries_sd": _sample_sd(avg_queries),
        "total_queries_mean": _mean(total_queries),
        "cumulative_regret_mean": _mean(regret),
        "normalized_regret_mean": _mean(norm_regret),
        "query_reduction_mean": _mean(query_reduction),
        "invalid_sql_mean": _mean([float(x) for x in invalid_sql]),
        "malformed_action_mean": _mean([float(x) for x in malformed]),
        "blocked_instance_count": blocked,
        "sql_categories": _sum_dicts([r.get("sql_categories") or {} for r in runs]),
        "db_runtime_summaries": db_summaries,
        "db_runtime_means": {
            "fact_count": _mean([float(s.get("fact_count") or 0) for s in db_summaries]),
            "basin_count": _mean([float(s.get("basin_count") or 0) for s in db_summaries]),
            "write_count": _mean([float(s.get("write_count") or 0) for s in db_summaries]),
            "drift_event_count": _mean([float(s.get("drift_event_count") or 0) for s in db_summaries]),
        },
        "stage_metrics_first_run": runs[0].get("stage_metrics") if runs else {},
    }


def _paired_diffs(a: dict[str, Any], b: dict[str, Any], key: str) -> tuple[bool, list[float]]:
    if len(a["runs"]) != len(b["runs"]):
        return False, []
    paired = all(ar["instance_ids"] == br["instance_ids"] for ar, br in zip(a["runs"], b["runs"]))
    if not paired:
        return False, []
    return True, [float(ar[key]) - float(br[key]) for ar, br in zip(a["runs"], b["runs"])]


def _paired_summary(diffs: list[float], boot: int) -> dict[str, Any] | None:
    if not diffs:
        return None
    se = _se(diffs)
    return {
        "diffs": diffs,
        "mean": _mean(diffs),
        "sd": _sample_sd(diffs),
        "se": se,
        "t_like": _mean(diffs) / se if se > 0 else None,
        "bootstrap_95_ci": _bootstrap_mean_ci(diffs, seed=0, n=boot),
    }


def _compare(a: dict[str, Any], b: dict[str, Any], *, boot: int) -> dict[str, Any]:
    out: dict[str, Any] = {"a_label": a["label"], "b_label": b["label"]}
    for key in ["score", "accuracy", "avg_queries_per_question", "cumulative_regret"]:
        av = [float(r[key]) for r in a["runs"]]
        bv = [float(r[key]) for r in b["runs"]]
        diff = _mean(av) - _mean(bv)
        se_ind = math.sqrt((_sample_sd(av) ** 2) / len(av) + (_sample_sd(bv) ** 2) / len(bv))
        paired, diffs = _paired_diffs(a, b, key)
        out[key] = {
            "mean_diff": diff,
            "independent_se": se_ind,
            "independent_z_like": diff / se_ind if se_ind > 0 else None,
            "paired": paired,
            "paired_summary": _paired_summary(diffs, boot),
        }
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--a", required=True, help="Trace dir or final-results archive")
    parser.add_argument("--b", required=True, help="Trace dir or final-results archive")
    parser.add_argument("--a-label", default="A")
    parser.add_argument("--b-label", default="B")
    parser.add_argument("--bootstrap", type=int, default=10000)
    parser.add_argument("--output", help="Optional JSON output path")
    args = parser.parse_args()

    arm_a = _load_source(args.a, args.a_label)
    arm_b = _load_source(args.b, args.b_label)
    payload = {
        "task": "database_exploration",
        "unit": "normalized score; queries/question; regret units",
        "arms": [_summarize_arm(arm_a), _summarize_arm(arm_b)],
        "comparison": _compare(arm_a, arm_b, boot=args.bootstrap),
    }
    text = json.dumps(payload, indent=2, sort_keys=True)
    print(text)
    if args.output:
        out = Path(args.output)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text + "\n")


if __name__ == "__main__":
    main()
