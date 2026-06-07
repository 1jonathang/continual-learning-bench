#!/usr/bin/env python3
"""Compare Cohort Studies CLBench runs and archives."""

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


def _mean(values: list[float]) -> float:
    return stats.mean(values) if values else float("nan")


def _sd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _se(values: list[float]) -> float:
    return _sd(values) / math.sqrt(len(values)) if values else float("nan")


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
    return ordered[lo] * (1.0 - frac) + ordered[hi] * frac


def _bootstrap_ci(values: list[float], *, seed: int = 17, n: int = 10_000) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    return {"low": _percentile(means, 0.025), "high": _percentile(means, 0.975)}


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


def _cohort_artifact(trace: dict[str, Any]) -> dict[str, Any]:
    artifacts = trace.get("system_artifacts") or trace.get("artifacts") or {}
    if isinstance(artifacts, dict):
        cohort = artifacts.get("cohort_runtime")
        return cohort if isinstance(cohort, dict) else {}
    return {}


def _tool_from_interaction(interaction: dict[str, Any]) -> str | None:
    action = (interaction.get("response") or {}).get("action")
    if not isinstance(action, dict):
        return None
    tool_call = action.get("tool_call")
    if isinstance(tool_call, dict):
        raw = tool_call.get("tool")
        return str(raw) if raw else None
    if any(str(key).endswith("__s12") for key in action):
        return "cohort_submission"
    return None


def _extract_result(trace: dict[str, Any]) -> dict[str, Any]:
    result = trace["result"]
    metrics = result.get("metrics", {})
    eval_metrics = result.get("eval_metrics", {})
    outcomes = result.get("instance_outcomes", [])
    cohort = _cohort_artifact(trace)
    stage_scores = metrics.get("stage_scores") or {}
    instance_history = metrics.get("instance_history") or []
    layer_scores = _aggregate_layer_scores(instance_history)
    tool_counts: dict[str, int] = {}
    for interaction in trace.get("interactions") or []:
        tool = _tool_from_interaction(interaction)
        if tool:
            tool_counts[tool] = tool_counts.get(tool, 0) + 1
    return {
        "score": float(result.get("score", metrics.get("mean_information_gain_bits", 0.0))),
        "mean_information_gain_bits": float(metrics.get("mean_information_gain_bits", result.get("score", 0.0))),
        "mean_kl_divergence": float(metrics.get("mean_kl_divergence", 0.0)),
        "mean_reference_kl": float(metrics.get("mean_reference_kl", 0.0)),
        "actual_performance": float(eval_metrics.get("actual_performance", 0.0)),
        "optimal_performance": float(eval_metrics.get("optimal_performance", 0.0)),
        "normalized_score_against_ceiling": _safe_div(
            float(eval_metrics.get("actual_performance", 0.0)),
            float(eval_metrics.get("optimal_performance", 0.0)),
        ),
        "first_half_avg": float(metrics.get("first_half_avg", 0.0)),
        "second_half_avg": float(metrics.get("second_half_avg", 0.0)),
        "stage_scores": stage_scores,
        "layer_scores": layer_scores,
        "total_interactions": int(metrics.get("total_interactions", 0)),
        "instance_ids": [o.get("instance_id") for o in outcomes],
        "instance_rewards": [float(o.get("reward", 0.0)) for o in outcomes],
        "tool_counts": tool_counts,
        "cohort_runtime": {
            "mode": cohort.get("mode"),
            "fact_count": cohort.get("fact_count"),
            "basin_count": cohort.get("basin_count"),
            "facts_created": cohort.get("facts_created"),
            "facts_evicted": cohort.get("facts_evicted"),
            "facts_contradicted": cohort.get("facts_contradicted"),
            "read_count": cohort.get("read_count"),
            "write_count": cohort.get("write_count"),
            "tool_event_count": cohort.get("tool_event_count"),
            "submission_count": cohort.get("submission_count"),
            "facts_by_type": cohort.get("facts_by_type") or {},
            "facts_by_study": cohort.get("facts_by_study") or {},
            "facts_by_variable": cohort.get("facts_by_variable") or {},
            "facts_by_cohort_layer": cohort.get("facts_by_cohort_layer") or {},
            "landscape_metrics": cohort.get("landscape_metrics") or {},
            "recent_submissions": cohort.get("recent_submissions") or [],
        },
    }


def _safe_div(a: float, b: float) -> float:
    return a / b if abs(b) > 1e-12 else float("nan")


def _aggregate_layer_scores(instance_history: list[dict[str, Any]]) -> dict[str, float]:
    buckets: dict[str, list[float]] = {}
    for item in instance_history:
        layer_scores = item.get("layer_scores")
        if not isinstance(layer_scores, dict):
            continue
        for key, value in layer_scores.items():
            try:
                buckets.setdefault(str(key), []).append(float(value))
            except (TypeError, ValueError):
                pass
    return {key: _mean(vals) for key, vals in sorted(buckets.items())}


def _load_trace_dir(path: Path, label: str) -> dict[str, Any]:
    run_paths = sorted(path.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json files found in {path}")
    runs = [
        _extract_result(_attach_saved_artifacts(_load_json(run_path), run_path))
        for run_path in run_paths
    ]
    baseline = None
    baseline_path = path / "baseline.json"
    if baseline_path.exists():
        baseline = _extract_result(_attach_saved_artifacts(_load_json(baseline_path), baseline_path))
    return {
        "label": label,
        "source": str(path),
        "kind": "trace_dir",
        "runs": runs,
        "baseline": baseline,
    }


def _load_archive(path: Path, label: str) -> dict[str, Any]:
    payload = _load_json(path)
    if "run_traces" not in payload:
        raise ValueError(f"{path} does not look like a CLBench final-results archive")
    runs = [_extract_result(item["trace"]) for item in payload["run_traces"]]
    baseline = _extract_result(payload["baseline_trace"]) if payload.get("baseline_trace") else None
    return {
        "label": label,
        "source": str(path),
        "kind": "archive",
        "system": (payload.get("summary") or {}).get("system", {}),
        "runs": runs,
        "baseline": baseline,
    }


def _load_source(path_text: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if path.is_dir():
        return _load_trace_dir(path, label)
    return _load_archive(path, label)


def _summarize(arm: dict[str, Any]) -> dict[str, Any]:
    scores = [run["score"] for run in arm["runs"]]
    actual = [run["actual_performance"] for run in arm["runs"]]
    optimal = [run["optimal_performance"] for run in arm["runs"]]
    first = [run["first_half_avg"] for run in arm["runs"]]
    second = [run["second_half_avg"] for run in arm["runs"]]
    interactions = [run["total_interactions"] for run in arm["runs"]]
    return {
        "label": arm["label"],
        "source": arm["source"],
        "kind": arm["kind"],
        "n": len(scores),
        "scores": scores,
        "score_mean": _mean(scores),
        "score_sd": _sd(scores),
        "score_se": _se(scores),
        "actual_performance_mean": _mean(actual),
        "optimal_performance_mean": _mean(optimal),
        "normalized_score_against_ceiling_mean": _safe_div(_mean(actual), _mean(optimal)),
        "first_half_avg_mean": _mean(first),
        "second_half_avg_mean": _mean(second),
        "total_interactions_mean": _mean([float(x) for x in interactions]),
        "tool_counts_total": _merge_counts([run["tool_counts"] for run in arm["runs"]]),
        "cohort_runtime_summaries": [run["cohort_runtime"] for run in arm["runs"]],
    }


def _merge_counts(rows: list[dict[str, int]]) -> dict[str, int]:
    out: dict[str, int] = {}
    for row in rows:
        for key, value in row.items():
            out[key] = out.get(key, 0) + int(value)
    return out


def _compare(left: dict[str, Any], right: dict[str, Any]) -> dict[str, Any]:
    left_scores = [run["score"] for run in left["runs"]]
    right_scores = [run["score"] for run in right["runs"]]
    diff = _mean(left_scores) - _mean(right_scores)
    se_ind = math.sqrt(
        (_sd(left_scores) ** 2) / max(1, len(left_scores))
        + (_sd(right_scores) ** 2) / max(1, len(right_scores))
    )
    paired = False
    paired_diffs: list[float] = []
    if len(left["runs"]) == len(right["runs"]):
        paired = all(
            left_run["instance_ids"] == right_run["instance_ids"]
            for left_run, right_run in zip(left["runs"], right["runs"])
        )
        if paired:
            paired_diffs = [
                left_run["score"] - right_run["score"]
                for left_run, right_run in zip(left["runs"], right["runs"])
            ]

    instance_paired = []
    for l_run, r_run in zip(left["runs"], right["runs"]):
        if l_run["instance_ids"] != r_run["instance_ids"]:
            continue
        diffs = [
            l_reward - r_reward
            for l_reward, r_reward in zip(l_run["instance_rewards"], r_run["instance_rewards"])
        ]
        instance_paired.extend(diffs)

    return {
        "score_diff_mean": diff,
        "score_diff_se_independent": se_ind,
        "score_diff_z_independent": diff / se_ind if se_ind > 0 else None,
        "paired_by_run_order": paired,
        "paired_run_diffs": paired_diffs,
        "paired_run_diff_mean": _mean(paired_diffs) if paired_diffs else None,
        "paired_run_diff_se": _se(paired_diffs) if paired_diffs else None,
        "paired_run_bootstrap_95ci": _bootstrap_ci(paired_diffs) if paired_diffs else None,
        "paired_instance_diff_mean": _mean(instance_paired) if instance_paired else None,
        "paired_instance_bootstrap_95ci": _bootstrap_ci(instance_paired) if instance_paired else None,
        "paired_instance_count": len(instance_paired),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--energy-run", required=True)
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--energy-label", default="energy")
    parser.add_argument("--baseline-label", default="baseline")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left = _load_source(args.energy_run, args.energy_label)
    right = _load_source(args.baseline_run, args.baseline_label)
    payload = {
        "left": _summarize(left),
        "right": _summarize(right),
        "comparison": _compare(left, right),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload["comparison"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
