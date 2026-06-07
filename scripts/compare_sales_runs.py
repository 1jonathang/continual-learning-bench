#!/usr/bin/env python3
"""Compare Sales Prediction CLBench runs and archives."""

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


def _sales_artifact(trace: dict[str, Any]) -> dict[str, Any]:
    artifacts = trace.get("system_artifacts") or trace.get("artifacts") or {}
    if isinstance(artifacts, dict):
        sales = artifacts.get("sales_runtime")
        return sales if isinstance(sales, dict) else {}
    return {}


def _extract_result(trace: dict[str, Any]) -> dict[str, Any]:
    result = trace["result"]
    metrics = result.get("metrics", {})
    eval_metrics = result.get("eval_metrics", {})
    outcomes = result.get("instance_outcomes", [])
    sales = _sales_artifact(trace)
    rewards = [float(o.get("reward", 0.0)) for o in outcomes]
    return {
        "score": float(result.get("score", _mean(rewards))),
        "total_reward": float(metrics.get("total_reward", sum(rewards))),
        "actual_performance": float(eval_metrics.get("actual_performance", sum(rewards))),
        "optimal_performance": float(eval_metrics.get("optimal_performance", len(rewards))),
        "loss_curve": [float(v) for v in eval_metrics.get("loss_curve", [])],
        "num_instances": int(metrics.get("num_instances", len(rewards))),
        "instance_ids": [o.get("instance_id") for o in outcomes],
        "instance_rewards": rewards,
        "format_valid_count": sum(
            1
            for o in outcomes
            if ((o.get("metadata") or {}).get("format_valid") is True)
        ),
        "timed_out_count": sum(
            1
            for o in outcomes
            if ((o.get("metadata") or {}).get("timed_out") is True)
        ),
        "sales_runtime": {
            "mode": sales.get("mode"),
            "record_count": sales.get("record_count"),
            "current_record_count": sales.get("current_record_count"),
            "catalog_size": sales.get("catalog_size"),
            "location_count": sales.get("location_count"),
            "write_count": sales.get("write_count"),
            "prediction_count": sales.get("prediction_count"),
            "source_counts": sales.get("source_counts") or {},
            "landscape_metrics": sales.get("landscape_metrics") or {},
            "last_update": sales.get("last_update") or {},
        },
    }


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
        baseline = _extract_result(
            _attach_saved_artifacts(_load_json(baseline_path), baseline_path)
        )
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
    return {
        "label": arm["label"],
        "source": arm["source"],
        "kind": arm["kind"],
        "n": len(scores),
        "score_mean": _mean(scores),
        "score_sd": _sd(scores),
        "score_se": _se(scores),
        "scores": scores,
        "actual_performance_mean": _mean(actual),
        "actual_performance_sd": _sd(actual),
        "num_instances": [run["num_instances"] for run in arm["runs"]],
        "format_valid_count": [run["format_valid_count"] for run in arm["runs"]],
        "timed_out_count": [run["timed_out_count"] for run in arm["runs"]],
        "sales_runtime": [
            run.get("sales_runtime") for run in arm["runs"] if run.get("sales_runtime")
        ],
    }


def _paired(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    pairs = []
    instance_diffs: list[float] = []
    aligned = True
    for arun, brun in zip(a["runs"], b["runs"]):
        if arun["instance_ids"] != brun["instance_ids"]:
            aligned = False
        pairs.append(float(arun["score"]) - float(brun["score"]))
        if len(arun["instance_rewards"]) == len(brun["instance_rewards"]):
            instance_diffs.extend(
                float(x) - float(y)
                for x, y in zip(arun["instance_rewards"], brun["instance_rewards"])
            )
        else:
            aligned = False
    return {
        "run_order_aligned": aligned and len(a["runs"]) == len(b["runs"]),
        "run_diffs": pairs,
        "run_diff_mean": _mean(pairs),
        "run_diff_sd": _sd(pairs),
        "run_diff_se": _se(pairs),
        "run_bootstrap_ci": _bootstrap_ci(pairs),
        "instance_diff_n": len(instance_diffs),
        "instance_diff_mean": _mean(instance_diffs),
        "instance_diff_sd": _sd(instance_diffs),
        "instance_diff_se": _se(instance_diffs),
        "instance_bootstrap_ci": _bootstrap_ci(instance_diffs),
    }


def _independent(a: dict[str, Any], b: dict[str, Any]) -> dict[str, Any]:
    ascores = [run["score"] for run in a["runs"]]
    bscores = [run["score"] for run in b["runs"]]
    diff = _mean(ascores) - _mean(bscores)
    se = math.sqrt(_se(ascores) ** 2 + _se(bscores) ** 2)
    return {
        "mean_diff": diff,
        "independent_se": se,
        "z_like": diff / se if se > 0 else None,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--left-label", default="left")
    parser.add_argument("--right-label", default="right")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left = _load_source(args.left, args.left_label)
    right = _load_source(args.right, args.right_label)
    payload = {
        "left": _summarize(left),
        "right": _summarize(right),
        "independent": _independent(left, right),
        "paired": _paired(left, right),
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True))
    print(json.dumps(payload["independent"], indent=2, sort_keys=True))
    print(json.dumps(payload["paired"], indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
