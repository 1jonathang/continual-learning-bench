#!/usr/bin/env python3
"""Compare BSM memory traces against an archived or trace-dir ICL run.

The script is intentionally dependency-free so it can be run inside or outside
the CLBench uv environment.  It treats run-level rollout scores as the primary
statistical unit and reports paired statistics only when run indices have
identical instance IDs in the same order.
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


def _sample_sd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


def _mean(values: list[float]) -> float:
    return stats.mean(values) if values else float("nan")


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


def _bootstrap_mean_ci(values: list[float], *, seed: int, n: int) -> dict[str, float]:
    if not values:
        return {"low": float("nan"), "high": float("nan")}
    rng = random.Random(seed)
    means = []
    for _ in range(n):
        sample = [values[rng.randrange(len(values))] for _ in values]
        means.append(_mean(sample))
    return {"low": _percentile(means, 0.025), "high": _percentile(means, 0.975)}


def _extract_result(trace: dict[str, Any]) -> dict[str, Any]:
    result = trace["result"]
    metrics = result.get("metrics", {})
    eval_metrics = result.get("eval_metrics", {})
    outcomes = result.get("instance_outcomes", [])
    return {
        "score": float(result["score"]),
        "actual_performance": float(eval_metrics.get("actual_performance", 0.0)),
        "first_half_score": metrics.get("first_half_score"),
        "second_half_score": metrics.get("second_half_score"),
        "learning_delta": metrics.get("learning_delta"),
        "instance_ids": [o.get("instance_id") for o in outcomes],
        "instance_rewards": [float(o.get("reward", 0.0)) for o in outcomes],
        "blocked_instances": trace.get("blocked_instances", []),
        "total_interactions": trace.get("execution", {}).get("total_interactions"),
    }


def _load_trace_dir(path: Path, label: str) -> dict[str, Any]:
    run_paths = sorted(path.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json files found in {path}")
    runs = [_extract_result(_load_json(p)) for p in run_paths]

    baseline = None
    baseline_path = path / "baseline.json"
    if baseline_path.exists():
        baseline_trace = _load_json(baseline_path)
        baseline = _extract_result(baseline_trace)

    return {
        "label": label,
        "source": str(path),
        "kind": "trace_dir",
        "runs": runs,
        "baseline": baseline,
    }


def _load_archive(path: Path, label: str) -> dict[str, Any]:
    data = _load_json(path)
    if "run_traces" not in data:
        raise ValueError(f"{path} does not look like a CLBench final-results archive")
    runs = [_extract_result(item["trace"]) for item in data["run_traces"]]
    baseline = None
    if data.get("baseline_trace"):
        baseline = _extract_result(data["baseline_trace"])
    summary = data.get("summary", {})
    system = summary.get("system", {})
    return {
        "label": label,
        "source": str(path),
        "kind": "archive",
        "system": system,
        "runs": runs,
        "baseline": baseline,
    }


def _load_source(path_text: str, label: str) -> dict[str, Any]:
    path = Path(path_text)
    if path.is_dir():
        return _load_trace_dir(path, label)
    return _load_archive(path, label)


def _summarize_arm(arm: dict[str, Any]) -> dict[str, Any]:
    scores = [r["score"] for r in arm["runs"]]
    actual = [r["actual_performance"] for r in arm["runs"]]
    first = [r["first_half_score"] for r in arm["runs"] if r["first_half_score"] is not None]
    second = [
        r["second_half_score"] for r in arm["runs"] if r["second_half_score"] is not None
    ]
    delta = [r["learning_delta"] for r in arm["runs"] if r["learning_delta"] is not None]
    baseline_reward = None
    gain = None
    if arm.get("baseline") is not None:
        baseline_score = arm["baseline"]["score"]
        n_instances = len(arm["runs"][0]["instance_ids"])
        baseline_reward = baseline_score * n_instances
        gain = _mean(actual) - baseline_reward
    blocked = sum(len(r.get("blocked_instances") or []) for r in arm["runs"])
    return {
        "label": arm["label"],
        "source": arm["source"],
        "n": len(scores),
        "scores": scores,
        "score_mean": _mean(scores),
        "score_sd": _sample_sd(scores),
        "score_se": _sample_sd(scores) / math.sqrt(len(scores)) if scores else float("nan"),
        "actual_performance_mean": _mean(actual),
        "baseline_reward": baseline_reward,
        "cumulative_gain": gain,
        "first_half_score_mean": _mean(first) if first else None,
        "second_half_score_mean": _mean(second) if second else None,
        "learning_delta_mean": _mean(delta) if delta else None,
        "blocked_instance_count": blocked,
    }


def _compare(a: dict[str, Any], b: dict[str, Any], *, boot: int) -> dict[str, Any]:
    a_scores = [r["score"] for r in a["runs"]]
    b_scores = [r["score"] for r in b["runs"]]
    diff = _mean(a_scores) - _mean(b_scores)
    se_ind = math.sqrt(
        (_sample_sd(a_scores) ** 2) / len(a_scores)
        + (_sample_sd(b_scores) ** 2) / len(b_scores)
    )

    paired = False
    paired_diffs: list[float] = []
    if len(a["runs"]) == len(b["runs"]):
        paired = all(
            ar["instance_ids"] == br["instance_ids"]
            for ar, br in zip(a["runs"], b["runs"])
        )
        if paired:
            paired_diffs = [ar["score"] - br["score"] for ar, br in zip(a["runs"], b["runs"])]

    paired_summary = None
    if paired_diffs:
        paired_se = _sample_sd(paired_diffs) / math.sqrt(len(paired_diffs))
        paired_summary = {
            "diffs": paired_diffs,
            "mean": _mean(paired_diffs),
            "sd": _sample_sd(paired_diffs),
            "se": paired_se,
            "t_like": _mean(paired_diffs) / paired_se if paired_se > 0 else None,
            "bootstrap_95_ci": _bootstrap_mean_ci(paired_diffs, seed=0, n=boot),
        }

    return {
        "a_label": a["label"],
        "b_label": b["label"],
        "mean_diff": diff,
        "independent_se": se_ind,
        "independent_z_like": diff / se_ind if se_ind > 0 else None,
        "paired": paired,
        "paired_summary": paired_summary,
    }


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
        "arms": [_summarize_arm(arm_a), _summarize_arm(arm_b)],
        "comparison": _compare(arm_a, arm_b, boot=args.bootstrap),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
