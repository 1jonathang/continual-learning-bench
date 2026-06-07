#!/usr/bin/env python3
"""Compare Exploitable Poker CLBench runs.

This is a Poker-specific companion to ``compare_bsm_runs.py``.  It uses
run-level rollout scores as the statistical unit, reports paired statistics
when run instance orders match, and adds Poker diagnostics that matter for
attribution: BB/hand scores, cumulative chip profit, blocked instances, and
visible invalid action proposals.
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


def _mean(values: list[float]) -> float:
    return stats.mean(values) if values else float("nan")


def _sample_sd(values: list[float]) -> float:
    return stats.stdev(values) if len(values) > 1 else 0.0


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


def _action_from_response(interaction: dict[str, Any]) -> str | None:
    response = interaction.get("response") or {}
    action = response.get("action")
    if isinstance(action, dict):
        value = action.get("action")
        return str(value).upper() if value is not None else None
    return None


def _legal_actions(interaction: dict[str, Any]) -> list[str]:
    query = interaction.get("query") or {}
    metadata = query.get("metadata") or {}
    return [str(item).upper() for item in metadata.get("legal_actions") or []]


def _count_invalid_action_outputs(trace: dict[str, Any]) -> int:
    invalid = 0
    for interaction in trace.get("interactions") or []:
        action = _action_from_response(interaction)
        legal = _legal_actions(interaction)
        if action is not None and legal and action not in legal:
            invalid += 1
    return invalid


def _fallback_poker_runtime_from_interactions(trace: dict[str, Any]) -> dict[str, Any]:
    """Recover final runtime summary from response metadata when artifacts are null."""

    for interaction in reversed(trace.get("interactions") or []):
        response = interaction.get("response") or {}
        metadata = response.get("metadata") or {}
        if metadata.get("task") != "exploitable_poker":
            continue
        last_update = metadata.get("runtime_last_update") or {}
        return {
            "mode": metadata.get("mode") or last_update.get("mode"),
            "identity_mode": metadata.get("identity_mode")
            or last_update.get("identity_mode"),
            "model_count": last_update.get("model_count"),
            "step_index": None,
            "recent_event_count": None,
        }
    return {}


def _extract_result(trace: dict[str, Any]) -> dict[str, Any]:
    result = trace["result"]
    metrics = result.get("metrics", {})
    eval_metrics = result.get("eval_metrics", {})
    outcomes = result.get("instance_outcomes", [])
    execution = trace.get("execution", {})
    artifacts = trace.get("system_artifacts") or trace.get("artifacts") or {}
    poker_artifact = {}
    if isinstance(artifacts, dict):
        poker_artifact = artifacts.get("poker_runtime") or {}
    if not poker_artifact:
        poker_artifact = _fallback_poker_runtime_from_interactions(trace)

    total_profit = metrics.get("total_profit", eval_metrics.get("actual_performance"))
    hands_played = metrics.get("hands_played", len(outcomes))
    return {
        "score": float(result["score"]),
        "total_profit": float(total_profit) if total_profit is not None else None,
        "actual_performance": float(eval_metrics.get("actual_performance", 0.0)),
        "hands_played": int(hands_played) if hands_played is not None else len(outcomes),
        "first_half_avg": metrics.get("first_half_avg"),
        "second_half_avg": metrics.get("second_half_avg"),
        "improvement": metrics.get("improvement"),
        "normalized_loss": metrics.get("normalized_loss"),
        "instance_ids": [o.get("instance_id") for o in outcomes],
        "instance_rewards": [float(o.get("reward", 0.0)) for o in outcomes],
        "blocked_instances": trace.get("blocked_instances", []),
        "total_interactions": execution.get("total_interactions"),
        "invalid_action_output_count": _count_invalid_action_outputs(trace),
        "poker_runtime": {
            "mode": poker_artifact.get("mode"),
            "identity_mode": poker_artifact.get("identity_mode"),
            "model_count": poker_artifact.get("model_count"),
            "step_index": poker_artifact.get("step_index"),
            "recent_event_count": poker_artifact.get("recent_event_count"),
        },
    }


def _load_trace_dir(path: Path, label: str) -> dict[str, Any]:
    run_paths = sorted(path.glob("run_*.json"))
    if not run_paths:
        raise ValueError(f"no run_*.json files found in {path}")
    runs = [_extract_result(_load_json(p)) for p in run_paths]

    baseline = None
    baseline_path = path / "baseline.json"
    if baseline_path.exists():
        baseline = _extract_result(_load_json(baseline_path))

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
    return {
        "label": label,
        "source": str(path),
        "kind": "archive",
        "system": summary.get("system", {}),
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
    profits = [
        float(r["total_profit"])
        for r in arm["runs"]
        if r.get("total_profit") is not None
    ]
    improvements = [
        float(r["improvement"]) for r in arm["runs"] if r.get("improvement") is not None
    ]
    first = [
        float(r["first_half_avg"])
        for r in arm["runs"]
        if r.get("first_half_avg") is not None
    ]
    second = [
        float(r["second_half_avg"])
        for r in arm["runs"]
        if r.get("second_half_avg") is not None
    ]
    blocked = sum(len(r.get("blocked_instances") or []) for r in arm["runs"])
    invalid = sum(int(r.get("invalid_action_output_count") or 0) for r in arm["runs"])
    interactions = [
        int(r["total_interactions"])
        for r in arm["runs"]
        if r.get("total_interactions") is not None
    ]
    return {
        "label": arm["label"],
        "source": arm["source"],
        "n": len(scores),
        "scores_bb_per_hand": scores,
        "score_mean_bb_per_hand": _mean(scores),
        "score_sd": _sample_sd(scores),
        "score_se": _sample_sd(scores) / math.sqrt(len(scores)) if scores else float("nan"),
        "total_profit_mean_chips": _mean(profits) if profits else None,
        "total_profit_sd_chips": _sample_sd(profits) if profits else None,
        "first_half_avg_chips_mean": _mean(first) if first else None,
        "second_half_avg_chips_mean": _mean(second) if second else None,
        "improvement_chips_mean": _mean(improvements) if improvements else None,
        "blocked_instance_count": blocked,
        "invalid_action_output_count": invalid,
        "total_interactions_mean": _mean(interactions) if interactions else None,
        "poker_runtime_summaries": [r.get("poker_runtime") for r in arm["runs"]],
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
            paired_diffs = [
                ar["score"] - br["score"] for ar, br in zip(a["runs"], b["runs"])
            ]

    paired_summary = None
    if paired_diffs:
        paired_se = _sample_sd(paired_diffs) / math.sqrt(len(paired_diffs))
        paired_summary = {
            "diffs_bb_per_hand": paired_diffs,
            "mean_bb_per_hand": _mean(paired_diffs),
            "sd": _sample_sd(paired_diffs),
            "se": paired_se,
            "t_like": _mean(paired_diffs) / paired_se if paired_se > 0 else None,
            "bootstrap_95_ci": _bootstrap_mean_ci(paired_diffs, seed=0, n=boot),
        }

    return {
        "a_label": a["label"],
        "b_label": b["label"],
        "mean_diff_bb_per_hand": diff,
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
        "task": "exploitable_poker",
        "unit": "BB/hand",
        "arms": [_summarize_arm(arm_a), _summarize_arm(arm_b)],
        "comparison": _compare(arm_a, arm_b, boot=args.bootstrap),
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
