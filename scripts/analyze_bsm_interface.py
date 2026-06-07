#!/usr/bin/env python3
"""Summarize visible BSM interface failures in CLBench traces.

This diagnostic intentionally works only from persisted trace artifacts.  It can
count parsed-action failures, blocked instances, and empty reports; it cannot
recover raw malformed generations that were successfully repaired before the
trace stored the structured action.
"""

from __future__ import annotations

import argparse
import gzip
import json
import statistics as stats
from pathlib import Path
from typing import Any


REQUIRED_TRANSMITTER_FIELDS = {
    "center_freq": (int, float),
    "bandwidth": (int, float),
    "estimated_power": (int, float),
    "currently_active": (bool,),
}


def _load_json(path: Path) -> dict[str, Any]:
    if path.suffix == ".gz":
        with gzip.open(path, "rt") as fh:
            return json.load(fh)
    with path.open() as fh:
        return json.load(fh)


def _load_traces(path: Path) -> list[tuple[str, dict[str, Any]]]:
    if path.is_dir():
        return [(p.stem, _load_json(p)) for p in sorted(path.glob("run_*.json"))]
    data = _load_json(path)
    if "run_traces" not in data:
        return [(path.stem, data)]
    return [
        (f"run_{idx:04d}", item["trace"])
        for idx, item in enumerate(data.get("run_traces", []))
    ]


def _reward_by_instance(trace: dict[str, Any]) -> dict[str, float]:
    result = trace.get("result", {})
    outcomes = result.get("instance_outcomes") or trace.get("instance_outcomes") or []
    return {
        str(outcome.get("instance_id")): float(outcome.get("reward", 0.0))
        for outcome in outcomes
    }


def _empty_report(action: Any) -> bool:
    if not isinstance(action, dict):
        return False
    transmitters = action.get("transmitters")
    return isinstance(transmitters, list) and len(transmitters) == 0


def _validate_action(action: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(action, dict):
        return ["action_not_dict"]
    if "transmitters" not in action:
        return ["missing_transmitters"]
    transmitters = action["transmitters"]
    if not isinstance(transmitters, list):
        return ["transmitters_not_list"]
    for idx, transmitter in enumerate(transmitters):
        if not isinstance(transmitter, dict):
            errors.append(f"transmitter_{idx}_not_dict")
            continue
        for field, types in REQUIRED_TRANSMITTER_FIELDS.items():
            if field not in transmitter:
                errors.append(f"transmitter_{idx}_missing_{field}")
                continue
            value = transmitter[field]
            if not isinstance(value, types):
                errors.append(f"transmitter_{idx}_bad_{field}_type")
    return errors


def _mean(values: list[float]) -> float | None:
    return stats.mean(values) if values else None


def _summarize_run(label: str, trace: dict[str, Any]) -> dict[str, Any]:
    rewards = _reward_by_instance(trace)
    blocked = trace.get("blocked_instances") or []
    interactions = trace.get("interactions") or []
    action_type_counts: dict[str, int] = {}
    validation_error_counts: dict[str, int] = {}
    empty_rewards: list[float] = []
    nonempty_rewards: list[float] = []

    missing_response = 0
    missing_action = 0
    malformed_action = 0
    empty_reports = 0
    total_transmitters = 0

    for interaction in interactions:
        response = interaction.get("response")
        instance_id = str(interaction.get("query", {}).get("instance_id"))
        reward = rewards.get(instance_id)
        if not isinstance(response, dict):
            missing_response += 1
            continue
        action_type = str(response.get("action_type"))
        action_type_counts[action_type] = action_type_counts.get(action_type, 0) + 1
        action = response.get("action")
        if action is None:
            missing_action += 1
            continue
        errors = _validate_action(action)
        if errors:
            malformed_action += 1
            for error in errors:
                validation_error_counts[error] = validation_error_counts.get(error, 0) + 1
        transmitters = action.get("transmitters") if isinstance(action, dict) else None
        if isinstance(transmitters, list):
            total_transmitters += len(transmitters)
        if _empty_report(action):
            empty_reports += 1
            if reward is not None:
                empty_rewards.append(reward)
        elif reward is not None:
            nonempty_rewards.append(reward)

    return {
        "label": label,
        "num_interactions": len(interactions),
        "blocked_instances": len(blocked),
        "missing_response": missing_response,
        "missing_action": missing_action,
        "malformed_action": malformed_action,
        "validation_error_counts": validation_error_counts,
        "action_type_counts": action_type_counts,
        "empty_reports": empty_reports,
        "empty_report_rate": empty_reports / len(interactions) if interactions else None,
        "avg_transmitters_reported": total_transmitters / len(interactions) if interactions else None,
        "empty_report_reward_mean": _mean(empty_rewards),
        "nonempty_report_reward_mean": _mean(nonempty_rewards),
    }


def _combine(runs: list[dict[str, Any]]) -> dict[str, Any]:
    keys = [
        "num_interactions",
        "blocked_instances",
        "missing_response",
        "missing_action",
        "malformed_action",
        "empty_reports",
    ]
    combined = {key: sum(int(run.get(key) or 0) for run in runs) for key in keys}
    combined["run_count"] = len(runs)
    combined["empty_report_rate"] = (
        combined["empty_reports"] / combined["num_interactions"]
        if combined["num_interactions"]
        else None
    )
    combined["malformed_action_rate"] = (
        combined["malformed_action"] / combined["num_interactions"]
        if combined["num_interactions"]
        else None
    )
    combined_errors: dict[str, int] = {}
    combined_action_types: dict[str, int] = {}
    for run in runs:
        for error, count in run.get("validation_error_counts", {}).items():
            combined_errors[error] = combined_errors.get(error, 0) + int(count)
        for action_type, count in run.get("action_type_counts", {}).items():
            combined_action_types[action_type] = (
                combined_action_types.get(action_type, 0) + int(count)
            )
    combined["validation_error_counts"] = combined_errors
    combined["action_type_counts"] = combined_action_types
    return combined


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", help="Trace dir, trace JSON, or final-results archive")
    parser.add_argument("--label", default="source")
    parser.add_argument("--output", help="Optional output JSON path")
    args = parser.parse_args()

    traces = _load_traces(Path(args.source))
    runs = [_summarize_run(label, trace) for label, trace in traces]
    payload = {
        "label": args.label,
        "source": args.source,
        "combined": _combine(runs),
        "runs": runs,
        "limitations": [
            "Counts visible parsed-action failures only.",
            "Raw malformed generations repaired before trace storage are not recoverable here.",
        ],
    }

    text = json.dumps(payload, indent=2, sort_keys=True)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)


if __name__ == "__main__":
    main()
