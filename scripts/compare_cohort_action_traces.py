#!/usr/bin/env python3
"""Compare Cohort Studies action/submission traces across two runs."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any


def _load_json(path: Path) -> dict[str, Any]:
    with path.open() as fh:
        return json.load(fh)


def _hash_obj(obj: Any) -> str:
    raw = json.dumps(obj, sort_keys=True, separators=(",", ":"))
    return hashlib.blake2b(raw.encode("utf-8"), digest_size=10).hexdigest()


def _load_saved_artifacts(trace_path: Path) -> dict[str, Any]:
    artifact_path = trace_path.parent / "artifacts" / trace_path.stem / "artifacts.json"
    if not artifact_path.exists():
        return {}
    payload = _load_json(artifact_path)
    return payload if isinstance(payload, dict) else {}


def _submission_fields(action: dict[str, Any]) -> dict[str, float]:
    return {
        key: float(value)
        for key, value in action.items()
        if isinstance(key, str) and "__s" in key and isinstance(value, int | float)
    }


def _normalize_trace(path: Path) -> list[dict[str, Any]]:
    trace = _load_json(path)
    saved = _load_saved_artifacts(path)
    cohort_artifact = saved.get("cohort_runtime") if isinstance(saved, dict) else {}
    recent_reads = []
    if isinstance(cohort_artifact, dict):
        recent_reads = cohort_artifact.get("recent_reads") or []
    rows = []
    read_idx = 0
    for idx, interaction in enumerate(trace.get("interactions") or []):
        response = interaction.get("response") or {}
        metadata = response.get("metadata") or {}
        action = response.get("action")
        query = interaction.get("query") or {}
        query_meta = query.get("metadata") or {}
        row: dict[str, Any] = {
            "interaction_index": idx,
            "instance_idx": query_meta.get("instance_idx", query.get("instance_index")),
            "study_name": query_meta.get("study_name"),
            "step": query_meta.get("step"),
            "mode": metadata.get("mode"),
            "top_fact_ids": metadata.get("top_fact_ids") or [],
            "read_policy": metadata.get("read_policy"),
        }
        if isinstance(action, dict) and isinstance(action.get("tool_call"), dict):
            tool_call = action["tool_call"]
            row["kind"] = "tool"
            row["tool"] = tool_call.get("tool")
            row["tool_payload_hash"] = _hash_obj(tool_call)
            row["sql_or_group_expression_hash"] = _hash_obj(
                tool_call.get("sql") or tool_call.get("group_expression") or ""
            )
            row["submission_hash"] = None
        elif isinstance(action, dict) and _submission_fields(action):
            fields = _submission_fields(action)
            row["kind"] = "submission"
            row["tool"] = "cohort_submission"
            row["tool_payload_hash"] = None
            row["sql_or_group_expression_hash"] = None
            row["submission_hash"] = _hash_obj(fields)
            row["submission_fields"] = fields
        else:
            row["kind"] = "unknown"
            row["tool"] = None
            row["tool_payload_hash"] = None
            row["sql_or_group_expression_hash"] = None
            row["submission_hash"] = None
        if read_idx < len(recent_reads):
            row["artifact_top_fact_ids"] = recent_reads[read_idx].get("top_fact_ids") or []
            read_idx += 1
        rows.append(row)
    return rows


def _levenshtein(a: list[str], b: list[str]) -> int:
    prev = list(range(len(b) + 1))
    for i, av in enumerate(a, start=1):
        cur = [i]
        for j, bv in enumerate(b, start=1):
            cur.append(
                min(
                    prev[j] + 1,
                    cur[j - 1] + 1,
                    prev[j - 1] + (0 if av == bv else 1),
                )
            )
        prev = cur
    return prev[-1]


def _compare_run(left_path: Path, right_path: Path) -> dict[str, Any]:
    left = _normalize_trace(left_path)
    right = _normalize_trace(right_path)
    left_tools = [str(row.get("tool")) + ":" + str(row.get("sql_or_group_expression_hash")) for row in left if row.get("kind") == "tool"]
    right_tools = [str(row.get("tool")) + ":" + str(row.get("sql_or_group_expression_hash")) for row in right if row.get("kind") == "tool"]
    first_divergence = None
    for idx, (lrow, rrow) in enumerate(zip(left, right)):
        comparable_keys = [
            "kind",
            "tool",
            "sql_or_group_expression_hash",
            "submission_hash",
        ]
        if any(lrow.get(key) != rrow.get(key) for key in comparable_keys):
            first_divergence = {
                "interaction_index": idx,
                "left": {key: lrow.get(key) for key in comparable_keys},
                "right": {key: rrow.get(key) for key in comparable_keys},
            }
            break

    left_subs = [row for row in left if row.get("kind") == "submission"]
    right_subs = [row for row in right if row.get("kind") == "submission"]
    field_delta_count = 0
    mean_abs_delta = 0.0
    divergent_fields = []
    deltas = []
    for sub_idx, (left_sub, right_sub) in enumerate(zip(left_subs, right_subs)):
        l_fields = left_sub.get("submission_fields") or {}
        r_fields = right_sub.get("submission_fields") or {}
        all_fields = sorted(set(l_fields) | set(r_fields))
        for field in all_fields:
            lv = float(l_fields.get(field, 0.0))
            rv = float(r_fields.get(field, 0.0))
            delta = abs(lv - rv)
            deltas.append(delta)
            if delta > 1e-9:
                field_delta_count += 1
                if len(divergent_fields) < 16:
                    divergent_fields.append(
                        {
                            "submission_index": sub_idx,
                            "field": field,
                            "left": lv,
                            "right": rv,
                            "abs_delta": delta,
                        }
                    )
    mean_abs_delta = sum(deltas) / len(deltas) if deltas else 0.0
    all_submission_hashes_match = (
        len(left_subs) == len(right_subs)
        and all(
            left_sub.get("submission_hash") == right_sub.get("submission_hash")
            for left_sub, right_sub in zip(left_subs, right_subs)
        )
    )

    return {
        "left_run": left_path.name,
        "right_run": right_path.name,
        "tool_trace_identical": left_tools == right_tools,
        "tool_trace_edit_distance": _levenshtein(left_tools, right_tools),
        "submission_hash_identical": all_submission_hashes_match,
        "left_submission_count": len(left_subs),
        "right_submission_count": len(right_subs),
        "first_divergent_tool": first_divergence,
        "submission_field_divergence_count": field_delta_count,
        "mean_abs_submission_delta": mean_abs_delta,
        "divergent_cohort_fields_sample": divergent_fields,
        "top_fact_overlap_mean": _top_fact_overlap(left, right),
    }


def _top_fact_overlap(left: list[dict[str, Any]], right: list[dict[str, Any]]) -> float:
    vals = []
    for lrow, rrow in zip(left, right):
        lset = set(lrow.get("top_fact_ids") or [])
        rset = set(rrow.get("top_fact_ids") or [])
        if not lset and not rset:
            continue
        vals.append(len(lset & rset) / max(1, len(lset | rset)))
    return sum(vals) / len(vals) if vals else 0.0


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--left", required=True)
    parser.add_argument("--right", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    left = Path(args.left)
    right = Path(args.right)
    left_runs = sorted(left.glob("run_*.json"))
    right_runs = sorted(right.glob("run_*.json"))
    if not left_runs or not right_runs:
        raise ValueError("both --left and --right must be trace directories with run_*.json")
    rows = [
        _compare_run(lpath, rpath)
        for lpath, rpath in zip(left_runs, right_runs)
    ]
    payload = {
        "left": str(left),
        "right": str(right),
        "n_paired_runs": len(rows),
        "tool_trace_identical_all": all(row["tool_trace_identical"] for row in rows),
        "submission_hash_identical_all": all(row["submission_hash_identical"] for row in rows),
        "mean_tool_trace_edit_distance": sum(row["tool_trace_edit_distance"] for row in rows) / len(rows),
        "mean_submission_field_divergence_count": sum(row["submission_field_divergence_count"] for row in rows) / len(rows),
        "mean_abs_submission_delta": sum(row["mean_abs_submission_delta"] for row in rows) / len(rows),
        "mean_top_fact_overlap": sum(row["top_fact_overlap_mean"] for row in rows) / len(rows),
        "runs": rows,
    }
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
