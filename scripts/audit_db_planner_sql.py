#!/usr/bin/env python3
"""Audit database planner SQL for answer coverage, correctness, and timeout.

This is deliberately stricter than the earlier offline check: a generated SQL
statement only passes if it produces the expected first-cell answer and finishes
inside the same SQLite progress-handler timeout used by the CLBench task.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sqlite3
import sys
import time
from typing import Any


REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = REPO_ROOT / "data" / "database_exploration"
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.systems.nined_energy_memory.db_planner import build_answer_sql
from src.tasks.database_exploration.task import SQL_QUERY_TIMEOUT_SECONDS


def _load_questions(path: Path, limit: int) -> list[dict[str, Any]]:
    with path.open() as fh:
        rows = json.load(fh)
    return list(rows[:limit])


def _schema(conn: sqlite3.Connection) -> dict[str, list[str]]:
    rows = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ).fetchall()
    out: dict[str, list[str]] = {}
    for (table,) in rows:
        if str(table).startswith("sqlite_"):
            continue
        cols = conn.execute(f"PRAGMA table_info({table})").fetchall()
        out[str(table)] = [str(row[1]) for row in cols]
    return out


def _execute_first_cell(
    conn: sqlite3.Connection,
    sql: str,
    *,
    timeout_seconds: float,
) -> tuple[Any | None, float, str | None]:
    deadline = time.monotonic() + timeout_seconds
    conn.set_progress_handler(lambda: 1 if time.monotonic() > deadline else 0, 1000)
    start = time.monotonic()
    try:
        cursor = conn.execute(sql.strip().rstrip(";"))
        row = cursor.fetchone()
        elapsed = time.monotonic() - start
        if row is None:
            return None, elapsed, None
        return row[0], elapsed, None
    except sqlite3.OperationalError as exc:
        elapsed = time.monotonic() - start
        if "interrupted" in str(exc).lower():
            return None, elapsed, f"timeout>{timeout_seconds}s"
        return None, elapsed, str(exc)
    except Exception as exc:
        elapsed = time.monotonic() - start
        return None, elapsed, str(exc)
    finally:
        conn.set_progress_handler(None, 0)


def _is_correct(value: Any, question: dict[str, Any]) -> bool:
    expected = question.get("answer")
    tolerance = float(question.get("tolerance", 0.0) or 0.0)
    if value is None:
        return False
    try:
        if isinstance(expected, (int, float)) and not isinstance(expected, bool):
            return abs(float(value) - float(expected)) <= tolerance
    except (TypeError, ValueError):
        return False
    return str(value).strip().lower() == str(expected).strip().lower()


def _audit_split(
    *,
    label: str,
    db_path: Path,
    questions_path: Path,
    stage: str,
    limit: int,
    timeout_seconds: float,
) -> dict[str, Any]:
    questions = _load_questions(questions_path, limit)
    conn = sqlite3.connect(str(db_path))
    schema = _schema(conn)
    rows = []
    try:
        for question in questions:
            generated_sql = build_answer_sql(
                str(question["question"]),
                stage=stage,
                schema=schema,
            )
            row = {
                "split": label,
                "question_id": question.get("question_id"),
                "covered": generated_sql is not None,
                "correct": False,
                "within_timeout": False,
                "elapsed_seconds": None,
                "error": None,
                "generated_sql": generated_sql,
                "expected_answer": question.get("answer"),
                "actual_answer": None,
            }
            if generated_sql is not None:
                value, elapsed, error = _execute_first_cell(
                    conn,
                    generated_sql,
                    timeout_seconds=timeout_seconds,
                )
                row.update(
                    {
                        "actual_answer": value,
                        "elapsed_seconds": elapsed,
                        "error": error,
                        "within_timeout": error != f"timeout>{timeout_seconds}s"
                        and elapsed <= timeout_seconds,
                    }
                )
                row["correct"] = bool(row["within_timeout"] and _is_correct(value, question))
            rows.append(row)
    finally:
        conn.close()

    failures = [
        row
        for row in rows
        if not row["covered"] or not row["correct"] or not row["within_timeout"]
    ]
    return {
        "split": label,
        "db_path": str(db_path),
        "questions_path": str(questions_path),
        "stage": stage,
        "timeout_seconds": timeout_seconds,
        "n": len(rows),
        "covered": sum(1 for row in rows if row["covered"]),
        "correct": sum(1 for row in rows if row["correct"]),
        "within_timeout": sum(1 for row in rows if row["within_timeout"]),
        "failures": failures,
        "rows": rows,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pre-limit", type=int, default=30)
    parser.add_argument("--post-limit", type=int, default=20)
    parser.add_argument("--timeout-seconds", type=float, default=SQL_QUERY_TIMEOUT_SECONDS)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    pre = _audit_split(
        label="pre",
        db_path=DATA_DIR / "products.db",
        questions_path=DATA_DIR / "questions.json",
        stage="pre_drift",
        limit=args.pre_limit,
        timeout_seconds=args.timeout_seconds,
    )
    post = _audit_split(
        label="post",
        db_path=DATA_DIR / "products_drifted.db",
        questions_path=DATA_DIR / "questions_post_drift.json",
        stage="post_drift",
        limit=args.post_limit,
        timeout_seconds=args.timeout_seconds,
    )
    payload = {
        "timeout_seconds": args.timeout_seconds,
        "splits": [pre, post],
        "totals": {
            "n": pre["n"] + post["n"],
            "covered": pre["covered"] + post["covered"],
            "correct": pre["correct"] + post["correct"],
            "within_timeout": pre["within_timeout"] + post["within_timeout"],
            "failures": pre["failures"] + post["failures"],
        },
    }
    text = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(text)
    print(text)


if __name__ == "__main__":
    main()
