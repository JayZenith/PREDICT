"""pass@k reporting, frontier selection, and paired checkpoint comparison."""

from __future__ import annotations

import json
import math
from collections import defaultdict
from pathlib import Path


def _passes(row: dict) -> bool:
    metrics = row.get("metrics") or {}
    if "passed" in metrics:
        return float(metrics["passed"]) > 0
    rewards = row.get("rewards") or {}
    return float(row.get("reward", rewards.get("mbpp_reward", 0)) or 0) > 0


def _task_id(row: dict, fallback: int) -> str:
    task = row.get("task") or {}
    data = task.get("data") or row.get("task_data") or {}
    return str(data.get("case_id", data.get("idx", row.get("task_id", fallback))))


def grouped_results(path: Path) -> dict[str, list[bool]]:
    grouped: dict[str, list[bool]] = defaultdict(list)
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            grouped[_task_id(row, line_no)].append(_passes(row))
    if not grouped:
        raise ValueError(f"no traces in {path}")
    return dict(grouped)


def pass_at_k(n: int, correct: int, k: int) -> float:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if n < k:
        raise ValueError(f"task has {n} samples, fewer than k={k}")
    if n - correct < k:
        return 1.0
    return 1.0 - math.prod((n - correct - i) / (n - i) for i in range(k))


def summarize(path: Path, k: int) -> dict[str, float | int]:
    grouped = grouped_results(path)
    estimates = [pass_at_k(len(samples), sum(samples), k) for samples in grouped.values()]
    return {
        "k": k,
        "tasks": len(grouped),
        "tasks_with_any_pass": sum(any(samples) for samples in grouped.values()),
        "pass_at_k": sum(estimates) / len(estimates),
    }


def select_frontier(traces: Path, tasks: Path, output: Path, k: int = 8) -> dict[str, int]:
    grouped = grouped_results(traces)
    wrong = {task_id: len(samples) for task_id, samples in grouped.items() if len(samples) != k}
    if wrong:
        preview = ", ".join(f"{task_id}={count}" for task_id, count in list(wrong.items())[:5])
        raise ValueError(f"frontier screening requires exactly {k} samples per task: {preview}")
    selected = {task_id for task_id, samples in grouped.items() if 0 < sum(samples) < k}
    rows: list[dict] = []
    with tasks.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            row = json.loads(line)
            case_id = str(row.get("case_id", row.get("task_id", line_no)))
            if case_id in selected:
                rows.append(row)
    found = {str(row.get("case_id", row.get("task_id"))) for row in rows}
    missing = selected - found
    if missing:
        raise ValueError(f"screened tasks are missing from {tasks}: {sorted(missing)[:5]}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows), encoding="utf-8")
    return {
        "screened": len(grouped),
        "frontier": len(rows),
        "all_failed": sum(not any(samples) for samples in grouped.values()),
        "all_passed": sum(all(samples) for samples in grouped.values()),
    }


def compare(sft: Path, rlvr: Path, k: int) -> dict[str, float | int]:
    left = grouped_results(sft)
    right = grouped_results(rlvr)
    if left.keys() != right.keys():
        raise ValueError("SFT and RLVR traces must contain the same task IDs")
    left_scores = {key: pass_at_k(len(value), sum(value), k) for key, value in left.items()}
    right_scores = {key: pass_at_k(len(value), sum(value), k) for key, value in right.items()}
    tasks = len(left_scores)
    sft_score = sum(left_scores.values()) / tasks
    rlvr_score = sum(right_scores.values()) / tasks
    return {
        "k": k,
        "tasks": tasks,
        "sft_pass_at_k": sft_score,
        "rlvr_pass_at_k": rlvr_score,
        "delta": rlvr_score - sft_score,
        "rlvr_wins": sum(right_scores[key] > left_scores[key] for key in left_scores),
        "sft_wins": sum(left_scores[key] > right_scores[key] for key in left_scores),
        "ties": sum(left_scores[key] == right_scores[key] for key in left_scores),
    }
