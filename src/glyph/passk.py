"""Task-level correctness and PREDICT experiment reporting."""

from __future__ import annotations

import json
import math
from collections import Counter, defaultdict
from pathlib import Path

from .program import OUTCOME_CLASSES, PASS


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


def _rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: trace must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"no traces in {path}")
    return rows


def _state(row: dict) -> tuple[list[dict], dict[str, dict], list[dict]]:
    state = ((row.get("info") or {}).get("glyph") or row.get("glyph") or {})
    return (
        list(state.get("calls") or []),
        dict(state.get("results") or {}),
        list(state.get("prediction_targets") or []),
    )


def _outcome(result: dict) -> str:
    outcome = result.get("outcome")
    if outcome in OUTCOME_CLASSES:
        return str(outcome)
    return PASS if result.get("success") else "OTHER"


def _token_count(row: dict) -> int | None:
    direct = row.get("num_total_tokens")
    if isinstance(direct, int):
        return direct
    nodes = row.get("nodes") or []
    if nodes and all(isinstance(node.get("token_ids"), list) for node in nodes):
        token_ids = sum(len(node["token_ids"]) for node in nodes)
        if token_ids:
            return token_ids
    usage_tokens = sum(
        int((node.get("usage") or {}).get("prompt_tokens") or 0)
        + int((node.get("usage") or {}).get("completion_tokens") or 0)
        for node in nodes
    )
    if usage_tokens:
        return usage_tokens
    return None


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


def report(path: Path) -> dict:
    rows = _rows(path)
    tasks = {_task_id(row, index) for index, row in enumerate(rows, 1)}
    if len(tasks) != len(rows):
        raise ValueError("PREDICT report requires exactly one pass@1 trace per task")

    passed = 0
    first_correct = 0
    executed_failure_tasks = 0
    executed_recoveries = 0
    visible_tests = 0
    visible_tests_solved = 0
    visible_tools = 0
    revisions = 0
    predictions: list[tuple[str, str]] = []
    bad = bad_rejected = good = good_rejected = 0
    token_counts: list[int] = []

    for row in rows:
        solved = _passes(row)
        passed += solved
        calls, results, targets = _state(row)
        tests = [
            _outcome(results.get(call.get("id")) or {})
            for call in calls
            if call.get("tool") == "python_test"
        ]
        if targets:
            first_correct += targets[0].get("actual") == PASS
        elif tests:
            first_correct += tests[0] == PASS
        first_failure = next(
            (
                index
                for index, outcome in enumerate(tests)
                if outcome != PASS
            ),
            None,
        )
        had_executed_failure = first_failure is not None
        visibly_recovered = bool(
            first_failure is not None
            and any(outcome == PASS for outcome in tests[first_failure + 1 :])
        )
        executed_failure_tasks += had_executed_failure
        executed_recoveries += bool(solved and visibly_recovered)
        visible_tests += len(tests)
        visible_tests_solved += len(tests) if solved else 0
        visible_tools += len(calls)
        revisions += max(
            0, sum(call.get("tool") == "apply_patch" for call in calls) - 1
        )
        for target in targets:
            predicted, actual = target.get("sampled_prediction"), target.get("actual")
            if actual in OUTCOME_CLASSES:
                predictions.append(
                    (
                        str(predicted)
                        if predicted in OUTCOME_CLASSES
                        else "INVALID",
                        str(actual),
                    )
                )
            if actual == PASS:
                good += 1
                good_rejected += target.get("decision") == "REVISE"
            else:
                bad += 1
                bad_rejected += target.get("decision") == "REVISE"
        if (count := _token_count(row)) is not None:
            token_counts.append(count)

    output: dict[str, object] = {
        "tasks": len(rows),
        "final_pass_at_1": passed / len(rows),
        "first_patch_success": first_correct / len(rows),
        "executed_failure_recovery_rate": (
            executed_recoveries / executed_failure_tasks
            if executed_failure_tasks
            else 0.0
        ),
        "tasks_with_executed_failure": executed_failure_tasks,
        "visible_python_tests_per_solved_task": (
            visible_tests_solved / passed if passed else 0.0
        ),
        "average_visible_python_tests": visible_tests / len(rows),
        "average_visible_tool_calls": visible_tools / len(rows),
        "average_revisions": revisions / len(rows),
    }
    if token_counts:
        output["average_total_tokens"] = sum(token_counts) / len(token_counts)
    if predictions:
        confusion = Counter(predictions)
        per_class: dict[str, float] = {}
        for label in sorted(OUTCOME_CLASSES):
            true_positive = confusion[(label, label)]
            predicted_total = sum(
                count for (predicted, _), count in confusion.items() if predicted == label
            )
            actual_total = sum(
                count for (_, actual), count in confusion.items() if actual == label
            )
            precision = true_positive / predicted_total if predicted_total else 0.0
            recall = true_positive / actual_total if actual_total else 0.0
            per_class[label] = (
                2 * precision * recall / (precision + recall)
                if precision + recall
                else 0.0
            )
        output.update(
            {
                "prediction_count": len(predictions),
                "prediction_accuracy": (
                    sum(predicted == actual for predicted, actual in predictions)
                    / len(predictions)
                ),
                "prediction_macro_f1": sum(per_class.values()) / len(per_class),
                "prediction_f1_by_class": per_class,
                "bad_patch_rejection_rate": bad_rejected / bad if bad else 0.0,
                "good_patch_unnecessary_rejection_rate": (
                    good_rejected / good if good else 0.0
                ),
            }
        )
    return output


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
