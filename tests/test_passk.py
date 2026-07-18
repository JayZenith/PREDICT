import json
from pathlib import Path

import pytest

from glyph.passk import compare, pass_at_k, select_frontier, summarize


def _trace(case_id: str, passed: bool) -> dict:
    return {"task": {"data": {"case_id": case_id}}, "metrics": {"passed": int(passed)}}


def _write(path: Path, rows: list[dict]) -> None:
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def test_pass_at_k_and_summary(tmp_path: Path) -> None:
    assert pass_at_k(4, 1, 2) == pytest.approx(0.5)
    traces = tmp_path / "traces.jsonl"
    _write(traces, [_trace("a", value) for value in (0, 1)] + [_trace("b", 0)] * 2)
    assert summarize(traces, 2) == {
        "k": 2,
        "tasks": 2,
        "tasks_with_any_pass": 1,
        "pass_at_k": 0.5,
    }


def test_frontier_keeps_only_mixed_pass8_groups(tmp_path: Path) -> None:
    traces = tmp_path / "screen.jsonl"
    _write(
        traces,
        [_trace("mixed", value) for value in ([1] + [0] * 7)]
        + [_trace("easy", 1)] * 8
        + [_trace("hard", 0)] * 8,
    )
    tasks = tmp_path / "tasks.jsonl"
    _write(tasks, [{"case_id": case_id} for case_id in ("mixed", "easy", "hard")])
    output = tmp_path / "rl.jsonl"
    assert select_frontier(traces, tasks, output) == {
        "screened": 3,
        "frontier": 1,
        "all_failed": 1,
        "all_passed": 1,
    }
    assert json.loads(output.read_text())["case_id"] == "mixed"


def test_compare_requires_paired_tasks(tmp_path: Path) -> None:
    sft, rl = tmp_path / "sft.jsonl", tmp_path / "rl.jsonl"
    _write(sft, [_trace("a", False), _trace("a", False)])
    _write(rl, [_trace("a", True), _trace("a", False)])
    report = compare(sft, rl, 2)
    assert report["delta"] == 1.0
    assert report["rlvr_wins"] == 1
