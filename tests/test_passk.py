import json
from pathlib import Path

import pytest

from glyph.passk import compare, pass_at_k, report, summarize


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


def test_predict_report_aggregates_decisions_and_efficiency(tmp_path: Path) -> None:
    traces = tmp_path / "traces.jsonl"
    rows = []
    for case_id, passed, actual, decision in (
        ("a", True, "PASS", "KEEP"),
        ("b", True, "ASSERTION_FAILURE", "REVISE"),
    ):
        row = _trace(case_id, passed)
        row["info"] = {
            "glyph": {
                "calls": [
                    {"tool": "apply_patch", "id": "c1"},
                    {"tool": "python_test", "id": "c2"},
                ],
                "results": {
                    "c2": {
                        "success": actual == "PASS",
                        "outcome": actual,
                    }
                },
                "prediction_targets": [
                    {
                        "sampled_prediction": actual,
                        "actual": actual,
                        "decision": decision,
                    }
                ],
            }
        }
        rows.append(row)
    _write(traces, rows)
    result = report(traces)
    assert result["final_pass_at_1"] == 1.0
    assert result["first_patch_success"] == 0.5
    assert result["prediction_accuracy"] == 1.0
    assert result["bad_patch_rejection_rate"] == 1.0
    assert result["good_patch_unnecessary_rejection_rate"] == 0.0


def test_compare_requires_paired_tasks(tmp_path: Path) -> None:
    sft, rl = tmp_path / "sft.jsonl", tmp_path / "rl.jsonl"
    _write(sft, [_trace("a", False), _trace("a", False)])
    _write(rl, [_trace("a", True), _trace("a", False)])
    report = compare(sft, rl, 2)
    assert report["delta"] == 1.0
    assert report["rlvr_wins"] == 1
