import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from data import prepare, recovery as recovery_generator
from data.validate import validate_sft
from glyph.program import run_hidden_tests


def _write_mbpp(path: Path, ids: list[int]) -> None:
    pq.write_table(
        pa.table(
            {
                "task_id": ids,
                "text": [f"Return {task_id}." for task_id in ids],
                "code": [f"def answer():\n    return {task_id}" for task_id in ids],
                "test_list": [[f"assert answer() == {task_id}"] for task_id in ids],
                "test_setup_code": ["" for _ in ids],
            }
        ),
        path,
    )


def _write_mbppplus(path: Path) -> None:
    pq.write_table(
        pa.table(
            {
                "task_id": [11, 602],
                "prompt": ["Return eleven.", "Overlaps training."],
                "code": ["def answer():\n    return 11", "def answer():\n    return 602"],
                "test": ["assert answer() == 11", "assert answer() == 602"],
            }
        ),
        path,
    )


def test_prepare_data_builds_disjoint_agentic_splits(tmp_path: Path, monkeypatch) -> None:
    paths = {
        "train": tmp_path / "train.parquet",
        "validation": tmp_path / "validation.parquet",
        "mbppplus": tmp_path / "plus.parquet",
    }
    _write_mbpp(paths["train"], [601, 602, 603, 604])
    _write_mbpp(paths["validation"], [511, 512])
    _write_mbppplus(paths["mbppplus"])
    monkeypatch.setattr(prepare, "download_source", lambda source, cache: paths[source.name])

    output = tmp_path / "data"
    output.mkdir()
    (output / "README.md").write_text("data documentation\n")
    manifest = prepare.prepare_data(
        output,
        tmp_path / "cache",
        sft_count=2,
        rl_count=2,
        recovery_one=1,
        recovery_two=0,
        seed=7,
    )
    counts = json.loads(manifest.read_text())["split"]
    assert counts == {
        "seed": 7,
        "sft": 2,
        "sft_recovery": 1,
        "sft_recovery_one": 1,
        "sft_recovery_two": 0,
        "rl_candidates": 2,
        "dev": 2,
        "test": 1,
        "test_rule": "MBPP+ task_id 11-510; disjoint from MBPP train and validation",
    }
    assert (output / "README.md").read_text() == "data documentation\n"
    sft_rows = [json.loads(line) for line in (tmp_path / "data/sft.jsonl").read_text().splitlines()]
    recovery = next(row for row in sft_rows if row["recovery_phases"] == 1)
    direct = next(row for row in sft_rows if row["recovery_phases"] == 0)
    assert [message["role"] for message in recovery["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant",
        "tool", "assistant", "tool", "assistant", "tool", "assistant",
    ]
    assert "status: failed" in recovery["messages"][7]["content"]
    assert "CALL apply_patch" in recovery["messages"][8]["content"]
    assert "CALL python_test" in recovery["messages"][10]["content"]
    assert "status: success" in recovery["messages"][11]["content"]
    assert [message["role"] for message in direct["messages"]] == [
        "system", "user", "assistant", "tool", "assistant", "tool", "assistant",
        "tool", "assistant",
    ]
    test_row = json.loads((tmp_path / "data/test.jsonl").read_text())
    assert test_row["task_id"] == 11
    assert test_row["test_code"] not in test_row["prompt"][-1]["content"]


def test_split_train_is_deterministic_and_disjoint() -> None:
    tasks = [prepare.MBPPTask(i, str(i), "pass\n", "pass\n", "mbpp") for i in range(10)]
    sft, rl = prepare.split_train(tasks, sft_count=6, rl_count=4, seed=42)
    again = prepare.split_train(tasks, sft_count=6, rl_count=4, seed=42)
    assert (sft, rl) == again
    assert {task.task_id for task in sft}.isdisjoint(task.task_id for task in rl)


def test_two_phase_recovery_fails_twice_then_passes(tmp_path: Path) -> None:
    code = "def score(x):\n    left = x + 1\n    right = x * 2\n    return left + right\n"
    tests = "assert score(1) == 4\nassert score(2) == 7\n"
    trace = recovery_generator.generate_recovery(code, tests, "score", phases=2)
    assert trace is not None
    assert len(trace.patches) == 2

    solution = tmp_path / "solution.py"
    solution.write_text(trace.initial_code)
    assert not run_hidden_tests(tmp_path, tests, 5).success
    for index, patch in enumerate(trace.patches):
        text = solution.read_text()
        assert text.count(patch.find) == 1
        solution.write_text(text.replace(patch.find, patch.replace, 1))
        assert run_hidden_tests(tmp_path, tests, 5).success is (index == 1)


def test_sft_validation_rejects_truncation_and_incomplete_final(tmp_path: Path) -> None:
    class Tokenizer:
        def encode(self, text: str, add_special_tokens: bool = False) -> list[str]:
            assert not add_special_tokens
            return text.split()

    path = tmp_path / "sft.jsonl"
    row = {
        "case_id": "case",
        "messages": [
            {"role": "system", "content": "system"},
            {"role": "user", "content": "task"},
            {"role": "assistant", "content": "FINAL: done"},
        ],
    }
    path.write_text(json.dumps(row) + "\n")
    summary = validate_sft(path, max_tokens=20, tokenizer=Tokenizer())
    assert summary["traces"] == 1

    with pytest.raises(ValueError, match="limit is 2"):
        validate_sft(path, max_tokens=2, tokenizer=Tokenizer())

    row["messages"][-1]["content"] = "done"
    path.write_text(json.dumps(row) + "\n")
    with pytest.raises(ValueError, match="terminal FINAL"):
        validate_sft(path, max_tokens=20, tokenizer=Tokenizer())
