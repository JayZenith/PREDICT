import json
from pathlib import Path

import pytest

from data import prepare, recovery as recovery_generator
from data.validate import validate_sft
from glyph.program import PASS, run_hidden_tests


def test_official_mbpp_split_contract_is_fixed() -> None:
    assert list(prepare.TRAIN_IDS) == list(range(601, 975))
    assert list(prepare.VALIDATION_IDS) == list(range(511, 601))
    assert list(prepare.TEST_IDS) == list(range(11, 511))
    assert prepare.DIRECT_COUNT == 250
    assert prepare.RECOVERY_COUNT == 124
    assert prepare.SEED == 42
    assert set(prepare.SOURCES) == {"train", "validation", "test"}
    assert all(source.revision == prepare.MBPP_REVISION for source in prepare.SOURCES.values())
    assert all(len(source.sha256) == 64 for source in prepare.SOURCES.values())


def test_recovery_is_real_one_step_and_patch_is_unambiguous(tmp_path: Path) -> None:
    code = (
        "def is_key_present(d, x):\n"
        "  if x in d:\n"
        "    return True\n"
        "  else:\n"
        "     return False\n"
    )
    tests = (
        "assert is_key_present({'a': 1}, 'a') is True\n"
        "assert is_key_present({'a': 1}, 'b') is False\n"
    )
    trace = recovery_generator.generate_recovery(code, tests, "key")
    assert trace is not None
    assert trace.initial_code.count(trace.patch.find) == 1

    solution = tmp_path / "solution.py"
    solution.write_text(trace.initial_code)
    assert not run_hidden_tests(tmp_path, tests, 5).success
    solution.write_text(
        trace.initial_code.replace(trace.patch.find, trace.patch.replace, 1)
    )
    assert run_hidden_tests(tmp_path, tests, 5).outcome == PASS


def test_arm_sft_rows_are_task_and_candidate_matched() -> None:
    task = prepare.MBPPTask(
        601,
        "Return whether n is even.",
        "def is_even(n):\n    return n % 2 == 0\n",
        "assert is_even(2)\nassert not is_even(3)\n",
        "train",
    )
    recovery = recovery_generator.generate_recovery(
        task.code, task.test_code, task.case_id
    )
    assert recovery is not None
    arm_a = prepare.sft_row(task, "a", recovery)
    arm_b = prepare.sft_row(task, "b", recovery)
    for key in (
        "candidate_code_sha256",
        "candidate_outcome",
        "final_code_sha256",
        "final_outcome",
        "matched_key",
        "task_id",
        "trace_type",
    ):
        assert arm_a[key] == arm_b[key]
    assert "status: failed" in "\n".join(
        message["content"] for message in arm_a["messages"]
    )
    arm_b_text = "\n".join(message["content"] for message in arm_b["messages"])
    assert f"<PREDICTION>{recovery.outcome}</PREDICTION>" in arm_b_text
    assert "<DECISION>REVISE</DECISION>" in arm_b_text
    assert "status: failed" not in arm_b_text
    assert arm_a["messages"][-1]["content"].startswith("FINAL:")
    assert arm_b["messages"][-1]["content"].startswith("FINAL:")


def test_sft_validation_rejects_truncation_and_incomplete_final(
    tmp_path: Path,
) -> None:
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
