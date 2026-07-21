"""Exhaustively validate generated PREDICT datasets before training."""

from __future__ import annotations

import argparse
import hashlib
import json
import tempfile
from collections import Counter
from pathlib import Path
from typing import Any

from glyph.chat import ARM_A_SYSTEM_PROMPT, ARM_B_SYSTEM_PROMPT, render_messages
from glyph.program import (
    PASS,
    Call,
    parse_calls,
    parse_prediction_decision,
    run_hidden_tests,
)

from .prepare import (
    DEEP_SHADOW_RECOVERY_COUNT,
    DEEP_VISIBLE_RECOVERY_COUNT,
    DIRECT_COUNT,
    DATA_ARTIFACTS,
    PLACEHOLDER,
    RECOVERY_COUNT,
    RL_TRAIN_COUNT,
    SHADOW_RECOVERY_COUNT,
    SOURCES,
    SFT_COUNT,
    TEST_IDS,
    VALIDATION_COUNT,
    VISIBLE_RECOVERY_COUNT,
)


DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"
SFT_MAX_TOKENS = 1280


def _rows(path: Path) -> list[dict]:
    rows: list[dict] = []
    with path.open(encoding="utf-8") as source:
        for line_no, line in enumerate(source, 1):
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc.msg}") from exc
            if not isinstance(row, dict):
                raise ValueError(f"{path}:{line_no}: row must be an object")
            rows.append(row)
    return rows


def _tokenizer(model: str, tokenizer: Any | None) -> Any:
    if tokenizer is not None:
        return tokenizer
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model)


def validate_sft(
    path: Path,
    *,
    max_tokens: int = SFT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
    tokenizer: Any | None = None,
) -> dict[str, int | str]:
    if max_tokens <= 0:
        raise ValueError("max_tokens must be positive")
    tokenizer = _tokenizer(model, tokenizer)
    longest = 0
    longest_case = ""
    count = 0
    for line_no, row in enumerate(_rows(path), 1):
        messages = row.get("messages") or []
        case_id = str(row.get("case_id", line_no))
        if (
            not messages
            or messages[-1].get("role") != "assistant"
            or not str(messages[-1].get("content", "")).strip().startswith("FINAL:")
        ):
            raise ValueError(f"{path}:{line_no} ({case_id}) lacks a terminal FINAL:")
        tokens = len(
            tokenizer.encode(
                render_messages(messages),
                add_special_tokens=False,
            )
        )
        if tokens > max_tokens:
            raise ValueError(
                f"{path}:{line_no} ({case_id}) has {tokens} tokens; "
                f"limit is {max_tokens}"
            )
        if tokens > longest:
            longest = tokens
            longest_case = case_id
        count += 1
    if not count:
        raise ValueError(f"{path} contains no SFT traces")
    return {
        "traces": count,
        "max_tokens": longest,
        "longest_case": longest_case,
    }


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _apply(call: Call, code: str) -> str:
    find = call.params.get("find")
    replace = call.params.get("replace")
    if find is None or replace is None:
        raise ValueError("apply_patch requires find and replace")
    if code.count(find) != 1:
        raise ValueError(
            f"apply_patch find must occur exactly once; found {code.count(find)}"
        )
    return code.replace(find, replace, 1)


def _verify_trace(
    row: dict, task: dict, project: Path, timeout: int
) -> Counter[str]:
    arm = row["arm"]
    code = PLACEHOLDER
    seen_ids: set[str] = set()
    visible_outcomes: list[str] = []
    predicted_outcomes: list[str] = []
    verified_outcomes: list[str] = []
    decisions: list[str] = []
    first_candidate_code: str | None = None
    solution = project / "solution.py"

    for message in row["messages"]:
        if message.get("role") != "assistant":
            continue
        content = str(message.get("content", ""))
        prediction, decision, prediction_errors = parse_prediction_decision(content)
        has_prediction = "<PREDICTION>" in content or "<DECISION>" in content
        if has_prediction:
            if arm != "b" or prediction_errors:
                raise ValueError(
                    f"{row['case_id']}: invalid prediction turn: {prediction_errors}"
                )
            solution.write_text(code, encoding="utf-8")
            actual = run_hidden_tests(project, task["test_code"], timeout).outcome
            predicted_outcomes.append(prediction or "")
            verified_outcomes.append(actual or "")
            decisions.append(decision or "")

        calls, errors = parse_calls(content, seen_ids)
        if errors:
            raise ValueError(f"{row['case_id']}: {errors}")
        for call in calls:
            seen_ids.add(call.id)
            if call.tool == "read_file":
                continue
            if call.tool == "apply_patch":
                try:
                    code = _apply(call, code)
                except ValueError as exc:
                    raise ValueError(f"{row['case_id']}: {exc}") from exc
                if first_candidate_code is None:
                    first_candidate_code = code
                continue
            if call.tool != "python_test":
                raise ValueError(f"{row['case_id']}: unknown tool {call.tool}")
            solution.write_text(code, encoding="utf-8")
            result = run_hidden_tests(project, task["test_code"], timeout)
            visible_outcomes.append(result.outcome or "")

    if first_candidate_code is None:
        raise ValueError(f"{row['case_id']}: trace never applied a candidate")
    if _sha256(first_candidate_code) != row["candidate_code_sha256"]:
        raise ValueError(f"{row['case_id']}: first candidate hash drifted")
    if _sha256(code) != row["final_code_sha256"]:
        raise ValueError(f"{row['case_id']}: final code hash drifted")
    if row["final_outcome"] != PASS or not visible_outcomes or visible_outcomes[-1] != PASS:
        raise ValueError(f"{row['case_id']}: final visible test did not pass")

    expected_first = row["candidate_outcome"]
    expected_second = row.get("second_candidate_outcome")
    mode = row.get("recovery_mode")
    is_deep = mode in ("deep_shadow", "deep_visible")
    if arm == "a":
        if row["trace_type"] == "direct":
            expected = [PASS]
        elif is_deep:
            expected = [expected_first, expected_second, PASS]
        else:
            expected = [expected_first, PASS]
        if visible_outcomes != expected:
            raise ValueError(f"{row['case_id']}: Arm A recovery sequence drifted")
    else:
        if row["trace_type"] == "direct":
            expected = {
                "predicted": [PASS],
                "verified": [PASS],
                "decisions": ["KEEP"],
                "visible": [PASS],
            }
        elif mode == "shadow":
            expected = {
                "predicted": [expected_first, PASS],
                "verified": [expected_first, PASS],
                "decisions": ["REVISE", "KEEP"],
                "visible": [PASS],
            }
        elif mode == "visible":
            # The first prediction is an intentional honest mistake: the model
            # keeps a bad candidate it expects to pass, then recovers from the
            # visible failure. RL prediction CE targets the verified label.
            expected = {
                "predicted": [PASS, PASS],
                "verified": [expected_first, PASS],
                "decisions": ["KEEP", "KEEP"],
                "visible": [expected_first, PASS],
            }
        elif mode == "deep_shadow":
            expected = {
                "predicted": [expected_first, expected_second, PASS],
                "verified": [expected_first, expected_second, PASS],
                "decisions": ["REVISE", "REVISE", "KEEP"],
                "visible": [PASS],
            }
        elif mode == "deep_visible":
            # Both early predictions are honest mistakes; each is corrected
            # only after a genuine visible failure, exactly like the shallow
            # visible mode but with one more recovery cycle.
            expected = {
                "predicted": [PASS, PASS, PASS],
                "verified": [expected_first, expected_second, PASS],
                "decisions": ["KEEP", "KEEP", "KEEP"],
                "visible": [expected_first, expected_second, PASS],
            }
        else:
            raise ValueError(f"{row['case_id']}: unknown recovery_mode {mode!r}")
        if predicted_outcomes != expected["predicted"]:
            raise ValueError(f"{row['case_id']}: Arm B predicted labels drifted")
        if verified_outcomes != expected["verified"]:
            raise ValueError(f"{row['case_id']}: Arm B verified outcomes drifted")
        if decisions != expected["decisions"]:
            raise ValueError(f"{row['case_id']}: Arm B decision sequence drifted")
        if visible_outcomes != expected["visible"]:
            raise ValueError(f"{row['case_id']}: Arm B recovery sequence drifted")

    return Counter(verified_outcomes)


def validate_prepared(
    data_dir: Path,
    *,
    max_tokens: int = SFT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
    tokenizer: Any | None = None,
    timeout: int = 5,
) -> dict:
    data_dir = data_dir.expanduser().resolve()
    tokenizer = _tokenizer(model, tokenizer)
    sft = {
        arm: _rows(data_dir / "sft" / f"arm_{arm}" / "train.jsonl")
        for arm in ("a", "b")
    }
    tasks = {
        (arm, split): _rows(data_dir / f"arm_{arm}_{split}.jsonl")
        for arm in ("a", "b")
        for split in ("train", "validation", "test")
    }
    manifest = json.loads((data_dir / "manifest.json").read_text())
    split_manifest = manifest.get("split") or {}

    expected_counts = {
        "train": RL_TRAIN_COUNT,
        "validation": VALIDATION_COUNT,
        "test": 500,
    }
    expected_ids = {
        "train": set(split_manifest.get("rl_train_task_ids") or []),
        "validation": set(split_manifest.get("validation_task_ids") or []),
        "test": set(TEST_IDS),
    }
    sft_ids = set(split_manifest.get("sft_task_ids") or [])
    if len(sft_ids) != SFT_COUNT:
        raise ValueError("manifest SFT task IDs are missing or wrong length")
    if len(expected_ids["train"]) != RL_TRAIN_COUNT:
        raise ValueError("manifest RL train task IDs are missing or wrong length")
    if len(expected_ids["validation"]) != VALIDATION_COUNT:
        raise ValueError("manifest validation task IDs are missing or wrong length")
    if sft_ids & expected_ids["train"]:
        raise ValueError("SFT and RL train task IDs overlap")
    if sft_ids & expected_ids["validation"]:
        raise ValueError("SFT and validation task IDs overlap")
    if (sft_ids | expected_ids["train"] | expected_ids["validation"]) & set(
        TEST_IDS
    ):
        raise ValueError("non-test task IDs overlap the 500-task test split")
    for arm in ("a", "b"):
        if len(sft[arm]) != SFT_COUNT:
            raise ValueError(f"Arm {arm.upper()} SFT must contain {SFT_COUNT} traces")
        if {row.get("task_id") for row in sft[arm]} != sft_ids:
            raise ValueError(f"Arm {arm.upper()} SFT task IDs differ from manifest")
        for split, expected in expected_counts.items():
            rows = tasks[(arm, split)]
            if len(rows) != expected:
                raise ValueError(
                    f"Arm {arm.upper()} {split} must contain {expected} tasks"
                )
            if {row.get("task_id") for row in rows} != expected_ids[split]:
                raise ValueError(
                    f"Arm {arm.upper()} {split} task IDs differ from manifest"
                )
            for row in rows:
                if row.get("arm") != arm or row.get("split") != split:
                    raise ValueError(
                        f"{row.get('case_id')}: arm or split metadata drifted"
                    )
                system = (
                    ARM_A_SYSTEM_PROMPT if arm == "a" else ARM_B_SYSTEM_PROMPT
                )
                if row["prompt"][0] != {"role": "system", "content": system}:
                    raise ValueError(
                        f"{row['case_id']}: wrong Arm {arm.upper()} system prompt"
                    )
                blueprint = data_dir / row["blueprint_root"]
                if (blueprint / "solution.py").read_text() != PLACEHOLDER:
                    raise ValueError(
                        f"{row['case_id']}: blueprint is not the blank solution"
                    )

    for split in expected_counts:
        left = [row["case_id"] for row in tasks[("a", split)]]
        right = [row["case_id"] for row in tasks[("b", split)]]
        if left != right:
            raise ValueError(f"Arm task ordering differs for {split}")
        for arm_a, arm_b in zip(
            tasks[("a", split)], tasks[("b", split)], strict=True
        ):
            for key in (
                "blueprint_root",
                "case_id",
                "source",
                "split",
                "task_id",
                "test_code",
                "trace_prefix",
            ):
                if arm_a[key] != arm_b[key]:
                    raise ValueError(
                        f"{arm_a['case_id']}: task arms differ on {key}"
                    )
            if arm_a["prompt"][1] != arm_b["prompt"][1]:
                raise ValueError(
                    f"{arm_a['case_id']}: task arms have different user prompts"
                )
        for row in [*tasks[("a", split)], *tasks[("b", split)]]:
            if row["test_code"].rstrip() not in row["prompt"][-1]["content"]:
                raise ValueError(f"{row['case_id']}: tests missing from prompt")

    split_ids = {
        split: {row["task_id"] for row in tasks[("a", split)]}
        for split in expected_counts
    }
    if (
        split_ids["train"] & split_ids["validation"]
        or split_ids["train"] & split_ids["test"]
        or split_ids["validation"] & split_ids["test"]
    ):
        raise ValueError("MBPP task IDs overlap across splits")

    by_arm = {
        arm: {row["case_id"]: row for row in sft[arm]} for arm in ("a", "b")
    }
    if by_arm["a"].keys() != by_arm["b"].keys():
        raise ValueError("SFT arms contain different task IDs")
    for case_id in by_arm["a"]:
        left, right = by_arm["a"][case_id], by_arm["b"][case_id]
        if left.get("arm") != "a" or right.get("arm") != "b":
            raise ValueError(f"{case_id}: SFT arm metadata drifted")
        if left["messages"][0] != {
            "role": "system",
            "content": ARM_A_SYSTEM_PROMPT,
        } or right["messages"][0] != {
            "role": "system",
            "content": ARM_B_SYSTEM_PROMPT,
        }:
            raise ValueError(f"{case_id}: SFT system prompt drifted")
        for key in (
            "candidate_code_sha256",
            "candidate_outcome",
            "second_candidate_outcome",
            "final_code_sha256",
            "final_outcome",
            "matched_key",
            "recovery_mode",
            "task_id",
            "test_code",
            "trace_type",
        ):
            if left[key] != right[key]:
                raise ValueError(f"{case_id}: arms differ on matched field {key}")

    trace_counts = Counter(row["trace_type"] for row in sft["a"])
    if trace_counts != {"direct": DIRECT_COUNT, "recovery": RECOVERY_COUNT}:
        raise ValueError(f"SFT composition drifted: {dict(trace_counts)}")
    for arm in ("a", "b"):
        mode_counts = Counter(
            row["recovery_mode"]
            for row in sft[arm]
            if row["trace_type"] == "recovery"
        )
        if mode_counts != {
            "shadow": SHADOW_RECOVERY_COUNT,
            "visible": VISIBLE_RECOVERY_COUNT,
            "deep_shadow": DEEP_SHADOW_RECOVERY_COUNT,
            "deep_visible": DEEP_VISIBLE_RECOVERY_COUNT,
        }:
            raise ValueError(
                f"Arm {arm.upper()} recovery mode split drifted: {dict(mode_counts)}"
            )

    assignments = json.loads((data_dir / "assignments.json").read_text())
    expected_assignments = {
        row["case_id"]: (
            row["trace_type"],
            row["candidate_outcome"],
            row["recovery_mode"],
        )
        for row in sft["a"]
    }
    actual_assignments = {
        row["case_id"]: (
            row["trace_type"],
            row["candidate_outcome"],
            row.get("recovery_mode"),
        )
        for row in assignments
    }
    if actual_assignments != expected_assignments:
        raise ValueError("assignments.json differs from the generated SFT data")

    for relative in DATA_ARTIFACTS:
        path = data_dir / relative
        recorded = manifest["artifacts"].get(relative) or {}
        if (
            recorded.get("sha256") != _file_sha256(path)
            or recorded.get("bytes") != path.stat().st_size
        ):
            raise ValueError(f"generated artifact proof drifted for {relative}")
    for name, source in SOURCES.items():
        recorded = manifest["sources"].get(name) or {}
        if (
            recorded.get("revision") != source.revision
            or recorded.get("sha256") != source.sha256
            or recorded.get("license") != source.license
        ):
            raise ValueError(f"manifest source proof drifted for {name}")

    token_summaries = {
        arm: validate_sft(
            data_dir / "sft" / f"arm_{arm}" / "train.jsonl",
            max_tokens=max_tokens,
            model=model,
            tokenizer=tokenizer,
        )
        for arm in ("a", "b")
    }
    prediction_outcomes: Counter[str] = Counter()
    with tempfile.TemporaryDirectory(prefix="predict-dataset-audit-") as temporary:
        project = Path(temporary)
        for arm in ("a", "b"):
            for row in sft[arm]:
                prediction_outcomes.update(
                    _verify_trace(
                        row,
                        row,
                        project,
                        timeout,
                    )
                )

    return {
        "sft": token_summaries,
        "composition": dict(sorted(trace_counts.items())),
        "prediction_targets": dict(sorted(prediction_outcomes.items())),
        "train": RL_TRAIN_COUNT,
        "validation": VALIDATION_COUNT,
        "test": 500,
        "verified_sft_traces": SFT_COUNT * 2,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("data_dir", type=Path, nargs="?", default=Path("data"))
    parser.add_argument("--max-tokens", type=int, default=SFT_MAX_TOKENS)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    print(
        json.dumps(
            validate_prepared(
                args.data_dir,
                max_tokens=args.max_tokens,
                model=args.model,
                timeout=args.timeout,
            ),
            indent=2,
            sort_keys=True,
        )
    )


if __name__ == "__main__":
    main()
