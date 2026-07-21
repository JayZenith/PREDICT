"""Download, verify, and format the official MBPP experiment splits."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import tempfile
import urllib.request
from collections import Counter
from dataclasses import dataclass, replace
from pathlib import Path

import pyarrow.parquet as pq

from glyph.chat import ARM_A_SYSTEM_PROMPT, ARM_B_SYSTEM_PROMPT, render_messages
from glyph.program import (
    ASSERTION_FAILURE,
    OTHER,
    PASS,
    RUNTIME_ERROR,
    SYNTAX_ERROR,
    TIMEOUT,
    run_hidden_tests,
)

from .recovery import (
    RecoveryTrace,
    TwoStepRecoveryTrace,
    generate_recovery,
    generate_two_step_recovery,
)


PLACEHOLDER = "# Write your function here.\n"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
OFFICIAL_TRAIN_IDS = range(601, 975)
OFFICIAL_VALIDATION_IDS = range(511, 601)
TEST_IDS = range(11, 511)
VALIDATION_TO_TRAIN_COUNT = 50
VALIDATION_COUNT = 40
SFT_COUNT = 212
RL_TRAIN_COUNT = 212
RECOVERY_COUNT = 70
SHADOW_RECOVERY_COUNT = 25
VISIBLE_RECOVERY_COUNT = 25
DEEP_SHADOW_RECOVERY_COUNT = 10
DEEP_VISIBLE_RECOVERY_COUNT = 10
DIRECT_COUNT = 142
SEED = 42
SFT_MAX_TOKENS = 1280
DEFAULT_MODEL = "Qwen/Qwen3-4B-Base"
DATA_ARTIFACTS = (
    "sft/arm_a/train.jsonl",
    "sft/arm_b/train.jsonl",
    "arm_a_train.jsonl",
    "arm_a_validation.jsonl",
    "arm_a_test.jsonl",
    "arm_b_train.jsonl",
    "arm_b_validation.jsonl",
    "arm_b_test.jsonl",
    "assignments.json",
)


@dataclass(frozen=True)
class SourceFile:
    name: str
    path: str
    sha256: str
    repository: str = "google-research-datasets/mbpp"
    revision: str = MBPP_REVISION
    license: str = "CC-BY-4.0"

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/"
            f"{self.revision}/{self.path}"
        )


SOURCES = {
    "train": SourceFile(
        "train",
        "full/train-00000-of-00001.parquet",
        "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae",
    ),
    "validation": SourceFile(
        "validation",
        "full/validation-00000-of-00001.parquet",
        "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80",
    ),
    "test": SourceFile(
        "test",
        "full/test-00000-of-00001.parquet",
        "566fd53060ffba5766dace1d1e2f4c38906781526de222b0dfbdbc325b696c77",
    ),
}


@dataclass(frozen=True)
class MBPPTask:
    task_id: int
    prompt: str
    code: str
    test_code: str
    split: str

    @property
    def case_id(self) -> str:
        return f"mbpp_{self.task_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _text_sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def download_source(source: SourceFile, cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{source.name}.parquet"
    if destination.exists() and _sha256(destination) == source.sha256:
        return destination
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    request = urllib.request.Request(
        source.url, headers={"User-Agent": "PREDICT/0.1"}
    )
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open(
        "wb"
    ) as output:
        shutil.copyfileobj(response, output)
    if _sha256(temporary) != source.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{source.name} failed its pinned SHA-256 check")
    os.replace(temporary, destination)
    return destination


def load_mbpp(path: Path, split: str) -> list[MBPPTask]:
    table = pq.read_table(
        path,
        columns=["task_id", "text", "code", "test_list", "test_setup_code"],
    )
    tasks: list[MBPPTask] = []
    for row in table.to_pylist():
        task_id = int(row["task_id"])
        prompt = str(row["text"] or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        code = str(row["code"] or "").replace("\r\n", "\n").replace("\r", "\n").strip()
        tests = [
            str(test).replace("\r\n", "\n").replace("\r", "\n").strip()
            for test in row["test_list"] or []
            if str(test).strip()
        ]
        setup = (
            str(row["test_setup_code"] or "")
            .replace("\r\n", "\n")
            .replace("\r", "\n")
            .strip()
        )
        if not prompt or not code or not tests:
            raise ValueError(
                f"{split} task {task_id} is missing prompt, code, or tests"
            )
        test_code = "\n".join(part for part in (setup, *tests) if part) + "\n"
        tasks.append(MBPPTask(task_id, prompt, code + "\n", test_code, split))
    return tasks


def _require_ids(tasks: list[MBPPTask], expected: range, split: str) -> None:
    actual = {task.task_id for task in tasks}
    wanted = set(expected)
    if actual != wanted:
        raise RuntimeError(
            f"{split} IDs differ from official MBPP: "
            f"missing={sorted(wanted - actual)[:5]} extra={sorted(actual - wanted)[:5]}"
        )


def _assignment_key(task: MBPPTask, seed: int) -> str:
    return hashlib.sha256(
        f"{seed}\0{task.task_id}\0{task.prompt}".encode()
    ).hexdigest()


def _with_split(task: MBPPTask, split: str) -> MBPPTask:
    return replace(task, split=split)


def _split_experiment_tasks(
    official_train: list[MBPPTask],
    official_validation: list[MBPPTask],
    seed: int,
) -> tuple[list[MBPPTask], list[MBPPTask], list[MBPPTask]]:
    ordered_validation = sorted(
        official_validation,
        key=lambda task: (_assignment_key(task, seed), task.task_id),
    )
    validation_to_train = ordered_validation[:VALIDATION_TO_TRAIN_COUNT]
    heldout_validation = ordered_validation[VALIDATION_TO_TRAIN_COUNT:]
    if len(heldout_validation) != VALIDATION_COUNT:
        raise RuntimeError("validation split size drifted")

    training_pool = [*official_train, *validation_to_train]
    ordered_pool = sorted(
        training_pool,
        key=lambda task: (_assignment_key(task, seed), task.task_id),
    )
    sft_tasks = ordered_pool[:SFT_COUNT]
    rl_train_tasks = ordered_pool[SFT_COUNT : SFT_COUNT + RL_TRAIN_COUNT]
    if len(sft_tasks) != SFT_COUNT or len(rl_train_tasks) != RL_TRAIN_COUNT:
        raise RuntimeError("SFT/RL split size drifted")
    if {task.task_id for task in sft_tasks} & {
        task.task_id for task in rl_train_tasks
    }:
        raise RuntimeError("SFT and RL train task IDs overlap")
    return sft_tasks, rl_train_tasks, heldout_validation


def task_prompt(
    task: MBPPTask, trace_prefix: str, arm: str
) -> list[dict[str, str]]:
    system = ARM_A_SYSTEM_PROMPT if arm == "a" else ARM_B_SYSTEM_PROMPT
    user = (
        "Implement the requested Python function in solution.py. Run the tests.\n\n"
        f"{task.prompt}\n\n"
        "Your code must pass these tests:\n"
        f"{task.test_code.rstrip()}\n\n"
        f"The project is at {trace_prefix}."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
    ]


def task_row(task: MBPPTask, arm: str) -> dict:
    blueprint_root = f"blueprints/{task.case_id}"
    trace_prefix = f"data/{blueprint_root}"
    return {
        "arm": arm,
        "blueprint_root": blueprint_root,
        "case_id": task.case_id,
        "prompt": task_prompt(task, trace_prefix, arm),
        "source": "mbpp",
        "split": task.split,
        "task_id": task.task_id,
        "test_code": task.test_code,
        "trace_prefix": trace_prefix,
    }


def _call(tool: str, payload: dict[str, str]) -> str:
    return f"CALL {tool} " + json.dumps(payload, separators=(",", ":"))


def _failed_result(call_id: str, outcome: str) -> str:
    detail = {
        ASSERTION_FAILURE: "tests failed",
        RUNTIME_ERROR: "generated solution raised a runtime error",
        SYNTAX_ERROR: "generated solution has a syntax error",
        TIMEOUT: "tests timed out",
        OTHER: "tests failed",
    }[outcome]
    return f"RESULT {call_id}:\nstatus: failed\nstderr:\n{detail}"


def _candidate_outcome(recovery: RecoveryTrace | TwoStepRecoveryTrace | None) -> str:
    if recovery is None:
        return PASS
    if isinstance(recovery, TwoStepRecoveryTrace):
        return recovery.first_outcome
    return recovery.outcome


def _matched_key(task: MBPPTask, initial_code: str, initial_outcome: str) -> str:
    payload = json.dumps(
        {
            "task_id": task.task_id,
            "initial_code": initial_code,
            "initial_outcome": initial_outcome,
            "final_code": task.code,
            "final_outcome": PASS,
        },
        sort_keys=True,
    )
    return _text_sha256(payload)


DEEP_RECOVERY_MODES = ("deep_shadow", "deep_visible")


def sft_row(
    task: MBPPTask,
    arm: str,
    recovery: RecoveryTrace | TwoStepRecoveryTrace | None,
    recovery_mode: str | None = None,
) -> dict:
    is_deep = isinstance(recovery, TwoStepRecoveryTrace)
    if recovery is None and recovery_mode is not None:
        raise ValueError("direct traces cannot carry a recovery_mode")
    if recovery is not None and is_deep and recovery_mode not in DEEP_RECOVERY_MODES:
        raise ValueError("two-step recovery traces require recovery_mode 'deep_shadow' or 'deep_visible'")
    if recovery is not None and not is_deep and recovery_mode not in ("shadow", "visible"):
        raise ValueError("recovery traces require recovery_mode 'shadow' or 'visible'")
    trace_prefix = f"data/blueprints/{task.case_id}"
    file_path = f"{trace_prefix}/solution.py"
    initial_code = recovery.first_code if is_deep else (recovery.initial_code if recovery else task.code)
    initial_outcome = _candidate_outcome(recovery)
    messages: list[dict[str, str]] = [
        *task_prompt(task, trace_prefix, arm),
        {
            "role": "assistant",
            "content": _call(
                "read_file", {"id": "c1", "file_path": file_path}
            ),
        },
        {
            "role": "tool",
            "content": (
                f"RESULT c1:\nstatus: success\nstdout:\n{PLACEHOLDER.rstrip()}"
            ),
        },
        {
            "role": "assistant",
            "content": _call(
                "apply_patch",
                {
                    "id": "c2",
                    "file_path": file_path,
                    "find": PLACEHOLDER,
                    "replace": initial_code,
                },
            ),
        },
        {
            "role": "tool",
            "content": "RESULT c2:\nstatus: success\nstdout:\npatch applied",
        },
    ]

    if arm == "a":
        messages.append(
            {
                "role": "assistant",
                "content": _call(
                    "python_test",
                    {"id": "c3", "project_path": trace_prefix},
                ),
            }
        )
        if is_deep:
            messages.extend(
                [
                    {"role": "tool", "content": _failed_result("c3", recovery.first_outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c4",
                                "file_path": file_path,
                                "find": recovery.second_patch.find,
                                "replace": recovery.second_patch.replace,
                            },
                        ),
                    },
                    {"role": "tool", "content": "RESULT c4:\nstatus: success\nstdout:\npatch applied"},
                    {
                        "role": "assistant",
                        "content": _call(
                            "python_test",
                            {"id": "c5", "project_path": trace_prefix},
                        ),
                    },
                    {"role": "tool", "content": _failed_result("c5", recovery.second_outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c6",
                                "file_path": file_path,
                                "find": recovery.final_patch.find,
                                "replace": recovery.final_patch.replace,
                            },
                        ),
                    },
                    {"role": "tool", "content": "RESULT c6:\nstatus: success\nstdout:\npatch applied"},
                    {
                        "role": "assistant",
                        "content": _call(
                            "python_test",
                            {"id": "c7", "project_path": trace_prefix},
                        ),
                    },
                    {"role": "tool", "content": "RESULT c7:\nstatus: success\nstdout:\ntests passed"},
                ]
            )
        elif recovery:
            messages.extend(
                [
                    {"role": "tool", "content": _failed_result("c3", recovery.outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c4",
                                "file_path": file_path,
                                "find": recovery.patch.find,
                                "replace": recovery.patch.replace,
                            },
                        ),
                    },
                    {
                        "role": "tool",
                        "content": (
                            "RESULT c4:\nstatus: success\nstdout:\npatch applied"
                        ),
                    },
                    {
                        "role": "assistant",
                        "content": _call(
                            "python_test",
                            {"id": "c5", "project_path": trace_prefix},
                        ),
                    },
                    {
                        "role": "tool",
                        "content": (
                            "RESULT c5:\nstatus: success\nstdout:\ntests passed"
                        ),
                    },
                ]
            )
        else:
            messages.append(
                {
                    "role": "tool",
                    "content": (
                        "RESULT c3:\nstatus: success\nstdout:\ntests passed"
                    ),
                }
            )
    else:
        if is_deep and recovery_mode == "deep_shadow":
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            f"<PREDICTION>{recovery.first_outcome}</PREDICTION>\n"
                            "<DECISION>REVISE</DECISION>\n"
                            + _call(
                                "apply_patch",
                                {
                                    "id": "c3",
                                    "file_path": file_path,
                                    "find": recovery.second_patch.find,
                                    "replace": recovery.second_patch.replace,
                                },
                            )
                        ),
                    },
                    {"role": "tool", "content": "RESULT c3:\nstatus: success\nstdout:\npatch applied"},
                    {
                        "role": "assistant",
                        "content": (
                            f"<PREDICTION>{recovery.second_outcome}</PREDICTION>\n"
                            "<DECISION>REVISE</DECISION>\n"
                            + _call(
                                "apply_patch",
                                {
                                    "id": "c4",
                                    "file_path": file_path,
                                    "find": recovery.final_patch.find,
                                    "replace": recovery.final_patch.replace,
                                },
                            )
                        ),
                    },
                    {"role": "tool", "content": "RESULT c4:\nstatus: success\nstdout:\npatch applied"},
                ]
            )
            test_call = "c5"
        elif is_deep:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            "<PREDICTION>PASS</PREDICTION>\n"
                            "<DECISION>KEEP</DECISION>\n"
                            + _call(
                                "python_test",
                                {"id": "c3", "project_path": trace_prefix},
                            )
                        ),
                    },
                    {"role": "tool", "content": _failed_result("c3", recovery.first_outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c4",
                                "file_path": file_path,
                                "find": recovery.second_patch.find,
                                "replace": recovery.second_patch.replace,
                            },
                        ),
                    },
                    {"role": "tool", "content": "RESULT c4:\nstatus: success\nstdout:\npatch applied"},
                    {
                        "role": "assistant",
                        "content": (
                            "<PREDICTION>PASS</PREDICTION>\n"
                            "<DECISION>KEEP</DECISION>\n"
                            + _call(
                                "python_test",
                                {"id": "c5", "project_path": trace_prefix},
                            )
                        ),
                    },
                    {"role": "tool", "content": _failed_result("c5", recovery.second_outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c6",
                                "file_path": file_path,
                                "find": recovery.final_patch.find,
                                "replace": recovery.final_patch.replace,
                            },
                        ),
                    },
                    {"role": "tool", "content": "RESULT c6:\nstatus: success\nstdout:\npatch applied"},
                ]
            )
            test_call = "c7"
        elif recovery and recovery_mode == "shadow":
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            f"<PREDICTION>{recovery.outcome}</PREDICTION>\n"
                            "<DECISION>REVISE</DECISION>\n"
                            + _call(
                                "apply_patch",
                                {
                                    "id": "c3",
                                    "file_path": file_path,
                                    "find": recovery.patch.find,
                                    "replace": recovery.patch.replace,
                                },
                            )
                        ),
                    },
                    {
                        "role": "tool",
                        "content": (
                            "RESULT c3:\nstatus: success\nstdout:\npatch applied"
                        ),
                    },
                ]
            )
            test_call = "c4"
        elif recovery:
            messages.extend(
                [
                    {
                        "role": "assistant",
                        "content": (
                            "<PREDICTION>PASS</PREDICTION>\n"
                            "<DECISION>KEEP</DECISION>\n"
                            + _call(
                                "python_test",
                                {"id": "c3", "project_path": trace_prefix},
                            )
                        ),
                    },
                    {"role": "tool", "content": _failed_result("c3", recovery.outcome)},
                    {
                        "role": "assistant",
                        "content": _call(
                            "apply_patch",
                            {
                                "id": "c4",
                                "file_path": file_path,
                                "find": recovery.patch.find,
                                "replace": recovery.patch.replace,
                            },
                        ),
                    },
                    {
                        "role": "tool",
                        "content": (
                            "RESULT c4:\nstatus: success\nstdout:\npatch applied"
                        ),
                    },
                ]
            )
            test_call = "c5"
        else:
            test_call = "c3"
        messages.extend(
            [
                {
                    "role": "assistant",
                    "content": (
                        "<PREDICTION>PASS</PREDICTION>\n"
                        "<DECISION>KEEP</DECISION>\n"
                        + _call(
                            "python_test",
                            {"id": test_call, "project_path": trace_prefix},
                        )
                    ),
                },
                {
                    "role": "tool",
                    "content": (
                        f"RESULT {test_call}:\n"
                        "status: success\nstdout:\ntests passed"
                    ),
                },
            ]
        )

    messages.append(
        {
            "role": "assistant",
            "content": "FINAL: implemented the function and passed the tests.",
        }
    )
    return {
        "arm": arm,
        "case_id": task.case_id,
        "candidate_code_sha256": _text_sha256(initial_code),
        "candidate_outcome": initial_outcome,
        "second_candidate_outcome": recovery.second_outcome if is_deep else None,
        "final_code_sha256": _text_sha256(task.code),
        "final_outcome": PASS,
        "matched_key": _matched_key(task, initial_code, initial_outcome),
        "messages": messages,
        "recovery_mode": recovery_mode,
        "task_id": task.task_id,
        "test_code": task.test_code,
        "trace_type": "recovery" if recovery else "direct",
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def _verify_gold(tasks: list[MBPPTask], timeout: int) -> None:
    with tempfile.TemporaryDirectory(prefix="predict-gold-audit-") as temporary:
        project = Path(temporary)
        solution = project / "solution.py"
        for task in tasks:
            solution.write_text(task.code, encoding="utf-8")
            result = run_hidden_tests(project, task.test_code, timeout)
            if result.outcome != PASS:
                raise RuntimeError(
                    f"official gold failed for {task.case_id}: {result.outcome}"
                )


def _assign_recoveries(
    train: list[MBPPTask], *, seed: int, count: int, deep_count: int, timeout: int
) -> tuple[dict[str, RecoveryTrace], dict[str, TwoStepRecoveryTrace]]:
    """Fill the deep (two-step) quota first, since a verified two-step chain
    is the harder search; remaining recovery slots take a one-step trace.
    Each task contributes at most one recovery (deep or shallow, never both)."""
    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(DEFAULT_MODEL)
    ordered = sorted(train, key=lambda task: (_assignment_key(task, seed), task.task_id))
    shallow_count = count - deep_count

    def _fits(task: MBPPTask, mode: str, trace: RecoveryTrace | TwoStepRecoveryTrace) -> bool:
        tokens = len(
            tokenizer.encode(
                render_messages(sft_row(task, "b", trace, recovery_mode=mode)["messages"]),
                add_special_tokens=False,
            )
        )
        return tokens <= SFT_MAX_TOKENS

    shallow: dict[str, RecoveryTrace] = {}
    deep: dict[str, TwoStepRecoveryTrace] = {}
    for task in ordered:
        if len(deep) < deep_count:
            deep_trace = generate_two_step_recovery(
                task.code, task.test_code, task.case_id, timeout=timeout, gold_verified=True,
            )
            # Gate on deep_visible: the longer of the two deep shapes.
            if deep_trace is not None and _fits(task, "deep_visible", deep_trace):
                deep[task.case_id] = deep_trace
                continue
        if len(shallow) < shallow_count:
            trace = generate_recovery(
                task.code, task.test_code, task.case_id, timeout=timeout, gold_verified=True,
            )
            # Gate on visible: the longer of the two shallow shapes.
            if trace is not None and _fits(task, "visible", trace):
                shallow[task.case_id] = trace
        if len(deep) == deep_count and len(shallow) == shallow_count:
            break
    if len(deep) != deep_count or len(shallow) != shallow_count:
        raise RuntimeError(
            f"constructed {len(deep)} deep + {len(shallow)} shallow recoveries; "
            f"needed {deep_count} deep + {shallow_count} shallow"
        )
    return shallow, deep


def prepare_data(
    output: Path = Path("data"),
    cache_dir: Path = Path(".cache/predict"),
    *,
    seed: int = SEED,
    recovery_count: int = RECOVERY_COUNT,
    timeout: int = 5,
) -> Path:
    if recovery_count != RECOVERY_COUNT:
        raise ValueError(
            f"the experiment requires exactly {RECOVERY_COUNT} recovery traces"
        )
    paths = {
        name: download_source(source, cache_dir) for name, source in SOURCES.items()
    }
    official_train = load_mbpp(paths["train"], "train")
    official_validation = load_mbpp(paths["validation"], "validation")
    final = load_mbpp(paths["test"], "test")
    _require_ids(official_train, OFFICIAL_TRAIN_IDS, "train")
    _require_ids(official_validation, OFFICIAL_VALIDATION_IDS, "validation")
    _require_ids(final, TEST_IDS, "test")
    all_tasks = [*official_train, *official_validation, *final]
    if len({task.task_id for task in all_tasks}) != len(all_tasks):
        raise RuntimeError("official MBPP splits overlap by task ID")

    _verify_gold(all_tasks, timeout)
    sft_tasks, rl_train_tasks, validation_tasks = _split_experiment_tasks(
        official_train, official_validation, seed
    )
    deep_recovery_count = DEEP_SHADOW_RECOVERY_COUNT + DEEP_VISIBLE_RECOVERY_COUNT
    shallow_recoveries, deep_recoveries = _assign_recoveries(
        sft_tasks,
        seed=seed,
        count=recovery_count,
        deep_count=deep_recovery_count,
        timeout=timeout,
    )
    recoveries: dict[str, RecoveryTrace | TwoStepRecoveryTrace] = {
        **shallow_recoveries,
        **deep_recoveries,
    }
    if len(sft_tasks) - len(recoveries) != DIRECT_COUNT:
        raise RuntimeError("SFT direct/recovery composition drifted")
    # Alternate over each seeded selection order so every spec-required Arm B
    # recovery mode is covered deterministically.
    recovery_modes = {
        **{
            case_id: "shadow" if index % 2 == 0 else "visible"
            for index, case_id in enumerate(shallow_recoveries)
        },
        **{
            case_id: "deep_shadow" if index % 2 == 0 else "deep_visible"
            for index, case_id in enumerate(deep_recoveries)
        },
    }
    mode_counts = Counter(recovery_modes.values())
    if mode_counts != {
        "shadow": SHADOW_RECOVERY_COUNT,
        "visible": VISIBLE_RECOVERY_COUNT,
        "deep_shadow": DEEP_SHADOW_RECOVERY_COUNT,
        "deep_visible": DEEP_VISIBLE_RECOVERY_COUNT,
    }:
        raise RuntimeError(f"recovery mode split drifted: {dict(mode_counts)}")

    output = output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "blueprints").mkdir(parents=True)
    (staging / "sft" / "arm_a").mkdir(parents=True)
    (staging / "sft" / "arm_b").mkdir(parents=True)
    for task in all_tasks:
        project = staging / "blueprints" / task.case_id
        project.mkdir()
        (project / "solution.py").write_text(PLACEHOLDER, encoding="utf-8")

    arm_a_sft = [
        sft_row(
            task,
            "a",
            recoveries.get(task.case_id),
            recovery_mode=recovery_modes.get(task.case_id),
        )
        for task in sft_tasks
    ]
    arm_b_sft = [
        sft_row(
            task,
            "b",
            recoveries.get(task.case_id),
            recovery_mode=recovery_modes.get(task.case_id),
        )
        for task in sft_tasks
    ]
    _write_jsonl(staging / "sft" / "arm_a" / "train.jsonl", arm_a_sft)
    _write_jsonl(staging / "sft" / "arm_b" / "train.jsonl", arm_b_sft)
    for arm in ("a", "b"):
        _write_jsonl(
            staging / f"arm_{arm}_train.jsonl",
            [task_row(_with_split(task, "train"), arm) for task in rl_train_tasks],
        )
        _write_jsonl(
            staging / f"arm_{arm}_validation.jsonl",
            [
                task_row(_with_split(task, "validation"), arm)
                for task in validation_tasks
            ],
        )
        _write_jsonl(
            staging / f"arm_{arm}_test.jsonl",
            [task_row(task, arm) for task in final],
        )

    ordered_sft = sorted(sft_tasks, key=lambda task: task.task_id)
    assignment = [
        {
            "task_id": task.task_id,
            "case_id": task.case_id,
            "trace_type": (
                "recovery" if task.case_id in recoveries else "direct"
            ),
            "candidate_outcome": _candidate_outcome(recoveries.get(task.case_id)),
            "recovery_mode": recovery_modes.get(task.case_id),
        }
        for task in ordered_sft
    ]
    (staging / "assignments.json").write_text(
        json.dumps(assignment, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    outcome_counts = Counter(trace.outcome for trace in shallow_recoveries.values())
    for trace in deep_recoveries.values():
        outcome_counts[trace.first_outcome] += 1
        outcome_counts[trace.second_outcome] += 1
    manifest = {
        "artifacts": {
            relative: {
                "sha256": _sha256(staging / relative),
                "bytes": (staging / relative).stat().st_size,
            }
            for relative in DATA_ARTIFACTS
        },
        "sources": {
            name: {
                "repository": source.repository,
                "revision": source.revision,
                "path": source.path,
                "sha256": source.sha256,
                "license": source.license,
            }
            for name, source in SOURCES.items()
        },
        "split": {
            "seed": seed,
            "sft": len(sft_tasks),
            "train": len(rl_train_tasks),
            "validation": len(validation_tasks),
            "test": len(final),
            "official_train_ids": "601-974",
            "official_validation_ids": "511-600",
            "validation_to_train": VALIDATION_TO_TRAIN_COUNT,
            "sft_task_ids": sorted(task.task_id for task in sft_tasks),
            "rl_train_task_ids": sorted(task.task_id for task in rl_train_tasks),
            "validation_task_ids": sorted(task.task_id for task in validation_tasks),
            "test_ids": "11-510",
            "sft_per_arm": len(sft_tasks),
            "sft_direct": DIRECT_COUNT,
            "sft_recovery": len(recoveries),
            "sft_recovery_shadow": mode_counts["shadow"],
            "sft_recovery_visible": mode_counts["visible"],
            "sft_recovery_deep_shadow": mode_counts["deep_shadow"],
            "sft_recovery_deep_visible": mode_counts["deep_visible"],
            "recovery_outcomes": dict(sorted(outcome_counts.items())),
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    output.mkdir(parents=True, exist_ok=True)
    generated_names = {path.name for path in staging.iterdir()}
    for existing in output.iterdir():
        if existing.name == "README.md" or existing.name.startswith("__"):
            continue
        if existing.name in generated_names or existing.name.endswith(".jsonl"):
            if existing.is_dir():
                shutil.rmtree(existing)
            else:
                existing.unlink()
    for generated in staging.iterdir():
        os.replace(generated, output / generated.name)
    staging.rmdir()
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/predict"))
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--timeout", type=int, default=5)
    args = parser.parse_args()
    print(
        prepare_data(
            args.output,
            args.cache_dir,
            seed=args.seed,
            timeout=args.timeout,
        )
    )


if __name__ == "__main__":
    main()
