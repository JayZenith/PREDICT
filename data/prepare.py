"""Download, split, and format pinned MBPP data."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import urllib.request
from dataclasses import dataclass
from pathlib import Path

import pyarrow.parquet as pq

from .recovery import RecoveryTrace, generate_recovery
from glyph.chat import DEFAULT_SYSTEM_PROMPT


PLACEHOLDER = "# Write your function here.\n"
MBPP_REVISION = "4bb6404fdc6cacfda99d4ac4205087b89d32030c"
MBPP_PLUS_REVISION = "b2d74c91837c3f2a20c1299ae98133cbe7cfa077"


@dataclass(frozen=True)
class SourceFile:
    name: str
    repository: str
    revision: str
    path: str
    sha256: str
    license: str

    @property
    def url(self) -> str:
        return (
            f"https://huggingface.co/datasets/{self.repository}/resolve/"
            f"{self.revision}/{self.path}"
        )


SOURCES = {
    "train": SourceFile(
        "train",
        "google-research-datasets/mbpp",
        MBPP_REVISION,
        "full/train-00000-of-00001.parquet",
        "09d125ca31edacb7800be8c67c45abff618faf0214ff551291817d06bdb914ae",
        "CC-BY-4.0",
    ),
    "validation": SourceFile(
        "validation",
        "google-research-datasets/mbpp",
        MBPP_REVISION,
        "full/validation-00000-of-00001.parquet",
        "3f0ec060987432d99fe8fb409d31e6c67445b208a01741c5583517c80a10fe80",
        "CC-BY-4.0",
    ),
    "mbppplus": SourceFile(
        "mbppplus",
        "evalplus/mbppplus",
        MBPP_PLUS_REVISION,
        "data/test-00000-of-00001-d5781c9c51e02795.parquet",
        "dc20030b3788fccf617444edcb34138ef13d7e4fafd17bfcb8c1279dbb12399b",
        "Apache-2.0",
    ),
}


@dataclass(frozen=True)
class MBPPTask:
    task_id: int
    prompt: str
    code: str
    test_code: str
    source: str

    @property
    def case_id(self) -> str:
        return f"{self.source}_{self.task_id}"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def download_source(source: SourceFile, cache_dir: Path) -> Path:
    cache_dir = cache_dir.expanduser().resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)
    destination = cache_dir / f"{source.name}.parquet"
    if destination.exists() and _sha256(destination) == source.sha256:
        return destination
    destination.unlink(missing_ok=True)
    temporary = destination.with_suffix(".parquet.part")
    request = urllib.request.Request(source.url, headers={"User-Agent": "GLYPH/2.0"})
    with urllib.request.urlopen(request, timeout=120) as response, temporary.open("wb") as output:
        shutil.copyfileobj(response, output)
    if _sha256(temporary) != source.sha256:
        temporary.unlink(missing_ok=True)
        raise RuntimeError(f"{source.name} failed its pinned SHA-256 check")
    os.replace(temporary, destination)
    return destination


def load_mbpp(path: Path, source: str) -> list[MBPPTask]:
    table = pq.read_table(
        path,
        columns=["task_id", "text", "code", "test_list", "test_setup_code"],
    )
    tasks: list[MBPPTask] = []
    for row in table.to_pylist():
        task_id = int(row["task_id"])
        prompt = str(row["text"] or "").strip()
        code = str(row["code"] or "").strip()
        tests = [str(test).strip() for test in row["test_list"] or [] if str(test).strip()]
        setup = str(row["test_setup_code"] or "").strip()
        if not prompt or not code or not tests:
            raise ValueError(f"{source} task {task_id} is missing prompt, code, or tests")
        test_code = "\n".join(part for part in (setup, *tests) if part) + "\n"
        tasks.append(MBPPTask(task_id, prompt, code + "\n", test_code, source))
    return tasks


def load_mbppplus(path: Path) -> list[MBPPTask]:
    table = pq.read_table(path, columns=["task_id", "prompt", "code", "test"])
    tasks: list[MBPPTask] = []
    for row in table.to_pylist():
        task_id = int(row["task_id"])
        # The official MBPP test partition is task IDs 11-510. Keeping only this
        # range removes every overlap with the MBPP train/validation partitions.
        if not 11 <= task_id <= 510:
            continue
        prompt = str(row["prompt"] or "").strip()
        code = str(row["code"] or "").strip()
        test_code = str(row["test"] or "").strip()
        if not prompt or not code or not test_code:
            raise ValueError(f"MBPP+ task {task_id} is missing prompt, code, or tests")
        tasks.append(MBPPTask(task_id, prompt, code + "\n", test_code + "\n", "mbppplus"))
    return tasks


def _split_key(task: MBPPTask, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0{task.task_id}\0{task.prompt}".encode()).hexdigest()


def split_train(
    tasks: list[MBPPTask], *, sft_count: int, rl_count: int, seed: int
) -> tuple[list[MBPPTask], list[MBPPTask]]:
    if sft_count < 1 or rl_count < 1:
        raise ValueError("sft_count and rl_count must be positive")
    if sft_count + rl_count > len(tasks):
        raise ValueError(
            f"requested {sft_count + rl_count} MBPP train tasks but only {len(tasks)} exist"
        )
    ordered = sorted(tasks, key=lambda task: (_split_key(task, seed), task.task_id))
    return ordered[:sft_count], ordered[sft_count : sft_count + rl_count]


def task_prompt(task: MBPPTask, trace_prefix: str) -> list[dict[str, str]]:
    user = (
        "Implement the requested Python function in solution.py. Run the tests.\n\n"
        f"{task.prompt}\n\nThe project is at {trace_prefix}."
    )
    return [
        {"role": "system", "content": DEFAULT_SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def task_row(task: MBPPTask) -> dict:
    blueprint_root = f"blueprints/{task.case_id}"
    trace_prefix = f"data/{blueprint_root}"
    return {
        "blueprint_root": blueprint_root,
        "case_id": task.case_id,
        "prompt": task_prompt(task, trace_prefix),
        "source": task.source,
        "task_id": task.task_id,
        "test_code": task.test_code,
        "trace_prefix": trace_prefix,
    }


def sft_row(task: MBPPTask, recovery: RecoveryTrace | None = None) -> dict:
    trace_prefix = f"data/blueprints/{task.case_id}"
    file_path = f"{trace_prefix}/solution.py"
    initial_code = recovery.initial_code if recovery else task.code
    messages = [
        *task_prompt(task, trace_prefix),
        {
            "role": "assistant",
            "content": f'CALL read_file {{"id":"c1","file_path":"{file_path}"}}',
        },
        {
            "role": "tool",
            "content": f"RESULT c1:\nstatus: success\nstdout:\n{PLACEHOLDER.rstrip()}",
        },
        {
            "role": "assistant",
            "content": "CALL apply_patch "
            + json.dumps(
                {
                    "id": "c2",
                    "file_path": file_path,
                    "find": PLACEHOLDER,
                    "replace": initial_code,
                },
                separators=(",", ":"),
            ),
        },
        {
            "role": "tool",
            "content": "RESULT c2:\nstatus: success\nstdout:\npatch applied",
        },
        {
            "role": "assistant",
            "content": f'CALL python_test {{"id":"c3","project_path":"{trace_prefix}"}}',
        },
    ]
    test_call = 3
    for patch in recovery.patches if recovery else ():
        patch_call = test_call + 1
        next_test = patch_call + 1
        messages.extend(
            [
                {
                    "role": "tool",
                    "content": (
                        f"RESULT c{test_call}:\nstatus: failed\n"
                        "stderr:\nhidden tests failed"
                    ),
                },
                {
                    "role": "assistant",
                    "content": "CALL apply_patch "
                    + json.dumps(
                        {
                            "id": f"c{patch_call}",
                            "file_path": file_path,
                            "find": patch.find,
                            "replace": patch.replace,
                        },
                        separators=(",", ":"),
                    ),
                },
                {
                    "role": "tool",
                    "content": (
                        f"RESULT c{patch_call}:\nstatus: success\nstdout:\npatch applied"
                    ),
                },
                {
                    "role": "assistant",
                    "content": (
                        f'CALL python_test {{"id":"c{next_test}",'
                        f'"project_path":"{trace_prefix}"}}'
                    ),
                },
            ]
        )
        test_call = next_test
    messages.extend(
        [
            {
                "role": "tool",
                "content": (
                    f"RESULT c{test_call}:\n"
                    "status: success\nstdout:\nhidden tests passed"
                ),
            },
            {
                "role": "assistant",
                "content": "FINAL: implemented the function and passed the hidden tests.",
            },
        ]
    )
    return {
        "case_id": task.case_id,
        "messages": messages,
        "recovery_phases": len(recovery.patches) if recovery else 0,
    }


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    path.write_text(
        "".join(json.dumps(row, sort_keys=True) + "\n" for row in rows),
        encoding="utf-8",
    )


def prepare_data(
    output: Path = Path("data"),
    cache_dir: Path = Path(".cache/glyph"),
    *,
    sft_count: int = 240,
    rl_count: int = 134,
    recovery_one: int = 40,
    recovery_two: int = 20,
    seed: int = 42,
) -> Path:
    paths = {name: download_source(source, cache_dir) for name, source in SOURCES.items()}
    train = load_mbpp(paths["train"], "mbpp")
    validation = load_mbpp(paths["validation"], "mbpp_dev")
    final = load_mbppplus(paths["mbppplus"])
    sft, rl = split_train(train, sft_count=sft_count, rl_count=rl_count, seed=seed)
    if recovery_one < 0 or recovery_two < 0 or recovery_one + recovery_two > len(sft):
        raise ValueError("recovery counts must be non-negative and fit inside the SFT split")
    recovery_by_id: dict[str, RecoveryTrace] = {}
    recovery_counts = {1: recovery_one, 2: recovery_two}
    for phases in (2, 1):
        for task in sft:
            found = sum(len(trace.patches) == phases for trace in recovery_by_id.values())
            if found == recovery_counts[phases]:
                break
            if task.case_id in recovery_by_id:
                continue
            trace = generate_recovery(
                task.code, task.test_code, task.case_id, phases=phases
            )
            if trace is not None:
                recovery_by_id[task.case_id] = trace
        found = sum(len(trace.patches) == phases for trace in recovery_by_id.values())
        if found != recovery_counts[phases]:
            raise RuntimeError(
                f"constructed {found} {phases}-phase recovery traces; "
                f"{recovery_counts[phases]} requested"
            )

    output = output.expanduser().resolve()
    staging = output.with_name(f".{output.name}.staging")
    shutil.rmtree(staging, ignore_errors=True)
    (staging / "blueprints").mkdir(parents=True)
    selected = [*sft, *rl, *validation, *final]
    if len({task.case_id for task in selected}) != len(selected):
        raise RuntimeError("task identifiers overlap across prepared splits")
    for task in selected:
        project = staging / "blueprints" / task.case_id
        project.mkdir()
        (project / "solution.py").write_text(PLACEHOLDER, encoding="utf-8")

    _write_jsonl(
        staging / "sft.jsonl",
        [sft_row(task, recovery_by_id.get(task.case_id)) for task in sft],
    )
    _write_jsonl(staging / "rl_candidates.jsonl", [task_row(task) for task in rl])
    _write_jsonl(staging / "dev.jsonl", [task_row(task) for task in validation])
    _write_jsonl(staging / "test.jsonl", [task_row(task) for task in final])
    manifest = {
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
            "sft": len(sft),
            "sft_recovery": len(recovery_by_id),
            "sft_recovery_one": recovery_one,
            "sft_recovery_two": recovery_two,
            "rl_candidates": len(rl),
            "dev": len(validation),
            "test": len(final),
            "test_rule": "MBPP+ task_id 11-510; disjoint from MBPP train and validation",
        },
    }
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    output.mkdir(parents=True, exist_ok=True)
    (output / "rl.jsonl").unlink(missing_ok=True)
    for generated in staging.iterdir():
        destination = output / generated.name
        if destination.is_dir():
            shutil.rmtree(destination)
        else:
            destination.unlink(missing_ok=True)
        os.replace(generated, destination)
    staging.rmdir()
    return output / "manifest.json"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/glyph"))
    parser.add_argument("--sft", type=int, default=240)
    parser.add_argument("--rl", type=int, default=134)
    parser.add_argument("--recovery-one", type=int, default=40)
    parser.add_argument("--recovery-two", type=int, default=20)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(
        prepare_data(
            args.output,
            args.cache_dir,
            sft_count=args.sft,
            rl_count=args.rl,
            recovery_one=args.recovery_one,
            recovery_two=args.recovery_two,
            seed=args.seed,
        )
    )


if __name__ == "__main__":
    main()
