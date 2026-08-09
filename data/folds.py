"""Split the SFT pool in two so a policy can be sampled on unseen tasks.

Harvesting the policy's real failures needs tasks the policy was not fine-tuned
on, and every MBPP task is already spoken for: 212 SFT, 212 RL, 40 validation,
500 final test. Sampling the RL pool would collapse the SFT/RL split the
experiment depends on, and sampling the SFT pool with a checkpoint trained on
all of it measures memorisation rather than failure.

Cross-fitting resolves both. The 212 SFT tasks are split into two folds; a
probe is trained on each fold and samples the other. Every harvested failure
then comes from a model that never saw that task, and the union covers the
whole SFT pool without touching RL, validation or test.

    uv run python -m data.folds
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from .prepare import (
    SEED,
    SOURCES,
    _split_experiment_tasks,
    _write_jsonl,
    download_source,
    load_mbpp,
    task_row,
)
from .validate import _rows

FOLDS = ("a", "b")


def _fold_key(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}\0fold\0{case_id}".encode()).hexdigest()


def split(case_ids: list[str], *, seed: int) -> dict[str, list[str]]:
    """Deal the SFT case ids into two halves, deterministically from `seed`."""
    ordered = sorted(case_ids, key=lambda case_id: (_fold_key(case_id, seed), case_id))
    half = len(ordered) // 2
    return {"a": sorted(ordered[:half]), "b": sorted(ordered[half:])}


def sft_tasks(cache_dir: Path, *, seed: int):
    paths = {name: download_source(source, cache_dir) for name, source in SOURCES.items()}
    tasks, _, _ = _split_experiment_tasks(
        load_mbpp(paths["train"], "train"),
        load_mbpp(paths["validation"], "validation"),
        seed,
    )
    return tasks


def build(data_dir: Path, cache_dir: Path, *, seed: int, arm: str) -> dict:
    sft_rows = _rows(data_dir / "sft" / f"arm_{arm}" / "train.jsonl")
    tasks = {task.case_id: task for task in sft_tasks(cache_dir, seed=seed)}
    if {row["case_id"] for row in sft_rows} != set(tasks):
        raise RuntimeError("the SFT traces and the SFT task split disagree")
    folds = split(list(tasks), seed=seed)
    root = data_dir / "folds"

    report: dict = {"seed": seed, "arm": arm, "folds": {}}
    for name, case_ids in folds.items():
        members = set(case_ids)
        (root / f"fold_{name}" / "sft" / f"arm_{arm}").mkdir(parents=True, exist_ok=True)
        # The probe trains on the published traces for its half, unchanged, so
        # it differs from the released checkpoint only in how many tasks it saw.
        _write_jsonl(
            root / f"fold_{name}" / "sft" / f"arm_{arm}" / "train.jsonl",
            [row for row in sft_rows if row["case_id"] in members],
        )
        # The task rows the *other* probe will sample. Same write-from-scratch
        # format as every other task set in the experiment, except that
        # blueprint_root is resolved against the task file's own directory --
        # these live one level deeper than data/arm_b_train.jsonl.
        _write_jsonl(
            root / f"fold_{name}_tasks.jsonl",
            [
                {
                    **task_row(tasks[case_id], arm),
                    "blueprint_root": f"../blueprints/{case_id}",
                    "fold": name,
                }
                for case_id in case_ids
            ],
        )
        report["folds"][name] = {"tasks": len(case_ids)}

    (root / "assignments.json").write_text(
        json.dumps(folds, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/predict"))
    parser.add_argument("--arm", default="b", choices=("a", "b"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = build(args.data_dir, args.cache_dir, seed=args.seed, arm=args.arm)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
