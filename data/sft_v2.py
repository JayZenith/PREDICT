"""Build a v2 Arm B SFT set that covers every outcome class.

The published SFT data teaches prediction from 37 assertion failures and 8
runtime errors, and nothing else -- its failing candidates are operator flips
of the gold solution, which break the answer without breaking the run. The Arm
B checkpoint it produces predicts PASS almost everywhere, and RL never learned
to name the class the data never showed it.

This keeps the published recipe and changes only which failures it draws on:
the same 212 SFT tasks, the same trace shapes, the same budget, but the
recovery traces are spread across the outcome classes the environment defines
and weighted towards the shadow mode, which is the only one whose assistant
turn names the failure. `data/sft/` is untouched, so the published runs still
reproduce.

    uv run python -m data.sft_v2
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from pathlib import Path

from glyph.program import ASSERTION_FAILURE, RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT

from .prepare import (
    SEED,
    SOURCES,
    _split_experiment_tasks,
    _write_jsonl,
    download_source,
    load_mbpp,
    sft_row,
)
from .recovery import generate_recovery

# Assertion failures carry twice the weight of the other classes. Splitting the
# quota evenly would cover the missing classes at the cost of showing fewer
# assertion failures than the published set did, and that is the class the
# policy most needs to predict: it is the one RL never learned, and the one a
# wrong prediction is most expensive to discover by running the tests.
CLASS_WEIGHTS = {ASSERTION_FAILURE: 2, RUNTIME_ERROR: 1, SYNTAX_ERROR: 1, TIMEOUT: 1}
RECOVERY_COUNT = 130
SHADOW_FRACTION = 0.8


def build(
    data_dir: Path,
    cache_dir: Path,
    *,
    seed: int = SEED,
    recovery_count: int = RECOVERY_COUNT,
    shadow_fraction: float = SHADOW_FRACTION,
    timeout: int = 5,
) -> dict:
    paths = {name: download_source(source, cache_dir) for name, source in SOURCES.items()}
    tasks, _, _ = _split_experiment_tasks(
        load_mbpp(paths["train"], "train"),
        load_mbpp(paths["validation"], "validation"),
        seed,
    )
    rng = random.Random(seed)
    ordered = sorted(tasks, key=lambda task: task.task_id)
    rng.shuffle(ordered)

    rows = []
    counts: Counter[str] = Counter()
    unmatched: Counter[str] = Counter()
    recoveries = 0
    for task in ordered:
        if recoveries >= recovery_count:
            # Everything past the recovery quota is a direct trace, exactly as
            # in the published set: the policy still has to see itself succeed.
            rows.append((task, None, None))
            continue
        # Ask for whichever class is furthest behind. Most tasks can supply
        # most classes, so taking the first that works would let one class --
        # timeouts, which almost any function can be made to hit -- swallow the
        # quota and reproduce the imbalance this set exists to remove.
        recovery = None
        behind = sorted(
            CLASS_WEIGHTS, key=lambda name: (counts[name] / CLASS_WEIGHTS[name], name)
        )
        for candidate_class in behind:
            recovery = generate_recovery(
                task.code, task.test_code, task.case_id, timeout=timeout, want=candidate_class
            )
            if recovery is not None:
                break
            unmatched[candidate_class] += 1
        if recovery is None:
            rows.append((task, None, None))
            continue
        rows.append((task, recovery, None))
        counts[recovery.outcome] += 1
        recoveries += 1

    # Assign the modes per class rather than by coin flip. Only shadow traces
    # name the failure in the assistant turn, so letting chance decide leaves
    # the count of labels the policy is actually taught to emit varying by
    # several per class between runs.
    seen: Counter[str] = Counter()
    for index, (task, recovery, _) in enumerate(rows):
        if recovery is None:
            continue
        shadow_quota = round(counts[recovery.outcome] * shadow_fraction)
        mode = "shadow" if seen[recovery.outcome] < shadow_quota else "visible"
        seen[recovery.outcome] += 1
        rows[index] = (task, recovery, mode)

    rng.shuffle(rows)
    out = data_dir / "sft_v2" / "arm_b"
    out.mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        out / "train.jsonl",
        [sft_row(task, "b", recovery, recovery_mode=mode) for task, recovery, mode in rows],
    )

    emitted: Counter[str] = Counter()
    for _, recovery, mode in rows:
        emitted["PASS"] += 1
        if recovery is not None and mode == "shadow":
            emitted[recovery.outcome] += 1
    return {
        "rows": len(rows),
        "recovery_rows": sum(1 for _, recovery, _ in rows if recovery),
        "recovery_modes": dict(Counter(mode for _, _, mode in rows if mode)),
        "verified_outcomes": dict(counts),
        "emitted_prediction_labels": dict(emitted),
        "classes_unavailable": dict(unmatched),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/predict"))
    parser.add_argument("--recovery-count", type=int, default=RECOVERY_COUNT)
    parser.add_argument("--shadow-fraction", type=float, default=SHADOW_FRACTION)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = build(
        args.data_dir,
        args.cache_dir,
        seed=args.seed,
        recovery_count=args.recovery_count,
        shadow_fraction=args.shadow_fraction,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
