"""Seed the RL tasks with failing code the SFT baseline actually wrote.

The published RL set starts every task from an empty `solution.py`, so the
policy only reaches a prediction point after it has written something, and the
outcome it is asked to predict is whatever it happened to produce. Assertion
failures are the class it never learned, and they are also the class it rarely
has to face on turn one.

This builds a parallel task set whose blueprints already contain a candidate
the SFT baseline wrote and the verifier scored as failing. The task pool, the
tests and the prompts are otherwise identical, so the first decision the policy
makes is a prediction about real broken code. The original
`data/arm_{a,b}_train.jsonl` and `data/blueprints/` are left alone: they remain
the baseline the published runs reproduce against.

    uv run python -m data.seed_rl --traces <run>/traces.jsonl
"""

from __future__ import annotations

import argparse
import json
import shutil
from collections import Counter, defaultdict
from pathlib import Path

from glyph.program import PASS

from .harvest import Candidate, _round_robin, load_candidates
from .prepare import PLACEHOLDER, SEED, _text_sha256, _write_jsonl
from .validate import _rows

BLUEPRINTS = "data/blueprints"
SEEDED_BLUEPRINTS = "data/blueprints_seeded"


def assign(
    candidates: list[Candidate], case_ids: list[str], *, seed: int
) -> dict[str, Candidate]:
    """Give each task one failing candidate, levelling the outcome classes.

    A task usually fails several different ways across a sampling run, so the
    class a task contributes is a free choice. Spending that choice on the
    rarest class first is what stops the seeded set from being almost entirely
    assertion failures, which is the distribution the policy already sees.
    """
    import random

    rng = random.Random(seed)
    wanted = set(case_ids)
    by_class: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.outcome != PASS and candidate.case_id in wanted:
            by_class[candidate.outcome].append(candidate)

    assigned: dict[str, Candidate] = {}
    # Rarest class first: a task it can cover may be the only one that can.
    for outcome in sorted(by_class, key=lambda name: len(by_class[name])):
        pool = _round_robin(by_class[outcome], key=lambda c: c.case_id, rng=rng)
        share = len(wanted) // len(by_class)
        taken = 0
        for candidate in pool:
            if taken == share:
                break
            if candidate.case_id in assigned:
                continue
            assigned[candidate.case_id] = candidate
            taken += 1

    # Whatever is left over goes to any class that can still cover it.
    leftovers = _round_robin(
        [c for group in by_class.values() for c in group],
        key=lambda c: c.case_id,
        rng=rng,
    )
    for candidate in leftovers:
        if candidate.case_id not in assigned:
            assigned[candidate.case_id] = candidate
    return assigned


def build(traces: list[Path], data_dir: Path, *, seed: int, suffix: str) -> dict:
    candidates, skipped = load_candidates(traces)
    rows = {arm: _rows(data_dir / f"arm_{arm}_train.jsonl") for arm in ("a", "b")}
    case_ids = [row["case_id"] for row in rows["b"]]
    assigned = assign(candidates, case_ids, seed=seed)

    seeded_root = data_dir / Path(SEEDED_BLUEPRINTS).name
    shutil.rmtree(seeded_root, ignore_errors=True)
    seeded_root.mkdir(parents=True)
    for case_id in case_ids:
        project = seeded_root / case_id
        project.mkdir()
        candidate = assigned.get(case_id)
        # Tasks the baseline never failed keep the empty blueprint, so the pool
        # stays the same 212 tasks rather than shrinking to the ones it broke.
        (project / "solution.py").write_text(
            candidate.code if candidate else PLACEHOLDER, encoding="utf-8"
        )

    manifest: list[dict] = []
    for arm, arm_rows in rows.items():
        seeded_rows = []
        for row in arm_rows:
            case_id = row["case_id"]
            blueprint_root = f"{Path(SEEDED_BLUEPRINTS).name}/{case_id}"
            trace_prefix = f"{SEEDED_BLUEPRINTS}/{case_id}"
            prompt = [
                {
                    **message,
                    "content": str(message["content"]).replace(
                        f"{BLUEPRINTS}/{case_id}", trace_prefix
                    ),
                }
                for message in row["prompt"]
            ]
            candidate = assigned.get(case_id)
            seeded_rows.append(
                {
                    **row,
                    "blueprint_root": blueprint_root,
                    "prompt": prompt,
                    "trace_prefix": trace_prefix,
                    "seed_outcome": candidate.outcome if candidate else None,
                    "seed_code_sha256": (
                        _text_sha256(candidate.code) if candidate else None
                    ),
                }
            )
        _write_jsonl(data_dir / f"arm_{arm}_train_{suffix}.jsonl", seeded_rows)
        if arm == "b":
            manifest = [
                {
                    "case_id": row["case_id"],
                    "seed_code_sha256": row["seed_code_sha256"],
                    "seed_outcome": row["seed_outcome"],
                }
                for row in seeded_rows
            ]

    (data_dir / f"{suffix}_assignments.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return {
        "candidates": len(candidates),
        "skipped": dict(skipped),
        "tasks": len(case_ids),
        "seeded_tasks": len(assigned),
        "empty_tasks": len(case_ids) - len(assigned),
        "seed_outcomes": dict(Counter(c.outcome for c in assigned.values())),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--suffix", default="seeded")
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = build(args.traces, args.data_dir, seed=args.seed, suffix=args.suffix)
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
