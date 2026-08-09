"""Build predictive SFT data from the policy's own verified mistakes.

`data.prepare` writes SFT traces whose failing candidate is a mutation of the
gold solution: an operator flip that nearly always yields an assertion failure.
The resulting Arm B SFT checkpoint predicts PASS almost everywhere, because the
failures it was shown do not look like the failures it makes.

This module builds the same trace shapes from a sampling run instead. Every
candidate is code the policy actually wrote, and every label is the outcome the
verifier actually observed for that code, so the prediction target matches the
distribution the policy meets during RL.

The traces must come from probes sampled on SFT tasks they were held out of --
see `data.folds`. Harvesting the RL pool would put SFT and RL on the same
tasks, which the experiment's split exists to prevent.

    uv run python -m data.harvest --traces <run>/traces.jsonl --output data/sft_harvested
"""

from __future__ import annotations

import argparse
import json
import random
from collections import Counter, defaultdict
from collections.abc import Callable
from dataclasses import dataclass, replace
from pathlib import Path

from glyph.chat import render_messages
from glyph.program import PASS, parse_calls

from .prepare import (
    DEFAULT_MODEL,
    PLACEHOLDER,
    SEED,
    SFT_MAX_TOKENS,
    SOURCES,
    MBPPTask,
    _split_experiment_tasks,
    _text_sha256,
    _write_jsonl,
    download_source,
    load_mbpp,
    sft_row,
)
from .recovery import Patch, RecoveryTrace

# A "visible" recovery trace predicts PASS, keeps a bad candidate and recovers
# from the executed failure, so it never demonstrates a failure label. Only
# "shadow" traces teach the policy to name the failure before running the test,
# which is the behaviour the harvested data exists to supply.
SHADOW_FRACTION = 0.75


@dataclass(frozen=True)
class Candidate:
    """One patch the policy wrote, with the outcome the verifier observed."""

    case_id: str
    code: str
    outcome: str
    shadow: bool


def _candidate_code(context_messages: list[dict]) -> str:
    """Replay the apply_patch calls in a rollout prefix onto the blueprint."""
    code = PLACEHOLDER
    for message in context_messages:
        if message.get("role") != "assistant":
            continue
        calls, errors = parse_calls(message.get("content") or "")
        if errors:
            raise ValueError(f"unparsable assistant turn: {errors[0]}")
        for call in calls:
            if call.tool != "apply_patch":
                continue
            find = call.params.get("find")
            new = call.params.get("replace")
            if find is None or new is None:
                raise ValueError("apply_patch requires find and replace")
            if code.count(find) != 1:
                raise ValueError("apply_patch find is not unique")
            code = code.replace(find, new, 1)
    return code


def load_candidates(paths: list[Path]) -> tuple[list[Candidate], Counter[str]]:
    """Extract every verified prediction target from one or more trace files.

    The reconstructed code is checked against the candidate digest the harness
    recorded, so a candidate is only kept when the replay provably reproduces
    the file the verifier ran.
    """
    seen: set[tuple[str, str]] = set()
    candidates: list[Candidate] = []
    skipped: Counter[str] = Counter()
    for path in paths:
        with path.open(encoding="utf-8") as source:
            for line in source:
                trace = json.loads(line)
                if trace.get("errors"):
                    skipped["errored rollout"] += 1
                    continue
                glyph = (trace.get("info") or {}).get("glyph") or {}
                case_id = ((trace.get("task") or {}).get("data") or {}).get("name")
                for target in glyph.get("prediction_targets") or []:
                    try:
                        code = _candidate_code(target.get("context_messages") or [])
                    except ValueError:
                        skipped["unreplayable prefix"] += 1
                        continue
                    if _text_sha256(code) != target.get("candidate_sha256"):
                        skipped["digest mismatch"] += 1
                        continue
                    key = (case_id, code)
                    if key in seen:
                        skipped["duplicate candidate"] += 1
                        continue
                    seen.add(key)
                    candidates.append(
                        Candidate(
                            case_id=case_id,
                            code=code,
                            outcome=target["actual"],
                            shadow=bool(target.get("shadow")),
                        )
                    )
    return candidates, skipped


def _round_robin(items: list, *, key, rng: random.Random) -> list:
    """Order items so every group contributes once before any contributes twice.

    A single task can fail dozens of ways in a large sampling run. Draining one
    task before touching the next would spend a class quota on a handful of
    problems, so the pool is dealt out one candidate per task at a time.
    """
    groups: dict = defaultdict(list)
    for item in items:
        groups[key(item)].append(item)
    for group in groups.values():
        rng.shuffle(group)
    order = sorted(groups)
    rng.shuffle(order)
    ordered: list = []
    for index in range(max(len(group) for group in groups.values()) if groups else 0):
        for name in order:
            if index < len(groups[name]):
                ordered.append(groups[name][index])
    return ordered


def select(
    candidates: list[Candidate],
    tasks: dict[str, MBPPTask],
    *,
    size: int,
    recovery_fraction: float,
    shadow_fraction: float,
    seed: int,
    fits: Callable[[MBPPTask, RecoveryTrace | None, str | None], bool] = lambda *_: True,
) -> list[tuple[MBPPTask, RecoveryTrace | None, str | None]]:
    """Choose a class-balanced set of traces from the harvested candidates.

    Failing candidates become recovery traces; the repair is another candidate
    the policy wrote and the verifier passed, falling back to the gold solution
    only when the policy never solved that task. Keeping both sides of a
    recovery in the policy's own style is what stops the prediction target from
    collapsing into "gold-looking code passes, model-looking code fails".

    `fits` rejects a trace the SFT context cannot hold; a rejected draw is
    replaced from the same class so the balance survives the filter.
    """
    rng = random.Random(seed)
    passing: dict[str, list[str]] = defaultdict(list)
    failing: dict[str, list[Candidate]] = defaultdict(list)
    for candidate in candidates:
        if candidate.case_id not in tasks:
            continue
        if candidate.outcome == PASS:
            passing[candidate.case_id].append(candidate.code)
        else:
            failing[candidate.outcome].append(candidate)

    def repair(case_id: str) -> str:
        pool = passing.get(case_id)
        return rng.choice(pool) if pool else tasks[case_id].code

    recovery_target = min(round(size * recovery_fraction), sum(map(len, failing.values())))
    classes = sorted(failing)
    quota = {name: 0 for name in classes}
    remaining = recovery_target
    # Level the classes up together so a rare outcome contributes everything it
    # has before a common one takes the rest.
    while remaining > 0 and any(len(failing[n]) > quota[n] for n in classes):
        for name in sorted(classes, key=lambda n: (quota[n], n)):
            if remaining == 0:
                break
            if quota[name] < len(failing[name]):
                quota[name] += 1
                remaining -= 1

    chosen: list[tuple[MBPPTask, RecoveryTrace | None, str | None]] = []
    for name in classes:
        pool = _round_robin(failing[name], key=lambda c: c.case_id, rng=rng)
        kept = 0
        shadow_count = round(quota[name] * shadow_fraction)
        for candidate in pool:
            if kept == quota[name]:
                break
            fixed = repair(candidate.case_id)
            if fixed == candidate.code:
                continue
            entry = (
                replace(tasks[candidate.case_id], code=fixed),
                RecoveryTrace(
                    initial_code=candidate.code,
                    patch=Patch(find=candidate.code, replace=fixed),
                    outcome=candidate.outcome,
                ),
                "shadow" if kept < shadow_count else "visible",
            )
            if not fits(*entry):
                continue
            chosen.append(entry)
            kept += 1

    # Direct traces show the policy its own correct code, so PASS is not only
    # ever seen on a repair.
    direct_pool = _round_robin(
        [(case_id, code) for case_id, codes in passing.items() for code in codes],
        key=lambda pair: pair[0],
        rng=rng,
    )
    for case_id, code in direct_pool:
        if len(chosen) == size:
            break
        entry = (replace(tasks[case_id], code=code), None, None)
        if fits(*entry):
            chosen.append(entry)

    rng.shuffle(chosen)
    return chosen


def build(
    traces: list[Path],
    output: Path,
    *,
    cache_dir: Path,
    size: int,
    recovery_fraction: float,
    shadow_fraction: float,
    seed: int,
    arm: str = "b",
    max_tokens: int = SFT_MAX_TOKENS,
    model: str = DEFAULT_MODEL,
) -> dict:
    candidates, skipped = load_candidates(traces)
    paths = {name: download_source(source, cache_dir) for name, source in SOURCES.items()}
    official_train = load_mbpp(paths["train"], "train")
    official_validation = load_mbpp(paths["validation"], "validation")
    sft_tasks, rl_train_tasks, validation_tasks = _split_experiment_tasks(
        official_train, official_validation, seed
    )
    tasks = {task.case_id: task for task in sft_tasks}
    # The whole point of cross-fitting is that the traces came from the SFT
    # pool. A candidate from anywhere else means the sampling run was pointed
    # at the wrong task file, and silently dropping it would hide that.
    reserved = {task.case_id for task in (*rl_train_tasks, *validation_tasks)}
    trespassing = sorted({c.case_id for c in candidates} & reserved)
    if trespassing:
        raise ValueError(
            "harvested candidates come from tasks reserved for RL or validation: "
            + ", ".join(trespassing[:5])
        )

    from transformers import AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model)
    oversize = Counter()

    def fits(task: MBPPTask, recovery: RecoveryTrace | None, mode: str | None) -> bool:
        # The policy writes longer code than the gold solutions the synthetic
        # traces were cut from, so the SFT context is the binding constraint.
        row = sft_row(task, arm, recovery, recovery_mode=mode)
        text = render_messages(row["messages"])
        if len(tokenizer.encode(text, add_special_tokens=False)) > max_tokens:
            oversize[recovery.outcome if recovery else PASS] += 1
            return False
        return True

    chosen = select(
        candidates,
        tasks,
        size=size,
        recovery_fraction=recovery_fraction,
        shadow_fraction=shadow_fraction,
        seed=seed,
        fits=fits,
    )

    # Only the harvested arm is written. Arm A has no prediction turn, so
    # nothing here would change what it learns, and leaving its published SFT
    # data alone keeps its runs reproducible.
    output = output.expanduser().resolve()
    (output / f"arm_{arm}").mkdir(parents=True, exist_ok=True)
    _write_jsonl(
        output / f"arm_{arm}" / "train.jsonl",
        [sft_row(task, arm, recovery, recovery_mode=mode) for task, recovery, mode in chosen],
    )

    # Only shadow recoveries put a failure label in the assistant turn; visible
    # ones keep PASS and let the executed failure do the teaching.
    taught: Counter[str] = Counter()
    verified: Counter[str] = Counter()
    for _, recovery, mode in chosen:
        if recovery is None:
            taught[PASS] += 1
            verified[PASS] += 1
            continue
        taught[recovery.outcome if mode == "shadow" else PASS] += 1
        taught[PASS] += 1
        verified[recovery.outcome] += 1
        verified[PASS] += 1
    return {
        "candidates": len(candidates),
        "candidate_outcomes": dict(Counter(c.outcome for c in candidates)),
        "skipped": dict(skipped),
        "oversize_rejected": dict(oversize),
        "rows": len(chosen),
        "recovery_rows": sum(1 for _, recovery, _ in chosen if recovery),
        "recovery_modes": dict(Counter(mode for _, _, mode in chosen if mode)),
        "emitted_prediction_labels": dict(taught),
        "verified_outcomes": dict(verified),
        "tasks_covered": len({task.case_id for task, _, _ in chosen}),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--traces", nargs="+", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/sft_harvested"))
    parser.add_argument("--cache-dir", type=Path, default=Path(".cache/predict"))
    parser.add_argument("--size", type=int, default=212)
    parser.add_argument("--recovery-fraction", type=float, default=0.6)
    parser.add_argument("--shadow-fraction", type=float, default=SHADOW_FRACTION)
    parser.add_argument("--arm", default="b", choices=("a", "b"))
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    report = build(
        args.traces,
        args.output,
        cache_dir=args.cache_dir,
        size=args.size,
        recovery_fraction=args.recovery_fraction,
        shadow_fraction=args.shadow_fraction,
        arm=args.arm,
        seed=args.seed,
    )
    print(json.dumps(report, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
