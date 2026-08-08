import json
import random

import pytest

from data import harvest, prepare
from glyph.program import ASSERTION_FAILURE, PASS, RUNTIME_ERROR

TASK = prepare.MBPPTask(
    601,
    "Return whether n is even.",
    "def is_even(n):\n    return n % 2 == 0\n",
    "assert is_even(2)\nassert not is_even(3)\n",
    "train",
)
OTHER_TASK = prepare.MBPPTask(
    602,
    "Return n doubled.",
    "def twice(n):\n    return n * 2\n",
    "assert twice(2) == 4\n",
    "train",
)


def _prefix(*codes: str) -> list[dict]:
    """A rollout prefix that patches the blueprint into each code in turn."""
    messages = [{"role": "user", "content": "implement it"}]
    code = prepare.PLACEHOLDER
    for index, replacement in enumerate(codes, start=2):
        messages.append(
            {
                "role": "assistant",
                "content": prepare._call(
                    "apply_patch",
                    {
                        "id": f"c{index}",
                        "file_path": "data/blueprints/mbpp_601/solution.py",
                        "find": code,
                        "replace": replacement,
                    },
                ),
            }
        )
        messages.append({"role": "tool", "content": f"RESULT c{index}:\nstatus: success"})
        code = replacement
    return messages


def _trace(case_id: str, targets: list[dict]) -> str:
    return json.dumps(
        {
            "task": {"data": {"name": case_id}},
            "info": {"glyph": {"prediction_targets": targets}},
        }
    )


def _target(code: str, outcome: str, *, digest: str | None = None) -> dict:
    return {
        "context_messages": _prefix(code),
        "actual": outcome,
        "sampled_prediction": PASS,
        "decision": "KEEP",
        "shadow": False,
        "candidate_sha256": digest or prepare._text_sha256(code),
    }


def test_candidate_replay_must_reproduce_the_verified_file(tmp_path) -> None:
    bug = "def is_even(n):\n    return n % 2\n"
    path = tmp_path / "traces.jsonl"
    path.write_text(
        _trace("mbpp_601", [_target(bug, ASSERTION_FAILURE)])
        + "\n"
        + _trace("mbpp_601", [_target(bug, RUNTIME_ERROR, digest="0" * 64)])
        + "\n",
        encoding="utf-8",
    )

    candidates, skipped = harvest.load_candidates([path])

    assert [(c.case_id, c.code, c.outcome) for c in candidates] == [
        ("mbpp_601", bug, ASSERTION_FAILURE)
    ]
    assert skipped["digest mismatch"] == 1


def test_repair_prefers_code_the_policy_itself_passed() -> None:
    bug = "def is_even(n):\n    return n % 2\n"
    own = "def is_even(n):\n    return n % 2 == 0  # mine\n"
    chosen = harvest.select(
        [
            harvest.Candidate("mbpp_601", bug, ASSERTION_FAILURE, False),
            harvest.Candidate("mbpp_601", own, PASS, False),
        ],
        {"mbpp_601": TASK},
        size=1,
        recovery_fraction=1.0,
        shadow_fraction=1.0,
        seed=0,
    )

    (task, recovery, mode) = chosen[0]
    assert recovery is not None and mode == "shadow"
    assert recovery.initial_code == bug
    # The repair is the policy's own passing code, not the gold solution, so a
    # failing candidate cannot be identified by style alone.
    assert recovery.patch.replace == own
    assert task.code == own


def test_repair_falls_back_to_gold_when_the_policy_never_solved_the_task() -> None:
    bug = "def is_even(n):\n    return n % 2\n"
    chosen = harvest.select(
        [harvest.Candidate("mbpp_601", bug, ASSERTION_FAILURE, False)],
        {"mbpp_601": TASK},
        size=1,
        recovery_fraction=1.0,
        shadow_fraction=1.0,
        seed=0,
    )

    (_, recovery, _) = chosen[0]
    assert recovery is not None
    assert recovery.patch.replace == TASK.code


def test_selection_levels_the_outcome_classes() -> None:
    candidates = [
        harvest.Candidate("mbpp_601", f"def f():\n    return {i}\n", ASSERTION_FAILURE, False)
        for i in range(50)
    ] + [
        harvest.Candidate("mbpp_601", f"def g():\n    return {i}\n", RUNTIME_ERROR, False)
        for i in range(3)
    ]

    chosen = harvest.select(
        candidates,
        {"mbpp_601": TASK},
        size=10,
        recovery_fraction=1.0,
        shadow_fraction=1.0,
        seed=0,
    )

    counts = {}
    for _, recovery, _ in chosen:
        counts[recovery.outcome] = counts.get(recovery.outcome, 0) + 1
    # The rare class contributes everything it has before the common one takes
    # the remainder, instead of the common class swamping the quota.
    assert counts == {RUNTIME_ERROR: 3, ASSERTION_FAILURE: 7}


def test_only_shadow_recoveries_emit_the_failure_label() -> None:
    bug = "def is_even(n):\n    return n % 2\n"
    candidates = [
        harvest.Candidate("mbpp_601", bug, ASSERTION_FAILURE, False),
        harvest.Candidate("mbpp_602", bug, ASSERTION_FAILURE, False),
    ]
    tasks = {"mbpp_601": TASK, "mbpp_602": OTHER_TASK}

    chosen = harvest.select(
        candidates, tasks, size=2, recovery_fraction=1.0, shadow_fraction=0.5, seed=0
    )

    modes = sorted(mode for _, _, mode in chosen)
    assert modes == ["shadow", "visible"]
    for task, recovery, mode in chosen:
        text = "\n".join(
            message["content"]
            for message in prepare.sft_row(task, "b", recovery, recovery_mode=mode)["messages"]
        )
        emitted = f"<PREDICTION>{ASSERTION_FAILURE}</PREDICTION>" in text
        assert emitted is (mode == "shadow")


def test_oversized_traces_are_replaced_from_the_same_class() -> None:
    keep = "def is_even(n):\n    return n % 2\n"
    huge = "def is_even(n):\n" + "    x = 1\n" * 200
    chosen = harvest.select(
        [
            harvest.Candidate("mbpp_601", huge, ASSERTION_FAILURE, False),
            harvest.Candidate("mbpp_602", keep, ASSERTION_FAILURE, False),
        ],
        {"mbpp_601": TASK, "mbpp_602": OTHER_TASK},
        size=1,
        recovery_fraction=1.0,
        shadow_fraction=1.0,
        seed=0,
        fits=lambda task, recovery, mode: recovery is None or "x = 1" not in recovery.initial_code,
    )

    assert len(chosen) == 1
    assert chosen[0][1].initial_code == keep


def test_round_robin_spreads_a_quota_across_tasks() -> None:
    items = [("a", 1), ("a", 2), ("a", 3), ("b", 1), ("c", 1)]

    ordered = harvest._round_robin(items, key=lambda pair: pair[0], rng=random.Random(0))

    assert sorted(name for name, _ in ordered[:3]) == ["a", "b", "c"]
    assert len(ordered) == len(items)


@pytest.mark.parametrize("outcome", [ASSERTION_FAILURE, RUNTIME_ERROR])
def test_harvested_rows_carry_the_verified_outcome(outcome: str) -> None:
    bug = "def is_even(n):\n    return n % 2\n"
    chosen = harvest.select(
        [harvest.Candidate("mbpp_601", bug, outcome, False)],
        {"mbpp_601": TASK},
        size=1,
        recovery_fraction=1.0,
        shadow_fraction=1.0,
        seed=0,
    )
    row = prepare.sft_row(chosen[0][0], "b", chosen[0][1], recovery_mode=chosen[0][2])

    assert row["candidate_outcome"] == outcome
    assert row["candidate_code_sha256"] == prepare._text_sha256(bug)
    assert row["messages"][-1]["content"].startswith("FINAL:")
