import json
from pathlib import Path

from data import seed_rl
from data.harvest import Candidate
from data.prepare import PLACEHOLDER, _text_sha256
from glyph.program import ASSERTION_FAILURE, PASS, RUNTIME_ERROR, TIMEOUT


def _candidates() -> list[Candidate]:
    # Every task can fail an assertion; only one can time out.
    items = [
        Candidate(f"mbpp_{n}", f"def f():\n    return {n}\n", ASSERTION_FAILURE, False)
        for n in range(601, 605)
    ]
    items.append(Candidate("mbpp_601", "def f():\n    while True:\n        pass\n", TIMEOUT, False))
    items.append(Candidate("mbpp_602", "def f():\n    return 1 / 0\n", RUNTIME_ERROR, False))
    return items


def test_rare_classes_claim_the_tasks_only_they_can_cover() -> None:
    case_ids = [f"mbpp_{n}" for n in range(601, 605)]

    assigned = seed_rl.assign(_candidates(), case_ids, seed=0)

    assert set(assigned) == set(case_ids)
    outcomes = sorted(candidate.outcome for candidate in assigned.values())
    # mbpp_601 is the only timeout available, so it must not be spent on the
    # assertion failure it could equally have supplied.
    assert assigned["mbpp_601"].outcome == TIMEOUT
    assert outcomes.count(RUNTIME_ERROR) == 1
    assert outcomes.count(ASSERTION_FAILURE) == 2


def test_passing_candidates_are_never_seeded() -> None:
    assigned = seed_rl.assign(
        [
            Candidate("mbpp_601", "def f():\n    return 1\n", PASS, False),
            Candidate("mbpp_602", "def f():\n    return 2\n", ASSERTION_FAILURE, False),
        ],
        ["mbpp_601", "mbpp_602"],
        seed=0,
    )

    assert set(assigned) == {"mbpp_602"}


def _task_row(case_id: str) -> dict:
    return {
        "arm": "b",
        "blueprint_root": f"blueprints/{case_id}",
        "case_id": case_id,
        "prompt": [
            {"role": "system", "content": "be a coding agent"},
            {
                "role": "user",
                "content": f"Implement it.\n\nThe project is at data/blueprints/{case_id}.",
            },
        ],
        "source": "mbpp",
        "split": "train",
        "task_id": int(case_id.removeprefix("mbpp_")),
        "test_code": "assert f() == 0\n",
        "trace_prefix": f"data/blueprints/{case_id}",
    }


def test_seeded_set_is_written_beside_an_untouched_baseline(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    case_ids = [f"mbpp_{n}" for n in range(601, 605)]
    for arm in ("a", "b"):
        (data_dir / f"arm_{arm}_train.jsonl").write_text(
            "".join(json.dumps({**_task_row(c), "arm": arm}) + "\n" for c in case_ids),
            encoding="utf-8",
        )
    baseline = (data_dir / "arm_b_train.jsonl").read_text(encoding="utf-8")
    traces = tmp_path / "traces.jsonl"
    traces.write_text("", encoding="utf-8")

    report = seed_rl.build([traces], data_dir, seed=0, suffix="seeded")

    # No candidates, so every task keeps the empty blueprint and the pool size
    # is preserved rather than shrinking to the tasks that failed.
    assert report["tasks"] == 4
    assert report["empty_tasks"] == 4
    assert (data_dir / "arm_b_train.jsonl").read_text(encoding="utf-8") == baseline
    for case_id in case_ids:
        seeded_file = data_dir / "blueprints_seeded" / case_id / "solution.py"
        assert seeded_file.read_text(encoding="utf-8") == PLACEHOLDER

    rows = [json.loads(line) for line in (data_dir / "arm_b_train_seeded.jsonl").open()]
    assert [row["case_id"] for row in rows] == case_ids
    for row in rows:
        assert row["trace_prefix"] == f"data/blueprints_seeded/{row['case_id']}"
        assert row["blueprint_root"] == f"blueprints_seeded/{row['case_id']}"
        # The prompt tells the agent where the project is; a stale path would
        # send every tool call at the unseeded copy.
        assert row["prompt"][1]["content"].endswith(
            f"The project is at data/blueprints_seeded/{row['case_id']}."
        )
        assert row["seed_outcome"] is None


def test_seeded_blueprint_carries_the_assigned_candidate(tmp_path: Path) -> None:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    case_ids = ["mbpp_601", "mbpp_602"]
    for arm in ("a", "b"):
        (data_dir / f"arm_{arm}_train.jsonl").write_text(
            "".join(json.dumps({**_task_row(c), "arm": arm}) + "\n" for c in case_ids),
            encoding="utf-8",
        )
    bug = "def f():\n    return 9\n"
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "task": {"data": {"name": "mbpp_601"}},
                "info": {
                    "glyph": {
                        "prediction_targets": [
                            {
                                "context_messages": [
                                    {
                                        "role": "assistant",
                                        "content": (
                                            'CALL apply_patch {"id":"c2",'
                                            '"file_path":"data/blueprints/mbpp_601/solution.py",'
                                            f'"find":{json.dumps(PLACEHOLDER)},'
                                            f'"replace":{json.dumps(bug)}}}'
                                        ),
                                    }
                                ],
                                "actual": ASSERTION_FAILURE,
                                "sampled_prediction": PASS,
                                "decision": "KEEP",
                                "shadow": False,
                                "candidate_sha256": _text_sha256(bug),
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    report = seed_rl.build([traces], data_dir, seed=0, suffix="seeded")

    assert report["seed_outcomes"] == {ASSERTION_FAILURE: 1}
    assert (data_dir / "blueprints_seeded" / "mbpp_601" / "solution.py").read_text() == bug
    assert (
        data_dir / "blueprints_seeded" / "mbpp_602" / "solution.py"
    ).read_text() == PLACEHOLDER
    rows = {
        row["case_id"]: row
        for row in (json.loads(line) for line in (data_dir / "arm_b_train_seeded.jsonl").open())
    }
    assert rows["mbpp_601"]["seed_outcome"] == ASSERTION_FAILURE
    assert rows["mbpp_601"]["seed_code_sha256"] == _text_sha256(bug)
    assert json.loads((data_dir / "seeded_assignments.json").read_text()) == [
        {
            "case_id": "mbpp_601",
            "seed_code_sha256": _text_sha256(bug),
            "seed_outcome": ASSERTION_FAILURE,
        },
        {"case_id": "mbpp_602", "seed_code_sha256": None, "seed_outcome": None},
    ]
