import json
from pathlib import Path

import pytest

from data import folds, harvest
from data.harvest import Candidate
from glyph.program import ASSERTION_FAILURE

CASE_IDS = [f"mbpp_{n}" for n in range(601, 701)]


def test_split_is_balanced_disjoint_and_seed_stable() -> None:
    first = folds.split(CASE_IDS, seed=42)
    again = folds.split(list(reversed(CASE_IDS)), seed=42)
    other_seed = folds.split(CASE_IDS, seed=43)

    assert first == again
    assert len(first["a"]) == len(first["b"]) == 50
    assert not set(first["a"]) & set(first["b"])
    assert sorted(first["a"] + first["b"]) == sorted(CASE_IDS)
    # A different seed has to actually redraw, or "cross-fitted" would be a
    # claim about one arbitrary partition.
    assert other_seed != first


def test_fold_task_files_resolve_their_blueprints() -> None:
    """GlyphTaskset resolves blueprint_root against the task file's own
    directory. The fold sets sit one level deeper than data/arm_b_train.jsonl,
    so a path copied from there points at a directory that does not exist and
    the sampling run dies on load."""
    from glyph.chat import message_content
    from glyph.taskset import GlyphTaskset, GlyphTasksetConfig

    root = Path(__file__).resolve().parents[1]
    for fold in ("a", "b"):
        path = root / "data" / "folds" / f"fold_{fold}_tasks.jsonl"
        tasks = GlyphTaskset(GlyphTasksetConfig(id="glyph", data_path=str(path))).load()
        assert len(tasks) == 106
        for task in tasks:
            assert Path(task.data.blueprint_root).is_dir()
            # The prompt and the sandbox agree on where the project lives.
            assert task.data.trace_prefix == f"data/blueprints/{task.data.case_id}"
            assert task.data.trace_prefix in message_content(task.data.prompt[-1])


def test_harvest_refuses_candidates_from_the_rl_or_validation_pools(tmp_path: Path) -> None:
    """The split is the experiment's whole defence against SFT/RL contact.

    A sampling run pointed at the wrong task file would otherwise produce a
    perfectly ordinary-looking dataset built on tasks RL is about to train on.
    """
    from data.prepare import SOURCES, _split_experiment_tasks, download_source, load_mbpp

    cache = Path(".cache/predict")
    paths = {name: download_source(source, cache) for name, source in SOURCES.items()}
    _, rl_tasks, _ = _split_experiment_tasks(
        load_mbpp(paths["train"], "train"),
        load_mbpp(paths["validation"], "validation"),
        42,
    )
    stolen = rl_tasks[0]
    code = "def f():\n    return 1\n"
    traces = tmp_path / "traces.jsonl"
    traces.write_text(
        json.dumps(
            {
                "task": {"data": {"name": stolen.case_id}},
                "info": {
                    "glyph": {
                        "prediction_targets": [
                            {
                                "context_messages": [],
                                "actual": ASSERTION_FAILURE,
                                "sampled_prediction": "PASS",
                                "decision": "KEEP",
                                "shadow": False,
                                "candidate_sha256": harvest._text_sha256(
                                    harvest.PLACEHOLDER
                                ),
                            }
                        ]
                    }
                },
            }
        )
        + "\n",
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="reserved for RL or validation"):
        harvest.build(
            [traces],
            tmp_path / "out",
            cache_dir=cache,
            size=1,
            recovery_fraction=1.0,
            shadow_fraction=1.0,
            seed=42,
        )


def test_probe_holds_epochs_not_steps_against_the_baseline() -> None:
    """A probe sees half the traces, so equal steps would double each trace's
    exposure and overfit it relative to the checkpoint it stands in for."""
    import tomllib

    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = tomllib.loads((root / "arm_b_sft.toml").read_text())
    for fold in ("a", "b"):
        probe = tomllib.loads((root / f"probe_{fold}_sft.toml").read_text())
        assert probe["data"]["name"] == f"data/folds/fold_{fold}/sft/arm_b"
        assert probe["data"]["batch_size"] == baseline["data"]["batch_size"]
        assert probe["max_steps"] * 2 == baseline["max_steps"]
        assert probe["optim"] == baseline["optim"]
        assert probe["scheduler"]["warmup_steps"] + probe["scheduler"]["decay_steps"] == (
            probe["max_steps"]
        )


def test_v2_rl_config_changes_only_the_starting_checkpoint() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = tomllib.loads((root / "arm_b_rl.toml").read_text())
    v2 = tomllib.loads((root / "arm_b_rl_v2.toml").read_text())

    assert v2.pop("model")["name"] == "JayZenith/SFT_ARM_B_v2"
    assert baseline.pop("model")["name"] == "JayZenith/SFT_ARM_B"
    assert v2.pop("output_dir") == "outputs/arm_b_rl_v2"
    baseline.pop("output_dir")
    v2.pop("wandb")
    baseline.pop("wandb")
    # Taskset, algorithm, sampler, validation set and trainer must be identical,
    # or the comparison against the published Arm B run gains a second cause.
    assert v2 == baseline


def test_v2_sft_config_keeps_the_published_budget() -> None:
    import tomllib

    root = Path(__file__).resolve().parents[1] / "configs"
    baseline = tomllib.loads((root / "arm_b_sft.toml").read_text())
    v2 = tomllib.loads((root / "arm_b_sft_v2.toml").read_text())

    assert v2.pop("data")["name"] == "data/sft_harvested/arm_b"
    assert baseline.pop("data")["name"] == "data/sft/arm_b"
    assert v2.pop("output_dir") == "outputs/arm_b_sft_v2"
    baseline.pop("output_dir")
    v2.pop("wandb")
    baseline.pop("wandb")
    assert v2 == baseline


def test_arm_a_data_and_configs_are_untouched() -> None:
    import subprocess

    root = Path(__file__).resolve().parents[1]
    changed = subprocess.run(
        ["git", "diff", "--name-only", "bdc4ffa", "HEAD", "--"],
        cwd=root,
        capture_output=True,
        text=True,
    ).stdout.split()
    forbidden = [
        name
        for name in changed
        if name.startswith(("data/sft/arm_a", "configs/arm_a", "data/arm_a"))
        or name
        in {
            "data/arm_b_train.jsonl",
            "data/arm_b_validation.jsonl",
            "data/arm_b_test.jsonl",
            "configs/arm_b_rl.toml",
            "configs/arm_b_sft.toml",
        }
    ]
    assert forbidden == []
