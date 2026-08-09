import tomllib
from pathlib import Path

import pytest

from data import recovery as recovery_generator
from data.validate import validate_sft
from glyph.program import (
    ASSERTION_FAILURE,
    PASS,
    RUNTIME_ERROR,
    SYNTAX_ERROR,
    TIMEOUT,
    run_hidden_tests,
)

ROOT = Path(__file__).resolve().parents[1]
CODE = "def volume(b, h):\n    return (b * h) / 3\n"
TESTS = "assert volume(3, 4) == 4.0\nassert volume(6, 2) == 4.0\n"


@pytest.mark.parametrize("want", [RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT])
def test_targeted_mutations_produce_the_class_they_claim(want: str, tmp_path: Path) -> None:
    trace = recovery_generator.generate_recovery(CODE, TESTS, "mbpp_x", want=want)

    assert trace is not None
    assert trace.outcome == want
    # The label is only worth training on if the environment agrees, so run it.
    (tmp_path / "solution.py").write_text(trace.initial_code, encoding="utf-8")
    assert run_hidden_tests(tmp_path, TESTS, 5).outcome == want
    # And the repair has to land back exactly on the gold text.
    repaired = trace.initial_code.replace(trace.patch.find, trace.patch.replace, 1)
    assert repaired == CODE


def test_default_generator_is_unchanged_by_the_new_families() -> None:
    """The published SFT data was cut with the untargeted generator; if that
    path moved, data/sft/ would no longer regenerate from data.prepare."""
    trace = recovery_generator.generate_recovery(CODE, TESTS, "mbpp_x")

    assert trace is not None
    assert trace.outcome == ASSERTION_FAILURE
    assert trace.initial_code == "def volume(b, h):\n    return (b + h) / 3\n"


def test_unreachable_class_is_declined_rather_than_mislabelled() -> None:
    # The breaking families rewrite a return expression. A function that never
    # returns one gives them nothing to work with, and the generator has to
    # decline rather than hand back some other class under the wrong label.
    code = "def fill(target):\n    target.append(1)\n"
    tests = "xs = []\nfill(xs)\nassert xs == [1]\n"

    for want in (RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT):
        assert recovery_generator.generate_recovery(code, tests, "mbpp_y", want=want) is None


def test_v2_dataset_covers_every_class_and_fits_the_budget() -> None:
    path = ROOT / "data" / "sft_v2" / "arm_b" / "train.jsonl"
    if not path.exists():
        pytest.skip("run: uv run python -m data.sft_v2")
    import json

    rows = [json.loads(line) for line in path.open()]
    assert len(rows) == 212
    assert len({row["case_id"] for row in rows}) == 212

    emitted: dict[str, int] = {}
    for row in rows:
        text = "\n".join(message["content"] for message in row["messages"])
        for outcome in (ASSERTION_FAILURE, RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT, PASS):
            if f"<PREDICTION>{outcome}</PREDICTION>" in text:
                emitted[outcome] = emitted.get(outcome, 0) + 1
    published = {ASSERTION_FAILURE: 37, RUNTIME_ERROR: 8, SYNTAX_ERROR: 0, TIMEOUT: 0}
    for outcome, before in published.items():
        assert emitted.get(outcome, 0) > before, f"{outcome} did not improve on {before}"

    validate_sft(path)


def test_v2_sft_config_changes_only_the_dataset() -> None:
    baseline = tomllib.loads((ROOT / "configs" / "arm_b_sft.toml").read_text())
    v2 = tomllib.loads((ROOT / "configs" / "arm_b_sft_v2.toml").read_text())

    assert v2.pop("data")["name"] == "data/sft_v2/arm_b"
    assert baseline.pop("data")["name"] == "data/sft/arm_b"
    assert v2.pop("output_dir") == "outputs/arm_b_sft_v2"
    baseline.pop("output_dir")
    v2.pop("wandb")
    baseline.pop("wandb")
    assert v2 == baseline


def test_v2_rl_config_changes_only_the_starting_checkpoint() -> None:
    """RL has to stay attributable to the checkpoint it starts from: same
    tasks, same algorithm, same sampler, same validation set."""
    baseline = tomllib.loads((ROOT / "configs" / "arm_b_rl.toml").read_text())
    v2 = tomllib.loads((ROOT / "configs" / "arm_b_rl_v2.toml").read_text())

    assert v2.pop("model")["name"] == "JayZenith/SFT_ARM_B_v2"
    assert baseline.pop("model")["name"] == "JayZenith/SFT_ARM_B"
    assert v2.pop("output_dir") == "outputs/arm_b_rl_v2"
    baseline.pop("output_dir")
    v2.pop("wandb")
    baseline.pop("wandb")
    assert v2 == baseline
