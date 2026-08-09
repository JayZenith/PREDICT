import os
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(
    0, str(ROOT / ".vendor/prime-rl/packages/prime-rl-configs/src")
)

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

from data.validate import SFT_MAX_TOKENS
from glyph.chat import GLYPH_CHAT_TEMPLATE


def _read(name: str) -> dict:
    return tomllib.loads((ROOT / "configs" / name).read_text())


def test_matched_sft_configs_are_one_gpu_and_full_trace() -> None:
    configs = [
        SFTConfig.model_validate(_read(f"arm_{arm}_sft.toml"))
        for arm in ("a", "b")
    ]
    for arm, config in zip(("a", "b"), configs, strict=True):
        assert config.model.name == "Qwen/Qwen3-4B-Base"
        assert config.max_steps == 60
        assert config.deployment.num_gpus == 1
        assert config.model.seq_len == SFT_MAX_TOKENS
        assert config.data.seq_len == SFT_MAX_TOKENS
        assert config.data.name == f"data/sft/arm_{arm}"
        assert config.data.pack_function == "stack"
        assert config.data.loss_mask.assistant
        assert not config.data.loss_mask.tool
        assert config.renderer is None
        assert config.tokenizer.chat_template == "configs/chat_template.jinja"
        assert config.tokenizer.eos_token == "<|im_end|>"
        assert (
            ROOT / config.tokenizer.chat_template
        ).read_text().rstrip("\n") == GLYPH_CHAT_TEMPLATE
    assert configs[0].optim == configs[1].optim
    assert configs[0].scheduler == configs[1].scheduler


def test_matched_rl_configs_differ_only_in_arm_and_algorithm() -> None:
    arm_a = RLConfig.model_validate(_read("arm_a_rl.toml"))
    arm_b_raw = _read("arm_b_rl.toml")
    assert arm_a.orchestrator.algo.type == "grpo"
    assert arm_a.model.name == "JayZenith/SFT_ARM_A"
    assert arm_b_raw["model"]["name"] == "JayZenith/SFT_ARM_B"
    assert arm_b_raw["orchestrator"]["algo"] == {
        "type": "predict",
        "alpha": 0.1,
        "max_aux_tokens": 4096,
        "renderer": {"name": "default"},
    }
    # PredictAlgorithm builds its own renderer in setup(); if it disagreed with
    # the orchestrator's, the auxiliary prefix would tokenize differently from
    # the rollout whose label it corrects.
    assert (
        arm_b_raw["orchestrator"]["algo"]["renderer"]
        == arm_b_raw["orchestrator"]["renderer"]
    )
    assert arm_a.seq_len == 4096
    assert arm_a.orchestrator.train.sampling.max_completion_tokens == 512
    assert arm_a.orchestrator.train.sampling.extra_body == {
        "stop_token_ids": [151645],
        "top_k": 20,
    }
    assert arm_a.tokenizer.chat_template == "configs/chat_template.jinja"
    assert arm_a.tokenizer.eos_token == "<|im_end|>"
    assert (
        ROOT / arm_a.tokenizer.chat_template
    ).read_text().rstrip("\n") == GLYPH_CHAT_TEMPLATE
    assert arm_a.trainer.loss.kl_tau == 0.0
    assert arm_a.deployment.num_train_gpus == 1
    assert arm_a.deployment.num_infer_gpus == 1

    for arm in ("a", "b"):
        raw = _read(f"arm_{arm}_rl.toml")
        [train] = raw["orchestrator"]["train"]["env"]
        [validation] = raw["orchestrator"]["eval"]["env"]
        assert train["taskset"]["data_path"] == f"data/arm_{arm}_train.jsonl"
        assert validation["taskset"]["data_path"] == (
            f"data/arm_{arm}_validation.jsonl"
        )
        assert train["harness"]["arm"] == arm
        assert train["harness"]["max_tool_calls"] == 8
        assert raw["orchestrator"]["train"]["sampling"]["extra_body"] == {
            "stop_token_ids": [151645],
            "top_k": 20,
        }
        assert raw["orchestrator"]["eval"]["sampling"]["extra_body"] == {
            "stop_token_ids": [151645]
        }
        assert train["harness"]["runtime"] == {"type": "subprocess"}
        assert validation["harness"]["runtime"] == {"type": "subprocess"}


def test_seeded_rl_config_changes_only_the_training_taskset() -> None:
    """The seeded run has to be attributable to the tasks it starts from.

    Anything else that drifted -- the checkpoint it initialises from, the
    algorithm, the sampler, the validation set it is scored on -- would give
    the comparison against the published Arm B run a second explanation.
    """
    baseline = _read("arm_b_rl.toml")
    seeded = _read("arm_b_rl_seeded.toml")
    RLConfig.model_validate(_read("arm_b_rl_seeded.toml"))

    [baseline_train] = baseline["orchestrator"]["train"].pop("env")
    [seeded_train] = seeded["orchestrator"]["train"].pop("env")
    assert seeded_train.pop("taskset")["data_path"] == "data/arm_b_train_seeded.jsonl"
    assert baseline_train.pop("taskset")["data_path"] == "data/arm_b_train.jsonl"
    assert seeded_train.pop("name") == "arm-b-train-seeded"
    assert baseline_train.pop("name") == "arm-b-train"
    assert seeded_train == baseline_train

    # Distinct output and run names keep the two runs from overwriting each
    # other; nothing else about the recipe may differ.
    for key, value in (("output_dir", "outputs/arm_b_rl_seeded"), ):
        assert seeded.pop(key) == value
        baseline.pop(key)
    assert seeded.pop("wandb")["name"] == "arm-b-predict-seeded"
    assert baseline.pop("wandb")["name"] == "arm-b-predict"
    assert seeded == baseline
    assert seeded["model"]["name"] == "JayZenith/SFT_ARM_B"


def test_predict_patch_only_registers_the_algorithm() -> None:
    """PREDICT extends PRIME-RL through its documented hook -- a new algorithm
    class plus its config, registered. Everything else (building the renderer,
    the tokenizer) belongs in our tree and happens in ``Algorithm.setup()``.
    Touching the orchestrator, envs, or the Algorithm base to thread the
    policy's own renderer through would make the integration unforkable.
    """
    patch = (ROOT / "patches/prime-rl-predict.patch").read_text()
    touched = {
        line.split(" b/")[-1].strip()
        for line in patch.splitlines()
        if line.startswith("diff --git ")
    }
    assert touched == {
        "packages/prime-rl-configs/src/prime_rl/configs/algorithm.py",
        "src/prime_rl/orchestrator/algo/__init__.py",
    }


def test_patched_prime_rl_schema_accepts_predict_algorithm() -> None:
    env = {
        **os.environ,
        "PYTHONPATH": str(
            ROOT / ".vendor/prime-rl/packages/prime-rl-configs/src"
        ),
    }
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import tomllib,sys;"
                "from prime_rl.configs.rl import RLConfig;"
                "c=RLConfig.model_validate(tomllib.load(open(sys.argv[1],'rb')));"
                "assert c.orchestrator.algo.type=='predict';"
                "assert c.orchestrator.algo.alpha==0.1"
            ),
            str(ROOT / "configs/arm_b_rl.toml"),
        ],
        env=env,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr
