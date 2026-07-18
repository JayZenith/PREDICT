import os
import subprocess
import sys
import tomllib
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

from data.validate import SFT_MAX_TOKENS
from glyph.chat import GLYPH_CHAT_TEMPLATE


ROOT = Path(__file__).resolve().parents[1]


def _read(name: str) -> dict:
    return tomllib.loads((ROOT / "configs" / name).read_text())


def test_matched_sft_configs_are_one_gpu_and_full_trace() -> None:
    configs = [
        SFTConfig.model_validate(_read(f"arm_{arm}_sft.toml"))
        for arm in ("a", "b")
    ]
    for arm, config in zip(("a", "b"), configs, strict=True):
        assert config.model.name == "Qwen/Qwen3-4B-Base"
        assert config.max_steps == 24
        assert config.deployment.num_gpus == 1
        assert config.model.seq_len == SFT_MAX_TOKENS
        assert config.data.seq_len == SFT_MAX_TOKENS
        assert config.data.name == f"data/sft/arm_{arm}"
        assert config.data.pack_function == "stack"
        assert config.data.loss_mask.assistant
        assert not config.data.loss_mask.tool
        assert config.renderer.name == "default"
        assert config.tokenizer.chat_template == GLYPH_CHAT_TEMPLATE
    assert configs[0].optim == configs[1].optim
    assert configs[0].scheduler == configs[1].scheduler


def test_matched_rl_configs_differ_only_in_arm_and_algorithm() -> None:
    arm_a = RLConfig.model_validate(_read("arm_a_rl.toml"))
    arm_b_raw = _read("arm_b_rl.toml")
    assert arm_a.orchestrator.algo.type == "grpo"
    assert arm_b_raw["orchestrator"]["algo"] == {
        "type": "predict",
        "alpha": 0.1,
        "max_aux_tokens": 4096,
    }
    assert arm_a.seq_len == 4096
    assert arm_a.orchestrator.train.sampling.max_completion_tokens == 512
    assert arm_a.tokenizer.chat_template == GLYPH_CHAT_TEMPLATE
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
        assert train["harness"]["runtime"]["image"] == (
            "python:3.12-slim-bookworm"
        )


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
