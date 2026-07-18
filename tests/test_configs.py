import tomllib
from pathlib import Path

from prime_rl.configs.rl import RLConfig
from prime_rl.configs.sft import SFTConfig

from data.validate import SFT_MAX_TOKENS
from glyph.chat import GLYPH_CHAT_TEMPLATE


ROOT = Path(__file__).resolve().parents[1]


def test_sft_config_is_small_one_gpu_run() -> None:
    config = SFTConfig.model_validate(tomllib.loads((ROOT / "configs/sft.toml").read_text()))
    assert config.model.name == "Qwen/Qwen3-4B-Base"
    assert config.max_steps == 16
    assert config.deployment.num_gpus == 1
    assert config.model.seq_len == SFT_MAX_TOKENS
    assert config.data.seq_len == SFT_MAX_TOKENS
    assert config.data.pack_function == "stack"
    assert config.renderer.name == "default"
    assert config.tokenizer.chat_template == GLYPH_CHAT_TEMPLATE


def test_rl_config_is_binary_grpo_without_extra_kl() -> None:
    config = RLConfig.model_validate(tomllib.loads((ROOT / "configs/rl.toml").read_text()))
    assert config.orchestrator.algo.type == "grpo"
    assert config.orchestrator.renderer.name == "default"
    assert config.seq_len == 4096
    assert config.orchestrator.train.sampling.max_completion_tokens == 512
    assert config.tokenizer.chat_template == GLYPH_CHAT_TEMPLATE
    assert config.trainer.loss.kl_tau == 0.0
    [environment] = config.orchestrator.train.env
    assert environment.taskset.data_path == "data/rl.jsonl"
    assert environment.taskset.task.max_trace_tokens == config.seq_len
    assert environment.harness.runtime.image == "python:3.12-slim-bookworm"
    assert environment.harness.max_tool_calls == 8
    assert config.deployment.num_train_gpus == 1
    assert config.deployment.num_infer_gpus == 1
