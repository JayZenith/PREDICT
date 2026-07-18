import asyncio
import importlib
import sys
import types
from dataclasses import dataclass
from types import SimpleNamespace


@dataclass
class _TrainingSample:
    token_ids: list[int]
    mask: list[bool]
    logprobs: list[float]
    temperatures: list[float]
    env_name: str
    rl_weights: list[float] | None = None
    ce_weights: list[float] | None = None


class _Tokenizer:
    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert not add_special_tokens and return_offsets_mapping
        return {
            "input_ids": [ord(character) for character in text],
            "offset_mapping": [(index, index + 1) for index in range(len(text))],
        }


class _Renderer:
    def render_ids(self, messages, *, add_generation_prompt: bool) -> list[int]:
        assert messages and add_generation_prompt
        return [1, 2, 3]


def test_predict_algorithm_uses_verified_label_ce_and_masks_sampled_label(
    monkeypatch,
) -> None:
    class GRPOAlgorithm:
        action_loss_type = "rl"

        def __init__(self, config, policy_pool):
            self.policy_pool = policy_pool
            self.policy_renderer = None
            self.policy_tokenizer = None

    grpo = types.ModuleType("prime_rl.orchestrator.algo.grpo")
    grpo.GRPOAlgorithm = GRPOAlgorithm
    trajectories = types.ModuleType("prime_rl.orchestrator.trajectories")
    trajectories.iter_trainable_branches = lambda rollout: iter(
        [(rollout.branch, rollout.samples[0].mask)]
    )
    transport = types.ModuleType("prime_rl.transport")
    transport.TrainingSample = _TrainingSample
    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.algo.grpo", grpo)
    monkeypatch.setitem(
        sys.modules, "prime_rl.orchestrator.trajectories", trajectories
    )
    monkeypatch.setitem(sys.modules, "prime_rl.transport", transport)
    sys.modules.pop("glyph.prime_rl", None)
    module = importlib.import_module("glyph.prime_rl")

    content = (
        "<PREDICTION>PASS</PREDICTION>\n"
        "<DECISION>KEEP</DECISION>"
    )
    node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content),
        sampled=True,
        token_ids=[99, *map(ord, content)],
    )
    sample = _TrainingSample(
        token_ids=node.token_ids,
        mask=[False, *([True] * len(content))],
        logprobs=[0.0] * (len(content) + 1),
        temperatures=[],
        env_name="arm-b",
    )
    rollout = SimpleNamespace(
        branch=SimpleNamespace(nodes=[node]),
        samples=[sample],
        info={
            "glyph": {
                "prediction_targets": [
                    {
                        "context_messages": [
                            {"role": "user", "content": "problem and candidate"}
                        ],
                        "sampled_prediction": "PASS",
                        "actual": "ASSERTION_FAILURE",
                    }
                ]
            }
        },
        env_name="arm-b",
    )
    algorithm = module.PredictAlgorithm(
        SimpleNamespace(alpha=0.25, max_aux_tokens=256),
        None,
    )
    algorithm.policy_tokenizer = _Tokenizer()
    algorithm.policy_renderer = _Renderer()
    asyncio.run(algorithm.score_rollout(rollout))

    assert len(rollout.samples) == 2
    raw, auxiliary = rollout.samples
    pass_start = content.index("PASS") + 1
    assert raw.rl_weights[pass_start : pass_start + 4] == [0.0] * 4
    assert raw.rl_weights[1] == 1.0
    verified = "ASSERTION_FAILURE"
    assert sum(weight != 0 for weight in auxiliary.ce_weights) == len(verified)
    assert set(weight for weight in auxiliary.ce_weights if weight) == {0.25}
    assert auxiliary.rl_weights == [0.0] * len(auxiliary.token_ids)
    sys.modules.pop("glyph.prime_rl", None)
