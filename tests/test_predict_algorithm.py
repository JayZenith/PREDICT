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


def test_predict_algorithm_adds_verified_label_ce_without_masking_sampled_span(
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
    transport = types.ModuleType("prime_rl.transport")
    transport.TrainingSample = _TrainingSample
    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.algo.grpo", grpo)
    monkeypatch.setitem(sys.modules, "prime_rl.transport", transport)
    sys.modules.pop("glyph.prime_rl", None)
    module = importlib.import_module("glyph.prime_rl")

    content = "<PREDICTION>PASS</PREDICTION>\n<DECISION>KEEP</DECISION>"
    token_ids = [99, *map(ord, content)]
    original_rl_weights = [1.0] * len(token_ids)
    sample = _TrainingSample(
        token_ids=token_ids,
        mask=[False, *([True] * (len(token_ids) - 1))],
        logprobs=[0.0] * len(token_ids),
        temperatures=[],
        env_name="arm-b",
        rl_weights=list(original_rl_weights),
    )
    rollout = SimpleNamespace(
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

    # The sampled prediction span keeps its full RL weight -- RLVR still
    # trains on it directly, on top of the CE auxiliary sample.
    assert raw.rl_weights == original_rl_weights

    verified = "ASSERTION_FAILURE"
    assert sum(weight != 0 for weight in auxiliary.ce_weights) == len(verified)
    assert set(weight for weight in auxiliary.ce_weights if weight) == {0.25}
    assert auxiliary.rl_weights == [0.0] * len(auxiliary.token_ids)
    sys.modules.pop("glyph.prime_rl", None)


def test_predict_algorithm_amplifies_alpha_for_rare_failure_classes(
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
    transport = types.ModuleType("prime_rl.transport")
    transport.TrainingSample = _TrainingSample
    monkeypatch.setitem(sys.modules, "prime_rl.orchestrator.algo.grpo", grpo)
    monkeypatch.setitem(sys.modules, "prime_rl.transport", transport)
    sys.modules.pop("glyph.prime_rl", None)
    module = importlib.import_module("glyph.prime_rl")

    algorithm = module.PredictAlgorithm(
        SimpleNamespace(alpha=0.25, max_aux_tokens=256),
        None,
    )
    algorithm.policy_tokenizer = _Tokenizer()
    algorithm.policy_renderer = _Renderer()

    def ce_weight_for(actual: str) -> float:
        rollout = SimpleNamespace(env_name="arm-b")
        sample = algorithm._auxiliary_sample(
            rollout,
            {
                "context_messages": [
                    {"role": "user", "content": "problem and candidate"}
                ],
                "actual": actual,
            },
        )
        return max(sample.ce_weights)

    # ASSERTION_FAILURE is the dominant failure class -- no amplification.
    assert ce_weight_for("ASSERTION_FAILURE") == 0.25
    # RUNTIME_ERROR/SYNTAX_ERROR show up far less often as a verified label,
    # so each occurrence needs to pull harder to get comparable gradient mass.
    assert ce_weight_for("RUNTIME_ERROR") == 0.5
    assert ce_weight_for("SYNTAX_ERROR") == 0.5
    sys.modules.pop("glyph.prime_rl", None)
