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
    """One token per character, so ids and offsets are easy to reason about."""

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        return [(character, index) for index, character in enumerate(text)]

    def __call__(
        self,
        text: str,
        *,
        add_special_tokens: bool,
        return_offsets_mapping: bool,
    ) -> dict:
        assert not add_special_tokens and return_offsets_mapping
        pieces = self._pieces(text)
        return {
            "input_ids": [self._id(piece) for piece, _ in pieces],
            "offset_mapping": [
                (start, start + len(piece)) for piece, start in pieces
            ],
        }

    def _id(self, piece: str) -> int:
        return ord(piece)

    def decode(self, ids, **kwargs) -> str:
        return "".join(chr(identifier) for identifier in ids)

    def batch_decode(self, groups, **kwargs) -> list[str]:
        return [self.decode(group, **kwargs) for group in groups]


class _MergingTokenizer(_Tokenizer):
    """A tokenizer that merges ``>`` with whatever follows it.

    Qwen's byte-level BPE does exactly this: ``<PREDICTION>PASS`` ends on a
    single ``">P"`` token, and ``PASS`` on its own is one token that appears
    nowhere in the sampled ids. Anything that locates the label by re-encoding
    it and searching for that subsequence misses it here.
    """

    _vocab: dict[str, int] = {}

    def _pieces(self, text: str) -> list[tuple[str, int]]:
        pieces: list[tuple[str, int]] = []
        index = 0
        while index < len(text):
            width = 2 if text[index] == ">" and index + 1 < len(text) else 1
            pieces.append((text[index : index + width], index))
            index += width
        return pieces

    def _id(self, piece: str) -> int:
        return self._vocab.setdefault(piece, 1000 + len(self._vocab))

    def decode(self, ids, **kwargs) -> str:
        inverse = {identifier: piece for piece, identifier in self._vocab.items()}
        return "".join(inverse[identifier] for identifier in ids)


class _Renderer:
    def render_ids(self, messages, *, add_generation_prompt: bool) -> list[int]:
        assert messages and add_generation_prompt
        return [1, 2, 3]


def _load_module(monkeypatch):
    """Import glyph.prime_rl against stubs for the PRIME-RL it builds on."""

    class GRPOAlgorithm:
        action_loss_type = "rl"

        def __init__(self, config, policy_pool):
            self.policy_pool = policy_pool

        async def setup(self) -> None:
            return None

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
    return importlib.import_module("glyph.prime_rl")


def _algorithm(module, tokenizer):
    algorithm = module.PredictAlgorithm(
        SimpleNamespace(alpha=0.25, max_aux_tokens=256, renderer=None),
        SimpleNamespace(model_name="stub"),
    )
    algorithm.tokenizer = tokenizer
    algorithm.renderer = _Renderer()
    return algorithm


def test_predict_algorithm_uses_verified_label_ce_and_masks_sampled_label(
    monkeypatch,
) -> None:
    module = _load_module(monkeypatch)

    content = "<PREDICTION>PASS</PREDICTION>\n<DECISION>KEEP</DECISION>"
    second_content = "<PREDICTION>RUNTIME_ERROR</PREDICTION>"
    second_label = "RUNTIME_ERROR"
    node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content),
        sampled=True,
        token_ids=[99, *map(ord, content)],
    )
    second_node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=second_content),
        sampled=True,
        token_ids=list(map(ord, second_content)),
    )
    token_ids = [*node.token_ids, *second_node.token_ids]
    sample = _TrainingSample(
        token_ids=token_ids,
        mask=[False, *([True] * (len(token_ids) - 1))],
        logprobs=[0.0] * len(token_ids),
        temperatures=[],
        env_name="arm-b",
    )
    rollout = SimpleNamespace(
        branch=SimpleNamespace(nodes=[node, second_node]),
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
    asyncio.run(_algorithm(module, _Tokenizer()).score_rollout(rollout))

    assert len(rollout.samples) == 2
    raw, auxiliary = rollout.samples
    pass_start = content.index("PASS") + 1
    assert raw.rl_weights[pass_start : pass_start + 4] == [0.0] * 4
    assert raw.rl_weights[1] == 1.0
    second_start = len(node.token_ids) + second_content.index(second_label)
    assert (
        raw.rl_weights[second_start : second_start + len(second_label)]
        == [0.0] * len(second_label)
    )
    verified = "ASSERTION_FAILURE"
    assert sum(weight != 0 for weight in auxiliary.ce_weights) == len(verified)
    assert set(weight for weight in auxiliary.ce_weights if weight) == {0.25}
    assert auxiliary.rl_weights == [0.0] * len(auxiliary.token_ids)
    sys.modules.pop("glyph.prime_rl", None)


def test_label_is_masked_when_tokens_merge_across_its_edges(monkeypatch) -> None:
    """The label's own encoding need not appear in the ids that carried it.

    A live run died here: byte-level BPE put the label's first character in the
    same token as the closing ``>`` of the opening tag, so searching the node
    for a re-encoding of the label -- or of the whole message -- found nothing
    and every rollout was rejected.
    """
    module = _load_module(monkeypatch)
    tokenizer = _MergingTokenizer()

    content = "<PREDICTION>PASS</PREDICTION>"
    node_ids, _ = module._encoding(tokenizer, content)
    label_ids, _ = module._encoding(tokenizer, "PASS")
    assert not any(
        node_ids[start : start + len(label_ids)] == label_ids
        for start in range(len(node_ids))
    ), "the merging tokenizer must not leave the label findable by re-encoding"

    node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content),
        sampled=True,
        token_ids=node_ids,
    )
    sample = _TrainingSample(
        token_ids=list(node_ids),
        mask=[True] * len(node_ids),
        logprobs=[0.0] * len(node_ids),
        temperatures=[],
        env_name="arm-b",
    )
    rollout = SimpleNamespace(
        branch=SimpleNamespace(nodes=[node]),
        samples=[sample],
        info={"glyph": {"prediction_targets": []}},
        env_name="arm-b",
    )
    asyncio.run(_algorithm(module, tokenizer).score_rollout(rollout))

    # ">P", "A", "S", "S" -- the boundary token goes with the label it carries.
    weights = rollout.samples[0].rl_weights
    zeroed = [index for index, weight in enumerate(weights) if weight == 0.0]
    assert [tokenizer.decode([node_ids[i]]) for i in zeroed] == [">P", "A", "S", "S"]
    sys.modules.pop("glyph.prime_rl", None)


def test_unmaskable_sampled_label_fails_the_run(monkeypatch) -> None:
    """A label the masker cannot find would keep its RL credit.

    That is the one outcome the algorithm exists to prevent -- reward would
    reinforce whatever the policy guessed -- and it is invisible from the
    outside, so it has to stop the run rather than degrade it.
    """
    module = _load_module(monkeypatch)

    content = "<PREDICTION>PASS</PREDICTION>"
    # Ids that decode to something without the label, as happens when the
    # renderer the algorithm built disagrees with the one that sampled.
    node = SimpleNamespace(
        message=SimpleNamespace(role="assistant", content=content),
        sampled=True,
        token_ids=[7, 7, 7, 7],
    )
    sample = _TrainingSample(
        token_ids=list(node.token_ids),
        mask=[True] * len(node.token_ids),
        logprobs=[0.0] * len(node.token_ids),
        temperatures=[],
        env_name="arm-b",
    )
    rollout = SimpleNamespace(
        branch=SimpleNamespace(nodes=[node]),
        samples=[sample],
        info={"glyph": {"prediction_targets": []}},
        env_name="arm-b",
    )

    try:
        asyncio.run(_algorithm(module, _Tokenizer()).score_rollout(rollout))
    except RuntimeError as error:
        assert "missing from the token ids" in str(error)
    else:
        raise AssertionError("an unmaskable label must not pass silently")
    finally:
        sys.modules.pop("glyph.prime_rl", None)
