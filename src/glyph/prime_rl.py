"""PREDICT's algorithm wiring for pinned PRIME-RL.

Hybrid objective on the prediction span: the span stays fully sampled and
RL-trainable (plain GRPO credit reaches it, shaped by the taskset's
prediction_reward bonus), and on top of that we append an auxiliary CE
sample that pulls log p(verified_label) up directly. CE gives rare outcome
classes (RUNTIME_ERROR, ASSERTION_FAILURE, ...) a gradient path that doesn't
depend on the policy having sampled them, which breaks the PASS/KEEP
collapse pure on-policy RLVR hits from an SFT init that rarely samples them
in the first place; RLVR then keeps refining once those tokens have real
probability mass to shape.
"""

from __future__ import annotations

from typing import Any

from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.transport import TrainingSample

from .program import OUTCOME_CLASSES

#: ASSERTION_FAILURE dominates the failure distribution (candidates that run
#: but return the wrong answer), so RUNTIME_ERROR/SYNTAX_ERROR/TIMEOUT show up
#: far less often as a verified label -- the auxiliary CE sample only fires
#: when one appears in a rollout, so rarer classes get proportionally fewer
#: gradient updates no matter how correct-answer reward is shaped. Multiply
#: alpha up for those classes so each occurrence pulls harder, compensating
#: for how rarely they occur at all. Measured over a real run (steps 11-34,
#: 3449 verified labels): ASSERTION_FAILURE ~58%, RUNTIME_ERROR ~11%,
#: SYNTAX_ERROR ~1.4%, TIMEOUT effectively unseen -- so SYNTAX_ERROR and
#: TIMEOUT need noticeably more amplification than RUNTIME_ERROR, not the
#: same flat "rare" bucket.
_ALPHA_MULTIPLIER_BY_CLASS = {
    "PASS": 1.0,
    "ASSERTION_FAILURE": 1.0,
    "OTHER": 2.0,
    "RUNTIME_ERROR": 4.0,
    "SYNTAX_ERROR": 6.0,
    "TIMEOUT": 8.0,
}


def _encoding(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return list(encoded["input_ids"]), [
        (int(start), int(end)) for start, end in encoded["offset_mapping"]
    ]


class PredictAlgorithm(GRPOAlgorithm):
    """GRPO over the full rollout plus a CE auxiliary sample on the verified label."""

    action_loss_type = "rl"

    def __init__(self, config, policy_pool):
        super().__init__(config, policy_pool)
        self.alpha = float(config.alpha)
        self.max_aux_tokens = int(config.max_aux_tokens)

    def _auxiliary_sample(self, rollout, target: dict) -> TrainingSample:
        renderer = self.policy_renderer
        tokenizer = self.policy_tokenizer
        if renderer is None or tokenizer is None:
            raise RuntimeError(
                "PREDICT requires PRIME-RL's policy renderer and tokenizer"
            )
        context = target.get("context_messages")
        actual = target.get("actual")
        if not isinstance(context, list) or actual not in OUTCOME_CLASSES:
            raise ValueError("invalid PREDICT auxiliary target in trace.info")

        prefix_ids = list(renderer.render_ids(context, add_generation_prompt=True))
        opening = "<PREDICTION>"
        continuation = f"{opening}{actual}</PREDICTION>"
        continuation_ids, offsets = _encoding(tokenizer, continuation)
        label_start = len(opening)
        label_end = label_start + len(actual)
        label_mask = [
            end > label_start and start < label_end for start, end in offsets
        ]
        if not any(label_mask):
            raise RuntimeError("verified prediction label produced no CE tokens")

        token_ids = [*prefix_ids, *continuation_ids]
        if len(token_ids) > self.max_aux_tokens:
            raise RuntimeError(
                f"PREDICT auxiliary sample has {len(token_ids)} tokens; "
                f"limit is {self.max_aux_tokens}"
            )
        class_alpha = self.alpha * _ALPHA_MULTIPLIER_BY_CLASS.get(actual, 1.0)
        ce_weights = [0.0] * len(prefix_ids) + [
            class_alpha if selected else 0.0 for selected in label_mask
        ]
        return TrainingSample(
            token_ids=token_ids,
            mask=[False] * len(token_ids),
            logprobs=[0.0] * len(token_ids),
            temperatures=[],
            env_name=rollout.env_name,
            rl_weights=[0.0] * len(token_ids),
            ce_weights=ce_weights,
        )

    async def score_rollout(self, rollout) -> None:
        state = rollout.info.get("glyph") or {}
        targets = state.get("prediction_targets") or []
        rollout.samples.extend(
            self._auxiliary_sample(rollout, target) for target in targets
        )


__all__ = ["PredictAlgorithm"]
