"""PREDICT's verified-label auxiliary objective for pinned PRIME-RL."""

from __future__ import annotations

import re
from typing import Any

from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm
from prime_rl.orchestrator.trajectories import iter_trainable_branches
from prime_rl.transport import TrainingSample

from .program import OUTCOME_CLASSES


PREDICTION_LABEL_RE = re.compile(
    r"<PREDICTION>\s*([A-Z_]+)\s*</PREDICTION>", re.DOTALL
)


def _find_subsequence(values: list[int], target: list[int]) -> int | None:
    if not target:
        return None
    end = len(values) - len(target) + 1
    for start in range(max(0, end)):
        if values[start : start + len(target)] == target:
            return start
    return None


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
    """GRPO actions plus CE on verified pre-execution outcome labels."""

    action_loss_type = "rl"

    def __init__(self, config, policy_pool):
        super().__init__(config, policy_pool)
        self.alpha = float(config.alpha)
        self.max_aux_tokens = int(config.max_aux_tokens)

    def _mask_sampled_labels(self, rollout) -> None:
        tokenizer = self.policy_tokenizer
        if tokenizer is None:
            raise RuntimeError("PREDICT requires PRIME-RL's policy tokenizer")
        branches = [branch for branch, _ in iter_trainable_branches(rollout)]
        original_samples = rollout.samples[: len(branches)]
        for sample, branch in zip(original_samples, branches, strict=True):
            rl_weights = [1.0 if keep else 0.0 for keep in sample.mask]
            masked = 0
            branch_offset = 0
            for node in branch.nodes:
                content = str(getattr(node.message, "content", "") or "")
                if node.sampled and node.message.role == "assistant":
                    matches = list(PREDICTION_LABEL_RE.finditer(content))
                    if matches:
                        content_ids, offsets = _encoding(tokenizer, content)
                        node_ids = list(node.token_ids)
                        content_start = _find_subsequence(node_ids, content_ids)
                        if content_start is None:
                            raise RuntimeError(
                                "could not align a sampled PREDICTION label to rollout tokens"
                            )
                        for match in matches:
                            label_start, label_end = match.span(1)
                            for token_idx, (start, end) in enumerate(offsets):
                                if end > label_start and start < label_end:
                                    index = branch_offset + content_start + token_idx
                                    if index >= len(rl_weights):
                                        raise RuntimeError(
                                            "PREDICTION label token exceeds its training sample"
                                        )
                                    rl_weights[index] = 0.0
                                    masked += 1
                branch_offset += len(node.token_ids)
            if masked:
                sample.rl_weights = rl_weights

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
        ce_weights = [0.0] * len(prefix_ids) + [
            self.alpha if selected else 0.0 for selected in label_mask
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
        self._mask_sampled_labels(rollout)
        state = rollout.info.get("glyph") or {}
        targets = state.get("prediction_targets") or []
        rollout.samples.extend(
            self._auxiliary_sample(rollout, target) for target in targets
        )


__all__ = ["PredictAlgorithm"]
