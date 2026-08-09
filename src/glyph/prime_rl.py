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



def _encoding(tokenizer: Any, text: str) -> tuple[list[int], list[tuple[int, int]]]:
    encoded = tokenizer(
        text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    return list(encoded["input_ids"]), [
        (int(start), int(end)) for start, end in encoded["offset_mapping"]
    ]


def _mask_label_tokens(
    tokenizer: Any,
    node_ids: list[int],
    branch_offset: int,
    rl_weights: list[float],
) -> int:
    """Zero the RL weight on every token that carries a sampled label.

    The label has to be found in the node's *own* ids, and re-encoding the
    label text does not find it: byte-level BPE merges straight through the
    label's edges, so ``<PREDICTION>PASS`` ends on the single token ``">P"``
    and ``PASS`` on its own is one token that appears nowhere in what was
    sampled. Decoding the node and encoding that exact string does round-trip,
    which lines the offset mapping up with ``node_ids`` one for one.

    A token that straddles the boundary takes the whole token's weight with it.
    That is the side to err on -- the structural ``>`` it also covers is fully
    determined by the format, while leaving a label token trained is the one
    thing this algorithm exists to prevent.
    """
    text = tokenizer.decode(
        node_ids, skip_special_tokens=False, clean_up_tokenization_spaces=False
    )
    ids, offsets = _encoding(tokenizer, text)
    if ids != node_ids:
        raise RuntimeError(
            "sampled node does not survive a decode/encode round trip, so its "
            "PREDICTION label cannot be located in its own token ids"
        )

    masked = 0
    for match in PREDICTION_LABEL_RE.finditer(text):
        label_start, label_end = match.span(1)
        hits = 0
        for token_idx, (start, end) in enumerate(offsets):
            if end > label_start and start < label_end:
                index = branch_offset + token_idx
                if index >= len(rl_weights):
                    raise RuntimeError(
                        "PREDICTION label token exceeds its training sample"
                    )
                rl_weights[index] = 0.0
                hits += 1
        if not hits:
            raise RuntimeError(
                "could not locate the sampled PREDICTION label "
                f"{match.group(1)!r} in its own token ids"
            )
        masked += hits
    return masked


class PredictAlgorithm(GRPOAlgorithm):
    """GRPO actions plus CE on verified pre-execution outcome labels."""

    action_loss_type = "rl"

    def __init__(self, config, policy_pool):
        super().__init__(config, policy_pool)
        self.alpha = float(config.alpha)
        self.max_aux_tokens = int(config.max_aux_tokens)
        self.renderer_config = config.renderer
        self.renderer = None
        self.tokenizer = None

    async def setup(self) -> None:
        """Build PREDICT's own renderer and tokenizer from the policy model,
        the way opsd builds its hint renderer -- PRIME-RL does not hand
        algorithms the policy's, and asking it to would mean patching the
        orchestrator. ``load_tokenizer`` reads the policy checkpoint, so the
        chat template and EOS baked into it at SFT are what the auxiliary
        sample renders through, matching the rollout it corrects.

        The tokenizer has to be a fast one: offsets are how the verified label
        is located inside its own encoding, and only the Rust backend answers
        ``return_offsets_mapping``. ``load_tokenizer`` gives us one whenever the
        checkpoint ships a ``tokenizer.json``, so check rather than assume --
        falling back to the slow backend would raise deep inside masking, one
        rollout into the run.
        """
        await super().setup()
        from renderers.base import create_renderer, load_tokenizer

        self.tokenizer = load_tokenizer(self.policy_pool.model_name)
        if not getattr(self.tokenizer, "is_fast", False):
            raise RuntimeError(
                f"{self.policy_pool.model_name} loaded a slow tokenizer; PREDICT "
                "needs offset mappings to mask the sampled label"
            )
        self.renderer = create_renderer(self.tokenizer, self.renderer_config)

    def _mask_sampled_labels(self, rollout) -> None:
        tokenizer = self.tokenizer
        if tokenizer is None:
            raise RuntimeError("PredictAlgorithm.setup() must run first")
        branches = [branch for branch, _ in iter_trainable_branches(rollout)]
        original_samples = rollout.samples[: len(branches)]
        for sample, branch in zip(original_samples, branches, strict=True):
            rl_weights = [1.0 if keep else 0.0 for keep in sample.mask]
            masked = 0
            branch_offset = 0
            for node in branch.nodes:
                content = str(getattr(node.message, "content", "") or "")
                if (
                    node.sampled
                    and node.message.role == "assistant"
                    and PREDICTION_LABEL_RE.search(content)
                ):
                    found = _mask_label_tokens(
                        tokenizer, list(node.token_ids), branch_offset, rl_weights
                    )
                    # The node's message carries a label, so its own ids have to
                    # as well. Leaving one unmasked would train the policy to
                    # keep saying whatever it guessed, which is the one thing
                    # this algorithm exists to prevent, so fail the run instead.
                    if not found:
                        raise RuntimeError(
                            "a sampled PREDICTION label is missing from the "
                            "token ids of the node that produced it"
                        )
                    masked += found
                branch_offset += len(node.token_ids)
            if masked:
                sample.rl_weights = rl_weights

    def _auxiliary_sample(self, rollout, target: dict) -> TrainingSample:
        renderer = self.renderer
        tokenizer = self.tokenizer
        if renderer is None or tokenizer is None:
            raise RuntimeError("PredictAlgorithm.setup() must run first")
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
