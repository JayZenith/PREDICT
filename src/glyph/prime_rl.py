"""PREDICT's algorithm wiring for pinned PRIME-RL.

The prediction span gets no special treatment here: it's sampled like any
other assistant tokens, so plain GRPO credit (shaped by the taskset's
prediction_reward bonus) reaches it via the normal policy gradient.
"""

from __future__ import annotations

from prime_rl.orchestrator.algo.grpo import GRPOAlgorithm


class PredictAlgorithm(GRPOAlgorithm):
    """GRPO over the full rollout, including the verified-label prediction span."""

    action_loss_type = "rl"


__all__ = ["PredictAlgorithm"]
