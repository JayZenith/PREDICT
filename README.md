# PREDICT — Making a Coding Agent Predict the Outcome of Its Environment and Update Its World Model

[Blog](docs/index.html) ·
[blog.md](docs/blog.md) ·
[REPRODUCTION.md](docs/REPRODUCTION.md) ·
[research_specs.md](docs/research_specs.md) ·
[ARCHITECTURE.md](docs/ARCHITECTURE.md) ·
[agent_trace.md](docs/agent_trace.md) ·
[SFT_ARM_A](https://huggingface.co/JayZenith/SFT_ARM_A) ·
[SFT_ARM_B](https://huggingface.co/JayZenith/SFT_ARM_B) ·
[X](https://x.com/jayz3nith) ·
[jayzenith.me](https://jayzenith.me/)

**CE and GRPO working together during RLVR:** Arm B predicts whether its patch
will pass or fail, then chooses to KEEP or REVISE it. The patch is executed
either way, and cross-entropy trains the prediction against the verified
outcome, while GRPO trains the rest of the rollout but is masked from the
prediction label. Arm A is the matched test-and-recover baseline: patch, test,
react.

**Environment prediction shows limited promise:** starting from an SFT
checkpoint that always predicted PASS, RLVR produced a RUNTIME_ERROR detector
with **62.5% and 64.1% precision** across two seeds, versus a 16% base rate,
after only 8 SFT examples. This did not improve pass@1 because the test reveals
runtime errors one turn later anyway. ASSERTION_FAILURE, 57% of outcomes and
the more useful class to catch early, was never predicted correctly.

**Next:** sample the SFT checkpoint's own failures, label them by execution,
and re-SFT on that data to improve ASSERTION_FAILURE predictions.

| Arm | Loop | Training |
|---|---|---|
| A — reactive | patch → test → recover | SFT → GRPO |
| B — predictive | patch → predict outcome → `KEEP`/`REVISE` → test (or shadow-test) | SFT+prediction → GRPO + verified-label CE |

Built on [PRIME-RL](https://github.com/PrimeIntellect-ai/prime-rl) and
[Verifiers](https://github.com/PrimeIntellect-ai/verifiers) V1 environments,
over Qwen3-4B-Base on [MBPP](https://arxiv.org/abs/2108.07732). The
verified-label auxiliary CE follows
[ECHO](https://arxiv.org/abs/2605.24517) (Shrivastava et al., 2026). Arm A is a
matched comparator, not a single-variable ablation — Arm B bundles the
protocol, action space, loss, and trace format together
([limits](docs/research_specs.md#limits-to-report)).

## Results (n=500, greedy pass@1)

| step | Arm A (seed 42) | Arm A (seed 43) | Arm B (seed 42) | Arm B (seed 43) |
|---|---:|---:|---:|---:|
| SFT | 50.6% | 50.6% (same ckpt) | 48.2% | 48.2% (same ckpt) |
| RL 100 | 56.4% | 54.2% | 52.0% | 53.6% |

**RLVR improves both arms over their own SFT baseline**, replicated across both
seeds (Arm A p=0.0003 / 0.028; Arm B p=0.033 / 0.0017). **No checkpoint shows a
confirmed difference between the arms** across seed pairings (McNemar,
p=0.026–0.86): the seed-42 step-100 gap (56.4% vs 52.0%) becomes a coin flip in
seed 43 (54.2% vs 53.6%, p=0.86). Per-step numbers, stats, and the HF
checkpoint map: [REPRODUCTION.md](docs/REPRODUCTION.md),
[`RESULTS_PUBLISHED/`](RESULTS_PUBLISHED/).

## Reproduce

Python 3.12, `uv`, 1 GPU for SFT, 2 GPUs for RL (1 train + 1 inference).

```bash
git clone https://github.com/JayZenith/PREDICT.git && cd PREDICT
bash scripts/setup.sh
uv run python -m data.prepare && uv run python -m data.validate data
bash scripts/train_sft.sh a          # and b; from Qwen/Qwen3-4B-Base
bash scripts/train_rl.sh a           # and b; 100 steps from the SFT ckpts
bash scripts/evaluate.sh a MODEL test
```

Exact per-checkpoint eval commands and significance tests:
[REPRODUCTION.md](docs/REPRODUCTION.md). Development narrative and bugs found
along the way: [blog.md](docs/blog.md).
