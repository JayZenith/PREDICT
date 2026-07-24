# PREDICT — Toward a Ground-Truth-Grounded World Model for Coding Agents

How does adding explicit outcome prediction and keep-or-revise control change
a long-horizon coding agent? Arm A is a standard test-and-recover system
trained with SFT and RLVR. Arm B adds a custom prediction-and-decision trace
trained with verified-label cross-entropy alongside RLVR. The arms are
comparative systems, not a single-variable ablation.

| Arm | Loop | Training |
|---|---|---|
| A — reactive | patch → test → recover | SFT → GRPO |
| B — predictive | patch → predict outcome → `KEEP`/`REVISE` → test (or shadow-test) | SFT+prediction → GRPO + verified-label CE |

Arm B predicts one class before testing — `PASS`, `ASSERTION_FAILURE`,
`RUNTIME_ERROR`, `SYNTAX_ERROR`, `TIMEOUT`, `OTHER` — then commits to `KEEP`
(runs the real test) or `REVISE` (the rejected candidate is shadow-tested,
hidden from the agent, then it patches again). The verified execution label
trains the prediction; the agent never sees it. Details:
[research_specs.md](docs/research_specs.md) ·
[agent_trace.md](docs/agent_trace.md) · [ARCHITECTURE.md](docs/ARCHITECTURE.md).

**Scope**: Arm A is a matched comparator, not a single-variable ablation.
Arm B bundles the predict/decide protocol, an added action space, the
auxiliary CE loss, and a different SFT trace format together — this compares
two complete systems, it does not isolate "prediction" as one causal factor.
Full caveat: [research_specs.md § Limits to report](docs/research_specs.md#limits-to-report).

## Data

MBPP (Mostly Basic Python Problems): ~1,000 short, crowd-sourced Python
problems, each with a description and gold-standard test assertions, from
Austin et al., ["Program Synthesis with Large Language Models"](https://arxiv.org/abs/2108.07732)
(2021). Used here as the source task distribution. RLVR-on-MBPP-with-small-Qwen
setup directly inspired by Skopin & Kotelnikov,
["Improving Small Language Models for Code Generation with Reinforcement
Learning from Verification Feedback"](https://arxiv.org/abs/2605.30478) (2026).
The verified-label auxiliary CE design was inspired by Shrivastava, Kauffmann,
Awadallah & Papailiopoulos, ["ECHO: Terminal Agents Learn World Models for
Free"](https://arxiv.org/abs/2605.24517) (2026), which trains a
complementary cross-entropy loss on environment-observation tokens within the
same GRPO rollout; PREDICT's difference from ECHO is detailed in
[research_specs.md § Novelty relative to ECHO](docs/research_specs.md#novelty-relative-to-echo).
Official MBPP, split by seed 42, disjoint SFT/RL/validation/test:

| Split | Task IDs | n | Use |
|---|---|---:|---|
| Train pool | 601–974 + 50 moved from validation | 424 | → 212 SFT + 212 RL |
| Validation | remaining 511–600 | 40 | checkpoint/λ selection |
| Test | 11–510 | 500 | final eval, touched once |

212 SFT traces per arm: 142 direct-success + 70 recovery (25 shadow / 25
visible one-step, 10 deep-shadow / 10 deep-visible two-step for Arm B). No
MBPP+, no frontier screening.

## Results (n=500, greedy pass@1)

Both arms were trained twice, independently, from the same SFT checkpoints
with different seeds, to check whether any Arm A vs Arm B difference
replicates:

| step | Arm A (seed 42) | Arm A (seed 43) | Arm B (seed 42) | Arm B (seed 43) |
|---|---:|---:|---:|---:|
| SFT | 50.6% | 50.6% (same ckpt) | 48.2% | 48.2% (same ckpt) |
| RL 25 | 51.4% | 50.4% | 45.2% | 47.8% |
| RL 50 | 52.2% | 52.8% | 50.0% | 48.6% |
| RL 75 | 53.6% | 54.8% | 52.6% | 51.2% |
| RL 100 | 56.4% | 54.2% | 52.0% | 53.6% |

The first seed pair made Arm A look ahead at step 100 (56.4% vs 52.0%) and
significantly ahead at step 25 (p=0.006). Neither holds up: **across all four
seed combinations, no checkpoint step shows a statistically confirmed
difference between Arm A and Arm B** (McNemar, p=0.026-0.86) — the step-100
gap shrinks to a coin flip in the seed43-vs-seed43 pairing (54.2% vs 53.6%,
p=0.86), and the step-25 "win" traces back to Arm B's seed-42 run being its
own outlier, not a reproducible effect. What *is* solid: **RLVR improves both
arms over their own SFT baseline by step 100**, replicated across both seeds
for each arm (Arm A: p=0.0003 and p=0.028; Arm B: p=0.033 and p=0.0017). Both
arms show good within-arm reproducibility (seed 42 vs seed 43 never
significantly differ, either arm, any step). Full stats and efficiency
numbers: [REPRODUCTION.md](docs/REPRODUCTION.md),
[`PREDICT_RL_RESULTS/`](PREDICT_RL_RESULTS/).

Checkpoints: [`SFT_ARM_A`](https://huggingface.co/JayZenith/SFT_ARM_A),
[`SFT_ARM_B`](https://huggingface.co/JayZenith/SFT_ARM_B),
`JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0` (seed 42), plus
`RLVR_ARM_{A,B}_STEP{25,50,75,100}_V1` (seed-43 replication runs).

## Reproduce

Python 3.12, `uv`, 1 GPU for SFT, 2 GPUs for RL (1 train + 1 inference).

```bash
git clone https://github.com/JayZenith/PREDICT.git
cd PREDICT
bash scripts/setup.sh
uv run python -m data.prepare
uv run python -m data.validate data
```

SFT (commit [`6884983`](https://github.com/JayZenith/PREDICT/commit/6884983)),
from `Qwen/Qwen3-4B-Base`:

```bash
bash scripts/train_sft.sh a   # -> outputs/arm_a_sft/weights/step_60
bash scripts/train_sft.sh b   # -> outputs/arm_b_sft/weights/step_60
```

RLVR (commit [`9eefac7`](https://github.com/JayZenith/PREDICT/commit/9eefac7)),
from the SFT checkpoints above — 100 steps, group size 16, batch 64,
checkpoints every 25 steps (all 4 kept), val40 evaluated in-loop at each:

```bash
bash scripts/train_rl.sh a
bash scripts/train_rl.sh b
```

```bash
bash scripts/evaluate.sh a MODEL validation   # or: test
```

Exact per-checkpoint eval commands, HF checkpoint map, and the significance
tests behind the results table: [REPRODUCTION.md](docs/REPRODUCTION.md).
Development narrative and bugs found along the way: [blog.md](docs/blog.md).
