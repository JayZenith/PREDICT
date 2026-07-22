# PREDICT

Does sandbox-supervised outcome prediction before execution make a coding
agent decide better? Two matched arms, same base model, same MBPP tasks,
same budgets, same reward.

| Arm | Loop | Training |
|---|---|---|
| A — reactive | patch → test → recover | SFT → GRPO |
| B — predictive | patch → predict outcome → `KEEP`/`REVISE` → test (or shadow-test) | SFT+prediction → GRPO + verified-label CE |

Arm B predicts one class before testing — `PASS`, `ASSERTION_FAILURE`,
`RUNTIME_ERROR`, `SYNTAX_ERROR`, `TIMEOUT`, `OTHER` — then commits to `KEEP`
(runs the real test) or `REVISE` (the rejected candidate is shadow-tested,
hidden from the agent, then it patches again). The verified sandbox label
trains the prediction; the agent never sees it. Details:
[research_specs.md](docs/research_specs.md) ·
[agent_trace.md](docs/agent_trace.md) · [ARCHITECTURE.md](docs/ARCHITECTURE.md).

## Data

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

| step | Arm A | Arm B |
|---|---:|---:|
| SFT | 51.6% | 48.2% |
| RL 25 | 51.4% | 45.2% |
| RL 50 | 52.2% | 50.0% |
| RL 75 | 53.6% | 52.6% |
| RL 100 | **56.4%** | **52.0%** |

Arm A leads at every RL checkpoint, but of 8 McNemar tests run (corrected for
multiple comparisons, BH-FDR 5%), only 2 survive: Arm A beats Arm B at step 25
(p=0.006), and Arm B's RL beats its own SFT by step 75 (p=0.010). The step-100
headline gap above is **not** statistically significant (p=0.068, CI crosses
0) — and Arm B has only one training run, so none of this has been checked
against training-seed variance. Full stats, corrections, and efficiency
numbers: [REPRODUCTION.md](docs/REPRODUCTION.md),
[`PREDICT_RL_RESULTS/`](PREDICT_RL_RESULTS/).

Checkpoints: [`SFT_ARM_A`](https://huggingface.co/JayZenith/SFT_ARM_A),
[`SFT_ARM_B`](https://huggingface.co/JayZenith/SFT_ARM_B),
`JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0`.

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
