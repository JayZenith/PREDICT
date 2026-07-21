# PREDICT

PREDICT tests one claim: **does sandbox-supervised consequence prediction help
a coding agent make better decisions before execution?**

It extends the MBPP unit-test GRPO experiment in
[arXiv:2605.30478](https://arxiv.org/abs/2605.30478) from single-shot function
generation to a verified multi-turn agent.

Both agents solve the same official MBPP tasks with the same Qwen3-4B base,
ChatML protocol, budgets, optimizer, visible test assertions in the task
prompt, and binary final reward.

| Arm | Agent loop | Training |
|---|---|---|
| A | patch → test → recover | agent SFT → GRPO |
| B | patch → predict outcome → KEEP or REVISE | matched SFT → GRPO + verified-label CE |

Arm B predicts one objective class before testing:
`PASS`, `ASSERTION_FAILURE`, `RUNTIME_ERROR`, `SYNTAX_ERROR`, `TIMEOUT`, or
`OTHER`. Rejected patches are shadow-tested. The result trains the prediction
label but is never shown to the agent.

Read the [experiment specification](docs/research_specs.md) and
[canonical traces](docs/agent_trace.md).

## Fixed data

| Official MBPP split | IDs | Use |
|---|---:|---|
| Train | 601–974 plus 50 seed-selected validation IDs (424) | split into disjoint SFT/RL task pools |
| Validation | remaining 40 seed-selected IDs from 511–600 | tune λ and check overfitting |
| Test | 11–510 (500) | one final pass@1 evaluation |

Each arm gets 212 SFT traces (142 direct-success, 70 verified one-recovery).
Arm B's 70 recovery traces split evenly across both recovery modes: 35 shadow
(predict failure → `REVISE` → fix → predict `PASS` → `KEEP` → pass) and 35
visible (predict `PASS` as an honest mistake → `KEEP` → visible failure → fix
→ predict `PASS` → `KEEP` → pass). The 212 SFT task IDs and 212 RL task IDs
are disjoint. There is no frontier screen, MBPP+, or two-step synthetic
recovery.

## Train

Requirements: Python 3.12, `uv`, one GPU for SFT, and two GPUs for RL.
RL environments run as isolated host subprocess workspaces on the training
instance; no Prime Sandbox access is required.

### Reproduce the published SFT models

The published [Arm A](https://huggingface.co/JayZenith/SFT_ARM_A) and
[Arm B](https://huggingface.co/JayZenith/SFT_ARM_B) checkpoints were trained
from commit
[`6884983`](https://github.com/JayZenith/PREDICT/commit/6884983).

```bash
git clone https://github.com/JayZenith/PREDICT.git
cd PREDICT
git checkout 6884983

bash scripts/setup.sh
uv run python -m data.prepare
uv run python -m data.validate data

bash scripts/train_sft.sh a
bash scripts/train_sft.sh b
```

Each command starts from `Qwen/Qwen3-4B-Base` and writes its final nine-epoch
checkpoint (60 steps × batch 32 over 212 traces) to
`outputs/arm_a_sft/weights/step_60` or `outputs/arm_b_sft/weights/step_60`.
Nine epochs is what it took for `<|im_end|>` to become the dominant sampled
token at turn boundaries; at five epochs (step 30) the model still emitted
malformed tokens under RL sampling temperature. `[ckpt] weights_only = true`
keeps only the ~7.6 GB HF export, skipping the ~47 GB full optimizer-state
checkpoint neither arm needs after a one-shot SFT run. `seq_len = 1280` (not
1024): on RTX PRO 6000 Blackwell, `seq_len=1024` specifically triggers a CUDA
illegal-memory-access under `torch.compile`, bisected as a narrow bug in the
pinned torch/PRIME-RL stack, not this repo — 768 and 1280 both run cleanly.
The reference runs used one 96 GB GPU and peaked at 76.4 GiB. Validation-split
(`val40`) greedy pass@1 at step 60: Arm A 60%, Arm B 53% (prediction accuracy
48%) — both up sharply from an earlier blind-function-signature harness
design (17%/10%); see [`PREDICT_SFT_RESULTS/`](PREDICT_SFT_RESULTS/) for full
training and eval artifacts.

### Continue to RLVR

The published RLVR checkpoints were trained from commit
[`9eefac7`](https://github.com/JayZenith/PREDICT/commit/9eefac7):

```bash
git checkout 9eefac7
bash scripts/train_rl.sh a
bash scripts/train_rl.sh b
```

These start directly from `JayZenith/SFT_ARM_A` and `JayZenith/SFT_ARM_B`.
100 steps, group size 16, batch size 64, `zero_advantage` filter enforced,
checkpoints every 25 steps with all four (25/50/75/100) retained and val40
evaluated in-loop at each. SFT uses 1280-token whole traces and aborts rather
than truncating or excluding one. RL allows 512 new tokens inside a
4096-token full trace; a truncated training rollout is logged and dropped
from its GRPO group rather than aborting the run. Arm A and B stay in one
codebase; arm-specific data and configs prevent experimental drift.

Run validation before freezing λ. Touch the official 500-task test set once
per checkpoint, standalone after the weights save (not wired into the live
RL loop):

```bash
bash scripts/evaluate.sh a MODEL validation
bash scripts/evaluate.sh a NAME test --client.base-url http://localhost:PORT/v1 --client.api-key-var HOME
uv run glyph report TRACES_JSONL
```

Published checkpoints:
`JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0`. Final (step 100, n=500,
greedy pass@1): **Arm A 56.4%, Arm B 52.0%** — Arm A leads at every
checkpoint, but McNemar + bootstrap CI show only the step-25 gap is
statistically solid (p=0.006); the step-100 headline gap is suggestive, not
confirmed (p=0.068, CI crosses zero). See
[reproduction](docs/REPRODUCTION.md) for the full per-checkpoint table, exact
commands, and statistics, and [architecture](docs/ARCHITECTURE.md) for the
loss path.
