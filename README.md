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
[`0245ce3`](https://github.com/JayZenith/PREDICT/commit/0245ce3).

```bash
git clone https://github.com/JayZenith/PREDICT.git
cd PREDICT
git checkout 0245ce3

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
checkpoint neither arm needs after a one-shot SFT run. The reference runs used
one 96 GB GPU and peaked at 76.4 GiB. Validation-split (`val40`) greedy pass@1
at step 60: Arm A 17%, Arm B 10% (see
[`PREDICT_SFT_RESULTS/`](PREDICT_SFT_RESULTS/) for full training and eval
artifacts).

### Continue to RLVR

```bash
bash scripts/train_rl.sh a
bash scripts/train_rl.sh b
```

These start directly from `JayZenith/SFT_ARM_A` and
`JayZenith/SFT_ARM_B`, respectively.

SFT uses 1024-token whole traces and aborts rather than truncating or excluding
one. RL allows 512 new tokens inside a 4096-token full trace; a truncated
training rollout is logged and dropped from its GRPO group rather than
aborting the run. Arm A and B stay in one codebase; arm-specific data and
configs prevent experimental drift.

Run validation before freezing λ. Touch the official 500-task test set once:

```bash
bash scripts/evaluate.sh a MODEL validation
bash scripts/evaluate.sh b MODEL validation

bash scripts/evaluate.sh a FINAL_MODEL test
bash scripts/evaluate.sh b FINAL_MODEL test
uv run glyph report TRACES_JSONL
```

The main result is Arm A RLVR versus Arm B RLVR. Also report base and SFT
checkpoints as controls. See [reproduction](docs/REPRODUCTION.md) for the exact
checkpoint map and [architecture](docs/ARCHITECTURE.md) for the loss path.
