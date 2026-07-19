# Reproduction

These are full-parameter checkpoints. A 50 GB disk is not a safe budget for
both arms and retained optimizer states; use a larger persistent volume or
archive each selected checkpoint before starting the next arm.

## 1. Prepare

```bash
git clone git@github.com:JayZenith/PREDICT.git
cd PREDICT
bash scripts/setup.sh
uv run python -m data.prepare
uv run python -m data.validate data
uv run pytest
```

The validator must report 374 tasks per training arm, 90 validation tasks, 500
test tasks, and 748 verified SFT traces.

## 2. SFT

One GPU per run:

```bash
bash scripts/train_sft.sh a
bash scripts/train_sft.sh b
```

Both start from `Qwen/Qwen3-4B-Base`, see the same 374 tasks for roughly five
epochs, and use `<|im_end|>` as both the ChatML turn boundary and model EOS.
The trainer must not print a missing-EOS warning. Checkpoints write:

```text
outputs/arm_a_sft/weights/step_60
outputs/arm_b_sft/weights/step_60
```

## 3. RL

Each run needs one training GPU and one inference GPU. Environments use fresh
host subprocess workspaces, so no Prime Sandbox key is required:

```bash
bash scripts/train_rl.sh a
bash scripts/train_rl.sh b
```

Both run 94 updates over the same 374 tasks with group size 8, batch size 32,
512 completion tokens, eight visible tool calls, binary final reward, and no
reference KL. Arm A starts from `JayZenith/SFT_ARM_A`; Arm B starts from
`JayZenith/SFT_ARM_B`. Arm B starts with `λ=0.1` (`alpha` in the config); override it
only for validation-backed tuning:

```bash
bash scripts/train_rl.sh b \
  --orchestrator.algo.alpha 0.05 \
  --output-dir outputs/arm_b_rl_lambda_005 \
  --wandb.name arm-b-lambda-005
```

Do not compare arms trained with different non-arm settings.

## 4. Select without touching test

Evaluate base, SFT, and RL checkpoints on the 90 validation tasks:

```bash
bash scripts/evaluate.sh a Qwen/Qwen3-4B-Base validation
bash scripts/evaluate.sh b Qwen/Qwen3-4B-Base validation
bash scripts/evaluate.sh a outputs/arm_a_sft/weights/step_60 validation
bash scripts/evaluate.sh b outputs/arm_b_sft/weights/step_60 validation
bash scripts/evaluate.sh a ARM_A_RL_CHECKPOINT validation
bash scripts/evaluate.sh b ARM_B_RL_CHECKPOINT validation
```

Freeze λ and checkpoint choices from validation. Record the choice before the
final run. `evaluate.sh` calls an OpenAI-compatible endpoint; pass
`--client.base-url` and `--client.api-key-var` when evaluating a checkpoint
served by your own vLLM instance.

## 5. Final test once

```bash
bash scripts/evaluate.sh a ARM_A_FINAL test
bash scripts/evaluate.sh b ARM_B_FINAL test

uv run glyph report ARM_A_TRACES
uv run glyph report ARM_B_TRACES
```

Report final pass@1, first-patch success, executed-failure recovery, visible
tests per solved task, tokens, and tool calls. For Arm B also report prediction
accuracy, six-class macro-F1, bad-patch rejection, and unnecessary rejection
of good patches.

The primary comparison is Arm A RLVR versus Arm B RLVR. Base and SFT results
show where each training stage changed behavior.
