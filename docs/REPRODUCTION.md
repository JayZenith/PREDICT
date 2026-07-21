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

The validator must report 212 RL tasks per training arm, 40 validation tasks,
500 test tasks, and 424 verified SFT traces.

## 2. SFT

One GPU per run:

```bash
bash scripts/train_sft.sh a
bash scripts/train_sft.sh b
```

Both start from `Qwen/Qwen3-4B-Base`, see the same 212 SFT tasks for roughly
nine epochs, and use `<|im_end|>` as both the ChatML turn boundary and model EOS.
The trainer must not print a missing-EOS warning. Checkpoints write:

```text
outputs/arm_a_sft/weights/step_60
outputs/arm_b_sft/weights/step_60
```

## 3. RL

The published RLVR checkpoints were trained from commit
[`9eefac7`](https://github.com/JayZenith/PREDICT/commit/9eefac7) (`git checkout 9eefac7`
after step 1-2 above). Each run needs one training GPU and one inference GPU.
Environments use fresh host subprocess workspaces, so no Prime Sandbox key is
required:

```bash
bash scripts/train_rl.sh a
bash scripts/train_rl.sh b
```

Both run 100 steps over the same 212 disjoint RL tasks with group size 16,
batch size 64, 512 completion tokens, eight visible tool calls, binary final
reward, no reference KL, and the `zero_advantage` post-batch filter enforced
(drops GRPO groups with uniform-zero reward). Arm A starts from
`JayZenith/SFT_ARM_A`; Arm B starts from `JayZenith/SFT_ARM_B` with `λ=0.1`
(`alpha` in the config). `configs/arm_{a,b}_rl.toml` set `[ckpt] interval=25,
keep_last=4` and `[orchestrator.eval] interval=25`, so all four checkpoints
(steps 25/50/75/100) are retained and val40 runs in-loop at each. Override λ
only for validation-backed tuning:

```bash
bash scripts/train_rl.sh b \
  --orchestrator.algo.alpha 0.05 \
  --output-dir outputs/arm_b_rl_lambda_005 \
  --wandb.name arm-b-lambda-005
```

Do not compare arms trained with different non-arm settings.

Published checkpoints (HF, one repo per arm per step):
`JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0`.

## 4. Select without touching test

In-loop val40 evals run automatically during RL at each retained checkpoint
(see W&B / trainer logs). To evaluate any checkpoint by hand:

```bash
bash scripts/evaluate.sh a Qwen/Qwen3-4B-Base validation
bash scripts/evaluate.sh a outputs/arm_a_sft/weights/step_60 validation
bash scripts/evaluate.sh a outputs/arm_a_rl/weights/step_25 validation
```

`evaluate.sh` calls an OpenAI-compatible endpoint. For a checkpoint served by
your own vLLM instance (`vllm serve PATH --served-model-name NAME --port PORT`),
pass the served name plus base URL:

```bash
bash scripts/evaluate.sh a NAME test \
  --client.base-url http://localhost:PORT/v1 --client.api-key-var HOME
```

`--client.api-key-var` names an env var to read as the bearer token; any var
that exists works against an unauthenticated local vLLM server (`HOME` is a
convenient no-op choice, not a real credential).

## 5. Final test once, per checkpoint, standalone

The full 500-task test set is evaluated once per checkpoint, after the weights
save, outside the live RL loop (keeps training fast and keeps "test set
touched once for the final report" honest):

```bash
for step in 25 50 75 100; do
  vllm serve outputs/arm_a_rl/weights/step_${step} \
    --served-model-name arm_a_step${step} --port 802${step:0:1} &
  bash scripts/evaluate.sh a arm_a_step${step} test \
    --client.base-url http://localhost:802${step:0:1}/v1 --client.api-key-var HOME
done
# repeat for arm b
```

Results from the runs at commit `9eefac7` (n=500, greedy pass@1):

| step | Arm A | Arm B |
|---|---:|---:|
| SFT | 51.6% | 48.2% |
| 25 | 51.4% | 45.2% |
| 50 | 52.2% | 50.0% |
| 75 | 53.6% | 52.6% |
| 100 | 56.4% | 52.0% |

Raw traces, eval/serve logs, and training artifacts (configs, W&B, trainer
logs) for both arms are archived under the gitignored
[`PREDICT_RL_RESULTS/`](../PREDICT_RL_RESULTS/) directory
(`RL_ARM_{A,B}_{25,50,75,100}/eval/` and `RL_ARM_{A,B}_shared/`).

## 6. Statistics

Compare paired per-task pass/fail outcomes with McNemar's test
(continuity-corrected) and a paired bootstrap CI on the pass-rate difference —
never trust the raw percentage gap alone at n=500. Script:
[`docs/stats.py`](stats.py) (`python3 docs/stats.py TRACES_A.jsonl TRACES_B.jsonl`).

Confirmed (p<0.05, CI excludes 0) at commit `9eefac7`:
- Arm B step75 vs Arm B SFT: +4.4 pts (p=0.010, CI [1.2, 7.6])
- Arm B step100 vs Arm B SFT: +3.8 pts (p=0.033, CI [0.4, 7.2])
- Arm B step25 vs Arm B SFT: −3.0 pts (p=0.033, CI [−5.6, −0.6]) — early RL regresses
- Arm A step25 vs Arm B step25: −6.2 pts (p=0.006, CI [−10.6, −2.0]) — A ahead

Not significant (CI crosses 0): Arm B step50 vs SFT; Arm A vs Arm B at steps
50, 75, 100 (including the final step-100 headline gap, p=0.068).

Report final pass@1, first-patch success, executed-failure recovery, visible
tests per solved task, tokens, and tool calls. For Arm B also report prediction
accuracy, six-class macro-F1, bad-patch rejection, and unnecessary rejection
of good patches.

The primary comparison is Arm A RLVR versus Arm B RLVR. Base and SFT results
show where each training stage changed behavior.
