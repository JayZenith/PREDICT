# Reproduction

These are full-parameter checkpoints. A 50 GB disk is not a safe budget for
both arms and retained optimizer states; use a larger persistent volume or
archive each selected checkpoint before starting the next arm.

## 1. Prepare

```bash
git clone https://github.com/JayZenith/PREDICT.git
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
touched once for the final report" honest). Serve one checkpoint, evaluate it,
move to the next port:

```bash
vllm serve outputs/arm_a_rl/weights/step_25 \
  --served-model-name arm_a_step25 --port 8021 &
bash scripts/evaluate.sh a arm_a_step25 test \
  --client.base-url http://localhost:8021/v1 --client.api-key-var HOME
```

Repeat per checkpoint (steps 25/50/75/100 and the SFT baseline) and per arm,
each on its own port.

Results from the runs at commit `9eefac7` (n=500, greedy pass@1). Both arms
were trained twice, independently, from the same SFT checkpoint
(`inference.seed` 42 then 43 — the only difference between each pair of
runs):

| step | Arm A (seed 42) | Arm A (seed 43) | Arm B (seed 42) | Arm B (seed 43) |
|---|---:|---:|---:|---:|
| SFT | 51.6% | 51.6% (same ckpt) | 48.2% | 48.2% (same ckpt) |
| 25 | 51.4% | 50.4% | 45.2% | 47.8% |
| 50 | 52.2% | 52.8% | 50.0% | 48.6% |
| 75 | 53.6% | 54.8% | 52.6% | 51.2% |
| 100 | 56.4% | 54.2% | 52.0% | 53.6% |

Raw traces, eval/serve logs, and training artifacts (configs, W&B, trainer
logs) for both arms are archived under the gitignored
[`PREDICT_RL_RESULTS/`](../PREDICT_RL_RESULTS/) directory
(`RL_ARM_{A,B}_{25,50,75,100}/eval/` for seed 42,
`RL_ARM_{A,B}_V1_{25,50,75,100}/eval/` for seed 43, `RL_ARM_A_sft/eval/` for
the freshly re-evaluated Arm A SFT baseline, and the four `*_shared/` dirs).
HF: `RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0` (seed 42), `..._V1` (seed 43).

## 6. Statistics

Compare paired per-task pass/fail outcomes with McNemar's test
(continuity-corrected) and a paired bootstrap CI on the pass-rate difference —
never trust the raw percentage gap alone at n=500. Script:
[`docs/stats.py`](stats.py) (`python3 docs/stats.py TRACES_A.jsonl TRACES_B.jsonl`).

A first pass (one run each, seed 42) found exactly one comparison that
survived correction for multiple comparisons: "Arm A beats Arm B at step 25."
Both arms were then retrained independently with seed 43 (same SFT
checkpoint, same everything else) specifically to test whether that held up.
It didn't — see below.

**Each arm, seed 42 vs seed 43, same checkpoint step (is training
reproducible?):**

| step | Arm A diff (p) | Arm B diff (p) |
|---|---|---|
| 25 | −1.0 (p=0.44) | +2.6 (p=0.055) |
| 50 | +0.6 (p=0.74) | −1.4 (p=0.39) |
| 75 | +1.2 (p=0.44) | −1.4 (p=0.44) |
| 100 | −2.2 (p=0.20) | +1.6 (p=0.37) |

No step shows a significant difference between either arm's two independent
training runs — both arms' RL training is reasonably stable across seeds.

**Arm A RL vs its own SFT baseline, both seeds** (SFT re-evaluated fresh —
50.6% here vs the 51.6% point estimate used earlier, same fixed checkpoint,
within eval noise):

| step | seed 42 diff (p) | seed 43 diff (p) |
|---|---|---|
| 25 | +0.8 (p=0.54) | −0.2 (p=1.0) |
| 50 | +1.6 (p=0.23) | +2.2 (p=0.091) |
| 75 | +3.0 (p=0.041) | **+4.2 (p=0.0035)** |
| 100 | **+5.8 (p=0.0003)** | +3.6 (p=0.028) |

**Arm B RL vs its own SFT baseline, both seeds:**

| step | seed 42 diff (p) | seed 43 diff (p) |
|---|---|---|
| 25 | −3.0 (p=0.033) | −0.4 (p=0.88) |
| 50 | +1.8 (p=0.30) | +0.4 (p=0.90) |
| 75 | +4.4 (p=0.010) | +3.0 (p=0.064) |
| 100 | +3.8 (p=0.033) | **+5.4 (p=0.0017)** |

For both arms, step 100 vs SFT is significant in both seeds — the most
solid result in this whole project, and Arm A's is the strongest single
number here (p=0.0003). Arm A's step 75 is significant in both seeds too;
Arm B's is significant in one and marginal in the other (p=0.064). Neither
arm shows a significant step-25 effect in both seeds — the step-25
"regression" reported from Arm B's seed 42 alone did not replicate (p=0.88
in seed 43) and was noise, not a real early-RL effect.

**Arm A vs Arm B, matched by step, all four seed combinations:**

| step | A42 vs B42 | A42 vs B43 | A43 vs B42 | A43 vs B43 |
|---|---|---|---|---|
| 25 | **−6.2 (p=0.006)** | −3.6 (p=0.11) | **−5.2 (p=0.026)** | −2.6 (p=0.27) |
| 50 | −2.2 (p=0.37) | −3.6 (p=0.13) | −2.8 (p=0.25) | −4.2 (p=0.076) |
| 75 | −1.0 (p=0.71) | −2.4 (p=0.32) | −2.2 (p=0.37) | −3.6 (p=0.13) |
| 100 | −4.4 (p=0.068) | −2.8 (p=0.25) | −2.2 (p=0.39) | −0.6 (p=0.86) |

**At step 100, none of the four seed combinations are significant** — the
seed43-vs-seed43 pair is a near dead heat (54.2% vs 53.6%, p=0.86). At step
25, two of four combinations are nominally significant (p=0.006, p=0.026),
and both involve Arm B's seed-42 run specifically — its own lowest point
(45.2%, and the one seed that showed a significant dip vs its own SFT
baseline above). That reads as Arm B seed 42's step-25 checkpoint being an
outlier, not a reproducible Arm A advantage: swap in Arm B's seed-43 run at
the same step and the gap is cut roughly in half and loses significance both
times (p=0.11, p=0.27).

**No checkpoint step shows a difference between Arm A and Arm B that holds up
across seed combinations.** The step-100 headline gap that first looked like
Arm A leading (56.4% vs 52.0%) shrinks to a coin flip once both arms have a
second independent training run (54.2% vs 53.6%).

**Bottom line.** RLVR reliably improves *both* arms over their own SFT
baseline by step 100 — replicated across two independent training runs per
arm, strong evidence for each (Arm A p=0.0003 seed42 / p=0.028 seed43; Arm B
p=0.033 seed42 / p=0.0017 seed43). Whether Arm A's reactive design or Arm B's
predictive design performs *better than the other* is a separate question and
remains **unconfirmed** at every checkpoint, across every seed combination
tested — the one result that once suggested Arm A's edge traced back to a
single outlier training run, not a reproducible arm-level effect. Both arms
show good within-arm reproducibility (seed 42 vs seed 43 never significantly
differ, for either arm, at any step).

Report final pass@1, first-patch success, executed-failure recovery, visible
tests per solved task, tokens, and tool calls. For Arm B also report prediction
accuracy, six-class macro-F1, bad-patch rejection, and unnecessary rejection
of good patches.

The primary comparison is Arm A RLVR versus Arm B RLVR. Base and SFT results
show where each training stage changed behavior.

## 7. Efficiency

Seed-42 eval traces, from the harness's own `visible_tool_calls` /
`visible_test_calls` metrics plus assistant-turn character counts (a token
proxy, not exact tokenization):

| | tool calls | visible tests | assistant chars* |
|---|---:|---:|---:|
| Arm A (25/50/75/100) | 5.3–5.5 | 1.94–1.99 | 835–856 |
| Arm B (SFT/25/50/75/100) | 5.4–5.7 | 1.62–1.99 | 996–1124 |

Arm B does not use fewer turns or tool calls — slightly more. It does use
fewer visible test executions (shadow-testing on `REVISE` moves some test
cycles off the visible ledger, as designed), but spends ~20-30% more
generation length per task on `<PREDICTION>`/`<DECISION>` tags. Not a clean
efficiency win either direction — fewer visible failures, more tokens to get
there.

A related data-quality note: 3 of 2500 Arm B eval rows (steps 25/50/75, a
different task each time) hit a harness turn/token-budget truncation and
scored as fails with empty metrics; 0 of 2000 Arm A rows did. Moves no
reported number by more than 0.2 points, but is a real, Arm-B-specific
overhead signature consistent with the token-cost finding above.
