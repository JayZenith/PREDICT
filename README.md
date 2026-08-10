# PREDICT

**A post-training study of reactive vs. predictive coding agents.**

Two Qwen3-4B agents on the same verifier-backed MBPP environment, same tasks, same reward, same
SFT → RL pipeline.

```text
Arm A   patch → test → recover                     GRPO
Arm B   patch → predict outcome → KEEP/REVISE      GRPO + verified-label CE
```

Arm B predicts the verified outcome of its patch before executing, and that prediction decides
whether the patch survives. GRPO trains the actions and decisions; the prediction-label tokens are
held out of RL credit and trained separately with cross-entropy against the outcome the environment
verifies afterward.

[Full write-up](https://jayzenith.github.io/PREDICT/) ·
[blog.md](docs/blog.md) ·
[REPRODUCTION.md](docs/REPRODUCTION.md) ·
[research_specs.md](docs/research_specs.md) ·
[SFT_ARM_A](https://huggingface.co/JayZenith/SFT_ARM_A) ·
[SFT_ARM_B](https://huggingface.co/JayZenith/SFT_ARM_B)

## Headline result

The predictive agent did **not** beat the reactive baseline on pass@1. The useful finding is why,
and it holds on both seeds: the prediction learned, the decision gate discriminated, and rollouts
that acted on the prediction almost never recovered.

```text
step-100 greedy pass@1, 500 held-out tasks

              seed 42   seed 43
Arm A          56.4%     54.2%
Arm B          52.0%     53.6%
```

Splitting Arm B's trajectories by whether it ever chose REVISE:

```text
                     seed 42            seed 43
                    n    pass@1        n    pass@1
never revised     409     63.3%      439     60.4%
revised            91      1.1%       61      4.9%
```

Both seeds. Arm B's observed deficit concentrates in the REVISE path.

The gate had useful discrimination; REVISE was the dominant observed failure mode:

```text
chose REVISE on...      seed 42   seed 43
a genuinely bad patch    22.2%     14.3%
a good patch              8.5%      4.2%
```

## Why REVISE failed

REVISE hides the test result by design. That is what makes the prediction load-bearing. But it
still costs a tool call, and the agent gets nothing back except `patch applied`. So it rewrites a
patch it never tested, with no information about how the previous one was wrong, against a fixed
8-call budget:

```text
                mean tool calls   hit the 8-call cap
seed 42  KEEP        4.90               35%
         REVISE      7.99               99%
seed 43  KEEP        5.03               37%
         REVISE      7.95               95%
```

Effectively every REVISE trajectory exhausts its budget and terminates before it can run a test.

That makes always predicting `PASS` the rational policy, not a training failure:

```text
predict PASS  → KEEP   → test → see the real failure → repair
predict fail  → REVISE → no feedback, one call spent → run out of budget
```

RL learned the first one.

**Scope.** Splitting by REVISE is correlational, because harder patches attract REVISE. The 8.5% / 4.2%
false-positive rate on patches that would have passed is the control that makes it more than
selection bias, but it is not a controlled ablation.

## The auxiliary objective did work

The Arm B SFT checkpoint effectively predicted `PASS` for everything. After RLVR it learned a real
execution-outcome distinction:

```text
RUNTIME_ERROR precision      seed 42: 62.5%   seed 43: 64.1%
```

against a ~16% base rate, and not by over-predicting the class (18.0% of predictions against 18.2%
of real outcomes). It did not learn `ASSERTION_FAILURE` at all: 0% recall against ~57% of
outcomes. `SYNTAX_ERROR` and `TIMEOUT` were never shown during SFT and cannot be judged.

The distinction is learned during GRPO+CE training and replicates across both seeds. Attributing it
to the CE term specifically would need an `alpha = 0` ablation, which was not run. Either way, the
agent protocol is what prevents that distinction from paying for itself.

## Core idea

Example:

```text
Policy predicts:   PASS
Verified outcome:  RUNTIME_ERROR
```

The normal rollout keeps the sampled prediction but masks its prediction-label tokens from RL
credit.

PREDICT then creates a separate CE-only training sample:

```text
same rollout prefix
→ <PREDICTION>RUNTIME_ERROR</PREDICTION>
```

with:

```text
rl_weights = 0
ce_weights = alpha on verified-label tokens
```

This lets the policy learn the correct outcome even when that target never appeared in the visible
rollout.

## Why this differs from ECHO

ECHO applies CE supervision to environment observations already present in the rollout.

PREDICT instead supervises an **earlier prediction point** using the verified future outcome
obtained after execution.

```text
ECHO:
action → environment observation
         ↑ CE target already in rollout

PREDICT:
state → predicted outcome → action → verified outcome
        ↑ train this point           ↑ provides target later
```

## Agent protocol

```text
Arm A    patch → test → react
Arm B    patch → predict outcome → KEEP / REVISE → execute
```

Prediction classes:

```text
PASS  ASSERTION_FAILURE  RUNTIME_ERROR  SYNTAX_ERROR  TIMEOUT  OTHER
```

Rejected patches are still evaluated through shadow execution, so the policy receives no result
during the rollout while training can still recover the verified target afterward.

## Stack

Built on Prime Intellect's open-source post-training stack.

**Verifiers** runs and scores the coding environment. `GlyphTaskset` handles task setup,
reward/metrics, and verified trace extraction; `GlyphHarness` launches the agent program in the
sandbox; `program.py` implements the interaction loop. Verified prediction targets are recovered
from the Glyph trace and attached to the rollout metadata.

**PRIME-RL** performs training. `PredictAlgorithm` extends GRPO by masking sampled prediction-label
tokens from RL credit, constructing CE-only verified-label samples, and assigning CE weight only to
the verified prediction tokens. It registers through PRIME-RL's algorithm hook; the pinned
integration patch touches only the algorithm registry and its config schema.

**Renderer / tokenizer** builds the synthetic CE sample with exactly the same formatting and
tokenization the policy uses, so CE weight lands only on the intended tokens.

```text
src/glyph/taskset.py      Verifiers task/environment integration
src/glyph/harness.py      Sandbox harness
src/glyph/program.py      Agent loop
src/glyph/prime_rl.py     PREDICT training algorithm
src/glyph/chat.py         Chat-format consistency checks

configs/                  SFT/RL configs and chat template
scripts/                  setup, training, evaluation
patches/                  pinned PRIME-RL integration patches
tests/                    config, integration, algorithm tests
docs/                     research specification and reproduction docs
```

## Experiment

```text
Qwen3-4B-Base · MBPP · SFT → RLVR · two RL seeds per arm

212 SFT tasks   212 RL tasks   40 validation   500 final test
```

Splits are disjoint. The 500-task test set was evaluated once, after the design was frozen.

## Reproduction

Python 3.12, `uv`, 1 GPU for SFT, 2 GPUs for RL (1 train + 1 inference).

```bash
git clone https://github.com/JayZenith/PREDICT.git && cd PREDICT
bash scripts/setup.sh
uv run python -m data.prepare && uv run python -m data.validate data
bash scripts/train_sft.sh a          # and b; from Qwen/Qwen3-4B-Base
bash scripts/train_rl.sh a           # and b; 100 steps from the SFT ckpts
bash scripts/evaluate.sh a MODEL test
```

See `docs/REPRODUCTION.md` and `docs/research_specs.md`. Upstream dependencies and the PRIME-RL
commit are pinned. All headline numbers come from the four Arm A / Arm B runs across seeds 42 and
43 in [`RESULTS_PUBLISHED/`](RESULTS_PUBLISHED/).

## What this establishes

PREDICT does **not** show that explicit outcome prediction improves coding-agent pass@1. It does
show:

* verifier-derived future outcomes can supervise an earlier prediction point during RL;
* GRPO and auxiliary CE coexist cleanly through token-level loss routing;
* the policy learns an execution-outcome distinction absent from its SFT behavior, across two seeds;
* **a correct prediction is worthless if the action it gates is worse than simply observing the
  environment.**

The next environment worth testing is one where acting first is expensive, irreversible, or
delayed, so that foresight has something real to buy.
