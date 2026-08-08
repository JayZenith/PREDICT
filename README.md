# PREDICT

PREDICT adds **verified pre-action outcome supervision** to GRPO for tool-using agents.

Before executing an action, the policy predicts its outcome and decides whether to KEEP or REVISE
it. After execution, the environment provides the verified outcome. GRPO trains the agent's
actions, while an auxiliary CE objective trains the earlier prediction against that verified
target.

[Full write-up](https://jayzenith.github.io/PREDICT/) ·
[blog.md](docs/blog.md) ·
[REPRODUCTION.md](docs/REPRODUCTION.md) ·
[research_specs.md](docs/research_specs.md) ·
[SFT_ARM_A](https://huggingface.co/JayZenith/SFT_ARM_A) ·
[SFT_ARM_B](https://huggingface.co/JayZenith/SFT_ARM_B)

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

### Arm A

```text
patch → test → react
```

### Arm B / PREDICT

```text
patch → predict outcome → KEEP / REVISE → execute
```

Prediction classes:

```text
PASS
ASSERTION_FAILURE
RUNTIME_ERROR
SYNTAX_ERROR
TIMEOUT
OTHER
```

Rejected patches can still be evaluated through shadow execution, so the policy receives no result
during the rollout while training can still recover the verified target afterward.

## Stack

PREDICT is built on Prime Intellect's open-source post-training stack.

### Verifiers

Runs and scores the coding environment.

```text
src/glyph/taskset.py
src/glyph/harness.py
src/glyph/program.py
```

`GlyphTaskset` handles task setup, reward/metrics, and verified trace extraction. `GlyphHarness`
launches the agent program in the sandbox. `program.py` implements the coding-agent interaction
loop. Verified prediction targets are recovered from the Glyph trace and attached to the rollout
metadata.

### PRIME-RL

Performs model training.

```text
src/glyph/prime_rl.py
```

`PredictAlgorithm` extends GRPO by:

1. masking sampled prediction-label tokens from RL credit;
2. constructing CE-only verified-label samples;
3. assigning CE weight only to the verified prediction tokens.

It registers through PRIME-RL's algorithm hook; the pinned integration patch touches only the
algorithm registry and its config schema.

### Renderer / tokenizer

PREDICT uses the policy's chat template and tokenizer to construct the synthetic CE sample with
exactly the same formatting and tokenization as the policy.

## Experiment

Model:

```text
Qwen3-4B-Base
```

Task environment:

```text
MBPP
```

Pipeline:

```text
SFT → RLVR
```

Data:

```text
212 SFT tasks
212 RL tasks
 40 validation tasks
500 final test tasks
```

Two RL seeds were run for each arm.

## Results

Step-100 pass@1:

```text
Arm A
seed 42: 56.4%
seed 43: 54.2%

Arm B
seed 42: 52.0%
seed 43: 53.6%
```

PREDICT did not establish a pass@1 advantage over the test-and-recover baseline.

The stronger result was outcome prediction. The Arm B SFT checkpoint effectively predicted `PASS`
for every case. After RLVR, the policy learned to detect runtime failures:

```text
RUNTIME_ERROR precision
seed 42: 62.5%
seed 43: 64.1%
```

At step 100, seed 42:

```text
PASS recall:              92%
RUNTIME_ERROR recall:     63%
ASSERTION_FAILURE recall:  0%
```

So the experiment shows that verifier-derived auxiliary supervision can teach the policy a
nontrivial future execution-outcome distinction, but it does not yet show improved overall
coding-agent performance.

## Current limitations

The policy failed to learn assertion-failure prediction, the class that would be most useful to
catch early.

MBPP also makes many execution outcomes cheap to observe immediately, reducing the behavioral value
of predicting them one step earlier.

The next experiments should therefore focus on:

```text
better predictive SFT
alpha = 0 ablation
native ECHO baseline
environments where actions are costly or irreversible
```

## Repository structure

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

See:

```text
docs/REPRODUCTION.md
docs/research_specs.md
```

The project pins its upstream dependencies and PRIME-RL commit for reproducibility. All headline
results are derived from the four Arm A / Arm B runs across seeds 42 and 43, in
[`RESULTS_PUBLISHED/`](RESULTS_PUBLISHED/).

## Status

PREDICT currently demonstrates:

* verifier-derived future outcomes can supervise an earlier prediction point;
* GRPO and auxiliary CE can be separated through token-level loss routing;
* the policy can learn at least one execution-outcome class that was absent from its SFT prediction
  behavior.

The main open question is whether this predictive supervision improves agent decisions in
environments where consequences cannot simply be observed immediately after acting.
