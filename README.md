# PREDICT

PREDICT tests whether a coding agent can predict the sandbox consequence of its
own patch and use that prediction to keep or revise the patch before execution.
It starts from GLYPH v2's minimal Python agent stack: a Qwen3-4B policy reads
`solution.py`, patches it, runs hidden MBPP tests, and retries before emitting
`FINAL`.

The matched Arm A/Arm B experiment is specified in
[research specs](docs/research_specs.md), with complete
[canonical agent traces](docs/agent_trace.md). The current code is the imported
GLYPH v2 baseline from which both arms will be implemented.

## Current GLYPH baseline

The data is small and fixed:

- 240 MBPP agent traces for SFT: 180 direct, 40 one-recovery, and 20
  two-recovery trajectories.
- 134 disjoint MBPP problems for pass@8 frontier screening.
- 90 MBPP validation problems for development.
- 224 held-out MBPP+ problems from the official MBPP test-ID range.

## SFT run

SFT is a one-GPU full-parameter run from `Qwen/Qwen3-4B-Base`. The 240 traces
contain 180 direct successes, 40 one-recovery traces, and 20 two-recovery
traces. Run:

```bash
bash scripts/setup.sh
uv run python -m data.prepare
bash scripts/train_sft.sh
```

The launcher tokenizes every complete ChatML trace before step 1. The longest
is 747 tokens against a 1024-token limit. It aborts if any trace exceeds that
limit or lacks terminal `FINAL:`. Stack packing preserves whole traces; none
are truncated or excluded. The final checkpoint is written to
`outputs/sft/weights/step_16`.

## RL run

Screen the disjoint candidates, retain only the mixed pass@8 frontier, then
run GRPO:

```bash

# Sample the SFT checkpoint eight times on every RL candidate.
uv run eval glyph --harness.id glyph \
  --taskset.data-path data/rl_candidates.jsonl \
  -m outputs/sft/weights/step_16 -n 134 -r 8 --no-push

# Replace SCREEN_TRACES with that evaluation's traces.jsonl.
uv run glyph frontier SCREEN_TRACES

# GRPO on only mixed pass@8 groups (one training GPU + one inference GPU).
uv run --project .vendor/prime-rl rl @ "$PWD/configs/rl.toml"
```

SFT uses the rented GPU directly. RL uses both rented GPUs for training and
vLLM inference; each agent rollout runs in a disposable Prime Sandbox and
therefore also requires `PRIME_API_KEY` and Prime Sandbox balance. The pinned
PRIME-RL patch installed by `setup.sh` records any truncated training rollout
in the audit log, then aborts before it reaches the trainer. It never silently
drops one and continues.

Evaluate both checkpoints on the same held-out MBPP+ tasks, then compare them:

```bash
uv run eval glyph --harness.id glyph --taskset.data-path data/test.jsonl \
  -m outputs/sft/weights/step_16 -n 224 -r 8 --no-push

uv run eval glyph --harness.id glyph --taskset.data-path data/test.jsonl \
  -m outputs/rl/weights/step_25 -n 224 -r 8 --no-push

uv run glyph compare SFT_TRACES RLVR_TRACES -k 8
```

Generated code is arbitrary Python. Prime Sandboxes are the default. The local
subprocess runtime is only for disposable development environments.

See [architecture](docs/ARCHITECTURE.md) and [reproduction](docs/REPRODUCTION.md).
