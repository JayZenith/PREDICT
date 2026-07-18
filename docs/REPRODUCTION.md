# Reproduction

Requirements: Git, Python 3.12, `uv`, one GPU for SFT, and two GPUs for the
disaggregated PRIME-RL run.

```bash
bash scripts/setup.sh
uv run python -m data.prepare
bash scripts/train_sft.sh
```

Screen the 134 RL candidates with eight SFT samples each:

```bash
uv run eval glyph --harness.id glyph \
  --taskset.data-path data/rl_candidates.jsonl \
  -m outputs/sft/weights/step_16 -n 134 -r 8 --no-push
uv run glyph frontier SCREEN_TRACES
```

If `frontier` reports too few mixed groups, stop. Training on uniform pass@8
groups cannot produce a group-relative learning signal.

```bash
uv run --project .vendor/prime-rl rl @ "$PWD/configs/rl.toml"
```

Run both checkpoints on `data/test.jsonl` with the same sampling settings and
eight samples per task. Then report the paired result:

```bash
uv run glyph compare SFT_TRACES RLVR_TRACES -k 8
```

## Data contract

`data.prepare` pins every source revision and SHA-256 in
`data/manifest.json`. MBPP train IDs are deterministically divided between SFT
and RL. Development uses MBPP validation. Final evaluation uses only MBPP+
task IDs 11-510, so it cannot overlap the official train or validation ranges.
The SFT set contains 180 direct, 40 one-recovery, and 20 two-recovery traces.
The SFT launcher aborts before training if any trace exceeds 1024 tokens or
lacks terminal `FINAL:`. Stack packing preserves trace boundaries instead of
slicing them; no SFT row is shortened or excluded.

Hidden assertions are stored in task metadata, never in the prompt or editable
project. Failed test calls reveal only an error category. The runtime record—not
assistant claims—determines the binary reward.
The environment marks length-truncated or over-4096-token RL traces as fatal.
The pinned PRIME-RL patch records the trace, then makes the orchestrator abort
before it can reach the trainer; it does not discard the trace and continue.

## Local verification

```bash
uv sync --locked --group dev
uv run pytest
uv run python -m data.prepare --output /tmp/glyph-mbpp
```

Do not run model-generated Python through the subprocess runtime on a machine
you care about. Use Prime Sandboxes for real evaluation and training.
