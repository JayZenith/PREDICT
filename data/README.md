# GLYPH data

Run from the repository root:

```bash
uv run python -m data.prepare
```

`prepare.py` downloads pinned parquet files into `.cache/glyph/`, checks their
SHA-256 hashes, and writes the processed datasets and editable `solution.py`
blueprints into this directory. Exact revisions and hashes are recorded in
`manifest.json`.

Pinned sources:

- `google-research-datasets/mbpp` at `4bb6404fdc6cacfda99d4ac4205087b89d32030c`
- `evalplus/mbppplus` at `b2d74c91837c3f2a20c1299ae98133cbe7cfa077`

| File | Source | Count | Purpose |
|---|---|---:|---|
| `sft.jsonl` | MBPP train IDs 601–974 | 240 | Full agent-trace SFT |
| `rl_candidates.jsonl` | Remaining MBPP train tasks | 134 | SFT pass@8 screening |
| `dev.jsonl` | MBPP validation IDs 511–600 | 90 | Development checks |
| `test.jsonl` | MBPP+ tasks with IDs 11–510 | 224 | Held-out final evaluation |

The train split is deterministic with seed 42. SFT and RLVR candidates never
share tasks. The 134 RLVR candidates are not all trained on: `glyph frontier`
retains only tasks where eight SFT samples contain both passes and failures.

SFT contains 180 direct traces, 40 one-recovery traces, and 20 two-recovery
traces. `recovery.py` introduces one or two independently failing semantic
mutations. It executes every failed stage, applies targeted corrections, then
executes the final pass before writing the trace. These are deterministic
counterfactual traces, not teacher-model rollouts.

The original 500-row MBPP test parquet and 10-row prompt split are not
downloaded. Final evaluation uses the stronger MBPP+ tests instead. MBPP+
contains 378 rows; 154 outside the official test IDs 11–510 are omitted. This
excludes prompt tasks and prevents train/validation overlap. Thus the download
contains 842 rows and the processed split uses 688 tasks.

Hidden tests are stored in task records but never placed in model prompts or
editable blueprints. Generated artifacts are ignored by Git. `README.md`,
`prepare.py`, `recovery.py`, and `validate.py` are tracked and preserved on
every rebuild.

Before SFT, `scripts/train_sft.sh` runs `data.validate` with the exact Qwen3
tokenizer. It aborts if a trace exceeds 1024 tokens or lacks a terminal
`FINAL:`. The current longest trace is 747 tokens.
