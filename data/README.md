# Data

Generate from the repository root:

```bash
uv run python -m data.prepare
uv run python -m data.validate data
```

`prepare.py` downloads the official full MBPP parquet files from
`google-research-datasets/mbpp`, pinned to revision
`4bb6404fdc6cacfda99d4ac4205087b89d32030c`. It verifies each SHA-256 before
use. `manifest.json` records the same source proof.
It also records the byte size and SHA-256 of every generated dataset artifact.

| Generated file | Rows | Purpose |
|---|---:|---|
| `sft/arm_a/train.jsonl` | 212 | reactive full-trace SFT |
| `sft/arm_b/train.jsonl` | 212 | matched prediction SFT |
| `arm_{a,b}_train.jsonl` | 212 each | RL environments |
| `arm_{a,b}_validation.jsonl` | 40 each | λ/development evaluation |
| `arm_{a,b}_test.jsonl` | 500 each | final evaluation |
| `assignments.json` | 212 | direct/recovery assignment audit |
| `manifest.json` | — | source, hash, split, and outcome audit |

The official source files are unchanged:

- Train: IDs 601–974.
- Validation: IDs 511–600.
- Test: IDs 11–510.

Seed 42 deterministically moves 50 official-validation IDs into the train pool,
then splits the 424 non-test tasks into 212 SFT IDs and 212 disjoint RL IDs.
The remaining 40 official-validation IDs are validation. The 500 official test
IDs stay untouched.

Seed 42 also chooses 70 SFT tasks where `recovery.py` can construct one real
semantic failure. The other 142 are direct successes.
Every Arm A/Arm B pair uses the same task, candidate, corrected solution, and
sandbox outcomes.

`validate.py` is exhaustive. It:

- executes all 964 official gold solutions;
- replays all 424 SFT traces;
- verifies every failed candidate and final pass;
- verifies every Arm B prediction label;
- checks matched hashes, split IDs, hidden-test isolation, terminal `FINAL:`,
  and exact tokenizer length.

Current longest traces are 703 tokens for Arm A and 766 for Arm B. The
768-token SFT limit fits every row. Nothing is shortened or omitted.

Generated files are committed so a fresh clone is immediately trainable. They
are not hand-edited; re-run the two commands above to reproduce them
byte-for-byte.
