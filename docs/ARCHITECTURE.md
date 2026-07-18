# Architecture

GLYPH has four moving parts:

1. `python -m data.prepare` downloads pinned MBPP and MBPP+ parquet files, verifies
   their hashes, and writes disjoint SFT, RL-candidate, development, and test
   JSONL files.
2. SFT demonstrations preserve the full agent trace: 180 direct successes, 40
   one-recovery traces, and 20 two-recovery traces. Every recorded failure and
   final pass is executed during curation.
   `glyph.chat` defines the shared system prompt and ChatML template. SFT and
   RL load that exact template; parity tests prevent role-marker drift. Exact
   tokenizer preflight plus stack packing prevent partial traces entering SFT.
3. `GlyphHarness` runs that text protocol inside one Verifiers v1 runtime. The
   model can edit only `solution.py`; `python_test` executes hidden assertions
   and returns sanitized pass/fail feedback.
4. `GlyphTask` reads the runtime execution record. A real successful
   `python_test`, its matching tool turn, and a terminal `FINAL:` earn `1`;
   every other trace earns `0`. A length-truncated or trainer-overflowing RL
   trace aborts the run before the orchestrator sends it to the trainer.

There are no formatting rewards, compiler rewards, protocol penalties, OPD,
length shaping, or reference KL. `configs/rl.toml` is plain GRPO and explicitly
sets the trainer KL coefficient to zero.

`glyph frontier` keeps only pass@8 groups containing both failures and
successes. `glyph compare` performs the paired SFT-versus-RLVR report on
identical held-out task IDs.
