# RLVR results, now with a second Arm B seed (commit 9eefac7)

Both arms trained 100 GRPO steps (group size 16, batch 64, `zero_advantage`
filter enforced) from their SFT checkpoints, with all four checkpoints
(steps 25/50/75/100) retained and evaluated once on the full 500-task test
set, standalone, after the weights save. A first pass turned up exactly one
significance result that survived correcting for multiple comparisons: "Arm A
beats Arm B at step 25." That's the kind of claim that shouldn't rest on one
training run, so Arm B was retrained from scratch with a different seed
(same SFT checkpoint, same everything else) to see if it held up.

It didn't.

| step | Arm A | Arm B (seed 42) | Arm B (seed 43) |
|---|---:|---:|---:|
| SFT | 51.6% | 48.2% | 48.2% (same checkpoint) |
| 25 | 51.4% | 45.2% | 47.8% |
| 50 | 52.2% | 50.0% | 48.6% |
| 75 | 53.6% | 52.6% | 51.2% |
| 100 | **56.4%** | 52.0% | 53.6% |

McNemar (continuity-corrected) + paired bootstrap CI on per-task pass/fail
([`docs/stats.py`](stats.py)):

- **Arm B seed 42 vs seed 43, same step**: no significant difference at any
  of the 4 checkpoints (p=0.055–0.44). Arm B's own training is reasonably
  reproducible across seeds.
- **Arm B RL vs its own SFT baseline, both seeds**: step 100 is now a solid,
  two-seed-replicated result — seed 42 gave +3.8 pts (p=0.033, weak alone),
  seed 43 gave +5.4 pts (p=0.0017, clears correction on its own). The step-25
  "regression" reported from seed 42 (−3.0 pts, p=0.033) did **not**
  replicate in seed 43 (−0.4 pts, p=0.88) — that was noise, not a real early
  RL effect.
- **Arm A vs Arm B, matched by step, against each Arm B seed**: the step-25
  gap that survived correction against seed 42 (−6.2 pts, p=0.006) shrinks and
  loses significance against seed 43 (−3.6 pts, p=0.11). Every other step was
  already non-significant against seed 42, and stays that way against seed 43
  (p=0.13–0.32). **No checkpoint step shows a confirmed Arm A vs Arm B
  difference once a second Arm B training run is in the picture.**

Read plainly: the RL training itself works — Arm B reliably improves over its
own SFT starting point, and that's now backed by two independent runs, not
one. Whether Arm A's reactive design or Arm B's predictive design is
*better* remains unconfirmed at every step tested. The one number that once
suggested Arm A had an edge (step 25) turned out to be exactly the kind of
single-run noise a second seed exists to catch. Arm A has two independent
training runs of its own (step100: 56%, 56.4% — consistent); a second Arm A
seed hasn't been run yet.

Efficiency (from the original seed-42 traces): Arm B does not use fewer tool
calls or turns (5.4–5.7 vs Arm A's 5.3–5.5) — it uses slightly more. It does
use fewer visible test executions (1.62–1.99 vs 1.94–1.99), since
shadow-testing on `REVISE` moves some test cycles off the visible ledger, but
spends ~20-30% more assistant-turn generation length per task on
`<PREDICTION>`/`<DECISION>` tags (996–1124 vs 835–856 chars). Not a clean
efficiency win — a trade.

Checkpoints: `JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0` (seed 42),
`RLVR_ARM_B_STEP{25,50,75,100}_V1` (seed 43). Raw traces, eval/serve logs, and
training artifacts archived under the gitignored
[`PREDICT_RL_RESULTS/`](../PREDICT_RL_RESULTS/) directory. Full reproduction
steps and the complete comparison tables: [`docs/REPRODUCTION.md`](REPRODUCTION.md).

# SFT complete: moving to RLVR

Both arms were full-fine-tuned from `Qwen3-4B-Base` on verified MBPP agent
traces: 60 optimizer steps (nine epochs over 212 traces), 1280-token sequence
limit, no trace truncated or excluded.

| Checkpoint | Final loss | val40 pass@1 (greedy) |
|---|---:|---:|
| [Arm A SFT](https://huggingface.co/JayZenith/SFT_ARM_A) | 0.0287 | 24/40 (60%) |
| [Arm B SFT](https://huggingface.co/JayZenith/SFT_ARM_B) | 0.0273 | 21/40 (53%), prediction accuracy 48% |

## The harness was testing the wrong thing

Earlier checkpoints scored 17%/10% here. Digging into the failure traces
against the full 500-task test set showed the dominant failure mode by far
was `RUNTIME_ERROR` outnumbering `ASSERTION_FAILURE` roughly 20:1 — not "close
but wrong logic," but code crashing outright. The cause: the harness hid the
MBPP test assertions from the prompt entirely, so the agent had to blind-guess
the exact function name and signature (e.g. writing `remove_characters` when
the hidden test called `remove_dirty_chars`), and the tool result only ever
reported `"generated solution raised a runtime error"` — never the traceback
or exception type — leaving the agent with no way to diagnose what it got
wrong. A standard-MBPP completion check (tests shown, no agent loop) on the
*untuned* base model scored 64.6% pass@1 on the same 500 tasks where the full
SFT pipeline scored 6%.

Arm B's actual research question — does explicit outcome prediction improve
reasoning, first-pass correctness, and reduce revision attempts in a
realistic test-driven workflow — doesn't require blind signature guessing to
be interesting; it requires the agent to know what it's graded against and
still have to execute to find out if its candidate works. Blind-signature
inference was accidentally the dominant difficulty, swamping the actual
experimental variable. The task prompt now shows the exact test assertions
(matching standard MBPP), for both arms, everywhere the prompt is built (SFT
traces and RL train/validation/test tasksets share one prompt function).
Arm A's val40 score landing right at the base model's own ceiling (60% vs.
64.6%) is the confirmation this worked: the harness is no longer eating
capability, and the remaining gap is agent-loop overhead, not blind guessing.

## Two infrastructure bugs, found and fixed along the way

**A CUDA crash specific to `seq_len=1024`.** Longer prompts (test assertions
now embedded) needed a higher token cap than the old 768. Both arms crashed
with a CUDA illegal-memory-access under `torch.compile` on RTX PRO 6000
Blackwell — reproduced deterministically on three separate fresh instances,
ruling out instance degradation. A minimal 8-row synthetic run bisected it to
the exact value: `seq_len=768` and `seq_len=1280` both train cleanly on
identical hardware; `1024` alone doesn't. This is a narrow bug in the pinned
torch/PRIME-RL/Blackwell stack, not this repo — the fix is using `1280`.

**Arm B's SFT set needed deeper recovery chains.** An earlier version of the
70 recovery traces only ever demonstrated one revision cycle (at most two
`<PREDICTION>`/`<DECISION>` turns per trace), so under RL exploration —
which routinely needs 3+ revision cycles — the model had no template and
degraded into malformed tags. 20 of the 70 recovery traces (10 `deep_shadow`,
10 `deep_visible`) now chain two independently-verified failing mutations of
the gold code, giving a genuine three-cycle example; the remaining 50 keep
the one-step shadow/visible split (25/25).

Local configs, logs, W&B runs, and raw sampling traces for the SFT stage are
archived under the gitignored `PREDICT_SFT_RESULTS/` directory (RLVR
artifacts are in `PREDICT_RL_RESULTS/`, see the results section above).
