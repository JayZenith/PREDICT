# RLVR results (commit 9eefac7)

Both arms trained 100 GRPO steps (group size 16, batch 64, `zero_advantage`
filter enforced) from their SFT checkpoints, with all four checkpoints
(steps 25/50/75/100) retained and evaluated once on the full 500-task test
set, standalone, after the weights save.

| step | Arm A | Arm B |
|---|---:|---:|
| SFT | 51.6% | 48.2% |
| 25 | 51.4% | 45.2% |
| 50 | 52.2% | 50.0% |
| 75 | 53.6% | 52.6% |
| 100 | **56.4%** | **52.0%** |

McNemar (continuity-corrected) + paired bootstrap CI on the per-task
pass/fail outcomes ([`docs/stats.py`](stats.py)), 8 comparisons total. At
uncorrected p<0.05 five of them look significant; correcting for running 8
tests (Benjamini-Hochberg, FDR 5%) leaves exactly two:

- **Arm A step25 vs Arm B step25: −6.2 pts, p=0.006 — survives correction.**
- **Arm B step75 vs its own SFT: +4.4 pts, p=0.010 — survives correction.**
- Arm B step100 vs SFT (+3.8, p=0.033), Arm B step25 vs SFT (−3.0, p=0.033),
  and Arm A step100 vs Arm B step100 (−4.4, p=0.068, the headline gap) do
  **not** survive correction — indistinguishable from test-set sampling noise.
- Arm A's own SFT-vs-RL was never tested: those raw traces lived on the
  original training instance, deleted before the rerun that produced
  everything else here. Only the point estimate (51.6% → 56.4%) exists.

Read as: the step-100 headline (Arm A ahead by 4.4 pts) is not something you
can hang a claim on yet. Also worth saying plainly — Arm A has two
independent training runs (56%, 56.4%, consistent); Arm B has exactly one,
because its original run stalled on disk-full mid-training. Nothing here has
been checked against training-seed variance for Arm B.

Efficiency, same traces: Arm B does not use fewer tool calls or turns
(5.4–5.7 vs Arm A's 5.3–5.5) — it uses slightly more. It does use fewer
visible test executions (1.62–1.99 vs 1.94–1.99), since shadow-testing on
`REVISE` moves some test cycles off the visible ledger, but spends ~20-30%
more assistant-turn generation length per task on `<PREDICTION>`/`<DECISION>`
tags (996–1124 vs 835–856 chars). Not a clean efficiency win — a trade.

Checkpoints: `JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0`. Raw traces,
eval/serve logs, and training artifacts archived under the gitignored
[`PREDICT_RL_RESULTS/`](../PREDICT_RL_RESULTS/) directory. Full reproduction
steps, correction method, and the full comparison table:
[`docs/REPRODUCTION.md`](REPRODUCTION.md).

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
