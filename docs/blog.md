# RLVR results, now with a second seed for both arms (commit 9eefac7)

Both arms trained 100 GRPO steps (group size 16, batch 64, `zero_advantage`
filter enforced) from their SFT checkpoints, with all four checkpoints
(steps 25/50/75/100) retained and evaluated once on the full 500-task test
set, standalone, after the weights save. A first pass (one run each) turned
up exactly one significance result that survived correcting for multiple
comparisons: "Arm A beats Arm B at step 25." That's the kind of claim that
shouldn't rest on one training run each, so both arms were retrained from
scratch with a different seed (same SFT checkpoint, same everything else) to
see if it held up.

It didn't.

| step | Arm A (seed 42) | Arm A (seed 43) | Arm B (seed 42) | Arm B (seed 43) |
|---|---:|---:|---:|---:|
| SFT | 51.6% | 51.6% (same ckpt) | 48.2% | 48.2% (same ckpt) |
| 25 | 51.4% | 50.4% | 45.2% | 47.8% |
| 50 | 52.2% | 52.8% | 50.0% | 48.6% |
| 75 | 53.6% | 54.8% | 52.6% | 51.2% |
| 100 | **56.4%** | 54.2% | 52.0% | 53.6% |

McNemar (continuity-corrected) + paired bootstrap CI on per-task pass/fail
([`docs/stats.py`](stats.py)):

- **Each arm, seed 42 vs seed 43, same step**: no significant difference for
  either arm at any of the 4 checkpoints (p=0.20–0.74 for Arm A, p=0.055–0.44
  for Arm B). Both arms' training is reasonably reproducible.
- **Each arm's RL vs its own SFT baseline, both seeds** (Arm A's SFT baseline
  was re-evaluated fresh for this — 50.6%, matching the earlier 51.6% point
  estimate within eval noise): **step 100 is significant for both arms in
  both seeds** — Arm A gave p=0.0003 (seed 42) and p=0.028 (seed 43); Arm B
  gave p=0.033 (seed 42) and p=0.0017 (seed 43). Arm A's step 75 is
  significant in both seeds too (p=0.041, p=0.0035). The step-25
  "regression" reported from Arm B's seed 42 alone (−3.0 pts, p=0.033) did
  **not** replicate in seed 43 (−0.4 pts, p=0.88) — that was noise, not a
  real early RL effect, and neither arm shows a significant step-25 result in
  both seeds.
- **Arm A vs Arm B, matched by step, all four seed combinations**: at step
  100, none of the four pairings are significant (p=0.068–0.86) — the
  seed43-vs-seed43 pairing is a near dead heat (54.2% vs 53.6%, p=0.86). At
  step 25, two of the four pairings are nominally significant (p=0.006,
  p=0.026), and both involve Arm B's seed-42 run — its own lowest point and
  the one seed that dipped significantly below its own SFT baseline. Swap in
  Arm B's seed-43 run at the same step and the gap halves and loses
  significance (p=0.11, p=0.27). **No checkpoint step shows a difference
  between Arm A and Arm B that holds up across seed combinations.**

Read plainly: the RL training itself works for **both** arms — each reliably
improves over its own SFT starting point by step 100, backed by two
independent runs per arm. Whether Arm A's reactive design or Arm B's
predictive design is *better than the other* is a different, separate
question, and it remains unconfirmed at every step, across every seed
combination tested. The one number that once suggested Arm A had an edge
(step 25) traced back to a single outlier training run on Arm B's side, not
a reproducible arm-level effect. Both arms show good within-arm
reproducibility of their own.

**Scope**: none of this isolates "prediction" as a single causal variable —
Arm B bundles the predict/decide protocol, an added action space, the
auxiliary CE loss, and a different SFT trace format together, so this is a
system-vs-system comparison. Detail:
[research_specs.md § Limits to report](research_specs.md#limits-to-report).

Efficiency (from the original seed-42 traces): Arm B does not use fewer tool
calls or turns (5.4–5.7 vs Arm A's 5.3–5.5) — it uses slightly more. It does
use fewer visible test executions (1.62–1.99 vs 1.94–1.99), since
shadow-testing on `REVISE` moves some test cycles off the visible ledger, but
spends ~20-30% more assistant-turn generation length per task on
`<PREDICTION>`/`<DECISION>` tags (996–1124 vs 835–856 chars). Not a clean
efficiency win — a trade.

Checkpoints: `JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0` (seed 42),
`RLVR_ARM_{A,B}_STEP{25,50,75,100}_V1` (seed 43). Raw traces, eval/serve logs, and
training artifacts archived under the gitignored
[`PREDICT_RL_RESULTS/`](../PREDICT_RL_RESULTS/) directory. Full reproduction
steps and the complete comparison tables: [`docs/REPRODUCTION.md`](REPRODUCTION.md).

# SFT complete: moving to RLVR

Both arms were full-fine-tuned from `Qwen3-4B-Base` on verified MBPP (Mostly
Basic Python Problems — Austin et al.,
["Program Synthesis with Large Language Models"](https://arxiv.org/abs/2108.07732),
2021) agent traces: 60 optimizer steps (nine epochs over 212 traces),
1280-token sequence limit, no trace truncated or excluded. RLVR-on-MBPP with a
small Qwen model was directly inspired by Skopin & Kotelnikov,
["Improving Small Language Models for Code Generation with Reinforcement
Learning from Verification Feedback"](https://arxiv.org/abs/2605.30478) (2026).

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

## What was actually hindering Arm B

The stats above establish that Arm A vs Arm B is unsettled — not that
prediction-before-execution doesn't work, just that this run doesn't prove it
does. Digging into what Arm B's prediction head actually learned narrows down
why.

Decision-following is solid: `KEEP` follows a `PASS` prediction 100% of the
time, `REVISE` follows a non-`PASS` prediction 97-99% of the time, in both
seeds. That's not the problem.

The problem is prediction coverage, by outcome class (step 100, both seeds):

| actual outcome | share of failures | recall |
|---|---:|---:|
| ASSERTION_FAILURE | 57-59% | **0%** |
| RUNTIME_ERROR | 15-16% | 50-63% |
| PASS | — | 92-96% |

The model predicts only `PASS` or `RUNTIME_ERROR`, ever. It has not once
correctly predicted `ASSERTION_FAILURE` — the dominant failure mode, code
that runs but fails the assertion — at step 50, 75, or 100, in either
independent run. The trajectory: 100% `PASS` at SFT (fully collapsed), a
brief ~1-2% `ASSERTION_FAILURE` recall at step 25, extinguished back to 0% by
step 50 and never recovering.

Likely cause: correctly predicting `ASSERTION_FAILURE` and choosing `REVISE`
saves at most one turn over just testing and reacting — the real
`python_test` catches wrong logic for free either way, and final task reward
is identical either path. GRPO has no reward gradient pushing toward that
discrimination skill; only the auxiliary CE loss (`λ=0.1`) could teach it,
and it isn't strong enough to survive against GRPO's pull. `RUNTIME_ERROR` is
detectable from surface code features (undefined vars, index risk) without
simulating the algorithm against the test cases — a cheaper pattern, and the
only one that stuck.

Net: Arm B pays the full token overhead of predicting on every turn (see
efficiency numbers above) but the mechanism only covers ~15% of real
failures and is blind to the other ~57%. That's sufficient on its own to
explain why Arm B never pulled ahead, independent of the arm-vs-arm
significance question.

**Where to go next**: sweep `λ` (`orchestrator.algo.alpha` in
`configs/arm_b_rl.toml`, currently `0.1`) upward, or reweight the auxiliary
CE loss toward the rare/hard classes instead of uniform per-token weighting.
That would tell you whether the `ASSERTION_FAILURE` collapse is a
loss-weight problem or a harder ceiling on what this SFT curriculum can
teach the model to generalize to its own generated code.
