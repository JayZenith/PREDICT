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
| SFT | 50.6% | 50.6% (same ckpt) | 48.2% | 48.2% (same ckpt) |
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
  is 50.6% — the only archived 500-task Arm A SFT eval in this repo; an
  earlier 51.6% figure had no corresponding raw eval file and has been
  corrected): **step 100 is significant for both arms in
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
The verified-label auxiliary CE design was inspired by Shrivastava, Kauffmann,
Awadallah & Papailiopoulos, ["ECHO: Terminal Agents Learn World Models for
Free"](https://arxiv.org/abs/2605.24517) (2026) — ECHO trains a complementary
CE loss on environment-observation tokens within the same GRPO rollout, no
separate reasoning step, and doubles pass@1 on TerminalBench-2.0 (Qwen3-8B:
2.70%→5.17%; Qwen3-14B: 5.17%→10.79%). PREDICT's difference from ECHO:
[research_specs.md § Novelty relative to ECHO](research_specs.md#novelty-relative-to-echo).

| SFT behavior family | Count | Arm A | Arm B |
|---|---:|---|---|
| Direct success | 142 | patch → test passes | correct patch → predict PASS → KEEP → test passes |
| One-step recovery | 50 | faulty patch → test fails → fix → test passes | 25 shadow: predict failure → REVISE → fix → predict PASS → KEEP → test passes; 25 visible: predict PASS (honest mistake) → KEEP → test fails → fix → predict PASS → KEEP → test passes |
| Two-step recovery | 20 | faulty patch → fails → different faulty patch → fails → fix → test passes | 10 deep shadow: predict failure → REVISE → predict failure → REVISE → predict PASS → KEEP → test passes; 10 deep visible: predict PASS (mistake) → KEEP → fails → predict PASS (mistake again) → KEEP → fails → fix → predict PASS → KEEP → test passes |

Total: 212 traces per arm (70 recovery, split 50 one-step / 20 two-step). Full
detail: [research_specs.md § SFT composition](research_specs.md#sft-composition).

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

Arm B's actual research question — how outcome prediction changes reasoning,
first-pass correctness, and revision attempts in a realistic test-driven
workflow, compared to Arm A — doesn't require blind signature guessing to
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
Blackwell — reproduced deterministically on two separate fresh instances,
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

Decision-following is solid: `REVISE` follows a non-`PASS` prediction 100% of
the time in both seeds; `KEEP` follows a `PASS` prediction 96.4% of the time
(seed 42) and 99.4% of the time (seed 43). That's not the problem.

The problem is prediction coverage, by outcome class (step 100, both seeds):

| actual outcome | share of verified outcomes | recall |
|---|---:|---:|
| ASSERTION_FAILURE | 57-59% | **0%** |
| RUNTIME_ERROR | 15-16% | 50-63% |
| PASS | 22-23% | 92-96% |

The model predicts only `PASS` or `RUNTIME_ERROR`, ever. It has not once
correctly predicted `ASSERTION_FAILURE` — the dominant failure mode, code
that runs but fails the assertion — at RL step 50, 75, or 100, in either
independent run (these are RLVR/GRPO steps; SFT is already finished and
frozen before this trajectory starts). The trajectory: 100% `PASS` at the
SFT checkpoint, i.e. RL step 0 (fully collapsed), a brief ~1-2%
`ASSERTION_FAILURE` recall at RL step 25, extinguished back to 0% by RL step
50 and never recovering.

Root cause, in two parts:

1. **GRPO can't fix it directly.** Prediction-label tokens are explicitly
   masked out of the GRPO loss ([research_specs.md § Arm B — consequence
   predictor](research_specs.md#arm-b--consequence-predictor)): final task
   reward depends only on whether `apply_patch`/`python_test`/`FINAL`
   eventually succeed, never on what `<PREDICTION>` said, so GRPO carries no
   direct loss term at those specific positions. That's narrower than "GRPO
   never touches it" — GRPO and the auxiliary CE update the same shared
   transformer weights, so GRPO's gradient at every other position in the
   rollout (the `<DECISION>` and action tokens) still reshapes the hidden
   representations feeding the masked positions, and can shift the
   prediction distribution indirectly. What the mask actually guarantees:
   no term in GRPO's loss directly rewards or penalizes a specific
   predicted label; the auxiliary CE (`λ=0.1`) is the only *direct*
   supervision on correctness there, not the only thing capable of moving
   those probabilities at all. The step-25 blip is early-optimization noise
   that nothing directly defends, so it doesn't last.
2. **SFT starts collapsed by construction.** Across all 212 Arm B SFT
   traces, `<PREDICTION>` labels split roughly 257 PASS : 45 real-failure
   (85%/15%) — and half the 70 recovery traces (25 `visible` + 10
   `deep_visible`) hardcode the label `PASS` on the buggy candidate itself,
   demonstrating the "honest mistake, caught by the real test" recovery
   path. Only the 25 `shadow` + 10 `deep_shadow` traces ever show a correct
   non-PASS label. So SFT doesn't just fail to build the skill robustly —
   for half its recovery examples it directly demonstrates that guessing
   PASS and letting `python_test` sort it out is an acceptable pattern,
   which is exactly the shortcut the RL policy settles back into.
   `RUNTIME_ERROR` is detectable from surface code features (undefined
   vars, index risk) without simulating the algorithm against the test
   cases — a cheaper pattern, and the only non-PASS one that stuck.

Not reward hacking — the masked-out tokens mean there's no direct term for
GRPO to game, only indifference. Call it poor reward shaping: a low-weight,
uniformly-per-token CE loss was the only direct correctness signal on that
skill, so the 85%-majority PASS label dominates that gradient too. The two
directions
below target these directly — reweighting CE toward rare classes
attacks part 2, reward shaping attacks part 1 by finally giving GRPO's own
reward a reason to tell the two kinds of rollouts apart.

Net: Arm B pays the full token overhead of predicting on every turn (see
efficiency numbers above) but the mechanism only covers ~15% of real
failures and is blind to the other ~57%. That's sufficient on its own to
explain why Arm B never pulled ahead, independent of the arm-vs-arm
significance question.

## What I learned

Not the results table — the methodology and architecture lessons that came
from digging into why the results looked the way they did.

1. **A matched comparison isn't an ablation.** Arm A vs. Arm B tests two
   complete, multi-part systems against each other — the predict/decide
   protocol, the auxiliary loss, and the SFT trace format all change
   together. It can tell you the bundle didn't clearly win; it can't tell
   you which piece of the bundle mattered. Isolating "prediction" as a
   single causal factor needs a compute/action-matched ablation, not this
   two-arm design.
2. **An auxiliary loss needs its own reward path, or it's the only
   teacher.** Masking prediction-label tokens out of GRPO's loss is a
   reasonable design choice — it stops the RL objective from getting noisy
   gradient on tokens final reward doesn't grade. But it also means the
   low-weight, uniformly-weighted auxiliary CE became the *entire* training
   signal for that skill. A weak sole teacher produces a weak skill,
   independent of how good the rest of the system is.
3. **SFT curricula can bake in the shortcut they're trying to prevent.**
   Half of Arm B's recovery traces demonstrate "guess PASS, let the real
   test catch the mistake" as a valid, reward-preserving pattern — because
   that's a real recovery mode worth training. It also happens to be the
   exact shortcut the collapsed policy falls back on. A curriculum can be
   correct about the behavior it demonstrates and still work against a
   different goal of the same system.
4. **Rare at one stage isn't rare at another.** Only 37 of 302 SFT
   prediction labels are real `ASSERTION_FAILURE` examples — genuinely
   thin. But checking real per-step RL logs (not just final eval) showed
   the model sees this outcome constantly during RL, a third to half of
   every step. Diagnosing a persistent 0% recall as "not enough examples"
   would have been wrong; checking the actual training-time logs instead
   of assuming the SFT-time distribution still applied caught that.
5. **One training run is an anecdote.** The first seed made Arm A look
   ahead at step 100 and significantly ahead at step 25. Both effects were
   seed noise — gone under a second independent run with the same setup.
   Two-seed replication (now standard for both arms here) is what turned
   an appealing headline into a checked claim.
6. **"No scratchpad" doesn't mean "no computation," and it isn't
   automatically a dead end.** Every `<PREDICTION>` tag in every trace is
   emitted immediately after `apply_patch`, with no reasoning tokens in
   between — but this project's own inspiration, ECHO (Shrivastava,
   Kauffmann, Awadallah & Papailiopoulos, ["Terminal Agents Learn World
   Models for Free"](https://arxiv.org/abs/2605.24517), 2026), trains a CE
   loss on environment-observation tokens with no separate reasoning step
   either, reusing the same GRPO rollout — and it doubles pass@1 on
   TerminalBench-2.0 (Qwen3-8B: 2.70%→5.17%; Qwen3-14B: 5.17%→10.79%). So
   scratchpad-free auxiliary prediction plainly can work. The likely
   difference: ECHO's target is the full, dense, multi-token environment
   observation — the actual output text, forcing token-by-token computation
   through what happened — while ours is a single terse label from a 6-way
   enum. A denser prediction target closer to ECHO's (the specific failing
   assertion or expected value, not just an outcome class) may be a more
   direct fix for the `ASSERTION_FAILURE` blind spot than reweighting the
   current, thin classification target.

**Where to go next**, sweep first, then the harder tests:

1. **Sweep `λ`** (`orchestrator.algo.alpha` in `configs/arm_b_rl.toml`,
   currently `0.1`) upward, or reweight the auxiliary CE loss toward the
   rare/hard classes instead of uniform per-token weighting. "Rare" is a
   claim about the *SFT set*, not RL: counting the 212 Arm B SFT traces
   directly, `<PREDICTION>` labels split 257 PASS : 37 `ASSERTION_FAILURE` :
   8 `RUNTIME_ERROR` — only 37 real examples, ever, none with any reasoning
   shown before the label (the tag is emitted immediately after
   `apply_patch`, no scratchpad). During RL itself `ASSERTION_FAILURE` is
   *not* rare — it's routinely a third to half of every step's real outcomes
   (118 of 236 at step 1, 72 of 103 at step 50) — so the model isn't short
   on raw exposure during RL, it just never learns from it. That points at
   *hard*, not just *rare*. Reweighting raises the gradient on examples the
   model already sees; it doesn't add a missing demonstration of *how* to
   simulate. Tells you whether this is a fixable weighting problem or a real
   ceiling on one-shot, no-scratchpad code simulation at this model size.
2. **Reward shaping.** Final task reward pays out identically regardless of
   whether the prediction was right, so GRPO carries no direct incentive
   toward good predictions. Hypothesis: give extra reward when
   `true failure + predicted failure + REVISE` and when
   `true PASS + predicted PASS + KEEP`. One nuance, verified in
   `src/glyph/prime_rl.py`: GRPO already masks sampled `<PREDICTION>` label
   tokens out of its own loss (`rl_weights=0`), so a shaped reward's
   *direct* gradient still lands on the surrounding `<DECISION>`/action
   tokens, not the label positions themselves. Because every position
   shares the same transformer weights, that's not nothing — reward tied
   to the decision can still reshape the hidden states feeding the label
   positions indirectly — but it isn't a term that directly grades the
   label choice, and decision-following is already 96-100% consistent,
   so there's limited headroom on the direct side. Making this experiment
   squarely target prediction correctness likely still means lifting the
   mask, not just adding a reward term.
3. **Ablate the gate.** Test prediction-with-behavioral-gating (current
   Arm B) against prediction-as-pure-auxiliary-signal (ECHO-style, no
   `KEEP`/`REVISE` control), holding the rest of the bundle fixed. This is
   the one piece that's actually new relative to ECHO, and it's never been
   tested in isolation.
4. **Denser prediction target.** Predict the specific failing assertion or
   expected/actual value instead of a 6-way outcome class. A coarse label
   may be why the model never learned to simulate; a denser target forces
   token-by-token reasoning the way ECHO's dense observation-token target
   does.
5. **Causal, not correlational, evaluation.** Decision-following is ~100%
   by construction, so "predicted X" and "did X" are nearly tautological
   here. Counterfactually rerun rollouts and measure whether reward
   actually tracks prediction correctness, holding the rest of the policy
   fixed.
6. **Scale beyond one model and one benchmark.** n=500 on MBPP with a 4B
   model that may have seen MBPP in pretraining can't rule out idiosyncrasy.
   Replicating across model scale or on a benchmark unlikely to be
   memorized is what would make this a claim, not just a run.

The actual crux: GRPO and the auxiliary CE update the same shared
transformer weights, so calling them fully separate is too strong — but
they're not doing the same job either. GRPO carries no direct loss term on
the prediction-label positions (masked out by design); the auxiliary CE is
the only signal that directly supervises which label gets predicted there.
Reweighting CE strengthens that direct signal; reward shaping strengthens
the indirect one, by making the surrounding decision/action tokens' reward
actually depend on prediction quality. Neither is a full fix alone:
reweighting doesn't add missing information about *how* to simulate, and
reward shaping's effect on the label positions, absent a direct term there,
still depends on shared-weight spillover from everywhere else in the
rollout — real, but unmeasured and not targeted.

## Appendix

### Beyond Code Verification: Research Judgment

One unexpected lesson came from reviewing the experiment itself. I initially
described Arm A as a baseline and Arm B as a single added variable. That
framing was too clean: Arm B also changed the trace format, action protocol,
auxiliary loss, and decision space. The two arms are better understood as
comparative training systems, not a strict causal ablation.

The coding agent helped build and document the project, but it did not
challenge that experimental claim. Code has hard verifiers—tests, syntax,
execution, and logs. Research judgment has no equivalent oracle. Auditing a
claim requires reconstructing the entire experiment, identifying every
differing factor, and testing whether the conclusion is actually supported.

A stronger research agent would need to do more than make a project coherent
and functional. It would need to actively falsify the researcher's framing.
