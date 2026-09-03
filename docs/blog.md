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
  is 50.6%, the only archived 500-task Arm A SFT eval in this repo; an
  earlier 51.6% figure had no corresponding raw eval file and has been
  corrected): **step 100 is significant for both arms in
  both seeds**: Arm A gave p=0.0003 (seed 42) and p=0.028 (seed 43); Arm B
  gave p=0.033 (seed 42) and p=0.0017 (seed 43). Arm A's step 75 is
  significant in both seeds too (p=0.041, p=0.0035). The step-25
  "regression" reported from Arm B's seed 42 alone (−3.0 pts, p=0.033) did
  **not** replicate in seed 43 (−0.4 pts, p=0.88); that was noise, not a
  real early RL effect, and neither arm shows a significant step-25 result in
  both seeds.
- **Arm A vs Arm B, matched by step, all four seed combinations**: at step
  100, none of the four pairings are significant (p=0.068–0.86); the
  seed43-vs-seed43 pairing is a near dead heat (54.2% vs 53.6%, p=0.86). At
  step 25, two of the four pairings are nominally significant (p=0.006,
  p=0.026), and both involve Arm B's seed-42 run, its own lowest point and
  the one seed that dipped significantly below its own SFT baseline. Swap in
  Arm B's seed-43 run at the same step and the gap halves and loses
  significance (p=0.11, p=0.27). **No checkpoint step shows a difference
  between Arm A and Arm B that holds up across seed combinations.**

Bottom line: RLVR reliably improves **both** arms over their own SFT baseline
by step 100, across two independent runs each. Whether either design is
*better than the other* remains unconfirmed at every step and seed
combination; the one number that once suggested Arm A had an edge (step 25)
traced back to a single outlier run, not a reproducible effect. Step 100 is
an interim checkpoint, not a destination: two of the four runs are still
climbing there (+2.8 and +2.4 points over step 75), so longer runs are the
first next step. What this phase establishes is the pipeline: custom SFT agent
traces plus RLVR produce real, McNemar-confirmed gains, and the between-arm
comparison is context. The decisive experiments compare Arm B to itself, one
factor at a time (see "Where to go next").

**Scope**: none of this isolates "prediction" as a single causal variable;
Arm B bundles the predict/decide protocol, an added action space, the
auxiliary CE loss, and a different SFT trace format together, so this is a
system-vs-system comparison. Detail:
[research_specs.md § Limits to report](research_specs.md#limits-to-report).

Efficiency (from the original seed-42 traces): Arm B does not use fewer tool
calls or turns (5.4–5.7 vs Arm A's 5.3–5.5); it uses slightly more. It does
use fewer visible test executions (1.62–1.99 vs 1.94–1.99), since
shadow-testing on `REVISE` moves some test cycles off the visible ledger, but
spends ~20-30% more assistant-turn generation length per task on
`<PREDICTION>`/`<DECISION>` tags (996–1124 vs 835–856 chars). Not a clean
efficiency win, just a trade.

Checkpoints: `JayZenith/RLVR_ARM_{A,B}_STEP{25,50,75,100}_V0` (seed 42),
`RLVR_ARM_{A,B}_STEP{25,50,75,100}_V1` (seed 43). Raw traces, eval/serve logs, and
training artifacts archived under the gitignored
[`RESULTS_PUBLISHED/`](../RESULTS_PUBLISHED/) directory. Full reproduction
steps and the complete comparison tables: [`docs/REPRODUCTION.md`](REPRODUCTION.md).
Which file backs which claim, and the four ways to misread this data:
[Provenance](#provenance-how-to-verify-every-claim-in-the-write-up), at the
bottom of this file.

# SFT complete: moving to RLVR

Both arms were full-fine-tuned from `Qwen3-4B-Base` on verified MBPP (Mostly
Basic Python Problems, Austin et al.,
["Program Synthesis with Large Language Models"](https://arxiv.org/abs/2108.07732),
2021) agent traces: 60 optimizer steps (nine epochs over 212 traces),
1280-token sequence limit, no trace truncated or excluded. RLVR-on-MBPP with a
small Qwen model was directly inspired by Skopin & Kotelnikov,
["Improving Small Language Models for Code Generation with Reinforcement
Learning from Verification Feedback"](https://arxiv.org/abs/2605.30478) (2026).
The verified-label auxiliary CE design was inspired by Shrivastava, Kauffmann,
Awadallah & Papailiopoulos, ["ECHO: Terminal Agents Learn World Models for
Free"](https://arxiv.org/abs/2605.24517) (2026). ECHO trains a complementary
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

**Limitation.** These recovery traces are synthetic: the faulty patches are
deterministic, verifier-confirmed mutations of gold MBPP solutions, not
failures naturally sampled from the model. So the SFT failure distribution is
one I chose, not one the policy actually produces, and the two need not
match. That bears directly on the `ASSERTION_FAILURE` result below: the
curriculum's mutations skew toward failures that are easy to construct and
confirm, while the failures the model generates under RL are its own, and the
outcome class it never learned to predict is exactly the one where that gap
would show up.

| Checkpoint | Final loss | val40 pass@1 (greedy) |
|---|---:|---:|
| [Arm A SFT](https://huggingface.co/JayZenith/SFT_ARM_A) | 0.0287 | 24/40 (60%) |
| [Arm B SFT](https://huggingface.co/JayZenith/SFT_ARM_B) | 0.0273 | 21/40 (53%), prediction accuracy 48% |

## The harness was testing the wrong thing

Earlier checkpoints scored 17%/10% here. Digging into the failure traces
against the full 500-task test set showed the dominant failure mode by far
was `RUNTIME_ERROR` outnumbering `ASSERTION_FAILURE` roughly 20:1, not "close
but wrong logic," but code crashing outright. The cause: the harness hid the
MBPP test assertions from the prompt entirely, so the agent had to blind-guess
the exact function name and signature (e.g. writing `remove_characters` when
the hidden test called `remove_dirty_chars`), and the tool result only ever
reported `"generated solution raised a runtime error"`, never the traceback
or exception type, leaving the agent with no way to diagnose what it got
wrong. A standard-MBPP completion check (tests shown, no agent loop) on the
*untuned* base model scored 64.6% pass@1 on the same 500 tasks where the full
SFT pipeline scored 6%.

That's not the actual research question: Arm B needs the agent to know what
it's graded against and still have to execute to find out if its candidate
works, not blind-guess a function signature. The task prompt now shows the exact test assertions
(matching standard MBPP), for both arms, everywhere the prompt is built (SFT
traces and RL train/validation/test tasksets share one prompt function).
Arm A's val40 score landing right at the base model's own ceiling (60% vs.
64.6%) is the confirmation this worked: the harness is no longer eating
capability, and the remaining gap is agent-loop overhead, not blind guessing.

## Two infrastructure bugs, found and fixed along the way

**A CUDA crash specific to `seq_len=1024`.** Longer prompts (test assertions
now embedded) needed a higher token cap than the old 768. Both arms crashed
with a CUDA illegal-memory-access under `torch.compile` on RTX PRO 6000
Blackwell, reproduced deterministically on two separate fresh instances,
ruling out instance degradation. A minimal 8-row synthetic run bisected it to
the exact value: `seq_len=768` and `seq_len=1280` both train cleanly on
identical hardware; `1024` alone doesn't. This is a narrow bug in the pinned
torch/PRIME-RL/Blackwell stack, not this repo; the fix is using `1280`.

**Arm B's SFT set needed deeper recovery chains.** An earlier version of the
70 recovery traces only ever demonstrated one revision cycle (at most two
`<PREDICTION>`/`<DECISION>` turns per trace), so under RL exploration, which routinely needs 3+ revision cycles, the model had no template and
degraded into malformed tags. 20 of the 70 recovery traces (10 `deep_shadow`,
10 `deep_visible`) now chain two independently-verified failing mutations of
the gold code, giving a genuine three-cycle example; the remaining 50 keep
the one-step shadow/visible split (25/25).

Local configs, logs, W&B runs, and raw sampling traces for the SFT stage are
archived under the gitignored `RESULTS_SFT/` directory (RLVR
artifacts are in `RESULTS_PUBLISHED/`, see the results section above).

## What was actually hindering Arm B

The stats above establish that Arm A vs Arm B is unsettled, not that
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
correctly predicted `ASSERTION_FAILURE`, the dominant failure mode, code
that runs but fails the assertion, at RL step 50, 75, or 100, in either
independent run (these are RLVR/GRPO steps; SFT is already finished and
frozen before this trajectory starts). The trajectory: 100% `PASS` at the
SFT checkpoint, i.e. RL step 0 (fully collapsed), a brief ~1-2%
`ASSERTION_FAILURE` recall at RL step 25, extinguished back to 0% by RL step
50 and never recovering.

Root cause, in two parts:

1. **GRPO can't fix it directly.** Prediction-label tokens are masked out of
   the GRPO loss ([research_specs.md § Arm B, consequence
   predictor](research_specs.md#arm-b-consequence-predictor)): final reward
   depends only on whether `apply_patch`/`python_test`/`FINAL` succeed,
   never on what `<PREDICTION>` said. GRPO carries no *direct* loss term on
   those tokens, but GRPO and the auxiliary CE update the same shared
   transformer weights, so gradient at the surrounding `<DECISION>`/action
   tokens can still reshape the masked positions indirectly. The auxiliary
   CE (`λ=0.1`) is the only *direct* teacher here, not the only thing
   capable of moving those probabilities. The step-25 blip is
   early-optimization noise nothing directly defends, so it doesn't last.
2. **SFT starts collapsed by construction, but the symmetry was
   deliberate.** Across all 212 Arm B SFT traces, `<PREDICTION>` labels
   split roughly 257 PASS : 45 real-failure (85%/15%). The 70 recovery
   traces were built symmetric on purpose: half (25 `visible` + 10
   `deep_visible`) teach recovery from a *wrong* PASS, honest mistake,
   caught by the real test, then fixed, and half (25 `shadow` + 10
   `deep_shadow`) teach recovery via a *correct* failure prediction and
   REVISE. The intent was for RL to sample both modes and improve both. RL
   did sample both, but with reward paying the two identically, the policy
   settled into the easier one: guess PASS and let `python_test` sort it
   out.
   `RUNTIME_ERROR` is detectable from surface code features (undefined
   vars, index risk) without simulating the algorithm against the test
   cases, a cheaper pattern, and the only non-PASS one that stuck.

Not reward hacking: the masked-out tokens mean there's nothing for GRPO to
game, only indifference. Call it poor reward shaping: a low-weight,
uniformly-per-token CE loss was the only direct correctness signal on that
skill, dominated by the 85%-majority PASS label. The two directions below
target this: reweighting CE toward rare classes attacks part 2, reward
shaping attacks part 1 by giving GRPO's own reward a reason to tell the two
kinds of rollouts apart.

Net: Arm B pays the full token overhead of predicting on every turn (see
efficiency numbers above) but the mechanism only covers ~15% of real
failures and is blind to the other ~57%. That's sufficient on its own to
explain why Arm B never pulled ahead, independent of the arm-vs-arm
significance question.

## What I learned

Not the results table, the methodology and architecture lessons that came
from digging into why the results looked the way they did.

1. **A matched comparison isn't an ablation.** Testing two complete,
   multi-part systems against each other (see **Scope**, above) can tell
   you the bundle didn't clearly win; it can't tell you which piece
   mattered. Isolating "prediction" as a single causal factor needs a
   compute/action-matched ablation, not this two-arm design.
2. **An auxiliary loss needs its own reward path, or it's the only
   *direct* teacher.** Masking prediction-label tokens out of GRPO's loss
   (see Root cause, above) is a reasonable design choice, but it leaves the
   low-weight, uniformly-weighted auxiliary CE as the only *direct* teacher
   for that skill. A weak sole teacher produces a weak skill, independent
   of how good the rest of the system is.
3. **A curriculum can teach both paths; reward decides which survives.**
   Arm B's recovery traces deliberately teach two modes: recover from a
   wrong PASS (visible), and recover via a correct failure prediction
   (shadow), split half and half so RL would sample and improve both. With
   reward paying the two identically, the easier mode won and the shadow
   skill decayed. The curriculum sets the menu; reward sets the diet.
4. **Rare at one stage isn't rare at another.** Only 37 of 302 SFT
   prediction labels are real `ASSERTION_FAILURE` examples, genuinely
   thin. But checking real per-step RL logs (not just final eval) showed
   the model sees this outcome constantly during RL, a third to half of
   every step. Diagnosing a persistent 0% recall as "not enough examples"
   would have been wrong; checking the actual training-time logs instead
   of assuming the SFT-time distribution still applied caught that.
5. **One training run is an anecdote.** The first seed made Arm A look
   ahead at step 100 and significantly ahead at step 25. Both effects were
   seed noise, gone under a second independent run with the same setup.
   Two-seed replication (now standard for both arms here) is what turned
   an appealing headline into a checked claim.
6. **No visible tokens doesn't mean "no computation," and it isn't
   automatically a dead end.** Every `<PREDICTION>` tag is emitted
   immediately after `apply_patch`, no reasoning tokens in between, but
   ECHO (Shrivastava, Kauffmann, Awadallah & Papailiopoulos,
   ["Terminal Agents Learn World Models for
   Free"](https://arxiv.org/abs/2605.24517), 2026, this project's own
   inspiration) trains a CE loss with no visible reasoning either, and
   still doubles pass@1 on TerminalBench-2.0. So auxiliary prediction with
   no visible tokens plainly can work. The likely difference: ECHO's target is
   the full, dense, multi-token environment observation, forcing
   token-by-token computation through what happened; ours is a single
   terse label from a 6-way enum. A denser prediction target (the specific
   failing assertion or expected value, not just an outcome class) may be a
   more direct fix for the `ASSERTION_FAILURE` blind spot than reweighting
   the current, thin classification target.

**Where to go next**, cheap experiments first, then the harder tests. The
real experiment is within-arm: most of these compare Arm B to itself with
one factor changed; the A-vs-B table was the guardrail that kept seed noise
out of the claims, not the verdict:

1. **Turn up `λ`** (`orchestrator.algo.alpha` in `configs/arm_b_rl.toml`,
   currently `0.1`). `λ` is the weight on the auxiliary CE loss, so the only
   thing that directly trains the prediction is also the smallest term in
   the loss. Turn it up, and weight the rare failure classes above `PASS`
   instead of uniform per-token weighting, then see whether prediction
   quality moves at all. Cheapest experiment on this list, and it separates
   "the signal was too weak" from "a 4B model can't do this."
2. **`λ = 0` ablation.** The mirror of the item above, and the one this
   write-up is missing (see lesson 1, above): run Arm B with the same
   prediction / `KEEP`-`REVISE` protocol, same trace format, same action
   budget, but `alpha = 0`: no verified-label CE at all. Every claim about
   the auxiliary objective teaching a real distinction currently rests on
   comparing Arm B to Arm A and to its own SFT checkpoint, neither of which
   holds the protocol fixed. If the learned outcome distinction survives at
   `λ = 0`, it came from the protocol or from GRPO on the surrounding tokens,
   not from the CE. Cheapest test of the project's central mechanism claim.
3. **Train longer than 100 steps.** 100 GRPO steps may simply be too short
   to judge this. Nothing has plateaued: two of the four runs are still
   climbing at step 100 (+2.8 and +2.4 points over step 75), and the other
   two are down only 0.6, inside the noise. Comparing the arms at a point
   where both are still moving risks reading a transient as a result. Run to
   convergence, then compare.
4. **Reward shaping.** Final task reward pays out identically regardless of
   whether the prediction was right, so GRPO carries no direct incentive
   toward good predictions. Hypothesis: extra reward for
   `true failure + predicted failure + REVISE` and
   `true PASS + predicted PASS + KEEP`. One nuance (see Root cause, above):
   GRPO already masks prediction-label tokens from its own loss
   (`rl_weights=0`), so a shaped reward's *direct* gradient still lands on
   the surrounding `<DECISION>`/action tokens, not the label positions, a
   real indirect effect via shared weights, but not a direct grade on the
   label choice. Decision-following is already 96-100% consistent, so
   there's limited headroom there. Squarely targeting prediction
   correctness likely still means lifting the mask, not just adding a
   reward term.
5. **Ablate the gate.** Test prediction-with-behavioral-gating (current
   Arm B) against prediction-as-pure-auxiliary-signal (ECHO-style, no
   `KEEP`/`REVISE` control), holding the rest of the bundle fixed. This is
   the one piece that's actually new relative to ECHO, and it's never been
   tested in isolation.
6. **Forced-`KEEP` ablation.** Take Arm B exactly as trained and disable
   `REVISE` at rollout time: keep the `<PREDICTION>` tag, keep the CE, but
   make every decision a `KEEP`. If pass@1 recovers toward Arm A, the
   decision gate, not prediction itself, is the failure mode. As it
   stands, the `REVISE`-path analysis above is correlational: `REVISE`
   rollouts are also the rollouts the model already judged risky, so their
   low recovery rate could be selection rather than damage. Forcing the
   decision breaks that confound.
7. **Budget-neutral `REVISE` ablation.** The strongest mechanism claim here
   is that `REVISE` hurts because it burns actions without returning
   feedback. Test it directly: make `REVISE` free (don't charge it against
   the tool budget), or raise the budget enough that revision can't cause
   exhaustion, and hold everything else fixed. If the deficit disappears,
   the cost is action exhaustion; if it survives a budget that can't be
   exhausted, the problem is that `REVISE` discards a patch without ever
   observing why, and the fix is a different repair action, not a bigger
   budget.
8. **Denser prediction target.** Not another SFT pass on the same label:
   change what's being predicted. Today's target is a single token from a
   6-way enum. Predicting the specific failing assertion or expected/actual
   value instead forces multi-token, generative simulation of the test,
   closer to how ECHO's dense observation-token target works, and may be why
   the model never learned to simulate in the first place.
9. **Isolate the causal effect.** Right now the prediction and the
   decision always agree: `KEEP` follows `PASS`, `REVISE` follows a
   predicted failure, ~100% of the time. So when a rollout goes well, there
   is no way to tell whether it went well because the *prediction* was
   right, or just because of the *action* that prediction triggered. Break
   the link: on some rollouts, pick `KEEP` or `REVISE` at random instead of
   following the prediction. If reward still tracks whether the prediction
   was correct, the prediction is doing real work.
10. **Scale beyond one model and one benchmark.** n=500 on MBPP with a 4B
   model that may have seen MBPP in pretraining can't rule out idiosyncrasy.
   Replicating across model scale or on a benchmark unlikely to be
   memorized is what would make this a claim, not just a run.

## Provenance: how to verify every claim in the write-up

Everything asserted in [`index.html`](index.html) is derived from files in this
repo or in `RESULTS_PUBLISHED/`. This section says which file, which field, and
which mistakes to avoid, so any of it can be re-derived or falsified without
trusting the prose.

**If you are a coding agent auditing this project, read this whole section
first.** Every mistake listed under "Four ways to get this wrong" below was
actually made during a review of this work, and each one produced a confident,
wrong conclusion that survived several rounds before the data caught it.

### What each part of the project is responsible for

| path | responsibility |
|---|---|
| `src/glyph/program.py` | the rollout program, run inside the sandbox. The agent loop, turn-shape validation, `<PREDICTION>`/`<DECISION>` parsing, shadow-test execution, and the `prediction_targets` records everything downstream reads. |
| `src/glyph/taskset.py` | task loading, real test execution, and **all** reward and metric definitions. If you want to know what the reward actually paid for, it is here and nowhere else. |
| `src/glyph/prime_rl.py` | `PredictAlgorithm`: the verified-label CE auxiliary sample, and (at the run commit) the `rl_weights = 0.0` masking of the prediction span. |
| `src/glyph/harness.py` | Verifiers v1 harness; ships `program.py` into the sandbox verbatim. |
| `src/glyph/chat.py` | system prompts and ChatML rendering, source of the `<PREDICTION>OUTCOME</PREDICTION>` placeholder that skews naive label counts. |
| `src/glyph/passk.py`, `cli.py` | eval-trace reporting |
| `data/prepare.py`, `recovery.py`, `validate.py` | generate and validate the MBPP splits and the hand-designed SFT traces. **The curriculum's class balance is decided here**, which is why two outcome classes were never demonstrated. |
| `data/sft/arm_{a,b}/train.jsonl` | the actual SFT traces, in-repo. Ground truth for any claim about what the model was shown. |
| `configs/arm_{a,b}_{sft,rl}.toml` | training configs. Note these are the *current* configs; what actually ran is `RESULTS_PUBLISHED/*_shared/run_default/control/orch.toml`. |
| `docs/stats.py` | paired McNemar + bootstrap CI. Use this rather than rolling your own. |
| `docs/REPRODUCTION.md` | exact commands, checkpoint map, full comparison tables |
| `docs/index.html` | the published write-up. Every claim in it should be traceable through this section. |
| `docs/blog.md` (this file) | the contemporaneous development record, reverse-chronological |
| `~/Desktop/portfolio` (separate repo) | the portfolio site. Carries a condensed version of the same claims and must not drift from `index.html`. |

### Scope

The published numbers come from **four runs only**: Arm A and Arm B, seeds 42
and 43. `_V1` in a directory name means seed 43; no suffix means seed 42.

| directory | what it is |
|---|---|
| `RESULTS_PUBLISHED/RL_ARM_{A,B}[_V1]_shared/` | the training run: `run_default/control/orch.toml`, per-step rollouts, checkpoints, W&B, trainer logs. "shared" because the four checkpoint evals below all came out of this one run. |
| `RESULTS_PUBLISHED/RL_ARM_{A,B}[_V1]_{25,50,75,100}/eval/` | eval of one RL checkpoint, n=500, one directory each |
| `RESULTS_PUBLISHED/RL_ARM_{A,B}_sft/eval/` | eval of the SFT checkpoint (RL step 0). No `_V1_sft` exists; both seeds start from the *same* SFT checkpoint, which is why the results table reads "50.6% (same ckpt)". |
| `RESULTS_EXPLORATORY/` | everything that does **not** back a published number. See its README. |
| `RESULTS_SFT/` | SFT training artifacts, both arms |

### Which code ran

SFT at [`6884983`](https://github.com/JayZenith/PREDICT/commit/6884983), RLVR at
[`9eefac7`](https://github.com/JayZenith/PREDICT/commit/9eefac7) (21 Jul 00:15;
the first run started 01:26). Both pinned in the top-level `README.md`.

The configs that actually ran are at
`RESULTS_PUBLISHED/RL_ARM_*_shared/run_default/control/orch.toml`: Arm A is
`algo.type = "grpo"`, Arm B is `algo.type = "predict"` with `alpha = 0.1`.
Neither sets `prediction_reward_weight`, so it is 0.0 in both.

**This matters more than it looks.** At `9eefac7`,
`PredictAlgorithm._mask_sampled_labels` sets `rl_weights = 0.0` on exactly the
sampled outcome-label tokens, so in these runs the prediction is trained by
**CE alone** and the policy gradient never reaches it. That method was *deleted*
on 1 Aug by `c5b261b` ("RLVR the prediction span directly"), which lives only on
the `rl/*` branches. `main` still carries the masked version, byte-identical to
`9eefac7`.

So: reading `src/glyph/prime_rl.py` on a feature branch will tell you the
opposite of what these runs did. Check the branch first.

### Four ways to get this wrong

1. **`mask` in a trace is not the RL weight.** `nodes[].mask` is the
   *tokenization* loss mask; it is true on the prediction tokens whether or not
   they were masked from RL. The `rl_weights` stream lives on PRIME-RL's
   `TrainingSample` and is never serialized to `traces.jsonl`. PRIME-RL fills
   only *absent* streams with 1.0 (`trainer/batch.py`, `STREAM_FILL`), and
   `stamp_loss_routing` returns early for `action_loss_type = "rl"` without
   clobbering a stream the algorithm already wrote, which is why the explicit
   zeros survive. **No amount of trace inspection can settle this; read the
   source at the right commit.**
2. **Pair on `task.data.name`.** Not `task.name`, and not `id`; `id` is a
   per-run UUID, so pairing on it silently matches nothing and McNemar returns
   n=0 instead of erroring.
3. **Count SFT prediction labels in assistant turns only.** Every trace's system
   prompt contains a literal `<PREDICTION>OUTCOME</PREDICTION>` placeholder, so a
   naive regex over the whole record over-counts by exactly 212.
4. **Don't mix eval predictions with training-rollout predictions.** The recall
   and precision figures come from `*/eval/`, which is greedy. The exposure
   table and the per-step trajectories come from `rollouts/step_*/`, which is
   sampled at temperature. The two distributions differ and are not
   interchangeable.

### Claim → source

| claim in `index.html` | source | field |
|---|---|---|
| pass@1 table, all 20 cells | `RL_ARM_*/eval/**/traces.jsonl` | `metrics.passed` |
| within-arm RLVR gains (p=0.0002–0.032) | same, SFT vs RL 100 | McNemar on `metrics.passed` |
| first-patch-correct vs recovery decomposition | same | `metrics.first_patch_correct`, `metrics.recovered_after_executed_failure`, `metrics.had_executed_failure` |
| recovery is underpowered, not flat | same | raw counts: Arm A 39/277 → 43/257, Arm B 36/282 → 33/266. ~40 successes per run, so the p=0.58–0.74 is weak evidence of no change, not evidence of none. Do not report it as "RLVR does not train recovery" |
| prediction recall chart (92/63/0%) | `RL_ARM_B_100/eval/**/traces.jsonl` | `info.glyph.prediction_targets[]`, `sampled_prediction` vs `actual` |
| RUNTIME_ERROR precision 62.5% / 64.1%, 208 predicted vs 207 actual, 16.1% base rate | `RL_ARM_B{,_V1}_100/eval/**/traces.jsonl` | same. Precision = correct RUNTIME_ERROR predictions ÷ all RUNTIME_ERROR predictions |
| SFT predicts PASS on 100% of candidates | `RL_ARM_B_sft/eval/**/traces.jsonl` | same |
| decision-following (96–99% / 100%) | `RL_ARM_B{,_V1}_100/eval/**` | `prediction_targets[].decision` |
| exposure table, 15,122 labels | `RL_ARM_B_shared/run_default/rollouts/step_*/train/all/traces.jsonl` | same |
| per-step prediction trajectories | `RL_ARM_B{,_V1}_shared/.../rollouts/step_*/` | same. Per-step n is 36–236, so read trends across steps and seeds, never a single point. |
| SFT curriculum: 257 PASS / 37 ASSERTION_FAILURE / 8 RUNTIME_ERROR / 0 SYNTAX_ERROR / 0 TIMEOUT | `data/sft/arm_b/train.jsonl` (212 traces, in-repo) | `<PREDICTION>` labels, assistant turns only |
| no reward term scores the prediction | `RL_ARM_B{,_V1}_shared/run_default/control/orch.toml` | `prediction_reward_weight` absent → 0.0; `rewards` in traces contains only `mbpp_reward` |

### Not recorded

No git sha is written into the run artifacts, not in `orch.toml`, the W&B
files, or the eval logs. The only sha-shaped string in the eval logs is a vLLM
compile-cache hash. The commit is known only because `README.md` pins it by
hand. **Stamp the sha into the run config next time.**

One eval record, `mbpp_231` in Arm B seed 43 step 100, is a truncated rollout
with an empty `metrics` dict. The published 53.6% counts it as not-passed
(268/500).

## Appendix

### Beyond Code Verification: Research Judgment

One unexpected lesson came from reviewing the experiment itself. I initially
described Arm A as a baseline and Arm B as a single added variable. That
framing was too clean: Arm B also changed the trace format, action protocol,
auxiliary loss, and decision space. The two arms are better understood as
comparative training systems, not a strict causal ablation.

The coding agent helped build and document the project, but it did not
challenge that experimental claim. Code has hard verifiers: tests, syntax,
execution, and logs. Research judgment has no equivalent oracle. Auditing a
claim requires reconstructing the entire experiment, identifying every
differing factor, and testing whether the conclusion is actually supported.

A stronger research agent would need to do more than make a project coherent
and functional. It would need to actively falsify the researcher's framing.
