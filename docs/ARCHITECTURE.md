# Architecture

## Shared agent

`GlyphHarness` launches one Verifiers v1 program in a fresh host subprocess
workspace on the training instance. The policy sees structured ChatML and can:

```text
read_file → apply_patch → python_test → FINAL
```

The MBPP assertions are shown verbatim in the task prompt, so the agent knows
the exact function signature and expected behavior up front; they are never
added to the editable project files. The agent still must run `python_test`
to find out whether its candidate satisfies them—test output is reduced to an
outcome class, not an assertion diff. Runtime-recorded calls and results—not
model claims—determine success.

Reward is binary:

```text
1 = visible passing python_test + passing final-state check + terminal FINAL
0 = everything else
```

The final-state check is hidden and does not count as an agent tool call. It
prevents an earlier passing test from masking a later broken edit.

There are no protocol penalties, partial-test rewards, compiler rewards,
length shaping, OPD, or reference KL.

## Matched arms

Arm A executes each applied candidate and reacts to test output.

Arm B must emit:

```text
<PREDICTION>OUTCOME</PREDICTION>
<DECISION>KEEP|REVISE</DECISION>
```

`KEEP` visibly tests the candidate. `REVISE` shadow-tests it, hides that result
from the agent, then permits another patch. Every candidate yields a runtime
record containing the sampled label, verified label, decision, and candidate
hash.

## Arm B loss

The patched PRIME-RL algorithm keeps two training views:

1. Raw rollout: GRPO trains agent actions. Sampled prediction-label tokens have
   zero GRPO weight.
2. Auxiliary view: the exact pre-prediction context is followed by the verified
   sandbox label. Only that label receives CE weight.

```text
loss = GRPO(actions) + λ × CE(verified outcome | problem, candidate, history)
```

An incorrect sampled label is never used as the CE target. Uniform-reward
groups remain available because Arm B can still receive verified-label CE.

## Failure policy

SFT preflight rejects any row over 1280 tokens or without terminal `FINAL:`.
Stack packing preserves whole traces. RL rejects truncated or over-4096-token
traces before they reach the trainer.

The repository pins PRIME-RL v0.7.0 (commit `d334ea5`) and Verifiers at commit
`6c64ce6`. `setup.sh` applies
the PREDICT objective, truncation-warning, and explicit-tokenizer-EOS patches
to the pinned checkout. A truncated training rollout is logged, dropped from
its GRPO group by the train sink, and never reaches the trainer; it does not
abort the run.
