# Research Experiment Specs

## Research question

Can a coding agent trained to predict the sandbox consequence of its concrete
patch use that prediction to revise bad patches before execution, improving
correctness or reducing test usage?

The claim stays narrow:

> Adding sandbox-supervised pre-execution consequence prediction to an
> otherwise matched coding-agent pipeline improves patch decisions,
> correctness, or execution efficiency.

## Data

Use official full MBPP:

| Split | Task IDs | Purpose |
|---|---:|---|
| Train | 601–974 (374) | SFT traces and RL environments |
| Validation | 511–600 (90) | Select checkpoints and λ; detect overfitting |
| Final test | 11–510 (500) | Untouched final generalization result |

No MBPP+, frontier screening, internal train split, or task-ID split between SFT
and RL. The same 374 training tasks have different SFT-trace and RL-environment
representations:

```text
374 train tasks
├── Arm A SFT traces
├── Arm B prediction-augmented SFT traces
├── Arm A RL environments
└── Arm B RL environments
```

Each task becomes a blank `solution.py` plus hidden tests.

## SFT composition

| Trace type | Count | Arm A | Arm B |
|---|---:|---|---|
| Direct success | 250 | patch → test passes | correct patch → predict PASS → KEEP → test passes |
| One-step recovery | 124 | faulty patch → test fails → fix → test passes | faulty patch → predict failure → REVISE → corrected patch → predict PASS → KEEP → test passes |

Total: 374 traces per arm.

Assign tasks reproducibly with seed 42 and save the assignment manifest. Every
faulty state must genuinely fail and every final state must pass. Dataset
generation must fail if it cannot construct the required verified trace.

The arms match on task, candidate code, final solution, and real outcomes—not
exact ChatML. Arm B adds `PREDICTION` and `KEEP/REVISE`.

## Shared controls

Both arms start from `Qwen3-4B-Base` and use:

- Identical MBPP tasks.
- Matching direct-success and verified-recovery examples.
- The same optimizer, GRPO groups, seeds, visible rollout-token budget, and
  visible tool budget.
- The same final reward: `1` after a visible passing test, a passing hidden
  final-state check, and terminal `FINAL`; otherwise `0`.

## Arm A — reactive baseline

```text
read_file → apply_patch → python_test → inspect result → revise if needed → FINAL
```

Training:

```text
Base → ordinary agent SFT → ordinary GRPO
```

## Arm B — consequence predictor

```text
read_file
→ apply_patch candidate
→ PREDICT what python_test would return
→ KEEP or REVISE
```

- `KEEP` → execute `python_test` → inspect the real result.
- `REVISE` → snapshot the applied candidate for hidden shadow testing → apply
  another patch → predict again.

The candidate is already applied to the task workspace before prediction. It has
not yet been executed against the tests.

### Outcome classes

```text
PASS
ASSERTION_FAILURE
RUNTIME_ERROR
SYNTAX_ERROR
TIMEOUT
OTHER
```

The prediction is the expected test outcome for this problem and this exact
candidate patch. It is not free-form reasoning or traceback reproduction.

Training:

```text
Base
→ prediction-augmented agent SFT
→ GRPO + λ × prediction cross-entropy
```

Both losses happen in the same RL update:

```text
total loss = GRPO loss + λ × prediction loss
```

- GRPO trains patches, decisions, revisions, tool calls, and `FINAL` from final
  task success.
- Prediction CE trains the outcome label from the real sandbox result.
- λ controls how strongly prediction learning influences the model.
- Mask prediction-label tokens out of the GRPO loss. Predictions receive no
  scalar reward; only final test success determines RL reward.

## Prediction CE — exact training target

For every candidate:

```text
context =
    user problem
    + current solution.py
    + proposed patch
    + previous tool observations
    + <PREDICTION>

target = actual sandbox outcome

prediction CE = -log P(actual outcome | context)
```

Example:

```text
Problem: implement is_even
Candidate: return n % 2 == 1
Model sampled: PASS
Shadow sandbox: ASSERTION_FAILURE
```

The auxiliary target must be:

```text
<PREDICTION>ASSERTION_FAILURE</PREDICTION>
```

Do **not** calculate CE against the sampled `PASS`; that would reinforce the
model's mistake.

Implementation rule:

1. Preserve the raw sampled trajectory for metrics and the agent's actual
   `KEEP/REVISE` decision.
2. Construct an auxiliary training view ending immediately before the prediction
   label.
3. Teacher-force the verified sandbox label and compute CE only on its label
   tokens.
4. Mask the remaining auxiliary tokens from prediction CE.

This auxiliary view is necessary when the sampled and verified labels differ.
Unlike ECHO, the correct target is not already present in the raw rollout.

The intended learning signal is:

```text
problem + candidate code → likely execution consequence
```

## Shadow execution

If Arm B chooses `REVISE`, the evaluator copies the rejected candidate into an
isolated hidden test process inside the disposable sandbox. The result is never
shown to the agent and does not count as a visible tool call.

```text
Agent:     PREDICT ASSERTION_FAILURE → REVISE
Evaluator: secretly tests rejected candidate → ASSERTION_FAILURE
Trainer:   uses ASSERTION_FAILURE as the prediction-CE target
```

Shadow execution answers:

- Was the prediction correct?
- Did the agent reject a genuinely bad patch?
- Did it unnecessarily reject a good patch?
- What verified label should train the prediction?

## Checkpoints

```text
base
arm_a_sft
arm_a_rlvr
arm_b_sft
arm_b_rlvr
```

The main comparison is `arm_a_rlvr` versus `arm_b_rlvr`.

Use only the 90 validation tasks to select checkpoints and λ. Touch the 500-task
test split once, after freezing every choice.

## Final evaluation

Run both final policies once on all 500 test tasks with identical visible
budgets.

Report:

- Final pass@1.
- First-patch correctness.
- Prediction accuracy.
- Prediction macro-F1, so rare outcome classes count equally.
- Bad-patch rejection rate.
- Good-patch unnecessary-rejection rate.
- Recovery after an executed failure.
- Visible `python_test` calls per solved task.
- Average total tokens, revisions, and visible tool calls.

**Measure context savings; do not assume them.** Arm B removes some failed test
outputs but adds prediction and revision tokens.

## Hypothesis

Training consequence prediction improves patch judgment:

- Reject bad patches before visible testing.
- Keep good patches without unnecessary revision.
- Eventually generate better patches from stronger execution understanding.
- Reduce failed executions and context growth if prediction quality becomes
  useful.

Hidden-representation improvement is a hypothesis, not a result. The behavioral
claim must stand on prediction quality, decisions, correctness, and efficiency.

## Limits to report

- Arm B adds auxiliary CE sequences, so it uses more trainer tokens than Arm A.
  Report trainer tokens and GPU-hours. A compute-matched control is follow-up
  work, not part of this two-arm test.
- MBPP may appear in model pretraining. The matched comparison can test the
  intervention, but absolute MBPP performance is not a clean measure of novel
  Python capability.

## Novelty relative to ECHO

ECHO applies CE to the environment's actual output tokens, but those predictions
are not emitted as a pre-execution assistant action and do not control whether
the agent executes or revises.

This experiment combines:

- The same policy generates the patch and predicts its outcome.
- Prediction happens before execution.
- Prediction explicitly controls `KEEP` versus `REVISE`.
- Real sandbox outcomes supervise predictions during GRPO.
- Shadow execution supervises rejected candidates.
- Final correctness remains the only RL reward.
- Correct rejection and unnecessary rejection are directly measurable.

Agents may already predict consequences implicitly. This experiment makes that
prediction explicit, verifiable, trainable, and causally connected to action.
