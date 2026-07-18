# Canonical Agent Traces

Both arms use explicit ChatML, the same task, and matched candidate patches.
Arm A reacts to test output. Arm B predicts the outcome before testing and uses
that prediction to choose `KEEP` or `REVISE`.

## Arm A — reactive recovery

```text
<|im_start|>system
You are a Python coding agent. Use tools when needed. After FINAL, stop immediately.
<|im_end|>

<|im_start|>user
Implement is_even(n) in solution.py. Run the tests.

The project is at data/blueprints/mbpp_example.
<|im_end|>

<|im_start|>assistant
CALL read_file {"id":"c1","file_path":"data/blueprints/mbpp_example/solution.py"}
<|im_end|>

<|im_start|>tool
RESULT c1:
status: success
stdout:
# Write your function here.
<|im_end|>

<|im_start|>assistant
CALL apply_patch {"id":"c2","file_path":"data/blueprints/mbpp_example/solution.py","find":"# Write your function here.\n","replace":"def is_even(n):\n    return n % 2 == 1\n"}
<|im_end|>

<|im_start|>tool
RESULT c2:
status: success
stdout:
patch applied
<|im_end|>

<|im_start|>assistant
CALL python_test {"id":"c3","project_path":"data/blueprints/mbpp_example"}
<|im_end|>

<|im_start|>tool
RESULT c3:
status: failed
stderr:
hidden tests failed
<|im_end|>

<|im_start|>assistant
CALL apply_patch {"id":"c4","file_path":"data/blueprints/mbpp_example/solution.py","find":"    return n % 2 == 1","replace":"    return n % 2 == 0"}
<|im_end|>

<|im_start|>tool
RESULT c4:
status: success
stdout:
patch applied
<|im_end|>

<|im_start|>assistant
CALL python_test {"id":"c5","project_path":"data/blueprints/mbpp_example"}
<|im_end|>

<|im_start|>tool
RESULT c5:
status: success
stdout:
hidden tests passed
<|im_end|>

<|im_start|>assistant
FINAL: implemented is_even and passed the hidden tests.
<|im_end|>
```

## Arm B — pre-execution consequence prediction

```text
<|im_start|>system
You are a Python coding agent. Before testing an applied candidate, emit <PREDICTION>OUTCOME</PREDICTION>. OUTCOME is PASS, ASSERTION_FAILURE, RUNTIME_ERROR, SYNTAX_ERROR, TIMEOUT, or OTHER. Then emit <DECISION>KEEP</DECISION> to test or <DECISION>REVISE</DECISION> to patch again. After FINAL, stop.
<|im_end|>

<|im_start|>user
Implement is_even(n) in solution.py. Run the tests.

The project is at data/blueprints/mbpp_example.
<|im_end|>

<|im_start|>assistant
CALL read_file {"id":"c1","file_path":"data/blueprints/mbpp_example/solution.py"}
<|im_end|>

<|im_start|>tool
RESULT c1:
status: success
stdout:
# Write your function here.
<|im_end|>

<|im_start|>assistant
CALL apply_patch {"id":"c2","file_path":"data/blueprints/mbpp_example/solution.py","find":"# Write your function here.\n","replace":"def is_even(n):\n    return n % 2 == 1\n"}
<|im_end|>

<|im_start|>tool
RESULT c2:
status: success
stdout:
patch applied
<|im_end|>

<|im_start|>assistant
<PREDICTION>ASSERTION_FAILURE</PREDICTION>
<DECISION>REVISE</DECISION>
CALL apply_patch {"id":"c3","file_path":"data/blueprints/mbpp_example/solution.py","find":"    return n % 2 == 1","replace":"    return n % 2 == 0"}
<|im_end|>

<|im_start|>tool
RESULT c3:
status: success
stdout:
patch applied
<|im_end|>

<|im_start|>assistant
<PREDICTION>PASS</PREDICTION>
<DECISION>KEEP</DECISION>
CALL python_test {"id":"c4","project_path":"data/blueprints/mbpp_example"}
<|im_end|>

<|im_start|>tool
RESULT c4:
status: success
stdout:
hidden tests passed
<|im_end|>

<|im_start|>assistant
FINAL: implemented is_even and passed the hidden tests.
<|im_end|>
```

The rejected `c2` candidate is copied into an isolated hidden test process:

```json
{
  "candidate_call_id": "c2",
  "predicted": "ASSERTION_FAILURE",
  "actual": "ASSERTION_FAILURE",
  "decision": "REVISE"
}
```

That shadow result is training metadata. It never enters the agent's visible
ChatML and does not count as a visible tool call.

## Rules

1. Use `<|im_start|>` and `<|im_end|>` around every system, user, assistant,
   and tool turn.
2. Every `CALL` has a unique ID and exactly one matching runtime `RESULT`.
3. Arm B predicts only after its concrete candidate has been applied but before
   that candidate has been tested.
4. Predictions use one verified class:
   `PASS`, `ASSERTION_FAILURE`, `RUNTIME_ERROR`, `SYNTAX_ERROR`, `TIMEOUT`, or
   `OTHER`.
5. `KEEP` visibly executes `python_test`. `REVISE` snapshots and shadow-tests
   the rejected candidate before applying the next patch.
6. Runtime executions are authoritative. Assistant text cannot claim a pass.
7. `FINAL:` is valid only after a real visible passing test and must be the last
   assistant turn. A hidden final-state check must also pass.
8. Preserve sampled predictions for metrics. Prediction CE targets the verified
   sandbox label in a separate auxiliary view.
9. Mask prediction-label tokens from GRPO. Final task success remains the only
   scalar reward.
