# SFT complete: moving to RLVR

Both arms were full-fine-tuned from `Qwen3-4B-Base` on verified MBPP
agent traces. Each run used 60 optimizer steps and a 768-token sequence limit. No trace was
truncated or excluded.

| Checkpoint | Final loss | RL-style sampling check |
|---|---:|---|
| [Arm A SFT](https://huggingface.co/JayZenith/SFT_ARM_A) | 0.0327 | 22/32 passed on 4 train tasks |
| [Arm B SFT](https://huggingface.co/JayZenith/SFT_ARM_B) | 0.0334 | 74/128 passed on 16 train tasks |

These are training-task readiness checks at `temperature=0.8`, `top_k=20`, and
eight samples per task—not held-out results.

Arm A produced one mixed-reward group, two all-pass groups, and one all-fail
group. Seven of its ten invalid traces exhausted the tool budget on one hard
task.

Arm B produced mixed rewards on 14/16 tasks, with no all-fail group or length
truncation. Its prediction accuracy was 84/160 (52.5%). Its main remaining
protocol error was extra text in a `CALL` turn: 41/128 rollouts.

The SFT checkpoints learned enough of each agent loop to start RLVR, while
leaving useful errors to optimize. Arm B especially has the mixed outcomes and
imperfect consequence predictions needed to test the auxiliary prediction
loss.

## Next

Train both arms with identical RL tasks, seeds, sampling, and tool budgets. Arm A
uses binary final-task GRPO. Arm B adds verified-label prediction CE. Select
checkpoints and λ on the 40-task validation split, then run the untouched
500-task test split once.

Local configs, logs, W&B runs, and raw sampling traces are archived under the
gitignored `PREDICT_SFT_RESULTS/` directory.
