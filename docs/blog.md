# SFT complete: moving to RLVR

Both arms were full-fine-tuned from `Qwen3-4B-Base` on verified MBPP agent
traces: 60 optimizer steps (nine epochs over 212 traces), 768-token sequence
limit, no trace truncated or excluded.

| Checkpoint | Final loss | val40 pass@1 (greedy) |
|---|---:|---:|
| [Arm A SFT](https://huggingface.co/JayZenith/SFT_ARM_A) | 0.0293 | 7/40 (17.5%) |
| [Arm B SFT](https://huggingface.co/JayZenith/SFT_ARM_B) | 0.0277 | 4/40 (10.0%) |

Nine epochs was a deliberate choice, not the default: a five-epoch checkpoint
(step 30) converged on loss but left `<|im_end|>` at only 0.5% probability at
turn boundaries, so `temperature=0.8` sampling — the RL sampling config —
produced malformed tokens on roughly 90% of turns and zero-reward rollouts
everywhere. At step 60, `<|im_end|>` is the dominant sampled token (~27%
probability) at the same position, and a 16-sample check at RL temperature
produced zero malformed turns.

Arm B's SFT set also needed a composition fix. An earlier version of the
recovery traces trained "predict failure, `KEEP` anyway" on all 70 recovery
examples — coherent predictions, an incoherent decision rule. The current set
splits the 70 recovery traces into 35 shadow (predict failure → `REVISE` →
fix → predict `PASS` → `KEEP` → pass) and 35 visible (predict `PASS` as an
honest mistake → `KEEP` → visible failure → fix → predict `PASS` → `KEEP` →
pass) traces, so every failure prediction in the data pairs with `REVISE` and
every `PASS` prediction pairs with `KEEP`. Sampling the fixed checkpoint at
RL temperature against an obviously-bad candidate now produces `REVISE`
correctly paired with a failure prediction (3/16 samples); the old data made
this pairing structurally impossible to learn.

Both checkpoints still solve MBPP tasks well below the base model's general
capability and Arm B's prediction head still defaults to `PASS`/`KEEP` under
greedy decoding — expected at this stage. The learning signal for
consequence prediction (`λ`-weighted CE against verified sandbox labels) and
for `REVISE` as a rewarded behavior (GRPO) only exists at the RL stage.

## Next

Train both arms with identical RL tasks, seeds, sampling, and tool budgets.
Arm A uses binary final-task GRPO. Arm B adds verified-label prediction CE.
Select checkpoints and λ on the 40-task validation split, then run the
untouched 500-task test split once.

Local configs, logs, W&B runs, and raw sampling traces are archived under the
gitignored `PREDICT_SFT_RESULTS/` directory.
