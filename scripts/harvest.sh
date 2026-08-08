#!/usr/bin/env bash
# Sample an Arm B policy over the RL training pool so its own patches, and the
# outcomes the verifier observes for them, can be turned into predictive SFT
# data. Sampling matches the RL training temperature: the point is to collect
# the mistakes the policy actually makes while it is being trained.
set -euo pipefail

cd "$(dirname "$0")/.."

model="${1:-}"
rollouts="${2:-32}"
if [[ -z "$model" ]]; then
  echo "usage: bash scripts/harvest.sh MODEL [rollouts] [extra eval arguments]" >&2
  exit 2
fi
shift 1
[[ $# -gt 0 ]] && shift 1

uv run python -m data.validate data
exec uv run eval glyph \
  --harness.id glyph \
  --harness.arm b \
  --taskset.data-path data/arm_b_train.jsonl \
  --sampling.temperature 0.8 \
  --sampling.max-tokens 512 \
  --max-total-tokens 4096 \
  -m "$model" -n 212 -r "$rollouts" --no-push "$@"
