#!/usr/bin/env bash
# Sample a probe over the SFT fold it was held out of, so its patches and the
# outcomes the verifier observes for them can become predictive SFT data.
#
# Sampling matches the RL training temperature: the point is to collect the
# mistakes a policy of this kind actually makes. The task file must be a fold
# from data/folds -- never the RL, validation or test set.
set -euo pipefail

cd "$(dirname "$0")/.."

model="${1:-}"
tasks="${2:-}"
rollouts="${3:-32}"
if [[ -z "$model" || -z "$tasks" ]]; then
  echo "usage: bash scripts/harvest.sh MODEL data/folds/fold_a_tasks.jsonl [rollouts]" >&2
  exit 2
fi
if [[ "$tasks" != data/folds/* ]]; then
  echo "refusing to harvest outside data/folds: $tasks" >&2
  exit 2
fi
shift 2
[[ $# -gt 0 ]] && shift 1

count="$(grep -c . "$tasks")"
exec uv run eval glyph \
  --harness.id glyph \
  --harness.arm b \
  --taskset.data-path "$tasks" \
  --sampling.temperature 0.8 \
  --sampling.max-tokens 512 \
  --max-total-tokens 4096 \
  -m "$model" -n "$count" -r "$rollouts" --no-push "$@"
