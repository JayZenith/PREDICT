#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
if [[ "$arm" != "a" && "$arm" != "b" ]]; then
  echo "usage: bash scripts/train_rl.sh a|b [extra PRIME-RL arguments]" >&2
  exit 2
fi
shift

test -d "outputs/arm_${arm}_sft/weights/step_24" || {
  echo "SFT checkpoint missing; run: bash scripts/train_sft.sh $arm" >&2
  exit 1
}

uv run python -m data.validate data
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project .vendor/prime-rl \
  rl @ "$PWD/configs/arm_${arm}_rl.toml" "$@"
