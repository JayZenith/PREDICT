#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
if [[ "$arm" != "a" && "$arm" != "b" ]]; then
  echo "usage: bash scripts/train_rl.sh a|b [v2] [extra PRIME-RL arguments]" >&2
  exit 2
fi
shift

# An optional dataset variant picks a sibling config. Everything else about the
# run is identical, so the suffix is the only thing that moves.
variant=""
if [[ "${1:-}" == v[0-9]* ]]; then
  variant="_$1"
  shift
fi

config="$PWD/configs/arm_${arm}_rl${variant}.toml"
test -f "$config" || { echo "no such config: $config" >&2; exit 2; }

uv run python -m data.validate data
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project .vendor/prime-rl \
  --extra flash-attn \
  rl @ "$config" "$@"
