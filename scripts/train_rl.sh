#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
if [[ "$arm" != "a" && "$arm" != "b" ]]; then
  echo "usage: bash scripts/train_rl.sh a|b [variant] [extra PRIME-RL arguments]" >&2
  exit 2
fi
shift

# An optional variant selects a sibling config, e.g. "seeded" for the run over
# tasks that start from a verified failing candidate. The bare arm always means
# the published run.
config="configs/arm_${arm}_rl.toml"
if [[ $# -gt 0 && -f "configs/arm_${arm}_rl_${1}.toml" ]]; then
  config="configs/arm_${arm}_rl_${1}.toml"
  shift
fi

uv run python -m data.validate data
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"
exec uv run --project .vendor/prime-rl \
  --extra flash-attn \
  rl @ "$PWD/$config" "$@"
