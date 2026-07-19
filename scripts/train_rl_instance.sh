#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

arm="${1:-}"
case "$arm" in
  a)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1}"
    ;;
  b)
    export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-2,3}"
    ;;
  *)
    echo "usage: bash scripts/train_rl_instance.sh a|b [extra PRIME-RL arguments]" >&2
    exit 2
    ;;
esac
shift

export HF_HOME="${HF_HOME:-/workspace/.hf_home}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONUNBUFFERED=1

exec bash scripts/train_rl.sh "$arm" "$@"
