#!/usr/bin/env bash
# Run the whole Arm B cross-fit: train a probe on each SFT fold, sample each
# probe on the fold it was held out of, and build the harvested SFT set from
# the union. One GPU is enough; RL is not part of this.
#
# Every stage is skipped if its output already exists, so the script can be
# rerun after an interruption.
set -euo pipefail

cd "$(dirname "$0")/.."

rollouts="${1:-32}"
port="${PORT:-8024}"
export WANDB_MODE="${WANDB_MODE:-offline}"
export PYTHONPATH="$PWD/src${PYTHONPATH:+:$PYTHONPATH}"

log() { echo "[cross_fit] $*" >&2; }

test -s data/folds/fold_a_tasks.jsonl || uv run python -m data.folds

probe_weights() {
  # PRIME-RL writes the servable weights under weights/step_N; the sibling
  # checkpoints/ directory holds resumable training state and stays empty when
  # the run only saves weights.
  find "outputs/probe_${1}_sft/weights" -mindepth 1 -maxdepth 1 -type d 2>/dev/null \
    | sort -t_ -k2 -n | tail -1
}

for fold in a b; do
  if [[ -n "$(probe_weights "$fold")" ]]; then
    log "probe $fold already trained"
    continue
  fi
  log "training probe $fold"
  uv run --project .vendor/prime-rl --extra flash-attn \
    sft @ "configs/probe_${fold}_sft.toml"
  test -n "$(probe_weights "$fold")" || { log "probe $fold saved no checkpoint"; exit 1; }
done

serve() {
  local weights="$1"
  uv run --project .vendor/prime-rl vllm serve "$weights" \
    --port "$port" --served-model-name probe \
    --gpu-memory-utilization 0.85 > "outputs/serve_probe.log" 2>&1 &
  echo $! > outputs/serve.pid
  until curl -sf "http://localhost:${port}/v1/models" | grep -q '"probe"'; do
    kill -0 "$(cat outputs/serve.pid)" 2>/dev/null || { log "server died"; exit 1; }
    sleep 10
  done
}

stop_serving() {
  [[ -f outputs/serve.pid ]] || return 0
  kill "$(cat outputs/serve.pid)" 2>/dev/null || true
  rm -f outputs/serve.pid
  while pgrep -f "vllm serve" >/dev/null; do sleep 3; done
}
trap stop_serving EXIT

# Probe a samples fold b and vice versa: each probe only ever sees tasks it was
# held out of.
for fold in a b; do
  other=$([[ "$fold" == a ]] && echo b || echo a)
  marker="outputs/harvest_fold_${other}.path"
  if [[ -s "$marker" ]]; then
    log "fold $other already harvested"
    continue
  fi
  log "serving probe $fold to sample fold $other"
  serve "$(probe_weights "$fold")"
  BASE_URL="http://localhost:${port}/v1" \
    bash scripts/harvest.sh probe "data/folds/fold_${other}_tasks.jsonl" "$rollouts" \
    2>&1 | tee "outputs/harvest_fold_${other}.log"
  stop_serving
  # The eval CLI prints the run directory it wrote; keep it for the builder.
  sed -e 's/\x1b\[[0-9;]*[a-zA-Z]//g' "outputs/harvest_fold_${other}.log" \
    | grep -oE 'outputs/glyph--[^ ]+' | tail -1 > "$marker"
  traces="$(cat "$marker")/traces.jsonl"
  # A run that cannot reach the server still finishes, writing a full file of
  # errored rollouts. Refuse to treat that as a harvest.
  if ! uv run python -c "
import json, sys
ok = sum(1 for line in open(sys.argv[1]) if not json.loads(line).get('errors'))
print(f'{ok} usable rollouts')
sys.exit(0 if ok else 1)
" "$traces"; then
    log "fold $other harvest produced no usable rollouts"
    rm -f "$marker"
    exit 1
  fi
  log "fold $other traces at $(cat "$marker")"
done

traces=()
for fold in a b; do
  traces+=("$(cat "outputs/harvest_fold_${fold}.path")/traces.jsonl")
done

log "building the harvested SFT set"
uv run python -m data.harvest --traces "${traces[@]}" --output data/sft_harvested
log "verifying every generated trace by execution"
uv run python -m data.verify_harvest --sft data/sft_harvested
