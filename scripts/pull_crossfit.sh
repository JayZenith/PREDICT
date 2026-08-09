#!/usr/bin/env bash
# Pull the evidence for a cross-fit run off a training instance before it is
# destroyed. The dataset itself belongs in git; everything here is the record
# of how it was produced -- sampling traces, probe logs, resolved configs.
#
# Model weights are deliberately excluded. The probes are throwaway: they exist
# only to sample the fold they were held out of, and the harvested traces
# already carry every candidate they wrote.
#
#   bash scripts/pull_crossfit.sh root@1.2.3.4 60704
set -euo pipefail

cd "$(dirname "$0")/.."

host="${1:-}"
port="${2:-22}"
remote="${3:-/workspace/PREDICT}"
if [[ -z "$host" ]]; then
  echo "usage: bash scripts/pull_crossfit.sh USER@HOST [PORT] [REMOTE_DIR]" >&2
  exit 2
fi

dest="RESULTS_CROSSFIT"
mkdir -p "$dest"

rsync -az --info=stats1 -e "ssh -p ${port}" \
  --prune-empty-dirs \
  --include='*/' \
  --include='traces.jsonl' \
  --include='*.log' \
  --include='*.toml' \
  --include='*.json' \
  --exclude='*' \
  "${host}:${remote}/outputs/" "${dest}/outputs/"

# The dataset is the one artifact that is version-controlled, so it goes back
# into the tree rather than into RESULTS_CROSSFIT.
rsync -az -e "ssh -p ${port}" \
  "${host}:${remote}/data/sft_harvested/" "data/sft_harvested/"

echo
echo "dataset  -> data/sft_harvested (commit this)"
echo "evidence -> ${dest}/outputs (gitignored)"
du -sh "$dest" data/sft_harvested
