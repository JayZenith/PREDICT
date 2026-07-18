#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."
uv run python -m data.validate data/sft.jsonl
exec uv run --project .vendor/prime-rl sft @ "$PWD/configs/sft.toml"
