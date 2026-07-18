#!/usr/bin/env bash
set -euo pipefail

readonly PRIME_RL_COMMIT="d334ea52940b47f426293a7d146239e3fbf91caa"
readonly VERIFIERS_COMMIT="6c64ce6a3a01e8edde7c3c0e8e5315fb236e9faa"
readonly PRIME_DIR=".vendor/prime-rl"
readonly PRIME_PATCH="$PWD/patches/prime-rl-fail-on-truncation.patch"

command -v uv >/dev/null || { echo "uv is required: https://docs.astral.sh/uv/" >&2; exit 1; }

if [[ ! -d "$PRIME_DIR/.git" ]]; then
  mkdir -p .vendor
  git clone https://github.com/PrimeIntellect-ai/prime-rl.git "$PRIME_DIR"
fi
git -C "$PRIME_DIR" fetch --tags origin
git -C "$PRIME_DIR" checkout --detach "$PRIME_RL_COMMIT"
git -C "$PRIME_DIR" submodule update --init deps/verifiers deps/renderers deps/pydantic-config deps/research-environments

if ! git -C "$PRIME_DIR" apply --reverse --check "$PRIME_PATCH" 2>/dev/null; then
  git -C "$PRIME_DIR" apply --check "$PRIME_PATCH"
  git -C "$PRIME_DIR" apply "$PRIME_PATCH"
fi

test "$(git -C "$PRIME_DIR/deps/verifiers" rev-parse HEAD)" = "$VERIFIERS_COMMIT"
uv sync --locked --group dev

echo "GLYPH is ready. PRIME-RL training dependencies sync on the first SFT/RL command."
