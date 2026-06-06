#!/usr/bin/env bash
# Thin wrapper -> bin/setup_pyproject.sh --target=linux-cuda
# Kept for backward compatibility. Prefer running setup_pyproject.sh directly
# (interactive + cross-platform detection).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/bin/setup_pyproject.sh" \
    --target=linux-cuda \
    --extras=gpu \
    --non-interactive \
    --no-sync \
    "$@"
