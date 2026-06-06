#!/usr/bin/env bash
# Thin wrapper -> bin/setup_pyproject.sh --target=cpu
# Kept for backward compatibility. Prefer running setup_pyproject.sh directly
# (interactive + cross-platform detection).
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
exec bash "$ROOT/bin/setup_pyproject.sh" \
    --target=cpu \
    --non-interactive \
    --no-sync \
    "$@"
