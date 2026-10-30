#!/usr/bin/env bash
# Practical-RIFE 4.25 installer. Updated: 29 October 2026; Version: 0.4.12.
# Uses the current vailá environment without changing CUDA groups or extras.
set -euo pipefail
RIFE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$RIFE_REPO_ROOT"
uv run --no-sync python bin/setup_rife.py "$@"
