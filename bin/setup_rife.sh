#!/usr/bin/env bash
# Practical-RIFE 4.25 installer. Updated: 30 October 2026; Version: 0.4.13.
# Uses the current vailá environment without changing CUDA groups or extras.
# Optional and relocatable: --home DIR installs into DIR, --export DIR copies it (no network).
set -euo pipefail
RIFE_REPO_ROOT="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
# No cd: a relative --home/--export DIR is relative to where you ran this, not to the repo.
uv run --no-sync --project "$RIFE_REPO_ROOT" python "$RIFE_REPO_ROOT/bin/setup_rife.py" "$@"
