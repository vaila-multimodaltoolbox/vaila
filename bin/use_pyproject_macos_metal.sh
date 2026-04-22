#!/usr/bin/env bash
# Switch vailá to macOS Metal/MPS manifest (Apple Silicon or Intel template file).
# Regenerates uv.lock; run uv sync afterward.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SRC="pyproject_macos.toml"
if [[ ! -f "$SRC" ]]; then
	echo "error: missing $SRC" >&2
	exit 1
fi
cp "$SRC" pyproject.toml
echo "pyproject.toml <- $SRC"
echo "Running uv lock ..."
uv lock
echo ""
echo "Next (examples):"
echo "  uv sync"
echo "  uv sync --extra sam        # SAM 3 video path still expects NVIDIA CUDA in vaila_sam.py"
