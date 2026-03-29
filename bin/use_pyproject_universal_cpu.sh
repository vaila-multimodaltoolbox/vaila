#!/usr/bin/env bash
# Switch vailá to the portable CPU manifest (laptops / no NVIDIA CUDA).
# Regenerates uv.lock for this matrix; run uv sync afterward.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SRC="pyproject_universal_cpu.toml"
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
echo "  uv sync --extra sam        # optional SAM 3 stack (video still needs NVIDIA CUDA at runtime)"
echo "  uv sync --extra upscaler"
