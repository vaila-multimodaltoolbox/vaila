#!/usr/bin/env bash
# Switch vailá to Linux + NVIDIA CUDA 12.8 (PyTorch cu128 index, TensorRT extra).
# Regenerates uv.lock; then use: uv sync --extra gpu [--extra sam]
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"
SRC="pyproject_linux_cuda12.toml"
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
echo "  uv sync --extra gpu"
echo "  uv sync --extra gpu --extra sam"
