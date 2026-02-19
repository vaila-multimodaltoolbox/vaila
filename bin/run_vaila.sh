#!/bin/bash

# Get the directory where this script resides (vaila/bin)
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# Project root is one level up (vaila/bin -> vaila/)
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

# Navigate to project root
cd "$PROJECT_ROOT" || exit 1

# Detect OS
OS_NAME="$(uname -s)"
if [[ "$OS_NAME" == "Darwin" ]]; then
    # Start vaila with GPU disabled for macOS (to avoid TensorRT/MPS issues in uv)
    # Using --no-sync to speed up start if already synced
    uv run --no-sync --no-extra gpu vaila.py
else
    # Linux/Windows(WSL)
    uv run --no-sync vaila.py
fi
