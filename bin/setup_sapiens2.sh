#!/usr/bin/env bash
# ============================================================================
# setup_sapiens2.sh — Sapiens2 Pose + DETR detector bootstrap for vailá
# ============================================================================
# - Clones facebookresearch/sapiens2 into .local/third_party/sapiens2 (editable install).
# - Downloads default pose (1B) + DETR person detector into vaila/models/sapiens2/.
#
# Prerequisites: git, network, uv environment active from repo root.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SAPIENS2_DIR="${REPO_ROOT}/.local/third_party/sapiens2"
WEIGHTS_DIR="${REPO_ROOT}/vaila/models/sapiens2"

mkdir -p "$(dirname "${SAPIENS2_DIR}")"

echo ">> vailá — Sapiens2 Pose setup"
echo "   repo root: ${REPO_ROOT}"

if [[ -d "${SAPIENS2_DIR}/.git" ]]; then
  echo ">> [1/4] sapiens2 already cloned at ${SAPIENS2_DIR}; pulling latest..."
  git -C "${SAPIENS2_DIR}" pull --ff-only || {
    echo "   (warning) git pull failed; continuing with existing checkout"
  }
else
  echo ">> [1/4] Cloning facebookresearch/sapiens2 into ${SAPIENS2_DIR}..."
  git clone --depth 1 \
    https://github.com/facebookresearch/sapiens2.git \
    "${SAPIENS2_DIR}"
fi

echo ">> [2/4] Installing sapiens2 (editable) into the uv environment..."
(cd "${REPO_ROOT}" && uv pip install -e "${SAPIENS2_DIR}")

echo ">> [3/4] Downloading pose + detector weights to ${WEIGHTS_DIR}..."
mkdir -p "${WEIGHTS_DIR}/pose" "${WEIGHTS_DIR}/detector"
# Use vaila_sapiens (huggingface_hub API). ``uv run hf download`` can exit 1 via click.Exit(0).
(cd "${REPO_ROOT}" && uv run vaila/vaila_sapiens.py --download-weights --model 1b)

echo ">> [4/4] Validating layout..."
missing=0
for f in \
  "${WEIGHTS_DIR}/pose/sapiens2_1b_pose.safetensors" \
  "${WEIGHTS_DIR}/detector/detr-resnet-101-dc5/config.json"; do
  if [[ ! -f "${f}" ]]; then
    echo "   MISSING: ${f}"
    missing=1
  else
    echo "   OK:      ${f}"
  fi
done

if [[ "${missing}" -eq 0 ]]; then
  echo ""
  echo ">> Done. Sapiens2 Pose is ready."
  echo "   Optional: uv sync --extra sapiens"
  echo "   Quick test:"
  echo "     uv run vaila/vaila_sapiens.py \\"
  echo "       -i tests/markerless_2d_analysis/ \\"
  echo "       -o /tmp/sapiens_out --model 1b --dry-run"
  echo "   Full run:"
  echo "     uv run vaila/vaila_sapiens.py \\"
  echo "       -i tests/markerless_2d_analysis/ \\"
  echo "       -o /tmp/sapiens_out --model 1b"
  exit 0
else
  echo ""
  echo ">> Some weights are missing. Check Hugging Face access and retry:"
  echo "   uv run hf auth login"
  exit 1
fi
