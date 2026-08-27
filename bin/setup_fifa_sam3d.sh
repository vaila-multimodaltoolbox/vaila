#!/usr/bin/env bash
# ============================================================================
# setup_fifa_sam3d.sh — SAM 3D Body + weights bootstrap for the FIFA pipeline
# ============================================================================
# - Clones the Meta `sam-3d-body` repository into `sam_3d_body/` at the project
#   root. Upstream ships no pyproject.toml/setup.py, so it cannot be pip
#   installed; only its runtime dependencies are installed (--no-deps), and
#   `vaila/sam3dinov3.py::ensure_sam3d_importable()` puts the checkout root on
#   sys.path at runtime.
# - Downloads the gated Hugging Face weights for ``facebook/sam-3d-body-dinov3``
#   into ``vaila/models/sam-3d-dinov3/`` (``model.ckpt`` + ``assets/mhr_model.pt``).
# - Validates the layout and prints the next-step command.
#
# Prerequisites: git, network access, Hugging Face CLI (``uv run hf auth login``)
# and acceptance of the `facebook/sam-3d-body-dinov3` license on huggingface.co.
# ============================================================================
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
SAM3D_DIR="${REPO_ROOT}/sam_3d_body"
WEIGHTS_DIR="${REPO_ROOT}/vaila/models/sam-3d-dinov3"

echo ">> vaila FIFA — SAM 3D Body setup"
echo "   repo root: ${REPO_ROOT}"

# ---------------------------------------------------------------- sam_3d_body
if [[ -d "${SAM3D_DIR}/.git" ]]; then
  echo ">> [1/3] sam_3d_body already cloned at ${SAM3D_DIR}; pulling latest..."
  git -C "${SAM3D_DIR}" pull --ff-only || {
    echo "   (warning) git pull failed; continuing with existing checkout"
  }
else
  echo ">> [1/3] Cloning facebookresearch/sam-3d-body into ${SAM3D_DIR}..."
  git clone --depth 1 \
    https://github.com/facebookresearch/sam-3d-body.git \
    "${SAM3D_DIR}"
fi

# Upstream `sam-3d-body` ships NO pyproject.toml / setup.py, so `uv pip install -e`
# cannot work. The package is imported by putting the checkout root on sys.path;
# `vaila/sam3dinov3.py` does that automatically (see `ensure_sam3d_importable`),
# and VAILA_SAM3D_BODY_DIR overrides the location. Here we only install the
# runtime dependencies it needs, with --no-deps so the CUDA torch build is
# never resolved away.
echo ">> [1/3] Installing sam_3d_body runtime dependencies (--no-deps)..."
(cd "${REPO_ROOT}" && uv pip install --no-deps \
  mhr yacs omegaconf "antlr4-python3-runtime==4.9.3" roma trimesh braceexpand \
  pytorch-lightning torchmetrics lightning-utilities termcolor)
# termcolor: not pulled in transitively by anything else in a typical vailá
# venv (unlike timm/tqdm, which usually are) -- without it, torch.hub loading
# the DINOv3 backbone (dinov3/logging/__init__.py) fails with
# ModuleNotFoundError: No module named 'termcolor' on the very first GPU run
# (found running vaila/sam3dinov3.py end to end, 2026-08-05).

# ----------------------------------------------------------------- weights DL
echo ">> [2/3] Downloading facebook/sam-3d-body-dinov3 weights to ${WEIGHTS_DIR}..."
mkdir -p "${WEIGHTS_DIR}"
# Prefer huggingface_hub API (same pattern as bin/setup_sapiens2.sh). Older
# huggingface_hub + typer>=0.26 made ``hf download`` print click.Exit(0) and
# return exit 1 even after a successful transfer, aborting this script under
# ``set -e``. Stale local-dir locks can also leave progress at "0.00B".
if [[ -d "${WEIGHTS_DIR}/.cache/huggingface/download" ]]; then
  find "${WEIGHTS_DIR}/.cache/huggingface/download" -name '*.lock' -delete 2>/dev/null || true
fi
(cd "${REPO_ROOT}" && uv run python -c "
from pathlib import Path
from huggingface_hub import snapshot_download

root = Path(r'''${WEIGHTS_DIR}''')
root.mkdir(parents=True, exist_ok=True)
print(f'>> snapshot_download facebook/sam-3d-body-dinov3 -> {root}', flush=True)
snapshot_download(repo_id='facebook/sam-3d-body-dinov3', local_dir=str(root))
print('>> download finished', flush=True)
")

# -------------------------------------------------------------------- checks
echo ">> [3/3] Validating layout..."
missing=0
for f in "${WEIGHTS_DIR}/model.ckpt" "${WEIGHTS_DIR}/assets/mhr_model.pt"; do
  if [[ ! -f "${f}" ]]; then
    echo "   MISSING: ${f}"
    missing=1
  else
    echo "   OK:      ${f}"
  fi
done

if [[ "${missing}" -eq 0 ]]; then
  echo ""
  echo ">> Done. SAM 3D Body is ready for the FIFA Skeletal Tracking Light pipeline."
  echo "   Next steps:"
  echo "     uv run vaila/vaila_sam.py fifa bootstrap \\"
  echo "       --videos-dir /path/to/FIFA_Challenge_2026_Video_Data/Videos \\"
  echo "       --data-root  /path/to/FIFA/data"
  echo "     uv run vaila/vaila_sam.py fifa prepare --data-root /path/to/FIFA/data \\"
  echo "       --video-source /path/to/FIFA_Challenge_2026_Video_Data/Videos"
  echo "     uv run vaila/vaila_sam.py fifa preprocess --data-root /path/to/FIFA/data \\"
  echo "       --sequences /path/to/FIFA/data/sequences_full.txt"
  echo "     uv run vaila/vaila_sam.py fifa baseline   --data-root /path/to/FIFA/data \\"
  echo "       --sequences /path/to/FIFA/data/sequences_full.txt -o outputs/submission_full.npz"
  echo "     uv run vaila/vaila_sam.py fifa pack       --submission-full outputs/submission_full.npz \\"
  echo "       --data-root /path/to/FIFA/data --output-dir outputs/ --split val"
  exit 0
else
  echo ""
  echo ">> Weights are missing. Most likely:"
  echo "   1) You have not accepted the license on https://huggingface.co/facebook/sam-3d-body-dinov3"
  echo "   2) You have not run 'uv run hf auth login' with an account that has access."
  exit 2
fi
