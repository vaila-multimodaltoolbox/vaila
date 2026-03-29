# SAM 3 smoke tests

Place a short MP4 here as `test1000.mp4` (or any name — adjust the CLI). Prefer a small file (under 24 MB) so it stays compatible with the repo pre-commit hook.

## Hugging Face

1. Open [facebook/sam3](https://huggingface.co/facebook/sam3), log in, accept the license (wait if access is pending).
2. From the project root:

   ```bash
   uv run hf auth login
   ```

   Use `--force` if the cached account is not the one with access.

3. Optional: download weights once (skips Hub at run time if the file is present):

   ```bash
   uv run vaila/vaila_sam.py --download-weights
   ```

## VRAM (8GB cards)

SAM3 loads **all session frames** onto the GPU at once. Long clips (e.g. 1000 frames) can exceed VRAM. By default vailá caps to **256 frames** (evenly subsampled) via `SAM3_MAX_FRAMES`. Override:

- Full length (if you have enough VRAM): `SAM3_MAX_FRAMES=0` or `--max-frames 0`
- Custom cap: `--max-frames 400`

Weights belong under **`vaila/models/sam3/`** as `sam3.pt` (flat), or inside **`vaila/models/sam3/sam3_weights/`** if you moved the whole HF folder — both are auto-detected. Repo root `./sam3_weights/` is still a fallback but deprecated.

## Manual run (full pipeline)

Requires **NVIDIA CUDA**, `uv sync --extra sam` (and `--extra gpu` on the CUDA `pyproject` template if you use TensorRT/CUDA wheels).

```bash
cd /path/to/vaila
uv run vaila/vaila_sam.py -i tests/SAM/test1000.mp4 -o tests/SAM/ -t person
```

With an explicit frame cap:

```bash
uv run vaila/vaila_sam.py -i tests/SAM/test1000.mp4 -o tests/SAM/ -t person --max-frames 256
```

With an explicit checkpoint:

```bash
uv run vaila/vaila_sam.py -i tests/SAM/test1000.mp4 -o tests/SAM/ -t person \
  --checkpoint vaila/models/sam3/sam3.pt
```

Outputs appear under `tests/SAM/processed_sam_YYYYMMDD_HHMMSS/`.

## Automated pytest (optional, slow)

Runs only if **all** are true: `sam3` installed, CUDA available, `test1000.mp4` present, and env flag set:

```bash
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

Default `pytest tests/test_vaila_sam.py` keeps fast unit tests only.
