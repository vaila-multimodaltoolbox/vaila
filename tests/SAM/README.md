# SAM 3 smoke tests

Place a short MP4 here as `test1000.mp4` (or any name — adjust the CLI). Prefer a small file (under 20 MiB) so it stays compatible with the repo pre-commit hook.

## Sample videos

| File | Frames | Notes |
|------|--------|-------|
| `test1000.mp4` | ~1000 | general smoke test |

Both files are gitignored (place locally; `*.mp4` is in `.gitignore`).

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

## Weights / Checkpoints

The resolver auto-detects checkpoint files under `vaila/models/sam3/` in this order:

1. `sam3.pt` (original SAM 3)
2. `sam3.1_multiplex.pt` (SAM 3.1 Multiplex)

Both flat (`vaila/models/sam3/<name>`) and nested (`vaila/models/sam3/sam3_weights/<name>`) layouts are supported. Legacy repo root `./sam3_weights/` is still a fallback but deprecated.

**Important:** `vaila/models/sam-3d-dinov3/` is **not** for this tool — that's the SAM 3D Body / FIFA pipeline.

## VRAM (8 GiB cards)

SAM3 loads **all session frames** for a session onto the GPU. Long clips exceed VRAM easily.

- **Default:** vailá picks a frame cap from your GPU size (see `[SAM3 VRAM]` in the terminal). It also sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` to reduce fragmentation OOMs.
- **Override:** `SAM3_MAX_FRAMES=80` or `--max-frames 80` if you still hit CUDA OOM.
- **Full clip (large VRAM only):** `SAM3_MAX_FRAMES=0` or `--max-frames 0`

## Manual run (full pipeline)

Requires **NVIDIA CUDA**, `uv sync --extra sam` (and `--extra gpu` on the CUDA `pyproject` template if you use TensorRT/CUDA wheels).

### BRA_KOR 60-frame example

```bash
cd /path/to/vaila
uv run vaila/vaila_sam.py \
  -i tests/SAM/BRA_KOR_230503_frame_1_to_61_h264.mp4 \
  -o tests/SAM/ \
  -w vaila/models/sam3/ \
  -t person
```

Or pass the full checkpoint path explicitly:

```bash
uv run vaila/vaila_sam.py \
  -i tests/SAM/BRA_KOR_230503_frame_1_to_61_h264.mp4 \
  -o tests/SAM/ \
  -w vaila/models/sam3/sam3.1_multiplex.pt \
  -t person
```

### test1000 example

```bash
uv run vaila/vaila_sam.py -i tests/SAM/test1000.mp4 -o tests/SAM/ -t person
```

With an explicit frame cap:

```bash
uv run vaila/vaila_sam.py -i tests/SAM/test1000.mp4 -o tests/SAM/ -t person --max-frames 80
```

### CLI flags

| Flag | Short | Description |
|------|-------|-------------|
| `--input` | `-i` | Input video file or directory (batch) |
| `--output` | `-o` | Output base directory |
| `--weights` / `--checkpoint` | `-w` | Path to SAM 3 weights (file or folder); auto-detected if omitted |
| `--text` | `-t` | Text prompt (default: `person`) |
| `--frame` | `-f` | Prompt frame index (default: `0`) |
| `--max-frames` | | Max frames on GPU (default: auto from VRAM; `0` = full clip) |
| `--no-overlay` | | Skip overlay MP4 output |
| `--no-png` | | Skip mask PNG output |

Outputs appear under `tests/SAM/processed_sam_YYYYMMDD_HHMMSS/`.

## Automated pytest (optional, slow)

Runs only if **all** are true: `sam3` installed, CUDA available, `test1000.mp4` present, and env flag set:

```bash
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

Default `pytest tests/test_vaila_sam.py` keeps fast unit tests only.
