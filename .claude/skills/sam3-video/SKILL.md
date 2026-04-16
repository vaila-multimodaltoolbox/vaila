# SAM 3 Video Segmentation (vailá)

Use when the user works on **SAM 3 video segmentation**, text-prompt masks, `vaila_sam.py` (non-FIFA mode), or troubleshoots SAM 3 checkpoint/VRAM/BFloat16 issues.

---

## Overview

`vaila/vaila_sam.py` performs video segmentation using Meta's **SAM 3** (Segment Anything Model 3). It supports:

- Text-prompt masks (e.g., "person", "ball", "car")
- Single video or batch directory processing
- GUI mode (Tkinter dialog) and headless CLI
- Automatic VRAM management sized to GPU memory
- Overlay MP4 + per-frame mask PNG outputs

---

## Setup

```bash
# Standard (CPU pyproject template — inference still needs CUDA)
uv sync --extra sam

# Workstation (CUDA pyproject template)
bash bin/use_pyproject_linux_cuda.sh   # or Windows equivalent
uv sync --extra gpu --extra sam
```

### Weights (Hugging Face, gated)

The repo `facebook/sam3` is gated. Accept the license on [huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3), then:

```bash
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights
```

Or manually:

```bash
uv run hf download facebook/sam3 sam3.pt --local-dir vaila/models/sam3
```

---

## Checkpoint Resolution

The resolver tries these locations in order:

1. CLI `-w` / `--checkpoint` argument
2. Environment: `SAM3_CHECKPOINT` or `VAILA_SAM3_CHECKPOINT`
3. `vaila/models/sam3/sam3.pt`
4. `vaila/models/sam3/sam3.1_multiplex.pt`
5. `vaila/models/sam3/sam3_weights/sam3.pt` (nested HF layout)
6. Legacy: `<repo>/sam3_weights/sam3.pt`
7. Auto-download from Hugging Face Hub

Supported checkpoints: `sam3.pt` (original) and `sam3.1_multiplex.pt` (SAM 3.1 Multiplex).

**Important:** `vaila/models/sam-3d-dinov3/` contains SAM 3D Body / FIFA weights — these are **not** SAM 3 video. Passing them will raise `ValueError`.

---

## CLI Usage

```bash
# Single video
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person

# Batch (directory)
uv run vaila/vaila_sam.py -i videos_dir/ -o output/ -t person

# With VRAM cap
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person --max-frames 80

# Frame-by-frame fallback (low VRAM)
uv run vaila/vaila_sam.py -i video.mp4 -o output/ -t person --frame-by-frame

# GUI mode (no args)
uv run vaila/vaila_sam.py

# Download weights only
uv run vaila/vaila_sam.py --download-weights

# Open help in browser
uv run vaila/vaila_sam.py --open-help
```

### All CLI Flags

| Flag | Short | Default | Description |
|------|-------|---------|-------------|
| `--input` | `-i` | — | Input video file or directory |
| `--output` | `-o` | — | Output base directory |
| `--text` | `-t` | `person` | Text prompt |
| `--frame` | `-f` | `0` | Prompt frame index |
| `--weights`/`--checkpoint` | `-w` | auto | Checkpoint path (file or folder) |
| `--max-frames` | — | auto | VRAM frame cap; `0` = full clip |
| `--no-overlay` | — | — | Skip overlay MP4 |
| `--no-png` | — | — | Skip mask PNGs |
| `--frame-by-frame` | — | — | Per-frame fallback (no temporal tracking) |
| `--download-weights` | — | — | Download from HF |
| `--open-help` | — | — | Open help in browser |

---

## VRAM Management

SAM 3 loads **all session frames** onto the GPU. Long clips exceed VRAM easily.

| Strategy | How |
|----------|-----|
| Auto (default) | Frame cap computed from GPU VRAM (logged as `[SAM3 VRAM]`) |
| Environment | `SAM3_MAX_FRAMES=80` |
| CLI | `--max-frames 80` |
| Full clip | `--max-frames 0` (large VRAM only) |
| Per-frame fallback | `--frame-by-frame` (loses temporal tracking) |

vailá auto-sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first `import torch` to reduce fragmentation OOMs.

---

## Output Format

```
output/processed_sam_YYYYMMDD_HHMMSS/
  video_name/
    masks/              PNG per frame (unless --no-png)
    overlay.mp4         coloured mask overlay (unless --no-overlay)
    sam_frames_meta.csv frame-level metadata
    README_sam.txt      run parameters
```

---

## Key Functions

| Function | Description |
|----------|-------------|
| `run_sam3_on_video()` | Core: run SAM 3 on a single video (CLI batch path) |
| `run_sam_video()` | GUI entry: Tkinter dialog → batch |
| `download_sam3_weights_to_vaila_models()` | Download gated SAM 3 from HF Hub |
| `_resolve_sam3_checkpoint_file()` | Auto-detect checkpoint location |
| `_maybe_subsample_video_for_vram()` | VRAM auto-sizing logic |
| `_composite_masks_bgr()` | Overlay coloured masks on BGR frames |
| `_resolve_bpe_path()` | Find CLIP BPE vocabulary file |
| `_patch_sam3_disable_perpetual_tracker_autocast()` | BFloat16 fix |
| `_patch_sam3_force_fp32_tracker_backbone_features()` | BFloat16 fix |
| `main()` | CLI entry point + `fifa` dispatch |

---

## Troubleshooting

### BFloat16 type mismatch

```
ERROR: Input type (c10::BFloat16) and bias type (float) should be the same
```

vailá monkey-patches SAM 3's tracker to disable autocast and force FP32 on backbone features. If the error persists:
- Ensure you have the latest `vaila_sam.py`
- Try `--frame-by-frame` as a workaround
- Check PyTorch version compatibility with SAM 3

### CUDA OOM

```
torch.cuda.OutOfMemoryError
```

- Reduce frame cap: `--max-frames 64` (or lower)
- Use `--frame-by-frame` for very long clips
- Close other GPU processes
- Check `nvidia-smi` for available VRAM

### HF 403 / Gated Repo

```
GatedRepoError: ... not in the authorized list
```

1. Accept the license at https://huggingface.co/facebook/sam3
2. `uv run hf auth login --force` (ensure the correct account)
3. Retry: `uv run vaila/vaila_sam.py --download-weights`
4. Or set `HF_TOKEN=hf_...` environment variable

### Missing BPE File

```
FileNotFoundError: ... bpe_simple_vocab_16e6.txt.gz
```

The SAM 3 PyPI wheel may omit the CLIP BPE vocabulary. vailá falls back to `boxmot`'s copy. Ensure `boxmot` is installed or place the file manually under the `sam3` package's `assets/` directory.

---

## Testing

```bash
# Unit tests (no GPU, no weights)
uv run pytest tests/test_vaila_sam.py -v

# GPU smoke test (needs CUDA + sam3 + test1000.mp4)
export VAILA_TEST_SAM_GPU=1
uv run pytest tests/test_vaila_sam.py -v -k sam3_smoke
```

Place a short MP4 at `tests/SAM/test1000.mp4` for smoke tests (see `tests/SAM/README.md`).

---

## Code Map

| File | Role |
|------|------|
| `vaila/vaila_sam.py` | All SAM 3 video logic + CLI + GUI + FIFA dispatch |
| `vaila/models/sam3/` | Checkpoint directory (sam3.pt, sam3.1_multiplex.pt) |
| `vaila/models/sam3/merges.txt` | BPE fallback vocabulary |
| `vaila/help/vaila_sam.html` | Browser help page |
| `vaila/help/vaila_sam.md` | Markdown help source |
| `tests/test_vaila_sam.py` | Unit + GPU smoke tests |
| `tests/SAM/README.md` | Test setup instructions |
