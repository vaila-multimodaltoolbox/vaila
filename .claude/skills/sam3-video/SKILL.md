# SAM 3 Video Segmentation (vailá)

Use when the user works on **SAM 3 video segmentation**, text-prompt masks, `vaila_sam.py` (non-FIFA mode), or troubleshoots SAM 3 checkpoint/VRAM/BFloat16 issues.

---

## Overview

`vaila/vaila_sam.py` performs video segmentation using Meta's **SAM 3** (Segment Anything Model 3). It supports:

- Text-prompt masks (e.g., "person", "ball", "car") — open-vocabulary, any free-text prompt
- Single video or batch directory processing
- GUI mode (Tkinter dialog) and headless CLI
- Automatic VRAM management sized to GPU memory
- Overlay MP4 + per-frame mask PNG outputs

### GUI Help button (v0.0.3+)

`SamVideoDialog` and `SamBatchProgress` both expose a **Help** button that
opens `vaila/help/vaila_sam.html` in the user's default browser (or falls back
to the GitHub raw version). Point users to this when they ask "where is the
documentation?" — no need to quote the `.md` in chat.

### Prompt presets

The text-prompt input in `SamVideoDialog` is an editable `ttk.Combobox`
populated from the module-level tuple:

```python
SAM3_PROMPT_PRESETS = (
    "person", "player", "goalkeeper", "referee",
    "ball", "soccer ball", "basketball", "volleyball",
    "coach", "crowd", "car", "bike", "dog", "cat",
)
```

Because SAM 3 is open-vocabulary, any free-text prompt works; the combobox is
only a convenience. CLI: `-t "goalkeeper"`.

### Platform (read before promising CPU / macOS)

| Environment | SAM 3 *video* in vailá |
|---------------|------------------------|
| Linux / Windows + NVIDIA CUDA | Supported |
| CPU-only Windows / no CUDA | **Not supported** — CLI/GUI exit early with an explanation |
| macOS (MPS / integrated GPU) | **Not supported** for this predictor (CUDA-only stack) |

`--frame-by-frame` reduces **VRAM on CUDA**; it is **not** a CPU fallback. Without CUDA, point users to Markerless 2D / YOLO or a remote CUDA box.

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
6. `models/sam3/sam3.pt` (repo-root models layout)
7. `models/sam3/sam3.1_multiplex.pt` (repo-root models layout)
8. `models/sam3/sam3_weights/sam3.pt` (repo-root nested HF layout)
9. Legacy: `<repo>/sam3_weights/sam3.pt`
10. Auto-download from Hugging Face Hub

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

Text prompt is open-vocabulary (no fixed class list). `person` is only the default.

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
| `--no-isolate-batch` | — | — | Disable subprocess-per-video isolation in batch mode (legacy in-process loop). Default isolation is ON when batch has >1 video; do not disable unless debugging. |
| `--video-output-dir` | — | — | **Internal** (hidden from `--help`): used by parent batch loop to instruct a child process to write outputs directly to this dir instead of creating its own `processed_sam_TS/` wrapper. Do not set manually. |

---

## VRAM Management

SAM 3 loads **all session frames** onto the GPU. Long clips exceed VRAM easily.

| Strategy | How |
|----------|-----|
| Auto (default) | Frame cap computed from **currently free** GPU VRAM (logged as `[SAM3 VRAM]`) |
| Environment | `SAM3_MAX_FRAMES=80` |
| CLI | `--max-frames 80` |
| Full clip | `--max-frames 0` (large VRAM only) |
| Per-frame fallback | `--frame-by-frame` (loses temporal tracking) |

vailá auto-sets `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` before the first `import torch` to reduce fragmentation OOMs.

### Batch loop guarantees

- **Subprocess-per-video isolation (default for batch >1 video)**: each video is spawned in a fresh `python vaila_sam.py --input <single.mp4> --video-output-dir <out>` subprocess. When the subprocess exits (success or OOM crash), the OS releases 100% of GPU memory, so the next video always starts with `alloc=0`. Disable with `--no-isolate-batch` to fall back to the legacy in-process loop.
- Between videos in legacy in-process mode: `del predictor` + `gc.collect()` + `torch.cuda.empty_cache()` (helper `_release_sam3_gpu_memory`).
- On CUDA OOM (inside each subprocess): **auto-retry** with `max_frames=64`, then `max_frames=32` (helper `_process_one_video_with_oom_retry`) before giving up. Top-of-loop release of `Exception e` happens at the start of each retry iteration so its traceback frames stop pinning ~7 GiB of inner SAM3 tensors.
- On final failure: write `FAILED_sam.txt` inside the per-video output folder (no more silent empty dirs); subprocess exits with code 3 and the parent batch loop logs `subprocess exit=3` and continues with the next video.
- GUI: progress window (`SamBatchProgress`) running the batch on a background thread; supports Cancel.

### Why subprocess-per-video (debug session 42b4a5, Apr 2026)

Multi-video batches kept failing with cascading `CUDA OOM while loading the video into SAM3` after the first OOM, even with the 256→64→32 retry ladder. Runtime instrumentation (`_dbg_log` → `.cursor/debug-42b4a5.log`) tested and rejected:

| Hyp | Status | Evidence |
|-----|--------|----------|
| H1 weights accumulate across videos | REJECTED | run1 (no OOM) showed `alloc=0.009` between videos |
| H2 `outputs_by_frame` retains GPU tensors | REJECTED | same as H1 |
| H3 `predictor = None` ineffective | REJECTED | run1 cleanup proved it works without OOM |
| H5 allocator fragmentation | REJECTED | post-cleanup `alloc ≈ reserved` (no fragmentation) |
| **H4 OOM `Exception e` retains traceback locals** | **CONFIRMED** | `_release_sam3_gpu_memory()` inside `except` was a no-op (alloc unchanged); ~7-8 GiB only freed AFTER `e` died (top of next iteration). Fix: call release at top of each retry loop iteration, not inside `except`. |
| **H7 SAM3 C++-level state leak on failed start_session** | **CONFIRMED (irrecoverable in-process)** | After `e` death + gc.collect + empty_cache, ~13 GiB of orphan tensors still reported as `allocated` by PyTorch but unreachable by Python gc. These persist across videos and `_release_sam3_gpu_memory()` calls. Only killing the Python process releases them. |

**Fix**: subprocess-per-video isolation in the CLI batch loop (`vaila_sam.py`, around `main()`). Each video runs in its own Python process via `subprocess.call([sys.executable, vaila_sam.py, "--input", vid, "--video-output-dir", out, ...])`. The new internal flag `--video-output-dir` makes the child write directly into the supplied dir (no `processed_sam_TS` wrapper). Cost: ~5-7 s startup per video for predictor build — acceptable vs. cascading failures.

If you need to revisit this with full context, see the agent transcript: [SAM3 OOM cascade debug](42b4a5fd-2701-458b-b7ec-e554e2265426).

---

## Output Format

```
output/processed_sam_YYYYMMDD_HHMMSS/
  video_name/
    masks/              PNG per frame (unless --no-png)
    <video_stem>_sam_overlay.mp4    coloured mask overlay (unless --no-overlay)
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

#### Cascading OOM in batch (multiple videos failing in a row)

If a batch run shows the first OOM-failing video poisoning every subsequent video with the same error, you are hitting the SAM3 C++-level GPU leak documented in *Why subprocess-per-video* above. Confirm subprocess isolation is active:

- Look for `[batch] subprocess-per-video isolation: ENABLED` in the CLI stdout.
- Each video log line should say `(isolated)` next to the video name.
- If isolation is disabled (you passed `--no-isolate-batch` or there is only 1 video in the batch), the legacy in-process loop is used and a CUDA OOM in `predictor.handle_request("start_session")` will leak ~13 GiB of orphan tensors that no in-process `gc.collect()` / `empty_cache()` can free. The next video then OOMs immediately.

**Recovery options**:
1. Re-run the batch without `--no-isolate-batch` so each video is isolated.
2. Lower `--max-frames` (e.g. `32`) so no video OOMs in the first place.
3. If running through the GUI, the GUI batch path already isolates per video by spawning the CLI; the CLI's own batch loop adds a second layer of subprocess isolation when given a directory.

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
| `vaila/models/sam3/` and `models/sam3/` | Checkpoint directories (sam3.pt, sam3.1_multiplex.pt) |
| `vaila/models/sam3/merges.txt` | BPE fallback vocabulary |
| `vaila/help/vaila_sam.html` | Browser help page |
| `vaila/help/vaila_sam.md` | Markdown help source |
| `tests/test_vaila_sam.py` | Unit + GPU smoke tests |
| `tests/SAM/README.md` | Test setup instructions |
