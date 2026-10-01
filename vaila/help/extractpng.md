# extractpng

## Module Information

- **Category:** Tools
- **File:** `vaila/extractpng.py`
- **Version:** 0.4.6
- **Updated:** 30 September 2026
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** Yes (one window, Frame C → Video and Image → C_B_r1_c1)

## Description

*vailá* Video ↔ PNG tool: high-performance video-to-PNG extraction and PNG-to-video creation accelerated by NVIDIA GPU hardware (NVDEC decoding & NVENC encoding via FFmpeg CUDA) with automatic CPU software fallback.

## Key Optimizations & Features

1. **NVIDIA GPU Decoding (`--hwaccel auto|cuda|cpu`):**
   Automatically detects physical NVIDIA GPUs (e.g. RTX 4090) and configures FFmpeg `-hwaccel cuda` to decode H.264, HEVC, AV1, VP9 streams via NVDEC on the GPU. Falls back gracefully to CPU software decoding if hardware decoding is unsupported.
2. **Fast Lossless PNG Extraction (`-c` / `--compression 0-9`):**
   Defaults to compression level 1 with bypass of Paeth filtering (`-pred none`). Delivers bit-for-bit identical, lossless PNG frames ~3x faster than standard level 6 with minimal size difference (~4%).
3. **Redundant Rescaler Bypass:**
   Skips unnecessary CPU Lanczos/bicubic resampling when the target resolution matches the native video stream, saving massive CPU cycles.
4. **Parallel Multi-Video Batching (`-j` / `--workers`):**
   Extracts folders of multiple videos concurrently using `ThreadPoolExecutor`, fully saturating GPU NVDEC and multi-core CPU pipelines.
5. **NVIDIA NVENC Video Creation:**
   Builds MP4 videos from PNG folders at high quality using `h264_nvenc` or `hevc_nvenc`.
   Every PNG uses a temporary numbered image2 sequence, in natural filename order: `frame_2.png` before `frame_10.png`.
6. **Stratified Random Frame Sampling (`sample` subcommand):**
   Extracts $N$ well-spaced, non-contiguous random frames per video (default 20, configurable with `-n`)
   using stratified jittered sampling to guarantee uniform temporal coverage across the whole timeline
   without clumping. Supports recursive directory scanning (`--recursive`), parallel workers, and per-video manifest export.

## GUI

Frame C → **Video and Image** → **C_B_r1_c1 - Video<-->PNG**, or:

```bash
uv run vaila/extractpng.py
```

One window:

1. View detected GPU acceleration badge (e.g. `🚀 NVIDIA CUDA: NVIDIA GeForce RTX 4090`).
2. Choose mode: **Video → PNG** / **PNG → Video** / **Select frames** / **Random sample**.
3. Browse input directory/file (and optional output directory).
4. Set options (Acceleration mode, PNG Speed/Compression, Workers, Codec, Num frames, Recursive).
5. Click **Run**. Prints copy-paste CLI mirror (`>> vaila/extractpng`).

## CLI

```bash
# High-speed GPU extraction (auto-detects NVIDIA CUDA, level 1 compression)
uv run vaila/extractpng.py extract -i /path/to/videos

# Explicit GPU acceleration with parallel workers
uv run vaila/extractpng.py extract -i /path/to/videos -o /path/to/out --hwaccel cuda --compression 1 --workers 4

# Create video using GPU NVENC hardware encoder
uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264_nvenc --hwaccel cuda

# Grab select frames with GPU acceleration
uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5,7 --hwaccel cuda

# Extract 20 well-spaced random frames per video recursively across subdirectories
uv run vaila/extractpng.py sample -i /path/to/videos -n 20 --recursive

# Flat folder output with custom seed
uv run vaila/extractpng.py sample -i /path/to/videos -o /path/to/out -n 20 --flat --seed 42
```

## Notes

- FFmpeg `-hwaccel cuda` is an input option placed before `-i`.
- Default PNG pattern: `%09d.png` (used when extracting; creation always includes every PNG in natural order).
- **PNG → Video** accepts any directory of `.png` files, or immediate subfolders that contain them. Every PNG in the folder is sorted by filename and joined into one MP4.
- Batch extract writes `vaila_extractpng_<timestamp>/<stem>_png/` plus `video_info.txt` containing extraction metrics and hardware profile per video.
- Random sampling writes `vaila_random_samples_<timestamp>/<stem>_random<N>/frame_<number>.png` and `manifest.csv` per video, plus `sampling_summary.csv` in the output root. Use `--flat` to output all frames into a single flat directory.

---

**Part of** *vailá* — Multimodal Toolbox


## PNG → MP4 quality and independent frames

PNG → MP4 defaults: quality 18 (1–51, lower is better), GOP 1 (every frame independent), no B-frames. CLI: --quality 18 --gop 1. NVENC uses p7/hq VBR CQ; CPU uses slow/CRF. H.264 and H.265 preserve these settings on CPU fallback. Every PNG produces exactly one frame in natural filename order, including gaps and names with spaces. Original dimensions and chosen FPS are preserved; mixed dimensions and odd dimensions incompatible with yuv420p are rejected. MP4/yuv420p remains lossy and cannot restore detail already blurred in the PNG. GOP 1 generally produces larger files.
