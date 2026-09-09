---
name: extractpng-gpu-accel-loop
category: Vailá
trigger: manual
verification-level: 1
theory-base: arXiv:2607.00038
---

# vailá Video ↔ PNG Hardware Acceleration (NVIDIA GPU / NVDEC & NVENC) Loop

## Description
Accelerate video frame extraction (`extractpng.py`) using NVIDIA GPU NVDEC hardware decoding (`-hwaccel cuda`) via FFmpeg, bypass redundant software resampling, optimize PNG compression (fast level 1 with `-pred none` for lossless pixel parity), enable multi-threaded batch video extraction via `ThreadPoolExecutor`, and accelerate video assembly via NVIDIA NVENC hardware encoders (`h264_nvenc`, `hevc_nvenc`).

## Use When
- Extracting PNG frames from high-resolution, high-framerate biomechanics videos (e.g. 1080p @ 120/240 fps, 4K).
- Converting PNG image sequences into MP4 video with high throughput.
- Running on systems with NVIDIA GeForce / RTX / Tesla GPUs (RTX 4090, etc.) with CUDA-enabled FFmpeg.
- Extracting single or batch videos in both interactive Tkinter GUI (Frame C → Video and Image → C_B_r1_c1) and CLI headless mode (`uv run vaila/extractpng.py`).
- Requiring bit-for-bit lossless pixel representation with ~3.4x speedup over standard FFmpeg defaults.

## Inputs
1. `extractpng` — `vaila/extractpng.py` (CLI, Tkinter GUI, core extraction routines).
2. `tests` — `tests/test_extractpng.py`.
3. `help` — `vaila/help/extractpng.md` and `vaila/help/extractpng.html`.
4. `index` — `vaila/help/index.md` and `vaila/help/index.html`.

## Goal
An objectively verifiable high-performance extraction pipeline:
1. **NVIDIA GPU Detection:** Auto-detect GPU hardware name and FFmpeg CUDA capabilities via `get_cuda_status()`.
2. **GPU Video Decoding:** Use `-hwaccel cuda` in FFmpeg input options with automatic CPU fallback.
3. **Lossless Compression Optimization:** Set PNG `-compression_level 1` and `-pred none` as default, reducing per-video extraction from 10.22s to 3.88s (2.6x speedup per video).
4. **Redundant Rescale Bypass:** Omit `-vf scale=W:H` when target dimensions match native video stream.
5. **Parallel Multi-Video Batching:** Run video extractions concurrently across multi-core CPU and GPU NVDEC via `ThreadPoolExecutor`, dropping 3-video batch time from 33.16s to 9.80s (3.4x faster, 188.6 fps aggregate).
6. **NVENC Video Creation:** Encode MP4 videos via `h264_nvenc` and `hevc_nvenc` with automatic libx264/libx265 fallback.
7. **GUI Modernization:** Display hardware badge (`🚀 NVIDIA CUDA: NVIDIA GeForce RTX 4090`), dropdowns for hwaccel, compression speed, parallel workers, and copy-paste CLI mirror.
8. **Automated Testing & Static Typing:** All unit tests pass, ruff clean, ty clean.
