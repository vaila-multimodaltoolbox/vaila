# Stroboscopic & Stromotion Generator

**Script:** `vaila/vaila_stroboscopic.py`  
**Version:** `0.3.105`  
**Updated:** `15 August 2026`

## Overview

This tool generates **stroboscopic (chronophotography) images and videos** from biomechanical and sports video recordings. It supports four distinct modes:

1. **`stromotion` (Default / AI Mode):** Dartfish-style chronophotography using **MediaPipe Selfie Segmentation** (`.tflite`). Automatically extracts the moving athlete and seamlessly composites them onto a clean estimated background (median or first frame). Produces both high-resolution composite PNG and MP4 video.
2. **`pose` (Skeleton Mode):** Overlays 2D joint skeleton landmarks across sampled frames from a CSV coordinate file.
3. **`motion` (Motion Difference Mode):** Extracts moving pixels via frame differencing without neural networks.
4. **`stack` (Multishot Blend Mode):** Blends sampled frames using `max` (peak exposure) or `add` (exposure accumulation).

---

## Usage

### 1. GUI Method (Recommended)

1. Launch `vaila.py`.
2. In **Frame C (Tools & Visualization)**, click **Stroboscopic**.
3. In the interactive settings dialog:
   - Select your input **Video**.
   - Choose the **Mode** (`stromotion`, `pose`, `motion`, or `stack`).
   - Adjust **Frame Interval**, **Background Mode**, **Outline**, or **Segmentation Threshold**.
   - Click **▶ Run**.
4. The GUI prints a copy-pasteable `>> Equivalent CLI` mirror command to the terminal for easy automation.

---

### 2. Command Line Interface (CLI)

Run directly from the terminal for batch processing or scripts:

#### A. AI Stromotion (Default)
```bash
uv run python -u vaila/vaila_stroboscopic.py \
    -v /path/to/video.mp4 \
    -o /path/to/output_dir \
    --mode stromotion \
    -i 10 \
    --bg-mode median \
    --bg-samples 10 \
    --seg-threshold 0.5 \
    --feather-px 5 \
    --outline
```

#### B. Pose Skeleton Mode (with CSV)
```bash
uv run python -u vaila/vaila_stroboscopic.py \
    -v /path/to/video.mp4 \
    -c /path/to/coordinates.csv \
    --mode pose \
    -i 10
```

#### C. Motion Difference Mode
```bash
uv run python -u vaila/vaila_stroboscopic.py \
    -v /path/to/video.mp4 \
    --mode motion \
    -i 5
```

#### D. Multishot Stack Mode
```bash
uv run python -u vaila/vaila_stroboscopic.py \
    -v /path/to/video.mp4 \
    --mode stack \
    -i 5 \
    --stack-op max
```

---

## CLI Options & Parameters

| Option | Description | Default |
| :--- | :--- | :--- |
| `-v, --video` | Path to input video file | GUI picker |
| `-o, --output` | Path to output directory or image file | Video dir |
| `-i, --interval` | Frame interval for sampling | `10` |
| `--mode` | Effect mode: `stromotion`, `pose`, `motion`, `stack` | `stromotion` |
| `-c, --csv` | Path to 2D landmarks CSV (for `pose` mode) | Auto-detect |
| `--bg-mode` | Background mode: `median` or `first` | `median` |
| `--bg-samples` | Number of frame samples to compute median background | `10` |
| `--seg-threshold`| Confidence threshold for person segmentation (0.0–1.0) | `0.5` |
| `--feather-px` | Edge blur / feathering radius in pixels | `5` |
| `--outline` | Draw colored outline around the segmented subjects | `False` |
| `--outline-color`| Outline RGB color (e.g. `255,255,255` or `0,255,0`) | `255,255,255` |
| `--outline-thickness`| Outline thickness in pixels | `2` |
| `--no-video` | Disable generation of output `.mp4` video | `False` |
| `--no-frames` | Disable saving individual PNG cutouts | `False` |
| `--stack-op` | Stack operator for `stack` mode: `max` or `add` | `max` |
| `--model-selection`| 1 = Landscape (144x256), 0 = Square (256x256) | `1` |
| `--open-help` | Open this documentation in default web browser | `False` |
