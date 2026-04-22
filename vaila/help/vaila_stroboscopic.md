# Stroboscopic Effect Generator

**Script:** `vaila/vaila_stroboscopic.py`

## Overview

This tool generates a **stroboscopic (chronophotography) image** from a video. It supports:

- **Pose overlay** from a CSV of landmarks (legacy vailá mode).
- **Video-only motion strobe** (videoStrobe-style): extract moving pixels and composite them into one image.
- **Video-only multishot stack** (multishot-style): stack sampled frames using `max` or `add` accumulation.

## Key Features

- **Temporal Gradient:** Skeletons are colored from **Blue (Start)** to **Red (End)**, clearly showing the direction of movement.
- **Enhanced Visualization:** Uses the vaila-standard "Enhanced Skeleton" style, which includes:
  - Explicit segments (Arm, Forearm, Thigh, Leg, etc.)
  - Computed midpoints (Neck, Mid-Hip, Mid-Shoulder)
  - Color-coded sides (Left=Sky Blue, Right=Coral)
- **Automatic Data Detection:** Can automatically find the corresponding CSV file if naming conventions are followed.
- **Flexible Input:** Works via GUI (file picker) or Command Line Interface (CLI).
- **Video-only modes:** `--mode motion` and `--mode stack` do **not** require a CSV.

## Usage

### 1. GUI Method (Recommended)

1. Launch `vaila.py`.
2. Navigate to the **Visualization** frame (Frame C).
3. Click the **"Stroboscopic"** button.
4. Select your video file in the dialog window.
   - The script will look for a CSV with the same name (e.g., `video.mp4` -> `video.csv` or `video_vaila_analyzed.csv`).
5. The output image (`_stroboscopic.png`) will be saved in the same directory.

### 2. Command Line Interface (CLI)

You can run the script directly from the terminal for batch processing or advanced control.

```bash
uv run vaila/vaila_stroboscopic.py -v /path/to/video.mp4 -i 10
```

**Arguments:**

- `-v`, `--video`: Path to the input video file (Required if not using GUI).
- `-c`, `--csv`: Path to the pixel coordinates CSV file (Optional, auto-detected if omitted).
- `-o`, `--output`: Path for the output PNG image (Optional).
- `-i`, `--interval`: Frame interval for the strobe effect (Default: 10). A higher number means fewer skeletons drawn.
- `--mode`: `pose` (default), `motion`, or `stack`.

### Modes (CLI)

#### `--mode pose` (default)

Overlay multiple skeleton instances from CSV landmarks.

```bash
uv run vaila/vaila_stroboscopic.py --mode pose -v /path/to/video.mp4 -c /path/to/coords.csv -i 10
```

#### `--mode motion` (video-only)

videoStrobe-style motion extraction: computes a motion mask by frame differencing, then accumulates only the moving pixels.

```bash
uv run vaila/vaila_stroboscopic.py --mode motion -v /path/to/video.mp4 \
  --threshold 50 --blend-ratio 1.0 --blur-size 5 --open-kernel-size 5 \
  --frame-interval 1
```

Optional stable background (median of sampled frames, `strobe2.py` style):

```bash
uv run vaila/vaila_stroboscopic.py --mode motion -v /path/to/video.mp4 \
  --stable-background --background-samples 10
```

#### `--mode stack` (video-only)

multishot-style stacking from sampled frames.

```bash
# Max exposure (default)
uv run vaila/vaila_stroboscopic.py --mode stack -v /path/to/video.mp4 --frame-interval 5 --stack-op max

# Add + normalize
uv run vaila/vaila_stroboscopic.py --mode stack -v /path/to/video.mp4 --frame-interval 5 --stack-op add
```

## Codec note (H.265 / HEVC)

Some Linux OpenCV builds cannot decode HEVC/H.265 videos unless FFmpeg/GStreamer support is available.
If you see **\"Error opening video\"** or frame count is zero, convert the input to H.264 (AVC) before running.

## Output

The tool produces a high-resolution PNG image overlaying the skeletons onto the first frame of the video (background darkened for contrast).

**Example Output:**
![Stroboscopic Example](images/stroboscopic_example.png)
_(Note: Example image placeholder)_
