# Stroboscopic Effect Generator

**Script:** `vaila/vaila_stroboscopic.py`

## Overview

This tool generates a **stroboscopic (chronophotography) image** from a video file and associated pose data (CSV). It visualizes human motion by overlaying multiple skeleton instances on a single background image, creating a "motion trail" effect.

## Key Features

- **Temporal Gradient:** Skeletons are colored from **Blue (Start)** to **Red (End)**, clearly showing the direction of movement.
- **Enhanced Visualization:** Uses the vaila-standard "Enhanced Skeleton" style, which includes:
  - Explicit segments (Arm, Forearm, Thigh, Leg, etc.)
  - Computed midpoints (Neck, Mid-Hip, Mid-Shoulder)
  - Color-coded sides (Left=Sky Blue, Right=Coral)
- **Automatic Data Detection:** Can automatically find the corresponding CSV file if naming conventions are followed.
- **Flexible Input:** Works via GUI (file picker) or Command Line Interface (CLI).

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

## Output

The tool produces a high-resolution PNG image overlaying the skeletons onto the first frame of the video (background darkened for contrast).

**Example Output:**
![Stroboscopic Example](images/stroboscopic_example.png)
_(Note: Example image placeholder)_
