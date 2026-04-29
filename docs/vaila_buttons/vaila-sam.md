# SAM (Segment Anything) Button

The **SAM (Segment Anything)** button launches the `vaila_sam.py` module for advanced video segmentation using Meta's SAM 3 model.

## Overview

SAM 3 allows for high-quality, zero-shot segmentation of objects in video based on text prompts or point clicks. In vailá, it is primarily used for segmenting players and referees in soccer footage to create binary masks and extract tracking points.

## Key Features

- **Text Prompting:** Segment objects by name (e.g., "player", "ball", "referee").
- **CUDA Optimized:** Specifically designed for NVIDIA GPUs to handle video tracking.
- **VRAM Management:** Includes an "OOM Retry Ladder" and spatial downscaling to handle high-resolution 4K footage on consumer GPUs.
- **Batch Processing:** Processes multiple videos in isolated subprocesses to prevent memory leaks.
- **FIFA Integration:** Includes a dedicated `fifa` subcommand for the FIFA Skeletal Tracking Light pipeline.

## Usage

1. Click **YOLO and SAM** -> **SAM (Segment Anything)** in Frame B.
2. Select the input video or folder.
3. Enter a **Text prompt** (e.g., `person`).
4. (Optional) Select **Save overlay MP4** to visualize the segmentation.
5. Click **Run**.

## Requirements
- **NVIDIA GPU:** CUDA is strictly required for the SAM 3 video stack.
- **Weights:** Must be downloaded from Hugging Face (`uv run vaila/vaila_sam.py --download-weights`).
- **Dependencies:** `uv sync --extra sam`.

---
See also: [FIFA Workflow](../../docs/fifa_workflow.md), [SAM 3 Help Index](../../vaila/help/vaila_sam.html)
