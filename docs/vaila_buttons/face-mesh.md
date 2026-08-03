# Face Mesh — via Markerless 2D Chooser

> **v0.3.98:** this used to be its own button (`B5_r6_c2`, text "Face Mesh").
> It is now under the **"Other 2D tools"** section of the **Markerless 2D**
> coringa chooser (`B1_r1_c4`, method `markerless_2d_analysis`) — the
> underlying handler and script (`vaila/mp_facemesh.py`) are unchanged.

The **Face Mesh** option launches the `mp_facemesh.py` module, which uses MediaPipe to track 468 3D facial landmarks in real-time or from video files.

## Overview

Face Mesh provides high-fidelity facial tracking for research into facial expressions, speech kinematics, and non-verbal communication.

## Key Features

- **High Density:** Tracks 468 landmarks in 3D.
- **Real-time & Batch:** Supports live webcam input and batch processing of video directories.
- **Metric Space:** Can estimate 3D coordinates in millimetres.
- **Output:** Saves landmark coordinates to CSV for further analysis.

## Usage

1. Click **Markerless 2D** in Frame B, then **Face Mesh** in the "Other 2D tools" section of the chooser.
2. Select the input source (video file or webcam).
3. Run the tracking to visualize and save facial landmark data.

---
See also: [Face Mesh Help Index](../../vaila/help/mp_facemesh_help.html)
