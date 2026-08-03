# Markerless Live — via Markerless 2D Chooser

## Overview

> **v0.3.98:** this used to be its own button (`B4_r4_c5`, text
> "Markerless Live"). It is now under the **"Other 2D tools"** section of
> the **Markerless 2D** coringa chooser (`B1_r1_c4`, method
> `markerless_2d_analysis`) — the underlying handler and script
> (`vaila/markerless_live.py`) are unchanged.

**Method Name:** `markerless_live`
**Button Text (in chooser):** Markerless Live

## Description

Real-time pose estimation and joint-angle calculation from a live webcam feed, using either a YOLO or MediaPipe pose engine (selectable at launch).

## Usage

1. Click **Markerless 2D** in the vailá GUI, then **Markerless Live** in the "Other 2D tools" section of the chooser
2. Select a camera and pose engine (YOLO or MediaPipe) in the dialogs that follow
3. The live overlay window shows the skeleton, bounding box, and joint angles; angle data and plots are saved on exit

## Related Scripts

- `vaila/markerless_live.py` — script documentation: `vaila/help/markerless_live.md`

---

**Last Updated:** 02 August 2026
**Part of vailá - Multimodal Toolbox**
**License:** AGPLv3.0
