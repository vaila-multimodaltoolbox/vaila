# Soccer-Field Calib Button

The **Soccer-Field Calib** button launches the `soccerfield_calib.py` module, which fits a DLT2D homography to a single video frame using detected soccer pitch keypoints.

## Overview

This tool calculates the transformation between 2D pixel coordinates and 2D field coordinates (metres) based on the FIFA canonical pitch model (105x68m). It is essential for projecting player positions from broadcast video onto a 2D field map.

## Key Features

- **FIFA Reference:** Uses the 29/32-keypoint FIFA pitch model.
- **Homography Fitting:** Calculates 8 DLT coefficients from at least 4 (ideally 6+) point correspondences.
- **Interactive Refinement:** Works in tandem with **Get Pixel Coord** to refine point locations.
- **Output:** Generates a `.dlt2d` file containing the calibration parameters for a specific frame.

## Usage

1. **Prerequisite:** Obtain pixel coordinates for pitch keypoints using **Field KPs (AI)** or manual clicking in **Get Pixel Coord**.
2. Click **Soccer-Field Calib** in Frame B.
3. Select the video and the CSV file containing the pixel coordinates.
4. Specify the frame index to calibrate.
5. The tool will calculate the homography and save a `.dlt2d` file in the output directory.

## Broadcast vs. Static
- For **static cameras**, a single DLT row is sufficient.
- For **moving/broadcast cameras**, use the **FIFA cams→DLT** workflow to generate per-frame calibration.

---
See also: [FIFA Workflow](../../docs/fifa_workflow.md), [DLT Reconstruction](../../docs/dlt2d.md)
