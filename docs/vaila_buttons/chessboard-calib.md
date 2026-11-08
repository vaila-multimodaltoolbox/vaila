# Chessboard Calib Button

The **Chessboard Calib** button (Frame C-A, below **DLT/REC 2D-3D**) calibrates several cameras at once from chessboard videos and writes the files that *vailá*'s 3D reconstruction uses.

## Overview

Record one video per camera of the same session while a person shows a chessboard to the cameras. The module (`vaila/chessboard_calib.py`) finds the board corners, estimates each camera's lens (intrinsics), the frame offsets between cameras and the camera positions (extrinsics), and exports a `.dlt3d` file per camera for **Rec3D 1DLT / MultiDLT**.

## Key Features

- **Small boards:** the detector retries on an upscaled crop around the last detection (`--zoom`).
- **Same corner numbering in every camera**, also for boards that look the same after a 180-degree turn.
- **Lens model chosen from the data:** with `--intrinsics auto` the model is restricted when the board covers little of the image; the coverage image shows where the calibration is measured.
- **Static-board warning:** when the board kept almost one orientation (tilt below 20°), the focal length cannot be told apart from the distance; the log and README say so. Load the lens of a calibration where the board moved with `--intrinsics-toml-dir <processed_chessboard_calib_* folder>`; cut and equalized copies find their camera by name.
- **Frame offsets** between cameras from the board's rotation speed (`--sync auto`), measured to a fraction of a frame and reported before rounding. Manual offsets may have decimals. Fractions are used by `undistort --subframe` or by a ready RIFE command (Video Event Equalizer) that aligns the videos themselves.
- **Frame numbers from the video timestamps**, as in getpixelvideo: frames a camera dropped are listed and do not shift the later frames.
- **World frame:** the board itself (`--world board`) or measured floor/control points (`--world control`).
- **Checks:** reprojection error and the reconstructed square size, in the calibration frames and in time blocks left out of the extrinsics fit.

## Usage

1. Click **Chessboard Calib** in Frame C-A.
2. Choose the folder with one chessboard video per camera.
3. Set the inner corners (for a board of 7 x 9 squares: `6x8`) and the square side in metres (default 0.098, the lab board; measure yours over several squares).
4. Click **Run calibration**. The terminal prints the equivalent command and the progress.

## Afterwards: 3D reconstruction

1. In the same window click **Undistort pixels for Rec3D...**, choose the calibration folder and the folder with the pixel CSVs (each name starts with or contains its camera video name), and answer whether to apply the frame offsets and whether to use the sub-frame offsets. CLI: `uv run vaila/chessboard_calib.py undistort --calib <run> -i <pixels> --apply-offsets` (or `--subframe`).
2. Run **Rec3D 1DLT** with the `<camera>.dlt3d` files and the undistorted CSVs; the terminal prints the exact command.

Do not use **Distort CSV** for step 1: it reprojects with a different camera matrix than the one in the `.dlt3d` files.

Record the board tilted 30–45° in several directions. A board held still gives a wrong focal length and unreliable sub-frame offsets. The offsets belong to the calibration recording; apply them only to pixel CSVs of the same recordings. If the report lists dropped frames, run Markerless 2D or Video to PNG on constant-frame-rate copies (`ffmpeg -i cam.mp4 -fps_mode cfr -r 30 -c:v libx264 -crf 12 cam_cfr.mp4`), because those tools count frames while reading.

---
See also: [Help page](../../vaila/help/chessboard_calib.md), [DLT3D and 3D reconstruction](dlt3d-reconstruction.md)
