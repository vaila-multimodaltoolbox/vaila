"""
===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 06 February 2026
Version: 0.1.4
Python Version: 3.12.12

Description:
------------
This script corrects lens distortion in videos using a camera calibration file
generated usually from 'camera_calibration.py'. It undistorts every frame of the
input video(s) and reconstructs the video with high quality, maintaining the
original metadata (FPS, duration, resolution) as precisely as possible.

Key Features:
-------------
1. Precise metadata extraction using `ffprobe`.
2. Hardware-accelerated encoding (optional/auto-detect).
3. Batch processing of multiple videos.
4. CLI support for pipeline integration.

Usage:
------
GUI Mode (Default):
    python vaila_lensdistortvideo.py

CLI Mode:
    python vaila_lensdistortvideo.py --input_dir /path/to/videos --params_file /path/to/params.toml [--output_dir /path/to/output]

    Arguments:
      --input_dir   Directory containing the videos to be processed.
      --params_file Path to the camera calibration parameters TOML file.
      --output_dir  (Optional) Directory to save the corrected videos.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================


Camera Calibration Parameters and Their Meanings
=================================================

This script processes videos by applying lens distortion correction based on
intrinsic camera parameters and distortion coefficients. It also demonstrates
how to calculate these parameters using field of view (FOV) and resolution.

Intrinsic Camera Parameters:
-----------------------------
1. fx, fy (Focal Length):
   - Represent the focal length of the lens in pixels along the x-axis (fx) and y-axis (fy).
   - Larger values indicate a narrower field of view.
   - Calculated using the formula:
     fx = (width / 2) / tan(horizontal FOV / 2)
     fy = (height / 2) / tan(vertical FOV / 2)
   - Example for a 2.8 mm lens with 105 deg horizontal FOV, 58 deg vertical FOV, and 1920x1080 resolution:
     fx = (1920 / 2) / tan(105 deg / 2) = 949.41 pixels
     fy = (1080 / 2) / tan(58 deg / 2) = 950.63 pixels

2. cx, cy (Optical Center):
   - Represent the x and y coordinates of the camera's optical center in pixels.
   - Typically close to the image's center:
     cx = width / 2 = 1920 / 2 = 960 pixels
     cy = height / 2 = 1080 / 2 = 540 pixels

Distortion Coefficients:
-------------------------
1. k1, k2, k3 (Radial Distortion):
   - Radial distortion causes straight lines to appear curved, especially near the edges of the image.
     - Negative values indicate "barrel distortion" (image bulges outward).
     - Positive values indicate "pincushion distortion" (image compresses inward).
   - k3 is usually smaller and accounts for higher-order distortion effects.

2. p1, p2 (Tangential Distortion):
   - Tangential distortion occurs when the lens and image sensor are not perfectly parallel.
   - These coefficients adjust for minor misalignments.

Example Calibration Output:
----------------------------
Camera Matrix (Intrinsic Parameters):
    [[fx   0  cx]
     [ 0  fy  cy]
     [ 0   0   1]]

Distortion Coefficients:
    [k1, k2, p1, p2, k3]

Example TOML Format:
--------------------
Below is an example of a TOML file containing calibration parameters for use with this script:

    fx = 949.41
    fy = 950.63
    cx = 960.0
    cy = 540.0
    k1 = -0.28871370110181493
    k2 = 0.1374614711665278
    k3 = -0.025511562284832402
    p1 = 0.00044281215436799446
    p2 = -0.00042111749309847274

Applications:
-------------
1. Lens Distortion Correction:
   - Use the camera matrix and distortion coefficients to undistort images and videos.
   - Example:
     * Remove "barrel distortion" in wide-angle lenses.
     * Improve the accuracy of feature detection and matching in computer vision tasks.

2. Image Simulation:
   - Simulate distorted images for augmented reality (AR) or testing vision algorithms.

3. Augmented Reality:
   - Use the corrected intrinsic parameters to align virtual objects with the real world.

Usage in OpenCV:
----------------
The OpenCV library provides functions for:
- Calibrating cameras (`cv2.calibrateCamera`).
- Correcting distortion in images (`cv2.undistort`).
- Optimizing the camera matrix for specific resolutions (`cv2.getOptimalNewCameraMatrix`).

Example Workflow:
-----------------
1. Calibrate the camera using multiple images of a chessboard pattern.
2. Obtain the intrinsic parameters (fx, fy, cx, cy) and distortion coefficients (k1, k2, k3, p1, p2).
3. Use the parameters to undistort captured images or videos.

References:
-----------
- OpenCV Camera Calibration Documentation:
  https://docs.opencv.org/4.x/dc/dbb/tutorial_py_calibration.html
- Radial and Tangential Distortion:
  https://docs.opencv.org/4.x/d9/d0c/group__calib3d.html
===============================================================================
"""

import argparse
import json
import os
import subprocess
import time
import tkinter as tk
from datetime import datetime
from pathlib import Path
from tkinter import filedialog

try:
    import tomllib
except ImportError:
    import tomli as tomllib  # pyright: ignore[reportMissingImports]

import cv2
import numpy as np
from rich import print as rprint
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TextColumn,
    TimeElapsedColumn,
)


def load_distortion_parameters(toml_path):
    """
    Load distortion parameters from a TOML file (fx, fy, cx, cy, k1, k2, k3, p1, p2).
    """
    with open(toml_path, "rb") as f:
        data = tomllib.load(f)
    params = {
        k: float(v)
        for k, v in data.items()
        if k in ("fx", "fy", "cx", "cy", "k1", "k2", "k3", "p1", "p2")
    }
    # #region agent log
    try:
        with open(
            "/home/preto/Preto/vaila/.cursor/debug-a5f5a000-975d-4bfc-9676-f9748629bda8.log", "a"
        ) as _f:
            _f.write(
                json.dumps(
                    {
                        "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                        "id": "lensdistort_load_toml",
                        "timestamp": int(time.time() * 1000),
                        "location": "vaila_lensdistortvideo.load_distortion_parameters",
                        "message": "TOML params loaded",
                        "data": {
                            "script": "vaila_lensdistortvideo",
                            "path": toml_path,
                            "ext": os.path.splitext(toml_path)[1],
                            "keys_count": len(params),
                        },
                        "runId": "distort",
                        "hypothesisId": "A",
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    return params


def get_precise_video_metadata(video_path):
    """
    Get precise video metadata using ffprobe to avoid rounding errors.
    Returns dict with fps (float), width, height, codec, etc.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, check=True
        )
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            # Fallback to OpenCV if ffprobe fails
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {
                "fps": fps,
                "width": width,
                "height": height,
                "codec": "unknown",
                "r_frame_rate": None,
                "avg_frame_rate": None,
            }

        # Get precise FPS from r_frame_rate or avg_frame_rate
        r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
        avg_frame_rate_str = video_stream.get("avg_frame_rate", "0/0")

        # Convert fraction strings to float
        def fraction_to_float(frac_str):
            try:
                if "/" in frac_str:
                    num, den = map(int, frac_str.split("/"))
                    return float(num) / den if den != 0 else 0.0
                return float(frac_str)
            except (ValueError, ZeroDivisionError):
                return None

        r_fps = fraction_to_float(r_frame_rate_str)
        avg_fps = fraction_to_float(avg_frame_rate_str)

        # Use avg_frame_rate if available, otherwise r_frame_rate
        fps = avg_fps if avg_fps and avg_fps > 0 else (r_fps if r_fps and r_fps > 0 else 30.0)

        # Get frame count if available
        nb_frames = None
        try:
            if "nb_frames" in video_stream and video_stream["nb_frames"] not in (None, "N/A", ""):
                nb_frames = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            nb_frames = None

        # Calculate frame count from duration and FPS if nb_frames not available
        duration = float(data.get("format", {}).get("duration", 0))
        if nb_frames is None and duration > 0 and fps > 0:
            nb_frames = int(round(duration * fps))

        return {
            "fps": fps,
            "width": int(video_stream.get("width")),
            "height": int(video_stream.get("height")),
            "codec": video_stream.get("codec_name", "unknown"),
            "r_frame_rate": r_frame_rate_str,
            "avg_frame_rate": avg_frame_rate_str,
            "duration": duration if duration > 0 else None,
            "nb_frames": nb_frames,
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        # Fallback to OpenCV if ffprobe is not available
        rprint(
            f"[yellow]Warning: ffprobe not available or failed, using OpenCV fallback: {e}[/yellow]"
        )
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "total_frames": total_frames,
            "codec": "unknown",
            "r_frame_rate": None,
            "avg_frame_rate": None,
        }


def process_video(input_path, output_path, parameters):
    """Process video applying lens distortion correction."""
    console = Console()

    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")

    # Get video properties using precise metadata
    metadata = get_precise_video_metadata(input_path)
    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]

    # Get total frames (prefer metadata, fallback to cap)
    if "nb_frames" in metadata and metadata["nb_frames"]:
        total_frames = metadata["nb_frames"]
    elif "total_frames" in metadata:
        total_frames = metadata["total_frames"]
    else:
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    rprint(f"[dim]Metadata: {width}x{height} @ {fps:.4f} fps, {total_frames} frames[/dim]")

    # Create camera matrix and distortion coefficients
    camera_matrix = np.array(
        [
            [parameters["fx"], 0, parameters["cx"]],
            [0, parameters["fy"], parameters["cy"]],
            [0, 0, 1],
        ]
    )

    dist_coeffs = np.array(
        [
            parameters["k1"],
            parameters["k2"],
            parameters["p1"],
            parameters["p2"],
            parameters["k3"],
        ]
    )

    # Get optimal new camera matrix
    new_camera_matrix, roi = cv2.getOptimalNewCameraMatrix(
        camera_matrix, dist_coeffs, (width, height), 1, (width, height)
    )

    # Create temporary directory for frames
    temp_dir = os.path.join(os.path.dirname(output_path), "temp_frames")
    os.makedirs(temp_dir, exist_ok=True)

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
            TimeElapsedColumn(),
            console=console,
        ) as progress:
            # Add tasks
            process_task = progress.add_task("[cyan]Processing frames...", total=total_frames)

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                # Undistort frame
                undistorted = cv2.undistort(
                    frame, camera_matrix, dist_coeffs, None, new_camera_matrix
                )

                # Save frame as PNG (lossless)
                frame_path = os.path.join(temp_dir, f"frame_{frame_count:06d}.png")
                cv2.imwrite(frame_path, undistorted)

                # Update progress
                frame_count += 1
                progress.update(process_task, advance=1)

                # Show additional info every 100 frames
                if frame_count % 100 == 0:
                    elapsed = progress.tasks[0].elapsed
                    if elapsed:
                        fps_processing = frame_count / elapsed
                        remaining = (total_frames - frame_count) / fps_processing
                        progress.console.print(
                            f"[dim]Processing speed: {fps_processing:.1f} fps | "
                            f"Estimated time remaining: {remaining:.1f}s[/dim]"
                        )

        # Use FFmpeg to create high-quality video
        rprint("\n[yellow]Creating final video with FFmpeg...[/yellow]")
        input_pattern = os.path.join(temp_dir, "frame_%06d.png")
        ffmpeg_cmd = [
            "ffmpeg",
            "-y",  # Overwrite output file if it exists
            "-framerate",
            str(fps),
            "-i",
            input_pattern,
            "-c:v",
            "libx264",  # Use H.264 codec
            "-preset",
            "slow",  # Higher quality encoding
            "-crf",
            "18",  # High quality (0-51, lower is better)
            "-pix_fmt",
            "yuv420p",  # Standard pixel format
            output_path,
        ]

        subprocess.run(ffmpeg_cmd, check=True)

    finally:
        # Release video capture
        cap.release()

        # Clean up temporary files
        if os.path.exists(temp_dir):
            for file in os.listdir(temp_dir):
                os.remove(os.path.join(temp_dir, file))
            os.rmdir(temp_dir)

    rprint("\n[green]Video processing complete![/green]")
    rprint(f"[blue]Output saved as: {output_path}[/blue]")


def select_directory(title="Select a directory"):
    """
    Open a dialog to select a directory.
    """
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory


def select_file(title="Select a file", filetypes=(("TOML Files", "*.toml"),)):
    """
    Open a dialog to select a file.
    """
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    return file_path


def run_distortvideo():
    """Main function to run batch lens distortion correction."""
    rprint("[yellow]Running batch lens distortion correction...[/yellow]")

    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")

    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Batch lens distortion correction for videos.")
    parser.add_argument("--input_dir", type=str, help="Directory containing videos to process")
    parser.add_argument(
        "--params_file", type=str, help="Path to the camera calibration parameters TOML file"
    )
    parser.add_argument(
        "--output_dir", type=str, help="Optional output directory for processed videos"
    )
    args = parser.parse_args()

    # Determine input directory
    if args.input_dir:
        input_dir = args.input_dir
        if not os.path.isdir(input_dir):
            rprint(f"[red]Error: Input directory not found: {input_dir}[/red]")
            return
        rprint(f"\n[cyan]Using input directory: {input_dir}[/cyan]")
    else:
        # Select input directory via GUI
        rprint("\nSelect the directory containing videos:")
        input_dir = select_directory(title="Select Directory with Videos")
        if not input_dir:
            rprint("[red]No directory selected. Exiting.[/red]")
            return

    # Determine parameters file
    # #region agent log
    _log_path_lens = (
        "/home/preto/Preto/vaila/.cursor/debug-a5f5a000-975d-4bfc-9676-f9748629bda8.log"
    )
    # #endregion
    if args.params_file:
        parameters_path = args.params_file
        if not os.path.isfile(parameters_path):
            rprint(f"[red]Error: Parameters file not found: {parameters_path}[/red]")
            return
        rprint(f"\n[cyan]Using parameters file: {parameters_path}[/cyan]")
        # #region agent log
        try:
            with open(_log_path_lens, "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                            "id": "lensdistort_mode",
                            "timestamp": int(time.time() * 1000),
                            "location": "vaila_lensdistortvideo.run_distortvideo",
                            "message": "Params source",
                            "data": {
                                "script": "vaila_lensdistortvideo",
                                "mode": "cli",
                                "params_path": parameters_path,
                            },
                            "runId": "distort",
                            "hypothesisId": "B",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
    else:
        # Select parameters file via GUI
        rprint("\nSelect the camera calibration parameters file:")
        parameters_path = select_file(
            title="Select Parameters File",
            filetypes=(("TOML Files", "*.toml"), ("All Files", "*.*")),
        )
        if not parameters_path:
            rprint("[red]No parameters file selected. Exiting.[/red]")
            return
        # #region agent log
        try:
            with open(_log_path_lens, "a") as _f:
                _f.write(
                    json.dumps(
                        {
                            "sessionId": "a5f5a000-975d-4bfc-9676-f9748629bda8",
                            "id": "lensdistort_mode",
                            "timestamp": int(time.time() * 1000),
                            "location": "vaila_lensdistortvideo.run_distortvideo",
                            "message": "Params source",
                            "data": {
                                "script": "vaila_lensdistortvideo",
                                "mode": "gui",
                                "params_path": parameters_path,
                            },
                            "runId": "distort",
                            "hypothesisId": "B",
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion

    # Load parameters
    try:
        parameters = load_distortion_parameters(parameters_path)
    except Exception as e:
        rprint(f"[red]Error loading parameters: {e}[/red]")
        return

    # Get all video files in the directory
    video_extensions = (".mp4", ".avi", ".mov")
    video_files = [
        f
        for f in os.listdir(input_dir)
        if os.path.isfile(os.path.join(input_dir, f)) and f.lower().endswith(video_extensions)
    ]

    if not video_files:
        rprint("[red]No video files found in the selected directory.[/red]")
        return

    rprint(f"\n[cyan]Found {len(video_files)} video files to process.[/cyan]")

    # Create output directory
    if args.output_dir:
        output_dir = args.output_dir
    else:
        # Create output directory with a timestamp in the name: vaila_lensdistort_timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = os.path.join(input_dir, f"vaila_lensdistort_{timestamp}")

    os.makedirs(output_dir, exist_ok=True)
    rprint(f"[cyan]Output directory: {output_dir}[/cyan]")

    # Process each video
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        base_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(output_dir, f"{base_name}_undistorted.mp4")

        try:
            rprint(f"\n[cyan]Processing video: {video_file}[/cyan]")
            process_video(input_path, output_path, parameters)
        except Exception as e:
            rprint(f"[red]Error processing video {video_file}: {e}[/red]")
            continue

    # Try to open the output folder
    try:
        if os.name == "nt":  # Windows
            os.startfile(output_dir)
        elif os.name == "posix":  # macOS and Linux
            subprocess.run(["xdg-open", output_dir])
    except Exception as e:
        rprint(f"[red]Could not open output directory: {e}[/red]")

    rprint("\n[green]Batch processing complete![/green]")


if __name__ == "__main__":
    run_distortvideo()
