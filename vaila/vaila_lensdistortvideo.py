"""
===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 20 December 2024
Version: 0.0.1
Python Version: 3.12.8
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
   - Example for a 2.8 mm lens with 105° horizontal FOV, 58° vertical FOV, and 1920×1080 resolution:
     fx ≈ (1920 / 2) / tan(105° / 2) ≈ 949.41 pixels
     fy ≈ (1080 / 2) / tan(58° / 2) ≈ 950.63 pixels

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

Example CSV Format:
-------------------
Below is an example of a CSV file containing calibration parameters for use with this script:

    fx,fy,cx,cy,k1,k2,k3,p1,p2
    949.41,950.63,960.00,540.00,-0.28871370110181493,0.1374614711665278,-0.025511562284832402,0.00044281215436799446,-0.00042111749309847274

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

import cv2
import numpy as np
import pandas as pd
import os
from rich import print
import tkinter as tk
from tkinter import filedialog
from datetime import datetime
from rich.progress import Progress, SpinnerColumn, TimeElapsedColumn, BarColumn, TextColumn
from rich.console import Console
from rich import print as rprint
import subprocess


def load_distortion_parameters(csv_path):
    """
    Load distortion parameters from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return df.iloc[0].to_dict()


def process_video(input_path, output_path, parameters):
    """Process video applying lens distortion correction."""
    console = Console()
    
    # Open video capture
    cap = cv2.VideoCapture(input_path)
    if not cap.isOpened():
        raise ValueError("Error opening video file")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # Create camera matrix and distortion coefficients
    camera_matrix = np.array([
        [parameters["fx"], 0, parameters["cx"]],
        [0, parameters["fy"], parameters["cy"]],
        [0, 0, 1]
    ])
    
    dist_coeffs = np.array([
        parameters["k1"],
        parameters["k2"],
        parameters["p1"],
        parameters["p2"],
        parameters["k3"]
    ])
    
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
            process_task = progress.add_task(
                "[cyan]Processing frames...", 
                total=total_frames
            )
            
            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Undistort frame
                undistorted = cv2.undistort(
                    frame, 
                    camera_matrix, 
                    dist_coeffs, 
                    None, 
                    new_camera_matrix
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
            "-framerate", str(fps),
            "-i", input_pattern,
            "-c:v", "libx264",  # Use H.264 codec
            "-preset", "slow",  # Higher quality encoding
            "-crf", "18",  # High quality (0-51, lower is better)
            "-pix_fmt", "yuv420p",  # Standard pixel format
            output_path
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
    
    rprint(f"\n[green]Video processing complete![/green]")
    rprint(f"[blue]Output saved as: {output_path}[/blue]")


def select_directory(title="Select a directory"):
    """
    Open a dialog to select a directory.
    """
    root = tk.Tk()
    root.withdraw()
    directory = filedialog.askdirectory(title=title)
    return directory


def select_file(title="Select a file", filetypes=(("CSV Files", "*.csv"),)):
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
    
    # Select input directory
    rprint("\nSelect the directory containing videos:")
    input_dir = select_directory(title="Select Directory with Videos")
    if not input_dir:
        rprint("[red]No directory selected. Exiting.[/red]")
        return
    
    # Select parameters file
    rprint("\nSelect the camera calibration parameters file:")
    parameters_path = select_file(
        title="Select Parameters File",
        filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
    )
    if not parameters_path:
        rprint("[red]No parameters file selected. Exiting.[/red]")
        return
    
    # Load parameters
    try:
        parameters = load_distortion_parameters(parameters_path)
    except Exception as e:
        rprint(f"[red]Error loading parameters: {e}[/red]")
        return
    
    # Get all video files in the directory
    video_extensions = ('.mp4', '.avi', '.mov')
    video_files = [f for f in os.listdir(input_dir) 
                  if os.path.isfile(os.path.join(input_dir, f)) 
                  and f.lower().endswith(video_extensions)]
    
    if not video_files:
        rprint("[red]No video files found in the selected directory.[/red]")
        return
    
    rprint(f"\n[cyan]Found {len(video_files)} video files to process.[/cyan]")
    
    # Process each video
    for video_file in video_files:
        input_path = os.path.join(input_dir, video_file)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(video_file)[0]
        output_path = os.path.join(input_dir, f"{base_name}_undistorted_{timestamp}.mp4")
        
        try:
            rprint(f"\n[cyan]Processing video: {video_file}[/cyan]")
            process_video(input_path, output_path, parameters)
        except Exception as e:
            rprint(f"[red]Error processing video {video_file}: {e}[/red]")
            continue
    
    # Try to open output folder
    try:
        if os.name == 'nt':  # Windows
            os.startfile(input_dir)
        elif os.name == 'posix':  # macOS and Linux
            subprocess.run(['xdg-open', input_dir])
    except:
        pass
    
    rprint("\n[green]Batch processing complete![/green]")


if __name__ == "__main__":
    run_distortvideo()
