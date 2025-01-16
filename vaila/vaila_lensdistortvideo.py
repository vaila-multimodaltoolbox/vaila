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


def load_distortion_parameters(csv_path):
    """
    Load distortion parameters from a CSV file.
    """
    df = pd.read_csv(csv_path)
    return df.iloc[0].to_dict()


def process_video(video_path, output_path, parameters):
    """
    Process a single video to apply lens distortion correction.
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return

    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")

    out = cv2.VideoWriter(output_path, fourcc, fps, (frame_width, frame_height))

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

    new_camera_matrix, _ = cv2.getOptimalNewCameraMatrix(
        camera_matrix,
        dist_coeffs,
        (frame_width, frame_height),
        1,
        (frame_width, frame_height),
    )

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        undistorted_frame = cv2.undistort(
            frame, camera_matrix, dist_coeffs, None, new_camera_matrix
        )
        out.write(undistorted_frame)

    cap.release()
    out.release()


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
    """
    Main function to process videos in a directory using distortion parameters.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    print("Select the video directory:")
    video_directory = select_directory()
    if not video_directory:
        print("No directory selected. Exiting.")
        return

    print("Select the distortion parameters CSV file:")
    parameters_path = select_file()
    if not parameters_path:
        print("No parameters file selected. Exiting.")
        return

    parameters = load_distortion_parameters(parameters_path)

    # Generate output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d")
    output_directory = os.path.join(video_directory, f"vaila_distort_{timestamp}")
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    for file in os.listdir(video_directory):
        file_path = os.path.join(video_directory, file)
        if os.path.isfile(file_path) and file.lower().endswith(
            (".mp4", ".avi", ".mov", ".mkv")
        ):
            output_path = os.path.join(output_directory, f"distorted_{file}")
            print(f"Processing: {file_path}")
            process_video(file_path, output_path, parameters)

    print(f"Processed videos saved in: {output_directory}")


if __name__ == "__main__":
    run_distortvideo()
