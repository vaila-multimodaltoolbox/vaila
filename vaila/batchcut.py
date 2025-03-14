"""
Batch Video Cutting Script with GPU Acceleration
Author: Prof. PhD. Paulo Santiago
Date: September 29, 2024
Version: 1.1

Description:
This script performs batch video cutting by processing a list of videos, extracting specified segments
based on frame ranges, and saving them in a structured output directory. The script supports GPU
acceleration via NVIDIA NVENC when available, defaulting to CPU-based processing if a GPU is not detected.

The script reads a list file where each line specifies the original video name, the desired name
for the cut video, the start frame, and the end frame. The videos are processed and saved in a "cut_videos"
subdirectory inside the specified output directory.

List file format:
<original_name> <new_name> <start_frame> <end_frame>

Example:
PC001_STS_02_FLIRsagital.avi PC001_STS_02_FLIRsagital_cut.mp4 100 300

The script automatically removes duplicate ".mp4" extensions from the new file name if necessary.

### Key Features:
1. **Batch Video Processing**: Processes multiple video files in a batch based on the segments specified
   in the list file.
2. **Frame-Based Cutting**: Allows precise cutting of videos by specifying start and end frames for each segment.
3. **GPU Acceleration**: Uses NVIDIA GPU with NVENC for accelerated video processing if available. Falls back
   to CPU-based processing using `libx264` if no GPU is detected.
4. **Organized Output**: Saves all cut videos in a "cut_videos" subdirectory inside the specified output directory.

### Dependencies:
- FFmpeg (installed via Conda or available in PATH)
- tkinter (for file and directory selection dialogs)
- rich (for enhanced console output)

### Usage:
1. Run the script.
2. Select the directory containing the videos to be processed.
3. Select the list file containing video names and frame ranges.
4. Select the output directory where the cut videos will be saved.

The script will then process the videos and save the cut segments in the "cut_videos" folder inside the output directory.

### GPU Support:
- If an NVIDIA GPU with NVENC support is detected, the script will use the `h264_nvenc` codec for hardware-accelerated video cutting.
- If no GPU is available, the script will fall back to using the `libx264` codec for CPU-based processing.

### Example list file format:
<original_video_name> <new_video_name> <start_frame> <end_frame>
PC001_STS_02_FLIRsagital.avi PC001_STS_02_FLIRsagital_cut.mp4 100 300

### Changelog:
Version 1.1 (2024-09-29):
- Added GPU acceleration using NVIDIA NVENC when available.
- Improved file name handling to avoid duplicate ".mp4" extensions.
- Enhanced error handling and output organization.

Version 1.0 (2024-08-12):
- Initial release with basic batch video cutting functionality.
"""

import os
import subprocess
import platform
import sys
import tkinter as tk
from tkinter import filedialog
from rich import print


def is_nvidia_gpu_available():
    """Check if an NVIDIA GPU is available using nvidia-smi."""
    system_platform = platform.system()
    try:
        if system_platform == "Windows" or system_platform == "Linux":
            result = subprocess.run(["nvidia-smi"], capture_output=True, text=True)
            if "NVIDIA" in result.stdout:
                print("NVIDIA GPU detected.")
                return True
            else:
                print("NVIDIA GPU not detected.")
        else:
            print(f"{system_platform} does not support NVIDIA GPUs.")
    except FileNotFoundError:
        print("nvidia-smi not found. No NVIDIA GPU available.")
    return False


def batch_cut_videos(video_directory, list_file_path, output_directory):
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting batch video cutting...")

    if not os.path.isfile(list_file_path):
        print(f"List file {list_file_path} does not exist.")
        return

    # Perform GPU check once at the start
    use_gpu = is_nvidia_gpu_available()
    codec = "h264_nvenc" if use_gpu else "libx264"

    if use_gpu:
        print("Using h264_nvenc for video processing (GPU).")
    else:
        print("Using libx264 for video processing (CPU).")

    # Create the "cut_videos" subdirectory inside the output directory
    output_directory = os.path.join(output_directory, "cut_videos")
    os.makedirs(output_directory, exist_ok=True)

    with open(list_file_path, "r") as file:
        lines = file.readlines()

    for line in lines:
        if not line.strip():
            continue

        parts = line.split()
        if len(parts) != 4:
            print(
                f"Line format error: {line.strip()} - expected 4 parts, got {len(parts)}"
            )
            continue

        original_name, new_name, start_frame, end_frame = parts
        start_frame, end_frame = int(start_frame), int(end_frame)

        original_path = os.path.join(video_directory, original_name)

        # Remove the duplicate .mp4 extension if it already exists in new_name
        if new_name.endswith(".mp4"):
            new_name = new_name[:-4]

        new_path = os.path.join(output_directory, f"{new_name}.mp4")

        try:
            print(f"Processing {original_name} from frame {start_frame} to {end_frame}")

            command = [
                "ffmpeg",
                "-y",  # overwrite output files
                "-i",
                original_path,  # input file
                "-vf",
                f"select=between(n\\,{start_frame}\\,{end_frame})",  # video filter
                "-vsync",
                "vfr",  # variable frame rate
                "-c:v",
                codec,  # Use GPU if available, otherwise CPU
                new_path,  # output file
            ]

            subprocess.run(command, check=True)
            print(f"{new_name} completed!")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {original_name}: {str(e)}", file=sys.stderr)


def cut_videos():
    root = tk.Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos"
    )
    if not video_directory:
        print("No video directory selected.")
        return

    list_file_path = filedialog.askopenfilename(
        title="Select the list file",
        filetypes=(("Text files", "*.txt"), ("All files", "*.*")),
    )
    if not list_file_path:
        print("No list file selected.")
        return

    output_directory = filedialog.askdirectory(title="Select the output directory")
    if not output_directory:
        print("No output directory selected.")
        return

    root.destroy()

    batch_cut_videos(video_directory, list_file_path, output_directory)


if __name__ == "__main__":
    cut_videos()
