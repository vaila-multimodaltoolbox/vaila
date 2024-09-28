"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# compress_videos_h264.py
# This script compresses videos in a specified directory to H.264 format
# using the FFmpeg tool. It provides a GUI for selecting the directory
# containing the videos, and then processes each video, saving the
# compressed versions in a subdirectory named 'compressed_h264'.
#
# Usage:
# Run the script to open the GUI, select the directory containing videos,
# and the compression process will start automatically.
#
# Requirements:
# - FFmpeg must be installed and accessible in the system PATH.
# - This script is designed to work in a Conda environment where FFmpeg is
#   installed via conda-forge.
#
# Dependencies:
# - Python 3.11.8
# - Tkinter (included with Python)
# - FFmpeg (installed via Conda or available in PATH)
#
# Installation of FFmpeg in Conda:
#   conda install -c conda-forge ffmpeg
#
# Note:
# This process may take several hours depending on the size of the videos
# and the performance of your computer.
"""

import os
import subprocess
import platform
from tkinter import filedialog, messagebox, Tk
import tempfile


def is_nvidia_gpu_available():
    """Check if an NVIDIA GPU is available and NVENC is supported."""
    try:
        # Check for NVIDIA GPU with NVENC support
        result = subprocess.run(["ffmpeg", "-encoders"], capture_output=True, text=True)
        return "h264_nvenc" in result.stdout
    except Exception as e:
        print(f"Error checking for NVIDIA GPU: {e}")
        return False


def find_videos_recursively(directory, output_directory):
    """Find all video files recursively in the directory, avoiding the output directory."""
    video_files = []
    for root, dirs, files in os.walk(directory):
        # Ignore the output directory
        if output_directory in root:
            continue
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
                video_files.append(os.path.join(root, file))
    return video_files


def create_temp_file_with_videos(video_files):
    """Create a temporary file containing the list of video files to process."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode='w')
    for video in video_files:
        temp_file.write(f"{video}\n")
    temp_file.close()
    return temp_file.name


def run_compress_videos_h264(temp_file_path, output_directory, preset="medium", crf=22, use_gpu=False):
    """Compress the list of video files stored in the temporary file to H.264 format using either CPU or GPU."""
    print("!!!ATTENTION!!!")
    print(
        "This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!"
    )

    os.makedirs(output_directory, exist_ok=True)

    # Read the video files from the temp file
    with open(temp_file_path, 'r') as temp_file:
        video_files = [line.strip() for line in temp_file]

    for video_file in video_files:
        input_path = video_file
        # Create corresponding output file path in the new directory
        relative_path = os.path.relpath(input_path, os.path.dirname(output_directory))
        output_path = os.path.join(output_directory, f"{os.path.splitext(relative_path)[0]}_h264.mp4")

        # Ensure output directory for the specific file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Compressing {video_file}...")

        if use_gpu:
            # If GPU is available, use NVIDIA NVENC for encoding
            command = [
                "ffmpeg",
                "-y",  # overwrite output files
                "-i", input_path,  # input file
                "-c:v", "h264_nvenc",  # Use NVIDIA NVENC for H.264
                "-preset", preset,  # preset for encoding speed
                "-b:v", "5M",  # bitrate (optional, adjust as needed)
                output_path,  # output file
            ]
        else:
            # Fallback to CPU-based encoding (libx264)
            command = [
                "ffmpeg",
                "-y",  # overwrite output files
                "-i", input_path,  # input file
                "-c:v", "libx264",  # video codec
                "-preset", preset,  # preset for encoding speed
                "-crf", str(crf),  # constant rate factor for quality
                output_path,  # output file
            ]

        try:
            subprocess.run(command, check=True)
            print(f"Done compressing {video_file} to H.264.")
        except subprocess.CalledProcessError as e:
            print(f"Error compressing {video_file}: {e}")

    print("All videos have been compressed successfully!")


def compress_videos_h264_gui():
    root = Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Create output directory
    output_directory = os.path.join(video_directory, "compressed_h264")

    # Detect system and GPU availability
    os_type = platform.system()
    print(f"Operating System: {os_type}")
    
    if os_type == "Linux" or os_type == "Windows":
        # Check if NVIDIA GPU with NVENC is available
        use_gpu = is_nvidia_gpu_available()
        if use_gpu:
            print("NVIDIA GPU with NVENC detected. Using GPU for video compression.")
        else:
            print("No NVIDIA GPU detected. Falling back to CPU-based compression.")
    elif os_type == "Darwin":  # macOS
        print("macOS detected. Using CPU-based compression (NVENC not available).")
        use_gpu = False
    else:
        print("Unsupported operating system. Using CPU-based compression.")
        use_gpu = False

    # Find all video files recursively, ignoring the output directory
    video_files = find_videos_recursively(video_directory, output_directory)

    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    # Create a temporary file with the list of video files
    temp_file_path = create_temp_file_with_videos(video_files)

    # Run the compression for all found videos
    preset = "medium"
    crf = 23

    run_compress_videos_h264(temp_file_path, output_directory, preset, crf, use_gpu)

    # Remove temporary file
    os.remove(temp_file_path)

    messagebox.showinfo(
        "Success",
        "Video compression completed. All videos have been compressed successfully!",
    )


if __name__ == "__main__":
    compress_videos_h264_gui()

