"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# videoprocessor.py
#
# Description:
# This script allows users to process and edit video files, enabling batch processing of videos with the ability to concatenate 
# a portion of one video (`video_A`) to another (`video_B`). Users can specify the percentage of `video_A` to use and 
# can optionally reverse the segment before concatenation. The script supports custom text files for batch processing and 
# includes a GUI for directory and file selection.
#
# Key Features:
# - Graphical User Interface (GUI) for easy selection of directories and file inputs.
# - Configurable percentage of `video_A` to concatenate onto `video_B`.
# - Option to reverse the concatenated segment of `video_A` (using the `vfx.time_mirror` from MoviePy).
# - Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for each video.
# - Automatic creation of output directories based on a timestamp for organized file management.
# - Detailed console output for tracking progress and handling errors.
#
# Usage:
# - Run the script to open a graphical interface. After selecting the source and target directories, 
#   the script will either process videos based on user input or use custom instructions from a text file.
# - The processed videos will be saved in a new output directory named with a timestamp.
#
# Requirements:
# - FFmpeg must be installed and accessible in the system PATH.
# - Python 3.x environment with the `moviepy` library installed.
# - Tkinter for the GUI components (usually included with Python).
#
# Installation of FFmpeg (for video processing):
# - **Conda (recommended)**: 
#   ```bash
#   conda install -c conda-forge ffmpeg
#   ```
# - **Direct download**: 
#   - Download precompiled builds for Windows/macOS/Linux from: https://ffmpeg.org/download.html
# 
# NVIDIA GPU Installation and FFmpeg NVENC Support
#
# To utilize NVIDIA GPU acceleration for video processing, follow the steps below:
#
# ## Windows:
# 1. **Install NVIDIA Drivers**:
#    - Download and install the latest NVIDIA drivers from: https://www.nvidia.com/Download/index.aspx.
#    - Ensure your GPU supports NVENC (Kepler series or newer).
#
# 2. **Install FFmpeg with NVENC Support**:
#    - Download the FFmpeg build with NVENC support from: https://www.gyan.dev/ffmpeg/builds/.
#    - Add the FFmpeg `bin` folder to your system's PATH.
#
# ## Linux:
# 1. **Install NVIDIA Drivers**:
#    - Use the following commands for Ubuntu-based distributions:
#      ```bash
#      sudo add-apt-repository ppa:graphics-drivers/ppa
#      sudo apt update
#      sudo apt install nvidia-driver-<version>
#      ```
# 
# 2. **Install CUDA**:
#    - Install CUDA from: https://developer.nvidia.com/cuda-downloads and follow the installation instructions.
#
# 3. **Compile FFmpeg with NVENC Support**:
#    - Download FFmpeg source and compile it with NVENC:
#      ```bash
#      sudo apt install ffmpeg
#      git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
#      cd ffmpeg
#      ./configure --enable-nonfree --enable-nvenc --enable-gpl
#      make
#      sudo make install
#      ```
# 
# 4. **Verify Installation**:
#    - Run:
#      ```bash
#      ffmpeg -encoders | grep nvenc
#      ```
#    - Look for `h264_nvenc` and `hevc_nvenc` encoders.
#
# More details on NVIDIA support for FFmpeg can be found at:
# - [NVIDIA Developer Guide for FFmpeg](https://developer.nvidia.com/ffmpeg)
# - [FFmpeg Documentation](https://ffmpeg.org/documentation.html)
#
# Dependencies:
# - Python 3.x
# - moviepy library for video editing: `pip install moviepy`
# - Tkinter for GUI components: `pip install tk` (usually bundled with Python)
# - FFmpeg for handling video processing (installed via Conda or manually).
#
# Note:
# Processing large video files may take significant time depending on the system’s performance.
# The script supports multiple formats, including `.mp4`, `.avi`, `.mov`, and `.mkv`.
#
"""

import os
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
from tkinter import filedialog, messagebox, simpledialog


def process_videos(
    source_dir,
    target_dir,
    percentage=50,
    reverse=False,
    use_text_file=False,
    text_file_path=None,
):
    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"mergedvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # Use provided text file if specified
    if use_text_file and text_file_path:
        video_files = []
        with open(text_file_path, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if "," in line:
                    source_video, percent, target_video = line.split(",")
                else:
                    source_video, percent, target_video = line.split()

                percent = float(percent)
                video_files.append(
                    (
                        os.path.join(source_dir, source_video),
                        percent,
                        os.path.join(output_dir, target_video),
                    )
                )
    else:
        video_files = []
        for file in os.listdir(source_dir):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append(
                    (
                        os.path.join(source_dir, file),
                        percentage,
                        os.path.join(output_dir, file),
                    )
                )

    for source_video, percent, target_video in video_files:
        try:
            video_original = VideoFileClip(source_video)

            # If percentage is 100, use the entire video
            if percent == 100:
                video_A = video_original
            else:
                video_A_duration = video_original.duration * (percent / 100)
                video_A = video_original.subclip(0, video_A_duration)

            if reverse:
                video_A = video_A.fx(vfx.time_mirror)  # Reverse the video segment

            # Use video_B if exists, otherwise use video_A for concatenation
            video_B = (
                VideoFileClip(target_video)
                if os.path.exists(target_video)
                else video_original
            )

            video_final = concatenate_videoclips([video_A, video_B])

            # Save the concatenated video
            base_name = os.path.splitext(os.path.basename(target_video))[0]
            inserted_frames = int(
                video_A.fps * video_A.duration
            )  # Number of frames inserted
            output_suffix = (
                f"rframes_{inserted_frames}" if reverse else f"frames_{inserted_frames}"
            )
            output_file = os.path.join(output_dir, f"{base_name}_{output_suffix}.mp4")
            video_final.write_videofile(
                output_file, fps=video_final.fps, codec="libx264", bitrate="8000k"
            )

            print(f"Video processed and saved to: {output_file}")
        except Exception as e:
            print(f"Error processing video {source_video} into {target_video}: {e}")


def process_videos_gui():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting video processing...")

    # Ask user to select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Ask user to select target directory
    target_dir = filedialog.askdirectory(title="Select Target Directory")
    if not target_dir:
        messagebox.showerror("Error", "No target directory selected.")
        return

    # Ask if the user wants to use a text file
    use_text_file = messagebox.askyesno(
        "Use Text File", "Do you want to use a text file (videos_e_frames.txt)?"
    )
    text_file_path = None
    if use_text_file:
        text_file_path = filedialog.askopenfilename(
            title="Select videos_e_frames.txt", filetypes=[("Text files", "*.txt")]
        )
        if not text_file_path:
            messagebox.showerror("Error", "No text file selected.")
            return

    # Ask user for the percentage and reverse option
    user_input = simpledialog.askstring(
        "Percentage and Reverse",
        "Enter the percentage of video_A to concatenate to video_B (default is 50%). "
        "To reverse the video, add 'R' (e.g., '30,R' for 30% reversed):",
    )

    if user_input is None:
        percentage = 50  # Default value if user cancels input
        reverse = False
    else:
        user_input = user_input.strip().split(",")
        percentage = float(user_input[0]) if user_input[0] else 50
        reverse = "R" in user_input[1].upper() if len(user_input) > 1 else False

    # Call the process_videos function with the collected inputs
    process_videos(
        source_dir, target_dir, percentage, reverse, use_text_file, text_file_path
    )


if __name__ == "__main__":
    process_videos_gui()
