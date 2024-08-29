"""
Batch Video Cutting Script
Author: Prof. PhD. Paulo Santiago
Date: August 12, 2024
Version: 1.0

Description:
This script performs batch video cutting, processing a list of videos and saving the specified parts
in an output directory. The video list should contain the original video name, the new name for the cut
video, the start frame, and the end frame.

List file format:
<original_name> <new_name> <start_frame> <end_frame>

Example:
PC001_STS_02_FLIRsagital.avi PC001_STS_02_FLIRsagital_200_100_300.mp4 100 300

The script creates a "cut_videos" subdirectory in the specified output directory, where the cut videos
will be saved. If the new name already contains the .mp4 extension, it will be removed to avoid duplication.

Dependencies:
- FFmpeg (installed via Conda or available in PATH)
- tkinter

Usage:
Run the script and follow the prompts to select the video directory, the list file, and the output
directory.

"""

import os
import sys
import subprocess
import tkinter as tk
from tkinter import filedialog


def batch_cut_videos(video_directory, list_file_path, output_directory):
    if not os.path.isfile(list_file_path):
        print(f"List file {list_file_path} does not exist.")
        return

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
