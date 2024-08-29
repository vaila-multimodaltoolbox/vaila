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
from tkinter import filedialog, messagebox, Tk


def run_compress_videos_h264(video_directory, preset="medium", crf=23):
    output_directory = os.path.join(video_directory, "compressed_h264")
    os.makedirs(output_directory, exist_ok=True)

    print("!!!ATTENTION!!!")
    print(
        "This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!"
    )

    for video_file in sorted(
        [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    ):
        input_path = os.path.join(video_directory, video_file)
        output_path = os.path.join(
            output_directory, f"{os.path.splitext(video_file)[0]}_h264.mp4"
        )

        print(f"Compressing {video_file}...")

        command = [
            "ffmpeg",
            "-y",  # overwrite output files
            "-i",
            input_path,  # input file
            "-c:v",
            "libx264",  # video codec
            "-preset",
            preset,  # preset for encoding speed
            "-crf",
            str(crf),  # constant rate factor for quality
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

    preset = "medium"
    crf = 23

    run_compress_videos_h264(video_directory, preset, crf)
    messagebox.showinfo(
        "Success",
        "Video compression completed. All videos have been compressed successfully!",
    )


if __name__ == "__main__":
    compress_videos_h264_gui()
