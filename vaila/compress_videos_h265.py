"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# compress_videos_h265.py
# This script compresses videos in a specified directory to H.265/HEVC format.
# It identifies the operating system and calls the appropriate shell script
# for compression: compress_videos_h265.sh for Linux/macOS and 
# compress_videos_h265.bat for Windows.
#
# Usage:
# Run the script to open the GUI, select the directory containing videos,
# and the appropriate shell script for your OS will handle the compression.
#
# Requirements:
# - A valid shell script for your OS (compress_videos_h265.sh or compress_videos_h265.bat)
#   must be available in the same directory as this Python script.
#
# Dependencies:
# - Python 3.11.8
# - Tkinter (included with Python)
# - subprocess (included with Python)
#
# Note:
# This process may take several hours depending on the size of the videos
# and the performance of your computer.
"""

import os
import subprocess
import platform
import sys


def compress_video_h265(input_file, output_file, preset="medium", crf=23):
    """
    Compress a video file to H.265/HEVC format using FFmpeg.
    
    Parameters:
        input_file (str): Path to the input video file.
        output_file (str): Path to the output compressed video file.
        preset (str): Compression preset for FFmpeg (default is "medium").
        crf (int): Constant Rate Factor (default is 23).
    """
    command = [
        "ffmpeg",
        "-y",
        "-i",
        input_file,
        "-c:v",
        "libx265",
        "-preset",
        preset,
        "-crf",
        str(crf),
        output_file,
    ]

    try:
        print(f"Compressing {input_file}...")
        subprocess.check_call(command)
        print(f"Compression completed successfully: {output_file}")
    except subprocess.CalledProcessError as e:
        print(f"Error compressing video: {e}")
        sys.exit(1)


def run_compress_videos_h265(video_directory, preset="medium", crf=23):
    """
    Compress all video files in a specified directory to H.265/HEVC format.
    
    Parameters:
        video_directory (str): Directory containing videos to compress.
        preset (str): Compression preset for FFmpeg (default is "medium").
        crf (int): Constant Rate Factor (default is 23).
    """
    output_directory = os.path.join(video_directory, "compressed_h265")
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
            output_directory, f"{os.path.splitext(video_file)[0]}_h265.mp4"
        )

        compress_video_h265(input_path, output_path, preset, crf)

    print("All videos have been compressed successfully!")


def compress_videos_h265_gui():
    from tkinter import filedialog, messagebox, Tk

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

    run_compress_videos_h265(video_directory, preset, crf)
    messagebox.showinfo(
        "Success",
        "Video compression completed. All videos have been compressed successfully!",
    )


if __name__ == "__main__":
    compress_videos_h265_gui()

