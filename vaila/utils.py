"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# utils.py
# This script provides various utility functions, including drawing boxes on videos.
# The script uses FFmpeg to process videos, applying a rectangular mask based on user-defined coordinates.
#
# Usage:
# Run the script with the argument 'draw_box' to open a GUI for selecting the coordinates
# and apply them to all videos in the specified directory.
#
# Requirements:
# - FFmpeg must be installed and accessible in the system PATH.
# - This script is designed to work in a Conda environment where FFmpeg is
#   installed via conda-forge.
#
# Dependencies:
# - Python 3.12.9
# - Tkinter (included with Python)
# - FFmpeg (installed via Conda or available in PATH)
#
# Installation of FFmpeg in Conda:
#   conda install -c conda-forge ffmpeg
#
# Note:
# Ensure that the video files are in a format supported by FFmpeg.
"""

import os
import subprocess
import sys

import cv2
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


def get_box_coordinates(image_path):
    img = mpimg.imread(image_path)
    plt.imshow(img)
    plt.title("Click on the top-left and bottom-right corners of the box")
    points = plt.ginput(2)
    plt.close()

    if len(points) != 2:
        raise ValueError("Two points were not selected.")

    (XT, YT), (XD, YD) = points
    return int(XT), int(YT), int(XD), int(YD)


def save_first_frame(video_path, frame_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(frame_path, image)
    vidcap.release()


def draw_box_on_videos(video_directory, coordinates):
    XT, YT, XD, YD = coordinates
    output_directory = os.path.join(video_directory, "video_2_drawbox_boxedge_h265")
    os.makedirs(output_directory, exist_ok=True)

    for video_file in sorted(
        [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    ):
        input_path = os.path.join(video_directory, video_file)
        output_path = os.path.join(output_directory, f"{os.path.splitext(video_file)[0]}_boxe.mp4")

        print(f"Processing {video_file}...")

        command = [
            "ffmpeg",
            "-i",
            input_path,
            "-vf",
            f"drawbox=0:0:in_w:{YT}:color=black:t=fill,"
            f"drawbox=0:{YT}:{XT}:in_h:color=black:t=fill,"
            f"drawbox={XT}:{YD}:in_w:in_h:color=black:t=fill,"
            f"drawbox={XD}:{YT}:in_w:{YD}:color=black:t=fill",
            "-c:v",
            "libx265",
            "-crf",
            "22",
            "-hide_banner",
            "-nostats",
            "-loglevel",
            "quiet",
            output_path,
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Done DRAW BOX EDGES -------> {video_file}")
        except subprocess.CalledProcessError as e:
            print(f"Error processing {video_file}: {e}")


def run_drawbox():
    video_directory = "./data/videos_2_edition/video_2_drawbox"
    video_files = sorted(
        [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )

    if not video_files:
        print(f"{' ' * 10}==================================================")
        print(f"{' ' * 10}##################################################")
        print(f"{' ' * 20}No video files found in the directory.       ")
        print(f"{' ' * 10}##################################################")
        print(f"{' ' * 10}==================================================")
        return

    first_video = video_files[0]
    first_frame_path = os.path.join(video_directory, "first_frame.jpg")
    save_first_frame(os.path.join(video_directory, first_video), first_frame_path)

    coordinates = get_box_coordinates(first_frame_path)
    os.remove(first_frame_path)
    draw_box_on_videos(video_directory, coordinates)


if __name__ == "__main__":
    if len(sys.argv) == 2 and sys.argv[1] == "draw_box":
        run_drawbox()
    else:
        print("Usage:")
        print("  python utils.py draw_box")
