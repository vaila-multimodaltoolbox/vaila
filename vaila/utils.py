"""
utils.py
Version: 2024-07-15 20:00:00
"""

import os
import sys
import pandas as pd
from ffmpeg import FFmpeg
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import tkinter as tk
from tkinter import filedialog
from common_utils import determine_header_lines, headersidx, reshapedata


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
        output_path = os.path.join(
            output_directory, f"{os.path.splitext(video_file)[0]}_boxe.mp4"
        )

        print(f"Processing {video_file}...")

        ffmpeg = (
            FFmpeg()
            .input(input_path)
            .output(
                output_path,
                vf=f"drawbox=0:0:in_w:{YT}:color=black:t=fill,"
                f"drawbox=0:{YT}:{XT}:in_h:color=black:t=fill,"
                f"drawbox={XT}:{YD}:in_w:in_h:color=black:t=fill,"
                f"drawbox={XD}:{YT}:in_w:{YD}:color=black:t=fill",
                c="libx265",
                crf=22,
            )
            .global_args("-hide_banner", "-nostats", "-loglevel", "quiet")
        )

        try:
            ffmpeg.execute()
            print(f"Done DRAW BOX EDGES -------> {video_file}")
        except Exception as e:
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
