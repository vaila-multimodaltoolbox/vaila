"""
File: videoprocessor.py

Description:
This script is designed to process videos by allowing the user to concatenate a portion of one video onto another. The user can define the percentage of the first video (`video_A`) that will be concatenated to the second video (`video_B`). If no percentage is provided, the default is 50%. The script supports selecting source and target directories through a graphical user interface (GUI) and can optionally use a provided text file (`videos_e_frames.txt`) for processing. Additionally, there is an option to reverse the concatenated video portion.

Version: 1.4
Last Updated: August 16, 2024
Author: Prof. Paulo Santiago

Dependencies:
- Python 3.x
- moviepy
- Tkinter
"""

import os
import time
from moviepy.editor import VideoFileClip, concatenate_videoclips, vfx
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog


def process_videos(
    source_dir, target_dir, percentage=50, reverse=False, use_text_file=False, text_file_path=None
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
                if ',' in line:
                    source_video, percent, target_video = line.split(',')
                else:
                    source_video, percent, target_video = line.split()
                
                percent = float(percent)
                video_files.append((os.path.join(source_dir, source_video), percent, os.path.join(output_dir, target_video)))
    else:
        video_files = []
        for file in os.listdir(source_dir):
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                video_files.append((os.path.join(source_dir, file), percentage, os.path.join(output_dir, file)))

    for source_video, percent, target_video in video_files:
        try:
            video_original = VideoFileClip(source_video)
            video_A_duration = video_original.duration * (percent / 100)
            video_A = video_original.subclip(0, video_A_duration)
            if reverse:
                video_A = video_A.fx(vfx.time_mirror)  # Reverse the video segment
            video_B = VideoFileClip(target_video) if os.path.exists(target_video) else video_original

            video_final = concatenate_videoclips([video_A, video_B])

            # Save the concatenated video
            base_name = os.path.splitext(os.path.basename(target_video))[0]
            inserted_frames = int(video_A.fps * video_A.duration)  # Number of frames inserted
            output_suffix = f"rframes_{inserted_frames}" if reverse else f"frames_{inserted_frames}"
            output_file = os.path.join(
                output_dir, f"{base_name}_{output_suffix}.mp4"
            )
            video_final.write_videofile(
                output_file, fps=video_final.fps, codec="libx264", bitrate="8000k"
            )

            print(f"Video processed and saved to: {output_file}")
        except Exception as e:
            print(f"Error processing video {source_video} into {target_video}: {e}")


def process_videos_gui():
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
    process_videos(source_dir, target_dir, percentage, reverse, use_text_file, text_file_path)


if __name__ == "__main__":
    process_videos_gui()
