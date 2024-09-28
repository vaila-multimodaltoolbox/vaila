"""
Module: videoprocessor.py

Description:
This module provides a user-friendly interface for processing and editing video files. The primary functionality allows users to concatenate a portion of one video (`video_A`) onto another (`video_B`). Users can specify the percentage of `video_A` to be used, with a default of 50% if unspecified. The script supports batch processing through directory selection and can utilize a text file (`videos_e_frames.txt`) for customized processing instructions. Additional features include reversing the concatenated video segment and handling various video formats.

Key Features:
- Graphical User Interface (GUI) for easy directory and file selection.
- Customizable percentage of `video_A` to concatenate onto `video_B`.
- Option to reverse the concatenated segment of `video_A`.
- Support for batch processing multiple video files.
- Ability to use a text file for detailed processing instructions.
- Automatic creation of timestamped output directories for organized file management.
- Console output for progress tracking and error handling.

Inputs:
- `process_videos(source_dir, target_dir, percentage=50, reverse=False, use_text_file=False, text_file_path=None)`:
  - `source_dir` (str): Path to the directory containing source videos (`video_A`).
  - `target_dir` (str): Path to the directory where processed videos will be saved.
  - `percentage` (float, optional): Percentage of `video_A` to concatenate. Defaults to 50%.
  - `reverse` (bool, optional): If `True`, reverses the concatenated segment of `video_A`. Defaults to `False`.
  - `use_text_file` (bool, optional): If `True`, uses a text file for processing instructions. Defaults to `False`.
  - `text_file_path` (str, optional): Path to the text file containing custom instructions.

- `process_videos_gui()`:
  - Initiates the GUI for interactive directory selection and parameter input.

Outputs:
- Processed video files saved in a new directory within `target_dir`, named with a timestamp (e.g., `mergedvid_20240816123045`).
- Filenames indicate the number of frames inserted and whether the segment was reversed (e.g., `video_name_frames_150.mp4` or `video_name_rframes_150.mp4`).
- Console messages detailing the processing status and any errors encountered.

Usage:
- Run the script directly to launch the GUI and follow on-screen prompts for processing videos.
- Use the `process_videos()` function in other scripts for automated processing with predefined parameters.

Example:

if __name__ == "__main__":
    process_videos_gui()

Example Workflow:

User executes videoprocessor.py.
GUI prompts appear for selecting the source and target directories.
User is asked whether to use a text file (videos_e_frames.txt) for processing instructions.
If yes, the user selects the text file, which should contain lines formatted as source_video, percentage, target_video.
User inputs the desired percentage of video_A to concatenate and specifies if the segment should be reversed.
The script processes each video according to the provided inputs and saves them in the output directory.
Progress and any errors are displayed in the console.
Dependencies:

Python 3.x
moviepy library for video editing (pip install moviepy)
Tkinter for GUI components (usually included with Python)
Standard Python libraries: os, time
Author:

Prof. Paulo Santiago
Version:

1.4
Date:

August 16, 2024
Changelog:

Version 1.4 (August 16, 2024):

Added the option to reverse the concatenated video segment.
Enhanced GUI prompts for better user experience.
Implemented support for processing instructions via a text file.
Improved error handling and console messaging.
References:

MoviePy Documentation: https://zulko.github.io/moviepy/
Tkinter Documentation: https://docs.python.org/3/library/tkinter.html
Python OS Module: https://docs.python.org/3/library/os.html
Python Time Module: https://docs.python.org/3/library/time.html
License:

This script is released under the MIT License.
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

