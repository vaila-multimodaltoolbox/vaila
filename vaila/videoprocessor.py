"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

videoprocessor.py

Description:
This script allows users to process and edit video files, enabling batch processing of videos. Users can choose between two main operations:
1. Merging a video with its reversed version, resulting in a video with double the frames.
2. Splitting each video into two halves and saving only the second half.
The script supports custom text files for batch processing and includes a GUI for directory and file selection.

Key Features:
- Graphical User Interface (GUI) for easy selection of directories and file inputs.
- Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for specifying which videos to process.
- If no text file is provided, the script processes all videos in the source directory.
- Merge option: Creates a video with the original and its reversed version merged.
- Split option: Processes each video to save only the second half.
- Automatic creation of output directories based on a timestamp for organized file management.
- Detailed console output for tracking progress and handling errors.

Usage:
- Run the script to open a graphical interface. After selecting the source and target directories, choose between merging or splitting the videos.
- The processed videos will be saved in a new output directory named with a timestamp.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- Python 3.x environment.
- Tkinter for the GUI components (usually included with Python).

Installation of FFmpeg (for video processing):
- **Conda (recommended)**: 
  ```bash
  conda install -c conda-forge ffmpeg
  ```

"""

import os
import time
import subprocess
from tkinter import filedialog, messagebox, simpledialog

def process_videos_merge(source_dir, target_dir, use_text_file=False, text_file_path=None):
    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"mergedvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    video_files = []

    # Use provided text file if specified
    if use_text_file and text_file_path:
        with open(text_file_path, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    video_files.append(os.path.join(source_dir, line.strip()))
    else:
        # No text file provided, process all videos in source_dir
        with os.scandir(source_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_files.append(entry.path)

    # Iterate over video files and apply the merge process
    for video_path in video_files:
        try:
            print(f"Processing video: {video_path}")

            # Output video path
            output_video = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_double.mp4")

            # Command to reverse the video and concatenate with the original
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", "reverse",
                "-c:v", "libx264",
                "-preset", "fast",
                "-f", "mpegts",
                "pipe:1"
            ]
            reverse_process = subprocess.Popen(ffmpeg_command, stdout=subprocess.PIPE)

            ffmpeg_concat_command = [
                "ffmpeg",
                "-i", video_path,
                "-f", "mpegts",
                "-i", "pipe:0",
                "-filter_complex", "[1:v][0:v]concat=n=2:v=1:a=0",
                "-c:v", "libx264",
                "-preset", "fast",
                output_video
            ]
            subprocess.run(ffmpeg_concat_command, stdin=reverse_process.stdout, check=True)

            print(f"Video processed and saved to: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error processing video {video_path}: {e}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

def process_videos_split(source_dir, target_dir, use_text_file=False, text_file_path=None):
    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"splitvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    video_files = []

    # Use provided text file if specified
    if use_text_file and text_file_path:
        with open(text_file_path, "r") as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    video_files.append(os.path.join(source_dir, line.strip()))
    else:
        # No text file provided, process all videos in source_dir
        with os.scandir(source_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith((".mp4", ".avi", ".mov", ".mkv")):
                    video_files.append(entry.path)

    # Iterate over video files and apply the split process
    for video_path in video_files:
        try:
            print(f"Processing video: {video_path}")

            # Get total number of frames using ffprobe
            ffprobe_frames_command = [
                "ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames", "-show_entries",
                "stream=nb_read_frames", "-of", "default=nokey=1:noprint_wrappers=1", video_path
            ]
            frames_result = subprocess.run(ffprobe_frames_command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            total_frames = int(frames_result.stdout.strip())
            half_frame = total_frames // 2

            # Output video path
            output_video = os.path.join(output_dir, f"{os.path.splitext(os.path.basename(video_path))[0]}_2half.mp4")

            # Command to extract the second half of the video by frames
            ffmpeg_command = [
                "ffmpeg",
                "-i", video_path,
                "-vf", f"select='gte(n,{half_frame})'",  # Select frames starting from the half
                "-vsync", "vfr",
                "-c:v", "libx264",
                output_video
            ]
            subprocess.run(ffmpeg_command, check=True)

            print(f"Video processed and saved to: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error processing video {video_path}: {e}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")

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

    # Ask user to select the operation (Merge or Split)
    operation = simpledialog.askstring("Operation", "Enter 'm' for merge or 's' for split:").strip().lower()
    if not operation or operation not in ["m", "s"]:
        messagebox.showerror("Error", "Invalid operation selected. Please enter 'm' for merge or 's' for split.")
        return

    # Ask if the user wants to use a text file
    use_text_file = messagebox.askyesno(
        "Use Text File", "Do you want to use a text file (videos_e_frames.txt) to specify videos to process?"
    )
    text_file_path = None
    if use_text_file:
        text_file_path = filedialog.askopenfilename(
            title="Select videos_e_frames.txt", filetypes=[("Text files", "*.txt")]
        )
        if not text_file_path:
            messagebox.showerror("Error", "No text file selected.")
            return

    # Call the appropriate function based on the selected operation
    if operation == 'm':
        process_videos_merge(source_dir, target_dir, use_text_file, text_file_path)
    elif operation == 's':
        process_videos_split(source_dir, target_dir, use_text_file, text_file_path)

if __name__ == "__main__":
    process_videos_gui()
