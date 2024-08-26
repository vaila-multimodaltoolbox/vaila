"""
File: numberframes.py

Description:
This script allows users to analyze video files within a selected directory and extract metadata such as frame count, frame rate (FPS), resolution, codec, and duration. The script generates a summary of this information, displays it in a user-friendly graphical interface, and saves the metadata to text files. The "basic" file contains essential metadata, while the "full" file includes all possible metadata extracted using `ffprobe`.

Version: 1.2
Last Updated: August 25, 2024
Author: Prof. Paulo Santiago

Dependencies:
- Python 3.x
- OpenCV
- Tkinter
- FFmpeg/FFprobe

Usage:
- Run the script, select a directory containing video files, and let the tool analyze the videos.
- View the metadata in the GUI and check the saved text files in the selected directory for details.
- Use the "full" file for complete metadata in JSON format.
"""

import os
import cv2
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime
import subprocess


def get_video_info(video_path):
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"Error: Cannot open video file {video_path}")
            return None
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        codec = int(cap.get(cv2.CAP_PROP_FOURCC))
        codec_str = f"{chr(codec & 0xFF)}{chr((codec >> 8) & 0xFF)}{chr((codec >> 16) & 0xFF)}{chr((codec >> 24) & 0xFF)}"

        duration = frame_count / fps if fps else 0

        cap.release()

        return {
            "file_name": os.path.basename(video_path),
            "frame_count": frame_count,
            "fps": fps,
            "resolution": f"{width}x{height}",
            "codec": codec_str,
            "duration": duration,
        }
    except Exception as e:
        print(f"Error parsing video info for {video_path}: {str(e)}")
        return None


def count_frames_in_videos():
    root = tk.Tk()
    root.withdraw()

    directory_path = filedialog.askdirectory(
        title="Select the directory containing videos"
    )
    if not directory_path:
        messagebox.showerror("Error", "No directory selected.")
        return

    video_files = sorted(
        [
            f
            for f in os.listdir(directory_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    video_infos = []
    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        video_info = get_video_info(video_path)
        if video_info is not None:
            video_infos.append(video_info)
        else:
            video_infos.append(
                {"file_name": video_file, "error": "Error retrieving video info"}
            )

    output_basic_file = save_basic_metadata_to_file(video_infos, directory_path)
    output_full_file = save_full_metadata_to_file(directory_path, video_files)

    print(f"Basic metadata saved to: {output_basic_file}")
    print(f"Full metadata saved to: {output_full_file}")
    display_video_info(video_infos, output_basic_file)


def display_video_info(video_infos, output_file):
    def on_closing():
        root.destroy()
        show_save_success_message(output_file)

    root = tk.Tk()
    root.title("Video Information")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    canvas = tk.Canvas(frame, width=1080, height=720)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    # Adding headers
    headers = ["Video File", "Frames", "FPS", "Resolution", "Codec", "Duration (s)"]
    for i, header in enumerate(headers):
        ttk.Label(scrollable_frame, text=header, font=("Arial", 10, "bold")).grid(
            row=0, column=i, padx=10, pady=5, sticky=tk.W
        )

    for i, info in enumerate(video_infos, start=1):
        ttk.Label(
            scrollable_frame, text=info["file_name"], font=("Arial", 10, "bold")
        ).grid(row=i, column=0, sticky=tk.W, pady=5)
        if "error" in info:
            ttk.Label(scrollable_frame, text=info["error"]).grid(
                row=i, column=1, columnspan=5, sticky=tk.W, padx=10
            )
        else:
            ttk.Label(scrollable_frame, text=info["frame_count"]).grid(
                row=i, column=1, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=info["fps"]).grid(
                row=i, column=2, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=info["resolution"]).grid(
                row=i, column=3, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=info["codec"]).grid(
                row=i, column=4, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=f"{info['duration']:.2f}").grid(
                row=i, column=5, sticky=tk.W, padx=10
            )

    root.mainloop()


def save_basic_metadata_to_file(video_infos, directory_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(directory_path, f"video_metadata_basic_{timestamp}.txt")

    with open(output_file, "w") as f:
        for info in video_infos:
            if "error" in info:
                f.write(f"File: {info['file_name']}\nError: {info['error']}\n\n")
            else:
                f.write(f"File: {info['file_name']}\n")
                f.write(f"Frames: {info['frame_count']}\n")
                f.write(f"FPS: {info['fps']}\n")
                f.write(f"Resolution: {info['resolution']}\n")
                f.write(f"Codec: {info['codec']}\n")
                f.write(f"Duration (s): {info['duration']:.2f}\n\n")

    return output_file


def save_full_metadata_to_file(directory_path, video_files):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(directory_path, f"metadata_full_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for video_file in video_files:
        video_path = os.path.join(directory_path, video_file)
        json_file = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}.json")
        command = [
            "ffprobe",
            "-v", "quiet",
            "-print_format", "json",
            "-show_format",
            "-show_streams",
            video_path,
        ]
        try:
            with open(json_file, "w") as f:
                subprocess.run(command, stdout=f)
            print(f"Full metadata for {video_file} saved to {json_file}")
        except Exception as e:
            print(f"Error saving full metadata for {video_file}: {str(e)}")

    return output_dir


def show_save_success_message(output_file):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Save Success", f"Metadata successfully saved!\n\nOutput file: {output_file}"
    )


if __name__ == "__main__":
    count_frames_in_videos()
