"""
compress_videos_h266.py

Created by Paulo Santiago
Date: 06 September 2025
Update: 06 September 2025
Version: 0.0.1
Python Version: 3.12.11

Description:
This script compresses videos in a specified directory to H.266/VVC format using the FFmpeg tool
and the libvvenc encoder. It provides a GUI for selecting the directory and compression settings.
The compressed versions are saved in a subdirectory named 'compressed_h266'.


vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Description:
This script compresses videos in a specified directory to H.266/VVC format using the FFmpeg tool
and the libvvenc encoder. It provides a GUI for selecting the directory and compression settings.
The compressed versions are saved in a subdirectory named 'compressed_h266'.

!!! IMPORTANT !!!
- H.266/VVC encoding is EXTREMELY SLOW and CPU-intensive. Expect long processing times.
- GPU acceleration (like NVIDIA NVENC) is NOT available for VVC in common FFmpeg builds.
  This script uses CPU-only encoding.

Usage:
- Run the script to open a GUI, select the directory containing the videos,
  adjust the settings, and the compression process will start automatically.

Requirements:
- A modern version of FFmpeg (version 7.1+ recommended) that was compiled with libvvenc support.
- The script is designed to work on Windows, Linux, and macOS.

Dependencies:
- Python 3.x
- Tkinter (included with Python)
- FFmpeg with libvvenc (available in PATH)

How to get FFmpeg with libvvenc support:

The version of FFmpeg from standard system repositories (like Ubuntu's apt) usually DOES NOT
include libvvenc. You need to download a pre-compiled "full" or "git" build.

## Windows:
- Download a "full" build from: https://www.gyan.dev/ffmpeg/builds/
- Extract the files and add the `bin` directory to your system's PATH.

## Linux / macOS:
- Download a static "git" build from: https://johnvansickle.com/ffmpeg/
- Extract the `ffmpeg` executable and either place it in a directory in your PATH
  (like /usr/local/bin) or call it directly using its path.

To verify your installation, run this command in your terminal:
  ffmpeg -encoders | findstr vvenc  (Windows)
  ffmpeg -encoders | grep vvenc   (Linux/macOS)

You should see a line containing "libvvenc". If not, your FFmpeg build is not compatible.
"""

import os
import subprocess
import platform
import tempfile
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from rich import print

# Global variables
success_count = 0
failure_count = 0


def find_videos(directory):
    """Find all video files in the specified directory."""
    video_files = []
    for file in os.listdir(directory):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
            video_files.append(os.path.join(directory, file))
    return video_files


def create_temp_file_with_videos(video_files):
    """Create temporary file with list of videos."""
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    for video in video_files:
        temp.write(f"{video}\n")
    temp.close()
    return temp.name


def run_compress_videos_h266(input_list, output_dir, preset, qp, resolution):
    """Run the actual compression using libvvenc."""
    global success_count, failure_count
    success_count = 0
    failure_count = 0

    print("\n[DEBUG] Compression Parameters:")
    print(f"[DEBUG] - Preset: {preset}")
    print(f"[DEBUG] - QP: {qp}")
    print(f"[DEBUG] - Resolution: {resolution}")
    print(f"[DEBUG] - Encoder: libvvenc (CPU-only)")

    print("\n[bold red]!!! ATTENTION !!![/bold red]")
    print(
        "[yellow]H.266/VVC encoding is [u]extremely slow[/u]. This process may take many hours depending on your computer and videos. Please be patient![/yellow]"
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(input_list, "r") as f:
        video_paths = [line.strip() for line in f]

    for video_path in video_paths:
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_h266.mp4"
        )

        try:
            # Base command
            cmd = ["ffmpeg", "-y", "-i", video_path]

            # Add scale filter if resolution is not original
            if resolution != "original":
                scale_filter = f"scale={resolution.replace('x', ':')}:force_original_aspect_ratio=decrease,pad={resolution.replace('x', ':')}:(ow-iw)/2:(oh-ih)/2"
                print(f"[DEBUG] Applying scale filter: {scale_filter}")
                cmd.extend(["-vf", scale_filter])
            else:
                print("[DEBUG] Keeping original resolution (no scale filter)")

            # VVC CPU encoding settings using libvvenc
            # Parameters are passed via -vvenc-params
            vvenc_params = f"preset={preset}:qp={qp}"
            cmd.extend(
                [
                    "-c:v",
                    "libvvenc",
                    "-vvenc-params",
                    vvenc_params,
                ]
            )

            # Add audio settings and output path
            cmd.extend(["-c:a", "copy", output_path])

            print("\n[DEBUG] Complete ffmpeg command:")
            print(" ".join(cmd))

            print(f"\nProcessing: {os.path.basename(video_path)}")
            subprocess.run(cmd, check=True)

            success_count += 1
            print(f"Successfully compressed: {os.path.basename(video_path)}")

        except subprocess.CalledProcessError as e:
            failure_count += 1
            print(f"Failed to compress: {os.path.basename(video_path)}")
            print(f"Error: {str(e)}")
            print(
                f"[DEBUG] ffmpeg error output: {e.stderr if hasattr(e, 'stderr') else 'Not available'}"
            )


def get_compression_parameters():
    """Create a dialog window for user to select compression settings."""
    params = {}
    preset_options = [
        "ultrafast", "superfast", "veryfast", "faster",
        "fast", "medium", "slow", "slower", "veryslow",
    ]
    resolution_options = [
        "original", "3840x2160", "2560x1440", "1920x1080",
        "1280x720", "854x480", "640x360",
    ]

    dialog = tk.Toplevel()
    dialog.title("VVC/H.266 Compression Settings")
    dialog.grab_set()

    main_frame = tk.Frame(dialog, padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)

    tk.Label(
        main_frame, text="H.266/VVC Compression Settings", font=("Arial", 12, "bold")
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

    # 1. Preset
    tk.Label(main_frame, text="Preset (enter number):", font=("Arial", 10, "bold")).grid(
        row=1, column=0, sticky="w", pady=5
    )
    preset_var = tk.StringVar(value="6")  # Default to medium
    preset_entry = tk.Entry(main_frame, textvariable=preset_var, width=5)
    preset_entry.grid(row=1, column=1, sticky="w", pady=5)
    preset_help_text = "Options:\n" + "   ".join([f"{i+1}={p}" for i, p in enumerate(preset_options)])
    tk.Label(main_frame, text=preset_help_text, font=("Arial", 8, "italic"), justify="left", wraplength=400).grid(
        row=2, column=0, columnspan=2, sticky="w", padx=20
    )

    # 2. QP Value (replaces CRF)
    tk.Label(main_frame, text="QP Value (0-51):", font=("Arial", 10, "bold")).grid(
        row=3, column=0, sticky="w", pady=5
    )
    qp_var = tk.StringVar(value="32")  # Default QP for VVC
    qp_entry = tk.Entry(main_frame, textvariable=qp_var, width=5)
    qp_entry.grid(row=3, column=1, sticky="w", pady=5)
    tk.Label(
        main_frame, text="Lower = better quality. 32 is a good default.", font=("Arial", 8, "italic")
    ).grid(row=4, column=0, columnspan=2, sticky="w", padx=20)

    # 3. Resolution
    tk.Label(main_frame, text="Resolution (enter number):", font=("Arial", 10, "bold")).grid(
        row=5, column=0, sticky="w", pady=5
    )
    resolution_var = tk.StringVar(value="1")  # Default to original
    resolution_entry = tk.Entry(main_frame, textvariable=resolution_var, width=5)
    resolution_entry.grid(row=5, column=1, sticky="w", pady=5)
    resolution_help_text = "Options:\n" + "   ".join([f"{i+1}={r}" for i, r in enumerate(resolution_options)])
    tk.Label(main_frame, text=resolution_help_text, font=("Arial", 8, "italic"), justify="left", wraplength=400).grid(
        row=6, column=0, columnspan=2, sticky="w", padx=20
    )

    tk.Frame(main_frame, height=1, bg="gray").grid(row=9, column=0, columnspan=2, sticky="ew", pady=15)
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=10, column=0, columnspan=2, pady=10)

    def on_ok():
        try:
            preset_idx = int(preset_var.get().strip())
            if not (1 <= preset_idx <= len(preset_options)):
                raise ValueError(f"Preset must be between 1 and {len(preset_options)}")
            preset = preset_options[preset_idx - 1]

            qp = int(qp_var.get().strip())
            if not (0 <= qp <= 51):
                raise ValueError("QP must be between 0 and 51")

            resolution_idx = int(resolution_var.get().strip())
            if not (1 <= resolution_idx <= len(resolution_options)):
                raise ValueError(f"Resolution must be between 1 and {len(resolution_options)}")
            resolution = resolution_options[resolution_idx - 1]

            params["preset"] = preset
            params["qp"] = qp
            params["resolution"] = resolution

            confirm_msg = (
                f"Selected compression settings:\n\n"
                f"• Preset: {preset}\n"
                f"• QP: {qp}\n"
                f"• Resolution: {resolution}\n\n"
                f"Remember: H.266 compression is VERY SLOW.\n\n"
                f"Continue?"
            )
            if messagebox.askyesno("Confirm Settings", confirm_msg):
                dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def on_cancel():
        dialog.destroy()

    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=5)
    dialog.wait_window()
    return params if params else None


def compress_videos_h266_gui():
    """Main function to run the GUI and compression process."""
    print(f"Running script: {os.path.basename(__file__)}")
    print("Starting compress_videos_h266_gui...")

    compression_config = get_compression_parameters()
    if not compression_config:
        print("Operation canceled by user.")
        return

    video_directory = filedialog.askdirectory(title="Select the directory containing videos to compress")
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(video_directory, f"compressed_h266_{timestamp}")

    video_files = find_videos(video_directory)
    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    temp_file_path = create_temp_file_with_videos(video_files)

    run_compress_videos_h266(
        temp_file_path,
        output_directory,
        preset=compression_config["preset"],
        qp=compression_config["qp"],
        resolution=compression_config["resolution"],
    )

    os.remove(temp_file_path)

    messagebox.showinfo(
        "Success",
        f"Video compression completed.\n\n{success_count} succeeded, {failure_count} failed.",
    )


if __name__ == "__main__":
    compress_videos_h266_gui()
