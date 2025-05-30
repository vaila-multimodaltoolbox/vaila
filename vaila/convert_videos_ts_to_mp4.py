"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

convert_videos_ts_to_mp4.py

Description:
    This script converts all .TS video files in a selected directory to .mp4 format
    and removes the audio track using FFmpeg.
    It provides a GUI for selecting the directory containing the .ts videos and
    saves the converted files in a subdirectory named 'converted_mp4'.

Usage:
    - Run the script to open a GUI, select the directory containing the .ts videos,
      and the conversion process will start automatically.

Requirements:
    - FFmpeg installed and accessible in the system PATH.
    - Python 3.x
    - Tkinter (included with Python)
"""

import os
import subprocess
from datetime import datetime
import tkinter as tk
from tkinter import filedialog, messagebox
from rich import print
from rich.progress import (
    Progress,
    SpinnerColumn,
    TextColumn,
    BarColumn,
    TimeElapsedColumn,
)
from pathlib import Path

# Global counters
success_count = 0
failure_count = 0


def find_ts_videos(directory: str) -> list:
    """Return a list of all .ts files in the given directory."""
    return [
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith(".ts")
    ]


def run_convert_ts_to_mp4(input_dir: str, output_dir: str):
    """Convert each .ts file in input_dir to .mp4 without audio, saving into output_dir."""
    global success_count, failure_count
    success_count = failure_count = 0

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    ts_files = find_ts_videos(input_dir)

    if not ts_files:
        print(f"[red]No .ts files found in {input_dir}[/red]")
        return

    # Setup progress bar
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        BarColumn(),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        TimeElapsedColumn(),
    ) as progress:
        task = progress.add_task("[cyan]Converting videos...", total=len(ts_files))

        for ts_path in ts_files:
            base = os.path.splitext(os.path.basename(ts_path))[0]
            mp4_path = os.path.join(output_dir, f"{base}.mp4")

            # Skip if output file already exists
            if os.path.exists(mp4_path):
                print(f"[yellow]Skipping existing file:[/yellow] {base}.mp4")
                progress.advance(task)
                continue

            cmd = [
                "ffmpeg",
                "-y",
                "-i",
                ts_path,
                "-c:v",
                "copy",  # Copy video stream without re-encoding
                "-an",  # Remove audio
                "-movflags",
                "+faststart",  # Enable fast start for web playback
                mp4_path,
            ]

            try:
                print(
                    f"[cyan]Converting:[/cyan] {os.path.basename(ts_path)} → {base}.mp4"
                )
                subprocess.run(
                    cmd, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
                )
                success_count += 1
            except subprocess.CalledProcessError as e:
                print(f"[red]Failed:[/red] {os.path.basename(ts_path)}")
                print(f"[yellow]Error:[/yellow] {e.stderr.decode().strip()}")
                failure_count += 1
            except Exception as e:
                print(f"[red]Unexpected error:[/red] {str(e)}")
                failure_count += 1

            progress.advance(task)


def convert_ts_to_mp4_gui():
    """Main GUI for directory selection and conversion process."""
    root = tk.Tk()
    root.withdraw()  # hide main window

    messagebox.showinfo(
        "Select Folder", "Select the directory containing .ts videos to convert."
    )
    input_dir = filedialog.askdirectory(title="Select .ts Videos Directory")
    if not input_dir:
        messagebox.showwarning("No Selection", "No directory selected, exiting.")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(input_dir, f"converted_mp4_{timestamp}")

    run_convert_ts_to_mp4(input_dir, output_dir)

    # Show completion message
    messagebox.showinfo(
        "Conversion Complete",
        f"Finished converting videos.\n\n"
        f"Succeeded: {success_count}\n"
        f"Failed: {failure_count}\n\n"
        f"Converted files are in:\n{output_dir}",
    )


if __name__ == "__main__":
    convert_ts_to_mp4_gui()
