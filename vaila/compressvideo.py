"""
# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# compress_videos.py
# This script compresses videos in a specified directory to either H.264 or H.265/HEVC format
# using the FFmpeg tool. It provides a GUI for selecting the directory containing the videos,
# allows the user to choose the desired codec, and then processes each video, saving the
# compressed versions in a subdirectory named 'compressed_[codec]'.
#
# Usage:
# Run the script to open the GUI, select the directory containing videos, choose the codec,
# and the compression process will start automatically.
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
# This process may take several hours depending on the size of the videos
# and the performance of your computer.
"""

import os
import subprocess
import warnings
from tkinter import (
    Button,
    Label,
    Radiobutton,
    StringVar,
    Tk,
    Toplevel,
    W,
    filedialog,
    messagebox,
)

from rich import print
from rich import print as rprint

# Deprecation notice
_DEPRECATION_MSG = (
    "compressvideo.py is deprecated. "
    "Use compress_videos_h264.py or compress_videos_h265.py instead, "
    "which support GPU acceleration, resolution selection, and CLI mode."
)


def check_ffmpeg_encoder(encoder):
    try:
        command = [
            "ffmpeg",
            "-f",
            "lavfi",  # use lavfi format for dummy input
            "-i",
            "nullsrc=s=64x64:d=1",  # generate dummy input
            "-vcodec",
            encoder,
            "-f",
            "null",
            "-hide_banner",
            "-nostats",
            "-",
        ]
        subprocess.run(command, check=True, capture_output=True)
        return True
    except subprocess.CalledProcessError as e:
        if "Unknown encoder" in e.stderr.decode():
            return False
        else:
            raise


def run_compress_videos(video_directory, codec, preset="medium", crf=23):
    output_directory = os.path.join(video_directory, "compressed_" + codec)
    os.makedirs(output_directory, exist_ok=True)

    print("!!!ATTENTION!!!")
    print(
        "This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!"
    )

    if not check_ffmpeg_encoder(codec):
        print(f"Error: Your ffmpeg installation does not support the {codec} encoder.")
        return

    for video_file in sorted(
        [
            f
            for f in os.listdir(video_directory)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    ):
        input_path = os.path.join(video_directory, video_file)
        output_path = os.path.join(
            output_directory, f"{os.path.splitext(video_file)[0]}_{codec}.mp4"
        )

        print(f"Compressing {video_file}...")

        command = [
            "ffmpeg",
            "-y",  # overwrite output files
            "-i",
            input_path,  # input file
            "-vcodec",
            codec,  # video codec
            "-preset",
            preset,  # preset for encoding speed
            "-crf",
            str(crf),  # constant rate factor for quality
            "-hide_banner",
            "-nostats",
            output_path,
        ]

        try:
            subprocess.run(command, check=True)
            print(f"Done compressing {video_file} to {codec}.")
        except subprocess.CalledProcessError as e:
            print(f"Error compressing {video_file}: {e}")


def ask_codec_selection(root):
    codecs = [("H264", "libx264"), ("H265/HEVC", "libx265")]

    codec_var = StringVar(value="libx264")
    selection_dialog = Toplevel(root)
    selection_dialog.title("Select Codec")

    Label(selection_dialog, text="Select the codec to use:").pack(pady=10)

    for text, codec in codecs:
        Radiobutton(selection_dialog, text=text, variable=codec_var, value=codec).pack(
            anchor=W, padx=20
        )

    def on_ok():
        selection_dialog.destroy()

    Button(selection_dialog, text="OK", command=on_ok).pack(pady=10)

    root.wait_window(selection_dialog)
    return codec_var.get()


def compress_videos_gui():
    warnings.warn(_DEPRECATION_MSG, DeprecationWarning, stacklevel=2)
    rprint(f"[yellow][WARNING] {_DEPRECATION_MSG}[/yellow]")
    # Create the main window# Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    root = Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    codec = ask_codec_selection(root)
    if not codec:
        messagebox.showerror("Error", "No codec selected.")
        return

    preset = "medium"  # You can add GUI components to select these values if needed
    crf = 23  # You can add GUI components to select these values if needed

    run_compress_videos(video_directory, codec, preset, crf)
    messagebox.showinfo("Success", "Video compression completed.")


if __name__ == "__main__":
    compress_videos_gui()
