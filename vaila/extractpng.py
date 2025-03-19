"""
================================================================================
Extract PNG Tool - extractpng.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Laboratory of Biomechanics and Motor Control (LaBioCoM)
School of Physical Education and Sport of Ribeirão Preto
University of São Paulo (USP)

Contact: paulosantiago@usp.br
Laboratory website: https://github.com/vaila-multimodaltoolbox/vaila

Created: December 15, 2023
Last Updated: January 27, 2025
Version: 2.1.0

Description:
------------
This module offers functionality to either extract PNG frames from video files or
create videos from a sequence of PNG images. It is designed for applications in
machine learning, computer vision, and biomechanics, ensuring that all frames and
videos maintain consistent quality and formatting.

The main features of the module include:
- Extract PNG frames from videos in RGB format
- Create videos from PNG sequences
- GPU-accelerated video processing when available
- Customizable frame naming patterns
- Comprehensive video format support

Dependencies:
------------
- Python 3.x
- ffmpeg (installed via Conda or available in PATH)
- OpenCV (cv2)
- Tkinter
- NumPy

References:
-----------
FFmpeg Documentation: https://ffmpeg.org/documentation.html

Changelog:
---------
Version 2.1.0 (2025-01-27):
- Added hardware acceleration support
- Improved PNG extraction compatibility
- Added detailed video information logging

Version 2.0.0 (2024-08-25):
- Added customizable PNG filename pattern
- Improved error handling
- Added batch processing support

Version 1.0.0 (2023-12-15):
- Initial release
================================================================================
"""

import os
import subprocess
import time
from tkinter import filedialog, messagebox, simpledialog, Tk, Toplevel, Label, Button
from rich import print
import shutil
import numpy as np
import cv2


class VideoProcessor:
    def __init__(self):
        self.pattern = "%09d.png"  # Default pattern

    def extract_png_from_videos(self):
        print("Starting extraction of PNG frames from videos...")

        root = Tk()
        root.withdraw()

        src = filedialog.askdirectory(
            title="Select the source directory containing videos"
        )
        if not src:
            messagebox.showerror("Error", "No source directory selected.")
            return

        pattern = simpledialog.askstring(
            "PNG Filename Pattern",
            "Enter the filename pattern for PNG files (e.g., frame%07d.png):\nLeave empty for default (%09d.png):",
        )
        if pattern:
            self.pattern = pattern

        timestamp = time.strftime("%Y%m%d%H%M%S")
        dest_main_dir = os.path.join(src, f"vaila_extractpng_{timestamp}")
        os.makedirs(dest_main_dir, exist_ok=True)

        try:
            video_files = [
                f
                for f in os.listdir(src)
                if f.endswith(
                    (".avi", ".mp4", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
                )
            ]

            for item in video_files:
                video_path = os.path.join(src, item)
                video_name = os.path.splitext(item)[0]
                output_dir = os.path.join(dest_main_dir, f"{video_name}_png")
                os.makedirs(output_dir, exist_ok=True)
                output_pattern = os.path.join(output_dir, self.pattern)

                # Get video dimensions and FPS
                width, height, fps = self.get_video_info(video_path)

                # Updated command for better HEVC decoding and PNG compatibility
                command = [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vf",
                    f"scale={width}:{height}:flags=lanczos",  # Removed format=rgb24
                    "-q:v",
                    "1",
                    "-fps_mode",
                    "passthrough",
                    "-hwaccel",
                    "auto",
                    "-c:v",
                    "hevc_cuvid",
                    "-drop_second_field",
                    "1",
                    "-sws_flags",
                    "bicubic",
                    "-pix_fmt",
                    "rgb24",  # Kept for correct color
                    "-f",
                    "image2",  # Forces image format
                    "-compression_level",
                    "6",  # PNG compression level (0-9)
                    output_pattern,
                ]

                try:
                    # Try first with hardware acceleration
                    try:
                        print(f"\nProcessing {item} with hardware acceleration...")
                        subprocess.run(command, check=True)

                    except subprocess.CalledProcessError:
                        print(
                            "\nHardware acceleration failed, trying software decoder..."
                        )

                        # Remove hardware acceleration to try software decoder
                        command = [
                            "ffmpeg",
                            "-i",
                            video_path,
                            "-vf",
                            f"scale={width}:{height}:flags=lanczos",
                            "-q:v",
                            "1",
                            "-fps_mode",
                            "passthrough",
                            "-sws_flags",
                            "bicubic",
                            "-pix_fmt",
                            "rgb24",
                            "-f",
                            "image2",
                            "-compression_level",
                            "6",
                            output_pattern,
                        ]

                        subprocess.run(command, check=True)

                    print(f"\n\nChecking frames in {output_dir}...")
                    total_frames = len(
                        [f for f in os.listdir(output_dir) if f.endswith(".png")]
                    )
                    print(f"Total frames extracted: {total_frames}")

                    # Save basic video information
                    with open(os.path.join(output_dir, "video_info.txt"), "w") as f:
                        f.write(f"Original video: {item}\n")
                        f.write(f"FPS: {fps}\n")
                        f.write(f"Resolution: {width}x{height}\n")
                        f.write(f"Total frames: {total_frames}\n")
                        f.write(f"Extraction timestamp: {timestamp}\n")

                    print(f"Successfully extracted frames from {item}")
                    print(f"Resolution: {width}x{height}, FPS: {fps}")

                except Exception as e:
                    print(f"Error processing {item}: {str(e)}")
                    raise

            self.show_completion_message("PNG extraction completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting PNG frames: {e}")

    def extract_select_frames_from_video(self):
        print("Starting extraction of specific frames from video...")

        root = Tk()
        root.withdraw()

        video_file = filedialog.askopenfilename(
            title="Select the video file",
            filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")],
        )
        if not video_file:
            messagebox.showerror("Error", "No video file selected.")
            return

        frame_numbers = simpledialog.askstring(
            "Frame Numbers", "Enter the frame numbers to extract (e.g., 0,3,5,7,9):"
        )
        if not frame_numbers:
            messagebox.showerror("Error", "No frame numbers provided.")
            return

        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(
            os.path.dirname(video_file), f"vaila_grabframes_{timestamp}"
        )
        os.makedirs(output_dir, exist_ok=True)

        frames = frame_numbers.replace(" ", "").split(",")
        try:
            for frame in frames:
                frame_number = int(frame)
                output_path = os.path.join(output_dir, f"frame_{frame_number:03d}.png")

                command = [
                    "ffmpeg",
                    "-i",
                    video_file,
                    "-vf",
                    f"select=eq(n\\,{frame_number})",
                    "-vframes",
                    "1",
                    output_path,
                    "-loglevel",
                    "quiet",
                    "-nostats",
                    "-hide_banner",
                ]
                subprocess.run(command, check=True)
                print(f"Extracted frame {frame_number} to {output_path}")

            self.show_completion_message("Frame extraction completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting frames: {e}")

    def get_fps_from_user(self):
        """Prompt user to enter desired FPS for output video."""
        root = Tk()
        root.withdraw()

        fps = simpledialog.askfloat(
            "FPS Configuration",
            "Enter the desired FPS for the output video:",
            initialvalue=30.0,
            minvalue=1.0,
            maxvalue=240.0,
        )

        return fps if fps is not None else 30.0

    def create_video_from_png(self):
        print("Starting creation of videos from PNG sequences...")

        root = Tk()
        root.withdraw()

        src = filedialog.askdirectory(
            title="Select the main directory containing PNG subdirectories"
        )
        if not src:
            messagebox.showerror("Error", "No source directory selected.")
            return

        # Get FPS from user
        fps = self.get_fps_from_user()

        # Get codec choice from user
        codec_choice = simpledialog.askstring(
            "Codec Selection",
            "Choose codec (type '264' for H.264 or '265' for H.265):",
            initialvalue="264",
        )

        # Set codec and preset based on user choice
        if codec_choice == "265":
            codec = "libx265"
            preset = "medium"
            # Add specific H.265 parameters
            extra_params = ["-x265-params", "log-level=error"]
        else:  # default to H.264
            codec = "libx264"
            preset = "medium"
            extra_params = []

        timestamp = time.strftime("%Y%m%d%H%M%S")
        dest_main_dir = os.path.join(src, f"vaila_png2videos_{timestamp}")
        os.makedirs(dest_main_dir, exist_ok=True)

        try:
            for subdir, _, files in os.walk(src):
                png_files = sorted([f for f in files if f.endswith(".png")])

                if png_files:
                    output_video_name = os.path.basename(subdir) + ".mp4"
                    output_video_path = os.path.join(dest_main_dir, output_video_name)

                    # Base command
                    command = [
                        "ffmpeg",
                        "-framerate",
                        str(fps),
                        "-i",
                        os.path.join(subdir, "%09d.png"),
                        "-c:v",
                        codec,
                        "-preset",
                        preset,
                        "-pix_fmt",
                        "yuv420p",
                    ]

                    # Add extra parameters if any
                    if extra_params:
                        command.extend(extra_params)

                    # Add output path
                    command.append(output_video_path)

                    subprocess.run(command, check=True)
                    print(f"Video created: {output_video_path}")
                    print(f"Using codec: {codec}")

            self.show_completion_message("Video creation completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating videos: {e}")

    def is_pattern_consistent(self, files):
        """
        Check if the files in the directory follow the expected numeric pattern.
        """
        print("Checking file naming pattern...")
        try:
            # Expected file names based on the default pattern
            expected_files = [self.pattern % i for i in range(len(files))]
            actual_files = sorted(files)
            return actual_files == expected_files
        except ValueError:
            # Return False if file names do not follow the numeric pattern
            return False

    def is_nvidia_gpu_available(self):
        try:
            result = subprocess.run(
                ["nvidia-smi"], stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
            return result.returncode == 0
        except FileNotFoundError:
            return False

    def get_fps(self, video_path):
        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

    def get_video_info(self, video_path):
        """Get video dimensions and FPS using ffprobe."""
        try:
            width = int(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                )
                .decode()
                .strip()
            )

            height = int(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=height",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                )
                .decode()
                .strip()
            )

            fps = float(
                subprocess.check_output(
                    [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=r_frame_rate",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                )
                .decode()
                .strip()
                .split("/")[0]
            )

            return width, height, fps

        except subprocess.CalledProcessError as e:
            print(f"Error getting video info: {e.stderr}")
            raise
        except Exception as e:
            print(f"Unexpected error getting video info: {str(e)}")
            raise

    def show_completion_message(self, message):
        root = Tk()
        root.withdraw()

        confirmation_window = Toplevel(root)
        confirmation_window.title("Process Completed")

        label = Label(confirmation_window, text=message, padx=20, pady=20)
        label.pack()

        ok_button = Button(
            confirmation_window,
            text="OK",
            command=confirmation_window.destroy,
            padx=10,
            pady=5,
        )
        ok_button.pack()

        confirmation_window.grab_set()
        confirmation_window.mainloop()

    def run(self):
        # Print the directory and name of the script being executed
        print(f"Running script: {os.path.basename(__file__)}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print("Starting vailá video processing...")

        root = Tk()
        root.withdraw()
        choice = simpledialog.askstring(
            "Choose Action",
            "Type 'e' to extract PNGs, 'c' to create a video from PNGs, or 'f' to extract specific frames from video:",
        )

        if choice == "e":
            self.extract_png_from_videos()
        elif choice == "c":
            self.create_video_from_png()
        elif choice == "f":
            self.extract_select_frames_from_video()
        else:
            messagebox.showerror("Error", "Invalid choice.")


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()
