"""
Module: extractpng.py

Description:
This module offers functionality to either extract PNG frames from video files or create videos from a sequence of PNG images. It is designed for applications in machine learning, computer vision, and biomechanics, ensuring that all frames and videos maintain consistent quality and formatting.

The main features of the module include:
- `extract_png_from_videos`: Extracts PNG frames from videos in RGB format to ensure compatibility with various machine learning models and image processing tasks. The function processes all video files in the specified source directory, creates a dedicated output directory for each video, and stores the frames using a customizable naming pattern.
- `create_video_from_png`: Converts a sequence of PNG images into a video using the YUV420p color space and either the libx264 codec (CPU) or the h264_nvenc codec (GPU), depending on the availability of an NVIDIA GPU. This ensures efficient compression while leveraging hardware acceleration when available.

Key Features:
1. **Frame Extraction in RGB**: Extracts PNG frames in RGB format to ensure compatibility with image-based models in machine learning and computer vision.
2. **GPU-Accelerated Video Creation**: Automatically detects if an NVIDIA GPU is available and uses `h264_nvenc` for video compression. Falls back to CPU-based encoding (`libx264`) if no GPU is detected.
3. **Customizable Naming Pattern**: Users can define their own filename pattern for the output PNGs, enabling flexibility in frame numbering and organization.
4. **Automated Directory Management**: The script creates timestamped directories for organizing output, ensuring a clean and structured workflow.
5. **Comprehensive Video Support**: Works with common video formats like `.mp4`, `.avi`, `.mov`, and `.mkv`, ensuring broad compatibility across different media types.

Usage:
- `extract_png_from_videos`: Processes all video files in the specified directory and extracts PNG frames into organized subdirectories. The function also saves essential metadata like frame rate (FPS) for future reference.
- `create_video_from_png`: Converts a sequence of PNG frames into a video, with options for selecting the compression codec and frame rate. Automatically uses GPU acceleration if available.

Dependencies:
- Python 3.x
- ffmpeg (installed via Conda or available in PATH)
- Tkinter (for file and directory dialogs)

Example usage:

from extractpng import VideoProcessor

# Create an instance of the VideoProcessor class
processor = VideoProcessor()

# Extract PNG frames from videos
processor.extract_png_from_videos()

# Create a video from PNG frames (uses GPU if available)
processor.create_video_from_png()

Version: 2.1 Last Updated: September 29, 2024 Author: Prof. Paulo Santiago

Changelog:

Version 2.1 (2024-09-29): Added NVIDIA GPU detection for hardware-accelerated video creation using h264_nvenc. Fallback to libx264 for CPU encoding if no GPU is found.
Version 2.0 (2024-08-25): Added customizable PNG filename pattern and video codec selection. Improved error handling and user interface for batch processing multiple videos and directories.
Version 1.0 (2023-12-15): Initial release with basic functionality for extracting PNG frames from videos and creating videos from PNG sequences.

References:

FFmpeg Documentation: https://ffmpeg.org/documentation.html
"""

import os
import subprocess
import time
from tkinter import filedialog, messagebox, simpledialog, Tk, Toplevel, Label, Button
from rich import print
import shutil


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

                fps = self.get_fps(video_path)

                command = [
                    "ffmpeg",
                    "-i",
                    video_path,
                    "-vf",
                    "scale=in_range=pc:out_range=pc,format=rgb24",
                    "-q:v",
                    "1",
                    output_pattern,
                ]
                subprocess.run(command, check=True)

                with open(os.path.join(output_dir, "info.txt"), "w") as f:
                    f.write(f"FPS: {fps}\n")

                print(f"Extraction completed for {item}")

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

        timestamp = time.strftime("%Y%m%d%H%M%S")
        dest_main_dir = os.path.join(src, f"vaila_png2videos_{timestamp}")
        os.makedirs(dest_main_dir, exist_ok=True)

        try:
            for subdir, _, files in os.walk(src):
                png_files = sorted([f for f in files if f.endswith(".png")])

                if png_files:
                    output_video_name = os.path.basename(subdir) + ".mp4"
                    output_video_path = os.path.join(dest_main_dir, output_video_name)

                    # Modified FFmpeg command to use image sequence directly
                    command = [
                        "ffmpeg",
                        "-framerate",
                        str(fps),
                        "-i",
                        os.path.join(subdir, "%09d.png"),  # Use 9-digit pattern
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        "-pix_fmt",
                        "yuv420p",
                        output_video_path,
                    ]
                    subprocess.run(command, check=True)
                    print(f"Video created: {output_video_path}")

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
        import cv2

        cap = cv2.VideoCapture(video_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        cap.release()
        return fps

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
