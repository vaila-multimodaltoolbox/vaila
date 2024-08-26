"""
File: extractpng.py

Description:
This script allows users to either extract PNG frames from video files or create a video from a sequence of PNG images. The script ensures consistency and quality in extracted frames and generated videos, making them suitable for machine learning, computer vision, e biomechanics applications.

Version: 2.0
Last Updated: August 25, 2024
Author: Prof. Paulo Santiago

Features:
- Extract PNG frames in RGB format to ensure compatibility with machine learning models.
- Create videos in YUV420p format, with the option to choose between libx264 or libx265 codecs.
- Automatically creates a directory for saving the output PNGs, named `vaila_extractpng_<timestamp>`.
- Processes all video files in the selected source directory, creating a directory for PNG files extracted from each video.
- Ensures consistent resolution, frame rate, and color space across all operations.

Dependencies:
- Python 3.x
- ffmpeg (installed via Conda or available in PATH)
- Tkinter
"""

import os
import subprocess
import time
from tkinter import filedialog, messagebox, simpledialog, Tk, Toplevel, Label, Button


class VideoProcessor:
    def __init__(self):
        self.pattern = "%09d.png"  # Default pattern

    def extract_png_from_videos(self):
        root = Tk()
        root.withdraw()

        src = filedialog.askdirectory(
            title="Select the source directory containing videos"
        )
        if not src:
            messagebox.showerror("Error", "No source directory selected.")
            return

        # Ask the user for the PNG filename pattern
        pattern = simpledialog.askstring(
            "PNG Filename Pattern",
            "Enter the filename pattern for PNG files (e.g., frame%07d.png):\nLeave empty for default (%09d.png):"
        )
        if pattern:
            self.pattern = pattern

        # Create a new main directory inside the selected destination directory for saving the output PNGs
        timestamp = time.strftime("%Y%m%d%H%M%S")
        dest_main_dir = os.path.join(src, f"vaila_extractpng_{timestamp}")
        os.makedirs(dest_main_dir, exist_ok=True)

        try:
            video_files = [
                f
                for f in os.listdir(src)
                if f.endswith((".avi", ".mp4", ".mov", ".mkv"))
            ]

            for item in video_files:
                video_path = os.path.join(src, item)
                video_name = os.path.splitext(item)[0]
                output_dir = os.path.join(dest_main_dir, f"{video_name}_png")
                os.makedirs(output_dir, exist_ok=True)
                output_pattern = os.path.join(output_dir, self.pattern)

                # Extract FPS using OpenCV
                fps = self.get_fps(video_path)

                # Extract frames with RGB color space
                command = [
                    "ffmpeg",
                    "-i", video_path,
                    "-vf", "scale=in_range=pc:out_range=pc,format=rgb24",
                    "-q:v", "1",
                    output_pattern
                ]
                subprocess.run(command, check=True)

                with open(os.path.join(output_dir, "info.txt"), "w") as f:
                    f.write(f"FPS: {fps}\n")

                print(f"Extraction completed for {item}")

            self.show_completion_message("PNG extraction completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error extracting PNG frames: {e}")

    def create_video_from_png(self):
        root = Tk()
        root.withdraw()

        src = filedialog.askdirectory(
            title="Select the source directory containing PNG files"
        )
        if not src:
            messagebox.showerror("Error", "No source directory selected.")
            return

        # Ask the user for the PNG filename pattern
        pattern = simpledialog.askstring(
            "PNG Filename Pattern",
            "Enter the filename pattern for PNG files (e.g., 'frame%07d.png'):\nLeave empty for default ('%09d.png'):"
        )
        if pattern:
            self.pattern = pattern

        # Ask the user to choose the codec
        codec_choice = simpledialog.askstring(
            "Codec Choice",
            "Enter 'h264' for libx264 or 'h265' for libx265 (default is 'h264'):"
        )
        if not codec_choice:
            codec_choice = 'h264'
        elif codec_choice not in ['h264', 'h265']:
            messagebox.showerror("Error", "Invalid codec choice. Defaulting to 'h264'.")
            codec_choice = 'h264'

        codec_map = {
            'h264': 'libx264',
            'h265': 'libx265'
        }
        codec = codec_map[codec_choice]

        # Create a new directory inside the selected source directory for saving the output videos
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(src, f"vaila_png2mp4_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        try:
            # Verifica se há arquivos PNG diretamente no diretório selecionado
            png_files = [f for f in os.listdir(src) if f.endswith(".png")]

            if png_files:
                # Se encontrar arquivos PNG no diretório raiz, processa-os diretamente
                input_pattern = os.path.join(src, self.pattern)
                # Usando o nome do diretório em vez de "output.mp4"
                output_video_name = os.path.basename(src)
                output_video_path = os.path.join(output_dir, f"{output_video_name}.mp4")

                # Check for the info.txt file for FPS, default to 30 if not found
                info_file = os.path.join(src, "info.txt")
                if os.path.exists(info_file):
                    with open(info_file, "r") as f:
                        lines = f.readlines()
                        fps = float(lines[0].split(":")[1].strip())
                else:
                    fps = 30.0  # Default FPS

                try:
                    # Create video in YUV420p color space for compatibility
                    command = [
                        "ffmpeg",
                        "-framerate", str(fps),
                        "-i", input_pattern,
                        "-c:v", codec,
                        "-pix_fmt", "yuv420p",
                        output_video_path
                    ]
                    subprocess.run(command, check=True)

                    print(f"Video creation completed and saved to {output_video_path}")
                except Exception as e:
                    print(f"Error creating video from PNG files: {e}")
                    messagebox.showerror(
                        "Error", f"Error creating video from PNG files: {e}"
                    )

            # Loop through the immediate subdirectories of the selected source directory
            for dir_name in os.listdir(src):
                dir_path = os.path.join(src, dir_name)
                if os.path.isdir(dir_path):  # Check if it's a directory
                    # Check for the existence of PNG files in the directory
                    png_files = [f for f in os.listdir(dir_path) if f.endswith(".png")]
                    if not png_files:
                        continue

                    # Check for the info.txt file for FPS, default to 30 if not found
                    info_file = os.path.join(dir_path, "info.txt")
                    if os.path.exists(info_file):
                        with open(info_file, "r") as f:
                            lines = f.readlines()
                            fps = float(lines[0].split(":")[1].strip())
                    else:
                        fps = 30.0  # Default FPS

                    input_pattern = os.path.join(dir_path, self.pattern)
                    output_video_path = os.path.join(output_dir, f"{dir_name}.mp4")

                    try:
                        # Create video in YUV420p color space for compatibility
                        command = [
                            "ffmpeg",
                            "-framerate", str(fps),
                            "-i", input_pattern,
                            "-c:v", codec,
                            "-pix_fmt", "yuv420p",
                            output_video_path
                        ]
                        subprocess.run(command, check=True)

                        print(
                            f"Video creation completed and saved to {output_video_path}"
                        )
                    except Exception as e:
                        print(f"Error creating video for {dir_name}: {e}")
                        messagebox.showerror(
                            "Error", f"Error creating video for {dir_name}: {e}"
                        )

            self.show_completion_message(
                "Video creation for all directories completed successfully."
            )

        except Exception as e:
            messagebox.showerror("Error", f"Error processing videos: {e}")

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
        root = Tk()
        root.withdraw()
        choice = simpledialog.askstring(
            "Choose Action",
            "Type 'e' to extract PNGs or 'c' to create a video from PNGs:",
        )

        if choice == "e":
            self.extract_png_from_videos()
        elif choice == "c":
            self.create_video_from_png()
        else:
            messagebox.showerror("Error", "Invalid choice.")


if __name__ == "__main__":
    processor = VideoProcessor()
    processor.run()
