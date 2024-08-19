"""
File: extractpng.py

Description:
This script allows users to either extract PNG frames from video files or create a video from a sequence of PNG images. The script ensures consistency and quality in extracted frames and generated videos, making them suitable for machine learning, computer vision, and biomechanics applications.

Version: 1.5
Last Updated: August 18, 2024
Author: Prof. Paulo Santiago

Features:
- Extract PNG frames in RGB format to ensure compatibility with machine learning models.
- Create videos in YUV420p format, which is a widely used standard in computer vision.
- Automatically creates a directory for saving the output videos, named `vaila_png2mp4_` followed by a timestamp.
- Processes all immediate subdirectories of the selected source directory, creating a video from PNG files found within each subdirectory.
- Uses the `info.txt` file within each subdirectory to determine the FPS for video creation, with a default of 30 FPS if the file is not found.
- User-friendly GUI for directory and file selection.
- Ensures consistent resolution, frame rate, and color space across all operations.

Dependencies:
- Python 3.x
- ffmpeg-python
- Tkinter
"""

import os
import time
from ffmpeg import FFmpeg
from tkinter import filedialog, messagebox, simpledialog, Tk, Toplevel, Label, Button


class VideoProcessor:

    def extract_png_from_videos(self):
        root = Tk()
        root.withdraw()

        src = filedialog.askdirectory(
            title="Select the source directory containing videos"
        )
        if not src:
            messagebox.showerror("Error", "No source directory selected.")
            return

        dest = filedialog.askdirectory(
            title="Select the destination directory for PNG files"
        )
        if not dest:
            messagebox.showerror("Error", "No destination directory selected.")
            return

        try:
            video_files = [
                f
                for f in os.listdir(src)
                if f.endswith((".avi", ".mp4", ".mov", ".mkv"))
            ]

            for item in video_files:
                video_path = os.path.join(src, item)
                video_name = os.path.splitext(item)[0]
                output_dir = os.path.join(dest, f"{video_name}_png")
                os.makedirs(output_dir, exist_ok=True)
                output_pattern = os.path.join(output_dir, "%09d.png")

                # Extract FPS using OpenCV
                fps = self.get_fps(video_path)

                # Extract frames with RGB color space
                ffmpeg = (
                    FFmpeg()
                    .input(video_path)
                    .output(
                        output_pattern,
                        vf="scale=in_range=pc:out_range=pc,format=rgb24",
                        vcodec="png",
                        q=1,
                    )
                )

                ffmpeg.execute()

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

        # Create a new directory inside the selected source directory for saving the output videos
        timestamp = time.strftime("%Y%m%d%H%M%S")
        output_dir = os.path.join(src, f"vaila_png2mp4_{timestamp}")
        os.makedirs(output_dir, exist_ok=True)

        try:
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

                    input_pattern = os.path.join(dir_path, "%09d.png")
                    output_video_path = os.path.join(output_dir, f"{dir_name}.mp4")

                    try:
                        # Create video in YUV420p color space for compatibility
                        ffmpeg = (
                            FFmpeg()
                            .input(input_pattern, framerate=fps)
                            .output(
                                output_video_path, vcodec="libx264", pix_fmt="yuv420p"
                            )
                        )

                        ffmpeg.execute()

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
