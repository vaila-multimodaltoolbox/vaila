"""
File: extractpng.py

Description:
This script allows users to either extract PNG frames from video files or create a video from a sequence of PNG images. When extracting frames, the script saves them in a specified directory and records the frame rate (FPS) and other relevant information in an `info.txt` file. This information is then used to correctly reassemble the PNG images back into a video, ensuring the final video has the correct frame rate and format. The script provides a graphical user interface (GUI) for selecting source directories, destination directories, and output files.

Version: 1.2
Last Updated: August 17, 2024
Author: Prof. Paulo Santiago

Features:
- Extract PNG frames from videos and save them with correct frame rate information.
- Create videos from PNG images using the saved frame rate.
- User-friendly GUI for directory and file selection.
- Supports multiple video formats including .avi, .mp4, .mov, and .mkv.
- Progress bar displayed in the terminal during the extraction or creation process.
- Confirmation message after process completion to ensure the main GUI remains responsive.

Usage:
- Run the script and choose the desired action (extract PNGs or create a video).
- Follow the prompts to select the source and destination directories.
- The user can type 'e' to extract PNGs or 'c' to create a video.

Dependencies:
- Python 3.x
- OpenCV
- ffmpeg-python
- Tkinter
- tqdm
"""

import os
import cv2
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

                cap = cv2.VideoCapture(video_path)
                fps = cap.get(cv2.CAP_PROP_FPS)

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

                cap.release()

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

        dest = filedialog.asksaveasfilename(
            title="Save the video file as",
            defaultextension=".mp4",
            filetypes=[
                ("MP4 files", "*.mp4"),
                ("AVI files", "*.avi"),
                ("MOV files", "*.mov"),
                ("MKV files", "*.mkv"),
            ],
        )
        if not dest:
            messagebox.showerror("Error", "No destination file selected.")
            return

        try:
            info_file = os.path.join(src, "info.txt")
            with open(info_file, "r") as f:
                lines = f.readlines()
                fps = float(lines[0].split(":")[1].strip())

            input_pattern = os.path.join(src, "%09d.png")

            ffmpeg = (
                FFmpeg()
                .input(input_pattern, framerate=fps)
                .output(dest, vcodec="libx264", pix_fmt="yuv420p")
            )

            ffmpeg.execute()

            print(f"Video creation completed and saved to {dest}")

            self.show_completion_message("Video creation completed successfully.")
        except Exception as e:
            messagebox.showerror("Error", f"Error creating video: {e}")

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
