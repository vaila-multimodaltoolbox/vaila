"""
================================================================================
Remove Duplicate Frames Tool - rm_duplicateframes.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Laboratory of Biomechanics and Motor Control (LaBioCoM)
School of Physical Education and Sport of Ribeirão Preto
University of São Paulo (USP)

Contact: paulosantiago@usp.br
Laboratory website: https://github.com/vaila-multimodaltoolbox/vaila

Created: March 19, 2025
Last Updated: March 19, 2025
Version: 0.0.1

Description:
------------
This script removes frames based on a specified pattern (e.g., every 6th frame)
from a sequence of PNG images. It creates a backup of the removed frames and
can regenerate a video with the remaining frames at a specified FPS.

Dependencies:
------------
- Python 3.x
- Tkinter
- FFmpeg (for video creation)

Usage:
------
Run the script and select the directory containing PNG frame sequences.
Specify the pattern for removing frames (e.g., 6 to remove frames 6, 12, 18...).
Enter the desired FPS for the output video.

================================================================================
"""

import os
import re
import shutil
import subprocess
import time
import tkinter as tk
from tkinter import Button, Label, Toplevel, filedialog, messagebox, simpledialog


class FrameRemover:
    def __init__(self):
        self.backup_dir = None
        self.removed_count = 0
        self.new_fps = 30.0  # Default FPS

    def extract_frame_number(self, filename):
        """Extract the numeric frame number from filename"""
        match = re.search(r"(\d+)", os.path.basename(filename))
        if match:
            return int(match.group(1))
        return None

    def remove_frames_by_pattern(self, frame_dir, pattern):
        """Remove frames based on the specified pattern (e.g., every 6th frame)"""
        # Get all PNG files
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

        if not frame_files:
            print("No PNG files found in the directory")
            return 0

        print(f"Found {len(frame_files)} PNG files")

        # Create backup directory
        timestamp = time.strftime("%Y%m%d%H%M%S")
        self.backup_dir = os.path.join(frame_dir, f"removed_frames_{timestamp}")
        os.makedirs(self.backup_dir, exist_ok=True)

        # Identify frames to remove based on pattern
        frames_to_remove = []
        frame_numbers = {
            self.extract_frame_number(f): f
            for f in frame_files
            if self.extract_frame_number(f) is not None
        }

        # Generate the list of frame numbers to remove
        max_frame = max(frame_numbers.keys())
        frames_to_remove = list(range(pattern, max_frame + 1, pattern))

        print(
            f"Will remove {len(frames_to_remove)} frames: {frames_to_remove[:5]}... (pattern: every {pattern}th frame)"
        )

        # Move the frames to backup directory
        removed_count = 0
        for frame_num in frames_to_remove:
            if frame_num in frame_numbers:
                frame_file = frame_numbers[frame_num]
                src_path = os.path.join(frame_dir, frame_file)
                dst_path = os.path.join(self.backup_dir, frame_file)

                # Move the frame to backup directory
                shutil.move(src_path, dst_path)
                removed_count += 1

                # Show progress periodically
                if removed_count % 10 == 0:
                    print(f"Removed {removed_count}/{len(frames_to_remove)} frames")

        print(f"Removed {removed_count} frames")
        self.removed_count = removed_count
        return removed_count

    def update_video_info(self, frame_dir, removed_count, pattern, new_fps=None):
        """Update the video_info.txt file with new FPS and frame count"""
        info_file = os.path.join(frame_dir, "video_info.txt")
        if not os.path.exists(info_file):
            print("No video_info.txt file found to update")
            return

        # Read the original info file
        with open(info_file) as f:
            lines = f.readlines()

        # Extract values
        original_frames = None
        modified_lines = []

        for line in lines:
            if line.startswith("FPS:"):
                float(line.split(":")[-1].strip())
                # Keep original FPS line
                modified_lines.append(line)
            elif line.startswith("Total frames:"):
                original_frames = int(line.split(":")[-1].strip())
                # Update the frame count
                new_count = original_frames - removed_count
                modified_lines.append(f"Total frames: {new_count}\n")
            else:
                modified_lines.append(line)

        # Add new user-specified FPS
        if new_fps:
            modified_lines.append(f"New FPS (user specified): {new_fps:.2f}\n")

        # Add information about removal
        modified_lines.append(f"Frames removed: {removed_count}\n")
        modified_lines.append(f"Removal pattern: Every {pattern}th frame\n")
        modified_lines.append(f"Removal date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Write the updated info file
        with open(info_file, "w") as f:
            f.writelines(modified_lines)

        print(f"Updated video information file: {info_file}")

    def create_video(self, frame_dir, fps):
        """Create a video from the remaining PNG frames"""
        # Get all remaining PNG files
        frame_files = sorted([f for f in os.listdir(frame_dir) if f.endswith(".png")])

        if not frame_files:
            print("No PNG files found in the directory")
            return None

        print(f"Found {len(frame_files)} PNG files for video creation")

        # Determine pattern from first file
        first_file = frame_files[0]
        match = re.search(r"(\d+)", first_file)
        if not match:
            print(f"Could not extract frame numbering pattern from {first_file}")
            return None

        digits = len(match.group(1))
        pattern = f"%0{digits}d.png"

        print(f"Using file pattern: {pattern} (based on {first_file})")

        # Use the name of the PNG directory as the base for output names
        png_dir_name = os.path.basename(frame_dir)

        # Create output directory for video using the PNG directory name
        # Use parent directory of the frame_dir as base location
        output_dir = os.path.join(os.path.dirname(frame_dir), f"{png_dir_name}_video")
        os.makedirs(output_dir, exist_ok=True)

        # Output video path - use the same name for the video file
        output_video = os.path.join(output_dir, f"{png_dir_name}.mp4")

        # Try with temporary directory and renamed sequential files
        temp_dir = os.path.join(output_dir, "temp_sequential")
        os.makedirs(temp_dir, exist_ok=True)

        try:
            # Copy and rename files sequentially
            print(
                f"Copying {len(frame_files)} files to temporary directory with sequential naming..."
            )
            for i, file in enumerate(frame_files, 1):
                src = os.path.join(frame_dir, file)
                dst = os.path.join(temp_dir, f"{i:09d}.png")
                shutil.copy2(src, dst)

            # Build FFmpeg command with absolute paths
            input_pattern = os.path.join(temp_dir, "%09d.png").replace("\\", "/")
            output_path = output_video.replace("\\", "/")

            cmd = [
                "ffmpeg",
                "-y",
                "-framerate",
                str(fps),
                "-i",
                input_pattern,
                "-c:v",
                "libx264",
                "-pix_fmt",
                "yuv420p",
                "-crf",
                "18",
                output_path,
            ]

            print(f"Running FFmpeg command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True, capture_output=True, text=True)
            print(f"Video created successfully: {output_video}")

        except Exception as e:
            print(f"Error creating video: {str(e)}")
            return None
        finally:
            # Clean up temp directory
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        return output_video

    def restore_duplicate_frames(self, frame_dir):
        """Copy the duplicate frames back to the original directory"""
        if not self.backup_dir or not os.path.exists(self.backup_dir):
            print("No backup directory found to restore frames from")
            return False

        duplicate_files = [f for f in os.listdir(self.backup_dir) if f.endswith(".png")]

        if not duplicate_files:
            print("No duplicate frames found in backup directory")
            return False

        print(f"Found {len(duplicate_files)} duplicate frames to restore")

        # Copy files from backup to original directory
        restored_count = 0
        for file in duplicate_files:
            src_path = os.path.join(self.backup_dir, file)
            dst_path = os.path.join(frame_dir, file)

            # If the destination already exists, don't overwrite it
            if os.path.exists(dst_path):
                print(f"File {file} already exists in the original directory, skipping")
                continue

            # Copy the file (don't move - maintain backup)
            shutil.copy2(src_path, dst_path)
            restored_count += 1

            # Show progress periodically
            if restored_count % 10 == 0:
                print(f"Restored {restored_count}/{len(duplicate_files)} frames")

        print(f"Successfully restored {restored_count} duplicate frames")
        return True

    def copy_backup_to_video_dir(self, output_video_dir):
        """Copy the backup directory to the video output directory"""
        if not self.backup_dir or not os.path.exists(self.backup_dir):
            print("No backup directory found to copy")
            return False

        # Get the name of the backup directory (just the folder name)
        backup_folder_name = os.path.basename(self.backup_dir)

        # Create target directory in the video output directory
        target_dir = os.path.join(output_video_dir, backup_folder_name)

        try:
            print("Copying backup directory to video output directory...")
            # Use shutil.copytree to copy the entire directory
            shutil.copytree(self.backup_dir, target_dir)
            print(f"Successfully copied backup directory to: {target_dir}")
            return True
        except Exception as e:
            print(f"Error copying backup directory: {str(e)}")
            return False

    @classmethod
    def run_rm_duplicateframes(cls):
        """
        Static method to run the frame removal process.
        Creates its own instance of FrameRemover and executes the workflow.
        """
        # Create an instance of FrameRemover
        remover = cls()

        print(f"Running script: {os.path.basename(__file__)}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

        root = tk.Tk()
        root.withdraw()

        # Get target directory
        frame_dir = filedialog.askdirectory(title="Select directory containing PNG frames")
        if not frame_dir:
            print("No directory selected. Exiting...")
            return

        # Ask for pattern
        pattern = simpledialog.askinteger(
            "Frame Removal Pattern",
            "Enter the pattern for frame removal (e.g., 6 to remove frames 6, 12, 18...):",
            initialvalue=6,
            minvalue=2,
        )

        if not pattern:
            print("No pattern specified. Exiting...")
            return

        try:
            # Remove frames based on pattern
            removed_count = remover.remove_frames_by_pattern(frame_dir, pattern)

            if removed_count > 0:
                # Ask for new FPS
                new_fps = simpledialog.askfloat(
                    "New FPS",
                    "Enter the desired FPS for the output video:",
                    initialvalue=30.0,
                    minvalue=1.0,
                    maxvalue=120.0,
                )

                if new_fps:
                    remover.new_fps = new_fps

                    # Update video info file
                    remover.update_video_info(frame_dir, removed_count, pattern, new_fps)

                    # Create video directly (without asking)
                    output_video = remover.create_video(frame_dir, new_fps)

                    if output_video:
                        # Get the output video directory
                        output_video_dir = os.path.dirname(output_video)

                        # Copy backup directory to video output directory
                        remover.copy_backup_to_video_dir(output_video_dir)

                        # Restore frames to original directory
                        remover.restore_duplicate_frames(frame_dir)

                        # Show completion message
                        remover.show_completion_message(
                            f"Process completed successfully!\n\n"
                            f"1. Removed {removed_count} frames temporarily\n"
                            f"2. Created video at:\n   {output_video}\n"
                            f"3. Copied duplicate frames to video directory\n"
                            f"4. Restored all frames to original directory\n\n"
                            f"Your original directory now contains all frames,\n"
                            f"and the video has been created with non-duplicate frames."
                        )
                    else:
                        # If video creation failed, just restore frames
                        remover.restore_duplicate_frames(frame_dir)
                        remover.show_completion_message(
                            "Failed to create video, but removed frames \n"
                            "have been restored to the original directory."
                        )
            else:
                messagebox.showinfo("No Frames Removed", "No frames were removed.")

        except Exception as e:
            # Ensure frames are restored even if an error occurs
            if (
                hasattr(remover, "backup_dir")
                and remover.backup_dir
                and os.path.exists(remover.backup_dir)
            ):
                remover.restore_duplicate_frames(frame_dir)

            messagebox.showerror(
                "Error",
                f"An error occurred: {str(e)}\nAny removed frames have been restored.",
            )

    def show_completion_message(self, message):
        """Show a completion message to the user"""
        root = tk.Tk()
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


def run_rm_duplicateframes():
    """
    Module-level function to run the frame removal process.
    This is a wrapper to call the class method, allowing the function
    to be imported directly from the module.
    """
    # Call the class method
    FrameRemover.run_rm_duplicateframes()


if __name__ == "__main__":
    run_rm_duplicateframes()
