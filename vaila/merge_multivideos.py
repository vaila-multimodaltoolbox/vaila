"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

merge_multvideos.py

Description:
This script allows users to merge multiple video files into a single video in a specified order.
It provides two methods of selecting videos:
1. By choosing multiple video files through a file selection dialog and arranging their order.
2. By providing a text file with a list of video files to merge.

The script processes videos using FFmpeg without requiring GPU acceleration, making it compatible
with all systems.

Key Features:
- Graphical User Interface (GUI) for selecting and arranging multiple videos
- Option to load video list from a text file
- Preview of selected videos and their order
- Ability to reorder videos before processing
- Detailed console output for tracking progress and handling errors
- Creation of a timestamped output directory for organized file management

Usage:
- Run the script to open a graphical interface.
- Select videos either by choosing multiple files or by loading a text file.
- Arrange the videos in the desired order.
- Specify an output directory and start the merge process.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- Python 3.x environment.
- Tkinter for the GUI components (usually included with Python).

Installation of FFmpeg (for video processing):
- **Conda (recommended)**: 
  ```bash
  conda install -c conda-forge ffmpeg
  ```
"""

import os
import time
import subprocess
import tkinter as tk
from tkinter import filedialog, messagebox, ttk, simpledialog
from datetime import datetime
import threading


class VideoMergeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Merge Multiple Videos")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        self.video_files = []  # List to store the selected video paths
        self.output_dir = ""  # Output directory

        # Create style for selected frame
        self.style = ttk.Style()
        self.style.configure(
            "Selected.TFrame", background="#ADD8E6"
        )  # Light blue background

        # Create main frame
        self.main_frame = ttk.Frame(root, padding="10")
        self.main_frame.pack(fill=tk.BOTH, expand=True)

        # Create top frame for buttons
        self.button_frame = ttk.Frame(self.main_frame)
        self.button_frame.pack(fill=tk.X, pady=(0, 10))

        # Create buttons for selecting videos and directories
        ttk.Button(
            self.button_frame,
            text="Select Multiple Videos",
            command=self.select_multiple_videos,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.button_frame,
            text="Load from Text File",
            command=self.load_from_text_file,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(
            self.button_frame,
            text="Set Output Directory",
            command=self.set_output_directory,
        ).pack(side=tk.LEFT, padx=5)

        ttk.Button(self.button_frame, text="Clear All", command=self.clear_all).pack(
            side=tk.LEFT, padx=5
        )

        # Label for output directory
        self.output_dir_label = ttk.Label(
            self.main_frame, text="Output Directory: Not selected"
        )
        self.output_dir_label.pack(fill=tk.X, pady=(0, 10))

        # Frame for the video list
        self.video_list_frame = ttk.LabelFrame(
            self.main_frame, text="Selected Videos (in order of merging)"
        )
        self.video_list_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Scrollable frame for the video list
        self.canvas = tk.Canvas(self.video_list_frame)
        self.scrollbar = ttk.Scrollbar(
            self.video_list_frame, orient="vertical", command=self.canvas.yview
        )
        self.scrollable_frame = ttk.Frame(self.canvas)

        self.scrollable_frame.bind(
            "<Configure>",
            lambda e: self.canvas.configure(scrollregion=self.canvas.bbox("all")),
        )

        self.canvas.create_window((0, 0), window=self.scrollable_frame, anchor="nw")
        self.canvas.configure(yscrollcommand=self.scrollbar.set)

        self.canvas.pack(side="left", fill="both", expand=True)
        self.scrollbar.pack(side="right", fill="y")

        # Button frame at the bottom
        self.bottom_frame = ttk.Frame(self.main_frame)
        self.bottom_frame.pack(fill=tk.X, pady=(0, 10))

        # Order manipulation buttons
        self.order_frame = ttk.Frame(self.bottom_frame)
        self.order_frame.pack(side=tk.LEFT)

        ttk.Button(self.order_frame, text="Move Up", command=self.move_up).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(self.order_frame, text="Move Down", command=self.move_down).pack(
            side=tk.LEFT, padx=5
        )

        ttk.Button(
            self.order_frame, text="Remove Selected", command=self.remove_selected
        ).pack(side=tk.LEFT, padx=5)

        # Start merge button
        self.merge_button = ttk.Button(
            self.bottom_frame,
            text="Merge Videos",
            command=self.start_merge,
            style="Accent.TButton",
        )
        self.merge_button.pack(side=tk.RIGHT, padx=5)

        # Create a style for accent button
        style = ttk.Style()
        style.configure("Accent.TButton", background="blue", foreground="white")

        # Variables for tracking
        self.selected_index = None
        self.video_frames = []

        # Progress bar frame
        self.progress_frame = ttk.Frame(self.main_frame)
        self.progress_frame.pack(fill=tk.X)

        self.progress_bar = ttk.Progressbar(
            self.progress_frame, length=100, mode="indeterminate"
        )
        self.progress_label = ttk.Label(self.progress_frame, text="")

    def select_multiple_videos(self):
        """Open a file dialog to select multiple video files"""
        filetypes = [
            ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV *.MKV"),
            ("All files", "*.*"),
        ]

        video_paths = filedialog.askopenfilenames(
            title="Select Video Files to Merge", filetypes=filetypes
        )

        if video_paths:  # If user selected files
            # Add to existing list
            self.video_files.extend(video_paths)
            # Update the display
            self.update_video_list()

    def load_from_text_file(self):
        """Load video list from a text file"""
        # Show example format in a message box
        messagebox.showinfo(
            "Text File Format",
            "Expected format of the text file:\n\nvideo1.mp4\nvideo2.mp4\nvideo3.mp4\n\n"
            + "Each line should contain just the filename (if in the same directory as the text file)\n"
            + "or the full path to the video file.\n\n"
            + "Videos will be merged in the order they appear in the file.",
        )

        text_file_path = filedialog.askopenfilename(
            title="Select Text File with Video Paths",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )

        if not text_file_path:
            return

        try:
            base_dir = os.path.dirname(text_file_path)  # Get directory of the text file

            with open(text_file_path, "r") as file:
                for line in file.readlines():
                    line = line.strip()
                    if line:
                        # Check if it's a relative or absolute path
                        if os.path.isabs(line):
                            video_path = line
                        else:
                            video_path = os.path.join(base_dir, line)

                        # Verify the file exists
                        if os.path.exists(video_path):
                            self.video_files.append(video_path)
                        else:
                            print(f"Warning: File not found: {video_path}")

            # Update the display
            self.update_video_list()

            if not self.video_files:
                messagebox.showwarning(
                    "Warning", "No valid video files found in the text file."
                )

        except Exception as e:
            messagebox.showerror(
                "Error", f"Failed to load videos from text file: {str(e)}"
            )

    def set_output_directory(self):
        """Set the output directory for the merged video"""
        directory = filedialog.askdirectory(title="Select Output Directory")
        if directory:
            self.output_dir = directory
            self.output_dir_label.config(text=f"Output Directory: {directory}")

    def clear_all(self):
        """Clear all selected videos"""
        self.video_files = []
        self.update_video_list()

    def update_video_list(self):
        """Update the display of the video list"""
        # Clear previous frames
        for frame in self.video_frames:
            frame.destroy()

        self.video_frames = []

        # Create a frame for each video
        for i, video_path in enumerate(self.video_files):
            # Use tk.Frame instead of ttk.Frame for better styling control
            frame = tk.Frame(self.scrollable_frame, bd=1, relief=tk.FLAT)
            frame.pack(fill=tk.X, pady=2, padx=2)

            # Add selection indicator
            self.selection_indicator = tk.Label(frame, text="►", fg="blue", width=2)
            self.selection_indicator.pack(side=tk.LEFT, padx=(2, 0))
            # Hide the indicator initially
            if self.selected_index != i:
                self.selection_indicator.pack_forget()

            # Add video index and name
            index_label = ttk.Label(frame, text=f"{i+1}.", width=3)
            index_label.pack(side=tk.LEFT, padx=(5, 0))

            video_name = os.path.basename(video_path)
            name_label = ttk.Label(frame, text=video_name, anchor="w")
            name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

            # Add selection functionality
            frame.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))
            index_label.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))
            name_label.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))

            self.video_frames.append(frame)

    def select_video(self, index):
        """Handle selection of a video in the list"""
        # Clear previous selection
        if self.selected_index is not None and 0 <= self.selected_index < len(
            self.video_frames
        ):
            self.video_frames[self.selected_index].config(
                bg=self.root.cget("bg"), bd=1, relief=tk.FLAT
            )
            # Remove all selection indicators from previous selected frame
            for widget in self.video_frames[self.selected_index].winfo_children():
                if isinstance(widget, tk.Label) and widget.cget("text") == "►":
                    widget.pack_forget()

        self.selected_index = index

        # Highlight new selection
        if 0 <= index < len(self.video_frames):
            # Set border and background
            self.video_frames[index].config(bg="#e6f2ff", bd=1, relief=tk.RAISED)

            # Add selection indicator
            selection_indicator = None
            for widget in self.video_frames[index].winfo_children():
                if isinstance(widget, tk.Label) and widget.cget("text") == "►":
                    selection_indicator = widget
                    break

            if selection_indicator:
                selection_indicator.pack(
                    side=tk.LEFT,
                    padx=(2, 0),
                    before=self.video_frames[index].winfo_children()[1],
                )

        # Print debug info
        print(f"Selected video at index {index}")

    def move_up(self):
        """Move the selected video up in the list"""
        if self.selected_index is None or self.selected_index <= 0:
            return

        # Swap videos
        (
            self.video_files[self.selected_index],
            self.video_files[self.selected_index - 1],
        ) = (
            self.video_files[self.selected_index - 1],
            self.video_files[self.selected_index],
        )

        # Update selected index
        self.selected_index -= 1

        # Update display
        self.update_video_list()

        # Update selection
        if 0 <= self.selected_index < len(self.video_frames):
            self.video_frames[self.selected_index].configure(style="Selected.TFrame")

    def move_down(self):
        """Move the selected video down in the list"""
        if (
            self.selected_index is None
            or self.selected_index >= len(self.video_files) - 1
        ):
            return

        # Swap videos
        (
            self.video_files[self.selected_index],
            self.video_files[self.selected_index + 1],
        ) = (
            self.video_files[self.selected_index + 1],
            self.video_files[self.selected_index],
        )

        # Update selected index
        self.selected_index += 1

        # Update display
        self.update_video_list()

        # Update selection
        if 0 <= self.selected_index < len(self.video_frames):
            self.video_frames[self.selected_index].configure(style="Selected.TFrame")

    def remove_selected(self):
        """Remove the selected video from the list"""
        if self.selected_index is None or not (
            0 <= self.selected_index < len(self.video_files)
        ):
            return

        # Remove video
        del self.video_files[self.selected_index]

        # Reset selection if list is empty
        if not self.video_files:
            self.selected_index = None
        # Adjust selection if removing the last item
        elif self.selected_index >= len(self.video_files):
            self.selected_index = len(self.video_files) - 1

        # Update display
        self.update_video_list()

        # Update selection
        if self.selected_index is not None and 0 <= self.selected_index < len(
            self.video_frames
        ):
            self.video_frames[self.selected_index].configure(style="Selected.TFrame")

    def start_merge(self):
        """Start the video merging process"""
        if not self.video_files:
            messagebox.showerror(
                "Error", "No videos selected. Please select videos to merge."
            )
            return

        if not self.output_dir:
            messagebox.showerror(
                "Error", "Output directory not set. Please select an output directory."
            )
            return

        # Get output filename from user
        output_filename = simpledialog.askstring(
            "Output Filename",
            "Enter name for the merged video (without extension):",
            initialvalue="merged_video",
        )

        if not output_filename:
            return

        # Create timestamp directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_subdir = os.path.join(self.output_dir, f"merged_videos_{timestamp}")
        os.makedirs(output_subdir, exist_ok=True)

        # Full output path
        output_video_path = os.path.join(output_subdir, f"{output_filename}.mp4")

        # Show progress elements
        self.progress_bar.pack(fill=tk.X, pady=(5, 0))
        self.progress_label.pack(fill=tk.X, pady=(5, 0))
        self.progress_bar.start()
        self.progress_label.config(text="Preparing to merge videos...")

        # Start merge process in a separate thread
        merge_thread = threading.Thread(
            target=self.do_merge, args=(output_video_path, output_subdir), daemon=True
        )
        merge_thread.start()

    def do_merge(self, output_video_path, output_subdir):
        """Execute the merge process in a background thread"""
        try:
            # Create temporary filelist.txt for ffmpeg
            filelist_path = os.path.join(output_subdir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for video_path in self.video_files:
                    # Escape single quotes in file path
                    escaped_path = video_path.replace("'", "'\\''")
                    f.write(f"file '{escaped_path}'\n")

            # Update progress
            self.root.after(
                0,
                lambda: self.progress_label.config(
                    text="Creating list of videos to merge..."
                ),
            )

            # Create FFmpeg command for merging
            ffmpeg_command = [
                "ffmpeg",
                "-y",  # Override output if exists
                "-f",
                "concat",
                "-safe",
                "0",  # Allow absolute paths
                "-i",
                filelist_path,
                "-c",
                "copy",  # Copy streams without re-encoding (fast)
                output_video_path,
            ]

            # Update progress
            self.root.after(
                0, lambda: self.progress_label.config(text="Starting merge process...")
            )

            # Run FFmpeg command
            process = subprocess.Popen(
                ffmpeg_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            # Read FFmpeg output for progress updates
            for line in process.stderr:
                # Update progress label with the current status
                if "frame=" in line or "time=" in line:
                    self.root.after(
                        0,
                        lambda l=line: self.progress_label.config(
                            text=f"Merging: {l.strip()}"
                        ),
                    )

            # Wait for process to complete
            process.wait()

            # Write a log file with information about the merged videos
            log_file_path = os.path.join(
                output_subdir, f"{os.path.basename(output_video_path)}_merge_info.txt"
            )
            with open(log_file_path, "w") as log_file:
                log_file.write(f"Merged Video: {output_video_path}\n")
                log_file.write(
                    f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
                )
                log_file.write("Videos merged in this order:\n")

                for i, video_path in enumerate(self.video_files, 1):
                    log_file.write(f"{i}. {video_path}\n")

            # If process succeeded, show success message
            if process.returncode == 0:
                self.root.after(0, lambda: self.merge_complete(True, output_video_path))
            else:
                self.root.after(
                    0,
                    lambda: self.merge_complete(
                        False, f"FFmpeg error: return code {process.returncode}"
                    ),
                )

        except Exception as e:
            self.root.after(0, lambda: self.merge_complete(False, str(e)))

    def merge_complete(self, success, message):
        """Handle completion of the merge process"""
        # Stop progress bar
        self.progress_bar.stop()
        self.progress_bar.pack_forget()
        self.progress_label.pack_forget()

        if success:
            messagebox.showinfo(
                "Success", f"Videos merged successfully!\nOutput: {message}"
            )
        else:
            messagebox.showerror("Error", f"Failed to merge videos: {message}")


def run_merge_multivideos():
    """Main function to run the multi-video merger application"""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting multi-video merger...")

    root = tk.Tk()
    app = VideoMergeApp(root)
    root.mainloop()


if __name__ == "__main__":
    run_merge_multivideos()
