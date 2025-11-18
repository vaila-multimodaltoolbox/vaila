"""
merge_multivideos.py

Author: Paulo R. P. Santiago
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

Created: 25 February 2025
Update: 13 March 2025
Version updated: 0.2.0

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
import subprocess
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, simpledialog, ttk

from rich import print


class VideoMergeApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Merge Multiple Videos")
        self.root.geometry("800x600")
        self.root.minsize(700, 500)

        self.video_files = []  # List to store the selected video paths
        self.output_dir = ""  # Output directory
        self.video_metadata = {}  # Dictionary to store video metadata

        # Usar uma string diretamente para maior controle
        self.selected_mode = "frame_accurate"  # Modo padrão

        # Create style for selected frame
        self.style = ttk.Style()
        self.style.configure("Selected.TFrame", background="#ADD8E6")  # Light blue background

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

        # Add a frame for merge mode selection
        self.mode_frame = ttk.LabelFrame(self.main_frame, text="Merge Mode")
        self.mode_frame.pack(fill=tk.X, pady=(0, 10))

        # Create mode buttons using the new function
        self.create_mode_buttons()

        # Label for output directory
        self.output_dir_label = ttk.Label(self.main_frame, text="Output Directory: Not selected")
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

        ttk.Button(self.order_frame, text="Remove Selected", command=self.remove_selected).pack(
            side=tk.LEFT, padx=5
        )

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

        # Init progress bar and label, but don't show them yet
        try:
            self.progress_bar = ttk.Progressbar(
                self.progress_frame, length=100, mode="indeterminate"
            )
            self.progress_label = ttk.Label(self.progress_frame, text="")
            print("DEBUG: Progress bar and label initialized")
        except Exception as e:
            print(f"ERROR: Failed to initialize progress bar: {str(e)}")

        # Initially, don't show the progress bar and label
        # They will be shown only when the merge process starts

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
            # Get metadata for each video
            self.load_video_metadata(video_paths)
            # Update the display
            self.update_video_list()

    def load_video_metadata(self, video_paths):
        """Load metadata for selected videos"""
        for video_path in video_paths:
            if video_path not in self.video_metadata:
                try:
                    # Use separate ffprobe commands for each property to avoid parsing issues

                    # Get resolution
                    res_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                    res_result = subprocess.run(res_cmd, capture_output=True, text=True)
                    try:
                        width, height = map(int, res_result.stdout.strip().split(","))
                    except:
                        width, height = 0, 0

                    # Get FPS
                    fps_cmd = [
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
                    fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)
                    try:
                        r_frame_rate = fps_result.stdout.strip()
                        if "/" in r_frame_rate:
                            num, den = map(int, r_frame_rate.split("/"))
                            fps = round(num / den, 2)
                        else:
                            fps = float(r_frame_rate)
                    except:
                        fps = 0

                    # Get codec
                    codec_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                    codec_result = subprocess.run(codec_cmd, capture_output=True, text=True)
                    codec = codec_result.stdout.strip()

                    # Get frame count
                    frames_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_frames",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                    frames_result = subprocess.run(frames_cmd, capture_output=True, text=True)
                    try:
                        nb_frames = int(frames_result.stdout.strip())
                    except (ValueError, IndexError):
                        # If frame count is not available, estimate it from duration
                        duration_cmd = [
                            "ffprobe",
                            "-v",
                            "error",
                            "-show_entries",
                            "format=duration",
                            "-of",
                            "csv=p=0",
                            video_path,
                        ]
                        duration_result = subprocess.run(
                            duration_cmd, capture_output=True, text=True
                        )
                        try:
                            duration = float(duration_result.stdout.strip())
                            nb_frames = int(duration * fps) if fps > 0 else 0
                        except:
                            nb_frames = 0

                    # Store metadata
                    self.video_metadata[video_path] = {
                        "width": width,
                        "height": height,
                        "resolution": f"{width}x{height}",
                        "fps": fps,
                        "frames": nb_frames,
                        "codec": codec,
                    }

                except Exception as e:
                    print(f"Error loading metadata for {video_path}: {str(e)}")
                    self.video_metadata[video_path] = {
                        "resolution": "Unknown",
                        "fps": "Unknown",
                        "frames": "Unknown",
                        "codec": "Unknown",
                    }

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

            with open(text_file_path) as file:
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
                messagebox.showwarning("Warning", "No valid video files found in the text file.")

        except Exception as e:
            messagebox.showerror("Error", f"Failed to load videos from text file: {str(e)}")

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

        print(f"DEBUG: Atualizando lista de vídeos com {len(self.video_files)} arquivos")

        # Create a frame for each video
        for i, video_path in enumerate(self.video_files):
            try:
                # Use tk.Frame instead of ttk.Frame for better styling control
                frame = tk.Frame(self.scrollable_frame, bd=1, relief=tk.FLAT)
                frame.pack(fill=tk.X, pady=2, padx=2)

                # Add selection indicator
                selection_indicator = tk.Label(frame, text="►", fg="blue", width=2)
                selection_indicator.pack(side=tk.LEFT, padx=(2, 0))
                # Hide the indicator initially
                if self.selected_index != i:
                    selection_indicator.pack_forget()

                # Add video index and name
                index_label = ttk.Label(frame, text=f"{i + 1}.", width=3)
                index_label.pack(side=tk.LEFT, padx=(5, 0))

                video_name = os.path.basename(video_path)
                name_label = ttk.Label(frame, text=video_name, anchor="w")
                name_label.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

                # Add video metadata if available
                if video_path in self.video_metadata:
                    metadata = self.video_metadata[video_path]
                    # Check if all metadata values are available
                    resolution = metadata.get("resolution", "Unknown")
                    fps = metadata.get("fps", "Unknown")
                    codec = metadata.get("codec", "Unknown")

                    meta_text = f"[{resolution}, {fps} FPS, {codec}]"
                    meta_label = ttk.Label(frame, text=meta_text, foreground="blue")
                    meta_label.pack(side=tk.RIGHT, padx=5)
                else:
                    print(f"DEBUG: Metadata not available for: {video_path}")

                # Add selection functionality
                frame.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))
                index_label.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))
                name_label.bind("<Button-1>", lambda e, idx=i: self.select_video(idx))

                self.video_frames.append(frame)
                print(f"DEBUG: Added frame for video {i + 1}: {video_name}")
            except Exception as e:
                print(f"ERROR: Failed to create frame for video {i + 1}: {str(e)}")

    def select_video(self, index):
        """Handle selection of a video in the list"""
        # Clear previous selection
        if self.selected_index is not None and 0 <= self.selected_index < len(self.video_frames):
            try:
                self.video_frames[self.selected_index].config(
                    bg=self.root.cget("bg"), bd=1, relief=tk.FLAT
                )
                # Remove all selection indicators from previous selected frame
                for widget in self.video_frames[self.selected_index].winfo_children():
                    if isinstance(widget, tk.Label) and widget.cget("text") == "►":
                        widget.pack_forget()
            except Exception as e:
                print(f"DEBUG: Error clearing previous selection: {str(e)}")

        self.selected_index = index

        # Highlight new selection
        if 0 <= index < len(self.video_frames):
            try:
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
            except Exception as e:
                print(f"DEBUG: Error highlighting new selection: {str(e)}")

        # Print debug info
        print(f"DEBUG: Selected video at index {index}")

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

        # Update selection - corrigido para usar config em vez de configure(style=...)
        if 0 <= self.selected_index < len(self.video_frames):
            # Usar config(bg=...) em vez de configure(style=...) para tk.Frame
            self.video_frames[self.selected_index].config(bg="#e6f2ff", relief=tk.RAISED)

    def move_down(self):
        """Move the selected video down in the list"""
        if self.selected_index is None or self.selected_index >= len(self.video_files) - 1:
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

        # Update selection - corrected to use config instead of configure(style=...)
        if 0 <= self.selected_index < len(self.video_frames):
            # Use config(bg=...) instead of configure(style=...) for tk.Frame
            self.video_frames[self.selected_index].config(bg="#e6f2ff", relief=tk.RAISED)

    def remove_selected(self):
        """Remove the selected video from the list"""
        if self.selected_index is None or not (0 <= self.selected_index < len(self.video_files)):
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

        # Update selection - corrected to use config instead of configure(style=...)
        if self.selected_index is not None and 0 <= self.selected_index < len(self.video_frames):
            # Use config(bg=...) instead of configure(style=...) for tk.Frame
            self.video_frames[self.selected_index].config(bg="#e6f2ff", relief=tk.RAISED)

    def start_merge(self):
        """Start the video merging process"""
        try:
            # Adicione esta linha para debug
            print(f"DEBUG: Modo selecionado ao iniciar mesclagem: '{self.selected_mode}'")

            if not self.video_files:
                messagebox.showerror("Error", "No videos selected. Please select videos to merge.")
                return

            if not self.output_dir:
                messagebox.showerror(
                    "Error",
                    "Output directory not set. Please select an output directory.",
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

            # Adicionar o nome do método de mesclagem ao diretório de saída
            method_name = ""
            if self.selected_mode == "fast":
                method_name = "fast"
            elif self.selected_mode == "precise":
                method_name = "precise"
            elif self.selected_mode == "frame_accurate":
                method_name = "accurate"

            output_subdir = os.path.join(self.output_dir, f"merge_{method_name}_{timestamp}")
            os.makedirs(output_subdir, exist_ok=True)

            # Full output path
            output_video_path = os.path.join(output_subdir, f"{output_filename}.mp4")

            # Ensure progress bar and label are configured correctly
            self.progress_bar.configure(mode="indeterminate", maximum=100, value=0)
            self.progress_label.configure(text="Preparing to merge videos...")

            # Show progress elements
            self.progress_bar.pack(fill=tk.X, pady=(5, 0))
            self.progress_label.pack(fill=tk.X, pady=(5, 0))
            self.progress_bar.start(10)  # Atualizar a cada 10ms

            print("DEBUG: Progress bar started")

            # Start merge process in a separate thread
            merge_thread = threading.Thread(
                target=self.do_merge,
                args=(output_video_path, output_subdir),
                daemon=True,
            )
            merge_thread.start()
            print("DEBUG: Merge thread started")
        except Exception as e:
            import traceback

            print(f"Error starting merge process: {str(e)}")
            print(traceback.format_exc())
            messagebox.showerror("Error", f"Failed to start merge process: {str(e)}")

    def do_merge(self, output_video_path, output_subdir):
        """Execute the merge process in a background thread"""
        try:
            mode = self.selected_mode
            print(f"DEBUG: Iniciando processo de mesclagem no modo: '{mode}'")

            # Select the appropriate merging method based on user choice
            if mode == "precise":
                print("DEBUG: Using precise mode")
                self.do_precise_merge(output_video_path, output_subdir)
            elif mode == "frame_accurate":
                print("DEBUG: Using frame-accurate mode")
                self.do_frame_accurate_merge(output_video_path, output_subdir)
            elif mode == "fast":
                print("DEBUG: Using fast mode")
                self.do_fast_merge(output_video_path, output_subdir)
            else:
                print(f"DEBUG: Unknown mode: '{mode}', using default (frame-accurate)")
                self.do_frame_accurate_merge(output_video_path, output_subdir)
        except Exception as e:
            import traceback

            print(f"ERROR in merge process: {str(e)}")
            print(traceback.format_exc())
            # Capture error message as string to avoid issues with 'e' variable
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.merge_complete(False, msg))

    def do_precise_merge(self, output_video_path, output_subdir):
        """Execute precise merging that ensures all frames are included"""
        try:
            # Create temporary directory for intermediate files
            temp_dir = os.path.join(output_subdir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Update progress
            self.root.after(0, lambda: self.progress_label.config(text="Analyzing videos..."))

            # First pass: analyze videos to determine optimal parameters
            highest_res = [0, 0]
            target_fps = 0

            for i, video_path in enumerate(self.video_files):
                if video_path in self.video_metadata:
                    metadata = self.video_metadata[video_path]
                    width, height = metadata["width"], metadata["height"]
                    fps = metadata["fps"]
                else:
                    # Fallback to analysis if metadata isn't available
                    analyze_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=width,height,r_frame_rate",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]

                    result = subprocess.run(analyze_cmd, capture_output=True, text=True)
                    width, height, r_frame_rate = result.stdout.strip().split(",")
                    width, height = int(width), int(height)

                    # Parse fractional frame rate (e.g., "30000/1001")
                    if "/" in r_frame_rate:
                        num, den = map(int, r_frame_rate.split("/"))
                        fps = num / den
                    else:
                        fps = float(r_frame_rate)

                # Update highest resolution
                if width * height > highest_res[0] * highest_res[1]:
                    highest_res = [width, height]

                # Update target FPS (use highest for best quality)
                target_fps = max(target_fps, fps)

                # Use captured values in lambda
                current_i = i
                current_width = width
                current_height = height
                current_fps = fps
                self.root.after(
                    0,
                    lambda i=current_i,
                    w=current_width,
                    h=current_height,
                    fps=current_fps: self.progress_label.config(
                        text=f"Analyzed video {i + 1}/{len(self.video_files)}: {w}x{h} at {fps:.2f} FPS"
                    ),
                )

            # Second pass: convert each video to same format
            temp_files = []
            for i, video_path in enumerate(self.video_files):
                # Use captured values in lambda
                current_i = i
                self.root.after(
                    0,
                    lambda i=current_i: self.progress_label.config(
                        text=f"Processing video {i + 1}/{len(self.video_files)}..."
                    ),
                )

                temp_file = os.path.join(temp_dir, f"temp_{i}.mp4")
                temp_files.append(temp_file)

                # Use full reencoding to ensure consistent frame rates and resolution
                convert_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-c:v",
                    "libx264",
                    "-preset",
                    "medium",  # Use medium preset for balance of speed/quality
                    "-r",
                    str(target_fps),  # Enforce consistent frame rate
                    "-vf",
                    f"scale={highest_res[0]}:{highest_res[1]}:force_original_aspect_ratio=decrease,pad={highest_res[0]}:{highest_res[1]}:(ow-iw)/2:(oh-ih)/2",  # Scale and pad
                    "-pix_fmt",
                    "yuv420p",  # Standard pixel format
                    "-an",  # Remove audio (we're focused on video frames)
                    temp_file,
                ]

                process = subprocess.Popen(
                    convert_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                )

                # Read output for progress updates
                for line in process.stderr:
                    if "frame=" in line or "time=" in line:
                        # Capture current line for lambda
                        current_line = line.strip()
                        current_i = i
                        self.root.after(
                            0,
                            lambda i=current_i, l=current_line: self.progress_label.config(
                                text=f"Processing video {i + 1}/{len(self.video_files)}: {l}"
                            ),
                        )

                process.wait()
                if process.returncode != 0:
                    raise Exception(f"Error processing video {i + 1}")

            # Create filelist for concat
            filelist_path = os.path.join(temp_dir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for temp_file in temp_files:
                    # Use relative paths
                    rel_path = os.path.relpath(temp_file, temp_dir)
                    f.write(f"file '{rel_path}'\n")

            # Final merge using the preprocessed files
            self.root.after(
                0,
                lambda: self.progress_label.config(text="Merging preprocessed videos..."),
            )

            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist_path,
                "-c",
                "copy",  # Now we can safely use copy since all videos have same format
                output_video_path,
            ]

            process = subprocess.Popen(
                concat_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            for line in process.stderr:
                if "frame=" in line or "time=" in line:
                    # Capture current line for lambda
                    current_line = line.strip()
                    self.root.after(
                        0,
                        lambda l=current_line: self.progress_label.config(text=f"Merging: {l}"),
                    )

            process.wait()
            if process.returncode != 0:
                raise Exception("Error during final merge")

            # Verify final output frame count
            verify_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                output_video_path,
            ]

            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            final_frames = int(result.stdout.strip())

            # Calcular o total de frames dos vídeos originais
            total_original_frames = 0
            for video_path in self.video_files:
                meta = self.video_metadata.get(video_path, {})
                frames = meta.get("frames", 0)
                if frames != "Unknown":
                    total_original_frames += int(frames)

            # Write log file with information
            base_name = os.path.splitext(os.path.basename(output_video_path))[0]
            log_file_path = os.path.join(output_subdir, f"{base_name}_merge_info.txt")
            frame_report_path = os.path.join(output_subdir, f"{base_name}_frame_report.txt")

            with open(log_file_path, "w") as log_file:
                log_file.write(f"Merged Video: {output_video_path}\n")
                log_file.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Target Resolution: {highest_res[0]}x{highest_res[1]}\n")
                log_file.write(f"Target FPS: {target_fps}\n")
                log_file.write("Merge Mode: Precise (reencoded for consistent frames)\n\n")
                log_file.write(f"Sum of original frames: {total_original_frames}\n")
                log_file.write(f"Frames in final video: {final_frames}\n")
                log_file.write(f"Difference: {final_frames - total_original_frames} frames\n\n")
                log_file.write("Videos merged in this order:\n")

                for i, video_path in enumerate(self.video_files, 1):
                    meta = self.video_metadata.get(video_path, {})
                    resolution = meta.get("resolution", "Unknown")
                    fps = meta.get("fps", "Unknown")
                    codec = meta.get("codec", "Unknown")
                    frames = meta.get("frames", "Unknown")
                    log_file.write(
                        f"{i}. {video_path}\n   [{resolution}, {fps} FPS, {codec}, {frames} frames]\n"
                    )

            # Adicionar um relatório de frames também para o modo preciso
            with open(frame_report_path, "w") as report_file:
                report_file.write("FRAME COUNT REPORT\n")
                report_file.write("==============================\n\n")
                report_file.write(f"Final Video: {base_name}.mp4\n")
                report_file.write(f"Target FPS: {target_fps}\n")
                report_file.write(f"Total Frames in Final Video: {final_frames}\n\n")
                report_file.write("Input Videos:\n")
                report_file.write("------------------\n")

                for i, video_path in enumerate(self.video_files, 1):
                    meta = self.video_metadata.get(video_path, {})
                    frames = meta.get("frames", "Unknown")
                    fps = meta.get("fps", "Unknown")
                    video_name = os.path.basename(video_path)

                    report_file.write(f"{i}. {video_name}\n")
                    report_file.write(f"   Original Frames: {frames}\n")
                    report_file.write(f"   Original FPS: {fps}\n")
                    if fps != target_fps:
                        report_file.write(f"   Note: FPS converted from {fps} to {target_fps}\n")
                    report_file.write("\n")

                report_file.write("Summary:\n")
                report_file.write("-------\n")
                report_file.write(f"Sum of original frames: {total_original_frames}\n")
                report_file.write(f"Frames in final video: {final_frames}\n")
                report_file.write(f"Difference: {final_frames - total_original_frames} frames\n\n")

                if total_original_frames != final_frames:
                    report_file.write(
                        "Note: Frame count difference is expected in Precise mode due to FPS conversion\n"
                    )
                    # Calcular a diferença percentual
                    if total_original_frames > 0:
                        percent_diff = abs(
                            (final_frames - total_original_frames) / total_original_frames * 100
                        )
                        report_file.write(f"Percent difference: {percent_diff:.2f}%\n")
                else:
                    report_file.write(
                        "UNUSUAL: Frame count was exactly preserved despite FPS conversion\n"
                    )

                report_file.write(
                    f"All videos reencoded to {target_fps} FPS for consistent playback\n"
                )

            # If process succeeded, clean up temp files and show success message
            if process.returncode == 0:
                # Clean up temp files automatically to save disk space
                self.root.after(
                    0,
                    lambda: self.progress_label.config(text="Cleaning up temporary files..."),
                )

                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Error removing temp file {temp_file}: {str(e)}")

                try:
                    os.remove(filelist_path)
                    os.rmdir(temp_dir)
                except Exception as e:
                    print(f"Error removing temp directory: {str(e)}")

                # Pass the output video path directly to the lambda
                output_path = output_video_path
                report_path = frame_report_path
                self.root.after(
                    0,
                    lambda path=output_path, report=report_path: self.merge_complete(
                        True, f"{path}\n\nFrame report: {report}"
                    ),
                )
            else:
                # Capturar o código de retorno para a lambda
                return_code = process.returncode
                self.root.after(
                    0,
                    lambda code=return_code: self.merge_complete(
                        False, f"FFmpeg error: return code {code}"
                    ),
                )

        except Exception as e:
            import traceback

            print(f"ERROR in precise mode: {str(e)}")
            print(traceback.format_exc())
            # Capture error message as string to avoid issues with 'e' variable
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.merge_complete(False, msg))

    def do_fast_merge(self, output_video_path, output_subdir):
        """Execute fast merging using direct concat with copy codec"""
        try:
            print(f"DEBUG: Iniciando mesclagem rápida com {len(self.video_files)} vídeos")
            print(f"DEBUG: Diretório de saída: {output_subdir}")
            print(f"DEBUG: Arquivo de saída: {output_video_path}")

            # Check each video file before adding
            for i, video_path in enumerate(self.video_files):
                if not os.path.exists(video_path):
                    print(f"ERROR: Video {i + 1} not found: {video_path}")
                    raise FileNotFoundError(f"Vídeo não encontrado: {video_path}")

                # Check if the video can be read by FFmpeg
                try:
                    probe_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=codec_name",
                        "-of",
                        "csv=p=0",
                        video_path,
                    ]
                    print(f"DEBUG: Verificando codec do vídeo {i + 1}: {' '.join(probe_cmd)}")
                    result = subprocess.run(probe_cmd, capture_output=True, text=True, timeout=10)

                    if result.returncode != 0 or not result.stdout.strip():
                        print(f"ERROR: Unable to read codec of video {i + 1}")
                        print(f"ERROR stderr: {result.stderr}")
                        raise ValueError(f"Video {i + 1} cannot be read by FFmpeg")

                    codec = result.stdout.strip()
                    print(f"DEBUG: Video {i + 1} codec: {codec}")
                except subprocess.TimeoutExpired:
                    print(f"ERROR: Timeout checking video {i + 1}")
                    raise TimeoutError(f"Timeout checking video {i + 1}")
                except Exception as probe_error:
                    print(f"ERROR: Error checking video {i + 1}: {str(probe_error)}")
                    raise

                # Update progress with captured value
                current_i = i
                self.root.after(
                    0,
                    lambda i=current_i: self.progress_label.config(
                        text=f"Checking video {i + 1}/{len(self.video_files)}..."
                    ),
                )

            # Create a filelist for concat
            temp_dir = os.path.join(output_subdir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Try a different approach: first copy the files to the temporary directory
            # to avoid problems with paths containing special characters
            copied_files = []

            for i, video_path in enumerate(self.video_files):
                # Create a simple filename to avoid problems with special characters
                temp_file = os.path.normpath(os.path.join(temp_dir, f"input_{i}.mp4"))
                copied_files.append(temp_file)

                # Copy the file using FFmpeg (more reliable than copying directly)
                copy_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    video_path,
                    "-c",
                    "copy",
                    "-v",
                    "warning",
                    temp_file,
                ]

                print(f"DEBUG: Copying video {i + 1} to temporary directory: {' '.join(copy_cmd)}")

                # Update progress
                current_i = i
                self.root.after(
                    0,
                    lambda i=current_i: self.progress_label.config(
                        text=f"Preparing video {i + 1}/{len(self.video_files)}..."
                    ),
                )

                result = subprocess.run(copy_cmd, capture_output=True, text=True)

                if result.returncode != 0:
                    print(f"ERRO ao copiar vídeo {i + 1}: {result.stderr}")
                    raise Exception(f"Erro ao copiar vídeo {i + 1}: {result.stderr}")

                print(f"DEBUG: Vídeo {i + 1} copiado com sucesso para {temp_file}")

            # Create the filelist with the copied files
            filelist_path = os.path.join(temp_dir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for temp_file in copied_files:
                    # Use relative paths to avoid problems
                    rel_path = os.path.relpath(temp_file, temp_dir)
                    f.write(f"file '{rel_path}'\n")

            print(f"DEBUG: Arquivo de lista criado: {filelist_path}")

            # Use the concat demuxer method that is simpler and more reliable
            ffmpeg_command = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist_path,
                "-c",
                "copy",  # Copy without reencoding
                "-stats",  # Show simple statistics
                output_video_path,
            ]

            print(f"DEBUG: Comando FFmpeg: {' '.join(ffmpeg_command)}")

            # Update progress
            self.root.after(
                0,
                lambda: self.progress_label.config(
                    text="Executando mesclagem rápida (sem recodificação)..."
                ),
            )

            # Execute the command and capture output
            try:
                process = subprocess.Popen(
                    ffmpeg_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    universal_newlines=True,
                    bufsize=1,
                    encoding="utf-8",
                    errors="replace",  # Handle problematic characters
                )

                # Collect all stderr output for debugging
                stderr_output = []

                # Read output for progress updates
                for line in process.stderr:
                    stderr_output.append(line)
                    if "frame=" in line or "time=" in line:
                        # Capturar a linha atual para a lambda
                        current_line = line.strip()
                        self.root.after(
                            0,
                            lambda l=current_line: self.progress_label.config(
                                text=f"Mesclando: {l}"
                            ),
                        )

                process.wait()

                # Log the output for debugging
                print(f"DEBUG: FFmpeg código de retorno: {process.returncode}")
                if process.returncode != 0:
                    print("DEBUG: FFmpeg erro (stderr):")
                    for line in stderr_output:
                        print(line.strip())

            except Exception as subprocess_error:
                print(f"ERRO no subprocess: {str(subprocess_error)}")
                raise subprocess_error

            # Após a conclusão do processamento, verificar o número de frames do vídeo final
            verify_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                output_video_path,
            ]

            # Tentar obter a contagem de frames do vídeo final
            final_frames = 0
            try:
                result = subprocess.run(verify_cmd, capture_output=True, text=True, timeout=60)
                if result.returncode == 0 and result.stdout.strip():
                    final_frames = int(result.stdout.strip())
                else:
                    print(f"WARNING: Could not count frames in final video: {result.stderr}")
            except Exception as e:
                print(f"ERROR: Exception counting frames in final video: {str(e)}")

            # Calcular o total de frames dos vídeos originais
            total_original_frames = 0
            for video_path in self.video_files:
                meta = self.video_metadata.get(video_path, {})
                frames = meta.get("frames", 0)
                if frames != "Unknown":
                    total_original_frames += int(frames)
                else:
                    print(f"WARNING: Unknown frame count for {video_path}")

            # Create log file
            base_name = os.path.splitext(os.path.basename(output_video_path))[0]
            log_file_path = os.path.join(output_subdir, f"{base_name}_merge_info.txt")
            frame_report_path = os.path.join(output_subdir, f"{base_name}_frame_report.txt")

            with open(log_file_path, "w", encoding="utf-8") as log_file:
                log_file.write(f"Merged Video: {output_video_path}\n")
                log_file.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write("Merge Mode: Fast (direct concat without reencoding)\n\n")
                log_file.write(f"Sum of original frames: {total_original_frames}\n")
                log_file.write(
                    f"Frames in final video: {final_frames if final_frames > 0 else 'Could not determine'}\n"
                )
                if final_frames > 0:
                    log_file.write(f"Difference: {final_frames - total_original_frames} frames\n")
                log_file.write("\nVideos merged in this order:\n")
                for i, video_path in enumerate(self.video_files, 1):
                    meta = self.video_metadata.get(video_path, {})
                    resolution = meta.get("resolution", "Unknown")
                    fps = meta.get("fps", "Unknown")
                    codec = meta.get("codec", "Unknown")
                    frames = meta.get("frames", "Unknown")
                    log_file.write(
                        f"{i}. {video_path}\n   [{resolution}, {fps} FPS, {codec}, {frames} frames]\n"
                    )

                    # Add error information, if any
                    if process.returncode != 0:
                        log_file.write("\n\nERROR DURING MERGE PROCESS:\n")
                        for line in stderr_output:
                            log_file.write(line)

            # Adicionar um relatório de frames para o modo rápido
            with open(frame_report_path, "w", encoding="utf-8") as report_file:
                report_file.write("FRAME COUNT REPORT\n")
                report_file.write("==============================\n\n")
                report_file.write(f"Final Video: {base_name}.mp4\n")
                report_file.write("Fast Mode: Direct concatenation without reencoding\n\n")
                report_file.write("Input Videos:\n")
                report_file.write("------------------\n")

                total_frames = 0
                for i, video_path in enumerate(self.video_files, 1):
                    meta = self.video_metadata.get(video_path, {})
                    frames = meta.get("frames", "Unknown")
                    fps = meta.get("fps", "Unknown")
                    video_name = os.path.basename(video_path)
                    if frames != "Unknown":
                        total_frames += int(frames)

                    report_file.write(f"{i}. {video_name}\n")
                    report_file.write(f"   Original Frames: {frames}\n")
                    report_file.write(f"   FPS: {fps}\n")
                    report_file.write("\n")

                report_file.write("Summary:\n")
                report_file.write("-------\n")
                report_file.write(f"Sum of original frames: {total_original_frames}\n")
                report_file.write(
                    f"Frames in final video: {final_frames if final_frames > 0 else 'Could not determine'}\n"
                )

                if final_frames > 0:
                    report_file.write(
                        f"Difference: {final_frames - total_original_frames} frames\n\n"
                    )

                    if total_original_frames == final_frames:
                        report_file.write("\nSUCCESS: The frame count was precisely preserved!\n")
                    else:
                        report_file.write(
                            f"\nWARNING: Difference of {final_frames - total_original_frames} frames in the final video!\n"
                        )
                        # Calcular a diferença percentual
                        if total_original_frames > 0:
                            percent_diff = abs(
                                (final_frames - total_original_frames) / total_original_frames * 100
                            )
                            report_file.write(f"Percent difference: {percent_diff:.2f}%\n")

                report_file.write(
                    "\nNote: In Fast mode, exact frame count should be preserved because no reencoding occurs\n"
                )
                report_file.write(
                    "Different FPS between videos may affect playback unless all videos have identical properties\n"
                )

            # Clean up temp files
            try:
                for temp_file in copied_files:
                    try:
                        os.remove(temp_file)
                        print(f"DEBUG: Temporary file removed: {temp_file}")
                    except Exception as e:
                        print(f"WARNING: Unable to remove temporary file {temp_file}: {str(e)}")

                os.remove(filelist_path)
                print(f"DEBUG: Filelist removed: {filelist_path}")

                os.rmdir(temp_dir)
                print(f"DEBUG: Temporary directory removed: {temp_dir}")
            except Exception as e:
                print(f"ERROR: Error removing temporary files: {str(e)}")

            # If process succeeded, show success message
            if process.returncode == 0:
                print("DEBUG: Fast merge completed successfully!")
                # Pass the output video path directly to the lambda
                output_path = output_video_path
                report_path = frame_report_path
                self.root.after(
                    0,
                    lambda path=output_path, report=report_path: self.merge_complete(
                        True, f"{path}\n\nFrame report: {report}"
                    ),
                )
            else:
                # Capture the error message for the lambda
                error_msg = f"FFmpeg error (code {process.returncode}): {' '.join([line.strip() for line in stderr_output if 'Error' in line])}"
                print(f"ERROR: {error_msg}")
                self.root.after(0, lambda msg=error_msg: self.merge_complete(False, msg))

        except Exception as e:
            import traceback

            print(f"ERROR: Exception in fast merge: {str(e)}")
            print(traceback.format_exc())
            # Capture the error message as a string to avoid problems with the 'e' variable
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.merge_complete(False, msg))

    def do_frame_accurate_merge(self, output_video_path, output_subdir):
        """Execute frame-accurate merging that preserves exact frame count"""
        try:
            # Create temporary directory for intermediate files
            temp_dir = os.path.join(output_subdir, "temp")
            os.makedirs(temp_dir, exist_ok=True)

            # Update progress
            self.root.after(0, lambda: self.progress_label.config(text="Analyzing videos..."))

            # First pass: analyze videos to determine optimal parameters
            # In this mode, we keep each video's original FPS but standardize resolution and codec
            highest_res = [0, 0]
            total_frames = 0
            frame_counts = []

            for i, video_path in enumerate(self.video_files):
                try:
                    print(f"DEBUG: Analyzing video {i + 1}: {video_path}")
                    if not os.path.exists(video_path):
                        print(f"ERROR: Video {i + 1} not found: {video_path}")
                        raise FileNotFoundError(f"Video not found: {video_path}")

                    if video_path in self.video_metadata:
                        metadata = self.video_metadata[video_path]
                        width, height = metadata["width"], metadata["height"]
                        frames = metadata["frames"]
                        print(f"DEBUG: Using existing metadata: {width}x{height}, {frames} frames")
                    else:
                        # Fallback to analysis if metadata isn't available
                        print("DEBUG: Metadata not available, analyzing video...")
                        analyze_cmd = [
                            "ffprobe",
                            "-v",
                            "error",
                            "-select_streams",
                            "v:0",
                            "-show_entries",
                            "stream=width,height,nb_frames",
                            "-of",
                            "csv=p=0",
                            video_path,
                        ]

                        print(f"DEBUG: Comando de análise: {' '.join(analyze_cmd)}")
                        result = subprocess.run(analyze_cmd, capture_output=True, text=True)
                        print(f"DEBUG: Resultado da análise: {result.stdout.strip()}")

                        parts = result.stdout.strip().split(",")
                        if len(parts) < 2:
                            print(f"ERROR: Invalid analysis result: {result.stdout}")
                            print(f"ERROR stderr: {result.stderr}")
                            raise ValueError(f"Invalid analysis result for video {i + 1}")

                        width, height = int(parts[0]), int(parts[1])

                        # Try to get frame count
                        if len(parts) > 2 and parts[2]:
                            frames = int(parts[2])
                            print(f"DEBUG: Frame count obtained: {frames}")
                        else:
                            print("DEBUG: Frame count not available, estimating it...")
                            # If nb_frames not available, estimate it from duration
                            duration_cmd = [
                                "ffprobe",
                                "-v",
                                "error",
                                "-show_entries",
                                "format=duration",
                                "-of",
                                "csv=p=0",
                                video_path,
                            ]
                            fps_cmd = [
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

                            duration_result = subprocess.run(
                                duration_cmd, capture_output=True, text=True
                            )
                            fps_result = subprocess.run(fps_cmd, capture_output=True, text=True)

                            duration = float(duration_result.stdout.strip())
                            r_frame_rate = fps_result.stdout.strip()

                            if "/" in r_frame_rate:
                                num, den = map(int, r_frame_rate.split("/"))
                                fps_value = num / den
                            else:
                                fps_value = float(r_frame_rate)

                            frames = int(duration * fps_value)
                            print(
                                f"DEBUG: Estimated frames: {frames} (duration: {duration}s, fps: {fps_value})"
                            )

                    # Update highest resolution
                    if width * height > highest_res[0] * highest_res[1]:
                        highest_res = [width, height]

                    # Track total frames
                    total_frames += frames
                    frame_counts.append(frames)

                    # Use captured values in lambda
                    current_i = i
                    current_width = width
                    current_height = height
                    current_frames = frames
                    self.root.after(
                        0,
                        lambda i=current_i,
                        w=current_width,
                        h=current_height,
                        f=current_frames: self.progress_label.config(
                            text=f"Analyzed video {i + 1}/{len(self.video_files)}: {w}x{h}, {f} frames"
                        ),
                    )
                except Exception as analysis_error:
                    import traceback

                    print(f"ERROR: Error analyzing video {i + 1}: {str(analysis_error)}")
                    print(traceback.format_exc())
                    raise Exception(f"Error analyzing video {i + 1}: {str(analysis_error)}")

            # Second pass: convert each video to consistent resolution and codec but keep original FPS
            temp_files = []
            segment_info = []

            for i, video_path in enumerate(self.video_files):
                try:
                    # Use captured value in lambda
                    current_i = i
                    self.root.after(
                        0,
                        lambda i=current_i: self.progress_label.config(
                            text=f"Processing video {i + 1}/{len(self.video_files)}..."
                        ),
                    )

                    temp_file = os.path.normpath(os.path.join(temp_dir, f"temp_{i}.mp4"))
                    temp_files.append(temp_file)

                    # Get FPS of this video
                    if video_path in self.video_metadata:
                        fps = self.video_metadata[video_path]["fps"]
                    else:
                        fps_cmd = [
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
                        result = subprocess.run(fps_cmd, capture_output=True, text=True)
                        r_frame_rate = result.stdout.strip()

                        # Parse fractional frame rate
                        if "/" in r_frame_rate:
                            num, den = map(int, r_frame_rate.split("/"))
                            fps = num / den
                        else:
                            fps = float(r_frame_rate)

                    print(f"DEBUG: Processing video {i + 1}: {video_path}")
                    print(
                        f"DEBUG: FPS: {fps}, Target resolution: {highest_res[0]}x{highest_res[1]}"
                    )

                    # Use encoding that preserves frame count exactly
                    convert_cmd = [
                        "ffmpeg",
                        "-y",
                        "-i",
                        video_path,
                        "-c:v",
                        "libx264",
                        "-preset",
                        "medium",
                        # Ensure no frames are dropped or added
                        "-vsync",
                        "passthrough",
                        "-copyts",  # Copy timestamps exactly
                        # Remove any potential padding frames
                        "-vf",
                        f"scale={highest_res[0]}:{highest_res[1]}:force_original_aspect_ratio=decrease,pad={highest_res[0]}:{highest_res[1]}:(ow-iw)/2:(oh-ih)/2,setpts=PTS-STARTPTS",
                        "-pix_fmt",
                        "yuv420p",
                        "-an",  # Remove audio
                        # Additional options to ensure frame count preservation
                        "-fflags",
                        "+genpts",
                        "-start_at_zero",
                        temp_file,
                    ]

                    print(f"DEBUG: Conversion command: {' '.join(convert_cmd)}")

                    process = subprocess.Popen(
                        convert_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        universal_newlines=True,
                        bufsize=1,
                    )

                    # Collect all stderr output for debugging
                    stderr_output = []

                    # Read output for progress updates
                    for line in process.stderr:
                        stderr_output.append(line)
                        if "frame=" in line or "time=" in line:
                            # Capturar a linha atual para a lambda
                            current_line = line.strip()
                            current_i = i
                            self.root.after(
                                0,
                                lambda i=current_i, l=current_line: self.progress_label.config(
                                    text=f"Processing video {i + 1}/{len(self.video_files)}: {l}"
                                ),
                            )

                    process.wait()
                    if process.returncode != 0:
                        print(
                            f"ERROR: Error processing video {i + 1}. Return code: {process.returncode}"
                        )
                        print("FFmpeg error output:")
                        for line in stderr_output:
                            print(line.strip())
                        raise Exception(
                            f"Error processing video {i + 1}: FFmpeg returned code {process.returncode}"
                        )

                    # Verify output frame count matches expected
                    verify_cmd = [
                        "ffprobe",
                        "-v",
                        "error",
                        "-count_frames",
                        "-select_streams",
                        "v:0",
                        "-show_entries",
                        "stream=nb_read_frames",
                        "-of",
                        "csv=p=0",
                        temp_file,
                    ]

                    result = subprocess.run(verify_cmd, capture_output=True, text=True)
                    output_frames = int(result.stdout.strip())

                    # Save segment info
                    segment_info.append(
                        {
                            "file": os.path.basename(video_path),
                            "original_frames": frame_counts[i],
                            "processed_frames": output_frames,
                            "fps": fps,
                        }
                    )

                    # Check if frames match
                    if output_frames != frame_counts[i]:
                        print(
                            f"Warning: Frame count mismatch for video {i + 1}. Expected {frame_counts[i]}, got {output_frames}"
                        )
                except Exception as process_error:
                    import traceback

                    print(f"ERROR: Error processing video {i + 1}: {str(process_error)}")
                    print(traceback.format_exc())
                    raise Exception(f"Error processing video {i + 1}: {str(process_error)}")

            # Create filelist for concat
            filelist_path = os.path.join(temp_dir, "filelist.txt")
            with open(filelist_path, "w") as f:
                for temp_file in temp_files:
                    # Use relative paths
                    rel_path = os.path.relpath(temp_file, temp_dir)
                    f.write(f"file '{rel_path}'\n")

            # Final merge using the preprocessed files
            self.root.after(
                0,
                lambda: self.progress_label.config(text="Merging preprocessed videos..."),
            )

            concat_cmd = [
                "ffmpeg",
                "-y",
                "-f",
                "concat",
                "-safe",
                "0",
                "-i",
                filelist_path,
                # Use copy to maintain exact frames
                "-c",
                "copy",
                output_video_path,
            ]

            print(f"DEBUG: Final concatenation command: {' '.join(concat_cmd)}")

            process = subprocess.Popen(
                concat_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
                bufsize=1,
            )

            # Collect all stderr output for debugging
            stderr_output = []

            for line in process.stderr:
                stderr_output.append(line)
                if "frame=" in line or "time=" in line:
                    # Capture the current line for the lambda
                    current_line = line.strip()
                    self.root.after(
                        0,
                        lambda l=current_line: self.progress_label.config(text=f"Merging: {l}"),
                    )

            process.wait()
            if process.returncode != 0:
                print(f"ERROR: Final merge error. Return code: {process.returncode}")
                print("FFmpeg error output:")
                for line in stderr_output:
                    print(line.strip())
                raise Exception(
                    f"Error during final merge: FFmpeg returned code {process.returncode}"
                )

            # Verify final output frame count
            verify_cmd = [
                "ffprobe",
                "-v",
                "error",
                "-count_frames",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "csv=p=0",
                output_video_path,
            ]

            result = subprocess.run(verify_cmd, capture_output=True, text=True)
            final_frames = int(result.stdout.strip())

            # Write log file with detailed frame count information
            base_name = os.path.splitext(os.path.basename(output_video_path))[0]
            log_file_path = os.path.join(output_subdir, f"{base_name}_merge_info.txt")
            frame_report_path = os.path.join(output_subdir, f"{base_name}_frame_report.txt")

            with open(log_file_path, "w") as log_file:
                log_file.write(f"Merged Video: {output_video_path}\n")
                log_file.write(f"Created on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                log_file.write(f"Target Resolution: {highest_res[0]}x{highest_res[1]}\n")
                log_file.write("Merge Mode: Frame Accurate (preserves exact frame counts)\n\n")
                log_file.write(f"Sum of original frames: {total_frames}\n")
                log_file.write(f"Frames in final video: {final_frames}\n")
                log_file.write(f"Difference: {final_frames - total_frames} frames\n\n")
                log_file.write("Videos merged in this order:\n")

                for i, video_path in enumerate(self.video_files, 1):
                    meta = self.video_metadata.get(video_path, {})
                    resolution = meta.get("resolution", "Unknown")
                    fps = meta.get("fps", "Unknown")
                    codec = meta.get("codec", "Unknown")
                    frames = segment_info[i - 1]["original_frames"]
                    processed_frames = segment_info[i - 1]["processed_frames"]

                    log_file.write(f"{i}. {video_path}\n")
                    log_file.write(
                        f"   [{resolution}, {fps} FPS, {codec}, Original Frames: {frames}, Processed Frames: {processed_frames}]\n"
                    )

            # Write dedicated frame report file
            with open(frame_report_path, "w") as report_file:
                report_file.write("FRAME COUNT REPORT\n")
                report_file.write("==============================\n\n")
                report_file.write(f"Final Video: {base_name}.mp4\n")
                report_file.write(f"Total Frames in Final Video: {final_frames}\n\n")
                report_file.write("Input Videos:\n")
                report_file.write("------------------\n")

                for i, info in enumerate(segment_info, 1):
                    report_file.write(f"{i}. {info['file']}\n")
                    report_file.write(f"   Original Frames: {info['original_frames']}\n")
                    report_file.write(f"   Processed Frames: {info['processed_frames']}\n")
                    report_file.write(f"   FPS: {info['fps']}\n")
                    if info["original_frames"] != info["processed_frames"]:
                        report_file.write(
                            f"   WARNING: Difference of {info['processed_frames'] - info['original_frames']} frames!\n"
                        )
                    report_file.write("\n")

                report_file.write("Summary:\n")
                report_file.write("-------\n")
                report_file.write(f"Sum of original frames: {total_frames}\n")
                report_file.write(f"Frames in final video: {final_frames}\n")
                report_file.write(f"Difference: {final_frames - total_frames} frames\n")

                if total_frames == final_frames:
                    report_file.write("\nSUCCESS: The frame count was preserved with precision!\n")
                else:
                    report_file.write(
                        f"\nWARNING: Difference of {final_frames - total_frames} frames in the final video!\n"
                    )
                    # Calcular a diferença percentual
                    if total_frames > 0:
                        percent_diff = abs((final_frames - total_frames) / total_frames * 100)
                        report_file.write(f"Percent difference: {percent_diff:.2f}%\n")

                report_file.write(
                    "\nNote: In Frame Accurate mode, the frame count should be preserved exactly\n"
                )
                report_file.write(
                    "Different FPS between videos may affect playback unless all videos have identical properties\n"
                )

            # Clean up temp files
            try:
                for temp_file in temp_files:
                    try:
                        os.remove(temp_file)
                    except Exception as e:
                        print(f"Error removing temp file {temp_file}: {str(e)}")

                os.remove(filelist_path)
                print(f"DEBUG: Filelist removed: {filelist_path}")

                os.rmdir(temp_dir)
                print(f"DEBUG: Temporary directory removed: {temp_dir}")
            except Exception as e:
                print(f"ERROR: Error removing temporary files: {str(e)}")

            # If process succeeded, show success message
            if process.returncode == 0:
                print("DEBUG: Frame Accurate merge completed successfully!")
                # Pass the output video path directly to the lambda
                output_path = output_video_path
                report_path = frame_report_path
                self.root.after(
                    0,
                    lambda path=output_path, report=report_path: self.merge_complete(
                        True, f"{path}\n\nFrame report: {report}"
                    ),
                )
            else:
                # Capture the return code for the lambda
                return_code = process.returncode
                self.root.after(
                    0,
                    lambda code=return_code: self.merge_complete(
                        False, f"FFmpeg error: return code {code}"
                    ),
                )

        except Exception as e:
            # Capture the error message as a string to avoid problems with the 'e' variable
            error_message = str(e)
            self.root.after(0, lambda msg=error_message: self.merge_complete(False, msg))

    def show_mode_help(self, mode):
        """Show help information for the selected merge mode"""
        try:
            if mode == "frame_accurate":
                title = "Frame Accurate Mode"
                message = (
                    "This mode preserves the exact number of frames from each original video.\n\n"
                    "Features:\n"
                    "• Maintains each video's original frame rate\n"
                    "• Standardizes resolution and codec only\n"
                    "• Creates detailed frame count reports\n"
                    "• Verifies frame counts match original videos\n\n"
                    "Best for:\n"
                    "• Biomechanical analysis\n"
                    "• Scientific research requiring precise frame timing\n"
                    "• Motion analysis where exact frame counts are critical\n"
                    "• When you need to know exactly where each frame came from"
                )
            elif mode == "precise":
                title = "Precise Mode"
                message = (
                    "This mode creates a consistent high-quality output with standardized properties.\n\n"
                    "Features:\n"
                    "• Reencodes all videos to the highest frame rate found\n"
                    "• Standardizes resolution to the highest found\n"
                    "• May change original frame counts when increasing frame rates\n"
                    "• Prioritizes visual quality and consistency\n\n"
                    "Best for:\n"
                    "• General video production\n"
                    "• Presentations and demonstrations\n"
                    "• When visual smoothness is more important than exact frame preservation\n"
                    "• When working with videos that have different properties"
                )
            else:  # fast mode
                title = "Fast Mode"
                message = (
                    "This mode directly concatenates videos without reencoding.\n\n"
                    "Features:\n"
                    "• Much faster processing (no reencoding)\n"
                    "• Preserves original video data exactly\n"
                    "• No quality loss from recompression\n"
                    "• May cause playback issues if videos have different properties\n\n"
                    "Best for:\n"
                    "• Quick merging of videos with identical properties\n"
                    "• When processing time is a priority\n"
                    "• When you want to avoid any quality loss from recompression\n"
                    "• Working with already-compatible video files"
                )

            # Use after to ensure the message is displayed in the main thread
            self.root.after(0, lambda t=title, m=message: messagebox.showinfo(t, m))

        except Exception as e:
            print(f"Error showing mode help: {str(e)}")
            # Use after to ensure the error message is displayed in the main thread
            error_message = str(e)
            self.root.after(
                0,
                lambda msg=error_message: messagebox.showerror(
                    "Error", f"Failed to show help: {msg}"
                ),
            )

    def merge_complete(self, success, message):
        """Handle completion of the merge process"""
        try:
            print(f"DEBUG: Finalizing merge process. Success: {success}")

            # Stop progress bar
            try:
                self.progress_bar.stop()
                self.progress_bar.pack_forget()
                self.progress_label.pack_forget()
                print("DEBUG: Progress bar stopped and removed")
            except Exception as e:
                print(f"Error stopping progress bar: {str(e)}")

            if success:
                print(f"DEBUG: Merge completed successfully: {message}")
                messagebox.showinfo("Success", f"Videos merged successfully!\nOutput: {message}")
            else:
                print(f"DEBUG: Error merging videos: {message}")
                messagebox.showerror("Error", f"Failed to merge videos: {message}")
        except Exception as e:
            import traceback

            print(f"Error finishing merge process: {str(e)}")
            print(traceback.format_exc())
            # Try to show a basic error message
            try:
                messagebox.showerror(
                    "Error",
                    f"An error occurred while completing the merge process: {str(e)}",
                )
            except:
                print("Could not show error message")

    def create_mode_buttons(self):
        """Create buttons for merge mode selection instead of radiobuttons"""
        # First create a label frame to hold the buttons
        self.mode_buttons_frame = ttk.Frame(self.mode_frame)
        self.mode_buttons_frame.pack(fill=tk.X, pady=5)

        # Create a label to show the currently selected mode
        self.mode_label = ttk.Label(
            self.mode_buttons_frame,
            text="Selected: Frame Accurate Mode",
            font=("Helvetica", 10, "bold"),
        )
        self.mode_label.pack(fill=tk.X, pady=5)

        # Create each mode button
        modes = [
            (
                "frame_accurate",
                "Frame Accurate Mode (Keeps the exact frame count - Ideal for biomechanics)",
            ),
            (
                "precise",
                "Precise Mode (Reencodes all videos - Best quality, consistent FPS)",
            ),
            (
                "fast",
                "Fast Mode (Direct concat - Faster but may have issues with different videos)",
            ),
        ]

        # Create a style for selected button
        style = ttk.Style()
        style.configure("Selected.TButton", background="lightblue", font=("Helvetica", 9, "bold"))
        style.configure("Normal.TButton", font=("Helvetica", 9))

        self.mode_buttons = []

        for mode, text in modes:
            button_frame = ttk.Frame(self.mode_buttons_frame)
            button_frame.pack(fill=tk.X, anchor=tk.W, padx=5, pady=2)

            # Print for debug
            print(f"DEBUG: Creating button for mode: '{mode}'")

            # Each button calls select_mode with its corresponding mode
            button = ttk.Button(
                button_frame,
                text=text,
                style="Normal.TButton",
                command=lambda m=mode: self.select_mode(m),
            )
            button.pack(side=tk.LEFT, anchor=tk.W, fill=tk.X, expand=True)
            self.mode_buttons.append((mode, button))

            help_button = ttk.Button(
                button_frame,
                text="?",
                width=2,
                command=lambda m=mode: self.show_mode_help(m),
            )
            help_button.pack(side=tk.LEFT, padx=5)

        # Highlight the default mode button
        self.update_mode_buttons()

    def select_mode(self, mode):
        """Explicitly set the merge mode"""
        print(f"DEBUG: Selected mode explicitly: '{mode}'")
        self.selected_mode = mode

        # Update the label that shows the selected mode
        mode_names = {
            "frame_accurate": "Frame Accurate Mode",
            "precise": "Precise Mode",
            "fast": "Fast Mode",
        }
        mode_display_name = mode_names.get(mode, "Unknown Mode")
        self.mode_label.config(text=f"Selected: {mode_display_name}")

        # Atualizar o título da janela com o modo selecionado
        self.root.title(f"Merge Multiple Videos - {mode_display_name}")

        # Update the appearance of mode buttons
        self.update_mode_buttons()

    def update_mode_buttons(self):
        """Update the appearance of mode buttons"""
        for mode, button in self.mode_buttons:
            if mode == self.selected_mode:
                button.config(style="Selected.TButton")
            else:
                button.config(style="Normal.TButton")


def run_merge_multivideos():
    """Main function to run the multi-video merger application"""
    try:
        print(f"Running script: {os.path.basename(__file__)}")
        print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
        print("Starting multi-video merger...")

        root = tk.Tk()

        # Configuração mais robusta para Windows 11
        root.attributes("-topmost", True)
        root.lift()  # Levanta a janela acima de todas as outras
        root.focus_force()  # Força o foco
        root.deiconify()  # Garante que a janela não está minimizada

        # Pequeno delay para garantir que a janela seja exibida corretamente
        root.after(100, lambda: root.attributes("-topmost", False))

        app = VideoMergeApp(root)
        root.mainloop()
    except Exception as e:
        import traceback

        print(f"FATAL ERROR in application: {str(e)}")
        print(traceback.format_exc())
        try:
            messagebox.showerror("Fatal Error", f"A fatal error occurred: {str(e)}")
        except:
            print("Could not show error message")


if __name__ == "__main__":
    run_merge_multivideos()
