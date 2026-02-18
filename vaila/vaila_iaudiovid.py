"""
================================================================================
Video Audio Processing Tool - vaila_iaudiovid.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Create: 01 March 2025
Update: 18 February 2026
Version: 0.1.4

Description:
------------
This script provides audio insertion functionality:

- Insert audio into videos (replace existing audio or add to silent videos)
- Support for looping short audio files to match longer videos
- Batch processing of multiple videos
- User-selectable output directory

Key Features:
- Batch processing of multiple videos
- Audio replacement/insertion for videos
- Preserves original video quality when processing
- User-friendly interface with progress tracking

Requirements:
- ffmpeg (must be installed on your system and in PATH)
- Python 3.6+
- tkinter (for GUI)
================================================================================
"""

import contextlib
import math
import os
import shutil
import subprocess
import sys
import threading
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox, scrolledtext, ttk


class AudioVideoProcessor:
    """Handles the processing of adding/replacing audio in videos."""

    def __init__(self):
        """Initialize the processor."""
        self.output_dir = os.path.join(os.path.expanduser("~"), "Processed_Videos")
        self.video_dir = ""
        self.audio_file = ""
        self.video_files = []
        self.processing_log = []
        self.status_callback = None
        self.progress_callback = None
        self.process_thread = None

        # Check for ffmpeg
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            print("Error: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
            sys.exit(1)

    def _check_ffmpeg(self):
        """Check if ffmpeg is available in the system path."""
        return shutil.which("ffmpeg") is not None

    def set_video_directory(self, directory):
        """Set the directory containing videos to process."""
        self.video_dir = directory
        self.video_files = self._scan_for_videos(directory)
        return len(self.video_files)

    def _scan_for_videos(self, directory):
        """Scan the directory for video files."""
        video_extensions = [".mp4", ".avi", ".mov", ".mkv", ".wmv", ".flv", ".webm"]
        video_files = []

        try:
            for file in os.listdir(directory):
                file_path = os.path.join(directory, file)
                if os.path.isfile(file_path):
                    ext = os.path.splitext(file)[1].lower()
                    if ext in video_extensions:
                        video_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory: {str(e)}")

        return sorted(video_files)

    def set_audio_file(self, audio_file):
        """Set the audio file to insert into videos."""
        self.audio_file = audio_file
        return os.path.exists(audio_file)

    def set_output_directory(self, directory):
        """Set the directory where processed files will be saved."""
        if directory and os.path.isdir(directory):
            self.output_dir = directory
            return True
        return False

    def check_video_has_audio(self, video_file):
        """Check if a video file already has audio streams."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "a",
                "-show_entries",
                "stream=codec_type",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_file,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            # If ffprobe finds audio streams, the output will contain "audio"
            return "audio" in result.stdout

        except Exception as e:
            print(f"Error checking audio in {video_file}: {str(e)}")
            # Assume it has audio if we can't check
            return True

    def get_duration(self, file_path):
        """Get the duration of a media file in seconds using ffprobe."""
        try:
            cmd = [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                file_path,
            ]

            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                check=False,
            )

            if result.returncode == 0 and result.stdout.strip():
                return float(result.stdout.strip())
            return 0
        except Exception as e:
            if self.status_callback:
                self.status_callback(f"Error getting duration: {str(e)}")
            return 0

    def process_video(self, video_file, output_file):
        """Process a single video to add or replace audio."""
        temp_audio_file = None

        try:
            has_audio = self.check_video_has_audio(video_file)
            action = "Replacing audio" if has_audio else "Adding audio"

            if self.status_callback:
                self.status_callback(f"{action} for {os.path.basename(video_file)}")

            # Get durations
            video_duration = self.get_duration(video_file)
            audio_duration = self.get_duration(self.audio_file)

            # Check if video is longer than audio
            if video_duration > audio_duration and audio_duration > 0:
                if self.status_callback:
                    self.status_callback(
                        f"Video ({video_duration:.2f}s) is longer than audio ({audio_duration:.2f}s). Creating looped audio..."
                    )

                # Create a temporary file for the looped audio
                temp_dir = os.path.dirname(output_file)
                os.makedirs(temp_dir, exist_ok=True)
                temp_audio_file = os.path.join(
                    temp_dir,
                    f"temp_looped_audio_{datetime.now().strftime('%Y%m%d%H%M%S')}.wav",
                )

                # Calculate how many loops we need
                loops_needed = int(math.ceil(video_duration / audio_duration))

                if self.status_callback:
                    self.status_callback(
                        f"Creating looped audio with {loops_needed} repetitions..."
                    )

                # Build a complex filter that concatenates the audio file with itself multiple times
                concat_parts = []
                for _i in range(loops_needed):
                    concat_parts.append("[0:a]")

                filter_complex = f"{' '.join(concat_parts)}concat=n={loops_needed}:v=0:a=1[aout]"

                # Create looped audio file using concat filter
                loop_cmd = [
                    "ffmpeg",
                    "-y",
                    "-i",
                    self.audio_file,
                    "-filter_complex",
                    filter_complex,
                    "-map",
                    "[aout]",
                    "-t",
                    str(video_duration),  # Limit duration to video length
                    "-acodec",
                    "pcm_s16le",  # Use uncompressed audio to avoid encoding issues
                    temp_audio_file,
                ]

                loop_process = subprocess.run(
                    loop_cmd,
                    capture_output=True,
                    text=True,
                    check=False,
                )

                if loop_process.returncode == 0:
                    if self.status_callback:
                        self.status_callback("Looped audio created successfully")
                    # Use the temporary looped audio instead of original
                    audio_input = temp_audio_file
                else:
                    if self.status_callback:
                        self.status_callback(
                            f"Error creating looped audio: {loop_process.stderr}. Trying alternative method..."
                        )

                    # Fallback to a simpler approach - concatenate the file with itself
                    temp_list_file = os.path.join(
                        temp_dir,
                        f"temp_list_{datetime.now().strftime('%Y%m%d%H%M%S')}.txt",
                    )

                    # Create a temporary file listing
                    with open(temp_list_file, "w") as f:
                        for _i in range(loops_needed):
                            f.write(f"file '{self.audio_file}'\n")

                    # Use the concat demuxer which is more reliable
                    fallback_cmd = [
                        "ffmpeg",
                        "-y",
                        "-f",
                        "concat",
                        "-safe",
                        "0",
                        "-i",
                        temp_list_file,
                        "-t",
                        str(video_duration),  # Limit duration to video length
                        "-c",
                        "copy",
                        temp_audio_file,
                    ]

                    fallback_process = subprocess.run(
                        fallback_cmd,
                        capture_output=True,
                        text=True,
                        check=False,
                    )

                    with contextlib.suppress(OSError):
                        os.remove(temp_list_file)  # Clean up the temporary list file

                    if fallback_process.returncode == 0:
                        if self.status_callback:
                            self.status_callback(
                                "Looped audio created successfully with fallback method"
                            )
                        audio_input = temp_audio_file
                    else:
                        if self.status_callback:
                            self.status_callback(
                                "Fallback method also failed. Using original audio."
                            )
                        audio_input = self.audio_file
            else:
                # Use original audio file
                audio_input = self.audio_file

            # Command to add/replace audio
            cmd = [
                "ffmpeg",
                "-y",  # Overwrite output file if it exists
                "-i",
                video_file,  # First input: video file
                "-i",
                audio_input,  # Second input: audio file (original or looped)
                "-c:v",
                "copy",  # Copy video codec (no re-encoding)
                "-map",
                "0:v",  # Use video from first input
                "-map",
                "1:a",  # Use audio from second input
                output_file,
            ]

            # Run the ffmpeg command
            process = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True,
            )

            # Process output to track progress
            for line in process.stderr:
                if "time=" in line and self.progress_callback:
                    # Extract time information to track progress
                    time_parts = line.split("time=")[1].split()[0].split(":")
                    if len(time_parts) == 3:
                        hours, minutes, seconds = time_parts
                        progress = float(hours) * 3600 + float(minutes) * 60 + float(seconds)
                        # Update progress (simplified - actual calculation would need video duration)
                        self.progress_callback(progress)

            # Wait for process to complete
            process.wait()

            # Clean up temporary file if it exists
            if temp_audio_file and os.path.exists(temp_audio_file):
                try:
                    os.remove(temp_audio_file)
                    if self.status_callback:
                        self.status_callback("Temporary audio file cleaned up")
                except Exception as e:
                    if self.status_callback:
                        self.status_callback(f"Error removing temporary file: {str(e)}")

            if process.returncode == 0:
                if self.status_callback:
                    self.status_callback(f"Completed: {os.path.basename(output_file)}")
                return True, output_file
            else:
                if self.status_callback:
                    self.status_callback(f"Error processing {os.path.basename(video_file)}")
                return False, None

        except Exception as e:
            # Clean up temporary file if it exists
            if temp_audio_file and os.path.exists(temp_audio_file):
                with contextlib.suppress(BaseException):
                    os.remove(temp_audio_file)

            if self.status_callback:
                self.status_callback(f"Error: {str(e)}")
            return False, None

    def process_all_videos(self):
        """Process all videos in the video directory with the selected audio."""
        if not self.video_files:
            if self.status_callback:
                self.status_callback("No video files found to process")
            return []

        if not self.audio_file or not os.path.exists(self.audio_file):
            if self.status_callback:
                self.status_callback("No valid audio file selected")
            return []

        # Create output directory with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Use the selected output directory plus a subfolder with timestamp
        process_dir = os.path.join(self.output_dir, f"AudioInsert_{timestamp}")
        os.makedirs(process_dir, exist_ok=True)

        # Create log file
        log_file = os.path.join(process_dir, "processing_log.txt")
        with open(log_file, "w", encoding="utf-8") as f:
            f.write(f"Processing started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Audio file: {self.audio_file}\n")
            f.write(f"Number of videos: {len(self.video_files)}\n\n")
            f.write("Results:\n")
            f.write("=" * 60 + "\n")

        processed_files = []

        for i, video_file in enumerate(self.video_files, 1):
            try:
                base_name = os.path.basename(video_file)
                output_file = os.path.join(process_dir, base_name)

                if self.status_callback:
                    self.status_callback(f"Processing ({i}/{len(self.video_files)}): {base_name}")

                # Process the video
                success, actual_output = self.process_video(video_file, output_file)

                # Log the result
                with open(log_file, "a", encoding="utf-8") as f:
                    result = "SUCCESS" if success else "FAILED"
                    f.write(f"{i}. {result}: {base_name}\n")

                if success:
                    processed_files.append(actual_output)
                    self.processing_log.append(f"Successfully processed: {base_name}")
                else:
                    self.processing_log.append(f"Failed to process: {base_name}")

            except Exception as e:
                self.processing_log.append(
                    f"Error processing {os.path.basename(video_file)}: {str(e)}"
                )

                with open(log_file, "a", encoding="utf-8") as f:
                    f.write(f"{i}. ERROR: {os.path.basename(video_file)} - {str(e)}\n")

        # Write completion to log
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(
                "\nProcessing completed: " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n"
            )
            f.write(f"Successfully processed: {len(processed_files)}/{len(self.video_files)}\n")

        if self.status_callback:
            self.status_callback(f"Processing complete. Files saved to: {process_dir}")

        return processed_files

    def start_processing_thread(self):
        """Start processing in a separate thread."""
        if self.process_thread and self.process_thread.is_alive():
            return False  # Already processing

        self.process_thread = threading.Thread(target=self.process_all_videos, daemon=True)
        self.process_thread.start()
        return True


class AudioVideoGUI:
    """GUI for the audio-video processor."""

    def __init__(self, root):
        """Initialize the GUI."""
        self.root = root
        self.root.title("vailá Audio Processing Tool")
        self.root.geometry("800x650")
        self.root.minsize(750, 600)

        # Create processor
        self.processor = AudioVideoProcessor()
        self.processor.status_callback = self.update_status
        self.processor.progress_callback = self.update_progress

        # Main frame com padding reduzido
        main_frame = ttk.Frame(self.root, padding=10)
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Title with vailá in italic
        title_frame = ttk.Frame(main_frame)
        title_frame.pack(pady=(0, 3), fill=tk.X)

        # Use two labels to have vailá in italic and the rest normal
        vaila_label = ttk.Label(title_frame, text="vailá", font=("Arial", 16, "bold", "italic"))
        vaila_label.pack(side=tk.LEFT)

        title_label = ttk.Label(
            title_frame, text=" AUDIO INSERTION TOOL", font=("Arial", 16, "bold")
        )
        title_label.pack(side=tk.LEFT)

        # Description
        desc_label = ttk.Label(
            main_frame,
            text="Add or replace audio in videos (supports looping audio)",
            font=("Arial", 11),
        )
        desc_label.pack(pady=(0, 8))

        # Video directory section
        video_frame = ttk.LabelFrame(main_frame, text="Step 1: Select Video Directory", padding=5)
        video_frame.pack(fill=tk.X, pady=5)

        ttk.Label(video_frame, text="Directory containing video files:").pack(
            anchor=tk.W, pady=(0, 2)
        )

        video_entry_frame = ttk.Frame(video_frame)
        video_entry_frame.pack(fill=tk.X, pady=2)

        self.video_dir_var = tk.StringVar()
        video_entry = ttk.Entry(video_entry_frame, textvariable=self.video_dir_var)
        video_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(video_entry_frame, text="Browse", command=self.browse_video_dir).pack(
            side=tk.RIGHT
        )

        # File counter display
        self.file_count_var = tk.StringVar(value="No videos selected")
        ttk.Label(video_frame, textvariable=self.file_count_var).pack(anchor=tk.W, pady=2)

        # Audio file section
        audio_frame = ttk.LabelFrame(main_frame, text="Step 2: Select Audio File", padding=5)
        audio_frame.pack(fill=tk.X, pady=5)

        ttk.Label(audio_frame, text="Audio file to insert:").pack(anchor=tk.W, pady=(0, 2))

        audio_entry_frame = ttk.Frame(audio_frame)
        audio_entry_frame.pack(fill=tk.X, pady=2)

        self.audio_file_var = tk.StringVar()
        audio_entry = ttk.Entry(audio_entry_frame, textvariable=self.audio_file_var)
        audio_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(audio_entry_frame, text="Browse", command=self.browse_audio_file).pack(
            side=tk.RIGHT
        )

        # Output directory section
        output_frame = ttk.LabelFrame(main_frame, text="Step 3: Select Output Directory", padding=5)
        output_frame.pack(fill=tk.X, pady=5)

        ttk.Label(output_frame, text="Directory to save processed files:").pack(
            anchor=tk.W, pady=(0, 2)
        )

        output_entry_frame = ttk.Frame(output_frame)
        output_entry_frame.pack(fill=tk.X, pady=2)

        self.output_dir_var = tk.StringVar(value=os.path.expanduser("~/Processed_Videos"))
        output_entry = ttk.Entry(output_entry_frame, textvariable=self.output_dir_var)
        output_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=(0, 5))

        ttk.Button(output_entry_frame, text="Browse", command=self.browse_output_dir).pack(
            side=tk.RIGHT
        )

        # Process button
        process_frame = ttk.LabelFrame(main_frame, text="Step 4: Start Processing", padding=5)
        process_frame.pack(fill=tk.X, pady=5)

        self.process_btn = ttk.Button(
            process_frame,
            text="START PROCESSING",
            command=self.start_processing,
            style="Accent.TButton",
        )
        self.process_btn.pack(pady=5)

        # Log display
        log_frame = ttk.LabelFrame(main_frame, text="Processing Log", padding=5)
        log_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.log_text = scrolledtext.ScrolledText(log_frame, height=8, width=80, wrap=tk.WORD)
        self.log_text.pack(fill=tk.BOTH, expand=True)

        # Status bar
        status_frame = ttk.Frame(root, padding=(5, 2))
        status_frame.pack(fill=tk.X, side=tk.BOTTOM)

        self.status_var = tk.StringVar(value="Ready")
        ttk.Label(status_frame, textvariable=self.status_var, font=("Arial", 9, "bold")).pack(
            side=tk.LEFT
        )

        self.progress_bar = ttk.Progressbar(
            status_frame, orient=tk.HORIZONTAL, length=100, mode="determinate"
        )
        self.progress_bar.pack(fill=tk.X, expand=True, side=tk.RIGHT, padx=(10, 0))

        # Initialize
        self.log("Audio insertion tool started. Please select a video directory and audio file.")

    def log(self, message):
        """Add a message to the log display."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.log_text.insert(tk.END, f"[{timestamp}] {message}\n")
        self.log_text.see(tk.END)  # Scroll to the end
        print(f"[LOG] {message}")  # Also print to console

    def update_status(self, message):
        """Update the status message."""
        self.status_var.set(message)
        self.log(message)
        self.root.update_idletasks()

    def update_progress(self, value):
        """Update the progress bar."""
        if value > 100:
            value = 100
        elif value < 0:
            value = 0

        self.progress_bar["value"] = value
        self.root.update_idletasks()

    def browse_video_dir(self):
        """Open directory browser dialog for videos."""
        try:
            directory = filedialog.askdirectory(
                initialdir=os.path.expanduser("~"),
                title="Select Directory Containing Videos",
            )

            if directory:
                self.video_dir_var.set(directory)
                count = self.processor.set_video_directory(directory)

                if count > 0:
                    self.file_count_var.set(f"Found {count} video files")
                    self.log(f"Selected directory: {directory} with {count} videos")
                else:
                    self.file_count_var.set("No video files found in this directory")
                    self.log(f"No video files found in: {directory}")
        except Exception as e:
            self.log(f"Error selecting directory: {str(e)}")

    def browse_audio_file(self):
        """Open file browser dialog for audio file."""
        try:
            file_path = filedialog.askopenfilename(
                initialdir=os.path.expanduser("~"),
                title="Select Audio File",
                filetypes=(
                    ("Audio files", "*.mp3 *.wav *.aac *.m4a *.ogg"),
                    ("All files", "*.*"),
                ),
            )

            if file_path:
                self.audio_file_var.set(file_path)
                if self.processor.set_audio_file(file_path):
                    self.log(f"Selected audio file: {file_path}")
                else:
                    self.log(f"Error: Could not access audio file: {file_path}")
        except Exception as e:
            self.log(f"Error selecting audio file: {str(e)}")

    def browse_output_dir(self):
        """Open directory browser dialog for output directory."""
        try:
            directory = filedialog.askdirectory(
                initialdir=self.output_dir_var.get(), title="Select Output Directory"
            )

            if directory:
                self.output_dir_var.set(directory)
                self.processor.set_output_directory(directory)
                self.log(f"Output directory set to: {directory}")
        except Exception as e:
            self.log(f"Error selecting output directory: {str(e)}")

    def start_processing(self):
        """Start the processing operation."""
        if not self.processor.video_files:
            messagebox.showwarning("Warning", "Please select a directory with video files first.")
            return

        if not self.processor.audio_file:
            messagebox.showwarning("Warning", "Please select an audio file first.")
            return

        # Set the output directory
        output_dir = self.output_dir_var.get()
        if not os.path.isdir(output_dir):
            try:
                os.makedirs(output_dir, exist_ok=True)
                self.log(f"Created output directory: {output_dir}")
            except Exception as e:
                messagebox.showerror("Error", f"Could not create output directory: {str(e)}")
                return

        self.processor.set_output_directory(output_dir)

        # Confirm before proceeding
        confirm = messagebox.askyesno(
            "Confirm Processing",
            f"Ready to insert audio into {len(self.processor.video_files)} videos?\n\n"
            f"Files will be saved to:\n{output_dir}",
        )

        if not confirm:
            return

        # Disable the process button while processing
        self.process_btn.configure(state=tk.DISABLED)

        # Reset progress bar
        self.progress_bar["value"] = 0

        # Start processing thread
        if self.processor.start_processing_thread():
            self.update_status("Processing started...")
            self.root.after(100, self.check_processing)
        else:
            self.update_status("Error: Processing already in progress")
            self.process_btn.configure(state=tk.NORMAL)

    def check_processing(self):
        """Check if processing is still running."""
        if self.processor.process_thread and self.processor.process_thread.is_alive():
            # Still processing, check again after 100ms
            self.root.after(100, self.check_processing)
        else:
            # Processing complete
            self.process_btn.configure(state=tk.NORMAL)

            # Show completion message
            messagebox.showinfo(
                "Processing Complete",
                f"All videos have been processed.\n\n"
                f"Output directory:\n{self.processor.output_dir}",
            )

            # Update progress bar to 100%
            self.progress_bar["value"] = 100


def run_iaudiovid():
    """Main entry point for the application."""
    root = tk.Tk()

    # Add custom style for better visibility
    style = ttk.Style()
    style.configure("TButton", font=("Arial", 10))
    style.configure("Accent.TButton", font=("Arial", 12, "bold"))
    style.configure("TLabel", font=("Arial", 10))
    style.configure("TLabelframe.Label", font=("Arial", 10, "bold"))

    AudioVideoGUI(root)
    root.mainloop()


if __name__ == "__main__":
    run_iaudiovid()
