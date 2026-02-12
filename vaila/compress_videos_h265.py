"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

compress_videos_h265.py

Description:
This script compresses videos in a specified directory to H.265/HEVC format using the FFmpeg tool.
It provides a GUI for selecting the directory containing the videos, and processes each video,
saving the compressed versions in a subdirectory named 'compressed_h265'.
The script supports GPU acceleration using NVIDIA NVENC if available, or falls back to CPU encoding
with libx265.

The script has been updated to work on Windows, Linux, and macOS.
It includes cross-platform detection of NVIDIA GPUs to utilize GPU acceleration where possible.
On systems without an NVIDIA GPU (e.g., macOS), the script defaults to CPU-based compression.

Usage:
- Run the script to open a GUI, select the directory containing the videos, and the compression process
  will start automatically.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- The script is designed to work on Windows, Linux, and macOS.

Dependencies:
- Python 3.x
- Tkinter (included with Python)
- FFmpeg (available in PATH)

NVIDIA GPU Installation and FFmpeg NVENC Support

To use NVIDIA GPU acceleration for video encoding in FFmpeg, follow the steps below for your operating system:

## Windows:
1. **Install NVIDIA Drivers**:
   - Download and install the latest NVIDIA drivers from the official site: https://www.nvidia.com/Download/index.aspx.
   - Ensure your GPU supports NVENC (Kepler series or newer).

2. **Install FFmpeg**:
   - Download the FFmpeg build with NVENC support from: https://www.gyan.dev/ffmpeg/builds/.
   - Extract the files, add the `bin` directory to your system's PATH, and verify installation by running:
     ```bash
     ffmpeg -encoders | findstr nvenc
     ```
   - Look for `h264_nvenc` and `hevc_nvenc` in the output.

## Linux:
1. **Install NVIDIA Drivers**:
   - Install the appropriate NVIDIA drivers for your GPU. For Ubuntu, you can add the graphics drivers PPA:
     ```bash
     sudo add-apt-repository ppa:graphics-drivers/ppa
     sudo apt update
     sudo apt install nvidia-driver-<version>
     ```
   - Verify your GPU and driver installation with:
     ```bash
     nvidia-smi
     ```

2. **Install CUDA Toolkit (if necessary)**:
   - Download and install the CUDA Toolkit from: https://developer.nvidia.com/cuda-downloads.
   - Follow the installation instructions for your Linux distribution.

3. **Install or Compile FFmpeg with NVENC Support**:
   - Some Linux distributions provide FFmpeg packages with NVENC support. Check if `h265_nvenc` is available:
     ```bash
     ffmpeg -encoders | grep nvenc
     ```
   - If not available, you may need to compile FFmpeg with NVENC support:
     ```bash
     sudo apt install build-essential pkg-config
     sudo apt build-dep ffmpeg
     git clone https://git.ffmpeg.org/ffmpeg.git ffmpeg/
     cd ffmpeg
     ./configure --enable-gpl --enable-nonfree --enable-cuda-nvcc --enable-libnpp --enable-libx264 --enable-libx265 --enable-nvenc --enable-cuvid --enable-cuda
     make -j$(nproc)
     sudo make install
     ```

## macOS:
- Recent versions of macOS do not support NVIDIA GPUs; NVENC acceleration is not available.
- The script will default to CPU-based encoding on macOS.

Note:
- Ensure that FFmpeg is installed and accessible in your system PATH.
- This process may take several hours depending on the size of the videos and the performance of your computer.
"""

import os
import platform
import subprocess
import tempfile
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

from rich import print

# Global variables for success and failure counts
success_count = 0
failure_count = 0


def is_nvidia_gpu_available():
    """Check if an NVIDIA GPU is available in the system."""
    os_type = platform.system()

    try:
        if os_type == "Windows":
            # Specific command for Windows
            result = subprocess.run(
                ["wmic", "path", "win32_VideoController", "get", "name"],
                capture_output=True,
                text=True,
                check=True,
            )
            return "NVIDIA" in result.stdout
        elif os_type == "Linux":
            # Specific command for Linux
            result = subprocess.run(["lspci"], capture_output=True, text=True, check=True)
            return "NVIDIA" in result.stdout
        elif os_type == "Darwin":  # macOS
            # On macOS, there are no recent NVIDIA GPUs supported
            return False
        else:
            # Unsupported operating system
            return False
    except Exception as e:
        print(f"Error checking for NVIDIA GPU: {e}")
        return False


def find_videos(directory):
    """Find all video files in the specified directory without searching subdirectories."""
    video_files = []
    for file in os.listdir(directory):
        if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
            video_files.append(os.path.join(directory, file))
    return video_files


def create_temp_file_with_videos(video_files):
    """Create a temporary file containing the list of video files to process."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    for video in video_files:
        temp_file.write(f"{video}\n")
    temp_file.close()
    return temp_file.name


def run_compress_videos_h265(input_list, output_dir, preset, crf, resolution, use_gpu):
    """Compress the list of video files stored in the temporary file to H.265/HEVC format."""
    global success_count, failure_count
    success_count = 0
    failure_count = 0

    print("\n[DEBUG] Compression Parameters:")
    print(f"[DEBUG] - Preset: {preset}")
    print(f"[DEBUG] - CRF: {crf}")
    print(f"[DEBUG] - Resolution: {resolution}")
    print(f"[DEBUG] - Use GPU: {use_gpu}")

    print("!!!ATTENTION!!!")
    print(
        "This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!"
    )

    os.makedirs(output_dir, exist_ok=True)

    # Read the video files from the temp file
    with open(input_list) as temp_file:
        video_paths = [line.strip() for line in temp_file]

    for video_path in video_paths:
        output_path = os.path.join(
            output_dir, os.path.splitext(os.path.basename(video_path))[0] + "_h265.mp4"
        )

        try:
            # Get original video resolution for debug
            try:
                cmd_probe = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    video_path,
                ]
                original_resolution = subprocess.check_output(cmd_probe).decode().strip()
                print(f"[DEBUG] Original video resolution: {original_resolution}")
            except Exception as e:
                print(f"[DEBUG] Error getting original resolution: {str(e)}")
                original_resolution = "unknown"

            # Base command
            cmd = ["ffmpeg", "-y", "-i", video_path]

            # Add scale filter if resolution is not original
            if resolution != "original":
                scale_filter = f"scale={resolution.replace('x', ':')}:force_original_aspect_ratio=decrease,pad={resolution.replace('x', ':')}:(ow-iw)/2:(oh-ih)/2"
                print(f"[DEBUG] Applying scale filter: {scale_filter}")
                cmd.extend(["-vf", scale_filter])
            else:
                print("[DEBUG] Keeping original resolution (no scale filter)")

            # Add encoding settings based on GPU availability
            if use_gpu:
                cmd.extend(
                    [
                        "-c:v",
                        "hevc_nvenc",
                        "-preset",
                        preset,
                        "-b:v",
                        "5M",
                        "-maxrate",
                        "5M",
                        "-bufsize",
                        "10M",
                    ]
                )
            else:
                cmd.extend(
                    [
                        "-c:v",
                        "libx265",
                        "-preset",
                        preset,
                        "-crf",
                        str(crf),
                    ]
                )

            # Add audio settings and output path
            cmd.extend(["-c:a", "copy", output_path])

            print("\n[DEBUG] Complete ffmpeg command:")
            print(" ".join(cmd))

            print(f"\nProcessing: {os.path.basename(video_path)}")
            subprocess.run(cmd, check=True)

            # Verify output video resolution
            try:
                cmd_check = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "csv=s=x:p=0",
                    output_path,
                ]
                output_resolution = subprocess.check_output(cmd_check).decode().strip()
                print(f"[DEBUG] Output video resolution: {output_resolution}")
            except Exception as e:
                print(f"[DEBUG] Error checking output resolution: {str(e)}")

            success_count += 1
            print(f"Successfully compressed: {os.path.basename(video_path)}")

        except subprocess.CalledProcessError as e:
            failure_count += 1
            print(f"Failed to compress: {os.path.basename(video_path)}")
            print(f"Error: {str(e)}")
            print(
                f"[DEBUG] ffmpeg error output: {e.stderr if hasattr(e, 'stderr') else 'Not available'}"
            )


def get_compression_parameters():
    """
    Create a single dialog window where user selects options by entering numbers.
    """
    # Create a dictionary to store parameters
    params = {}

    # Define options for reference
    preset_options = [
        "ultrafast",
        "superfast",
        "veryfast",
        "faster",
        "fast",
        "medium",
        "slow",
        "slower",
        "veryslow",
    ]
    resolution_options = [
        "original",
        "3840x2160",
        "2560x1440",
        "1920x1080",
        "1280x720",
        "854x480",
        "640x360",
    ]

    # Create dialog window
    dialog = tk.Toplevel()
    dialog.title("Video Compression Settings")
    dialog.grab_set()  # Make modal

    # Main frame with padding
    main_frame = tk.Frame(dialog, padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)

    # Title
    tk.Label(main_frame, text="H.265 Video Compression Settings", font=("Arial", 12, "bold")).grid(
        row=0, column=0, columnspan=2, sticky="w", pady=(0, 15)
    )

    # 1. Preset field with numbered options
    tk.Label(main_frame, text="Preset (enter number):", font=("Arial", 10, "bold")).grid(
        row=1, column=0, sticky="w", pady=5
    )
    preset_var = tk.StringVar(value="5")  # Default to medium (index 5)
    preset_entry = tk.Entry(main_frame, textvariable=preset_var, width=5)
    preset_entry.grid(row=1, column=1, sticky="w", pady=5)

    # Preset help text with numbered options
    preset_help_text = "Options:\n"
    for i, preset in enumerate(preset_options, 1):
        preset_help_text += f"{i} = {preset}"
        if i < len(preset_options):
            preset_help_text += "   "
            if i % 3 == 0:  # Break line every 3 options
                preset_help_text += "\n"

    tk.Label(main_frame, text=preset_help_text, font=("Arial", 8, "italic"), justify="left").grid(
        row=2, column=0, columnspan=2, sticky="w", padx=20
    )

    # 2. CRF field (keep as is - already a number)
    tk.Label(main_frame, text="CRF Value (0-51):", font=("Arial", 10, "bold")).grid(
        row=3, column=0, sticky="w", pady=5
    )
    crf_var = tk.StringVar(value="28")  # Default CRF for H.265 is higher than H.264
    crf_entry = tk.Entry(main_frame, textvariable=crf_var, width=5)
    crf_entry.grid(row=3, column=1, sticky="w", pady=5)

    # CRF help text
    tk.Label(
        main_frame,
        text="Lower = better quality, larger files",
        font=("Arial", 8, "italic"),
    ).grid(row=4, column=0, columnspan=2, sticky="w", padx=20)

    # 3. Resolution field with numbered options
    tk.Label(main_frame, text="Resolution (enter number):", font=("Arial", 10, "bold")).grid(
        row=5, column=0, sticky="w", pady=5
    )
    resolution_var = tk.StringVar(value="1")  # Default to original (index 1)
    resolution_entry = tk.Entry(main_frame, textvariable=resolution_var, width=5)
    resolution_entry.grid(row=5, column=1, sticky="w", pady=5)

    # Resolution help text with numbered options
    resolution_help_text = "Options:\n"
    for i, res in enumerate(resolution_options, 1):
        resolution_help_text += f"{i} = {res}"
        if i < len(resolution_options):
            resolution_help_text += "   "
            if i % 2 == 0:  # Break line every 2 options
                resolution_help_text += "\n"

    tk.Label(
        main_frame,
        text=resolution_help_text,
        font=("Arial", 8, "italic"),
        justify="left",
    ).grid(row=6, column=0, columnspan=2, sticky="w", padx=20)

    # 4. GPU field with numbered options
    tk.Label(main_frame, text="Use GPU (enter number):", font=("Arial", 10, "bold")).grid(
        row=7, column=0, sticky="w", pady=5
    )
    gpu_var = tk.StringVar(value="2")  # Default to No
    gpu_entry = tk.Entry(main_frame, textvariable=gpu_var, width=5)
    gpu_entry.grid(row=7, column=1, sticky="w", pady=5)

    # GPU help text with numbered options
    tk.Label(
        main_frame,
        text="Options: 1 = Yes (NVIDIA GPUs only)   2 = No (CPU encoding)",
        font=("Arial", 8, "italic"),
    ).grid(row=8, column=0, columnspan=2, sticky="w", padx=20)

    # Separator
    tk.Frame(main_frame, height=1, bg="gray").grid(
        row=9, column=0, columnspan=2, sticky="ew", pady=15
    )

    # Button frame
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=10, column=0, columnspan=2, pady=10)

    # Function for OK button
    def on_ok():
        try:
            # Validate preset
            try:
                preset_idx = int(preset_var.get().strip())
                if not (1 <= preset_idx <= len(preset_options)):
                    messagebox.showerror(
                        "Error",
                        f"Preset number must be between 1 and {len(preset_options)}",
                    )
                    return
                preset = preset_options[preset_idx - 1]
            except ValueError:
                messagebox.showerror("Error", "Preset must be a number")
                return

            # Validate CRF
            try:
                crf = int(crf_var.get().strip())
                if not (0 <= crf <= 51):
                    messagebox.showerror("Error", "CRF value must be between 0 and 51")
                    return
            except ValueError:
                messagebox.showerror("Error", "CRF value must be a number")
                return

            # Validate resolution
            try:
                resolution_idx = int(resolution_var.get().strip())
                if not (1 <= resolution_idx <= len(resolution_options)):
                    messagebox.showerror(
                        "Error",
                        f"Resolution number must be between 1 and {len(resolution_options)}",
                    )
                    return
                resolution = resolution_options[resolution_idx - 1]
            except ValueError:
                messagebox.showerror("Error", "Resolution must be a number")
                return

            # Validate GPU choice
            try:
                gpu_choice = int(gpu_var.get().strip())
                if gpu_choice not in [1, 2]:
                    messagebox.showerror("Error", "GPU option must be 1 (Yes) or 2 (No)")
                    return
                use_gpu = gpu_choice == 1
            except ValueError:
                messagebox.showerror("Error", "GPU option must be a number")
                return

            # Store parameters
            params["preset"] = preset
            params["crf"] = crf
            params["resolution"] = resolution
            params["use_gpu"] = use_gpu

            # Show confirmation
            confirm_msg = (
                f"Selected compression settings:\n\n"
                f"• Preset: {preset}\n"
                f"• CRF: {crf}\n"
                f"• Resolution: {resolution}\n"
                f"• Use GPU: {'Yes' if use_gpu else 'No'}\n\n"
                f"Continue with these settings?"
            )

            if messagebox.askyesno("Confirm Settings", confirm_msg):
                dialog.destroy()
            # If No, keep dialog open

        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {str(e)}")

    # Function for Cancel button
    def on_cancel():
        dialog.destroy()

    # Add buttons
    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(side="left", padx=5)

    # Function to show built-in help
    def show_help():
        help_text = """
COMPRESSION SETTINGS HELP:

PRESET: (speed vs quality tradeoff)
1 = ultrafast: Fastest encoding, lowest quality
2 = superfast: Very fast encoding, lower quality
3 = veryfast: Fast encoding, decent quality
4 = faster: Reasonably fast encoding, good quality
5 = fast: Balanced encoding speed and quality
6 = medium: Default preset with good balance
7 = slow: Better quality, slower encoding
8 = slower: Very good quality, much slower encoding
9 = veryslow: Best quality, very slow encoding

CRF VALUE (0-51): (quality setting)
- Lower values = better quality, larger files
- Higher values = lower quality, smaller files
- 0 = lossless (largest files)
- 28 = default value for H.265 (good balance)
- 51 = worst quality (smallest files)
- Note: For same quality, H.265 typically uses ~40% lower CRF than H.264

RESOLUTION: (output size)
1 = original: Keep the original video resolution
2 = 3840x2160: 4K UHD
3 = 2560x1440: 2K QHD
4 = 1920x1080: Full HD
5 = 1280x720: HD
6 = 854x480: SD
7 = 640x360: Low resolution

GPU ACCELERATION:
1 = Yes: Use NVIDIA GPU for encoding (faster)
2 = No: Use CPU for encoding (works on all systems)

NOTE: H.265 encoding is significantly slower than H.264 but produces smaller files
at the same quality level. Expect longer encoding times, especially with CPU encoding.
        """

        help_dialog = tk.Toplevel(dialog)
        help_dialog.title("Compression Settings Help")
        help_dialog.transient(dialog)
        help_dialog.grab_set()

        text_widget = tk.Text(help_dialog, wrap="word", width=70, height=25)
        text_widget.pack(padx=20, pady=20, fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")

        tk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)

        # Center the help dialog
        help_dialog.update_idletasks()
        width = help_dialog.winfo_width()
        height = help_dialog.winfo_height()
        x = dialog.winfo_rootx() + (dialog.winfo_width() // 2) - (width // 2)
        y = dialog.winfo_rooty() + (dialog.winfo_height() // 2) - (height // 2)
        help_dialog.geometry(f"{width}x{height}+{x}+{y}")

    # Add help button
    tk.Button(main_frame, text="Help", command=show_help).grid(
        row=11, column=0, columnspan=2, pady=5
    )

    # Center the dialog on screen
    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")

    # Wait for window to close
    dialog.wait_window()

    # Return parameters if valid, otherwise None
    return params if params else None


def compress_videos_h265_gui():
    """Main function to run the GUI and compression process."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting compress_videos_h265_gui...")

    global success_count, failure_count
    success_count = 0  # Reset counters at the beginning
    failure_count = 0

    # Get compression parameters through dialog
    compression_config = get_compression_parameters()

    # Check if user cancelled
    if not compression_config:
        print("User canceled the operation")
        return

    # Select video directory
    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Create output directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(video_directory, f"compressed_h265_{timestamp}")

    # Check if NVIDIA GPU is available when GPU is selected
    use_gpu = compression_config["use_gpu"] and is_nvidia_gpu_available()
    if compression_config["use_gpu"] and not use_gpu:
        print("GPU acceleration requested but no NVIDIA GPU detected. Using CPU instead.")
        messagebox.showwarning(
            "GPU Not Available",
            "GPU acceleration was requested but no compatible NVIDIA GPU was detected.\n"
            "Compression will proceed using CPU instead.",
        )

    # Find all video files
    video_files = find_videos(video_directory)

    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    # Create a temporary file with the list of video files
    temp_file_path = create_temp_file_with_videos(video_files)

    # Run the compression with user-defined settings
    run_compress_videos_h265(
        temp_file_path,
        output_directory,
        preset=compression_config["preset"],
        crf=compression_config["crf"],
        resolution=compression_config["resolution"],
        use_gpu=use_gpu,
    )

    # Remove temporary file
    os.remove(temp_file_path)

    messagebox.showinfo(
        "Success",
        f"Video compression completed.\n\n{success_count} succeeded, {failure_count} failed.",
    )


if __name__ == "__main__":
    compress_videos_h265_gui()
