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
import subprocess
import platform
import tempfile
from tkinter import filedialog, messagebox, Tk

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
            if "NVIDIA" in result.stdout:
                return True
            else:
                return False

        elif os_type == "Linux":
            # Specific command for Linux
            result = subprocess.run(
                ["lspci"], capture_output=True, text=True, check=True
            )
            if "NVIDIA" in result.stdout:
                return True
            else:
                return False

        elif os_type == "Darwin":  # macOS
            # On macOS, there are no recent NVIDIA GPUs supported
            return False

        else:
            # Unsupported operating system
            return False

    except Exception as e:
        print(f"Error checking for NVIDIA GPU: {e}")
        return False


def find_videos_recursively(directory, output_directories):
    """Find all video files recursively in the directory, avoiding the output directories."""
    video_files = []
    for root, dirs, files in os.walk(directory):
        # Ignore the output directories
        if any(
            os.path.abspath(output_dir) in os.path.abspath(root)
            for output_dir in output_directories
        ):
            continue
        for file in files:
            if file.lower().endswith((".mp4", ".avi", ".mov", ".mkv", ".wmv")):
                video_files.append(os.path.join(root, file))
    return video_files


def create_temp_file_with_videos(video_files):
    """Create a temporary file containing the list of video files to process."""
    temp_file = tempfile.NamedTemporaryFile(delete=False, mode="w")
    for video in video_files:
        temp_file.write(f"{video}\n")
    temp_file.close()
    return temp_file.name


def run_compress_videos_h265(
    temp_file_path, output_directory, preset="slower", crf=28, use_gpu=False
):
    """Compress the list of video files stored in the temporary file to H.265/HEVC format."""
    global success_count, failure_count

    print("!!!ATTENTION!!!")
    print(
        "This process might take several hours depending on your computer and the size of your videos. Please be patient or use a high-performance computer!"
    )

    os.makedirs(output_directory, exist_ok=True)

    # Read the video files from the temp file
    with open(temp_file_path, "r") as temp_file:
        video_files = [line.strip() for line in temp_file]

    for video_file in video_files:
        input_path = video_file
        # Create corresponding output file path in the new directory
        relative_path = os.path.relpath(input_path, os.path.dirname(output_directory))
        output_path = os.path.join(
            output_directory, f"{os.path.splitext(relative_path)[0]}_h265.mp4"
        )

        # Ensure output directory for the specific file exists
        os.makedirs(os.path.dirname(output_path), exist_ok=True)

        print(f"Compressing {video_file}...")

        if use_gpu:
            # If GPU is available, use NVIDIA NVENC for encoding
            command = [
                "ffmpeg",
                "-y",  # overwrite output files
                "-i",
                input_path,  # input file
                "-c:v",
                "hevc_nvenc",  # Use NVIDIA NVENC for H.265
                "-preset",
                preset,  # preset for encoding speed
                "-crf",
                str(crf),  # constant rate factor for quality
                output_path,  # output file
            ]
        else:
            # Fallback to CPU-based encoding (libx265)
            command = [
                "ffmpeg",
                "-y",  # overwrite output files
                "-i",
                input_path,  # input file
                "-c:v",
                "libx265",  # video codec for H.265
                "-preset",
                preset,  # preset for encoding speed
                "-crf",
                str(crf),  # constant rate factor for quality
                output_path,  # output file
            ]

        try:
            subprocess.run(command, check=True)
            print(f"Done compressing {video_file} to H.265.")
            success_count += 1
        except subprocess.CalledProcessError as e:
            print(f"Error compressing {video_file}: {e}")
            failure_count += 1

    print(f"Compression completed: {success_count} succeeded, {failure_count} failed.")


def compress_videos_h265_gui():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    global success_count, failure_count
    success_count = 0  # Reset counters at the beginning
    failure_count = 0

    root = Tk()
    root.withdraw()

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    # Create output directory
    output_directory = os.path.join(video_directory, "compressed_h265")

    # Detect system and GPU availability
    os_type = platform.system()
    print(f"Operating System: {os_type}")

    # Check if an NVIDIA GPU is available
    use_gpu = is_nvidia_gpu_available()

    if use_gpu:
        print("NVIDIA GPU detected. Using GPU for video compression.")
    else:
        print("No NVIDIA GPU detected. Using CPU-based compression.")

    # Exclude both 'compressed_h264' and 'compressed_h265' directories
    output_directories = [
        os.path.join(video_directory, "compressed_h264"),
        output_directory,
    ]

    # Find all video files recursively, ignoring the output directories
    video_files = find_videos_recursively(video_directory, output_directories)

    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    # Create a temporary file with the list of video files
    temp_file_path = create_temp_file_with_videos(video_files)

    # Run the compression for all found videos
    preset = "medium"
    crf = 23  # Adjusted for better compression

    run_compress_videos_h265(temp_file_path, output_directory, preset, crf, use_gpu)

    # Remove temporary file
    os.remove(temp_file_path)

    messagebox.showinfo(
        "Completed",
        f"Video compression completed.\n\n{success_count} succeeded, {failure_count} failed.",
    )


if __name__ == "__main__":
    compress_videos_h265_gui()
