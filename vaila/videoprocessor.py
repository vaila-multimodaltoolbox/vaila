"""
videoprocessor.py
vailá - Multimodal Toolbox
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.
Created by Paulo Santiago
Date: 03 April 2025
Updated: 12 January 2026

Description:
This script allows users to process and edit video files, enabling batch processing of videos. Users can choose between three main operations:
1. Merging a video with its reversed version, resulting in a video with double the frames.
2. Splitting each video into two halves and saving only the second half.
3. Merging multiple videos into a single video in a specified order.
The script supports custom text files for batch processing and includes a GUI for directory and file selection.

Key Features:
- Graphical User Interface (GUI) for easy selection of directories and file inputs.
- Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for specifying which videos to process.
- If no text file is provided, the script processes all videos in the source directory.
- Merge option: Creates a video with the original and its reversed version merged.
- Split option: Processes each video to save only the second half.
- Automatic creation of output directories based on a timestamp for organized file management.
- Detailed console output for tracking progress and handling errors.
- Support for merging multiple videos into a single video in a specified order.

Usage:
- Run the script to open a graphical interface.
- Select the operation to perform.
- Select the source directory.
- Select the target directory.
- Select the text file to use for batch processing.
- Click the "Start" button to begin the processing.
- The processed videos will be saved in a new output directory named with a timestamp.

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- Python 3.12.12 environment.
- Tkinter for the GUI components (usually included with Python).
- rich for enhanced console output.
- tqdm for progress tracking.
- pathlib for path manipulation.
- subprocess for subprocess management.
- time for time management.
- json for JSON manipulation.
- os for operating system management.
- tkinter for the GUI components.
- simpledialog for simple dialogs.
- messagebox for message boxes.
- filedialog for file dialogs.
- tqdm for progress tracking.
- rich for enhanced console output.
- pathlib for path manipulation.
- subprocess for subprocess management.

Installation:
uv run videoprocessor.py

License:
Affero General Public License v3.0
https://www.gnu.org/licenses/agpl-3.0.html
Visit the project repository: https://github.com/vaila-multimodaltoolbox
"""

import json
import os
import pathlib
import subprocess
import time
from tkinter import filedialog, messagebox, simpledialog

import tqdm
from rich import print


def check_ffmpeg_installed():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        return True
    except (subprocess.SubprocessError, FileNotFoundError):
        return False


def detect_hardware_encoder():
    """Detect hardware acceleration support with verification"""
    try:
        # First check available encoders
        result = subprocess.run(
            ["ffmpeg", "-encoders"],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
        )

        # Test if NVIDIA is actually working
        if "h264_nvenc" in result.stdout:
            try:
                test_cmd = [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=32x32:r=1:d=1",
                    "-c:v",
                    "h264_nvenc",
                    "-f",
                    "null",
                    "-",
                ]
                test_result = subprocess.run(
                    test_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3,
                )

                if (
                    "Cannot load nvcuda.dll" not in test_result.stderr
                    and "Error" not in test_result.stderr
                ):
                    print("NVIDIA GPU hardware encoding confirmed working (h264_nvenc)")
                    return {
                        "encoder": "h264_nvenc",
                        "quality_param": "preset",
                        "quality_values": {
                            "high": "p7",  # slow (previously "hq")
                            "medium": "p5",  # medium
                            "fast": "p3",  # fast (previously "hp")
                        },
                    }
                else:
                    print("NVIDIA NVENC found but not working, falling back to CPU")
            except Exception as e:
                print(f"NVIDIA NVENC test failed: {e}, falling back to CPU")

        # Check for Intel QSV
        if "h264_qsv" in result.stdout:
            try:
                test_cmd = [
                    "ffmpeg",
                    "-f",
                    "lavfi",
                    "-i",
                    "color=black:s=32x32:r=1:d=1",
                    "-c:v",
                    "h264_qsv",
                    "-f",
                    "null",
                    "-",
                ]
                test_result = subprocess.run(
                    test_cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                    timeout=3,
                )

                if "Error" not in test_result.stderr:
                    print("Intel Quick Sync hardware encoding confirmed working (h264_qsv)")
                    return {
                        "encoder": "h264_qsv",
                        "quality_param": "preset",
                        "quality_values": {
                            "high": "veryslow",
                            "medium": "medium",
                            "fast": "veryfast",
                        },
                    }
                else:
                    print("Intel QSV found but not working, falling back to CPU")
            except Exception as e:
                print(f"Intel QSV test failed: {e}, falling back to CPU")

        # Default to CPU encoding
        print("Using CPU software encoding (libx264)")
        return {
            "encoder": "libx264",
            "quality_param": "preset",
            "quality_values": {
                "high": "p7",  # slow
                "medium": "p5",  # medium
                "fast": "p2",  # veryfast
            },
        }

    except Exception as e:
        print(f"Error detecting encoders: {e}, falling back to libx264")
        return {
            "encoder": "libx264",
            "quality_param": "preset",
            "quality_values": {
                "high": "p7",  # slow
                "medium": "p5",  # medium
                "fast": "p2",  # veryfast
            },
        }


def check_video_size(video_path):
    """Check video size before processing"""
    try:
        result = subprocess.run(
            [
                "ffprobe",
                "-v",
                "error",
                "-show_entries",
                "format=size",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                video_path,
            ],
            stdout=subprocess.PIPE,
            text=True,
        )
        size_bytes = int(result.stdout.strip())
        size_gb = size_bytes / (1024**3)

        if size_gb > 4:  # Limit of 4GB for example
            return False, f"Video is too large ({size_gb:.2f} GB)"
        return True, ""
    except:
        return True, ""  # Proceed in case of error


def process_videos_merge(source_dir, target_dir, use_text_file=False, text_file_path=None):
    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"mergedvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    video_files = []

    # Use provided text file if specified
    if use_text_file and text_file_path:
        with open(text_file_path) as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    video_files.append(os.path.join(source_dir, line.strip()))
    else:
        # No text file provided, process all videos in source_dir
        with os.scandir(source_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv", ".MP4", ".AVI", ".MOV", ".MKV")
                ):
                    video_files.append(entry.path)

    # Detect hardware encoder and available presets - moved outside the loop
    encoder_info = detect_hardware_encoder()
    encoder = encoder_info["encoder"]
    quality_param = encoder_info["quality_param"]
    quality_values = encoder_info["quality_values"]

    # Ask user to choose quality by number (1-9) - moved outside the loop
    quality_msg = (
        "Choose quality level (1-9):\n1-3: Fast (lower quality)\n4-6: Medium\n7-9: High (slower)"
    )
    quality_num = simpledialog.askinteger(
        "Quality Level", quality_msg, minvalue=1, maxvalue=9, initialvalue=5
    )

    # Handle case where user cancels dialog
    if quality_num is None:
        quality_num = 5  # Default to medium quality

    # Map the number to a quality setting - moved outside the loop
    if quality_num <= 3:
        quality = "fast"
    elif quality_num <= 6:
        quality = "medium"
    else:
        quality = "high"

    # Get preset value - moved outside the loop
    preset_value = quality_values[quality]

    # Simplified approach: use numerical presets for libx264 - moved outside the loop
    if encoder == "libx264":
        # Convert p1-p9 to actual preset names
        preset_map = {
            "p1": "ultrafast",
            "p2": "veryfast",
            "p3": "faster",
            "p4": "fast",
            "p5": "medium",
            "p6": "slow",
            "p7": "slower",
            "p8": "veryslow",
            "p9": "placebo",
        }
        preset_value = preset_map.get(preset_value, "medium")

    print(f"Using encoder: {encoder} with {quality_param}={preset_value}")

    # Iterate over video files and apply the merge process
    for video_path in tqdm.tqdm(video_files, desc="Processing videos"):
        try:
            print(f"Processing video: {video_path}")

            # Output video path
            output_video = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_double.mp4",
            )

            # Verificar se já processou este vídeo anteriormente
            if os.path.exists(output_video):
                if not messagebox.askyesno(
                    "File exists",
                    f"Output file already exists:\n{output_video}\n\nOverwrite?",
                ):
                    print(f"Skipping {video_path} (output exists)")
                    continue

            # Prepare command
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter_complex",
                "[0:v]reverse[r];[r][0:v]concat=n=2:v=1:a=0[out]",
                "-map",
                "[out]",
                "-c:v",
                encoder,
                "-threads",
                "4",
                f"-{quality_param}",
                preset_value,
                "-pix_fmt",
                "yuv420p",
                output_video,
            ]

            # Then get metadata
            try:
                # 1. Obter duração do vídeo
                duration_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-show_entries",
                    "format=duration",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                duration = float(
                    subprocess.run(
                        duration_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ).stdout.strip()
                )

                # 2. Obter resolução
                resolution_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=width,height",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                resolution = (
                    subprocess.run(
                        resolution_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    )
                    .stdout.strip()
                    .split("\n")
                )
                width = int(resolution[0])
                height = int(resolution[1])

                # 3. Obter número de frames
                frames_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-count_frames",
                    "-show_entries",
                    "stream=nb_read_frames",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                total_frames = int(
                    subprocess.run(
                        frames_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                        text=True,
                    ).stdout.strip()
                )

                # 4. Obter taxa de quadros
                fps_cmd = [
                    "ffprobe",
                    "-v",
                    "error",
                    "-select_streams",
                    "v:0",
                    "-show_entries",
                    "stream=r_frame_rate",
                    "-of",
                    "default=noprint_wrappers=1:nokey=1",
                    video_path,
                ]
                fps_str = subprocess.run(
                    fps_cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
                ).stdout.strip()
                frame_rate = eval(fps_str)  # Convert "30000/1001" to float

                # Calcular informações do vídeo resultante
                merged_frames = total_frames * 2
                merged_duration = duration * 2
                reverse_start_frame = total_frames

                print(
                    f"Video info: {width}x{height}, {frame_rate} fps, {duration:.2f}s, {total_frames} frames"
                )

            except Exception as e:
                print(f"Warning: Could not get detailed video info: {e}")
                width = height = "Unknown"
                duration = 0
                frame_rate = "Unknown"
                total_frames = 0
                merged_frames = 0
                merged_duration = 0
                reverse_start_frame = 0

            subprocess.run(ffmpeg_command, check=True)

            # Após o processamento, escrever log detalhado
            log_file_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_merge_frames.txt",
            )
            with open(log_file_path, "w") as log_file:
                log_file.write("DETAILED VIDEO MERGE REPORT\n")
                log_file.write("=========================\n\n")
                log_file.write(f"Source Video: {video_path}\n")
                log_file.write(f"Output Video: {output_video}\n")
                log_file.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                log_file.write("ORIGINAL VIDEO DETAILS\n")
                log_file.write("---------------------\n")
                log_file.write(f"Resolution: {width} x {height}\n")
                log_file.write(f"Frame Rate: {frame_rate} fps\n")
                log_file.write(f"Duration: {duration:.2f} seconds\n")
                log_file.write(f"Total Frames: {total_frames}\n\n")

                log_file.write("MERGED VIDEO STRUCTURE\n")
                log_file.write("---------------------\n")
                log_file.write("Part 1 (Original Video):\n")
                log_file.write("  - Start Frame: 0\n")
                log_file.write(f"  - End Frame: {total_frames - 1}\n")
                log_file.write(f"  - Duration: {duration:.2f} seconds\n\n")

                log_file.write("Part 2 (Reversed Video):\n")
                log_file.write(f"  - Start Frame: {reverse_start_frame}\n")
                log_file.write(f"  - End Frame: {merged_frames - 1}\n")
                log_file.write(f"  - Duration: {duration:.2f} seconds\n\n")

                log_file.write("MERGED VIDEO DETAILS\n")
                log_file.write("-------------------\n")
                log_file.write(f"Total Frames: {merged_frames}\n")
                log_file.write(f"Total Duration: {merged_duration:.2f} seconds\n")
                log_file.write(f"Encoder: {encoder}\n")
                log_file.write(f"Quality Setting: {quality_param}={preset_value}\n")

                # Adicionar tamanho do arquivo
                output_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                log_file.write(f"File Size: {output_size_mb:.2f} MB\n")

                log_file.write("\nFFmpeg Command Used:\n")
                log_file.write(f"{' '.join(ffmpeg_command)}\n")

            # Registre tempo de execução por vídeo
            start_time = time.time()
            # ... processamento ...
            elapsed = time.time() - start_time
            print(
                f"Processed in {elapsed:.2f} seconds ({os.path.getsize(output_video) / 1024 / 1024:.2f} MB)"
            )

            print(f"Video processed and saved to: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error processing video {video_path}: {e}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")


def process_videos_split(source_dir, target_dir, use_text_file=False, text_file_path=None):
    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"splitvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    video_files = []

    # Use provided text file if specified
    if use_text_file and text_file_path:
        with open(text_file_path) as file:
            for line in file.readlines():
                line = line.strip()
                if line:
                    video_files.append(os.path.join(source_dir, line.strip()))
    else:
        # No text file provided, process all videos in source_dir
        with os.scandir(source_dir) as entries:
            for entry in entries:
                if entry.is_file() and entry.name.lower().endswith(
                    (".mp4", ".avi", ".mov", ".mkv")
                ):
                    video_files.append(entry.path)

    # Iterate over video files and apply the split process
    for video_path in tqdm.tqdm(video_files, desc="Processing videos"):
        try:
            print(f"Processing video: {video_path}")

            # Get total number of frames using ffprobe
            ffprobe_frames_command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-count_frames",
                "-show_entries",
                "stream=nb_read_frames",
                "-of",
                "default=nokey=1:noprint_wrappers=1",
                video_path,
            ]
            frames_result = subprocess.run(
                ffprobe_frames_command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            total_frames = int(frames_result.stdout.strip())
            half_frame = total_frames // 2

            # Output video path
            output_video = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_2half.mp4",
            )

            # Obter informações detalhadas do vídeo
            ffprobe_info_command = [
                "ffprobe",
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "stream=width,height,r_frame_rate",
                "-show_entries",
                "format=duration",
                "-of",
                "json",
                video_path,
            ]

            try:
                video_info_result = subprocess.run(
                    ffprobe_info_command,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True,
                )
                video_info = json.loads(video_info_result.stdout)

                # Extract values from the JSON structure
                width = int(video_info["streams"][0]["width"])
                height = int(video_info["streams"][0]["height"])
                frame_rate_str = video_info["streams"][0]["r_frame_rate"]
                duration = float(video_info["format"]["duration"])

                # Convert frame rate from string format (e.g. "30000/1001")
                if "/" in frame_rate_str:
                    num, den = map(int, frame_rate_str.split("/"))
                    frame_rate = num / den
                else:
                    frame_rate = float(frame_rate_str)

                # Calcular informações do vídeo resultante
                half_duration = duration / 2
                second_half_duration = duration - half_duration

                print(
                    f"Video info: {width}x{height}, {frame_rate:.2f} fps, {duration:.2f}s, {total_frames} frames"
                )

            except Exception as e:
                print(f"Warning: Could not get detailed video info: {e}")
                width = height = "Unknown"
                duration = 0
                frame_rate = "Unknown"
                half_duration = 0
                second_half_duration = 0

            # Detect hardware encoder but FORCE libx264 for split operation
            encoder_info = detect_hardware_encoder()

            # Force libx264 for split operation to avoid compatibility issues
            print("Split mode: forcing software encoder (libx264)")
            encoder = "libx264"
            quality_param = "preset"
            preset_value = "medium"  # Use medium preset for a good balance

            print(f"Using encoder: {encoder} with {quality_param}={preset_value}")

            # Command to extract the second half of the video by frames
            ffmpeg_command = [
                "ffmpeg",
                "-i",
                video_path,
                "-vf",
                f"select='gte(n,{half_frame})',setpts=N/FR/TB",
                "-vsync",
                "0",
                "-c:v",
                encoder,
                f"-{quality_param}",
                preset_value,
                "-pix_fmt",
                "yuv420p",  # libx264 works well with yuv420p
                "-threads",
                "4",
                output_video,
            ]
            subprocess.run(ffmpeg_command, check=True)

            # Após processamento, escrever log detalhado
            log_file_path = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_split_frames.txt",
            )
            with open(log_file_path, "w") as log_file:
                log_file.write("DETAILED VIDEO SPLIT REPORT\n")
                log_file.write("=========================\n\n")
                log_file.write(f"Source Video: {video_path}\n")
                log_file.write(f"Output Video: {output_video}\n")
                log_file.write(f"Processing Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                log_file.write("ORIGINAL VIDEO DETAILS\n")
                log_file.write("---------------------\n")
                log_file.write(f"Resolution: {width} x {height}\n")
                log_file.write(
                    f"Frame Rate: {frame_rate if isinstance(frame_rate, str) else frame_rate:.2f} fps\n"
                )
                log_file.write(f"Duration: {duration:.2f} seconds\n")
                log_file.write(f"Total Frames: {total_frames}\n\n")

                log_file.write("SPLIT DETAILS\n")
                log_file.write("------------\n")
                log_file.write("First Half (Discarded):\n")
                log_file.write("  - Start Frame: 0\n")
                log_file.write(f"  - End Frame: {half_frame - 1}\n")
                log_file.write(f"  - Duration: {half_duration:.2f} seconds\n\n")

                log_file.write("Second Half (Kept):\n")
                log_file.write(f"  - Start Frame: {half_frame}\n")
                log_file.write(f"  - End Frame: {total_frames - 1}\n")
                log_file.write(f"  - Duration: {second_half_duration:.2f} seconds\n\n")

                log_file.write("OUTPUT VIDEO DETAILS\n")
                log_file.write("-------------------\n")
                log_file.write(f"Total Frames: {total_frames - half_frame}\n")
                log_file.write(f"Total Duration: {second_half_duration:.2f} seconds\n")
                log_file.write(f"Encoder: {encoder}\n")
                log_file.write(f"Encoding Parameters: {quality_param}={preset_value}\n")
                log_file.write("Pixel Format: yuv420p\n")

                # Adicionar tamanho do arquivo
                output_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                log_file.write(f"File Size: {output_size_mb:.2f} MB\n")

                log_file.write("\nFFmpeg Command Used:\n")
                log_file.write(f"{' '.join(ffmpeg_command)}\n")

            print(f"Video processed and saved to: {output_video}")
        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error processing video {video_path}: {e}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")


def process_videos_gui():
    # Print the directory and name of the script being executed
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent}")
    print("Starting video processing...")

    if not check_ffmpeg_installed():
        messagebox.showerror(
            "Error",
            "FFmpeg is not installed or not in PATH. Please install FFmpeg to use this feature.",
        )
        return

    # Ask user to select one of the three options
    operation_input = simpledialog.askstring(
        "Operation",
        "Enter operation:\n'm' for merge (original+reverse)\n's' for split (keep second half)\n'multi' for multi-video merge",
    )

    # Check if user cancelled the dialog
    if operation_input is None:
        messagebox.showerror("Error", "No operation selected.")
        return

    operation = operation_input.strip().lower()

    if not operation or operation not in ["m", "s", "multi"]:
        messagebox.showerror(
            "Error",
            "Invalid operation selected. Please enter 'm', 's', or 'multi'.",
        )
        return

    # Print the selected operation
    operation_names = {
        "m": "merge (original+reverse)",
        "s": "split (keep second half)",
        "multi": "multi-video merge"
    }
    print(f"Selected operation: '{operation}' - {operation_names[operation]}")

    # For multi-video merge, call the new module
    if operation == "multi":
        try:
            from vaila import merge_multivideos

            merge_multivideos.run_merge_multivideos()
            return
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start multi-video merger: {str(e)}")
            return

    # For other operations, continue with existing code
    # Ask user to select source directory
    source_dir = filedialog.askdirectory(title="Select Source Directory")
    if not source_dir:
        messagebox.showerror("Error", "No source directory selected.")
        return

    # Ask user to select target directory
    target_dir = filedialog.askdirectory(title="Select Target Directory")
    if not target_dir:
        messagebox.showerror("Error", "No target directory selected.")
        return

    # Ask if the user wants to use a text file with example format
    txt_file_example = (
        "Example format of videos_e_frames.txt:\n\nvideo1.mp4\nvideo2.mp4\nvideo3.mp4\n\n"
        + "Each line should contain just the filename (if in source directory)\n"
        + "or full path (if located elsewhere)"
    )
    use_text_file = messagebox.askyesno(
        "Use Text File",
        f"Do you want to use a text file to specify videos to process?\n\n{txt_file_example}",
        detail="Files will be processed in the order listed in the text file.",
    )

    text_file_path = None
    if use_text_file:
        text_file_path = filedialog.askopenfilename(
            title="Select videos_e_frames.txt", filetypes=[("Text files", "*.txt")]
        )
        if not text_file_path:
            messagebox.showerror("Error", "No text file selected.")
            return

    # Call the appropriate function based on the selected operation
    if operation == "m":
        process_videos_merge(source_dir, target_dir, use_text_file, text_file_path)
    elif operation == "s":
        process_videos_split(source_dir, target_dir, use_text_file, text_file_path)


if __name__ == "__main__":
    process_videos_gui()
