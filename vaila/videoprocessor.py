"""
videoprocessor.py
vailá - Multimodal Toolbox
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.
Created by Paulo Santiago
Date: 03 April 2025
Updated: 25 July 2025

Licensed under GNU Lesser General Public License v3.0

Description:
This script allows users to process and edit video files, enabling batch processing of videos. Users can choose between two main operations:
1. Merging a video with its reversed version, resulting in a video with double the frames.
2. Splitting each video into two halves and saving only the second half.
The script supports custom text files for batch processing and includes a GUI for directory and file selection.

Key Features:
- Graphical User Interface (GUI) for easy selection of directories and file inputs.
- Batch processing using a text file (`videos_e_frames.txt`) with custom instructions for specifying which videos to process.
- If no text file is provided, the script processes all videos in the source directory.
- Merge option: Creates a video with the original and its reversed version merged.
- Split option: Processes each video to save only the second half.
- Automatic creation of output directories based on a timestamp for organized file management.
- Detailed console output for tracking progress and handling errors.

Usage:
- python videoprocessor.py
- Run the script to open a graphical interface. After selecting the source and target directories, choose between merging or splitting the videos.
- The processed videos will be saved in a new output directory named with a timestamp.

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

import json
import os
import pathlib
import subprocess
import time
from tkinter import filedialog, messagebox, simpledialog

import tqdm
from rich import print

# Try to import toml for TOML file support (toml → tomli → tomllib)
try:
    import toml
except ImportError:
    try:
        import tomli as toml  # type: ignore[import-untyped]
    except ImportError:
        try:
            import tomllib as toml
        except ImportError:
            toml = None


def check_ffmpeg_installed():
    """Check if FFmpeg is available on the system"""
    try:
        subprocess.run(
            ["ffmpeg", "-version"],
            capture_output=True,
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
            capture_output=True,
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
                    capture_output=True,
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
                    capture_output=True,
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
    except Exception:
        return True, ""  # Proceed in case of error


def process_videos_merge(source_dir, target_dir, use_text_file=False, text_file_path=None):
    print("\n" + "=" * 60)
    print("METHOD: MERGE (original + reverse)")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Using text file: {use_text_file}")
    if use_text_file and text_file_path:
        print(f"Text file path: {text_file_path}")
    print("=" * 60 + "\n")

    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"mergedvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

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
    print("\nDetecting hardware encoder...")
    encoder_info = detect_hardware_encoder()
    encoder = encoder_info["encoder"]
    quality_param = encoder_info["quality_param"]
    quality_values = encoder_info["quality_values"]
    print(f"Selected encoder: {encoder}")

    # Ask user to choose reverse percentage
    reverse_percent_msg = (
        "Choose percentage of video to reverse:\n\n"
        "Options: 10, 20, 30, 40, 50, 60, 70, 80, 90, 100\n\n"
        "Example: 50 = reverse the first 50% of the video\n"
        "The reversed portion will be concatenated with the full original video."
    )
    reverse_percent = simpledialog.askinteger(
        "Reverse Percentage", reverse_percent_msg, minvalue=10, maxvalue=100, initialvalue=100
    )

    # Handle case where user cancels dialog
    if reverse_percent is None:
        reverse_percent = 100  # Default to full reverse (original behavior)

    print(f"Reverse percentage selected: {reverse_percent}%")

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
            if os.path.exists(output_video) and not messagebox.askyesno(
                "File exists",
                f"Output file already exists:\n{output_video}\n\nOverwrite?",
            ):
                print(f"Skipping {video_path} (output exists)")
                continue

            # Initialize variables for logging
            reverse_frames = 0
            reverse_duration = 0

            # Get metadata first to calculate reverse duration
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
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )

                # Calculate reverse duration based on percentage
                reverse_duration = duration * (reverse_percent / 100.0)
                reverse_duration_str = f"{reverse_duration:.6f}"

                print(
                    f"Video duration: {duration:.2f}s | Reverse portion: {reverse_duration:.2f}s ({reverse_percent}%)"
                )

            except Exception as e:
                print(f"Warning: Could not get video duration: {e}")
                # Fallback: use full video reverse (original behavior)
                reverse_duration_str = None
                duration = 0

            # Prepare command based on reverse percentage
            if reverse_percent == 100 or reverse_duration_str is None:
                # Full reverse (original behavior or fallback)
                filter_complex = "[0:v]reverse[r];[r][0:v]concat=n=2:v=1:a=0[out]"
            else:
                # Partial reverse: trim first X% of video, reverse it, then concat with full original
                # Part 1: First X% of video (reversed)
                # Part 2: Full original video
                filter_complex = (
                    f"[0:v]trim=start=0:end={reverse_duration_str},reverse,setpts=PTS-STARTPTS[rev];"
                    f"[rev][0:v]concat=n=2:v=1:a=0[out]"
                )

            ffmpeg_command = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                "-c:v",
                encoder,
                "-threads",
                "4",
                f"-{quality_param}",
                preset_value,
                output_video,
            ]

            # Continue getting metadata for logging
            try:
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
                        capture_output=True,
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
                        capture_output=True,
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
                fps_str = subprocess.run(fps_cmd, capture_output=True, text=True).stdout.strip()
                # Safely parse fractional fps like "30000/1001"
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    frame_rate = int(num) / int(den)
                else:
                    frame_rate = float(fps_str)

                # Calcular informações do vídeo resultante baseado na porcentagem
                reverse_frames = int(total_frames * (reverse_percent / 100.0))
                if reverse_percent == 100:
                    merged_frames = total_frames * 2
                    merged_duration = duration * 2
                else:
                    merged_frames = reverse_frames + total_frames
                    merged_duration = reverse_duration + duration

                print(
                    f"Video info: {width}x{height}, {frame_rate} fps, {duration:.2f}s, {total_frames} frames"
                )
                print(
                    f"Reverse: {reverse_frames} frames ({reverse_percent}%) | Merged: {merged_frames} frames, {merged_duration:.2f}s"
                )

            except Exception as e:
                print(f"Warning: Could not get detailed video info: {e}")
                width = height = "Unknown"
                duration = 0
                frame_rate = "Unknown"
                total_frames = 0
                merged_frames = 0
                merged_duration = 0
                reverse_frames = 0
                reverse_duration = 0

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
                log_file.write(f"Reverse Percentage: {reverse_percent}%\n\n")

                if reverse_percent == 100 or reverse_duration == 0:
                    log_file.write("Part 1 (Reversed Video - Full):\n")
                    log_file.write("  - Start Frame: 0 (reversed to end)\n")
                    log_file.write(f"  - End Frame: {total_frames - 1} (reversed to 0)\n")
                    log_file.write(f"  - Duration: {duration:.2f} seconds\n\n")
                    log_file.write("Part 2 (Original Video - Full):\n")
                    log_file.write("  - Start Frame: 0\n")
                    log_file.write(f"  - End Frame: {total_frames - 1}\n")
                    log_file.write(f"  - Duration: {duration:.2f} seconds\n\n")
                else:
                    log_file.write(f"Part 1 (Reversed Video - First {reverse_percent}%):\n")
                    log_file.write(f"  - Start Frame: 0 (reversed to frame {reverse_frames - 1})\n")
                    log_file.write(f"  - End Frame: {reverse_frames - 1} (reversed to 0)\n")
                    log_file.write(f"  - Duration: {reverse_duration:.2f} seconds\n\n")
                    log_file.write("Part 2 (Original Video - Full):\n")
                    log_file.write("  - Start Frame: 0\n")
                    log_file.write(f"  - End Frame: {total_frames - 1}\n")
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


def save_frame_merge_toml(
    source_video,
    output_video,
    selected_frame,
    reverse_frames_count,
    original_start_frame,
    original_total_frames,
    fps,
    output_dir,
    video_basename,
):
    """
    Save frame reverse merge metadata to TOML file.

    Parameters:
    -----------
    source_video : str
        Path to original source video
    output_video : str
        Path to merged output video
    selected_frame : int
        Frame number selected by user
    reverse_frames_count : int
        Number of frames in the reversed portion
    original_start_frame : int
        Frame where original video starts in merged video
    original_total_frames : int
        Total frames in original video
    fps : float
        Frames per second
    output_dir : str
        Directory to save TOML file
    video_basename : str
        Base name for output file
    """
    if toml is None:
        print("Warning: TOML library not available. Cannot save metadata file.")
        return

    toml_path = os.path.join(output_dir, f"{video_basename}_frame_reverse_merge.toml")

    toml_data = {
        "frame_reverse_merge": {
            "source_video": source_video,
            "output_video": output_video,
            "selected_frame": selected_frame,
            "reverse_frames_count": reverse_frames_count,
            "original_start_frame": original_start_frame,
            "original_total_frames": original_total_frames,
            "fps": float(fps),
            "processing_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
    }

    try:
        with open(toml_path, "w", encoding="utf-8") as f:
            toml.dump(toml_data, f)
        print(f"TOML metadata saved to: {toml_path}")
    except Exception as e:
        print(f"Error saving TOML file: {e}")


def process_videos_frame_reverse_merge(
    source_dir, target_dir, use_text_file=False, text_file_path=None
):
    print("\n" + "=" * 60)
    print("METHOD: FRAME-BASED REVERSE MERGE (frame N→0 reverse + original)")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Using text file: {use_text_file}")
    if use_text_file and text_file_path:
        print(f"Text file path: {text_file_path}")
    print("=" * 60 + "\n")

    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"framereverse_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

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

    # Ask user to choose frame number (once for all videos)
    frame_msg = (
        "Enter the frame number to reverse from:\n\n"
        "Example: 500 = reverse frames from 0 to 500 (backward)\n"
        "The reversed portion (frame N→0) will be concatenated\n"
        "with the full original video at the beginning."
    )
    selected_frame = simpledialog.askinteger(
        "Frame Number", frame_msg, minvalue=1, initialvalue=500
    )

    # Handle case where user cancels dialog
    if selected_frame is None:
        messagebox.showerror("Error", "Frame number is required. Operation cancelled.")
        return

    if selected_frame < 1:
        messagebox.showerror("Error", "Frame number must be at least 1. Operation cancelled.")
        return

    print(f"Selected frame: {selected_frame}")

    # Detect hardware encoder and available presets - moved outside the loop
    print("\nDetecting hardware encoder...")
    encoder_info = detect_hardware_encoder()
    encoder = encoder_info["encoder"]
    quality_param = encoder_info["quality_param"]
    quality_values = encoder_info["quality_values"]
    print(f"Selected encoder: {encoder}")

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

    # Iterate over video files and apply the frame reverse merge process
    for video_path in tqdm.tqdm(video_files, desc="Processing videos"):
        try:
            print(f"Processing video: {video_path}")

            # Get total number of frames first to validate
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
            try:
                total_frames = int(
                    subprocess.run(
                        frames_cmd,
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )
            except Exception as e:
                print(f"Error getting frame count: {e}")
                print(f"Skipping {video_path}")
                continue

            # Validate frame number
            if selected_frame > total_frames:
                print(
                    f"Warning: Selected frame {selected_frame} exceeds total frames {total_frames}"
                )
                print(f"Skipping {video_path}")
                continue

            if selected_frame == 0:
                print("Warning: Frame 0 selected, nothing to reverse")
                print(f"Skipping {video_path}")
                continue

            # Output video path
            output_video = os.path.join(
                output_dir,
                f"{os.path.splitext(os.path.basename(video_path))[0]}_framereverse.mp4",
            )

            # Verificar se já processou este vídeo anteriormente
            if os.path.exists(output_video) and not messagebox.askyesno(
                "File exists",
                f"Output file already exists:\n{output_video}\n\nOverwrite?",
            ):
                print(f"Skipping {video_path} (output exists)")
                continue

            # Get video metadata for logging
            try:
                # Get resolution
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
                        capture_output=True,
                        text=True,
                    )
                    .stdout.strip()
                    .split("\n")
                )
                width = int(resolution[0])
                height = int(resolution[1])

                # Get frame rate
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
                fps_str = subprocess.run(fps_cmd, capture_output=True, text=True).stdout.strip()
                # Safely parse fractional fps like "30000/1001"
                if "/" in fps_str:
                    num, den = fps_str.split("/")
                    frame_rate = int(num) / int(den)
                else:
                    frame_rate = float(fps_str)

                # Get duration
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
                        capture_output=True,
                        text=True,
                    ).stdout.strip()
                )

                print(
                    f"Video info: {width}x{height}, {frame_rate:.2f} fps, {duration:.2f}s, {total_frames} frames"
                )

            except Exception as e:
                print(f"Warning: Could not get detailed video info: {e}")
                width = height = "Unknown"
                duration = 0
                frame_rate = "Unknown"

            # Calculate reverse frames count (frames 0 to selected_frame-1, inclusive)
            # Note: FFmpeg uses 0-indexed frames, so frame N means frames 0 to N-1
            reverse_frames_count = selected_frame
            original_start_frame = reverse_frames_count  # Original starts after reversed portion

            # Calculate durations
            reverse_duration = reverse_frames_count / frame_rate if frame_rate != "Unknown" else 0
            merged_duration = reverse_duration + duration
            merged_frames = reverse_frames_count + total_frames

            print(
                f"Reverse: frames 0-{selected_frame - 1} ({reverse_frames_count} frames) | "
                f"Merged: {merged_frames} frames, {merged_duration:.2f}s"
            )

            # Create FFmpeg filter_complex for frame-based reverse
            # Select frames 0 to selected_frame-1, reverse them, then concat with full original
            # Note: FFmpeg select filter uses 0-indexed frames
            # 'lt(n,selected_frame)' selects frames where n < selected_frame (i.e., 0 to selected_frame-1)
            filter_complex = (
                f"[0:v]select='lt(n\\,{selected_frame})',reverse,setpts=PTS-STARTPTS[rev];"
                f"[rev][0:v]concat=n=2:v=1:a=0[out]"
            )

            ffmpeg_command = [
                "ffmpeg",
                "-i",
                video_path,
                "-filter_complex",
                filter_complex,
                "-map",
                "[out]",
                "-c:v",
                encoder,
                "-threads",
                "4",
                f"-{quality_param}",
                preset_value,
                output_video,
            ]

            # Process video
            start_time = time.time()
            subprocess.run(ffmpeg_command, check=True)
            elapsed = time.time() - start_time

            # Save TOML metadata
            video_basename = os.path.splitext(os.path.basename(video_path))[0]
            save_frame_merge_toml(
                source_video=video_path,
                output_video=output_video,
                selected_frame=selected_frame,
                reverse_frames_count=reverse_frames_count,
                original_start_frame=original_start_frame,
                original_total_frames=total_frames,
                fps=frame_rate if frame_rate != "Unknown" else 0.0,
                output_dir=output_dir,
                video_basename=video_basename,
            )

            # Write detailed log file
            log_file_path = os.path.join(
                output_dir,
                f"{video_basename}_frame_reverse_frames.txt",
            )
            with open(log_file_path, "w") as log_file:
                log_file.write("DETAILED FRAME REVERSE MERGE REPORT\n")
                log_file.write("==================================\n\n")
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

                log_file.write("FRAME REVERSE MERGE STRUCTURE\n")
                log_file.write("----------------------------\n")
                log_file.write(f"Selected Frame: {selected_frame}\n\n")

                log_file.write(f"Part 1 (Reversed Video - Frames 0 to {selected_frame - 1}):\n")
                log_file.write(f"  - Start Frame: 0 (reversed to frame {selected_frame - 1})\n")
                log_file.write(f"  - End Frame: {selected_frame - 1} (reversed to 0)\n")
                log_file.write(f"  - Frames Count: {reverse_frames_count}\n")
                log_file.write(f"  - Duration: {reverse_duration:.2f} seconds\n\n")

                log_file.write("Part 2 (Original Video - Full):\n")
                log_file.write("  - Start Frame: 0\n")
                log_file.write(f"  - End Frame: {total_frames - 1}\n")
                log_file.write(f"  - Frames Count: {total_frames}\n")
                log_file.write(f"  - Duration: {duration:.2f} seconds\n\n")

                log_file.write("MERGED VIDEO DETAILS\n")
                log_file.write("-------------------\n")
                log_file.write(f"Total Frames: {merged_frames}\n")
                log_file.write(f"Total Duration: {merged_duration:.2f} seconds\n")
                log_file.write(f"Original Starts at Frame: {original_start_frame}\n")
                log_file.write(f"Encoder: {encoder}\n")
                log_file.write(f"Quality Setting: {quality_param}={preset_value}\n")

                # Add file size
                output_size_mb = os.path.getsize(output_video) / (1024 * 1024)
                log_file.write(f"File Size: {output_size_mb:.2f} MB\n")

                log_file.write("\nFFmpeg Command Used:\n")
                log_file.write(f"{' '.join(ffmpeg_command)}\n")

            print(
                f"Processed in {elapsed:.2f} seconds ({os.path.getsize(output_video) / 1024 / 1024:.2f} MB)"
            )
            print(f"Video processed and saved to: {output_video}")

        except subprocess.CalledProcessError as e:
            print(f"FFmpeg error processing video {video_path}: {e}")
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")


def process_videos_split(source_dir, target_dir, use_text_file=False, text_file_path=None):
    print("\n" + "=" * 60)
    print("METHOD: SPLIT (keep second half)")
    print("=" * 60)
    print(f"Source directory: {source_dir}")
    print(f"Target directory: {target_dir}")
    print(f"Using text file: {use_text_file}")
    if use_text_file and text_file_path:
        print(f"Text file path: {text_file_path}")
    print("=" * 60 + "\n")

    # Create a new directory with timestamp
    timestamp = time.strftime("%Y%m%d%H%M%S")
    output_dir = os.path.join(target_dir, f"splitvid_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory created: {output_dir}")

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
                capture_output=True,
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
                    capture_output=True,
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
            detect_hardware_encoder()

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

    # Ask user to select one of the four options
    operation_input = simpledialog.askstring(
        "Operation",
        "Enter operation:\n'm' for merge (original+reverse)\n's' for split (keep second half)\n'f' for frame-based reverse merge\n'multi' for multi-video merge",
    )

    # Check if user cancelled the dialog
    if operation_input is None:
        messagebox.showerror("Error", "No operation selected.")
        return

    operation = operation_input.strip().lower()

    if not operation or operation not in ["m", "s", "f", "multi"]:
        messagebox.showerror(
            "Error",
            "Invalid operation selected. Please enter 'm', 's', 'f', or 'multi'.",
        )
        return

    # Print selected operation method
    operation_names = {
        "m": "MERGE (original + reverse)",
        "s": "SPLIT (keep second half)",
        "f": "FRAME-BASED REVERSE MERGE",
        "multi": "MULTI-VIDEO MERGE",
    }
    print(f"Selected operation: '{operation}' - {operation_names.get(operation, 'UNKNOWN')}")

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
    elif operation == "f":
        process_videos_frame_reverse_merge(source_dir, target_dir, use_text_file, text_file_path)


if __name__ == "__main__":
    process_videos_gui()
