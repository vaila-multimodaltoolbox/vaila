"""
Project: vailá Multimodal Toolbox
Script: cutvideo.py - Cut Video

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 08 January 2026
Version: 0.1.1

Description:
This script performs batch processing of videos for cutting videos.
verview:
This script performs batch processing of videos for cutting videos.
It allows users to visually mark cut points in videos, save cut information, and generate precisely cut video segments while preserving original metadata with high accuracy.
It supports scientific precision for research applications.
It uses ffmpeg to preserve exact frame rates (e.g., 59.94005994005994 fps) without rounding.
It uses OpenCV to fallback to less precise but always available.
It uses pygame to create a graphical interface for selecting the input directory containing video files (.mp4, .avi, .mov), the output directory, and for specifying the cuts.
It uses tomllib to load the cuts from a TOML file.
It uses ffmpeg to cut the videos.

Features:
- Added support for TOML files.
- Added support for audio waveform visualization.
- Added support for audio playback.
- Added support for loop control.
- Added support for auto-fit window.
- Added support for marker navigation.
- Added support for manual FPS input.
- Added support for help dialog.
- Added support for save and generate videos.
- Added support for batch processing of videos.

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov, .mkv), the output directory, and for
  specifying the cuts.

Requirements:
- Python 3.12.12
- OpenCV (`pip install opencv-python`)
- pygame (`pip install pygame`)
- Tkinter (usually included with Python installations)
- tomllib (`pip install tomllib`)
- rich (`pip install rich`)
- numpy (`pip install numpy`)
- scipy (`pip install scipy`)
- matplotlib (`pip install matplotlib`)
- pandas (`pip install pandas`)
- seaborn (`pip install seaborn`)
- plotly (`pip install plotly`)
- plotly-express (`pip install plotly-express`)
- plotly-orca (`pip install plotly-orca`)

Output:
The following files are generated for each processed video:
- Cuts information saved in TOML format.
- Cut videos with the cuts applied.
- Batch output directory with all cut videos from batch operation.

How to run:
python cutvideo.py

License:
    This project is licensed under the terms of AGPLv3.
"""

import datetime
import json
import os
import subprocess
import tempfile
import tomllib
import wave
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import cv2
import numpy as np
import pygame
from rich import print


def get_precise_video_metadata(video_path):
    """
    Get precise video metadata using ffprobe to avoid rounding errors.
    Returns dict with fps (float), width, height, codec, etc.
    """
    try:
        cmd = [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, check=True
        )
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            # Fallback to OpenCV if ffprobe fails
            cap = cv2.VideoCapture(str(video_path))
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            cap.release()
            return {
                "fps": fps,
                "width": width,
                "height": height,
                "codec": "unknown",
                "r_frame_rate": None,
                "avg_frame_rate": None,
            }

        # Get precise FPS from r_frame_rate or avg_frame_rate
        r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
        avg_frame_rate_str = video_stream.get("avg_frame_rate", "0/0")

        # Convert fraction strings to float
        def fraction_to_float(frac_str):
            try:
                if "/" in frac_str:
                    num, den = map(int, frac_str.split("/"))
                    return float(num) / den if den != 0 else 0.0
                return float(frac_str)
            except (ValueError, ZeroDivisionError):
                return None

        r_fps = fraction_to_float(r_frame_rate_str)
        avg_fps = fraction_to_float(avg_frame_rate_str)

        # Use avg_frame_rate if available, otherwise r_frame_rate
        fps = avg_fps if avg_fps and avg_fps > 0 else (r_fps if r_fps and r_fps > 0 else 30.0)

        # Get frame count if available
        nb_frames = None
        try:
            if "nb_frames" in video_stream and video_stream["nb_frames"] not in (None, "N/A", ""):
                nb_frames = int(video_stream["nb_frames"])
        except (ValueError, TypeError):
            nb_frames = None

        # Calculate frame count from duration and FPS if nb_frames not available
        duration = float(data.get("format", {}).get("duration", 0))
        if nb_frames is None and duration > 0 and fps > 0:
            nb_frames = int(round(duration * fps))

        return {
            "fps": fps,
            "width": video_stream.get("width"),
            "height": video_stream.get("height"),
            "codec": video_stream.get("codec_name", "unknown"),
            "r_frame_rate": r_frame_rate_str,
            "avg_frame_rate": avg_frame_rate_str,
            "duration": duration if duration > 0 else None,
            "nb_frames": nb_frames,
        }
    except (subprocess.CalledProcessError, json.JSONDecodeError, FileNotFoundError) as e:
        # Fallback to OpenCV if ffprobe is not available
        print(f"Warning: ffprobe not available or failed, using OpenCV fallback: {e}")
        cap = cv2.VideoCapture(str(video_path))
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        cap.release()
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "codec": "unknown",
            "r_frame_rate": None,
            "avg_frame_rate": None,
        }


def cut_video_with_ffmpeg(video_path, output_path, start_frame, end_frame, metadata):
    """
    Cut video using ffmpeg to preserve precise metadata.
    Uses frame-accurate cutting with copy codec when possible.
    """
    try:
        # Check if ffmpeg is available
        subprocess.run(
            ["ffmpeg", "-version"], stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        # Fallback to OpenCV if ffmpeg not available
        return cut_video_with_opencv(video_path, output_path, start_frame, end_frame, metadata)

    fps = metadata["fps"]

    # Calculate precise frame count (inclusive) and start time
    frame_count = end_frame - start_frame + 1
    start_time = start_frame / fps if fps > 0 else 0.0

    # Frame-accurate cutting using ffmpeg with re-encoding and explicit frame count
    cmd_reencode = [
        "ffmpeg",
        "-y",
        "-ss",
        f"{start_time:.6f}",
        "-i",
        str(video_path),
        "-frames:v",
        str(frame_count),
        "-c:v",
        "libx264",  # Re-encode for accuracy
        "-preset",
        "medium",
        "-crf",
        "18",  # High quality
        "-c:a",
        "aac",
        "-b:a",
        "192k",
        "-r",
        f"{fps:.6f}",  # Preserve fps
        "-avoid_negative_ts",
        "make_zero",
        str(output_path),
    ]
    try:
        subprocess.run(
            cmd_reencode,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=True,
        )
        return True
    except subprocess.CalledProcessError as e2:
        print(f"Error with ffmpeg re-encoding: {e2.stderr}")
        # Final fallback to OpenCV
        return cut_video_with_opencv(video_path, output_path, start_frame, end_frame, metadata)


def cut_video_with_opencv(video_path, output_path, start_frame, end_frame, metadata):
    """
    Fallback function to cut video using OpenCV (less precise but always available).
    """
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]

    # Use float FPS, not int
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    for _ in range(end_frame - start_frame + 1):
        ret, frame = cap.read()
        if not ret:
            break
        out.write(frame)

    out.release()
    cap.release()
    return True


def save_cuts_to_toml(video_path, cuts, fps=None, output_dir=None, per_cut_outputs=None):
    """Save cuts information to a TOML file.

    output_dir: optional Path/str for planned output directory.
    per_cut_outputs: optional list of filenames (one per cut) to record planned outputs.
    """
    try:
        video_name = Path(video_path).stem
        # Convert path to POSIX format (forward slashes) for universal compatibility
        # This works on Windows, Linux, and macOS
        video_path_posix = Path(video_path).absolute().as_posix()
        # Escape only quotes for TOML string (forward slashes don't need escaping)
        escaped_video_path = video_path_posix.replace('"', '\\"')
        toml_path = Path(video_path).parent / f"{video_name}_cuts.toml"

        # Current timestamp
        created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Build TOML content manually (no external library needed for writing)
        toml_content = "# Video metadata\n"
        toml_content += f'video_name = "{video_name}"\n'
        if fps is not None:
            toml_content += f"fps = {fps:.6f}\n"
        toml_content += f'created = "{created_time}"\n'
        toml_content += f'source_file = "{escaped_video_path}"\n'
        if output_dir is not None:
            output_dir_posix = Path(output_dir).absolute().as_posix()
            toml_content += f'output_dir = "{output_dir_posix}"\n'
        toml_content += "\n# List of cuts\n"

        for i, (start, end) in enumerate(cuts, 1):
            # Calculate frame_count and times
            # Note: start and end are 0-based internally, convert to 1-based for TOML (as shown in UI)
            start_frame_1based = start + 1
            end_frame_1based = end + 1
            # frame_count = end - start + 1 (inclusive count)
            frame_count = end - start + 1
            if fps is not None and fps > 0:
                # Use frame-count-based duration to avoid losing the last frame in timing
                start_time = start_frame_1based / fps
                duration = frame_count / fps
                end_time = start_time + duration
            else:
                start_time = None
                end_time = None
                duration = None

            toml_content += "[[cuts]]\n"
            toml_content += f"index = {i}\n"
            toml_content += f"start_frame = {start_frame_1based}\n"
            toml_content += f"end_frame = {end_frame_1based}\n"
            toml_content += f"frame_count = {frame_count}\n"
            if per_cut_outputs and i - 1 < len(per_cut_outputs):
                toml_content += f'output_file = "{per_cut_outputs[i - 1]}"\n'
            if output_dir is not None:
                toml_content += f'output_dir = "{output_dir_posix}"\n'
            if start_time is not None:
                toml_content += f"start_time = {start_time:.6f}\n"
                toml_content += f"end_time = {end_time:.6f}\n"
                toml_content += f"duration = {duration:.6f}\n"
            toml_content += "\n"

        with open(str(toml_path), "w", encoding="utf-8", errors="replace") as f:
            f.write(toml_content)

        return toml_path
    except Exception as e:
        print(f"Error saving TOML file: {e}")
        # Fallback para nomes com caracteres especiais
        try:
            safe_name = "".join(c if c.isalnum() or c in "._- " else "_" for c in video_name)
            toml_path = Path(video_path).parent / f"{safe_name}_cuts.toml"

            created_time = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Convert path to POSIX format (forward slashes) for universal compatibility
            # This works on Windows, Linux, and macOS
            video_path_posix = Path(video_path).absolute().as_posix()
            # Escape only quotes for TOML string (forward slashes don't need escaping)
            escaped_video_path = video_path_posix.replace('"', '\\"')

            toml_content = "# Video metadata\n"
            toml_content += f'video_name = "{video_name}"\n'
            if fps is not None:
                toml_content += f"fps = {fps:.6f}\n"
            toml_content += f'created = "{created_time}"\n'
            toml_content += f'source_file = "{escaped_video_path}"\n'
            toml_content += "\n# List of cuts\n"

            for i, (start, end) in enumerate(cuts, 1):
                # Note: start and end are 0-based internally, convert to 1-based for TOML (as shown in UI)
                start_frame_1based = start + 1
                end_frame_1based = end + 1
                # frame_count = end - start + 1 (inclusive)
                frame_count = end - start + 1
                if fps is not None and fps > 0:
                    # Use frame-count-based duration to avoid losing the last frame in timing
                    start_time = start_frame_1based / fps
                    duration = frame_count / fps
                    end_time = start_time + duration
                else:
                    start_time = None
                    end_time = None
                    duration = None

                toml_content += "[[cuts]]\n"
                toml_content += f"index = {i}\n"
                toml_content += f"start_frame = {start_frame_1based}\n"
                toml_content += f"end_frame = {end_frame_1based}\n"
                toml_content += f"frame_count = {frame_count}\n"
                if start_time is not None:
                    toml_content += f"start_time = {start_time:.6f}\n"
                    toml_content += f"end_time = {end_time:.6f}\n"
                    toml_content += f"duration = {duration:.6f}\n"
                toml_content += "\n"

            with open(str(toml_path), "w", encoding="utf-8", errors="replace") as f:
                f.write(toml_content)

            return toml_path
        except Exception as e2:
            print(f"Error in fallback save: {e2}")
            return None


def load_sync_file(video_path):
    """Load synchronization data from sync file generated by syncvid.py."""
    video_dir = Path(video_path).parent
    video_name = Path(video_path).name

    # Look for sync files in the directory
    sync_files = list(video_dir.glob("*.txt"))

    for sync_file in sync_files:
        try:
            with open(sync_file, encoding="utf-8") as f:
                lines = f.readlines()

            # Check if this is a sync file (contains video file names and frame data)
            for line in lines:
                if line.strip() and video_name in line:
                    # Parse sync file format: video_file new_name initial_frame final_frame
                    parts = line.strip().split()
                    if len(parts) >= 4:
                        try:
                            initial_frame = int(parts[2])
                            final_frame = int(parts[3])
                            return [(initial_frame, final_frame)]
                        except ValueError:
                            continue
        except Exception:
            continue

    return []


def load_cuts_from_toml(video_path):
    """Load existing cuts from a TOML file if it exists. Falls back to old TXT format if needed."""
    video_name = Path(video_path).stem
    toml_path = Path(video_path).parent / f"{video_name}_cuts.toml"
    txt_path = Path(video_path).parent / f"{video_name}_cuts.txt"

    cuts = []

    # Try to load from TOML file first
    if toml_path.exists():
        try:
            # Convert Path to string explicitly to avoid any issues
            toml_file_str = str(toml_path)
            with open(toml_file_str, "rb") as f:
                # Use tomllib.load() which accepts a file-like object in binary mode
                toml_data = tomllib.load(f)

            # Extract cuts from TOML data
            if "cuts" in toml_data and toml_data["cuts"]:
                for cut in toml_data["cuts"]:
                    start_frame = cut.get("start_frame")
                    end_frame = cut.get("end_frame")
                    if start_frame is not None and end_frame is not None:
                        # TOML stores 1-based frames (as shown in UI), convert to 0-based for internal use
                        start_frame_0based = int(start_frame) - 1
                        end_frame_0based = int(end_frame) - 1
                        cuts.append((start_frame_0based, end_frame_0based))

            # Return cuts (even if empty list) - TOML file exists and was parsed successfully
            if cuts:
                return cuts
            # If TOML file exists but has no cuts, return empty list (don't fall through to TXT)
            return []
        except Exception as e:
            print(f"Error loading TOML file: {e}")
            import traceback

            traceback.print_exc()
            # Fall through to try TXT file

    # Fallback: Try to load from old TXT format for backward compatibility
    if txt_path.exists():
        try:
            with open(txt_path, encoding="utf-8") as f:
                lines = f.readlines()

            # Parse each line looking for cut information
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a cut line (format: "Cut N: Frame X to Y" or "Cut N: Frame X to Y (Time: ...)")
                if line.startswith("Cut ") and "Frame" in line:
                    # Extract frame numbers
                    # Format: "Cut N: Frame X to Y" or "Cut N: Frame X to Y (Time: ...)"
                    try:
                        # Find "Frame " and extract numbers
                        frame_part = line.split("Frame ")[1].split(" (")[
                            0
                        ]  # Remove time part if present
                        parts = frame_part.split(" to ")
                        # Note: TXT format uses 1-based frames, TOML uses 0-based
                        # So we convert: subtract 1 from each number
                        start = int(parts[0]) - 1
                        end = int(parts[1]) - 1
                        cuts.append((start, end))
                    except (ValueError, IndexError) as e:
                        print(f"Error parsing line: {line}, error: {e}")
                        continue

            return cuts
        except Exception as e:
            print(f"Error loading TXT file: {e}")

    return cuts


def load_cuts_or_sync(video_path):
    """Load cuts from either cut file or sync file."""
    # First try to load from sync file
    sync_cuts = load_sync_file(video_path)
    if sync_cuts:
        return sync_cuts, True  # True indicates sync file was used

    # If no sync file, try regular cuts file (TOML or TXT fallback)
    regular_cuts = load_cuts_from_toml(video_path)
    return regular_cuts, False  # False indicates regular cuts file was used


def extract_audio_data(video_path, target_sr=44100):
    """
    Extract raw mono PCM audio from video using ffmpeg and return (audio_array, sample_rate).
    If there is no audio stream or extraction fails, returns (None, None).
    """
    try:
        probe_cmd = [
            "ffprobe",
            "-v",
            "error",
            "-select_streams",
            "a:0",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            str(video_path),
        ]
        has_audio = subprocess.run(probe_cmd, stdout=subprocess.PIPE, text=True).stdout.strip()
        if not has_audio:
            return None, None

        cmd = [
            "ffmpeg",
            "-i",
            str(video_path),
            "-f",
            "s16le",
            "-ac",
            "1",
            "-ar",
            str(target_sr),
            "-acodec",
            "pcm_s16le",
            "-vn",
            "-",
        ]

        print(f"Loading audio waveform for {Path(video_path).name}...")
        process = subprocess.run(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            bufsize=10**8,
        )
        if process.returncode != 0:
            print(f"FFmpeg audio extraction failed: {process.stderr}")
            return None, None

        audio_data = np.frombuffer(process.stdout, dtype=np.int16)
        audio_data = audio_data.astype(np.float32) / 32768.0
        return audio_data, target_sr
    except Exception as e:
        print(f"Error extracting audio: {e}")
        return None, None


def write_wav_from_pcm(audio_data: np.ndarray, sample_rate: int) -> str:
    """
    Write mono float32 PCM (-1..1) to a temp WAV file and return its path.
    """
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    tmp_path = tmp.name
    tmp.close()

    pcm_int16 = np.clip(audio_data * 32767.0, -32768, 32767).astype(np.int16)
    with wave.open(tmp_path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_int16.tobytes())
    return tmp_path


def load_cuts_from_toml_file(toml_file_path):
    """Load cuts from a specific TOML file path.

    TOML file stores frames as 1-based (as shown in UI),
    converts to 0-based for internal use.
    """
    cuts = []
    try:
        # Ensure path is converted to string if it's a Path object
        toml_file_str = str(toml_file_path)
        with open(toml_file_str, "rb") as f:
            # Use tomllib.load() which accepts a file-like object in binary mode
            toml_data = tomllib.load(f)

        # Extract cuts from TOML data
        if "cuts" in toml_data and toml_data["cuts"]:
            for cut in toml_data["cuts"]:
                start_frame = cut.get("start_frame")
                end_frame = cut.get("end_frame")
                if start_frame is not None and end_frame is not None:
                    # TOML stores 1-based frames (as shown in UI), convert to 0-based for internal use
                    start_frame_0based = int(start_frame) - 1
                    end_frame_0based = int(end_frame) - 1
                    cuts.append((start_frame_0based, end_frame_0based))
        else:
            print(
                f"Warning: TOML file {toml_file_str} does not contain 'cuts' section or it is empty"
            )

        return cuts
    except Exception as e:
        print(f"Error loading TOML file {toml_file_path}: {e}")
        import traceback

        traceback.print_exc()
        return []


def load_sync_file_from_dialog(video_path):
    """Open dialog to select sync file or TOML cuts file and load cuts from it."""
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()

    selected_file = filedialog.askopenfilename(
        title="Select Sync File or Cuts TOML File",
        filetypes=[
            ("TOML cuts files", "*.toml"),
            ("Text files (sync)", "*.txt"),
            ("All files", "*.*"),
        ],
        initialdir=Path(video_path).parent,
    )

    if not selected_file:
        return [], False, None

    file_path = Path(selected_file)

    # Check if it's a TOML file (cuts file)
    if file_path.suffix.lower() == ".toml":
        try:
            cuts = load_cuts_from_toml_file(selected_file)
            if cuts:
                print(f"Loaded {len(cuts)} cuts from TOML file")
                return cuts, False, None  # False = not a sync file, just cuts
            else:
                messagebox.showwarning(
                    "No Cuts Found", "The selected TOML file does not contain any cuts."
                )
                return [], False, None
        except Exception as e:
            messagebox.showerror("Error", f"Error loading TOML file: {e}")
            return [], False, None

    # Otherwise, treat as sync file (TXT format)
    video_name = Path(video_path).name
    cuts = []
    sync_data = {}  # Store all sync data for batch processing

    try:
        with open(selected_file, encoding="utf-8") as f:
            lines = f.readlines()

        for line in lines:
            if line.strip():
                # Parse sync file format: video_file new_name initial_frame final_frame
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        video_file = parts[0]
                        new_name = parts[1]
                        initial_frame = int(parts[2])
                        final_frame = int(parts[3])

                        # Store sync data for all videos
                        sync_data[video_file] = {
                            "new_name": new_name,
                            "initial_frame": initial_frame,
                            "final_frame": final_frame,
                        }

                        # If this is the current video, add to cuts
                        if video_name in video_file:
                            cuts.append((initial_frame, final_frame))

                    except ValueError:
                        continue

        return cuts, True, sync_data
    except Exception as e:
        print(f"Error loading sync file: {e}")
        messagebox.showerror("Error", f"Error loading sync file: {e}")
        return [], False, None


def batch_process_sync_videos(video_path, sync_data):
    """Process all videos in directory according to sync file."""
    if not sync_data:
        return False

    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = video_dir / f"vailacut_sync_{video_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    processed_count = 0

    for video_file, sync_info in sync_data.items():
        video_path_full = video_dir / video_file

        if not video_path_full.exists():
            print(f"Warning: Video file {video_file} not found")
            continue

        try:
            # Get precise video metadata
            metadata = get_precise_video_metadata(video_path_full)
            total_frames = metadata.get("nb_frames") or (
                int(metadata.get("duration", 0) * metadata["fps"])
                if metadata.get("duration")
                else int(cv2.VideoCapture(str(video_path_full)).get(cv2.CAP_PROP_FRAME_COUNT))
            )

            # Process the cut
            start_frame = sync_info["initial_frame"]
            end_frame = sync_info["final_frame"]

            # Skip if start frame is beyond video length
            if start_frame >= total_frames:
                print(f"Warning: Start frame {start_frame} beyond video length for {video_file}")
                continue

            # Adjust end frame if needed
            actual_end_frame = min(end_frame, total_frames - 1)

            output_path = output_dir / sync_info["new_name"]

            # Use ffmpeg for precise cutting
            success = cut_video_with_ffmpeg(
                video_path_full, output_path, start_frame, actual_end_frame, metadata
            )

            if success:
                processed_count += 1
                print(
                    f"Processed: {video_file} -> {sync_info['new_name']} (FPS: {metadata['fps']:.6f})"
                )
            else:
                print(f"Error processing: {video_file}")

        except Exception as e:
            print(f"Error processing {video_file}: {str(e)}")

    return processed_count > 0


def play_video_with_cuts(video_path):
    pygame.init()

    # Initialize video capture
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print("Error opening video file")
        return

    # Get precise video metadata using ffprobe
    metadata = get_precise_video_metadata(video_path)
    fps = metadata["fps"]  # Use precise float FPS
    original_width = metadata["width"]
    original_height = metadata["height"]

    # Calculate total frames from metadata if available, otherwise use OpenCV
    total_frames = metadata.get("nb_frames")
    if total_frames is None:
        if metadata.get("duration") and fps > 0:
            total_frames = int(round(metadata["duration"] * fps))
        else:
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(
        f"Video metadata: {original_width}x{original_height}, FPS: {fps:.6f}, Frames: {total_frames}"
    )

    # Initialize window with adjusted size
    screen_info = pygame.display.Info()
    max_width = screen_info.current_w - 100  # Leave some margin
    max_height = screen_info.current_h - 100  # Leave some margin

    # Calculate aspect ratio
    aspect_ratio = original_width / original_height

    # Calculate initial window size maintaining aspect ratio
    if original_width / max_width > original_height / max_height:
        # Width is the limiting factor
        window_width = max_width
        window_height = int(window_width / aspect_ratio)
    else:
        # Height is the limiting factor
        window_height = max_height
        window_width = int(window_height * aspect_ratio)

    # Ensure minimum size
    window_width = max(640, min(window_width, max_width))
    window_height = max(480, min(window_height, max_height))

    # UI layout constants
    control_height = 80
    audio_height = 150
    show_audio = False
    audio_data = None
    audio_loaded = False
    sample_rate = 44100
    audio_wave_path = None
    audio_muted = False
    audio_ready = False
    audio_music_loaded = False
    loop_enabled = True

    # Get video filename for window title
    video_filename = Path(video_path).name

    def set_display_mode():
        total_h = window_height + control_height + (audio_height if show_audio else 0)
        return pygame.display.set_mode((window_width, total_h), pygame.RESIZABLE)

    def update_caption():
        pygame.display.set_caption(
            f"{video_filename} (FPS: {fps:.2f}) | A:Audio M:Mute 0:AutoFit | Space:Play/Pause | ←→:Frame | S:Start E:End R:Reset DEL/D:Remove | L:List | F:Load TOML | Home/End:Jump Cut | PgUp/PgDn:Next Marker | G:Frame T:Time I/P:FPS | H:Help ESC:Save"
        )

    def auto_fit_window():
        """Auto-fit window to current display while respecting margins and aspect ratio."""
        nonlocal window_width, window_height
        screen_info = pygame.display.Info()
        # Margins to avoid covering taskbar/title; keep similar to previous 100px margin
        max_w = max(640, screen_info.current_w - 100)
        max_h = max(480, screen_info.current_h - 140)  # allow for title bar and taskbar

        # Available height for video after controls/audio
        avail_h = max_h - control_height - (audio_height if show_audio else 0)
        avail_h = max(240, avail_h)

        # Fit by aspect ratio
        if original_width / max_w > original_height / avail_h:
            window_width = max_w
            window_height = int(window_width / aspect_ratio)
        else:
            window_height = avail_h
            window_width = int(window_height * aspect_ratio)

        # Clamp to bounds
        window_width = max(640, min(window_width, max_w))
        window_height = max(480, min(window_height, avail_h))

        return set_display_mode()

    # Initialize window
    screen = auto_fit_window()
    update_caption()

    # Initialize variables
    clock = pygame.time.Clock()
    frame_count = 0
    paused = True
    cuts = []  # List to store (start, end) frame pairs
    current_start = None
    using_sync_file = False
    sync_data = None  # Store sync data for batch processing

    # Load existing cuts if available
    cuts = load_cuts_from_toml(video_path)
    if len(cuts) > 0:
        print(f"Loaded {len(cuts)} cuts from cuts file")

    def ensure_audio_player():
        nonlocal audio_ready
        if audio_ready:
            return True
        try:
            pygame.mixer.pre_init(frequency=sample_rate, size=-16, channels=1)
            pygame.mixer.init()
            audio_ready = True
            return True
        except Exception as e:
            print(f"Audio init failed: {e}")
            return False

    def sync_audio_to_frame(frame_idx: int):
        """Seek audio playback to match current frame position."""
        if not audio_ready or not audio_loaded or audio_muted:
            return
        try:
            pos_seconds = (frame_idx / fps) if fps > 0 else 0.0
            pygame.mixer.music.set_pos(pos_seconds)
        except Exception as e:
            print(f"Audio seek failed: {e}")

    def start_audio_playback(frame_idx: int):
        if not audio_ready or not audio_loaded or audio_muted:
            return
        try:
            pos_seconds = (frame_idx / fps) if fps > 0 else 0.0
            # start playback; looping once is enough for single video
            pygame.mixer.music.play(loops=0, start=pos_seconds)
        except Exception as e:
            print(f"Audio play failed: {e}")

    def stop_audio_playback():
        if audio_ready:
            try:
                pygame.mixer.music.stop()
            except Exception:
                pass

    def ensure_music_loaded():
        nonlocal audio_music_loaded
        if audio_music_loaded:
            return True
        if not (audio_ready and audio_loaded and audio_wave_path):
            return False
        try:
            pygame.mixer.music.load(audio_wave_path)
            audio_music_loaded = True
            return True
        except Exception as e:
            print(f"Audio load failed: {e}")
            return False

    def draw_waveform(surface, current_f, fps_val, sr, data, x_start, y_start, w, h):
        """Draw the audio waveform centered on the current frame."""
        if data is None:
            font = pygame.font.Font(None, 24)
            text = font.render("No Audio Data (or Loading...)", True, (120, 120, 120))
            surface.blit(text, (x_start + 10, y_start + h // 2))
            return

        pygame.draw.rect(surface, (0, 0, 20), (x_start, y_start, w, h))

        # Center line (playhead)
        center_x = w // 2
        current_time = current_f / fps_val if fps_val > 0 else 0.0
        center_sample_idx = int(current_time * sr)

        # Display about 2 seconds of audio across the width
        samples_per_pixel = max(1, int((sr * 2.0) / w))
        half_w_samples = (w // 2) * samples_per_pixel
        start_idx = max(0, center_sample_idx - half_w_samples)
        end_idx = min(len(data), center_sample_idx + half_w_samples)
        if end_idx <= start_idx:
            return

        chunk = data[start_idx:end_idx]
        step = max(1, samples_per_pixel)
        pixel_offset = (start_idx - (center_sample_idx - half_w_samples)) // samples_per_pixel
        screen_x_start = x_start + pixel_offset

        points = []
        for i in range(0, len(chunk), step):
            screen_x = screen_x_start + (i // step)
            if screen_x >= x_start + w:
                break
            amp = chunk[i]
            screen_y = y_start + (h // 2) - int(amp * (h / 2))
            points.append((screen_x, screen_y))

        if len(points) > 1:
            pygame.draw.lines(surface, (255, 140, 0), False, points, 1)

        pygame.draw.line(
            surface,
            (0, 200, 255),
            (x_start + center_x, y_start),
            (x_start + center_x, y_start + h),
            1,
        )

        font = pygame.font.Font(None, 20)
        ts = font.render(f"{current_time:.6f}s", True, (255, 255, 255))
        surface.blit(ts, (x_start + 5, y_start + 5))

    def draw_controls():
        base_y = window_height + (audio_height if show_audio else 0)
        slider_surface = pygame.Surface((window_width, control_height))
        slider_surface.fill((30, 30, 30))

        # Draw slider bar
        slider_width = int(window_width * 0.8)
        slider_x = (window_width - slider_width) // 2
        slider_y = 30
        slider_height = 10
        pygame.draw.rect(
            slider_surface,
            (60, 60, 60),
            (slider_x, slider_y, slider_width, slider_height),
        )

        # Draw slider handle
        slider_pos = slider_x + int((frame_count / total_frames) * slider_width)
        pygame.draw.circle(
            slider_surface,
            (255, 255, 255),
            (slider_pos, slider_y + slider_height // 2),
            8,
        )

        # Draw frame information and cut markers
        font = pygame.font.Font(None, 24)
        time_seconds = frame_count / fps if fps > 0 else 0.0
        time_total = total_frames / fps if fps > 0 else 0.0
        frame_text = font.render(
            f"Frame: {frame_count + 1}/{total_frames} ({time_seconds:.6f}s/{time_total:.6f}s)",
            True,
            (255, 255, 255),
        )
        slider_surface.blit(frame_text, (10, 10))

        # Draw current cut information
        if current_start is not None:
            cut_text = font.render(f"Current Cut Start: {current_start + 1}", True, (0, 255, 0))
            slider_surface.blit(cut_text, (10, 50))

        # Draw number of cuts and sync status
        sync_status = " (SYNC)" if using_sync_file else ""
        cuts_text = font.render(f"Cuts: {len(cuts)}{sync_status}", True, (255, 255, 255))
        slider_surface.blit(cuts_text, (window_width - 150, 50))

        # Smaller font for buttons
        button_font = pygame.font.Font(None, 15)

        # Button width and x position (aligned)
        button_width = 70
        button_x = window_width - button_width - 10

        # Loop button (above Help)
        loop_button_rect = pygame.Rect(button_x, 10, button_width, 22)
        loop_color = (60, 120, 60) if loop_enabled else (90, 90, 90)
        pygame.draw.rect(slider_surface, loop_color, loop_button_rect)
        loop_text = button_font.render(
            "Loop" if loop_enabled else "Loop off", True, (255, 255, 255)
        )
        loop_text_rect = loop_text.get_rect(center=loop_button_rect.center)
        slider_surface.blit(loop_text, loop_text_rect)

        # Help button (below Loop)
        help_button_rect = pygame.Rect(button_x, 35, button_width, 22)
        pygame.draw.rect(slider_surface, (100, 100, 100), help_button_rect)
        help_text = button_font.render("Help", True, (255, 255, 255))
        text_rect = help_text.get_rect(center=help_button_rect.center)
        slider_surface.blit(help_text, text_rect)

        if show_audio:
            audio_indicator = font.render("[AUDIO ON]", True, (0, 255, 0))
            slider_surface.blit(audio_indicator, (window_width - 260, 10))

        screen.blit(slider_surface, (0, base_y))
        return (
            slider_x,
            slider_width,
            slider_y,
            slider_height,
            help_button_rect,
            loop_button_rect,
            base_y,
        )

    def get_all_cut_markers():
        """Get a sorted flat list of all cut markers (start and end frames)."""
        markers = []
        for start, end in cuts:
            markers.append(start)
            markers.append(end)
        # Remove duplicates and sort
        markers = sorted(list(set(markers)))
        return markers

    def find_next_marker(frame_pos):
        """Find the next marker after the current frame position."""
        markers = get_all_cut_markers()
        for marker in markers:
            if marker > frame_pos:
                return marker
        return None

    def find_previous_marker(frame_pos):
        """Find the previous marker before the current frame position."""
        markers = get_all_cut_markers()
        for marker in reversed(markers):
            if marker < frame_pos:
                return marker
        return None

    def find_current_cut(frame_pos):
        """Find which cut contains the current frame, or return None."""
        for i, (start, end) in enumerate(cuts):
            if start <= frame_pos <= end:
                return i, start, end
        return None, None, None

    def find_next_cut(frame_pos):
        """Find the next cut after the current frame position."""
        for i, (start, end) in enumerate(cuts):
            if start > frame_pos:
                return i, start, end
        return None, None, None

    def find_previous_cut(frame_pos):
        """Find the previous cut before the current frame position."""
        for i in range(len(cuts) - 1, -1, -1):
            start, end = cuts[i]
            if end < frame_pos:
                return i, start, end
        return None, None, None

    def show_help_dialog():
        """Display help information directly in pygame window with scroll support."""
        help_lines = [
            "Video Cutting Controls:",
            "",
            "Navigation:",
            "- Space: Play/Pause",
            "- Right Arrow: Next Frame (when paused)",
            "- Left Arrow: Previous Frame (when paused)",
            "- Up Arrow: Fast Forward (60 frames)",
            "- Down Arrow: Rewind (60 frames)",
            "- G: Go to Frame Number (enter frame number as int)",
            "- T: Go to Time (enter time in seconds as float)",
            "- 0: Auto-fit window to screen",
            "",
            "Audio Controls:",
            "- A: Toggle Audio Waveform Panel",
            "- M: Mute/Unmute Audio",
            "- Loop Button: Enable/Disable video/audio looping",
            "",
            "Cutting Operations:",
            "- S: Mark Start Frame",
            "- E: Mark End Frame",
            "- R: Reset Current Cut",
            "- DELETE or D: Remove Last Cut",
            "- L: List All Cuts",
            "",
            "Cut Navigation:",
            "- Home: Jump to Start of Current Cut",
            "- End: Jump to End of Current Cut",
            "- Page Up: Jump to Previous Marker",
            "- Page Down: Jump to Next Marker",
            "",
            "File Operations:",
            "- F: Load Sync File or Cuts TOML File",
            "- I or P: Input Manual FPS",
            "- ESC: Save cuts to TOML file and optionally generate videos",
            "",
            "Help:",
            "- H: Show this help dialog",
            "",
            "Mouse Controls:",
            "- Click on slider: Jump to frame",
            "- Click 'Loop' button: Toggle looping",
            "- Click 'Help' button: Show this dialog",
            "- Mouse Wheel: Scroll help text",
            "- Arrow Up/Down: Scroll help text",
            "- Drag window edges: Resize window",
            "",
            "Display Features:",
            "- Audio waveform shows synchronized audio with orange line",
            "- Time precision: 6 decimal places (.6f) for scientific accuracy",
            "- Auto-fit adjusts window to maximize use of screen space",
            "",
            "Press ESC or click to close this help",
        ]

        total_overlay_h = window_height + control_height + (audio_height if show_audio else 0)
        overlay = pygame.Surface((window_width, total_overlay_h))
        overlay.set_alpha(230)
        overlay.fill((0, 0, 0))

        # Render help text
        font = pygame.font.Font(None, 24)
        line_height = 28

        # Calculate total height needed
        total_height = len(help_lines) * line_height + 40
        visible_height = total_overlay_h - 40  # Leave 20px margin top and bottom
        scroll_offset = 0
        max_scroll = max(0, total_height - visible_height)

        waiting_for_input = True
        while waiting_for_input:
            # Clear overlay
            overlay.fill((0, 0, 0))

            # Render visible portion of help text
            start_line = max(0, scroll_offset // line_height)
            end_line = min(len(help_lines), start_line + (visible_height // line_height) + 1)
            y_offset = 20 - (scroll_offset % line_height)

            for i in range(start_line, end_line):
                line = help_lines[i]
                text_surface = font.render(line, True, (255, 255, 255))
                y_pos = y_offset + (i - start_line) * line_height
                if 0 <= y_pos <= visible_height:
                    overlay.blit(text_surface, (20, y_pos))

            # Display scroll indicator if needed
            if max_scroll > 0:
                scroll_bar_height = int(visible_height * (visible_height / total_height))
                scroll_bar_y = 20 + int(
                    (scroll_offset / max_scroll) * (visible_height - scroll_bar_height)
                )
                pygame.draw.rect(
                    overlay,
                    (100, 100, 100),
                    (window_width - 20, scroll_bar_y, 10, scroll_bar_height),
                )

                # Show scroll hint
                hint_font = pygame.font.Font(None, 20)
                hint_text = hint_font.render(
                    "Use mouse wheel or arrows to scroll", True, (200, 200, 200)
                )
                overlay.blit(hint_text, (window_width - 280, total_overlay_h - 25))

            # Display help and wait for key/click
            screen.blit(overlay, (0, 0))
            pygame.display.flip()

            for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        waiting_for_input = False
                    elif event.key == pygame.K_UP:
                        scroll_offset = max(0, scroll_offset - line_height * 3)
                    elif event.key == pygame.K_DOWN:
                        scroll_offset = min(max_scroll, scroll_offset + line_height * 3)
                    else:
                        waiting_for_input = False
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    if event.button == 4:  # Scroll up
                        scroll_offset = max(0, scroll_offset - line_height * 3)
                    elif event.button == 5:  # Scroll down
                        scroll_offset = min(max_scroll, scroll_offset + line_height * 3)
                    else:
                        waiting_for_input = False
                elif event.type == pygame.MOUSEWHEEL:
                    scroll_offset = max(
                        0, min(max_scroll, scroll_offset - event.y * line_height * 3)
                    )
                elif event.type == pygame.QUIT:
                    waiting_for_input = False
                    global running
                    running = False

    def save_and_generate_videos():
        nonlocal cuts, video_path, using_sync_file, sync_data, fps

        if not cuts:
            messagebox.showinfo("Info", "No cuts were marked!")
            return False

        # First save cuts to TOML file with FPS information
        # Precompute timestamp and planned output dir for TOML/processing consistency
        timestamp_now = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        planned_dir_name = (
            f"{video_name}_sync_vailacut_{timestamp_now}"
            if using_sync_file
            else f"{video_name}_vailacut_{timestamp_now}"
        )
        planned_output_dir = Path(video_path).parent / planned_dir_name
        per_cut_files = [f"{video_name}_frame_{start + 1}_to_{end + 1}.mp4" for start, end in cuts]

        save_cuts_to_toml(
            video_path, cuts, fps, output_dir=planned_output_dir, per_cut_outputs=per_cut_files
        )

        # Close pygame temporarily instead of fully quitting it
        pygame.display.quit()
        print("Pygame display closed before video processing")

        # If using sync file, process all videos in batch
        if using_sync_file and sync_data:
            if messagebox.askyesno(
                "Sync Mode",
                "Sync mode detected. Do you want to process all videos in the directory according to the sync file?",
            ):
                success = batch_process_sync_videos(video_path, sync_data)
                if success:
                    messagebox.showinfo(
                        "Sync Processing Complete",
                        "All videos have been processed according to the sync file!",
                    )
                return success
        else:
            # Regular cut processing
            if messagebox.askyesno(
                "Generate Videos",
                "Cuts saved to text file. Do you want to generate video files now?",
            ):
                success = save_cuts(
                    video_path, cuts, using_sync_file, fixed_timestamp=timestamp_now
                )

                # Ask if user wants to apply the same cuts to all videos in the directory
                if success and messagebox.askyesno(
                    "Batch Processing",
                    "Do you want to apply these same cuts to all other videos in this directory?",
                ):
                    batch_process_videos(video_path, cuts, using_sync_file)

                return success
        return True

    def batch_process_videos(source_video_path, cuts, from_sync_file=False):
        """Apply the same cuts to all videos in the same directory."""
        if not cuts:
            messagebox.showinfo("Info", "No cuts to apply!")
            return

        # Get the directory of the source video
        source_dir = Path(source_video_path).parent
        source_name = Path(source_video_path).name

        # Get all video files in the directory
        video_extensions = [
            ".mp4",
            ".MP4",
            ".avi",
            ".AVI",
            ".mov",
            ".MOV",
            ".mkv",
            ".MKV",
        ]
        video_files = []

        for ext in video_extensions:
            video_files.extend(list(source_dir.glob(f"*{ext}")))

        # Remove the source video from the list
        video_files = [v for v in video_files if v.name != source_name]

        if not video_files:
            messagebox.showinfo("Info", "No other video files found in this directory.")
            return

        # Create a progress dialog
        root = Tk()
        root.title("Batch Processing")
        root.geometry("400x150")

        from tkinter import ttk

        label = ttk.Label(root, text="Processing videos in batch...")
        label.pack(pady=10)

        progress = ttk.Progressbar(root, orient="horizontal", length=300, mode="determinate")
        progress.pack(pady=10)
        progress["maximum"] = len(video_files)

        status_label = ttk.Label(root, text="")
        status_label.pack(pady=5)

        # Create output directory with improved naming including source video basename
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        source_name = Path(source_video_path).stem
        prefix = "sync_" if from_sync_file else ""
        output_dir = source_dir / f"vailacut_{prefix}{source_name}_batch_{timestamp}"
        output_dir.mkdir(exist_ok=True)

        processed_count = 0

        def process_next_video():
            nonlocal processed_count

            if processed_count < len(video_files):
                video_path = str(video_files[processed_count])
                video_name = Path(video_path).stem

                # Get metadata to get FPS for save_cuts_to_toml
                try:
                    metadata = get_precise_video_metadata(video_path)
                    video_fps = metadata.get("fps", None)
                except Exception:
                    video_fps = None

                # Salvar informações de corte para cada vídeo processado em TOML
                save_cuts_to_toml(video_path, cuts, video_fps)

                status_label.config(text=f"Processing: {video_name}")

                try:
                    # Get precise video metadata
                    metadata = get_precise_video_metadata(video_path)
                    total_frames = metadata.get("nb_frames") or (
                        int(metadata.get("duration", 0) * metadata["fps"])
                        if metadata.get("duration")
                        else int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
                    )

                    # Process each cut
                    for i, (start_frame, end_frame) in enumerate(cuts):
                        # Skip if end frame is beyond video length
                        if start_frame >= total_frames:
                            continue

                        # Adjust end frame if needed
                        actual_end_frame = min(end_frame, total_frames - 1)

                        output_path = (
                            output_dir
                            / f"{video_name}_frame_{start_frame}_to_{actual_end_frame}.mp4"
                        )

                        # Use ffmpeg for precise cutting
                        success = cut_video_with_ffmpeg(
                            video_path, output_path, start_frame, actual_end_frame, metadata
                        )
                        if not success:
                            status_label.config(
                                text=f"Warning: Cut {i + 1} failed for {video_name}"
                            )

                except Exception as e:
                    status_label.config(text=f"Error processing {video_name}: {str(e)}")

                processed_count += 1
                progress["value"] = processed_count
                root.after(100, process_next_video)
            else:
                status_label.config(text="Batch processing complete!")
                root.after(2000, root.destroy)

        # Start processing
        root.after(100, process_next_video)
        root.mainloop()

        messagebox.showinfo(
            "Batch Complete",
            f"Processed {processed_count} videos. Output saved to {output_dir}",
        )

    def save_cuts(video_path, cuts, from_sync_file=False, fixed_timestamp=None):
        if not cuts:
            messagebox.showinfo("Info", "No cuts were marked!")
            return False

        # Create output directory with improved naming including video basename
        # Format: {video_name}_vailacut_{timestamp} (not batch processing)
        timestamp = fixed_timestamp or datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        video_name = Path(video_path).stem
        if from_sync_file:
            # For sync files, keep the sync prefix before vailacut
            output_dir = Path(video_path).parent / f"{video_name}_sync_vailacut_{timestamp}"
        else:
            # For regular processing: basename before vailacut
            output_dir = Path(video_path).parent / f"{video_name}_vailacut_{timestamp}"
        output_dir.mkdir(exist_ok=True)

        # Get precise video metadata
        metadata = get_precise_video_metadata(video_path)

        # Process each cut
        for i, (start_frame, end_frame) in enumerate(cuts):
            # Use 1-based numbering in filenames to match TOML/UI and avoid off-by-one confusion
            output_path = (
                output_dir / f"{video_name}_frame_{start_frame + 1}_to_{end_frame + 1}.mp4"
            )

            # Use ffmpeg for precise cutting
            success = cut_video_with_ffmpeg(
                video_path, output_path, start_frame, end_frame, metadata
            )
            if not success:
                print(f"Warning: Failed to create cut {i + 1} for {video_name}")

        return True

    running = True
    while running:
        if paused:
            # Quando pausado, vamos usar o método set para posicionar no frame exato
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        else:
            # Quando em reprodução, apenas leia o próximo frame sem reposicionar
            ret, frame = cap.read()
            if ret:
                frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                # Final do vídeo alcançado
                if loop_enabled:
                    frame_count = 0
                    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                    ret, frame = cap.read()
                    if (
                        ret
                        and not audio_muted
                        and audio_loaded
                        and ensure_audio_player()
                        and ensure_music_loaded()
                    ):
                        start_audio_playback(frame_count)
                else:
                    # Stop at last frame and pause
                    frame_count = max(total_frames - 1, 0)
                    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                    ret, frame = cap.read()
                    paused = True
                    stop_audio_playback()

        if not ret:
            break

        # Calculate scaling factors for width and height
        scale_w = window_width / original_width
        scale_h = window_height / original_height
        scale = min(scale_w, scale_h)  # Use the smaller scale to fit in window

        # Calculate new dimensions
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)

        # Calculate position to center the frame
        x_offset = (window_width - new_width) // 2
        y_offset = (window_height - new_height) // 2

        # Resize frame while maintaining aspect ratio
        frame = cv2.resize(frame, (new_width, new_height))

        # Convert frame to pygame surface
        frame_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        # Fill screen with black
        screen.fill((0, 0, 0))

        # Draw frame at centered position
        screen.blit(frame_surface, (x_offset, y_offset))

        # Draw audio waveform panel if enabled
        if show_audio:
            draw_waveform(
                screen,
                frame_count,
                fps,
                sample_rate,
                audio_data,
                0,
                window_height,
                window_width,
                audio_height,
            )

        # Draw controls
        (
            slider_x,
            slider_width,
            slider_y,
            slider_height,
            help_button_rect,
            loop_button_rect,
            base_y,
        ) = draw_controls()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                min_total_h = control_height + (audio_height if show_audio else 0) + 100
                if new_h > min_total_h:
                    # Determine available height for video region
                    target_video_h = new_h - control_height - (audio_height if show_audio else 0)
                    target_video_h = max(240, target_video_h)

                    window_width = max(640, new_w)
                    video_h_from_width = int(window_width / aspect_ratio)

                    if video_h_from_width > target_video_h:
                        window_height = target_video_h
                        window_width = int(window_height * aspect_ratio)
                    else:
                        window_height = video_h_from_width

                    screen = set_display_mode()

            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    if save_and_generate_videos():
                        messagebox.showinfo(
                            "Success",
                            "Cuts saved to text file and videos generated (if selected)!",
                        )
                    running = False
                elif event.key == pygame.K_a:
                    show_audio = not show_audio
                    if show_audio and not audio_loaded:
                        # Quick feedback
                        font = pygame.font.Font(None, 36)
                        msg = font.render("Loading Audio...", True, (255, 255, 255))
                        screen.blit(
                            msg, (max(0, window_width // 2 - 120), max(0, window_height // 2 - 20))
                        )
                        pygame.display.flip()

                        data, sr = extract_audio_data(video_path)
                        if data is not None:
                            audio_data = data
                            sample_rate = sr
                            audio_loaded = True
                            if ensure_audio_player():
                                if audio_wave_path is None:
                                    audio_wave_path = write_wav_from_pcm(audio_data, sample_rate)
                                if ensure_music_loaded() and not audio_muted and not paused:
                                    start_audio_playback(frame_count)
                            print("Audio loaded successfully.")
                        else:
                            print("Failed to load audio or no audio track.")

                    screen = auto_fit_window()
                    update_caption()
                elif event.key == pygame.K_m:
                    audio_muted = not audio_muted
                    if audio_muted:
                        stop_audio_playback()
                    else:
                        if (
                            audio_loaded
                            and ensure_audio_player()
                            and ensure_music_loaded()
                            and not paused
                        ):
                            sync_audio_to_frame(frame_count)
                            pygame.mixer.music.unpause() if pygame.mixer.music.get_busy() else start_audio_playback(
                                frame_count
                            )
                    update_caption()
                elif event.key == pygame.K_0:
                    screen = auto_fit_window()
                    update_caption()
                elif event.key == pygame.K_SPACE:
                    paused = not paused
                    if paused:
                        stop_audio_playback()
                    else:
                        if frame_count >= total_frames - 1 and loop_enabled:
                            frame_count = 0
                            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
                        if (
                            audio_loaded
                            and ensure_audio_player()
                            and ensure_music_loaded()
                            and not audio_muted
                        ):
                            start_audio_playback(frame_count)
                elif event.key == pygame.K_RIGHT and paused:
                    frame_count = min(frame_count + 1, total_frames - 1)
                elif event.key == pygame.K_LEFT and paused:
                    frame_count = max(frame_count - 1, 0)
                elif event.key == pygame.K_UP and paused:
                    frame_count = min(frame_count + 60, total_frames - 1)
                elif event.key == pygame.K_DOWN and paused:
                    frame_count = max(frame_count - 60, 0)
                elif event.key == pygame.K_s and paused:
                    current_start = frame_count
                    print(f"Start frame marked: {frame_count + 1}")
                elif event.key == pygame.K_e and paused and current_start is not None:
                    if frame_count > current_start:
                        cuts.append((current_start, frame_count))
                        print(f"Cut marked: {current_start + 1} to {frame_count + 1}")
                        current_start = None
                    else:
                        print("Error: End frame must be after start frame!")
                elif event.key == pygame.K_r:  # Reset current cut
                    current_start = None
                    print("Current cut reset")
                elif event.key == pygame.K_DELETE:  # Delete last cut
                    if cuts:
                        cuts.pop()
                        print("Last cut removed")
                elif event.key == pygame.K_d:  # Also support D key for delete (common expectation)
                    if cuts:
                        cuts.pop()
                        print("Last cut removed (D key)")
                elif event.key == pygame.K_l:  # List all cuts
                    if cuts:
                        cuts_info = "\n".join(
                            [
                                f"Cut {i + 1}: Frame {start + 1} to {end + 1}"
                                for i, (start, end) in enumerate(cuts)
                            ]
                        )
                        messagebox.showinfo("Cuts List", cuts_info)
                    else:
                        messagebox.showinfo("Cuts List", "No cuts marked yet")
                elif event.key == pygame.K_f:  # Load sync file or TOML cuts file
                    new_cuts, is_sync, new_sync_data = load_sync_file_from_dialog(video_path)
                    if new_cuts:
                        cuts = new_cuts
                        using_sync_file = is_sync
                        sync_data = new_sync_data
                        if is_sync:
                            print(f"Loaded {len(cuts)} cuts from sync file")
                            messagebox.showinfo(
                                "Sync File Loaded",
                                f"Loaded {len(cuts)} cuts from sync file",
                            )
                        else:
                            print(f"Loaded {len(cuts)} cuts from TOML file")
                            messagebox.showinfo(
                                "Cuts File Loaded",
                                f"Loaded {len(cuts)} cuts from TOML file",
                            )
                    else:
                        print("No file selected or error loading file")
                elif event.key == pygame.K_HOME and paused:  # Jump to start of current cut
                    cut_idx, start, end = find_current_cut(frame_count)
                    if cut_idx is not None:
                        frame_count = start
                        print(f"Jumped to start of cut {cut_idx + 1}: Frame {start + 1}")
                    elif cuts:
                        # If not in a cut, jump to start of first cut
                        frame_count = cuts[0][0]
                        print(f"Jumped to start of first cut: Frame {cuts[0][0] + 1}")
                    else:
                        print("No cuts available")
                elif event.key == pygame.K_END and paused:  # Jump to end of current cut
                    cut_idx, start, end = find_current_cut(frame_count)
                    if cut_idx is not None:
                        frame_count = end
                        print(f"Jumped to end of cut {cut_idx + 1}: Frame {end + 1}")
                    elif cuts:
                        # If not in a cut, jump to end of last cut
                        frame_count = cuts[-1][1]
                        print(f"Jumped to end of last cut: Frame {cuts[-1][1] + 1}")
                    else:
                        print("No cuts available")
                elif event.key == pygame.K_PAGEUP and paused:  # Jump to previous marker
                    prev_marker = find_previous_marker(frame_count)
                    if prev_marker is not None:
                        frame_count = prev_marker
                        print(f"Jumped to previous marker: Frame {prev_marker + 1}")
                    else:
                        markers = get_all_cut_markers()
                        if markers:
                            frame_count = markers[0]
                            print(f"Jumped to first marker: Frame {markers[0] + 1}")
                        else:
                            print("No markers available")
                elif event.key == pygame.K_PAGEDOWN and paused:  # Jump to next marker
                    next_marker = find_next_marker(frame_count)
                    if next_marker is not None:
                        frame_count = next_marker
                        print(f"Jumped to next marker: Frame {next_marker + 1}")
                    else:
                        markers = get_all_cut_markers()
                        if markers:
                            frame_count = markers[-1]
                            print(f"Jumped to last marker: Frame {markers[-1] + 1}")
                        else:
                            print("No markers available")
                elif event.key == pygame.K_i or event.key == pygame.K_p:  # Input manual FPS
                    # Temporarily close pygame display to show tkinter dialog
                    pygame.display.quit()
                    root_fps = Tk()
                    root_fps.withdraw()
                    new_fps = simpledialog.askfloat(
                        "Input FPS",
                        f"Enter new FPS value (current: {fps:.6f}):",
                        initialvalue=fps,
                        minvalue=0.1,
                        maxvalue=1000.0,
                    )
                    root_fps.destroy()
                    # Reinitialize pygame display
                    screen = set_display_mode()
                    if new_fps is not None and new_fps > 0:
                        fps = float(new_fps)
                        print(f"FPS updated to: {fps:.6f}")
                        # Update window title with new FPS
                        update_caption()
                        messagebox.showinfo("FPS Updated", f"FPS set to {fps:.6f}")
                    else:
                        # Keep original caption if FPS was not updated
                        update_caption()
                    pygame.display.flip()
                elif event.key == pygame.K_h:  # Show help dialog
                    show_help_dialog()
                elif event.key == pygame.K_g and paused:  # Go to frame number
                    # Temporarily close pygame display to show tkinter dialog
                    pygame.display.quit()
                    root_frame = Tk()
                    root_frame.withdraw()
                    target_frame = simpledialog.askinteger(
                        "Go to Frame",
                        f"Enter frame number (current: {frame_count + 1}, max: {total_frames}):",
                        initialvalue=frame_count + 1,
                        minvalue=1,
                        maxvalue=total_frames,
                    )
                    root_frame.destroy()
                    # Reinitialize pygame display
                    screen = set_display_mode()
                    update_caption()
                    if target_frame is not None:
                        # Convert to 0-based frame index
                        frame_count = min(max(0, target_frame - 1), total_frames - 1)
                        print(f"Jumped to frame: {frame_count + 1}")
                        paused = True  # Pause when jumping to frame
                    pygame.display.flip()
                elif event.key == pygame.K_t and paused:  # Go to time
                    # Temporarily close pygame display to show tkinter dialog
                    pygame.display.quit()
                    root_time = Tk()
                    root_time.withdraw()
                    current_time = frame_count / fps if fps > 0 else 0.0
                    max_time = total_frames / fps if fps > 0 else 0.0
                    target_time = simpledialog.askfloat(
                        "Go to Time",
                        f"Enter time in seconds (current: {current_time:.2f}s, max: {max_time:.2f}s):",
                        initialvalue=current_time,
                        minvalue=0.0,
                        maxvalue=max_time,
                    )
                    root_time.destroy()
                    # Reinitialize pygame display
                    screen = set_display_mode()
                    update_caption()
                    if target_time is not None and fps > 0:
                        # Convert time to frame (0-based)
                        target_frame_float = target_time * fps
                        frame_count = min(max(0, int(round(target_frame_float))), total_frames - 1)
                        actual_time = frame_count / fps
                        print(f"Jumped to time: {actual_time:.2f}s (frame: {frame_count + 1})")
                        paused = True  # Pause when jumping to time
                    pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if help_button_rect.collidepoint(x, y - base_y):
                    show_help_dialog()
                elif loop_button_rect.collidepoint(x, y - base_y):
                    loop_enabled = not loop_enabled
                    if not loop_enabled:
                        stop_audio_playback()
                    screen = set_display_mode()
                    update_caption()
                elif slider_y <= y - base_y <= slider_y + slider_height:
                    rel_x = x - slider_x
                    frame_count = int((rel_x / slider_width) * total_frames)
                    frame_count = max(0, min(frame_count, total_frames - 1))
                    paused = True

        if paused:
            # Se pausado, não limitamos a taxa de FPS para que a interface seja responsiva
            clock.tick(60)  # Taxa de atualização da interface
        else:
            # Se em reprodução, limitamos à taxa de FPS do vídeo
            clock.tick(fps)

    cap.release()
    stop_audio_playback()
    if audio_ready:
        try:
            pygame.mixer.quit()
        except Exception:
            pass
    if audio_wave_path:
        try:
            os.remove(audio_wave_path)
        except Exception:
            pass
    pygame.quit()


def get_video_path():
    root = Tk()
    root.withdraw()
    video_path = filedialog.askopenfilename(
        title="Select Video File",
        filetypes=[("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")],
    )
    return video_path


def cleanup_resources():
    """Ensure all resources are properly released without killing the main process."""
    # Close OpenCV windows but don't destroy all windows globally
    try:
        cap = cv2.VideoCapture(0)  # Dummy capture to reset OpenCV state
        cap.release()
    except Exception:
        pass

    # Close pygame display but don't fully quit pygame
    if pygame.get_init():
        pygame.display.quit()

    # Don't create a new Tkinter root window
    # This was causing problems by creating new instances

    # Don't force garbage collection - this can cause lockups
    # Let Python handle memory cleanup naturally


def run_cutvideo():
    # Print the directory and name of the script being executed
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting cutvideo.py...")

    # Platform-specific adjustments
    import platform

    if platform.system() == "Linux":
        try:
            # Check if we're on Linux and if NVIDIA drivers are present
            has_nvidia = os.path.exists("/proc/driver/nvidia")
            if has_nvidia:
                print("NVIDIA GPU detected, applying OpenGL compatibility settings")

            # Set OpenGL to software rendering as a fallback for Mesa/OpenGL issues on Linux
            os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
            os.environ["SDL_VIDEODRIVER"] = "x11"
        except Exception:
            print("If you experience graphics issues, try running: export LIBGL_ALWAYS_SOFTWARE=1")

    video_path = get_video_path()
    if not video_path:
        print("No video selected. Exiting.")
        return

    try:
        play_video_with_cuts(video_path)
    except Exception as e:
        print(f"Error in cutvideo: {e}")

        # More helpful error message for Linux users
        if platform.system() == "Linux":
            print("\nPossible Linux graphics driver issue detected.")
            print("Try running these commands before starting the application:")
            print("export LIBGL_ALWAYS_SOFTWARE=1")
            print("export SDL_VIDEODRIVER=x11")
    finally:
        # Clean up resources more gently
        cleanup_resources()
        print("Video cutting process completed")


if __name__ == "__main__":
    run_cutvideo()
