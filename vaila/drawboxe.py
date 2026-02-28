"""
Project: vailá Multimodal Toolbox
Script: drawboxe.py - Draw boxes on videos

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 28 October 2024
Update Date: 11 January 2026
Version: 0.0.10
Description:
    Draw boxes on videos.
    This script is a modified version of the original drawboxe.py script.
    Draw trapezoid and free polygon boxes.

Usage:
    Run the script from the command line:
        python drawboxe.py

Requirements:
    - Python 3.x
    - OpenCV
    - Tkinter (for GUI operations)

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.9: Audio and metadata preservation - preserves original video audio and all metadata (compatible with numberframes.py and other metadata tools)
    - v0.0.8: Frame-accurate preservation - ensures exact frame count and precise FPS are maintained for biomechanical data synchronization (similar to cutvideo.py)
    - v0.0.7: Added hatching to indicate outside mode
    - v0.0.6: Added support for free polygon boxes
    - v0.0.5: Added support for trapezoid boxes
    - v0.0.4: Added support for rectangle boxes
    - v0.0.3: Added support for frame intervals
    - v0.0.2: Added support for multiple videos
    - v0.0.1: First version
"""

import contextlib
import datetime
import json
import os
import shutil
import subprocess
import time
import tkinter as tk
from tkinter import filedialog, messagebox

import cv2
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import toml  # type: ignore[import-untyped]


def get_precise_video_metadata(video_path):
    """
    Get precise video metadata using ffprobe to avoid rounding errors.
    Returns dict with fps (float), width, height, codec, nb_frames, etc.
    Similar to cutvideo.py for consistency in frame preservation.
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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
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
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            cap.release()
            return {
                "fps": fps,
                "width": width,
                "height": height,
                "codec": "unknown",
                "r_frame_rate": None,
                "avg_frame_rate": None,
                "nb_frames": total_frames,
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
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return {
            "fps": fps,
            "width": width,
            "height": height,
            "codec": "unknown",
            "r_frame_rate": None,
            "avg_frame_rate": None,
            "nb_frames": total_frames,
        }


def save_first_frame(video_path, frame_path):
    vidcap = cv2.VideoCapture(video_path)
    success, image = vidcap.read()
    if success:
        cv2.imwrite(frame_path, image)
    vidcap.release()


def extract_frames(video_path, frames_dir):
    """
    Extract all frames from video using ffmpeg.
    Ensures all frames are extracted without skipping.
    """
    try:
        os.makedirs(frames_dir, exist_ok=True)
        video_path = os.path.normpath(os.path.abspath(video_path))
        frames_dir = os.path.normpath(os.path.abspath(frames_dir))

        print(f"  Extracting frames from {os.path.basename(video_path)} (ffmpeg)...")
        # Use ffmpeg with frame-accurate extraction
        # -vsync 0 ensures no frame dropping or duplication
        # -start_number 1 ensures frame numbering starts at 1
        command = [
            "ffmpeg",
            "-i",
            video_path,
            "-vsync",
            "0",  # Don't drop or duplicate frames
            "-start_number",
            "1",  # Start numbering at 1
            os.path.join(frames_dir, "frame_%09d.png"),
        ]

        if os.name == "nt":
            subprocess.run(command, check=True, capture_output=False, text=True, shell=True)
        else:
            subprocess.run(command, check=True, capture_output=False, text=True)

        # Count extracted frames
        frame_files = [
            f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")
        ]
        print(f"Extracted {len(frame_files)} frames from {os.path.basename(video_path)}")

    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", None) or str(e)
        print(f"Error running ffmpeg: {err}")
        raise
    except Exception as e:
        print(f"Error extracting frames: {str(e)}")
        raise


def apply_boxes_directly_to_video(input_path, output_path, coordinates, selections, colors):
    """
    Apply boxes directly to video using OpenCV for processing, then ffmpeg to preserve
    audio and metadata. Ensures all frames are preserved with original audio and metadata.
    """
    import tempfile

    # Get precise metadata first
    metadata = get_precise_video_metadata(input_path)
    width = metadata["width"]
    height = metadata["height"]
    fps = metadata["fps"]
    total_frames = metadata.get("nb_frames")

    vidcap = cv2.VideoCapture(input_path)
    if not vidcap.isOpened():
        print(f"Error: Could not open video {input_path}")
        return False

    # Verify dimensions match
    actual_width = int(vidcap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(vidcap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if actual_width != width or actual_height != height:
        print(
            f"Warning: Metadata dimensions ({width}x{height}) don't match video ({actual_width}x{actual_height})"
        )
        width, height = actual_width, actual_height

    # Get actual frame count if metadata didn't provide it
    if total_frames is None:
        total_frames = int(vidcap.get(cv2.CAP_PROP_FRAME_COUNT))

    print(f"Applying boxes to {total_frames} frames: {os.path.basename(input_path)} -> {os.path.basename(output_path)}")
    # Create temporary video file for processed frames (without audio)
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    # Use float FPS to preserve precision
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(temp_video_path, fourcc, fps, (width, height))

    if not out.isOpened():
        print(f"Error: Could not create temporary video {temp_video_path}")
        vidcap.release()
        return False

    frame_count = 0
    frames_written = 0

    # Process ALL frames, ensuring none are skipped
    while frame_count < total_frames:
        # Explicitly set frame position to ensure we don't skip frames
        vidcap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
        ret, frame = vidcap.read()

        if not ret:
            # If we can't read this frame, try to continue but warn
            print(f"\nWarning: Could not read frame {frame_count + 1}, skipping...")
            frame_count += 1
            continue

        # Apply boxes to frame
        for coords, selection, color in zip(coordinates, selections, colors, strict=False):
            mode = selection[0]
            shape_type = selection[1]
            # Converter cor de matplotlib (0-1) para OpenCV (0-255) BGR
            bgr_color = (int(color[2] * 255), int(color[1] * 255), int(color[0] * 255))

            if shape_type == "rectangle":
                x1, y1 = int(coords[0][0]), int(coords[0][1])
                x2, y2 = int(coords[2][0]), int(coords[2][1])
                x1, x2 = min(x1, x2), max(x1, x2)
                y1, y2 = min(y1, y2), max(y1, y2)
                if mode == "inside":
                    frame[y1:y2, x1:x2] = bgr_color
                else:
                    # For outside mode, fill everything except the rectangle
                    mask = np.ones(frame.shape[:2], dtype=np.uint8)
                    cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
                    frame[mask == 1] = bgr_color
            elif shape_type in ("trapezoid", "free"):
                pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                if mode == "inside":
                    cv2.fillPoly(frame, [pts], bgr_color)
                else:
                    # For outside mode, fill everything except the polygon
                    mask = np.ones(frame.shape[:2], dtype=np.uint8)
                    cv2.fillPoly(mask, [pts], 0)
                    frame[mask == 1] = bgr_color

        out.write(frame)
        frames_written += 1
        frame_count += 1

        step = 10 if total_frames > 50 else 5
        if frame_count % step == 0 or frame_count == total_frames:
            print(
                f"  Processed {frame_count}/{total_frames} frames for {os.path.basename(input_path)}",
                end="\r",
            )

    print(f"\n  Completed processing frames: {frames_written}/{total_frames}")

    out.release()
    vidcap.release()

    # Verify frame count preservation
    if frames_written != total_frames:
        print(f"Warning: Frame count mismatch! Expected {total_frames}, wrote {frames_written}")
        with contextlib.suppress(BaseException):
            os.remove(temp_video_path)
        return False

    # Now use ffmpeg to combine processed video with original audio and preserve metadata
    try:
        # Check if ffmpeg is available
        subprocess.run(["ffmpeg", "-version"], capture_output=True, check=True)

        fps_str = f"{fps:.6f}"

        # Check if original video has audio stream
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
            str(input_path),
        ]
        probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
        has_audio = probe_result.returncode == 0 and probe_result.stdout.strip() == "audio"

        # Build ffmpeg command to combine video with audio and preserve metadata
        cmd = [
            "ffmpeg",
            "-y",  # Overwrite output
            "-i",
            temp_video_path,  # Processed video (no audio)
            "-i",
            str(input_path),  # Original video (for audio and metadata)
            "-c:v",
            "libx264",  # Video codec
            "-preset",
            "medium",  # Encoding preset
            "-crf",
            "18",  # High quality
            "-r",
            fps_str,  # Preserve exact FPS
            "-map",
            "0:v",  # Use video from processed file
            "-map_metadata",
            "1",  # Copy all metadata from original video
        ]

        # Add audio mapping if available
        if has_audio:
            cmd.extend(["-map", "1:a?", "-c:a", "copy"])  # Copy audio codec (no re-encoding)
        else:
            print("Info: Original video has no audio stream, video will be saved without audio")

        cmd.extend(
            [
                "-pix_fmt",
                "yuv420p",  # Ensure compatibility
                "-avoid_negative_ts",
                "make_zero",
                str(output_path),
            ]
        )

        print(f"  Combining video with audio and metadata (ffmpeg)...")
        subprocess.run(cmd, capture_output=False, text=True, check=True)

        print(f"  Done: {os.path.basename(output_path)}")

        # Clean up temporary file
        with contextlib.suppress(BaseException):
            os.remove(temp_video_path)

        return True

    except (subprocess.CalledProcessError, FileNotFoundError) as e:
        # If ffmpeg fails or is not available, fallback to just using the temp video
        print(f"Warning: ffmpeg not available or failed, saving without audio/metadata: {e}")
        try:
            import shutil

            shutil.move(temp_video_path, output_path)
            print(f"Saved video without audio/metadata: {os.path.basename(output_path)}")
            return True
        except Exception as e2:
            print(f"Error moving temp file: {e2}")
            with contextlib.suppress(BaseException):
                os.remove(temp_video_path)
            return False


def apply_boxes_to_frames(frames_dir, coordinates, selections, colors, frame_intervals):
    for filename in sorted(os.listdir(frames_dir)):
        frame_number = int(filename.split("_")[1].split(".")[0])
        for start_frame, end_frame in frame_intervals:
            if start_frame <= frame_number <= end_frame:
                frame_path = os.path.join(frames_dir, filename)
                img = cv2.imread(frame_path)

                for coords, selection, color in zip(coordinates, selections, colors, strict=False):
                    mode = selection[0]
                    shape_type = selection[1]
                    # Converter cor de matplotlib (0-1) para OpenCV (0-255) BGR
                    bgr_color = (
                        int(color[2] * 255),
                        int(color[1] * 255),
                        int(color[0] * 255),
                    )

                    if shape_type == "rectangle":
                        x1, y1 = int(coords[0][0]), int(coords[0][1])
                        x2, y2 = int(coords[2][0]), int(coords[2][1])
                        x1, x2 = min(x1, x2), max(x1, x2)
                        y1, y2 = min(y1, y2), max(y1, y2)
                        if mode == "inside":
                            img[y1:y2, x1:x2] = bgr_color
                        else:
                            # For outside mode, fill everything except the rectangle
                            mask = np.ones(img.shape[:2], dtype=np.uint8)
                            cv2.rectangle(mask, (x1, y1), (x2, y2), 0, -1)
                            img[mask == 1] = bgr_color
                    elif shape_type in ("trapezoid", "free"):
                        pts = np.array(coords, np.int32).reshape((-1, 1, 2))
                        if mode == "inside":
                            cv2.fillPoly(img, [pts], bgr_color)
                        else:
                            # For outside mode, fill everything except the polygon
                            mask = np.ones(img.shape[:2], dtype=np.uint8)
                            cv2.fillPoly(mask, [pts], 0)
                            img[mask == 1] = bgr_color

                cv2.imwrite(frame_path, img)


def reassemble_video(frames_dir, output_path, fps, total_frames=None, original_video_path=None):
    """
    Reassemble video from frames using ffmpeg with precise FPS preservation.
    If original_video_path is provided, copies audio and metadata from original video.
    Ensures all frames are included and FPS is preserved exactly.
    """
    import tempfile

    # Count actual frames in directory
    frame_files = sorted(
        [f for f in os.listdir(frames_dir) if f.startswith("frame_") and f.endswith(".png")]
    )
    actual_frame_count = len(frame_files)

    if total_frames is not None and actual_frame_count != total_frames:
        print(f"Warning: Expected {total_frames} frames, found {actual_frame_count} in directory")

    # Use precise FPS (6 decimal places) for scientific accuracy
    fps_str = f"{fps:.6f}"

    # Create temporary video file (without audio) first
    temp_video = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
    temp_video_path = temp_video.name
    temp_video.close()

    # Build ffmpeg command to create video from frames (no audio)
    command = [
        "ffmpeg",
        "-y",  # Overwrite output file
        "-framerate",
        fps_str,  # Input framerate (precise)
        "-i",
        os.path.join(frames_dir, "frame_%09d.png"),
        "-c:v",
        "libx264",  # Video codec
        "-preset",
        "medium",  # Encoding preset
        "-crf",
        "18",  # High quality
        "-r",
        fps_str,  # Output framerate (precise, must match input)
        "-pix_fmt",
        "yuv420p",  # Ensure compatibility
        "-avoid_negative_ts",
        "make_zero",  # Avoid timestamp issues
    ]

    # If we know the exact frame count, specify it
    if total_frames is not None:
        command.extend(["-frames:v", str(total_frames)])
    elif actual_frame_count > 0:
        # Use actual frame count as fallback
        command.extend(["-frames:v", str(actual_frame_count)])

    command.append(str(temp_video_path))

    try:
        print(f"  Creating video from {actual_frame_count} frames at {fps_str} fps (ffmpeg)...")
        subprocess.run(command, check=True, capture_output=False, text=True)
        print(f"  Created video from frames: {actual_frame_count} frames")
    except subprocess.CalledProcessError as e:
        err = getattr(e, "stderr", None) or str(e)
        print(f"Error creating video from frames: {err}")
        with contextlib.suppress(BaseException):
            os.remove(temp_video_path)
        raise

    # If original video path is provided, combine with audio and metadata
    if original_video_path and os.path.exists(original_video_path):
        try:
            # Check if original video has audio stream
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
                str(original_video_path),
            ]
            probe_result = subprocess.run(probe_cmd, capture_output=True, text=True)
            has_audio = probe_result.returncode == 0 and probe_result.stdout.strip() == "audio"

            # Build command to combine video with audio and metadata
            combine_cmd = [
                "ffmpeg",
                "-y",
                "-i",
                temp_video_path,  # Processed video (no audio)
                "-i",
                str(original_video_path),  # Original video (for audio and metadata)
                "-c:v",
                "libx264",  # Video codec
                "-preset",
                "medium",
                "-crf",
                "18",
                "-r",
                fps_str,  # Preserve exact FPS
                "-map",
                "0:v",  # Use video from processed file
                "-map_metadata",
                "1",  # Copy all metadata from original
            ]

            # Add audio mapping if available (use ? to make it optional)
            if has_audio:
                combine_cmd.extend(["-map", "1:a?", "-c:a", "copy"])  # Copy audio codec
            else:
                print("Info: Original video has no audio stream, video will be saved without audio")

            combine_cmd.extend(
                [
                    "-pix_fmt",
                    "yuv420p",
                    "-avoid_negative_ts",
                    "make_zero",
                    str(output_path),
                ]
            )

            print(f"  Combining with audio and metadata (ffmpeg)...")
            subprocess.run(combine_cmd, check=True, capture_output=False, text=True)

            print(f"  Done: {os.path.basename(output_path)}")

            # Clean up temporary file
            with contextlib.suppress(BaseException):
                os.remove(temp_video_path)

            return True

        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            print(f"Warning: Could not combine with audio/metadata, using video only: {e}")
            # Fallback: just use the temp video
            try:
                import shutil

                shutil.move(temp_video_path, output_path)
                print(f"Saved video without audio/metadata: {os.path.basename(output_path)}")
                return True
            except Exception as e2:
                print(f"Error moving temp file: {e2}")
                with contextlib.suppress(BaseException):
                    os.remove(temp_video_path)
                return False
    else:
        # No original video provided, just use the temp video
        try:
            import shutil

            shutil.move(temp_video_path, output_path)
            print(f"Saved video: {os.path.basename(output_path)}")
            return True
        except Exception as e:
            print(f"Error moving temp file: {e}")
            with contextlib.suppress(BaseException):
                os.remove(temp_video_path)
            return False


def clean_up(directory):
    for filename in os.listdir(directory):
        file_path = os.path.join(directory, filename)
        os.remove(file_path)
    os.rmdir(directory)


def save_config_toml(video_path, coordinates, selections, colors, frame_intervals=None):
    """Save box configuration to TOML file"""
    basename = os.path.splitext(os.path.basename(video_path))[0]
    config_path = os.path.join(os.path.dirname(video_path), f"{basename}_dbox.toml")

    # Escape the video path for TOML (replace backslashes with forward slashes)
    escaped_video_path = video_path.replace("\\", "/")

    # Create TOML content with comments
    toml_content = f"""# ================================================================
# DrawBoxe Configuration File
# Generated automatically by drawboxe.py in vaila Multimodal Analysis Toolbox
# Created: {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}
# ================================================================
#
# HOW TO USE THIS FILE:
# 1. This file contains polygon/box definitions for video processing
# 2. Load this configuration to apply the same boxes to multiple videos
# 3. Use the 'Load Config' button or 'q' key to load this file
# 4. The configuration will be applied to the current video
#
# IMPORTANT: Keep the format exactly as shown!
# - All coordinates are stored as x,y pairs
# - Colors are stored as RGB values (0-1 range)
# - Modes: "inside" (fills selected area) or "outside" (fills everything except selected area)
# - Shapes: "rectangle", "trapezoid", or "free"
# - Frame intervals (optional): define which frames to process (start_frame, end_frame)
# ================================================================

[description]
title = "DrawBoxe Configuration File"
purpose = "This file contains polygon/box definitions for video processing"
usage = "Load this configuration to apply the same boxes to multiple videos"
author = "Prof. Dr. Paulo R. P. Santiago"
github = "https://github.com/paulopreto/vaila-multimodaltoolbox"
version = "0.0.7"
date = "{datetime.datetime.now().strftime("%Y-%m-%d")}"

[video_info]
basename = "{basename}"
original_path = "{escaped_video_path}"

"""

    # Add frame intervals section (always include commented example)
    toml_content += """
# ================================================================
# FRAME INTERVALS (OPTIONAL)
# ================================================================
# Define which frame ranges to process. If not specified, all frames will be processed.
# Format: start_frame, end_frame (inclusive, 0-indexed)
#
# Example: To process frames 100-200 and 500-600, uncomment and modify:
# [[frame_intervals]]
# id = 1
# start_frame = 100
# end_frame = 200
#
# [[frame_intervals]]
# id = 2
# start_frame = 500
# end_frame = 600
# ================================================================
"""

    # Add actual frame intervals if provided
    if frame_intervals:
        for i, (start, end) in enumerate(frame_intervals):
            toml_content += "[[frame_intervals]]\n"
            toml_content += f"id = {i + 1}\n"
            toml_content += f"start_frame = {start}\n"
            toml_content += f"end_frame = {end}\n\n"

    for i, (coords, selection, color) in enumerate(
        zip(coordinates, selections, colors, strict=False)
    ):
        toml_content += f"""
# ================================================================
# BOX {i + 1}: {selection[1].upper()} - {selection[0].upper()} MODE
# ================================================================
[[boxes]]
id = {i + 1}
mode = "{selection[0]}"
shape = "{selection[1]}"
color_r = {color[0]:.6f}
color_g = {color[1]:.6f}
color_b = {color[2]:.6f}
"""
        for j, coord in enumerate(coords):
            toml_content += f"coord_{j + 1}_x = {coord[0]:.6f}\n"
            toml_content += f"coord_{j + 1}_y = {coord[1]:.6f}\n"

    with open(config_path, "w") as f:
        f.write(toml_content)

    print(f"Configuration saved to: {config_path}")
    return config_path


def load_config_toml(config_path):
    """Load box configuration from TOML file"""
    try:
        with open(config_path) as f:
            config_data = toml.load(f)

        coordinates = []
        selections = []
        colors = []
        frame_intervals = []

        # Load frame intervals if present
        for interval in config_data.get("frame_intervals", []):
            start = interval.get("start_frame")
            end = interval.get("end_frame")
            if start is not None and end is not None:
                frame_intervals.append((int(start), int(end)))

        # Load boxes
        for box in config_data.get("boxes", []):
            # Convert coordinates
            coords = []
            i = 1
            while f"coord_{i}_x" in box and f"coord_{i}_y" in box:
                x = box[f"coord_{i}_x"]
                y = box[f"coord_{i}_y"]
                coords.append((x, y))
                i += 1

            if coords:
                coordinates.append(coords)

                # Convert selection
                color = (box["color_r"], box["color_g"], box["color_b"])
                selections.append((box["mode"], box["shape"], color))
                colors.append(color)

        # Return None for frame_intervals if empty (for backward compatibility)
        return coordinates, selections, colors, (frame_intervals if frame_intervals else None)
    except Exception as e:
        print(f"Error loading config: {e}")
        return [], [], [], None


def get_box_coordinates(image_path, video_path=None):
    img = plt.imread(image_path)
    fig = plt.figure(figsize=(14, 9))

    # Main image axis
    ax = plt.axes([0.15, 0.1, 0.65, 0.8])  # [left, bottom, width, height]

    selection_mode = {
        "mode": "inside",
        "shape": "rectangle",
        "color": (0, 0, 0),
    }  # Default black

    # Add color palette on the side
    from matplotlib.patches import Rectangle as ColorRect
    from matplotlib.widgets import Button, RadioButtons, Slider

    # Left panel for mode and shape selection
    ax_mode = plt.axes([0.02, 0.7, 0.1, 0.15])
    radio_mode = RadioButtons(ax_mode, ("Inside", "Outside"), active=0)
    ax_mode.set_title("Mode", fontsize=10)

    ax_shape = plt.axes([0.02, 0.45, 0.1, 0.2])
    radio_shape = RadioButtons(ax_shape, ("Rectangle", "Trapezoid", "Free"), active=0)
    ax_shape.set_title("Shape", fontsize=10)

    # Instructions on the left
    # Polygon tracking
    polygon_counter = {"current": 0, "total": 0}

    # Status display
    ax_status = plt.axes([0.02, 0.12, 0.1, 0.03])
    ax_status.axis("off")
    status_text = ax_status.text(0, 0, "Polygon: 0/0", fontsize=9, color="blue")

    ax_inst = plt.axes([0.02, 0.15, 0.1, 0.25])
    ax_inst.axis("off")
    instructions = (
        "Keys:\n"
        "• e: toggle mode\n"
        "• t: toggle shape\n"
        "• c: pick color\n"
        "• z: finish polygon\n"
        "• d: done polygon\n"
        "• r: clear all\n"
        "• w: save config\n"
        "• q: load config\n"
        "• a: abort script\n"
        "• Enter: save & exit\n"
        "• Right-click: undo"
    )
    ax_inst.text(0, 1, instructions, fontsize=8, verticalalignment="top")

    # Right panel for color controls
    # Create sliders for RGB values
    ax_r = plt.axes([0.85, 0.6, 0.02, 0.25])
    ax_g = plt.axes([0.89, 0.6, 0.02, 0.25])
    ax_b = plt.axes([0.93, 0.6, 0.02, 0.25])

    slider_r = Slider(ax_r, "R", 0, 255, valinit=0, orientation="vertical", valstep=1)
    slider_g = Slider(ax_g, "G", 0, 255, valinit=0, orientation="vertical", valstep=1)
    slider_b = Slider(ax_b, "B", 0, 255, valinit=0, orientation="vertical", valstep=1)

    # Color preview box
    ax_color = plt.axes([0.85, 0.45, 0.1, 0.1])
    ax_color.set_xlim(0, 1)
    ax_color.set_ylim(0, 1)
    ax_color.axis("off")
    color_preview = ColorRect((0, 0), 1, 1, color=(0, 0, 0))
    ax_color.add_patch(color_preview)
    ax_color.text(
        0.5,
        -0.2,
        "Color Preview",
        ha="center",
        fontsize=10,
        transform=ax_color.transAxes,
    )

    # Button to pick color from image
    ax_pick = plt.axes([0.85, 0.35, 0.1, 0.04])
    btn_pick = Button(ax_pick, "Pick Color")

    pick_mode = {"active": False}
    pick_rect = {"start": None, "rect": None}

    def validate_color(color):
        """Ensures that the color is in the 0-1 range"""
        if any(c > 1.0 for c in color):
            return tuple(c / 255.0 if c > 1.0 else c for c in color)
        return color

    def update_mode(label):
        selection_mode["mode"] = label.lower()
        update_title()

    def update_shape(label):
        selection_mode["shape"] = label.lower()
        update_title()

    radio_mode.on_clicked(update_mode)
    radio_shape.on_clicked(update_shape)

    def update_color(val=None):
        r = slider_r.val / 255
        g = slider_g.val / 255
        b = slider_b.val / 255
        selection_mode["color"] = (r, g, b)
        color_preview.set_color((r, g, b))
        fig.canvas.draw_idle()

    slider_r.on_changed(update_color)
    slider_g.on_changed(update_color)
    slider_b.on_changed(update_color)

    def update_title():
        current_id = polygon_counter["current"] + 1
        ax.set_title(f"Drawing Shape #{current_id} - Press Enter when done", fontsize=12)
        fig.canvas.draw_idle()

    def update_status():
        """Updates the polygon status display"""
        polygon_counter["total"] = len(shapes)
        current_shape = selection_mode["shape"].capitalize()
        current_mode = selection_mode["mode"].capitalize()
        status_text.set_text(
            f"ID: {polygon_counter['current']}/{polygon_counter['total']} | {current_shape} | {current_mode}"
        )
        fig.canvas.draw_idle()

    ax.imshow(img)
    update_title()
    points = []
    shapes = []
    selections = []
    temp_points = []
    free_polygon_sizes = []
    free_lines = []
    temp_shapes = []  # For preview

    # Additional controls
    ax_clear = plt.axes([0.02, 0.02, 0.08, 0.04])
    btn_clear = Button(ax_clear, "Clear All")

    ax_done = plt.axes([0.02, 0.07, 0.08, 0.04])
    btn_done = Button(ax_done, "Done Polygon")

    ax_save = plt.axes([0.85, 0.25, 0.08, 0.04])
    btn_save = Button(ax_save, "Save Config")

    ax_load = plt.axes([0.85, 0.2, 0.08, 0.04])
    btn_load = Button(ax_load, "Load Config")

    ax_help = plt.axes([0.85, 0.15, 0.08, 0.04])
    btn_help = Button(ax_help, "Help")

    ax_abort = plt.axes([0.85, 0.1, 0.08, 0.04])
    btn_abort = Button(ax_abort, "Abort")

    def on_pick_color(event):
        pick_mode["active"] = True
        pick_rect["start"] = None
        pick_rect["rect"] = None
        ax.set_title("Color Picker Mode: Click and drag to select area")
        fig.canvas.draw()

    def on_clear_all(event):
        nonlocal points, shapes, selections, temp_points, free_polygon_sizes
        # Clear all shapes from screen
        for shape in shapes:
            shape.remove()
        clear_temp_shapes()

        # Reset all lists
        points.clear()
        shapes.clear()
        selections.clear()
        temp_points.clear()
        free_polygon_sizes.clear()
        polygon_counter["current"] = 0
        polygon_counter["total"] = 0

        # Reset selection color
        pick_mode["active"] = False
        pick_rect["start"] = None
        if pick_rect["rect"]:
            pick_rect["rect"].remove()
            pick_rect["rect"] = None

        ax.set_title("All shapes cleared - Drawing boxes on image - Press Enter when done")
        update_status()
        fig.canvas.draw()

    def on_done_polygon(event):
        nonlocal temp_points, free_polygon_sizes
        # Finalize current polygon if in progress
        if selection_mode["shape"] == "free" and len(temp_points) >= 3:
            # Close the polygon by connecting the last point to the first
            if temp_points[0] != temp_points[-1]:
                temp_points.append(temp_points[0])

            color = validate_color(selection_mode["color"])
            if selection_mode["mode"] == "inside":
                poly = patches.Polygon(
                    temp_points,
                    linewidth=3,
                    edgecolor=color,
                    facecolor=color,
                    alpha=0.6,
                )
            else:
                poly = patches.Polygon(
                    temp_points,
                    linewidth=3,
                    edgecolor=color,
                    facecolor="none",
                    alpha=1.0,
                )
                # Add overlay for outside mode
                overlay = patches.Polygon(
                    temp_points,
                    linewidth=0,
                    facecolor=color,
                    alpha=0.2,
                    hatch="///",  # Add hatching to indicate outside mode
                )
                ax.add_patch(overlay)
                temp_shapes.append(overlay)
            ax.add_patch(poly)
            shapes.append(poly)
            points.extend(temp_points)
            selections.append((selection_mode["mode"], "free", color))
            free_polygon_sizes.append(len(temp_points))
            polygon_counter["current"] = len(shapes)
            clear_temp_shapes()
            temp_points.clear()
            update_status()
            plt.draw()

    def on_save_config(event):
        if video_path and shapes:
            try:
                current_coords = []
                current_selections = []
                current_colors = []

                # Process current shapes
                i = 0
                free_index = 0
                while i < len(points) and len(current_selections) < len(selections):
                    if len(selections) > len(current_selections):
                        current_selection = selections[len(current_selections)]
                        if current_selection[1] == "free":
                            if free_index < len(free_polygon_sizes):
                                n_points = free_polygon_sizes[free_index]
                                # For free polygons, remove the duplicate point from closing
                                box_points = [
                                    (points[j][0], points[j][1]) for j in range(i, i + n_points - 1)
                                ]
                                current_coords.append(box_points)
                                current_selections.append(current_selection)
                                current_colors.append(current_selection[2])
                                i += n_points
                                free_index += 1
                        else:
                            if i + 4 <= len(points):
                                box_points = [(points[j][0], points[j][1]) for j in range(i, i + 4)]
                                current_coords.append(box_points)
                                current_selections.append(current_selection)
                                current_colors.append(current_selection[2])
                                i += 4
                    else:
                        break

                config_path = save_config_toml(
                    video_path, current_coords, current_selections, current_colors
                )
                ax.set_title(f"Configuration saved! - {os.path.basename(config_path)}")
                fig.canvas.draw_idle()
            except Exception as e:
                ax.set_title(f"Error saving config: {str(e)}")
                fig.canvas.draw_idle()
        else:
            ax.set_title("No video path or shapes to save!")
            fig.canvas.draw_idle()

    def show_help(event):
        """Shows help window with detailed instructions"""
        help_text = """
DRAWBOXE - HELP

 HOW TO USE:
1. Select mode (Inside/Outside) and shape (Rectangle/Trapezoid/Free)
2. Choose a color using RGB sliders or "Pick Color" button
3. Left-click on the image to add points
4. Use keys or buttons to finalize shapes

 KEYS:
• e: Toggle mode (Inside/Outside)
• t: Toggle shape (Rectangle/Trapezoid/Free)
• c: Activate color picker
• z: Finish free polygon (close current polygon)
• d: Complete current polygon (move to next)
• r: Clear all shapes
• w: Save configuration
• q: Load configuration
• a: Abort script (exit without processing)
• Enter: Save and exit
• Right-click: Undo last point/shape

 MOUSE:
• Left-click: Add point (definition)
• Right-click: Remove last point/shape
• Drag with "Pick Color": Select area for color

 MODES:
• Inside: Fills the selected area
• Outside: Fills everything except the selected area

🔷 SHAPES:
• Rectangle: 2 clicks (opposite corners)
• Trapezoid: 4 clicks (trapezoid points)
• Free: Multiple clicks + 'z' key to close

 FILES:
• Configuration saved as: [video]_dbox.toml
• Processed video: [video]_dbox.mp4

[WARNING] IMPORTANT:
• Clicks outside the image are ignored
• Free polygons need at least 3 points
• Use "Help" button for more information
        """

        # Create help window
        help_window = tk.Toplevel()
        help_window.title("DrawBoxe - Help")
        help_window.geometry("600x700")
        help_window.resizable(True, True)

        # Create text widget
        text_widget = tk.Text(help_window, wrap=tk.WORD, padx=10, pady=10)
        text_widget.pack(fill=tk.BOTH, expand=True)

        # Insert help text
        text_widget.insert(tk.END, help_text)
        text_widget.config(state=tk.DISABLED)  # Make it read-only

        # Close button
        close_btn = tk.Button(help_window, text="Close", command=help_window.destroy)
        close_btn.pack(pady=10)

        # Center the window
        help_window.transient()  # Modal window
        help_window.grab_set()  # Capture events
        help_window.focus_set()  # Focus on the window

    def on_abort(event):
        """Abort the script without processing anything"""
        import sys

        print("Script aborted by user")
        plt.close()
        sys.exit(0)

    def on_load_config(event):
        # Open file dialog to select TOML configuration file
        # Create a root window for the dialog
        dialog_root = tk.Tk()
        dialog_root.withdraw()  # Hide the root window
        dialog_root.attributes("-topmost", True)  # Keep dialog on top

        config_path = filedialog.askopenfilename(
            parent=dialog_root,
            title="Select TOML configuration file",
            filetypes=[("TOML files", "*.toml"), ("All files", "*.*")],
            initialdir=os.path.dirname(video_path) if video_path else os.getcwd(),
        )

        dialog_root.destroy()  # Clean up the dialog root

        if config_path and os.path.exists(config_path):
            try:
                loaded_coords, loaded_selections, loaded_colors, _ = load_config_toml(config_path)
                if loaded_coords:
                    # Clear current shapes
                    on_clear_all(None)

                    # Load saved shapes
                    for coords, selection, color in zip(
                        loaded_coords, loaded_selections, loaded_colors, strict=False
                    ):
                        if selection[1] == "rectangle":
                            x1, y1 = coords[0]
                            x2, y2 = coords[2]
                            rect = patches.Rectangle(
                                (min(x1, x2), min(y1, y2)),
                                abs(x2 - x1),
                                abs(y2 - y1),
                                linewidth=3,
                                edgecolor=color,
                                facecolor=color if selection[0] == "inside" else "none",
                                alpha=0.6 if selection[0] == "inside" else 1.0,
                            )
                            ax.add_patch(rect)
                            shapes.append(rect)
                            points.extend(coords)
                            selections.append(selection)
                            polygon_counter["current"] = len(shapes)

                        elif selection[1] in ["trapezoid", "free"]:
                            # For polygons, add first point at end to close
                            if coords[0] != coords[-1]:
                                closed_coords = coords + [coords[0]]
                            else:
                                closed_coords = coords

                            poly = patches.Polygon(
                                closed_coords,
                                linewidth=3,
                                edgecolor=color,
                                facecolor=color if selection[0] == "inside" else "none",
                                alpha=0.6 if selection[0] == "inside" else 1.0,
                            )
                            ax.add_patch(poly)
                            shapes.append(poly)
                            points.extend(closed_coords)
                            selections.append(selection)
                            polygon_counter["current"] = len(shapes)
                            if selection[1] == "free":
                                free_polygon_sizes.append(len(closed_coords))

                    ax.set_title(f"Configuration loaded! - {os.path.basename(config_path)}")
                    update_status()
                    fig.canvas.draw()
                else:
                    ax.set_title("No shapes found in config file!")
                    fig.canvas.draw_idle()
            except Exception as e:
                ax.set_title(f"Error loading config: {str(e)}")
                fig.canvas.draw_idle()
        elif config_path:
            ax.set_title("Selected file does not exist!")
            fig.canvas.draw_idle()

    btn_pick.on_clicked(on_pick_color)
    btn_clear.on_clicked(on_clear_all)
    btn_done.on_clicked(on_done_polygon)
    btn_save.on_clicked(on_save_config)
    btn_load.on_clicked(on_load_config)
    btn_help.on_clicked(show_help)
    btn_abort.on_clicked(on_abort)

    def clear_temp_shapes():
        for shape in temp_shapes:
            shape.remove()
        temp_shapes.clear()
        for line in free_lines:
            line.remove()
        free_lines.clear()

    def on_key(event):
        nonlocal shapes, points, selections, temp_points, free_polygon_sizes

        if event.key == "e":
            # Toggle mode and update radio button
            new_mode = "outside" if selection_mode["mode"] == "inside" else "inside"
            selection_mode["mode"] = new_mode
            radio_mode.set_active(0 if new_mode == "inside" else 1)
            update_title()
        elif event.key == "t":
            # Toggle shape and update radio button
            shape_list = ["rectangle", "trapezoid", "free"]
            current_idx = shape_list.index(selection_mode["shape"])
            new_idx = (current_idx + 1) % 3
            selection_mode["shape"] = shape_list[new_idx]
            radio_shape.set_active(new_idx)
            temp_points.clear()
            clear_temp_shapes()
            update_title()
        elif event.key == "c":
            # Activate color picker mode
            pick_mode["active"] = True
            ax.set_title("Color Picker Mode: Click and drag to select area")
            fig.canvas.draw()
        elif event.key == "r":
            # Clear all
            on_clear_all(None)
        elif event.key == "d":
            # Done polygon - finalize current polygon and move to next
            on_done_polygon(None)
        elif event.key == "w":
            # Save config
            on_save_config(None)
        elif event.key == "q":
            # Load config
            on_load_config(None)
        elif event.key == "a":
            # Abort script
            on_abort(None)
        elif event.key == "z":
            # Finish polygon - close current free polygon
            if selection_mode["shape"] == "free" and len(temp_points) >= 3:
                # Close polygon by connecting last point to first
                if temp_points[0] != temp_points[-1]:
                    temp_points.append(temp_points[0])

                color = validate_color(selection_mode["color"])
                if selection_mode["mode"] == "inside":
                    poly = patches.Polygon(
                        temp_points,
                        linewidth=3,
                        edgecolor=color,
                        facecolor=color,
                        alpha=0.6,
                    )
                else:
                    # For outside mode, show the polygon border and indicate exterior fill
                    poly = patches.Polygon(
                        temp_points,
                        linewidth=3,
                        edgecolor=color,
                        facecolor="none",
                        alpha=1.0,
                    )
                    # Add a semi-transparent overlay to indicate outside mode
                    overlay = patches.Polygon(
                        temp_points,
                        linewidth=0,
                        facecolor=color,
                        alpha=0.2,
                        hatch="///",  # Add hatching to indicate outside mode
                    )
                    ax.add_patch(overlay)
                    temp_shapes.append(overlay)
                ax.add_patch(poly)
                shapes.append(poly)
                points.extend(temp_points)
                selections.append((selection_mode["mode"], "free", color))
                free_polygon_sizes.append(len(temp_points))
                polygon_counter["current"] = len(shapes)
                clear_temp_shapes()
                temp_points.clear()
                update_status()
                plt.draw()
        elif event.key == "enter":
            plt.close()

    def on_click(event):
        nonlocal points, shapes, selections, temp_points

        # Check if click is inside the main image axes
        if event.inaxes != ax:
            return

        # Validate if click is inside the image
        if event.xdata is None or event.ydata is None:
            return

        # Ensure coordinates are valid
        x, y = event.xdata, event.ydata
        if x < 0 or y < 0:
            return

        # Color picker mode
        if pick_mode["active"]:
            if event.button == 1:
                pick_rect["start"] = (event.xdata, event.ydata)
            return

        if event.button == 3:  # Right button - remove last shape or temporary points
            if temp_points:
                # If there are temporary points, remove the last point
                temp_points.pop()
                clear_temp_shapes()
                # Redraw remaining points
                color = validate_color(selection_mode["color"])
                for i, point in enumerate(temp_points):
                    pt = ax.plot(point[0], point[1], "o", color=color, markersize=6)[0]
                    temp_shapes.append(pt)
                    if i > 0:
                        line = ax.plot(
                            [temp_points[i - 1][0], point[0]],
                            [temp_points[i - 1][1], point[1]],
                            color=color,
                            linewidth=2,
                        )[0]
                        free_lines.append(line)
                plt.draw()
            elif shapes:
                # If no temporary points, remove the last complete shape
                shapes[-1].remove()
                shapes.pop()
                if selections:
                    last_selection = selections.pop()
                    if last_selection[1] == "free":
                        # Remove points from last free polygon
                        if free_polygon_sizes:
                            num_points = free_polygon_sizes.pop()
                            points = points[:-num_points] if len(points) >= num_points else []
                    else:
                        # Remove points from rectangle/trapezoid (4 points)
                        points = points[:-4] if len(points) >= 4 else []

                # Update counter
                polygon_counter["current"] = len(shapes)
                update_status()
                plt.draw()
            return

        if event.button == 1:  # Left button for definition
            if selection_mode["shape"] == "rectangle":
                temp_points.append((x, y))
                if len(temp_points) == 1:
                    # Show initial point
                    color = validate_color(selection_mode["color"])
                    point = ax.plot(x, y, "o", color=color, markersize=8)[0]
                    temp_shapes.append(point)
                elif len(temp_points) == 2:
                    x1, y1 = temp_points[0]
                    x2, y2 = temp_points[1]
                    rect_points = [
                        (x1, y1),
                        (x2, y1),
                        (x2, y2),
                        (x1, y2),
                    ]
                    points.extend(rect_points)
                    color = validate_color(selection_mode["color"])
                    if selection_mode["mode"] == "inside":
                        rect = patches.Rectangle(
                            (min(x1, x2), min(y1, y2)),
                            abs(x2 - x1),
                            abs(y2 - y1),
                            linewidth=3,
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.6,
                        )
                    else:
                        rect = patches.Rectangle(
                            (min(x1, x2), min(y1, y2)),
                            abs(x2 - x1),
                            abs(y2 - y1),
                            linewidth=3,
                            edgecolor=color,
                            facecolor="none",
                            alpha=1.0,
                        )
                        # Add overlay for outside mode
                        overlay = patches.Rectangle(
                            (min(x1, x2), min(y1, y2)),
                            abs(x2 - x1),
                            abs(y2 - y1),
                            linewidth=0,
                            facecolor=color,
                            alpha=0.2,
                            hatch="///",  # Add hatching to indicate outside mode
                        )
                        ax.add_patch(overlay)
                        temp_shapes.append(overlay)
                    ax.add_patch(rect)
                    shapes.append(rect)
                    selections.append(
                        (
                            selection_mode["mode"],
                            "rectangle",
                            validate_color(selection_mode["color"]),
                        )
                    )
                    polygon_counter["current"] = len(shapes)
                    temp_points.clear()
                    clear_temp_shapes()
                    update_status()
                    plt.draw()
            elif selection_mode["shape"] == "trapezoid":
                temp_points.append((x, y))
                # Show points and temporary lines
                color = validate_color(selection_mode["color"])
                point = ax.plot(x, y, "o", color=color, markersize=8)[0]
                temp_shapes.append(point)
                if len(temp_points) > 1:
                    line = ax.plot(
                        [temp_points[-2][0], temp_points[-1][0]],
                        [temp_points[-2][1], temp_points[-1][1]],
                        color=color,
                        linewidth=2,
                        alpha=0.5,
                    )[0]
                    temp_shapes.append(line)
                if len(temp_points) == 4:
                    # Close the trapezoid
                    line = ax.plot(
                        [temp_points[-1][0], temp_points[0][0]],
                        [temp_points[-1][1], temp_points[0][1]],
                        color=color,
                        linewidth=2,
                        alpha=0.5,
                    )[0]
                    temp_shapes.append(line)
                    plt.draw()
                    # Create final shape
                    if selection_mode["mode"] == "inside":
                        trap = patches.Polygon(
                            temp_points,
                            linewidth=3,
                            edgecolor=color,
                            facecolor=color,
                            alpha=0.6,
                        )
                    else:
                        trap = patches.Polygon(
                            temp_points,
                            linewidth=3,
                            edgecolor=color,
                            facecolor="none",
                            alpha=1.0,
                        )
                        # Add overlay for outside mode
                        overlay = patches.Polygon(
                            temp_points,
                            linewidth=0,
                            facecolor=color,
                            alpha=0.2,
                            hatch="///",  # Add hatching to indicate outside mode
                        )
                        ax.add_patch(overlay)
                        temp_shapes.append(overlay)
                    ax.add_patch(trap)
                    shapes.append(trap)
                    points.extend(temp_points)
                    selections.append((selection_mode["mode"], "trapezoid", color))
                    polygon_counter["current"] = len(shapes)
                    temp_points.clear()
                    clear_temp_shapes()
                    update_status()
                    plt.draw()
            elif selection_mode["shape"] == "free":
                temp_points.append((x, y))
                # Show point
                color = validate_color(selection_mode["color"])
                point = ax.plot(x, y, "o", color=color, markersize=6)[0]
                temp_shapes.append(point)
                # Draw line between points for visual feedback
                if len(temp_points) > 1:
                    line = ax.plot(
                        [temp_points[-2][0], temp_points[-1][0]],
                        [temp_points[-2][1], temp_points[-1][1]],
                        color=color,
                        linewidth=2,
                    )[0]
                    free_lines.append(line)
                    plt.draw()

    def on_motion(event):
        # Preview rectangle
        if (
            not pick_mode["active"]
            and len(temp_points) == 1
            and selection_mode["shape"] == "rectangle"
        ):
            # Clear previous preview
            for shape in [s for s in temp_shapes if isinstance(s, patches.Rectangle)]:
                shape.remove()
                temp_shapes.remove(shape)

            if event.xdata and event.ydata:
                x1, y1 = temp_points[0]
                x2, y2 = event.xdata, event.ydata
                color = validate_color(selection_mode["color"])
                preview_rect = patches.Rectangle(
                    (min(x1, x2), min(y1, y2)),
                    abs(x2 - x1),
                    abs(y2 - y1),
                    linewidth=1,
                    edgecolor=color,
                    facecolor=color if selection_mode["mode"] == "inside" else "none",
                    alpha=0.3 if selection_mode["mode"] == "inside" else 1.0,
                    linestyle="--",
                )
                ax.add_patch(preview_rect)
                temp_shapes.append(preview_rect)
                plt.draw()

        if pick_mode["active"] and pick_rect["start"] and event.button == 1:
            # Clear previous rectangle if it exists
            if pick_rect["rect"]:
                pick_rect["rect"].remove()

            x1, y1 = pick_rect["start"]
            x2, y2 = event.xdata, event.ydata

            if x1 and y1 and x2 and y2:
                rect = patches.Rectangle(
                    (min(x1, x2), min(y1, y2)),
                    abs(x2 - x1),
                    abs(y2 - y1),
                    linewidth=2,
                    edgecolor="white",
                    facecolor="none",
                    linestyle="--",
                )
                ax.add_patch(rect)
                pick_rect["rect"] = rect
                plt.draw()

    def on_release(event):
        if pick_mode["active"] and pick_rect["start"]:
            x1, y1 = pick_rect["start"]
            x2, y2 = event.xdata, event.ydata

            if x1 and y1 and x2 and y2:
                # Calculate average color of the selected area
                x_min, x_max = int(min(x1, x2)), int(max(x1, x2))
                y_min, y_max = int(min(y1, y2)), int(max(y1, y2))

                # Ensure we are within image bounds
                h, w = img.shape[:2]
                x_min = max(0, x_min)
                x_max = min(w, x_max)
                y_min = max(0, y_min)
                y_max = min(h, y_max)

                # Extract region and calculate average color
                region = img[y_min:y_max, x_min:x_max]
                if region.size > 0:
                    avg_color = np.mean(region.reshape(-1, region.shape[-1]), axis=0)

                    # Ensure colors are in the 0-1 range
                    if avg_color.max() > 1.0:
                        avg_color = avg_color / 255.0

                    if len(avg_color) >= 3:
                        # RGB or RGBA
                        selection_mode["color"] = (
                            float(avg_color[0]),
                            float(avg_color[1]),
                            float(avg_color[2]),
                        )
                        slider_r.set_val(int(avg_color[0] * 255))
                        slider_g.set_val(int(avg_color[1] * 255))
                        slider_b.set_val(int(avg_color[2] * 255))

                        # Update color preview
                        color_preview.set_color(selection_mode["color"])
                        update_status()

            # Clear color picker mode
            if pick_rect["rect"]:
                pick_rect["rect"].remove()
            pick_rect["start"] = None
            pick_rect["rect"] = None
            pick_mode["active"] = False
            update_title()
            plt.draw()

    fig.canvas.mpl_connect("button_press_event", on_click)
    fig.canvas.mpl_connect("button_release_event", on_release)
    fig.canvas.mpl_connect("motion_notify_event", on_motion)
    fig.canvas.mpl_connect("key_press_event", on_key)
    plt.show()
    if temp_points:
        raise ValueError("An incomplete shape was defined.")

    # Process boxes correctly
    boxes = []
    colors = []
    i = 0
    free_index = 0

    while i < len(points):
        if selections and len(selections) > len(boxes):
            current_selection = selections[len(boxes)]
            if current_selection[1] == "free":
                # Use the correct free polygon size
                if free_index < len(free_polygon_sizes):
                    n_points = free_polygon_sizes[free_index]
                    box_points = [
                        (int(points[j][0]), int(points[j][1])) for j in range(i, i + n_points)
                    ]
                    boxes.append(box_points)
                    color = current_selection[2] if len(current_selection) > 2 else (0, 0, 0)
                    colors.append(validate_color(color))
                    i += n_points
                    free_index += 1
                else:
                    break
            else:
                # For rectangle and trapezoid, use 4 points
                if i + 4 <= len(points):
                    box_points = [(int(points[j][0]), int(points[j][1])) for j in range(i, i + 4)]
                    boxes.append(box_points)
                    color = current_selection[2] if len(current_selection) > 2 else (0, 0, 0)
                    colors.append(validate_color(color))
                    i += 4
                else:
                    break
        else:
            break

    # Automatically save config if there are shapes
    if boxes and video_path:
        try:
            save_config_toml(video_path, boxes, selections, colors)
        except Exception as e:
            print(f"Warning: Could not save config automatically: {e}")

    return boxes, selections, colors


def load_frame_intervals(file_path):
    intervals = []
    with open(file_path) as file:
        for line in file:
            start, end = map(int, line.strip().split(","))
            intervals.append((start, end))
    return intervals


def show_feedback_message():
    print("vailá!")
    time.sleep(2)


def run_drawboxe():
    print("Running DrawBoxe...")
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("--------------------------------")
    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Keep dialog on top

    initial_dir = os.path.expanduser("~") if os.name == "nt" or os.name == "posix" else os.getcwd()
    video_directory = filedialog.askdirectory(
        parent=root,
        title="Select the directory containing videos",
        initialdir=initial_dir,
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return
    video_directory = os.path.normpath(os.path.abspath(video_directory))
    if not os.path.exists(video_directory):
        messagebox.showerror("Error", f"Directory does not exist: {video_directory}")
        return
    try:
        video_files = sorted(
            [
                f
                for f in os.listdir(video_directory)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
        )
    except PermissionError:
        messagebox.showerror("Error", f"Permission denied to access directory: {video_directory}")
        return
    except Exception as e:
        messagebox.showerror("Error", f"Error accessing directory: {str(e)}")
        return
    if not video_files:
        messagebox.showerror("Error", "No video files found in the selected directory.")
        return
    first_video = video_files[0]
    first_frame_path = os.path.join(video_directory, "first_frame.jpg")
    save_first_frame(os.path.join(video_directory, first_video), first_frame_path)
    coordinates, selections, colors = get_box_coordinates(
        first_frame_path, os.path.join(video_directory, first_video)
    )
    os.remove(first_frame_path)

    # Check if there's a TOML config file with frame intervals
    frame_intervals = None
    basename = os.path.splitext(first_video)[0]
    config_path = os.path.join(video_directory, f"{basename}_dbox.toml")

    if os.path.exists(config_path):
        try:
            _, _, _, loaded_intervals = load_config_toml(config_path)
            if loaded_intervals:
                use_toml_intervals = messagebox.askyesno(
                    "Frame Intervals Found",
                    f"Frame intervals found in TOML config file.\n\n"
                    f"Intervals: {loaded_intervals}\n\n"
                    f"Do you want to use these intervals?",
                )
                if use_toml_intervals:
                    frame_intervals = loaded_intervals
        except Exception as e:
            print(f"Warning: Could not load frame intervals from TOML: {e}")

    # If no intervals from TOML, ask user
    if frame_intervals is None:
        use_intervals = messagebox.askyesno(
            "Frame Intervals", "Do you want to use frame intervals from a .txt file?"
        )
        if use_intervals:
            intervals_file = filedialog.askopenfilename(
                parent=root,
                title="Select the .txt file with frame intervals",
                filetypes=[("Text files", "*.txt")],
            )
            if intervals_file:
                frame_intervals = load_frame_intervals(intervals_file)
            else:
                messagebox.showerror("Error", "No .txt file selected.")
                return
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(video_directory, f"video_2_drawbox_{timestamp}")
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    n_videos = len(video_files)
    for idx, video_file in enumerate(video_files):
        input_path = os.path.join(video_directory, video_file)
        final_output_path = os.path.join(output_dir, f"{os.path.splitext(video_file)[0]}_dbox.mp4")

        # Get precise video metadata to preserve frame count and FPS
        print(f"\n--- Video {idx + 1}/{n_videos}: {video_file} ---")
        metadata = get_precise_video_metadata(input_path)
        fps = metadata["fps"]
        total_frames = metadata.get("nb_frames")

        if frame_intervals:
            frames_dir = os.path.join(video_directory, "frames_temp")
            if os.path.exists(frames_dir):
                shutil.rmtree(frames_dir)
            extract_frames(input_path, frames_dir)
            print(f"  Applying boxes to extracted frames...")
            apply_boxes_to_frames(frames_dir, coordinates, selections, colors, frame_intervals)
            # Pass precise FPS, total frame count, and original video path to preserve audio and metadata
            reassemble_video(
                frames_dir, final_output_path, fps, total_frames, original_video_path=input_path
            )
            clean_up(frames_dir)
            print(f"  Done: {os.path.basename(final_output_path)}")
        else:
            # apply_boxes_directly_to_video now uses precise metadata internally
            success = apply_boxes_directly_to_video(
                input_path, final_output_path, coordinates, selections, colors
            )
            if not success:
                print(f"  Warning: Processing may have issues for {video_file}")
    show_feedback_message()
    print("All videos processed and saved to the output directory.")
    messagebox.showinfo("Completed", "All videos have been processed successfully!")


if __name__ == "__main__":
    run_drawboxe()
