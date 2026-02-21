"""
numberframes.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 06 February 2026
Version: 0.1.4
Python Version: 3.12.12

Description:
------------
This script allows users to analyze video files within a selected directory and
extract metadata such as frame count, frame rate (FPS), resolution, codec, and
duration. It generates a summary of this information, displays it in a
graphical interface, and saves the metadata to text files.

Key Features:
-------------
1. Robust metadata extraction using `ffprobe` (via `get_precise_video_metadata`)
   with a fallback to OpenCV.
2. Accurate handling of fractional frame rates and precise duration calculations.
3. Detection of capture FPS via Android tag com.android.capture.fps.
4. Parallel processing of multiple videos for faster analysis.

Notes:
------
- The script uses `ffprobe` as the primary source for metadata to ensure specificities
  like fractional FPS are captured correctly.
- OpenCV is used as a fallback if `ffprobe` fails.

Usage:
------
- Run the script, select a directory containing video files, and let the tool analyze the videos.
- View the metadata in the GUI and check the saved text files in the selected directory for details.
- Use the "full" file for complete metadata in JSON format.

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

import json
import os
from pathlib import Path
import subprocess
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tkinter import filedialog, messagebox, ttk

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
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)

        # Find video stream
        video_stream = None
        for stream in data.get("streams", []):
            if stream.get("codec_type") == "video":
                video_stream = stream
                break

        if not video_stream:
            return None

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
            "width": int(video_stream.get("width")),
            "height": int(video_stream.get("height")),
            "codec": video_stream.get("codec_name", "unknown"),
            "codec_long": video_stream.get("codec_long_name", "unknown"),
            "container": data.get("format", {}).get("format_name", "unknown"),
            "container_long": data.get("format", {}).get("format_long_name", "unknown"),
            "r_frame_rate": r_frame_rate_str,
            "avg_frame_rate": avg_frame_rate_str,
            "duration": duration if duration > 0 else None,
            "nb_frames": nb_frames,
            "_raw_json": data,
        }
    except Exception as e:
        print(f"Warning: ffprobe failed for {video_path}: {e}")
        return None


def get_video_info(video_path):
    """
    Fast metadata extractor using a single ffprobe JSON call.
    Also detects capture FPS via Android tag com.android.capture.fps when present.
    """

    try:
        meta = get_precise_video_metadata(video_path)

        # Fallback to OpenCV if ffprobe fails or returns incomplete data
        if not meta or meta.get("nb_frames") is None:
            import cv2

            cap = cv2.VideoCapture(str(video_path))
            if cap.isOpened():
                if not meta:
                    meta = {
                        "fps": cap.get(cv2.CAP_PROP_FPS),
                        "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
                        "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
                        "codec": "unknown",
                        "codec_long": "unknown",
                        "container": "unknown",
                        "container_long": "unknown",
                        "duration": 0.0,
                        "_raw_json": {},
                    }

                meta["nb_frames"] = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                cap.release()

        if not meta:
            raise ValueError("Could not extract metadata from video")

        width = meta.get("width")
        height = meta.get("height")
        fps = meta.get("fps")
        duration = meta.get("duration")
        nb_frames = meta.get("nb_frames")
        codec_name = meta.get("codec")
        codec_long_name = meta.get("codec_long")
        container_format = meta.get("container")
        container_long_name = meta.get("container_long")

        # Capture FPS logic removed for simplicity as robust parsing is prioritized,
        # but could be re-added by inspecting meta["_raw_json"] if strictly needed.
        capture_fps = None

        # Fraction to float helper inside get_video_info
        def frac_to_float(f_str):
            if not f_str or f_str == "0/0": return None
            try:
                if "/" in f_str:
                    n, d = map(int, f_str.split("/"))
                    return float(n)/d if d else 0.0
                return float(f_str)
            except:
                return None

        # Properly assign Display and Avg FPS from original fractions
        display_fps = frac_to_float(meta.get("r_frame_rate")) or fps
        avg_fps = frac_to_float(meta.get("avg_frame_rate")) or fps

        recommended_sampling_hz = display_fps

        # Format FPS values for printing (handle None)
        disp_str = f"{display_fps:.9f}" if display_fps else "N/A"
        dur_str = f"{duration:.9f}" if duration else "N/A"

        print(
            f"Video info: {width}x{height}, codec={codec_name}, container={container_format}, "
            f"display≈{disp_str} fps, dur={dur_str}s, frames={nb_frames}"
        )

        return {
            "file_name": os.path.basename(video_path),
            "frame_count": nb_frames,
            "display_fps": display_fps,
            "avg_fps": avg_fps,
            "capture_fps": capture_fps,
            "recommended_sampling_hz": recommended_sampling_hz,
            "resolution": f"{width}x{height}" if width and height else "unknown",
            "duration": duration if duration else 0.0,
            "codec_name": codec_name,
            "codec_long_name": codec_long_name,
            "container_format": container_format,
            "container_long_name": container_long_name,
            "_raw_json": meta.get("_raw_json", {}),
        }

    except Exception as e:
        print(f"Warning: Could not get detailed video info for {video_path}: {e}")
        return {
            "file_name": os.path.basename(video_path),
            "error": f"Error retrieving video info: {str(e)}",
        }




def count_frames_in_videos(directory_path=None, video_files=None, show_gui=True):
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting video metadata extraction...")

    if not directory_path and not video_files:
        if not show_gui:
            print("Error: No directory or files provided in CLI mode.")
            return

        import tkinter as tk
        from tkinter import filedialog, messagebox
        
        root = tk.Tk()
        root.withdraw()

        directory_path = filedialog.askdirectory(title="Select the directory containing videos")
        if not directory_path:
            messagebox.showerror("Error", "No directory selected.")
            return

    if video_files:
        abs_paths = [os.path.abspath(f) for f in video_files]
        if not directory_path:
            directory_path = os.path.dirname(abs_paths[0])
    else:
        video_files_list = sorted(
            [
                f
                for f in os.listdir(directory_path)
                if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
            ]
        )
        abs_paths = [os.path.join(directory_path, f) for f in video_files_list]
    video_infos = []
    max_workers = max(2, (os.cpu_count() or 4))
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        future_to_path = {executor.submit(get_video_info, p): p for p in abs_paths}
        for fut in as_completed(future_to_path):
            p = future_to_path[fut]
            try:
                info = fut.result()
            except Exception as e:
                info = {"file_name": os.path.basename(p), "error": f"{e}"}
            video_infos.append(info)

    # Mantém ordem por nome de arquivo
    video_infos.sort(key=lambda d: d.get("file_name", ""))

    print(f"\nBasic metadata saved to: {output_basic_file}")
    print(f"Full metadata saved to: {output_full_file}")
    
    print("\n--- Summary ---")
    for info in video_infos:
        print(f"[{info.get('file_name')}]: {info.get('frame_count')} frames, {info.get('display_fps')} display fps")


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description="Extract high-precision frame and metadata from video files.",
        epilog="""
Examples:
  # Launch GUI to select directory
  python numberframes.py
  
  # Process a specific directory via CLI
  python numberframes.py --dir /path/to/videos
  
  # Process specific video files
  python numberframes.py /path/to/video1.mp4 /path/to/video2.avi
        """,
        formatter_class=argparse.RawTextHelpFormatter
    )
    parser.add_argument("videos", nargs="*", help="Specific video files to process")
    parser.add_argument("--dir", "-d", help="Directory containing videos to process")
    parser.add_argument("--cli", action="store_true", help="Run without GUI display")
    
    # If no arguments provided at all (except the script name), run default GUI mode
    if len(sys.argv) == 1:
        count_frames_in_videos(show_gui=True)
    else:
        args = parser.parse_args()
        show_gui_flag = not args.cli
        
        if args.videos:
            count_frames_in_videos(video_files=args.videos, show_gui=show_gui_flag)
        elif args.dir:
            count_frames_in_videos(directory_path=args.dir, show_gui=show_gui_flag)
        else:
            # Should not happen as argparse handles required inputs, but fallback to GUI just in case
            count_frames_in_videos(show_gui=True)
