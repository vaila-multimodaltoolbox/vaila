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
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

        display_fps = fps
        avg_fps = fps

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


def display_video_info(video_infos, output_file):
    def on_closing():
        root.destroy()
        show_save_success_message(output_file)

    root = tk.Tk()
    root.title("Video Information")
    root.protocol("WM_DELETE_WINDOW", on_closing)

    frame = ttk.Frame(root, padding="10")
    frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

    canvas = tk.Canvas(frame, width=1080, height=720)
    scrollbar = ttk.Scrollbar(frame, orient="vertical", command=canvas.yview)
    scrollable_frame = ttk.Frame(canvas)

    scrollable_frame.bind(
        "<Configure>", lambda e: canvas.configure(scrollregion=canvas.bbox("all"))
    )

    canvas.create_window((0, 0), window=scrollable_frame, anchor="nw")
    canvas.configure(yscrollcommand=scrollbar.set)

    canvas.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
    scrollbar.grid(row=0, column=1, sticky=(tk.N, tk.S))

    # Adding headers (exibe Display e Capture FPS lado a lado)
    headers = [
        "Video File",
        "Frames",
        "Display FPS",
        "Capture FPS",
        "Codec",
        "Container",
        "Resolution",
        "Duration (s)",
    ]
    for i, header in enumerate(headers):
        ttk.Label(scrollable_frame, text=header, font=("Arial", 10, "bold")).grid(
            row=0, column=i, padx=10, pady=5, sticky=tk.W
        )

    for i, info in enumerate(video_infos, start=1):
        ttk.Label(scrollable_frame, text=info["file_name"], font=("Arial", 10, "bold")).grid(
            row=i, column=0, sticky=tk.W, pady=5
        )
        if "error" in info:
            ttk.Label(scrollable_frame, text=info["error"]).grid(
                row=i, column=1, columnspan=4, sticky=tk.W, padx=10
            )
        else:
            ttk.Label(scrollable_frame, text=info["frame_count"]).grid(
                row=i, column=1, sticky=tk.W, padx=10
            )
            # Display with high precision in GUI (6 decimal places for readability)
            ttk.Label(scrollable_frame, text=f"{(info.get('display_fps') or 0):.6f}").grid(
                row=i, column=2, sticky=tk.W, padx=10
            )
            ttk.Label(
                scrollable_frame,
                text=(f"{info.get('capture_fps'):.6f}" if info.get("capture_fps") else "N/A"),
            ).grid(row=i, column=3, sticky=tk.W, padx=10)
            ttk.Label(scrollable_frame, text=(info.get("codec_name") or "N/A")).grid(
                row=i, column=4, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=(info.get("container_format") or "N/A")).grid(
                row=i, column=5, sticky=tk.W, padx=10
            )
            ttk.Label(scrollable_frame, text=info["resolution"]).grid(
                row=i, column=6, sticky=tk.W, padx=10
            )
            # Display duration with high precision (6 decimal places for readability)
            ttk.Label(scrollable_frame, text=f"{info['duration']:.6f}").grid(
                row=i, column=7, sticky=tk.W, padx=10
            )

    root.mainloop()


def save_basic_metadata_to_file(video_infos, directory_path):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = os.path.join(directory_path, f"video_metadata_basic_{timestamp}.txt")

    with open(output_file, "w", encoding="utf-8") as f:
        for info in video_infos:
            if "error" in info:
                f.write(f"File: {info['file_name']}\nError: {info['error']}\n\n")
            else:
                f.write(f"File: {info['file_name']}\n")
                f.write(f"Frames: {info['frame_count']}\n")
                # Mostrar ambos para evitar ambiguidade
                disp = info.get("display_fps")
                avg = info.get("avg_fps")
                cap = info.get("capture_fps")
                rec = info.get("recommended_sampling_hz")
                # Use high precision for scientific accuracy (9 decimal places)
                f.write(f"Display_FPS: {disp:.9f}\n" if disp else "Display_FPS: N/A\n")
                f.write(f"Avg_FPS: {avg:.9f}\n" if avg else "Avg_FPS: N/A\n")
                f.write(f"Capture_FPS: {cap:.9f}\n" if cap else "Capture_FPS: N/A\n")
                f.write(
                    f"Recommended_Sampling_Hz: {rec:.9f}\n"
                    if rec
                    else "Recommended_Sampling_Hz: N/A\n"
                )
                if cap and disp:
                    try:
                        slowmo = cap / disp
                        f.write(f"SlowMo_Factor: {slowmo:.9f}\n")
                    except Exception:
                        pass
                # Codec/Container
                codec = info.get("codec_name") or "N/A"
                codec_long = info.get("codec_long_name") or ""
                container = info.get("container_format") or "N/A"
                container_long = info.get("container_long_name") or ""
                f.write(f"Codec: {codec}{(' - ' + codec_long) if codec_long else ''}\n")
                f.write(
                    f"Container: {container}{(' - ' + container_long) if container_long else ''}\n"
                )
                f.write(f"Resolution: {info['resolution']}\n")
                # Use high precision for duration (9 decimal places for scientific accuracy)
                f.write(f"Duration (s): {info['duration']:.9f}\n\n")

    return output_file


def save_full_metadata_to_file(directory_path, video_infos):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(directory_path, f"metadata_full_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    for info in video_infos:
        json_file = os.path.join(output_dir, f"{os.path.splitext(info['file_name'])[0]}.json")
        try:
            with open(json_file, "w", encoding="utf-8") as f:
                json.dump(info.get("_raw_json", {}), f, ensure_ascii=False, indent=2)
            print(f"Full metadata for {info['file_name']} saved to {json_file}")
        except Exception as e:
            print(f"Error saving full metadata for {info['file_name']}: {str(e)}")

    return output_dir


def show_save_success_message(output_file):
    root = tk.Tk()
    root.withdraw()
    messagebox.showinfo(
        "Save Success", f"Metadata successfully saved!\n\nOutput file: {output_file}"
    )


def count_frames_in_videos():
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting video frame counting...")

    root = tk.Tk()
    root.withdraw()

    directory_path = filedialog.askdirectory(title="Select the directory containing videos")
    if not directory_path:
        messagebox.showerror("Error", "No directory selected.")
        return

    video_files = sorted(
        [
            f
            for f in os.listdir(directory_path)
            if f.lower().endswith((".mp4", ".avi", ".mov", ".mkv"))
        ]
    )
    # Processamento em paralelo para acelerar
    abs_paths = [os.path.join(directory_path, f) for f in video_files]
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

    output_basic_file = save_basic_metadata_to_file(video_infos, directory_path)
    output_full_file = save_full_metadata_to_file(directory_path, video_infos)

    print(f"Basic metadata saved to: {output_basic_file}")
    print(f"Full metadata saved to: {output_full_file}")
    display_video_info(video_infos, output_basic_file)


if __name__ == "__main__":
    count_frames_in_videos()
