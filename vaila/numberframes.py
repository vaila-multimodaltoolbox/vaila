"""
numberframes.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 13 August 2025
Version: 0.1.2
Python Version: 3.12.11

Description:
------------
This script allows users to analyze video files within a selected directory and extract metadata such as frame count, frame rate (FPS), resolution, codec, and duration. The script generates a summary of this information, displays it in a user-friendly graphical interface, and saves the metadata to text files. The "basic" file contains essential metadata, while the "full" file includes all possible metadata extracted using `ffprobe`.

Key Features:
-------------
1. Fast metadata extraction using a single ffprobe JSON call.
2. Detection of capture FPS via Android tag com.android.capture.fps when present.
3. Parallel processing of multiple videos for faster analysis.

Notes:
------
- The script uses the `ffprobe` command-line tool to extract metadata.
- The script uses the `ThreadPoolExecutor` to process multiple videos in parallel.
- The script uses the `as_completed` function to process the videos in parallel.
- The script uses the `tkinter` library to create a user-friendly graphical interface.


Usage:
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
from fractions import Fraction
from tkinter import filedialog, messagebox, ttk

from rich import print


def _ffprobe_json(video_path: str) -> dict:
    """
    Single, fast ffprobe call returning JSON with format + all streams and related sections.
    Avoids -count_frames and -show_frames for speed.
    """
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_format",
        "-show_streams",
        "-show_chapters",
        "-show_programs",
        "-show_stream_groups",
        video_path,
    ]
    try:
        proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
        out = proc.stdout.strip() or "{}"
        return json.loads(out)
    except Exception as exc:  # noqa: BLE001
        print(f"Warning: ffprobe JSON failed for {video_path}: {exc}")
        return {}


def _to_float_fps(fr_str: str | None) -> float | None:
    if not fr_str or fr_str == "0/0":
        return None
    try:
        return float(Fraction(fr_str))
    except Exception:
        try:
            return float(fr_str)
        except Exception:
            return None


def _extract_capture_fps(meta: dict) -> float | None:
    """Search known Android slow-motion capture FPS tags across format and streams."""
    keys = [
        "com.android.capture.fps",
        "com.android.capturer.fps",
        "com.android.slowMotion.capture.fps",
    ]
    # Check format tags first (as in user's JSON)
    fmt_tags = (meta.get("format", {}) or {}).get("tags", {}) or {}
    for k in keys:
        val = fmt_tags.get(k)
        if val is not None:
            try:
                return float(val)
            except Exception:
                pass
    # Then check any stream tags (video stream usually index 0)
    for st in meta.get("streams", []) or []:
        tags = st.get("tags", {}) or {}
        for k in keys:
            val = tags.get(k)
            if val is not None:
                try:
                    return float(val)
                except Exception:
                    pass
    return None


def get_video_info(video_path):
    """
    Fast metadata extractor using a single ffprobe JSON call.
    Also detects capture FPS via Android tag com.android.capture.fps when present.
    """
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    try:
        meta = _ffprobe_json(video_path)
        fmt = meta.get("format", {}) or {}
        streams = meta.get("streams", []) or []
        v0 = streams[0] if streams else {}

        width = v0.get("width")
        height = v0.get("height")
        r_frame_rate = _to_float_fps(v0.get("r_frame_rate"))
        avg_frame_rate = _to_float_fps(v0.get("avg_frame_rate"))
        duration = float(fmt.get("duration")) if fmt.get("duration") else None

        # Codec and container info
        codec_name = v0.get("codec_name")
        codec_long_name = v0.get("codec_long_name")
        container_format = fmt.get("format_name")
        container_long_name = fmt.get("format_long_name")

        # container-reported frame count (fast). If absent, estimate by avg_fps * duration
        nb_frames = None
        try:
            if "nb_frames" in v0 and v0["nb_frames"] not in (None, "N/A", ""):
                nb_frames = int(v0["nb_frames"])
        except Exception:
            nb_frames = None

        if nb_frames is None and duration and (avg_frame_rate or r_frame_rate):
            fps_for_estimation = avg_frame_rate or r_frame_rate
            if fps_for_estimation:
                nb_frames = int(round(fps_for_estimation * duration))

        # Real capture rate for slow-motion, if present (scan format and streams)
        capture_fps = _extract_capture_fps(meta)

        display_fps = r_frame_rate
        avg_fps = avg_frame_rate

        recommended_sampling_hz = None
        if capture_fps:
            recommended_sampling_hz = capture_fps
        elif duration and nb_frames:
            recommended_sampling_hz = nb_frames / duration
        else:
            recommended_sampling_hz = display_fps

        # Format FPS values for printing (handle None)
        disp_str = f"{display_fps:.9f}" if display_fps else "N/A"
        avg_str = f"{avg_fps:.9f}" if avg_fps else "N/A"
        cap_str = f"{capture_fps:.9f}" if capture_fps else "N/A"
        dur_str = f"{duration:.9f}" if duration else "N/A"

        print(
            f"Video info: {width}x{height}, codec={codec_name}, container={container_format}, "
            f"display≈{disp_str} fps, avg≈{avg_str} fps, cap={cap_str} Hz, "
            f"dur={dur_str}s, frames={nb_frames}"
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
            "_raw_json": meta,
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
