"""
Project: vailá Multimodal Toolbox
Script: cutvideo.py - Cut Video

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 15 July 2026
Version: 0.3.83

Description:
This script performs batch processing of videos for cutting videos.
verview:
This script performs batch processing of videos for cutting videos.
It allows users to visually mark cut points in videos, save cut information, and generate precisely cut video segments while preserving original metadata with high accuracy.
It supports scientific precision for research applications.
It uses ffmpeg to preserve exact frame rates (e.g., 59.94005994005994 fps) without rounding.
It uses hardware H.264 when available (NVIDIA NVENC on Linux/Windows, Apple
VideoToolbox on macOS) and falls back to CPU libx264 on all platforms.
It uses OpenCV to fallback to less precise but always available.
It uses pygame to create a graphical interface for selecting the input directory containing video files (.mp4, .avi, .mov), the output directory, and for specifying the cuts.
It uses tomllib to load the cuts from a TOML file.
It uses ffmpeg to cut the videos.

Features:
- Added support for TOML files.
- Added support for audio waveform visualization.
- Added support for audio playback.
- Added support for loop control.
- Playback speed: `[` / `]` halve/double speed (0.0625×–16×), same as getpixelvideo.
- Added support for auto-fit window.
- Added support for marker navigation.
- Added clickable timeline feedback for cut ranges, start/end markers, and pending start.
- Added Shift+Left/Right navigation across cut start/end timeline markers.
- Added responsive, cancellable progress dialogs while final video cuts render.
- Added support for manual FPS input.
- Added support for help dialog.
- Added support for save and generate videos.
- Added support for batch processing of videos.
- Optional custom output base name for cut files (GUI button or B key).
- Optional per-cut output names from a CSV/TXT list (GUI **Cut names** button or **N** / **V**):
  one name per line → cut 1 → name1.mp4, cut 2 → name2.mp4, …
- Hardware H.264 encode when available (NVENC / VideoToolbox; auto CPU libx264 fallback).

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov, .mkv), the output directory, and for
  specifying the cuts.

Requirements:
- Python 3.12.13
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

import bisect
import contextlib
import datetime
import json
import os
import platform

# Configure SDL environment variables BEFORE importing pygame
# to prevent EGL/OpenGL warnings and window manager crashes on Linux systems
if platform.system() == "Linux":
    os.environ["SDL_VIDEODRIVER"] = "x11"
    os.environ["SDL_RENDER_DRIVER"] = "software"
    os.environ["LIBGL_ALWAYS_SOFTWARE"] = "1"
    os.environ["SDL_VIDEO_X11_FORCE_EGL"] = "0"
    os.environ["QT_LOGGING_RULES"] = "*.debug=false;qt.qpa.*=false"

import subprocess
import tempfile
import threading
import time
import tomllib
import wave
from collections.abc import Callable
from pathlib import Path

import cv2
import numpy as np
import pygame
from rich import print
from rich.progress import Progress, SpinnerColumn, TextColumn, TimeElapsedColumn

try:
    from .ffmpeg_utils import (
        describe_video_encoder,
        encoder_device_tag,
        encoders_with_cpu_fallback,
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_ffprobe_path,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )
except ImportError:
    from ffmpeg_utils import (
        describe_video_encoder,
        encoder_device_tag,
        encoders_with_cpu_fallback,
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_ffprobe_path,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )

MAX_RENDER_PIXELS = 4_000_000
CUT_RANGE_COLOR = (42, 86, 112)
CUT_START_COLOR = (70, 220, 110)
CUT_END_COLOR = (255, 135, 70)
CUT_PENDING_COLOR = (255, 220, 70)
CUT_PLAYHEAD_COLOR = (245, 245, 245)

# SDL2 on Linux often posts WINDOWCLOSE instead of (or without) QUIT for the title-bar X.
_CLOSE_EVENTS: tuple[int, ...] = (pygame.QUIT,)
if hasattr(pygame, "WINDOWCLOSE"):
    _CLOSE_EVENTS = (pygame.QUIT, pygame.WINDOWCLOSE)

# Discrete playback-speed ladder (always includes 1.0× so [ / ] never skip normal speed).
PLAYBACK_SPEED_STEPS: tuple[float, ...] = (0.0625, 0.125, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0)


def _playback_speed_index(speed: float) -> int:
    best_i = 0
    best_d = float("inf")
    for i, step in enumerate(PLAYBACK_SPEED_STEPS):
        d = abs(step - speed)
        if d < best_d:
            best_d = d
            best_i = i
    return best_i


def _step_playback_speed(speed: float, direction: int) -> float:
    idx = _playback_speed_index(speed)
    new_idx = max(0, min(len(PLAYBACK_SPEED_STEPS) - 1, idx + direction))
    return PLAYBACK_SPEED_STEPS[new_idx]


def _format_playback_speed_label(speed: float) -> str:
    canonical = PLAYBACK_SPEED_STEPS[_playback_speed_index(speed)]
    if canonical >= 1.0 and canonical == int(canonical):
        return str(int(canonical))
    return f"{canonical:g}"


def _draw_playback_speed_hud(surface: pygame.Surface, speed: float, viewport_width: int) -> None:
    """Upper-right overlay on the video area (same placement style as getpixelvideo HUD)."""
    label = f"Speed: {_format_playback_speed_label(speed)}×"
    font = pygame.font.SysFont("verdana", 11)
    text = font.render(label, True, (238, 238, 245))
    pad_x, pad_y = 8, 5
    bg = pygame.Surface((text.get_width() + pad_x * 2, text.get_height() + pad_y + 3))
    bg.set_alpha(200)
    bg.fill((18, 18, 26))
    x = viewport_width - bg.get_width() - 8
    y = 8
    surface.blit(bg, (x, y))
    surface.blit(text, (x + pad_x, y + 3))


def clamp_frame_index(frame_idx: int, total_frames: int) -> int:
    """Clamp a zero-based frame index to a video timeline."""
    if total_frames <= 0:
        return 0
    return max(0, min(int(frame_idx), total_frames - 1))


def timeline_x_for_frame(
    frame_idx: int, total_frames: int, strip_left: int, strip_width: int
) -> int:
    """Map a zero-based frame index to an X coordinate on a timeline strip."""
    if strip_width <= 0 or total_frames <= 1:
        return strip_left
    frame_idx = clamp_frame_index(frame_idx, total_frames)
    return strip_left + int(round((frame_idx / (total_frames - 1)) * strip_width))


def frame_index_from_cut_timeline_x(
    *,
    mouse_x: int,
    strip_left: int,
    strip_width: int,
    total_frames: int,
    cut_markers: list[int],
) -> int:
    """Map a timeline click to a frame, snapping to a cut marker in that pixel column."""
    if strip_width <= 0 or total_frames <= 1:
        return 0
    rel_x = max(0.0, min(float(mouse_x - strip_left), float(strip_width)))
    px_col = min(strip_width - 1, int(rel_x))
    marker_frames = sorted(
        {
            clamp_frame_index(marker, total_frames)
            for marker in cut_markers
            if 0 <= marker < total_frames
        }
    )
    f0 = int(px_col * total_frames / strip_width)
    f1 = int((px_col + 1) * total_frames / strip_width)
    f1 = max(f0 + 1, min(f1, total_frames))
    lo = bisect.bisect_left(marker_frames, f0)
    hi = bisect.bisect_left(marker_frames, f1)
    markers_in_column = marker_frames[lo:hi]
    if markers_in_column:
        return markers_in_column[len(markers_in_column) // 2]
    target = int(round((rel_x / strip_width) * (total_frames - 1)))
    return clamp_frame_index(target, total_frames)


def adjacent_cut_marker_frame(frame_idx: int, cut_markers: list[int], direction: int) -> int | None:
    """Return previous/next cut marker with wraparound, or None when no markers exist."""
    markers = sorted(set(cut_markers))
    if not markers:
        return None
    if direction > 0:
        return next((marker for marker in markers if marker > frame_idx), markers[0])
    if direction < 0:
        return next((marker for marker in reversed(markers) if marker < frame_idx), markers[-1])
    raise ValueError("direction must be negative or positive")


def _continue_processing(progress_callback: Callable[[], bool | None] | None) -> bool:
    """Run an optional UI callback and report whether processing should continue."""
    return progress_callback is None or progress_callback() is not False


class RenderProgressDialog:
    """Small Tk progress window kept responsive while ffmpeg renders cuts."""

    def __init__(self, title: str, total_steps: int):
        self.cancelled = False
        self._root = None
        self._status = None
        self._progress = None
        root = None
        try:
            from tkinter import Tk, ttk

            root = Tk()
            root.title(title)
            root.geometry("520x155")
            root.resizable(False, False)
            root.protocol("WM_DELETE_WINDOW", self.request_cancel)

            ttk.Label(root, text=title, font=("Arial", 11, "bold")).pack(pady=(18, 8))
            self._status = ttk.Label(root, text="Preparing...")
            self._status.pack(pady=(0, 8))
            self._progress = ttk.Progressbar(
                root,
                orient="horizontal",
                length=450,
                mode="determinate",
                maximum=max(1, total_steps),
            )
            self._progress.pack(pady=(0, 10))
            ttk.Button(root, text="Cancel", command=self.request_cancel).pack()
            self._root = root
            self.update()
        except Exception as exc:
            print(f"Could not create render progress dialog: {exc}")
            if root is not None:
                with contextlib.suppress(Exception):
                    root.destroy()

    def request_cancel(self):
        """Request cancellation; active ffmpeg subprocess is terminated by its polling loop."""
        self.cancelled = True
        if self._status is not None:
            with contextlib.suppress(Exception):
                self._status.config(text="Cancelling after active operation stops...")

    def update(self, message: str | None = None, completed_steps: int | None = None) -> bool:
        """Refresh window and return False when user requested cancellation."""
        if self._root is None:
            return not self.cancelled
        try:
            if message is not None and self._status is not None:
                self._status.config(text=message)
            if completed_steps is not None and self._progress is not None:
                self._progress["value"] = completed_steps
            self._root.update_idletasks()
            self._root.update()
        except Exception as exc:
            print(f"Render progress dialog closed: {exc}")
            self._root = None
        return not self.cancelled

    def close(self):
        """Close progress window if it was created."""
        if self._root is not None:
            with contextlib.suppress(Exception):
                self._root.destroy()
            self._root = None


def sanitize_output_basename(name: str) -> str:
    """Return a filesystem-safe base name (no extension, no frame range suffix)."""
    cleaned = name.strip().rstrip("._- ")
    if not cleaned:
        return ""
    safe = "".join(c if c.isalnum() or c in "_-" else "_" for c in cleaned)
    safe = safe.strip("._- ")
    return safe


def effective_cut_basename(video_path: str | Path, custom_basename: str | None) -> str:
    """Resolve the prefix used in generated cut filenames."""
    stem = Path(video_path).stem
    if custom_basename:
        safe = sanitize_output_basename(custom_basename)
        if safe:
            return safe
    return stem


def cut_output_filename(basename: str, start_frame: int, end_frame: int, ext: str = ".mp4") -> str:
    """Build output filename with 1-based inclusive frame range (matches TOML/UI)."""
    return f"{basename}_frame_{start_frame + 1}_to_{end_frame + 1}{ext}"


def parse_basename_list(file_path: str | Path) -> list[str]:
    """Read a CSV/TXT list of output base names (one per cut).

    Accepts either a single comma-separated line or one name per line. When a
    line has several comma-separated fields, the first non-empty field is used
    (so a 2-column ``name,label`` CSV also works). Returns sanitized names;
    empty/invalid entries are kept as ``""`` so positions still map to cuts.
    """
    names: list[str] = []
    with open(file_path, encoding="utf-8") as f:
        content = f.read().strip()
    if not content:
        return names
    if "," in content and "\n" not in content:
        raw_items = content.split(",")
    else:
        raw_items = [line for line in content.splitlines() if line.strip()]
    for item in raw_items:
        first_field = item.split(",")[0]
        names.append(sanitize_output_basename(first_field))
    return names


def build_cut_output_filenames(
    video_path: str | Path,
    cuts: list[tuple[int, int]],
    custom_basename: str | None,
    per_cut_basenames: list[str] | None = None,
    ext: str = ".mp4",
) -> list[str]:
    """Return one output filename per cut, collision-safe.

    Priority per cut: CSV per-cut basename > single custom basename > video stem.
    When a per-cut basename is supplied it names the file exactly
    ``<basename><ext>`` (clean, professional); otherwise the classic
    ``<base>_frame_<start>_to_<end><ext>`` keeps full frame traceability.
    Duplicate names are disambiguated with ``_2``, ``_3`` suffixes.
    """
    names: list[str] = []
    used: dict[str, int] = {}
    for i, (start, end) in enumerate(cuts):
        per_cut = per_cut_basenames[i] if per_cut_basenames and i < len(per_cut_basenames) else ""
        if per_cut:
            candidate = f"{per_cut}{ext}"
        else:
            base = effective_cut_basename(video_path, custom_basename)
            candidate = cut_output_filename(base, start, end, ext)
        key = candidate.lower()
        if key in used:
            used[key] += 1
            stem = candidate[: -len(ext)] if ext and candidate.endswith(ext) else candidate
            candidate = f"{stem}_{used[key]}{ext}"
            used[candidate.lower()] = 1
        else:
            used[key] = 1
        names.append(candidate)
    return names


def get_precise_video_metadata(video_path):
    """
    Get precise video metadata using ffprobe to avoid rounding errors.
    Returns dict with fps (float), width, height, codec, etc.
    """
    try:
        cmd = [
            get_ffprobe_path(),
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(video_path),
        ]
        result = subprocess.run(
            cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", check=True
        )
        if result.stdout is None or not result.stdout.strip():
            raise ValueError("ffprobe returned no output")
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

        # Convert fraction strings to float and extract exact numerator/denominator components
        fps_num, fps_den = None, None

        def parse_fraction(frac_str):
            try:
                if "/" in frac_str:
                    n, d = map(int, frac_str.split("/"))
                    return (n, d, float(n) / d if d != 0 else 0.0)
                val = float(frac_str)
                return (int(val * 1000), 1000, val)
            except (ValueError, ZeroDivisionError):
                return (None, None, None)

        r_n, r_d, r_fps = parse_fraction(r_frame_rate_str)
        a_n, a_d, avg_fps = parse_fraction(avg_frame_rate_str)

        # Use avg_frame_rate if available and valid, otherwise r_frame_rate
        if avg_fps and avg_fps > 0:
            fps, fps_num, fps_den = avg_fps, a_n, a_d
        elif r_fps and r_fps > 0:
            fps, fps_num, fps_den = r_fps, r_n, r_d
        else:
            fps, fps_num, fps_den = 30.0, 30, 1

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

        # Extract rotation angle
        rotation = 0
        for sd in video_stream.get("side_data_list", []):
            if sd.get("side_data_type") == "Display Matrix" and "rotation" in sd:
                try:
                    rotation = int(float(sd["rotation"]))
                    break
                except (ValueError, TypeError):
                    pass
        if rotation == 0 and "tags" in video_stream:
            rotate_tag = video_stream["tags"].get("rotate")
            if rotate_tag:
                with contextlib.suppress(ValueError, TypeError):
                    rotation = int(float(rotate_tag))

        rotation = rotation % 360
        raw_width = int(video_stream.get("width"))
        raw_height = int(video_stream.get("height"))

        # Swap width and height if rotated by 90 or 270 degrees
        if rotation in (90, 270):
            width = raw_height
            height = raw_width
        else:
            width = raw_width
            height = raw_height

        return {
            "fps": fps,
            "width": width,
            "height": height,
            "codec": video_stream.get("codec_name", "unknown"),
            "r_frame_rate": r_frame_rate_str,
            "avg_frame_rate": avg_frame_rate_str,
            "fps_num": fps_num,
            "fps_den": fps_den,
            "duration": duration if duration > 0 else None,
            "nb_frames": nb_frames,
            "rotation": rotation,
        }
    except (
        subprocess.CalledProcessError,
        json.JSONDecodeError,
        FileNotFoundError,
        ValueError,
        TypeError,
    ) as e:
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
            "fps_num": int(fps) if fps else 30,
            "fps_den": 1,
            "rotation": 0,
        }


def check_and_rotate_frame(frame, metadata):
    """
    Ensure the frame matches the display dimensions by rotating it if needed.
    """
    if frame is None or not metadata or "rotation" not in metadata:
        return frame
    rot = metadata["rotation"] % 360
    if rot in (90, 180, 270):
        h, w = frame.shape[:2]
        target_w = metadata["width"]
        target_h = metadata["height"]
        if w == target_w and h == target_h:
            # Already rotated by OpenCV/backend
            return frame
        # Apply manual rotation
        if rot == 90:
            return cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        elif rot == 180:
            return cv2.rotate(frame, cv2.ROTATE_180)
        elif rot == 270:
            return cv2.rotate(frame, cv2.ROTATE_90_COUNTERCLOCKWISE)
    return frame


def cut_video_with_ffmpeg(
    video_path,
    output_path,
    start_frame,
    end_frame,
    metadata,
    progress_callback: Callable[[], bool | None] | None = None,
):
    """Cut video with ffmpeg while keeping an optional progress UI responsive.

    Prefers verified hardware H.264 (NVENC / VideoToolbox); falls back to CPU
    ``libx264`` on all OS.
    """
    ffmpeg_check = get_ffmpeg_path()
    try:
        subprocess.run(
            [ffmpeg_check, "-version"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            check=True,
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return cut_video_with_opencv(
            video_path,
            output_path,
            start_frame,
            end_frame,
            metadata,
            progress_callback=progress_callback,
        )

    fps = metadata["fps"]
    frame_count = end_frame - start_frame + 1
    start_time = start_frame / fps if fps > 0 else 0.0

    for encoder in encoders_with_cpu_fallback():
        device = encoder_device_tag(encoder)
        ffmpeg = get_video_encode_ffmpeg_path(encoder)
        print(
            f"  [FFmpeg][{device}] Cutting with {encoder} "
            f"({describe_video_encoder(encoder)}) via {ffmpeg}..."
        )
        cmd_reencode = [
            ffmpeg,
            "-y",
            "-ss",
            f"{start_time:.6f}",
            "-i",
            str(video_path),
            "-frames:v",
            str(frame_count),
            *get_ffmpeg_video_encoding_args(encoder),
            "-c:a",
            "aac",
            "-b:a",
            "192k",
            "-r",
            f"{fps:.6f}",
            "-avoid_negative_ts",
            "make_zero",
            str(output_path),
        ]
        try:
            process = subprocess.Popen(cmd_reencode)
            while process.poll() is None:
                if not _continue_processing(progress_callback):
                    print("Video cut cancelled by user")
                    process.terminate()
                    try:
                        process.wait(timeout=2)
                    except subprocess.TimeoutExpired:
                        process.kill()
                        process.wait()
                    return False
                time.sleep(0.05)
            if process.returncode != 0:
                raise subprocess.CalledProcessError(process.returncode, cmd_reencode)
            print(f"  [FFmpeg][{device}] Finished cut encode with {encoder}")
            return _continue_processing(progress_callback)
        except (OSError, subprocess.CalledProcessError) as exc:
            if is_hardware_video_encoder(encoder):
                print(
                    f"  [FFmpeg][GPU] Warning: {encoder} cut failed ({exc}); "
                    "retrying with CPU libx264"
                )
                continue
            print(f"Error with ffmpeg re-encoding: {exc}")
            return cut_video_with_opencv(
                video_path,
                output_path,
                start_frame,
                end_frame,
                metadata,
                progress_callback=progress_callback,
            )

    return cut_video_with_opencv(
        video_path,
        output_path,
        start_frame,
        end_frame,
        metadata,
        progress_callback=progress_callback,
    )


def cut_video_with_opencv(
    video_path,
    output_path,
    start_frame,
    end_frame,
    metadata,
    progress_callback: Callable[[], bool | None] | None = None,
):
    """Fallback video cutter with cooperative progress UI updates."""
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return False

    fps = metadata["fps"]
    width = metadata["width"]
    height = metadata["height"]
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    if not out.isOpened():
        cap.release()
        return False

    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
        for frame_offset in range(end_frame - start_frame + 1):
            if frame_offset % 10 == 0 and not _continue_processing(progress_callback):
                print("Video cut cancelled by user")
                return False
            ret, frame = cap.read()
            if not ret:
                break
            frame = check_and_rotate_frame(frame, metadata)
            out.write(frame)
        return _continue_processing(progress_callback)
    finally:
        out.release()
        cap.release()


def _gcd(a: int, b: int) -> int:
    while b:
        a, b = b, a % b
    return abs(a)


def _float_fps_to_rational(fps_val: float) -> tuple[int, int]:
    """Map a user-entered float FPS to a reduced numerator/denominator pair."""
    fps_num = int(round(fps_val * 1000))
    fps_den = 1000
    g = _gcd(fps_num, fps_den)
    return fps_num // g, fps_den // g


def _resolve_cuts_timing_fps(
    fps_override: float | None, video_path: str | Path
) -> tuple[float, int, int]:
    """Return (fps, fps_num, fps_den) for TOML cut timing fields."""
    try:
        mm = get_precise_video_metadata(video_path)
        meta_fps = float(mm["fps"])
        if fps_override is None:
            return (
                meta_fps,
                int(mm.get("fps_num", 30)),
                int(mm.get("fps_den", 1)),
            )
        if abs(fps_override - meta_fps) < 0.001:
            return (
                meta_fps,
                int(mm.get("fps_num", 30)),
                int(mm.get("fps_den", 1)),
            )
    except Exception:
        if fps_override is not None and fps_override > 0:
            fps_num, fps_den = _float_fps_to_rational(fps_override)
            return fps_override, fps_num, fps_den
        return 30.0, 30, 1

    fps_num, fps_den = _float_fps_to_rational(fps_override)
    return fps_override, fps_num, fps_den


def _cut_times_for_toml(
    start_frame_1based: int, frame_count: int, fps_num: int, fps_den: int
) -> tuple[float, float, float]:
    """Compute start/end/duration seconds using the same convention as the cut UI."""
    start_time = (start_frame_1based * fps_den) / fps_num
    duration = (frame_count * fps_den) / fps_num
    return start_time, start_time + duration, duration


def save_cuts_to_toml(
    video_path, cuts, fps=None, output_dir=None, per_cut_outputs=None, labels=None
):
    """Save cuts information to a TOML file.

    output_dir: optional Path/str for planned output directory.
    per_cut_outputs: optional list of filenames (one per cut) to record planned outputs.
    labels: optional list of string labels for each cut.
    """

    effective_fps, fps_num, fps_den = _resolve_cuts_timing_fps(fps, video_path)
    if fps is None:
        fps = effective_fps
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
            if fps is not None and fps > 0 and fps_num > 0:
                start_time, end_time, duration = _cut_times_for_toml(
                    start_frame_1based, frame_count, fps_num, fps_den
                )
            else:
                start_time = None
                end_time = None
                duration = None

            toml_content += "[[cuts]]\n"
            toml_content += f"index = {i}\n"
            if labels and i - 1 < len(labels):
                safe_label = labels[i - 1].replace('"', '\\"')
                toml_content += f'label = "{safe_label}"\n'
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
                if fps is not None and fps > 0 and fps_num > 0:
                    start_time, end_time, duration = _cut_times_for_toml(
                        start_frame_1based, frame_count, fps_num, fps_den
                    )
                else:
                    start_time = None
                    end_time = None
                    duration = None

                toml_content += "[[cuts]]\n"
                toml_content += f"index = {i}\n"
                if labels and i - 1 < len(labels):
                    safe_label = labels[i - 1].replace('"', '\\"')
                    toml_content += f'label = "{safe_label}"\n'
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


def parse_sync_file_content(selected_file, video_path):
    """Common logic for parsing sync file content (from dialog or auto-loading)."""
    video_path_obj = Path(video_path).absolute()
    video_name = video_path_obj.name
    video_stem = video_path_obj.stem

    cuts = []
    sync_data = {}
    is_sync_file = False

    try:
        with open(selected_file, encoding="utf-8", errors="replace") as f:
            lines = f.readlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            # Parse sync file format: video_file new_name initial_frame final_frame
            parts = line.split()
            if len(parts) >= 4:
                try:
                    video_file = parts[0]
                    new_name = parts[1]
                    initial_frame_val = int(parts[2])
                    final_frame_val = int(parts[3])

                    # It's definitely a sync file format if it has 4+ parts and ints at the end
                    is_sync_file = True

                    # Store sync data for all videos in the file
                    sync_data[video_file] = {
                        "new_name": new_name,
                        "initial_frame": initial_frame_val,
                        "final_frame": final_frame_val,
                    }

                    # Matching logic:
                    # 1. Exact match (case-insensitive)
                    # 2. Stem match (case-insensitive)
                    # 3. Substring match
                    match_found = False
                    if (
                        video_name.lower() == video_file.lower()
                        or video_stem.lower() == video_file.lower()
                        or video_file.lower() == video_stem.lower()
                        or video_name.lower() in video_file.lower()
                        or video_file.lower() in video_name.lower()
                    ):
                        match_found = True

                    if match_found:
                        # Use 1-based to 0-based conversion
                        # Safety clamp to 0
                        start = max(0, initial_frame_val - 1)
                        end = max(0, final_frame_val - 1)
                        cuts.append((start, end))
                        print(
                            f"  [green]Match found in sync file:[/] {video_file} -> {start + 1} to {end + 1}"
                        )

                except (ValueError, IndexError):
                    continue

        return cuts, is_sync_file, sync_data
    except Exception as e:
        print(f"  [red]Error parsing sync file {selected_file}:[/] {e}")
        return [], False, {}


def load_sync_file(video_path):
    """Automatically search and load synchronization data from sync file (*.txt)."""
    try:
        video_path_obj = Path(video_path).absolute()
        video_dir = video_path_obj.parent
        video_name = video_path_obj.name

        print(f"Searching for sync files for {video_name} in {video_dir}...")

        # Look for sync files in the directory (*.txt)
        sync_files = list(video_dir.glob("*.txt")) + list(video_dir.glob("*.TXT"))
        sync_files = list(set(sync_files))  # Deduplicate

        for sync_file in sync_files:
            # Skip common non-sync files if they are huge or clearly irrelevant
            if sync_file.name.lower() in ["requirements.txt", "readme.txt"]:
                continue

            print(f"Trying sync file: {sync_file.name}")
            cuts, is_sync, sync_data = parse_sync_file_content(sync_file, video_path)
            if cuts:
                return cuts, sync_data

    except Exception as e:
        print(f"Error in load_sync_file: {e}")

    return [], None


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
            # Fall through to try TXT file

    # Fallback: Try to load from old TXT format for backward compatibility
    if txt_path.exists():
        try:
            with open(txt_path, encoding="utf-8", errors="replace") as f:
                lines = f.readlines()

            # Parse each line looking for cut information
            for line in lines:
                line = line.strip()
                if not line:
                    continue

                # Check if this is a cut line (format: "Cut N: Frame X to Y" or "Cut N: Frame X to Y (Time: ...)")
                if line.startswith("Cut ") and "Frame" in line:
                    try:
                        # Find "Frame " and extract numbers
                        frame_part = line.split("Frame ")[1].split(" (")[0]
                        parts = frame_part.split(" to ")
                        # TXT format uses 1-based frames
                        start = int(parts[0]) - 1
                        end = int(parts[1]) - 1
                        cuts.append((start, end))
                    except (ValueError, IndexError):
                        continue

            return cuts
        except Exception as e:
            print(f"Error loading TXT file: {e}")

    return cuts


def load_cuts_or_sync(video_path):
    """Load cuts from either cut file or sync file. Returns (cuts, is_sync, sync_data)."""
    # First try to load from sync file
    sync_cuts, sync_data = load_sync_file(video_path)
    if sync_cuts:
        return sync_cuts, True, sync_data

    # If no sync file, try regular cuts file (TOML or TXT fallback)
    regular_cuts = load_cuts_from_toml(video_path)
    return regular_cuts, False, None


def extract_audio_data(video_path, target_sr=44100):
    """
    Extract raw mono PCM audio from video using ffmpeg and return (audio_array, sample_rate).
    If there is no audio stream or extraction fails, returns (None, None).
    """
    try:
        probe_cmd = [
            get_ffprobe_path(),
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
        has_audio = subprocess.run(
            probe_cmd, stdout=subprocess.PIPE, text=True, encoding="utf-8", errors="replace"
        ).stdout.strip()
        if not has_audio:
            return None, None

        cmd = [
            get_ffmpeg_path(),
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
            capture_output=True,
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
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp_path = tmp.name

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
            ("Sync files (TXT)", "*.txt *.TXT"),
            ("TOML cuts files", "*.toml *.TOML"),
            ("All files", "*.*"),
        ],
        initialdir=Path(video_path).parent,
    )
    root.destroy()

    if not selected_file:
        return [], False, None

    file_path = Path(selected_file)

    # Check if it's a TOML file (cuts file)
    if file_path.suffix.lower() == ".toml":
        try:
            cuts = load_cuts_from_toml_file(selected_file)
            if cuts:
                print(f"Loaded {len(cuts)} cuts from TOML file")
                return cuts, False, None
            else:
                from tkinter import messagebox

                messagebox.showwarning(
                    "No Cuts Found", "The selected TOML file does not contain any cuts."
                )
                return [], False, None
        except Exception as e:
            from tkinter import messagebox

            messagebox.showerror("Error", f"Error loading TOML file: {e}")
            return [], False, None

    # Otherwise, treat as sync file (TXT format)
    print(f"Manual loading sync file: {selected_file}")
    cuts, is_sync, sync_data = parse_sync_file_content(selected_file, video_path)

    if is_sync:
        if not cuts:
            from tkinter import messagebox

            video_name = Path(video_path).name
            messagebox.showwarning(
                "Video Not Found in Sync File",
                f"Sync file loaded, but video '{video_name}' was not found in it.\n\n"
                "Check if the filename in the TXT matches exactly.",
            )
        return cuts, True, sync_data
    else:
        from tkinter import messagebox

        messagebox.showerror("Error", "The selected file is not in a recognized sync file format.")
        return [], False, None


def batch_process_sync_videos(video_path, sync_data):
    """Process all videos in a sync file with visible, cancellable progress."""
    if not sync_data:
        return False

    video_dir = Path(video_path).parent
    video_name = Path(video_path).stem
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = video_dir / f"vailacut_sync_{video_name}_{timestamp}"
    output_dir.mkdir(exist_ok=True)

    processed_count = 0
    progress_dialog = RenderProgressDialog("Processing synchronized videos", len(sync_data))
    try:
        for item_idx, (video_file, sync_info) in enumerate(sync_data.items()):
            if not progress_dialog.update(f"Preparing {video_file}", item_idx):
                break
            video_path_full = video_dir / video_file
            if not video_path_full.exists():
                print(f"Warning: Video file {video_file} not found")
                continue

            try:
                metadata = get_precise_video_metadata(video_path_full)
                total_frames = metadata.get("nb_frames") or (
                    int(metadata.get("duration", 0) * metadata["fps"])
                    if metadata.get("duration")
                    else int(cv2.VideoCapture(str(video_path_full)).get(cv2.CAP_PROP_FRAME_COUNT))
                )
                start_frame = sync_info["initial_frame"]
                end_frame = sync_info["final_frame"]
                if start_frame >= total_frames:
                    print(
                        f"Warning: Start frame {start_frame} beyond video length for {video_file}"
                    )
                    continue

                actual_end_frame = min(end_frame, total_frames - 1)
                output_path = output_dir / sync_info["new_name"]
                progress_dialog.update(
                    f"Rendering {video_file} ({item_idx + 1}/{len(sync_data)})",
                    item_idx,
                )
                success = cut_video_with_ffmpeg(
                    video_path_full,
                    output_path,
                    start_frame,
                    actual_end_frame,
                    metadata,
                    progress_callback=progress_dialog.update,
                )
                if progress_dialog.cancelled:
                    break
                if success:
                    processed_count += 1
                    print(
                        f"Processed: {video_file} -> {sync_info['new_name']} "
                        f"(FPS: {metadata['fps']:.6f})"
                    )
                else:
                    print(f"Error processing: {video_file}")
            except Exception as exc:
                print(f"Error processing {video_file}: {exc}")
            finally:
                progress_dialog.update(completed_steps=item_idx + 1)
    finally:
        progress_dialog.close()

    if progress_dialog.cancelled:
        print("Synchronized video processing cancelled by user")
    return processed_count > 0


def play_video_with_cuts(video_path):
    from tkinter import Tk, filedialog, messagebox, simpledialog, ttk

    pygame.init()

    # Initialize video capture with fallback conversion logic
    cap = cv2.VideoCapture(video_path)
    video_valid = cap.isOpened()

    if video_valid:
        # Try reading the first frame to ensure codec support (e.g. AV1 fallback)
        ret, _ = cap.read()
        if not ret:
            video_valid = False
        else:
            cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    if not video_valid:
        cap.release()
        print(f"Error opening or reading video: {video_path}")

        # Initialize Tk root if needed for the dialog.
        root = None
        try:
            root = Tk()
            root.withdraw()
        except Exception:
            if root is not None:
                with contextlib.suppress(Exception):
                    root.destroy()
            return

        if messagebox.askyesno(
            "Video Load Error",
            f"Could not open/read video: {Path(video_path).name}\n\n"
            "This usually happens when the video codec (e.g., AV1) is not supported "
            "by your OpenCV installation.\n\n"
            "Do you want to automatically convert it to H.264 (MP4) format?",
            parent=root,
        ):
            converted_path = str(
                Path(video_path).parent / (Path(video_path).stem + "_converted.mp4")
            )
            print(
                f"[bold yellow]Preparing to convert video to H.264:[/bold yellow] {converted_path}"
            )

            # Helper function to run conversion with visual feedback
            def convert_with_feedback():
                conversion_success = False
                conversion_error = None

                # Create a progress window
                progress_win = Tk()
                progress_win.title("Converting Video")
                progress_win.geometry("400x150")
                progress_win.resizable(False, False)
                if root:
                    # Center relative to parent if possible, otherwise screen center
                    with contextlib.suppress(Exception):
                        x = root.winfo_x() + (root.winfo_width() // 2) - 200
                        y = root.winfo_y() + (root.winfo_height() // 2) - 75
                        progress_win.geometry(f"+{x}+{y}")

                ttk.Label(
                    progress_win,
                    text="Converting video to compatible format...",
                    font=("Arial", 11),
                ).pack(pady=(20, 10))
                ttk.Label(
                    progress_win,
                    text="Please wait, this may take a moment based on video size.",
                    font=("Arial", 9),
                ).pack(pady=(0, 15))

                pbar = ttk.Progressbar(progress_win, mode="indeterminate")
                pbar.pack(fill="x", padx=30, pady=5)
                pbar.start(10)

                # Conversion logic in a thread
                def run_ffmpeg():
                    nonlocal conversion_success, conversion_error
                    try:
                        last_error = None
                        for encoder in encoders_with_cpu_fallback():
                            device = encoder_device_tag(encoder)
                            ffmpeg = get_video_encode_ffmpeg_path(encoder)
                            print(
                                f"  [FFmpeg][{device}] Converting with {encoder} "
                                f"({describe_video_encoder(encoder)}) via {ffmpeg}..."
                            )
                            cmd = [
                                ffmpeg,
                                "-y",
                                "-i",
                                str(video_path),
                                *get_ffmpeg_video_encoding_args(encoder),
                                "-c:a",
                                "aac",
                                "-b:a",
                                "192k",
                                "-pix_fmt",
                                "yuv420p",
                                str(converted_path),
                            ]

                            # Use rich progress in terminal
                            with Progress(
                                SpinnerColumn(),
                                TextColumn("[bold blue]{task.description}"),
                                TimeElapsedColumn(),
                                transient=True,
                            ) as progress:
                                progress.add_task(
                                    f"Converting {Path(video_path).name}...", total=None
                                )

                                # Run subprocess (UTF-8 to avoid UnicodeDecodeError on Windows cp1252)
                                try:
                                    subprocess.run(
                                        cmd,
                                        check=True,
                                        capture_output=True,
                                        text=True,
                                        encoding="utf-8",
                                        errors="replace",
                                    )
                                    print(f"  [FFmpeg][{device}] Finished convert with {encoder}")
                                    conversion_success = True
                                    last_error = None
                                    break
                                except subprocess.CalledProcessError as e:
                                    last_error = e
                                    if is_hardware_video_encoder(encoder):
                                        print(
                                            f"  [FFmpeg][GPU] Warning: {encoder} convert "
                                            "failed; retrying with CPU libx264"
                                        )
                                        continue
                                    raise

                        if not conversion_success and last_error is not None:
                            raise last_error
                    except subprocess.CalledProcessError as e:
                        err_text = (
                            e.stderr
                            if isinstance(e.stderr, str)
                            else (
                                e.stderr.decode("utf-8", errors="replace")
                                if e.stderr
                                else "Unknown error"
                            )
                        )
                        conversion_error = f"FFmpeg failed: {err_text}"
                    except Exception as e:
                        conversion_error = str(e)
                    finally:
                        # Schedule window close on main thread
                        def cleanup():
                            try:
                                pbar.stop()  # Important: stop timer events before destroying
                                progress_win.destroy()
                            except Exception:
                                pass

                        progress_win.after(0, cleanup)

                t = threading.Thread(target=run_ffmpeg)
                t.daemon = True
                t.start()

                # Keep window open until thread finishes
                progress_win.mainloop()

                return conversion_success, conversion_error

            # Run the conversion
            success, error = convert_with_feedback()

            if success:
                print(f"[bold green]Conversion successful:[/bold green] {converted_path}")
                messagebox.showinfo(
                    "Success",
                    f"Video converted successfully!\nNow loading: {Path(converted_path).name}",
                    parent=root,
                )

                # Switch to the new video
                video_path = converted_path
                cap = cv2.VideoCapture(video_path)

                if not cap.isOpened():
                    messagebox.showerror(
                        "Error", "Could not open the converted video.", parent=root
                    )
                    root.destroy()
                    return
            else:
                error_msg = error if error else "Unknown error during conversion"
                print(f"[bold red]Conversion failed:[/bold red] {error_msg}")
                messagebox.showerror(
                    "Conversion Failed",
                    f"Failed to convert video:\n{error_msg}\n\nEnsure ffmpeg is installed.",
                    parent=root,
                )
                root.destroy()
                return
        else:
            root.destroy()
            return
        root.destroy()

    # Get precise video metadata using ffprobe
    metadata = get_precise_video_metadata(video_path)
    fps = metadata["fps"]  # Use precise float FPS
    fps_num = metadata.get("fps_num", 30)
    fps_den = metadata.get("fps_den", 1)
    original_fps = fps

    def get_time_s(f_count, current_fps):
        if current_fps == original_fps and fps_num and fps_den:
            return (f_count * fps_den) / fps_num
        return f_count / current_fps if current_fps > 0 else 0.0

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

    # UI layout constants (cut-marker strip + scrub slider + option buttons)
    control_height = 104
    CUT_TIMELINE_ROW_Y = 28
    CUT_TIMELINE_HEIGHT = 9
    SLIDER_ROW_Y = 45
    BUTTON_ROW_Y = 74
    audio_height = 150
    show_audio = False
    audio_data = None
    audio_loaded = False
    sample_rate = 44100
    audio_wave_path = None
    audio_muted = False
    audio_ready = False
    audio_music_loaded = False
    loop_enabled = False

    # Get video filename for window title
    video_filename = Path(video_path).name
    output_basename = (
        None  # Must exist before nested funcs that read it (update_caption, draw_controls)
    )
    output_basenames: list[str] = []  # Optional per-cut base names loaded from a CSV/TXT list

    def set_display_mode():
        total_h = window_height + control_height + (audio_height if show_audio else 0)
        return pygame.display.set_mode((window_width, total_h), pygame.RESIZABLE)

    def restore_pygame_display():
        """Re-open the display after pygame.display.quit() (required on Linux for WM events)."""
        if not pygame.display.get_init():
            pygame.display.init()
        screen_local = set_display_mode()
        update_caption()
        return screen_local

    def update_caption():
        if output_basenames:
            base_info = f"Names:CSV({len(output_basenames)})"
        else:
            base_info = f"Base:{effective_cut_basename(video_path, output_basename)}"
        pygame.display.set_caption(
            f"{video_filename} (FPS: {fps:.2f}) | {base_info} | "
            "A:Audio M:Mute B:BaseName N:NamesCSV 0:AutoFit +/-:Zoom | Space:Play/Pause | ←→:Frame | "
            "S:Start E:End R:Reset DEL/D:Remove | L:List | F:Load | Home/End | PgUp/PgDn | "
            "G:Frame T:Time I/P:FPS Shift+←/→:Markers | H:Help ESC:Save"
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
    playback_speed = 1.0
    slow_mo_accumulator = 0.0
    last_valid_frame = None
    cuts = []  # List to store (start, end) frame pairs
    cut_labels = []  # List to store labels for each cut
    current_start = None
    using_sync_file = False
    sync_data = None  # Store sync data for batch processing
    zoom_level = 1.0
    offset_x = 0.0
    offset_y = 0.0
    panning = False
    dragging_cut_timeline = False

    def clamp_pan_offsets():
        nonlocal offset_x, offset_y
        scale_fit = min(window_width / original_width, window_height / original_height)
        zoomed_width = max(1, int(original_width * scale_fit * zoom_level))
        zoomed_height = max(1, int(original_height * scale_fit * zoom_level))
        max_x = max(0, zoomed_width - window_width)
        max_y = max(0, zoomed_height - window_height)
        offset_x = max(0.0, min(float(max_x), offset_x))
        offset_y = max(0.0, min(float(max_y), offset_y))

    def reset_zoom_pan():
        nonlocal zoom_level, offset_x, offset_y
        zoom_level = 1.0
        offset_x = 0.0
        offset_y = 0.0

    def get_max_safe_zoom():
        scale_fit_local = min(window_width / original_width, window_height / original_height)
        base_w = max(1.0, float(original_width) * scale_fit_local)
        base_h = max(1.0, float(original_height) * scale_fit_local)
        max_by_pixels = (MAX_RENDER_PIXELS / (base_w * base_h)) ** 0.5
        return max(1.0, min(10.0, max_by_pixels))

    def clamp_zoom_level():
        nonlocal zoom_level
        safe_zoom = get_max_safe_zoom()
        if zoom_level > safe_zoom:
            zoom_level = safe_zoom

    # Load existing cuts or sync file if available
    cuts, using_sync_file, sync_data = load_cuts_or_sync(video_path)
    if len(cuts) > 0:
        if using_sync_file:
            print(f"Loaded {len(cuts)} sync points from sync file")
        else:
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
            pos_seconds = get_time_s(frame_idx, fps)
            pygame.mixer.music.set_pos(pos_seconds)
        except Exception as e:
            print(f"Audio seek failed: {e}")

    def start_audio_playback(frame_idx: int):
        if not audio_ready or not audio_loaded or audio_muted:
            return
        try:
            pos_seconds = get_time_s(frame_idx, fps)
            # start playback; looping once is enough for single video
            pygame.mixer.music.play(loops=0, start=pos_seconds)
        except Exception as e:
            print(f"Audio play failed: {e}")

    def stop_audio_playback():
        if audio_ready:
            with contextlib.suppress(Exception):
                pygame.mixer.music.stop()

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
        current_time = get_time_s(current_f, fps_val)
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

        # Draw clickable cut-marker strip above main scrub slider, matching getpixelvideo.
        slider_width = int(window_width * 0.8)
        slider_x = (window_width - slider_width) // 2
        slider_y = SLIDER_ROW_Y
        slider_height = 10
        cut_timeline_rect = pygame.Rect(
            slider_x,
            CUT_TIMELINE_ROW_Y,
            slider_width,
            CUT_TIMELINE_HEIGHT,
        )
        pygame.draw.rect(slider_surface, (48, 48, 48), cut_timeline_rect)
        for start, end in cuts:
            start_x = timeline_x_for_frame(start, total_frames, slider_x, slider_width)
            end_x = timeline_x_for_frame(end, total_frames, slider_x, slider_width)
            pygame.draw.line(
                slider_surface,
                CUT_RANGE_COLOR,
                (start_x, cut_timeline_rect.centery),
                (end_x, cut_timeline_rect.centery),
                5,
            )
            pygame.draw.line(
                slider_surface,
                CUT_START_COLOR,
                (start_x, cut_timeline_rect.top),
                (start_x, cut_timeline_rect.bottom - 1),
                2,
            )
            pygame.draw.line(
                slider_surface,
                CUT_END_COLOR,
                (end_x, cut_timeline_rect.top),
                (end_x, cut_timeline_rect.bottom - 1),
                2,
            )
        if current_start is not None:
            pending_x = timeline_x_for_frame(current_start, total_frames, slider_x, slider_width)
            pygame.draw.line(
                slider_surface,
                CUT_PENDING_COLOR,
                (pending_x, cut_timeline_rect.top - 2),
                (pending_x, cut_timeline_rect.bottom + 1),
                3,
            )
        playhead_x = timeline_x_for_frame(frame_count, total_frames, slider_x, slider_width)
        pygame.draw.line(
            slider_surface,
            CUT_PLAYHEAD_COLOR,
            (playhead_x, cut_timeline_rect.top),
            (playhead_x, cut_timeline_rect.bottom - 1),
            1,
        )

        pygame.draw.rect(
            slider_surface,
            (60, 60, 60),
            (slider_x, slider_y, slider_width, slider_height),
        )
        slider_pos = timeline_x_for_frame(frame_count, total_frames, slider_x, slider_width)
        pygame.draw.circle(
            slider_surface,
            (255, 255, 255),
            (slider_pos, slider_y + slider_height // 2),
            8,
        )

        # Draw frame information and cut markers
        font = pygame.font.Font(None, 24)
        time_seconds = get_time_s(frame_count, fps)
        time_total = get_time_s(total_frames, fps)
        frame_text = font.render(
            f"Frame: {frame_count + 1}/{total_frames} ({time_seconds:.6f}s/{time_total:.6f}s)",
            True,
            (255, 255, 255),
        )
        slider_surface.blit(frame_text, (10, 10))

        # Draw cut status on the right without overlapping pending-start feedback.
        status_font = pygame.font.Font(None, 20)
        sync_status = " (SYNC)" if using_sync_file else ""
        cuts_text = status_font.render(f"Cuts: {len(cuts)}{sync_status}", True, (255, 255, 255))
        cuts_x = max(10, window_width - cuts_text.get_width() - 12)
        slider_surface.blit(cuts_text, (cuts_x, 8))
        if current_start is not None:
            cut_text = status_font.render(
                f"Current Cut Start: {current_start + 1}", True, CUT_PENDING_COLOR
            )
            cut_x = max(10, cuts_x - cut_text.get_width() - 18)
            slider_surface.blit(cut_text, (cut_x, 8))

        # Smaller font for option buttons (second row, below timeline)
        button_font = pygame.font.Font(None, 15)
        button_width = 70
        button_h = 22
        button_gap = 6
        buttons_right = window_width - 10

        base_button_rect = pygame.Rect(
            buttons_right - button_width, BUTTON_ROW_Y, button_width, button_h
        )
        names_button_rect = pygame.Rect(
            base_button_rect.left - button_gap - button_width,
            BUTTON_ROW_Y,
            button_width,
            button_h,
        )
        help_button_rect = pygame.Rect(
            names_button_rect.left - button_gap - button_width,
            BUTTON_ROW_Y,
            button_width,
            button_h,
        )
        loop_button_rect = pygame.Rect(
            help_button_rect.left - button_gap - button_width,
            BUTTON_ROW_Y,
            button_width,
            button_h,
        )

        loop_color = (60, 120, 60) if loop_enabled else (90, 90, 90)
        pygame.draw.rect(slider_surface, loop_color, loop_button_rect)
        loop_text = button_font.render(
            "Loop" if loop_enabled else "Loop off", True, (255, 255, 255)
        )
        slider_surface.blit(loop_text, loop_text.get_rect(center=loop_button_rect.center))

        pygame.draw.rect(slider_surface, (100, 100, 100), help_button_rect)
        help_text = button_font.render("Help", True, (255, 255, 255))
        slider_surface.blit(help_text, help_text.get_rect(center=help_button_rect.center))

        base_active = output_basename is not None and bool(
            sanitize_output_basename(output_basename)
        )
        base_color = (80, 100, 140) if base_active else (70, 70, 70)
        pygame.draw.rect(slider_surface, base_color, base_button_rect)
        base_label = "Base ✓" if base_active else "Base name"
        base_text = button_font.render(base_label, True, (255, 255, 255))
        slider_surface.blit(base_text, base_text.get_rect(center=base_button_rect.center))

        names_active = bool(output_basenames)
        names_color = (140, 110, 70) if names_active else (70, 70, 70)
        pygame.draw.rect(slider_surface, names_color, names_button_rect)
        names_label = f"Names ✓{len(output_basenames)}" if names_active else "Cut names"
        names_text = button_font.render(names_label, True, (255, 255, 255))
        slider_surface.blit(names_text, names_text.get_rect(center=names_button_rect.center))

        base_hint_font = pygame.font.Font(None, 18)
        if output_basenames:
            hint_str = f"Out: per-cut CSV names ({len(output_basenames)})"
        else:
            file_base = effective_cut_basename(video_path, output_basename)
            hint_str = f"Out: {file_base}_frame_…"
        base_hint = base_hint_font.render(hint_str, True, (180, 200, 255))
        slider_surface.blit(base_hint, (10, BUTTON_ROW_Y + 2))

        if show_audio:
            audio_indicator = font.render("[AUDIO ON]", True, (0, 255, 0))
            slider_surface.blit(audio_indicator, (window_width - 260, 10))

        screen.blit(slider_surface, (0, base_y))
        return (
            slider_x,
            slider_width,
            slider_y,
            slider_height,
            cut_timeline_rect,
            help_button_rect,
            loop_button_rect,
            base_button_rect,
            names_button_rect,
            base_y,
        )

    def get_all_cut_markers():
        """Get a sorted flat list of all cut markers (start and end frames)."""
        markers = []
        for start, end in cuts:
            markers.append(start)
            markers.append(end)
        # Remove duplicates and sort
        markers = sorted(set(markers))
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
            "- [ / ]: Decrease / Increase playback speed (halve / double; 0.0625×–16×)",
            "- Right Arrow: Next Frame (when paused)",
            "- Left Arrow: Previous Frame (when paused)",
            "- Shift+Right / Shift+Left: Jump to next/previous cut marker",
            "    (start/end points shown in the timeline strip)",
            "- Up Arrow: Fast Forward (60 frames)",
            "- Down Arrow: Rewind (60 frames)",
            "- G: Go to Frame Number (enter frame number as int)",
            "- T: Go to Time (enter time in seconds as float)",
            "- 0: Auto-fit window to screen",
            "- + or =: Zoom In",
            "- -: Zoom Out",
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
            "- F: Load Sync File or Cuts TOML File (cuts only; same as before)",
            "- V or N: Load per-cut output .mp4 names from CSV/TXT (one name per line)",
            "     Example (3 cuts): CarlosMiguel_cod_02 / Coutinho_cod_02 / Heittor_cod_02",
            "     → CarlosMiguel_cod_02.mp4, Coutinho_cod_02.mp4, … (or 'Cut names' button)",
            "     Does not change cuts or TOML load; only renames exported video files.",
            "- C: Load cut labels only (metadata in TOML; does not rename output files)",
            "- B: Set a single output base name for all cuts (or 'Base name' button)",
            "- I or P: Input Manual FPS",
            "- ESC: Save cuts to TOML file and optionally generate videos",
            "",
            "Help:",
            "- H: Show this help dialog",
            "",
            "Mouse Controls:",
            "- Click or drag cut strip: Jump to cut markers (green start / orange end)",
            "- Yellow strip marker: Pending start selected with S",
            "- Click on slider: Jump to frame",
            "- Click 'Loop' / 'Help' / 'Cut names' / 'Base name' (row below the timeline):",
            "  Loop, Help, per-cut name list (CSV/TXT), single Base name",
            "- Mouse Wheel (video area): Zoom in/out",
            "- Middle mouse drag (video area): Pan when zoomed",
            "- Mouse Wheel in this help: Scroll help text",
            "- Arrow Up/Down: Scroll help text",
            "- Drag window edges: Resize window",
            "",
            "Display Features:",
            "- Cut strip: blue ranges, green starts, orange ends, yellow pending start",
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
                elif event.type in _CLOSE_EVENTS:
                    waiting_for_input = False
                    nonlocal running
                    running = False

    def prompt_output_basename_dialog():
        """Ask user for custom output file prefix (Tk dialog; pygame display paused)."""
        nonlocal output_basename
        stem = Path(video_path).stem
        current = output_basename if output_basename is not None else ""
        pygame.display.quit()
        root_bn = Tk()
        root_bn.withdraw()
        new_val = simpledialog.askstring(
            "Output base name",
            "Base name for cut output files (prefix before _frame_X_to_Y.mp4):\n\n"
            f"Leave empty to use video name: {stem}\n"
            "Example: trial01 → trial01_frame_10_to_20.mp4",
            initialvalue=current,
        )
        root_bn.destroy()
        screen_local = restore_pygame_display()
        pygame.display.flip()
        if new_val is None:
            return screen_local
        if not new_val.strip():
            output_basename = None
            print(f"Output base name reset to video stem: {stem}")
        else:
            safe = sanitize_output_basename(new_val)
            if not safe:
                msg_root = Tk()
                msg_root.withdraw()
                messagebox.showwarning(
                    "Invalid base name",
                    "Use letters, numbers, underscore, or hyphen only.",
                    parent=msg_root,
                )
                msg_root.destroy()
            else:
                output_basename = safe
                print(f"Output base name set: {safe}")
        return screen_local

    def prompt_output_basenames_csv():
        """Load a CSV/TXT list of per-cut output base names (pygame display paused).

        Each line/field becomes the base name of the corresponding cut, in order:
        cut 1 -> name[0], cut 2 -> name[1], ... Files are then written as
        ``<name>.mp4`` instead of ``<video>_frame_X_to_Y.mp4``.
        """
        nonlocal output_basenames
        pygame.display.quit()
        root_csv = Tk()
        root_csv.withdraw()
        csv_file = filedialog.askopenfilename(
            title="Per-cut output names (one per line → name.mp4)",
            filetypes=[
                ("CSV files", "*.csv"),
                ("TXT files", "*.txt"),
                ("All files", "*.*"),
            ],
            parent=root_csv,
        )
        names: list[str] = []
        load_error = None
        if csv_file:
            try:
                names = parse_basename_list(csv_file)
            except Exception as exc:  # noqa: BLE001 - surfaced to the user below
                load_error = str(exc)
        root_csv.destroy()
        screen_local = restore_pygame_display()
        pygame.display.flip()

        if not csv_file:
            return screen_local

        msg_root = Tk()
        msg_root.withdraw()
        if load_error is not None:
            messagebox.showerror(
                "Base-name list", f"Could not read file:\n{load_error}", parent=msg_root
            )
        elif not names:
            messagebox.showwarning(
                "Base-name list", "No usable base names found in the file.", parent=msg_root
            )
        else:
            output_basenames = names
            n_cuts = len(cuts)
            extra = ""
            if n_cuts and len(names) < n_cuts:
                extra = (
                    f"\n\nNote: {len(names)} name(s) for {n_cuts} cut(s); "
                    "remaining cuts use the default name."
                )
            elif n_cuts and len(names) > n_cuts:
                extra = f"\n\nNote: {len(names)} name(s) for {n_cuts} cut(s); extras are ignored."
            preview = ", ".join(names[:10]) + (" …" if len(names) > 10 else "")
            messagebox.showinfo(
                "Cut names loaded",
                f"Loaded {len(names)} name(s) for output files:\n{preview}{extra}",
                parent=msg_root,
            )
            print(f"Per-cut base names loaded ({len(names)}): {names}")
        msg_root.destroy()
        return screen_local

    def save_and_generate_videos():
        nonlocal cuts, video_path, using_sync_file, sync_data, fps, cut_labels
        from tkinter import messagebox

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
        per_cut_files = build_cut_output_filenames(
            video_path, cuts, output_basename, output_basenames
        )

        save_cuts_to_toml(
            video_path,
            cuts,
            fps,
            output_dir=planned_output_dir,
            per_cut_outputs=per_cut_files,
            labels=cut_labels,
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
        batch_cancelled = False

        def request_batch_cancel():
            nonlocal batch_cancelled
            batch_cancelled = True
            status_label.config(text="Cancelling batch...")

        def refresh_batch_progress():
            root.update_idletasks()
            root.update()
            return not batch_cancelled

        root.protocol("WM_DELETE_WINDOW", request_batch_cancel)
        ttk.Button(root, text="Cancel", command=request_batch_cancel).pack()

        def process_next_video():
            nonlocal processed_count

            if batch_cancelled:
                status_label.config(text="Batch processing cancelled")
                root.after(300, root.destroy)
                return
            if processed_count < len(video_files):
                video_path = str(video_files[processed_count])
                video_name = Path(video_path).stem

                # Use the FPS chosen in the cut UI (may differ from container metadata).
                save_cuts_to_toml(video_path, cuts, fps)

                status_label.config(text=f"Processing: {video_name}")
                print(f"\n--- Video {processed_count + 1}/{len(video_files)}: {video_name} ---")

                try:
                    # Get precise video metadata
                    metadata = get_precise_video_metadata(video_path)
                    total_frames = metadata.get("nb_frames") or (
                        int(metadata.get("duration", 0) * metadata["fps"])
                        if metadata.get("duration")
                        else int(cv2.VideoCapture(video_path).get(cv2.CAP_PROP_FRAME_COUNT))
                    )

                    # Process each cut
                    valid_cuts = [(i, s, e) for i, (s, e) in enumerate(cuts) if s < total_frames]
                    for idx, (i, start_frame, end_frame) in enumerate(valid_cuts):
                        actual_end_frame = min(end_frame, total_frames - 1)
                        output_path = (
                            output_dir
                            / f"{video_name}_frame_{start_frame}_to_{actual_end_frame}.mp4"
                        )
                        print(
                            f"  Cut {idx + 1}/{len(valid_cuts)}: frames {start_frame}-{actual_end_frame} -> {output_path.name}"
                        )
                        success = cut_video_with_ffmpeg(
                            video_path,
                            output_path,
                            start_frame,
                            actual_end_frame,
                            metadata,
                            progress_callback=refresh_batch_progress,
                        )
                        if batch_cancelled:
                            break
                        if not success:
                            status_label.config(
                                text=f"Warning: Cut {i + 1} failed for {video_name}"
                            )
                            print(f"  Warning: Cut {i + 1} failed")
                        else:
                            print(f"  Done: {output_path.name}")

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

        if batch_cancelled:
            messagebox.showinfo("Batch Cancelled", f"Processed {processed_count} videos.")
        else:
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

        # Resolve one output filename per cut (per-cut CSV names take priority)
        out_names = build_cut_output_filenames(video_path, cuts, output_basename, output_basenames)
        naming_mode = (
            "per-cut CSV names"
            if output_basenames
            else effective_cut_basename(video_path, output_basename)
        )
        print(f"Saving {len(cuts)} cut(s) to {output_dir} (naming: {naming_mode})")
        # Get precise video metadata
        metadata = get_precise_video_metadata(video_path)

        # Process each cut with visible progress after the pygame display closes.
        n_cuts = len(cuts)
        progress_dialog = RenderProgressDialog("Generating cut videos", n_cuts)
        try:
            for i, (start_frame, end_frame) in enumerate(cuts):
                output_path = output_dir / out_names[i]
                status = f"Rendering cut {i + 1}/{n_cuts}: {output_path.name}"
                if not progress_dialog.update(status, i):
                    print("Video cut generation cancelled by user")
                    return False
                print(
                    f"Processing cut {i + 1}/{n_cuts}: frames {start_frame + 1}-{end_frame + 1} -> {output_path.name}"
                )
                success = cut_video_with_ffmpeg(
                    video_path,
                    output_path,
                    start_frame,
                    end_frame,
                    metadata,
                    progress_callback=progress_dialog.update,
                )
                if progress_dialog.cancelled:
                    print("Video cut generation cancelled by user")
                    return False
                if not success:
                    print(f"Warning: Failed to create cut {i + 1} for {video_name}")
                else:
                    print(f"  Done: {output_path.name}")
                progress_dialog.update(completed_steps=i + 1)
        finally:
            progress_dialog.close()

        return True

    running = True
    while running:
        # Handle title-bar X early (Linux SDL2 often sends WINDOWCLOSE, not QUIT).
        for close_event in pygame.event.get(_CLOSE_EVENTS):
            if close_event.type in _CLOSE_EVENTS:
                running = False
                break
        if not running:
            break

        if paused:
            # Quando pausado, vamos usar o método set para posicionar no frame exato
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_count)
            ret, frame = cap.read()
        else:
            if playback_speed >= 1.0:
                skip = int(playback_speed)
                for _ in range(skip):
                    ret, frame = cap.read()
                    if not ret:
                        break
                if ret:
                    frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
            else:
                slow_mo_accumulator += playback_speed
                if slow_mo_accumulator >= 1.0:
                    slow_mo_accumulator -= 1.0
                    ret, frame = cap.read()
                    if ret:
                        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1
                elif last_valid_frame is not None:
                    frame = last_valid_frame.copy()
                    ret = True
                else:
                    ret, frame = cap.read()
                    if ret:
                        frame_count = int(cap.get(cv2.CAP_PROP_POS_FRAMES)) - 1

            if not ret:
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

        last_valid_frame = frame.copy()
        frame = check_and_rotate_frame(frame, metadata)

        # Base scale so that at zoom_level=1.0 the whole video fits in the window
        clamp_zoom_level()
        scale_fit = min(window_width / original_width, window_height / original_height)
        zoomed_width = max(1, int(original_width * scale_fit * zoom_level))
        zoomed_height = max(1, int(original_height * scale_fit * zoom_level))
        clamp_pan_offsets()
        zoomed_frame = cv2.resize(frame, (zoomed_width, zoomed_height))
        crop_x = int(offset_x)
        crop_y = int(offset_y)
        visible_w = max(1, min(window_width, zoomed_width - crop_x))
        visible_h = max(1, min(window_height, zoomed_height - crop_y))
        cropped_frame = zoomed_frame[crop_y : crop_y + visible_h, crop_x : crop_x + visible_w]
        frame_surface = pygame.surfarray.make_surface(
            cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB).swapaxes(0, 1)
        )

        screen.fill((0, 0, 0))
        screen.blit(frame_surface, (0, 0))
        _draw_playback_speed_hud(screen, playback_speed, window_width)

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
            cut_timeline_rect,
            help_button_rect,
            loop_button_rect,
            base_button_rect,
            names_button_rect,
            base_y,
        ) = draw_controls()
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type in _CLOSE_EVENTS:
                running = False

            elif event.type == pygame.VIDEORESIZE:
                new_w, new_h = event.w, event.h
                min_total_h = control_height + (audio_height if show_audio else 0) + 100
                if new_h > min_total_h:
                    requested_video_h = new_h - control_height - (audio_height if show_audio else 0)
                    if requested_video_h <= 0:
                        continue

                    # Keep the window manager requested geometry to avoid resize feedback loops.
                    next_window_width = max(1, int(new_w))
                    next_window_height = max(1, int(requested_video_h))

                    if next_window_width != window_width or next_window_height != window_height:
                        window_width = next_window_width
                        window_height = next_window_height
                        clamp_zoom_level()
                        screen = set_display_mode()
                        clamp_pan_offsets()

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
                    reset_zoom_pan()
                    screen = auto_fit_window()
                    clamp_zoom_level()
                    clamp_pan_offsets()
                    update_caption()
                elif event.key in (pygame.K_PLUS, pygame.K_EQUALS, pygame.K_KP_PLUS):
                    zoom_level = min(10.0, zoom_level * 1.2)
                    clamp_zoom_level()
                    clamp_pan_offsets()
                elif event.key in (pygame.K_MINUS, pygame.K_KP_MINUS):
                    zoom_level = max(0.1, zoom_level / 1.2)
                    clamp_zoom_level()
                    clamp_pan_offsets()
                elif event.key == pygame.K_RIGHTBRACKET:  # ]
                    playback_speed = _step_playback_speed(playback_speed, 1)
                    slow_mo_accumulator = 0.0
                    print(f"Playback speed: {_format_playback_speed_label(playback_speed)}×")
                elif event.key == pygame.K_LEFTBRACKET:  # [
                    playback_speed = _step_playback_speed(playback_speed, -1)
                    slow_mo_accumulator = 0.0
                    print(f"Playback speed: {_format_playback_speed_label(playback_speed)}×")
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
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        next_marker = adjacent_cut_marker_frame(
                            frame_count, get_all_cut_markers(), 1
                        )
                        if next_marker is not None:
                            frame_count = next_marker
                            print(f"Jumped to next marker: Frame {next_marker + 1}")
                        else:
                            print("No markers available")
                    else:
                        frame_count = min(frame_count + 1, total_frames - 1)
                elif event.key == pygame.K_LEFT and paused:
                    if pygame.key.get_mods() & pygame.KMOD_SHIFT:
                        previous_marker = adjacent_cut_marker_frame(
                            frame_count, get_all_cut_markers(), -1
                        )
                        if previous_marker is not None:
                            frame_count = previous_marker
                            print(f"Jumped to previous marker: Frame {previous_marker + 1}")
                        else:
                            print("No markers available")
                    else:
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
                        planned = build_cut_output_filenames(
                            video_path, cuts, output_basename, output_basenames
                        )
                        cuts_list_strs = []
                        for i, (start, end) in enumerate(cuts):
                            label_str = f" [{cut_labels[i]}]" if i < len(cut_labels) else ""
                            out_file = planned[i] if i < len(planned) else "?"
                            cuts_list_strs.append(
                                f"Cut {i + 1}{label_str}: Frame {start + 1} to {end + 1}"
                                f" → {out_file}"
                            )
                        messagebox.showinfo("Cuts List", "\n".join(cuts_list_strs))
                    else:
                        messagebox.showinfo("Cuts List", "No cuts marked yet")
                elif event.key == pygame.K_c:  # Load cut labels from CSV
                    from tkinter import Tk, filedialog, messagebox

                    temp_root = Tk()
                    temp_root.withdraw()
                    csv_file = filedialog.askopenfilename(
                        title="Select Labels CSV/TXT",
                        filetypes=[
                            ("CSV files", "*.csv"),
                            ("TXT files", "*.txt"),
                            ("All files", "*.*"),
                        ],
                    )
                    temp_root.destroy()
                    if csv_file:
                        try:
                            parsed_labels = []
                            with open(csv_file, encoding="utf-8") as f:
                                # Simple parsing: comma separated inline, or newline separated
                                content = f.read().strip()
                                if "," in content and "\n" not in content:
                                    parsed_labels = [label.strip() for label in content.split(",")]
                                else:
                                    parsed_labels = [
                                        line.strip().strip(",")
                                        for line in content.splitlines()
                                        if line.strip()
                                    ]

                            cut_labels.clear()
                            cut_labels.extend(parsed_labels)

                            info_root = Tk()
                            info_root.withdraw()
                            messagebox.showinfo(
                                "Cut Labels Loaded",
                                f"Successfully loaded {len(cut_labels)} labels:\n"
                                + ", ".join(cut_labels),
                                parent=info_root,
                            )
                            info_root.destroy()
                            print(f"Cut labels loaded: {cut_labels}")
                        except Exception as e:
                            print(f"Error loading cut labels: {e}")
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
                    new_fps_str = simpledialog.askstring(
                        "Input FPS",
                        f"Enter new FPS value (current: {fps:.6f}):\n(You can enter a float or a fraction like 60000/1001)",
                        initialvalue=str(fps),
                    )
                    root_fps.destroy()
                    screen = restore_pygame_display()
                    if new_fps_str:
                        try:
                            val = None
                            if "/" in new_fps_str:
                                num, den = map(int, new_fps_str.split("/"))
                                if den != 0:
                                    val = float(num) / den
                                    fps_num, fps_den = num, den
                            else:
                                val = float(new_fps_str)
                                fps_num, fps_den = int(val * 1000), 1000

                            if val is not None and val > 0:
                                fps = val
                                original_fps = fps  # Force exact matching to get_time_s
                                print(f"FPS updated to: {fps:.6f} ({fps_num}/{fps_den})")
                                update_caption()

                                # Spawn temporary root for messagebox to ensure it closes properly on Linux
                                msg_root = Tk()
                                msg_root.withdraw()
                                messagebox.showinfo(
                                    "FPS Updated",
                                    f"FPS set to {fps:.6f} ({fps_num}/{fps_den})",
                                    parent=msg_root,
                                )
                                msg_root.destroy()

                            else:
                                print("Invalid FPS value entered.")
                        except ValueError:
                            print("Invalid FPS format entered.")
                    else:
                        # Keep original caption if FPS was not updated
                        update_caption()
                    pygame.display.flip()
                elif event.key == pygame.K_h:  # Show help dialog
                    show_help_dialog()
                elif event.key == pygame.K_b:  # Set output base name for cut files
                    screen = prompt_output_basename_dialog()
                elif event.key in (pygame.K_n, pygame.K_v):  # Per-cut output names (CSV/TXT)
                    screen = prompt_output_basenames_csv()
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
                    screen = restore_pygame_display()
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
                    current_time = get_time_s(frame_count, fps)
                    max_time = get_time_s(total_frames, fps)
                    target_time = simpledialog.askfloat(
                        "Go to Time",
                        f"Enter time in seconds (current: {current_time:.2f}s, max: {max_time:.2f}s):",
                        initialvalue=current_time,
                        minvalue=0.0,
                        maxvalue=max_time,
                    )
                    root_time.destroy()
                    screen = restore_pygame_display()
                    if target_time is not None and fps > 0:
                        # Convert time to frame (0-based)
                        if fps == original_fps and fps_num and fps_den:
                            target_frame_float = (target_time * fps_num) / fps_den
                        else:
                            target_frame_float = target_time * fps
                        frame_count = min(max(0, int(round(target_frame_float))), total_frames - 1)
                        actual_time = get_time_s(frame_count, fps)
                        print(f"Jumped to time: {actual_time:.2f}s (frame: {frame_count + 1})")
                        paused = True  # Pause when jumping to time
                    pygame.display.flip()
            elif event.type == pygame.MOUSEBUTTONDOWN:
                x, y = event.pos
                if event.button == 2 and y < window_height:
                    panning = True
                    pygame.mouse.get_rel()
                elif event.button == 1:
                    if help_button_rect.collidepoint(x, y - base_y):
                        show_help_dialog()
                    elif base_button_rect.collidepoint(x, y - base_y):
                        screen = prompt_output_basename_dialog()
                    elif names_button_rect.collidepoint(x, y - base_y):
                        screen = prompt_output_basenames_csv()
                    elif loop_button_rect.collidepoint(x, y - base_y):
                        loop_enabled = not loop_enabled
                        if not loop_enabled:
                            stop_audio_playback()
                        screen = set_display_mode()
                        clamp_pan_offsets()
                        update_caption()
                    elif cut_timeline_rect.collidepoint(x, y - base_y):
                        dragging_cut_timeline = True
                        markers = get_all_cut_markers()
                        if current_start is not None:
                            markers.append(current_start)
                        frame_count = frame_index_from_cut_timeline_x(
                            mouse_x=x,
                            strip_left=slider_x,
                            strip_width=slider_width,
                            total_frames=total_frames,
                            cut_markers=markers,
                        )
                        paused = True
                    elif slider_y <= y - base_y <= slider_y + slider_height:
                        frame_count = frame_index_from_cut_timeline_x(
                            mouse_x=x,
                            strip_left=slider_x,
                            strip_width=slider_width,
                            total_frames=total_frames,
                            cut_markers=[],
                        )
                        paused = True
            elif event.type == pygame.MOUSEBUTTONUP:
                if event.button == 1:
                    dragging_cut_timeline = False
                elif event.button == 2:
                    panning = False
            elif event.type == pygame.MOUSEMOTION:
                if dragging_cut_timeline:
                    markers = get_all_cut_markers()
                    if current_start is not None:
                        markers.append(current_start)
                    frame_count = frame_index_from_cut_timeline_x(
                        mouse_x=event.pos[0],
                        strip_left=slider_x,
                        strip_width=slider_width,
                        total_frames=total_frames,
                        cut_markers=markers,
                    )
                    paused = True
                elif panning:
                    rel_dx, rel_dy = pygame.mouse.get_rel()
                    offset_x -= rel_dx
                    offset_y -= rel_dy
                    clamp_pan_offsets()
            elif event.type == pygame.MOUSEWHEEL:
                mx, my = pygame.mouse.get_pos()
                if my < window_height and event.y != 0:
                    scale_fit = min(window_width / original_width, window_height / original_height)
                    old_zoom = zoom_level
                    zoom_factor = 1.1
                    if event.y > 0:
                        zoom_level = min(10.0, zoom_level * zoom_factor)
                    else:
                        zoom_level = max(0.1, zoom_level / zoom_factor)
                    clamp_zoom_level()

                    if zoom_level != old_zoom:
                        old_effective = scale_fit * old_zoom
                        new_effective = scale_fit * zoom_level
                        target_vx = (mx + offset_x) / old_effective
                        target_vy = (my + offset_y) / old_effective
                        offset_x = (target_vx * new_effective) - mx
                        offset_y = (target_vy * new_effective) - my
                        clamp_pan_offsets()

        if paused:
            # Se pausado, não limitamos a taxa de FPS para que a interface seja responsiva
            clock.tick(60)  # Taxa de atualização da interface
        else:
            # Se em reprodução, limitamos à taxa de FPS do vídeo
            clock.tick(fps)

    cap.release()
    stop_audio_playback()
    if audio_ready:
        with contextlib.suppress(Exception):
            pygame.mixer.quit()
    if audio_wave_path:
        with contextlib.suppress(Exception):
            os.remove(audio_wave_path)
    pygame.quit()


def get_video_path():
    from tkinter import Tk, filedialog

    root = Tk()
    root.withdraw()
    try:
        return filedialog.askopenfilename(
            title="Select Video File",
            filetypes=[("Video Files", "*.mp4 *.MP4 *.avi *.AVI *.mov *.MOV *.mkv *.MKV")],
        )
    finally:
        root.destroy()


def cleanup_resources():
    """Ensure all resources are properly released without killing the main process."""
    # Avoid opening camera (VideoCapture(0)): on machines without a camera or with
    # no permission to /dev/video*, OpenCV would log V4L2/FFMPEG/obsensor errors.
    # Closing pygame display is sufficient for cleanup here.

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

    import platform

    if platform.system() == "Linux":
        has_nvidia = os.path.exists("/proc/driver/nvidia")
        if has_nvidia:
            print("NVIDIA GPU detected, applying OpenGL compatibility settings")

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
