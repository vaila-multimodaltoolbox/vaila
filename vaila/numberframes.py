"""
numberframes.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 10 October 2024
Update Date: 27 October 2026
Version: 0.4.12
Python Version: 3.12.14

Description:
------------
This script allows users to analyze video files within a selected directory and
extract metadata such as frame count, frame rate (FPS), resolution, codec, and
duration. It generates a summary of this information, displays it in a
graphical interface, and saves the metadata to text files.

It also retags container FPS (Adjust real FPS) so Cut Video / Make Sync file
read the capture rate from the file without per-session overrides.

Key Features:
-------------
1. Robust metadata extraction using `ffprobe` (via `get_precise_video_metadata`)
   with a fallback to OpenCV.
2. Accurate handling of fractional frame rates and precise duration calculations.
3. Detection of capture FPS via Android tag com.android.capture.fps.
4. Parallel processing of multiple videos for faster analysis.
5. Adjust real FPS: remux (stream copy) with re-encode fallback; writes
   ``processed_fpsfix_<timestamp>/`` + ``fpsfix_summary.csv``.

Notes:
------
- The script uses `ffprobe` as the primary source for metadata to ensure specificities
  like fractional FPS are captured correctly.
- OpenCV is used as a fallback if `ffprobe` fails.
- Cut Video and syncvid use container FPS; fix stamped rates here before cut/sync.

Usage:
------
- Run the script, select a directory containing video files, and let the tool analyze
  the videos.
- View the metadata in the GUI and check the saved text files in the selected directory
  for details.
- Use the "full" file for complete metadata in JSON format.
- Adjust real FPS: ``python numberframes.py --set-fps 29.97 --dir /path/to/videos``

License:
--------
This program is licensed under the GNU Lesser General Public License v3.0.
For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
===============================================================================
"""

from __future__ import annotations

import contextlib
import csv
import json
import os
import shlex
import subprocess
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from fractions import Fraction
from pathlib import Path
from typing import Any

from rich import print

try:
    from .ffmpeg_utils import (
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_video_encode_ffmpeg_path,
        run_ffmpeg_encode_with_fallback,
    )
except ImportError:
    from ffmpeg_utils import (  # ty: ignore[unresolved-import]
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_video_encode_ffmpeg_path,
        run_ffmpeg_encode_with_fallback,
    )

VIDEO_EXTENSIONS = (".mp4", ".avi", ".mov", ".mkv", ".m4v", ".webm")
FPS_MATCH_ABS = 0.05
FPS_MATCH_REL = 1e-3


def parse_fps_fraction(text: str) -> Fraction:
    """Parse a positive FPS from a float or ``num/den`` string as a Fraction."""
    raw = (text or "").strip().replace(",", ".")
    if not raw:
        raise ValueError("FPS value is empty")
    try:
        if "/" in raw:
            num_s, den_s = raw.split("/", 1)
            num = Fraction(num_s.strip())
            den = Fraction(den_s.strip())
            if den == 0:
                raise ValueError("FPS denominator cannot be zero")
            rate = num / den
        else:
            rate = Fraction(raw).limit_denominator(100_000)
    except (ValueError, ZeroDivisionError) as exc:
        raise ValueError(f"Invalid FPS value: {text!r}") from exc
    if not 0 < float(rate) < 10_000.0:
        raise ValueError(f"FPS must be between 0 and 10000 (got {float(rate)})")
    return Fraction(rate)


def parse_fps_value(text: str) -> float:
    """Parse a positive FPS from a float or ``num/den`` fraction string."""
    return float(parse_fps_fraction(text))


def fps_rates_match(a: float, b: float) -> bool:
    """True when two FPS rates are close enough for container retag checks."""
    return abs(float(a) - float(b)) <= max(FPS_MATCH_ABS, FPS_MATCH_REL * max(abs(a), abs(b), 1.0))


def fps_to_rational(fps: float | Fraction | str, *, max_den: int = 1001) -> tuple[int, int]:
    """Map FPS to a reduced ``num/den`` pair for ffmpeg timescale/setts."""
    if isinstance(fps, str):
        frac = parse_fps_fraction(fps)
    elif isinstance(fps, Fraction):
        frac = fps
    else:
        rate = float(fps)
        if not 0.0 < rate < 10_000.0:
            raise ValueError(f"FPS must be between 0 and 10000 (got {rate})")
        frac = Fraction(rate).limit_denominator(max_den)
    frac = Fraction(frac).limit_denominator(max(max_den, frac.denominator))
    if frac.numerator <= 0 or frac.denominator <= 0:
        raise ValueError(f"Could not rationalize FPS {fps}")
    return int(frac.numerator), int(frac.denominator)


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
            "codec_long": video_stream.get("codec_long_name", "unknown"),
            "container": data.get("format", {}).get("format_name", "unknown"),
            "container_long": data.get("format", {}).get("format_long_name", "unknown"),
            "r_frame_rate": r_frame_rate_str,
            "avg_frame_rate": avg_frame_rate_str,
            "duration": duration if duration > 0 else None,
            "nb_frames": nb_frames,
            "rotation": rotation,
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
            if not f_str or f_str == "0/0":
                return None
            try:
                if "/" in f_str:
                    n, d = map(int, f_str.split("/"))
                    return float(n) / d if d else 0.0
                return float(f_str)
            except Exception:
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


def _list_video_paths(directory_path: str | None, video_files: list[str] | None) -> list[str]:
    """Resolve absolute video paths from a directory and/or explicit file list."""
    if video_files:
        abs_paths = [os.path.abspath(f) for f in video_files]
        return abs_paths
    if not directory_path:
        return []
    names = sorted(f for f in os.listdir(directory_path) if f.lower().endswith(VIDEO_EXTENSIONS))
    return [os.path.join(directory_path, f) for f in names]


def _fpsfix_comment(source_name: str, source_fps: float, target_fps: float, method: str) -> str:
    return (
        f"vaila numberframes fpsfix: source={source_name} "
        f"source_fps={source_fps:.6f} target_fps={target_fps:.6f} method={method}"
    )


def _run_ffmpeg(cmd: list[str]) -> tuple[bool, str]:
    """Run ffmpeg; return (ok, stderr_tail)."""
    result = subprocess.run(cmd, capture_output=True, text=True, check=False)
    err = (result.stderr or result.stdout or "").strip()
    if result.returncode != 0:
        return False, err[-2000:] if err else f"ffmpeg exit {result.returncode}"
    return True, err[-500:] if err else ""


def _fraction_to_float(frac_str: str | None) -> float | None:
    if not frac_str or frac_str == "0/0":
        return None
    try:
        if "/" in frac_str:
            num, den = map(int, frac_str.split("/", 1))
            return float(num) / den if den else None
        return float(frac_str)
    except (ValueError, ZeroDivisionError):
        return None


def _verify_fpsfix(
    out_path: Path,
    *,
    target_fps: float,
    frames_in: int | None,
) -> tuple[bool, dict[str, Any]]:
    """Probe output; require matching FPS and (when known) unchanged frame count."""
    meta = get_precise_video_metadata(out_path)
    if not meta:
        return False, {"error": "ffprobe failed on output"}
    r_fps = _fraction_to_float(meta.get("r_frame_rate"))
    avg_fps = _fraction_to_float(meta.get("avg_frame_rate"))
    out_fps = float(r_fps or avg_fps or meta.get("fps") or 0.0)
    frames_out = meta.get("nb_frames")
    duration_out = meta.get("duration")
    ok_fps = fps_rates_match(out_fps, target_fps) or (
        avg_fps is not None and fps_rates_match(avg_fps, target_fps)
    )
    ok_frames = True
    if frames_in is not None and frames_out is not None:
        ok_frames = int(frames_out) == int(frames_in)
    return ok_fps and ok_frames, {
        "fps_out": out_fps,
        "frames_out": frames_out,
        "duration_out": duration_out,
        "ok_fps": ok_fps,
        "ok_frames": ok_frames,
    }


def build_fpsfix_copy_cmd(
    src: Path,
    out_path: Path,
    real_fps: float | Fraction | str,
    *,
    comment: str,
) -> list[str]:
    """Build stream-copy remux that rewrites PTS to ``real_fps`` (no pixel re-encode).

    Uses the ``setts`` bitstream filter. For MP4/MOV also sets
    ``-video_track_timescale`` so avg/r rates and OpenCV agree.
    """
    num, den = fps_to_rational(real_fps)
    fps_expr = f"({num}/{den})" if den != 1 else str(num)
    cmd = [
        get_ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-map",
        "0",
        "-c",
        "copy",
        "-bsf:v",
        f"setts=ts=N/TB/{fps_expr}:duration=1/TB/{fps_expr}",
        "-metadata",
        f"comment={comment}",
    ]
    if src.suffix.lower() in {".mp4", ".mov", ".m4v"} or out_path.suffix.lower() in {
        ".mp4",
        ".mov",
        ".m4v",
    }:
        cmd.extend(["-video_track_timescale", str(num)])
        if out_path.suffix.lower() in {".mp4", ".m4v", ".mov"}:
            cmd.extend(["-movflags", "+faststart"])
    cmd.append(str(out_path))
    return cmd


def run_fpsfix_reencode(
    src: Path,
    out_path: Path,
    real_fps: float | Fraction | str,
    *,
    comment: str,
) -> tuple[bool, str]:
    """Re-encode fallback that rewrites PTS at ``real_fps`` (GPU then CPU)."""
    num, den = fps_to_rational(real_fps)
    rate = float(num) / float(den)
    fps_s = f"{num}/{den}" if den != 1 else str(num)
    fps_float_s = f"{rate:.6f}".rstrip("0").rstrip(".")
    prefix = [
        get_video_encode_ffmpeg_path(),
        "-y",
        "-hide_banner",
        "-loglevel",
        "error",
        "-i",
        str(src),
        "-vf",
        f"setpts=N/({fps_float_s}*TB)",
        "-r",
        fps_s,
    ]
    suffix = [
        "-c:a",
        "copy",
        "-metadata",
        f"comment={comment}",
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    try:
        run_ffmpeg_encode_with_fallback(prefix, suffix)
        return True, ""
    except (subprocess.CalledProcessError, RuntimeError, OSError) as exc:
        # Last resort: force libx264 args without hardware path.
        cmd = [
            get_ffmpeg_path(),
            "-y",
            "-hide_banner",
            "-loglevel",
            "error",
            "-i",
            str(src),
            "-vf",
            f"setpts=N/({fps_float_s}*TB)",
            "-r",
            fps_s,
            *get_ffmpeg_video_encoding_args("libx264"),
            "-c:a",
            "copy",
            "-metadata",
            f"comment={comment}",
            "-movflags",
            "+faststart",
            str(out_path),
        ]
        ok, err = _run_ffmpeg(cmd)
        if ok:
            return True, ""
        return False, err or str(exc)


def adjust_video_fps(
    src: str | Path,
    real_fps: float | Fraction | str,
    out_path: str | Path,
) -> dict[str, Any]:
    """Retag one video to ``real_fps`` (stream copy, then re-encode fallback).

    Keeps every frame; reported duration becomes ``nb_frames / real_fps``.
    """
    src_path = Path(src)
    dest = Path(out_path)
    try:
        if isinstance(real_fps, str):
            rate_frac = parse_fps_fraction(real_fps)
        elif isinstance(real_fps, Fraction):
            rate_frac = real_fps
        else:
            rate_frac = Fraction(float(real_fps)).limit_denominator(1001)
        rate = float(rate_frac)
        if not 0.0 < rate < 10_000.0:
            raise ValueError(f"Invalid real FPS: {rate}")
    except (ValueError, ZeroDivisionError, TypeError) as exc:
        return {
            "file_name": src_path.name,
            "ok": False,
            "error": f"Invalid real FPS: {exc}",
            "method": None,
        }

    dest.parent.mkdir(parents=True, exist_ok=True)
    meta_in = get_precise_video_metadata(src_path) or {}
    fps_in = float(meta_in.get("fps") or 0.0)
    frames_in = meta_in.get("nb_frames")
    duration_in = meta_in.get("duration")

    row: dict[str, Any] = {
        "file_name": src_path.name,
        "source_path": str(src_path),
        "output_path": str(dest),
        "fps_in": fps_in if fps_in > 0 else None,
        "fps_out": None,
        "frames_in": frames_in,
        "frames_out": None,
        "duration_in": duration_in,
        "duration_out": None,
        "method": None,
        "ok": False,
        "error": None,
    }

    if fps_in > 0 and fps_rates_match(fps_in, rate):
        # Already correct: still write a copy so the batch folder is complete.
        comment = _fpsfix_comment(src_path.name, fps_in, rate, "copy")
        cmd = build_fpsfix_copy_cmd(src_path, dest, rate_frac, comment=comment)
        ok, err = _run_ffmpeg(cmd)
        if not ok:
            row["error"] = err
            return row
        verified, probe = _verify_fpsfix(dest, target_fps=rate, frames_in=frames_in)
        row.update(
            method="copy",
            ok=verified,
            fps_out=probe.get("fps_out"),
            frames_out=probe.get("frames_out"),
            duration_out=probe.get("duration_out"),
            error=None if verified else "Output FPS/frames did not match after copy",
        )
        return row

    comment_copy = _fpsfix_comment(src_path.name, fps_in or 0.0, rate, "copy")
    cmd_copy = build_fpsfix_copy_cmd(src_path, dest, rate_frac, comment=comment_copy)
    ok, err = _run_ffmpeg(cmd_copy)
    if ok:
        verified, probe = _verify_fpsfix(dest, target_fps=rate, frames_in=frames_in)
        if verified:
            row.update(
                method="copy",
                ok=True,
                fps_out=probe.get("fps_out"),
                frames_out=probe.get("frames_out"),
                duration_out=probe.get("duration_out"),
            )
            return row
        _unlink_quiet(dest)
        copy_fail_reason = (
            f"copy verification failed (fps_ok={probe.get('ok_fps')}, "
            f"frames_ok={probe.get('ok_frames')})"
        )
    else:
        copy_fail_reason = err
        _unlink_quiet(dest)

    comment_re = _fpsfix_comment(src_path.name, fps_in or 0.0, rate, "reencode")
    ok, err = run_fpsfix_reencode(src_path, dest, rate_frac, comment=comment_re)
    if not ok:
        row["error"] = f"copy failed ({copy_fail_reason}); reencode failed ({err})"
        _unlink_quiet(dest)
        return row

    # Re-encode may alter nb_frames slightly; require FPS match only.
    verified, probe = _verify_fpsfix(dest, target_fps=rate, frames_in=None)
    row.update(
        method="reencode",
        ok=bool(probe.get("ok_fps")),
        fps_out=probe.get("fps_out"),
        frames_out=probe.get("frames_out"),
        duration_out=probe.get("duration_out"),
        error=None
        if probe.get("ok_fps")
        else f"reencode verification failed after copy failure ({copy_fail_reason})",
    )
    if not row["ok"]:
        _unlink_quiet(dest)
    return row


def _unlink_quiet(path: Path) -> None:
    """Best-effort remove of a failed output file."""
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


def _format_fpsfix_cli(
    *,
    real_fps: float,
    directory_path: str | None,
    video_files: list[str] | None,
) -> str:
    fps_s = f"{real_fps:.6g}"
    parts = [f"uv run vaila/numberframes.py --set-fps {shlex.quote(fps_s)}"]
    if directory_path:
        parts.append(f"--dir {shlex.quote(str(directory_path))}")
    if video_files:
        parts.extend(shlex.quote(str(p)) for p in video_files)
    parts.append("--cli")
    return " ".join(parts)


def _write_fpsfix_summary(out_dir: Path, rows: list[dict[str, Any]], real_fps: float) -> Path:
    summary = out_dir / "fpsfix_summary.csv"
    fieldnames = [
        "file_name",
        "ok",
        "method",
        "target_fps",
        "fps_in",
        "fps_out",
        "frames_in",
        "frames_out",
        "duration_in",
        "duration_out",
        "output_path",
        "error",
    ]
    with open(summary, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "file_name": row.get("file_name"),
                    "ok": row.get("ok"),
                    "method": row.get("method"),
                    "target_fps": real_fps,
                    "fps_in": row.get("fps_in"),
                    "fps_out": row.get("fps_out"),
                    "frames_in": row.get("frames_in"),
                    "frames_out": row.get("frames_out"),
                    "duration_in": row.get("duration_in"),
                    "duration_out": row.get("duration_out"),
                    "output_path": row.get("output_path"),
                    "error": row.get("error") or "",
                }
            )
    return summary


def adjust_videos_fps(
    directory_path: str | None = None,
    video_files: list[str] | None = None,
    real_fps: float | Fraction | str | None = None,
    show_gui: bool = True,
) -> Path | None:
    """Batch-retag videos to a shared real FPS; write ``processed_fpsfix_<ts>/``."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting Adjust real FPS...")

    rate_frac: Fraction
    if real_fps is None:
        if not show_gui:
            print("Error: --set-fps is required in CLI mode.")
            return None
        import tkinter as tk
        from tkinter import simpledialog

        root = tk.Tk()
        root.withdraw()
        typed = simpledialog.askstring(
            "Adjust real FPS",
            "Enter the real (capture) FPS for all selected videos.\n"
            "Float or fraction (e.g. 29.97 or 30000/1001).\n"
            "Keeps every frame; duration becomes frames/real_fps.",
            parent=root,
        )
        root.destroy()
        if not typed:
            print("Cancelled: no FPS entered.")
            return None
        try:
            rate_frac = parse_fps_fraction(typed)
        except ValueError as exc:
            print(f"Error: {exc}")
            return None
    else:
        try:
            if isinstance(real_fps, str):
                rate_frac = parse_fps_fraction(real_fps)
            elif isinstance(real_fps, Fraction):
                rate_frac = real_fps
            else:
                rate = float(real_fps)
                if not 0.0 < rate < 10_000.0:
                    raise ValueError(f"FPS must be between 0 and 10000 (got {real_fps})")
                rate_frac = Fraction(rate).limit_denominator(1001)
        except ValueError as exc:
            print(f"Error: {exc}")
            return None
    rate = float(rate_frac)

    if not directory_path and not video_files:
        if not show_gui:
            print("Error: No directory or files provided in CLI mode.")
            return None
        import tkinter as tk
        from tkinter import filedialog, messagebox

        root = tk.Tk()
        root.withdraw()
        directory_path = filedialog.askdirectory(
            title="Select the directory containing videos to retag"
        )
        if not directory_path:
            messagebox.showerror("Error", "No directory selected.")
            root.destroy()
            return None
        root.destroy()

    abs_paths = _list_video_paths(directory_path, video_files)
    if not abs_paths:
        print("Error: No video files found.")
        return None

    base_dir = directory_path or os.path.dirname(abs_paths[0])
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(base_dir) / f"processed_fpsfix_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)

    cli = _format_fpsfix_cli(real_fps=rate, directory_path=directory_path, video_files=video_files)
    print("\n" + "=" * 60)
    print(">> vaila/numberframes: Equivalent CLI")
    print(f">>   {cli}")
    print("=" * 60 + "\n")
    print(f"Target real FPS: {rate:g}")
    print(f"Output directory: {out_dir}")

    rows: list[dict[str, Any]] = []
    for src in abs_paths:
        src_path = Path(src)
        dest = out_dir / src_path.name
        print(f"\nAdjusting {src_path.name} → {rate:g} fps ...")
        row = adjust_video_fps(src_path, rate_frac, dest)
        rows.append(row)
        if row.get("ok"):
            print(
                f"  OK [{row.get('method')}] "
                f"{row.get('fps_in')} → {row.get('fps_out')} fps, "
                f"frames {row.get('frames_in')} → {row.get('frames_out')}"
            )
        else:
            print(f"  FAILED: {row.get('error')}")

    summary = _write_fpsfix_summary(out_dir, rows, rate)
    ok_n = sum(1 for r in rows if r.get("ok"))
    print(f"\nDone: {ok_n}/{len(rows)} succeeded.")
    print(f"Summary: {summary}")

    if show_gui:
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showinfo(
                "Adjust real FPS",
                f"Finished {ok_n}/{len(rows)} videos.\n\nOutput:\n{out_dir}",
            )
            root.destroy()
        except Exception:
            pass

    return out_dir


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
    elif not directory_path:
        print("Error: No directory or files provided.")
        return
    else:
        video_files_list = sorted(
            f for f in os.listdir(directory_path) if f.lower().endswith(VIDEO_EXTENSIONS)
        )
        abs_paths = [str(Path(directory_path) / f) for f in video_files_list]

    if not directory_path:
        print("Error: Could not resolve output directory.")
        return
    base_dir: str = str(directory_path)

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

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_basic_file = os.path.join(base_dir, f"video_metadata_basic_{timestamp}.txt")
    output_full_file = os.path.join(base_dir, f"video_metadata_full_{timestamp}.json")

    # Mantém ordem por nome de arquivo
    video_infos.sort(key=lambda d: d.get("file_name", ""))

    try:
        with open(output_basic_file, "w", encoding="utf-8") as f:
            f.write(
                f"Video Metadata Summary\nDirectory: {base_dir}\n============================\n\n"
            )
            for info in video_infos:
                f.write(f"File: {info.get('file_name')}\n")
                if "error" in info:
                    f.write(f"  Error: {info['error']}\n\n")
                    continue
                f.write(f"  Frames:        {info.get('frame_count')}\n")
                f.write(f"  Duration:      {info.get('duration'):.3f} seconds\n")
                f.write(f"  Resolution:    {info.get('resolution')}\n")

                disp_fps = info.get("display_fps")
                if disp_fps:
                    f.write(f"  Display FPS:   {disp_fps:.9f}\n")
                else:
                    f.write("  Display FPS:   N/A\n")

                avg_fps = info.get("avg_fps")
                if avg_fps:
                    f.write(f"  Avg FPS:       {avg_fps:.9f}\n")
                else:
                    f.write("  Avg FPS:       N/A\n")

                cap_fps = info.get("capture_fps")
                if cap_fps:
                    f.write(f"  Capture FPS:   {cap_fps:.9f}\n")
                else:
                    f.write("  Capture FPS:   N/A\n")

                f.write(f"  Recommended Hz:{info.get('recommended_sampling_hz')}\n")
                f.write(
                    f"  Codec:         {info.get('codec_name')} ({info.get('codec_long_name')})\n"
                )
                f.write(
                    f"  Container:     {info.get('container_format')} ({info.get('container_long_name')})\n\n"
                )

        with open(output_full_file, "w", encoding="utf-8") as f:
            json.dump(video_infos, f, indent=4)

        print(f"\nBasic metadata saved to: {output_basic_file}")
        print(f"Full metadata saved to: {output_full_file}")
    except Exception as e:
        print(f"\nError saving metadata files: {e}")

    print("\n--- Summary ---")
    for info in video_infos:
        print(
            f"[{info.get('file_name')}]: {info.get('frame_count')} frames, {info.get('display_fps')} display fps"
        )


if __name__ == "__main__":
    import argparse
    import sys

    parser = argparse.ArgumentParser(
        description=(
            "Extract high-precision frame/metadata from videos, "
            "or retag container FPS (Adjust real FPS)."
        ),
        epilog="""
Examples:
  # Launch GUI to select directory (metadata)
  python numberframes.py

  # Process a specific directory via CLI
  python numberframes.py --dir /path/to/videos

  # Process specific video files
  python numberframes.py /path/to/video1.mp4 /path/to/video2.avi

  # Retag all videos in a directory to real capture FPS
  python numberframes.py --set-fps 29.97 --dir /path/to/videos --cli
  python numberframes.py --set-fps 30000/1001 video1.mp4 video2.mp4 --cli
        """,
        formatter_class=argparse.RawTextHelpFormatter,
    )
    parser.add_argument("videos", nargs="*", help="Specific video files to process")
    parser.add_argument("--dir", "-d", help="Directory containing videos to process")
    parser.add_argument("--cli", action="store_true", help="Run without GUI display")
    parser.add_argument(
        "--set-fps",
        metavar="FPS",
        help="Adjust real FPS (float or num/den). Writes processed_fpsfix_<timestamp>/",
    )

    # If no arguments provided at all (except the script name), run default GUI mode
    if len(sys.argv) == 1:
        count_frames_in_videos(show_gui=True)
    else:
        args = parser.parse_args()
        show_gui_flag = not args.cli

        if args.set_fps is not None:
            try:
                rate_frac = parse_fps_fraction(args.set_fps)
            except ValueError as exc:
                parser.error(str(exc))
            if args.videos:
                adjust_videos_fps(
                    video_files=args.videos,
                    real_fps=rate_frac,
                    show_gui=show_gui_flag,
                )
            elif args.dir:
                adjust_videos_fps(
                    directory_path=args.dir,
                    real_fps=rate_frac,
                    show_gui=show_gui_flag,
                )
            else:
                adjust_videos_fps(real_fps=rate_frac, show_gui=show_gui_flag)
        elif args.videos:
            count_frames_in_videos(video_files=args.videos, show_gui=show_gui_flag)
        elif args.dir:
            count_frames_in_videos(directory_path=args.dir, show_gui=show_gui_flag)
        else:
            # Should not happen as argparse handles required inputs, but fallback to GUI just in case
            count_frames_in_videos(show_gui=True)
