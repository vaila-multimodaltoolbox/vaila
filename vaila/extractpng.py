"""
================================================================================
Extract PNG Tool - extractpng.py
================================================================================
*vailá* – Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/vaila-multimodaltoolbox/vaila

Created: December 15, 2023
Update: 09 September 2026
Version: 0.3.131
Python Version: 3.12.14

Description:
------------
High-performance video ↔ PNG extraction and creation suite accelerated by
NVIDIA GPU hardware (NVDEC decoding & NVENC encoding via FFmpeg CUDA) with
automatic CPU software fallback.

Features:
- NVIDIA GPU Acceleration: Detects NVIDIA CUDA NVDEC decoding and NVENC encoding
  to dramatically accelerate video-to-PNG extraction and PNG-to-video encoding.
- High-Speed Lossless Extraction: Defaults to zlib compression level 1 with
  Paeth filtering bypass (`-pred none`), extracting lossless PNG frames ~3x faster
  than standard level 6 with zero quality loss.
- Redundant Rescaler Elimination: Automatically skips unnecessary Lanczos/bicubic
  CPU rescaling when target resolution matches native video stream dimensions.
- Parallel Batch Processing: Multi-video extraction via ThreadPoolExecutor,
  saturating NVDEC hardware decoders and multi-core CPU pipelines.
- Modern Tkinter GUI with hardware acceleration status badge, compression chooser,
  parallel worker controls, and CLI mirror reproduction.
- Full CLI supporting extract, create, and frames subcommands with headless parity.

CLI::

    uv run vaila/extractpng.py
    uv run vaila/extractpng.py extract -i /path/to/videos --hwaccel auto --compression 1
    uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264 --hwaccel auto
    uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5,7 --hwaccel auto

GUI (no args, or from Frame C → Video↔PNG): one window — pick mode, paths, Run.

================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import functools
import json
import os
import shutil
import subprocess
import sys
import time
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]

VIDEO_EXTENSIONS = (".avi", ".mp4", ".mov", ".mkv", ".webm", ".m4v")
DEFAULT_PATTERN = "%09d.png"


def _timestamp() -> str:
    return time.strftime("%Y%m%d%H%M%S")


def _is_video_file(name: str) -> bool:
    return name.lower().endswith(VIDEO_EXTENSIONS)


def list_videos_in_dir(directory: str | Path) -> list[Path]:
    directory = Path(directory)
    return sorted(p for p in directory.iterdir() if p.is_file() and _is_video_file(p.name))


@functools.lru_cache(maxsize=1)
def get_cuda_status() -> dict[str, Any]:
    """Detect if NVIDIA GPU is present and FFmpeg supports CUDA decoding / NVENC encoding."""
    has_nvidia = False
    device_name = "None"
    driver_version = "Unknown"

    if shutil.which("nvidia-smi"):
        try:
            res = subprocess.run(
                [
                    "nvidia-smi",
                    "--query-gpu=name,driver_version",
                    "--format=csv,noheader",
                ],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res.returncode == 0 and res.stdout.strip():
                first_line = res.stdout.strip().splitlines()[0]
                parts = [p.strip() for p in first_line.split(",")]
                if len(parts) >= 2:
                    device_name = parts[0]
                    driver_version = parts[1]
                    has_nvidia = True
        except Exception:
            pass

    if not has_nvidia:
        try:
            import torch

            if torch.cuda.is_available():
                device_name = torch.cuda.get_device_name(0)
                has_nvidia = True
        except Exception:
            pass

    ffmpeg_cuda = False
    ffmpeg_nvenc = False

    if shutil.which("ffmpeg"):
        try:
            res_hw = subprocess.run(
                ["ffmpeg", "-hide_banner", "-hwaccels"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res_hw.returncode == 0 and "cuda" in res_hw.stdout.lower():
                ffmpeg_cuda = True
        except Exception:
            pass

        try:
            res_enc = subprocess.run(
                ["ffmpeg", "-hide_banner", "-encoders"],
                capture_output=True,
                text=True,
                check=False,
                timeout=3,
            )
            if res_enc.returncode == 0 and "nvenc" in res_enc.stdout.lower():
                ffmpeg_nvenc = True
        except Exception:
            pass

    recommended = "cuda" if (has_nvidia and ffmpeg_cuda) else "auto"

    return {
        "has_nvidia": has_nvidia,
        "device_name": device_name,
        "driver_version": driver_version,
        "ffmpeg_cuda": ffmpeg_cuda,
        "ffmpeg_nvenc": ffmpeg_nvenc,
        "recommended_hwaccel": recommended,
    }


def get_video_info(video_path: str | Path) -> tuple[int, int, float]:
    """Return (width, height, fps), swapping dims for 90/270° display rotation."""
    video_path = Path(video_path)
    cmd = [
        "ffprobe",
        "-v",
        "error",
        "-print_format",
        "json",
        "-show_streams",
        str(video_path),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    data = json.loads(result.stdout)
    video_stream = next(
        (s for s in data.get("streams", []) if s.get("codec_type") == "video"),
        None,
    )
    if not video_stream:
        raise ValueError(f"No video stream found: {video_path}")

    raw_width = int(video_stream.get("width", 0))
    raw_height = int(video_stream.get("height", 0))

    r_frame_rate_str = video_stream.get("r_frame_rate", "0/0")
    fps = 30.0
    if "/" in r_frame_rate_str:
        try:
            num, den = map(int, r_frame_rate_str.split("/"))
            if den != 0:
                fps = float(num) / den
        except (ValueError, ZeroDivisionError):
            pass

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
    if rotation in (90, 270):
        width, height = raw_height, raw_width
    else:
        width, height = raw_width, raw_height
    return width, height, fps


def build_extract_png_command(
    video_path: str | Path,
    output_pattern: str | Path,
    *,
    width: int,
    height: int,
    orig_width: int | None = None,
    orig_height: int | None = None,
    hwaccel: bool | str = "auto",
    compression_level: int = 1,
) -> list[str]:
    """Build optimized ffmpeg argv for video → PNG. Input options precede ``-i``."""
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner"]

    # Configure hardware acceleration (must precede -i)
    if hwaccel is False or hwaccel == "cpu":
        pass
    elif hwaccel == "cuda":
        cmd.extend(["-hwaccel", "cuda"])
    elif hwaccel is True or hwaccel == "auto":
        cuda_status = get_cuda_status()
        if cuda_status["ffmpeg_cuda"] and cuda_status["has_nvidia"]:
            cmd.extend(["-hwaccel", "cuda"])
        else:
            cmd.extend(["-hwaccel", "auto"])

    cmd.extend(["-i", str(video_path)])

    # Avoid redundant Lanczos CPU rescaling if target resolution matches native video
    if (
        orig_width is not None
        and orig_height is not None
        and width == orig_width
        and height == orig_height
    ):
        filter_args: list[str] = []
    else:
        filter_args = ["-vf", f"scale={width}:{height}:flags=lanczos"]

    cmd.extend(filter_args)
    cmd.extend(
        [
            "-q:v",
            "1",
            "-fps_mode",
            "passthrough",
            "-sws_flags",
            "bicubic",
            "-pix_fmt",
            "rgb24",
            "-f",
            "image2",
            "-compression_level",
            str(compression_level),
        ]
    )

    # Fast prediction method for speed when compression level <= 2
    if compression_level <= 2:
        cmd.extend(["-pred", "none"])

    cmd.extend(["-threads", "0", str(output_pattern)])
    return cmd


def build_select_frame_command(
    video_path: str | Path,
    frame_number: int,
    output_path: str | Path,
    *,
    hwaccel: bool | str = "auto",
) -> list[str]:
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner"]

    if hwaccel is False or hwaccel == "cpu":
        pass
    elif hwaccel == "cuda":
        cmd.extend(["-hwaccel", "cuda"])
    elif hwaccel is True or hwaccel == "auto":
        cuda_status = get_cuda_status()
        if cuda_status["ffmpeg_cuda"] and cuda_status["has_nvidia"]:
            cmd.extend(["-hwaccel", "cuda"])
        else:
            cmd.extend(["-hwaccel", "auto"])

    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vf",
            f"select=eq(n\\,{frame_number})",
            "-vframes",
            "1",
            "-pix_fmt",
            "rgb24",
            "-compression_level",
            "1",
            str(output_path),
        ]
    )
    return cmd


def build_png_to_video_command(
    input_pattern: str | Path,
    output_video: str | Path,
    *,
    fps: float,
    codec: str = "264",
    hwaccel: bool | str = False,
) -> list[str]:
    use_nvenc = False
    if hwaccel in (True, "cuda") or str(codec).endswith("_nvenc"):
        use_nvenc = True
    elif hwaccel == "auto":
        cuda_status = get_cuda_status()
        if cuda_status["has_nvidia"] and cuda_status["ffmpeg_nvenc"]:
            use_nvenc = True

    if use_nvenc:
        if str(codec) in ("265", "hevc", "h265", "265_nvenc", "hevc_nvenc"):
            vcodec = "hevc_nvenc"
        else:
            vcodec = "h264_nvenc"
        extra = ["-preset", "p4", "-tune", "hq"]
    else:
        if str(codec) in ("265", "hevc", "h265"):
            vcodec = "libx265"
            extra = ["-x265-params", "log-level=error"]
        else:
            vcodec = "libx264"
            extra = []

    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-framerate",
        str(fps),
        "-i",
        str(input_pattern),
        "-c:v",
        vcodec,
        *extra,
        "-pix_fmt",
        "yuv420p",
        str(output_video),
    ]


def _run_ffmpeg(cmd: list[str]) -> None:
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        err_msg = res.stderr.strip() or res.stdout.strip()
        raise subprocess.CalledProcessError(res.returncode, cmd, output=res.stdout, stderr=err_msg)


def extract_png_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = DEFAULT_PATTERN,
    hwaccel: bool | str = "auto",
    compression: int = 1,
) -> int:
    """Extract all frames from one video into ``output_dir``. Returns frame count."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps = get_video_info(video_path)
    output_pattern = output_dir / pattern

    cuda_info = get_cuda_status()
    use_cuda = False
    if hwaccel == "cuda":
        use_cuda = True
    elif hwaccel in (True, "auto"):
        use_cuda = cuda_info["has_nvidia"] and cuda_info["ffmpeg_cuda"]

    if use_cuda:
        hw_label = f"NVIDIA CUDA ({cuda_info['device_name']})"
        hw_flag: bool | str = "cuda"
    elif hwaccel == "auto":
        hw_label = "hwaccel auto"
        hw_flag = "auto"
    else:
        hw_label = "CPU software decoder"
        hw_flag = False

    t0 = time.time()
    try:
        print(f"Processing {video_path.name} [{hw_label}]...")
        _run_ffmpeg(
            build_extract_png_command(
                video_path,
                output_pattern,
                width=width,
                height=height,
                orig_width=width,
                orig_height=height,
                hwaccel=hw_flag,
                compression_level=compression,
            )
        )
    except subprocess.CalledProcessError as exc:
        if use_cuda or hwaccel == "auto":
            err_snip = (exc.stderr or str(exc)).strip().splitlines()[-1] if exc.stderr else ""
            print(
                f"Hardware acceleration failed ({err_snip}), falling back to software CPU decoder..."
            )
            _run_ffmpeg(
                build_extract_png_command(
                    video_path,
                    output_pattern,
                    width=width,
                    height=height,
                    orig_width=width,
                    orig_height=height,
                    hwaccel=False,
                    compression_level=compression,
                )
            )
            hw_label = "CPU (Fallback)"
        else:
            raise

    elapsed = time.time() - t0
    total_frames = len([f for f in output_dir.iterdir() if f.suffix.lower() == ".png"])
    eff_fps = (total_frames / elapsed) if elapsed > 0 else 0.0

    info_path = output_dir / "video_info.txt"
    info_path.write_text(
        f"Original video: {video_path.name}\n"
        f"FPS: {fps}\n"
        f"Resolution: {width}x{height}\n"
        f"Total frames: {total_frames}\n"
        f"Extraction timestamp: {_timestamp()}\n"
        f"Hardware acceleration: {hw_label}\n"
        f"Compression level: {compression}\n"
        f"Processing time: {elapsed:.2f} s ({eff_fps:.1f} fps)\n",
        encoding="utf-8",
    )
    print(
        f"Extracted {total_frames} frames from {video_path.name} in {elapsed:.2f} s "
        f"({eff_fps:.1f} fps) → {output_dir}"
    )
    print(f"Resolution: {width}x{height}, Source FPS: {fps:.2f}")
    return total_frames


def extract_png_from_videos(
    src_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    pattern: str = DEFAULT_PATTERN,
    hwaccel: bool | str = "auto",
    compression: int = 1,
    workers: int | None = None,
) -> Path:
    """Batch-extract PNGs for every video in ``src_dir`` with parallel GPU/CPU processing."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    videos = list_videos_in_dir(src_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {src_dir}")

    dest = Path(output_dir) if output_dir else src_dir / f"vaila_extractpng_{_timestamp()}"
    dest.mkdir(parents=True, exist_ok=True)

    cuda_info = get_cuda_status()
    use_cuda = hwaccel == "cuda" or (
        hwaccel in (True, "auto") and cuda_info["has_nvidia"] and cuda_info["ffmpeg_cuda"]
    )
    hw_desc = f"NVIDIA CUDA ({cuda_info['device_name']})" if use_cuda else "CPU Software Decoder"

    if workers is None or workers <= 0:
        if use_cuda:
            max_workers = min(len(videos), 4)
        else:
            cpu_cores = os.cpu_count() or 4
            max_workers = min(len(videos), max(1, cpu_cores // 2))
    else:
        max_workers = min(len(videos), workers)

    print(
        "================================================================================\n"
        f"Starting extraction of PNG frames ({len(videos)} video(s))...\n"
        f"Hardware: {hw_desc}\n"
        f"Mode: Lossless PNG (Level {compression}) | Parallel Workers: {max_workers}\n"
        "================================================================================"
    )

    t_start = time.time()
    results: list[int] = []

    def _process_one(video: Path) -> int:
        out = dest / f"{video.stem}_png"
        return extract_png_from_video(
            video, out, pattern=pattern, hwaccel=hwaccel, compression=compression
        )

    if max_workers > 1 and len(videos) > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            results = list(executor.map(_process_one, videos))
    else:
        for video in videos:
            results.append(_process_one(video))

    t_total = time.time() - t_start
    total_frames = sum(results)
    overall_fps = (total_frames / t_total) if t_total > 0 else 0.0

    print(
        f"✓ Done. Extracted {total_frames} total frames across {len(videos)} video(s) "
        f"in {t_total:.2f} s ({overall_fps:.1f} fps) → {dest}"
    )
    return dest


def extract_select_frames(
    video_file: str | Path,
    frame_numbers: list[int],
    *,
    output_dir: str | Path | None = None,
    hwaccel: bool | str = "auto",
) -> Path:
    video_file = Path(video_file)
    if not video_file.is_file():
        raise FileNotFoundError(f"Video not found: {video_file}")
    if not frame_numbers:
        raise ValueError("No frame numbers provided")

    dest = (
        Path(output_dir) if output_dir else video_file.parent / f"vaila_grabframes_{_timestamp()}"
    )
    dest.mkdir(parents=True, exist_ok=True)

    cuda_info = get_cuda_status()
    hw_desc = (
        f"NVIDIA CUDA ({cuda_info['device_name']})"
        if (
            hwaccel == "cuda"
            or (hwaccel in (True, "auto") and cuda_info["has_nvidia"] and cuda_info["ffmpeg_cuda"])
        )
        else "CPU"
    )

    print(f"Extracting {len(frame_numbers)} frame(s) from {video_file.name} [{hw_desc}]...")
    for frame_number in frame_numbers:
        output_path = dest / f"frame_{frame_number:03d}.png"
        _run_ffmpeg(
            build_select_frame_command(video_file, frame_number, output_path, hwaccel=hwaccel)
        )
        print(f"  frame {frame_number} → {output_path.name}")
    print(f"Done. Output: {dest}")
    return dest


def _png_dirs_to_process(src: Path, exclude: Path | None = None) -> list[Path]:
    """Immediate subdirs of ``src`` that contain PNGs; or ``src`` itself if it does."""
    dirs: list[Path] = []
    if any(p.suffix.lower() == ".png" for p in src.iterdir() if p.is_file()):
        dirs.append(src)
    for child in sorted(src.iterdir()):
        if not child.is_dir():
            continue
        if exclude is not None and child.resolve() == exclude.resolve():
            continue
        if child.name.startswith("vaila_png2videos_"):
            continue
        if any(p.suffix.lower() == ".png" for p in child.iterdir() if p.is_file()):
            dirs.append(child)
    return dirs


def create_video_from_png(
    src_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    fps: float = 30.0,
    codec: str = "264",
    pattern: str = DEFAULT_PATTERN,
    hwaccel: bool | str = "auto",
) -> Path:
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    dest = Path(output_dir) if output_dir else src_dir / f"vaila_png2videos_{_timestamp()}"
    dest.mkdir(parents=True, exist_ok=True)

    png_dirs = _png_dirs_to_process(src_dir, exclude=dest)
    if not png_dirs:
        raise FileNotFoundError(f"No PNG sequences found under {src_dir}")

    cuda_info = get_cuda_status()
    use_nvenc = hwaccel == "cuda" or (
        hwaccel in (True, "auto") and cuda_info["has_nvidia"] and cuda_info["ffmpeg_nvenc"]
    )
    enc_desc = f"NVIDIA NVENC ({cuda_info['device_name']})" if use_nvenc else "CPU Software Encoder"

    print(f"Creating videos from {len(png_dirs)} PNG sequence(s) @ {fps} fps [{enc_desc}]...")
    for png_dir in png_dirs:
        output_video = dest / f"{png_dir.name}.mp4"
        input_pattern = png_dir / pattern
        t0 = time.time()
        try:
            _run_ffmpeg(
                build_png_to_video_command(
                    input_pattern, output_video, fps=fps, codec=codec, hwaccel=hwaccel
                )
            )
        except subprocess.CalledProcessError:
            if use_nvenc:
                print("Hardware NVENC encoding failed, falling back to CPU software encoder...")
                _run_ffmpeg(
                    build_png_to_video_command(
                        input_pattern, output_video, fps=fps, codec=codec, hwaccel=False
                    )
                )
            else:
                raise
        dt = time.time() - t0
        print(f"  video created in {dt:.2f} s: {output_video}")
    print(f"Done. Output: {dest}")
    return dest


def parse_frame_list(text: str) -> list[int]:
    """Parse ``0,3,5`` or ``0 3 5`` into sorted unique ints."""
    parts = [p for p in text.replace(" ", ",").split(",") if p.strip()]
    frames = sorted({int(p.strip()) for p in parts})
    if any(f < 0 for f in frames):
        raise ValueError("Frame numbers must be >= 0")
    return frames


def build_cli_argv(
    mode: str,
    *,
    input_path: str,
    output_path: str | None = None,
    pattern: str = DEFAULT_PATTERN,
    fps: float = 30.0,
    codec: str = "264",
    frames: str | None = None,
    hwaccel: str = "auto",
    compression: int = 1,
    workers: int | None = None,
) -> list[str]:
    argv = ["uv", "run", "vaila/extractpng.py", mode, "-i", input_path]
    if output_path:
        argv.extend(["-o", output_path])
    if hwaccel and hwaccel != "auto":
        argv.extend(["--hwaccel", hwaccel])
    if mode == "extract":
        if pattern != DEFAULT_PATTERN:
            argv.extend(["--pattern", pattern])
        if compression != 1:
            argv.extend(["--compression", str(compression)])
        if workers:
            argv.extend(["--workers", str(workers)])
    elif mode == "create":
        argv.extend(["--fps", str(fps), "--codec", str(codec)])
        if pattern != DEFAULT_PATTERN:
            argv.extend(["--pattern", pattern])
    elif mode == "frames" and frames:
        argv.extend(["--frames", frames])
    return argv


# ---------------------------------------------------------------------------
# GUI — one window
# ---------------------------------------------------------------------------


class ExtractPngApp:
    """Single easy GUI for extract / create / select-frames."""

    def __init__(self, parent: tk.Misc | None = None):
        default_root = getattr(tk, "_default_root", None)
        if parent is not None:
            self.root = tk.Toplevel(parent)
        elif default_root is not None:
            self.root = tk.Toplevel(default_root)
        else:
            self.root = tk.Tk()

        self.root.title("vailá — Video ↔ PNG (NVIDIA GPU Accelerated)")
        self.root.resizable(True, False)
        self.root.minsize(620, 320)

        self.mode = tk.StringVar(value="extract")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.pattern_var = tk.StringVar(value=DEFAULT_PATTERN)
        self.fps_var = tk.StringVar(value="30")
        self.codec_var = tk.StringVar(value="264")
        self.frames_var = tk.StringVar(value="0,3,5")
        self.hwaccel_var = tk.StringVar(value="auto")
        self.compression_var = tk.StringVar(value="1")
        self.workers_var = tk.StringVar(value="auto")
        self.status_var = tk.StringVar(value="Choose a mode, set paths, then Run.")

        self._build()
        self._on_mode_change()

    def _build(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        header_row = ttk.Frame(frm)
        header_row.grid(row=0, column=0, columnspan=3, sticky="ew", padx=10, pady=4)

        ttk.Label(header_row, text="Video ↔ PNG", font=("TkDefaultFont", 12, "bold")).pack(
            side="left"
        )

        cuda_info = get_cuda_status()
        if cuda_info["has_nvidia"] and cuda_info["ffmpeg_cuda"]:
            hw_badge = f"🚀 NVIDIA CUDA: {cuda_info['device_name']}"
            hw_color = "#007700"
        else:
            hw_badge = "💻 Hardware: CPU Software Mode"
            hw_color = "#555555"

        ttk.Label(
            header_row,
            text=hw_badge,
            font=("TkDefaultFont", 9, "bold"),
            foreground=hw_color,
        ).pack(side="right")

        mode_row = ttk.Frame(frm)
        mode_row.grid(row=1, column=0, columnspan=3, sticky="w", padx=10, pady=4)
        ttk.Label(mode_row, text="Mode:").pack(side="left", padx=(0, 8))
        for value, label in (
            ("extract", "Video → PNG"),
            ("create", "PNG → Video"),
            ("frames", "Select frames"),
        ):
            ttk.Radiobutton(
                mode_row,
                text=label,
                value=value,
                variable=self.mode,
                command=self._on_mode_change,
            ).pack(side="left", padx=4)

        ttk.Label(frm, text="Input:").grid(row=2, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(frm, textvariable=self.input_var, width=56).grid(
            row=2, column=1, sticky="ew", padx=10, pady=4
        )
        ttk.Button(frm, text="Browse…", command=self._browse_input).grid(
            row=2, column=2, padx=10, pady=4
        )

        ttk.Label(frm, text="Output:").grid(row=3, column=0, sticky="w", padx=10, pady=4)
        ttk.Entry(frm, textvariable=self.output_var, width=56).grid(
            row=3, column=1, sticky="ew", padx=10, pady=4
        )
        ttk.Button(frm, text="Browse…", command=self._browse_output).grid(
            row=3, column=2, padx=10, pady=4
        )
        ttk.Label(frm, text="(leave empty for timestamped folder next to input)").grid(
            row=4, column=1, sticky="w", padx=10
        )

        self.options_frame = ttk.LabelFrame(frm, text="Options & Acceleration", padding=8)
        self.options_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=4)

        self.pattern_label = ttk.Label(self.options_frame, text="PNG pattern:")
        self.pattern_entry = ttk.Entry(self.options_frame, textvariable=self.pattern_var, width=16)

        self.hwaccel_label = ttk.Label(self.options_frame, text="Acceleration:")
        self.hwaccel_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.hwaccel_var,
            values=("auto", "cuda", "cpu"),
            width=8,
            state="readonly",
        )

        self.compression_label = ttk.Label(self.options_frame, text="PNG Speed:")
        self.compression_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.compression_var,
            values=("1", "3", "6"),
            width=4,
            state="readonly",
        )

        self.workers_label = ttk.Label(self.options_frame, text="Workers:")
        self.workers_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.workers_var,
            values=("auto", "1", "2", "3", "4"),
            width=6,
            state="readonly",
        )

        self.fps_label = ttk.Label(self.options_frame, text="FPS:")
        self.fps_entry = ttk.Entry(self.options_frame, textvariable=self.fps_var, width=8)

        cuda_info = get_cuda_status()
        codec_vals = ("264", "265")
        if cuda_info["has_nvidia"] and cuda_info["ffmpeg_nvenc"]:
            codec_vals = ("264", "265", "264_nvenc", "265_nvenc")

        self.codec_label = ttk.Label(self.options_frame, text="Codec:")
        self.codec_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.codec_var,
            values=codec_vals,
            width=10,
            state="readonly",
        )
        self.frames_label = ttk.Label(self.options_frame, text="Frames (e.g. 0,3,5):")
        self.frames_entry = ttk.Entry(self.options_frame, textvariable=self.frames_var, width=28)

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=6, column=0, columnspan=3, sticky="e", padx=10, pady=4)
        ttk.Button(btn_row, text="Run", command=self._run).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Close", command=self.root.destroy).pack(side="right", padx=4)

        ttk.Label(frm, textvariable=self.status_var, wraplength=580).grid(
            row=7, column=0, columnspan=3, sticky="w", padx=10, pady=4
        )

        frm.columnconfigure(1, weight=1)

    def _on_mode_change(self) -> None:
        for w in self.options_frame.winfo_children():
            if isinstance(w, tk.Widget):
                w.grid_forget()
        mode = self.mode.get()
        if mode == "extract":
            self.pattern_label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.pattern_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
            self.hwaccel_label.grid(row=0, column=2, sticky="w", padx=4, pady=2)
            self.hwaccel_combo.grid(row=0, column=3, sticky="w", padx=4, pady=2)

            self.compression_label.grid(row=1, column=0, sticky="w", padx=4, pady=2)
            self.compression_combo.grid(row=1, column=1, sticky="w", padx=4, pady=2)
            self.workers_label.grid(row=1, column=2, sticky="w", padx=4, pady=2)
            self.workers_combo.grid(row=1, column=3, sticky="w", padx=4, pady=2)
            self.status_var.set(
                "Select a folder of videos. NVIDIA GPU acceleration and Fast Lossless extraction active."
            )
        elif mode == "create":
            self.fps_label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.fps_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
            self.codec_label.grid(row=0, column=2, sticky="w", padx=4, pady=2)
            self.codec_combo.grid(row=0, column=3, sticky="w", padx=4, pady=2)

            self.pattern_label.grid(row=1, column=0, sticky="w", padx=4, pady=2)
            self.pattern_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)
            self.hwaccel_label.grid(row=1, column=2, sticky="w", padx=4, pady=2)
            self.hwaccel_combo.grid(row=1, column=3, sticky="w", padx=4, pady=2)
            self.status_var.set(
                "Select a folder of PNG sequences (or subfolders). NVENC GPU encoding supported."
            )
        else:
            self.frames_label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.frames_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
            self.hwaccel_label.grid(row=0, column=2, sticky="w", padx=4, pady=2)
            self.hwaccel_combo.grid(row=0, column=3, sticky="w", padx=4, pady=2)
            self.status_var.set("Select one video and list frame indices to grab.")

    def _browse_input(self) -> None:
        mode = self.mode.get()
        if mode == "frames":
            path = filedialog.askopenfilename(
                parent=self.root,
                title="Select video",
                filetypes=[
                    ("Video", "*.mp4 *.avi *.mov *.mkv *.webm *.m4v"),
                    ("All", "*.*"),
                ],
            )
        else:
            path = filedialog.askdirectory(parent=self.root, title="Select input directory")
        if path:
            self.input_var.set(path)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(parent=self.root, title="Select output directory (optional)")
        if path:
            self.output_var.set(path)

    def _run(self) -> None:
        mode = self.mode.get()
        input_path = self.input_var.get().strip()
        output_path = self.output_var.get().strip() or None
        pattern = self.pattern_var.get().strip() or DEFAULT_PATTERN
        hwaccel = self.hwaccel_var.get().strip() or "auto"

        if not input_path:
            messagebox.showerror("Missing input", "Please choose an input path.", parent=self.root)
            return

        try:
            if mode == "extract":
                try:
                    comp = int(self.compression_var.get().strip() or "1")
                except ValueError:
                    comp = 1
                w_str = self.workers_var.get().strip()
                workers = int(w_str) if w_str.isdigit() else None

                cli = build_cli_argv(
                    "extract",
                    input_path=input_path,
                    output_path=output_path,
                    pattern=pattern,
                    hwaccel=hwaccel,
                    compression=comp,
                    workers=workers,
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = extract_png_from_videos(
                    input_path,
                    output_dir=output_path,
                    pattern=pattern,
                    hwaccel=hwaccel,
                    compression=comp,
                    workers=workers,
                )
                msg = f"PNG extraction done:\n{dest}"
            elif mode == "create":
                fps = float(self.fps_var.get().strip() or "30")
                codec = self.codec_var.get().strip() or "264"
                cli = build_cli_argv(
                    "create",
                    input_path=input_path,
                    output_path=output_path,
                    pattern=pattern,
                    fps=fps,
                    codec=codec,
                    hwaccel=hwaccel,
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = create_video_from_png(
                    input_path,
                    output_dir=output_path,
                    fps=fps,
                    codec=codec,
                    pattern=pattern,
                    hwaccel=hwaccel,
                )
                msg = f"Video creation done:\n{dest}"
            else:
                frames_text = self.frames_var.get().strip()
                frames = parse_frame_list(frames_text)
                cli = build_cli_argv(
                    "frames",
                    input_path=input_path,
                    output_path=output_path,
                    frames=frames_text,
                    hwaccel=hwaccel,
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = extract_select_frames(
                    input_path, frames, output_dir=output_path, hwaccel=hwaccel
                )
                msg = f"Frame grab done:\n{dest}"

            self.status_var.set(msg.replace("\n", " "))
            messagebox.showinfo("Done", msg, parent=self.root)
        except Exception as exc:
            self.status_var.set(f"Error: {exc}")
            messagebox.showerror("Error", str(exc), parent=self.root)

    def run(self) -> None:
        if isinstance(self.root, tk.Toplevel):
            self.root.grab_set()
            self.root.wait_window()
        else:
            self.root.mainloop()


def run_extractpng_gui(parent: tk.Misc | None = None) -> None:
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).resolve().parent}")
    print("Starting vailá Video ↔ PNG...")
    ExtractPngApp(parent=parent).run()


# ---------------------------------------------------------------------------
# Backward-compatible class API (used by vaila.py / __init__.py)
# ---------------------------------------------------------------------------


class VideoProcessor:
    """Legacy wrapper — prefer ``run_extractpng_gui`` / CLI subcommands."""

    def __init__(self):
        self.pattern = DEFAULT_PATTERN

    def extract_png_from_videos(self):
        run_extractpng_gui()

    def extract_select_frames_from_video(self):
        app = ExtractPngApp()
        app.mode.set("frames")
        app._on_mode_change()
        app.run()

    def create_video_from_png(self):
        app = ExtractPngApp()
        app.mode.set("create")
        app._on_mode_change()
        app.run()

    def run(self):
        run_extractpng_gui()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="extractpng",
        description="Extract PNG frames from video, or build video from PNG sequences (GPU accelerated).",
    )
    sub = parser.add_subparsers(dest="command")

    p_ex = sub.add_parser("extract", help="Video folder → PNG sequences")
    p_ex.add_argument("-i", "--input", required=True, help="Directory with videos")
    p_ex.add_argument("-o", "--output", default=None, help="Output directory")
    p_ex.add_argument(
        "--pattern",
        default=DEFAULT_PATTERN,
        help=f"PNG name pattern (default {DEFAULT_PATTERN})",
    )
    p_ex.add_argument(
        "--hwaccel",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Hardware acceleration method (default auto: uses NVIDIA CUDA if available)",
    )
    p_ex.add_argument(
        "-c",
        "--compression",
        type=int,
        default=1,
        choices=range(0, 10),
        help="PNG zlib compression level 0-9 (default 1 for fastest lossless extraction)",
    )
    p_ex.add_argument(
        "-j",
        "--workers",
        type=int,
        default=None,
        help="Number of parallel workers for multi-video extraction (default: auto)",
    )

    p_cr = sub.add_parser("create", help="PNG sequences → videos")
    p_cr.add_argument("-i", "--input", required=True, help="Directory with PNG folders")
    p_cr.add_argument("-o", "--output", default=None, help="Output directory")
    p_cr.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    p_cr.add_argument(
        "--codec",
        default="264",
        choices=("264", "265", "264_nvenc", "265_nvenc"),
        help="H.264 or H.265 (libx264/libx265 or NVENC)",
    )
    p_cr.add_argument("--pattern", default=DEFAULT_PATTERN, help="Input PNG pattern")
    p_cr.add_argument(
        "--hwaccel",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Hardware acceleration method (NVENC if available)",
    )

    p_fr = sub.add_parser("frames", help="Grab specific frames from one video")
    p_fr.add_argument("-i", "--input", required=True, help="Video file")
    p_fr.add_argument("-o", "--output", default=None, help="Output directory")
    p_fr.add_argument(
        "--frames",
        required=True,
        help="Comma-separated frame indices (e.g. 0,3,5)",
    )
    p_fr.add_argument(
        "--hwaccel",
        default="auto",
        choices=("auto", "cuda", "cpu"),
        help="Hardware acceleration method (CUDA if available)",
    )

    return parser


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv:
        run_extractpng_gui()
        return 0
    if argv[0] in ("-h", "--help"):
        build_arg_parser().print_help()
        return 0

    parser = build_arg_parser()
    args = parser.parse_args(argv)

    if args.command == "extract":
        extract_png_from_videos(
            args.input,
            output_dir=args.output,
            pattern=args.pattern,
            hwaccel=args.hwaccel,
            compression=args.compression,
            workers=args.workers,
        )
    elif args.command == "create":
        create_video_from_png(
            args.input,
            output_dir=args.output,
            fps=args.fps,
            codec=args.codec,
            pattern=args.pattern,
            hwaccel=args.hwaccel,
        )
    elif args.command == "frames":
        extract_select_frames(
            args.input,
            parse_frame_list(args.frames),
            output_dir=args.output,
            hwaccel=args.hwaccel,
        )
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
