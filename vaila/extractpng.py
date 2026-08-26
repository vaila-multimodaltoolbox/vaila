"""
================================================================================
Extract PNG Tool - extractpng.py
================================================================================
*vailá* – Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/vaila-multimodaltoolbox/vaila

Created: December 15, 2023
Update: 25 August 2026
Version: 0.3.114
Python Version: 3.12.14

Description:
------------
Extract PNG frames from videos, create videos from PNG sequences, or grab
selected frames. One simple Tkinter window for GUI; full CLI for headless use.

CLI::

    uv run vaila/extractpng.py
    uv run vaila/extractpng.py extract -i /path/to/videos
    uv run vaila/extractpng.py create -i /path/to/png_dirs --fps 30 --codec 264
    uv run vaila/extractpng.py frames -i VIDEO.mp4 --frames 0,3,5,7

GUI (no args, or from Frame C → Video↔PNG): one window — pick mode, paths, Run.

================================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import json
import subprocess
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

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
    hwaccel: bool = True,
) -> list[str]:
    """Build ffmpeg argv for video → PNG. Input options come before ``-i``."""
    cmd: list[str] = ["ffmpeg", "-y", "-hide_banner"]
    if hwaccel:
        # Must precede -i (input option). Do not force a specific cuvid decoder.
        cmd.extend(["-hwaccel", "auto"])
    cmd.extend(
        [
            "-i",
            str(video_path),
            "-vf",
            f"scale={width}:{height}:flags=lanczos",
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
            "6",
            str(output_pattern),
        ]
    )
    return cmd


def build_select_frame_command(
    video_path: str | Path, frame_number: int, output_path: str | Path
) -> list[str]:
    return [
        "ffmpeg",
        "-y",
        "-hide_banner",
        "-i",
        str(video_path),
        "-vf",
        f"select=eq(n\\,{frame_number})",
        "-vframes",
        "1",
        str(output_path),
    ]


def build_png_to_video_command(
    input_pattern: str | Path,
    output_video: str | Path,
    *,
    fps: float,
    codec: str = "264",
) -> list[str]:
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
        "-preset",
        "medium",
        "-pix_fmt",
        "yuv420p",
        *extra,
        str(output_video),
    ]


def _run_ffmpeg(cmd: list[str]) -> None:
    subprocess.run(cmd, check=True)


def extract_png_from_video(
    video_path: str | Path,
    output_dir: str | Path,
    *,
    pattern: str = DEFAULT_PATTERN,
) -> int:
    """Extract all frames from one video into ``output_dir``. Returns frame count."""
    video_path = Path(video_path)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    width, height, fps = get_video_info(video_path)
    output_pattern = output_dir / pattern

    try:
        print(f"Processing {video_path.name} (hwaccel auto)...")
        _run_ffmpeg(
            build_extract_png_command(
                video_path, output_pattern, width=width, height=height, hwaccel=True
            )
        )
    except subprocess.CalledProcessError:
        print("Hardware acceleration failed, trying software decoder...")
        _run_ffmpeg(
            build_extract_png_command(
                video_path, output_pattern, width=width, height=height, hwaccel=False
            )
        )

    total_frames = len([f for f in output_dir.iterdir() if f.suffix.lower() == ".png"])
    info_path = output_dir / "video_info.txt"
    info_path.write_text(
        f"Original video: {video_path.name}\n"
        f"FPS: {fps}\n"
        f"Resolution: {width}x{height}\n"
        f"Total frames: {total_frames}\n"
        f"Extraction timestamp: {_timestamp()}\n",
        encoding="utf-8",
    )
    print(f"Extracted {total_frames} frames from {video_path.name} → {output_dir}")
    print(f"Resolution: {width}x{height}, FPS: {fps}")
    return total_frames


def extract_png_from_videos(
    src_dir: str | Path,
    *,
    output_dir: str | Path | None = None,
    pattern: str = DEFAULT_PATTERN,
) -> Path:
    """Batch-extract PNGs for every video in ``src_dir``."""
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    videos = list_videos_in_dir(src_dir)
    if not videos:
        raise FileNotFoundError(f"No video files found in {src_dir}")

    dest = Path(output_dir) if output_dir else src_dir / f"vaila_extractpng_{_timestamp()}"
    dest.mkdir(parents=True, exist_ok=True)

    print(f"Starting extraction of PNG frames ({len(videos)} video(s))...")
    for video in videos:
        out = dest / f"{video.stem}_png"
        extract_png_from_video(video, out, pattern=pattern)
    print(f"Done. Output: {dest}")
    return dest


def extract_select_frames(
    video_file: str | Path,
    frame_numbers: list[int],
    *,
    output_dir: str | Path | None = None,
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

    print(f"Extracting {len(frame_numbers)} frame(s) from {video_file.name}...")
    for frame_number in frame_numbers:
        output_path = dest / f"frame_{frame_number:03d}.png"
        _run_ffmpeg(build_select_frame_command(video_file, frame_number, output_path))
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
) -> Path:
    src_dir = Path(src_dir)
    if not src_dir.is_dir():
        raise FileNotFoundError(f"Not a directory: {src_dir}")

    dest = Path(output_dir) if output_dir else src_dir / f"vaila_png2videos_{_timestamp()}"
    dest.mkdir(parents=True, exist_ok=True)

    png_dirs = _png_dirs_to_process(src_dir, exclude=dest)
    if not png_dirs:
        raise FileNotFoundError(f"No PNG sequences found under {src_dir}")

    print(f"Creating videos from {len(png_dirs)} PNG sequence(s) @ {fps} fps...")
    for png_dir in png_dirs:
        output_video = dest / f"{png_dir.name}.mp4"
        input_pattern = png_dir / pattern
        _run_ffmpeg(build_png_to_video_command(input_pattern, output_video, fps=fps, codec=codec))
        print(f"  video created: {output_video}")
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
) -> list[str]:
    argv = ["uv", "run", "vaila/extractpng.py", mode, "-i", input_path]
    if output_path:
        argv.extend(["-o", output_path])
    if mode == "extract":
        argv.extend(["--pattern", pattern])
    elif mode == "create":
        argv.extend(["--fps", str(fps), "--codec", str(codec), "--pattern", pattern])
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

        self.root.title("vailá — Video ↔ PNG")
        self.root.resizable(True, False)
        self.root.minsize(560, 280)

        self.mode = tk.StringVar(value="extract")
        self.input_var = tk.StringVar()
        self.output_var = tk.StringVar()
        self.pattern_var = tk.StringVar(value=DEFAULT_PATTERN)
        self.fps_var = tk.StringVar(value="30")
        self.codec_var = tk.StringVar(value="264")
        self.frames_var = tk.StringVar(value="0,3,5")
        self.status_var = tk.StringVar(value="Choose a mode, set paths, then Run.")

        self._build()
        self._on_mode_change()

    def _build(self) -> None:
        frm = ttk.Frame(self.root, padding=12)
        frm.pack(fill="both", expand=True)

        ttk.Label(frm, text="Video ↔ PNG", font=("TkDefaultFont", 12, "bold")).grid(
            row=0, column=0, columnspan=3, sticky="w", padx=10, pady=4
        )

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

        self.options_frame = ttk.LabelFrame(frm, text="Options", padding=8)
        self.options_frame.grid(row=5, column=0, columnspan=3, sticky="ew", padx=10, pady=4)

        self.pattern_label = ttk.Label(self.options_frame, text="PNG pattern:")
        self.pattern_entry = ttk.Entry(self.options_frame, textvariable=self.pattern_var, width=24)
        self.fps_label = ttk.Label(self.options_frame, text="FPS:")
        self.fps_entry = ttk.Entry(self.options_frame, textvariable=self.fps_var, width=8)
        self.codec_label = ttk.Label(self.options_frame, text="Codec:")
        self.codec_combo = ttk.Combobox(
            self.options_frame,
            textvariable=self.codec_var,
            values=("264", "265"),
            width=6,
            state="readonly",
        )
        self.frames_label = ttk.Label(self.options_frame, text="Frames (e.g. 0,3,5):")
        self.frames_entry = ttk.Entry(self.options_frame, textvariable=self.frames_var, width=28)

        btn_row = ttk.Frame(frm)
        btn_row.grid(row=6, column=0, columnspan=3, sticky="e", padx=10, pady=4)
        ttk.Button(btn_row, text="Run", command=self._run).pack(side="right", padx=4)
        ttk.Button(btn_row, text="Close", command=self.root.destroy).pack(side="right", padx=4)

        ttk.Label(frm, textvariable=self.status_var, wraplength=520).grid(
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
            self.status_var.set("Select a folder of videos. Output is optional.")
        elif mode == "create":
            self.fps_label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.fps_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
            self.codec_label.grid(row=0, column=2, sticky="w", padx=4, pady=2)
            self.codec_combo.grid(row=0, column=3, sticky="w", padx=4, pady=2)
            self.pattern_label.grid(row=1, column=0, sticky="w", padx=4, pady=2)
            self.pattern_entry.grid(row=1, column=1, sticky="w", padx=4, pady=2)
            self.status_var.set("Select a folder of PNG sequences (or a folder of subfolders).")
        else:
            self.frames_label.grid(row=0, column=0, sticky="w", padx=4, pady=2)
            self.frames_entry.grid(row=0, column=1, sticky="w", padx=4, pady=2)
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

        if not input_path:
            messagebox.showerror("Missing input", "Please choose an input path.", parent=self.root)
            return

        try:
            if mode == "extract":
                cli = build_cli_argv(
                    "extract",
                    input_path=input_path,
                    output_path=output_path,
                    pattern=pattern,
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = extract_png_from_videos(input_path, output_dir=output_path, pattern=pattern)
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
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = create_video_from_png(
                    input_path,
                    output_dir=output_path,
                    fps=fps,
                    codec=codec,
                    pattern=pattern,
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
                )
                print_gui_cli_mirror("vaila/extractpng", cli)
                dest = extract_select_frames(input_path, frames, output_dir=output_path)
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
        description="Extract PNG frames from video, or build video from PNG sequences.",
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

    p_cr = sub.add_parser("create", help="PNG sequences → videos")
    p_cr.add_argument("-i", "--input", required=True, help="Directory with PNG folders")
    p_cr.add_argument("-o", "--output", default=None, help="Output directory")
    p_cr.add_argument("--fps", type=float, default=30.0, help="Output FPS")
    p_cr.add_argument(
        "--codec",
        default="264",
        choices=("264", "265"),
        help="H.264 or H.265",
    )
    p_cr.add_argument("--pattern", default=DEFAULT_PATTERN, help="Input PNG pattern")

    p_fr = sub.add_parser("frames", help="Grab specific frames from one video")
    p_fr.add_argument("-i", "--input", required=True, help="Video file")
    p_fr.add_argument("-o", "--output", default=None, help="Output directory")
    p_fr.add_argument(
        "--frames",
        required=True,
        help="Comma-separated frame indices (e.g. 0,3,5)",
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
        extract_png_from_videos(args.input, output_dir=args.output, pattern=args.pattern)
    elif args.command == "create":
        create_video_from_png(
            args.input,
            output_dir=args.output,
            fps=args.fps,
            codec=args.codec,
            pattern=args.pattern,
        )
    elif args.command == "frames":
        extract_select_frames(args.input, parse_frame_list(args.frames), output_dir=args.output)
    else:
        parser.print_help()
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
