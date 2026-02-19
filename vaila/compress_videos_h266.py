"""
compress_videos_h266.py

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

Created by Paulo Santiago
Date: 06 September 2025
Update: 18 February 2026
Version: 0.0.2
Python Version: 3.12

Description:
This script compresses videos in a specified directory to H.266/VVC format using the FFmpeg tool
and the libvvenc encoder. It provides both a GUI and CLI for selecting the directory and
compression settings. The compressed versions are saved in a subdirectory named
'compressed_h266_<timestamp>'.

!!! IMPORTANT !!!
- H.266/VVC encoding is EXTREMELY SLOW and CPU-intensive. Expect long processing times.
- GPU acceleration (like NVIDIA NVENC) is NOT available for VVC in common FFmpeg builds.
  This script uses CPU-only encoding.

Usage:
- GUI: Run the script without arguments to open the GUI.
- CLI: python compress_videos_h266.py --dir /path/to/videos --preset medium --qp 32

Requirements:
- A modern version of FFmpeg (version 7.1+ recommended) that was compiled with libvvenc support.
- The script is designed to work on Windows, Linux, and macOS.

Dependencies:
- Python 3.x
- Tkinter (included with Python)
- FFmpeg with libvvenc (available in PATH)

How to get FFmpeg with libvvenc support:

The version of FFmpeg from standard system repositories (like Ubuntu's apt) usually DOES NOT
include libvvenc. You need to download a pre-compiled "full" or "git" build.

## Windows:
- Download a "full" build from: https://www.gyan.dev/ffmpeg/builds/
- Extract the files and add the `bin` directory to your system's PATH.

## Linux / macOS:
- Download a static "git" build from: https://johnvansickle.com/ffmpeg/
- Extract the `ffmpeg` executable and either place it in a directory in your PATH
  (like /usr/local/bin) or call it directly using its path.

To verify your installation, run this command in your terminal:
  ffmpeg -encoders | findstr vvenc  (Windows)
  ffmpeg -encoders | grep vvenc   (Linux/macOS)

You should see a line containing "libvvenc". If not, your FFmpeg build is not compatible.
"""

import argparse
import os
import subprocess
import sys
import tempfile
import tkinter as tk
from datetime import datetime
from tkinter import filedialog, messagebox

from rich import print

from vaila.ffmpeg_utils import get_ffmpeg_path

# Resolve FFmpeg path (local static → venv → system)
FFMPEG = get_ffmpeg_path()

# --- Options (shared between GUI and CLI) ---
PRESET_OPTIONS = [
    "ultrafast",
    "superfast",
    "veryfast",
    "faster",
    "fast",
    "medium",
    "slow",
    "slower",
    "veryslow",
]

RESOLUTION_OPTIONS = [
    "original",
    "3840x2160",
    "2560x1440",
    "1920x1080",
    "1280x720",
    "854x480",
    "640x360",
]


def check_libvvenc_available():
    """Check if FFmpeg was compiled with libvvenc support.

    Returns:
        bool: True if libvvenc is available.
    """
    try:
        result = subprocess.run(
            [FFMPEG, "-encoders"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        if "libvvenc" in result.stdout:
            print("[bold green]libvvenc encoder found in FFmpeg.[/bold green]")
            return True
        else:
            print(
                "[bold red]libvvenc encoder NOT found in your FFmpeg installation.[/bold red]\n"
                "[yellow]Your FFmpeg was not compiled with libvvenc support.\n"
                "See the script header for instructions on getting a compatible FFmpeg build.[/yellow]"
            )
            return False
    except FileNotFoundError:
        print("[bold red]FFmpeg not found. Please install FFmpeg first.[/bold red]")
        return False
    except Exception as e:
        print(f"[red]Error checking FFmpeg encoders: {e}[/red]")
        return False


def find_videos(directory):
    """Find all video files in the specified directory (non-recursive)."""
    video_extensions = (".mp4", ".avi", ".mov", ".mkv", ".wmv")
    video_files = []
    for file in sorted(os.listdir(directory)):
        if file.lower().endswith(video_extensions):
            video_files.append(os.path.join(directory, file))
    return video_files


def create_temp_file_with_videos(video_files):
    """Create temporary file with list of video paths."""
    temp = tempfile.NamedTemporaryFile(mode="w", delete=False, suffix=".txt")
    for video in video_files:
        temp.write(f"{video}\n")
    temp.close()
    return temp.name


def run_compress_videos_h266(input_list, output_dir, preset, qp, resolution):
    """Run the actual H.266/VVC compression using libvvenc (CPU-only).

    Returns:
        tuple: (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0

    print("\n[bold cyan]Compression Parameters:[/bold cyan]")
    print(f"  Preset: {preset}")
    print(f"  QP: {qp}")
    print(f"  Resolution: {resolution}")
    print("  Encoder: libvvenc (CPU-only)")

    print(
        "\n[bold red]!!! ATTENTION !!![/bold red]\n"
        "[yellow]H.266/VVC encoding is [u]extremely slow[/u]. "
        "This process may take many hours.[/yellow]\n"
    )

    os.makedirs(output_dir, exist_ok=True)

    with open(input_list) as f:
        video_paths = [line.strip() for line in f if line.strip()]

    if not video_paths:
        print("[red]No video files found in input list.[/red]")
        return 0, 0

    print(f"[bold]Processing {len(video_paths)} video(s)...[/bold]\n")

    for i, video_path in enumerate(video_paths, 1):
        basename = os.path.basename(video_path)
        output_path = os.path.join(
            output_dir, os.path.splitext(basename)[0] + "_h266.mp4"
        )

        try:
            print(f"[{i}/{len(video_paths)}] {basename}")

            # Base command
            cmd = [FFMPEG, "-y", "-i", video_path]

            # Add scale filter if resolution is not original
            if resolution != "original":
                w, h = resolution.split("x")
                scale_filter = (
                    f"scale={w}:{h}:force_original_aspect_ratio=decrease,"
                    f"pad={w}:{h}:(ow-iw)/2:(oh-ih)/2"
                )
                cmd.extend(["-vf", scale_filter])

            # VVC CPU encoding settings using libvvenc
            vvenc_params = f"preset={preset}:qp={qp}"
            cmd.extend([
                "-c:v", "libvvenc",
                "-vvenc-params", vvenc_params,
            ])

            # Add audio settings and output path
            cmd.extend(["-c:a", "copy", "-hide_banner", "-nostats", output_path])

            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
                output_size_mb = os.path.getsize(output_path) / (1024 * 1024)
                success_count += 1
                print(f"  [green][OK] Done ({output_size_mb:.1f} MB)[/green]")
            else:
                failure_count += 1
                print("  [red][FAIL] Output file is empty or missing[/red]")

        except subprocess.CalledProcessError as e:
            failure_count += 1
            print(f"  [red][FAIL] Failed: {e}[/red]")
            if e.stderr:
                stderr_text = e.stderr if isinstance(e.stderr, str) else e.stderr.decode()
                print(f"  [red]  ffmpeg stderr: {stderr_text[:200]}[/red]")

    print(f"\n[bold]Results: {success_count} succeeded, {failure_count} failed.[/bold]")
    return success_count, failure_count


# ---------------------------------------------------------------------------
# GUI Mode
# ---------------------------------------------------------------------------


def get_compression_parameters():
    """Create a dialog window for user to select VVC compression settings.

    Returns:
        dict or None: Parameters dict with keys preset, qp, resolution.
    """
    params = {}

    # Ensure a hidden root exists
    try:
        root = tk._default_root  # noqa: SLF001
        if root is None:
            root = tk.Tk()
            root.withdraw()
    except Exception:
        root = tk.Tk()
        root.withdraw()

    dialog = tk.Toplevel()
    dialog.title("VVC/H.266 Compression Settings")
    dialog.grab_set()

    main_frame = tk.Frame(dialog, padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)

    tk.Label(
        main_frame, text="H.266/VVC Compression Settings", font=("Arial", 12, "bold")
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

    # 1. Preset
    tk.Label(main_frame, text="Preset (enter number):", font=("Arial", 10, "bold")).grid(
        row=1, column=0, sticky="w", pady=5
    )
    preset_var = tk.StringVar(value="6")
    tk.Entry(main_frame, textvariable=preset_var, width=5).grid(
        row=1, column=1, sticky="w", pady=5
    )
    preset_help_text = "Options:\n" + "   ".join(
        [f"{i + 1}={p}" for i, p in enumerate(PRESET_OPTIONS)]
    )
    tk.Label(
        main_frame, text=preset_help_text, font=("Arial", 8, "italic"),
        justify="left", wraplength=400,
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=20)

    # 2. QP Value
    tk.Label(main_frame, text="QP Value (0-51):", font=("Arial", 10, "bold")).grid(
        row=3, column=0, sticky="w", pady=5
    )
    qp_var = tk.StringVar(value="23")
    tk.Entry(main_frame, textvariable=qp_var, width=5).grid(
        row=3, column=1, sticky="w", pady=5
    )
    tk.Label(
        main_frame,
        text="Lower = better quality. 32 is a good default.",
        font=("Arial", 8, "italic"),
    ).grid(row=4, column=0, columnspan=2, sticky="w", padx=20)

    # 3. Resolution
    tk.Label(main_frame, text="Resolution (enter number):", font=("Arial", 10, "bold")).grid(
        row=5, column=0, sticky="w", pady=5
    )
    resolution_var = tk.StringVar(value="1")
    tk.Entry(main_frame, textvariable=resolution_var, width=5).grid(
        row=5, column=1, sticky="w", pady=5
    )
    resolution_help_text = "Options:\n" + "   ".join(
        [f"{i + 1}={r}" for i, r in enumerate(RESOLUTION_OPTIONS)]
    )
    tk.Label(
        main_frame, text=resolution_help_text, font=("Arial", 8, "italic"),
        justify="left", wraplength=400,
    ).grid(row=6, column=0, columnspan=2, sticky="w", padx=20)

    tk.Frame(main_frame, height=1, bg="gray").grid(
        row=9, column=0, columnspan=2, sticky="ew", pady=15
    )
    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=10, column=0, columnspan=2, pady=10)

    def on_ok():
        try:
            preset_idx = int(preset_var.get().strip())
            if not (1 <= preset_idx <= len(PRESET_OPTIONS)):
                raise ValueError(f"Preset must be between 1 and {len(PRESET_OPTIONS)}")
            preset = PRESET_OPTIONS[preset_idx - 1]

            qp = int(qp_var.get().strip())
            if not (0 <= qp <= 51):
                raise ValueError("QP must be between 0 and 51")

            resolution_idx = int(resolution_var.get().strip())
            if not (1 <= resolution_idx <= len(RESOLUTION_OPTIONS)):
                raise ValueError(
                    f"Resolution must be between 1 and {len(RESOLUTION_OPTIONS)}"
                )
            resolution = RESOLUTION_OPTIONS[resolution_idx - 1]

            params["preset"] = preset
            params["qp"] = qp
            params["resolution"] = resolution

            confirm_msg = (
                f"Selected compression settings:\n\n"
                f"• Preset: {preset}\n"
                f"• QP: {qp}\n"
                f"• Resolution: {resolution}\n\n"
                f"Remember: H.266 compression is VERY SLOW.\n\n"
                f"Continue?"
            )
            if messagebox.askyesno("Confirm Settings", confirm_msg):
                dialog.destroy()
        except ValueError as e:
            messagebox.showerror("Input Error", str(e))

    def on_cancel():
        dialog.destroy()

    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(
        side="left", padx=5
    )
    dialog.wait_window()
    return params if params else None


def compress_videos_h266_gui():
    """Main function to run the GUI and compression process."""
    print(f"Running script: {os.path.basename(__file__)}")
    print("Starting compress_videos_h266_gui...")

    # Check encoder availability BEFORE showing any dialogs
    if not check_libvvenc_available():
        messagebox.showerror(
            "Encoder Not Available",
            "Your FFmpeg installation does not include the libvvenc encoder.\n\n"
            "H.266/VVC compression requires a special FFmpeg build with libvvenc.\n"
            "See the script documentation for instructions on getting a compatible build.",
        )
        return

    compression_config = get_compression_parameters()
    if not compression_config:
        print("Operation canceled by user.")
        return

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(video_directory, f"compressed_h266_{timestamp}")

    video_files = find_videos(video_directory)
    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    temp_file_path = create_temp_file_with_videos(video_files)

    success_count, failure_count = run_compress_videos_h266(
        temp_file_path,
        output_directory,
        preset=compression_config["preset"],
        qp=compression_config["qp"],
        resolution=compression_config["resolution"],
    )

    os.remove(temp_file_path)

    messagebox.showinfo(
        "Success",
        f"Video compression completed.\n\n{success_count} succeeded, {failure_count} failed.",
    )


# ---------------------------------------------------------------------------
# CLI Mode
# ---------------------------------------------------------------------------


def build_parser():
    """Build the argparse parser for CLI mode."""
    parser = argparse.ArgumentParser(
        description="Compress videos to H.266/VVC format using FFmpeg + libvvenc.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress with default settings (medium preset, QP 32, original resolution)
  python -m vaila.compress_videos_h266 --dir /path/to/videos

  # Compress to 1080p with slow preset
  python -m vaila.compress_videos_h266 --dir /path/to/videos --preset slow --qp 28 --resolution 1920x1080

  # Launch GUI mode
  python -m vaila.compress_videos_h266

Note: H.266/VVC encoding requires FFmpeg compiled with libvvenc.
      Standard FFmpeg packages usually do NOT include libvvenc.
        """,
    )
    parser.add_argument(
        "--dir", type=str, help="Directory containing videos to compress."
    )
    parser.add_argument(
        "--preset",
        type=str,
        default="medium",
        choices=PRESET_OPTIONS,
        help="Encoding preset (default: medium).",
    )
    parser.add_argument(
        "--qp",
        type=int,
        default=23,
        help="Quantization Parameter 0-51 (default: 23). Lower = better quality.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="original",
        help="Output resolution WIDTHxHEIGHT or 'original' (default: original).",
    )
    return parser


def main():
    """Entry point: CLI if arguments given, GUI otherwise."""
    if len(sys.argv) == 1:
        compress_videos_h266_gui()
        return

    parser = build_parser()
    args = parser.parse_args()

    if not args.dir:
        parser.error("--dir is required in CLI mode.")

    if not os.path.isdir(args.dir):
        parser.error(f"Directory does not exist: {args.dir}")

    if args.resolution != "original" and "x" not in args.resolution:
        parser.error(
            f"Invalid resolution format: {args.resolution}. Use WIDTHxHEIGHT or 'original'."
        )

    if not (0 <= args.qp <= 51):
        parser.error(f"QP must be between 0 and 51, got {args.qp}.")

    # Check encoder before starting
    if not check_libvvenc_available():
        sys.exit(1)

    video_files = find_videos(args.dir)
    if not video_files:
        print(f"[red]No video files found in {args.dir}[/red]")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) in {args.dir}")

    temp_file_path = create_temp_file_with_videos(video_files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(args.dir, f"compressed_h266_{timestamp}")

    try:
        success_count, failure_count = run_compress_videos_h266(
            temp_file_path,
            output_directory,
            preset=args.preset,
            qp=args.qp,
            resolution=args.resolution,
        )
    finally:
        os.remove(temp_file_path)

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
