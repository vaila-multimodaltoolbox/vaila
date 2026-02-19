"""
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

Licensed under GNU Lesser General Public License v3.0

compress_videos_h265.py

Description:
This script compresses videos in a specified directory to H.265/HEVC format using the FFmpeg tool.
It provides both a GUI and CLI for selecting the directory containing the videos and processes
each video, saving the compressed versions in a subdirectory named 'compressed_h265_<timestamp>'.
The script supports GPU acceleration using NVIDIA NVENC if available or falls back to CPU encoding
with libx265.

Usage:
- GUI: Run the script without arguments to open the GUI.
- CLI: python compress_videos_h265.py --dir /path/to/videos --preset medium --crf 28

Requirements:
- FFmpeg must be installed and accessible in the system PATH.
- The script is designed to work on Windows, Linux, and macOS.

Dependencies:
- Python 3.x
- Tkinter (included with Python)
- FFmpeg (available in PATH)

NVIDIA GPU Installation and FFmpeg NVENC Support

To use NVIDIA GPU acceleration for video encoding in FFmpeg, follow the steps below for your
operating system:

## Windows:
1. **Install NVIDIA Drivers**:
   - Download and install the latest NVIDIA drivers from the official site:
     https://www.nvidia.com/Download/index.aspx.
   - Ensure your GPU supports NVENC (Kepler series or newer).

2. **Install FFmpeg**:
   - Download the FFmpeg build with NVENC support from: https://www.gyan.dev/ffmpeg/builds/.
   - Extract the files, add the `bin` directory to your system's PATH, and verify installation by
     running:
     ```bash
     ffmpeg -encoders | findstr nvenc
     ```
   - Look for `h264_nvenc` and `hevc_nvenc` in the output.

## Linux:
1. **Install NVIDIA Drivers**:
   - Install the appropriate NVIDIA drivers for your GPU.
   - Verify your GPU and driver installation with:
     ```bash
     nvidia-smi
     ```

2. **Install or Compile FFmpeg with NVENC Support**:
   - Check if `hevc_nvenc` is available:
     ```bash
     ffmpeg -encoders | grep nvenc
     ```

## macOS:
- Recent versions of macOS do not support NVIDIA GPUs; NVENC acceleration is not available.
- The script will default to CPU-based encoding on macOS.

Note:
- Ensure that FFmpeg is installed and accessible in your system PATH.
- This process may take several hours depending on the size of the videos and the performance
  of your computer.
- H.265 encoding is significantly slower than H.264 but produces smaller files at the same
  quality level.
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

from vaila.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path

# Resolve FFmpeg/FFprobe paths (local static → venv → system)
FFMPEG = get_ffmpeg_path()
FFPROBE = get_ffprobe_path()

# --- Preset and resolution options (shared between GUI and CLI) ---
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

NVENC_PRESET_MAP = {
    "ultrafast": "p1",
    "superfast": "p2",
    "veryfast": "p3",
    "faster": "p4",
    "fast": "p4",
    "medium": "p5",
    "slow": "p6",
    "slower": "p7",
    "veryslow": "p7",
}

RESOLUTION_OPTIONS = [
    "original",
    "3840x2160",
    "2560x1440",
    "1920x1080",
    "1280x720",
    "854x480",
    "640x360",
]


def is_nvidia_gpu_available():
    """Check if an NVIDIA GPU with working HEVC NVENC is available.

    Uses nvidia-smi (cross-platform) to detect NVIDIA GPUs, then verifies
    that hevc_nvenc actually works by running a quick test encode.
    """
    try:
        result = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=5,
        )
        if result.returncode != 0:
            return False

        gpu_name = result.stdout.strip()
        if not gpu_name:
            return False

        print(f"[bold green]Detected NVIDIA GPU: {gpu_name}[/bold green]")

        # Verify hevc_nvenc actually works
        test_cmd = [
            FFMPEG,
            "-f", "lavfi",
            "-i", "color=black:s=32x32:r=1:d=1",
            "-c:v", "hevc_nvenc",
            "-f", "null",
            "-hide_banner",
            "-nostats",
            "-",
        ]
        test_result = subprocess.run(
            test_cmd,
            capture_output=True,
            text=True,
            timeout=10,
        )
        if test_result.returncode == 0:
            print("[bold green]NVIDIA NVENC HEVC encoder verified working.[/bold green]")
            return True
        else:
            print("[yellow]NVIDIA GPU found but hevc_nvenc encoder not working.[/yellow]")
            return False

    except FileNotFoundError:
        return False
    except subprocess.TimeoutExpired:
        print("[yellow]GPU detection timed out.[/yellow]")
        return False
    except Exception as e:
        print(f"[yellow]Error checking for NVIDIA GPU: {e}[/yellow]")
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


def run_compress_videos_h265(input_list, output_dir, preset, crf, resolution, use_gpu):
    """Run the actual H.265/HEVC compression.

    Returns:
        tuple: (success_count, failure_count)
    """
    success_count = 0
    failure_count = 0

    print("\n[bold cyan]Compression Parameters:[/bold cyan]")
    print(f"  Preset: {preset}")
    print(f"  CRF: {crf}")
    print(f"  Resolution: {resolution}")
    print(f"  Use GPU: {use_gpu}")

    os.makedirs(output_dir, exist_ok=True)

    with open(input_list) as f:
        video_paths = [line.strip() for line in f if line.strip()]

    if not video_paths:
        print("[red]No video files found in input list.[/red]")
        return 0, 0

    print(f"\n[bold]Processing {len(video_paths)} video(s)...[/bold]")
    print(
        "[yellow]H.265 encoding is significantly slower than H.264. "
        "This process might take several hours.[/yellow]\n"
    )

    for i, video_path in enumerate(video_paths, 1):
        basename = os.path.basename(video_path)
        output_path = os.path.join(
            output_dir, os.path.splitext(basename)[0] + "_h265.mp4"
        )

        try:
            # Get original video resolution for logging
            try:
                cmd_probe = [
                    FFPROBE,
                    "-v", "error",
                    "-select_streams", "v:0",
                    "-show_entries", "stream=width,height",
                    "-of", "csv=s=x:p=0",
                    video_path,
                ]
                original_resolution = subprocess.check_output(cmd_probe).decode().strip()
                print(f"[{i}/{len(video_paths)}] {basename} ({original_resolution})")
            except Exception:
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

            # Add encoding settings based on GPU availability
            if use_gpu:
                nvenc_preset = NVENC_PRESET_MAP.get(preset, "p5")
                cmd.extend([
                    "-c:v", "hevc_nvenc",
                    "-preset", nvenc_preset,
                    "-b:v", "5M",
                    "-maxrate", "5M",
                    "-bufsize", "10M",
                ])
            else:
                cmd.extend([
                    "-c:v", "libx265",
                    "-preset", preset,
                    "-crf", str(crf),
                ])

            # Add audio settings and output path
            cmd.extend(["-c:a", "copy", "-hide_banner", "-nostats", output_path])

            print(f"  Command: {' '.join(cmd)}")
            subprocess.run(cmd, check=True)

            # Verify output
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
    """Create a dialog window where user selects compression options.

    Returns:
        dict or None: Parameters dict with keys preset, crf, resolution, use_gpu.
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
    dialog.title("Video Compression Settings")
    dialog.grab_set()

    main_frame = tk.Frame(dialog, padx=20, pady=15)
    main_frame.pack(fill="both", expand=True)

    tk.Label(
        main_frame, text="H.265/HEVC Video Compression Settings", font=("Arial", 12, "bold")
    ).grid(row=0, column=0, columnspan=2, sticky="w", pady=(0, 15))

    # 1. Preset
    tk.Label(main_frame, text="Preset (enter number):", font=("Arial", 10, "bold")).grid(
        row=1, column=0, sticky="w", pady=5
    )
    preset_var = tk.StringVar(value="6")
    tk.Entry(main_frame, textvariable=preset_var, width=5).grid(
        row=1, column=1, sticky="w", pady=5
    )

    preset_help_text = "Options:\n"
    for i, preset in enumerate(PRESET_OPTIONS, 1):
        preset_help_text += f"{i} = {preset}"
        if i < len(PRESET_OPTIONS):
            preset_help_text += "   "
            if i % 3 == 0:
                preset_help_text += "\n"

    tk.Label(
        main_frame, text=preset_help_text, font=("Arial", 8, "italic"), justify="left"
    ).grid(row=2, column=0, columnspan=2, sticky="w", padx=20)

    # 2. CRF
    tk.Label(main_frame, text="CRF Value (0-51):", font=("Arial", 10, "bold")).grid(
        row=3, column=0, sticky="w", pady=5
    )
    crf_var = tk.StringVar(value="23")  # Same default as H.264 for fair comparison
    tk.Entry(main_frame, textvariable=crf_var, width=5).grid(
        row=3, column=1, sticky="w", pady=5
    )
    tk.Label(
        main_frame,
        text="Lower = better quality. 28 is default for H.265",
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

    resolution_help_text = "Options:\n"
    for i, res in enumerate(RESOLUTION_OPTIONS, 1):
        resolution_help_text += f"{i} = {res}"
        if i < len(RESOLUTION_OPTIONS):
            resolution_help_text += "   "
            if i % 2 == 0:
                resolution_help_text += "\n"

    tk.Label(
        main_frame, text=resolution_help_text, font=("Arial", 8, "italic"), justify="left"
    ).grid(row=6, column=0, columnspan=2, sticky="w", padx=20)

    # 4. GPU
    tk.Label(main_frame, text="Use GPU (enter number):", font=("Arial", 10, "bold")).grid(
        row=7, column=0, sticky="w", pady=5
    )
    gpu_var = tk.StringVar(value="2")
    tk.Entry(main_frame, textvariable=gpu_var, width=5).grid(
        row=7, column=1, sticky="w", pady=5
    )
    tk.Label(
        main_frame,
        text="Options: 1 = Yes (NVIDIA GPUs only)   2 = No (CPU encoding)",
        font=("Arial", 8, "italic"),
    ).grid(row=8, column=0, columnspan=2, sticky="w", padx=20)

    tk.Frame(main_frame, height=1, bg="gray").grid(
        row=9, column=0, columnspan=2, sticky="ew", pady=15
    )

    button_frame = tk.Frame(main_frame)
    button_frame.grid(row=10, column=0, columnspan=2, pady=10)

    def on_ok():
        try:
            preset_idx = int(preset_var.get().strip())
            if not (1 <= preset_idx <= len(PRESET_OPTIONS)):
                messagebox.showerror(
                    "Error", f"Preset number must be between 1 and {len(PRESET_OPTIONS)}"
                )
                return
            preset = PRESET_OPTIONS[preset_idx - 1]

            crf = int(crf_var.get().strip())
            if not (0 <= crf <= 51):
                messagebox.showerror("Error", "CRF value must be between 0 and 51")
                return

            resolution_idx = int(resolution_var.get().strip())
            if not (1 <= resolution_idx <= len(RESOLUTION_OPTIONS)):
                messagebox.showerror(
                    "Error",
                    f"Resolution number must be between 1 and {len(RESOLUTION_OPTIONS)}",
                )
                return
            resolution = RESOLUTION_OPTIONS[resolution_idx - 1]

            gpu_choice = int(gpu_var.get().strip())
            if gpu_choice not in [1, 2]:
                messagebox.showerror("Error", "GPU option must be 1 (Yes) or 2 (No)")
                return
            use_gpu = gpu_choice == 1

            params["preset"] = preset
            params["crf"] = crf
            params["resolution"] = resolution
            params["use_gpu"] = use_gpu

            confirm_msg = (
                f"Selected compression settings:\n\n"
                f"• Preset: {preset}\n"
                f"• CRF: {crf}\n"
                f"• Resolution: {resolution}\n"
                f"• Use GPU: {'Yes' if use_gpu else 'No'}\n\n"
                f"Note: H.265 encoding is slower than H.264.\n\n"
                f"Continue with these settings?"
            )

            if messagebox.askyesno("Confirm Settings", confirm_msg):
                dialog.destroy()

        except ValueError:
            messagebox.showerror("Error", "Please enter valid numbers for all fields.")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving parameters: {e}")

    def on_cancel():
        dialog.destroy()

    tk.Button(button_frame, text="OK", command=on_ok, width=10).pack(side="left", padx=5)
    tk.Button(button_frame, text="Cancel", command=on_cancel, width=10).pack(
        side="left", padx=5
    )

    def show_help():
        help_text = """
COMPRESSION SETTINGS HELP:

PRESET: (speed vs quality tradeoff)
1 = ultrafast: Fastest encoding, lowest quality
2 = superfast: Very fast encoding, lower quality
3 = veryfast: Fast encoding, decent quality
4 = faster: Reasonably fast encoding, good quality
5 = fast: Balanced encoding speed and quality
6 = medium: Default preset with good balance
7 = slow: Better quality, slower encoding
8 = slower: Very good quality, much slower encoding
9 = veryslow: Best quality, very slow encoding

CRF VALUE (0-51): (quality setting)
- Lower values = better quality, larger files
- Higher values = lower quality, smaller files
- 0 = lossless (largest files)
- 28 = default value for H.265 (good balance)
- 51 = worst quality (smallest files)
- Note: For same quality, H.265 typically needs ~6 higher CRF than H.264

RESOLUTION: (output size)
1 = original: Keep the original video resolution
2 = 3840x2160: 4K UHD
3 = 2560x1440: 2K QHD
4 = 1920x1080: Full HD
5 = 1280x720: HD
6 = 854x480: SD
7 = 640x360: Low resolution

GPU ACCELERATION:
1 = Yes: Use NVIDIA GPU for encoding (faster)
2 = No: Use CPU for encoding (works on all systems)

NOTE: H.265 encoding is significantly slower than H.264 but produces smaller files
at the same quality level. Expect longer encoding times, especially with CPU encoding.
        """
        help_dialog = tk.Toplevel(dialog)
        help_dialog.title("Compression Settings Help")
        help_dialog.transient(dialog)
        help_dialog.grab_set()

        text_widget = tk.Text(help_dialog, wrap="word", width=70, height=30)
        text_widget.pack(padx=20, pady=20, fill="both", expand=True)
        text_widget.insert("1.0", help_text)
        text_widget.config(state="disabled")

        tk.Button(help_dialog, text="Close", command=help_dialog.destroy).pack(pady=10)

    tk.Button(main_frame, text="Help", command=show_help).grid(
        row=11, column=0, columnspan=2, pady=5
    )

    dialog.update_idletasks()
    width = dialog.winfo_width()
    height = dialog.winfo_height()
    x = (dialog.winfo_screenwidth() // 2) - (width // 2)
    y = (dialog.winfo_screenheight() // 2) - (height // 2)
    dialog.geometry(f"{width}x{height}+{x}+{y}")

    dialog.wait_window()
    return params if params else None


def compress_videos_h265_gui():
    """Main function to run the GUI and compression process."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")
    print("Starting compress_videos_h265_gui...")

    compression_config = get_compression_parameters()

    if not compression_config:
        print("User canceled the operation")
        return

    video_directory = filedialog.askdirectory(
        title="Select the directory containing videos to compress"
    )
    if not video_directory:
        messagebox.showerror("Error", "No directory selected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(video_directory, f"compressed_h265_{timestamp}")

    use_gpu = compression_config["use_gpu"] and is_nvidia_gpu_available()
    if compression_config["use_gpu"] and not use_gpu:
        print("GPU acceleration requested but NVENC encoder not available. Using CPU instead.")
        messagebox.showwarning(
            "GPU Not Available",
            "GPU acceleration was requested but the hevc_nvenc encoder is not available.\n\n"
            "This can happen if:\n"
            "• No NVIDIA GPU is present\n"
            "• Your FFmpeg was not compiled with NVENC support\n"
            "• NVIDIA drivers are outdated\n\n"
            "Compression will proceed using CPU (libx265) instead.",
        )

    video_files = find_videos(video_directory)

    if not video_files:
        messagebox.showerror("Error", "No video files found.")
        return

    temp_file_path = create_temp_file_with_videos(video_files)

    success_count, failure_count = run_compress_videos_h265(
        temp_file_path,
        output_directory,
        preset=compression_config["preset"],
        crf=compression_config["crf"],
        resolution=compression_config["resolution"],
        use_gpu=use_gpu,
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
        description="Compress videos to H.265/HEVC format using FFmpeg.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compress with default settings (medium preset, CRF 28, original resolution, CPU)
  python -m vaila.compress_videos_h265 --dir /path/to/videos

  # Compress with GPU acceleration
  python -m vaila.compress_videos_h265 --dir /path/to/videos --gpu

  # Compress to 1080p with slow preset for better quality
  python -m vaila.compress_videos_h265 --dir /path/to/videos --preset slow --crf 24 --resolution 1920x1080

  # Launch GUI mode
  python -m vaila.compress_videos_h265
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
        help="Encoding preset (default: medium). Slower = better quality.",
    )
    parser.add_argument(
        "--crf",
        type=int,
        default=23,
        help="Constant Rate Factor 0-51 (default: 23). Lower = better quality.",
    )
    parser.add_argument(
        "--resolution",
        type=str,
        default="original",
        help="Output resolution WIDTHxHEIGHT or 'original' (default: original).",
    )
    parser.add_argument(
        "--gpu", action="store_true", default=False,
        help="Use NVIDIA GPU (NVENC) for encoding if available.",
    )
    parser.add_argument(
        "--no-gpu", action="store_true", default=False,
        help="Force CPU encoding (overrides --gpu).",
    )
    return parser


def main():
    """Entry point: CLI if arguments given, GUI otherwise."""
    if len(sys.argv) == 1:
        compress_videos_h265_gui()
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

    if not (0 <= args.crf <= 51):
        parser.error(f"CRF must be between 0 and 51, got {args.crf}.")

    use_gpu = args.gpu and not args.no_gpu
    if use_gpu:
        use_gpu = is_nvidia_gpu_available()
        if not use_gpu:
            print("[yellow]GPU requested but not available. Using CPU encoding.[/yellow]")

    video_files = find_videos(args.dir)
    if not video_files:
        print(f"[red]No video files found in {args.dir}[/red]")
        sys.exit(1)

    print(f"Found {len(video_files)} video(s) in {args.dir}")

    temp_file_path = create_temp_file_with_videos(video_files)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_directory = os.path.join(args.dir, f"compressed_h265_{timestamp}")

    try:
        success_count, failure_count = run_compress_videos_h265(
            temp_file_path,
            output_directory,
            preset=args.preset,
            crf=args.crf,
            resolution=args.resolution,
            use_gpu=use_gpu,
        )
    finally:
        os.remove(temp_file_path)

    if failure_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    main()
