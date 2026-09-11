"""
================================================================================
YouTube High Quality Downloader - vaila_ytdown.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Create: 10 October 2025
Update Date: 11 September 2026
Version: 0.3.137

Description:
------------
Review editable URLs (Load TXT only fills the list), select destination and MP4
or MP3, then Download. Cancellation is cooperative; completed files are kept.
MP4 uses yt-dlp bestvideo+bestaudio/best; MP3 uses bestaudio/best, converted at
192 kbps. Both need ffmpeg. GUI workers send queued events to Tk's main thread.
CLI: python -m vaila.vaila_ytdown --file urls.txt --output /data --audio-only --no-gui
Use --debug for technical details. See help/vaila_ytdown.md for outputs.

License:
---------
This script is licensed under the GNU Affero General Public License v3.0.
See the LICENSE file for more details.
Visit the project repository: https://github.com/vaila-multimodaltoolbox
"""

import argparse
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import webbrowser
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

try:
    from .task_feedback import Feedback, WorkerTask, command_text, redact
except ImportError:
    from task_feedback import Feedback, WorkerTask, command_text, redact

# Try to import yt-dlp
try:
    import yt_dlp
except ImportError:
    print("Error: yt-dlp package is required. Install it with:")
    print("pip install yt-dlp")
    sys.exit(1)

# Try to import tkinter for GUI
try:
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    TKINTER_AVAILABLE = True
except ImportError:
    TKINTER_AVAILABLE = False
    print("Warning: tkinter not available. Running in CLI mode only.")


# Preferred JS runtimes for YouTube EJS challenges (yt-dlp wiki/EJS).
_JS_RUNTIME_CANDIDATES = ("deno", "node", "qjs")


def get_help_html_path():
    """Return absolute path to vaila_ytdown.html (next to this script)."""
    script_dir = Path(__file__).resolve().parent
    return script_dir / "help" / "vaila_ytdown.html"


def detect_js_runtimes() -> dict[str, dict[str, str]]:
    """Return yt-dlp ``js_runtimes`` map for Deno/Node/QuickJS found on PATH.

    Deno is preferred (enabled by default upstream). Node needs explicit enable.
    """
    runtimes: dict[str, dict[str, str]] = {}
    for name in _JS_RUNTIME_CANDIDATES:
        path = shutil.which(name)
        if not path:
            continue
        key = "quickjs" if name == "qjs" else name
        runtimes[key] = {"path": path}
    return runtimes


def build_ytdlp_base_opts(**overrides) -> dict:
    """Shared yt-dlp options: cert skip, JS runtimes, EJS remote fallback."""
    opts: dict = {
        "no_check_certificate": True,
        "noplaylist": True,
        "retries": 10,
        "fragment_retries": 10,
        "extractor_retries": 3,
    }
    js_runtimes = detect_js_runtimes()
    if js_runtimes:
        opts["js_runtimes"] = js_runtimes
    # Prefer bundled yt-dlp-ejs; allow GitHub fetch if the package is missing/outdated.
    try:
        import yt_dlp_ejs  # noqa: F401
    except ImportError:
        opts["remote_components"] = {"ejs:github"}
    opts.update(overrides)
    return opts


# Simplified function to read URLs from file - no resolution parsing needed
def read_urls_from_file(file_path):
    """Read YouTube URLs from a text file (one per line)."""
    urls = []
    try:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                url = line.strip()
                if url and not url.startswith("#"):  # Ignore empty lines and comments
                    urls.append(url)
        return parse_urls(urls)
    except Exception as e:
        raise OSError(f"Error reading URL file: {e}") from e


def parse_urls(lines):
    if isinstance(lines, str):
        lines = lines.splitlines()
    return [
        line.strip().removeprefix("@")
        for line in lines
        if line.strip() and not line.strip().startswith("#")
    ]


def _make_directory(parent, prefix):
    parent = Path(parent)
    parent.mkdir(parents=True, exist_ok=True)
    base = parent / f"{prefix}_{datetime.now():%Y%m%d_%H%M%S}"
    candidate, index = base, 1
    while True:
        try:
            candidate.mkdir()
            return candidate
        except FileExistsError:
            candidate = base.with_name(f"{base.name}_{index}")
            index += 1


@dataclass
class DownloadResult:
    directory: str
    total: int
    files: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    cancelled: bool = False

    @property
    def exit_code(self):
        return 1 if self.errors else 130 if self.cancelled else 0

    def summary(self):
        state = (
            "Cancelled"
            if self.cancelled
            else "Finished with failures"
            if self.errors
            else "Completed"
        )
        return f"{state}: {len(self.files)}/{self.total} successful; {len(self.errors)} failed. Output: {self.directory}"


class DownloadCancelledError(Exception):
    pass


class _YTDLPLogger:
    def __init__(self, feedback):
        self.feedback = feedback

    def debug(self, message):
        self.feedback.debug(message)

    def warning(self, message):
        self.feedback(f"Warning: {message}")

    def error(self, message):
        self.feedback(message)


class YTDownloader:
    def _message(self, message="", **kwargs):
        self.feedback(
            re.sub(
                r"\[/?(?:bold |bold|red|green|yellow|blue|cyan)[^\]]*\]", "", str(message)
            ).strip()
        )

    def _event(self, kind, payload):
        if self.event_callback:
            self.event_callback(kind, payload)

    def _check_cancel(self):
        if self.cancel_event.is_set():
            raise DownloadCancelledError("Cancellation requested")

    def _check_ready(self):
        self._check_cancel()
        if not self.ffmpeg_available:
            raise RuntimeError("ffmpeg is required to produce MP4/MP3; install it and retry")
        self._final_path = None

    def _progress_hook(self, data):
        self._check_cancel()
        if data.get("status") == "downloading":
            total = data.get("total_bytes") or data.get("total_bytes_estimate")
            percent = min(100, 100 * data.get("downloaded_bytes", 0) / total) if total else None
            if self.progress_callback:
                self.progress_callback(data)
            now = time.monotonic()
            if now - self._last_gui_progress >= 0.1:
                self._event("progress", {"percent": percent})
                self._last_gui_progress = now
            if now - self._last_progress >= 1:
                self.feedback(
                    f"Downloading: {percent:.1f}%"
                    if percent is not None
                    else "Downloading: size unknown"
                )
                self._last_progress = now
        elif data.get("status") == "finished":
            self.feedback("Transfer finished; processing/conversion is still running.")
            if self.status_callback:
                self.status_callback("Processing / converting...")

    def _postprocessor_hook(self, data):
        # Do not interrupt ffmpeg midway; the cancellation flag is checked at the next safe point.
        message = (
            "Cancellation requested; waiting for processing"
            if self.cancel_event.is_set()
            else "Processing / converting..."
        )
        if self.status_callback:
            self.status_callback(message)
        self._event("phase", message)
        self.feedback.debug(f"Postprocessor: {data.get('postprocessor')} - {data.get('status')}")
        if data.get("status") == "finished":
            self._final_path = data.get("info_dict", {}).get("filepath") or self._final_path

    def _finished_filename(self, ydl, info, suffix):
        candidate = Path(
            self._final_path or info.get("filepath") or ydl.prepare_filename(info)
        ).with_suffix(suffix)
        if not candidate.is_file():
            raise OSError(
                f"Post-processing did not produce the expected {suffix} file: {candidate}"
            )
        return str(candidate)

    def __init__(self):
        """Initialize the downloader with default settings."""
        self.output_dir = os.path.join(os.path.expanduser("~"), "Downloads")
        self.current_video_title = ""
        self.progress_callback = None
        self.status_callback = None
        self._js_runtime_warned = False
        self.feedback = Feedback("vaila_ytdown")
        self.cancel_event = threading.Event()
        self.event_callback = None
        self._batch_lock = threading.Lock()
        self._last_progress = 0
        self._last_gui_progress = 0
        self._final_path = None
        self.last_result = None

        # Check if ffmpeg is available
        self.ffmpeg_available = self._check_ffmpeg()
        if not self.ffmpeg_available:
            self._message(
                "[yellow]Warning: ffmpeg not found in PATH. MP4/MP3 downloads require ffmpeg.[/yellow]"
            )
        self._warn_missing_js_runtime_once()

    def _check_ffmpeg(self):
        """Check if ffmpeg is available in the system path."""
        return shutil.which("ffmpeg") is not None

    def _warn_missing_js_runtime_once(self) -> None:
        """Warn once when no Deno/Node/QuickJS is available for YouTube EJS."""
        if self._js_runtime_warned or detect_js_runtimes():
            return
        self._js_runtime_warned = True
        self._message(
            "[yellow]Warning: No JavaScript runtime (deno/node) found for YouTube. "
            "Install Deno (recommended) or Node.js ≥22 to avoid HTTP 403 / missing formats. "
            "See https://github.com/yt-dlp/yt-dlp/wiki/EJS[/yellow]"
        )

    def get_video_info(self, url):
        """Get detailed information about the video with enhanced resolution and FPS tracking."""
        ydl_opts = build_ytdlp_base_opts(
            quiet=True,
            no_warnings=True,
            skip_download=True,
            format="best",
            simulate=True,
            dump_single_json=True,
            logger=_YTDLPLogger(self.feedback),
        )

        try:
            self._check_cancel()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                # Process all available formats to get comprehensive quality options
                available_formats = []

                if "formats" in info:
                    video_formats = [
                        f for f in info["formats"] if f.get("vcodec", "none") != "none"
                    ]

                    # Group formats by resolution and find best FPS for each
                    resolution_formats = {}
                    for fmt in video_formats:
                        height = fmt.get("height", 0)
                        fps = fmt.get("fps", 0)

                        key = f"{fmt.get('width', 0)}x{height}"
                        if key not in resolution_formats or fps > resolution_formats[key]["fps"]:
                            resolution_formats[key] = {
                                "resolution": key,
                                "height": height,
                                "width": fmt.get("width", 0),
                                "fps": fps,
                                "format_id": fmt.get("format_id", ""),
                                "ext": fmt.get("ext", ""),
                                "filesize": fmt.get("filesize", 0),
                                "vcodec": fmt.get("vcodec", ""),
                            }

                    # Convert to sorted list (highest resolution first)
                    available_formats = sorted(
                        resolution_formats.values(),
                        key=lambda x: (x["height"], x["fps"]),
                        reverse=True,
                    )

                return {
                    "title": info.get("title", "Unknown"),
                    "uploader": info.get("uploader", "Unknown"),
                    "duration": info.get("duration", 0),
                    "upload_date": info.get("upload_date", ""),
                    "available_formats": available_formats,
                    "url": url,
                }
        except Exception as e:
            self._message(f"[red]Error getting video info: {str(e)}[/red]")
            # Return basic info so download can still proceed
            return {
                "title": "Unknown",
                "url": url,
                "available_formats": [],
            }

    def download_video(self, url, output_dir=None, filename_prefix=""):
        """Download yt-dlp best video + audio and produce a verified MP4."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder
        self._check_ready()
        save_dir = str(_make_directory(self.output_dir, "vaila_ytdownload"))
        if self.feedback.log_file is None:
            self.feedback.log_file = Path(save_dir) / "download_log.txt"

        # Format spec: Let yt-dlp decide best quality available (default behavior)
        format_spec = "bestvideo+bestaudio/best"
        self._message("[blue]Downloading best available quality (video+audio)[/blue]")

        # Prepare filename template with prefix if provided
        outtmpl = os.path.join(
            save_dir,
            f"{filename_prefix + '_' if filename_prefix else ''}%(title)s.%(ext)s",
        )

        # Set up download options with max FPS preference + YouTube EJS/JS runtime
        ydl_opts = build_ytdlp_base_opts(
            format=format_spec,
            outtmpl=outtmpl,
            progress_hooks=[self._progress_hook],
            postprocessor_hooks=[self._postprocessor_hook],
            logger=_YTDLPLogger(self.feedback),
            quiet=True,
            no_warnings=False,
            merge_output_format="mp4",
            postprocessors=[
                {
                    "key": "FFmpegVideoConvertor",
                    "preferedformat": "mp4",
                }
            ],
            writethumbnail=False,
            writeinfojson=False,
        )

        try:
            # First get extended video information for the detailed info file
            try:
                self._message(f"[blue]Getting detailed info for: {url}[/blue]")
                video_info = self.get_video_info(url)
            except Exception as e:
                self._message(f"[yellow]Warning: Could not get detailed info: {str(e)}[/yellow]")
                video_info = {"url": url, "available_formats": []}

            # Now download the video
            self._check_cancel()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                self.current_video_title = info.get("title", "Unknown")

                actual_filename = self._finished_filename(ydl, info, ".mp4")

                # Create a comprehensive information file with available resolutions and FPS
                info_file = os.path.join(save_dir, "video_info.txt")
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {info.get('title', 'Unknown')}\n")
                    f.write(f"Channel: {info.get('uploader', 'Unknown')}\n")
                    f.write(f"URL: {redact(url)}\n")
                    f.write(
                        f"Downloaded resolution: {info.get('width', 0)}x{info.get('height', 0)}\n"
                    )
                    f.write(f"Downloaded FPS: {info.get('fps', 0)}\n")
                    f.write(f"Duration: {info.get('duration', 0)} seconds\n")
                    f.write(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # Add available formats information sorted by FPS first, then resolution
                    f.write("AVAILABLE RESOLUTIONS AND FPS OPTIONS (sorted by FPS):\n")
                    f.write("=================================================\n")

                    if video_info.get("available_formats"):
                        # Sort formats by FPS first, then by resolution
                        sorted_formats = sorted(
                            video_info["available_formats"],
                            key=lambda x: (x.get("fps", 0), x.get("height", 0)),
                            reverse=True,
                        )

                        for i, fmt in enumerate(sorted_formats, 1):
                            f.write(
                                f"{i}. FPS: {fmt.get('fps')} | Resolution: {fmt.get('resolution')}\n"
                            )
                            f.write(f"   Video codec: {fmt.get('vcodec')}\n")
                            filesize_mb = fmt.get("filesize", 0) / (1024 * 1024)
                            if filesize_mb > 0:
                                f.write(f"   Approximate size: {filesize_mb:.1f} MB\n")
                            f.write("\n")
                    else:
                        f.write("Could not retrieve detailed format information.\n")

                self._message(f"\n[green]Download successful:[/green] {self.current_video_title}")
                self._message(f"[blue]Saved to:[/blue] {actual_filename}")
                self._message(
                    f"[blue]Resolution:[/blue] {info.get('width', 0)}x{info.get('height', 0)}"
                )
                self._message(f"[blue]FPS:[/blue] {info.get('fps', 0)}")

                if self.status_callback:
                    self._event("phase", f"Saved: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading video: {str(e)}"
            self._message(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg) from e

    def download_audio(self, url, output_dir=None, filename_prefix=""):
        """Download audio only from a YouTube URL as MP3."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder (optional, maybe save directly to output_dir?)
        self._check_ready()
        save_dir = str(_make_directory(self.output_dir, "vaila_ytaudio"))
        if self.feedback.log_file is None:
            self.feedback.log_file = Path(save_dir) / "download_log.txt"

        self._message(f"[blue]Downloading audio only (MP3) for: {url}[/blue]")

        outtmpl = os.path.join(
            save_dir,
            f"{filename_prefix + '_' if filename_prefix else ''}%(title)s.%(ext)s",
        )

        ydl_opts = build_ytdlp_base_opts(
            format="bestaudio/best",
            outtmpl=outtmpl,
            progress_hooks=[self._progress_hook],
            postprocessor_hooks=[self._postprocessor_hook],
            logger=_YTDLPLogger(self.feedback),
            quiet=True,
            no_warnings=False,
            postprocessors=[
                {
                    "key": "FFmpegExtractAudio",
                    "preferredcodec": "mp3",
                    "preferredquality": "192",  # Pode ajustar a qualidade (ex: '320')
                }
            ],
            writethumbnail=False,
            writeinfojson=False,  # Pode querer manter True para ter info
        )

        try:
            self._check_cancel()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=True)
                self.current_video_title = info.get("title", "Unknown")

                actual_filename = self._finished_filename(ydl, info, ".mp3")

                self._message(
                    f"\n[green]Audio download successful:[/green] {self.current_video_title}"
                )
                self._message(f"[blue]Saved as MP3 to:[/blue] {actual_filename}")

                if self.status_callback:
                    self._event("phase", f"Saved: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading audio: {str(e)}"
            self._message(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg) from e

    def download_urls(self, urls, output_dir=None, audio_only=False, *, batch=None):
        """Single execution path for GUI, TXT, CLI and interactive input."""
        if not self._batch_lock.acquire(blocking=False):
            raise RuntimeError("A download is already running")
        try:
            self.feedback.log_file = None
            urls = parse_urls(urls)
            if not urls:
                raise ValueError("Enter at least one URL")
            destination = Path(output_dir or self.output_dir).expanduser().resolve()
            use_batch = len(urls) > 1 if batch is None else batch
            folder_type = "audio" if audio_only else "batch"
            run_dir = (
                _make_directory(destination, f"vaila_{folder_type}") if use_batch else destination
            )
            run_dir.mkdir(parents=True, exist_ok=True)
            result = DownloadResult(str(run_dir), total=len(urls))
            self.last_result = result
            self.feedback.log_file = run_dir / "download_log.txt" if use_batch else None
            argv = [
                sys.executable,
                "-m",
                "vaila.vaila_ytdown",
                "--no-gui",
                "--output",
                str(destination),
            ]
            if use_batch:
                url_file = run_dir / "urls.txt"
                url_file.write_text("\n".join(urls) + "\n", encoding="utf-8")
                argv.extend(["--file", str(url_file)])
            else:
                argv.extend(["--url", urls[0]])
            if audio_only:
                argv.append("--audio-only")
            if self.feedback.debug_enabled:
                argv.append("--debug")
            self.feedback("Equivalent CLI: " + command_text(argv))
            self.feedback(
                f"Starting {len(urls)} items; format={'MP3 (192 kbps)' if audio_only else 'MP4 (best video + audio)'}"
            )
            for index, url in enumerate(urls, 1):
                if self.cancel_event.is_set():
                    result.cancelled = True
                    break
                self.feedback(f"Item {index}/{len(urls)}: {url}")
                self._event("item", {"index": index, "total": len(urls), "url": redact(url)})
                item_dir = run_dir / f"{index:03d}" if use_batch else destination
                try:
                    download = self.download_audio if audio_only else self.download_video
                    output = download(
                        url,
                        output_dir=str(item_dir),
                        filename_prefix=f"{index:03d}" if use_batch else "",
                    )
                    result.files.append(output)
                    if not use_batch:
                        result.directory = str(Path(output).parent)
                        self.feedback.log_file = Path(result.directory) / "download_log.txt"
                    self.feedback(f"SUCCESS {index}: {output}")
                except Exception as error:
                    if self.cancel_event.is_set():
                        result.cancelled = True
                        self.feedback("Cancelled at a safe point; completed files preserved.")
                        break
                    result.errors.append(redact(str(error)))
                    self.feedback.error(error)
                self._event(
                    "counts",
                    {
                        "success": len(result.files),
                        "failed": len(result.errors),
                        "total": len(urls),
                    },
                )
            if self.cancel_event.is_set():
                result.cancelled = True
            self.feedback(result.summary())
            self._event("summary", result)
            return result
        finally:
            self._batch_lock.release()

    def download_from_file(self, file_path, output_dir=None, audio_only=False):
        """Compatibility entry point; detailed counts are also in last_result."""
        result = self.download_urls(
            read_urls_from_file(file_path), output_dir, audio_only, batch=True
        )
        return result.directory


class DownloaderGUI:
    def __init__(self, root):
        self.root = root
        self.task = WorkerTask()
        self.downloader = YTDownloader()
        self.downloader.cancel_event = self.task.cancel
        self.downloader.event_callback = self.task.emit
        self.downloader.feedback.callback = lambda line: self.task.emit("log", line)
        self.downloader.progress_callback = None
        self.downloader.status_callback = lambda message: self.task.emit("phase", message)
        self.last_directory = None
        self.close_requested = False
        root.title("vailá YouTube Downloader")
        root.geometry("850x680")
        root.minsize(640, 520)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="YouTube Downloader", font=("TkDefaultFont", 16, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame, text="1. Review URLs -> 2. Choose destination and format -> 3. Download"
        ).pack(anchor="w", pady=6)
        self.urls_text = tk.Text(frame, height=8, wrap="word", undo=True)
        self.urls_text.pack(fill="both", expand=True)
        self.urls_text.bind("<<Modified>>", self.update_count)
        url_buttons = ttk.Frame(frame)
        url_buttons.pack(fill="x", pady=5)
        self.load_button = ttk.Button(url_buttons, text="Load TXT...", command=self.load_txt)
        self.load_button.pack(side="left")
        self.clear_button = ttk.Button(url_buttons, text="Clear", command=self.clear_urls)
        self.clear_button.pack(side="left", padx=5)
        self.count_label = ttk.Label(url_buttons, text="0 entries")
        self.count_label.pack(side="left", padx=8)
        dest_row = ttk.Frame(frame)
        dest_row.pack(fill="x", pady=5)
        ttk.Label(dest_row, text="Destination").pack(side="left")
        self.output_dir_var = tk.StringVar(root, value=self.downloader.output_dir)
        self.dest_entry = ttk.Entry(dest_row, textvariable=self.output_dir_var)
        self.dest_entry.pack(side="left", fill="x", expand=True, padx=8)
        self.browse_button = ttk.Button(dest_row, text="Browse...", command=self.browse)
        self.browse_button.pack(side="right")
        self.audio_only = tk.BooleanVar(root, value=False)
        format_row = ttk.Frame(frame)
        format_row.pack(fill="x")
        self.video_button = ttk.Radiobutton(
            format_row, text="Video (MP4)", variable=self.audio_only, value=False
        )
        self.video_button.pack(side="left")
        self.audio_button = ttk.Radiobutton(
            format_row, text="Audio (MP3)", variable=self.audio_only, value=True
        )
        self.audio_button.pack(side="left", padx=12)
        ttk.Label(
            frame,
            text="MP4: yt-dlp best video + audio. MP3: best audio converted to 192 kbps.\n"
            "Both formats require ffmpeg. Completion includes merging/conversion.",
            wraplength=760,
        ).pack(anchor="w", pady=6)
        controls = ttk.Frame(frame)
        controls.pack(fill="x", pady=8)
        self.download_button = ttk.Button(controls, text="Download", command=self.start_download)
        self.download_button.pack(side="left")
        self.cancel_button = ttk.Button(
            controls, text="Cancel", command=self.cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=6)
        ttk.Button(controls, text="Open folder", command=self.open_folder).pack(side="left")
        ttk.Button(controls, text="Help", command=self.show_help).pack(side="right")
        self.status = ttk.Label(frame, text="Waiting for URLs", wraplength=760)
        self.status.pack(anchor="w")
        self.progress = ttk.Progressbar(frame, maximum=100)
        self.progress.pack(fill="x", pady=5)
        self.counts = ttk.Label(frame, text="Success: 0 · Failed: 0")
        self.counts.pack(anchor="w")
        details_row = ttk.Frame(frame)
        details_row.pack(fill="x")
        self.details_visible = tk.BooleanVar(root, value=False)
        ttk.Checkbutton(
            details_row,
            text="Show details",
            variable=self.details_visible,
            command=self.toggle_details,
        ).pack(side="left")
        self.debug = tk.BooleanVar(root, value=False)
        self.debug_button = ttk.Checkbutton(
            details_row, text="Diagnostic details (--debug)", variable=self.debug
        )
        self.debug_button.pack(side="left", padx=10)
        from tkinter.scrolledtext import ScrolledText

        self.log = ScrolledText(frame, height=7, state="disabled", wrap="word")
        self.input_widgets = [
            self.load_button,
            self.clear_button,
            self.dest_entry,
            self.browse_button,
            self.video_button,
            self.audio_button,
            self.debug_button,
            self.urls_text,
        ]
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-Return>", lambda event: self.start_download())
        root.bind("<Escape>", lambda event: self.cancel())
        self.urls_text.focus_set()
        self.poll_id = root.after(75, self.poll)

    def update_count(self, event=None):
        self.count_label.configure(
            text=f"{len(parse_urls(self.urls_text.get('1.0', 'end')))} entries"
        )
        self.urls_text.edit_modified(False)

    def load_txt(self):
        if self.task.busy:
            return
        path = filedialog.askopenfilename(
            parent=self.root,
            title="Load URLs for review",
            filetypes=[("Text files", "*.txt"), ("All files", "*")],
        )
        if path:
            try:
                urls = read_urls_from_file(path)
                self.urls_text.delete("1.0", "end")
                self.urls_text.insert("1.0", "\n".join(urls))
                self.update_count()
                self.downloader.feedback(
                    f"Loaded {len(urls)} entries. Review the list, then click Download."
                )
            except Exception as error:
                messagebox.showerror("Cannot load URLs", str(error), parent=self.root)

    def clear_urls(self):
        if not self.task.busy:
            self.urls_text.delete("1.0", "end")
            self.update_count()

    def browse(self):
        path = filedialog.askdirectory(parent=self.root, title="Download destination")
        if path:
            self.output_dir_var.set(path)

    def toggle_details(self):
        if self.details_visible.get():
            self.log.pack(fill="both", expand=True)
        else:
            self.log.pack_forget()

    def start_download(self):
        if self.task.busy:
            return False
        urls = parse_urls(self.urls_text.get("1.0", "end"))
        output = self.output_dir_var.get().strip()
        if not urls or not output:
            messagebox.showerror(
                "Missing parameters", "Enter URLs and a destination directory.", parent=self.root
            )
            return False
        audio_only = self.audio_only.get()
        self.downloader.feedback.debug_enabled = self.debug.get()
        self.counts.configure(text="Success: 0 · Failed: 0")
        self.progress.configure(value=0, mode="determinate")
        self.status.configure(text="Starting...")
        for widget in self.input_widgets:
            widget.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.task.start(lambda: self.downloader.download_urls(urls, output, audio_only))
        return True

    def cancel(self):
        if self.task.busy:
            self.task.cancel.set()
            self.status.configure(
                text="Cancellation requested; waiting for a safe point. Completed files are preserved."
            )
            self.downloader.feedback(
                "Cancellation requested; waiting for transfer/processing to reach a safe point."
            )

    def close(self):
        if self.task.busy:
            self.close_requested = True
            self.cancel()
        else:
            self.root.after_cancel(self.poll_id)
            self.root.destroy()

    def open_folder(self):
        path = Path(self.last_directory or self.output_dir_var.get()).expanduser().resolve()
        if path.is_dir():
            try:
                if sys.platform == "win32":
                    os.startfile(path)
                else:
                    subprocess.Popen(
                        ["open" if sys.platform == "darwin" else "xdg-open", str(path)]
                    )
            except OSError as error:
                messagebox.showerror("Open folder", str(error), parent=self.root)

    def show_help(self):
        webbrowser.open_new_tab(get_help_html_path().as_uri())

    def poll(self):
        for kind, payload in self.task.drain():
            if kind == "log":
                self.log.configure(state="normal")
                self.log.insert("end", payload + "\n")
                self.log.see("end")
                self.log.configure(state="disabled")
            elif kind == "item":
                self.status.configure(
                    text=f"Item {payload['index']}/{payload['total']}: {payload['url']}"
                )
                self.progress.stop()
                self.progress.configure(mode="determinate", value=0)
            elif kind == "progress":
                percent = payload["percent"]
                self.progress.stop()
                self.progress.configure(
                    mode="determinate" if percent is not None else "indeterminate"
                )
                if percent is None:
                    self.progress.start()
                else:
                    self.progress.configure(value=percent)
            elif kind == "phase":
                self.status.configure(
                    text="Cancellation requested; waiting for processing"
                    if self.task.cancel.is_set()
                    else payload
                )
                self.progress.configure(mode="indeterminate")
                self.progress.start()
            elif kind == "counts":
                self.counts.configure(
                    text=f"Success: {payload['success']} · Failed: {payload['failed']}"
                )
            elif kind == "result":
                self.last_directory = payload.directory
                self.status.configure(text=payload.summary())
                self.counts.configure(
                    text=f"Success: {len(payload.files)} · Failed: {len(payload.errors)}"
                )
            elif kind == "error":
                self.downloader.feedback.error(payload)
                self.status.configure(text=f"Failed: {payload}")
            elif kind == "done":
                self.progress.stop()
                self.progress.configure(mode="determinate")
                for widget in self.input_widgets:
                    widget.configure(state="normal")
                self.download_button.configure(state="normal")
                self.cancel_button.configure(state="disabled")
                if self.close_requested:
                    self.root.destroy()
                    return
        self.poll_id = self.root.after(75, self.poll)


def run_ytdown(argv=None):
    parser = argparse.ArgumentParser(
        description="Download best video + audio as MP4, or audio as MP3",
        epilog='Example: python -m vaila.vaila_ytdown --file "my urls.txt" --output "my videos" --audio-only --no-gui',
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("-u", "--url", help="Single video URL")
    inputs.add_argument("-f", "--file", help="TXT with one URL per line")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-a", "--audio-only", action="store_true", help="Produce MP3 (192 kbps)")
    parser.add_argument("--no-gui", action="store_true", help="CLI only; prompt for URL if absent")
    parser.add_argument(
        "--debug", action="store_true", help="Include technical details and traceback"
    )
    # Embedded launch does not consume the parent application's command line.
    embedded = TKINTER_AVAILABLE and tk._default_root is not None
    args = parser.parse_args([] if argv is None and embedded else argv)
    feedback = Feedback("vaila_ytdown", args.debug)
    if args.url or args.file or args.no_gui or not TKINTER_AVAILABLE:
        downloader = YTDownloader()
        downloader.feedback = feedback
        try:
            urls = read_urls_from_file(args.file) if args.file else [args.url] if args.url else []
            if not urls:
                feedback("Waiting for URL input:")
                urls = [input().strip()]
            result = downloader.download_urls(
                urls, args.output, args.audio_only, batch=bool(args.file)
            )
            return result.exit_code
        except (KeyboardInterrupt, EOFError):
            downloader.cancel_event.set()
            feedback("Cancelled; completed files preserved.")
            return 130
        except Exception as error:
            feedback.error(error)
            return 1
    try:
        parent = tk._default_root
        root = tk.Toplevel(parent) if parent else tk.Tk()
        if parent:
            root.transient(parent)
        app = DownloaderGUI(root)
        root.app = app
        app.debug.set(args.debug)
        app.audio_only.set(args.audio_only)
        if args.output:
            app.output_dir_var.set(args.output)
        if parent:
            parent.wait_window(root)
        else:
            root.mainloop()
        return 0
    except Exception as error:
        feedback.error(error)
        feedback("GUI unavailable. Use --no-gui with --url or --file.")
        return 1


if __name__ == "__main__":
    sys.exit(run_ytdown())
