"""
================================================================================
YouTube High Quality Downloader - vaila_ytdown.py
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Create: 10 October 2025
Update Date: 30 September 2026
Version: 0.4.6

Description:
------------
Review editable URLs (Load TXT only fills the list), select destination and MP4
or MP3, then Download. Cancellation is cooperative; completed files are kept.
MP4 offers per-video resolution/FPS choices, defaulting to highest FPS then
resolution. MP3 uses bestaudio/best, converted at
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
import contextlib
import copy
import os
import re
import shlex
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
    from task_feedback import (  # ty: ignore[unresolved-import]
        Feedback,
        WorkerTask,
        command_text,
        redact,
    )

try:
    from .ffmpeg_utils import opencv_compatible_copy, opencv_reads_video
except ImportError:
    from ffmpeg_utils import (  # ty: ignore[unresolved-import]
        opencv_compatible_copy,
        opencv_reads_video,
    )

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    try:
        from vaila.cli_highlight import print_gui_cli_mirror
    except ImportError:
        try:
            from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
        except ImportError:

            def print_gui_cli_mirror(  # type: ignore[misc]
                module_label: str,
                cli: list[str] | str,
                *,
                note: str = "Equivalent CLI (copy/paste to repeat this run):",
            ) -> None:
                cli_str = (
                    cli
                    if isinstance(cli, str)
                    else (
                        subprocess.list2cmdline(cli) if sys.platform == "win32" else shlex.join(cli)
                    )
                )
                header = f">> {module_label}: {note}"
                body = f">>   {cli_str}"
                banner = "=" * min(max(len(body), 40), 100)
                print()
                print(banner)
                print(header)
                print(body)
                print(banner)
                print()


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


def quality_key(fmt):
    return tuple(fmt.get(key) or 0 for key in ("width", "height", "fps"))


def codec_label(fmt):
    vcodec = str(fmt.get("vcodec") or "").lower()
    for prefix, name in (
        ("avc1", "H.264"), ("h264", "H.264"), ("vp09", "VP9"), ("vp9", "VP9"),
        ("av01", "AV1"), ("hev1", "H.265"), ("hvc1", "H.265"),
    ):  # fmt: skip
        if vcodec.startswith(prefix):
            return name
    return vcodec or "?"


def quality_label(fmt):
    width, height, fps = quality_key(fmt)
    fps_text = f"{fps:g}" if fps else "unknown"
    return f"{width or '?'}x{height or '?'} · {fps_text} FPS · {codec_label(fmt)}"


def video_formats(info):
    """Keep usable video formats in yt-dlp preference order (worst to best)."""
    return [
        fmt
        for fmt in info.get("formats", [])
        if fmt.get("vcodec") not in (None, "none")
        and not fmt.get("has_drm")
        and fmt.get("url")
        and fmt.get("format_id")
    ]


def opencv_codec_rank(fmt):
    """2 = H.264, 1 = VP9, 0 = other (AV1): what opencv-python can decode.

    Its bundled FFmpeg has no software AV1 decoder, so an AV1 download opens
    in every vailá video tool but never returns a frame.
    """
    vcodec = str(fmt.get("vcodec") or "").lower()
    if vcodec.startswith(("avc1", "h264")):
        return 2
    return 1 if vcodec.startswith(("vp9", "vp09")) else 0


def quality_options(info, *, keep_codec=False):
    """One variant per resolution/FPS, highest FPS first, then resolution.

    Default (vailá-ready): the most OpenCV-decodable codec of that quality
    (H.264 > VP9 > AV1). ``keep_codec``: YouTube's own best variant (often
    AV1, smaller). Among equals yt-dlp's order wins (last = best). The codec
    never lowers the resolution or FPS.
    """
    grouped = {}
    for fmt in video_formats(info):
        key = quality_key(fmt)
        if (
            key not in grouped
            or keep_codec
            or opencv_codec_rank(fmt) >= opencv_codec_rank(grouped[key])
        ):
            grouped[key] = fmt
    return sorted(
        grouped.values(),
        key=lambda f: (
            f.get("fps") or 0,
            min(f.get("width") or 0, f.get("height") or 0),
            max(f.get("width") or 0, f.get("height") or 0),
        ),
        reverse=True,
    )


def parse_video_selections(values, count):
    selections = {}
    for value in values:
        index, separator, format_id = value.partition("=")
        if not separator or not index.isdecimal() or not 1 <= int(index) <= count or not format_id:
            raise ValueError("--video-format requires INDEX=FORMAT_ID (index starts at 1)")
        if int(index) in selections:
            raise ValueError(f"Duplicate video selection: {index}")
        selections[int(index)] = format_id
    return selections


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


def format_ytdown_cli_command(
    urls: list[str],
    destination: Path | str,
    *,
    audio_only: bool = False,
    selections: dict[int, str] | None = None,
    debug: bool = False,
    url_file: Path | str | None = None,
    keep_codec: bool = False,
) -> list[str]:
    """Build the copy-paste CLI command that reproduces this run headlessly."""
    cmd = [
        "uv",
        "run",
        "--no-sync",
        "vaila/vaila_ytdown.py",
        "--output",
        str(destination),
        "--no-gui",
    ]
    if url_file:
        cmd.extend(["--file", str(url_file)])
    elif urls:
        cmd.extend(["--url", urls[0]])

    if audio_only:
        cmd.append("--audio-only")
    else:
        if keep_codec:
            cmd.append("--keep-codec")
        for idx, fmt_id in sorted((selections or {}).items()):
            cmd.extend(["--video-format", f"{idx}={fmt_id}"])

    if debug:
        cmd.append("--debug")

    return cmd


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
            fragments = data.get("fragment_count")
            fragment_index = data.get("fragment_index", 0)
            if fragments:
                percent = min(99, 100 * fragment_index / fragments)
            elif percent is not None:
                percent = min(99, percent)
            detail = f"Fragments: {fragment_index}/{fragments}" if fragments else None
            if self.progress_callback:
                self.progress_callback(data)
            now = time.monotonic()
            if now - self._last_gui_progress >= 0.1:
                self._event("progress", {"percent": percent, "detail": detail})
                self._last_gui_progress = now
            if now - self._last_progress >= 1:
                self.feedback(
                    detail or f"Downloading: {percent:.1f}%"
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
        # False (default) = vailá-ready MP4: OpenCV-decodable codec, H.264 re-encode
        # when needed. True = keep YouTube's best codec (often AV1), no re-encode.
        self.keep_codec = False

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
            format="bestvideo+bestaudio/best",
            format_sort=["fps", "res"],
            format_sort_force=True,
            simulate=True,
            logger=_YTDLPLogger(self.feedback),
        )

        try:
            self._check_cancel()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                info = ydl.extract_info(url, download=False)
                self._check_cancel()
                if not info or not quality_options(info):
                    raise ValueError("No downloadable video qualities found")
                return info
        except DownloadCancelledError:
            raise
        except Exception as e:
            raise ValueError(f"Cannot consult video qualities: {e}") from e

    def download_video(
        self, url, output_dir=None, filename_prefix="", *, video_info=None, format_id=None
    ):
        """Download the selected video quality with audio and verify the final MP4."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder
        self._check_ready()
        video_info = video_info or self.get_video_info(url)
        options = quality_options(video_info, keep_codec=self.keep_codec)
        if not options:
            raise ValueError("No downloadable video qualities found")
        selected = (
            next((fmt for fmt in video_formats(video_info) if fmt["format_id"] == format_id), None)
            if format_id
            else options[0]
        )
        if selected is None:
            raise ValueError(f"Video format unavailable: {format_id}")
        requested_quality = quality_key(selected)
        selected_format_id = selected["format_id"]
        save_dir = str(_make_directory(self.output_dir, "vaila_ytdownload"))
        if self.feedback.log_file is None:
            self.feedback.log_file = Path(save_dir) / "download_log.txt"

        self.feedback(f"Target folder for download: {save_dir}")
        self.feedback(
            f"Selected format: {quality_label(selected)} "
            f"(format ID: {selected['format_id']}, codec: {selected.get('vcodec') or 'auto'})"
        )
        self.feedback("Starting video/audio stream download with yt-dlp...")

        def selector(context):
            # IDs come from extraction, never interpolate them into selector syntax.
            video = next(
                (f for f in context["formats"] if f["format_id"] == selected_format_id), None
            )
            if video is None:
                raise ValueError("Selected video format is no longer available")
            if video.get("acodec") not in (None, "none"):
                yield video
                return
            audio = next(
                (
                    f
                    for f in reversed(context["formats"])
                    if f.get("vcodec") == "none"
                    and f.get("acodec") not in (None, "none")
                    and not f.get("has_drm")
                ),
                None,
            )
            if audio is None:
                raise ValueError("No audio stream available for selected video")
            yield {
                "format_id": f"{video['format_id']}+{audio['format_id']}",
                "ext": "mp4",
                "requested_formats": [video, audio],
                "width": video.get("width"),
                "height": video.get("height"),
                "fps": video.get("fps"),
                "vcodec": video.get("vcodec"),
                "acodec": audio.get("acodec"),
                "protocol": f"{video['protocol']}+{audio['protocol']}",
            }

        self._message(f"Selected: {quality_label(selected)} (format {selected['format_id']})")

        # Prepare filename template with prefix if provided
        outtmpl = os.path.join(
            save_dir,
            f"{filename_prefix + '_' if filename_prefix else ''}%(title)s.%(ext)s",
        )

        # Set up download options with max FPS preference + YouTube EJS/JS runtime
        ydl_opts = build_ytdlp_base_opts(
            format=selector,
            format_sort=["fps", "res"],
            format_sort_force=True,
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
            self._check_cancel()
            with yt_dlp.YoutubeDL(ydl_opts) as ydl:
                for attempt in range(2):
                    try:
                        info = ydl.process_ie_result(copy.deepcopy(video_info), download=True)
                        break
                    except yt_dlp.utils.DownloadError as error:
                        self._check_cancel()
                        if attempt or not re.search(r"\b(?:403|410)\b", str(error)):
                            raise
                        self.feedback("Media URL expired or denied; refreshing once.")
                        video_info = self.get_video_info(url)
                        same_quality = [
                            f
                            for f in reversed(video_formats(video_info))
                            if quality_key(f) == requested_quality
                        ]
                        selected = (
                            next(iter(same_quality), None)
                            if self.keep_codec
                            else max(same_quality, key=opencv_codec_rank, default=None)
                        )
                        if selected is None:
                            raise ValueError("Selected quality is no longer available") from error
                        selected_format_id = selected["format_id"]
                self.current_video_title = info.get("title", "Unknown")

                actual_filename = self._finished_filename(ydl, info, ".mp4")
                actual_filename = self._make_vaila_readable(actual_filename)
                file_size_mb = 0.0
                with contextlib.suppress(OSError):
                    file_size_mb = os.path.getsize(actual_filename) / (1024 * 1024)

                # Create a comprehensive information file with available resolutions and FPS
                info_file = os.path.join(save_dir, "video_info.txt")
                with open(info_file, "w", encoding="utf-8") as f:
                    f.write(f"Title: {info.get('title', 'Unknown')}\n")
                    f.write(f"Channel: {info.get('uploader', 'Unknown')}\n")
                    f.write(f"URL: {redact(url)}\n")
                    f.write(f"Requested quality (width, height, fps): {requested_quality}\n")
                    f.write(f"Downloaded format: {info.get('format_id', selected['format_id'])}\n")
                    f.write(
                        f"Downloaded resolution: {info.get('width', 0)}x{info.get('height', 0)}\n"
                    )
                    f.write(f"Downloaded FPS: {info.get('fps', 0)}\n")
                    f.write(f"Duration: {info.get('duration', 0)} seconds\n")
                    f.write(f"Download date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")

                    # Add available formats information sorted by FPS first, then resolution
                    f.write("AVAILABLE RESOLUTIONS AND FPS OPTIONS (sorted by FPS):\n")
                    f.write("=================================================\n")

                    if options:
                        for i, fmt in enumerate(options, 1):
                            f.write(f"{i}. {quality_label(fmt)} | ID: {fmt['format_id']}\n")
                            f.write(f"   Video codec: {fmt.get('vcodec')}\n")
                            filesize_mb = (
                                fmt.get("filesize") or fmt.get("filesize_approx") or 0
                            ) / (1024 * 1024)
                            if filesize_mb > 0:
                                f.write(f"   Approximate size: {filesize_mb:.1f} MB\n")
                            f.write("\n")
                    else:
                        f.write("Could not retrieve detailed format information.\n")

                self._message(f"\n[green]Download successful:[/green] {self.current_video_title}")
                self._message(f"[blue]Saved to:[/blue] {actual_filename} ({file_size_mb:.2f} MB)")
                self._message(
                    f"[blue]Resolution:[/blue] {info.get('width', 0)}x{info.get('height', 0)}"
                )
                self._message(f"[blue]FPS:[/blue] {info.get('fps', 0)}")
                self.feedback(f"Video metadata written to: {info_file}")

                if self.status_callback:
                    self._event("phase", f"Saved: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading video: {str(e)}"
            self._message(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg) from e

    def _make_vaila_readable(self, path):
        """Re-encode the MP4 to H.264 in place when OpenCV cannot decode it.

        Skipped with ``keep_codec`` (the user keeps YouTube's codec; getpixelvideo
        then offers an H.264 copy when opening an AV1 file).

        Every vailá video tool reads frames through OpenCV, whose bundled
        FFmpeg has no software AV1 decoder. Frames, timestamps and audio are
        kept (see ``ffmpeg_utils.opencv_compatible_copy``); the file keeps its name.
        """
        if self.keep_codec or opencv_reads_video(path):
            return path
        self._check_cancel()
        self.feedback(
            f"{Path(path).name}: codec not readable by OpenCV (e.g. AV1); "
            "re-encoding to H.264 so vailá tools can open it..."
        )
        if self.status_callback:
            self._event("phase", "Converting to H.264 for vailá")
        os.replace(opencv_compatible_copy(path), path)
        self.feedback(f"H.264 ready: {path}")
        return path

    def download_audio(self, url, output_dir=None, filename_prefix=""):
        """Download audio only from a YouTube URL as MP3."""
        if output_dir:
            self.output_dir = output_dir

        # Create timestamp for unique folder (optional, maybe save directly to output_dir?)
        self._check_ready()
        save_dir = str(_make_directory(self.output_dir, "vaila_ytaudio"))
        if self.feedback.log_file is None:
            self.feedback.log_file = Path(save_dir) / "download_log.txt"

        self.feedback(f"Target folder for audio: {save_dir}")
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
                file_size_mb = 0.0
                with contextlib.suppress(OSError):
                    file_size_mb = os.path.getsize(actual_filename) / (1024 * 1024)

                self._message(
                    f"\n[green]Audio download successful:[/green] {self.current_video_title}"
                )
                self._message(
                    f"[blue]Saved as MP3 to:[/blue] {actual_filename} ({file_size_mb:.2f} MB)"
                )

                if self.status_callback:
                    self._event("phase", f"Saved: {actual_filename}")

                return actual_filename
        except Exception as e:
            error_msg = f"Error downloading audio: {str(e)}"
            self._message(f"[red]{error_msg}[/red]")
            if self.status_callback:
                self.status_callback(f"Error: {error_msg}")
            raise Exception(error_msg) from e

    def download_urls(
        self, urls, output_dir=None, audio_only=False, *, batch=None, selections=None, previews=None
    ):
        """Single execution path for GUI, TXT, CLI and interactive input."""
        if not self._batch_lock.acquire(blocking=False):
            raise RuntimeError("A download is already running")
        try:
            self.feedback.log_file = None
            urls = parse_urls(urls)
            if not urls:
                raise ValueError("Enter at least one URL")
            selections = selections or {}
            previews = previews or {}
            if any(index < 1 or index > len(urls) for index in selections):
                raise ValueError("Video selection index is outside the URL list")
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
            url_file = None
            if use_batch:
                url_file = run_dir / "urls.txt"
                url_file.write_text("\n".join(urls) + "\n", encoding="utf-8")

            cli_argv = format_ytdown_cli_command(
                urls,
                destination,
                audio_only=audio_only,
                selections=selections,
                debug=self.feedback.debug_enabled,
                url_file=url_file,
                keep_codec=self.keep_codec,
            )

            format_label = (
                "Audio MP3 (192 kbps)"
                if audio_only
                else "Video MP4 (selected quality; default highest FPS, then resolution; "
                + (
                    "YouTube codec kept)"
                    if self.keep_codec
                    else "vailá-ready: H.264/VP9, AV1 re-encoded to H.264)"
                )
            )
            self.feedback("=" * 72)
            self.feedback(
                f"Starting YouTube download run ({len(urls)} item{'s' if len(urls) != 1 else ''})"
            )
            self.feedback(f"Destination parent directory: {destination}")
            self.feedback(f"Run output directory: {run_dir}")
            self.feedback(f"Format: {format_label}")
            self.feedback("=" * 72)
            self.feedback("Equivalent CLI: " + command_text(cli_argv))
            print_gui_cli_mirror(
                "vaila/vaila_ytdown",
                cli_argv,
                note="Equivalent CLI command for this run (copy/paste):",
            )

            for index, url in enumerate(urls, 1):
                if self.cancel_event.is_set():
                    result.cancelled = True
                    break
                item_dir = run_dir / f"{index:03d}" if use_batch else destination
                self.feedback(f"\n--- Item {index}/{len(urls)}: {redact(url)} ---")
                self.feedback(f"Item destination folder: {item_dir}")
                self._event("item", {"index": index, "total": len(urls), "url": redact(url)})
                try:
                    download = self.download_audio if audio_only else self.download_video
                    preview = previews.get(index)
                    if not audio_only and isinstance(preview, Exception):
                        raise ValueError(f"Quality consultation failed: {preview}")
                    extra = (
                        {}
                        if audio_only
                        else {"video_info": preview, "format_id": selections.get(index)}
                    )
                    output = download(
                        url,
                        output_dir=str(item_dir),
                        filename_prefix=f"{index:03d}" if use_batch else "",
                        **extra,
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
                self.feedback("Cancellation requested; completed files preserved.")

            self.feedback("=" * 72)
            self.feedback(result.summary())
            self.feedback(f"Final output directory: {result.directory}")
            if result.files:
                self.feedback(f"Completed files ({len(result.files)}):")
                for f in result.files:
                    self.feedback(f"  ✓ {f}")
            if result.errors:
                self.feedback(f"Failed items ({len(result.errors)}):")
                for err in result.errors:
                    self.feedback(f"  ✗ {err}")
            self.feedback("=" * 72)

            print_gui_cli_mirror(
                "vaila/vaila_ytdown",
                cli_argv,
                note="Equivalent CLI command for this run (copy/paste):",
            )
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
        self.preview_urls = []
        self.previews = {}
        self.selections = {}
        self.operation = None
        root.title("vailá YouTube Downloader")
        root.geometry("900x850")
        root.minsize(640, 520)
        frame = ttk.Frame(root, padding=16)
        frame.pack(fill="both", expand=True)
        ttk.Label(frame, text="YouTube Downloader", font=("TkDefaultFont", 16, "bold")).pack(
            anchor="w"
        )
        ttk.Label(
            frame,
            text="1. Review URLs -> 2. Choose destination -> 3. Consult qualities -> 4. Download",
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
        self.vaila_ready = tk.BooleanVar(root, value=not self.downloader.keep_codec)
        self.vaila_ready_button = ttk.Checkbutton(
            format_row,
            text="vailá-ready MP4 (H.264; AV1 re-encoded)",
            variable=self.vaila_ready,
            command=self.toggle_vaila_ready,
        )
        self.vaila_ready_button.pack(side="left", padx=12)
        self.consult_button = ttk.Button(
            frame, text="Consult qualities", command=self.consult_qualities
        )
        self.consult_button.pack(anchor="w", pady=5)
        self.quality_table = ttk.Treeview(
            frame, columns=("title", "quality"), show="headings", height=5
        )
        self.quality_table.heading("title", text="Video / consultation status")
        self.quality_table.heading("quality", text="Selected resolution / FPS")
        self.quality_table.pack(fill="x")
        self.quality_table.bind("<<TreeviewSelect>>", self.show_quality_choices)
        self.quality_choice = ttk.Combobox(frame, state="disabled")
        self.quality_choice.pack(fill="x", pady=5)
        self.quality_choice.bind("<<ComboboxSelected>>", self.choose_quality)
        ttk.Label(
            frame,
            text="MP4 default: highest FPS, then highest resolution; choose separately for each video.\n"
            "vailá-ready (default): same resolution/FPS in a codec OpenCV reads (H.264/VP9); an AV1-only "
            "quality is re-encoded to H.264. Untick to keep YouTube's codec (smaller, AV1).\n"
            "MP3: best audio converted to 192 kbps. "
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
            self.vaila_ready_button,
            self.debug_button,
            self.urls_text,
            self.consult_button,
        ]
        root.protocol("WM_DELETE_WINDOW", self.close)
        root.bind("<Control-Return>", lambda event: self.start_download())
        root.bind("<Escape>", lambda event: self.cancel())
        self.urls_text.focus_set()
        self.poll_id = root.after(75, self.poll)

    def quality_options(self, info):
        return quality_options(info, keep_codec=self.downloader.keep_codec)

    def toggle_vaila_ready(self):
        """Codec choice changes the variant of each quality: consult again."""
        self.downloader.keep_codec = not self.vaila_ready.get()
        self.preview_urls = []
        self.update_count()

    def update_count(self, event=None):
        urls = parse_urls(self.urls_text.get("1.0", "end"))
        if urls != self.preview_urls and not self.task.busy:
            self.previews.clear()
            self.selections.clear()
            self.preview_urls = []
            self.quality_table.delete(*self.quality_table.get_children())
            self.quality_choice.set("")
            self.quality_choice.configure(state="disabled")
        self.count_label.configure(
            text=f"{len(parse_urls(self.urls_text.get('1.0', 'end')))} entries"
        )
        self.urls_text.edit_modified(False)

    def set_busy(self):
        for widget in self.input_widgets:
            widget.configure(state="disabled")
        self.quality_choice.configure(state="disabled")
        self.download_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")

    def consult_qualities(self):
        if self.task.busy or self.audio_only.get():
            return False
        urls = parse_urls(self.urls_text.get("1.0", "end"))
        if not urls:
            self.status.configure(text="Enter URLs before consulting qualities")
            return False
        self.preview_urls = urls
        self.previews = {}
        self.selections = {}
        self.quality_table.delete(*self.quality_table.get_children())
        for index, url in enumerate(urls, 1):
            self.quality_table.insert("", "end", iid=str(index), values=(url, "Waiting"))
        self.operation = "consult"
        self.downloader.feedback.debug_enabled = self.debug.get()
        self.status.configure(text="Consulting available qualities...")
        self.downloader.feedback(f"Consulting available qualities for {len(urls)} video(s)...")
        self.set_busy()

        def consult():
            for index, url in enumerate(urls, 1):
                if self.task.cancel.is_set():
                    break
                try:
                    self.downloader.feedback(
                        f"Item {index}/{len(urls)}: fetching formats for {redact(url)}"
                    )
                    info = self.downloader.get_video_info(url)
                    opts = self.quality_options(info)
                    self.downloader.feedback(
                        f"Item {index}: found {len(opts)} format option(s) for '{info.get('title', 'Unknown')}' "
                        f"(default: {quality_label(opts[0]) if opts else 'none'})"
                    )
                except DownloadCancelledError:
                    break
                except Exception as error:
                    info = error
                    self.downloader.feedback.error(
                        f"Item {index} format consultation failed: {error}"
                    )
                self.task.emit("qualities", (index, info))

        self.task.start(consult)
        return True

    def show_quality_choices(self, event=None):
        selected = self.quality_table.selection()
        if self.task.busy or not selected:
            return
        index = int(selected[0])
        info = self.previews.get(index)
        options = self.quality_options(info) if isinstance(info, dict) else []
        self.quality_choice.configure(
            values=[quality_label(f) for f in options], state="readonly" if options else "disabled"
        )
        self.quality_choice.set("")
        for position, fmt in enumerate(options):
            if fmt["format_id"] == self.selections.get(index):
                self.quality_choice.current(position)

    def choose_quality(self, event=None):
        selected = self.quality_table.selection()
        if self.task.busy or not selected or self.quality_choice.current() < 0:
            return
        index = int(selected[0])
        fmt = self.quality_options(self.previews[index])[self.quality_choice.current()]
        self.selections[index] = fmt["format_id"]
        self.quality_table.set(str(index), "quality", quality_label(fmt))

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
        if not audio_only and (urls != self.preview_urls or len(self.previews) != len(urls)):
            return self.consult_qualities()
        if not audio_only and not self.selections:
            self.status.configure(text="No valid qualities; consult again before downloading")
            return False
        self.operation = "download"
        self.downloader.feedback.debug_enabled = self.debug.get()
        self.counts.configure(text="Success: 0 · Failed: 0")
        self.progress.configure(value=0, mode="determinate")
        self.status.configure(text="Starting...")
        self.set_busy()
        selections, previews = dict(self.selections), dict(self.previews)
        self.task.start(
            lambda: self.downloader.download_urls(
                urls, output, audio_only, selections=selections, previews=previews
            )
        )
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
                if payload.get("detail"):
                    self.status.configure(text=payload["detail"])
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
            elif kind == "qualities":
                index, info = payload
                self.previews[index] = info
                if isinstance(info, Exception):
                    self.quality_table.set(str(index), "quality", f"Failed: {redact(info)}")
                else:
                    fmt = self.quality_options(info)[0]
                    self.selections[index] = fmt["format_id"]
                    self.quality_table.item(
                        str(index), values=(info.get("title", "Unknown"), quality_label(fmt))
                    )
            elif kind == "result" and self.operation == "download":
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
                if self.operation == "consult":
                    self.status.configure(
                        text="Consultation cancelled"
                        if self.task.cancel.is_set()
                        else "Review each video's quality, then click Download. Failed items will be reported and skipped."
                    )
                    if self.quality_table.get_children():
                        self.quality_table.selection_set(self.quality_table.get_children()[0])
                self.show_quality_choices()
                if self.close_requested:
                    self.root.destroy()
                    return
        self.poll_id = self.root.after(75, self.poll)


def run_ytdown(argv=None):
    parser = argparse.ArgumentParser(
        description="Download MP4 with per-video quality selection (default: highest FPS, then resolution), or MP3",
        epilog='Example: python -m vaila.vaila_ytdown --file "my urls.txt" --output "my videos" --audio-only --no-gui',
    )
    inputs = parser.add_mutually_exclusive_group()
    inputs.add_argument("-u", "--url", help="Single video URL")
    inputs.add_argument("-f", "--file", help="TXT with one URL per line")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("-a", "--audio-only", action="store_true", help="Produce MP3 (192 kbps)")
    parser.add_argument("--no-gui", action="store_true", help="CLI only; prompt for URL if absent")
    parser.add_argument(
        "--list-formats",
        action="store_true",
        help="List resolution/FPS choices without downloading",
    )
    parser.add_argument(
        "--video-format",
        action="append",
        default=[],
        metavar="INDEX=FORMAT_ID",
        help="Select video format for a 1-based URL index; repeat for a batch",
    )
    parser.add_argument(
        "--keep-codec",
        action="store_true",
        help="Keep YouTube's best codec (often AV1, smaller files) and never re-encode. "
        "Default: vailá-ready MP4 (H.264/VP9 variant of the same quality; AV1 re-encoded to "
        "H.264) that opens in getpixelvideo and every OpenCV-based vailá tool.",
    )
    parser.add_argument(
        "--debug", action="store_true", help="Include technical details and traceback"
    )
    # Embedded launch does not consume the parent application's command line.
    embedded = TKINTER_AVAILABLE and getattr(tk, "_default_root", None) is not None
    args = parser.parse_args([] if argv is None and embedded else argv)
    if args.audio_only and (args.list_formats or args.video_format):
        parser.error("Video quality options cannot be combined with --audio-only")
    feedback = Feedback("vaila_ytdown", args.debug)
    if (
        args.url
        or args.file
        or args.no_gui
        or args.list_formats
        or args.video_format
        or not TKINTER_AVAILABLE
    ):
        downloader = YTDownloader()
        downloader.feedback = feedback
        downloader.keep_codec = args.keep_codec
        try:
            urls = read_urls_from_file(args.file) if args.file else [args.url] if args.url else []
            if not urls:
                feedback("Waiting for URL input:")
                urls = [input().strip()]
            try:
                selections = parse_video_selections(args.video_format, len(urls))
            except ValueError as error:
                parser.error(str(error))
            if args.list_formats:
                failed = False
                for index, url in enumerate(urls, 1):
                    try:
                        info = downloader.get_video_info(url)
                        feedback(f"Item {index}: {info.get('title', 'Unknown')}")
                        for position, fmt in enumerate(
                            quality_options(info, keep_codec=args.keep_codec)
                        ):
                            feedback(
                                f"{'DEFAULT ' if position == 0 else ''}{quality_label(fmt)} | "
                                f"--video-format {index}={fmt['format_id']}"
                            )
                    except Exception as error:
                        failed = True
                        feedback.error(error)
                return int(failed)
            result = downloader.download_urls(
                urls, args.output, args.audio_only, batch=bool(args.file), selections=selections
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
        parent = getattr(tk, "_default_root", None)
        root = tk.Toplevel(parent) if parent else tk.Tk()
        if parent:
            root.transient(parent)
        app = DownloaderGUI(root)
        root.app = app  # ty: ignore[invalid-assignment]
        app.debug.set(args.debug)
        app.audio_only.set(args.audio_only)
        if args.output:
            app.output_dir_var.set(args.output)
        if not parent:
            print(">> vaila/vaila_ytdown: launcher CLI")
            print_gui_cli_mirror(
                "vaila/vaila_ytdown",
                ["uv", "run", "--no-sync", "vaila/vaila_ytdown.py"],
                note="Launcher CLI (opens GUI; use --no-gui with -u URL or -f FILE for CLI):",
            )
        print(f">> vaila/vaila_ytdown: GUI ready. Default destination: {app.output_dir_var.get()}")
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
