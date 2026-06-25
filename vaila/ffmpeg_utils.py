# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# ffmpeg_utils.py
#
# Central helper module for FFmpeg binary discovery.
# Provides get_ffmpeg_path() and get_ffprobe_path() that resolve the
# best available FFmpeg binary in this order:
#
#   1. VAILA_FFMPEG_PATH / VAILA_FFPROBE_PATH env vars (user override)
#   2. <project_root>/bin/ffmpeg/ffmpeg  (local static binary)
#   3. <venv>/bin/ffmpeg                 (venv-installed)
#   4. System "ffmpeg" on PATH           (fallback)
#
# Usage:
#   from vaila.ffmpeg_utils import get_ffmpeg_path, get_ffprobe_path
#
#   FFMPEG  = get_ffmpeg_path()
#   FFPROBE = get_ffprobe_path()
#
#   subprocess.run([FFMPEG, "-i", "video.mp4", ...])

import contextlib
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path


def _get_project_root() -> Path:
    """Return the vaila project root directory.

    Walks up from this file (vaila/ffmpeg_utils.py) to find the project root,
    which is the parent of the 'vaila' package directory.
    """
    return Path(__file__).resolve().parent.parent


def _find_binary(name: str, env_var: str) -> str:
    """Find the best available binary for the given name.

    Search order:
        1. Environment variable override (e.g. VAILA_FFMPEG_PATH)
        2. Local static binary in <project_root>/bin/ffmpeg/<name>
        3. Virtual environment binary in <venv>/bin/<name>
        4. System binary on PATH

    Args:
        name: Binary name, e.g. "ffmpeg" or "ffprobe".
        env_var: Environment variable name for user override.

    Returns:
        Full path to the binary, or just the name if only found on PATH.
    """
    # 1. Environment variable override
    env_path = os.environ.get(env_var)
    if env_path and os.path.isfile(env_path) and os.access(env_path, os.X_OK):
        return env_path

    # 2. Local static binary in bin/ffmpeg/
    project_root = _get_project_root()
    local_bin = project_root / "bin" / "ffmpeg" / name
    if local_bin.is_file() and os.access(local_bin, os.X_OK):
        return str(local_bin)

    # 3. Virtual environment binary
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        venv_bin = Path(venv_path) / "bin" / name
        if venv_bin.is_file() and os.access(venv_bin, os.X_OK):
            return str(venv_bin)

    # 4. System binary on PATH
    system_bin = shutil.which(name)
    if system_bin:
        return system_bin

    # If nothing found, return the bare name and let subprocess raise the error
    return name


@lru_cache(maxsize=1)
def get_ffmpeg_path() -> str:
    """Get the path to the best available ffmpeg binary.

    Results are cached after the first call.

    Returns:
        Full path to ffmpeg, or "ffmpeg" if only on PATH.
    """
    path = _find_binary("ffmpeg", "VAILA_FFMPEG_PATH")
    return path


@lru_cache(maxsize=1)
def get_ffprobe_path() -> str:
    """Get the path to the best available ffprobe binary.

    Results are cached after the first call.

    Returns:
        Full path to ffprobe, or "ffprobe" if only on PATH.
    """
    path = _find_binary("ffprobe", "VAILA_FFPROBE_PATH")
    return path


def get_ffmpeg_version() -> str:
    """Get the version string of the resolved ffmpeg binary.

    Returns:
        Version string (e.g. "7.1.1") or "unknown" on failure.
    """
    ffmpeg = get_ffmpeg_path()
    try:
        result = subprocess.run(
            [ffmpeg, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        # First line is like: ffmpeg version 7.1.1 Copyright (c) ...
        first_line = result.stdout.split("\n")[0]
        parts = first_line.split()
        if len(parts) >= 3:
            return parts[2]
        return first_line
    except Exception:
        return "unknown"


def print_ffmpeg_info():
    """Print diagnostic info about the resolved FFmpeg binary."""
    ffmpeg = get_ffmpeg_path()
    ffprobe = get_ffprobe_path()
    version = get_ffmpeg_version()

    print(f"FFmpeg path:    {ffmpeg}")
    print(f"FFprobe path:   {ffprobe}")
    print(f"FFmpeg version: {version}")

    # Check if using local binary
    project_root = _get_project_root()
    local_bin = str(project_root / "bin" / "ffmpeg")
    if ffmpeg.startswith(local_bin):
        print("Source:         Local static binary (bin/ffmpeg/)")
    elif "venv" in ffmpeg.lower() or ".venv" in ffmpeg:
        print("Source:         Virtual environment")
    else:
        print("Source:         System PATH")


def probe_video_duration(path: str, ffprobe: str | None = None) -> float:
    """Probe video duration in seconds using ffprobe.

    Args:
        path: Absolute path to the input media file.
        ffprobe: Optional ffprobe binary; defaults to ``get_ffprobe_path()``.

    Returns:
        Duration in seconds; ``0.0`` if ffprobe fails (caller must treat
        zero as "unknown" — progress reporter will skip % / ETA).
    """
    if ffprobe is None:
        ffprobe = get_ffprobe_path()
    try:
        result = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-show_entries",
                "format=duration",
                "-of",
                "default=noprint_wrappers=1:nokey=1",
                path,
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
        if result.returncode == 0:
            value = (result.stdout or "").strip()
            if value and value.upper() != "N/A":
                return float(value)
    except Exception:
        pass
    return 0.0


def _format_hms(sec: float) -> str:
    """Format seconds as ``H:MM:SS`` (or ``M:SS`` below 1 h)."""
    if sec is None or sec < 0 or sec != sec:  # NaN guard
        return "--:--"
    total = int(sec)
    h, rem = divmod(total, 3600)
    m, s = divmod(rem, 60)
    if h > 0:
        return f"{h:d}:{m:02d}:{s:02d}"
    return f"{m:d}:{s:02d}"


def _format_bytes(num: float) -> str:
    """Format a byte count with binary multiples (B/KB/MB/GB/TB)."""
    num = float(num)
    for unit in ("B", "KB", "MB", "GB"):
        if abs(num) < 1024.0:
            return f"{num:.1f} {unit}"
        num /= 1024.0
    return f"{num:.1f} TB"


def run_ffmpeg_with_progress(
    cmd: list[str],
    total_duration_sec: float = 0.0,
    label: str = "ffmpeg",
    progress_interval_sec: float = 2.0,
    on_progress: Callable[[dict], None] | None = None,
    stream=None,
) -> tuple[int, str]:
    """Run an ffmpeg command and emit periodic progress lines on ``stream``.

    The helper injects ``-progress pipe:1`` (and ``-nostats``) immediately
    before the output path so it works with the existing per-encoder commands
    in ``compress_videos_h264.py`` / ``h265.py`` / ``h266.py``.

    Progress lines look like:

    ``[1/3] my.mp4]:  37.4% | speed=2.10x | fps= 60.0 | enc=0:01:12/0:03:13 | ETA 0:00:58 | size=42.3 MB | elapsed 0:00:35``

    Args:
        cmd: Full ffmpeg invocation (output path must be the last positional).
        total_duration_sec: Duration from ``probe_video_duration``; pass ``0``
            to disable percent / ETA (only ``frame`` / ``speed`` are shown).
        label: Short prefix printed before each progress line.
        progress_interval_sec: Minimum wall-clock seconds between prints.
        on_progress: Optional callback invoked with the parsed progress dict
            (keys: ``encoded_sec``, ``total_sec``, ``percent``, ``speed``,
            ``fps``, ``frame``, ``size_bytes``, ``eta_sec``, ``elapsed_sec``).
        stream: File-like target for progress prints; defaults to ``sys.stdout``.

    Returns:
        Tuple ``(returncode, stderr_tail)`` where ``stderr_tail`` keeps the
        last ~1000 chars of ffmpeg stderr (useful for error messages).
    """
    if stream is None:
        stream = sys.stdout

    cmd = list(cmd)
    if "-progress" not in cmd:
        cmd.insert(-1, "-progress")
        cmd.insert(-1, "pipe:1")
    if "-nostats" not in cmd:
        cmd.insert(-1, "-nostats")

    start = time.time()
    last_print = 0.0
    state: dict[str, str] = {}

    proc = subprocess.Popen(  # noqa: SIM117
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        bufsize=1,
    )

    try:
        assert proc.stdout is not None
        for raw_line in proc.stdout:
            line = raw_line.strip()
            if not line or "=" not in line:
                continue
            key, _, value = line.partition("=")
            state[key] = value

            if key != "progress":
                continue

            now = time.time()
            force = value == "end"
            if not force and (now - last_print) < progress_interval_sec:
                continue

            try:
                out_time_us = int(state.get("out_time_us", "0") or "0")
            except ValueError:
                out_time_us = 0
            encoded_sec = out_time_us / 1_000_000.0

            speed_str = (state.get("speed") or "0x").rstrip("x")
            try:
                speed = float(speed_str)
            except ValueError:
                speed = 0.0

            try:
                fps = float(state.get("fps", "0") or "0")
            except ValueError:
                fps = 0.0

            try:
                size_bytes = int(state.get("total_size", "0") or "0")
            except ValueError:
                size_bytes = 0

            try:
                frame = int(state.get("frame", "0") or "0")
            except ValueError:
                frame = 0

            percent = 0.0
            eta_sec = -1.0
            if total_duration_sec > 0:
                percent = min(100.0, 100.0 * encoded_sec / total_duration_sec)
                if speed > 0.0:
                    remaining = max(0.0, total_duration_sec - encoded_sec)
                    eta_sec = remaining / speed

            elapsed = now - start

            if total_duration_sec > 0:
                pct_str = f"{percent:5.1f}%"
                enc_str = f"{_format_hms(encoded_sec)}/{_format_hms(total_duration_sec)}"
            else:
                pct_str = "  --.-%"
                enc_str = f"{_format_hms(encoded_sec)}"

            eta_str = _format_hms(eta_sec) if eta_sec >= 0 else "--:--"
            size_disp = _format_bytes(size_bytes) if size_bytes > 0 else "--"

            msg = (
                f"{label}: {pct_str}"
                f" | speed={speed:5.2f}x"
                f" | fps={fps:6.1f}"
                f" | enc={enc_str}"
                f" | ETA {eta_str}"
                f" | size={size_disp}"
                f" | elapsed {_format_hms(elapsed)}"
            )
            with contextlib.suppress(Exception):
                print(msg, file=stream, flush=True)

            if on_progress is not None:
                with contextlib.suppress(Exception):
                    on_progress(
                        {
                            "encoded_sec": encoded_sec,
                            "total_sec": total_duration_sec,
                            "percent": percent,
                            "speed": speed,
                            "fps": fps,
                            "frame": frame,
                            "size_bytes": size_bytes,
                            "eta_sec": eta_sec,
                            "elapsed_sec": elapsed,
                        }
                    )

            last_print = now
            if force:
                state = {}

        stderr_text = proc.stderr.read() if proc.stderr is not None else ""
        proc.wait()
        return proc.returncode, (stderr_text or "")[-1000:]
    finally:
        try:
            if proc.stdout is not None:
                proc.stdout.close()
        except Exception:
            pass
        try:
            if proc.stderr is not None:
                proc.stderr.close()
        except Exception:
            pass
        if proc.poll() is None:
            try:
                proc.terminate()
                proc.wait(timeout=5)
            except Exception:
                with contextlib.suppress(Exception):
                    proc.kill()


if __name__ == "__main__":
    print_ffmpeg_info()
