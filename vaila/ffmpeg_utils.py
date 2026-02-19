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

import os
import shutil
import subprocess
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


if __name__ == "__main__":
    print_ffmpeg_info()
