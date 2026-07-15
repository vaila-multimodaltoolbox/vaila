# vailá - Multimodal Toolbox
# © Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
# https://github.com/paulopreto/vaila-multimodaltoolbox
# Please see AUTHORS for contributors.
#
# Licensed under GNU Lesser General Public License v3.0
#
# ffmpeg_utils.py
#
# Update Date: 15 July 2026
# Version: 0.3.83
#
# Central helper module for FFmpeg binary discovery and H.264 encoding helpers
# shared by cutvideo, drawboxe, and related video tools.
#
# Cross-platform encode selection (verified live when possible):
#   1. h264_nvenc        — NVIDIA GPU (Linux / Windows; Ada e.g. RTX 4090)
#   2. h264_videotoolbox  — Apple VideoToolbox (macOS)
#   3. libx264           — CPU software (all OS; laptops without discrete GPU)
#
# Recommended NVIDIA stack (Linux/Windows + RTX 40xx):
#   - FFmpeg 8.x (or 7.x) with NVENC + recent NVIDIA driver
#   - Default remains h264_nvenc (OpenCV-compatible); not av1/hevc by default
#
# Binary search order for get_ffmpeg_path() / get_ffprobe_path():
#
#   1. VAILA_FFMPEG_PATH / VAILA_FFPROBE_PATH env vars (user override)
#   2. <project_root>/bin/ffmpeg/<name>[.exe]  (bundled / local static)
#   3. <venv>/bin|Scripts/<name>[.exe]
#   4. System binary on PATH
#
# Optional env:
#   VAILA_NVENC_PRESET=p1..p7     (default p5)
#   VAILA_FFMPEG_ENCODER=libx264|h264_nvenc|h264_videotoolbox  (force)
#
# Usage:
#   from vaila.ffmpeg_utils import (
#       get_ffmpeg_path,
#       detect_ffmpeg_video_encoder,
#       get_ffmpeg_video_encoding_args,
#   )
#
#   subprocess.run([get_ffmpeg_path(), "-i", "video.mp4", ...])

import contextlib
import os
import shutil
import subprocess
import sys
import time
from collections.abc import Callable
from functools import lru_cache
from pathlib import Path

# Preferred NVIDIA / NVENC device index for H.264 hardware encoding.
NVENC_GPU_INDEX = 0
# Default NVENC preset: p5 balances quality and speed on Ada (RTX 40xx).
# Override with VAILA_NVENC_PRESET=p6 for slightly higher quality.
DEFAULT_NVENC_PRESET = "p5"
_VALID_NVENC_PRESETS = frozenset(f"p{i}" for i in range(1, 8))

CPU_VIDEO_ENCODER = "libx264"
NVENC_VIDEO_ENCODER = "h264_nvenc"
VIDEOTOOLBOX_VIDEO_ENCODER = "h264_videotoolbox"
HARDWARE_VIDEO_ENCODERS = frozenset({NVENC_VIDEO_ENCODER, VIDEOTOOLBOX_VIDEO_ENCODER})
_SUPPORTED_FORCE_ENCODERS = frozenset(
    {CPU_VIDEO_ENCODER, NVENC_VIDEO_ENCODER, VIDEOTOOLBOX_VIDEO_ENCODER}
)


def _get_project_root() -> Path:
    """Return the vaila project root directory.

    Walks up from this file (vaila/ffmpeg_utils.py) to find the project root,
    which is the parent of the 'vaila' package directory.
    """
    return Path(__file__).resolve().parent.parent


def _nvenc_preset() -> str:
    """Return NVENC preset ``p1``–``p7`` (default ``p5``)."""
    raw = os.environ.get("VAILA_NVENC_PRESET", DEFAULT_NVENC_PRESET).strip().lower()
    if raw in _VALID_NVENC_PRESETS:
        return raw
    return DEFAULT_NVENC_PRESET


def _is_windows() -> bool:
    return sys.platform == "win32"


def _is_macos() -> bool:
    return sys.platform == "darwin"


def _binary_basenames(name: str) -> list[str]:
    """Return candidate basenames for ``name`` (adds ``.exe`` on Windows)."""
    if _is_windows():
        lower = name.lower()
        if lower.endswith(".exe"):
            return [name]
        return [f"{name}.exe", name]
    return [name]


def _is_runnable_binary(path: Path) -> bool:
    """Return True if ``path`` looks like an executable ffmpeg/ffprobe binary."""
    if not path.is_file():
        return False
    if _is_windows():
        return path.suffix.lower() in {".exe", ".bat", ".cmd"} or os.access(path, os.X_OK)
    return os.access(path, os.X_OK)


def _venv_bin_dir(venv_path: str) -> Path:
    """Return the Scripts/ (Windows) or bin/ directory inside a virtualenv."""
    root = Path(venv_path)
    if _is_windows():
        return root / "Scripts"
    return root / "bin"


def _find_binary(name: str, env_var: str) -> str:
    """Find the best available binary for the given name.

    Search order (portable across Linux / macOS / Windows):
        1. Environment variable override (e.g. VAILA_FFMPEG_PATH)
        2. Local static binary in <project_root>/bin/ffmpeg/
        3. Virtual environment binary
        4. System binary on PATH

    Args:
        name: Binary name, e.g. "ffmpeg" or "ffprobe".
        env_var: Environment variable name for user override.

    Returns:
        Full path to the binary, or just the name if only found on PATH.
    """
    # 1. Environment variable override
    env_path = os.environ.get(env_var)
    if env_path:
        candidate = Path(env_path).expanduser()
        if _is_runnable_binary(candidate):
            return str(candidate)

    project_root = _get_project_root()

    # 2. Local static / bundled binary (Windows installers often ship this)
    for basename in _binary_basenames(name):
        local_bin = project_root / "bin" / "ffmpeg" / basename
        if _is_runnable_binary(local_bin):
            return str(local_bin)

    # 3. Virtual environment binary
    venv_path = os.environ.get("VIRTUAL_ENV")
    if venv_path:
        venv_dir = _venv_bin_dir(venv_path)
        for basename in _binary_basenames(name):
            venv_bin = venv_dir / basename
            if _is_runnable_binary(venv_bin):
                return str(venv_bin)

    # 4. System binary on PATH (shutil.which resolves .exe on Windows)
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


def _parse_ffmpeg_version_output(stdout: str) -> str:
    """Parse ``ffmpeg -version`` stdout into a short version token."""
    first_line = (stdout or "").split("\n", maxsplit=1)[0]
    parts = first_line.split()
    if len(parts) >= 3:
        return parts[2]
    return first_line or "unknown"


def get_ffmpeg_version(ffmpeg: str | None = None) -> str:
    """Get the version string of an ffmpeg binary (default: resolved path).

    Returns:
        Version string (e.g. "8.1.2") or "unknown" on failure.
    """
    binary = ffmpeg or get_ffmpeg_path()
    try:
        result = subprocess.run(
            [binary, "-version"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return _parse_ffmpeg_version_output(result.stdout)
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


@lru_cache(maxsize=1)
def get_nvidia_gpu_info() -> dict[str, str] | None:
    """Return nvidia-smi details for the NVIDIA GPU selected for NVENC."""
    try:
        result = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=index,name,uuid,pci.bus_id",
                "--format=csv,noheader,nounits",
            ],
            capture_output=True,
            text=True,
            check=True,
            timeout=5,
        )
        for line in result.stdout.splitlines():
            index, name, uuid, pci_bus_id = (part.strip() for part in line.split(",", maxsplit=3))
            if int(index) == NVENC_GPU_INDEX:
                return {
                    "index": index,
                    "name": name,
                    "uuid": uuid,
                    "pci_bus_id": pci_bus_id,
                }
    except (FileNotFoundError, subprocess.SubprocessError, OSError, ValueError):
        pass
    return None


def describe_nvenc_gpu() -> str:
    """Return a CLI-friendly description of the NVIDIA NVENC device."""
    gpu_info = get_nvidia_gpu_info()
    if not gpu_info:
        return f"NVIDIA NVENC GPU index {NVENC_GPU_INDEX}"
    return (
        f"NVIDIA NVENC GPU {gpu_info['index']}: {gpu_info['name']} "
        f"(PCI {gpu_info['pci_bus_id']}, UUID {gpu_info['uuid']})"
    )


# FFmpeg binary that passed a live hardware-encoder test (may differ from get_ffmpeg_path()).
_hw_ffmpeg_path: str | None = None
# Back-compat alias used by older call sites / tests.
_nvenc_ffmpeg_path: str | None = None


def is_hardware_video_encoder(encoder: str) -> bool:
    """Return True for NVIDIA NVENC or Apple VideoToolbox H.264 encoders."""
    return encoder in HARDWARE_VIDEO_ENCODERS


def encoders_with_cpu_fallback(selected: str | None = None) -> list[str]:
    """Return ``[selected, libx264]`` for hardware encoders, else ``[libx264]``."""
    encoder = selected or detect_ffmpeg_video_encoder()
    if is_hardware_video_encoder(encoder):
        return [encoder, CPU_VIDEO_ENCODER]
    return [CPU_VIDEO_ENCODER]


def encoder_device_tag(encoder: str) -> str:
    """Return ``GPU`` or ``CPU`` for log prefixes."""
    return "GPU" if is_hardware_video_encoder(encoder) else "CPU"


def describe_video_encoder(encoder: str) -> str:
    """Return a short human-readable description of the selected encoder."""
    if encoder == NVENC_VIDEO_ENCODER:
        return f"NVIDIA NVENC ({describe_nvenc_gpu()})"
    if encoder == VIDEOTOOLBOX_VIDEO_ENCODER:
        return "Apple VideoToolbox"
    return "CPU libx264"


def _iter_ffmpeg_candidates() -> list[str]:
    """Return unique FFmpeg binaries to try for hardware encode (preferred first)."""
    candidates: list[str] = []
    seen: set[str] = set()

    def _add(path: str | None) -> None:
        if not path:
            return
        resolved = str(Path(path).expanduser())
        key = os.path.realpath(resolved) if os.path.exists(resolved) else resolved.lower()
        if key in seen:
            return
        if _is_runnable_binary(Path(resolved)):
            seen.add(key)
            candidates.append(resolved)
            return
        which_path = shutil.which(path)
        if which_path:
            _add(which_path)

    _add(get_ffmpeg_path())
    _add(os.environ.get("VAILA_FFMPEG_PATH"))

    # Platform-typical install locations (harmless no-ops when missing).
    if _is_windows():
        for extra in (
            r"C:\ffmpeg\bin\ffmpeg.exe",
            r"C:\Program Files\ffmpeg\bin\ffmpeg.exe",
            r"C:\ProgramData\chocolatey\bin\ffmpeg.exe",
        ):
            _add(extra)
    else:
        for extra in (
            "/usr/bin/ffmpeg",
            "/bin/ffmpeg",
            "/usr/local/bin/ffmpeg",
            "/opt/homebrew/bin/ffmpeg",  # Apple Silicon Homebrew
            "/usr/local/opt/ffmpeg/bin/ffmpeg",
        ):
            _add(extra)

    # Local bundled copy (in case PATH preferred a different binary).
    project_root = _get_project_root()
    for basename in _binary_basenames("ffmpeg"):
        _add(str(project_root / "bin" / "ffmpeg" / basename))

    # Other ffmpeg binaries later on PATH.
    path_env = os.environ.get("PATH", "")
    for directory in path_env.split(os.pathsep):
        if not directory:
            continue
        for basename in _binary_basenames("ffmpeg"):
            _add(str(Path(directory) / basename))
    return candidates


def _encoder_error_summary(stderr: str, *keywords: str) -> str:
    """Extract useful failure line(s) from ffmpeg stderr."""
    keys = tuple(k.lower() for k in keywords) or ("error",)
    lines = []
    for line in (stderr or "").splitlines():
        lower = line.lower()
        if any(k in lower for k in keys):
            text = line.strip()
            if text and text not in lines:
                lines.append(text)
    if lines:
        return " | ".join(lines[:3])
    text = (stderr or "").strip()
    return text[-300:] if text else "unknown error"


def _ffmpeg_lists_encoder(ffmpeg: str, encoder_name: str) -> tuple[bool, str]:
    """Return whether ``encoder_name`` appears in ``ffmpeg -encoders``."""
    try:
        encoders_result = subprocess.run(
            [ffmpeg, "-hide_banner", "-encoders"],
            capture_output=True,
            text=True,
            check=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return False, str(e)
    if encoder_name not in encoders_result.stdout:
        return False, f"{encoder_name} encoder not listed in this FFmpeg build"
    return True, ""


def _test_h264_nvenc(ffmpeg: str) -> tuple[bool, str]:
    """Return (ok, error_summary) for a short h264_nvenc test encode."""
    listed, err = _ffmpeg_lists_encoder(ffmpeg, NVENC_VIDEO_ENCODER)
    if not listed:
        return False, err

    try:
        test_result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=1280x720:rate=30",
                "-t",
                "1",
                "-c:v",
                NVENC_VIDEO_ENCODER,
                "-gpu",
                str(NVENC_GPU_INDEX),
                "-preset",
                _nvenc_preset(),
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return False, str(e)

    if test_result.returncode == 0:
        return True, ""
    return False, _encoder_error_summary(
        test_result.stderr or test_result.stdout or "",
        "nvenc",
        "nvidia",
        "driver",
        "cuda",
        "nvcuda",
    )


def _test_h264_videotoolbox(ffmpeg: str) -> tuple[bool, str]:
    """Return (ok, error_summary) for a short Apple VideoToolbox test encode."""
    listed, err = _ffmpeg_lists_encoder(ffmpeg, VIDEOTOOLBOX_VIDEO_ENCODER)
    if not listed:
        return False, err

    try:
        test_result = subprocess.run(
            [
                ffmpeg,
                "-hide_banner",
                "-loglevel",
                "error",
                "-f",
                "lavfi",
                "-i",
                "testsrc=size=1280x720:rate=30",
                "-t",
                "1",
                "-c:v",
                VIDEOTOOLBOX_VIDEO_ENCODER,
                "-b:v",
                "8M",
                "-pix_fmt",
                "yuv420p",
                "-f",
                "null",
                "-",
            ],
            capture_output=True,
            text=True,
            timeout=15,
        )
    except (FileNotFoundError, subprocess.SubprocessError, OSError) as e:
        return False, str(e)

    if test_result.returncode == 0:
        return True, ""
    return False, _encoder_error_summary(
        test_result.stderr or test_result.stdout or "",
        "videotoolbox",
        "video toolbox",
        "error",
    )


def _forced_encoder_from_env() -> str | None:
    """Return a forced encoder from ``VAILA_FFMPEG_ENCODER`` when valid."""
    raw = (os.environ.get("VAILA_FFMPEG_ENCODER") or "").strip().lower()
    if raw in _SUPPORTED_FORCE_ENCODERS:
        return raw
    return None


@lru_cache(maxsize=1)
def detect_ffmpeg_video_encoder() -> str:
    """Return the best verified FFmpeg H.264 encoder for this machine/OS.

    Priority:
        1. ``VAILA_FFMPEG_ENCODER`` force override (when set)
        2. NVIDIA ``h264_nvenc`` (Linux/Windows with working NVENC)
        3. Apple ``h264_videotoolbox`` (macOS)
        4. CPU ``libx264`` (all platforms; no discrete GPU / no HW encode)

    NVENC tries alternate FFmpeg binaries when the preferred one lists NVENC
    but cannot open the encoder (common with a mismatched static build vs
    driver). Machines without NVIDIA skip NVENC quietly after a list check.
    """
    global _hw_ffmpeg_path, _nvenc_ffmpeg_path
    _hw_ffmpeg_path = None
    _nvenc_ffmpeg_path = None

    forced = _forced_encoder_from_env()
    if forced == CPU_VIDEO_ENCODER:
        print("  [FFmpeg][CPU] VAILA_FFMPEG_ENCODER=libx264 — using software encoding")
        return CPU_VIDEO_ENCODER

    preferred = get_ffmpeg_path()
    candidates = _iter_ffmpeg_candidates()
    last_nvenc_error = ""
    saw_nvenc_listed = False

    # 1) NVIDIA NVENC (Linux / Windows; rare legacy macOS NVIDIA)
    if forced in (None, NVENC_VIDEO_ENCODER):
        for ffmpeg in candidates:
            listed, list_err = _ffmpeg_lists_encoder(ffmpeg, NVENC_VIDEO_ENCODER)
            if not listed:
                if forced == NVENC_VIDEO_ENCODER:
                    last_nvenc_error = list_err
                continue
            saw_nvenc_listed = True
            ok, err = _test_h264_nvenc(ffmpeg)
            if ok:
                _hw_ffmpeg_path = ffmpeg
                _nvenc_ffmpeg_path = ffmpeg
                version = get_ffmpeg_version(ffmpeg)
                if os.path.realpath(ffmpeg) != os.path.realpath(preferred):
                    print(
                        f"  [FFmpeg][GPU] Preferred binary cannot use NVENC; "
                        f"using alternate for encode: {ffmpeg}"
                    )
                print(f"  [FFmpeg][GPU] h264_nvenc verified on {describe_nvenc_gpu()}")
                print(
                    f"  [FFmpeg][GPU] Encode binary: {ffmpeg} "
                    f"(FFmpeg {version}, preset {_nvenc_preset()})"
                )
                return NVENC_VIDEO_ENCODER
            last_nvenc_error = err
            if ffmpeg == candidates[0]:
                print(f"  [FFmpeg][GPU] NVENC unavailable with {ffmpeg}: {err}")
        if forced == NVENC_VIDEO_ENCODER:
            print(
                f"  [FFmpeg][CPU] Forced h264_nvenc unavailable "
                f"({last_nvenc_error or 'not found'}); using libx264"
            )
            return CPU_VIDEO_ENCODER
        if saw_nvenc_listed and last_nvenc_error:
            print(f"  [FFmpeg][CPU] NVENC listed but not usable ({last_nvenc_error})")

    # 2) Apple VideoToolbox (macOS)
    if forced in (None, VIDEOTOOLBOX_VIDEO_ENCODER) and (
        _is_macos() or forced == VIDEOTOOLBOX_VIDEO_ENCODER
    ):
        for ffmpeg in candidates:
            ok, err = _test_h264_videotoolbox(ffmpeg)
            if ok:
                _hw_ffmpeg_path = ffmpeg
                version = get_ffmpeg_version(ffmpeg)
                print("  [FFmpeg][GPU] h264_videotoolbox verified (Apple VideoToolbox)")
                print(f"  [FFmpeg][GPU] Encode binary: {ffmpeg} (FFmpeg {version})")
                return VIDEOTOOLBOX_VIDEO_ENCODER
            if forced == VIDEOTOOLBOX_VIDEO_ENCODER and ffmpeg == candidates[0]:
                print(f"  [FFmpeg][GPU] VideoToolbox unavailable with {ffmpeg}: {err}")
        if forced == VIDEOTOOLBOX_VIDEO_ENCODER:
            print("  [FFmpeg][CPU] Forced h264_videotoolbox unavailable; using libx264")
            return CPU_VIDEO_ENCODER

    # 3) Portable CPU fallback
    print("  [FFmpeg][CPU] Using libx264 software encoding")
    return CPU_VIDEO_ENCODER


def get_video_encode_ffmpeg_path(encoder: str | None = None) -> str:
    """Return the FFmpeg binary to use for H.264 video encoding.

    When hardware encode was verified on an alternate binary (e.g. system
    FFmpeg while PATH points at a mismatched static build), that alternate is
    returned for hardware encoders. Otherwise returns ``get_ffmpeg_path()``.
    """
    selected = encoder or detect_ffmpeg_video_encoder()
    if is_hardware_video_encoder(selected) and _hw_ffmpeg_path:
        return _hw_ffmpeg_path
    return get_ffmpeg_path()


def get_ffmpeg_video_encoding_args(encoder: str | None = None) -> list[str]:
    """Build quality-oriented FFmpeg H.264 arguments for GPU or CPU encoding.

    - ``h264_nvenc``: Ada-friendly defaults (preset p5, CQ 18, spatial AQ)
    - ``h264_videotoolbox``: Apple HW encode with bitrate target
    - ``libx264``: portable CPU encode (all OS)
    """
    selected_encoder = encoder or detect_ffmpeg_video_encoder()
    if selected_encoder == NVENC_VIDEO_ENCODER:
        return [
            "-c:v",
            NVENC_VIDEO_ENCODER,
            "-gpu",
            str(NVENC_GPU_INDEX),
            "-preset",
            _nvenc_preset(),
            "-tune",
            "hq",
            "-rc",
            "vbr",
            "-cq",
            "18",
            "-b:v",
            "0",
            "-spatial-aq",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]
    if selected_encoder == VIDEOTOOLBOX_VIDEO_ENCODER:
        return [
            "-c:v",
            VIDEOTOOLBOX_VIDEO_ENCODER,
            "-b:v",
            "8M",
            "-allow_sw",
            "1",
            "-pix_fmt",
            "yuv420p",
        ]
    return [
        "-c:v",
        CPU_VIDEO_ENCODER,
        "-preset",
        "medium",
        "-crf",
        "18",
        "-pix_fmt",
        "yuv420p",
    ]


def run_ffmpeg_encode_with_fallback(
    command_prefix: list[str],
    command_suffix: list[str],
) -> str:
    """Run an FFmpeg encode, retrying with CPU if a hardware encode fails.

    ``command_prefix`` should start with an FFmpeg binary path; it is replaced
    with ``get_video_encode_ffmpeg_path(encoder)`` for each attempt.
    """
    for encoder in encoders_with_cpu_fallback():
        device = encoder_device_tag(encoder)
        prefix = list(command_prefix)
        if prefix:
            prefix[0] = get_video_encode_ffmpeg_path(encoder)
        command = [
            *prefix,
            *get_ffmpeg_video_encoding_args(encoder),
            *command_suffix,
        ]
        print(
            f"  [FFmpeg][{device}] Starting H.264 encode with {encoder} "
            f"({describe_video_encoder(encoder)}) via {prefix[0]}..."
        )
        try:
            subprocess.run(command, check=True, capture_output=False, text=True)
            print(f"  [FFmpeg][{device}] Finished H.264 encode with {encoder}")
            return encoder
        except subprocess.CalledProcessError:
            if not is_hardware_video_encoder(encoder):
                raise
            print(
                f"  [FFmpeg][GPU] Warning: {encoder} failed; retrying with CPU {CPU_VIDEO_ENCODER}"
            )

    raise RuntimeError("FFmpeg encoding failed without producing an output file")


def clear_ffmpeg_encoder_cache() -> None:
    """Clear cached encoder detection / binary path results (useful in tests)."""
    global _hw_ffmpeg_path, _nvenc_ffmpeg_path
    _hw_ffmpeg_path = None
    _nvenc_ffmpeg_path = None
    get_ffmpeg_path.cache_clear()
    get_ffprobe_path.cache_clear()
    get_nvidia_gpu_info.cache_clear()
    detect_ffmpeg_video_encoder.cache_clear()


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
