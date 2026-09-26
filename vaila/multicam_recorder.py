"""
Project: vailá Multimodal Toolbox
Script: multicam_recorder.py - Record Cameras

Author: Carlos Pinto Jr
Email: carlos@vulcanum.com.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 26 September 2026
Update Date: 26 September 2026
Version: 0.4.5

Description:
Records video from multiple detected cameras simultaneously, through a simple
Tkinter GUI, using parallel ffmpeg subprocesses with stream-copy (no
re-encoding). Replaces hand-typed per-camera commands such as:

    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 \\
        -i /dev/video2 -c:v copy cam1.mkv &
    ffmpeg -f v4l2 -input_format mjpeg -framerate 30 -video_size 1280x720 \\
        -i /dev/video4 -c:v copy cam2.mkv &
    wait

which previously had to be launched by hand for multi-camera biomechanics
trials (e.g. markerless 2D/3D multi-view capture).

Camera device identity (the string passed to ffmpeg's ``-i``) is resolved per
OS rather than via OpenCV's capture index, since that index does not reliably
map to what ffmpeg needs on every platform (Windows DirectShow in particular
requires a device *name*, not an index). OpenCV is used only as a secondary,
best-effort pass to probe per-camera resolution/fps options shown in the GUI.

Usage:
GUI: click "Record Cameras" in vailá's Video and Image tools, or run
     ``uv run python vaila/multicam_recorder.py``.

Requirements:
- FFmpeg available (resolved via vaila.ffmpeg_utils.get_ffmpeg_path)
- OpenCV (only used to probe supported resolution/fps per camera)
"""

from __future__ import annotations

import glob
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import threading
import tkinter as tk
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Callable, cast

import cv2
import numpy as np

try:
    from .cutvideo import sanitize_output_basename
    from .ffmpeg_utils import get_ffmpeg_path
except ImportError:
    from cutvideo import sanitize_output_basename  # ty: ignore[unresolved-import]
    from ffmpeg_utils import get_ffmpeg_path  # ty: ignore[unresolved-import]


# ── Camera identity ────────────────────────────────────────────────────────


@dataclass(frozen=True)
class CameraDevice:
    """A camera as identified for ffmpeg capture on the current OS."""

    id: str
    """Exact value to pass to ffmpeg's ``-i`` (path on Linux, index on macOS,
    device/alternative name on Windows)."""

    label: str
    """Human-friendly name shown in the GUI."""

    backend: str
    """One of ``"v4l2"``, ``"avfoundation"``, ``"dshow"``."""


def enumerate_cameras() -> list[CameraDevice]:
    """Detect available cameras using an OS-native identity source."""
    if sys.platform.startswith("linux"):
        return _enumerate_cameras_linux()
    if sys.platform == "darwin":
        return _enumerate_cameras_macos(get_ffmpeg_path())
    if sys.platform.startswith("win"):
        return _enumerate_cameras_windows(get_ffmpeg_path())
    return []


def _video_index_key(path: str) -> int:
    match = re.search(r"(\d+)$", path)
    return int(match.group(1)) if match else 0


def _enumerate_cameras_linux() -> list[CameraDevice]:
    """Glob ``/dev/video*``, keeping only nodes that actually support capture.

    Many UVC webcams expose more than one ``/dev/videoN`` node per physical
    camera (e.g. a separate metadata-only node alongside the real capture
    node), which would otherwise show up as "ghost cameras" with no
    corresponding physical device. OpenCV's V4L2 backend refuses to open
    nodes that don't report the video-capture capability, so ``isOpened()``
    is used here purely as a real-camera filter (the device path itself, not
    OpenCV, remains the identity passed to ffmpeg).
    """
    paths = sorted(glob.glob("/dev/video*"), key=_video_index_key)
    return [
        CameraDevice(id=path, label=Path(path).name, backend="v4l2")
        for path in paths
        if _is_capturable_v4l2_device(path)
    ]


def _is_capturable_v4l2_device(path: str) -> bool:
    """Return True if OpenCV can actually open ``path`` as a capture device."""
    try:
        cap = cv2.VideoCapture(path)
    except Exception:
        return False
    try:
        return bool(cap.isOpened())
    finally:
        cap.release()


def _run_ffmpeg_device_listing(ffmpeg_path: str, args: list[str]) -> str:
    try:
        result = subprocess.run(
            [ffmpeg_path, *args],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
            timeout=10,
        )
    except (OSError, subprocess.TimeoutExpired):
        return ""
    return result.stderr or ""


def _enumerate_cameras_macos(ffmpeg_path: str) -> list[CameraDevice]:
    stderr_text = _run_ffmpeg_device_listing(
        ffmpeg_path, ["-f", "avfoundation", "-list_devices", "true", "-i", ""]
    )
    return _parse_avfoundation_listing(stderr_text)


def _parse_avfoundation_listing(stderr_text: str) -> list[CameraDevice]:
    devices: list[CameraDevice] = []
    in_video_section = False
    entry_pattern = re.compile(r"\[(\d+)\]\s+(.+)")
    for line in stderr_text.splitlines():
        lowered = line.lower()
        if "video devices" in lowered:
            in_video_section = True
            continue
        if "audio devices" in lowered:
            in_video_section = False
            continue
        if not in_video_section:
            continue
        match = entry_pattern.search(line)
        if match:
            index, name = match.group(1), match.group(2).strip()
            devices.append(CameraDevice(id=index, label=name, backend="avfoundation"))
    return devices


def _enumerate_cameras_windows(ffmpeg_path: str) -> list[CameraDevice]:
    stderr_text = _run_ffmpeg_device_listing(
        ffmpeg_path, ["-f", "dshow", "-list_devices", "true", "-i", "dummy"]
    )
    return _parse_dshow_listing(stderr_text)


def _parse_dshow_listing(stderr_text: str) -> list[CameraDevice]:
    """Parse ``ffmpeg -f dshow -list_devices`` stderr into CameraDevice entries.

    Each device is announced as a quoted display name line, optionally
    followed by an ``Alternative name "@device_pnp_..."`` line. The
    alternative name is preferred as the ffmpeg ``-i`` identity because it is
    unique even when two devices share the same display name.
    """
    devices: list[CameraDevice] = []
    in_video_section = False
    name_pattern = re.compile(r'"([^"]+)"')
    alt_pattern = re.compile(r'Alternative name\s+"([^"]+)"')
    pending_name: str | None = None

    def flush(alt_id: str | None = None) -> None:
        nonlocal pending_name
        if pending_name is not None:
            devices.append(
                CameraDevice(id=alt_id or pending_name, label=pending_name, backend="dshow")
            )
            pending_name = None

    for line in stderr_text.splitlines():
        if "DirectShow video devices" in line:
            in_video_section = True
            continue
        if "DirectShow audio devices" in line:
            flush()
            in_video_section = False
            continue
        if not in_video_section:
            continue
        alt_match = alt_pattern.search(line)
        if alt_match:
            flush(alt_match.group(1))
            continue
        name_match = name_pattern.search(line)
        if name_match:
            flush()
            pending_name = name_match.group(1)
    flush()
    return devices


# ── Resolution/FPS probing (best-effort, GUI display only) ─────────────────

_CANDIDATE_RESOLUTIONS = [(640, 480), (1280, 720), (1920, 1080)]
_CANDIDATE_FPS = [15, 24, 30, 60]


def default_resolutions() -> list[str]:
    return ["640x480", "1280x720", "1920x1080"]


def default_fps_values() -> list[int]:
    return [15, 30, 60]


def probe_camera_capabilities(cv_target: int | str) -> tuple[list[str], list[int]]:
    """Best-effort probe of resolutions/fps a camera supports, via OpenCV.

    Always releases the capture handle before returning so ffmpeg can open
    the same device immediately afterwards without a "device busy" error.
    Falls back to sane defaults if OpenCV can't open the device.
    """
    cap = cv2.VideoCapture(cv_target)
    try:
        if not cap.isOpened():
            return default_resolutions(), default_fps_values()

        resolutions: list[str] = []
        for width, height in _CANDIDATE_RESOLUTIONS:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            actual = f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}"
            if actual not in resolutions:
                resolutions.append(actual)

        fps_values: list[int] = []
        for fps in _CANDIDATE_FPS:
            cap.set(cv2.CAP_PROP_FPS, fps)
            actual_fps = int(cap.get(cv2.CAP_PROP_FPS)) or fps
            if actual_fps not in fps_values:
                fps_values.append(actual_fps)

        return resolutions or default_resolutions(), sorted(fps_values) or default_fps_values()
    finally:
        cap.release()


def probe_camera_for_gui(device: CameraDevice, order_index: int) -> tuple[list[str], list[int]]:
    """Probe a camera for the GUI, guessing an OpenCV target for it.

    On Linux the ffmpeg device path doubles as a valid OpenCV target. On
    macOS/Windows there is no guaranteed index mapping, so the camera's
    position in the enumerated list is used as a best-effort guess; if it's
    wrong, probing simply fails open and defaults are used instead (the
    resolution/fps fields remain editable either way).
    """
    target: int | str = device.id if device.backend == "v4l2" else order_index
    try:
        return probe_camera_capabilities(target)
    except Exception:
        return default_resolutions(), default_fps_values()


def _pick_default(options: list[str], preferred: str) -> str:
    return preferred if preferred in options else (options[0] if options else preferred)


# ── FFmpeg command construction (pure, per-OS) ──────────────────────────────


_PREVIEW_SNAPSHOT_FPS = 2
_PREVIEW_SNAPSHOT_WIDTH = 320


def build_record_command(
    ffmpeg_path: str,
    camera: CameraDevice,
    *,
    framerate: int,
    video_size: str,
    output_path: Path,
    preview_snapshot_path: Path | None = None,
) -> list[str]:
    """Build the ffmpeg argv to record ``camera`` losslessly to ``output_path``.

    Uses ``-c:v copy`` (stream copy, no re-encoding). Deliberately omits
    ``-nostdin`` and ``-t <duration>``: recording is open-ended and stopped
    interactively by sending ``q`` on stdin (see MulticamRecorderController).

    While ffmpeg holds exclusive access to the camera for recording, the GUI
    can't open its own capture for a live preview. If ``preview_snapshot_path``
    is given, a second output is added mapping the same input through a small
    decode+downscale branch that continuously overwrites that path with a low
    frame-rate JPEG — the recorded file stays untouched stream-copy, only this
    extra branch costs any CPU, and the preview mosaic just re-reads that file.
    """
    input_args = _record_input_args(camera, framerate=framerate, video_size=video_size)
    cmd = [ffmpeg_path, "-y", *input_args, "-map", "0:v", "-c:v", "copy", str(output_path)]
    if preview_snapshot_path is not None:
        cmd += [
            "-map",
            "0:v",
            "-vf",
            f"fps={_PREVIEW_SNAPSHOT_FPS},scale={_PREVIEW_SNAPSHOT_WIDTH}:-1",
            "-update",
            "1",
            str(preview_snapshot_path),
        ]
    return cmd


def _record_input_args(camera: CameraDevice, *, framerate: int, video_size: str) -> list[str]:
    if camera.backend == "v4l2":
        return _record_args_linux(camera, framerate=framerate, video_size=video_size)
    if camera.backend == "avfoundation":
        return _record_args_macos(camera, framerate=framerate, video_size=video_size)
    if camera.backend == "dshow":
        return _record_args_windows(camera, framerate=framerate, video_size=video_size)
    raise ValueError(f"Unsupported camera backend: {camera.backend}")


def _record_args_linux(camera: CameraDevice, *, framerate: int, video_size: str) -> list[str]:
    return [
        "-f",
        "v4l2",
        "-input_format",
        "mjpeg",
        "-framerate",
        str(framerate),
        "-video_size",
        video_size,
        "-i",
        camera.id,
    ]


def _record_args_macos(camera: CameraDevice, *, framerate: int, video_size: str) -> list[str]:
    return [
        "-f",
        "avfoundation",
        "-framerate",
        str(framerate),
        "-video_size",
        video_size,
        "-vcodec",
        "mjpeg",
        "-i",
        camera.id,
    ]


def _record_args_windows(camera: CameraDevice, *, framerate: int, video_size: str) -> list[str]:
    return [
        "-f",
        "dshow",
        "-framerate",
        str(framerate),
        "-video_size",
        video_size,
        "-vcodec",
        "mjpeg",
        "-i",
        f"video={camera.id}",
    ]


def format_record_cli_command(cmd: list[str]) -> str:
    """Render an ffmpeg argv as a copy-pasteable shell command (GUI→CLI mirror)."""
    return " ".join(shlex.quote(part) for part in cmd)


# ── Output naming ────────────────────────────────────────────────────────────


def build_session_dir(output_dir: Path, session_name: str, timestamp: str | None = None) -> Path:
    safe_name = sanitize_output_basename(session_name) or "session"
    ts = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    return output_dir / f"{safe_name}_{ts}"


def build_output_filename(camera_index: int, session_name: str) -> str:
    """Build the per-camera output filename.

    Uses ``.mp4`` (rather than e.g. ``.mkv``) since that's one of the
    extensions vailá's Markerless 2D tools scan for (``.mp4``/``.avi``/
    ``.mov``) — recordings are then ready to analyze without renaming.
    """
    safe_name = sanitize_output_basename(session_name) or "session"
    return f"cam{camera_index}_{safe_name}.mp4"


# ── Remembered output directory (~/.vaila/vaila_config.toml) ────────────────

_VAILA_CONFIG_PATH = Path.home() / ".vaila" / "vaila_config.toml"


def load_last_output_dir() -> str:
    """Read the last-used output directory, or "" if none saved/readable.

    Deliberately never raises: a corrupt or missing config must not stop the
    tool from opening, it just falls back to an empty output-dir field.
    """
    if not _VAILA_CONFIG_PATH.is_file():
        return ""
    try:
        import toml

        config = toml.load(_VAILA_CONFIG_PATH)
    except Exception as exc:  # noqa: BLE001
        print(f">> vaila/multicam_recorder: ignoring unreadable config {_VAILA_CONFIG_PATH}: {exc}")
        return ""
    return config.get("multicam_recorder", {}).get("last_output_dir", "")


def save_last_output_dir(output_dir: str) -> None:
    """Persist ``output_dir`` so the next session starts with it pre-filled."""
    try:
        import toml

        config = {}
        if _VAILA_CONFIG_PATH.is_file():
            config = toml.load(_VAILA_CONFIG_PATH)
        config.setdefault("multicam_recorder", {})["last_output_dir"] = output_dir
        _VAILA_CONFIG_PATH.parent.mkdir(parents=True, exist_ok=True)
        with open(_VAILA_CONFIG_PATH, "w", encoding="utf-8") as fh:
            toml.dump(config, fh)
    except Exception as exc:  # noqa: BLE001
        print(f">> vaila/multicam_recorder: could not save config {_VAILA_CONFIG_PATH}: {exc}")


# ── Live preview (owns OpenCV capture/window, pre-recording only) ──────────


def _restore_cv2_qt_plugin_env() -> None:
    """Re-point Qt at cv2's bundled plugins so cv2.imshow can create a window.

    vaila.syncvid scrubs QT_QPA_PLATFORM_PLUGIN_PATH/QT_QPA_FONTDIR right
    after cv2 sets them (see its ``_scrub_cv2_qt_plugin_env`` docstring), so
    that unrelated subprocesses this app spawns elsewhere (webbrowser, the
    Cut Video handoff) don't inherit cv2's bundled Qt runtime and crash.
    cutvideo -- imported below for ``sanitize_output_basename`` -- pulls in
    syncvid transitively, so that scrub has already run by the time this
    module loads. The live preview genuinely needs cv2's Qt-based imshow,
    so re-apply the same paths cv2 itself would set, but only while a
    preview window is actually open; :func:`_scrub_cv2_qt_plugin_env` undoes
    it again once the last one closes.
    """
    cv2_dir = Path(cv2.__file__).parent
    plugins_dir = cv2_dir / "qt" / "plugins"
    if plugins_dir.is_dir():
        os.environ["QT_QPA_PLATFORM_PLUGIN_PATH"] = str(plugins_dir)
    fonts_dir = cv2_dir / "qt" / "fonts"
    if fonts_dir.is_dir():
        os.environ["QT_QPA_FONTDIR"] = str(fonts_dir)


def _scrub_cv2_qt_plugin_env() -> None:
    """Undo :func:`_restore_cv2_qt_plugin_env` (mirrors vaila.syncvid's scrub)."""
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = os.environ.get(name, "")
        if "cv2" in value.replace("\\", "/").lower():
            os.environ.pop(name, None)


_PREVIEW_WINDOW_TITLE = "vailá - Camera Preview"
_PREVIEW_TILE_SIZE = (240, 180)  # (width, height) per camera, kept small on purpose


class _LivePreviewSlot:
    """One camera's capture, drained in real time by a dedicated thread.

    cv2.VideoCapture buffers frames internally (driver + OpenCV queues); if
    the display side only reads one frame per GUI tick and the camera can't
    keep up with that cadence, the backlog isn't discarded — it grows into a
    multi-second lag over time. cv2.read()/grab() are plain V4L2 I/O, not
    Qt/GUI calls, so — unlike imshow — they're safe to run off the main
    thread: a background thread continuously reads as fast as the camera
    produces frames and keeps only the newest one, so the mosaic always
    shows "now" instead of the stale head of a growing queue.
    """

    is_recording_snapshot = False

    def __init__(self, cap: cv2.VideoCapture, label: str, on_stopped: Callable[[], None]) -> None:
        self.label = label
        self.on_stopped = on_stopped
        self._cap = cap
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._latest: np.ndarray | None = None
        self._alive = True
        self._thread = threading.Thread(target=self._drain, daemon=True)
        self._thread.start()

    def _drain(self) -> None:
        while not self._stop_event.is_set():
            ok, frame = self._cap.read()
            with self._lock:
                if not ok:
                    self._alive = False
                    return
                self._latest = frame

    def latest_frame(self) -> tuple[bool, np.ndarray | None]:
        """Return (alive, frame). ``frame`` is None until the first read lands."""
        with self._lock:
            return self._alive, self._latest

    def close(self) -> None:
        self._stop_event.set()
        self._thread.join(timeout=1.0)
        self._cap.release()


class _SnapshotPreviewSlot:
    """Preview source that polls a JPEG ffmpeg keeps overwriting.

    Used while ffmpeg holds exclusive access to the camera for recording:
    since the GUI can no longer open that device itself, ffmpeg is asked to
    also continuously overwrite a small low-res JPEG from the same feed (see
    ``build_record_command``'s ``preview_snapshot_path``), and this just
    re-reads that file each tick instead.
    """

    is_recording_snapshot = True

    def __init__(self, jpg_path: Path, label: str, on_stopped: Callable[[], None]) -> None:
        self.label = label
        self.on_stopped = on_stopped
        self._jpg_path = jpg_path

    def latest_frame(self) -> tuple[bool, np.ndarray | None]:
        # Always "alive": a missing/mid-write file just isn't ready yet, and
        # this slot's lifetime is tied to the recording, not to read success.
        return True, cv2.imread(str(self._jpg_path))

    def close(self) -> None:
        pass  # the file belongs to ffmpeg; nothing here to release


def _build_mosaic(tiles: list[np.ndarray], tile_size: tuple[int, int]) -> np.ndarray:
    """Arrange same-sized tiles into a roughly-square grid, padding the last row."""
    width, height = tile_size
    count = len(tiles)
    cols = math.ceil(math.sqrt(count))
    rows = math.ceil(count / cols)
    canvas = np.zeros((rows * height, cols * width, 3), dtype=np.uint8)
    for index, tile in enumerate(tiles):
        r, c = divmod(index, cols)
        canvas[r * height : (r + 1) * height, c * width : (c + 1) * width] = tile
    return canvas


class CameraPreviewController:
    """Owns a single small live-preview window mosaicking every active camera.

    OpenCV's Qt-based highgui backend requires its *first* GUI call
    (``imshow``, ``waitKey``, window queries — Qt lazily creates a
    ``QApplication`` at that point) to happen on the process's main thread;
    calling it from a worker thread aborts the process. Since the Tkinter
    GUI already owns the main thread inside ``mainloop()``, previews are not
    run on a separate thread at all: ``poll()`` reads a frame from each
    active camera, tiles them into one small mosaic image, and shows that in
    a single window; it must be invoked periodically from the main thread
    via ``window.after(...)``, piggy-backing on Tk's own event loop instead
    of competing with it. Each capture handle is released as soon as its
    slot is removed, so ffmpeg can open the same device right afterwards
    without a "device busy" error.
    """

    def __init__(self) -> None:
        self._slots: dict[str, _LivePreviewSlot | _SnapshotPreviewSlot] = {}
        self._window_open = False

    def is_active(self, camera_id: str) -> bool:
        return camera_id in self._slots

    def start(
        self,
        camera_id: str,
        cv_target: int | str,
        label: str,
        on_stopped: Callable[[], None],
        *,
        framerate: int | None = None,
        video_size: str | None = None,
    ) -> None:
        if camera_id in self._slots:
            return

        cap = cv2.VideoCapture(cv_target)
        if not cap.isOpened():
            cap.release()
            on_stopped()
            return
        if video_size:
            try:
                width, height = (int(part) for part in video_size.lower().split("x"))
                cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
                cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            except ValueError:
                pass
        if framerate:
            cap.set(cv2.CAP_PROP_FPS, framerate)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # best-effort; ignored where unsupported

        if not self._slots:
            _restore_cv2_qt_plugin_env()
        self._slots[camera_id] = _LivePreviewSlot(cap, label, on_stopped)

    def show_snapshot(
        self, camera_id: str, jpg_path: Path, label: str, on_stopped: Callable[[], None]
    ) -> None:
        """Show/switch this camera's tile to ffmpeg's recording-time snapshot JPEG.

        Closes whatever preview source was active first (releasing a live
        V4L2 handle if there was one), so ffmpeg can hold the device.
        """
        existing = self._slots.pop(camera_id, None)
        if existing is not None:
            existing.close()
        if not self._slots:
            _restore_cv2_qt_plugin_env()
        self._slots[camera_id] = _SnapshotPreviewSlot(jpg_path, label, on_stopped)

    def show_live(
        self,
        camera_id: str,
        cv_target: int | str,
        label: str,
        on_stopped: Callable[[], None],
        *,
        framerate: int | None = None,
        video_size: str | None = None,
    ) -> None:
        """Switch this camera's tile back to a live capture once recording stops."""
        existing = self._slots.pop(camera_id, None)
        if existing is not None:
            existing.close()
        self.start(camera_id, cv_target, label, on_stopped, framerate=framerate, video_size=video_size)

    def stop(self, camera_id: str) -> None:
        slot = self._slots.pop(camera_id, None)
        if slot is None:
            return
        slot.close()
        if not self._slots:
            self._close_window()
            _scrub_cv2_qt_plugin_env()
        slot.on_stopped()

    def stop_all(self) -> None:
        for camera_id in list(self._slots):
            self.stop(camera_id)

    def _close_window(self) -> None:
        if self._window_open:
            try:
                cv2.destroyWindow(_PREVIEW_WINDOW_TITLE)
            except cv2.error:
                pass
            self._window_open = False

    def poll(self) -> None:
        """Read a frame from each active camera and refresh the mosaic window.

        Call periodically from the Tk main thread.
        """
        if not self._slots:
            return

        to_stop: set[str] = set()
        tiles: list[np.ndarray] = []
        for camera_id, slot in self._slots.items():
            alive, frame = slot.latest_frame()
            if not alive:
                to_stop.add(camera_id)
                continue
            if frame is None:
                continue  # first frame hasn't landed yet; skip this tick
            tile = cv2.resize(frame, _PREVIEW_TILE_SIZE)
            if slot.is_recording_snapshot:
                label_text, color = f"REC {slot.label}", (0, 0, 255)  # BGR red
            else:
                label_text, color = slot.label, (0, 255, 0)
            cv2.putText(
                tile,
                label_text,
                (6, 18),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                color,
                1,
                cv2.LINE_AA,
            )
            tiles.append(tile)

        if tiles:
            cv2.imshow(_PREVIEW_WINDOW_TITLE, _build_mosaic(tiles, _PREVIEW_TILE_SIZE))
            self._window_open = True
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):  # 'q' or Esc closes the whole mosaic
                to_stop.update(self._slots)
            else:
                try:
                    if cv2.getWindowProperty(_PREVIEW_WINDOW_TITLE, cv2.WND_PROP_VISIBLE) < 1:
                        to_stop.update(self._slots)
                except cv2.error:
                    to_stop.update(self._slots)

        for camera_id in to_stop:
            self.stop(camera_id)


# ── Recording controller (owns live ffmpeg subprocesses) ────────────────────


@dataclass
class ActiveRecording:
    camera: CameraDevice
    process: subprocess.Popen
    output_path: Path
    alive: bool = True


class MulticamRecorderController:
    """Owns the live ffmpeg subprocesses for one recording session."""

    def __init__(self, ffmpeg_path: str) -> None:
        self.ffmpeg_path = ffmpeg_path
        self._active: dict[str, ActiveRecording] = {}
        self._preview_snapshot_dir: Path | None = None

    @property
    def is_recording(self) -> bool:
        return bool(self._active)

    def start(
        self,
        cameras: list[tuple[CameraDevice, int, str]],
        session_dir: Path,
        session_name: str,
    ) -> tuple[list[str], dict[str, Path]]:
        """Launch one ffmpeg subprocess per (camera, framerate, video_size).

        Returns GUI→CLI mirror lines (one per camera, ready to print) and a
        ``{camera_id: snapshot_jpg_path}`` map so the live-preview mosaic can
        keep showing each camera (via that low-res JPEG, see
        ``build_record_command``) while ffmpeg holds the device.
        """
        session_dir.mkdir(parents=True, exist_ok=True)
        if self._preview_snapshot_dir is None:
            self._preview_snapshot_dir = Path(
                tempfile.mkdtemp(prefix="vaila_multicam_preview_")
            )
        cli_lines: list[str] = []
        snapshot_paths: dict[str, Path] = {}
        for index, (camera, framerate, video_size) in enumerate(cameras, start=1):
            output_path = session_dir / build_output_filename(index, session_name)
            snapshot_path = self._preview_snapshot_dir / f"cam{index}_preview.jpg"
            cmd = build_record_command(
                self.ffmpeg_path,
                camera,
                framerate=framerate,
                video_size=video_size,
                output_path=output_path,
                preview_snapshot_path=snapshot_path,
            )
            process = subprocess.Popen(
                cmd,
                stdin=subprocess.PIPE,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.PIPE,
            )
            self._active[camera.id] = ActiveRecording(
                camera=camera, process=process, output_path=output_path
            )
            snapshot_paths[camera.id] = snapshot_path
            cli_lines.append(
                f">> vaila/multicam_recorder: Equivalent CLI for cam{index} (copy/paste):\n"
                f">>   {format_record_cli_command(cmd)}"
            )
        return cli_lines, snapshot_paths

    def cleanup_preview_snapshots(self) -> None:
        """Remove the temp dir used for recording-time preview JPEGs, if any."""
        if self._preview_snapshot_dir is not None:
            shutil.rmtree(self._preview_snapshot_dir, ignore_errors=True)
            self._preview_snapshot_dir = None

    def poll_disconnected(self) -> list[ActiveRecording]:
        """Return recordings whose process exited unexpectedly, marking them dead."""
        newly_dead = []
        for rec in self._active.values():
            if rec.alive and rec.process.poll() is not None:
                rec.alive = False
                newly_dead.append(rec)
        return newly_dead

    def stop(self, timeout: float = 8.0) -> dict[str, Path]:
        """Gracefully stop every active recording.

        Sends ``q`` on each process's stdin (ffmpeg's interactive quit
        command, which finalizes the container properly), waits on all
        processes concurrently so one hung/disconnected camera can't block
        the others, then escalates to terminate()/kill() as a last resort.

        Returns ``{camera_id: output_path}`` for every recording that was
        active when Stop was pressed.
        """
        recordings = list(self._active.values())
        for rec in recordings:
            if rec.process.poll() is None:
                try:
                    if rec.process.stdin is not None:
                        rec.process.stdin.write(b"q")
                        rec.process.stdin.flush()
                except (BrokenPipeError, OSError):
                    pass

        def _wait(rec: ActiveRecording) -> None:
            try:
                rec.process.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                rec.process.terminate()
                try:
                    rec.process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    rec.process.kill()
                    rec.process.wait(timeout=3)

        if recordings:
            with ThreadPoolExecutor(max_workers=len(recordings)) as pool:
                list(pool.map(_wait, recordings))

        results = {rec.camera.id: rec.output_path for rec in recordings}
        self._active.clear()
        return results


# ── GUI ──────────────────────────────────────────────────────────────────────


class _CameraRow:
    """One camera's checkbox + resolution/fps pickers + status label."""

    def __init__(
        self,
        parent: tk.Widget,
        grid_row: int,
        device: CameraDevice,
        resolutions: list[str],
        fps_values: list[int],
        on_preview: Callable[["_CameraRow"], None],
    ) -> None:
        self.device = device
        self.var_selected = tk.BooleanVar(value=True)
        self.var_resolution = tk.StringVar(value=_pick_default(resolutions, "1280x720"))
        self.var_fps = tk.StringVar(value=_pick_default([str(f) for f in fps_values], "30"))
        self.status_var = tk.StringVar(value="idle")

        self.check = ttk.Checkbutton(parent, text=device.label, variable=self.var_selected)
        self.resolution_box = ttk.Combobox(
            parent, textvariable=self.var_resolution, values=resolutions, width=10
        )
        self.fps_box = ttk.Combobox(
            parent,
            textvariable=self.var_fps,
            values=[str(f) for f in fps_values],
            width=6,
        )
        self.status_label = ttk.Label(parent, textvariable=self.status_var, width=14)
        self.preview_btn = ttk.Button(
            parent, text="Preview", width=12, command=lambda: on_preview(self)
        )

        self.check.grid(row=grid_row, column=0, sticky="w", padx=4, pady=2)
        self.resolution_box.grid(row=grid_row, column=1, padx=4, pady=2)
        self.fps_box.grid(row=grid_row, column=2, padx=4, pady=2)
        self.status_label.grid(row=grid_row, column=3, padx=4, pady=2)
        self.preview_btn.grid(row=grid_row, column=4, padx=4, pady=2)

    def framerate(self) -> int:
        try:
            return int(self.var_fps.get())
        except ValueError:
            return 30

    def video_size(self) -> str:
        return self.var_resolution.get().strip() or "1280x720"

    def set_previewing(self, active: bool) -> None:
        self.preview_btn.config(text="Stop Preview" if active else "Preview")

    def set_enabled(self, enabled: bool) -> None:
        state = "readonly" if enabled else "disabled"
        check_state = "normal" if enabled else "disabled"
        self.check.config(state=check_state)
        self.resolution_box.config(state=state)
        self.fps_box.config(state=state)
        self.preview_btn.config(state=check_state)


class MulticamRecorderApp:
    """GUI controller: detected-cameras list + output settings + Start/Stop."""

    def __init__(self, window: tk.Toplevel) -> None:
        self.window = window
        self.window.title("vailá - Record Cameras")
        self.controller = MulticamRecorderController(get_ffmpeg_path())
        self.preview_controller = CameraPreviewController()
        self.rows: list[_CameraRow] = []
        self.output_dir_var = tk.StringVar(value=load_last_output_dir())
        self.session_name_var = tk.StringVar(value="trial01")
        self.is_recording = False
        self._watchdog_job: str | None = None

        self._build_widgets()
        self.rescan_cameras()
        self._preview_tick()

        self.window.protocol("WM_DELETE_WINDOW", self.on_close)

    def _preview_tick(self) -> None:
        self.preview_controller.poll()
        self.window.after(33, self._preview_tick)

    def _build_widgets(self) -> None:
        outer = ttk.Frame(self.window, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text="Detected cameras (select which to record, adjust resolution/fps):",
        ).pack(anchor="w")

        self.camera_frame = ttk.Frame(outer)
        self.camera_frame.pack(fill="both", expand=True, pady=(4, 8))

        self.rescan_btn = ttk.Button(outer, text="Rescan cameras", command=self.rescan_cameras)
        self.rescan_btn.pack(anchor="w", pady=(0, 10))

        output_row = ttk.Frame(outer)
        output_row.pack(fill="x", pady=4)
        ttk.Label(output_row, text="Output directory:").pack(side="left")
        self.output_entry = ttk.Entry(output_row, textvariable=self.output_dir_var, width=40)
        self.output_entry.pack(side="left", padx=6)
        self.output_btn = ttk.Button(output_row, text="Browse...", command=self._choose_output_dir)
        self.output_btn.pack(side="left")

        session_row = ttk.Frame(outer)
        session_row.pack(fill="x", pady=4)
        ttk.Label(session_row, text="Session/trial name:").pack(side="left")
        self.session_entry = ttk.Entry(session_row, textvariable=self.session_name_var, width=30)
        self.session_entry.pack(side="left", padx=6)

        self.toggle_btn = ttk.Button(outer, text="Start Recording", command=self._on_toggle)
        self.toggle_btn.pack(pady=(12, 0))

    def _choose_output_dir(self) -> None:
        selected = filedialog.askdirectory(
            parent=self.window,
            title="Select output directory",
            initialdir=self.output_dir_var.get().strip() or None,
        )
        if selected:
            self.output_dir_var.set(selected)
            save_last_output_dir(selected)

    def rescan_cameras(self) -> None:
        self.preview_controller.stop_all()
        for widget in self.camera_frame.winfo_children():
            widget.destroy()
        self.rows = []

        devices = enumerate_cameras()
        if not devices:
            ttk.Label(
                self.camera_frame,
                text="No cameras detected. Connect a camera and click Rescan.",
            ).grid(row=0, column=0, columnspan=4, padx=4, pady=8)
            return

        ttk.Label(self.camera_frame, text="Camera").grid(row=0, column=0, sticky="w", padx=4)
        ttk.Label(self.camera_frame, text="Resolution").grid(row=0, column=1, padx=4)
        ttk.Label(self.camera_frame, text="FPS").grid(row=0, column=2, padx=4)
        ttk.Label(self.camera_frame, text="Status").grid(row=0, column=3, padx=4)
        ttk.Label(self.camera_frame, text="Preview").grid(row=0, column=4, padx=4)

        for order_index, device in enumerate(devices):
            resolutions, fps_values = probe_camera_for_gui(device, order_index)
            row = _CameraRow(
                self.camera_frame,
                order_index + 1,
                device,
                resolutions,
                fps_values,
                self._on_preview_toggle,
            )
            self.rows.append(row)

        # Live preview starts automatically on every detected camera, so you
        # can check focus/framing right away without clicking anything; the
        # per-row button is only there to close/reopen an individual window.
        for row in self.rows:
            self._start_preview(row)

    def _on_preview_toggle(self, row: _CameraRow) -> None:
        if self.preview_controller.is_active(row.device.id):
            self.preview_controller.stop(row.device.id)
        else:
            self._start_preview(row)

    def _start_preview(self, row: _CameraRow) -> None:
        if self.preview_controller.is_active(row.device.id):
            return

        order_index = self.rows.index(row)
        cv_target: int | str = row.device.id if row.device.backend == "v4l2" else order_index

        row.set_previewing(True)
        self.preview_controller.start(
            row.device.id,
            cv_target,
            row.device.label,
            self._make_on_preview_stopped(row),
            framerate=row.framerate(),
            video_size=row.video_size(),
        )

    def _make_on_preview_stopped(self, row: _CameraRow) -> Callable[[], None]:
        def on_stopped() -> None:
            self.window.after(0, lambda: row.set_previewing(False))

        return on_stopped

    def _on_toggle(self) -> None:
        if self.is_recording:
            self._stop_recording()
        else:
            self._start_recording()

    def _start_recording(self) -> None:
        selected = [row for row in self.rows if row.var_selected.get()]
        if not selected:
            messagebox.showerror("vailá", "Select at least one camera.")
            return

        output_dir = self.output_dir_var.get().strip()
        if not output_dir or not Path(output_dir).is_dir():
            messagebox.showerror("vailá", "Choose a valid output directory.")
            return
        save_last_output_dir(output_dir)

        session_name = sanitize_output_basename(self.session_name_var.get())
        if not session_name:
            messagebox.showerror("vailá", "Enter a valid session/trial name.")
            return

        # Release live preview handles so ffmpeg can claim these devices; the
        # mosaic keeps showing each recorded camera below, re-sourced from
        # the low-res JPEG ffmpeg is asked to keep overwriting.
        self.preview_controller.stop_all()
        for row in self.rows:
            row.set_previewing(False)

        session_dir = build_session_dir(Path(output_dir), session_name)
        cameras = [(row.device, row.framerate(), row.video_size()) for row in selected]

        try:
            cli_lines, snapshot_paths = self.controller.start(cameras, session_dir, session_name)
        except OSError as exc:
            messagebox.showerror("vailá", f"Failed to start recording: {exc}")
            return

        for line in cli_lines:
            print(f"\n{line}", flush=True)

        for row in selected:
            row.status_var.set("recording")
            self.preview_controller.show_snapshot(
                row.device.id,
                snapshot_paths[row.device.id],
                row.device.label,
                self._make_on_preview_stopped(row),
            )
            row.set_previewing(True)
        self._set_inputs_enabled(False)
        self.toggle_btn.config(text="Stop Recording")
        self.is_recording = True
        self._watchdog_job = self.window.after(1000, self._poll_watchdog)

    def _poll_watchdog(self) -> None:
        if not self.is_recording:
            return
        dead = self.controller.poll_disconnected()
        dead_ids = {rec.camera.id for rec in dead}
        for row in self.rows:
            if row.device.id in dead_ids:
                row.status_var.set("disconnected")
        self._watchdog_job = self.window.after(1000, self._poll_watchdog)

    def _stop_recording(self) -> None:
        self.toggle_btn.config(state="disabled")

        def worker() -> None:
            results = self.controller.stop()
            self.window.after(0, lambda: self._on_stop_complete(results))

        threading.Thread(target=worker, daemon=True).start()

    def _on_stop_complete(self, results: dict[str, Path]) -> None:
        if self._watchdog_job is not None:
            self.window.after_cancel(self._watchdog_job)
            self._watchdog_job = None

        self.is_recording = False
        self._set_inputs_enabled(True)
        self.toggle_btn.config(text="Start Recording", state="normal")

        for row in self.rows:
            if row.device.id not in results:
                continue
            if row.status_var.get() != "disconnected":
                row.status_var.set("stopped")
            # ffmpeg has released the device by now (controller.stop() already
            # waited for it to exit), so a live capture can reopen it.
            order_index = self.rows.index(row)
            cv_target: int | str = row.device.id if row.device.backend == "v4l2" else order_index
            self.preview_controller.show_live(
                row.device.id,
                cv_target,
                row.device.label,
                self._make_on_preview_stopped(row),
                framerate=row.framerate(),
                video_size=row.video_size(),
            )
            row.set_previewing(True)

        summary = "\n".join(str(path) for path in results.values())
        messagebox.showinfo(
            "vailá",
            f"Recording stopped.\n\nFiles written:\n{summary or '(none)'}",
        )

    def _set_inputs_enabled(self, enabled: bool) -> None:
        for row in self.rows:
            row.set_enabled(enabled)
        state = "normal" if enabled else "disabled"
        self.rescan_btn.config(state=state)
        self.output_btn.config(state=state)
        self.output_entry.config(state=state)
        self.session_entry.config(state=state)

    def on_close(self) -> None:
        if self.is_recording:
            if not messagebox.askyesno(
                "vailá", "Recording is in progress. Stop and close the window?"
            ):
                return
            self.controller.stop()
        self.preview_controller.stop_all()
        self.controller.cleanup_preview_snapshots()
        self.window.destroy()


def run_multicam_recorder(parent: tk.Tk | tk.Toplevel | None = None) -> None:
    """Open the multi-camera simultaneous recording tool."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting multicam_recorder...")

    created_root = False
    default_root = cast("tk.Tk | tk.Toplevel | None", getattr(tk, "_default_root", None))
    root = parent or default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True

    window = tk.Toplevel(root)
    MulticamRecorderApp(window)

    if created_root:
        root.mainloop()


if __name__ == "__main__":
    run_multicam_recorder()
