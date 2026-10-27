"""
Project: vailá Multimodal Toolbox
Script: syncvid.py - Interactive Multi-Video Synchronization

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 27 October 2026
Version: 0.4.12

Description:
Create a frame-accurate synchronization plan with a fast Pygame player. Every
video in the target directory can be reviewed with the same playback/navigation
keys used by ``vaila.cutvideo``. Synchronization frames, reference video, and
reference start/end are entered directly in editable on-screen fields.

Sync by events calibrates retained-frame ordinals piecewise between ordered
events, using an assumed rational reference clock and a separate output grid.
It saves an event JSON manifest and linked checkpoint; Cut Video exports exact
rational-FPS videos, temporal-error CSVs and retained PNGs with physical times.

The versioned TSV sync file is consumed directly by ``vaila.cutvideo``. The
``Save + Open Cut Video`` button launches Cut Video with both the reference
video and the generated sync file already selected.

Holding Left/Right (or Up/Down) keeps stepping frame by frame, as in
getpixelvideo. The mouse wheel zooms at the cursor; drag pans; ``0`` resets.
``G`` or the Go to frame field jumps to a typed frame.

Cameras with different frame rates are aligned in time at their sync (key)
frame, each at its own fps (shown per camera). A non-reference camera may
instead be cut at its own typed Start/End.

Repeated frames: an editor that splices several cameras into one timeline
repeats frames. Motion thresholds and alternating vertical field shifts detect
re-encoded copies. ``Detect repeats`` marks them
in every video; ``D``/Delete marks or keeps the current frame by hand.
``Save w/o repeats`` writes repeat-free copies at the estimated real frame
rate, a frame map and per-frame motion scores to
``processed_norepeat_<timestamp>/``. The marks also go into the sync file, so
Cut Video saves each synchronized cut with and without them.

Frame convention:
- GUI and sync file: 1-based, inclusive.
- Internal Python state: 0-based, inclusive.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import datetime as dt
import io
import json
import math
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import webbrowser
from collections.abc import Callable, Collection, Generator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from tkinter import messagebox
from typing import Any, cast

import cv2
import numpy as np
import pygame
from rich import print

try:
    from .sync_events import (
        EventSyncError,
        manifest_from_checkpoint,
        parse_event_frames,
        read_event_manifest,
        temporal_maps,
    )
except ImportError:
    from sync_events import (  # ty: ignore[unresolved-import]
        EventSyncError,
        manifest_from_checkpoint,
        parse_event_frames,
        read_event_manifest,
        temporal_maps,
    )


def _scrub_cv2_qt_plugin_env() -> None:
    """Undo cv2's QT_QPA_PLATFORM_PLUGIN_PATH/QT_QPA_FONTDIR import side effect.

    Importing cv2 unconditionally overwrites these two variables with its own
    bundled Qt runtime path, even though this module only uses cv2.VideoCapture
    and never touches cv2's Qt GUI backend. Left in place, every child process
    this module spawns (the Help button's webbrowser tab, the Cut Video
    handoff subprocess) inherits the leaked value; if that child -- or
    something it execs, such as a desktop's xdg-open handler -- is itself
    Qt-based, it tries to load cv2's bundled xcb plugin against the system's
    Qt/X11 and aborts with "qt.qpa.plugin: Could not load the Qt platform
    plugin xcb" / "Aborted (core dumped)". Restore the system default by
    removing only the values cv2 itself set.
    """
    for name in ("QT_QPA_PLATFORM_PLUGIN_PATH", "QT_QPA_FONTDIR"):
        value = os.environ.get(name, "")
        if "cv2" in value.replace("\\", "/").lower():
            os.environ.pop(name, None)


_scrub_cv2_qt_plugin_env()


# Child-process Tk chooser body (no vailá/pygame imports — keeps the child light).
_ISOLATED_TK_DIALOG_SCRIPT = r"""
import json
import sys
from tkinter import Tk, filedialog

spec = json.load(open(sys.argv[1], encoding="utf-8"))
root = Tk()
root.withdraw()
try:
    root.attributes("-topmost", True)
except Exception:
    pass
root.lift()
root.update_idletasks()
kwargs = {"title": spec["title"], "parent": root}
if spec.get("initialdir"):
    kwargs["initialdir"] = spec["initialdir"]
if spec.get("initialfile"):
    kwargs["initialfile"] = spec["initialfile"]
if spec.get("defaultextension"):
    kwargs["defaultextension"] = spec["defaultextension"]
if spec.get("filetypes"):
    kwargs["filetypes"] = [tuple(item) for item in spec["filetypes"]]
kind = spec["kind"]
if kind == "save":
    result = filedialog.asksaveasfilename(**kwargs)
elif kind == "open":
    result = filedialog.askopenfilename(**kwargs)
elif kind == "dir":
    result = filedialog.askdirectory(**kwargs)
else:
    raise SystemExit(f"unknown dialog kind {kind!r}")
sys.stdout.write(result or "")
"""


def _zenity_path_dialog(
    kind: str,
    *,
    title: str,
    initialdir: str | Path | None = None,
    initialfile: str | None = None,
) -> str | None:
    """GNOME/GTK file chooser via zenity, or None if zenity is unavailable."""
    zenity = shutil.which("zenity")
    if not zenity:
        return None
    cmd = [zenity, "--file-selection", f"--title={title}"]
    if kind == "save":
        cmd.append("--save")
        cmd.append("--confirm-overwrite")
    elif kind == "dir":
        cmd.append("--directory")
    elif kind != "open":
        raise ValueError(f"unsupported dialog kind: {kind!r}")
    hint: Path | None = None
    if initialdir is not None:
        hint = Path(initialdir)
        if initialfile and kind == "save":
            hint = hint / initialfile
        elif kind != "dir" and hint.is_dir():
            hint = hint / "."
        cmd.append(f"--filename={hint}")
    proc = subprocess.run(cmd, capture_output=True, text=True, check=False)
    selected = (proc.stdout or "").strip()
    # Zenity: 0 = ok, 1 = cancel, anything else = error/unavailable.
    if proc.returncode == 1:
        return ""
    if proc.returncode != 0:
        return None
    return selected


def _isolated_tk_path_dialog(
    kind: str,
    *,
    title: str,
    initialdir: str | Path | None = None,
    initialfile: str | None = None,
    defaultextension: str | None = None,
    filetypes: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Run a Tk path chooser in a child process (no shared grab with pygame)."""
    if kind not in {"open", "save", "dir"}:
        raise ValueError(f"unsupported dialog kind: {kind!r}")
    spec = {
        "kind": kind,
        "title": title,
        "initialdir": str(initialdir) if initialdir is not None else None,
        "initialfile": initialfile,
        "defaultextension": defaultextension,
        "filetypes": [list(item) for item in filetypes] if filetypes else None,
    }
    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False, encoding="utf-8") as handle:
        json.dump(spec, handle)
        spec_path = handle.name
    try:
        proc = subprocess.run(
            [sys.executable, "-c", _ISOLATED_TK_DIALOG_SCRIPT, spec_path],
            capture_output=True,
            text=True,
            check=False,
        )
        selected = (proc.stdout or "").strip()
        if proc.returncode != 0:
            err = (proc.stderr or "").strip() or f"exit {proc.returncode}"
            raise OSError(f"File dialog failed: {err}")
        return selected
    finally:
        Path(spec_path).unlink(missing_ok=True)


def _portable_path_dialog(
    kind: str,
    *,
    title: str,
    initialdir: str | Path | None = None,
    initialfile: str | None = None,
    defaultextension: str | None = None,
    filetypes: Sequence[tuple[str, str]] | None = None,
) -> str:
    """Desktop-native path chooser (zenity) with isolated-Tk fallback.

    Prefer zenity on Linux so the chooser is a real GTK dialog the window
    manager stacks above SDL. Fall back to an out-of-process Tk chooser when
    zenity is missing (macOS/Windows CI, minimal containers).
    """
    zenity_result = _zenity_path_dialog(
        kind, title=title, initialdir=initialdir, initialfile=initialfile
    )
    if zenity_result is not None:
        return zenity_result
    return _isolated_tk_path_dialog(
        kind,
        title=title,
        initialdir=initialdir,
        initialfile=initialfile,
        defaultextension=defaultextension,
        filetypes=filetypes,
    )


def _resilient_open_local_html(path: Path) -> None:
    """Open a local HTML file, working around wslview/WSL-interop failures.

    ``webbrowser.open_new_tab()`` resolves to ``wslview`` whenever ``BROWSER``
    is unset and a ``wslview`` binary is on PATH -- which WSL's default
    Python distributions ship even when WSL interop itself is broken (no
    WSLg, a restricted/systemd-less distro, a container). wslview then shells
    out to ``reg.exe``/``explorer.exe`` through that interop and dies with
    ``grep: /proc/sys/fs/binfmt_misc/WSLInterop: No such file or directory``
    and ``wslview: line 216: .../reg.exe: No such file or directory`` --
    exactly the Help-button failure this works around. Native Linux GUI
    browsers are tried directly first; wslview is only considered once
    interop is confirmed to actually work; ``webbrowser`` is kept as a last
    library-level fallback for platforms (macOS, native Windows) where it
    already resolves correctly. A launch failure never reaches the caller --
    the last resort is printing the ``file://`` URI, so Help can never crash
    the pygame loop or spam a traceback to stderr.
    """
    uri = path.resolve().as_uri()
    candidates: list[list[str]] = []

    browser_env = os.environ.get("BROWSER", "").strip()
    if browser_env:
        candidates.append([browser_env, uri])

    for name in ("xdg-open", "google-chrome", "firefox", "chromium", "chromium-browser"):
        if shutil.which(name):
            candidates.append([name, uri])

    wsl_interop_available = Path("/proc/sys/fs/binfmt_misc/WSLInterop").exists()
    if wsl_interop_available and shutil.which("wslview"):
        candidates.append(["wslview", uri])

    for command in candidates:
        try:
            subprocess.Popen(
                command,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
            return
        except (OSError, subprocess.SubprocessError):
            continue

    with contextlib.suppress(webbrowser.Error, OSError):
        if webbrowser.open_new_tab(uri):
            return

    print(f">> vaila/syncvid: Open this file in a browser to view Help:\n   {uri}")


VIDEO_EXTENSIONS = frozenset({".mp4", ".avi", ".mov", ".mkv", ".webm", ".m4v"})
SYNC_FORMAT_HEADER = "# vaila sync file v2"
SYNC_COLUMNS = ("video_file", "output_file", "start_frame", "end_frame", "sync_frame")
# Optional sixth column, written only when frames were marked as repeated:
# space-separated 1-based source frames that Cut Video drops from its output.
SYNC_REMOVED_COLUMN = "removed_frames"
MAX_SYNC_FILE_BYTES = 1_000_000
MAX_SYNC_ENTRIES = 10_000
MAX_FRAME_NUMBER = 2_147_483_647
CHECKPOINT_SCHEMA_VERSION = 1
MAX_CHECKPOINT_FILE_BYTES = 1_000_000


PLAYBACK_SPEED_STEPS: tuple[float, ...] = (
    0.0625,
    0.125,
    0.25,
    0.5,
    1.0,
    2.0,
    4.0,
    8.0,
    16.0,
)
# Held navigation keys repeat like Cut Video / getpixelvideo (values from
# vaila.cutvideo, imported lazily because cutvideo imports this module).
PLAYER_NAV_KEYS: dict[int, int] = {
    pygame.K_RIGHT: 1,
    pygame.K_LEFT: -1,
    pygame.K_UP: 60,
    pygame.K_EQUALS: 60,
    pygame.K_KP_PLUS: 60,
    getattr(pygame, "K_PLUS", pygame.K_EQUALS): 60,
    pygame.K_DOWN: -60,
    pygame.K_MINUS: -60,
    pygame.K_KP_MINUS: -60,
}
PLAYER_ZOOM_MAX = 10.0
PLAYER_ZOOM_STEP = 1.1
PLAYER_STEP_BUTTONS: tuple[tuple[str, int], ...] = (
    ("-60", -60),
    ("-1", -1),
    ("+1", 1),
    ("+60", 60),
)


class SyncWorkflowError(ValueError):
    """User-facing synchronization validation error."""


class SyncFileError(SyncWorkflowError):
    """Invalid or unsafe synchronization-file content."""


@dataclass(frozen=True)
class VideoInfo:
    """Validated video metadata used by the synchronization UI."""

    path: Path
    frame_count: int
    fps: float
    width: int
    height: int

    @property
    def name(self) -> str:
        return self.path.name

    def with_fps(self, fps: float) -> VideoInfo:
        """Return a copy that uses ``fps`` for time-based sync math."""
        rate = float(fps)
        if not 0.0 < rate < 10_000.0:
            raise SyncWorkflowError(f"Real FPS must be between 0 and 10000 (got {fps})")
        if rate == self.fps:
            return self
        return VideoInfo(
            path=self.path,
            frame_count=self.frame_count,
            fps=rate,
            width=self.width,
            height=self.height,
        )


def parse_fps_field(text: str, *, field: str = "real FPS") -> float:
    """Parse a positive FPS typed in the UI."""
    raw = text.strip().replace(",", ".")
    if not raw:
        raise SyncWorkflowError(f"Enter a {field} value (e.g. 29.97 or 13.91)")
    try:
        rate = float(raw)
    except ValueError as exc:
        raise SyncWorkflowError(f"Invalid {field}: {text!r}") from exc
    if not 0.0 < rate < 10_000.0:
        raise SyncWorkflowError(f"{field} must be between 0 and 10000 (got {rate})")
    return rate


def fps_rates_match(a: float, b: float) -> bool:
    """True when two rates should share the same frame count in a sync cut.

    Tolerates tiny container vs typed differences (e.g. 29.970030 vs 29.97).
    """
    return math.isclose(float(a), float(b), rel_tol=1e-3, abs_tol=0.05)


def apply_fps_overrides(
    video_infos: Sequence[VideoInfo], overrides: Mapping[str, float] | None
) -> tuple[list[VideoInfo], list[str]]:
    """Apply real-FPS overrides that agree with each file's playback rate.

    Cut Video always seeks/writes using the **container** fps. An override that
    disagrees (e.g. typing 29.97 on a ``*_norepeat_13.91fps.mp4``) would force
    equal frame counts while the cut still plays at 13.91 fps — durations drift
    and the temporal window is wrong. Those overrides are ignored and listed in
    the returned warning strings.
    """
    if not overrides:
        return list(video_infos), []
    resolved: list[VideoInfo] = []
    warnings: list[str] = []
    for info in video_infos:
        if info.name not in overrides:
            resolved.append(info)
            continue
        rate = float(overrides[info.name])
        if fps_rates_match(rate, info.fps):
            resolved.append(info.with_fps(rate))
            continue
        warnings.append(
            f"{info.name}: ignored real FPS {rate:g} (file is {info.fps:.3f} fps); "
            "sync math keeps the file rate so Start/End stay the same events in time"
        )
        resolved.append(info)
    return resolved, warnings


def equal_fps_frame_span(reference_start: int, reference_end: int) -> int:
    """``end - start`` (0-based inclusive span) forced for equal-fps cameras."""
    return int(reference_end) - int(reference_start)


def fill_range_texts_from_plan(
    entries: Sequence[SyncPlanEntry],
    *,
    reference_video: str,
) -> dict[str, tuple[str, str]]:
    """1-based Start/End texts for every non-reference camera in ``entries``."""
    return {
        entry.video_file: (str(entry.start_frame + 1), str(entry.end_frame + 1))
        for entry in entries
        if entry.video_file != reference_video
    }


@dataclass(frozen=True)
class SyncPlanEntry:
    """One synchronized video range, stored internally as zero-based frames."""

    video_file: str
    output_file: str
    start_frame: int
    end_frame: int
    sync_frame: int | None = None
    # Source frames (0-based, inside start..end) marked as repeated.
    removed_frames: tuple[int, ...] = ()


@dataclass(frozen=True)
class SyncRunResult:
    """Result returned by the interactive dialog."""

    sync_file: Path
    reference_video: Path
    entries: tuple[SyncPlanEntry, ...]
    open_cutvideo: bool


def validate_sync_leaf_filename(
    value: str,
    *,
    field: str,
    allowed_extensions: frozenset[str] = VIDEO_EXTENSIONS,
) -> str:
    """Validate that a sync-file field is a safe filename, never a path."""
    if not isinstance(value, str) or not value:
        raise SyncFileError(f"{field} must be a non-empty filename")
    if value != value.strip():
        raise SyncFileError(f"{field} cannot start or end with whitespace: {value!r}")
    if any(ord(char) < 32 or ord(char) == 127 for char in value):
        raise SyncFileError(f"{field} contains control characters")

    posix = PurePosixPath(value)
    windows = PureWindowsPath(value)
    if (
        value in {".", ".."}
        or posix.is_absolute()
        or windows.is_absolute()
        or windows.drive
        or posix.name != value
        or windows.name != value
    ):
        raise SyncFileError(f"{field} must be a filename without directories: {value!r}")
    if Path(value).suffix.lower() not in allowed_extensions:
        extensions = ", ".join(sorted(allowed_extensions))
        raise SyncFileError(
            f"{field} must use a supported video extension ({extensions}): {value!r}"
        )
    return value


def discover_video_files(directory_path: str | Path) -> list[Path]:
    """Return validated direct-child videos from a target directory."""
    directory = Path(directory_path).expanduser().resolve()
    if not directory.is_dir():
        raise SyncWorkflowError(f"Video directory not found: {directory}")

    videos: list[Path] = []
    seen_names: set[str] = set()
    for candidate in sorted(directory.iterdir(), key=lambda path: path.name.casefold()):
        if candidate.name.startswith(".") or candidate.suffix.lower() not in VIDEO_EXTENSIONS:
            continue
        try:
            resolved = candidate.resolve(strict=True)
        except OSError as exc:
            raise SyncWorkflowError(f"Cannot resolve video path {candidate}: {exc}") from exc
        if resolved.parent != directory:
            raise SyncWorkflowError(
                f"Video symlink escapes the selected directory and was rejected: {candidate.name}"
            )
        if not resolved.is_file():
            continue
        folded = candidate.name.casefold()
        if folded in seen_names:
            raise SyncWorkflowError(
                f"Ambiguous video filenames differing only by letter case: {candidate.name}"
            )
        seen_names.add(folded)
        videos.append(resolved)
    return videos


def get_video_files(directory_path: str | Path) -> list[str]:
    """Backward-compatible filename-only wrapper."""
    return [path.name for path in discover_video_files(directory_path)]


def probe_video(video_path: str | Path) -> VideoInfo:
    """Open a video, validate the first frame, and return timeline metadata."""
    path = Path(video_path).expanduser().resolve()
    if not path.is_file():
        raise SyncWorkflowError(f"Video not found: {path}")

    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise SyncWorkflowError(f"OpenCV could not open video: {path}")
        ok, frame = cap.read()
        if not ok or frame is None:
            raise SyncWorkflowError(f"OpenCV could not decode the first frame: {path}")
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        height, width = frame.shape[:2]
    finally:
        cap.release()

    if frame_count <= 0:
        raise SyncWorkflowError(f"Video reports no frames: {path}")
    if not 0.0 < fps < 10_000.0:
        raise SyncWorkflowError(f"Video reports invalid FPS ({fps}): {path}")
    return VideoInfo(path=path, frame_count=frame_count, fps=fps, width=width, height=height)


def probe_video_directory(directory_path: str | Path) -> list[VideoInfo]:
    """Discover and validate every supported video in a directory."""
    videos = discover_video_files(directory_path)
    if len(videos) < 2:
        raise SyncWorkflowError(
            f"At least two videos are required for synchronization; found {len(videos)}"
        )
    return [probe_video(video) for video in videos]


def _frames_floor(value: float) -> int:
    """floor() tolerant to float noise (60 / 60 * 60 may give 59.999999)."""
    return math.floor(value + 1e-6)


def _frames_ceil(value: float) -> int:
    return math.ceil(value - 1e-6)


def common_reference_bounds(
    video_infos: Sequence[VideoInfo],
    sync_frames: Mapping[str, int],
    reference_video: str,
    keyframe_videos: Collection[str] | None = None,
) -> tuple[int, int]:
    """Return the largest reference-camera range present in every keyframe camera.

    Cameras are aligned in time at their sync (key) frames, each at its own
    fps. ``keyframe_videos`` (default: every video) are the cameras whose
    range follows the reference; cameras cut by their own Start/End are left
    out.
    """
    info_by_name = {info.name: info for info in video_infos}
    if reference_video not in info_by_name:
        raise SyncWorkflowError(
            f"Reference video is not in the target directory: {reference_video}"
        )
    names = set(info_by_name if keyframe_videos is None else keyframe_videos)
    names.add(reference_video)
    missing = [name for name in info_by_name if name in names and name not in sync_frames]
    if missing:
        raise SyncWorkflowError(
            "Mark a sync frame for every video without its own Start/End: " + ", ".join(missing)
        )

    # Common time window, in seconds relative to the sync frames.
    earliest = -float("inf")
    latest = float("inf")
    for name in names:
        info = info_by_name[name]
        sync_frame = int(sync_frames[name])
        if not 0 <= sync_frame < info.frame_count:
            raise SyncWorkflowError(
                f"Sync frame {sync_frame + 1} is outside {name} (1–{info.frame_count})"
            )
        earliest = max(earliest, -sync_frame / info.fps)
        latest = min(latest, (info.frame_count - 1 - sync_frame) / info.fps)

    reference = info_by_name[reference_video]
    reference_sync = int(sync_frames[reference_video])
    start = reference_sync + _frames_ceil(earliest * reference.fps)
    end = reference_sync + _frames_floor(latest * reference.fps)
    if start > end:
        raise SyncWorkflowError("The selected sync frames have no common video interval")
    return start, end


def _safe_output_name(
    video_name: str, sync_frame: int | None, start_frame: int, end_frame: int
) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(video_name).stem, flags=re.UNICODE).strip("._")
    stem = (stem or "video")[:120]
    sync_part = "" if sync_frame is None else f"_sync_{sync_frame + 1}"
    output_name = f"{stem}{sync_part}_frames_{start_frame + 1}_to_{end_frame + 1}.mp4"
    return validate_sync_leaf_filename(output_name, field="output_file")


def build_sync_plan(
    video_infos: Sequence[VideoInfo],
    sync_frames: Mapping[str, int],
    reference_video: str,
    reference_start: int,
    reference_end: int,
    removed_frames: Mapping[str, Collection[int]] | None = None,
    ranges: Mapping[str, tuple[int, int]] | None = None,
) -> list[SyncPlanEntry]:
    """Build synchronized ranges (0-based, inclusive) covering the same time for all videos.

    The reference is cut at ``reference_start``..``reference_end``. A camera
    listed in ``ranges`` keeps its Start as the same temporal event start; its
    End is forced to the reference frame span when fps matches the reference,
    otherwise the typed End is kept (different fps → different frame counts).
    Every other camera follows the reference in time from its sync (key) frame::

        start = sync + round((reference_start - reference_sync) / reference_fps * fps)
        end   = start + (reference_end - reference_start)   # same fps as reference
        end   = start + round((reference_end - reference_start) / reference_fps * fps)  # else

    Equal fps (within tolerance) always share the reference frame count so the
    cuts stay frame-locked; different fps stay time-aligned with different
    counts. ``removed_frames`` maps a video name to source frames (0-based)
    marked as repeated; those inside the range are kept on its entry.
    """
    if reference_start > reference_end:
        raise SyncWorkflowError("Reference start frame must not be after the end frame")
    info_by_name = {info.name: info for info in video_infos}
    ranges = {name: value for name, value in (ranges or {}).items() if name != reference_video}
    keyframe_videos = [info.name for info in video_infos if info.name not in ranges]
    if len(keyframe_videos) > 1:
        lower, upper = common_reference_bounds(
            video_infos, sync_frames, reference_video, keyframe_videos
        )
    else:
        lower, upper = 0, info_by_name[reference_video].frame_count - 1
    if reference_start < lower or reference_end > upper:
        raise SyncWorkflowError(
            f"Reference range is outside the common interval: choose frames {lower + 1}–{upper + 1}"
        )

    reference = info_by_name[reference_video]
    ref_span = equal_fps_frame_span(reference_start, reference_end)
    span_s = ref_span / reference.fps
    entries: list[SyncPlanEntry] = []
    for info in video_infos:
        sync_frame = sync_frames.get(info.name)
        if info.name == reference_video:
            start_frame, end_frame = int(reference_start), int(reference_end)
        elif info.name in ranges:
            start_frame, end_frame = (int(value) for value in ranges[info.name])
            if start_frame > end_frame:
                raise SyncWorkflowError(f"Start must not be after End for {info.name}")
            # Same temporal event start; if fps matches reference, lock frame count.
            if fps_rates_match(info.fps, reference.fps):
                end_frame = start_frame + ref_span
        else:
            reference_sync = int(sync_frames[reference_video])
            offset_s = (int(reference_start) - reference_sync) / reference.fps
            start_frame = int(sync_frames[info.name]) + round(offset_s * info.fps)
            if fps_rates_match(info.fps, reference.fps):
                end_frame = start_frame + ref_span
            else:
                end_frame = start_frame + round(span_s * info.fps)
        if not 0 <= start_frame <= end_frame < info.frame_count:
            raise SyncWorkflowError(
                f"Computed range {start_frame + 1}–{end_frame + 1} is outside "
                f"{info.name} (1–{info.frame_count})"
            )
        entries.append(
            SyncPlanEntry(
                video_file=validate_sync_leaf_filename(info.name, field="video_file"),
                output_file=_safe_output_name(
                    info.name, sync_frame=sync_frame, start_frame=start_frame, end_frame=end_frame
                ),
                start_frame=start_frame,
                end_frame=end_frame,
                sync_frame=None if sync_frame is None else int(sync_frame),
                removed_frames=tuple(
                    sorted(
                        frame
                        for frame in set((removed_frames or {}).get(info.name, ()))
                        if start_frame <= frame <= end_frame
                    )
                ),
            )
        )
    return entries


def describe_sync_plan(
    entries: Sequence[SyncPlanEntry], video_infos: Sequence[VideoInfo], reference_video: str
) -> list[str]:
    """One line per camera: range (1-based), frames, fps and duration.

    A camera whose duration differs from the reference by more than one of its
    frames (a hand-set Start/End, or a wrong fps) also shows the fps its range
    implies: (end - start) / reference duration.
    """
    info_by_name = {info.name: info for info in video_infos}
    reference_entry = next(entry for entry in entries if entry.video_file == reference_video)
    reference_s = (reference_entry.end_frame - reference_entry.start_frame) / info_by_name[
        reference_video
    ].fps
    lines = []
    for entry in entries:
        info = info_by_name[entry.video_file]
        duration_s = (entry.end_frame - entry.start_frame) / info.fps
        line = (
            f"{entry.video_file}: frames {entry.start_frame + 1}–{entry.end_frame + 1} "
            f"({entry.end_frame - entry.start_frame + 1}), {info.fps:.3f} fps, {duration_s:.3f} s"
        )
        if entry.video_file == reference_video:
            line += " [reference]"
        elif reference_s > 0 and abs(duration_s - reference_s) * info.fps > 1.0:
            implied = (entry.end_frame - entry.start_frame) / reference_s
            line += (
                f" | differs from reference {reference_s:.3f} s: Start/End imply {implied:.3f} fps"
            )
        lines.append(line)
    return lines


def _playback_speed_index(speed: float) -> int:
    return min(
        range(len(PLAYBACK_SPEED_STEPS)),
        key=lambda index: abs(PLAYBACK_SPEED_STEPS[index] - float(speed)),
    )


def step_playback_speed(speed: float, direction: int) -> float:
    """Move through the same discrete playback-speed ladder as Cut Video."""
    index = _playback_speed_index(speed)
    new_index = max(0, min(len(PLAYBACK_SPEED_STEPS) - 1, index + int(direction)))
    return PLAYBACK_SPEED_STEPS[new_index]


def format_playback_speed(speed: float) -> str:
    """Return a compact canonical playback-speed label."""
    canonical = PLAYBACK_SPEED_STEPS[_playback_speed_index(speed)]
    return str(int(canonical)) if canonical >= 1.0 and canonical.is_integer() else f"{canonical:g}"


def frame_after_navigation(current_frame: int, frame_count: int, delta: int) -> int:
    """Clamp one player navigation step to a zero-based video timeline."""
    if frame_count <= 0:
        return 0
    return max(0, min(int(current_frame) + int(delta), int(frame_count) - 1))


def resolve_reference_video(reference_text: str, video_infos: Sequence[VideoInfo]) -> str:
    """Resolve a typed row number, filename, stem, or unique text fragment."""
    query = reference_text.strip()
    if not query:
        raise SyncWorkflowError("Enter the reference video name or row number")

    if query.isdecimal():
        row = int(query)
        if 1 <= row <= len(video_infos):
            return video_infos[row - 1].name
        raise SyncWorkflowError(f"Reference video row must be between 1 and {len(video_infos)}")

    folded = query.casefold()
    exact_names = [info.name for info in video_infos if info.name.casefold() == folded]
    if len(exact_names) == 1:
        return exact_names[0]

    exact_stems = [info.name for info in video_infos if Path(info.name).stem.casefold() == folded]
    if len(exact_stems) == 1:
        return exact_stems[0]

    fragments = [info.name for info in video_infos if folded in info.name.casefold()]
    if len(fragments) == 1:
        return fragments[0]
    if len(fragments) > 1:
        raise SyncWorkflowError(
            f"Reference text {query!r} matches more than one video; type a unique fragment"
        )
    raise SyncWorkflowError(f"Reference video not found for typed text: {query!r}")


def parse_frame_field(text: str, *, field: str, frame_count: int) -> int:
    """Parse a user-entered 1-based frame into a validated zero-based index."""
    value_text = text.strip()
    if not value_text:
        raise SyncWorkflowError(f"Enter {field}")
    try:
        value = int(value_text)
    except ValueError as exc:
        raise SyncWorkflowError(f"{field} must be an integer frame number") from exc
    if not 1 <= value <= frame_count:
        raise SyncWorkflowError(f"{field} must be between 1 and {frame_count}")
    return value - 1


def build_sync_plan_from_fields(
    video_infos: Sequence[VideoInfo],
    sync_frame_texts: Mapping[str, str],
    reference_text: str,
    reference_start_text: str,
    reference_end_text: str,
    removed_frames: Mapping[str, Collection[int]] | None = None,
    range_texts: Mapping[str, tuple[str, str]] | None = None,
) -> tuple[str, list[SyncPlanEntry]]:
    """Validate the Pygame text fields and build the synchronization plan.

    ``range_texts`` holds each camera's optional (Start, End) texts. A camera
    with both is cut there and needs no sync frame; one with neither follows
    the reference from its sync frame (see ``build_sync_plan``).
    """
    reference_video = resolve_reference_video(reference_text, video_infos)
    info_by_name = {info.name: info for info in video_infos}
    ranges: dict[str, tuple[int, int]] = {}
    for info in video_infos:
        if info.name == reference_video:
            continue
        start_text, end_text = (
            text.strip() for text in (range_texts or {}).get(info.name, ("", ""))
        )
        if not start_text and not end_text:
            continue
        if not start_text or not end_text:
            raise SyncWorkflowError(f"Enter both Start and End for {info.name}, or neither")
        ranges[info.name] = (
            parse_frame_field(
                start_text, field=f"start frame for {info.name}", frame_count=info.frame_count
            ),
            parse_frame_field(
                end_text, field=f"end frame for {info.name}", frame_count=info.frame_count
            ),
        )
    needs_reference_sync = any(
        info.name not in ranges for info in video_infos if info.name != reference_video
    )
    sync_frames: dict[str, int] = {}
    for info in video_infos:
        text = sync_frame_texts.get(info.name, "")
        required = info.name not in ranges and (
            info.name != reference_video or needs_reference_sync
        )
        if required or text.strip():
            sync_frames[info.name] = parse_frame_field(
                text, field=f"sync frame for {info.name}", frame_count=info.frame_count
            )
    reference_info = info_by_name[reference_video]
    reference_start = parse_frame_field(
        reference_start_text,
        field="reference start frame",
        frame_count=reference_info.frame_count,
    )
    reference_end = parse_frame_field(
        reference_end_text,
        field="reference end frame",
        frame_count=reference_info.frame_count,
    )
    return reference_video, build_sync_plan(
        video_infos,
        sync_frames,
        reference_video,
        reference_start,
        reference_end,
        removed_frames,
        ranges,
    )


def _sync_file_text(entries: Sequence[SyncPlanEntry]) -> str:
    if not entries:
        raise SyncFileError("Cannot write an empty synchronization plan")
    stream = io.StringIO(newline="")
    stream.write(f"{SYNC_FORMAT_HEADER}\n")
    stream.write("# frame_base=1; frame ranges are inclusive\n")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    with_removed = any(entry.removed_frames for entry in entries)
    writer.writerow((*SYNC_COLUMNS, SYNC_REMOVED_COLUMN) if with_removed else SYNC_COLUMNS)
    seen: set[str] = set()
    for entry in entries:
        video_file = validate_sync_leaf_filename(entry.video_file, field="video_file")
        output_file = validate_sync_leaf_filename(entry.output_file, field="output_file")
        folded = video_file.casefold()
        if folded in seen:
            raise SyncFileError(f"Duplicate video in synchronization plan: {video_file}")
        seen.add(folded)
        if not 0 <= entry.start_frame <= entry.end_frame <= MAX_FRAME_NUMBER:
            raise SyncFileError(f"Invalid frame range for {video_file}")
        if entry.sync_frame is not None and not 0 <= entry.sync_frame <= MAX_FRAME_NUMBER:
            raise SyncFileError(f"Invalid sync frame for {video_file}")
        _validate_removed_frames(entry.removed_frames, entry, video_file)
        row: list[str | int] = [
            video_file,
            output_file,
            entry.start_frame + 1,
            entry.end_frame + 1,
            "" if entry.sync_frame is None else entry.sync_frame + 1,
        ]
        if with_removed:
            row.append(" ".join(str(frame + 1) for frame in entry.removed_frames))
        writer.writerow(row)
    return stream.getvalue()


def write_sync_file(entries: Sequence[SyncPlanEntry], output_file: str | Path) -> Path:
    """Atomically replace a sync file; never append stale synchronization rows."""
    output_path = Path(output_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = _sync_file_text(entries)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def _validate_removed_frames(removed: Sequence[int], entry: SyncPlanEntry, video_file: str) -> None:
    if list(removed) != sorted(set(removed)):
        raise SyncFileError(f"Removed frames for {video_file} must be unique and sorted")
    if any(not entry.start_frame <= frame <= entry.end_frame for frame in removed):
        raise SyncFileError(f"Removed frames for {video_file} must lie inside its frame range")
    if len(removed) > entry.end_frame - entry.start_frame:
        raise SyncFileError(f"Removed frames for {video_file} would leave no frame")


def _parse_sync_row(parts: Sequence[str], line_number: int) -> SyncPlanEntry:
    if len(parts) not in {4, 5, 6}:
        raise SyncFileError(f"Line {line_number}: expected 4 to 6 fields, received {len(parts)}")
    video_file = validate_sync_leaf_filename(parts[0], field=f"line {line_number} video_file")
    output_file = validate_sync_leaf_filename(parts[1], field=f"line {line_number} output_file")
    try:
        start_1based = int(parts[2])
        end_1based = int(parts[3])
        sync_1based = int(parts[4]) if len(parts) >= 5 and parts[4].strip() else None
        removed_1based = [int(value) for value in parts[5].split()] if len(parts) == 6 else []
    except ValueError as exc:
        raise SyncFileError(f"Line {line_number}: frame fields must be integers") from exc
    if not 1 <= start_1based <= end_1based <= MAX_FRAME_NUMBER:
        raise SyncFileError(f"Line {line_number}: expected 1 <= start_frame <= end_frame")
    if sync_1based is not None and not 1 <= sync_1based <= MAX_FRAME_NUMBER:
        raise SyncFileError(f"Line {line_number}: sync_frame must be positive")
    entry = SyncPlanEntry(
        video_file=video_file,
        output_file=output_file,
        start_frame=start_1based - 1,
        end_frame=end_1based - 1,
        sync_frame=None if sync_1based is None else sync_1based - 1,
        removed_frames=tuple(frame - 1 for frame in removed_1based),
    )
    try:
        _validate_removed_frames(entry.removed_frames, entry, video_file)
    except SyncFileError as exc:
        raise SyncFileError(f"Line {line_number}: {exc}") from exc
    return entry


def read_sync_file(sync_file: str | Path) -> list[SyncPlanEntry]:
    """Read v2 TSV or legacy whitespace sync files with strict validation."""
    path = Path(sync_file).expanduser().resolve()
    if not path.is_file():
        raise SyncFileError(f"Sync file not found: {path}")
    if path.suffix.lower() == ".json":
        try:
            manifest = read_event_manifest(path)
        except EventSyncError as exc:
            raise SyncFileError(str(exc)) from exc
        return [
            SyncPlanEntry(
                name,
                f"{name}_events.mp4",
                camera["events"][0] - 1,
                camera["events"][-1] - 1,
                removed_frames=tuple(
                    f - 1
                    for f in sorted(camera["removed_frames"])
                    if camera["events"][0] <= f <= camera["events"][-1]
                ),
            )
            for name, camera in manifest["cameras"].items()
        ]
    if path.stat().st_size > MAX_SYNC_FILE_BYTES:
        raise SyncFileError(
            f"Sync file is larger than {MAX_SYNC_FILE_BYTES} bytes and was rejected: {path}"
        )
    try:
        lines = path.read_text(encoding="utf-8-sig").splitlines()
    except UnicodeError as exc:
        raise SyncFileError(f"Sync file is not valid UTF-8: {path}") from exc

    entries: list[SyncPlanEntry] = []
    seen: set[str] = set()
    for line_number, raw_line in enumerate(lines, start=1):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        if stripped.casefold().startswith("video_file\t"):
            continue
        try:
            if "\t" in raw_line:
                parts = next(csv.reader([raw_line], delimiter="\t", strict=True))
            else:
                parts = shlex.split(raw_line, comments=False, posix=True)
        except (csv.Error, ValueError) as exc:
            raise SyncFileError(f"Line {line_number}: invalid quoting") from exc
        entry = _parse_sync_row(parts, line_number)
        folded = entry.video_file.casefold()
        if folded in seen:
            raise SyncFileError(f"Line {line_number}: duplicate video entry {entry.video_file!r}")
        seen.add(folded)
        entries.append(entry)
        if len(entries) > MAX_SYNC_ENTRIES:
            raise SyncFileError(f"Sync file exceeds {MAX_SYNC_ENTRIES} entries")
    if not entries:
        raise SyncFileError(f"No synchronization entries found in: {path}")
    return entries


def build_checkpoint_state(
    *,
    reference: str,
    start: str,
    end: str,
    sync_frames: Mapping[str, str],
    current_video: str,
    current_frame: int,
    output_file: str | Path | None,
    completed: bool,
    removed_frames: Mapping[str, Collection[int]] | None = None,
    ranges: Mapping[str, tuple[str, str]] | None = None,
    fps_overrides: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the JSON-serializable "Load / Resume" checkpoint for the current UI state.

    ``removed_frames`` (0-based) is stored 1-based, like every frame the user sees.
    ``ranges`` keeps each camera's optional Start/End texts (only non-empty ones).
    ``fps_overrides`` stores per-camera real FPS used for time-based sync math.
    """
    return {
        "ranges": {
            name: [start_text, end_text]
            for name, (start_text, end_text) in (ranges or {}).items()
            if start_text.strip() or end_text.strip()
        },
        "removed_frames": {
            name: sorted(int(frame) + 1 for frame in frames)
            for name, frames in (removed_frames or {}).items()
            if frames
        },
        "fps_overrides": {
            name: float(rate) for name, rate in (fps_overrides or {}).items() if float(rate) > 0.0
        },
        "vaila_checkpoint": CHECKPOINT_SCHEMA_VERSION,
        "reference": reference,
        "start": start,
        "end": end,
        "sync_frames": dict(sync_frames),
        "current_video": current_video,
        "current_frame": int(current_frame),
        "output_file": str(output_file) if output_file else None,
        "completed": bool(completed),
    }


def write_checkpoint_file(state: Mapping[str, Any], checkpoint_file: str | Path) -> Path:
    """Atomically write a "Load / Resume" checkpoint next to the sync file."""
    output_path = Path(checkpoint_file).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(state, indent=2, sort_keys=True, ensure_ascii=False)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            prefix=f".{output_path.name}.",
            suffix=".tmp",
            dir=output_path.parent,
            delete=False,
        ) as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
            temporary_path = Path(handle.name)
        os.replace(temporary_path, output_path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()
    return output_path


def read_checkpoint_file(checkpoint_file: str | Path) -> dict[str, Any]:
    """Read and strictly validate a "Load / Resume" checkpoint JSON file.

    Every validation failure -- missing keys, wrong types, corrupt JSON, an
    oversized file -- raises SyncFileError with a message the caller can show
    directly in the status bar; nothing here ever raises an opaque exception
    or crashes the player.
    """
    path = Path(checkpoint_file).expanduser().resolve()
    if not path.is_file():
        raise SyncFileError(f"Checkpoint file not found: {path}")
    if path.stat().st_size > MAX_CHECKPOINT_FILE_BYTES:
        raise SyncFileError(
            f"Checkpoint file is larger than {MAX_CHECKPOINT_FILE_BYTES} bytes: {path}"
        )
    try:
        text = path.read_text(encoding="utf-8-sig")
    except UnicodeError as exc:
        raise SyncFileError(f"Checkpoint file is not valid UTF-8: {path}") from exc
    try:
        data = json.loads(text)
    except json.JSONDecodeError as exc:
        raise SyncFileError(f"Checkpoint file is not valid JSON: {path} ({exc})") from exc
    if not isinstance(data, dict):
        raise SyncFileError(f"Checkpoint file must contain a JSON object: {path}")

    required = (
        "vaila_checkpoint",
        "reference",
        "start",
        "end",
        "sync_frames",
        "current_video",
        "current_frame",
    )
    missing = [key for key in required if key not in data]
    if missing:
        raise SyncFileError(f"Checkpoint file is missing key(s) {missing}: {path}")
    if not all(
        isinstance(data[key], str) for key in ("reference", "start", "end", "current_video")
    ):
        raise SyncFileError(
            f"Checkpoint 'reference'/'start'/'end'/'current_video' must be text: {path}"
        )
    current_frame = data["current_frame"]
    if isinstance(current_frame, bool) or not isinstance(current_frame, int) or current_frame < 0:
        raise SyncFileError(f"Checkpoint 'current_frame' must be a non-negative integer: {path}")
    sync_frames = data["sync_frames"]
    if not isinstance(sync_frames, dict) or not all(
        isinstance(name, str) and isinstance(value, str) for name, value in sync_frames.items()
    ):
        raise SyncFileError(f"Checkpoint 'sync_frames' must map video names to text: {path}")
    output_file = data.get("output_file")
    if output_file is not None and not isinstance(output_file, str):
        raise SyncFileError(f"Checkpoint 'output_file' must be text or null: {path}")
    removed = data.get("removed_frames", {})
    if not isinstance(removed, dict) or not all(
        isinstance(name, str)
        and isinstance(frames, list)
        and all(
            isinstance(frame, int) and not isinstance(frame, bool) and frame >= 1
            for frame in frames
        )
        for name, frames in removed.items()
    ):
        raise SyncFileError(
            f"Checkpoint 'removed_frames' must map video names to 1-based frame lists: {path}"
        )
    ranges = data.get("ranges", {})
    if not isinstance(ranges, dict) or not all(
        isinstance(name, str)
        and isinstance(texts, list)
        and len(texts) == 2
        and all(isinstance(text, str) for text in texts)
        for name, texts in ranges.items()
    ):
        raise SyncFileError(
            f"Checkpoint 'ranges' must map video names to [start, end] texts: {path}"
        )
    fps_overrides = data.get("fps_overrides", {})
    if not isinstance(fps_overrides, dict) or not all(
        isinstance(name, str)
        and isinstance(rate, (int, float))
        and not isinstance(rate, bool)
        and 0.0 < float(rate) < 10_000.0
        for name, rate in fps_overrides.items()
    ):
        raise SyncFileError(
            f"Checkpoint 'fps_overrides' must map video names to positive FPS rates: {path}"
        )
    data["fps_overrides"] = {name: float(rate) for name, rate in fps_overrides.items()}
    if data.get("sync_mode", "fps") not in ("fps", "events"):
        raise SyncFileError("Checkpoint sync_mode must be fps or events")
    event_texts = data.get("event_texts", {})
    if not isinstance(event_texts, dict) or not all(
        isinstance(k, str) and isinstance(v, str) for k, v in event_texts.items()
    ):
        raise SyncFileError("Checkpoint event_texts must map camera names to text")
    if not all(isinstance(data.get(k, ""), str) for k in ("reference_fps", "output_fps")):
        raise SyncFileError("Checkpoint reference_fps and output_fps must be rational text")
    return data


def find_sync_entry(
    entries: Sequence[SyncPlanEntry], video_path: str | Path
) -> SyncPlanEntry | None:
    """Match a video exactly by filename, or uniquely by stem for legacy files."""
    video = Path(video_path)
    name_matches = [
        entry for entry in entries if entry.video_file.casefold() == video.name.casefold()
    ]
    if len(name_matches) == 1:
        return name_matches[0]
    stem_matches = [
        entry
        for entry in entries
        if Path(entry.video_file).stem.casefold() == video.stem.casefold()
    ]
    return stem_matches[0] if len(stem_matches) == 1 else None


def sync_entries_to_cutvideo_data(
    entries: Sequence[SyncPlanEntry],
) -> dict[str, dict[str, Any]]:
    """Convert validated entries to cutvideo's batch mapping."""
    return {
        entry.video_file: {
            "new_name": entry.output_file,
            "initial_frame": entry.start_frame,
            "final_frame": entry.end_frame,
            "sync_frame": entry.sync_frame,
            "removed_frames": entry.removed_frames,
        }
        for entry in entries
    }


def build_cutvideo_command(
    reference_video: str | Path,
    sync_file: str | Path,
    *,
    python_executable: str | Path | None = None,
) -> list[str]:
    """Build a shell-free direct handoff command for Cut Video."""
    video = Path(reference_video).expanduser().resolve()
    sync = Path(sync_file).expanduser().resolve()
    if not video.is_file():
        raise SyncWorkflowError(f"Reference video not found: {video}")
    if not sync.is_file():
        raise SyncWorkflowError(f"Sync file not found: {sync}")
    return [
        str(python_executable or sys.executable),
        "-m",
        "vaila.cutvideo",
        "--video",
        str(video),
        "--sync-file",
        str(sync),
    ]


def launch_cutvideo(reference_video: str | Path, sync_file: str | Path) -> subprocess.Popen[Any]:
    """Launch Cut Video immediately with the generated synchronization selected."""
    command = build_cutvideo_command(reference_video, sync_file)
    print(f">> vaila/syncvid: Cut Video handoff\n{shlex.join(command)}")
    kwargs: dict[str, Any] = {
        "cwd": str(Path(__file__).resolve().parents[1]),
        "shell": False,
    }
    if os.name != "nt":
        kwargs["start_new_session"] = True
    return subprocess.Popen(command, **kwargs)


# Repeated-frame detection. Editors that splice several cameras into one
# timeline repeat a source frame whenever the camera (or replay) rate is lower
# than the timeline rate. Re-encoding makes the copy differ slightly from the
# original (H.264/H.265 noise along edges), so an exact pixel test misses it.
# Each frame is scored like a viewer would: the largest local change, i.e. the
# maximum over a 32 x 18 grid of the mean absolute difference (grey levels)
# between the blurred frame and the previous one. Repeats score at codec-noise
# level; real frames score where something moved. The cut is chosen per video
# at a gap of at least REPEAT_MIN_GAP_RATIO between sorted scores (see
# repeat_threshold). On the real 1080p examples copies scored <= 16.5 against
# motion >= 40.2, and <= 8.4 against >= 30.6; videos without repeats show a
# continuous range (largest gap 1.09x). Scores at or below
# REPEAT_INVISIBLE_SCORE are invisible and always count as repeats.
#
# Interlaced sources (e.g. 50i material on a 59.94 timeline) repeat a frame
# built from the other field: the same image shifted about +-1.2 px
# vertically. Its raw score (~20 on gol.mp4) falls among slow real frames, so
# a small, purely vertical global shift (phase correlation on the centre of
# the frame) is compensated before scoring again; the lower score is kept.
# Horizontal or larger shifts are real camera motion and are not compensated.
REPEAT_GRID = (32, 18)
REPEAT_INVISIBLE_SCORE = 1.0
REPEAT_MIN_GAP_RATIO = 2.0
REPEAT_FIELD_SHIFT_MAX_X = 0.5
REPEAT_FIELD_SHIFT_MAX_Y = 2.0
REPEAT_SHIFT_MARGIN = 8
REPEAT_ALTERNATION_MAX_X = 0.1
REPEAT_ALTERNATION_MIN_Y = 0.75
REPEAT_ALTERNATION_MAX_Y = 1.75
REPEAT_ALTERNATION_MIN_RESPONSE = 0.9
REPEAT_ALTERNATION_MIN_LENGTH = 3

ProgressCallback = Callable[[str, int, int], None]


@dataclass(frozen=True)
class RepeatedFrameReport:
    """Repeated-frame analysis of one video (0-based frame indices).

    ``motion_scores[i]`` scores frame ``i + 1`` against frame ``i``; a frame is
    classified by ``automatic_reasons[i]``. Reports created without explicit
    decisions retain the original score/threshold behavior. ``shifts[i]`` and
    ``phase_responses[i]`` describe phase correlation for the same pair.
    """

    video: Path
    fps: float
    width: int
    height: int
    motion_scores: tuple[float, ...]
    threshold: float
    gap_ratio: float
    shifts: tuple[tuple[float, float], ...] = ()
    phase_responses: tuple[float, ...] = ()
    automatic_reasons: tuple[str, ...] | None = None

    @property
    def repeat_reasons(self) -> tuple[str, ...]:
        if self.automatic_reasons is not None:
            return self.automatic_reasons
        return tuple(
            "motion_threshold" if score <= self.threshold else "none"
            for score in self.motion_scores
        )

    @property
    def total_frames(self) -> int:
        return len(self.motion_scores) + 1

    @property
    def info(self) -> VideoInfo:
        """The analyzed video, with the decoded frame count."""
        return VideoInfo(
            path=self.video,
            frame_count=self.total_frames,
            fps=self.fps,
            width=self.width,
            height=self.height,
        )

    @property
    def repeated_frames(self) -> tuple[int, ...]:
        return tuple(
            index + 1 for index, reason in enumerate(self.repeat_reasons) if reason != "none"
        )

    @property
    def kept_frames(self) -> tuple[int, ...]:
        repeated = set(self.repeated_frames)
        return tuple(index for index in range(self.total_frames) if index not in repeated)

    @property
    def estimated_fps(self) -> float:
        """Rate of distinct frames over the same duration (fps x unique / total)."""
        return self.fps * len(self.kept_frames) / self.total_frames


def frame_motion_score(previous_blurred: Any, blurred: Any) -> float:
    """Largest local mean grey-level change between two blurred grey frames."""
    diff = cv2.absdiff(previous_blurred, blurred).astype(np.float32)
    blocks = cv2.resize(diff, REPEAT_GRID, interpolation=cv2.INTER_AREA)
    return float(blocks.max())


def _centre_crop(image: Any) -> Any:
    height, width = image.shape[:2]
    return image[height // 4 : height // 4 + height // 2, width // 4 : width // 4 + width // 2]


def field_shift_motion_score(
    previous_blurred: Any, blurred: Any
) -> tuple[float, tuple[float, float], float]:
    """Motion score with field-shift compensation, measured shift and response.

    The global shift comes from phase correlation on the centre of the frames.
    When it is small and purely vertical (an interlaced-field copy), the
    previous frame is shifted back and scored again away from the borders;
    the lower of the two scores is returned.
    """
    score = frame_motion_score(previous_blurred, blurred)
    (dx, dy), response = cv2.phaseCorrelate(
        _centre_crop(previous_blurred).astype(np.float32),
        _centre_crop(blurred).astype(np.float32),
    )
    if abs(dx) <= REPEAT_FIELD_SHIFT_MAX_X and abs(dy) <= REPEAT_FIELD_SHIFT_MAX_Y:
        height, width = blurred.shape[:2]
        shifted = cv2.warpAffine(
            previous_blurred,
            np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]], dtype=np.float32),
            (width, height),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )
        m = REPEAT_SHIFT_MARGIN
        score = min(score, frame_motion_score(shifted[m:-m, m:-m], blurred[m:-m, m:-m]))
    return score, (float(dx), float(dy)), float(response)


def field_alternation_frames(
    shifts: Sequence[tuple[float, float]], responses: Sequence[float]
) -> set[int]:
    """Confirm runs of >=3 alternating field candidates (0-based frame indices).

    Runs follow the candidate sequence, allowing arbitrary intervals of real
    frames between candidates. A same-sign candidate ends a run and starts the
    next one; isolated candidates and pairs are never confirmed.
    """
    confirmed: set[int] = set()
    run: list[int] = []
    previous_dy = 0.0
    for index, ((dx, dy), response) in enumerate(zip(shifts, responses, strict=True), start=1):
        if not (
            abs(dx) <= REPEAT_ALTERNATION_MAX_X
            and REPEAT_ALTERNATION_MIN_Y <= abs(dy) <= REPEAT_ALTERNATION_MAX_Y
            and response >= REPEAT_ALTERNATION_MIN_RESPONSE
        ):
            continue
        if dy * previous_dy >= 0:
            if len(run) >= REPEAT_ALTERNATION_MIN_LENGTH:
                confirmed.update(run)
            run = []
        run.append(index)
        previous_dy = dy
    if len(run) >= REPEAT_ALTERNATION_MIN_LENGTH:
        confirmed.update(run)
    return confirmed


def repeat_threshold(scores: Sequence[float]) -> tuple[float, float]:
    """Return (threshold, gap ratio) separating codec noise from real motion.

    Candidate cuts are the gaps between sorted visible scores that reach
    REPEAT_MIN_GAP_RATIO. Of those, the cut that best separates the two groups
    (largest between-class variance of log scores, as in Otsu's method) wins,
    so a lone gap inside the noise group does not beat the noise/motion gap.
    Without a candidate, only invisible changes count as repeats. The returned
    ratio is that of the chosen gap, or the widest gap when none qualifies.
    """
    raw = np.sort([score for score in scores if score > REPEAT_INVISIBLE_SCORE])
    visible = np.log(raw)
    if visible.size < 2:
        return REPEAT_INVISIBLE_SCORE, 1.0
    gaps = np.diff(visible)
    total = visible.size
    best_cut, best_spread = -1, -1.0
    for cut in np.flatnonzero(gaps >= np.log(REPEAT_MIN_GAP_RATIO)):
        low, high = visible[: cut + 1], visible[cut + 1 :]
        spread = low.size * high.size / total**2 * (high.mean() - low.mean()) ** 2
        if spread > best_spread:
            best_cut, best_spread = int(cut), float(spread)
    if best_cut < 0:
        return REPEAT_INVISIBLE_SCORE, float(np.exp(gaps.max()))
    return float(raw[best_cut]), float(raw[best_cut + 1] / raw[best_cut])


def find_repeated_frames(
    video_path: str | Path, progress: ProgressCallback | None = None
) -> RepeatedFrameReport:
    """Combine motion thresholds and alternating field evidence for every frame."""
    info = probe_video(video_path)
    cap = cv2.VideoCapture(str(info.path))
    scores: list[float] = []
    shifts: list[tuple[float, float]] = []
    responses: list[float] = []
    previous = None
    index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            blurred = cv2.GaussianBlur(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY), (5, 5), 0)
            if previous is not None:
                score, shift, response = field_shift_motion_score(previous, blurred)
                scores.append(score)
                shifts.append(shift)
                responses.append(response)
            previous = blurred
            index += 1
            if progress is not None and index % 30 == 0:
                progress("analyze", index, info.frame_count)
    finally:
        cap.release()
    if index == 0:
        raise SyncWorkflowError(f"OpenCV could not decode any frame: {info.path}")
    threshold, gap_ratio = repeat_threshold(scores)
    alternation = field_alternation_frames(shifts, responses)
    reasons = tuple(
        "motion_threshold"
        if score <= threshold
        else "field_alternation"
        if index in alternation
        else "none"
        for index, score in enumerate(scores, start=1)
    )
    return RepeatedFrameReport(
        video=info.path,
        fps=info.fps,
        width=info.width,
        height=info.height,
        motion_scores=tuple(scores),
        threshold=threshold,
        gap_ratio=gap_ratio,
        shifts=tuple(shifts),
        phase_responses=tuple(responses),
        automatic_reasons=reasons,
    )


def _encode_without_frames(
    info: VideoInfo,
    removed: frozenset[int],
    output_fps: float,
    output_video: Path,
    encoder: str,
    progress: ProgressCallback | None,
) -> list[int]:
    """Pipe every decoded frame not in ``removed`` as raw BGR into one H.264 encode.

    Returns the source indices written, in order.
    """
    try:
        from .ffmpeg_utils import get_ffmpeg_video_encoding_args, get_video_encode_ffmpeg_path
    except ImportError:
        from ffmpeg_utils import (  # ty: ignore[unresolved-import]
            get_ffmpeg_video_encoding_args,
            get_video_encode_ffmpeg_path,
        )

    command = [
        get_video_encode_ffmpeg_path(encoder),
        "-hide_banner",
        "-loglevel",
        "error",
        "-y",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{info.width}x{info.height}",
        "-framerate",
        f"{output_fps:.6f}",
        "-i",
        "-",
    ]
    if info.width % 2 or info.height % 2:
        command += ["-vf", "crop=trunc(iw/2)*2:trunc(ih/2)*2"]
    command += [*get_ffmpeg_video_encoding_args(encoder), "-movflags", "+faststart"]
    command += [
        "-metadata",
        f"comment=vaila syncvid norepeat: source={info.name} source_fps={info.fps:.6f} "
        f"unique_frames={info.frame_count - len(removed)}/{info.frame_count} "
        f"estimated_real_fps={output_fps:.6f}",
        str(output_video),
    ]

    written: list[int] = []
    expected = info.frame_count - len(removed)
    cap = cv2.VideoCapture(str(info.path))
    process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=subprocess.PIPE)
    stdin = cast(io.BufferedWriter, process.stdin)
    try:
        index = 0
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                break
            if index not in removed:
                stdin.write(frame.tobytes())
                written.append(index)
                if progress is not None and len(written) % 30 == 0:
                    progress("encode", len(written), expected)
            index += 1
    except BrokenPipeError:
        pass
    finally:
        cap.release()
        with contextlib.suppress(BrokenPipeError):
            stdin.close()
        stderr = process.stderr.read().decode(errors="replace") if process.stderr else ""
        process.wait()
    if process.returncode != 0 or not written:
        raise RuntimeError(
            f"FFmpeg {encoder} encode failed (exit {process.returncode}, "
            f"{len(written)}/{expected} frames): {stderr.strip()[-400:]}"
        )
    return written


def write_video_without_frames(
    info: VideoInfo,
    removed: Collection[int],
    output_dir: str | Path,
    progress: ProgressCallback | None = None,
    *,
    frame_offset: int = 0,
) -> Path:
    """Write ``<stem>_norepeat_<fps>fps.mp4`` without ``removed`` (0-based) frames.

    The output keeps the original duration: its container frame rate is the
    estimated real rate (fps x kept / total), also written in the file name
    and the ``comment`` metadata. Audio is not copied. ``<stem>_norepeat_frames.csv``
    maps every new frame to its source frame (both 1-based) and source time;
    ``frame_offset`` is added when ``info`` is a cut that starts at that
    source frame. Returns the video path.
    """
    try:
        from .ffmpeg_utils import encoders_with_cpu_fallback, is_hardware_video_encoder
    except ImportError:
        from ffmpeg_utils import (  # ty: ignore[unresolved-import]
            encoders_with_cpu_fallback,
            is_hardware_video_encoder,
        )

    dropped = frozenset(int(frame) for frame in removed if 0 <= int(frame) < info.frame_count)
    if len(dropped) >= info.frame_count:
        raise SyncWorkflowError(f"Removing {len(dropped)} frames would leave {info.name} empty")
    output_fps = info.fps * (info.frame_count - len(dropped)) / info.frame_count
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = info.path.stem
    output_video = out_dir / f"{stem}_norepeat_{output_fps:.2f}fps.mp4"
    written: list[int] = []
    for encoder in encoders_with_cpu_fallback():
        try:
            written = _encode_without_frames(
                info, dropped, output_fps, output_video, encoder, progress
            )
            break
        except RuntimeError as exc:
            if not is_hardware_video_encoder(encoder):
                raise
            print(f"[yellow]{exc}; retrying with CPU encoder[/]")

    with open(out_dir / f"{stem}_norepeat_frames.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["new_frame", "original_frame", "original_time_s"])
        for new_index, original in enumerate(written):
            source = original + frame_offset
            writer.writerow([new_index + 1, source + 1, f"{source / info.fps:.6f}"])
    return output_video


def save_videos_without_frames(
    items: Sequence[tuple[VideoInfo, Collection[int], RepeatedFrameReport | None]],
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> Path:
    """Write repeat-free copies of the videos that have frames to remove.

    ``items`` holds (video, frames to remove 0-based, automatic analysis or
    None). Results go to ``processed_norepeat_<timestamp>/`` next to the first
    video (or ``output_dir``): one copy per video with removed frames,
    ``<stem>_motion_scores.csv`` for each analyzed video, and a line of
    ``norepeat_summary.csv`` for every video.
    """
    if not items:
        raise SyncWorkflowError("No videos were given for repeated-frame removal")
    if output_dir is None:
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = items[0][0].path.parent / f"processed_norepeat_{timestamp}"
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows: list[list[Any]] = []
    for info, removed_frames, report in items:
        removed = sorted({int(frame) for frame in removed_frames if 0 <= frame < info.frame_count})
        removed_set = set(removed)
        automatic = set(report.repeated_frames) if report is not None else set()
        motion_count = report.repeat_reasons.count("motion_threshold") if report else 0
        field_count = report.repeat_reasons.count("field_alternation") if report else 0
        added = len(removed_set - automatic)
        kept_by_review = len(automatic - removed_set)
        if report is not None:
            scores_csv = out_dir / f"{info.path.stem}_motion_scores.csv"
            with open(scores_csv, "w", newline="", encoding="utf-8") as f:
                writer = csv.writer(f)
                writer.writerow(
                    [
                        "frame",
                        "motion_score",
                        "repeated",
                        "shift_x_px",
                        "shift_y_px",
                        "phase_response",
                        "auto_repeated",
                        "auto_reason",
                    ]
                )
                writer.writerow([1, "", int(0 in removed_set), "", "", "", 0, "none"])
                shifts = report.shifts or ((0.0, 0.0),) * len(report.motion_scores)
                responses = report.phase_responses or (None,) * len(report.motion_scores)
                for index, (score, (dx, dy), response, reason) in enumerate(
                    zip(
                        report.motion_scores, shifts, responses, report.repeat_reasons, strict=True
                    ),
                    start=1,
                ):
                    writer.writerow(
                        [
                            index + 1,
                            f"{score:.3f}",
                            int(index in removed_set),
                            f"{dx:.2f}",
                            f"{dy:.2f}",
                            "" if response is None else f"{response:.6f}",
                            int(index in automatic),
                            reason,
                        ]
                    )
        output_name = ""
        if removed:
            output_name = write_video_without_frames(info, removed, out_dir, progress).name
        kept = info.frame_count - len(removed)
        estimated_fps = info.fps * kept / info.frame_count
        rows.append(
            [
                info.name,
                info.frame_count,
                kept,
                len(removed),
                f"{info.fps:.6f}",
                f"{estimated_fps:.6f}",
                "" if report is None else f"{report.threshold:.3f}",
                "" if report is None else f"{report.gap_ratio:.2f}",
                "" if report is None else len(report.repeated_frames),
                " ".join(str(index + 1) for index in removed),
                output_name,
                "" if report is None else motion_count,
                "" if report is None else field_count,
                added,
                kept_by_review,
            ]
        )
        print(
            f"  {info.name}: {info.frame_count} frames, {len(removed)} marked for removal, "
            f"{kept} kept, real fps ~{estimated_fps:.3f}; "
            + (
                f"auto={len(automatic)} (motion_threshold={motion_count}, "
                f"field_alternation={field_count}), "
                if report
                else "auto=not analyzed, "
            )
            + f"review added={added}, review kept={kept_by_review}"
            + (f" -> {output_name}" if output_name else " (no repeats, not rewritten)")
        )

    with open(out_dir / "norepeat_summary.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "video",
                "frames",
                "unique_frames",
                "repeated_frames",
                "fps",
                "estimated_real_fps",
                "repeat_threshold",
                "noise_motion_gap_ratio",
                "auto_detected_repeats",
                "repeated_frame_numbers",
                "output_video",
                "motion_threshold_repeats",
                "field_alternation_repeats",
                "manual_added_repeats",
                "manual_kept_frames",
            ]
        )
        writer.writerows(rows)
    return out_dir


def remove_repeated_frames(
    videos: Sequence[str | Path],
    output_dir: str | Path | None = None,
    progress: ProgressCallback | None = None,
) -> tuple[Path, list[RepeatedFrameReport]]:
    """Detect repeated frames in every video and save repeat-free copies (CLI path)."""
    if not videos:
        raise SyncWorkflowError("No videos were given for repeated-frame removal")
    reports = [find_repeated_frames(video, progress) for video in videos]
    out_dir = save_videos_without_frames(
        [(report.info, report.repeated_frames, report) for report in reports],
        output_dir,
        progress,
    )
    return out_dir, reports


class PygameTextField:
    """Small editable text field rendered directly in the Pygame window."""

    def __init__(self, key: str, text: str = "", *, max_length: int = 260) -> None:
        self.key = key
        self.text = text
        self.max_length = max_length
        self.rect = pygame.Rect(0, 0, 0, 0)
        self.cursor = len(text)
        self.active = False
        self.select_all = False
        # Grey hint shown while the field is empty and inactive.
        self.placeholder = ""

    def set_rect(self, rect: pygame.Rect) -> None:
        self.rect = rect

    def activate(self) -> None:
        self.active = True
        self.cursor = len(self.text)
        self.select_all = True

    def deactivate(self) -> None:
        self.active = False
        self.select_all = False

    def handle_key(self, event: pygame.event.Event) -> bool:
        """Edit the field and return True when Enter commits it."""
        if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            return True
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        if ctrl and event.key == pygame.K_a:
            self.cursor = len(self.text)
            self.select_all = True
            return False
        if event.key == pygame.K_HOME:
            self.cursor = 0
            self.select_all = False
            return False
        if event.key == pygame.K_END:
            self.cursor = len(self.text)
            self.select_all = False
            return False
        if event.key == pygame.K_LEFT:
            self.cursor = max(0, self.cursor - 1)
            self.select_all = False
            return False
        if event.key == pygame.K_RIGHT:
            self.cursor = min(len(self.text), self.cursor + 1)
            self.select_all = False
            return False
        if event.key in (pygame.K_BACKSPACE, pygame.K_DELETE):
            if self.select_all:
                self.text = ""
                self.cursor = 0
                self.select_all = False
            elif event.key == pygame.K_BACKSPACE and self.cursor > 0:
                self.text = self.text[: self.cursor - 1] + self.text[self.cursor :]
                self.cursor -= 1
            elif event.key == pygame.K_DELETE and self.cursor < len(self.text):
                self.text = self.text[: self.cursor] + self.text[self.cursor + 1 :]
            return False

        typed = getattr(event, "unicode", "")
        if typed and typed.isprintable() and not ctrl:
            if self.select_all:
                self.text = ""
                self.cursor = 0
                self.select_all = False
            if len(self.text) < self.max_length:
                self.text = self.text[: self.cursor] + typed + self.text[self.cursor :]
                self.cursor += len(typed)
        return False

    def draw(self, surface: pygame.Surface, font: pygame.font.Font) -> None:
        background = (246, 249, 252) if self.active else (222, 228, 235)
        border = (63, 158, 255) if self.active else (112, 122, 134)
        pygame.draw.rect(surface, background, self.rect, border_radius=4)
        pygame.draw.rect(surface, border, self.rect, 2, border_radius=4)

        inner = self.rect.inflate(-10, -6)
        old_clip = surface.get_clip()
        surface.set_clip(inner)
        if not self.text and not self.active and self.placeholder:
            rendered = font.render(self.placeholder, True, (128, 138, 150))
        else:
            rendered = font.render(self.text, True, (20, 25, 31))
        text_x = inner.x
        if rendered.get_width() > inner.width:
            text_x = inner.right - rendered.get_width()
        surface.blit(
            rendered, (text_x, inner.y + max(0, (inner.height - rendered.get_height()) // 2))
        )

        if self.active and (pygame.time.get_ticks() // 500) % 2 == 0:
            before = font.render(self.text[: self.cursor], True, (20, 25, 31))
            cursor_x = max(inner.x, min(inner.right, text_x + before.get_width()))
            pygame.draw.line(
                surface,
                (20, 25, 31),
                (cursor_x, inner.y + 2),
                (cursor_x, inner.bottom - 2),
                1,
            )
        surface.set_clip(old_clip)


class PygameSyncPlayer:
    """Fast multi-camera synchronization player using Cut Video-style controls."""

    PANEL_MIN_WIDTH = 370
    PANEL_MAX_WIDTH = 470
    CONTROL_HEIGHT = 180
    ROW_HEIGHT = 40
    # Reserved height (px) at the bottom of the side panel for the status
    # line, the action-button rows (incl. real-FPS controls), and the footer.
    ACTIONS_AREA_HEIGHT = 268

    def __init__(
        self,
        video_infos: Sequence[VideoInfo],
        *,
        output_file: str | Path | None = None,
        dialog_parent: tk.Tk | tk.Toplevel | None = None,
    ) -> None:
        if not video_infos:
            raise SyncWorkflowError("No videos were provided to the synchronization player")
        self.video_infos = list(video_infos)
        self.output_file = Path(output_file).expanduser().resolve() if output_file else None
        self.dialog_parent = dialog_parent
        self.result: SyncRunResult | None = None

        self.reference_field = PygameTextField("reference", self.video_infos[0].name)
        self.start_field = PygameTextField("reference_start", "1", max_length=16)
        self.end_field = PygameTextField(
            "reference_end", str(self.video_infos[0].frame_count), max_length=16
        )
        self.sync_fields = {
            info.name: PygameTextField(f"sync:{info.name}", "", max_length=16)
            for info in self.video_infos
        }
        # Optional per-camera cut range; empty = follow the reference from the
        # sync frame at the camera's own fps (shown as a grey preview).
        self.range_start_fields = {
            info.name: PygameTextField(f"start:{info.name}", "", max_length=16)
            for info in self.video_infos
        }
        self.range_end_fields = {
            info.name: PygameTextField(f"end:{info.name}", "", max_length=16)
            for info in self.video_infos
        }
        self.goto_field = PygameTextField("goto", "", max_length=16)
        # Typed real FPS applied to the current camera or every camera; empty
        # means "use the container fps" until Set FPS is clicked.
        self.real_fps_field = PygameTextField("real_fps", "", max_length=12)
        self.fps_overrides: dict[str, float] = {}
        self.event_mode = False
        self.event_fields = {
            info.name: PygameTextField(f"events:{info.name}", "", max_length=260)
            for info in self.video_infos
        }
        self.reference_clock_field = PygameTextField("reference_clock", "60000/1001")
        self.output_rate_field = PygameTextField("output_rate", "30000/1001")
        self.loaded_checkpoint: Path | None = None
        self.active_field: PygameTextField | None = None

        self.current_index = 0
        self.current_frame = 0
        self.positions = {info.name: 0 for info in self.video_infos}
        self.capture: cv2.VideoCapture | None = None
        self.frame: Any = None
        self.playing = False
        self.playback_speed = 1.0
        self.slow_motion_accumulator = 0.0
        self.running = True
        self.dragging_timeline = False
        self.list_scroll = 0
        self.status_text = (
            "Type each camera sync frame, then the reference video and inclusive start/end."
        )
        self.status_color = (190, 215, 235)

        pygame.display.init()
        pygame.font.init()
        display = pygame.display.Info()
        available_width = display.current_w - 100 if display.current_w else 1280
        available_height = display.current_h - 100 if display.current_h else 820
        self.window_width = max(900, min(1360, available_width))
        self.window_height = max(650, min(860, available_height))
        self.screen = pygame.display.set_mode(
            (self.window_width, self.window_height), pygame.RESIZABLE
        )
        self.clock = pygame.time.Clock()
        self.font = pygame.font.SysFont("verdana", 16)
        self.small_font = pygame.font.SysFont("verdana", 13)
        self.tiny_font = pygame.font.SysFont("verdana", 11)
        self.title_font = pygame.font.SysFont("verdana", 18, bold=True)
        self.title_font_italic = pygame.font.SysFont("verdana", 18, italic=True)
        self.large_font = pygame.font.SysFont("verdana", 24, bold=True)
        self.button_rects: dict[str, pygame.Rect] = {}
        self.video_row_rects: dict[str, pygame.Rect] = {}
        self.visible_fields: list[PygameTextField] = []
        self.timeline_rect = pygame.Rect(0, 0, 1, 1)
        self._cached_video_surface: pygame.Surface | None = None
        self._cached_video_key: tuple[Any, ...] | None = None
        self._video_dest = pygame.Rect(0, 0, 1, 1)
        self.zoom_level = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self.panning = False
        self.nav_hold_key: int | None = None
        self.nav_hold_started_ms = 0
        self.nav_hold_last_ms = 0
        # Frames (0-based) marked as repeated per video: D/Delete toggles,
        # "Detect repeats" adds the automatic ones. Saved into the sync file,
        # the checkpoint and the repeat-free copies.
        self.removed_frames: dict[str, set[int]] = {info.name: set() for info in self.video_infos}
        self.repeat_reports: dict[str, RepeatedFrameReport] = {}

        pygame.display.set_caption(
            "vailá Sync | Space Play/Pause | hold Left/Right 1 frame | "
            "Up/Down or -/+ 60 frames | wheel zoom, 0 reset | D mark repeat | PgUp/PgDn Video"
        )
        icon_path = Path(__file__).resolve().parent / "images" / "vaila_ico_mac.png"
        if icon_path.is_file():
            with contextlib.suppress(pygame.error):
                pygame.display.set_icon(pygame.image.load(str(icon_path)))
        self._open_video(0)

    @property
    def current_info(self) -> VideoInfo:
        return self.video_infos[self.current_index]

    def _set_status(self, text: str, *, error: bool = False) -> None:
        self.status_text = text
        self.status_color = (255, 130, 120) if error else (160, 225, 175)

    def _release_capture(self) -> None:
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _open_video(self, index: int, *, target_frame: int | None = None) -> None:
        if self.capture is not None:
            self.positions[self.current_info.name] = self.current_frame
        self.playing = False
        self._release_capture()
        self.current_index = max(0, min(int(index), len(self.video_infos) - 1))
        self._reset_zoom()
        info = self.current_info
        self.capture = cv2.VideoCapture(str(info.path))
        if not self.capture.isOpened():
            raise SyncWorkflowError(f"OpenCV could not open video: {info.path}")
        wanted = self.positions[info.name] if target_frame is None else target_frame
        if not self._decode_at(wanted):
            raise SyncWorkflowError(f"OpenCV could not decode video: {info.path}")
        if self.active_field is not self.real_fps_field:
            override = self.fps_overrides.get(info.name)
            self.real_fps_field.text = f"{override:.6g}" if override is not None else ""
        self._set_status(f"Camera {self.current_index + 1}/{len(self.video_infos)}: {info.name}")

    def _decode_at(self, frame_index: int) -> bool:
        if self.capture is None:
            return False
        target = frame_after_navigation(0, self.current_info.frame_count, int(frame_index))
        self.capture.set(cv2.CAP_PROP_POS_FRAMES, target)
        ok, frame = self.capture.read()
        if not ok or frame is None:
            self._set_status(f"Could not decode frame {target + 1}", error=True)
            return False
        self.current_frame = target
        self.positions[self.current_info.name] = target
        self.frame = frame
        self._cached_video_surface = None
        self._cached_video_key = None
        return True

    def _seek_relative(self, delta: int) -> None:
        self.playing = False
        target = frame_after_navigation(self.current_frame, self.current_info.frame_count, delta)
        if (
            target == self.current_frame + 1
            and self.capture is not None
            and int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) == target
        ):
            # Next frame: a sequential read avoids a keyframe seek while held.
            ok, frame = self.capture.read()
            if ok and frame is not None:
                self.current_frame = target
                self.positions[self.current_info.name] = target
                self.frame = frame
                self._cached_video_surface = None
                self._cached_video_key = None
                return
        self._decode_at(target)

    def _start_nav_hold(self, key: int) -> None:
        """Step once now; ``_tick_nav_hold`` repeats while the key stays down."""
        self._seek_relative(PLAYER_NAV_KEYS[key])
        self.nav_hold_key = key
        self.nav_hold_started_ms = pygame.time.get_ticks()
        self.nav_hold_last_ms = self.nav_hold_started_ms

    def _tick_nav_hold(self) -> None:
        if self.nav_hold_key is None:
            return
        if (
            self.playing
            or self.active_field is not None
            or not pygame.key.get_pressed()[self.nav_hold_key]
        ):
            self.nav_hold_key = None
            return
        try:
            from .cutvideo import nav_hold_should_step
        except ImportError:
            from cutvideo import nav_hold_should_step  # ty: ignore[unresolved-import]

        now_ms = pygame.time.get_ticks()
        delta = PLAYER_NAV_KEYS[self.nav_hold_key]
        if nav_hold_should_step(
            now_ms, self.nav_hold_started_ms, self.nav_hold_last_ms, vertical=abs(delta) > 1
        ):
            self._seek_relative(delta)
            self.nav_hold_last_ms = now_ms

    def _toggle_removed_frame(self) -> None:
        """Mark or unmark the current frame of the current video as repeated."""
        self.playing = False
        marks = self.removed_frames[self.current_info.name]
        frame = self.current_frame
        if frame in marks:
            marks.discard(frame)
            self._set_status(f"Frame {frame + 1} kept in {self.current_info.name}")
        elif len(marks) + 1 >= self.current_info.frame_count:
            self._set_status("Cannot mark every frame of a video as repeated", error=True)
        else:
            marks.add(frame)
            self._set_status(
                f"Frame {frame + 1} marked as repeated in {self.current_info.name} "
                f"({len(marks)} marked; D again to keep it)"
            )

    def _reset_zoom(self) -> None:
        self.zoom_level = 1.0
        self.offset_x = 0.0
        self.offset_y = 0.0
        self._cached_video_surface = None
        self._cached_video_key = None

    def _zoomed_size(self, viewport: pygame.Rect) -> tuple[float, int, int]:
        """Effective scale and zoomed frame size for the current zoom level."""
        height, width = self.frame.shape[:2]
        effective = min(viewport.width / width, viewport.height / height) * self.zoom_level
        return effective, max(1, round(width * effective)), max(1, round(height * effective))

    def _clamp_pan(self, viewport: pygame.Rect) -> None:
        if self.frame is None:
            return
        _effective, zoomed_w, zoomed_h = self._zoomed_size(viewport)
        self.offset_x = max(0.0, min(float(max(0, zoomed_w - viewport.width)), self.offset_x))
        self.offset_y = max(0.0, min(float(max(0, zoomed_h - viewport.height)), self.offset_y))

    def _zoom_at(self, pos: tuple[int, int], factor: float) -> None:
        """Zoom by ``factor`` keeping the frame point under ``pos`` in place."""
        if self.frame is None:
            return
        viewport = self._video_viewport()
        old_effective, _w, _h = self._zoomed_size(viewport)
        zoom = max(1.0, min(PLAYER_ZOOM_MAX, self.zoom_level * factor))
        if zoom == self.zoom_level:
            return
        source_x = (pos[0] - self._video_dest.x + self.offset_x) / old_effective
        source_y = (pos[1] - self._video_dest.y + self.offset_y) / old_effective
        self.zoom_level = zoom
        new_effective, zoomed_w, zoomed_h = self._zoomed_size(viewport)
        dest_x = viewport.x + max(0, (viewport.width - zoomed_w) // 2)
        dest_y = viewport.y + max(0, (viewport.height - zoomed_h) // 2)
        self.offset_x = source_x * new_effective - (pos[0] - dest_x)
        self.offset_y = source_y * new_effective - (pos[1] - dest_y)
        self._clamp_pan(viewport)

    def _advance_playback(self) -> None:
        if not self.playing or self.capture is None:
            return
        steps = 0
        if self.playback_speed >= 1.0:
            steps = max(1, int(self.playback_speed))
        else:
            self.slow_motion_accumulator += self.playback_speed
            if self.slow_motion_accumulator >= 1.0:
                self.slow_motion_accumulator -= 1.0
                steps = 1
        for _ in range(steps):
            ok, frame = self.capture.read()
            if not ok or frame is None:
                self.playing = False
                self._decode_at(self.current_info.frame_count - 1)
                return
            self.frame = frame
            self.current_frame = min(
                int(self.capture.get(cv2.CAP_PROP_POS_FRAMES)) - 1,
                self.current_info.frame_count - 1,
            )
            self.positions[self.current_info.name] = self.current_frame
            self._cached_video_surface = None
            self._cached_video_key = None
        if self.current_frame >= self.current_info.frame_count - 1:
            self.playing = False

    def _reference_name_or_none(self) -> str | None:
        try:
            return resolve_reference_video(self.reference_field.text, self.video_infos)
        except SyncWorkflowError:
            return None

    def _all_fields(self) -> list[PygameTextField]:
        """Tab order: reference fields, each camera's Sync/Start/End, then real FPS."""
        if getattr(self, "event_mode", False):
            return [
                self.reference_field,
                self.reference_clock_field,
                self.output_rate_field,
                *self.event_fields.values(),
            ]
        reference_name = self._reference_name_or_none()
        fields = [self.reference_field, self.start_field, self.end_field]
        for info in self.video_infos:
            fields.append(self.sync_fields[info.name])
            if info.name != reference_name:
                fields += [self.range_start_fields[info.name], self.range_end_fields[info.name]]
        fields.append(self.real_fps_field)
        return fields

    def _range_texts(self) -> dict[str, tuple[str, str]]:
        return {
            info.name: (
                self.range_start_fields[info.name].text,
                self.range_end_fields[info.name].text,
            )
            for info in self.video_infos
        }

    def _effective_video_infos(self) -> tuple[list[VideoInfo], list[str]]:
        """Video metadata with container-consistent real-FPS overrides applied."""
        return apply_fps_overrides(self.video_infos, self.fps_overrides)

    def _effective_fps(self, info: VideoInfo) -> float:
        rate = self.fps_overrides.get(info.name)
        if rate is not None and fps_rates_match(rate, info.fps):
            return float(rate)
        return float(info.fps)

    def _plan_preview(self) -> dict[str, tuple[int, int]]:
        """Ranges (0-based) the current fields would save, or {} while incomplete."""
        if self.event_mode:
            return {}
        try:
            infos, _warnings = self._effective_video_infos()
            _reference, entries = build_sync_plan_from_fields(
                infos,
                {name: field.text for name, field in self.sync_fields.items()},
                self.reference_field.text,
                self.start_field.text,
                self.end_field.text,
                range_texts=self._range_texts(),
            )
        except SyncWorkflowError:
            return {}
        return {entry.video_file: (entry.start_frame, entry.end_frame) for entry in entries}

    def _set_fps_override(self, *, all_cameras: bool) -> None:
        """Store typed real FPS only when it matches each file's playback rate.

        Cut Video uses container fps. Locking equal frame counts on a mismatched
        override (29.97 on a 13.91 norepeat file) makes cuts share frame counts
        but not duration — Start/End stop being the same temporal events.
        """
        try:
            rate = parse_fps_field(self.real_fps_field.text)
        except SyncWorkflowError as exc:
            self._set_status(str(exc), error=True)
            return
        targets = list(self.video_infos) if all_cameras else [self.current_info]
        applied: list[str] = []
        skipped: list[str] = []
        for info in targets:
            if fps_rates_match(rate, info.fps):
                self.fps_overrides[info.name] = rate
                applied.append(info.name)
            else:
                self.fps_overrides.pop(info.name, None)
                skipped.append(f"{info.name} ({info.fps:.3f} fps)")
        if applied and not skipped:
            self._set_status(
                f"Real FPS {rate:.6g} on {len(applied)} camera(s) — Fill Start/End: "
                "equal fps → equal frames; same temporal window"
            )
            return
        if applied and skipped:
            self._set_status(
                f"Real FPS {rate:.6g} on {len(applied)}; skipped mismatch: "
                + ", ".join(skipped)
                + " (use each file's real rate for time sync)",
                error=True,
            )
            return
        self._set_status(
            f"No camera plays at {rate:g} fps. Typed rate must match the file "
            f"(current {self.current_info.name} is {self.current_info.fps:.3f}). "
            "Norepeat cams keep the rate in the filename.",
            error=True,
        )

    def _clear_fps_override_current(self) -> None:
        """Drop the real-FPS override for the current camera (back to container fps)."""
        removed = self.fps_overrides.pop(self.current_info.name, None)
        if removed is None:
            self._set_status(f"{self.current_info.name} already uses container fps")
            return
        self.real_fps_field.text = ""
        self._set_status(
            f"Cleared real FPS for {self.current_info.name} (now {self.current_info.fps:.3f} fps)"
        )

    def _fill_ranges_from_fps(self) -> None:
        """Write each camera's Start/End from sync + real FPS so cuts share one time window.

        Clears empty-optional behaviour by materialising the computed ranges.
        Cameras that cannot cover the reference window report an error instead
        of writing a partial/wrong cut.
        """
        infos, override_warnings = self._effective_video_infos()
        try:
            reference, entries = build_sync_plan_from_fields(
                infos,
                {name: field.text for name, field in self.sync_fields.items()},
                self.reference_field.text,
                self.start_field.text,
                self.end_field.text,
                # Ignore prior typed Start/End so every camera is recomputed
                # from Sync + effective fps against the reference window.
                range_texts={info.name: ("", "") for info in self.video_infos},
            )
        except SyncWorkflowError as exc:
            self._set_status(str(exc), error=True)
            return
        filled = fill_range_texts_from_plan(entries, reference_video=reference)
        for name, (start_text, end_text) in filled.items():
            self.range_start_fields[name].text = start_text
            self.range_end_fields[name].text = end_text
        lines = describe_sync_plan(entries, infos, reference)
        print(">> vaila/syncvid: Fill Start/End (same temporal window)\n  " + "\n  ".join(lines))
        for warning in override_warnings:
            print(f">> vaila/syncvid: {warning}")
        info_by_name = {info.name: info for info in infos}
        ref_info = info_by_name[reference]
        ref_entry = next(entry for entry in entries if entry.video_file == reference)
        ref_count = ref_entry.end_frame - ref_entry.start_frame + 1
        equal_fps_names = [
            entry.video_file
            for entry in entries
            if entry.video_file != reference
            and fps_rates_match(info_by_name[entry.video_file].fps, ref_info.fps)
        ]
        other_counts = sorted(
            {
                entry.end_frame - entry.start_frame + 1
                for entry in entries
                if entry.video_file != reference and entry.video_file not in equal_fps_names
            }
        )
        if equal_fps_names and not other_counts:
            self._set_status(
                f"Filled Start/End: same temporal window; all {len(entries)} cuts have "
                f"{ref_count} frames (equal fps → locked count)"
            )
        elif equal_fps_names:
            self._set_status(
                f"Filled Start/End: same temporal window; {len(equal_fps_names)} equal-fps "
                f"cam(s) locked to {ref_count} frames; other fps keep time align "
                f"({min(other_counts)}–{max(other_counts)} frames)"
            )
        else:
            self._set_status(
                f"Filled Start/End: same temporal window; fps differ so frame counts "
                f"differ ({min(other_counts + [ref_count])}–{max(other_counts + [ref_count])}) "
                "while durations match"
            )

    def _set_active_field(self, field: PygameTextField | None) -> None:
        for candidate in (
            self.reference_field,
            self.start_field,
            self.end_field,
            self.goto_field,
            self.real_fps_field,
            *self.sync_fields.values(),
            *self.range_start_fields.values(),
            *self.range_end_fields.values(),
            *getattr(self, "event_fields", {}).values(),
            *(
                [self.reference_clock_field, self.output_rate_field]
                if hasattr(self, "reference_clock_field")
                else []
            ),
        ):
            candidate.deactivate()
        self.active_field = field
        if field is None:
            return
        self.playing = False
        if field is self.goto_field:
            field.text = str(self.current_frame + 1)
        if field is self.real_fps_field and not field.text.strip():
            field.text = f"{self._effective_fps(self.current_info):.6g}"
        field.activate()
        prefix, _sep, video_name = field.key.partition(":")
        if prefix in ("sync", "start", "end", "events"):
            index = next(i for i, info in enumerate(self.video_infos) if info.name == video_name)
            if index != self.current_index:
                self._open_video(index)
            self._ensure_row_visible(index)

    def _goto_frame(self, text: str) -> None:
        """Seek the current video to a typed 1-based frame."""
        try:
            value = parse_frame_field(
                text, field="Go to frame", frame_count=self.current_info.frame_count
            )
        except SyncWorkflowError as exc:
            self._set_status(str(exc), error=True)
            return
        self.playing = False
        if self._decode_at(value):
            self._set_status(f"Frame {value + 1} of {self.current_info.name}")

    def _cycle_active_field(self, direction: int = 1) -> None:
        fields = self._all_fields()
        if self.active_field not in fields:
            self._set_active_field(fields[0])
            return
        current = fields.index(self.active_field)
        self._set_active_field(fields[(current + direction) % len(fields)])

    def _ensure_row_visible(self, index: int) -> None:
        visible_count = self._visible_row_count()
        if index < self.list_scroll:
            self.list_scroll = index
        elif index >= self.list_scroll + visible_count:
            self.list_scroll = index - visible_count + 1
        self._clamp_list_scroll()

    def _visible_row_count(self) -> int:
        actions_top = self.window_height - self.ACTIONS_AREA_HEIGHT
        return max(1, (actions_top - 54 - 246) // self.ROW_HEIGHT)

    def _clamp_list_scroll(self) -> None:
        maximum = max(0, len(self.video_infos) - self._visible_row_count())
        self.list_scroll = max(0, min(self.list_scroll, maximum))

    def _commit_active_field(self) -> None:
        field = self.active_field
        if field is None:
            return
        try:
            if field is self.reference_field:
                name = resolve_reference_video(field.text, self.video_infos)
                self._set_status(f"Reference video: {name}")
            elif field is self.start_field or field is self.end_field:
                reference = resolve_reference_video(self.reference_field.text, self.video_infos)
                info = next(item for item in self.video_infos if item.name == reference)
                label = (
                    "reference start frame" if field is self.start_field else "reference end frame"
                )
                value = parse_frame_field(field.text, field=label, frame_count=info.frame_count)
                self._set_status(f"{label.capitalize()}: {value + 1}")
            elif field is self.goto_field:
                self._set_active_field(None)
                self._goto_frame(field.text)
                return
            elif field is self.real_fps_field:
                if not field.text.strip():
                    self._clear_fps_override_current()
                else:
                    self._set_fps_override(all_cameras=False)
                self._set_active_field(None)
                return
            elif self.event_mode and field in [
                self.reference_clock_field,
                self.output_rate_field,
                *self.event_fields.values(),
            ]:
                manifest = self._event_manifest()
                self._set_status(
                    f"Event timing valid: {manifest['output_frame_count']} output frames"
                )
            elif field.key.partition(":")[0] in ("sync", "start", "end"):
                prefix, _sep, video_name = field.key.partition(":")
                label = {"sync": "Sync frame", "start": "Start", "end": "End"}[prefix]
                info = next(item for item in self.video_infos if item.name == video_name)
                if prefix != "sync" and not field.text.strip():
                    self._set_status(f"{label} for {video_name} cleared: follows sync frame + fps")
                    self._set_active_field(None)
                    return
                value = parse_frame_field(
                    field.text,
                    field=f"{label.lower()} for {video_name}",
                    frame_count=info.frame_count,
                )
                index = self.video_infos.index(info)
                if index != self.current_index:
                    self._open_video(index, target_frame=value)
                else:
                    self._decode_at(value)
                self._set_status(f"{label} for {video_name}: {value + 1}")
            self._set_active_field(None)
        except (SyncWorkflowError, EventSyncError) as exc:
            self._set_status(str(exc), error=True)

    def _panel_width(self) -> int:
        return min(
            self.PANEL_MAX_WIDTH,
            max(self.PANEL_MIN_WIDTH, int(self.window_width * 0.34)),
        )

    def _video_viewport(self) -> pygame.Rect:
        panel_width = self._panel_width()
        return pygame.Rect(
            0,
            0,
            max(1, self.window_width - panel_width),
            max(1, self.window_height - self.CONTROL_HEIGHT),
        )

    def _get_video_surface(self, viewport: pygame.Rect) -> pygame.Surface | None:
        if self.frame is None:
            return None
        try:
            from .cutvideo import visible_source_rect
        except ImportError:
            from cutvideo import visible_source_rect  # ty: ignore[unresolved-import]

        self._clamp_pan(viewport)
        key = (
            self.current_frame,
            viewport.width,
            viewport.height,
            self.zoom_level,
            int(self.offset_x),
            int(self.offset_y),
        )
        if self._cached_video_surface is not None and self._cached_video_key == key:
            return self._cached_video_surface
        height, width = self.frame.shape[:2]
        # Resize only the visible crop, so a 10x zoom never builds a huge frame.
        x0, y0, x1, y1, dest_w, dest_h = visible_source_rect(
            width,
            height,
            viewport.width,
            viewport.height,
            self.zoom_level,
            self.offset_x,
            self.offset_y,
        )
        shrinking = dest_w < x1 - x0
        resized = cv2.resize(
            self.frame[y0:y1, x0:x1],
            (dest_w, dest_h),
            interpolation=cv2.INTER_AREA if shrinking else cv2.INTER_LINEAR,
        )
        rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        self._cached_video_surface = pygame.surfarray.make_surface(rgb.swapaxes(0, 1))
        self._cached_video_key = key
        return self._cached_video_surface

    @staticmethod
    def _fit_label(text: str, font: pygame.font.Font, width: int) -> str:
        if font.size(text)[0] <= width:
            return text
        suffix = "..."
        fitted = text
        while fitted and font.size(fitted + suffix)[0] > width:
            fitted = fitted[:-1]
        return fitted + suffix

    def _draw_button(
        self,
        rect: pygame.Rect,
        label: str,
        *,
        active: bool = False,
        accent: bool = False,
    ) -> None:
        if accent:
            color = (34, 120, 90)
        elif active:
            color = (45, 112, 172)
        else:
            color = (73, 82, 94)
        pygame.draw.rect(self.screen, color, rect, border_radius=5)
        pygame.draw.rect(self.screen, (132, 145, 158), rect, 1, border_radius=5)
        rendered = self.small_font.render(label, True, (248, 250, 252))
        self.screen.blit(rendered, rendered.get_rect(center=rect.center))

    def _draw_video_and_controls(self) -> None:
        viewport = self._video_viewport()
        pygame.draw.rect(self.screen, (8, 10, 13), viewport)
        video_surface = self._get_video_surface(viewport)
        if video_surface is not None:
            self._video_dest = video_surface.get_rect(center=viewport.center)
            self.screen.blit(video_surface, self._video_dest)
        marks = self.removed_frames[self.current_info.name]
        if self.current_frame in marks:
            pygame.draw.rect(self.screen, (230, 60, 60), self._video_dest.clip(viewport), 4)
            removed_label = self.font.render(
                "REPEATED - removed in saved videos (D to keep)", True, (255, 235, 235)
            )
            label_rect = removed_label.get_rect(midbottom=(viewport.centerx, viewport.bottom - 10))
            pygame.draw.rect(self.screen, (170, 35, 35), label_rect.inflate(16, 8), border_radius=4)
            self.screen.blit(removed_label, label_rect)

        overlay = pygame.Surface((viewport.width, 36), pygame.SRCALPHA)
        overlay.fill((10, 12, 16, 205))
        self.screen.blit(overlay, viewport.topleft)
        zoom_text = f"Zoom {self.zoom_level:.1f}x  " if self.zoom_level > 1.0 else ""
        speed_label = self.small_font.render(
            f"{zoom_text}Speed {format_playback_speed(self.playback_speed)}x",
            True,
            (245, 247, 250),
        )
        self.screen.blit(speed_label, (viewport.right - speed_label.get_width() - 12, 9))
        info = self.current_info
        info_text = self._fit_label(
            f"{self.current_index + 1}/{len(self.video_infos)}  {info.name}  |  "
            f"{info.fps:.3f} fps  |  {info.width}x{info.height}",
            self.small_font,
            viewport.width - speed_label.get_width() - 36,
        )
        self.screen.blit(self.small_font.render(info_text, True, (245, 247, 250)), (12, 9))

        controls = pygame.Rect(
            0,
            viewport.bottom,
            viewport.width,
            self.window_height - viewport.bottom,
        )
        pygame.draw.rect(self.screen, (28, 32, 38), controls)
        seconds = self.current_frame / self.current_info.fps
        duration = (self.current_info.frame_count - 1) / self.current_info.fps
        recount = ""
        if marks:
            kept_total = self.current_info.frame_count - len(marks)
            if self.current_frame in marks:
                recount = f"  (new -/{kept_total:,})"
            else:
                new_index = self.current_frame - sum(1 for f in marks if f < self.current_frame)
                recount = f"  (new {new_index + 1:,}/{kept_total:,})"
        frame_label = (
            f"Frame {self.current_frame + 1:,}/{self.current_info.frame_count:,}{recount}"
            f"    {seconds:.6f}s / {duration:.6f}s"
        )
        if self.event_mode:
            try:
                timeline, _rows = temporal_maps(self._event_manifest(), info.name)
                original = self.current_frame + 1
                if original in timeline.frames:
                    ordinal = timeline.frames.index(original)
                    frame_label = (
                        f"Frame {original} | kept {ordinal + 1}/{len(timeline.frames)} | "
                        f"physical {float(timeline.times[ordinal]):.6f}s"
                    )
                else:
                    reason = "removed" if self.current_frame in marks else "outside event interval"
                    frame_label = f"Frame {original} | {reason}: no assigned physical time"
            except EventSyncError:
                frame_label = (
                    f"Frame {self.current_frame + 1} | correct events to assign physical time"
                )
        frame_text = self.font.render(
            self._fit_label(frame_label, self.font, controls.width - 24),
            True,
            (238, 242, 247),
        )
        self.screen.blit(frame_text, (16, controls.y + 10))

        self.timeline_rect = pygame.Rect(
            20,
            controls.y + 44,
            max(40, controls.width - 40),
            12,
        )
        pygame.draw.rect(self.screen, (73, 79, 87), self.timeline_rect, border_radius=4)
        fraction = (
            self.current_frame / (self.current_info.frame_count - 1)
            if self.current_info.frame_count > 1
            else 0.0
        )
        knob_x = self.timeline_rect.left + round(fraction * self.timeline_rect.width)
        pygame.draw.rect(
            self.screen,
            (52, 143, 214),
            (
                self.timeline_rect.left,
                self.timeline_rect.top,
                max(1, knob_x - self.timeline_rect.left),
                self.timeline_rect.height,
            ),
            border_radius=4,
        )
        if self.current_info.frame_count > 1:
            for frame in marks:
                tick_x = self.timeline_rect.left + round(
                    frame / (self.current_info.frame_count - 1) * self.timeline_rect.width
                )
                pygame.draw.line(
                    self.screen,
                    (235, 70, 70),
                    (tick_x, self.timeline_rect.top - 4),
                    (tick_x, self.timeline_rect.bottom + 3),
                    2,
                )
        pygame.draw.circle(
            self.screen,
            (250, 252, 255),
            (knob_x, self.timeline_rect.centery),
            8,
        )

        self.button_rects = {
            key: rect for key, rect in self.button_rects.items() if key.startswith("side:")
        }
        step_labels = {delta: label for label, delta in PLAYER_STEP_BUTTONS}
        specs = [
            ("step:-60", step_labels[-60], 58),
            ("step:-1", step_labels[-1], 48),
            ("play", "Pause" if self.playing else "Play", 72),
            ("step:1", step_labels[1], 48),
            ("step:60", step_labels[60], 58),
            ("previous_video", "Prev video", 88),
            ("next_video", "Next video", 88),
        ]
        x = 16
        y = controls.y + 72
        for action, label, width in specs:
            rect = pygame.Rect(x, y, width, 30)
            self.button_rects[action] = rect
            self._draw_button(rect, label, active=action == "play" and self.playing)
            x += width + 7

        goto_y = controls.y + 110
        goto_label = self.small_font.render("Go to frame (G)", True, (198, 208, 218))
        self.screen.blit(goto_label, (16, goto_y + 6))
        goto_x = 16 + goto_label.get_width() + 8
        self.goto_field.set_rect(pygame.Rect(goto_x, goto_y, 84, 28))
        self.goto_field.placeholder = str(self.current_frame + 1)
        self.goto_field.draw(self.screen, self.small_font)
        go_rect = pygame.Rect(goto_x + 90, goto_y, 46, 28)
        self.button_rects["goto"] = go_rect
        self._draw_button(go_rect, "Go")

        hint = self.tiny_font.render(
            "Space play | hold arrows step | [/] speed | wheel zoom, drag pan, 0 reset | D repeat",
            True,
            (174, 188, 202),
        )
        self.screen.blit(hint, (16, controls.bottom - 24))

    def _draw_wrapped_status(self, rect: pygame.Rect) -> None:
        words = self.status_text.split()
        lines: list[str] = []
        current = ""
        for word in words:
            trial = f"{current} {word}".strip()
            if self.small_font.size(trial)[0] <= rect.width or not current:
                current = trial
            else:
                lines.append(current)
                current = word
                if len(lines) == 2:
                    break
        if current and len(lines) < 2:
            lines.append(current)
        for index, line in enumerate(lines[:2]):
            rendered = self.small_font.render(line, True, self.status_color)
            self.screen.blit(rendered, (rect.x, rect.y + index * 18))

    def _draw_side_panel(self) -> None:
        panel_width = self._panel_width()
        panel = pygame.Rect(
            self.window_width - panel_width,
            0,
            panel_width,
            self.window_height,
        )
        pygame.draw.rect(self.screen, (38, 43, 50), panel)
        pygame.draw.line(
            self.screen,
            (92, 103, 115),
            panel.topleft,
            panel.bottomleft,
            1,
        )
        x = panel.x + 14
        content_width = panel.width - 28
        vaila_label = self.title_font_italic.render("vailá", True, (248, 250, 252))
        self.screen.blit(vaila_label, (x, 14))
        self.screen.blit(
            self.title_font.render(" Sync", True, (248, 250, 252)),
            (x + vaila_label.get_width(), 14),
        )

        self.screen.blit(
            self.small_font.render(
                "Reference video (row, name, stem or unique text)",
                True,
                (198, 208, 218),
            ),
            (x, 48),
        )
        self.reference_field.set_rect(pygame.Rect(x, 68, content_width, 31))
        self.reference_field.draw(self.screen, self.small_font)

        half = (content_width - 10) // 2
        self.screen.blit(
            self.small_font.render(
                "Reference clock FPS" if self.event_mode else "Reference start",
                True,
                (198, 208, 218),
            ),
            (x, 109),
        )
        self.screen.blit(
            self.small_font.render(
                "Output grid FPS" if self.event_mode else "Reference end", True, (198, 208, 218)
            ),
            (x + half + 10, 109),
        )
        top_start = self.reference_clock_field if self.event_mode else self.start_field
        top_end = self.output_rate_field if self.event_mode else self.end_field
        top_start.set_rect(pygame.Rect(x, 130, half, 31))
        top_end.set_rect(pygame.Rect(x + half + 10, 130, half, 31))
        top_start.draw(self.screen, self.small_font)
        top_end.draw(self.screen, self.small_font)

        pygame.draw.line(
            self.screen,
            (82, 91, 102),
            (x, 178),
            (panel.right - 14, 178),
            1,
        )
        mode_rect = pygame.Rect(x, 184, content_width, 26)
        self.button_rects["side:event_mode"] = mode_rect
        self._draw_button(
            mode_rect, "Sync by events: ON" if self.event_mode else "Sync by events: OFF"
        )
        self.screen.blit(
            self.tiny_font.render(
                "Ordered events: start, intermediate(s), end (1-based)"
                if self.event_mode
                else "Empty Start/End = from sync frame + fps (grey preview)",
                True,
                (168, 182, 196),
            ),
            (x, 214),
        )
        field_width = 60
        field_gap = 4
        column_x = [
            panel.right - 14 - 4 - (3 - column) * field_width - (2 - column) * field_gap
            for column in range(3)
        ]
        for column, title in enumerate(
            ("Events",) if self.event_mode else ("Sync", "Start", "End")
        ):
            rendered = self.tiny_font.render(title, True, (198, 208, 218))
            self.screen.blit(
                rendered, (column_x[column] + (field_width - rendered.get_width()) // 2, 229)
            )

        self._clamp_list_scroll()
        visible_count = self._visible_row_count()
        start = self.list_scroll
        end = min(len(self.video_infos), start + visible_count)
        self.video_row_rects.clear()
        self.visible_fields = [
            self.reference_field,
            top_start,
            top_end,
            self.goto_field,
        ]
        for field in (
            *self.sync_fields.values(),
            *self.range_start_fields.values(),
            *self.range_end_fields.values(),
            *self.event_fields.values(),
        ):
            field.set_rect(pygame.Rect(-1, -1, 0, 0))

        reference_name = self._reference_name_or_none()
        preview = self._plan_preview()

        list_y = 246
        name_width = column_x[0] - x - 6
        for visible_index, index in enumerate(range(start, end)):
            info = self.video_infos[index]
            row_y = list_y + visible_index * self.ROW_HEIGHT
            row_rect = pygame.Rect(x, row_y, content_width, self.ROW_HEIGHT - 4)
            self.video_row_rects[info.name] = row_rect
            if index == self.current_index:
                row_color = (55, 91, 122)
            elif info.name == reference_name:
                row_color = (69, 78, 87)
            else:
                row_color = (48, 53, 60)
            pygame.draw.rect(self.screen, row_color, row_rect, border_radius=4)

            prefix = f"{index + 1}. "
            if info.name == reference_name:
                prefix += "[REF] "
            marked = len(self.removed_frames[info.name])
            suffix = f" (-{marked})" if marked else ""
            label = self._fit_label(prefix + info.name + suffix, self.tiny_font, name_width - 10)
            self.screen.blit(
                self.tiny_font.render(label, True, (235, 239, 243)),
                (row_rect.x + 6, row_rect.y + 3),
            )
            effective_fps = self._effective_fps(info)
            fps_overridden = info.name in self.fps_overrides
            fps_color = (250, 210, 120) if fps_overridden else (168, 182, 196)
            fps_label = f"{effective_fps:.3f} fps*" if fps_overridden else f"{info.fps:.3f} fps"
            self.screen.blit(
                self.tiny_font.render(f"{fps_label} · {info.frame_count:,} fr", True, fps_color),
                (row_rect.x + 6, row_rect.y + 19),
            )
            row_fields = (
                [self.event_fields[info.name]] if self.event_mode else [self.sync_fields[info.name]]
            )
            if not self.event_mode and info.name != reference_name:
                row_fields += [self.range_start_fields[info.name], self.range_end_fields[info.name]]
                planned = preview.get(info.name)
                self.range_start_fields[info.name].placeholder = (
                    str(planned[0] + 1) if planned else ""
                )
                self.range_end_fields[info.name].placeholder = (
                    str(planned[1] + 1) if planned else ""
                )
            for column, field in enumerate(row_fields):
                field.set_rect(
                    pygame.Rect(
                        column_x[column],
                        row_rect.y + 4,
                        panel.right - 18 - column_x[0] if self.event_mode else field_width,
                        row_rect.height - 8,
                    )
                )
                field.draw(self.screen, self.small_font)
                self.visible_fields.append(field)
            if info.name == reference_name and not self.event_mode:
                for column in (1, 2):
                    cell = pygame.Rect(
                        column_x[column], row_rect.y + 4, field_width, row_rect.height - 8
                    )
                    pygame.draw.rect(self.screen, (60, 67, 76), cell, border_radius=4)
                    rendered = self.tiny_font.render("ref", True, (150, 160, 172))
                    self.screen.blit(rendered, rendered.get_rect(center=cell.center))

        actions_top = self.window_height - self.ACTIONS_AREA_HEIGHT
        status_rect = pygame.Rect(x, actions_top - 47, content_width, 40)
        self._draw_wrapped_status(status_rect)

        gap = 8
        first_width = (content_width - gap) // 2
        save_rect = pygame.Rect(x, actions_top, first_width, 34)
        save_cut_rect = pygame.Rect(x + first_width + gap, actions_top, first_width, 34)
        help_rect = pygame.Rect(x, actions_top + 43, first_width, 32)
        cancel_rect = pygame.Rect(
            x + first_width + gap,
            actions_top + 43,
            first_width,
            32,
        )
        load_rect = pygame.Rect(x, actions_top + 86, content_width, 32)
        self.screen.blit(
            self.tiny_font.render(
                "Calibrate AFTER repeat review; inferred rates per segment:"
                if self.event_mode
                else "Real FPS (norepeat / all cams) — Enter sets current; * = overridden",
                True,
                (168, 182, 196),
            ),
            (x, actions_top + 122),
        )
        third = max(70, (content_width - 2 * gap) // 3)
        self.real_fps_field.set_rect(
            pygame.Rect(-1, -1, 0, 0)
            if self.event_mode
            else pygame.Rect(x, actions_top + 138, third, 28)
        )
        if not self.event_mode:
            self.real_fps_field.draw(self.screen, self.small_font)
            self.visible_fields.append(self.real_fps_field)
        fps_this_rect = pygame.Rect(x + third + gap, actions_top + 138, third, 28)
        fps_all_rect = pygame.Rect(x + 2 * (third + gap), actions_top + 138, third, 28)
        fill_rect = pygame.Rect(x, actions_top + 172, content_width, 28)
        detect_rect = pygame.Rect(x, actions_top + 206, first_width, 28)
        write_rect = pygame.Rect(x + first_width + gap, actions_top + 206, first_width, 28)
        self.button_rects["side:save"] = save_rect
        self.button_rects["side:save_cutvideo"] = save_cut_rect
        self.button_rects["side:help"] = help_rect
        self.button_rects["side:cancel"] = cancel_rect
        self.button_rects["side:load"] = load_rect
        if not self.event_mode:
            self.button_rects["side:fps_this"] = fps_this_rect
            self.button_rects["side:fps_all"] = fps_all_rect
        self.button_rects["side:fill_ranges"] = fill_rect
        self.button_rects["side:detect_repeats"] = detect_rect
        self.button_rects["side:save_without_repeats"] = write_rect
        self._draw_button(save_rect, "Save sync", accent=True)
        self._draw_button(save_cut_rect, "Save + Cut Video", accent=True)
        self._draw_button(help_rect, "Help")
        self._draw_button(cancel_rect, "Cancel")
        self._draw_button(load_rect, "Load / Resume")
        if self.event_mode:
            try:
                manifest = self._event_manifest()
                segments = manifest["cameras"][self.current_info.name]["segments"]
                rates = " / ".join(f"{s['effective_fps_decimal']:.3f}" for s in segments)
                label = self._fit_label(
                    f"Current: {rates} fps (model)", self.small_font, content_width
                )
                self.screen.blit(
                    self.small_font.render(label, True, (250, 210, 120)), (x, actions_top + 144)
                )
            except EventSyncError:
                self.screen.blit(
                    self.tiny_font.render(
                        "Enter valid events in every camera.", True, (250, 210, 120)
                    ),
                    (x, actions_top + 144),
                )
        else:
            self._draw_button(fps_this_rect, "FPS this")
            self._draw_button(fps_all_rect, "FPS all")
        self._draw_button(
            fill_rect,
            "Validate event timing" if self.event_mode else "Fill Start/End from FPS",
            accent=True,
        )
        self._draw_button(detect_rect, "Detect repeats")
        self._draw_button(write_rect, "Save w/o repeats")

        footer = self.tiny_font.render(
            "Tab fields | Enter validate | Ctrl+S save | Esc cancel",
            True,
            (164, 177, 190),
        )
        self.screen.blit(footer, (x, actions_top + 240))

    def _draw(self) -> None:
        self.screen.fill((18, 21, 25))
        self._draw_video_and_controls()
        self._draw_side_panel()
        pygame.display.flip()

    def _timeline_frame_from_x(self, mouse_x: int) -> int:
        if self.timeline_rect.width <= 0:
            return 0
        fraction = (mouse_x - self.timeline_rect.left) / self.timeline_rect.width
        fraction = max(0.0, min(1.0, fraction))
        return round(fraction * (self.current_info.frame_count - 1))

    def _handle_button(self, action: str) -> None:
        if action == "side:event_mode":
            self._set_active_field(None)
            self.event_mode = not self.event_mode
            if self.event_mode:
                reference = self._reference_name_or_none()
                for info in self.video_infos:
                    if self.event_fields[info.name].text.strip():
                        continue
                    start, end = (
                        (self.start_field.text, self.end_field.text)
                        if info.name == reference
                        else self._range_texts()[info.name]
                    )
                    self.event_fields[
                        info.name
                    ].text = f"{start}, {self.sync_fields[info.name].text}, {end}"
            self._set_status(
                "Event times use retained samples; review repeats before saving."
                if self.event_mode
                else "Sync frame + FPS mode"
            )
        elif action == "goto":
            self._goto_frame(self.goto_field.text or str(self.current_frame + 1))
        elif action.startswith("step:"):
            self._seek_relative(int(action.removeprefix("step:")))
        elif action == "play":
            self.playing = not self.playing
        elif action == "previous_video":
            self._open_video(self.current_index - 1)
            self._ensure_row_visible(self.current_index)
        elif action == "next_video":
            self._open_video(self.current_index + 1)
            self._ensure_row_visible(self.current_index)
        elif action == "side:save":
            self._save(False)
        elif action == "side:save_cutvideo":
            self._save(True)
        elif action == "side:help":
            self._open_help()
        elif action == "side:load":
            self._load_checkpoint()
        elif action == "side:fps_this":
            self._set_fps_override(all_cameras=False)
        elif action == "side:fps_all":
            self._set_fps_override(all_cameras=True)
        elif action == "side:fill_ranges":
            if self.event_mode:
                try:
                    manifest = self._event_manifest()
                    print(">> vaila/syncvid: Event calibration\n" + json.dumps(manifest, indent=2))
                    self._set_status(
                        f"Valid: {manifest['output_frame_count']} output frames; span {manifest['sample_span_seconds']:.6f} s"
                    )
                except (EventSyncError, SyncWorkflowError) as exc:
                    self._set_status(str(exc), error=True)
            else:
                self._fill_ranges_from_fps()
        elif action == "side:detect_repeats":
            self._detect_repeated_frames()
        elif action == "side:save_without_repeats":
            self._save_without_repeats()
        elif action == "side:cancel":
            self.running = False

    def _handle_player_key(self, event: pygame.event.Event) -> None:
        ctrl = bool(event.mod & pygame.KMOD_CTRL)
        if event.key == pygame.K_ESCAPE:
            self.running = False
        elif ctrl and event.key == pygame.K_s:
            self._save(False)
        elif ctrl and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._save(True)
        elif event.key == pygame.K_SPACE:
            self.playing = not self.playing
        elif event.key in PLAYER_NAV_KEYS and not self.playing:
            self._start_nav_hold(event.key)
        elif event.key in (pygame.K_0, pygame.K_KP0):
            self._reset_zoom()
        elif event.key == pygame.K_g:
            self._set_active_field(self.goto_field)
        elif event.key in (pygame.K_d, pygame.K_DELETE):
            self._toggle_removed_frame()
        elif event.key == pygame.K_RIGHTBRACKET:
            self.playback_speed = step_playback_speed(self.playback_speed, 1)
            self.slow_motion_accumulator = 0.0
        elif event.key == pygame.K_LEFTBRACKET:
            self.playback_speed = step_playback_speed(self.playback_speed, -1)
            self.slow_motion_accumulator = 0.0
        elif event.key == pygame.K_HOME and not self.playing:
            self._decode_at(0)
        elif event.key == pygame.K_END and not self.playing:
            self._decode_at(self.current_info.frame_count - 1)
        elif event.key == pygame.K_PAGEUP:
            self._open_video(self.current_index - 1)
            self._ensure_row_visible(self.current_index)
        elif event.key == pygame.K_PAGEDOWN:
            self._open_video(self.current_index + 1)
            self._ensure_row_visible(self.current_index)
        elif event.key == pygame.K_h:
            self._open_help()
        elif event.key == pygame.K_TAB:
            self._cycle_active_field(-1 if event.mod & pygame.KMOD_SHIFT else 1)

    def _handle_key(self, event: pygame.event.Event) -> None:
        if event.mod & pygame.KMOD_CTRL and event.key == pygame.K_s:
            self._save(False)
            return
        if event.mod & pygame.KMOD_CTRL and event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            self._save(True)
            return
        if self.active_field is None:
            self._handle_player_key(event)
            return
        if event.key == pygame.K_ESCAPE:
            self._set_active_field(None)
            return
        if event.key == pygame.K_TAB:
            self._cycle_active_field(-1 if event.mod & pygame.KMOD_SHIFT else 1)
            return
        if self.active_field.handle_key(event):
            self._commit_active_field()

    def _handle_mouse_down(self, event: pygame.event.Event) -> None:
        if event.button == 4:
            if event.pos[0] >= self.window_width - self._panel_width():
                self.list_scroll -= 1
                self._clamp_list_scroll()
            return
        if event.button == 5:
            if event.pos[0] >= self.window_width - self._panel_width():
                self.list_scroll += 1
                self._clamp_list_scroll()
            return
        if event.button == 2 and self._video_viewport().collidepoint(event.pos):
            self.panning = True
            return
        if event.button != 1:
            return

        for field in self.visible_fields:
            if field.rect.collidepoint(event.pos):
                self._set_active_field(field)
                return

        self._set_active_field(None)
        for action, rect in self.button_rects.items():
            if rect.collidepoint(event.pos):
                self._handle_button(action)
                return
        for name, rect in self.video_row_rects.items():
            if rect.collidepoint(event.pos):
                index = next(i for i, info in enumerate(self.video_infos) if info.name == name)
                self._open_video(index)
                return
        if self.timeline_rect.collidepoint(event.pos):
            self.dragging_timeline = True
            self.playing = False
            self._decode_at(self._timeline_frame_from_x(event.pos[0]))
        elif self.zoom_level > 1.0 and self._video_viewport().collidepoint(event.pos):
            self.panning = True

    def _handle_event(self, event: pygame.event.Event) -> None:
        close_event = getattr(pygame, "WINDOWCLOSE", -1)
        if event.type in (pygame.QUIT, close_event):
            self.running = False
        elif event.type == pygame.VIDEORESIZE:
            self.window_width = max(900, int(event.w))
            self.window_height = max(650, int(event.h))
            self.screen = pygame.display.set_mode(
                (self.window_width, self.window_height), pygame.RESIZABLE
            )
            self._cached_video_surface = None
            self._cached_video_key = None
            self._clamp_list_scroll()
        elif event.type == pygame.KEYDOWN:
            self._handle_key(event)
        elif event.type == pygame.MOUSEBUTTONDOWN:
            self._handle_mouse_down(event)
        elif event.type == pygame.MOUSEBUTTONUP and event.button in (1, 2):
            self.dragging_timeline = False
            self.panning = False
        elif event.type == pygame.MOUSEMOTION and self.dragging_timeline:
            self._decode_at(self._timeline_frame_from_x(event.pos[0]))
        elif event.type == pygame.MOUSEMOTION and self.panning:
            self.offset_x -= event.rel[0]
            self.offset_y -= event.rel[1]
            self._clamp_pan(self._video_viewport())
        elif event.type == pygame.MOUSEWHEEL:
            mouse_pos = pygame.mouse.get_pos()
            if mouse_pos[0] >= self.window_width - self._panel_width():
                self.list_scroll -= int(event.y)
                self._clamp_list_scroll()
            elif self._video_viewport().collidepoint(mouse_pos) and event.y:
                factor = PLAYER_ZOOM_STEP if event.y > 0 else 1.0 / PLAYER_ZOOM_STEP
                self._zoom_at(mouse_pos, factor)

    @contextlib.contextmanager
    def _sdl_released_for_dialog(self) -> Generator[None, None, None]:
        """Tear down the SDL window while a desktop file chooser owns the pointer.

        Iconify/topmost are ignored on several Linux WMs, so the only reliable
        way to keep the chooser clickable is to quit the display, then rebuild
        it (and refresh ``self.screen``) when the chooser returns.
        """
        size = self.screen.get_size()
        flags = self.screen.get_flags()
        title, icon = pygame.display.get_caption()
        pygame.display.quit()
        try:
            yield
        finally:
            pygame.display.init()
            self.screen = pygame.display.set_mode(size, flags)
            self._cached_video_surface = None
            self._cached_video_key = None
            if title:
                pygame.display.set_caption(title, icon)
            pygame.event.clear()
            pygame.event.pump()
            # A window recreated here can stay blank (the old content shows through) until
            # its size changes. Post the same event a user resize produces, so the normal
            # resize handler sets the mode again inside the event loop.
            pygame.event.post(pygame.event.Event(pygame.VIDEORESIZE, w=size[0], h=size[1]))

    def _choose_output_file(self) -> Path | None:
        if self.output_file is not None:
            return self.output_file
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with self._sdl_released_for_dialog():
            selected = _portable_path_dialog(
                "save",
                title="Save vailá synchronization file",
                initialdir=self.current_info.path.parent,
                initialfile=f"vaila_sync_{timestamp}.txt",
                defaultextension=".txt",
                filetypes=[("vailá sync file", "*.txt"), ("All files", "*.*")],
            )
        return Path(selected).expanduser().resolve() if selected else None

    def _checkpoint_state(self, *, completed: bool) -> dict[str, Any]:
        """Snapshot everything needed to resume this UI session later.

        ``completed`` marks whether this checkpoint came from "Save + Cut
        Video" (the session is handed off downstream) or plain "Save sync"
        (saved, but still open to being resumed/re-saved elsewhere).
        """
        state = build_checkpoint_state(
            reference=self.reference_field.text,
            start=self.start_field.text,
            end=self.end_field.text,
            sync_frames={name: field.text for name, field in self.sync_fields.items()},
            current_video=self.current_info.name,
            current_frame=self.current_frame,
            output_file=self.output_file,
            completed=completed,
            removed_frames=self.removed_frames,
            ranges=self._range_texts(),
            fps_overrides=self.fps_overrides,
        )
        state.update(
            sync_mode="events" if self.event_mode else "fps",
            event_texts={name: field.text for name, field in self.event_fields.items()},
            reference_fps=self.reference_clock_field.text,
            output_fps=self.output_rate_field.text,
        )
        return state

    def _event_manifest(self) -> dict[str, Any]:
        state = self._checkpoint_state(completed=False)
        try:
            state["reference"] = resolve_reference_video(
                self.reference_field.text, self.video_infos
            )
        except SyncWorkflowError as exc:
            raise EventSyncError(str(exc)) from exc
        state["events"] = {
            name: parse_event_frames(field.text, name) for name, field in self.event_fields.items()
        }
        return manifest_from_checkpoint(
            state,
            self.video_infos,
            reference_fps=self.reference_clock_field.text,
            output_fps=self.output_rate_field.text,
            checkpoint_file=self.loaded_checkpoint,
        )

    def _save_events(self, open_cutvideo: bool) -> None:
        try:
            manifest = self._event_manifest()
        except EventSyncError as exc:
            self._set_status(str(exc), error=True)
            return
        selected = self._choose_output_file()
        if selected is None:
            self._set_status("Save cancelled")
            return
        output = (
            selected
            if selected.name.endswith(".events.json")
            else selected.with_suffix(".events.json")
        )
        checkpoint = output.with_name(output.name.removesuffix(".events.json") + ".checkpoint.json")
        self.output_file = output
        manifest["checkpoint"] = str(checkpoint)
        manifest["imported_checkpoint"] = (
            str(self.loaded_checkpoint) if self.loaded_checkpoint else None
        )
        try:
            write_checkpoint_file(self._checkpoint_state(completed=open_cutvideo), checkpoint)
            write_checkpoint_file(manifest, output)
        except OSError as exc:
            self._set_status(str(exc), error=True)
            return
        entries = tuple(
            SyncPlanEntry(
                name,
                f"{name}_events.mp4",
                camera["events"][0] - 1,
                camera["events"][-1] - 1,
                removed_frames=tuple(
                    f - 1
                    for f in camera["removed_frames"]
                    if camera["events"][0] <= f <= camera["events"][-1]
                ),
            )
            for name, camera in manifest["cameras"].items()
        )
        reference_path = next(
            info.path for info in self.video_infos if info.name == manifest["reference"]
        )
        command = build_cutvideo_command(reference_path, output)
        print(">> vaila/syncvid: Equivalent event export CLI\n" + shlex.join(command))
        print(
            f">> Event sample span {manifest['sample_span_seconds']:.9f} s; "
            f"{manifest['output_frame_count']} frames at {manifest['output_fps']} fps; "
            f"playback {manifest['playback_duration_seconds']:.9f} s"
        )
        self.result = SyncRunResult(output, reference_path, entries, open_cutvideo)
        if open_cutvideo:
            self.running = False
        else:
            self._set_status(f"Saved events: {output.name} (GUI stays open)")

    def _apply_checkpoint(self, data: Mapping[str, Any]) -> None:
        """Restore session state (fields, position, UI) from a validated checkpoint.

        Cameras the checkpoint mentions that are not part of the currently
        loaded directory are skipped rather than rejected -- the checkpoint
        may have been written against a different take of the same shoot.
        """
        known = {info.name for info in self.video_infos}
        sync_frames = cast(dict[str, str], data["sync_frames"])
        skipped = sorted(name for name in sync_frames if name not in known)
        for name, text in sync_frames.items():
            field = self.sync_fields.get(name)
            if field is not None:
                field.text = text

        removed = cast(dict[str, list[int]], data.get("removed_frames", {}))
        for info in self.video_infos:
            self.removed_frames[info.name] = {
                frame - 1 for frame in removed.get(info.name, []) if frame <= info.frame_count
            }
        ranges = cast(dict[str, list[str]], data.get("ranges", {}))
        for info in self.video_infos:
            start_text, end_text = ranges.get(info.name, ["", ""])
            self.range_start_fields[info.name].text = start_text
            self.range_end_fields[info.name].text = end_text
        self.reference_field.text = cast(str, data["reference"])
        self.start_field.text = cast(str, data["start"])
        self.end_field.text = cast(str, data["end"])
        self.event_mode = data.get("sync_mode") == "events"
        for name, field in self.event_fields.items():
            field.text = data.get("event_texts", {}).get(name, "")
        self.reference_clock_field.text = data.get("reference_fps", "60000/1001")
        self.output_rate_field.text = data.get("output_fps", "30000/1001")
        overrides = cast(dict[str, float], data.get("fps_overrides", {}))
        info_by_name = {info.name: info for info in self.video_infos}
        self.fps_overrides = {}
        dropped_overrides: list[str] = []
        for name, rate in overrides.items():
            if name not in known:
                continue
            if fps_rates_match(float(rate), info_by_name[name].fps):
                self.fps_overrides[name] = float(rate)
            else:
                dropped_overrides.append(f"{name} ({rate:g}≠{info_by_name[name].fps:.3f})")
        if self.current_info.name in self.fps_overrides:
            self.real_fps_field.text = f"{self.fps_overrides[self.current_info.name]:.6g}"
        else:
            self.real_fps_field.text = ""

        output_file = data.get("output_file")
        self.output_file = Path(output_file).expanduser().resolve() if output_file else None

        current_video = cast(str, data["current_video"])
        index = next(
            (i for i, info in enumerate(self.video_infos) if info.name == current_video),
            self.current_index,
        )
        self._open_video(index, target_frame=int(data["current_frame"]))
        self._ensure_row_visible(index)

        state_label = "completed session" if data.get("completed") else "in-progress session"
        message = f"Resumed {state_label} from checkpoint"
        if skipped:
            message += f" (skipped unknown camera(s): {', '.join(skipped)})"
        if self.fps_overrides:
            message += f" ({len(self.fps_overrides)} real-FPS override(s))"
        if dropped_overrides:
            message += f" — dropped mismatched FPS: {', '.join(dropped_overrides)}"
            print(
                ">> vaila/syncvid: Dropped checkpoint FPS overrides that disagree "
                "with file playback rate:\n  " + "\n  ".join(dropped_overrides)
            )
        self._set_status(message, error=bool(dropped_overrides))

    def _load_checkpoint(self) -> None:
        with self._sdl_released_for_dialog():
            selected = _portable_path_dialog(
                "open",
                title="Load / Resume vailá synchronization checkpoint",
                initialdir=self.current_info.path.parent,
                filetypes=[("vailá checkpoint", "*.json"), ("All files", "*.*")],
            )
        if not selected:
            self._set_status("Load cancelled")
            return
        try:
            data = read_checkpoint_file(Path(selected))
            self._apply_checkpoint(data)
            self.loaded_checkpoint = Path(selected).resolve()
        except (SyncWorkflowError, OSError) as exc:
            self._set_status(f"Could not load checkpoint: {exc}", error=True)

    def _save(self, open_cutvideo: bool) -> None:
        if self.event_mode:
            self._save_events(open_cutvideo)
            return
        effective_infos, override_warnings = self._effective_video_infos()
        for warning in override_warnings:
            print(f">> vaila/syncvid: {warning}")
        try:
            reference_video, entries = build_sync_plan_from_fields(
                effective_infos,
                {name: field.text for name, field in self.sync_fields.items()},
                self.reference_field.text,
                self.start_field.text,
                self.end_field.text,
                self.removed_frames,
                self._range_texts(),
            )
        except SyncWorkflowError as exc:
            self._set_status(str(exc), error=True)
            return

        output = self._choose_output_file()
        if output is None:
            self._set_status("Save cancelled")
            return
        # Persist the resolved path so a Load/Resume checkpoint written below
        # records where this session's sync file actually went, not just
        # whatever (usually unset) value was preselected via --output.
        self.output_file = output
        try:
            sync_file = write_sync_file(entries, output)
        except (OSError, SyncWorkflowError) as exc:
            self._set_status(str(exc), error=True)
            return

        try:
            checkpoint_file = write_checkpoint_file(
                self._checkpoint_state(completed=open_cutvideo),
                sync_file.with_suffix(".json"),
            )
            print(f">> vaila/syncvid: Checkpoint saved for Load/Resume: {checkpoint_file}")
        except OSError as exc:
            print(f">> vaila/syncvid: Warning: checkpoint was not saved ({exc})")

        print(
            ">> vaila/syncvid: Synchronized ranges (same time window in every camera)\n  "
            + "\n  ".join(describe_sync_plan(entries, effective_infos, reference_video))
        )
        if self.fps_overrides:
            print(
                ">> vaila/syncvid: Real-FPS overrides used for time math\n  "
                + "\n  ".join(
                    f"{name}: {rate:.6g} fps" for name, rate in sorted(self.fps_overrides.items())
                )
            )
        reference_path = next(
            info.path for info in self.video_infos if info.name == reference_video
        )
        print(
            ">> vaila/syncvid: Equivalent interactive launcher\n"
            + shlex.join(
                [
                    sys.executable,
                    "-m",
                    "vaila.syncvid",
                    "--input-dir",
                    str(reference_path.parent),
                    "--output",
                    str(sync_file),
                ]
            )
        )
        self.result = SyncRunResult(
            sync_file=sync_file,
            reference_video=reference_path,
            entries=tuple(entries),
            open_cutvideo=open_cutvideo,
        )
        if open_cutvideo:
            self.running = False
        else:
            self._set_status(f"Saved sync: {sync_file.name} (GUI stays open)")

    def _repeat_progress(self, phase: str, done: int, total: int) -> None:
        label = "Analyzing" if phase == "analyze" else "Writing"
        self._set_status(f"{label} repeated frames: {done}/{total} ...")
        pygame.event.pump()
        self._draw()

    def _detect_repeated_frames(self) -> None:
        """Mark the automatically detected repeats of every video for review."""
        self.playing = False
        print(
            ">> vaila/syncvid: Equivalent CLI (detect and remove repeated frames)\n"
            + shlex.join(
                [
                    "uv",
                    "run",
                    "--no-sync",
                    "vaila/syncvid.py",
                    "--remove-repeated",
                    "-i",
                    str(self.current_info.path.parent),
                ]
            )
        )
        parts = []
        try:
            for info in self.video_infos:
                self._repeat_progress("analyze", 0, info.frame_count)
                report = find_repeated_frames(info.path, self._repeat_progress)
                self.repeat_reports[info.name] = report
                self.removed_frames[info.name] |= set(report.repeated_frames)
                print(
                    f"  {info.name}: auto={len(report.repeated_frames)} "
                    f"(motion_threshold={report.repeat_reasons.count('motion_threshold')}, "
                    f"field_alternation={report.repeat_reasons.count('field_alternation')}); "
                    f"current marks={len(self.removed_frames[info.name])}"
                )
                if report.repeated_frames:
                    parts.append(f"{info.name} {len(report.repeated_frames)}")
        except (OSError, SyncWorkflowError) as exc:
            self._set_status(f"Detect repeats failed: {exc}", error=True)
            return
        self._cached_video_surface = None
        if not parts:
            self._set_status(f"No repeated frames detected in {len(self.video_infos)} videos.")
            return
        self._set_status(
            f"Marked repeats: {'; '.join(parts)}. Review with D, then Save w/o repeats or Save sync."
        )

    def _save_without_repeats(self) -> None:
        """Write repeat-free copies of every video with marked frames."""
        self.playing = False
        if not any(self.removed_frames.values()):
            self._set_status("No frames marked. Use Detect repeats or D on a frame.", error=True)
            return
        try:
            out_dir = save_videos_without_frames(
                [
                    (info, self.removed_frames[info.name], self.repeat_reports.get(info.name))
                    for info in self.video_infos
                ],
                progress=self._repeat_progress,
            )
        except (OSError, RuntimeError, SyncWorkflowError) as exc:
            self._set_status(f"Save without repeats failed: {exc}", error=True)
            return
        parts = [
            f"{info.name} {info.frame_count}->{info.frame_count - len(marks)}"
            for info in self.video_infos
            if (marks := self.removed_frames[info.name])
        ]
        print(f">> vaila/syncvid: Repeat-free videos saved in {out_dir}")
        self._set_status(f"Saved {'; '.join(parts)} in {out_dir}")

    @staticmethod
    def _open_help() -> None:
        help_path = Path(__file__).resolve().parent / "help" / "syncvid.html"
        if help_path.is_file():
            _resilient_open_local_html(help_path)

    def run(self) -> SyncRunResult | None:
        """Run the player until Save + Cut Video, Cancel, or window close.

        Plain Save sync writes files and keeps the window open for more edits.
        """
        try:
            while self.running:
                for event in pygame.event.get():
                    self._handle_event(event)
                self._tick_nav_hold()
                self._advance_playback()
                self._draw()
                target_fps = max(1, round(self.current_info.fps)) if self.playing else 60
                self.clock.tick(target_fps)
            return self.result
        finally:
            self._release_capture()
            pygame.display.quit()
            pygame.font.quit()


def run_pygame_sync_player(
    video_infos: Sequence[VideoInfo],
    *,
    output_file: str | Path | None = None,
    dialog_parent: tk.Tk | tk.Toplevel | None = None,
) -> SyncRunResult | None:
    """Open the Pygame synchronization player."""
    player = PygameSyncPlayer(
        video_infos,
        output_file=output_file,
        dialog_parent=dialog_parent,
    )
    return player.run()


def sync_videos(
    input_directory: str | Path | None = None,
    output_file: str | Path | None = None,
    *,
    parent: tk.Tk | tk.Toplevel | None = None,
) -> SyncRunResult | None:
    """Open the interactive synchronizer and optionally hand off to Cut Video."""
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting interactive multi-video synchronization...")

    created_root = False
    default_root = cast(tk.Tk | tk.Toplevel | None, getattr(tk, "_default_root", None))
    root = parent or default_root
    if root is None:
        root = tk.Tk()
        root.withdraw()
        created_root = True

    try:
        if input_directory is None:
            selected = _portable_path_dialog(
                "dir",
                title="Select target directory containing synchronized videos",
            )
            if not selected:
                print("No video directory selected.")
                return None
            input_directory = selected
        try:
            video_infos = probe_video_directory(input_directory)
        except (OSError, SyncWorkflowError) as exc:
            messagebox.showerror("Video synchronization", str(exc), parent=root)
            return None

        try:
            result = run_pygame_sync_player(
                video_infos, output_file=output_file, dialog_parent=root
            )
        except (pygame.error, SyncWorkflowError) as exc:
            messagebox.showerror("Video synchronization", str(exc), parent=root)
            return None
    finally:
        if created_root:
            root.destroy()

    if result is not None:
        print(f"Synchronization file created: {result.sync_file}")
        if result.open_cutvideo:
            try:
                launch_cutvideo(result.reference_video, result.sync_file)
            except (OSError, SyncWorkflowError) as exc:
                if parent is not None:
                    messagebox.showerror("Cut Video handoff", str(exc), parent=parent)
                else:
                    print(f"[red]Cut Video handoff failed:[/] {exc}")
    return result


def build_dry_run_report(input_directory: str | Path) -> list[str]:
    """Probe real target videos without opening the interactive window."""
    infos = probe_video_directory(input_directory)
    lines = [
        ">> vaila/syncvid: Dry-run — videos decode successfully",
        f"directory={Path(input_directory).expanduser().resolve()}",
        f"videos={len(infos)}",
    ]
    for info in infos:
        lines.append(
            f"  {info.name}: {info.width}x{info.height}, {info.fps:.6f} fps, "
            f"{info.frame_count} frames"
        )
    return lines


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Visually synchronize every video in a target directory."
    )
    parser.add_argument("-i", "--input-dir", type=Path, help="Target directory containing videos")
    parser.add_argument("-o", "--output", type=Path, help="Destination sync TXT file")
    parser.add_argument(
        "--events",
        action="store_true",
        help="Build an event manifest from --checkpoint without opening a window",
    )
    parser.add_argument("--checkpoint", type=Path, help="Existing checkpoint for --events")
    parser.add_argument(
        "--reference-fps", default=None, help="Assumed reference clock, default 60000/1001"
    )
    parser.add_argument("--output-fps", default=None, help="Output grid, default 30000/1001")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Probe/decode all target videos and exit without opening the GUI",
    )
    parser.add_argument(
        "--remove-repeated",
        action="store_true",
        help=(
            "Find repeated (editor-duplicated) frames in every video of --input-dir "
            "(or one video file), write repeat-free copies to "
            "processed_norepeat_<timestamp>/ and exit without opening the GUI"
        ),
    )
    args = parser.parse_args(argv)

    if args.events:
        if args.checkpoint is None or args.input_dir is None or args.output is None:
            parser.error("--events requires --checkpoint, --input-dir and --output")
        try:
            state = read_checkpoint_file(args.checkpoint)
            infos = probe_video_directory(args.input_dir)
            state["reference"] = resolve_reference_video(state["reference"], infos)
            if state.get("sync_mode") == "events":
                state["events"] = {
                    name: parse_event_frames(text, name)
                    for name, text in state.get("event_texts", {}).items()
                }
            manifest = manifest_from_checkpoint(
                state,
                infos,
                reference_fps=args.reference_fps or state.get("reference_fps", "60000/1001"),
                output_fps=args.output_fps or state.get("output_fps", "30000/1001"),
                checkpoint_file=args.checkpoint.resolve(),
            )
            if args.output.resolve() == args.checkpoint.resolve():
                raise EventSyncError("Choose a manifest output separate from the checkpoint")
            write_checkpoint_file(manifest, args.output)
            print(">> vaila/syncvid: Event manifest saved: " + str(args.output))
            for name, camera in manifest["cameras"].items():
                print(
                    f">> {name}: events={camera['events']}; retained={camera['retained_count']}; "
                    + " / ".join(
                        f"{s['effective_fps_decimal']:.6f} fps (inferred)"
                        for s in camera["segments"]
                    )
                )
        except (OSError, SyncWorkflowError, EventSyncError) as exc:
            parser.error(str(exc))
        return 0

    if args.remove_repeated:
        if args.input_dir is None:
            parser.error("--remove-repeated requires --input-dir (directory or video file)")
        source = args.input_dir.expanduser().resolve()
        try:
            videos = [source] if source.is_file() else discover_video_files(source)
            if not videos:
                raise SyncWorkflowError(f"No supported videos found in {source}")
            out_dir, _reports = remove_repeated_frames(videos)
        except (OSError, RuntimeError, SyncWorkflowError) as exc:
            parser.error(str(exc))
        print(f"Saved in: {out_dir}")
        return 0

    if args.dry_run:
        if args.input_dir is None:
            parser.error("--dry-run requires --input-dir")
        try:
            for line in build_dry_run_report(args.input_dir):
                print(line)
        except SyncWorkflowError as exc:
            parser.error(str(exc))
        return 0

    sync_videos(args.input_dir, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
