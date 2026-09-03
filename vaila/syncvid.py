"""
Project: vailá Multimodal Toolbox
Script: syncvid.py - Interactive Multi-Video Synchronization

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 02 September 2026
Version: 0.3.119

Description:
Create a frame-accurate synchronization plan with a fast Pygame player. Every
video in the target directory can be reviewed with the same playback/navigation
keys used by ``vaila.cutvideo``. Synchronization frames, reference video, and
reference start/end are entered directly in editable on-screen fields.

The versioned TSV sync file is consumed directly by ``vaila.cutvideo``. The
``Save + Open Cut Video`` button launches Cut Video with both the reference
video and the generated sync file already selected.

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
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import tkinter as tk
import webbrowser
from collections.abc import Iterator, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from tkinter import filedialog, messagebox
from typing import Any, cast

import cv2
import pygame
from rich import print


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


@dataclass(frozen=True)
class SyncPlanEntry:
    """One synchronized video range, stored internally as zero-based frames."""

    video_file: str
    output_file: str
    start_frame: int
    end_frame: int
    sync_frame: int | None = None


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


def common_reference_bounds(
    video_infos: Sequence[VideoInfo],
    sync_frames: Mapping[str, int],
    reference_video: str,
) -> tuple[int, int]:
    """Return the largest reference-camera range present in every camera."""
    info_by_name = {info.name: info for info in video_infos}
    if reference_video not in info_by_name:
        raise SyncWorkflowError(
            f"Reference video is not in the target directory: {reference_video}"
        )
    missing = [name for name in info_by_name if name not in sync_frames]
    if missing:
        raise SyncWorkflowError("Mark a sync frame for every video: " + ", ".join(missing))

    min_relative = -MAX_FRAME_NUMBER
    max_relative = MAX_FRAME_NUMBER
    for name, info in info_by_name.items():
        sync_frame = int(sync_frames[name])
        if not 0 <= sync_frame < info.frame_count:
            raise SyncWorkflowError(
                f"Sync frame {sync_frame + 1} is outside {name} (1–{info.frame_count})"
            )
        min_relative = max(min_relative, -sync_frame)
        max_relative = min(max_relative, info.frame_count - 1 - sync_frame)

    reference_sync = int(sync_frames[reference_video])
    start = reference_sync + min_relative
    end = reference_sync + max_relative
    if start > end:
        raise SyncWorkflowError("The selected sync frames have no common video interval")
    return start, end


def _safe_output_name(video_name: str, sync_frame: int, start_frame: int, end_frame: int) -> str:
    stem = re.sub(r"[^\w.-]+", "_", Path(video_name).stem, flags=re.UNICODE).strip("._")
    stem = (stem or "video")[:120]
    output_name = f"{stem}_sync_{sync_frame + 1}_frames_{start_frame + 1}_to_{end_frame + 1}.mp4"
    return validate_sync_leaf_filename(output_name, field="output_file")


def build_sync_plan(
    video_infos: Sequence[VideoInfo],
    sync_frames: Mapping[str, int],
    reference_video: str,
    reference_start: int,
    reference_end: int,
) -> list[SyncPlanEntry]:
    """Build equal-duration synchronized ranges for all videos."""
    if reference_start > reference_end:
        raise SyncWorkflowError("Reference start frame must not be after the end frame")
    lower, upper = common_reference_bounds(video_infos, sync_frames, reference_video)
    if reference_start < lower or reference_end > upper:
        raise SyncWorkflowError(
            f"Reference range is outside the common interval: choose frames {lower + 1}–{upper + 1}"
        )

    reference_sync = int(sync_frames[reference_video])
    start_offset = int(reference_start) - reference_sync
    end_offset = int(reference_end) - reference_sync
    entries: list[SyncPlanEntry] = []
    for info in video_infos:
        sync_frame = int(sync_frames[info.name])
        start_frame = sync_frame + start_offset
        end_frame = sync_frame + end_offset
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
                sync_frame=sync_frame,
            )
        )
    return entries


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
) -> tuple[str, list[SyncPlanEntry]]:
    """Validate the Pygame text fields and build the regular synchronization plan."""
    reference_video = resolve_reference_video(reference_text, video_infos)
    info_by_name = {info.name: info for info in video_infos}
    sync_frames = {
        info.name: parse_frame_field(
            sync_frame_texts.get(info.name, ""),
            field=f"sync frame for {info.name}",
            frame_count=info.frame_count,
        )
        for info in video_infos
    }
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
    )


def _sync_file_text(entries: Sequence[SyncPlanEntry]) -> str:
    if not entries:
        raise SyncFileError("Cannot write an empty synchronization plan")
    stream = io.StringIO(newline="")
    stream.write(f"{SYNC_FORMAT_HEADER}\n")
    stream.write("# frame_base=1; frame ranges are inclusive\n")
    writer = csv.writer(stream, delimiter="\t", lineterminator="\n", quoting=csv.QUOTE_MINIMAL)
    writer.writerow(SYNC_COLUMNS)
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
        writer.writerow(
            (
                video_file,
                output_file,
                entry.start_frame + 1,
                entry.end_frame + 1,
                "" if entry.sync_frame is None else entry.sync_frame + 1,
            )
        )
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


def _parse_sync_row(parts: Sequence[str], line_number: int) -> SyncPlanEntry:
    if len(parts) not in {4, 5}:
        raise SyncFileError(f"Line {line_number}: expected 4 or 5 fields, received {len(parts)}")
    video_file = validate_sync_leaf_filename(parts[0], field=f"line {line_number} video_file")
    output_file = validate_sync_leaf_filename(parts[1], field=f"line {line_number} output_file")
    try:
        start_1based = int(parts[2])
        end_1based = int(parts[3])
        sync_1based = int(parts[4]) if len(parts) == 5 and parts[4].strip() else None
    except ValueError as exc:
        raise SyncFileError(f"Line {line_number}: frame fields must be integers") from exc
    if not 1 <= start_1based <= end_1based <= MAX_FRAME_NUMBER:
        raise SyncFileError(f"Line {line_number}: expected 1 <= start_frame <= end_frame")
    if sync_1based is not None and not 1 <= sync_1based <= MAX_FRAME_NUMBER:
        raise SyncFileError(f"Line {line_number}: sync_frame must be positive")
    return SyncPlanEntry(
        video_file=video_file,
        output_file=output_file,
        start_frame=start_1based - 1,
        end_frame=end_1based - 1,
        sync_frame=None if sync_1based is None else sync_1based - 1,
    )


def read_sync_file(sync_file: str | Path) -> list[SyncPlanEntry]:
    """Read v2 TSV or legacy whitespace sync files with strict validation."""
    path = Path(sync_file).expanduser().resolve()
    if not path.is_file():
        raise SyncFileError(f"Sync file not found: {path}")
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
) -> dict[str, Any]:
    """Build the JSON-serializable "Load / Resume" checkpoint for the current UI state."""
    return {
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
) -> dict[str, dict[str, int | str | None]]:
    """Convert validated entries to cutvideo's batch mapping."""
    return {
        entry.video_file: {
            "new_name": entry.output_file,
            "initial_frame": entry.start_frame,
            "final_frame": entry.end_frame,
            "sync_frame": entry.sync_frame,
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
    CONTROL_HEIGHT = 146
    ROW_HEIGHT = 40
    # Reserved height (px) at the bottom of the side panel for the status
    # line, the three action-button rows, and the footer hint.
    ACTIONS_AREA_HEIGHT = 179

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
        self._cached_video_key: tuple[int, int, int] | None = None

        pygame.display.set_caption(
            "vailá Sync | Space Play/Pause | Left/Right 1 frame | "
            "Up/Down or -/+ 60 frames | [/] Speed | PgUp/PgDn Video"
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
        info = self.current_info
        self.capture = cv2.VideoCapture(str(info.path))
        if not self.capture.isOpened():
            raise SyncWorkflowError(f"OpenCV could not open video: {info.path}")
        wanted = self.positions[info.name] if target_frame is None else target_frame
        if not self._decode_at(wanted):
            raise SyncWorkflowError(f"OpenCV could not decode video: {info.path}")
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
        self._decode_at(target)

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

    def _all_fields(self) -> list[PygameTextField]:
        return [
            self.reference_field,
            self.start_field,
            self.end_field,
            *(self.sync_fields[info.name] for info in self.video_infos),
        ]

    def _set_active_field(self, field: PygameTextField | None) -> None:
        for candidate in self._all_fields():
            candidate.deactivate()
        self.active_field = field
        if field is None:
            return
        self.playing = False
        field.activate()
        if field.key.startswith("sync:"):
            video_name = field.key.removeprefix("sync:")
            index = next(i for i, info in enumerate(self.video_infos) if info.name == video_name)
            if index != self.current_index:
                self._open_video(index)
            self._ensure_row_visible(index)

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
            elif field.key.startswith("sync:"):
                video_name = field.key.removeprefix("sync:")
                info = next(item for item in self.video_infos if item.name == video_name)
                value = parse_frame_field(
                    field.text,
                    field=f"sync frame for {video_name}",
                    frame_count=info.frame_count,
                )
                index = self.video_infos.index(info)
                if index != self.current_index:
                    self._open_video(index, target_frame=value)
                else:
                    self._decode_at(value)
                self._set_status(f"Sync frame for {video_name}: {value + 1}")
            self._set_active_field(None)
        except SyncWorkflowError as exc:
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
        key = (self.current_frame, viewport.width, viewport.height)
        if self._cached_video_surface is not None and self._cached_video_key == key:
            return self._cached_video_surface
        height, width = self.frame.shape[:2]
        scale = min(viewport.width / width, viewport.height / height)
        scaled_width = max(1, int(width * scale))
        scaled_height = max(1, int(height * scale))
        resized = cv2.resize(
            self.frame,
            (scaled_width, scaled_height),
            interpolation=cv2.INTER_AREA if scale < 1.0 else cv2.INTER_LINEAR,
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
            destination = video_surface.get_rect(center=viewport.center)
            self.screen.blit(video_surface, destination)

        overlay = pygame.Surface((viewport.width, 36), pygame.SRCALPHA)
        overlay.fill((10, 12, 16, 205))
        self.screen.blit(overlay, viewport.topleft)
        info_label = self.small_font.render(
            f"{self.current_index + 1}/{len(self.video_infos)}  {self.current_info.name}",
            True,
            (245, 247, 250),
        )
        self.screen.blit(info_label, (12, 9))
        speed_label = self.small_font.render(
            f"Speed {format_playback_speed(self.playback_speed)}x",
            True,
            (245, 247, 250),
        )
        self.screen.blit(speed_label, (viewport.right - speed_label.get_width() - 12, 9))

        controls = pygame.Rect(
            0,
            viewport.bottom,
            viewport.width,
            self.window_height - viewport.bottom,
        )
        pygame.draw.rect(self.screen, (28, 32, 38), controls)
        seconds = self.current_frame / self.current_info.fps
        duration = (self.current_info.frame_count - 1) / self.current_info.fps
        frame_text = self.font.render(
            f"Frame {self.current_frame + 1:,}/{self.current_info.frame_count:,}"
            f"    {seconds:.6f}s / {duration:.6f}s",
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

        hint = self.tiny_font.render(
            "Space play/pause | Left/Right 1 | Up/Down or -/+ 60 | [/] speed | timeline seek",
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
            self.small_font.render("Reference start", True, (198, 208, 218)),
            (x, 109),
        )
        self.screen.blit(
            self.small_font.render("Reference end", True, (198, 208, 218)),
            (x + half + 10, 109),
        )
        self.start_field.set_rect(pygame.Rect(x, 130, half, 31))
        self.end_field.set_rect(pygame.Rect(x + half + 10, 130, half, 31))
        self.start_field.draw(self.screen, self.small_font)
        self.end_field.draw(self.screen, self.small_font)

        pygame.draw.line(
            self.screen,
            (82, 91, 102),
            (x, 178),
            (panel.right - 14, 178),
            1,
        )
        self.screen.blit(
            self.font.render("Sync frame per video (1-based)", True, (238, 242, 247)),
            (x, 188),
        )
        self.screen.blit(
            self.tiny_font.render(
                "Click a camera name to open it; click its field to type.",
                True,
                (168, 182, 196),
            ),
            (x, 214),
        )

        self._clamp_list_scroll()
        visible_count = self._visible_row_count()
        start = self.list_scroll
        end = min(len(self.video_infos), start + visible_count)
        self.video_row_rects.clear()
        self.visible_fields = [
            self.reference_field,
            self.start_field,
            self.end_field,
        ]
        for field in self.sync_fields.values():
            field.set_rect(pygame.Rect(-1, -1, 0, 0))

        try:
            reference_name = resolve_reference_video(self.reference_field.text, self.video_infos)
        except SyncWorkflowError:
            reference_name = None

        list_y = 246
        sync_width = 92
        name_width = content_width - sync_width - 10
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
            label = self._fit_label(prefix + info.name, self.tiny_font, name_width - 10)
            self.screen.blit(
                self.tiny_font.render(label, True, (235, 239, 243)),
                (row_rect.x + 6, row_rect.y + 11),
            )
            field = self.sync_fields[info.name]
            field.set_rect(
                pygame.Rect(
                    row_rect.right - sync_width - 4,
                    row_rect.y + 4,
                    sync_width,
                    row_rect.height - 8,
                )
            )
            field.draw(self.screen, self.small_font)
            self.visible_fields.append(field)

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
        self.button_rects["side:save"] = save_rect
        self.button_rects["side:save_cutvideo"] = save_cut_rect
        self.button_rects["side:help"] = help_rect
        self.button_rects["side:cancel"] = cancel_rect
        self.button_rects["side:load"] = load_rect
        self._draw_button(save_rect, "Save sync", accent=True)
        self._draw_button(save_cut_rect, "Save + Cut Video", accent=True)
        self._draw_button(help_rect, "Help")
        self._draw_button(cancel_rect, "Cancel")
        self._draw_button(load_rect, "Load / Resume")

        footer = self.tiny_font.render(
            "Tab fields | Enter validate | Ctrl+S save | Esc cancel",
            True,
            (164, 177, 190),
        )
        self.screen.blit(footer, (x, actions_top + 127))

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
        if action.startswith("step:"):
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
        elif event.key == pygame.K_RIGHT and not self.playing:
            self._seek_relative(1)
        elif event.key == pygame.K_LEFT and not self.playing:
            self._seek_relative(-1)
        elif (
            event.key
            in (
                pygame.K_DOWN,
                pygame.K_MINUS,
                pygame.K_KP_MINUS,
            )
            and not self.playing
        ):
            self._seek_relative(-60)
        elif (
            event.key
            in (
                pygame.K_UP,
                pygame.K_EQUALS,
                pygame.K_KP_PLUS,
                getattr(pygame, "K_PLUS", pygame.K_EQUALS),
            )
            and not self.playing
        ):
            self._seek_relative(60)
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
        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            self.dragging_timeline = False
        elif event.type == pygame.MOUSEMOTION and self.dragging_timeline:
            self._decode_at(self._timeline_frame_from_x(event.pos[0]))
        elif event.type == pygame.MOUSEWHEEL:
            mouse_x, _mouse_y = pygame.mouse.get_pos()
            if mouse_x >= self.window_width - self._panel_width():
                self.list_scroll -= int(event.y)
                self._clamp_list_scroll()

    @contextlib.contextmanager
    def _sdl_window_minimized_for_dialog(self) -> Iterator[None]:
        """Iconify the SDL window for the duration of a native Tk dialog.

        Save and Load/Resume are the only places syncvid opens a native Tk
        dialog while its own SDL window is still alive (Help and the Cut
        Video handoff both fire-and-forget after the pygame loop has already
        exited). Tk's ask*filename() blocks inside its own nested event
        loop, so this process stops pumping the SDL window's queue for as
        long as the dialog is open. On X11, a window that stops answering
        the window manager while a second toolkit's top-level dialog grabs
        focus is a known trigger for the whole desktop appearing to hang,
        especially across multiple monitors. Iconifying the SDL window
        before the dialog opens -- and restoring it via the same set_mode()
        call already used for resize handling -- keeps only one top-level
        window fighting for the window manager's attention at a time.
        """
        with contextlib.suppress(pygame.error):
            pygame.display.iconify()
        try:
            yield
        finally:
            with contextlib.suppress(pygame.error):
                self.screen = pygame.display.set_mode(
                    (self.window_width, self.window_height), pygame.RESIZABLE
                )

    def _choose_output_file(self) -> Path | None:
        if self.output_file is not None:
            return self.output_file
        timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
        with self._sdl_window_minimized_for_dialog():
            selected = filedialog.asksaveasfilename(
                parent=self.dialog_parent,
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
        return build_checkpoint_state(
            reference=self.reference_field.text,
            start=self.start_field.text,
            end=self.end_field.text,
            sync_frames={name: field.text for name, field in self.sync_fields.items()},
            current_video=self.current_info.name,
            current_frame=self.current_frame,
            output_file=self.output_file,
            completed=completed,
        )

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

        self.reference_field.text = cast(str, data["reference"])
        self.start_field.text = cast(str, data["start"])
        self.end_field.text = cast(str, data["end"])

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
        self._set_status(message)

    def _load_checkpoint(self) -> None:
        with self._sdl_window_minimized_for_dialog():
            selected = filedialog.askopenfilename(
                parent=self.dialog_parent,
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
        except (SyncWorkflowError, OSError) as exc:
            self._set_status(f"Could not load checkpoint: {exc}", error=True)

    def _save(self, open_cutvideo: bool) -> None:
        try:
            reference_video, entries = build_sync_plan_from_fields(
                self.video_infos,
                {name: field.text for name, field in self.sync_fields.items()},
                self.reference_field.text,
                self.start_field.text,
                self.end_field.text,
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
        self.running = False

    @staticmethod
    def _open_help() -> None:
        help_path = Path(__file__).resolve().parent / "help" / "syncvid.html"
        if help_path.is_file():
            _resilient_open_local_html(help_path)

    def run(self) -> SyncRunResult | None:
        """Run the player until Save or Cancel."""
        try:
            while self.running:
                for event in pygame.event.get():
                    self._handle_event(event)
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
            selected = filedialog.askdirectory(
                parent=root, title="Select target directory containing synchronized videos"
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
        "--dry-run",
        action="store_true",
        help="Probe/decode all target videos and exit without opening the GUI",
    )
    args = parser.parse_args(argv)

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
