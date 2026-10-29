"""Independent video trims and explicit frame removals, without synchronization.

Update Date: 28 October 2026
Version: 0.4.12

Public frame numbers are 1-based and inclusive. Output preserves the source FPS;
removing frames shortens playback rather than changing the playback clock.
"""

from __future__ import annotations

import csv
import json
import shlex
from collections.abc import Callable, Mapping, Sequence
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

try:
    from .sync_events import (
        EventExportCancelledError,
        EventSyncError,
        _encode_samples,
        rational_rate,
        source_clock,
    )
except ImportError:
    from sync_events import (  # ty: ignore[unresolved-import]
        EventExportCancelledError,
        EventSyncError,
        _encode_samples,
        rational_rate,
        source_clock,
    )


def selected_frames(camera: Mapping[str, Any]) -> list[int]:
    """Return kept source identities; marks outside the trim have no effect."""
    removed = set(camera["removed_frames"])
    return [f for f in range(camera["start"], camera["end"] + 1) if f not in removed]


def validate_manual_cuts(data: Any, video_infos: Sequence[Any]) -> dict[str, Any]:
    """Validate every selected camera before creating output; recompute all counts."""
    if (
        not isinstance(data, dict)
        or type(data.get("vaila_manual_cuts")) is not int
        or data["vaila_manual_cuts"] != 1
    ):
        raise EventSyncError("Expected a vaila_manual_cuts version 1 file")
    cameras = data.get("cameras")
    if not isinstance(cameras, dict) or not cameras:
        raise EventSyncError("Choose at least one video to save")
    if data.get("fps_policy", "source") != "source":
        raise EventSyncError("Manual cuts preserve each source FPS")
    available = {info.name: info for info in video_infos}
    result = {
        "vaila_manual_cuts": 1,
        "method": "manual_trim_and_remove",
        "fps_policy": "source",
        "frame_base": 1,
        "audio": False,
        "cameras": {},
    }
    for name, camera in cameras.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or any(c in name for c in '\\/\n\r\t:*?"<>|')
            or name not in available
            or not isinstance(camera, dict)
        ):
            raise EventSyncError(f"Invalid or missing source video: {name!r}")
        info = available[name]
        start, end = camera.get("start"), camera.get("end")
        if (
            any(type(v) is not int for v in (start, end))
            or not 1 <= start <= end <= info.frame_count
        ):
            raise EventSyncError(
                f"{name}: Start/End must satisfy 1 <= Start <= End <= {info.frame_count}"
            )
        removed = camera.get("removed_frames", [])
        if (
            not isinstance(removed, list)
            or any(type(f) is not int or not 1 <= f <= info.frame_count for f in removed)
            or len(set(removed)) != len(removed)
        ):
            raise EventSyncError(f"{name}: removed_frames must be unique 1-based source frames")
        if camera.get("frame_count", info.frame_count) != info.frame_count:
            raise EventSyncError(f"{name}: source frame count differs from saved cuts")
        fps = rational_rate(source_clock(info), f"{name} FPS")
        if (
            "source_fps" in camera
            and rational_rate(camera["source_fps"], f"{name} saved FPS") != fps
        ):
            raise EventSyncError(f"{name}: source FPS differs from saved cuts")
        entry = {
            "start": start,
            "end": end,
            "removed_frames": sorted(removed),
            "frame_count": info.frame_count,
            "source_fps": str(fps),
            "output_file": f"{name}_manual.mp4",
        }
        count = end - start + 1 - sum(start <= f <= end for f in removed)
        if not count:
            raise EventSyncError(f"{name}: removal marks leave the selected interval empty")
        entry["output_frame_count"] = count
        entry["playback_duration_seconds"] = float(Fraction(count) / fps)
        result["cameras"][name] = entry
    if len({name.casefold() for name in cameras}) != len(cameras):
        raise EventSyncError("Video names must be distinct ignoring case")
    return result


def build_manual_cuts(
    video_infos: Sequence[Any],
    ranges: Mapping[str, tuple[str, str]],
    removed_frames: Mapping[str, Sequence[int]],
) -> dict[str, Any]:
    """Build an export request from independent UI intervals (1-based)."""
    cameras = {}
    for info in video_infos:
        try:
            start, end = (int(text.strip()) for text in ranges[info.name])
        except (ValueError, KeyError) as exc:
            raise EventSyncError(f"{info.name}: enter integer Start and End frames") from exc
        cameras[info.name] = {
            "start": start,
            "end": end,
            "removed_frames": list(removed_frames.get(info.name, [])),
        }
    return validate_manual_cuts({"vaila_manual_cuts": 1, "cameras": cameras}, video_infos)


def read_manual_cuts(path: str | Path, video_infos: Sequence[Any]) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise EventSyncError("Manual cuts file exceeds 10 MB")
    try:
        return validate_manual_cuts(json.loads(path.read_text(encoding="utf-8-sig")), video_infos)
    except (ValueError, TypeError) as exc:
        raise EventSyncError(f"Invalid manual cuts {path}: {exc}") from exc


def export_manual_cuts(
    data: dict[str, Any],
    video_infos: Sequence[Any],
    output_dir: str | Path,
    *,
    progress_callback: Callable[[str, int, int], bool | None] | None = None,
) -> Path:
    """Save only the requested trims, at source FPS and with no audio or resampling."""
    manifest = validate_manual_cuts(data, video_infos)
    sources = {
        info.name: info.path.resolve(strict=True)
        for info in video_infos
        if info.name in manifest["cameras"]
    }
    destination = Path(output_dir).expanduser().resolve()
    if any(
        destination == src.parent or destination in src.parent.parents for src in sources.values()
    ):
        raise EventSyncError("Choose a new output folder, separate from the source directory")
    total = sum(c["output_frame_count"] for c in manifest["cameras"].values())
    done = 0

    def tick(name: str) -> None:
        if progress_callback is not None and progress_callback(name, done, total) is False:
            raise EventExportCancelledError(
                "Manual export cancelled; partial folder is marked INCOMPLETE"
            )

    def advance(name: str) -> None:
        nonlocal done
        done += 1
        if done % max(1, total // 10) == 0 or done == total:
            print(f">> Manual cuts: {done}/{total} frames ({name})", flush=True)
        tick(name)

    tick("Preparing manual cuts")
    for name, source in sources.items():
        cap = cv2.VideoCapture(str(source))
        try:
            camera = manifest["cameras"][name]
            if (
                not cap.isOpened()
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != camera["frame_count"]
                or abs(cap.get(cv2.CAP_PROP_FPS) - float(Fraction(camera["source_fps"]))) > 0.00001
            ):
                raise EventSyncError(
                    f"{name}: source changed since it was loaded; reopen the videos"
                )
        finally:
            cap.release()
    destination.mkdir(parents=True, exist_ok=False)
    marker = destination / "INCOMPLETE.txt"
    marker.write_text(
        "Manual export incomplete (running, cancelled or failed).\n", encoding="utf-8"
    )
    intent = destination / "manual_cuts.json"
    intent.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    print(
        ">> Equivalent CLI\n"
        + shlex.join(
            [
                "uv",
                "run",
                "--no-sync",
                "python",
                "-m",
                "vaila.syncvid",
                "--manual-cuts",
                "--input-dir",
                str(next(iter(sources.values())).parent),
                "--cuts-file",
                str(intent),
                "--output-dir",
                str(destination.with_name(destination.name + "_rerun")),
            ]
        ),
        flush=True,
    )
    for name, camera in manifest["cameras"].items():
        frames = selected_frames(camera)
        _encode_samples(
            sources[name],
            frames,
            destination / camera["output_file"],
            camera["source_fps"],
            tick=lambda name=name: tick(name),
            frame_done=lambda name=name: advance(name),
        )
        rate = Fraction(camera["source_fps"])
        with (destination / f"{name}_manual_frames.csv").open(
            "w", newline="", encoding="utf-8"
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(
                ["output_frame", "original_frame", "source_time_rational", "output_time_rational"]
            )
            writer.writerows(
                (i + 1, f, str(Fraction(f - 1) / rate), str(Fraction(i) / rate))
                for i, f in enumerate(frames)
            )
    marker.unlink()
    return destination
