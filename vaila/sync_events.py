"""Physical-time synchronization from reviewed events and retained samples.

Update Date: 27 October 2026
Version: 0.4.12

All public frame numbers are 1-based. Fractions preserve the reference clock,
piecewise calibration, output grid and earlier-sample tie breaking exactly.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
import json
import subprocess
import tempfile
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2

try:
    from .ffmpeg_utils import get_ffmpeg_path
except ImportError:
    from ffmpeg_utils import get_ffmpeg_path  # ty: ignore[unresolved-import]


class EventSyncError(ValueError):
    """An event, clock or source needs correction before exporting."""


METHOD = "piecewise_linear_retained_samples"
ASSUMPTION = "Uniform physical spacing of retained samples within each event segment."
LIMITATION = (
    "Output correspondence does not imply simultaneous acquisition. Effective rates are "
    "inferred, not measured camera acquisition rates. For 3D, measure 2D coordinates on "
    "retained samples, interpolate coordinates to shared times, then triangulate. "
    "Interpolation creates no new physical observations. Video exports contain no audio."
)


def rational_rate(value: Any, label: str) -> Fraction:
    try:
        rate = Fraction(str(value))
        if not 0 < rate < 10000:
            raise ValueError
        return rate
    except (ValueError, ZeroDivisionError) as exc:
        raise EventSyncError(f"{label}: enter a positive rational FPS, e.g. 60000/1001") from exc


def _frame_list(values: Any, name: str) -> list[int]:
    if not isinstance(values, (list, tuple)) or any(
        isinstance(v, bool) or not isinstance(v, int) or v < 1 for v in values
    ):
        raise EventSyncError(f"{name}: expected positive 1-based integer frames")
    return list(values)


def parse_event_frames(text: str, name: str) -> list[int]:
    try:
        return _frame_list([int(v) for v in text.replace(",", " ").split()], name)
    except ValueError as exc:
        raise EventSyncError(f"{name}: enter event frames separated by spaces or commas") from exc


@dataclass(frozen=True)
class EventTimeline:
    frames: tuple[int, ...]
    times: tuple[Fraction, ...]
    event_indices: tuple[int, ...]


def calibrate_camera(
    name: str, camera: Mapping[str, Any], event_times: Sequence[Fraction]
) -> EventTimeline:
    events = _frame_list(camera.get("events"), f"{name} events")
    removed = _frame_list(camera.get("removed_frames", []), f"{name} removed_frames")
    count = camera.get("frame_count")
    if isinstance(count, bool) or not isinstance(count, int) or count < 1:
        raise EventSyncError(f"{name}: invalid source frame_count")
    if len(events) != len(event_times) or len(events) < 2:
        raise EventSyncError(f"{name}: every camera needs the same ordered events (at least two)")
    for index, event in enumerate(events):
        if event > count:
            raise EventSyncError(f"{name}: event {index + 1}, frame {event}, exceeds {count}")
        if index and event <= events[index - 1]:
            raise EventSyncError(f"{name}: event {index + 1}, frame {event}, is out of order")
        if event in removed:
            raise EventSyncError(f"{name}: event {index + 1}, frame {event}, is marked for removal")
    if len(set(removed)) != len(removed) or any(v > count for v in removed):
        raise EventSyncError(f"{name}: removed_frames must be unique and within the source")
    excluded = set(removed)
    kept = tuple(f for f in range(events[0], events[-1] + 1) if f not in excluded)
    indices = tuple(bisect.bisect_left(kept, event) for event in events)
    times: list[Fraction] = []
    for segment, (lo, hi) in enumerate(zip(indices, indices[1:], strict=False)):
        begin, end = event_times[segment : segment + 2]
        times.extend(begin + (end - begin) * Fraction(i - lo, hi - lo) for i in range(lo, hi))
    times.append(event_times[-1])
    return EventTimeline(kept, tuple(times), indices)


def validate_manifest(data: Any) -> dict[str, Any]:
    """Validate inputs and regenerate derived audit fields, never trusting cached maps."""
    if (
        not isinstance(data, dict)
        or type(data.get("vaila_event_sync")) is not int
        or data.get("vaila_event_sync") != 1
    ):
        raise EventSyncError("Unsupported event synchronization manifest (expected version 1)")
    if data.get("method") != METHOD:
        raise EventSyncError("Unsupported event temporal method")
    ref_rate = rational_rate(data.get("reference_fps"), "Reference clock")
    out_rate = rational_rate(data.get("output_fps"), "Output grid")
    cameras = data.get("cameras")
    reference = data.get("reference")
    if (
        not isinstance(cameras, dict)
        or len(cameras) < 2
        or not isinstance(reference, str)
        or reference not in cameras
    ):
        raise EventSyncError("Select at least two cameras and a reference camera")
    for name, camera in cameras.items():
        if (
            not isinstance(name, str)
            or Path(name).name != name
            or name in {".", ".."}
            or any(c in name for c in '\\/:\n\r\t*?"<>|')
            or not isinstance(camera, dict)
        ):
            raise EventSyncError(f"Invalid camera filename or entry: {name!r}")
    if len({name.casefold() for name in cameras}) != len(cameras):
        raise EventSyncError("Camera filenames must be distinct ignoring case")
    reference_events = _frame_list(cameras[reference].get("events"), f"{reference} events")
    if len(reference_events) < 2:
        raise EventSyncError("Reference needs at least two events")
    event_times = tuple(Fraction(f - reference_events[0], 1) / ref_rate for f in reference_events)
    if any(b <= a for a, b in zip(event_times, event_times[1:], strict=False)):
        raise EventSyncError(f"{reference}: reference events are out of order")
    result = dict(data)
    result.update(reference_fps=str(ref_rate), output_fps=str(out_rate))
    result["event_times"] = [str(t) for t in event_times]
    result["assumption"] = ASSUMPTION
    result["limitation"] = LIMITATION
    result["frame_base"] = 1
    result["sample_span_seconds"] = float(event_times[-1])
    result["output_frame_count"] = int(event_times[-1] * out_rate) + 1
    result["playback_duration_seconds"] = float(
        Fraction(result["output_frame_count"], 1) / out_rate
    )
    result["cameras"] = {}
    for name, camera in cameras.items():
        timeline = calibrate_camera(name, camera, event_times)
        segments = []
        for j in range(len(event_times) - 1):
            intervals = timeline.event_indices[j + 1] - timeline.event_indices[j]
            rate = Fraction(intervals, 1) / (event_times[j + 1] - event_times[j])
            segments.append({"effective_fps": str(rate), "effective_fps_decimal": float(rate)})
        result["cameras"][name] = dict(
            camera, segments=segments, retained_count=len(timeline.frames)
        )
    return result


def manifest_from_checkpoint(
    checkpoint: Mapping[str, Any],
    video_infos: Sequence[Any],
    *,
    reference_fps: str = "60000/1001",
    output_fps: str = "30000/1001",
    checkpoint_file: str | Path | None = None,
) -> dict[str, Any]:
    """Import existing Start/Sync/End as three events without FPS range snapping."""
    reference = checkpoint["reference"]
    explicit = checkpoint.get("events", {})
    cameras = {}
    for info in video_infos:
        events = explicit.get(info.name)
        if events is None:
            bounds = (
                [checkpoint["start"], checkpoint["end"]]
                if info.name == reference
                else checkpoint.get("ranges", {}).get(info.name, ["", ""])
            )
            text = f"{bounds[0]} {checkpoint['sync_frames'].get(info.name, '')} {bounds[1]}"
            if not all(
                str(v).strip() for v in (*bounds, checkpoint["sync_frames"].get(info.name, ""))
            ):
                raise EventSyncError(f"{info.name}: enter explicit Start, Sync and End events")
            events = parse_event_frames(text, info.name)
        cameras[info.name] = {
            "events": events,
            "frame_count": info.frame_count,
            "removed_frames": checkpoint.get("removed_frames", {}).get(info.name, []),
            "declared_fps": info.fps,
        }
    return validate_manifest(
        {
            "vaila_event_sync": 1,
            "method": METHOD,
            "reference": reference,
            "reference_fps": reference_fps,
            "output_fps": output_fps,
            "checkpoint": str(checkpoint_file) if checkpoint_file else None,
            "cameras": cameras,
        }
    )


def read_event_manifest(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    if path.stat().st_size > 10_000_000:
        raise EventSyncError("Event manifest exceeds 10 MB")
    try:
        return validate_manifest(json.loads(path.read_text(encoding="utf-8")))
    except (ValueError, TypeError, KeyError) as exc:
        raise EventSyncError(f"Invalid event manifest {path}: {exc}") from exc


def temporal_maps(
    manifest: dict[str, Any], name: str
) -> tuple[EventTimeline, list[dict[str, Any]]]:
    times = tuple(Fraction(t) for t in manifest["event_times"])
    timeline = calibrate_camera(name, manifest["cameras"][name], times)
    rate = Fraction(manifest["output_fps"])
    rows = []
    used: set[int] = set()
    for i in range(manifest["output_frame_count"]):
        requested = Fraction(i, 1) / rate
        right = bisect.bisect_left(timeline.times, requested)
        candidates = range(max(0, right - 1), min(len(timeline.times), right + 1))
        selected = min(candidates, key=lambda k: (abs(timeline.times[k] - requested), k))
        frame, physical = timeline.frames[selected], timeline.times[selected]
        rows.append(
            {
                "output_frame": i + 1,
                "requested_time_s": float(requested),
                "original_frame": frame,
                "physical_time_s": float(physical),
                "temporal_error_s": float(physical - requested),
                "reused": int(frame in used),
                "requested_time_rational": str(requested),
                "physical_time_rational": str(physical),
                "temporal_error_rational": str(physical - requested),
            }
        )
        used.add(frame)
    return timeline, rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _encode_samples(source: Path, frames: Sequence[int], destination: Path, rate: str) -> None:
    """Sequential decoding preserves original frame identity; FFmpeg uses exact FPS.

    Lossless RGB H.264 keeps each decoded observation intact, including odd sizes.
    Stderr goes to a file to avoid pipe deadlocks. Failure never leaves a video.
    """
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    command = [
        get_ffmpeg_path(),
        "-hide_banner",
        "-loglevel",
        "error",
        "-nostdin",
        "-n",
        "-f",
        "rawvideo",
        "-pix_fmt",
        "bgr24",
        "-s",
        f"{width}x{height}",
        "-framerate",
        rate,
        "-i",
        "pipe:0",
        "-an",
        "-c:v",
        "libx264rgb",
        "-crf",
        "0",
        "-preset",
        "fast",
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            assert process.stdin is not None
            written, original = 0, 0
            try:
                while written < len(frames):
                    ok, frame = cap.read()
                    original += 1
                    if not ok:
                        raise EventSyncError(
                            f"{source.name}: cannot decode original frame {original}"
                        )
                    while written < len(frames) and frames[written] == original:
                        process.stdin.write(frame.tobytes())
                        written += 1
            finally:
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.close()
                process.wait()
            if process.returncode:
                errors.seek(0)
                raise EventSyncError(
                    f"FFmpeg failed: {errors.read().decode(errors='replace')[-1000:]}"
                )
    except BaseException:
        destination.unlink(missing_ok=True)
        raise
    finally:
        cap.release()


def export_event_videos(
    manifest: dict[str, Any],
    source_dir: str | Path,
    output_dir: str | Path,
) -> Path:
    """Export common-grid videos, retained observations, maps and a versioned audit."""
    manifest = validate_manifest(manifest)
    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    prepared = []
    for name, camera in manifest["cameras"].items():
        source = (source_dir / name).resolve(strict=True)
        if source.parent != source_dir:
            raise EventSyncError(f"{name}: source escapes input directory")
        cap = cv2.VideoCapture(str(source))
        try:
            if (
                not cap.isOpened()
                or int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) != camera["frame_count"]
            ):
                raise EventSyncError(f"{name}: source frame count differs from manifest")
        finally:
            cap.release()
        timeline, rows = temporal_maps(manifest, name)
        # Including the extension avoids collisions for cameras sharing a stem.
        prepared.append((name, source, timeline, rows))
    # No output is created until every camera and event is validated. Never overwrite.
    if output_dir == source_dir or output_dir in source_dir.parents:
        raise EventSyncError("Output directory must be separate from the input directory")
    output_dir.mkdir(parents=True, exist_ok=False)
    manifest["exports"] = {}
    for name, source, timeline, rows in prepared:
        output_name = f"{name}_events.mp4"
        _encode_samples(
            source,
            [r["original_frame"] for r in rows],
            output_dir / output_name,
            manifest["output_fps"],
        )
        _write_csv(output_dir / f"{name}_time_map.csv", rows)
        retained_dir = output_dir / f"{name}_retained"
        retained_dir.mkdir()
        # PNGs preserve samples without imposing a misleading constant playback clock.
        cap = cv2.VideoCapture(str(source))
        retained_rows = []
        wanted = set(timeline.frames)
        ordinal = 0
        try:
            for original in range(1, timeline.frames[-1] + 1):
                ok, frame = cap.read()
                if not ok:
                    raise EventSyncError(f"{name}: cannot decode original frame {original}")
                if original not in wanted:
                    continue
                filename = f"sample_{ordinal + 1:06d}.png"
                if not cv2.imwrite(str(retained_dir / filename), frame):
                    raise OSError(f"Could not write {retained_dir / filename}")
                physical = timeline.times[ordinal]
                retained_rows.append(
                    {
                        "retained_frame": ordinal + 1,
                        "original_frame": original,
                        "physical_time_s": float(physical),
                        "physical_time_rational": str(physical),
                        "image": filename,
                    }
                )
                ordinal += 1
        finally:
            cap.release()
        _write_csv(output_dir / f"{name}_retained_times.csv", retained_rows)
        max_error = max(abs(r["temporal_error_s"]) for r in rows)
        manifest["exports"][name] = {"video": output_name, "max_absolute_error_s": max_error}
        print(
            f">> {name}: {len(rows)} output frames; {len(timeline.frames)} retained; "
            f"maximum temporal error {max_error:.9f} s"
        )
    (output_dir / "event_sync_manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output_dir / "README.txt").write_text(
        ASSUMPTION + "\n" + LIMITATION + "\n"
        "Physical time is authoritative in *_retained_times.csv; PNG sequences have no playback FPS.\n"
        "Signed temporal error = selected physical time - requested output time.\n"
        f"Sample span: {manifest['sample_span_seconds']:.9f} s. "
        f"Playback duration: {manifest['playback_duration_seconds']:.9f} s.\n",
        encoding="utf-8",
    )
    return output_dir
