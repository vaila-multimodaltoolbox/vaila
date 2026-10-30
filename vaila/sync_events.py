"""Physical-time synchronization from reviewed events and retained samples.

Update Date: 29 October 2026
Version: 0.4.12

All public frame numbers are 1-based. Fractions preserve the reference clock,
piecewise calibration, output grid and earlier-sample tie breaking exactly.
Optional Farnebäck interpolation preserves every original on the minimum exact
grid, with a configurable FPS ceiling (240 by default). Lossless RGB encoding
preserves decoded observations.
"""

from __future__ import annotations

import bisect
import contextlib
import csv
import json
import math
import subprocess
import tempfile
from collections.abc import Callable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from .ffmpeg_utils import get_ffmpeg_path
except ImportError:
    from ffmpeg_utils import get_ffmpeg_path  # ty: ignore[unresolved-import]


class EventSyncError(ValueError):
    """An event, clock or source needs correction before exporting."""


class EventExportCancelledError(EventSyncError):
    """Export stopped; any partial outputs are explicitly marked incomplete."""


METHOD = "piecewise_linear_retained_samples"
INTERPOLATION = "farneback_bidirectional"
AUTO_FPS = "auto_preserve_originals"
MAX_FPS = Fraction(240)
INTERPOLATION_WARNING = (
    "Interpolated images are estimates: ball, feet and occlusions may show artifacts. "
    "They do not recover missing physical measurements."
)
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
    max_rate = rational_rate(data.get("max_output_fps", str(MAX_FPS)), "Max output FPS")
    interpolation = data.get("interpolation_method", "nearest")
    if interpolation not in ("nearest", INTERPOLATION):
        raise EventSyncError("Unsupported interpolation method")
    preserving = interpolation == INTERPOLATION
    policy = data.get("fps_policy", AUTO_FPS if preserving else "manual")
    if policy != (AUTO_FPS if preserving else "manual"):
        raise EventSyncError("Interpolation requires auto_preserve_originals FPS policy")
    out_rate = None if preserving else rational_rate(data.get("output_fps"), "Output grid")
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
    if preserving and len(reference_events) < 3:
        raise EventSyncError("Interpolation needs at least Start, Sync and End events")
    event_times = tuple(Fraction(f - reference_events[0], 1) / ref_rate for f in reference_events)
    if any(b <= a for a, b in zip(event_times, event_times[1:], strict=False)):
        raise EventSyncError(f"{reference}: reference events are out of order")
    result = dict(data)
    result.pop("event_output_frames", None)
    result.pop("interpolation_warning", None)
    result.update(
        reference_fps=str(ref_rate),
        max_output_fps=str(max_rate),
        interpolation_method=interpolation,
        fps_policy=policy,
    )
    result["event_times"] = [str(t) for t in event_times]
    result["assumption"] = ASSUMPTION
    result["limitation"] = LIMITATION
    result["frame_base"] = 1
    result["sample_span_seconds"] = float(event_times[-1])
    result["cameras"] = {}
    time_gcd = Fraction(0)
    for name, camera in cameras.items():
        timeline = calibrate_camera(name, camera, event_times)
        if preserving:
            if any(
                timeline.frames[0] <= f <= timeline.frames[-1]
                for f in camera.get("removed_frames", [])
            ):
                raise EventSyncError(
                    f"{name}: removal marks inside the event interval; clear them or revise events"
                )
            for time in timeline.times:
                time_gcd = Fraction(
                    math.gcd(time_gcd.numerator, time.numerator),
                    math.lcm(time_gcd.denominator, time.denominator),
                )
        segments = []
        for j in range(len(event_times) - 1):
            intervals = timeline.event_indices[j + 1] - timeline.event_indices[j]
            rate = Fraction(intervals, 1) / (event_times[j + 1] - event_times[j])
            segments.append({"effective_fps": str(rate), "effective_fps_decimal": float(rate)})
        result["cameras"][name] = dict(
            camera, segments=segments, retained_count=len(timeline.frames)
        )
    if preserving:
        out_rate = 1 / time_gcd
        if out_rate > max_rate:
            raise EventSyncError(
                f"Preserving all originals requires {out_rate} FPS ({float(out_rate):.6f}), "
                f"above {max_rate} FPS; increase Max output FPS (--max-output-fps) "
                "to authorize this grid. Check event correspondences; keep the recorded clock "
                "unless a different clock is explicitly intended."
            )
        result["interpolation_warning"] = INTERPOLATION_WARNING
        result["event_output_frames"] = [int(t * out_rate) + 1 for t in event_times]
    assert out_rate is not None
    result["output_fps"] = str(out_rate)
    result["output_frame_count"] = int(event_times[-1] * out_rate) + 1
    for camera in result["cameras"].values():
        # Regenerate counts just like times; never trust imported preview values.
        camera.pop("original_output_count", None)
        camera.pop("interpolated_output_count", None)
        if preserving:
            camera["original_output_count"] = camera["retained_count"]
            camera["interpolated_output_count"] = (
                result["output_frame_count"] - camera["retained_count"]
            )
    result["playback_duration_seconds"] = float(
        Fraction(result["output_frame_count"], 1) / out_rate
    )
    return result


def source_clock(info: Any) -> str:
    """Recover the rational container rate from the probe's floating representation."""
    return str(Fraction(str(info.fps)).limit_denominator(100_000))


def clock_needs_confirmation(checkpoint: Mapping[str, Any], info: Any) -> bool:
    stored = checkpoint.get("reference_fps") or checkpoint.get("fps_overrides", {}).get(info.name)
    if stored is None:
        return False
    current = rational_rate(source_clock(info), "Source clock")
    if abs(float(rational_rate(stored, "Checkpoint clock") / current) - 1) < 0.00001:
        return False
    return not (
        checkpoint.get("reference_clock_confirmed") is True
        and checkpoint.get("reference_source_fps") == str(current)
    )


def manifest_from_checkpoint(
    checkpoint: Mapping[str, Any],
    video_infos: Sequence[Any],
    *,
    reference_fps: str | None = None,
    output_fps: str = "30000/1001",
    interpolation_method: str | None = None,
    max_output_fps: str | None = None,
    checkpoint_file: str | Path | None = None,
) -> dict[str, Any]:
    """Import existing Start/Sync/End as three events without FPS range snapping."""
    reference = checkpoint["reference"]
    interpolation_method = interpolation_method or checkpoint.get("interpolation_method", "nearest")
    ref_info = next((info for info in video_infos if info.name == reference), None)
    if ref_info is None:
        raise EventSyncError("Reference camera is not in the input directory")
    if interpolation_method == INTERPOLATION and reference_fps is None:
        if clock_needs_confirmation(checkpoint, ref_info):
            raise EventSyncError(
                f"Confirm reference clock: checkpoint records {checkpoint.get('reference_fps', checkpoint.get('fps_overrides', {}).get(reference))}, "
                f"current file is {source_clock(ref_info)} FPS. Pass --reference-fps explicitly."
            )
        reference_fps = checkpoint.get("reference_fps", source_clock(ref_info))
    reference_fps = reference_fps or checkpoint.get("reference_fps", "60000/1001")
    explicit = checkpoint.get("events")
    if explicit is None:
        explicit = (
            {
                name: parse_event_frames(text, name)
                for name, text in checkpoint.get("event_texts", {}).items()
            }
            if checkpoint.get("sync_mode") == "events"
            else {}
        )
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
            "reference_source_fps": source_clock(ref_info),
            "reference_clock_confirmed": True,
            "output_fps": output_fps,
            "max_output_fps": (
                max_output_fps
                if max_output_fps is not None
                else checkpoint.get("max_output_fps", str(MAX_FPS))
            ),
            "interpolation_method": interpolation_method,
            "fps_policy": AUTO_FPS if interpolation_method == INTERPOLATION else "manual",
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


def event_summary(manifest: Mapping[str, Any]) -> list[str]:
    """Describe the actual grid and each camera before saving or rendering."""
    preserving = manifest["interpolation_method"] == INTERPOLATION
    mode = "Events: interpolation" if preserving else "Events: nearest image"
    rate = Fraction(manifest["output_fps"])
    lines = [
        f"Mode: {mode}; {manifest['output_frame_count']} frames per camera at "
        f"{rate} FPS ({float(rate):.6f}); Max output FPS: {manifest['max_output_fps']}"
    ]
    for name, camera in manifest["cameras"].items():
        detail = (
            f"{camera['original_output_count']} originals + "
            f"{camera['interpolated_output_count']} interpolated"
            if preserving
            else f"{manifest['output_frame_count']} nearest source images; 0 interpolated; "
            f"{camera['retained_count']} retained samples"
        )
        lines.append(f"{name}: {detail}")
    return lines


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
        if manifest.get("interpolation_method") == INTERPOLATION:
            if right >= len(timeline.times) or requested < timeline.times[0]:
                raise EventSyncError("Output grid would extrapolate beyond events")
            exact = timeline.times[right] == requested
            left = right if exact else right - 1
            alpha = (
                Fraction(0)
                if exact
                else (
                    (requested - timeline.times[left])
                    / (timeline.times[right] - timeline.times[left])
                )
            )
            rows.append(
                {
                    "output_frame": i + 1,
                    "requested_time_s": float(requested),
                    "requested_time_rational": str(requested),
                    "type": "original" if exact else "interpolated",
                    "original_frame": timeline.frames[right] if exact else "",
                    "source_frame_left": timeline.frames[left],
                    "source_frame_right": timeline.frames[right],
                    "interpolation_fraction": str(alpha),
                    "physical_time_s": float(requested),
                    "physical_time_rational": str(requested),
                    "temporal_error_s": 0.0,
                    "temporal_error_rational": "0",
                    "reused": 0,
                }
            )
            continue
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


def bidirectional_flow(left: Any, right: Any) -> tuple[Any, Any]:
    """Dense forward/backward Farnebäck motion, cached for each neighboring pair."""
    gray_left = cv2.cvtColor(left, cv2.COLOR_BGR2GRAY)
    gray_right = cv2.cvtColor(right, cv2.COLOR_BGR2GRAY)

    def calculate(previous: Any, following: Any) -> Any:
        flow = np.empty((*previous.shape, 2), dtype=np.float32)
        return cv2.calcOpticalFlowFarneback(previous, following, flow, 0.5, 4, 25, 5, 7, 1.5, 0)

    return (
        calculate(gray_left, gray_right),
        calculate(gray_right, gray_left),
    )


def interpolate_image(left: Any, right: Any, alpha: Fraction, flows: tuple[Any, Any]) -> Any:
    """Inverse-warp both neighbors toward the requested time and blend in BGR.

    Fixed-point inverse coordinates account for spatially varying motion. Border
    replication handles image edges; visibility/occlusion is still an estimate.
    """
    if alpha == 0:
        return left
    if alpha == 1:
        return right
    weight = float(alpha)
    y, x = np.mgrid[: left.shape[0], : left.shape[1]].astype(np.float32)

    def warp(frame: Any, flow: Any, fraction: float) -> Any:
        mx, my = x.copy(), y.copy()
        for _ in range(3):
            motion = cv2.remap(flow, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
            mx, my = x - fraction * motion[..., 0], y - fraction * motion[..., 1]
        return cv2.remap(frame, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)

    return cv2.addWeighted(
        warp(left, flows[0], weight), 1 - weight, warp(right, flows[1], 1 - weight), weight, 0
    )


def _render_images(
    cap: Any, source: Path, rows: Sequence[Mapping[str, Any]], tick: Callable[[], None]
) -> Iterator[Any]:
    """Sequential decode with at most two originals and one pair of flows in memory."""
    decoded = 0
    cache: dict[int, Any] = {}
    flow_pair = None
    flows = None
    for row in rows:
        lo, hi = row["source_frame_left"], row["source_frame_right"]
        cache = {k: v for k, v in cache.items() if k in (lo, hi)}
        while decoded < hi:
            tick()
            ok, frame = cap.read()
            decoded += 1
            if not ok:
                raise EventSyncError(f"{source.name}: cannot decode original frame {decoded}")
            if decoded in (lo, hi):
                cache[decoded] = frame
        if lo == hi:
            yield cache[lo]
        else:
            if flow_pair != (lo, hi):
                tick()
                flows = bidirectional_flow(cache[lo], cache[hi])
                flow_pair = (lo, hi)
            tick()
            assert flows is not None
            yield interpolate_image(
                cache[lo], cache[hi], Fraction(row["interpolation_fraction"]), flows
            )


def _encode_samples(
    source: Path,
    frames: Sequence[int],
    destination: Path,
    rate: str,
    *,
    interpolation_rows: Sequence[Mapping[str, Any]] | None = None,
    tick: Callable[[], None] = lambda: None,
    frame_done: Callable[[], None] = lambda: None,
    render_images: Callable[..., Iterator[Any]] | None = None,
) -> None:
    """Sequential decoding preserves original frame identity; FFmpeg uses exact FPS.

    Lossless RGB H.264 keeps each decoded observation intact, including odd sizes.
    Stderr goes to a file to avoid pipe deadlocks. Failure never leaves a video.
    """
    cap = cv2.VideoCapture(str(source))
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    exact_rate = Fraction(rate)
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
        # FFmpeg may choose a rounded encoder time base even with exact input FPS.
        # Use one grid interval per encoder tick and integer MP4 sample durations.
        "-enc_time_base",
        str(1 / exact_rate),
        "-video_track_timescale",
        str(exact_rate.numerator),
        "-movflags",
        "+faststart",
        str(destination),
    ]
    try:
        with tempfile.TemporaryFile() as errors:
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stderr=errors)
            assert process.stdin is not None
            try:
                rows = (
                    interpolation_rows
                    if interpolation_rows is not None
                    else [{"source_frame_left": f, "source_frame_right": f} for f in frames]
                )
                renderer = render_images or _render_images
                for frame in renderer(cap, source, rows, tick):
                    tick()
                    process.stdin.write(frame.tobytes())
                    frame_done()
                process.stdin.close()
                while process.poll() is None:
                    tick()
                    with contextlib.suppress(subprocess.TimeoutExpired):
                        process.wait(timeout=0.1)
            except BaseException:
                process.terminate()
                try:
                    process.wait(timeout=3)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
                with contextlib.suppress(BrokenPipeError):
                    process.stdin.close()
                raise
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
    *,
    progress_callback: Callable[[str, int, int], bool | None] | None = None,
) -> Path:
    """Export common-grid videos, retained observations, maps and a versioned audit."""
    manifest = validate_manifest(manifest)
    for line in event_summary(manifest):
        print(">> " + line, flush=True)
    if manifest["interpolation_method"] == INTERPOLATION:
        print(">> " + INTERPOLATION_WARNING, flush=True)
    completed = 0
    total = len(manifest["cameras"]) * manifest["output_frame_count"]

    def tick(name: str) -> None:
        if progress_callback is not None and progress_callback(name, completed, total) is False:
            raise EventExportCancelledError("Event export cancelled; directory marked INCOMPLETE")

    def frame_done(name: str) -> None:
        nonlocal completed
        completed += 1
        if completed % max(1, total // 10) == 0 or completed == total:
            print(f">> Event export: {completed}/{total} frames ({name})", flush=True)
        tick(name)

    source_dir, output_dir = Path(source_dir).resolve(), Path(output_dir).resolve()
    prepared = []
    for name, camera in manifest["cameras"].items():
        tick(name)
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
    incomplete = output_dir / "INCOMPLETE.txt"
    incomplete.write_text(
        "Export incomplete (running, cancelled or failed). Do not use as a synchronized set.\n",
        encoding="utf-8",
    )
    manifest["exports"] = {}
    for name, source, timeline, rows in prepared:
        output_name = f"{name}_events.mp4"
        _encode_samples(
            source,
            [r["original_frame"] for r in rows],
            output_dir / output_name,
            manifest["output_fps"],
            interpolation_rows=rows if manifest["interpolation_method"] == INTERPOLATION else None,
            tick=lambda name=name: tick(name),
            frame_done=lambda name=name: frame_done(name),
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
                tick(name)
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
        ASSUMPTION
        + "\n"
        + LIMITATION
        + "\n"
        + (
            INTERPOLATION_WARNING + "\n"
            if manifest["interpolation_method"] == INTERPOLATION
            else ""
        )
        + "Physical time is authoritative in *_retained_times.csv; PNG sequences have no playback FPS.\n"
        "Signed temporal error = selected physical time - requested output time.\n"
        f"Sample span: {manifest['sample_span_seconds']:.9f} s. "
        f"Playback duration: {manifest['playback_duration_seconds']:.9f} s.\n",
        encoding="utf-8",
    )
    incomplete.unlink()
    return output_dir
