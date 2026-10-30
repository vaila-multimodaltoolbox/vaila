"""
================================================================================
Script: freekiki_stereo.py - binocular (two-camera) ball reconstruction for FreeKiki runs
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 27 October 2026
Update Date: 29 October 2026

Description:
    The monocular 3D run places the ball on the ground plane (z = 0.11 m), so its height is
    never measured. When two cameras see the same kick, each frame of each camera gives a
    pixel of the ball and the per-frame DLT3D of that camera (pitch metres -> pixels, from the
    field keypoints). Triangulating the two pixels gives the ball in metres with its real
    height, and the reprojection error in both cameras says how well the pair agrees.

    The two cameras must describe the same instant. The sync is the frame pair written in the
    file names (``*_sync_<n>_frames_<a>_to_<b>``): both cuts start at the same instant there.
    The two videos of ``video_event_equalizer`` carry no sync in their names: they share one
    grid, and the contact frame of the ``event_equalizer.json`` beside them is the sync of both
    (:func:`video_sync`).
    The default clock of a frame is ``frame / fps`` of its cut video. An original capture clock
    (ffprobe of the file the cut came from) can be given instead; the summary then reports both,
    so the clock the pair agrees with is visible, not assumed.

    * :func:`reconstruct_ball` - the per-frame triangulation of the reference camera's track;
    * :func:`run_binocular` - reads the two 3D folders, writes the CSV, the JSON summary and
      the comparison with the monocular track.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_detect3d import _single, _video_from_readme, work_owner
    from .freekiki_field3d import dlt_for_frames
    from .freekiki_physics import fit_flight_views
    from .monocular_dlt_align import dlt_projection_matrix
except ImportError:
    from freekiki_detect3d import (  # ty: ignore[unresolved-import]
        _single,
        _video_from_readme,
        work_owner,
    )
    from freekiki_field3d import dlt_for_frames  # ty: ignore[unresolved-import]
    from freekiki_physics import fit_flight_views  # ty: ignore[unresolved-import]
    from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]

SYNC_NAME = re.compile(r"_sync_(\d+)_frames_(\d+)_to_(\d+)")
EQUALIZER_MANIFEST = "event_equalizer.json"  # video_event_equalizer: two videos on one grid
MAX_RESIDUAL_PX = 3.0  # both views must reproduce the ball within this (operational choice)
MIN_ANGLE_DEG = 5.0  # below this the two rays are nearly parallel and depth is not trusted
OUTPUT_PREFIX = "processed_rec3d_binocular"


@dataclass
class CamView:
    """One camera: its per-frame DLT, its ball pixels and its clock, all in cut-video frames.

    ``dlt`` and ``markers`` are indexed by the frame of the cut video. ``pts_cut`` is the capture
    time (s) of each cut frame, and ``sync_cut`` the cut frame that happens at the same instant as
    the other camera's sync frame. ``cut_start`` is kept only for the record (where the cut begins
    in the original file).
    """

    name: str
    dlt: dict[int, np.ndarray]
    markers: dict[int, np.ndarray]
    cut_start: int
    sync_cut: int
    pts_cut: np.ndarray


def parse_sync_name(stem: str) -> tuple[int, int] | None:
    """``(sync_frame, cut_start)`` from a file stem such as ``c1_main_sync_15_frames_2_to_79``."""
    match = SYNC_NAME.search(stem)
    if match is None:
        return None
    sync, start, _end = (int(value) for value in match.groups())
    return sync, start


def _equalizer_manifest(folder: Path) -> dict | None:
    """The ``event_equalizer.json`` of a folder written by ``video_event_equalizer``, or None."""
    path = Path(folder) / EQUALIZER_MANIFEST
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    if not isinstance(data, dict) or "vaila_event_equalizer" not in data:
        return None
    events = data.get("output_event_frames")
    if not isinstance(events, list) or len(events) != 3:
        return None
    return data


def equalizer_videos(folder: Path | str) -> list[Path]:
    """The two videos of an event-equalizer folder, reference camera first.

    Empty when the folder has no manifest or a video the manifest names is missing.
    """
    folder = Path(folder)
    data = _equalizer_manifest(folder)
    cameras = (data or {}).get("cameras")
    if not isinstance(cameras, dict):
        return []
    videos = [
        folder / str(cameras[role].get("video", ""))
        for role in ("reference", "source")
        if isinstance(cameras.get(role), dict)
    ]
    return videos if len(videos) == 2 and all(v.is_file() for v in videos) else []


def equalizer_sync(video: Path | str) -> tuple[int, int] | None:
    """``(sync_frame, cut_start)`` of a video the event equalizer wrote, else None.

    Both cameras of the equalizer share one grid: the contact (the middle event) is the same output
    frame in each, so it is the sync, and the video is its own cut (``cut_start`` 0). The manifest
    counts frames from ``frame_base`` (1); the pipeline counts from 0.
    """
    video = Path(video)
    data = _equalizer_manifest(video.parent)
    if data is None or video not in equalizer_videos(video.parent):
        return None
    try:
        return int(data["output_event_frames"][1]) - int(data.get("frame_base", 1)), 0
    except (TypeError, ValueError):
        return None


def video_sync(video: Path | str) -> tuple[int, int] | None:
    """``(sync_frame, cut_start)`` of a cut video: the numbers in its name, else the equalizer's."""
    video = Path(video)
    return parse_sync_name(video.stem) or equalizer_sync(video)


def video_timestamps(path: Path | str) -> np.ndarray:
    """Capture time (s) of every frame of a video, from the container (ffprobe).

    Without ffprobe the frames are placed at the rate OpenCV reports; the caller is told
    through the summary, because a nominal rate is not the capture clock.
    """
    path = Path(path)
    ffprobe = shutil.which("ffprobe")
    if ffprobe is not None:
        out = subprocess.run(
            [
                ffprobe,
                "-v",
                "error",
                "-select_streams",
                "v:0",
                "-show_entries",
                "frame=best_effort_timestamp_time",
                "-of",
                "csv=p=0",
                str(path),
            ],
            capture_output=True,
            text=True,
            check=False,
            timeout=900,
        )
        values = []
        for line in out.stdout.splitlines():
            try:
                values.append(float(line.strip().rstrip(",")))
            except ValueError:
                values.append(np.nan)
        stamps = np.asarray(values, dtype=float)
        if stamps.size and np.isfinite(stamps).all() and np.all(np.diff(stamps) > 0):
            return stamps
        if stamps.size:
            return _fill_missing(stamps)
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"cannot read the frame timing of {path}")
    return np.arange(count) / fps


def nominal_clock(path: Path | str) -> np.ndarray:
    """Capture time (s) of each frame of a video at its nominal rate: ``frame / fps``.

    This is the default clock: for a cut video the frames are the same instants as the file's
    rate, and it is the clock the two cameras agree with (see :func:`run_binocular`).
    """
    import cv2

    cap = cv2.VideoCapture(str(path))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    finally:
        cap.release()
    if fps <= 0 or count <= 0:
        raise ValueError(f"cannot read the frame rate of {path}")
    return np.arange(count) / fps


def _fill_missing(stamps: np.ndarray) -> np.ndarray:
    """Interpolate the frames ffprobe could not time, keeping the order of the rest."""
    index = np.arange(stamps.size)
    good = np.isfinite(stamps)
    if good.sum() < 2:
        raise ValueError("fewer than two frames have a capture time")
    return np.interp(index, index[good], stamps[good])


def load_markers(path: Path | str) -> dict[int, np.ndarray]:
    """Ball pixels per cut frame from a ``frame,p0_x,p0_y`` file (empty rows are skipped)."""
    markers: dict[int, np.ndarray] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if reader.fieldnames is None or not {"frame", "p0_x", "p0_y"} <= set(reader.fieldnames):
            raise ValueError(f"{path} must have the columns frame,p0_x,p0_y")
        for row in reader:
            if not row["p0_x"] or not row["p0_y"]:
                continue
            markers[int(row["frame"])] = np.array([float(row["p0_x"]), float(row["p0_y"])])
    return markers


def find_dlt3d(run_dir: Path | str, video: Path | str | None) -> dict[int, np.ndarray]:
    """Frame -> DLT coefficients of this video's detect run (``<stem>.dlt3d`` in ``freekiki_predict_*``).

    Batch runs keep one detect folder per video under ``work/``, so the file is matched by the
    video's stem; a single unmatched file is used only when there is nothing else to choose.
    """
    owner = work_owner(run_dir, video)
    if owner is None:
        raise FileNotFoundError(f"No work/ folder for {run_dir}: the per-frame DLT3D is needed")
    found = sorted(p for p in (owner / "work").glob("**/freekiki_predict_*/*.dlt3d") if p.is_file())
    stem = Path(video).stem if video else ""
    matching = [p for p in found if p.stem == stem]
    dlt_file = _single(matching or found, "*.dlt3d for this video")
    if dlt_file is None:
        raise FileNotFoundError(f"No .dlt3d of {stem or 'the video'} in {owner / 'work'}")
    return dlt_for_frames(dlt_file)


def _interp(values: dict[int, np.ndarray], frame: float) -> np.ndarray | None:
    """Linear interpolation of a per-frame array at a fractional frame; None outside the data."""
    lo = int(np.floor(frame))
    weight = frame - lo
    if abs(weight) < 1e-9:
        return values.get(lo)
    first, second = values.get(lo), values.get(lo + 1)
    if first is None or second is None:
        return None
    return (1.0 - weight) * first + weight * second


def capture_time(view: CamView, cut_frame: float) -> float:
    """Seconds from this camera's sync instant to a cut frame (fractional allowed)."""
    frames = np.arange(view.pts_cut.size)
    at = np.interp(cut_frame, frames, view.pts_cut)
    sync_at = np.interp(view.sync_cut, frames, view.pts_cut)
    return float(at - sync_at)


def cut_frame_at(view: CamView, seconds: float) -> float | None:
    """The fractional cut frame of this camera at ``seconds`` from its sync instant."""
    target = seconds + float(np.interp(view.sync_cut, np.arange(view.pts_cut.size), view.pts_cut))
    if target < view.pts_cut[0] or target > view.pts_cut[-1]:
        return None
    return float(np.interp(target, view.pts_cut, np.arange(view.pts_cut.size)))


def projection_centre(P: np.ndarray) -> np.ndarray:
    """Optical centre of a 3x4 projection matrix (the point that maps to the null space)."""
    return -np.linalg.solve(P[:, :3], P[:, 3])


def project(P: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Pixel of a pitch point through a 3x4 projection matrix."""
    h = P @ np.append(xyz, 1.0)
    return h[:2] / h[2]


def triangulate(
    L_a: np.ndarray, L_b: np.ndarray, uv_a: np.ndarray, uv_b: np.ndarray
) -> tuple[np.ndarray, float, float, float]:
    """Pitch point (m) seen at ``uv_a`` and ``uv_b``; returns (xyz, residual_a_px, residual_b_px,
    angle_deg between the two rays). Uses the linear DLT of :mod:`vaila.rec3d`."""
    try:
        from .rec3d import rec3d_multicam
    except ImportError:
        from rec3d import rec3d_multicam  # ty: ignore[unresolved-import]

    xyz = np.asarray(
        rec3d_multicam(
            [L_a, L_b], [(float(uv_a[0]), float(uv_a[1])), (float(uv_b[0]), float(uv_b[1]))]
        ),
        dtype=float,
    )
    P_a, P_b = dlt_projection_matrix(L_a), dlt_projection_matrix(L_b)
    resid_a = float(np.linalg.norm(project(P_a, xyz) - uv_a))
    resid_b = float(np.linalg.norm(project(P_b, xyz) - uv_b))
    ray_a = projection_centre(P_a) - xyz
    ray_b = projection_centre(P_b) - xyz
    cos = float(np.dot(ray_a, ray_b) / (np.linalg.norm(ray_a) * np.linalg.norm(ray_b)))
    angle = float(np.degrees(np.arccos(np.clip(cos, -1.0, 1.0))))
    return xyz, resid_a, resid_b, angle


def reconstruct_ball(
    reference: CamView,
    other: CamView,
    *,
    max_residual_px: float = MAX_RESIDUAL_PX,
    min_angle_deg: float = MIN_ANGLE_DEG,
) -> list[dict]:
    """One row per reference frame that has a ball pixel: the ball in metres and its quality.

    The other camera is read at the instant of each reference frame (its clock, from the capture
    times). Rows that cannot be triangulated keep the reason; nothing is filled in.
    """
    rows: list[dict] = []
    for frame in sorted(reference.markers):
        t = capture_time(reference, frame)
        frame_other = cut_frame_at(other, t)
        row: dict = {
            "frame": frame,
            "time_s": round(t, 6),
            "x_m": np.nan,
            "y_m": np.nan,
            "z_m": np.nan,
            "resid_ref_px": np.nan,
            "resid_other_px": np.nan,
            "angle_deg": np.nan,
            "valid": False,
            "reason": "",
        }
        if frame_other is None:
            row["reason"] = "instant outside the other camera's clip"
            rows.append(row)
            continue
        uv_other = _interp(other.markers, frame_other)
        L_ref = _interp(reference.dlt, frame)
        L_other = _interp(other.dlt, frame_other)
        if uv_other is None or L_other is None:
            row["reason"] = "no ball pixel or DLT in the other camera"
            rows.append(row)
            continue
        if L_ref is None:
            row["reason"] = "no DLT in the reference camera"
            rows.append(row)
            continue
        xyz, resid_ref, resid_other, angle = triangulate(
            L_ref, L_other, reference.markers[frame], uv_other
        )
        row.update(
            x_m=float(xyz[0]),
            y_m=float(xyz[1]),
            z_m=float(xyz[2]),
            resid_ref_px=resid_ref,
            resid_other_px=resid_other,
            angle_deg=angle,
        )
        if max(resid_ref, resid_other) > max_residual_px:
            row["reason"] = "reprojection error above the limit"
        elif angle < min_angle_deg:
            row["reason"] = "rays nearly parallel: depth not trusted"
        else:
            row["valid"] = True
            row["reason"] = "ok"
        rows.append(row)
    return rows


def _median(values: list[float]) -> float | None:
    finite = [v for v in values if np.isfinite(v)]
    return float(np.median(finite)) if finite else None


def summarize(rows: list[dict]) -> dict:
    """Counts and medians of a reconstruction (valid rows only for the depth and angle)."""
    valid = [r for r in rows if r["valid"]]
    return {
        "frames": len(rows),
        "valid": len(valid),
        "median_resid_ref_px": _median([r["resid_ref_px"] for r in valid]),
        "median_resid_other_px": _median([r["resid_other_px"] for r in valid]),
        "median_angle_deg": _median([r["angle_deg"] for r in valid]),
        "median_z_m": _median([r["z_m"] for r in valid]),
        "max_z_m": float(np.max([r["z_m"] for r in valid])) if valid else None,
    }


def with_clock(view: CamView, pts_cut: np.ndarray) -> CamView:
    """The same camera on another clock (cut-frame capture times)."""
    return CamView(
        name=view.name,
        dlt=view.dlt,
        markers=view.markers,
        cut_start=view.cut_start,
        sync_cut=view.sync_cut,
        pts_cut=pts_cut,
    )


def monocular_comparison(rows: list[dict], monocular_csv: Path | None) -> dict | None:
    """Difference between the binocular ball and the monocular (ground plane) track, per frame."""
    if monocular_csv is None or not Path(monocular_csv).is_file():
        return None
    mono: dict[int, np.ndarray] = {}
    with Path(monocular_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["x_m"] and row["y_m"] and row["z_m"]:
                mono[int(row["frame"])] = np.array(
                    [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
                )
    dz, dxy = [], []
    for row in rows:
        if not row["valid"] or row["frame"] not in mono:
            continue
        m = mono[row["frame"]]
        dxy.append(float(np.hypot(row["x_m"] - m[0], row["y_m"] - m[1])))
        dz.append(float(row["z_m"] - m[2]))
    return {
        "frames_compared": len(dz),
        "median_horizontal_diff_m": _median(dxy),
        "median_height_diff_m": _median(dz),
    }


def write_rows(rows: list[dict], path: Path) -> Path:
    fields = [
        "frame",
        "time_s",
        "x_m",
        "y_m",
        "z_m",
        "resid_ref_px",
        "resid_other_px",
        "angle_deg",
        "valid",
        "reason",
    ]
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    k: ("" if isinstance(row[k], float) and not np.isfinite(row[k]) else row[k])
                    for k in fields
                }
            )
    return path


def load_view(
    run_dir: Path,
    *,
    markers: Path | None,
    timing: Path | None,
    sync_frame: int | None,
    cut_start: int | None,
    name: str,
) -> tuple[CamView, Path]:
    """A camera from its 3D folder. Returns the view and the cut video used by the run.

    ``timing`` is the original file whose capture times define the clock (default: the cut video
    itself); ``sync_frame`` and ``cut_start`` default to the numbers in the cut video's name.
    """
    cut_video = _video_from_readme(run_dir)
    if cut_video is None or not cut_video.is_file():
        raise FileNotFoundError(f"The video of {run_dir} is not in its README; pass it explicitly")
    parsed = video_sync(cut_video)
    if sync_frame is None or cut_start is None:
        if parsed is None:
            raise ValueError(
                f"{cut_video.name} has no sync numbers in its name; pass --sync and --cut-start"
            )
        sync_frame = parsed[0] if sync_frame is None else sync_frame
        cut_start = parsed[1] if cut_start is None else cut_start
    if markers is None:
        markers = cut_video.with_name(f"{cut_video.stem}_markers.csv")
    if not markers.is_file():
        raise FileNotFoundError(f"Ball markers not found: {markers}")
    cut_clock = nominal_clock(cut_video)
    if timing is None:
        pts_cut = cut_clock
    else:
        full = video_timestamps(timing)
        if full.size < cut_start + cut_clock.size:
            raise ValueError(
                f"{timing} has {full.size} frames; the cut needs frames {cut_start}"
                f" to {cut_start + cut_clock.size - 1}"
            )
        pts_cut = full[cut_start : cut_start + cut_clock.size]
    view = CamView(
        name=name,
        dlt=find_dlt3d(run_dir, cut_video),
        markers=load_markers(markers),
        cut_start=int(cut_start),
        sync_cut=int(sync_frame) - int(cut_start),
        pts_cut=pts_cut,
    )
    return view, cut_video


BARRIER_DISTANCE_M = 9.15  # the wall stands 9.15 m from the ball (FIFA rule)
PITCH_HALF_LENGTH_M = 52.45
PITCH_HALF_WIDTH_M = 33.95


def rough_flight(rows: list[dict], fps: float) -> np.ndarray | None:
    """A starting guess of the flight from the triangulated track: the rest point, the first
    launch (height above 0.3 m) and the velocity just after it. None if there is no launch."""
    valid = [r for r in rows if r["valid"]]
    if len(valid) < 6:
        return None
    frames = np.array([r["frame"] for r in valid])
    xyz = np.array([[r["x_m"], r["y_m"], r["z_m"]] for r in valid])
    lifted = np.flatnonzero(xyz[:, 2] > 0.3)
    if lifted.size == 0:
        return None
    k = int(lifted[0])
    v = np.zeros(3)
    if k + 3 < len(valid):
        dt = (frames[k + 3] - frames[k]) / fps
        if dt > 0:
            v = (xyz[k + 3] - xyz[k]) / dt
    tk = max(frames[k] - 1, 0) / fps
    return np.array([xyz[0, 0], xyz[0, 1], v[0], v[1], v[2], tk, 0.0, 0.0, 0.0])


def fit_binocular(
    view_a: CamView, view_b: CamView, rows: list[dict], fps: float, n_frames: int
) -> dict:
    """One flight fitted to the ball pixels of both cameras, on the reference camera's frames.

    Camera A gives its pixel at every frame; camera B gives the pixel at the instant of each A
    frame (its clock). Each camera adds its own reprojection residuals, so the depth comes from
    the pair, not from the ground plane.
    """
    pixels_a: dict[int, tuple[float, float]] = {}
    cams_a: dict[int, np.ndarray] = {}
    pixels_b: dict[int, tuple[float, float]] = {}
    cams_b: dict[int, np.ndarray] = {}
    for f in sorted(view_a.markers):
        if not 0 <= f < n_frames:
            continue
        L_a = _interp(view_a.dlt, f)
        if L_a is not None:
            uv = view_a.markers[f]
            pixels_a[f] = (float(uv[0]), float(uv[1]))
            cams_a[f] = dlt_projection_matrix(L_a)
        frame_b = cut_frame_at(view_b, capture_time(view_a, f))
        if frame_b is None:
            continue
        uv_b = _interp(view_b.markers, frame_b)
        L_b = _interp(view_b.dlt, frame_b)
        if uv_b is None or L_b is None:
            continue
        pixels_b[f] = (float(uv_b[0]), float(uv_b[1]))
        cams_b[f] = dlt_projection_matrix(L_b)
    return fit_flight_views(
        [(pixels_a, cams_a), (pixels_b, cams_b)],
        fps,
        n_frames,
        anchors={},
        theta0=rough_flight(rows, fps),
    )


def barrier_point(positions: dict[int, tuple[float, float, float]]) -> dict:
    """The point where the fitted ball is 9.15 m from its rest point: the wall's plane, measured."""
    start = np.array(positions[min(positions)][:2])
    for f in sorted(positions):
        x, y, z = positions[f]
        if float(np.hypot(x - start[0], y - start[1])) >= BARRIER_DISTANCE_M:
            return {
                "p0": {
                    "label": "barreira: plano a 9,15 m do chute (regra FIFA)",
                    "frame": int(f),
                    "xyz": [float(x), float(y), float(z)],
                }
            }
    return {}


def write_fit_run(
    out: Path,
    fit: dict,
    rows: list[dict],
    view_a: CamView,
    video_a: Path,
    owner_a: Path,
) -> dict:
    """The fitted ball as a 3D folder the report reads (ball_3d_on_field.csv, ball_flight.json,
    ball.csv, README with the video and the work folder of camera 1)."""
    positions = fit["positions"]
    fit_xyz = {f: np.array(p) for f, p in positions.items()}
    tri = {r["frame"]: np.array([r["x_m"], r["y_m"], r["z_m"]]) for r in rows if r["valid"]}
    gaps = [float(np.linalg.norm(fit_xyz[f] - tri[f])) for f in tri if f in fit_xyz]
    p90 = float(np.percentile(gaps, 90)) if gaps else float("nan")
    median = float(np.median(gaps)) if gaps else float("nan")
    record = {k: v for k, v in fit.items() if k not in ("positions", "per_frame_px")}
    record["per_frame_px"] = {str(f): round(e, 3) for f, e in fit["per_frame_px"].items()}
    record["method"] = "binocular"
    record["anchor_residuals"] = {"fit_vs_triangulation_p90_m": round(p90, 3)}
    record["anchor_points"] = barrier_point(positions)
    (out / "ball_flight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
    with (out / "ball_3d_on_field.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "cx", "cy", "score", "x_m", "y_m", "z_m", "on_field"])
        for f in sorted(fit_xyz):
            x, y, z = fit_xyz[f]
            uv = view_a.markers.get(f)
            on_field = abs(x) <= PITCH_HALF_LENGTH_M and abs(y) <= PITCH_HALF_WIDTH_M
            writer.writerow(
                [
                    f,
                    round(float(uv[0]), 2) if uv is not None else "",
                    round(float(uv[1]), 2) if uv is not None else "",
                    1.0 if uv is not None else "",
                    round(float(x), 6),
                    round(float(y), 6),
                    round(float(z), 6),
                    on_field,
                ]
            )
    with (out / "ball.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "cx", "cy", "x1", "y1", "x2", "y2", "score"])
        for f in sorted(view_a.markers):
            uv = view_a.markers[f]
            writer.writerow(
                [f, round(float(uv[0]), 2), round(float(uv[1]), 2), "", "", "", "", 1.0]
            )
    (out / "README.txt").write_text(
        "FreeKiki 3D output (vailá), binocular\n"
        f"video: {video_a}\n"
        f"work from: {owner_a}\n\n"
        "ball_3d_on_field.csv   ball in pitch metres, from the two-camera flight fit\n"
        "ball_flight.json       the fit (method binocular): rms, residuals, anchor point of the wall\n"
        "ball.csv               camera 1 ball pixels (the reference camera)\n"
        "ball_binocular.csv     per-frame triangulation of both cameras, with the reason of each frame\n"
        "summary.json           the clocks, the sync, the counts and the comparison\n",
        encoding="utf-8",
    )
    return {
        "rms_px": fit["rms_px"],
        "max_px": fit["max_px"],
        "kick_time_s": fit["kick_time_s"],
        "spin_rps": fit["spin_rps"],
        "peak_speed_mps": fit["peak_speed_mps"],
        "fit_vs_triangulation_median_m": round(median, 3),
        "fit_vs_triangulation_p90_m": round(p90, 3),
        "barrier": record["anchor_points"],
    }


def run_binocular(
    run_a: Path,
    run_b: Path,
    *,
    out_dir: Path | None = None,
    markers_a: Path | None = None,
    markers_b: Path | None = None,
    timing_a: Path | None = None,
    timing_b: Path | None = None,
    sync_a: int | None = None,
    sync_b: int | None = None,
    cut_start_a: int | None = None,
    cut_start_b: int | None = None,
    max_residual_px: float = MAX_RESIDUAL_PX,
    min_angle_deg: float = MIN_ANGLE_DEG,
) -> dict:
    """Reconstruct the ball from two 3D folders of the same instant and write a 3D run folder.

    The first folder is the reference (its frames are the output frames). The output folder
    (``processed_rec3d_binocular_<time>/`` next to the first run by default) holds the fitted
    flight in the layout the report reads, the per-frame triangulation and the summary.
    """
    view_a, video_a = load_view(
        Path(run_a),
        markers=markers_a,
        timing=timing_a,
        sync_frame=sync_a,
        cut_start=cut_start_a,
        name="cam_a",
    )
    view_b, video_b = load_view(
        Path(run_b),
        markers=markers_b,
        timing=timing_b,
        sync_frame=sync_b,
        cut_start=cut_start_b,
        name="cam_b",
    )
    rows = reconstruct_ball(
        view_a, view_b, max_residual_px=max_residual_px, min_angle_deg=min_angle_deg
    )
    clock_check: dict = {
        "used": "original capture clock" if timing_a or timing_b else "nominal frame/fps"
    }
    clock_check["used_stats"] = summarize(rows)
    if timing_a or timing_b:
        nominal = reconstruct_ball(
            with_clock(view_a, nominal_clock(video_a)),
            with_clock(view_b, nominal_clock(video_b)),
            max_residual_px=max_residual_px,
            min_angle_deg=min_angle_deg,
        )
        clock_check["nominal_frame_over_fps_stats"] = summarize(nominal)
    fps = 1.0 / float(np.median(np.diff(view_a.pts_cut)))
    n_frames = int(view_a.pts_cut.size)
    fit = fit_binocular(view_a, view_b, rows, fps, n_frames)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out = Path(out_dir) if out_dir is not None else Path(run_a).parent / f"{OUTPUT_PREFIX}_{stamp}"
    out.mkdir(parents=True, exist_ok=True)
    owner_a = work_owner(Path(run_a), video_a)
    if owner_a is None:
        raise FileNotFoundError(f"No work/ folder for {run_a}: the per-frame DLT3D is needed")
    fit_summary = write_fit_run(out, fit, rows, view_a, video_a, owner_a)
    write_rows(rows, out / "ball_binocular.csv")
    summary = {
        "created": datetime.now().isoformat(timespec="seconds"),
        "reference_run": str(Path(run_a)),
        "other_run": str(Path(run_b)),
        "reference_video": str(video_a),
        "other_video": str(video_b),
        "timing": {
            "reference_clock": str(timing_a or "nominal frame/fps"),
            "other_clock": str(timing_b or "nominal frame/fps"),
        },
        "sync": {
            "reference_sync_frame": view_a.sync_cut,
            "reference_cut_start": view_a.cut_start,
            "other_sync_frame": view_b.sync_cut,
            "other_cut_start": view_b.cut_start,
        },
        "limits": {"max_residual_px": max_residual_px, "min_angle_deg": min_angle_deg},
        "counts": summarize(rows),
        "clock_check": clock_check,
        "fit": fit_summary,
        "monocular_comparison": monocular_comparison(rows, Path(run_a) / "ball_3d_on_field.csv"),
    }
    (out / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    summary["out_dir"] = str(out)
    return summary
