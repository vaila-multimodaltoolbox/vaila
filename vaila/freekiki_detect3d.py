"""
================================================================================
Script: freekiki_detect3d.py - field DLT3D -> sam3dinov3, ball and players on pitch
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 25 October 2026
Update Date: 29 October 2026

Description:
    Runs on the output of a finished ``detect`` run (or batch) and adds, per video:

    1. the per-frame focal length of the field DLT3D, written as ``cam_int.npy``
       and handed to ``sam3dinov3`` (``--cam-int-npy``), so SAM 3D Body gets a
       metric focal for every frame that has a valid DLT3D;
    2. the SAM 3 + SAM 3D Body run itself, through the ``sam3dinov3.py`` CLI in a
       subprocess (the GPU stack stays out of the FreeKiki process);
    3. with ``players_on_field``: each tracked person is kept only when the feet
       lie on the pitch for at least ``field_min_frac`` of their frames;
    4. with ``ball``: the per-frame ball track, from the dedicated ball detector
       (``"net"``) or from the SAM 3 ``sports ball`` prompt (``"sam3"``);
    5. the Vitruvian re-fit of the frames whose camera is not a DLT3D (see
       :func:`vitruvian_refit`), written next to the other outputs;
    6. the 3D markers in pitch metres (see :mod:`vaila.freekiki_field3d`): the 49
       field keypoints per frame, the poses and the ball on the pitch, also written
       as one C3D file (see :mod:`vaila.freekiki_export`).

    The camera of every 3D output is the run's DLT3D, the same one SAM 3D Body used
    for its focal, so the poses and the field share a camera per frame. The Vitruvian
    camera only feeds its own diagnostics. The player filter runs on the finished
    ``sam3dinov3`` output, so people are still processed on the GPU before being
    discarded.
"""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_auto_flight import auto_flight, auto_rows_and_record
    from .freekiki_ball import (
        ball_candidates_net,
        ball_candidates_video,
        ball_row,
        track_ball,
        write_ball_csv,
    )
    from .freekiki_camera3d import (
        cam_int_sequence,
        focal_sequence,
        focal_summary,
        write_cam_int_npy,
    )
    from .freekiki_diag import load_field_points
    from .freekiki_export import write_c3d, write_c3d_csv
    from .freekiki_field3d import (
        BALL_3D_COLUMNS,
        BALL_RADIUS_M,
        FIELD_KP_COLUMNS,
        ball_3d,
        box_ground,
        dlt_for_frames,
        field_keypoints_3d,
        pitch_ball_candidates,
        pose_to_pitch,
        write_table,
    )
    from .freekiki_physics import BALL_MODEL, PLAYER_MODEL, fit_flight, smooth_track
    from .freekiki_players import PITCH_HALF_LENGTH_M, PITCH_HALF_WIDTH_M, homography_from_dlt2d
    from .freekiki_players import players_on_field as pitch_verdicts
    from .freekiki_stature import apply_stature_scale
    from .freekiki_topview import write_topview
    from .monocular_dlt_align import dlt_projection_matrix
    from .soccerfield_vitruvian_dlt3d import fit_vitruvian_frame
except ImportError:
    from freekiki_auto_flight import (  # ty: ignore[unresolved-import]
        auto_flight,
        auto_rows_and_record,
    )
    from freekiki_ball import (  # ty: ignore[unresolved-import]
        ball_candidates_net,
        ball_candidates_video,
        ball_row,
        track_ball,
        write_ball_csv,
    )
    from freekiki_camera3d import (  # ty: ignore[unresolved-import]
        cam_int_sequence,
        focal_sequence,
        focal_summary,
        write_cam_int_npy,
    )
    from freekiki_diag import load_field_points  # ty: ignore[unresolved-import]
    from freekiki_export import (  # ty: ignore[unresolved-import]
        write_c3d,
        write_c3d_csv,
    )
    from freekiki_field3d import (  # ty: ignore[unresolved-import]
        BALL_3D_COLUMNS,
        BALL_RADIUS_M,
        FIELD_KP_COLUMNS,
        ball_3d,
        box_ground,
        dlt_for_frames,
        field_keypoints_3d,
        pitch_ball_candidates,
        pose_to_pitch,
        write_table,
    )
    from freekiki_physics import (  # ty: ignore[unresolved-import]
        BALL_MODEL,
        PLAYER_MODEL,
        fit_flight,
        smooth_track,
    )
    from freekiki_players import (  # ty: ignore[unresolved-import]
        PITCH_HALF_LENGTH_M,
        PITCH_HALF_WIDTH_M,
        homography_from_dlt2d,
    )
    from freekiki_players import players_on_field as pitch_verdicts  # ty: ignore[unresolved-import]
    from freekiki_stature import apply_stature_scale  # ty: ignore[unresolved-import]
    from freekiki_topview import write_topview  # ty: ignore[unresolved-import]
    from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]
    from soccerfield_vitruvian_dlt3d import fit_vitruvian_frame  # ty: ignore[unresolved-import]

MODULE_DIR = Path(__file__).resolve().parent
SAM3DINOV3 = MODULE_DIR / "sam3dinov3.py"


def _log(message: str) -> None:
    print(f"[freekiki detect-3d] {message}", flush=True)


def _single(paths: list[Path], what: str) -> Path | None:
    """The one file of a run folder matching ``what``; None when there is none."""
    if not paths:
        return None
    if len(paths) > 1:
        raise ValueError(f"expected one {what} in the run folder, found {len(paths)}")
    return paths[0]


def _video_info(video: Path) -> tuple[int, int, int]:
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    try:
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    finally:
        cap.release()
    if n <= 0 or width <= 0 or height <= 0:
        raise ValueError(f"Invalid video metadata: {video}")
    return n, width, height


def _read_homographies(dlt2d: Path) -> dict[int, np.ndarray]:
    """Frame index -> pitch homography from a ``.dlt2d`` file (frame, L1..L8)."""
    out: dict[int, np.ndarray] = {}
    with dlt2d.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                frame = int(float(row["frame"]))
                coeffs = np.array([float(row[f"L{i}"]) for i in range(1, 9)])
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(coeffs).all():
                out[frame] = homography_from_dlt2d(coeffs)
    return out


def _read_person_tracks(camera_csv: Path) -> dict[int, list[tuple[int, np.ndarray]]]:
    """``person_id -> [(frame, xyxy), ...]`` from the sam3dinov3 camera CSV."""
    tracks: dict[int, list[tuple[int, np.ndarray]]] = {}
    with camera_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                pid = int(row["person_id"])
                frame = int(row["frame"])
                box = np.array(
                    [
                        float(row[k])
                        for k in ("bbox_x1_px", "bbox_y1_px", "bbox_x2_px", "bbox_y2_px")
                    ]
                )
            except (KeyError, TypeError, ValueError):
                continue
            if np.isfinite(box).all():
                tracks.setdefault(pid, []).append((frame, box))
    return tracks


def _write_person_verdicts(path: Path, verdicts: dict[int, dict]) -> Path:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["person_id", "fraction", "on", "checked", "keep"])
        writer.writeheader()
        for pid in sorted(verdicts):
            writer.writerow({"person_id": pid, **verdicts[pid]})
    return path


def _filter_person_csv(src: Path, dst: Path, kept: set[int]) -> Path:
    """Copy the rows of ``src`` whose ``person_id`` is in ``kept`` (camera or keypoints CSV)."""
    with src.open(newline="", encoding="utf-8") as fh_in:
        reader = csv.DictReader(fh_in)
        with dst.open("w", newline="", encoding="utf-8") as fh_out:
            writer = csv.DictWriter(fh_out, fieldnames=reader.fieldnames or [])
            writer.writeheader()
            for row in reader:
                try:
                    if int(row["person_id"]) in kept:
                        writer.writerow(row)
                except (KeyError, TypeError, ValueError):
                    continue
    return dst


VITRUVIAN_HEIGHT_M = 1.83  # the stature prior, freekiki_stature.STATURE_M
WEAK_CAMERAS = frozenset({"homography", "planar", "none"})


def _read_camera_sources(path: Path) -> dict[int, str]:
    """Frame -> camera source (dlt3d | homography | planar | none) of ``field_kps_geom_status.csv``."""
    out: dict[int, str] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                out[int(float(row["frame"]))] = row["camera"]
            except (KeyError, TypeError, ValueError):
                continue
    return out


def _read_field_pixels(path: Path) -> dict[int, np.ndarray]:
    """Frame -> ``(K, 2)`` accepted field keypoints in pixels, NaN where not accepted."""
    out: dict[int, np.ndarray] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        n_points = (len(next(reader, [])) - 1) // 2
        for row in reader:
            try:
                frame = int(float(row[0]))
            except (IndexError, ValueError):
                continue
            xy = np.full((n_points, 2), np.nan)
            for i in range(n_points):
                try:
                    xy[i] = (float(row[1 + 2 * i]), float(row[2 + 2 * i]))
                except (IndexError, ValueError):
                    continue
            out[frame] = xy
    return out


def vitruvian_refit(run_dir: Path, camera_csv: Path, out_dir: Path) -> dict:
    """Re-fit the field camera of the weak frames with the Vitruvian model.

    A frame whose camera came from a homography, a planar fit or nothing has no
    off-ground keypoint to pin its height. The people's boxes give that vertical
    scale instead (:func:`fit_vitruvian_frame`): bottom centre = ground point,
    top centre = head, height = ``VITRUVIAN_HEIGHT_M``. Only the planar schema
    points are used as ground references.

    Writes ``vitruvian.dlt3d`` (``frame,L1..L11``) and ``vitruvian_diagnostics.csv``
    in ``out_dir``. The run's own ``.dlt3d`` is left untouched, and the SAM 3D Body
    focal of those frames is not re-run: this is a calibration output.
    """
    sources_csv = run_dir / "field_kps_geom_status.csv"
    pixels_csv = run_dir / "field_kps_getpixelvideo.csv"
    if not (sources_csv.is_file() and pixels_csv.is_file()):
        return {"status": "skipped: no field keypoints or camera status in the run"}

    sources = _read_camera_sources(sources_csv)
    pixels = _read_field_pixels(pixels_csv)
    _, _, xyz = load_field_points()
    planar = xyz[:, 2] == 0
    boxes_by_frame: dict[int, list[np.ndarray]] = {}
    for boxes in _read_person_tracks(camera_csv).values():
        for frame, box in boxes:
            boxes_by_frame.setdefault(frame, []).append(box)

    coefficients: list[list[str]] = []
    diagnostics: list[list[str]] = []
    candidates = failed = 0
    for frame in sorted(f for f, src in sources.items() if src in WEAK_CAMERAS):
        px = pixels.get(frame)
        people = boxes_by_frame.get(frame, [])
        if px is None or len(people) < 2:
            continue
        candidates += 1
        ok = planar & np.isfinite(px).all(axis=1)
        boxes = np.asarray(people, dtype=float)
        centre_x = 0.5 * (boxes[:, 0] + boxes[:, 2])
        try:
            params, diag, _ = fit_vitruvian_frame(
                frame=frame,
                field_world_xy=xyz[ok, :2],
                field_pixels=px[ok],
                bbox_bottom_pixels=np.column_stack((centre_x, boxes[:, 3])),
                bbox_top_pixels=np.column_stack((centre_x, boxes[:, 1])),
                bbox_heights_m=np.full(len(boxes), VITRUVIAN_HEIGHT_M),
            )
        except (ValueError, np.linalg.LinAlgError):
            failed += 1
            continue
        coefficients.append([str(frame)] + [f"{v:.10g}" for v in params])
        diagnostics.append(
            [
                str(frame),
                str(diag.n_field),
                str(diag.n_bbox_verticals),
                f"{diag.field_reprojection_rms_px:.3f}",
            ]
        )

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "vitruvian.dlt3d").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", *[f"L{i}" for i in range(1, 12)]])
        writer.writerows(coefficients)
    with (out_dir / "vitruvian_diagnostics.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "n_field", "n_bbox_verticals", "field_rms_px"])
        writer.writerows(diagnostics)
    return {"status": "ok", "weak_frames": candidates, "refit": len(coefficients), "failed": failed}


POSE_COLUMNS = ["frame", "person_id", "kpt_idx", "kpt_name", "x_m", "y_m", "z_m"]
STATURE_COLUMNS = [
    "person_id",
    "frames",
    "nose_median_raw_m",
    "scale",
    "stature_before_m",
    "stature_after_m",
    "flag",
]
GROUND_COLUMNS = ["frame", "person_id", "raw_x_m", "raw_y_m", "x_m", "y_m"]
GROUND_KPT_IDX = 999  # index of the "ground" marker, after the 70 MHR keypoints
BALL_MODES = ("off", "net", "sam3")


def _ball_mode(ball) -> str:
    """``off | net | sam3``; a bool is kept for old callers (True = SAM 3, False = off)."""
    if ball is True:
        return "sam3"
    if ball is False or ball is None:
        return "off"
    if ball in BALL_MODES:
        return ball
    raise ValueError(f"ball must be one of {BALL_MODES}, got {ball!r}")


def _video_fps(video: Path) -> float:
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    finally:
        cap.release()
    if fps <= 0:
        raise ValueError(f"video fps unknown: {video}")
    return fps


def _float_or_nan(text: str) -> float:
    try:
        return float(text)
    except (TypeError, ValueError):
        return float("nan")


def _pose_rows(
    keypoints_csv: Path,
    cameras: dict[int, np.ndarray],
    boxes: dict[tuple[int, int], np.ndarray],
    kept: set[int] | None,
    fps: float,
    n_frames: int,
) -> tuple[list[dict], list[dict]]:
    """Poses of the kept people in pitch metres, plus each person's ground track.

    The ground point of a person is the bottom centre of its box, on the pitch through
    the frame's DLT3D. That raw point jumps when the box changes, so each person's track
    is smoothed (:func:`freekiki_physics.smooth_track`) and every pose is anchored on the smoothed
    point of its frame. The pose keeps its own shape: only the position moves.
    Returns ``(pose rows, ground rows)``; the ground rows carry the raw and smoothed point.
    """
    groups: dict[tuple[int, int], list[dict]] = {}
    with keypoints_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            try:
                pid = int(row["person_id"])
                frame = int(row["frame"])
            except (KeyError, TypeError, ValueError):
                continue
            if kept is not None and pid not in kept:
                continue
            groups.setdefault((frame, pid), []).append(row)

    raw_ground: dict[int, dict[int, tuple[float, float]]] = {}
    for frame, pid in groups:
        L = cameras.get(frame)
        box = boxes.get((frame, pid))
        if L is None or box is None:
            continue
        ground = box_ground(L, box)
        if ground is not None:
            raw_ground.setdefault(pid, {})[frame] = ground
    tracks = {
        pid: smooth_track(points, n_frames, fps, PLAYER_MODEL) for pid, points in raw_ground.items()
    }

    rows: list[dict] = []
    ground_rows: list[dict] = []
    for (frame, pid), people in sorted(groups.items()):
        track = tracks.get(pid, {})
        # a frame without its own DLT3D (or ground point) has no camera to place the pose
        # on the pitch; the smoothed track can still fill it, so it is skipped here
        if frame not in track or frame not in raw_ground.get(pid, {}):
            continue
        L = cameras[frame]
        box = boxes[(frame, pid)]
        n_kpts = max(int(r["kpt_idx"]) for r in people) + 1
        names = [""] * n_kpts
        xcam = np.full((n_kpts, 3), np.nan)
        for r in people:
            i = int(r["kpt_idx"])
            names[i] = r["kpt_name"]
            xcam[i] = (
                _float_or_nan(r["xcam_m"]),
                _float_or_nan(r["ycam_m"]),
                _float_or_nan(r["zcam_m"]),
            )
        pitch = pose_to_pitch(names, xcam, L, box, ground_xy=track[frame])
        if pitch is None:
            continue
        raw = raw_ground[pid][frame]
        ground_rows.append(
            {
                "frame": frame,
                "person_id": pid,
                "raw_x_m": raw[0],
                "raw_y_m": raw[1],
                "x_m": track[frame][0],
                "y_m": track[frame][1],
            }
        )
        for i, name in enumerate(names):
            if name and np.isfinite(pitch[i]).all():
                rows.append(
                    {
                        "frame": frame,
                        "person_id": pid,
                        "kpt_idx": i,
                        "kpt_name": name,
                        "x_m": pitch[i, 0],
                        "y_m": pitch[i, 1],
                        "z_m": pitch[i, 2],
                    }
                )
    return rows, ground_rows


def _c3d_groups(
    names: list[str],
    kp_rows: list[dict],
    pose_rows: list[dict],
    ball_on: list[dict],
    n_frames: int,
) -> dict[str, tuple[list[str], np.ndarray]]:
    """C3D files as ``name -> (labels, (frames, markers, 3))``.

    A C3D file keeps its marker names in one parameter that holds at most 255 labels
    (``ezc3d`` has no ``LABELS2``), so the markers are split: ``field_ball`` holds the
    49 field points and the ball, and each person has a file with their keypoints.
    """
    field = np.full((n_frames, len(names), 3), np.nan)
    for r in kp_rows:
        if r["frame"] < n_frames:
            field[r["frame"], r["point"]] = (r["x_m"], r["y_m"], r["z_m"])
    ball = np.full((n_frames, 1, 3), np.nan)
    for r in ball_on:
        if r["frame"] < n_frames:
            ball[r["frame"], 0] = (r["x_m"], r["y_m"], r["z_m"])
    groups: dict[str, tuple[list[str], np.ndarray]] = {
        # The field points are kept with the ball: the poses are placed inside the pitch
        # with them. ``ball`` repeats the ball alone, for a viewer that wants only the ball.
        "field_ball": (
            [f"FKP_{name}" for name in names] + ["BALL"],
            np.concatenate((field, ball), axis=1),
        ),
        "ball": (["BALL"], ball),
    }

    keys_by_person: dict[int, set[tuple[int, str]]] = {}
    for r in pose_rows:
        keys_by_person.setdefault(r["person_id"], set()).add((r["kpt_idx"], r["kpt_name"]))
    for pid in sorted(keys_by_person):
        keys = sorted(keys_by_person[pid])
        index = {key: i for i, key in enumerate(keys)}
        arr = np.full((n_frames, len(keys), 3), np.nan)
        for r in pose_rows:
            if r["person_id"] == pid and r["frame"] < n_frames:
                arr[r["frame"], index[(r["kpt_idx"], r["kpt_name"])]] = (
                    r["x_m"],
                    r["y_m"],
                    r["z_m"],
                )
        groups[f"person_{pid:03d}"] = ([f"P{pid}_{name}" for _, name in keys], arr)
    return groups


def physical_ball_rows(
    ball_all: list[dict], n_frames: int, fps: float, *, trusted: bool = False
) -> list[dict]:
    """The ball on the pitch as a physically consistent track (see ``freekiki_physics``).

    ``ball_all`` holds the raw ground point of every chosen ball box. Its on-pitch points
    are smoothed with a Kalman filter and a speed ceiling; a smoothed point is kept
    only when it is still inside the pitch. The raw columns stay in ``ball_3d.csv``.

    ``trusted`` is for a ball the user marked by hand: every marked frame is used, also
    when the ball is outside the pitch (over a goal line or touchline, out of play), so
    the track has no gap where the user placed the ball.
    """
    raw = {int(r["frame"]): r for r in ball_all if trusted or r["on_field"]}
    measurements = {f: (float(r["x_m"]), float(r["y_m"])) for f, r in raw.items()}
    track = smooth_track(measurements, n_frames, fps, BALL_MODEL)
    rows: list[dict] = []
    for f, (x, y) in sorted(track.items()):
        if not trusted and (abs(x) > PITCH_HALF_LENGTH_M or abs(y) > PITCH_HALF_WIDTH_M):
            continue
        src = raw.get(f, {})
        rows.append(
            {
                "frame": f,
                "cx": src.get("cx", ""),
                "cy": src.get("cy", ""),
                "score": src.get("score", ""),
                "x_m": x,
                "y_m": y,
                "z_m": BALL_RADIUS_M,
                "on_field": True,
            }
        )
    return rows


@dataclass(frozen=True)
class BallAnchorSpec:
    """The physical points that anchor a hand-corrected ball flight (see ``fit_flight``).

    ``kicker``: the ball rests at the kicker's ``kicker_part`` at frame 0. ``barrier``: at
    ``barrier_frame`` the ball is ``barrier_height_m`` high and ``barrier_offset_m`` (horizontal)
    from the ``barrier_part`` of the barrier player. ``keeper``: at ``keeper_frame`` (default:
    the last frame) the ball is at the mean of the keeper's ``keeper_parts``.
    """

    kicker_id: int
    barrier_id: int
    keeper_id: int
    kicker_part: str = "left-ankle"
    barrier_frame: int = 9
    barrier_part: str = "left-shoulder"
    barrier_height_m: float = 1.8
    barrier_offset_m: float = 0.3
    keeper_frame: int | None = None
    keeper_parts: tuple[str, ...] = ("left-wrist", "right-wrist")


@dataclass(frozen=True)
class BallAnchorPoints:
    """Anchors given as points: ``(frame, xyz in pitch metres, sigma in metres, label)``.

    At least three are needed. They come from the anchor dialog (the ball on the ground, a
    keypoint of a kept player or a point of the goal), or from a JSON with a ``points`` list.
    """

    points: tuple[dict, ...]


MIN_BALL_POINTS = 3
BALL_CENTRE_Z = 0.11  # the ball's centre on the ground (m)


def _point_from_json(item: dict, index: int) -> dict:
    frame = int(item["frame"])
    xyz = [float(v) for v in item["xyz"]]
    sigma = float(item.get("sigma", 0.3))
    if frame < 0 or len(xyz) != 3 or not np.isfinite(xyz).all() or sigma <= 0:
        raise ValueError(
            f"ball anchor {index}: needs frame >= 0, xyz of 3 finite numbers, sigma > 0"
        )
    return {
        "frame": frame,
        "xyz": xyz,
        "sigma": sigma,
        "label": str(item.get("label", f"p{index}")),
    }


def ball_anchor_spec_from_json(text: str) -> BallAnchorSpec | BallAnchorPoints:
    """Ball anchors from JSON: either the player form (``kicker_id``...) or ``{"points": [...]}``.

    Unknown keys are an error; the points form needs at least three points.
    """
    data = json.loads(text)
    if not isinstance(data, dict):
        raise ValueError("--ball-anchors must be a JSON object")
    if "points" in data:
        items = data["points"]
        if not isinstance(items, list) or len(items) < MIN_BALL_POINTS:
            raise ValueError(f"at least {MIN_BALL_POINTS} ball anchors are needed")
        return BallAnchorPoints(tuple(_point_from_json(it, j) for j, it in enumerate(items)))
    fields = set(BallAnchorSpec.__dataclass_fields__)
    unknown = sorted(set(data) - fields)
    if unknown:
        raise ValueError(f"unknown --ball-anchors keys: {unknown} (allowed: {sorted(fields)})")
    if "keeper_parts" in data:
        data["keeper_parts"] = tuple(data["keeper_parts"])
    return BallAnchorSpec(**data)


def anchor_points_json(points: list[dict]) -> str:
    """The ``{"points": [...]}`` text the anchor dialog writes into the form."""
    return json.dumps(
        {
            "points": [
                {
                    "frame": int(p["frame"]),
                    "xyz": [round(float(v), 4) for v in p["xyz"]],
                    "sigma": float(p["sigma"]),
                    "label": str(p.get("label", "")),
                }
                for p in points
            ]
        },
        ensure_ascii=False,
    )


def _pose_point(pose_rows: list[dict], pid: int, frame: int, parts) -> np.ndarray:
    """Mean position (metres) of the named keypoints of a person at a frame."""
    wanted = set(parts)
    pts = [
        (float(r["x_m"]), float(r["y_m"]), float(r["z_m"]))
        for r in pose_rows
        if int(r["person_id"]) == pid and int(r["frame"]) == frame and r["kpt_name"] in wanted
    ]
    if len(pts) != len(wanted):
        raise ValueError(
            f"person {pid} at frame {frame} has no {sorted(wanted)}: the ball anchors need that "
            "player among the kept people (players_on_field)"
        )
    return np.mean(np.array(pts), axis=0)


def anchors_from_points(spec: BallAnchorPoints) -> dict:
    """The ``anchors`` of :func:`freekiki_physics.fit_flight` for point anchors."""
    return {"points": [dict(p) for p in spec.points]}


def anchors_from_poses(spec: BallAnchorSpec, pose_rows: list[dict], n_frames: int) -> dict:
    """The ``anchors`` of :func:`freekiki_physics.fit_flight` read from the pose table."""
    foot = _pose_point(pose_rows, spec.kicker_id, 0, [spec.kicker_part])
    shoulder = _pose_point(pose_rows, spec.barrier_id, spec.barrier_frame, [spec.barrier_part])
    last = n_frames - 1 if spec.keeper_frame is None else int(spec.keeper_frame)
    hands = _pose_point(pose_rows, spec.keeper_id, last, spec.keeper_parts)
    return {
        "rest": (float(foot[0]), float(foot[1])),
        "rest_z": float(foot[2]),
        "barrier_shoulder_z": float(shoulder[2]),
        "barrier": {
            "frame": int(spec.barrier_frame),
            "xy": (float(shoulder[0]), float(shoulder[1])),
            "z": float(spec.barrier_height_m),
            "offset": float(spec.barrier_offset_m),
        },
        "keeper": {"frame": last, "xyz": tuple(float(v) for v in hands)},
    }


def flight_ball_rows(
    ball_rows: list[dict],
    cameras: dict[int, np.ndarray],
    fps: float,
    n_frames: int,
    anchors: dict,
) -> tuple[list[dict], dict]:
    """The ball of every frame from the fitted flight, and the fit as a JSON-ready dict."""
    pixels = {int(r["frame"]): (float(r["cx"]), float(r["cy"])) for r in ball_rows if r["cx"] != ""}
    projections = {f: dlt_projection_matrix(L) for f, L in cameras.items()}
    fit = fit_flight(pixels, projections, fps, n_frames, anchors=anchors)
    rows: list[dict] = []
    for f, (x, y, z) in sorted(fit["positions"].items()):
        if not np.isfinite([x, y, z]).all():
            continue
        px = pixels.get(f)
        rows.append(
            {
                "frame": f,
                "cx": round(px[0], 2) if px else "",
                "cy": round(px[1], 2) if px else "",
                "score": 1.0 if px else "",
                "x_m": x,
                "y_m": y,
                "z_m": z,
                "on_field": True,
            }
        )
    record = {k: v for k, v in fit.items() if k not in ("positions", "per_frame_px")}
    record["per_frame_px"] = {str(f): round(e, 3) for f, e in fit["per_frame_px"].items()}
    return rows, record


GOAL_LINE_M = 52.45  # the goal line (x), metres, as in soccerfield_kiki.csv
GOAL_POST_M = 3.66  # half the goal width
GOAL_CROSSBAR_M = 2.44  # the crossbar height
GOAL_OPTIONS: dict[str, tuple[float, float]] = {
    # option: (y, z) of the point on the goal line
    "goal centre, on the ground": (0.0, BALL_CENTRE_Z),
    "goal centre, at the crossbar": (0.0, GOAL_CROSSBAR_M),
    "post +y": (GOAL_POST_M, BALL_CENTRE_Z),
    "post -y": (-GOAL_POST_M, BALL_CENTRE_Z),
}
# the names of the first version (Portuguese), still accepted in the JSON and the functions
GOAL_OPTION_ALIASES = {
    "centro do gol, no chão": "goal centre, on the ground",
    "centro do gol, no travessão": "goal centre, at the crossbar",
    "trave lado +y": "post +y",
    "trave lado -y": "post -y",
}
SOURCE_ALIASES = {"bola": "ball", "jogador": "player", "gol": "goal"}


def goal_point(side: int, option: str) -> np.ndarray:
    """A point of a goal: ``side`` is -1 or +1 (the goal line at x = -52.45 or +52.45 m)."""
    option = GOAL_OPTION_ALIASES.get(option, option)
    if option not in GOAL_OPTIONS:
        raise ValueError(f"unknown goal point: {option!r} (allowed: {sorted(GOAL_OPTIONS)})")
    y, z = GOAL_OPTIONS[option]
    return np.array([float(side) * GOAL_LINE_M, y, z], dtype=float)


def anchor_sources(folder, video=None, markers_csv=None) -> dict:
    """What the anchor dialog can pick from a 3D folder.

    ``ball``: frame -> (x, y) of the ball on the ground. It is taken from ``markers_csv`` (the
    corrected track, projected through the DLT3D of each frame) when it exists, else from the
    folder's ``ball_3d.csv``. ``people``: person id -> keypoint -> frame -> (x, y, z), for the
    players kept on the pitch.
    """
    import csv as _csv

    folder = Path(folder)
    ball: dict[int, tuple[float, float]] = {}
    if markers_csv is not None and Path(markers_csv).is_file() and video is not None:
        owner = work_owner(folder, video)
        if owner is None:
            raise FileNotFoundError(f"No work/ for {folder}: the DLT3D is needed for the ball")
        runs = [p for p in sorted((owner / "work").glob("freekiki_predict_*")) if p.is_dir()]
        dlt3d = _single([f for r in runs for f in sorted(r.glob("*.dlt3d"))], "*.dlt3d")
        cameras = dlt_for_frames(dlt3d) if dlt3d is not None else {}
        n_frames = _video_info(Path(video))[0]
        for row in ball_3d(read_corrected_ball(Path(markers_csv), n_frames), cameras):
            if np.isfinite(row["x_m"]):
                ball[int(row["frame"])] = (float(row["x_m"]), float(row["y_m"]))
    else:
        with (folder / "ball_3d.csv").open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                if row["x_m"] != "":
                    ball[int(row["frame"])] = (float(row["x_m"]), float(row["y_m"]))
    pose_file = folder / "poses_3d_on_field.csv"
    if not pose_file.is_file():
        pose_file = folder / "poses_3d.csv"
    people: dict[int, dict[str, dict[int, tuple[float, float, float]]]] = {}
    if pose_file.is_file():
        with pose_file.open(newline="", encoding="utf-8") as fh:
            for row in _csv.DictReader(fh):
                pid = int(row["person_id"])
                by_kpt = people.setdefault(pid, {})
                by_kpt.setdefault(row["kpt_name"], {})[int(row["frame"])] = (
                    float(row["x_m"]),
                    float(row["y_m"]),
                    float(row["z_m"]),
                )
    return {"ball": ball, "people": people}


def source_point(
    sources: dict,
    source: str,
    frame: int,
    *,
    person: int | None = None,
    kpt: str | None = None,
    option: str | None = None,
    side: int = 1,
    z: float | None = None,
) -> np.ndarray | None:
    """The 3D point of one anchor row, or None when the source has no data at that frame."""
    source = SOURCE_ALIASES.get(source, source)
    if source == "ball":
        xy = sources["ball"].get(int(frame))
        if xy is None:
            return None
        return np.array([xy[0], xy[1], BALL_CENTRE_Z if z is None else float(z)], dtype=float)
    if source == "player":
        if person is None:
            return None
        by_kpt = sources["people"].get(int(person), {})
        if kpt == "hands":  # the mean of both wrists, the goalkeeper's hands in the fit
            wrists = [
                by_kpt.get(name, {}).get(int(frame)) for name in ("left-wrist", "right-wrist")
            ]
            if any(w is None for w in wrists):
                return None
            return np.mean(np.array(wrists, dtype=float), axis=0)
        pt = by_kpt.get(str(kpt), {}).get(int(frame))
        return None if pt is None else np.array(pt, dtype=float)
    if source == "goal":
        return goal_point(side, option or next(iter(GOAL_OPTIONS)))
    raise ValueError(f"unknown anchor source: {source!r} (ball, player or goal)")


def _project_point(P: np.ndarray, xyz) -> tuple[float, float] | None:
    """Pixel of a pitch point (metres) through a 3x4 projection, None if behind the camera."""
    h = np.asarray(P, dtype=np.float64) @ np.append(np.asarray(xyz, dtype=np.float64), 1.0)
    if not np.isfinite(h).all() or h[2] <= 1e-9:
        return None
    return float(h[0] / h[2]), float(h[1] / h[2])


def _anchor_points_record(anchors: dict) -> dict:
    """The anchor points as stored in ``ball_flight.json`` (what the report and the video draw)."""
    if "points" in anchors:
        return {
            f"p{j}": {
                "label": p["label"],
                "frame": int(p["frame"]),
                "xyz": [float(v) for v in p["xyz"]],
            }
            for j, p in enumerate(anchors["points"])
        }
    return {
        "kick_foot": {"frame": 0, "xyz": [*anchors["rest"], anchors["rest_z"]]},
        "barrier_shoulder": {
            "frame": anchors["barrier"]["frame"],
            "xyz": [*anchors["barrier"]["xy"], anchors["barrier_shoulder_z"]],
        },
        "barrier_target": {
            "frame": anchors["barrier"]["frame"],
            "xyz": [*anchors["barrier"]["xy"], anchors["barrier"]["z"]],
        },
        "keeper_hands": {
            "frame": anchors["keeper"]["frame"],
            "xyz": list(anchors["keeper"]["xyz"]),
        },
    }


def flight_check_points(anchors: dict) -> list[dict]:
    """The physical points of the fit, with the frame and the colour where they are drawn (BGR)."""
    if "points" in anchors:
        palette = [(255, 255, 0), (255, 0, 255), (0, 165, 255), (0, 255, 0), (255, 255, 255)]
        return [
            {
                "label": p["label"],
                "frame": int(p["frame"]),
                "xyz": tuple(p["xyz"]),
                "color": palette[j % len(palette)],
            }
            for j, p in enumerate(anchors["points"])
        ]
    sx, sy = anchors["barrier"]["xy"]
    b_frame = int(anchors["barrier"]["frame"])
    k_frame = int(anchors["keeper"]["frame"])
    return [
        {
            "label": "pe (frame 0)",
            "frame": 0,
            "xyz": (*anchors["rest"], anchors["rest_z"]),
            "color": (255, 255, 0),
        },  # fmt: skip
        {
            "label": "ombro",
            "frame": b_frame,
            "xyz": (sx, sy, anchors["barrier_shoulder_z"]),
            "color": (255, 0, 255),
        },  # fmt: skip
        {
            "label": "barreira 1.8 m",
            "frame": b_frame,
            "xyz": (sx, sy, anchors["barrier"]["z"]),
            "color": (0, 165, 255),
        },  # fmt: skip
        {
            "label": "maos GK",
            "frame": k_frame,
            "xyz": tuple(anchors["keeper"]["xyz"]),
            "color": (0, 255, 0),
        },  # fmt: skip
    ]


def write_flight_check_video(
    video: Path,
    out_path: Path,
    projections: dict[int, np.ndarray],
    positions: dict[int, tuple[float, float, float]],
    pixels: dict[int, tuple[float, float]],
    points: list[dict],
) -> Path:
    """The video with the marked pixels (yellow), the fitted ball (red) and the anchors.

    Each frame shows the corrected marker in yellow, the fitted 3D ball projected through that
    frame's DLT3D in red, and the physical anchors on their frames. A good fit has the red dot
    on the yellow one; a gap between them is the error of the flight at that frame.
    """
    import cv2

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        fps,
        (width, height),
    )
    try:
        for f in range(total):
            ok, frame = cap.read()
            if not ok:
                break
            if f in pixels:
                u, v = pixels[f]
                cv2.circle(frame, (round(u), round(v)), 10, (0, 255, 255), 2)
            if f in positions and f in projections:
                uv = _project_point(projections[f], positions[f])
                if uv is not None:
                    cv2.circle(frame, (round(uv[0]), round(uv[1])), 5, (0, 0, 255), -1)
            for point in points:
                if point["frame"] != f or f not in projections:
                    continue
                uv = _project_point(projections[f], point["xyz"])
                if uv is None:
                    continue
                x, y = round(uv[0]), round(uv[1])
                cv2.drawMarker(frame, (x, y), point["color"], cv2.MARKER_CROSS, 22, 2)
                cv2.putText(frame, point["label"], (x + 12, y - 8), 0, 0.6, point["color"], 2)
            cv2.putText(frame, f"frame {f}", (12, 28), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return out_path


def write_field_3d(
    run_dir: Path,
    out_dir: Path,
    *,
    cameras: dict[int, np.ndarray],
    n_frames: int,
    rate_hz: float,
    camera_csv: Path | None,
    keypoints_csv: Path | None,
    kept: set[int] | None,
    ball_rows: list[dict],
    ball_trusted: bool = False,
    ball_anchors: BallAnchorSpec | BallAnchorPoints | None = None,
    video: Path | None = None,
) -> dict:
    """Write the 3D markers in pitch metres and the C3D file. Returns a summary dict.

    A hand-corrected ball (``ball_trusted``) with ``ball_anchors`` follows the fitted flight
    (:func:`flight_ball_rows`) and writes ``ball_flight.json``. Without anchors it keeps the
    smoothed track of :func:`physical_ball_rows`.
    """
    names, _, xyz = load_field_points()
    pixels_csv = run_dir / "field_kps_getpixelvideo.csv"
    pixels = _read_field_pixels(pixels_csv) if pixels_csv.is_file() else {}
    kp_rows = field_keypoints_3d(names, xyz, cameras, pixels)
    write_table(out_dir / "field_kps_3d.csv", FIELD_KP_COLUMNS, kp_rows)

    pose_rows: list[dict] = []
    if keypoints_csv is not None and camera_csv is not None:
        boxes = {
            (frame, pid): box
            for pid, rows in _read_person_tracks(camera_csv).items()
            for frame, box in rows
        }
        pose_rows, ground_rows = _pose_rows(keypoints_csv, cameras, boxes, kept, rate_hz, n_frames)
        write_table(out_dir / "ground_tracks.csv", GROUND_COLUMNS, ground_rows)
        stature = apply_stature_scale(pose_rows)
        write_table(
            out_dir / "stature_by_person.csv", STATURE_COLUMNS, [asdict(s) for s in stature]
        )
        # The smoothed ground point is also a marker of each person ("ground"), for the C3D.
        pose_rows += [
            {
                "frame": g["frame"],
                "person_id": g["person_id"],
                "kpt_idx": GROUND_KPT_IDX,
                "kpt_name": "ground",
                "x_m": g["x_m"],
                "y_m": g["y_m"],
                "z_m": 0.0,
            }
            for g in ground_rows
        ]
        pose_name = "poses_3d_on_field.csv" if kept is not None else "poses_3d.csv"
        write_table(out_dir / pose_name, POSE_COLUMNS, pose_rows)

    ball_all = ball_3d(ball_rows, cameras)
    write_table(out_dir / "ball_3d.csv", BALL_3D_COLUMNS, ball_all)
    flight_summary: dict | None = None
    auto = None
    pixels_all = {
        int(r["frame"]): (float(r["cx"]), float(r["cy"])) for r in ball_rows if r["cx"] != ""
    }
    if ball_anchors is None and pixels_all:
        auto = auto_flight(ball_all, pixels_all, cameras, pose_rows, rate_hz, n_frames)
    if auto is not None:
        ball_on, record = auto_rows_and_record(auto, pixels_all, n_frames)
        (out_dir / "ball_flight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        flight_summary = {k: record[k] for k in ("kick_time_s", "spin_rps", "rms_px", "max_px")}
        flight_summary["anchor_residuals"] = record["anchor_residuals"]
        flight_summary["peak_speed_mps"] = record["peak_speed_mps"]
        flight_summary["arrival"] = record["arrival"]["type"] if record["arrival"] else None
        _log(f"ball flight (monocular, automatic anchors): {flight_summary}")
    elif ball_trusted and ball_anchors is not None:
        if isinstance(ball_anchors, BallAnchorPoints):
            anchors = anchors_from_points(ball_anchors)
        else:
            anchors = anchors_from_poses(ball_anchors, pose_rows, n_frames)
        ball_on, record = flight_ball_rows(ball_rows, cameras, rate_hz, n_frames, anchors)
        record["anchor_spec"] = asdict(ball_anchors)
        record["anchor_points"] = _anchor_points_record(anchors)
        (out_dir / "ball_flight.json").write_text(json.dumps(record, indent=2), encoding="utf-8")
        flight_summary = {k: record[k] for k in ("kick_time_s", "spin_rps", "rms_px", "max_px")}
        flight_summary["anchor_residuals"] = record["anchor_residuals"]
        flight_summary["peak_speed_mps"] = record["peak_speed_mps"]
        if video is not None:
            check = write_flight_check_video(
                video,
                out_dir / f"{Path(video).stem}_flight_check.mp4",
                {f: dlt_projection_matrix(L) for f, L in cameras.items()},
                {int(r["frame"]): (r["x_m"], r["y_m"], r["z_m"]) for r in ball_on},
                {
                    int(r["frame"]): (float(r["cx"]), float(r["cy"]))
                    for r in ball_rows
                    if r["cx"] != ""
                },
                flight_check_points(anchors),
            )
            flight_summary["check_video"] = check.name
        _log(f"ball flight fitted: {flight_summary}")
    else:
        ball_on = physical_ball_rows(ball_all, n_frames, rate_hz, trusted=ball_trusted)
    write_table(out_dir / "ball_3d_on_field.csv", BALL_3D_COLUMNS, ball_on)

    groups = _c3d_groups(names, kp_rows, pose_rows, ball_on, n_frames)
    markers = 0
    for group, (labels, points) in groups.items():
        write_c3d(out_dir / f"{group}.c3d", labels, points, rate_hz=rate_hz)
        write_c3d_csv(out_dir / f"{group}_c3d.csv", labels, points, rate_hz=rate_hz)
        markers += len(labels)
    return {
        "cameras": len(cameras),
        "pose_rows": len(pose_rows),
        "ball_on_field": len(ball_on),
        "ball_flight": flight_summary,
        "c3d_files": len(groups),
        "markers": markers,
    }


def detect_3d_run(
    run_dir: Path,
    video: Path,
    out_dir: Path,
    *,
    work_dir: Path | None = None,
    text: str = "person",
    weights_dir: Path | None = None,
    device: str | None = None,
    ball: bool | str = "off",
    ball_weights: Path | None = None,
    players_on_field: bool = False,
    field_min_frac: float = 0.5,
    python: str = sys.executable,
    ball_csv: Path | None = None,
) -> dict:
    """Run the 3D chain for one video of a finished detect run. Returns a summary dict.

    ``ball_csv`` is a corrected ball track in pixels (``frame,p0_x,p0_y``, as saved by
    getpixelvideo): it replaces the detector, and the physical smoothing still applies.

    The results sit flat in ``out_dir``: the CSVs, the C3D files and the full overlay
    video ``<video>_3d_overlay.mp4``. The intermediate files (SAM 3D Body output, focal
    array, Vitruvian re-fit, camera CSV) go to ``work_dir`` (default ``out_dir/work``).
    ``ball`` is ``"off"``, ``"net"`` (the dedicated ball detector) or ``"sam3"``;
    ``ball_weights`` is the file of that detector.
    """
    run_dir = Path(run_dir)
    video = Path(video)
    out_dir = Path(out_dir)
    work = Path(work_dir) if work_dir is not None else out_dir / "work"
    out_dir.mkdir(parents=True, exist_ok=True)
    work.mkdir(parents=True, exist_ok=True)
    n_frames, width, height = _video_info(video)
    fps = _video_fps(video)
    summary: dict = {"video": video.name, "run": run_dir.name, "out": str(out_dir)}
    ball_mode = _ball_mode(ball)

    dlt3d = _single(sorted(run_dir.glob("*.dlt3d")), "*.dlt3d")
    if dlt3d is None:
        focals = np.full(n_frames, np.nan)
        _log(f"{video.name}: no .dlt3d in {run_dir.name}; default intrinsics for every frame")
    else:
        focals = focal_sequence(dlt3d, n_frames, float(width))
    summary["focal"] = focal_summary(focals)
    npy = write_cam_int_npy(work / "cam_int.npy", cam_int_sequence(focals, width, height))
    _log(f"{video.name}: focal from DLT3D on {summary['focal']['valid']}/{n_frames} frames")

    sam_out = work / "sam3dinov3"
    cmd = [python, str(SAM3DINOV3), "-i", str(video), "-o", str(sam_out), "-t", text,
           "--cam-int-npy", str(npy)]  # fmt: skip
    if weights_dir is not None:
        cmd += ["--weights-dir", str(weights_dir)]
    if device:
        cmd += ["--device", device]
    _log("sam3dinov3: " + " ".join(cmd))
    proc = subprocess.run(cmd, check=False, capture_output=True, text=True)
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "").strip().splitlines()[-5:]
        raise RuntimeError(f"sam3dinov3 failed (exit {proc.returncode}): " + " | ".join(tail))
    camera_csv = _single(sorted(sam_out.rglob("*_sam3dinov3_camera.csv")), "camera csv")
    if camera_csv is None:
        raise RuntimeError(f"sam3dinov3 wrote no camera csv in {sam_out}")

    refit = vitruvian_refit(run_dir, camera_csv, work)
    summary["vitruvian"] = refit
    _log(f"{video.name}: Vitruvian re-fit {refit}")

    kept: set[int] | None = None
    if players_on_field:
        dlt2d = _single(sorted(run_dir.glob("*.dlt2d")), "*.dlt2d")
        if dlt2d is None:
            raise RuntimeError(f"No .dlt2d in {run_dir}: the pitch gate needs the 2D calibration")
        homographies = _read_homographies(dlt2d)
        tracks = _read_person_tracks(camera_csv)
        verdicts = pitch_verdicts(tracks, homographies, min_frac=field_min_frac)
        kept = {pid for pid, v in verdicts.items() if v["keep"]}
        _write_person_verdicts(out_dir / "players_on_field.csv", verdicts)
        _filter_person_csv(camera_csv, work / "players_on_field_camera.csv", kept)
        summary["players"] = {"tracks": len(verdicts), "kept": len(kept)}
        _log(f"{video.name}: players on pitch {len(kept)}/{len(verdicts)} tracks")

    ball_rows: list[dict] = []
    if ball_csv is not None:
        ball_rows = read_corrected_ball(ball_csv, n_frames)
        write_ball_csv(out_dir / "ball.csv", ball_rows)
        found = sum(1 for r in ball_rows if r["cx"] != "")
        summary["ball"] = {"mode": "corrected", "frames": len(ball_rows), "found": found}
        _log(f"{video.name}: ball from the corrected track {ball_csv.name} on {found} frames")
    elif ball_mode != "off":
        if ball_weights is None:
            raise ValueError(f"ball detection '{ball_mode}' needs its weights file")
        if ball_mode == "net":
            candidates = ball_candidates_net(video, ball_weights)
        else:
            candidates = ball_candidates_video(video, ball_weights)
        ball_rows = select_ball_rows(
            candidates, dlt_for_frames(dlt3d), work / "ball_candidates.csv"
        )
        write_ball_csv(out_dir / "ball.csv", ball_rows)
        found = sum(1 for r in ball_rows if r["cx"] != "")
        summary["ball"] = {"mode": ball_mode, "frames": len(ball_rows), "found": found}
        _log(f"{video.name}: ball ({ball_mode}) found on {found}/{len(ball_rows)} frames")

    keypoints_csv = _single(
        sorted(sam_out.rglob("*_sam3dinov3_keypoints3d.csv")), "keypoints3d csv"
    )
    write_ball_dependent_outputs(
        run_dir,
        out_dir,
        work,
        video,
        summary,
        dlt3d=dlt3d,
        n_frames=n_frames,
        fps=fps,
        camera_csv=camera_csv,
        keypoints_csv=keypoints_csv,
        kept=kept,
        ball_rows=ball_rows,
        ball_trusted=ball_csv is not None,
    )
    return summary


def write_ball_dependent_outputs(
    run_dir: Path,
    out_dir: Path,
    work: Path,
    video: Path,
    summary: dict,
    *,
    dlt3d: Path | None,
    n_frames: int,
    fps: float,
    camera_csv: Path,
    keypoints_csv: Path | None,
    kept: set[int] | None,
    ball_rows: list[dict],
    ball_trusted: bool = False,
    ball_anchors: BallAnchorSpec | BallAnchorPoints | None = None,
) -> dict:
    """Everything that depends on the ball, written into ``out_dir`` (shared by both paths).

    Used by :func:`detect_3d_run` after the sam3dinov3 run, and by :func:`rebuild_ball_3d`
    after a ball correction. The sam3dinov3 outputs and the detect run are only read.
    """
    summary["field3d"] = write_field_3d(
        run_dir,
        out_dir,
        cameras=dlt_for_frames(dlt3d),
        n_frames=n_frames,
        rate_hz=fps,
        camera_csv=camera_csv,
        keypoints_csv=keypoints_csv,
        kept=kept,
        ball_rows=ball_rows,
        ball_trusted=ball_trusted,
        ball_anchors=ball_anchors,
        video=video,
    )
    _log(f"{video.name}: 3D markers in pitch metres {summary['field3d']}")
    summary["topview"] = write_topview(out_dir, video.stem, fps=fps, n_frames=n_frames)
    _log(f"{video.name}: top view -> {summary['topview']}")
    overlay = write_3d_overlay(video, run_dir, out_dir, work, kept=kept)
    summary["overlay"] = overlay.name
    _log(f"{video.name}: overlay (field, poses, ball) -> {overlay}")
    write_run_readme(out_dir, video, summary)
    return summary


def _video_from_readme(folder: Path) -> Path | None:
    """The ``video:`` line written by :func:`write_run_readme` in a 3D folder, if present."""
    readme = folder / "README.txt"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8").splitlines():
        if line.startswith("video:"):
            path = Path(line.split(":", 1)[1].strip())
            return path if path.is_file() else None
    return None


def _work_from(folder: Path) -> Path | None:
    """The ``work from:`` folder named in a rebuilt folder's README, if any."""
    readme = folder / "README.txt"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("work from: "):
            return Path(line[len("work from: ") :].strip())
    return None


@dataclass(frozen=True)
class WorkBundle:
    """Resolved work tree for a 3D result folder (single-video or batch nested)."""

    owner: Path
    work: Path
    run_dir: Path
    players_dir: Path


def _detect_run_dir(work: Path, stem: str) -> Path | None:
    """``freekiki_predict_*`` for ``stem`` under ``work`` or a sibling batch detect folder."""
    named = sorted(p for p in work.glob(f"freekiki_predict_{stem}_*") if p.is_dir())
    if named:
        return named[-1]
    any_runs = sorted(p for p in work.glob("freekiki_predict_*") if p.is_dir())
    if len(any_runs) == 1:
        return any_runs[0]
    # Batch Detect + 3D: work is <batch>/work/<stem>; detects sit under
    # <batch>/work/processed_freekiki_batch_*/freekiki_predict_<stem>_*.
    search_roots = [work]
    if work.parent.name == "work":
        search_roots.append(work.parent)
    matches: list[Path] = []
    for root in search_roots:
        matches.extend(
            p for p in root.glob(f"processed_freekiki_*/freekiki_predict_{stem}_*") if p.is_dir()
        )
        matches.extend(p for p in root.glob(f"freekiki_predict_{stem}_*") if p.is_dir())
    unique = sorted(set(matches))
    return unique[-1] if unique else None


def locate_3d_work(folder, video=None) -> WorkBundle | None:
    """Find SAM 3D Body work + detect run for a 3D folder (flat or batch nested).

    Single-video layout: ``<stem>_3d_<ts>/work/{{sam3dinov3,freekiki_predict_*}}``.
    Batch layout: ``<batch>_3d_<ts>/<stem>/`` with intermediates in
    ``<batch>_3d_<ts>/work/<stem>/`` and detects in ``work/processed_freekiki_batch_*/``.
    """
    folder = Path(folder)
    video_path = Path(video) if video else _video_from_readme(folder)
    stem = video_path.stem if video_path is not None else folder.name

    def try_bundle(owner: Path, work: Path, players_dir: Path) -> WorkBundle | None:
        if not work.is_dir() or not (work / "sam3dinov3").is_dir():
            return None
        run_dir = _detect_run_dir(work, stem)
        if run_dir is None:
            return None
        return WorkBundle(owner=owner, work=work, run_dir=run_dir, players_dir=players_dir)

    got = try_bundle(folder, folder / "work", folder)
    if got is not None:
        return got

    parent = folder.parent
    for name in dict.fromkeys([folder.name, stem]):
        got = try_bundle(parent, parent / "work" / name, folder)
        if got is not None:
            return got

    origin = _work_from(folder)
    if origin is not None and origin.is_dir():
        got = try_bundle(origin, origin / "work", origin)
        if got is not None:
            return got
        for name in dict.fromkeys([folder.name, stem]):
            got = try_bundle(origin, origin / "work" / name, folder)
            if got is not None:
                return got

    if video_path is not None:
        for sib in sorted(folder.parent.glob(f"{stem}_3d_*"), reverse=True):
            got = try_bundle(sib, sib / "work", sib)
            if got is not None:
                return got
    return None


def work_owner(folder, video=None) -> Path | None:
    """The 3D folder whose ``work/`` holds the detect run, SAM 3D Body output and focal.

    In order: ``folder`` itself; a batch parent (``parent/work/<stem>/``); the
    ``work from:`` folder of its README; else the newest sibling ``<stem>_3d_*``
    of ``video`` that has a ``work/``. None when there is none.
    """
    folder = Path(folder)
    if (folder / "work").is_dir():
        return folder
    bundle = locate_3d_work(folder, video)
    if bundle is not None:
        return bundle.owner
    origin = _work_from(folder)
    if origin is not None and (origin / "work").is_dir():
        return origin
    if not video:
        # Batch child without a resolved video: parent/work/<folder.name> still counts.
        nested = folder.parent / "work" / folder.name
        if nested.is_dir():
            return folder.parent
        return None
    siblings = sorted(
        (f for f in folder.parent.glob(f"{Path(video).stem}_3d_*") if (f / "work").is_dir()),
        reverse=True,
    )
    if siblings:
        return siblings[0]
    nested = folder.parent / "work" / Path(video).stem
    if nested.is_dir() or (folder.parent / "work" / folder.name).is_dir():
        return folder.parent
    return None


def _attach_work(out_dir: Path, work: Path, owner: Path) -> tuple[Path, Path]:
    """Put a usable ``work/`` in ``out_dir`` and return ``(work, work_owner)``.

    Flat single-video runs store intermediates in ``owner/work/``. A ball rebuild
    used to leave only a README ``work from:`` pointer, so deleting the owner
    broke every later rebuild. For that layout, move ``work/`` into ``out_dir``
    and leave a symlink at the old path so anything still pointing at the owner
    keeps working. Batch nested trees (``owner/work/<stem>/``) stay shared via
    symlink — moving would break sibling videos.
    """
    out_dir = Path(out_dir)
    work = Path(work).resolve()
    owner = Path(owner)
    dest = out_dir / "work"
    if dest.exists():
        return dest.resolve(), out_dir
    owner_work = owner / "work"
    flat_owner_work = owner_work.is_dir() and work == owner_work.resolve()
    if flat_owner_work and owner.resolve() != out_dir.resolve():
        shutil.move(str(work), str(dest))
        try:
            owner_work.symlink_to(dest.resolve(), target_is_directory=True)
        except OSError:
            pass
        return dest.resolve(), out_dir
    try:
        dest.symlink_to(work, target_is_directory=True)
    except OSError:
        # Last resort: share by reference only (README work from: still names owner).
        return work, owner
    return dest.resolve(), out_dir if dest.is_dir() else owner


def rebuild_ball_3d(
    prev_dir: Path,
    ball_csv: Path,
    *,
    video: Path | None = None,
    out_root: Path | None = None,
    timestamp: str | None = None,
    ball_anchors: BallAnchorSpec | BallAnchorPoints | None = None,
) -> dict:
    """Rebuild only the ball-dependent outputs from a 3D folder and a corrected ball.

    ``prev_dir`` is a ``<video>_3d_<time>`` folder (or a batch per-video child
    ``<batch>_3d_<ts>/<stem>/``), or one rebuilt from another, whose ``work/`` is
    found by :func:`locate_3d_work`. That ``work/`` (detect run, SAM 3D Body output,
    focal) is attached into the new folder: moved for flat single-video owners
    (with a symlink left behind), or symlinked for batch nested trees. The new
    folder is ``<stem>_3d_<timestamp>`` next to ``prev_dir`` unless ``out_root`` is
    given; its README names the folder that owns the ``work/`` in a ``work from:``
    line, so it can be rebuilt again even if an older sibling is deleted.
    """
    prev = Path(prev_dir)
    ball_csv = Path(ball_csv)
    if not ball_csv.is_file():
        raise FileNotFoundError(f"Corrected ball CSV not found: {ball_csv}")
    with ball_csv.open(newline="", encoding="utf-8") as fh:
        header = next(csv.reader(fh), [])
    if not {"frame", "p0_x", "p0_y"} <= set(header):
        raise ValueError(f"{ball_csv} must have the columns frame,p0_x,p0_y (got {header})")

    video = Path(video) if video is not None else _video_from_readme(prev)
    if video is None or not video.is_file():
        raise FileNotFoundError("The video is not in the README; pass --video")
    bundle = locate_3d_work(prev, video)
    if bundle is None:
        missing = _work_from(prev)
        hint = ""
        if missing is not None and not missing.is_dir():
            hint = (
                f" README 'work from:' names {missing}, which is missing — "
                "restore that folder (or its work/) and retry"
            )
        raise FileNotFoundError(
            f"No work/ for {prev}: the folder has none, no batch parent "
            f"work/{video.stem}/, its 'work from:' folder has none, and no "
            f"sibling {video.stem}_3d_* folder has one.{hint}"
        )
    players_csv = bundle.players_dir / "players_on_field.csv"
    kept = _read_kept(players_csv)

    n_frames, _, _ = _video_info(video)
    fps = _video_fps(video)
    rows = read_corrected_ball(ball_csv, n_frames)

    stamp = timestamp or datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(out_root) if out_root is not None else prev.parent / f"{video.stem}_3d_{stamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    # Move/symlink work into the new folder first so outputs stay rebuildable if
    # the previous owner is deleted later.
    work, work_owner_dir = _attach_work(out_dir, bundle.work, bundle.owner)
    run_dir = _detect_run_dir(work, video.stem) or bundle.run_dir
    sam_out = work / "sam3dinov3"
    camera_csv = _single(sorted(sam_out.rglob("*_sam3dinov3_camera.csv")), "camera csv")
    if camera_csv is None:
        raise FileNotFoundError(f"No sam3dinov3 camera csv in {sam_out}")
    keypoints_csv = _single(
        sorted(sam_out.rglob("*_sam3dinov3_keypoints3d.csv")), "keypoints3d csv"
    )
    dlt3d = _single(sorted(run_dir.glob("*.dlt3d")), "*.dlt3d")
    if players_csv.is_file():
        (out_dir / "players_on_field.csv").write_bytes(players_csv.read_bytes())
    summary: dict = {
        "video": video.name,
        "rebuilt_from": str(prev),
        "work_from": str(work_owner_dir),
        "out": str(out_dir),
        "ball": {
            "mode": "corrected",
            "frames": len(rows),
            "found": sum(1 for r in rows if r["cx"] != ""),
        },
    }
    write_ball_csv(out_dir / "ball.csv", rows)
    _log(f"{video.name}: rebuilding ball outputs from {ball_csv.name} -> {out_dir}")
    return write_ball_dependent_outputs(
        run_dir,
        out_dir,
        work,
        video,
        summary,
        dlt3d=dlt3d,
        n_frames=n_frames,
        fps=fps,
        camera_csv=camera_csv,
        keypoints_csv=keypoints_csv,
        kept=kept,
        ball_rows=rows,
        ball_trusted=True,
        ball_anchors=ball_anchors,
    )


# --------------------------------------------------------------------------- #
# Full overlay video and the flat output folder
# --------------------------------------------------------------------------- #
BALL_OK_SCORE = 0.5  # ball boxes at or above this score are drawn green, the rest yellow


def read_corrected_ball(path: Path, n_frames: int) -> list[dict]:
    """Ball rows from a pixel track ``frame,p0_x,p0_y`` (blank = no ball), one per frame.

    The box is a 16 px square around the point, so ``cx, cy`` are the point itself. The
    score is 1.0: a corrected point is accepted by hand.
    """
    points: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        reader = csv.DictReader(fh)
        if "p1_x" in (reader.fieldnames or []):
            raise ValueError(
                f"{path.name} holds 49 field keypoints, not a ball track: the ball is "
                "frame,p0_x,p0_y in <video>_markers.csv, the keypoints in <video>_kiki49.csv"
            )
        for row in reader:
            x, y = row.get("p0_x", ""), row.get("p0_y", "")
            if x and y:
                points[int(float(row["frame"]))] = (float(x), float(y))
    rows: list[dict] = []
    for f in range(n_frames):
        if f in points:
            x, y = points[f]
            rows.append(ball_row(f, (1.0, np.array([x - 8, y - 8, x + 8, y + 8]))))
        else:
            rows.append(ball_row(f, None))
    return rows


def select_ball_rows(
    candidates: list[list[tuple[float, np.ndarray]]],
    cameras: dict[int, np.ndarray],
    diagnostics: Path | None = None,
) -> list[dict]:
    """One ball row per frame: the track of the candidates on the pitch.

    Off-pitch boxes (the stands, a spare ball) are dropped first, then ``track_ball``
    keeps one smooth track, so a frame with no consistent ball stays empty. With
    ``diagnostics`` every SAM 3 / ball-net box is written to that CSV with its on-pitch
    flag and whether it was chosen.
    """
    gated = pitch_ball_candidates(candidates, cameras)
    picks = track_ball([[(score, cx, cy) for score, cx, cy, _ in frame] for frame in gated])
    rows: list[dict] = []
    for t, pick in enumerate(picks):
        if pick is None:
            rows.append(ball_row(t, None))
        else:
            score, _, _, box = gated[t][pick]
            rows.append(ball_row(t, (score, box)))
    if diagnostics is not None:
        _write_ball_candidates(diagnostics, candidates, gated, picks)
    return rows


def _write_ball_candidates(
    path: Path,
    candidates: list[list[tuple[float, np.ndarray]]],
    gated: list[list[tuple[float, float, float, np.ndarray]]],
    picks: list[int | None],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "rank", "score", "x1", "y1", "x2", "y2", "on_pitch", "chosen"])
        for t, detections in enumerate(candidates):
            for rank, (score, box) in enumerate(detections):
                x1, y1, x2, y2 = (round(float(v), 2) for v in box)
                on_pitch = any(g[3] is box for g in gated[t])
                pick = picks[t]
                chosen = pick is not None and gated[t][pick][3] is box
                writer.writerow([t, rank, round(float(score), 4), x1, y1, x2, y2, on_pitch, chosen])


def _read_kept(players_csv: Path) -> set[int] | None:
    """Person ids with ``keep`` True in ``players_on_field.csv``; ``None`` when there is no gate."""
    if not players_csv.is_file():
        return None
    with players_csv.open(newline="", encoding="utf-8") as fh:
        return {int(r["person_id"]) for r in csv.DictReader(fh) if r["keep"] == "True"}


def _read_person_poses(work_dir: Path, kept: set[int] | None) -> dict[int, dict[int, np.ndarray]]:
    """Person id -> frame -> ``(K, 2)`` pixels, from the sam3dinov3 ``*_id_NN_markers.csv``."""
    poses: dict[int, dict[int, np.ndarray]] = {}
    for path in sorted(work_dir.rglob("*_markers.csv")):
        match = re.search(r"_id_(\d+)_markers\.csv$", path.name)
        if match is None:
            continue
        pid = int(match.group(1))
        if kept is None or pid in kept:
            poses[pid] = _read_field_pixels(path)  # same layout: frame, p0_x, p0_y, ...
    return poses


def _read_ball_pixels(ball_csv: Path) -> dict[int, tuple[float, float, float]]:
    """Frame -> ``(cx, cy, score)`` for the frames where the ball was found."""
    if not ball_csv.is_file():
        return {}
    out: dict[int, tuple[float, float, float]] = {}
    with ball_csv.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["cx"]:
                out[int(row["frame"])] = (float(row["cx"]), float(row["cy"]), float(row["score"]))
    return out


def _draw_field(frame, xy: np.ndarray, bones: list[tuple[int, int]]) -> None:
    import cv2

    ok: np.ndarray = np.all(np.isfinite(xy), axis=1)
    for a, b in bones:
        if a < len(ok) and b < len(ok) and ok[a] and ok[b]:
            pa, pb = (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(frame, pa, pb, (255, 255, 255), 1, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(frame, (int(xy[i, 0]), int(xy[i, 1])), 4, (0, 255, 0), -1)


def _draw_person(
    frame, pid: int, xy: np.ndarray, edges, colours, neck: int, nose: int, label: int | None = None
) -> None:
    """One person's skeleton; ``colours[i]`` is the BGR colour of keypoint ``i``.

    ``label`` is the id written beside the person (the track id ``pid`` when it is None).
    """
    import cv2

    ok: np.ndarray = np.all(np.isfinite(xy), axis=1)
    for a, b in edges:
        if ok[a] and ok[b]:
            pa, pb = (int(xy[a, 0]), int(xy[a, 1])), (int(xy[b, 0]), int(xy[b, 1]))
            cv2.line(frame, pa, pb, colours[b], 2, cv2.LINE_AA)
    for i in np.flatnonzero(ok):
        cv2.circle(frame, (int(xy[i, 0]), int(xy[i, 1])), 3, colours[i], -1)
    anchor = neck if ok[neck] else nose if ok[nose] else None
    if anchor is not None:
        x, y = int(xy[anchor, 0]), int(xy[anchor, 1])
        shown = pid if label is None else label
        cv2.putText(frame, f"ID {shown}", (x + 6, y - 6), 0, 0.5, (0, 255, 255), 1, cv2.LINE_AA)


def _draw_ball(frame, cx: float, cy: float, score: float) -> None:
    import cv2

    colour = (0, 255, 0) if score >= BALL_OK_SCORE else (0, 255, 255)
    center = (int(round(cx)), int(round(cy)))
    cv2.circle(frame, center, 10, colour, 2, cv2.LINE_AA)
    cv2.putText(frame, f"ball {score:.2f}", (center[0] + 12, center[1] - 8), 0, 0.5, colour, 1)


def write_3d_overlay(
    video: Path,
    run_dir: Path,
    out_dir: Path,
    work_dir: Path,
    *,
    kept: set[int] | None = None,
    labels: dict[int, int] | None = None,
) -> Path:
    """One MP4 with the field keypoints, the kept people's poses and the ball, frame by frame.

    ``labels`` maps a track id to the id written beside the person (the player id of a run with a
    player map: ID0 is the keeper and ID1 the kicker of a penalty); the track id otherwise.

    Reads the detect run's accepted field points, the sam3dinov3 2D markers of the kept
    people (pixels, from ``work_dir``) and ``ball.csv`` (from ``out_dir``). The video
    stops at the last frame with any data. The file is ``<video stem>_3d_overlay.mp4``.
    """
    import cv2

    # The sam3dinov3 names are imported here, with the same fallback as the package
    # imports above: freekiki.py runs as a script too, where relative imports fail.
    try:
        from .freekiki import load_bones
        from .sam3dinov3 import MHR70_NAMES, SKELETON_EDGE_NAMES, _side_color_bgr
    except ImportError:
        from freekiki import load_bones  # ty: ignore[unresolved-import]
        from sam3dinov3 import (  # ty: ignore[unresolved-import]
            MHR70_NAMES,
            SKELETON_EDGE_NAMES,
            _side_color_bgr,
        )

    field = _read_field_pixels(run_dir / "field_kps_getpixelvideo.csv")
    poses = _read_person_poses(work_dir, kept)
    balls = _read_ball_pixels(out_dir / "ball.csv")
    index = {name: i for i, name in enumerate(MHR70_NAMES)}
    edges = [(index[a], index[b]) for a, b in SKELETON_EDGE_NAMES if a in index and b in index]
    colours = [_side_color_bgr(name) for name in MHR70_NAMES]
    field_bones = load_bones()
    neck, nose = index["neck"], index["nose"]

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise OSError(f"Cannot open video: {video}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    data_frames = [
        *field,
        *balls,
        *(f for frames in poses.values() for f in frames),
    ]
    last = max(data_frames) + 1 if data_frames else total
    n_frames = min(total, last) if total else last

    out_path = Path(out_dir) / f"{Path(video).stem}_3d_overlay.mp4"
    writer = cv2.VideoWriter(
        str(out_path),
        cv2.VideoWriter_fourcc(*"mp4v"),  # ty: ignore[unresolved-attribute]
        fps,
        (width, height),
    )
    try:
        for f in range(n_frames):
            ok, frame = cap.read()
            if not ok:
                break
            if f in field:
                _draw_field(frame, field[f], field_bones)
            for pid, frames in poses.items():
                if f in frames:
                    shown = labels.get(pid, pid) if labels else None
                    _draw_person(frame, pid, frames[f], edges, colours, neck, nose, shown)
            if f in balls:
                _draw_ball(frame, *balls[f])
            cv2.putText(frame, f"frame {f}", (12, 28), 0, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        writer.release()
        cap.release()
    return out_path


def write_run_readme(out_dir: Path, video: Path, summary: dict) -> Path:
    """A short README beside the outputs: what each file is, for the next person who opens it."""
    lines = [
        "FreeKiki 3D output (vailá)",
        f"video: {video}",
        *([f"work from: {summary['work_from']}"] if summary.get("work_from") else []),
        "",
        f"{Path(video).stem}_3d_overlay.mp4  full video: field keypoints (green), people kept on",
        "                                  the pitch (skeleton, ID), ball (green = score >= 0.5)",
        "field_kps_3d.csv                   49 field points per frame, pitch metres, reprojection error",
        "poses_3d_on_field.csv             keypoints of each kept person, pitch metres",
        "players_on_field.csv              pitch gate verdict per tracked person",
        "ball.csv                          ball in pixels with score (SAM 3 or ball net)",
        "ball_3d_on_field.csv              ball on the ground plane, inside the pitch",
        "ground_tracks.csv                 each kept person's ground point: raw box base and smoothed track",
        "<video>_topview.mp4 / .png         top view: the field, each person's path and the ball's path",
        "field_ball.c3d                    field points + BALL (C3D, metres, rate = video fps)",
        "field_ball_c3d.csv                the same markers as field_ball.c3d (one row per frame)",
        "ball.c3d                          BALL only (C3D, same rate; NaN where no ball)",
        "ball_c3d.csv                      the same markers as ball.c3d (ball.csv is the pixel track)",
        "person_<id>.c3d                   one C3D per kept person (<= 255 marker names each)",
        "person_<id>_c3d.csv               the same markers as person_<id>.c3d",
        "work/                             intermediate files (SAM 3D Body, focal, re-fits)",
        "",
        f"focal valid frames: {summary.get('focal', {}).get('valid', 'n/a')}",
        f"field3d: {summary.get('field3d', {})}",
        f"players: {summary.get('players', {})}",
        f"ball: {summary.get('ball', {})}",
    ]
    path = Path(out_dir) / "README.txt"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return path


def write_ball_correction_csv(folder: Path) -> Path:
    """The ball pixel track of a 3D folder as ``frame,p0_x,p0_y`` for getpixelvideo.

    Blank cells are frames without a ball, which the marker loader leaves empty. The
    file is ``ball_for_correction.csv`` inside the folder; the corrected version comes
    back with ``detect-3d --ball-csv``.
    """
    folder = Path(folder)
    rows = []
    with (folder / "ball.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            rows.append((int(row["frame"]), row["cx"], row["cy"]))
    path = folder / "ball_for_correction.csv"
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["frame", "p0_x", "p0_y"])
        writer.writerows(rows)
    return path
