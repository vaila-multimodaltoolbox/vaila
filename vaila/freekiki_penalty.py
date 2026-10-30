"""
================================================================================
Script: freekiki_penalty.py - the penalty metrics of a run, from the data the freekiki analysis already has
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 28 October 2026
Update Date: 29 October 2026

Description:
    The penalty analysis of :mod:`pynalty_analysis` without the hand marking of the pynalty app. Every
    input comes from the reference camera's run of the freekiki analysis:

    * the four goal-mouth corners in the image: the posts of the attacked goal, projected by the
      camera's DLT in ``field_kps_3d.csv`` at the frame nearest the kick, ordered as pynalty asks
      (bottom-left, top-left, top-right, bottom-right, as seen in the image);
    * the kick frame (:func:`freekiki_auto_flight.kick_onset` on ``ball.csv``) and the ball path
      (:func:`penalty_path`): the spot at the kick, the arrival on the goal line at the last frame the
      ball is seen (the ray of that pixel in its camera), and the flight fitted between the two. The
      goal frame is the arrival frame;
    * the goalkeeper's pose: the MHR70 keypoints of its SAM track, mapped to the 33 pynalty landmarks
      (the landmarks the model does not have stay missing); its SAM box centre is the keeper point.

    A metric whose inputs are missing is left out with the reason. The pynalty thresholds (for example
    the elite dive speed) are the pynalty ones and are not changed here.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .freekiki_auto_flight import kick_onset
    from .freekiki_physics import fit_flight
    from .freekiki_pose_binocular import Keypoints
    from .pynalty_analysis import GoalGeometry, compute_penalty_metrics
    from .pynalty_report import ReportContext, write_html_reports
    from .pynalty_vision import LANDMARK_NAMES, PoseSequence, _first_movement_frame, gk_kinematics
except ImportError:
    from freekiki_auto_flight import kick_onset  # ty: ignore[unresolved-import]
    from freekiki_physics import fit_flight  # ty: ignore[unresolved-import]
    from freekiki_pose_binocular import Keypoints  # ty: ignore[unresolved-import]
    from pynalty_analysis import GoalGeometry, compute_penalty_metrics  # ty: ignore[unresolved-import]
    from pynalty_report import (  # ty: ignore[unresolved-import]
        ReportContext,
        write_html_reports,
    )
    from pynalty_vision import (  # ty: ignore[unresolved-import]
        LANDMARK_NAMES,
        PoseSequence,
        _first_movement_frame,
        gk_kinematics,
    )

GOAL_HEIGHT_M = 2.44
GK_MOVE_M = (
    0.10  # the goalkeeper has moved when the middle of his hips is this far from the kick position
)
MOVE_THRESHOLD_PX = 10.0  # operational: the keeper has moved when a shoulder or hip moves this much
MOVE_INDICES = (11, 12, 23, 24)  # left/right shoulder and hip, in the 33 landmarks
# a fitted point this far, on the ground, from the spot-arrival line is a curve
PATH_STRAIGHT_LIMIT_M = 0.30
SPOT_SIGMA_M = 0.05  # the penalty mark is fixed by the Laws of the Game: a tight anchor
# MHR70 keypoint name -> pynalty landmark name (the 33 MediaPipe landmarks)
MHR_TO_LANDMARK = {
    "nose": "nose",
    "left-eye": "left_eye",
    "right-eye": "right_eye",
    "left-ear": "left_ear",
    "right-ear": "right_ear",
    "left-shoulder": "left_shoulder",
    "right-shoulder": "right_shoulder",
    "left-elbow": "left_elbow",
    "right-elbow": "right_elbow",
    "left-wrist": "left_wrist",
    "right-wrist": "right_wrist",
    "left-index-tip": "left_index",
    "right-index-tip": "right_index",
    "left-pinky-tip": "left_pinky",
    "right-pinky-tip": "right_pinky",
    "left-thumb-tip": "left_thumb",
    "right-thumb-tip": "right_thumb",
    "left-hip": "left_hip",
    "right-hip": "right_hip",
    "left-knee": "left_knee",
    "right-knee": "right_knee",
    "left-ankle": "left_ankle",
    "right-ankle": "right_ankle",
    "left-heel": "left_heel",
    "right-heel": "right_heel",
    "left-big-toe-tip": "left_foot_index",
    "right-big-toe-tip": "right_foot_index",
}


def read_field_rows(path: Path | str) -> list[dict]:
    """The rows of ``field_kps_3d.csv`` (the field model projected by the camera, per frame)."""
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def read_boxes(path: Path | str) -> dict[int, dict[int, tuple[float, float, float, float]]]:
    """SAM box per track: ``obj_id -> frame -> (x1, y1, x2, y2)`` in pixels, from ``sam_bbox_tracks.csv``."""
    boxes: dict[int, dict[int, tuple[float, float, float, float]]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            x, y = float(row["x_px"]), float(row["y_px"])
            w, h = float(row["w_px"]), float(row["h_px"])
            boxes.setdefault(int(row["obj_id"]), {})[int(row["frame"])] = (x, y, x + w, y + h)
    return boxes


def keeper_sam_id(player_map: Path | str) -> int | None:
    """The SAM id of the track whose role is ``keeper`` in ``player_map.csv`` (None when there is none)."""
    with Path(player_map).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["role"] == "keeper":
                return int(row["sam_id"])
    return None


def goal_corners_px(
    field_rows: list[dict], goal_x: float, frame: int
) -> tuple[np.ndarray, dict] | None:
    """The four corners of the attacked goal mouth in the image, in pynalty's order.

    The posts are read at the frame of ``field_rows`` nearest to ``frame`` (the camera may move). The
    post with the smaller u is the left one in the image. None when the goal is not in the field model.
    """
    by_frame: dict[int, dict[float, dict[str, tuple[float, float]]]] = {}
    for row in field_rows:
        if "post" not in row["name"] or not row["u_px"] or not row["v_px"]:
            continue
        if abs(float(row["x_m"]) - goal_x) > 0.05:
            continue
        z = float(row["z_m"])
        if abs(z) < 0.01:
            level = "base"
        elif abs(z - GOAL_HEIGHT_M) < 0.01:
            level = "top"
        else:
            continue
        posts = by_frame.setdefault(int(row["frame"]), {})
        posts.setdefault(round(float(row["y_m"]), 2), {})[level] = (
            float(row["u_px"]),
            float(row["v_px"]),
        )
    complete = {
        f: [p for p in posts.values() if {"base", "top"} <= set(p)] for f, posts in by_frame.items()
    }
    usable = [f for f, posts in complete.items() if len(posts) == 2]
    if not usable:
        return None
    chosen = min(usable, key=lambda f: (abs(f - frame), f))
    left, right = sorted(complete[chosen], key=lambda p: p["base"][0])
    corners = np.array([left["base"], left["top"], right["top"], right["base"]], dtype=float)
    return corners, {"frame": chosen, "left_post_px": left, "right_post_px": right}


def arrival_on_goal_line(
    P: np.ndarray, uv: tuple[float, float], goal_x: float
) -> np.ndarray | None:
    """Where the ray of one pixel meets the goal line ``x = goal_x``, as pitch metres ``(x, y, z)``.

    ``P`` is the 3x4 projection of the frame (pitch metres -> pixels). With ``x`` fixed, the pixel gives
    two linear equations in ``y`` and ``z``. None when the ray is parallel to the line or the meeting
    point is behind the camera.
    """
    p = np.asarray(P, dtype=float)
    u, v = float(uv[0]), float(uv[1])
    a = np.array(
        [
            [u * p[2, 1] - p[0, 1], u * p[2, 2] - p[0, 2]],
            [v * p[2, 1] - p[1, 1], v * p[2, 2] - p[1, 2]],
        ]
    )
    b = -np.array(
        [
            (u * p[2, 0] - p[0, 0]) * goal_x + (u * p[2, 3] - p[0, 3]),
            (v * p[2, 0] - p[1, 0]) * goal_x + (v * p[2, 3] - p[1, 3]),
        ]
    )
    if abs(np.linalg.det(a)) < 1e-9:
        return None
    y, z = np.linalg.solve(a, b)
    point = np.array([goal_x, y, z], dtype=float)
    depth = p @ np.append(point, 1.0)
    if not np.isfinite(depth).all() or depth[2] <= 1e-9:
        return None
    return point


def penalty_path(
    pixels: dict[int, tuple[float, float]],
    cameras: dict[int, np.ndarray],
    *,
    kick: int,
    goal_x: float,
    fps: float,
    n_frames: int,
    track: dict[int, np.ndarray] | None = None,
) -> dict:
    """The ball path of a penalty: from the spot to the goal line, bent only where the flight is.

    With ``track`` (frame -> the ball's 3D position in metres, from the binocular fit of two cameras)
    the verdict is that track, the validated one. Without it, the flight of
    :func:`freekiki_physics.fit_flight` (gravity, drag and Magnus spin) is fitted to ``pixels`` (frame ->
    the ball's pixel, seen by ``cameras``, frame -> the 3x4 projection of that frame), with one anchor:
    the spot where the ball rests before the kick (the penalty mark, known exactly, at the first frame of
    the record). That verdict is provisional: a monocular track near the limit can flip with the model.
    Either way the goal frame is the first frame in which the ball is across the goal line, when the clip
    gets there, and the arrival is the ball's point of that frame. The path is straight when no point of
    the flight is further than ``PATH_STRAIGHT_LIMIT_M``, on the ground, from the segment spot-arrival;
    ``bow_side`` is the side the path bows to, seen by the kicker facing the goal. The vertical arc of a
    lob is reported as ``apex_z_m`` and is not a curve.

    The ray of the last pixel, met with the goal line in its camera, is reported as ``ray_arrival_m``
    for comparison only: the ball is not on the goal line in that frame in general, so the ray is not an
    anchor (a real penalty compared with the binocular fit showed the ray 2 m off the fitted ball).
    """
    if track is None:
        # the fit indexes the clip's frames: rows past the clip (a longer ball.csv) are left out
        pixels = {f: uv for f, uv in pixels.items() if f < n_frames}
        cameras = {f: P for f, P in cameras.items() if f < n_frames}
        seen = sorted(f for f in pixels if f in cameras)
        if not seen or seen[-1] <= kick:
            return {
                "available": False,
                "reason": "the ball is not seen after the kick with a camera",
            }
        rest, last = seen[0], seen[-1]  # the ball rests on the spot before the kick
    else:
        frames = sorted(f for f in track if f < n_frames)
        if not frames or frames[-1] <= kick:
            return {"available": False, "reason": "the binocular ball is not seen after the kick"}
        rest, last = frames[0], frames[-1]
    goal = GoalGeometry()
    sign = math.copysign(1.0, goal_x)
    spot = np.array([goal_x - sign * goal.penalty_distance, 0.0, goal.ball_radius], dtype=float)
    ray = None
    if last in pixels and last in cameras:
        ray = arrival_on_goal_line(cameras[last], pixels[last], goal_x)
    if track is None:
        anchors = {"points": [{"frame": rest, "xyz": spot.tolist(), "sigma": SPOT_SIGMA_M}]}
        # the contact is anywhere before the last frame; a window around the first visible movement
        # failed on a real clip, where the hand-marked ball drifts a few frames before the kick
        window = (0.0, (last - 1) / fps)
        fit = fit_flight(pixels, cameras, fps, n_frames, anchors=anchors, kick_window=window)
        positions = {f: np.asarray(fit["positions"][f], dtype=float) for f in range(n_frames)}
        kick_time_s = fit["kick_time_s"]
        source = "monocular fit"
        fit_summary = {
            "rms_px": fit["rms_px"],
            "kick_time_s": fit["kick_time_s"],
            "spin_rps": fit["spin_rps"],
            "success": fit["success"],
        }
    else:
        positions = {f: np.asarray(p, dtype=float) for f, p in track.items() if f < n_frames}
        kick_time_s = kick / fps
        source = "binocular"
        fit_summary = {
            "rms_px": None,
            "kick_time_s": round(kick_time_s, 4),
            "spin_rps": None,
            "success": None,
        }
    across = [f for f in sorted(positions) if f >= rest and sign * positions[f][0] >= abs(goal_x)]
    goal_frame = across[0] if across else None
    arrival = positions[goal_frame] if goal_frame is not None else positions[last]
    # how far the fit had to move the spot to meet the pixels at rest (calibration or ball placement)
    spot_offset = float(np.linalg.norm(positions[rest] - spot))
    # the bend is measured on the ground: the gravity arc of a lob is not curvature (no spin, no swerve)
    ground = arrival[:2] - spot[:2]
    length_sq = float(np.dot(ground, ground))
    if length_sq < 1e-6:
        return {"available": False, "reason": "the arrival is at the spot"}
    # the kicker's left, on the ground
    left = np.array([-ground[1], ground[0]]) / math.sqrt(length_sq)
    rows = []
    for f in range(kick, last + 1):
        if f not in positions:
            continue
        point = positions[f]
        t = float(np.clip(np.dot(point[:2] - spot[:2], ground) / length_sq, 0.0, 1.0))
        offset = point[:2] - (spot[:2] + t * ground)
        rows.append(
            {
                "frame": f,
                "x_m": round(float(point[0]), 4),
                "y_m": round(float(point[1]), 4),
                "z_m": round(float(point[2]), 4),
                "horiz_dev_m": round(float(np.linalg.norm(offset)), 4),
                "bow_left_m": round(float(np.dot(offset, left)), 4),
            }
        )
    # the bend is the flight's: the frames before the contact sit on the spot
    flight = [r for r in rows if r["frame"] / fps >= kick_time_s] or rows
    worst = max(flight, key=lambda r: r["horiz_dev_m"])
    max_dev = float(worst["horiz_dev_m"])
    shape = "straight" if max_dev <= PATH_STRAIGHT_LIMIT_M else "curved"
    bow_side = "none"
    if shape == "curved":
        bow_side = "left" if worst["bow_left_m"] > 0 else "right"
    inside = (
        goal_frame is not None
        and abs(arrival[1]) <= goal.width / 2
        and 0.0 <= arrival[2] <= goal.height
    )
    return {
        "available": True,
        "reason": "",
        "source": source,
        "kick_frame": int(kick),
        "last_frame": int(last),
        "goal_frame": goal_frame,
        "spot_m": [round(float(v), 3) for v in spot],
        "arrival_m": [round(float(v), 3) for v in arrival],
        "ray_arrival_m": None if ray is None else [round(float(v), 3) for v in ray],
        "arrival_in_goal_mouth": bool(inside),
        "shape": shape,
        "bow_side": bow_side,
        "max_dev_m": round(max_dev, 3),
        "apex_z_m": round(max(r["z_m"] for r in flight), 3),
        "straight_limit_m": PATH_STRAIGHT_LIMIT_M,
        "spot_fit_offset_m": round(spot_offset, 3),
        "fit": fit_summary,
        "rows": rows,
    }


def landmark_array(kps: np.ndarray, names: list[str]) -> np.ndarray:
    """One frame of MHR70 keypoints ``(70, 2)`` as the 33 pynalty landmarks ``(33, 3)``: x, y, visibility."""
    arr = np.full((len(LANDMARK_NAMES), 3), np.nan)
    index = {name: k for k, name in enumerate(names)}
    for mhr, landmark in MHR_TO_LANDMARK.items():
        k = index.get(mhr)
        if k is None:
            continue
        x, y = kps[k]
        if np.isfinite(x) and np.isfinite(y):
            arr[LANDMARK_NAMES.index(landmark)] = (x, y, 1.0)
    return arr


def pose_sequence(
    keypoints: Keypoints,
    sam_id: int,
    boxes: dict[int, tuple[float, float, float, float]],
) -> PoseSequence:
    """The keeper's pose over the frames it has keypoints, with its SAM boxes."""
    arr = keypoints.by_person.get(sam_id)
    landmarks: dict[int, np.ndarray] = {}
    if arr is not None:
        for f in range(len(arr)):
            if np.isfinite(arr[f]).any():
                landmarks[f] = landmark_array(arr[f], keypoints.names)
    frames = sorted(landmarks)

    def int_box(box: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
        x1, y1, x2, y2 = (int(round(v)) for v in box)
        return x1, y1, x2, y2

    kept = {f: int_box(boxes[f]) for f in frames if f in boxes}
    return PoseSequence(label=f"sam {sam_id}", frames=frames, landmarks=landmarks, boxes=kept)


def _centre(box: tuple[float, float, float, float]) -> tuple[float, float]:
    return (box[0] + box[2]) / 2.0, (box[1] + box[3]) / 2.0


def gk_partial(
    points: dict[str, np.ndarray], *, kick: int, last: int, fps: float, goal_x: float
) -> dict:
    """The goalkeeper's move from the kick to the last frame, without the goal line: a partial measure.

    ``points`` are the keeper's 3D joints per frame in metres (pitch frame, ``(n, 3)`` each). The move
    starts at the first frame in which the middle of the hips is ``GK_MOVE_M`` away, on the ground, from
    where it was at the kick. ``toward_right_m`` is the keeper's displacement at the last frame towards
    his own right (he faces the kicker, so it is the kicker's left); ``direction`` names that side, as
    the goalkeeper sees it. It is partial: a dive is only measured up to
    the goal line, and the clip may end before it, so the dive itself is not reported here. The dive kinematics this move
    is compared with in the literature: Monteiro et al. (2022, Scientific Reports, doi:10.1038/s41598-022-21508-6)
    and Monteiro et al. (2024, Scientific Reports, doi:10.1038/s41598-024-60074-x) on goalkeepers' penalty dives;
    Navarro et al. (2013, J Sports Sci, doi:10.1080/02640414.2012.762602) on the goalkeeper's presence and the kick.
    """
    left, right = points.get("left-hip"), points.get("right-hip")
    if left is None or right is None:
        return {"available": False, "reason": "the goalkeeper's hips are not in the poses"}
    mid = (left + right) / 2.0
    if not (0 <= kick <= last < len(mid)) or not np.isfinite(mid[kick]).all():
        return {"available": False, "reason": "no goalkeeper pose at the kick"}
    if not np.isfinite(mid[last]).all():
        return {"available": False, "reason": "no goalkeeper pose at the last frame"}
    start = mid[kick]
    move = np.hypot(mid[:, 0] - start[0], mid[:, 1] - start[1])
    base = {"partial": True, "kick_frame": int(kick), "last_frame": int(last)}
    moved = [f for f in range(kick, last + 1) if np.isfinite(move[f]) and move[f] >= GK_MOVE_M]
    if not moved:
        reason = f"the goalkeeper stays within {GK_MOVE_M:.2f} m of the kick position"
        return {**base, "available": True, "moved": False, "reason": reason}
    sign = math.copysign(1.0, goal_x)
    toward_right = sign * float(mid[last, 1] - start[1])  # the keeper's right, facing the kicker
    elapsed = (last - kick) / fps
    return {
        **base,
        "available": True,
        "moved": True,
        "reason": "",
        "onset_frame": int(moved[0]),
        "onset_ms_after_kick": round((moved[0] - kick) / fps * 1000.0, 1),
        "toward_right_m": round(toward_right, 3),
        "direction": "right" if toward_right > 0 else "left",
        "displacement_m": round(float(move[last]), 3),
        "mean_speed_mps": round(float(move[last]) / elapsed, 2) if elapsed > 0 else None,
    }


def _unavailable(reason: str, goal_x: float) -> dict:
    return {"available": False, "reason": reason, "goal_x_m": goal_x}


def penalty_metrics(
    *,
    field_rows: list[dict],
    ball_px: dict[int, tuple[float, float]],
    cameras: dict[int, np.ndarray] | None,
    keypoints: Keypoints,
    keeper_boxes: dict[int, tuple[float, float, float, float]],
    keeper_sam: int | None,
    fps: float,
    n_frames: int,
    goal_x: float,
    track: dict[int, np.ndarray] | None = None,
) -> dict:
    """The penalty metrics of one run (pynalty's keys), or the reason they cannot be computed.

    ``track``: the binocular ball in metres per frame when the run has two cameras; it sets the
    verdict of the ball path (see :func:`penalty_path`). Without it the path is the monocular fit.

    ``ball_px``: the reference camera's ball (``ball.csv``); ``cameras``: frame -> the 3x4 projection of
    the reference camera (its per-frame DLT3D), needed by the ball path; ``n_frames``: the clip length;
    ``goal_x``: the attacked goal line.
    """
    kick = kick_onset(ball_px)
    if kick is None:
        return _unavailable("the ball does not move in this camera", goal_x)
    corners = goal_corners_px(field_rows, goal_x, kick)
    if corners is None:
        return _unavailable("the goal posts are not in the field model of this camera", goal_x)
    calib, posts = corners
    if cameras is None and track is None:
        return _unavailable("no per-frame camera (DLT3D) of this run for the ball path", goal_x)
    path = penalty_path(
        ball_px, cameras or {}, kick=kick, goal_x=goal_x, fps=fps, n_frames=n_frames, track=track
    )
    if not path["available"]:
        return _unavailable(path["reason"], goal_x)
    if path["goal_frame"] is None:  # the ball path still stands: it is written and reported
        result = _unavailable(
            "the fitted flight does not reach the goal line within the clip "
            f"(last frame {path['last_frame']}: the ball is short of the goal)",
            goal_x,
        )
        result["path"] = path
        return result
    goal = path["goal_frame"]
    if keeper_sam is None:
        return _unavailable("no goalkeeper in the player map", goal_x)
    seq = pose_sequence(keypoints, keeper_sam, keeper_boxes)
    if seq.is_empty():
        return _unavailable("the goalkeeper has no keypoints in this camera", goal_x)
    if kick not in keeper_boxes or goal not in keeper_boxes:
        return _unavailable(
            "the goalkeeper box is missing at the kick or at the goal frame", goal_x
        )
    if kick not in ball_px or goal not in ball_px:
        return _unavailable("the ball is not seen at the kick or at the goal frame", goal_x)

    move = _first_movement_frame(seq, MOVE_INDICES, kick, MOVE_THRESHOLD_PX)
    body = seq.landmarks.get(goal)
    body_px, body_names = None, None
    if body is not None:
        finite = np.isfinite(body[:, :2]).all(axis=1)
        if finite.any():
            body_px = body[finite, :2]
            body_names = [name for name, ok in zip(LANDMARK_NAMES, finite, strict=True) if ok]

    metrics = compute_penalty_metrics(
        calib_pixels=calib,
        kick_frame=kick,
        goal_frame=goal,
        gk_move_frame=move if move is not None else goal,
        kick_ball_px=ball_px[kick],
        kick_gk_px=_centre(keeper_boxes[kick]),
        goal_ball_px=ball_px[goal],
        goal_gk_px=_centre(keeper_boxes[goal]),
        fps=fps,
        goal=GoalGeometry(),
        gk_body_points_px=body_px,
        gk_body_landmark_names=body_names,
    )
    dive = gk_kinematics(seq, kick, goal, fps, calib_pixels=calib, goal=GoalGeometry())
    return {
        "available": True,
        "reason": "",
        "goal_x_m": goal_x,
        "kick_frame": kick,
        "goal_frame": goal,
        "gk_move_frame": move,
        "gk_move_detected": move is not None,
        "calibration_px": calib.tolist(),
        "posts": posts,
        "metrics": metrics,
        "gk_dive": dive,
        "path": path,
    }


def _flat(prefix: str, value, rows: list[tuple[str, object]]) -> None:
    if isinstance(value, dict):
        for key, inner in value.items():
            _flat(f"{prefix}.{key}" if prefix else str(key), inner, rows)
    elif isinstance(value, (list, tuple)):
        rows.append((prefix, ";".join(str(v) for v in value)))
    elif isinstance(value, float) and not math.isfinite(value):
        rows.append((prefix, ""))
    elif isinstance(value, (np.generic,)):
        rows.append((prefix, value.item()))
    else:
        rows.append((prefix, value))


def write(data_dir: Path | str, result: dict) -> dict[str, Path]:
    """``penalty_metrics.csv`` (one row per value, the reason first when unavailable), ``.json``, and
    ``penalty_path.csv`` (the ball path frame by frame, when there is one)."""
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    path = result.get("path") or {}
    summary = {k: v for k, v in result.items() if k != "path"}
    if path.get("available"):
        summary["path"] = {k: v for k, v in path.items() if k != "rows"}  # rows: penalty_path.csv
    rows: list[tuple[str, object]] = []
    _flat("", summary, rows)
    csv_path = folder / "penalty_metrics.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.writer(fh)
        writer.writerow(["metric", "value"])
        writer.writerows(rows)
    json_path = folder / "penalty_metrics.json"

    def plain(value):
        if isinstance(value, dict):
            return {k: plain(v) for k, v in value.items()}
        if isinstance(value, (list, tuple, np.ndarray)):
            return [plain(v) for v in value]
        if isinstance(value, np.generic):
            return value.item()
        if isinstance(value, float) and not math.isfinite(value):
            return None
        return value

    json_path.write_text(json.dumps(plain(result), indent=2), encoding="utf-8")
    out: dict[str, Path] = {"csv": csv_path, "json": json_path}
    path_csv = folder / "penalty_path.csv"
    if path.get("rows"):
        with path_csv.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=list(path["rows"][0]))
            writer.writeheader()
            writer.writerows(path["rows"])
        out["path_csv"] = path_csv
    else:
        path_csv.unlink(missing_ok=True)  # a path of an earlier run must not outlive this result
    return out


def report(out_dir: Path | str, video: Path | str, fps: float, result: dict) -> dict[str, str]:
    """The pynalty HTML report of an available result (``report.html`` and ``report_pt.html``)."""
    if not result.get("available"):
        return {}
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    ctx = ReportContext(
        video_path=str(video),
        fps=float(fps),
        metrics=result["metrics"],
        goal=GoalGeometry(),
        pose_metrics=result.get("gk_dive") or {},
    )
    return write_html_reports(str(folder), ctx, lang="both")
