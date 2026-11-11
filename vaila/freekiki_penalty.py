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
Version: 0.4.15
Created: 28 October 2026
Update Date: 09 November 2026

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

    A save (``save_frame``, the keeper's hands stop the ball): the ball never reaches the goal line,
    so the metrics are taken from the 3D flight of the run (``ball_flight.json``, anchored on the
    keeper's hands) instead of the goal-plane pixels: the interception point and time, the mean
    speed from the spot to it, and the would-be crossing of the goal line had the keeper not
    touched the ball (:func:`extrapolate_crossing`: the same fitted flight, continued), which sets
    the placement (zone, posts, crossbar, HWR accuracy of Attaallah et al. 2021). The kick is the
    flight's contact, which a panning camera does not move.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .freekiki_auto_flight import kick_onset
    from .freekiki_physics import fit_flight, simulate_flight
    from .freekiki_pose_binocular import Keypoints
    from .pynalty_analysis import (
        Anthropometrics,
        GoalGeometry,
        compute_penalty_metrics,
        fit_ball_flight_3d,
        goal_zone,
        hwr_accuracy_pct,
    )
    from .pynalty_report import ReportContext, write_html_reports
    from .pynalty_vision import LANDMARK_NAMES, PoseSequence, _first_movement_frame, gk_kinematics
except ImportError:
    from freekiki_auto_flight import kick_onset  # ty: ignore[unresolved-import]
    from freekiki_physics import (  # ty: ignore[unresolved-import]
        fit_flight,
        simulate_flight,
    )
    from freekiki_pose_binocular import Keypoints  # ty: ignore[unresolved-import]
    from pynalty_analysis import (  # ty: ignore[unresolved-import]
        Anthropometrics,
        GoalGeometry,
        compute_penalty_metrics,
        fit_ball_flight_3d,
        goal_zone,
        hwr_accuracy_pct,
    )
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
    ys = {id(p): y for y, p in by_frame[chosen].items()}
    left, right = sorted(complete[chosen], key=lambda p: p["base"][0])
    corners = np.array([left["base"], left["top"], right["top"], right["base"]], dtype=float)
    return corners, {
        "frame": chosen,
        "left_post_px": left,
        "right_post_px": right,
        "left_post_y_m": float(ys[id(left)]),
        "right_post_y_m": float(ys[id(right)]),
    }


def goal_plane_xz(y_m: float, z_m: float, posts: dict) -> tuple[float, float]:
    """A pitch point's ``(y, z)`` as pynalty's goal-plane ``(x, z)``: x from the post that is on the
    left in the image (:func:`goal_corners_px`) toward the other post, z up."""
    left, right = float(posts["left_post_y_m"]), float(posts["right_post_y_m"])
    direction = 1.0 if right > left else -1.0
    return (float(y_m) - left) * direction, float(z_m)


def extrapolate_crossing(
    theta,
    goal_x: float,
    *,
    t_from: float,
    horizon_s: float = 1.5,
    dt: float = 1e-3,
) -> dict | None:
    """Where the fitted flight ``theta`` would cross the goal line ``x = goal_x`` after ``t_from``.

    The same flight model (gravity, drag, Magnus) continued past the save, as if the keeper had not
    touched the ball. Returns ``{"y_m", "z_m", "time_s"}`` (time from the clip start, like the
    flight's ``kick_time_s``), or None when the ball does not reach the line within ``horizon_s``.
    """
    times = np.arange(float(t_from), float(t_from) + horizon_s, dt)
    pos = simulate_flight(np.asarray(theta, dtype=float), times, GoalGeometry().ball_radius)
    sign = math.copysign(1.0, goal_x)
    line = abs(goal_x)
    for i in range(1, len(times)):
        a, b = sign * pos[i - 1, 0], sign * pos[i, 0]
        if np.isfinite(a) and np.isfinite(b) and a < line <= b:
            w = (line - a) / (b - a)
            point = pos[i - 1] + w * (pos[i] - pos[i - 1])
            return {
                "y_m": float(point[1]),
                "z_m": float(point[2]),
                "time_s": float(times[i - 1] + w * dt),
            }
    return None


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
    track_source: str = "binocular",
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
        source = track_source
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
    flight: dict | None = None,
    flight_track: dict[int, np.ndarray] | None = None,
    save_frame: int | None = None,
    shot_outcome: str | None = None,
    keeper_points: dict[str, np.ndarray] | None = None,
    gk_move_frame: int | None = None,
    anthro: Anthropometrics | None = None,
    alternatives: list[dict] | None = None,
) -> dict:
    """The penalty metrics of one run (pynalty's keys), or the reason they cannot be computed.

    ``flight`` is the run's ``ball_flight.json`` and ``flight_track`` its ball in metres per frame
    (``ball_3d_on_field.csv``): the kick is then the flight's contact (a pan does not move it).
    ``save_frame`` (or ``flight["save_frame"]``) with ``shot_outcome`` ``save``: the keeper's hands
    stop the ball in that frame, and the metrics come from the 3D flight (:func:`save_metrics`).
    ``keeper_points`` (``kpt_name -> (n, 3)`` metres) place the keeper on the goal plane;
    ``gk_move_frame`` is his movement onset when known (a hand-marked or 3D onset); ``anthro`` the
    measured body sizes; ``alternatives`` the flights refitted with the keeper's anchor moved along
    the camera ray (``{"shift_m", "theta"}``), whose spread is the uncertainty of the save.

    ``track``: the binocular ball in metres per frame when the run has two cameras; it sets the
    verdict of the ball path (see :func:`penalty_path`). Without it the path is the monocular fit.

    ``ball_px``: the reference camera's ball (``ball.csv``); ``cameras``: frame -> the 3x4 projection of
    the reference camera (its per-frame DLT3D), needed by the ball path; ``n_frames``: the clip length;
    ``goal_x``: the attacked goal line.
    """
    if save_frame is None and flight is not None and flight.get("save_frame") is not None:
        save_frame = int(flight["save_frame"])
    if shot_outcome is None and save_frame is not None:
        shot_outcome = "save"
    if save_frame is not None and shot_outcome == "save":
        if flight is None or flight_track is None:
            return _unavailable(
                "a save needs the run's 3D flight (ball_flight.json and ball_3d_on_field.csv)",
                goal_x,
            )
        return save_metrics(
            field_rows=field_rows,
            ball_px=ball_px,
            fps=fps,
            n_frames=n_frames,
            goal_x=goal_x,
            flight=flight,
            flight_track=flight_track,
            save_frame=int(save_frame),
            keeper_points=keeper_points,
            gk_move_frame=gk_move_frame,
            anthro=anthro,
            alternatives=alternatives,
        )
    kick = None
    if flight is not None and flight.get("contact_frame") is not None:
        kick = int(flight["contact_frame"])  # the flight's contact: a panning camera cannot move it
    if kick is None:
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


def _hips_mid(points: dict[str, np.ndarray] | None, frame: int) -> np.ndarray | None:
    """The middle of the keeper's hips at ``frame`` (metres), or None."""
    if not points or "left-hip" not in points or "right-hip" not in points:
        return None
    if not 0 <= frame < len(points["left-hip"]):
        return None
    mid = (points["left-hip"][frame] + points["right-hip"][frame]) / 2.0
    return mid if np.isfinite(mid).all() else None


def _would_be(
    theta, goal_x: float, t_from: float, posts: dict | None, goal: GoalGeometry
) -> dict | None:
    """The would-be crossing of the goal line, with its placement on pynalty's goal plane."""
    crossing = extrapolate_crossing(theta, goal_x, t_from=t_from)
    if crossing is None:
        return None
    y, z = crossing["y_m"], crossing["z_m"]
    inside = abs(y) <= goal.width / 2 and 0.0 <= z <= goal.height
    out = {
        "y_m": round(y, 4),
        "z_m": round(z, 4),
        "time_s": round(crossing["time_s"], 4),
        "inside_goal_mouth": bool(inside),
        "dist_to_nearest_post_m": round(goal.width / 2 - abs(y), 3),
        "dist_to_crossbar_m": round(goal.height - z, 3),
        "hwr_accuracy_pct": round(hwr_accuracy_pct(y, z, goal), 1),
    }
    if posts is not None:
        gx, gz = goal_plane_xz(y, z, posts)
        out["goal_plane_xz"] = [round(gx, 4), round(gz, 4)]
        out["zone_label"] = goal_zone(gx, gz, goal)[3]
    else:
        out["zone_label"] = ""
    return out


# The uncertainty of the one mean speed of a saved penalty (see :func:`speed_summary`): each end of the
# flight time is placed to a quarter of a frame, and the spot-to-hands distance to 0.3 m (the depth
# of a single camera, the weak axis).
SIGMA_KICK_FRAMES = 0.25
SIGMA_SAVE_FRAMES = 0.25
SIGMA_DISTANCE_M = 0.3
PIXEL_CONTACT_TOLERANCE_FRAMES = 0.25


def contact_from_pixels(ball_px: dict[int, tuple[float, float]], first_moved: int) -> dict | None:
    """The kick instant, as a fractional frame, from the marked ball pixels alone.

    No 3D, no flight model: the ball is still until the kick and then moves in the image at nearly
    the same number of pixels per frame, so the distance it covers by the frames after the first
    moved one, extrapolated back, reaches zero at the kick. The rest pixel drifts with a panning
    camera: that drift is taken from the three frames before the kick and removed.

    ``first_moved`` is the first frame with the ball already moved (``--contact-frame``). Returns
    ``{"frame", "low", "high", "variants"}`` (the median of four extrapolations and their range), or
    None when the rest (4 frames before) or the flight (5 frames from ``first_moved``) is not marked.
    """
    f1 = int(first_moved)
    if any(f not in ball_px for f in range(f1 - 4, f1 + 5)):
        return None
    last = np.asarray(ball_px[f1 - 1], dtype=float)
    drift = (last - np.asarray(ball_px[f1 - 4], dtype=float)) / 3.0

    def covered(f: int) -> float:
        return float(
            np.linalg.norm(np.asarray(ball_px[f], dtype=float) - (last + drift * (f - (f1 - 1))))
        )

    frames_sets = [(range(f1 + 1, f1 + 4), 1), (range(f1 + 1, f1 + 5), 2), (range(f1, f1 + 5), 2)]
    if f1 + 5 in ball_px:
        frames_sets.append((range(f1 + 1, f1 + 6), 2))
    found = []
    for frames, degree in frames_sets:
        xs = np.array(list(frames), dtype=float)
        coeffs = np.polyfit(xs, [covered(int(f)) for f in xs], degree)
        roots = np.roots(coeffs)
        roots = roots[np.isreal(roots)].real
        if roots.size:
            found.append(float(roots[np.argmin(np.abs(roots - f1))]))
    if not found:
        return None
    return {
        "frame": round(float(np.median(found)), 3),
        "low": round(min(found), 3),
        "high": round(max(found), 3),
        "variants": len(found),
    }


def _path_length(theta, t_from: float, t_to: float, radius: float, step: float = 1e-3) -> float:
    """Length of the modelled flight between two times (metres)."""
    times = np.append(np.arange(t_from, t_to, step), t_to)
    pos = simulate_flight(np.asarray(theta, dtype=float), times, radius)
    return float(np.linalg.norm(np.diff(pos, axis=0), axis=1).sum())


def speed_summary(
    *,
    fps: float,
    kick_time: float,
    contact: int,
    save_frame: int,
    theta,
    flight: dict,
    distance: float,
    to_line: float | None,
    goal: GoalGeometry,
    ball_px: dict[int, tuple[float, float]],
    alternative_distances: list[float] | None = None,
    alternative_thetas: list | None = None,
) -> dict:
    """One definition of the speed and the time of a saved penalty, with what it rests on.

    * **Time base:** ``fps`` and the frame period. The kick is the fit's sub-frame contact
      (``kick.frame``, a fraction of a frame), checked against the marked pixels alone
      (``kick.pixel_check``); ``kick.counted_frame`` is the whole frame a viewer sees at the ball.
      The save is a frame. The flight time is their difference, with the whole-frame count next to it.
    * **Mean speed** = the spot-to-hands distance over the flight time, the ball speed of Navarro et al.
      2013 (distance over the time from foot-ball contact to arrival). Its uncertainty puts each end of
      the time at a quarter of a frame and the distance at 0.3 m (more when the refits along the camera
      ray spread wider).
    * **Exit and at-the-save speeds** come from the fitted flight model (drag and spin): estimates,
      not measurements. ``exit_model.range_kmh`` is their spread over the refits along the camera ray.
    """
    t_save = save_frame / fps
    flight_time = t_save - kick_time
    kick_frame = kick_time * fps
    counted_frame = int(np.floor(round(kick_frame, 2)))
    counted_frames = int(save_frame) - counted_frame
    mean_ms = distance / flight_time
    counted_ms = distance / (counted_frames / fps) if counted_frames > 0 else None
    path_m = _path_length(theta, kick_time, t_save, goal.ball_radius)
    # the distance is uncertain by at least the camera's depth, and by as much as the definitions
    # differ (the mark to the hands, or the fitted flight from its own rest point) or the refits along
    # the camera ray spread
    sigma_d = max(SIGMA_DISTANCE_M, abs(path_m - distance))
    if alternative_distances:
        spread = [distance, *alternative_distances]
        sigma_d = max(sigma_d, (max(spread) - min(spread)) / 2.0)
    sigma_t = float(np.hypot(SIGMA_KICK_FRAMES, SIGMA_SAVE_FRAMES)) / fps
    uncertainty_kmh = mean_ms * 3.6 * float(np.hypot(sigma_d / distance, sigma_t / flight_time))
    mean_kmh = round(mean_ms * 3.6, 1)
    uncertainty_kmh = round(uncertainty_kmh, 1)
    pixels = contact_from_pixels(ball_px, contact)
    launch = np.asarray(theta, dtype=float)[2:5]
    launch_kmh = [float(np.linalg.norm(launch)) * 3.6]
    for alt in alternative_thetas or []:
        launch_kmh.append(float(np.linalg.norm(np.asarray(alt, dtype=float)[2:5])) * 3.6)
    per_frame = flight.get("speed_mps") or []
    at_save = (
        float(per_frame[save_frame]) if 0 <= save_frame < len(per_frame) else None
    )  # fmt: skip
    peak = flight.get("peak_speed_mps")
    return {
        "fps": float(fps),
        "frame_ms": round(1000.0 / fps, 2),
        "kick": {
            "frame": round(kick_frame, 2),
            "time_s": round(kick_time, 4),
            "first_moved_frame": int(contact),
            "counted_frame": counted_frame,
            "pixel_check": pixels,
            "agrees_with_pixels": None
            if pixels is None
            else bool(abs(kick_frame - pixels["frame"]) <= PIXEL_CONTACT_TOLERANCE_FRAMES),
        },
        "save": {"frame": int(save_frame), "time_s": round(t_save, 4)},
        "flight_time_s": round(flight_time, 4),
        "flight_frames_counted": counted_frames,
        "flight_time_counted_s": round(counted_frames / fps, 4),
        "distance": {
            "spot_to_hands_m": round(distance, 3),
            "path_m": round(path_m, 3),
            "to_line_m": None if to_line is None else round(to_line, 3),
            "nominal_m": float(goal.penalty_distance),
        },
        "mean": {
            "ms": round(mean_ms, 2),
            "kmh": mean_kmh,
            "kmh_counted": None if counted_ms is None else round(counted_ms * 3.6, 1),
            "uncertainty_kmh": uncertainty_kmh,
            "range_kmh": [
                round(mean_kmh - uncertainty_kmh, 1),
                round(mean_kmh + uncertainty_kmh, 1),
            ],
        },
        "exit_model": {
            "launch_ms": round(float(np.linalg.norm(launch)), 2),
            "launch_kmh": round(launch_kmh[0], 1),
            "peak_ms": None if peak is None else round(float(peak), 2),
            "peak_kmh": None if peak is None else round(float(peak) * 3.6, 1),
            "range_kmh": [round(min(launch_kmh), 1), round(max(launch_kmh), 1)],
        },
        "at_save_model": {
            "ms": None if at_save is None else round(at_save, 2),
            "kmh": None if at_save is None else round(at_save * 3.6, 1),
        },
        "sigma": {
            "kick_frames": SIGMA_KICK_FRAMES,
            "save_frames": SIGMA_SAVE_FRAMES,
            "distance_m": round(sigma_d, 3),
        },
    }


def save_metrics(
    *,
    field_rows: list[dict],
    ball_px: dict[int, tuple[float, float]],
    fps: float,
    n_frames: int,
    goal_x: float,
    flight: dict,
    flight_track: dict[int, np.ndarray],
    save_frame: int,
    keeper_points: dict[str, np.ndarray] | None = None,
    gk_move_frame: int | None = None,
    anthro: Anthropometrics | None = None,
    alternatives: list[dict] | None = None,
) -> dict:
    """The metrics of a saved penalty from the run's 3D flight (see :func:`penalty_metrics`).

    * ``interception``: the ball at the save frame (pitch metres), how far in front of the goal
      line, its height, the time from the kick and the mean speed from the spot to it;
    * ``counterfactual``: where the same flight would have crossed the goal line (zone, posts,
      crossbar, HWR accuracy), had the keeper not touched it;
    * ``uncertainty``: the same two, from the flights refitted with the keeper's anchor moved along
      the camera ray (``alternatives``), when given;
    * ``metrics``: pynalty's set, with the would-be crossing as the placement, the keeper's hips on
      the goal plane, the flight's sub-frame kick time and the spot-to-hands distance and time.
    """
    goal = GoalGeometry()
    sign = math.copysign(1.0, goal_x)
    spot = np.array([goal_x - sign * goal.penalty_distance, 0.0, goal.ball_radius], dtype=float)
    theta = flight.get("theta")
    kick_time = flight.get("kick_time_s")
    contact = flight.get("contact_frame")
    if theta is None or kick_time is None or contact is None:
        return _unavailable("the flight has no fitted kick (theta, kick time, contact)", goal_x)
    kick_time, contact = float(kick_time), int(contact)
    inter = flight_track.get(save_frame)
    if inter is None or not np.isfinite(np.asarray(inter, dtype=float)).all():
        return _unavailable(f"the 3D ball is missing at the save frame {save_frame}", goal_x)
    inter = np.asarray(inter, dtype=float)
    t_save = save_frame / fps
    flight_time = t_save - kick_time
    if flight_time <= 0:
        return _unavailable("the save is not after the kick", goal_x)
    distance = float(np.linalg.norm(inter - spot))
    interception = {
        "frame": int(save_frame),
        "xyz": [round(float(v), 4) for v in inter],
        "in_front_of_line_m": round(float(sign * (goal_x - inter[0])), 3),
        "height_m": round(float(inter[2]), 3),
        "lateral_m": round(float(inter[1]), 3),
        "time_s_after_kick": round(flight_time, 4),
        "distance_from_spot_m": round(distance, 3),
        "mean_speed_mps": round(distance / flight_time, 2),
    }
    corners = goal_corners_px(field_rows, goal_x, save_frame)
    posts = corners[1] if corners is not None else None
    counterfactual = _would_be(theta, goal_x, t_save, posts, goal)
    uncertainty = None
    alt_distances: list[float] = []
    alt_thetas: list = []
    if alternatives:
        rows = []
        for alt in alternatives:
            alt_theta = alt.get("theta")
            if alt_theta is None:
                continue
            at_save = simulate_flight(np.asarray(alt_theta, dtype=float), np.array([t_save]),
                                      goal.ball_radius)[0]  # fmt: skip
            alt_distances.append(float(np.linalg.norm(at_save - spot)))
            alt_thetas.append(alt_theta)
            rows.append(
                {
                    "shift_m": alt.get("shift_m"),
                    "interception_xyz": [round(float(v), 4) for v in at_save],
                    "counterfactual": _would_be(alt_theta, goal_x, t_save, posts, goal),
                    "rms_px": alt.get("rms_px"),
                }
            )
        uncertainty = {"along_camera_ray": rows}
    path = penalty_path(
        ball_px,
        {},
        kick=contact,
        goal_x=goal_x,
        fps=fps,
        n_frames=n_frames,
        track={f: p for f, p in flight_track.items() if f <= save_frame},
        track_source="monocular flight (ball_flight.json)",
    )
    result: dict = {
        "available": True,
        "reason": "",
        "event": "save",
        "goal_x_m": goal_x,
        "kick_frame": contact,
        "kick_time_s": round(kick_time, 4),
        "save_frame": int(save_frame),
        "interception": interception,
        "counterfactual": counterfactual,
        "uncertainty": uncertainty,
        "path": path,
        "speed": speed_summary(
            fps=fps,
            kick_time=kick_time,
            contact=contact,
            save_frame=int(save_frame),
            theta=theta,
            flight=flight,
            distance=distance,
            to_line=None
            if counterfactual is None
            else float(
                np.hypot(
                    np.hypot(goal.penalty_distance, counterfactual["y_m"]),
                    counterfactual["z_m"] - goal.ball_radius,
                )
            ),
            goal=goal,
            ball_px=ball_px,
            alternative_distances=alt_distances,
            alternative_thetas=alt_thetas,
        ),
    }
    if corners is None or counterfactual is None:
        result["metrics"] = {}
        result["metrics_reason"] = (
            "the goal posts are not in the field model"
            if corners is None
            else "the flight does not reach the goal line even without the keeper"
        )
        return result
    calib, posts = corners
    move = int(gk_move_frame) if gk_move_frame is not None else None
    plane: dict = {
        "goal_ball": tuple(counterfactual["goal_plane_xz"]),
        "kick_ball": goal_plane_xz(spot[1], spot[2], posts),
    }
    kick_hips = _hips_mid(keeper_points, contact)
    save_hips = _hips_mid(keeper_points, save_frame)
    if kick_hips is not None and save_hips is not None:
        plane["kick_gk"] = goal_plane_xz(kick_hips[1], kick_hips[2], posts)
        plane["goal_gk"] = goal_plane_xz(save_hips[1], save_hips[2], posts)
    body = []
    if keeper_points:
        for arr in keeper_points.values():
            if 0 <= save_frame < len(arr) and np.isfinite(arr[save_frame]).all():
                body.append(goal_plane_xz(arr[save_frame][1], arr[save_frame][2], posts))
    if body:
        plane["gk_body"] = np.asarray(body, dtype=float)
    timing = {"flight_time_s": flight_time}
    if move is not None:
        timing["gk_response_time_s"] = move / fps - kick_time
        timing["gk_dive_time_s"] = t_save - move / fps
    centre = (float(np.mean(calib[:, 0])), float(np.mean(calib[:, 1])))
    metrics = compute_penalty_metrics(
        calib_pixels=calib,
        kick_frame=contact,
        goal_frame=int(save_frame),
        gk_move_frame=move if move is not None else int(save_frame),
        kick_ball_px=ball_px.get(contact),
        kick_gk_px=centre,  # replaced by the keeper's hips on the goal plane when he has them
        goal_ball_px=ball_px.get(save_frame, centre),
        goal_gk_px=centre,
        fps=fps,
        goal=goal,
        anthro=anthro,
        shot_outcome="save",
        plane_xz=plane,
        timing=timing,
        flight_distance_m=distance,
    )
    metrics.pop("_flight_path", None)
    result.update(
        {
            "calibration_px": calib.tolist(),
            "posts": posts,
            "gk_move_frame": move,
            "metrics": metrics,
        }
    )
    return result


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


def flight_path_array(
    result: dict, fps: float, goal: GoalGeometry | None = None
) -> np.ndarray | None:
    """Canvas flight samples ``(n, 4)`` as ``t, X_off, Y_toward, Z`` in the pynalty goal frame.

    Prefer the measured ``path.rows`` (field metres → goal plane via :func:`goal_plane_xz`);
    otherwise fall back to the gravity parabola of :func:`fit_ball_flight_3d` from the entry point.
    """
    goal = goal or GoalGeometry()
    path = result.get("path") or {}
    rows = path.get("rows") or []
    posts = result.get("posts")
    kick = int(result.get("kick_frame") or path.get("kick_frame") or 0)
    if rows and isinstance(posts, dict) and {"left_post_y_m", "right_post_y_m"} <= posts.keys():
        spot = path.get("spot_m")
        if spot is None:
            spot = [rows[0]["x_m"], rows[0]["y_m"], rows[0]["z_m"]]
        spot_x = float(spot[0])
        samples: list[list[float]] = []
        for row in rows:
            t = (int(row["frame"]) - kick) / float(fps)
            x_plane, z_plane = goal_plane_xz(float(row["y_m"]), float(row["z_m"]), posts)
            samples.append(
                [t, x_plane - goal.mid_x, abs(float(row["x_m"]) - spot_x), float(z_plane)]
            )
        if samples:
            return np.asarray(samples, dtype=float)
    metrics = result.get("metrics") or {}
    entry_x = metrics.get("ball_entry_x_m")
    entry_z = metrics.get("ball_entry_z_m")
    flight_s = metrics.get("flight_time_s")
    if entry_x is None or entry_z is None or not flight_s:
        return None
    return fit_ball_flight_3d(float(entry_x), float(entry_z), float(flight_s), goal=goal)["path"]


def _grab_snapshots(
    video: Path | str, frames: dict[str, int | None], out_dir: Path
) -> dict[str, str]:
    """Write PNG frames for the HTML report; keys match :func:`pynalty_report._snapshots_html`."""
    import cv2

    video = Path(video)
    if not video.is_file():
        return {}
    wanted = {k: int(v) for k, v in frames.items() if v is not None and int(v) >= 0}
    if not wanted:
        return {}
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        return {}
    snaps_dir = out_dir / "snapshots"
    snaps_dir.mkdir(parents=True, exist_ok=True)
    out: dict[str, str] = {}
    try:
        for key, frame_idx in wanted.items():
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ok, bgr = cap.read()
            if not ok or bgr is None:
                continue
            path = snaps_dir / f"{key}.png"
            cv2.imwrite(str(path), bgr)
            out[key] = str(path)
    finally:
        cap.release()
    return out


def _collect_report_videos(cam3d: Path, out_dir: Path) -> dict[str, str]:
    """Copy available TV / overlay / Blender MP4s next to the penalty HTML for offline playback."""
    media = out_dir / "media"
    media.mkdir(parents=True, exist_ok=True)
    candidates: list[tuple[str, Path]] = [
        ("animacao_lance", cam3d / "report_tv" / "pt" / "animacao_lance.mp4"),
        ("kick_animation", cam3d / "report_tv" / "en" / "kick_animation.mp4"),
        ("overlay_3d", next(cam3d.glob("*_3d_overlay.mp4"), Path())),
        ("topview", next(cam3d.glob("*_topview.mp4"), Path())),
        ("animacao_3d", cam3d / "report_tv" / "pt" / "animacao_3d.mp4"),
        ("animacao_3d_en", cam3d / "report_tv" / "en" / "animacao_3d.mp4"),
        ("blender_preview", cam3d / "blender" / "animacao_3d.mp4"),
    ]
    import shutil

    videos: dict[str, str] = {}
    for key, src in candidates:
        if not src or not src.is_file():
            continue
        dest = media / src.name
        if dest.resolve() != src.resolve():
            shutil.copy2(src, dest)
        videos[key] = str(dest)
    return videos


def report(
    out_dir: Path | str,
    video: Path | str,
    fps: float,
    result: dict,
    *,
    cam3d: Path | str | None = None,
) -> dict[str, str]:
    """The pynalty HTML report of an available result (``report.html`` and ``report_pt.html``).

    Fills the canvas flight samples from the 3D path, embeds TV / overlay / Blender MP4s when
    present under ``cam3d`` (defaults to ``out_dir/../..``), and grabs event snapshots from ``video``.
    """
    if not result.get("available"):
        return {}
    folder = Path(out_dir)
    folder.mkdir(parents=True, exist_ok=True)
    root = Path(cam3d) if cam3d is not None else folder.resolve().parent.parent
    goal = GoalGeometry()
    flight = flight_path_array(result, float(fps), goal=goal)
    metrics = result["metrics"]
    snapshots = _grab_snapshots(
        video,
        {
            "snapshot_kick": result.get("kick_frame") or metrics.get("kick_frame"),
            "snapshot_gk_move": result.get("gk_move_frame") or metrics.get("gk_move_frame"),
            "snapshot_goal": result.get("save_frame")
            or result.get("goal_frame")
            or metrics.get("goal_frame"),
        },
        folder,
    )
    videos = _collect_report_videos(root, folder) if root.is_dir() else {}
    pose = result.get("gk_dive") or result.get("keeper") or {}
    ctx = ReportContext(
        video_path=str(video),
        fps=float(fps),
        metrics=metrics,
        goal=goal,
        snapshots=snapshots,
        flight_path=flight,
        pose_metrics=pose if isinstance(pose, dict) else {},
        videos=videos,
        anthro=result.get("anthropometrics") or {},
    )
    return write_html_reports(str(folder), ctx, lang="both")
