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
Update Date: 28 October 2026

Description:
    The penalty analysis of :mod:`pynalty_analysis` without the hand marking of the pynalty app. Every
    input comes from the reference camera's run of the freekiki analysis:

    * the four goal-mouth corners in the image: the posts of the attacked goal, projected by the
      camera's DLT in ``field_kps_3d.csv`` at the frame nearest the kick, ordered as pynalty asks
      (bottom-left, top-left, top-right, bottom-right, as seen in the image);
    * the kick frame (:func:`freekiki_auto_flight.kick_onset` on ``ball.csv``) and the goal frame (the
      first frame after the kick in which the fitted ball is across the goal line);
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
    from .freekiki_pose_binocular import Keypoints
    from .freekiki_setpiece import GOAL_LINE_X_M
    from .pynalty_analysis import GoalGeometry, compute_penalty_metrics
    from .pynalty_report import ReportContext, write_html_reports
    from .pynalty_vision import LANDMARK_NAMES, PoseSequence, _first_movement_frame, gk_kinematics
except ImportError:
    from freekiki_auto_flight import kick_onset  # ty: ignore[unresolved-import]
    from freekiki_pose_binocular import Keypoints  # ty: ignore[unresolved-import]
    from freekiki_setpiece import GOAL_LINE_X_M  # ty: ignore[unresolved-import]
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
MOVE_THRESHOLD_PX = 10.0  # operational: the keeper has moved when a shoulder or hip moves this much
MOVE_INDICES = (11, 12, 23, 24)  # left/right shoulder and hip, in the 33 landmarks
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


def goal_frame(ball_xyz: np.ndarray, goal_x: float, after: int) -> int | None:
    """The first frame after ``after`` in which the fitted ball is across the attacked goal line."""
    side = math.copysign(1.0, goal_x)
    x = ball_xyz[:, 0] * side
    for f in range(max(after + 1, 0), len(x)):
        if np.isfinite(x[f]) and x[f] >= GOAL_LINE_X_M:
            return f
    return None


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


def _unavailable(reason: str, goal_x: float) -> dict:
    return {"available": False, "reason": reason, "goal_x_m": goal_x}


def penalty_metrics(
    *,
    field_rows: list[dict],
    ball_px: dict[int, tuple[float, float]],
    ball_xyz: np.ndarray,
    keypoints: Keypoints,
    keeper_boxes: dict[int, tuple[float, float, float, float]],
    keeper_sam: int | None,
    fps: float,
    goal_x: float,
) -> dict:
    """The penalty metrics of one run (pynalty's keys), or the reason they cannot be computed.

    ``ball_px``: the reference camera's ball (``ball.csv``); ``ball_xyz``: the fitted ball in metres,
    ``(n, 3)``, from the ``ball_3d_on_field.csv`` of the analysis; ``goal_x``: the attacked goal line.
    """
    kick = kick_onset(ball_px)
    if kick is None:
        return _unavailable("the ball does not move in this camera", goal_x)
    corners = goal_corners_px(field_rows, goal_x, kick)
    if corners is None:
        return _unavailable("the goal posts are not in the field model of this camera", goal_x)
    calib, posts = corners
    goal = goal_frame(ball_xyz, goal_x, kick)
    if goal is None:
        return _unavailable(
            "no crossing of the goal line in the fitted flight (a save or a miss)", goal_x
        )
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
    """``penalty_metrics.csv`` (one row per value, the reason first when unavailable) and ``.json``."""
    folder = Path(data_dir)
    folder.mkdir(parents=True, exist_ok=True)
    rows: list[tuple[str, object]] = []
    _flat("", result, rows)
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
    return {"csv": csv_path, "json": json_path}


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
