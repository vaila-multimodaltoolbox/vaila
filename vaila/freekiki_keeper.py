"""
================================================================================
Script: freekiki_keeper.py - the goalkeeper's penalty dive from the 3D poses of a run
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.17
Created: 08 November 2026
Update Date: 12 November 2026

Description:
    The keeper's movement in a penalty, from his 3D keypoints in pitch metres
    (``freekiki_kicker.person_points``) and the ball of the run. Pitch frame of freekiki: +x
    toward the right goal, the attacked goal line at ``x = goal_x``, z up. The keeper faces the
    kicker, so his right is ``+sign(goal_x) * y``.

    * onset: the first frame from ``ONSET_SEARCH_S`` before the contact in which the lateral speed
      of the centre of mass (CoM, :func:`freekiki_com.com_track`) is above ``ONSET_SPEED_MPS`` for
      ``ONSET_FRAMES`` frames. It is compared with an onset marked by eye when one is given;
    * the dive: side (the keeper's own), lateral CoM displacement from the onset to the save, peak
      lateral and resultant CoM speed and the time to reach them, the CoM drop;
    * the variables of Monteiro et al. (2024, Sci Rep, doi:10.1038/s41598-024-60074-x)
      as far as one camera allows: the frontal step (the forward travel of the foot that steps
      first), the frontal departure angle (the CoM's horizontal path from the onset to the peak
      lateral speed, against the goal line) and the knee angles from the onset to the push-off;
    * the hands (the palms: wrist, index and pinky tips) and their gap to the ball at the save;
    * IFAB Law 14 at the kick: the signed distance of each foot point (heels, toes, ankles) to the
      goal line, positive in front of it. Since 2019 the keeper must have at least part of one foot
      touching, in line with or behind the line (see Monteiro et al. 2022, Sci Rep,
      doi:10.1038/s41598-022-21508-6). From the 3D pose the depth of the feet is a monocular
      estimate; :func:`law14_from_pixels` measures a foot pixel against the painted line pixels
      through the frame's ground homography, a relative measure with its pixel uncertainty.

    Every value carries the error of the monocular 3D pose and of the 1/fps sampling; they are
    measurements of one clip, not a verdict of the referee.
"""

from __future__ import annotations

import math

import numpy as np

try:
    from .freekiki_com import com_track
    from .freekiki_kicker import angle_between, local_velocity
except ImportError:
    from freekiki_com import com_track  # ty: ignore[unresolved-import]
    from freekiki_kicker import angle_between, local_velocity  # ty: ignore[unresolved-import]

BALL_RADIUS_M = 0.11
ONSET_SPEED_MPS = 0.5  # lateral CoM speed that marks the start of the keeper's movement
ONSET_FRAMES = 3  # ...held for this many frames
ONSET_SEARCH_S = 0.5  # searched from this long before the contact
STEP_MIN_M = 0.05  # a foot that travels less than this has not stepped
GOAL_LINE_WIDTH_M = 0.12  # the maximum width of the painted lines (Law 1)
FOOT_POINTS = (
    "left-heel",
    "right-heel",
    "left-big-toe-tip",
    "right-big-toe-tip",
    "left-small-toe-tip",
    "right-small-toe-tip",
    "left-ankle",
    "right-ankle",
)
PALM_PARTS = {
    "left": ("left-wrist", "left-index-tip", "left-pinky-tip"),
    "right": ("right-wrist", "right-index-tip", "right-pinky-tip"),
}


def _mean_points(points: dict[str, np.ndarray], names, n: int) -> np.ndarray:
    """Per frame, the mean of the named points that are finite; NaN where none is."""
    found = [points[name] for name in names if name in points]
    if not found:
        return np.full((n, 3), np.nan)
    stack = np.stack(found)  # (k, n, 3)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(stack).all(axis=2).sum(axis=0)
        total = np.nansum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    out = np.full((n, 3), np.nan)
    ok = counts > 0
    out[ok] = total[ok] / counts[ok][:, None]
    return out


def palms(points: dict[str, np.ndarray], n: int) -> np.ndarray:
    """The middle of the two palms per frame ``(n, 3)`` (each palm: wrist, index and pinky tips)."""
    sides = [_mean_points(points, names, n) for names in PALM_PARTS.values()]
    stack = np.stack(sides)
    with np.errstate(invalid="ignore"):
        counts = np.isfinite(stack).all(axis=2).sum(axis=0)
        total = np.nansum(np.where(np.isfinite(stack), stack, 0.0), axis=0)
    out = np.full((n, 3), np.nan)
    ok = counts > 0
    out[ok] = total[ok] / counts[ok][:, None]
    return out


def movement_onset(
    lateral_speed: np.ndarray,
    *,
    start: int,
    stop: int,
    speed: float = ONSET_SPEED_MPS,
    frames: int = ONSET_FRAMES,
) -> int | None:
    """The first frame in ``[start, stop]`` from which ``|lateral_speed|`` stays above ``speed``
    for ``frames`` frames, or None."""
    n = len(lateral_speed)
    for f in range(max(0, start), min(stop, n - frames) + 1):
        window = np.abs(lateral_speed[f : f + frames])
        if np.isfinite(window).all() and (window > speed).all():
            return int(f)
    return None


def _knee_angle(points: dict[str, np.ndarray], side: str) -> np.ndarray | None:
    """The knee angle (degrees, 180 = straight) of one leg per frame, or None."""
    hip, knee, ankle = (points.get(f"{side}-{p}") for p in ("hip", "knee", "ankle"))
    if hip is None or knee is None or ankle is None:
        return None
    return angle_between(hip - knee, ankle - knee)


def law14_feet(points: dict[str, np.ndarray], frame: int, goal_x: float) -> dict[str, float] | None:
    """Signed distance (m) of each foot point to the goal line at ``frame``: positive in front of
    it (on the pitch side). None when no foot point is there."""
    sign = math.copysign(1.0, goal_x)
    out = {}
    for name in FOOT_POINTS:
        arr = points.get(name)
        if arr is not None and 0 <= frame < len(arr) and np.isfinite(arr[frame]).all():
            out[name] = round(float(sign * (goal_x - arr[frame][0])), 3)
    return out or None


def law14_status(nearest_in_front_m: float, sigma_m: float = 0.0) -> str:
    """``on_or_behind_line``, ``touching_the_line``, ``in_front`` or ``undecided`` (within sigma).

    The foot point nearest the goal (the smallest signed distance) decides: at or behind the line
    (<= 0), on the painted line (within its width), or in front of it. An interval that crosses a
    limit is ``undecided``.
    """
    d = float(nearest_in_front_m)
    if d + sigma_m <= 0.0:
        return "on_or_behind_line"
    if d - sigma_m > GOAL_LINE_WIDTH_M:
        return "in_front"
    if d - sigma_m > 0.0 and d + sigma_m <= GOAL_LINE_WIDTH_M:
        return "touching_the_line"
    return "undecided"


def law14_from_pixels(
    foot_px: tuple[float, float],
    line_px: list[tuple[float, float]],
    L: np.ndarray,
    goal_x: float,
    *,
    sigma_px: float = 2.0,
) -> dict:
    """A foot's distance to the painted goal line measured in the image, through the ground plane.

    ``foot_px`` is the pixel where the foot touches the grass; ``line_px`` two or more pixels on the
    painted goal line near it; ``L`` the frame's ground homography (8 DLT2D coefficients, pitch
    ``(x, y, 0)`` -> pixel). Both are mapped to the pitch and the foot is compared with the line
    at the foot's ``y``: a relative measure, free of a calibration offset common to both. The
    uncertainty moves each pixel by ``sigma_px`` (vertical, the steep direction of a grazing view)
    and keeps the spread. Positive: in front of the line (pitch side).
    """
    H = np.array([[L[0], L[1], L[2]], [L[3], L[4], L[5]], [L[6], L[7], 1.0]], dtype=float)
    sign = math.copysign(1.0, goal_x)

    def to_ground(uv) -> np.ndarray:
        g = np.linalg.solve(H, np.array([float(uv[0]), float(uv[1]), 1.0]))
        return g[:2] / g[2]

    def distance(foot_uv, line_uvs) -> tuple[float, float]:
        foot = to_ground(foot_uv)
        line = np.array([to_ground(p) for p in line_uvs])
        if len(line) >= 2 and np.ptp(line[:, 1]) > 1e-6:
            slope, intercept = np.polyfit(line[:, 1], line[:, 0], 1)
            line_x = slope * foot[1] + intercept
        else:
            line_x = float(np.mean(line[:, 0]))
        return float(sign * (line_x - foot[0])), float(line_x)

    centre, line_x = distance(foot_px, line_px)
    spread = []
    for dv_foot in (-sigma_px, sigma_px):
        for dv_line in (-sigma_px, sigma_px):
            foot = (foot_px[0], foot_px[1] + dv_foot)
            line = [(u, v + dv_line) for u, v in line_px]
            spread.append(distance(foot, line)[0])
    sigma = float(max(abs(v - centre) for v in spread))
    return {
        "in_front_m": round(centre, 3),
        "sigma_m": round(sigma, 3),
        "line_x_m": round(line_x, 3),
        "line_offset_from_model_m": round(float(sign * (goal_x - line_x)), 3),
        "status": law14_status(centre, sigma),
    }


def keeper_kinematics(
    points: dict[str, np.ndarray],
    *,
    fps: float,
    contact_frame: int,
    save_frame: int,
    goal_x: float,
    kick_time_s: float | None = None,
    ball_xyz: np.ndarray | None = None,
    onset_frame_given: int | None = None,
) -> dict:
    """The keeper's dive, its timing and Law 14 at the kick (see the module description).

    ``contact_frame`` is the first frame in which the ball has left the foot (freekiki's contact);
    ``kick_time_s`` the fitted instant of the kick (between frames, default ``contact/fps``);
    ``save_frame`` the frame of the save (or the last frame studied); ``ball_xyz`` the ball
    ``(n, 3)`` for the hand-ball gap; ``onset_frame_given`` an onset marked by eye (0-based).
    Returns ``{"available", "reason", "metrics", "series"}``.
    """
    if not points:
        return {"available": False, "reason": "the keeper has no 3D keypoints", "metrics": {}}
    n = len(next(iter(points.values())))
    kick_t = float(kick_time_s) if kick_time_s is not None else contact_frame / fps
    if not 0 <= contact_frame < save_frame < n:
        return {"available": False, "reason": "the contact and the save are not in the clip",
                "metrics": {}}  # fmt: skip
    com = com_track(points)
    if not np.isfinite(com[contact_frame]).all() or not np.isfinite(com[save_frame]).all():
        return {"available": False, "reason": "no keeper centre of mass at the kick or the save",
                "metrics": {}}  # fmt: skip
    sign = math.copysign(1.0, goal_x)
    vel = local_velocity(com, fps)
    lateral = vel[:, 1]
    start = contact_frame - int(round(ONSET_SEARCH_S * fps))
    onset_auto = movement_onset(lateral, start=start, stop=save_frame)
    onset = onset_frame_given if onset_frame_given is not None else onset_auto
    if onset is None or not np.isfinite(com[onset]).all():
        onset = contact_frame
    dy = float(com[save_frame][1] - com[onset][1])
    side = "right" if sign * dy > 0 else "left"  # the keeper faces the kicker: his right is +sign*y
    window = range(onset, save_frame + 1)
    lat_speed = np.array([abs(lateral[f]) if np.isfinite(lateral[f]) else np.nan for f in window])
    res_speed = np.array([float(np.linalg.norm(vel[f])) if np.isfinite(vel[f]).all() else np.nan
                          for f in window])  # fmt: skip
    peak_lat_i = int(np.nanargmax(lat_speed)) if np.isfinite(lat_speed).any() else 0
    peak_res_i = int(np.nanargmax(res_speed)) if np.isfinite(res_speed).any() else 0
    peak_lat_f = onset + peak_lat_i
    # the frontal step: the forward travel (toward the kicker) of the foot that steps first
    step = {}
    for side_name in ("left", "right"):
        ankle = points.get(f"{side_name}-ankle")
        if ankle is None or not np.isfinite(ankle[onset]).all():
            continue
        forward = -sign * (ankle[onset : peak_lat_f + 1, 0] - ankle[onset, 0])
        travel = np.linalg.norm(ankle[onset : peak_lat_f + 1, :2] - ankle[onset, :2], axis=1)
        if np.isfinite(travel).any() and np.nanmax(travel) >= STEP_MIN_M:
            first = int(np.argmax(np.nan_to_num(travel) >= STEP_MIN_M))
            step[side_name] = {"first_frame": onset + first,
                               "forward_m": float(np.nanmax(forward))}  # fmt: skip
    step_foot = min(step, key=lambda s: step[s]["first_frame"]) if step else None
    # the frontal departure angle: the CoM's horizontal path against the goal line
    path = com[peak_lat_f][:2] - com[onset][:2]
    forward_path = -sign * path[0]
    departure = (
        math.degrees(math.atan2(forward_path, abs(path[1]))) if np.hypot(*path) > 0 else None
    )
    knees = {}
    for side_name in ("left", "right"):
        angle = _knee_angle(points, side_name)
        if angle is None:
            continue
        seg = angle[onset : peak_lat_f + 1]
        knees[side_name] = {
            "at_onset_deg": _round(angle[onset]),
            "at_push_off_deg": _round(angle[peak_lat_f]),
            "min_deg": _round(np.nanmin(seg)) if np.isfinite(seg).any() else None,
        }
    hands = palms(points, n)
    gap = None
    if ball_xyz is not None and np.isfinite(hands[save_frame]).all():
        ball = np.asarray(ball_xyz, dtype=float)[save_frame]
        if np.isfinite(ball).all():
            gap = float(np.linalg.norm(hands[save_frame] - ball) - BALL_RADIUS_M)
    feet = {f: law14_feet(points, f, goal_x) for f in sorted({contact_frame - 1, contact_frame})
            if f >= 0}  # fmt: skip
    nearest = {f: min(d.values()) for f, d in feet.items() if d}
    metrics = {
        "onset_frame": int(onset),
        "onset_source": "given"
        if onset_frame_given is not None
        else ("com_lateral_speed" if onset_auto is not None else "contact"),  # fmt: skip
        "onset_frame_auto": onset_auto,
        "onset_ms_before_kick": round((kick_t - onset / fps) * 1000.0, 1),
        "onset_auto_ms_before_kick": None
        if onset_auto is None
        else round((kick_t - onset_auto / fps) * 1000.0, 1),  # fmt: skip
        "save_ms_after_kick": round((save_frame / fps - kick_t) * 1000.0, 1),
        "movement_time_s": round((save_frame - onset) / fps, 4),
        "dive_side": side,
        "lateral_displacement_m": round(abs(dy), 3),
        "com_drop_m": round(float(com[onset][2] - com[save_frame][2]), 3),
        "peak_lateral_speed_mps": _round(lat_speed[peak_lat_i]),
        "peak_lateral_speed_frame": int(peak_lat_f),
        "time_to_peak_lateral_s": round(peak_lat_i / fps, 4),
        "peak_resultant_speed_mps": _round(res_speed[peak_res_i]),
        "time_to_peak_resultant_s": round(peak_res_i / fps, 4),
        "frontal_step_foot": step_foot,
        "frontal_step_m": round(step[step_foot]["forward_m"], 3) if step_foot else None,
        "departure_angle_deg": None if departure is None else round(departure, 1),
        "knees": knees,
        "hands_ball_gap_m": None if gap is None else round(gap, 3),
        "law14_feet_m": feet,
        "law14_nearest_m": nearest,
    }
    series = {
        "frame": list(range(n)),
        "com_y_m": [_round(v) for v in com[:, 1]],
        "com_z_m": [_round(v) for v in com[:, 2]],
        "lateral_speed_mps": [_round(v) for v in lateral],
        "palms_xyz": [[_round(c) for c in row] for row in hands],
    }
    return {"available": True, "reason": "", "metrics": metrics, "series": series}


def _round(value, digits: int = 3):
    value = float(value)
    return round(value, digits) if math.isfinite(value) else None


# ---------------------------------------------------------------------------------------------
# The rear camera's measure of the dive
# ---------------------------------------------------------------------------------------------

REAR_OVERRIDE_FILE = "keeper_rear_override.json"
# What the monocular 3D pose gets wrong when the keeper's depth is unknown: the lateral travel and
# everything derived from it (speeds, the step, the departure angle) and the hands against the ball.
REAR_NULLED_KEYS = (
    "peak_lateral_speed_mps",
    "peak_lateral_speed_frame",
    "time_to_peak_lateral_s",
    "peak_resultant_speed_mps",
    "time_to_peak_resultant_s",
    "hands_ball_gap_m",
    "frontal_step_m",
    "frontal_step_foot",
    "departure_angle_deg",
)


def load_rear_override(data_dir) -> dict | None:
    """``keeper_rear_override.json`` of a run's ``data/`` folder (None when there is none).

    It holds what the goal-plane view of a rear camera measured: the hip's lateral travel (and its
    range), how far the hip went forward and the interception point from both cameras.
    """
    import json
    from pathlib import Path

    path = Path(data_dir) / REAR_OVERRIDE_FILE
    if not path.is_file():
        return None
    data = json.loads(path.read_text(encoding="utf-8"))
    return (
        data if isinstance(data, dict) and data.get("lateral_displacement_m") is not None else None
    )


def apply_rear_override(keeper: dict, override: dict | None) -> dict:
    """The keeper's metrics with the rear camera's lateral travel in place of the monocular one.

    A single camera cannot place the keeper in depth, so its lateral travel can be off by a metre.
    The rear camera sees the goal plane face on. The monocular value stays as
    ``lateral_displacement_model_1cam_m``; the speeds, the step, the departure angle and the
    hands-ball gap that came from the same poses are removed (None) and a mean lateral speed
    (travel over the movement time) replaces the peak.
    """
    if not override:
        return keeper
    out = dict(keeper)
    out["lateral_displacement_model_1cam_m"] = keeper.get("lateral_displacement_m")
    out["lateral_displacement_m"] = float(override["lateral_displacement_m"])
    out["lateral_displacement_range_m"] = list(override.get("lateral_range_m") or [])
    out["lateral_source"] = "rear_camera"
    for key in REAR_NULLED_KEYS:
        if key in out:
            out[key] = None
    move = keeper.get("movement_time_s")
    out["mean_lateral_speed_mps"] = (
        round(out["lateral_displacement_m"] / float(move), 2) if move else None
    )
    if override.get("hip_forward_m") is not None:
        out["hip_forward_m"] = float(override["hip_forward_m"])
    for key in ("hands_lateral_m", "hands_travel_m"):  # onset hands to the two-camera save point
        if override.get(key) is not None:
            out[key] = float(override[key])
    return out


def apply_rear_to_metrics(metrics: dict, data_dir) -> dict:
    """``penalty_metrics.json`` with the rear override applied to its keeper block (a copy)."""
    override = load_rear_override(data_dir)
    if not override:
        return metrics
    out = dict(metrics)
    out["keeper"] = apply_rear_override(metrics.get("keeper") or {}, override)
    out["rear_override"] = override
    return out
