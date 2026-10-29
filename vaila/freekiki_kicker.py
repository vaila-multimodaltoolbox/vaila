"""
================================================================================
Script: freekiki_kicker.py - the kicker's metrics from the two-camera run
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
    The core kick metrics of one kicker, from ``poses_3d_binocular.csv`` (the triangulated
    keypoints, :mod:`freekiki_pose_binocular`) and the fitted ball (``ball_3d_on_field.csv``,
    ``ball.csv`` of the reference camera, ``ball_flight.json``):

    * contact: the first frame in which the ball leaves its rest pixel (:func:`freekiki_auto_flight.kick_onset`),
      checked against the fit's kick time;
    * kicking foot: the side whose foot is nearest the ball at contact (``kick_foot`` overrides);
    * foot speed (Vfoot) before contact, ball speed (Vball) after contact, and Vball/Vfoot;
    * distance from the support foot's centroid to the ball at contact;
    * approach angle: hip displacement before contact against the ball's direction after it;
    * knee angle (three points hip-knee-ankle), its extension speed, and the peak angular speeds of
      thigh, shank and foot before contact (proximal-distal order);
    * goal error: the fitted ball's entry on the goal plane against the centre of the goal mouth.
    * the posture of the kicking leg and the pelvis (Levanon & Dapena, 1998, operational definitions
      only): pelvis rotation, the angle between the thigh-shank and shank-foot planes, the hip and
      ankle angles, the lateral foot speed, and the share of the foot speed made by the knee's own
      rotation (a kinematic decomposition, not inverse dynamics). The figures are in
      :mod:`freekiki_kicker_figures`.

    Every number says where it comes from (``observado`` = triangulated keypoint, ``filtrado`` =
    quadratic local fit of a triangulated track, ``ajustado`` = the fitted flight, ``inferido``
    = computed from two of these). A metric without the points it needs is left empty with the
    reason; nothing is filled in.

    Scientific limits: at 59.94 fps the ball contact lasts about 1 ms, so the speed at impact is not
    measured. Vfoot is the mean over the frames before contact (about 33 ms, 2 frames), and Vball
    over the frames after it. Vieira et al. (2018) used 10 frames at 300 Hz, so the values are not
    comparable in absolute terms. The foot marker is the mean of big-toe-tip, small-toe-tip and heel,
    not the fifth metatarsal used by Vieira et al. The knee angle is a three-point angle, not an
    anatomical Euler angle. The segment angular speeds are the speed of each segment's direction, with
    no inverse dynamics.

    References (checked in the PDFs of the study folder): Vieira et al. 2018, RQES 89(2):210-220,
    doi:10.1080/02701367.2018.1439569 (Vfoot, Vball, Vball/Vfoot, support-foot distance); Ishii et al.
    2009, J Biomech 42:2712-2720 (attack angle and impact point, phase 2); Kellis & Katis 2007, JSSM
    6:154-165 (proximal-distal order). No threshold from the literature is used as an acceptance rule.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

try:
    from .freekiki_auto_flight import kick_onset
except ImportError:
    from freekiki_auto_flight import kick_onset  # ty: ignore[unresolved-import]

WINDOW_S = 0.033  # the window before the contact (foot) and after it (ball), ~2 frames at 60 Hz
SEARCH_S = (
    0.25  # the window before the contact in which the peaks of knee and segments are searched
)
SMOOTH_HALF = 2  # the quadratic fit uses 2 * SMOOTH_HALF + 1 frames (5 at 60 Hz)
SENSITIVITY_S = (0.017, 0.033, 0.067)  # windows reported to show how much the speeds depend on it
APPROACH_FRAMES = 5  # hip displacement: from 5 frames before the contact to 1 frame before it
FOOT_POINTS = ("big-toe-tip", "small-toe-tip", "heel")  # the foot marker: their mean
FOOT_CENTROID = FOOT_POINTS  # the foot's centroid: the points of the foot, not the ankle above it
GOAL_X_M = 52.45  # the goal line, |x| from the centre spot
GOAL_CENTRE_Z_M = 1.22  # the centre of the goal mouth (half the crossbar)
DEG = 180.0 / math.pi
SIDES = ("left", "right")
SENSITIVE_CHANGE = 0.5  # a speed that changes by more than this from 33 to 67 ms is flagged
SENSITIVE_METRICS = {  # metric -> the key of the sensitivity table it comes from
    "v_foot_pre_mps": "v_foot_pre_mps",
    "v_ball_post_mps": "v_ball_post_mps",
    "lateral_foot_speed_mps": "lateral_foot_mps",
    "knee_share_foot_speed": "knee_share",
}

DEFINITIONS = {
    "contact_frame": "first frame in which the ball leaves its rest pixel (5 px), reference camera",
    "kick_time_check_s": "fit kick time minus the contact time (positive: the fit starts later)",
    "kick_foot": "side whose foot centre is nearest the ball at contact (or the one requested)",
    "v_foot_pre_mps": "mean 3D speed of the kicking foot marker over the frames before contact",
    "v_ball_post_mps": "mean fitted ball speed over the frames after contact",
    "ratio_vball_vfoot": "v_ball_post / v_foot_pre (a.u.; Vieira et al. 2018 Vball/Vfoot)",
    "d_support_ball_m": "distance from the support foot centroid to the ball centre at contact",
    "approach_angle_deg": "angle in the pitch plane between the hip displacement before contact "
    "and the ball's horizontal velocity after it (operational definition)",
    "knee_angle_contact_deg": "three-point knee angle hip-knee-ankle of the kicking leg at contact",
    "knee_ext_peak_deg_s": "largest knee extension speed (d angle / dt) before contact",
    "peak_w_thigh_deg_s": "peak angular speed of the thigh direction before contact",
    "peak_w_shank_deg_s": "peak angular speed of the shank direction before contact",
    "peak_w_foot_deg_s": "peak angular speed of the foot direction before contact",
    "t_peak_thigh_ms": "time of the thigh peak relative to contact",
    "t_peak_shank_ms": "time of the shank peak relative to contact",
    "t_peak_foot_ms": "time of the foot peak relative to contact",
    "proximal_distal_order": "True when thigh, shank and foot peak in that order before contact",
    "goal_error_m": "distance from the fitted entry on the goal plane to the goal mouth centre",
    "goal_lateral_m": "y of the fitted entry on the goal plane (0 = centre of the goal)",
    "goal_height_m": "z of the fitted entry on the goal plane",
    "fit_peak_speed_mps": "peak speed of the fitted flight (ball_flight.json)",
    "fit_spin_rps": "spin of the fitted flight (ball_flight.json)",
    "pelvis_rotation_deg": "signed angle, in the pitch plane, from the kick direction to the hip line "
    "(left hip to right hip), folded into -90..90, at contact",
    "thigh_shank_plane_deg": "angle between the normals of the planes hip-knee-ankle and "
    "knee-ankle-toe, at contact (unsigned, 0-180; toe = big toe, small toe where the big one is missing)",
    "hip_angle_deg": "three-point angle shoulder-hip-knee of the kicking side, at contact",
    "ankle_angle_deg": "three-point angle knee-ankle-toe of the kicking foot, at contact",
    "lateral_foot_speed_mps": "mean speed of the foot across the kick direction over the frames "
    "before contact (the window of v_foot_pre)",
    "knee_share_foot_speed": "mean share of the foot speed made by the knee's own rotation over the "
    "frames before contact: (w_knee x (foot - knee)) . v_foot / |v_foot|^2 (a.u.)",
}


def window_frames(fps: float, window_s: float) -> int:
    """The number of frames in a window of ``window_s`` seconds (at least one)."""
    return max(1, int(round(window_s * fps)))


def local_velocity(values, fps: float, half: int = SMOOTH_HALF) -> np.ndarray:
    """Velocity of each frame from a quadratic fit to the ``2*half+1`` frames centred on it.

    ``values`` is ``(n, d)`` (or ``(n,)``); the result has the same shape, NaN where fewer than three
    frames of the window are finite. The real time between frames is used (``1 / fps``).
    """
    arr = np.asarray(values, dtype=float)
    squeeze = arr.ndim == 1
    if squeeze:
        arr = arr[:, None]
    n = len(arr)
    out = np.full(arr.shape, np.nan)
    for i in range(n):
        lo, hi = max(0, i - half), min(n, i + half + 1)
        idx = np.arange(lo, hi)
        ok = np.isfinite(arr[idx]).all(axis=1)
        if ok.sum() < 3:
            continue
        t = (idx[ok] - i) / fps
        for d in range(arr.shape[1]):
            coef = np.polyfit(t, arr[idx[ok], d], 2)
            out[i, d] = coef[1]
    return out[:, 0] if squeeze else out


def proximal_distal(thigh: int | None, shank: int | None, foot: int | None) -> bool | None:
    """True when the thigh, shank and foot peaks come in that order (frames); None when one is missing."""
    if thigh is None or shank is None or foot is None:
        return None
    return bool(thigh <= shank <= foot)


def _norm(vec) -> np.ndarray:
    return np.linalg.norm(np.asarray(vec, dtype=float), axis=-1)


def _unit(vec) -> np.ndarray:
    v = np.asarray(vec, dtype=float)
    length = np.linalg.norm(v, axis=-1, keepdims=True)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(length > 0, v / length, np.nan)


def angle_between(u, v) -> np.ndarray:
    """Angle (degrees) between two vectors per row; NaN when either is missing or zero."""
    u = np.asarray(u, dtype=float)
    v = np.asarray(v, dtype=float)
    dot = np.sum(u * v, axis=-1)
    norm = np.linalg.norm(u, axis=-1) * np.linalg.norm(v, axis=-1)
    with np.errstate(invalid="ignore", divide="ignore"):
        cosine = np.where(norm > 0, dot / norm, np.nan)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def read_poses(path: Path | str) -> list[dict]:
    """``poses_3d_binocular.csv`` as rows with numbers (empty fields become NaN)."""
    rows: list[dict] = []
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            out = dict(row)
            out["frame"] = int(row["frame"])
            out["player_id"] = int(row["player_id"])
            out["kpt_idx"] = int(row["kpt_idx"])
            for key in ("x_m", "y_m", "z_m"):
                out[key] = float(row[key]) if row[key] != "" else math.nan
            out["valid"] = row.get("valid", "True") == "True"  # monocular poses have no column
            rows.append(out)
    return rows


def person_points(rows: list[dict], player_id: int, n: int) -> dict[str, np.ndarray]:
    """``kpt_name -> (n, 3)`` metres of one paired person, NaN where the point is not valid."""
    out: dict[str, np.ndarray] = {}
    for row in rows:
        if row["player_id"] != player_id or row["kpt_idx"] < 0 or not row["valid"]:
            continue
        f = row["frame"]
        if not 0 <= f < n:
            continue
        arr = out.setdefault(row["kpt_name"], np.full((n, 3), np.nan))
        arr[f] = (row["x_m"], row["y_m"], row["z_m"])
    return out


def read_ball_xyz(path: Path | str, n: int) -> np.ndarray:
    """The fitted ball, ``(n, 3)`` metres per frame, from ``ball_3d_on_field.csv``."""
    arr = np.full((n, 3), np.nan)
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            f = int(row["frame"])
            if 0 <= f < n and row["x_m"] != "":
                arr[f] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    return arr


def read_ball_pixels(path: Path | str) -> dict[int, tuple[float, float]]:
    """Frame -> ball pixel of the reference camera (``ball.csv``)."""
    out: dict[int, tuple[float, float]] = {}
    with Path(path).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx"):
                out[int(row["frame"])] = (float(row["cx"]), float(row["cy"]))
    return out


def contact_frame(pixels: dict[int, tuple[float, float]]) -> int | None:
    """The first frame in which the ball moves from its rest pixel (None when it never does)."""
    return kick_onset(pixels)


def choose_kick_foot(
    points: dict[str, np.ndarray], ball: np.ndarray, frame: int
) -> tuple[str | None, dict]:
    """The side whose foot centroid is nearest the ball at ``frame``; the distances per side."""
    dist: dict[str, float] = {}
    for side in SIDES:
        pts = [points.get(f"{side}-{name}") for name in FOOT_CENTROID]
        pts = [p[frame] for p in pts if p is not None and np.isfinite(p[frame]).all()]
        if pts and np.isfinite(ball[frame]).all():
            dist[side] = float(np.linalg.norm(np.mean(pts, axis=0) - ball[frame]))
    if not dist:
        return None, dist
    return min(dist, key=lambda side: dist[side]), dist


def _foot_marker(points: dict[str, np.ndarray], side: str, n: int) -> np.ndarray:
    """The mean of the foot points of one side per frame (NaN where none is valid)."""
    arrays = [points[f"{side}-{name}"] for name in FOOT_POINTS if f"{side}-{name}" in points]
    if not arrays:
        return np.full((n, 3), np.nan)
    stack = np.stack(arrays)
    finite = np.isfinite(stack)
    count = finite.sum(axis=0)
    total = np.where(finite, stack, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _centroid(
    points: dict[str, np.ndarray], side: str, frame: int
) -> tuple[np.ndarray | None, int]:
    pts = [points[f"{side}-{name}"][frame] for name in FOOT_CENTROID if f"{side}-{name}" in points]
    good = [p for p in pts if np.isfinite(p).all()]
    if len(good) < 2:
        return None, len(good)
    return np.mean(good, axis=0), len(good)


def _speed_window(speed: np.ndarray, frames) -> tuple[float, int, int]:
    """Mean of ``speed`` over ``frames``; returns (mean or NaN, finite count, frame count)."""
    idx = [f for f in frames if 0 <= f < len(speed)]
    vals = speed[idx] if idx else np.array([])
    good = vals[np.isfinite(vals)]
    if good.size == 0:
        return math.nan, 0, len(idx)
    return float(np.mean(good)), int(good.size), len(idx)


def _metric(name, value, unit, source, quality, reason=""):
    return {
        "metric": name,
        "value": "" if value is None or (isinstance(value, float) and not math.isfinite(value)) else value,
        "unit": unit,
        "source": source,
        "quality": quality,
        "reason": reason,
        "definition": DEFINITIONS.get(name, ""),
    }  # fmt: skip


def _peak(series: np.ndarray, frames) -> tuple[float, int | None]:
    idx = [f for f in frames if 0 <= f < len(series) and np.isfinite(series[f])]
    if not idx:
        return math.nan, None
    best = max(idx, key=lambda f: series[f])
    return float(series[best]), best


def _finite_mean(values: np.ndarray) -> np.ndarray:
    """The mean over the first axis, ignoring NaN (NaN where no value is finite; no warning)."""
    ok = np.isfinite(values)
    count = ok.sum(axis=0)
    total = np.where(ok, values, 0.0).sum(axis=0)
    return np.where(count > 0, total / np.maximum(count, 1), np.nan)


def _value_at(series: np.ndarray, f: int) -> float:
    return float(series[f]) if 0 <= f < len(series) else math.nan


def _kick_direction(ball_vel: np.ndarray, contact: int, nw: int) -> np.ndarray | None:
    """The unit horizontal direction in which the ball goes after the contact; None if unknown."""
    vel = _finite_mean(ball_vel[contact : contact + nw, :2])
    norm = float(np.linalg.norm(vel)) if np.isfinite(vel).all() else 0.0
    return vel / norm if norm > 0 else None


def _signed_angle_2d(ref: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """The signed angle (degrees) from ``ref`` to each row of ``vec``, folded into (-90, 90].

    A line has no sense, so a rotation of 170 degrees is read as -10. NaN where ``vec`` is zero.
    """
    cross = ref[0] * vec[:, 1] - ref[1] * vec[:, 0]
    dot = ref[0] * vec[:, 0] + ref[1] * vec[:, 1]
    ang = np.degrees(np.arctan2(cross, dot))
    ang = np.where(ang > 90.0, ang - 180.0, ang)
    ang = np.where(ang <= -90.0, ang + 180.0, ang)
    return np.where(np.linalg.norm(vec, axis=1) > 0, ang, np.nan)


def _knee_share(hip, knee, ankle, foot, knee_rate_deg_s, fps: float) -> np.ndarray:
    """The share of the foot's speed that the knee's own rotation makes (kinematic decomposition).

    The knee turns about the normal of the hip-knee-ankle plane at the rate of the three-point knee
    angle, so the foot moves ``w x (foot - knee)`` because of the knee. The share is that velocity
    projected on the foot's velocity, divided by the foot's speed: 1 when the knee makes all of the
    foot's motion, 0 when the leg only translates.
    """
    axis = _unit(np.cross(hip - knee, ankle - knee))
    omega = axis * (np.asarray(knee_rate_deg_s, dtype=float) / DEG)[:, None]
    v_knee = np.cross(omega, foot - knee)
    v_foot = local_velocity(foot, fps)
    num = np.sum(v_knee * v_foot, axis=1)
    den = np.sum(v_foot * v_foot, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den > 0, num / den, np.nan)


def _toe(points: dict[str, np.ndarray], side: str) -> np.ndarray | None:
    """The big toe of a side, per frame; the small toe where the big one is missing (the foot's toe)."""
    big, small = points.get(f"{side}-big-toe-tip"), points.get(f"{side}-small-toe-tip")
    if big is None or small is None:
        return big if big is not None else small
    ok = np.all(np.isfinite(big), axis=1, keepdims=True)
    return np.where(ok, big, small)


def _posture(points, side, foot, fps, n, kick_dir, knee_rate) -> dict[str, np.ndarray]:
    """Per-frame pelvis, limb and foot quantities of the kicking side (NaN where a point is missing)."""
    out = {
        "pelvis_deg": np.full(n, np.nan),
        "plane_deg": np.full(n, np.nan),
        "hip_angle_deg": np.full(n, np.nan),
        "ankle_angle_deg": np.full(n, np.nan),
        "knee_share": np.full(n, np.nan),
        "lateral_foot_mps": np.full(n, np.nan),
    }  # fmt: skip
    hip, knee = points.get(f"{side}-hip"), points.get(f"{side}-knee")
    ankle, toe = points.get(f"{side}-ankle"), _toe(points, side)
    shoulders = [points[f"{s}-shoulder"] for s in SIDES if f"{s}-shoulder" in points]
    left, right = points.get("left-hip"), points.get("right-hip")
    if left is not None and right is not None and kick_dir is not None:
        out["pelvis_deg"] = _signed_angle_2d(kick_dir, (right - left)[:, :2])
    if shoulders and hip is not None and knee is not None:
        out["hip_angle_deg"] = angle_between(np.mean(shoulders, axis=0) - hip, knee - hip)
    if hip is not None and knee is not None and ankle is not None and toe is not None:
        out["plane_deg"] = angle_between(
            np.cross(hip - knee, ankle - knee), np.cross(knee - ankle, toe - ankle)
        )
        out["ankle_angle_deg"] = angle_between(knee - ankle, toe - ankle)
    if hip is not None and knee is not None and ankle is not None:
        out["knee_share"] = _knee_share(hip, knee, ankle, foot, knee_rate, fps)
    if kick_dir is not None:
        vel = local_velocity(foot, fps)[:, :2]
        along = vel @ kick_dir
        out["lateral_foot_mps"] = np.linalg.norm(vel - along[:, None] * kick_dir[None, :], axis=1)
    return out


def _flag_window_sensitive(metrics: list[dict], sensitivity: list[dict]) -> None:
    """Marks a speed whose value moves by more than ``SENSITIVE_CHANGE`` between 33 and 67 ms."""
    by_window = {round(row["window_s"], 3): row for row in sensitivity}
    short, long_ = by_window.get(0.033), by_window.get(0.067)
    if short is None or long_ is None:
        return
    for metric in metrics:
        key = SENSITIVE_METRICS.get(metric["metric"])
        if key is None or metric["value"] == "":
            continue
        a, b = short[key], long_[key]
        if a == "" or b == "" or a == 0:
            continue
        change = abs(b - a) / abs(a)
        if change > SENSITIVE_CHANGE:
            metric["quality"] = "window-sensitive"
            metric["reason"] = f"changes {change * 100:.0f}% between 33 and 67 ms"


def _goal_entry(ball: np.ndarray, after: int) -> tuple[np.ndarray | None, str]:
    """The fitted ball where it first crosses the goal plane after ``after`` (linear in x)."""
    x = ball[:, 0]
    for f in range(max(1, after), len(x)):
        if not (np.isfinite(x[f]) and np.isfinite(x[f - 1])):
            continue
        if abs(x[f]) >= GOAL_X_M and abs(x[f - 1]) < GOAL_X_M:
            w = (GOAL_X_M - abs(x[f - 1])) / (abs(x[f]) - abs(x[f - 1]))
            return ball[f - 1] + w * (ball[f] - ball[f - 1]), "crossing of the goal plane"
    return None, "no crossing of the goal plane in the fit (a save or a miss)"


def analyze_kicker(
    rows: list[dict],
    ball: np.ndarray,
    ball_pixels: dict[int, tuple[float, float]],
    *,
    kicker_id: int,
    fps: float,
    kick_foot: str = "auto",
    kick_time_s: float | None = None,
    fit_peak_speed_mps: float | None = None,
    fit_spin_rps: float | None = None,
    window_s: float = WINDOW_S,
) -> dict:
    """The metrics of one kicker. Returns ``metrics`` (rows), ``events``, ``series`` and ``sensitivity``."""
    n = len(ball)
    warnings: list[str] = []
    points = person_points(rows, kicker_id, n)
    contact = contact_frame(ball_pixels)
    events: dict = {"kicker_id": kicker_id, "fps": fps, "contact_frame": contact}
    metrics: list[dict] = []
    if contact is None or not points:
        reason = (
            "no ball movement in the reference camera"
            if contact is None
            else "kicker has no points"
        )
        metrics.append(_metric("contact_frame", None, "frame", "observado", "missing", reason))
        return {
            "metrics": metrics,
            "events": events,
            "series": None,
            "sensitivity": [],
            "warnings": [reason],
        }

    if kick_time_s is not None:
        check = kick_time_s - contact / fps
        events["kick_time_check_s"] = round(check, 4)
        if abs(check) > 1.0 / fps:
            warnings.append(f"fit kick time differs from the contact by {check * fps:+.1f} frames")
        metrics.append(_metric("kick_time_check_s", check, "s", "inferido", "ok"))
    metrics.append(_metric("contact_frame", contact, "frame", "observado", "ok"))
    metrics.append(_metric("contact_time_s", contact / fps, "s", "observado", "ok"))

    if kick_foot == "auto":
        side, dist = choose_kick_foot(points, ball, contact)
        foot_reason = "nearest foot to the ball at contact" if side else "no foot at contact"
    else:
        side = kick_foot
        _, dist = choose_kick_foot(points, ball, contact)
        foot_reason = "given by the user"
    if side is None:
        metrics.append(_metric("kick_foot", None, "", "observado", "missing", foot_reason))
        return {
            "metrics": metrics,
            "events": events,
            "series": None,
            "sensitivity": [],
            "warnings": warnings,
        }
    other = "right" if side == "left" else "left"
    events.update(kick_foot=side, support_foot=other, kick_foot_distance_m=dist)
    metrics.append(_metric("kick_foot", side, "", "inferido", "ok", foot_reason))

    foot = _foot_marker(points, side, n)
    foot_speed = _norm(local_velocity(foot, fps))
    ball_vel = local_velocity(ball, fps)
    ball_speed = _norm(ball_vel)
    nw = window_frames(fps, window_s)
    events["window_frames"] = nw
    events["window_s"] = round(nw / fps, 4)

    v_foot, got_f, all_f = _speed_window(foot_speed, range(contact - nw, contact))
    q_foot = "ok" if got_f == all_f and all_f else ("partial" if got_f else "missing")
    metrics.append(_metric("v_foot_pre_mps", v_foot, "m/s", "filtrado", q_foot,
                           "" if q_foot == "ok" else f"{got_f}/{all_f} foot frames valid"))  # fmt: skip
    v_ball, got_b, all_b = _speed_window(ball_speed, range(contact, contact + nw))
    q_ball = "ok" if got_b == all_b and all_b else ("partial" if got_b else "missing")
    metrics.append(_metric("v_ball_post_mps", v_ball, "m/s", "ajustado", q_ball))
    ratio = (
        v_ball / v_foot
        if math.isfinite(v_ball) and math.isfinite(v_foot) and v_foot > 0
        else math.nan
    )
    metrics.append(_metric("ratio_vball_vfoot", ratio, "a.u.", "inferido",
                           "ok" if math.isfinite(ratio) else "missing",
                           "" if math.isfinite(ratio) else "needs both speeds"))  # fmt: skip

    centroid, n_pts = _centroid(points, other, contact)
    if centroid is not None and np.isfinite(ball[contact]).all():
        d_support = float(np.linalg.norm(centroid - ball[contact]))
        metrics.append(_metric("d_support_ball_m", d_support, "m", "observado",
                               "ok" if n_pts == len(FOOT_CENTROID) else "partial",
                               "" if n_pts == len(FOOT_CENTROID) else f"{n_pts} of {len(FOOT_CENTROID)} support points"))  # fmt: skip
    else:
        metrics.append(_metric("d_support_ball_m", None, "m", "observado", "missing",
                               "support foot or ball missing at contact"))  # fmt: skip

    hip = np.nanmean(np.stack([points[f"{s}-hip"] for s in SIDES if f"{s}-hip" in points]), axis=0) \
        if any(f"{s}-hip" in points for s in SIDES) else np.full((n, 3), np.nan)  # fmt: skip
    if contact - APPROACH_FRAMES >= 0 and np.isfinite(hip[contact - 1]).all() and np.isfinite(hip[contact - APPROACH_FRAMES]).all() and np.isfinite(ball_vel[contact:contact + nw, :2]).all():  # fmt: skip
        d_hip = hip[contact - 1, :2] - hip[contact - APPROACH_FRAMES, :2]
        v_post = np.nanmean(ball_vel[contact : contact + nw, :2], axis=0)
        approach = float(angle_between(d_hip, v_post))
        metrics.append(_metric("approach_angle_deg", approach, "deg", "inferido", "ok"))
    else:
        metrics.append(_metric("approach_angle_deg", None, "deg", "inferido", "missing",
                               "hip or ball missing around contact"))  # fmt: skip

    hip_k = points.get(f"{side}-hip")
    knee = points.get(f"{side}-knee")
    ankle = points.get(f"{side}-ankle")
    if hip_k is not None and knee is not None and ankle is not None:
        knee_angle = angle_between(hip_k - knee, ankle - knee)
        knee_rate = local_velocity(knee_angle, fps)
        metrics.append(_metric("knee_angle_contact_deg", float(knee_angle[contact]), "deg",
                               "observado", "ok" if np.isfinite(knee_angle[contact]) else "missing"))  # fmt: skip
    else:
        knee_angle = knee_rate = np.full(n, np.nan)
        metrics.append(_metric("knee_angle_contact_deg", None, "deg", "observado", "missing",
                               "hip, knee or ankle missing"))  # fmt: skip

    search = range(contact - window_frames(fps, SEARCH_S), contact + 1)
    ext, _ = _peak(knee_rate, search)
    metrics.append(_metric("knee_ext_peak_deg_s", ext, "deg/s", "filtrado",
                           "ok" if math.isfinite(ext) else "missing"))  # fmt: skip

    segments = {}
    if hip_k is not None and knee is not None and ankle is not None:
        segments = {
            "thigh": _unit(knee - hip_k),
            "shank": _unit(ankle - knee),
            "foot": _unit(foot - ankle),
        }
    peaks: dict[str, int | None] = {}
    for name, unit_vec in segments.items():
        rate = _norm(local_velocity(unit_vec, fps)) * DEG
        value, frame = _peak(rate, search)
        peaks[name] = frame
        metrics.append(_metric(f"peak_w_{name}_deg_s", value, "deg/s", "filtrado",
                               "ok" if frame is not None else "missing"))  # fmt: skip
        metrics.append(_metric(f"t_peak_{name}_ms",
                               None if frame is None else round((frame - contact) / fps * 1000.0, 1),
                               "ms", "filtrado", "ok" if frame is not None else "missing"))  # fmt: skip
    order = proximal_distal(peaks.get("thigh"), peaks.get("shank"), peaks.get("foot"))
    metrics.append(_metric("proximal_distal_order", order, "", "filtrado",
                           "ok" if order is not None else "missing"))  # fmt: skip

    entry, how = _goal_entry(ball, contact)
    if entry is not None:
        err = math.hypot(entry[1], entry[2] - GOAL_CENTRE_Z_M)
        metrics.append(_metric("goal_error_m", err, "m", "ajustado", "ok"))
        metrics.append(_metric("goal_lateral_m", float(entry[1]), "m", "ajustado", "ok"))
        metrics.append(_metric("goal_height_m", float(entry[2]), "m", "ajustado", "ok"))
    else:
        for name in ("goal_error_m", "goal_lateral_m", "goal_height_m"):
            metrics.append(_metric(name, None, "m", "ajustado", "missing", how))
    if fit_peak_speed_mps is not None:
        metrics.append(_metric("fit_peak_speed_mps", fit_peak_speed_mps, "m/s", "ajustado", "ok"))
    if fit_spin_rps is not None:
        metrics.append(_metric("fit_spin_rps", fit_spin_rps, "rev/s", "ajustado", "ok"))

    kick_dir = _kick_direction(ball_vel, contact, nw)
    posture = _posture(points, side, foot, fps, n, kick_dir, knee_rate)
    pre = range(contact - nw, contact)
    at_contact = (
        ("pelvis_rotation_deg", "pelvis_deg", "observado", "hip line or kick direction missing"),
        ("thigh_shank_plane_deg", "plane_deg", "inferido", "hip, knee, ankle or toe missing"),
        ("hip_angle_deg", "hip_angle_deg", "observado", "shoulder, hip or knee missing"),
        ("ankle_angle_deg", "ankle_angle_deg", "observado", "knee, ankle or toe missing"),
    )  # fmt: skip
    for name, key, source, why in at_contact:
        value = _value_at(posture[key], contact)
        ok = math.isfinite(value)
        metrics.append(_metric(name, value, "deg", source, "ok" if ok else "missing",
                               "" if ok else why))  # fmt: skip
    for name, key, unit, source, why in (
        ("lateral_foot_speed_mps", "lateral_foot_mps", "m/s", "filtrado", "foot or kick direction missing"),
        ("knee_share_foot_speed", "knee_share", "a.u.", "inferido", "hip, knee, ankle or foot missing"),
    ):  # fmt: skip
        value, got, all_f = _speed_window(posture[key], pre)
        quality = "ok" if got == all_f and all_f else ("partial" if got else "missing")
        reason = why if not got else ("" if quality == "ok" else f"{got}/{all_f} frames valid")
        metrics.append(_metric(name, value, unit, source, quality, reason))  # fmt: skip

    sensitivity = []
    for w in SENSITIVITY_S:
        k = window_frames(fps, w)
        vf, _, _ = _speed_window(foot_speed, range(contact - k, contact))
        vb, _, _ = _speed_window(ball_speed, range(contact, contact + k))
        lat, _, _ = _speed_window(posture["lateral_foot_mps"], range(contact - k, contact))
        share, _, _ = _speed_window(posture["knee_share"], range(contact - k, contact))
        sensitivity.append({
            "window_s": w, "frames": k, "v_foot_pre_mps": _round(vf), "v_ball_post_mps": _round(vb),
            "ratio_vball_vfoot": _round(vb / vf if vf and math.isfinite(vf) and math.isfinite(vb) else math.nan),
            "lateral_foot_mps": _round(lat), "knee_share": _round(share),
        })  # fmt: skip
    _flag_window_sensitive(metrics, sensitivity)

    series = _series(contact, fps, foot_speed, ball_speed, knee_angle, knee_rate, segments, posture)
    context = {
        "points": points, "ball": ball, "rows": rows, "fps": fps, "contact": contact,
        "kicker_id": kicker_id, "kick_foot": side, "support_foot": other, "kick_dir": kick_dir,
        "window_s": nw / fps,
    }  # fmt: skip
    return {"metrics": metrics, "events": events, "series": series, "sensitivity": sensitivity,
            "warnings": warnings, "context": context}  # fmt: skip


def _round(value: float) -> float | str:
    return round(value, 4) if isinstance(value, float) and math.isfinite(value) else ""


def _series(
    contact, fps, foot_speed, ball_speed, knee_angle, knee_rate, segments, posture
) -> list[dict]:
    """Per-frame values from the search window before contact to twice that after it."""
    rates = {}
    for name, unit_vec in segments.items():
        rates[name] = _norm(local_velocity(unit_vec, fps)) * DEG
    span = window_frames(fps, SEARCH_S) + 5
    out = []
    for f in range(max(0, contact - span), min(len(ball_speed), contact + 2 * span)):
        out.append({
            "frame": f,
            "time_rel_s": round((f - contact) / fps, 4),
            "foot_speed_mps": _round(float(foot_speed[f])) if f < len(foot_speed) else "",
            "ball_speed_mps": _round(float(ball_speed[f])),
            "knee_angle_deg": _round(float(knee_angle[f])) if f < len(knee_angle) else "",
            "knee_ext_deg_s": _round(float(knee_rate[f])) if f < len(knee_rate) else "",
            "thigh_w_deg_s": _round(float(rates["thigh"][f])) if "thigh" in rates else "",
            "shank_w_deg_s": _round(float(rates["shank"][f])) if "shank" in rates else "",
            "foot_w_deg_s": _round(float(rates["foot"][f])) if "foot" in rates else "",
        })  # fmt: skip
        out[-1].update({key: _round(float(arr[f])) for key, arr in posture.items()})
    return out


def _figures():
    """The figures module, imported when a figure is written (it loads the stereo helpers)."""
    try:
        from . import freekiki_kicker_figures as figures
    except ImportError:
        import freekiki_kicker_figures as figures  # ty: ignore[unresolved-import]
    return figures


def write_outputs(
    result: dict, out_dir: Path | str, *, views: list[dict] | None = None
) -> dict[str, Path]:
    """``kicker_metrics.csv|json``, ``kicker_series.csv``, ``kicker_sensitivity.csv`` and the figures.

    ``views`` (reference camera first, see :func:`freekiki_kicker_figures.write_figures`) adds the
    reprojection overlays; without it the stick figures and curves are still written.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {
        "metrics": out / "kicker_metrics.csv",
        "json": out / "kicker_metrics.json",
        "series": out / "kicker_series.csv",
        "sensitivity": out / "kicker_sensitivity.csv",
    }
    with paths["metrics"].open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["metric", "value", "unit", "source", "quality", "reason", "definition"]
        )
        writer.writeheader()
        writer.writerows(result["metrics"])
    values = {m["metric"]: m["value"] for m in result["metrics"]}
    paths["json"].write_text(
        json.dumps({"events": result["events"], "metrics": values, "warnings": result["warnings"],
                    "sources": "observado = triangulated keypoint; filtrado = local quadratic fit of a "
                                "triangulated track; ajustado = fitted flight; inferido = from two of these"},
                   indent=2, default=str),
        encoding="utf-8",
    )  # fmt: skip
    if result["series"]:
        _write_rows(paths["series"], result["series"])
    if result["sensitivity"]:
        _write_rows(paths["sensitivity"], result["sensitivity"])
    if result["series"] and result.get("context"):
        paths.update(_figures().write_figures(result, out, views=views))
    return paths


def _write_rows(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
