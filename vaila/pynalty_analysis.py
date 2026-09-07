"""
Project: vailá Multimodal Toolbox
Script: pynalty_analysis.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 07 September 2026
Update Date: 07 September 2026
Version: 0.3.129

Description:
    Pure-numpy kinematics behind the Pynalty penalty analysis. Everything in
    this module is free of OpenCV, pygame, MediaPipe and Tkinter so it can be
    unit-tested headlessly and reused from a CLI batch.

    Coordinate systems
    ------------------
    Goal plane (2D), produced by the DLT2D homography of the four goal corners:
        X in [0, goal_width]  measured from the left post as seen in the video
        Z in [0, goal_height] measured up from the ground
    Flight space (3D), used by the ball-flight model:
        origin on the ground at the penalty spot
        X across the pitch (0 = goal centre line)
        Y toward the goal (0 = spot, goal_line at penalty_distance)
        Z up

License:
    This project is licensed under the terms of AGPLv3.0.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

# ==============================================================================
# Constants and geometry
# ==============================================================================

GRAVITY_MS2 = 9.80665

#: Fastest sustained horizontal translation reported for elite goalkeeper dives.
#: Used only as the ceiling that separates "unsaveable" from "saveable but late".
ELITE_DIVE_SPEED_MS = 4.5

#: Simple visual reaction time of a trained athlete. Reference band only.
ELITE_REACTION_TIME_S = 0.20

ZONE_ROW_LABELS = ("Low", "Middle", "High")
ZONE_COL_LABELS = ("Left", "Centre", "Right")

#: Typical elite male goalkeeper stature used when the user skips step 7.
DEFAULT_GK_HEIGHT_M = 1.88
DEFAULT_KICKER_HEIGHT_M = 1.80

#: User-reported shot result. Geometry alone cannot distinguish a save that
#: happens inside the posts from a goal that crossed the line.
SHOT_OUTCOMES = ("goal", "save", "miss", "woodwork")


@dataclass(frozen=True)
class GoalGeometry:
    """Goal-mouth dimensions and shot distance, all in metres.

    Defaults are the FIFA Laws of the Game full-size goal and penalty mark.
    Override for futsal, youth goals or a free kick taken from another
    distance.
    """

    width: float = 7.32
    height: float = 2.44
    penalty_distance: float = 11.0
    ball_radius: float = 0.11

    @property
    def mid_x(self) -> float:
        return self.width / 2.0

    @property
    def mid_z(self) -> float:
        return self.height / 2.0

    def corner_coords(self) -> np.ndarray:
        """Real (X, Z) of the four goal corners in the marking order.

        Order matches the on-screen instruction: bottom-left, top-left,
        top-right, bottom-right.
        """
        return np.array(
            [
                [0.0, 0.0],
                [0.0, self.height],
                [self.width, self.height],
                [self.width, 0.0],
            ],
            dtype=float,
        )


@dataclass
class Anthropometrics:
    """Goalkeeper and kicker body measurements driving the reach model.

    Only ``gk_height_m`` is really needed; ``arm_span`` and ``standing_reach``
    fall back to standard proportions when omitted. When stature itself is
    omitted, :meth:`resolved` fills :data:`DEFAULT_GK_HEIGHT_M` so reach
    analysis still runs (flagged via :attr:`used_defaults`).
    """

    gk_height_m: float | None = None
    gk_arm_span_m: float | None = None
    gk_standing_reach_m: float | None = None
    kicker_height_m: float | None = None
    #: Extra horizontal hand travel of a full-extension dive, as a fraction of
    #: stature, beyond the standing lateral reach. Exposed because it is an
    #: assumption, not a measurement.
    dive_extension_factor: float = 0.50
    #: True when stature came from :data:`DEFAULT_GK_HEIGHT_M` rather than input.
    used_defaults: bool = False

    def resolved(self, *, use_defaults: bool = True) -> Anthropometrics:
        """Return a copy with the missing measurements filled by proportion."""
        height = self.gk_height_m
        used_defaults = False
        if (not height or height <= 0) and use_defaults:
            height = DEFAULT_GK_HEIGHT_M
            used_defaults = True

        arm_span = self.gk_arm_span_m
        reach = self.gk_standing_reach_m
        kicker = self.kicker_height_m

        if height and height > 0:
            if not arm_span or arm_span <= 0:
                arm_span = 1.02 * height  # elite keepers sit slightly above 1.0
            if not reach or reach <= 0:
                reach = 1.25 * height
        if (not kicker or kicker <= 0) and use_defaults:
            kicker = DEFAULT_KICKER_HEIGHT_M

        return Anthropometrics(
            gk_height_m=height,
            gk_arm_span_m=arm_span,
            gk_standing_reach_m=reach,
            kicker_height_m=kicker,
            dive_extension_factor=self.dive_extension_factor,
            used_defaults=used_defaults or self.used_defaults,
        )

    def is_complete(self) -> bool:
        """True when measured stature is present (defaults alone do not count)."""
        return bool(self.gk_height_m and self.gk_height_m > 0)

    def is_ready(self) -> bool:
        """True when reach analysis can run (measured stature or defaults)."""
        r = self.resolved(use_defaults=True)
        return bool(r.gk_height_m and r.gk_arm_span_m and r.gk_standing_reach_m)


@dataclass
class ReachEnvelope:
    """Radii, in metres, of what the goalkeeper can touch on the goal plane."""

    height_m: float
    arm_span_m: float
    standing_reach_m: float
    radius_standing_m: float
    radius_dive_m: float
    dive_extension_m: float


@dataclass
class BallPathPoint:
    frame: int
    x_px: float
    y_px: float
    source: str = "manual"  # manual | auto | interp


@dataclass
class BallPath:
    points: list[BallPathPoint] = field(default_factory=list)

    def by_frame(self) -> dict[int, BallPathPoint]:
        return {p.frame: p for p in self.points}

    def sorted_points(self) -> list[BallPathPoint]:
        return sorted(self.points, key=lambda p: p.frame)


# ==============================================================================
# DLT2D homography
# ==============================================================================


def dlt2d(F, L) -> np.ndarray:
    """Solve the 8-parameter DLT2D that maps points in ``F`` onto points in ``L``.

    ``F`` holds the source coordinates (for the goal-plane calibration these
    are the real-world corner coordinates) and ``L`` the matching coordinates
    seen by the camera, in the same order. Four non-collinear pairs give an
    exact solution; more pairs are solved in the least-squares sense.

    Feed the result to :func:`rec2d`, which inverts the mapping: ``rec2d`` on
    ``dlt2d(real, pixel)`` reconstructs real coordinates from pixels.
    """
    F = np.asarray(F, dtype=float).reshape(-1, 2)
    L = np.asarray(L, dtype=float).reshape(-1, 2)
    if F.shape[0] != L.shape[0]:
        raise ValueError(f"dlt2d needs matching point counts, got {F.shape[0]} and {L.shape[0]}")
    if F.shape[0] < 4:
        raise ValueError(f"dlt2d needs at least 4 point pairs, got {F.shape[0]}")

    m = F.shape[0]
    C = L.reshape(-1)
    B = np.zeros((2 * m, 8), dtype=float)
    for i in range(m):
        B[2 * i, 0] = F[i, 0]
        B[2 * i, 1] = F[i, 1]
        B[2 * i, 2] = 1.0
        B[2 * i, 6] = -F[i, 0] * L[i, 0]
        B[2 * i, 7] = -F[i, 1] * L[i, 0]
        B[2 * i + 1, 3] = F[i, 0]
        B[2 * i + 1, 4] = F[i, 1]
        B[2 * i + 1, 5] = 1.0
        B[2 * i + 1, 6] = -F[i, 0] * L[i, 1]
        B[2 * i + 1, 7] = -F[i, 1] * L[i, 1]

    A, *_ = np.linalg.lstsq(B, C, rcond=None)
    return A.reshape(8, 1)


def rec2d(A, cc2d) -> np.ndarray:
    """Invert a :func:`dlt2d` mapping: recover source coords from target coords.

    ``A`` is the 8-parameter vector, ``cc2d`` an (n, 2) array of target-space
    points. Returns an (n, 2) array in source space.
    """
    a = np.asarray(A, dtype=float).reshape(-1)
    pts = np.asarray(cc2d, dtype=float).reshape(-1, 2)
    out = np.zeros((pts.shape[0], 2), dtype=float)
    for k in range(pts.shape[0]):
        x, y = pts[k, 0], pts[k, 1]
        M = np.array(
            [
                [a[0] - x * a[6], a[1] - x * a[7]],
                [a[3] - y * a[6], a[4] - y * a[7]],
            ],
            dtype=float,
        )
        b = np.array([x - a[2], y - a[5]], dtype=float)
        out[k, :] = np.linalg.solve(M, b)
    return out


def pixel_to_goal_plane(calib_pixels, points_px, goal: GoalGeometry | None = None) -> np.ndarray:
    """Map image points onto goal-plane (X, Z) metres via the corner homography."""
    goal = goal or GoalGeometry()
    dlt = dlt2d(goal.corner_coords(), np.asarray(calib_pixels, dtype=float))
    return rec2d(dlt, points_px)


def goal_plane_to_pixel(calib_pixels, points_real, goal: GoalGeometry | None = None) -> np.ndarray:
    """Project goal-plane (X, Z) metres back into image pixels."""
    goal = goal or GoalGeometry()
    dlt_inv = dlt2d(np.asarray(calib_pixels, dtype=float), goal.corner_coords())
    return rec2d(dlt_inv, points_real)


def calibration_residual_px(calib_pixels, goal: GoalGeometry | None = None) -> float:
    """RMS reprojection error, in pixels, of the calibration corners.

    With exactly four corners the homography is exact by construction and this
    is ~0; it only becomes informative if the corner set is ever extended.
    """
    goal = goal or GoalGeometry()
    calib = np.asarray(calib_pixels, dtype=float).reshape(-1, 2)
    reproj = goal_plane_to_pixel(calib, goal.corner_coords(), goal)
    return float(np.sqrt(np.mean(np.sum((reproj - calib) ** 2, axis=1))))


def is_convex_quadrilateral(pts) -> bool:
    """True when the four points form a convex, non-self-intersecting quad."""
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if p.shape[0] != 4:
        return False
    signs = []
    for i in range(4):
        a, b, c = p[i], p[(i + 1) % 4], p[(i + 2) % 4]
        cross = (b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])
        if abs(cross) < 1e-9:
            return False
        signs.append(cross > 0)
    return all(signs) or not any(signs)


def order_goal_corners(pts) -> np.ndarray:
    """Reorder four clicked corners into bottom-left, top-left, top-right, bottom-right.

    Image coordinates grow downward, so "top" means the smaller Y. Used to
    rescue a calibration whose corners were clicked out of order.
    """
    p = np.asarray(pts, dtype=float).reshape(-1, 2)
    if p.shape[0] != 4:
        raise ValueError(f"expected 4 corners, got {p.shape[0]}")
    left = p[np.argsort(p[:, 0])][:2]
    right = p[np.argsort(p[:, 0])][2:]
    bl, tl = left[np.argsort(-left[:, 1])]
    br, tr = right[np.argsort(-right[:, 1])]
    return np.array([bl, tl, tr, br], dtype=float)


# ==============================================================================
# Ball flight
# ==============================================================================


def ball_flight_distance(
    entry_x_m: float,
    entry_z_m: float,
    goal: GoalGeometry | None = None,
) -> float:
    """Straight-line 3D distance from the penalty spot to the ball's goal-line point."""
    goal = goal or GoalGeometry()
    z = max(entry_z_m, goal.ball_radius)
    start = np.array([0.0, 0.0, goal.ball_radius])
    end = np.array([entry_x_m - goal.mid_x, goal.penalty_distance, z])
    return float(np.linalg.norm(end - start))


def distvelball_penalti(phorz, pvert, nframes, fpsvideo=30):
    """Distance and speed of the shot. Kept for backward compatibility.

    ``phorz`` and ``pvert`` are the reconstructed goal-plane X and Z in metres,
    ``nframes`` the frame count between contact and the goal line.
    """
    goal = GoalGeometry()
    distball = ball_flight_distance(float(phorz), float(pvert), goal)
    if nframes > 0:
        velball_ms = distball / nframes * fpsvideo
        velball_kmh = velball_ms * 3.6
    else:
        velball_ms = 0.0
        velball_kmh = 0.0
    return distball, velball_ms, velball_kmh


def goal_zone(
    entry_x_m: float,
    entry_z_m: float,
    goal: GoalGeometry | None = None,
) -> tuple[int, int, int, str]:
    """Classify the entry point on the 3x3 target grid.

    Returns ``(row, col, index, label)`` where row 0 is low and col 0 is the
    left third as seen in the video. ``index`` runs 0..8 left-to-right,
    bottom-to-top.
    """
    goal = goal or GoalGeometry()
    col = int(np.clip(entry_x_m / (goal.width / 3.0), 0, 2.999))
    row = int(np.clip(max(entry_z_m, 0.0) / (goal.height / 3.0), 0, 2.999))
    label = f"{ZONE_ROW_LABELS[row]} {ZONE_COL_LABELS[col]}"
    return row, col, row * 3 + col, label


def placement_index(
    entry_x_m: float,
    entry_z_m: float,
    goal: GoalGeometry | None = None,
) -> float:
    """Heuristic 0-100 corner-seeking index for the shot placement.

    Rewards distance from the goal centre and penalises shots so close to a
    post or the crossbar that a small error would miss. This is a coaching
    heuristic for ranking shots within a session, not a validated metric.
    """
    goal = goal or GoalGeometry()
    corner_dist = math.hypot(goal.mid_x, goal.mid_z)
    centre_dist = math.hypot(entry_x_m - goal.mid_x, entry_z_m - goal.mid_z)
    spread = min(1.0, centre_dist / corner_dist) if corner_dist > 0 else 0.0

    margin = min(entry_x_m, goal.width - entry_x_m, goal.height - entry_z_m)
    danger_band = 2.0 * goal.ball_radius
    risk = float(np.clip((danger_band - margin) / danger_band, 0.0, 1.0))
    return float(np.clip(100.0 * spread * (1.0 - 0.5 * risk), 0.0, 100.0))


def is_inside_goal(entry_x_m: float, entry_z_m: float, goal: GoalGeometry | None = None) -> bool:
    goal = goal or GoalGeometry()
    return 0.0 <= entry_x_m <= goal.width and 0.0 <= entry_z_m <= goal.height


def fit_ball_flight_3d(
    entry_x_m: float,
    entry_z_m: float,
    flight_time_s: float,
    n_samples: int = 60,
    goal: GoalGeometry | None = None,
) -> dict:
    """Build a gravity-consistent 3D flight path from the spot to the goal line.

    Horizontal motion is treated as uniform and the vertical component as a
    parabola whose initial velocity is solved so the ball arrives at
    ``entry_z_m`` exactly at ``flight_time_s``. Aerodynamic drag and spin are
    neglected, so this is a plausible reconstruction consistent with the two
    measured endpoints and the measured flight time, not a measured trajectory.

    Returns a dict with the sampled path ``(n, 4)`` as ``t, X, Y, Z`` plus
    launch angles, apex height and the initial speed.
    """
    goal = goal or GoalGeometry()
    n_samples = max(2, int(n_samples))
    x0, y0, z0 = 0.0, 0.0, goal.ball_radius
    x1 = entry_x_m - goal.mid_x
    y1 = goal.penalty_distance
    z1 = max(entry_z_m, goal.ball_radius)

    T = float(flight_time_s)
    if not np.isfinite(T) or T <= 0:
        t = np.zeros(n_samples)
        path = np.column_stack(
            [
                t,
                np.linspace(x0, x1, n_samples),
                np.linspace(y0, y1, n_samples),
                np.full(n_samples, z1),
            ]
        )
        return {
            "path": path,
            "vx_ms": 0.0,
            "vy_ms": 0.0,
            "vz0_ms": 0.0,
            "speed0_ms": 0.0,
            "launch_elevation_deg": 0.0,
            "launch_azimuth_deg": 0.0,
            "apex_z_m": z1,
            "apex_time_s": 0.0,
        }

    vx = (x1 - x0) / T
    vy = (y1 - y0) / T
    vz0 = (z1 - z0) / T + 0.5 * GRAVITY_MS2 * T

    t = np.linspace(0.0, T, n_samples)
    X = x0 + vx * t
    Y = y0 + vy * t
    Z = z0 + vz0 * t - 0.5 * GRAVITY_MS2 * t**2
    path = np.column_stack([t, X, Y, Z])

    horiz = math.hypot(vx, vy)
    apex_time = float(np.clip(vz0 / GRAVITY_MS2, 0.0, T))
    apex_z = z0 + vz0 * apex_time - 0.5 * GRAVITY_MS2 * apex_time**2

    return {
        "path": path,
        "vx_ms": vx,
        "vy_ms": vy,
        "vz0_ms": vz0,
        "speed0_ms": math.sqrt(vx * vx + vy * vy + vz0 * vz0),
        "launch_elevation_deg": math.degrees(math.atan2(vz0, horiz)) if horiz > 0 else 0.0,
        "launch_azimuth_deg": math.degrees(math.atan2(vx, vy)) if vy != 0 else 0.0,
        "apex_z_m": float(apex_z),
        "apex_time_s": apex_time,
    }


def flight_speed_at(flight_path, times) -> np.ndarray:
    """Model speed, in m/s, of the fitted 3D flight at the requested times.

    Use this rather than differentiating the pixel path: pixel displacements
    can only be scaled to metres on the goal plane, so a ball still near the
    penalty spot would come out far too fast.
    """
    arr = np.asarray(flight_path, dtype=float)
    t = np.atleast_1d(np.asarray(times, dtype=float))
    if arr.ndim != 2 or arr.shape[0] < 2:
        return np.full(t.shape, np.nan)
    ts = arr[:, 0]
    speed = np.linalg.norm(np.gradient(arr[:, 1:4], ts, axis=0), axis=1)
    out = np.interp(t, ts, speed, left=np.nan, right=np.nan)
    return out


def interpolate_ball_path(
    path: BallPath,
    first_frame: int,
    last_frame: int,
) -> BallPath:
    """Fill frame gaps between marked ball positions by linear interpolation.

    Marked points are preserved verbatim; generated points are tagged
    ``interp``. Frames outside the marked span are not extrapolated.
    """
    known = path.by_frame()
    frames = sorted(known)
    if len(frames) < 2:
        return BallPath(points=list(path.sorted_points()))

    lo = max(first_frame, frames[0])
    hi = min(last_frame, frames[-1])
    xs = np.array([known[f].x_px for f in frames], dtype=float)
    ys = np.array([known[f].y_px for f in frames], dtype=float)
    fa = np.array(frames, dtype=float)

    out: list[BallPathPoint] = []
    for f in range(lo, hi + 1):
        if f in known:
            out.append(known[f])
        else:
            out.append(
                BallPathPoint(
                    frame=f,
                    x_px=float(np.interp(f, fa, xs)),
                    y_px=float(np.interp(f, fa, ys)),
                    source="interp",
                )
            )
    return BallPath(points=out)


def ball_path_speed_series(
    path: BallPath,
    calib_pixels,
    fps: float,
    goal: GoalGeometry | None = None,
) -> list[dict]:
    """Per-frame speed along the marked pixel path, in pixels and in metres.

    The metric conversion uses the goal-plane scale at the ball's height, which
    is only exact on the goal plane itself. Points closer to the camera are
    therefore over-estimated; the value is reported as an indicative rate and
    the report says so.
    """
    goal = goal or GoalGeometry()
    pts = path.sorted_points()
    if not pts:
        return []

    try:
        real = pixel_to_goal_plane(calib_pixels, [[p.x_px, p.y_px] for p in pts], goal)
    except Exception:
        real = np.full((len(pts), 2), np.nan)

    rows: list[dict] = []
    for i, p in enumerate(pts):
        row = {
            "frame": p.frame,
            "time_s": p.frame / fps if fps > 0 else 0.0,
            "x_px": p.x_px,
            "y_px": p.y_px,
            "source": p.source,
            "plane_x_m": float(real[i, 0]),
            "plane_z_m": float(real[i, 1]),
            "speed_px_s": float("nan"),
            "plane_speed_ms": float("nan"),
        }
        if i > 0 and fps > 0:
            prev = pts[i - 1]
            dframes = p.frame - prev.frame
            if dframes > 0:
                dt = dframes / fps
                row["speed_px_s"] = math.hypot(p.x_px - prev.x_px, p.y_px - prev.y_px) / dt
                if np.isfinite(real[i, 0]) and np.isfinite(real[i - 1, 0]):
                    d = math.hypot(real[i, 0] - real[i - 1, 0], real[i, 1] - real[i - 1, 1])
                    row["plane_speed_ms"] = d / dt
        rows.append(row)
    return rows


# ==============================================================================
# Goalkeeper reach and timing
# ==============================================================================


def gk_reach_envelope(anthro: Anthropometrics) -> ReachEnvelope | None:
    """Reach radii on the goal plane for the given body measurements.

    ``radius_standing_m`` is half the arm span: what the keeper covers with the
    feet planted. ``radius_dive_m`` adds ``dive_extension_factor`` x stature to
    represent a full-extension dive from a set position. Both are treated as
    isotropic radii around the keeper's marked centre, which is a
    simplification: real envelopes are wider laterally than vertically.
    """
    r = anthro.resolved()
    if not (r.gk_height_m and r.gk_arm_span_m and r.gk_standing_reach_m):
        return None
    radius_standing = r.gk_arm_span_m / 2.0
    dive_extension = r.dive_extension_factor * r.gk_height_m
    return ReachEnvelope(
        height_m=r.gk_height_m,
        arm_span_m=r.gk_arm_span_m,
        standing_reach_m=r.gk_standing_reach_m,
        radius_standing_m=radius_standing,
        radius_dive_m=radius_standing + dive_extension,
        dive_extension_m=dive_extension,
    )


def reaction_classification(reaction_time_s: float) -> tuple[str, str]:
    """Label a signed reaction time, where negative means the keeper anticipated.

    Returns ``(key, human_label)``. The key is stable for CSV/database use, the
    label is the phrasing used in the report.
    """
    t = float(reaction_time_s)
    if t < -0.05:
        return "anticipation", f"Anticipated contact by {abs(t) * 1000:.0f} ms"
    if t <= 0.05:
        return "simultaneous", "Moved with contact (within 50 ms)"
    if t <= 0.25:
        return "fast_reactive", f"Fast reaction ({t * 1000:.0f} ms)"
    if t <= 0.40:
        return "reactive", f"Reactive ({t * 1000:.0f} ms)"
    return "late", f"Late reaction ({t * 1000:.0f} ms)"


def nearest_body_point_to_ball(
    body_points_xz,
    ball_xz,
) -> tuple[np.ndarray, float, int]:
    """Pick the body point on the goal plane closest to the ball entry.

    ``body_points_xz`` is ``(N, 2)``. Returns ``(point_xz, gap_m, index)``.
    """
    pts = np.asarray(body_points_xz, dtype=float).reshape(-1, 2)
    ball = np.asarray(ball_xz, dtype=float).reshape(2)
    if pts.size == 0:
        raise ValueError("nearest_body_point_to_ball needs at least one body point")
    finite = np.isfinite(pts).all(axis=1)
    if not finite.any():
        raise ValueError("nearest_body_point_to_ball: all body points are non-finite")
    usable = pts[finite]
    d = np.linalg.norm(usable - ball, axis=1)
    i = int(np.argmin(d))
    # Map back to the original index among finite rows.
    orig = int(np.flatnonzero(finite)[i])
    return usable[i].copy(), float(d[i]), orig


def classify_shot_outcome(
    outcome: str | None,
    *,
    ball_inside_goal: bool,
) -> tuple[str, str]:
    """Normalise a user outcome and build a short coaching label.

    Returns ``(key, label)``. Unknown or empty input falls back to geometry:
    inside posts → ``goal``, otherwise ``miss``. A save must be stated by the
    user because the DLT entry point alone cannot prove a stop.
    """
    key = (outcome or "").strip().lower()
    if key in {"gol", "golo"}:
        key = "goal"
    if key in {"defesa", "parada", "saved"}:
        key = "save"
    if key in {"fora", "wide", "over"}:
        key = "miss"
    if key in {"trave", "crossbar", "post", "barra"}:
        key = "woodwork"
    if key not in SHOT_OUTCOMES:
        key = "goal" if ball_inside_goal else "miss"
    labels = {
        "goal": "Goal",
        "save": "Save",
        "miss": "Miss (off target)",
        "woodwork": "Woodwork (post/crossbar)",
    }
    return key, labels[key]


def gk_reach_time(
    gk_start_xz,
    ball_entry_xz,
    envelope: ReachEnvelope | None,
    flight_time_s: float,
    reaction_time_s: float,
    measured_dive_speed_ms: float,
    elite_dive_speed_ms: float = ELITE_DIVE_SPEED_MS,
    *,
    gk_reference_xz=None,
    gap_source: str = "marked_centre",
) -> dict:
    """Distance and time the goalkeeper needs to touch the ball.

    By default the gap is measured on the goal plane between the keeper's
    marked centre at contact and the ball's entry point. Pass
    ``gk_reference_xz`` (e.g. the nearest pose landmark or the defending hand)
    to measure from that body point instead; ``gap_source`` records which
    reference was used.
    """
    gk = np.asarray(
        gk_reference_xz if gk_reference_xz is not None else gk_start_xz, dtype=float
    ).reshape(2)
    ball = np.asarray(ball_entry_xz, dtype=float).reshape(2)
    dx = float(ball[0] - gk[0])
    dz = float(ball[1] - gk[1])
    gap = math.hypot(dx, dz)

    r_stand = envelope.radius_standing_m if envelope else float("nan")
    r_dive = envelope.radius_dive_m if envelope else float("nan")

    travel_standing = max(0.0, gap - r_stand) if envelope else float("nan")
    travel_dive = max(0.0, gap - r_dive) if envelope else float("nan")

    available = float(flight_time_s) - max(0.0, float(reaction_time_s))
    available = max(available, 0.0)

    required_speed = float("nan")
    if envelope and available > 0:
        required_speed = travel_dive / available if travel_dive > 0 else 0.0

    t_reach = float("nan")
    if envelope:
        if travel_dive <= 0:
            t_reach = 0.0
        elif measured_dive_speed_ms > 0:
            t_reach = travel_dive / measured_dive_speed_ms
        else:
            t_reach = float("inf")

    time_margin = available - t_reach if np.isfinite(t_reach) else float("-inf")

    return {
        "gap_m": gap,
        "gap_horizontal_m": abs(dx),
        "gap_vertical_m": abs(dz),
        "gap_source": gap_source,
        "gap_ref_x_m": float(gk[0]),
        "gap_ref_z_m": float(gk[1]),
        "reach_standing_m": r_stand,
        "reach_dive_m": r_dive,
        "travel_standing_m": travel_standing,
        "travel_dive_m": travel_dive,
        "available_time_s": available,
        "required_dive_speed_ms": required_speed,
        "time_to_reach_s": t_reach,
        "time_margin_s": time_margin,
        "reachable_standing": bool(envelope and gap <= r_stand),
        "reachable_dive": bool(envelope and gap <= r_dive),
        "elite_dive_speed_ms": elite_dive_speed_ms,
    }


def dive_side(
    gk_kick_x: float,
    gk_goal_x: float,
    ball_entry_x: float,
    min_move_m: float = 0.15,
) -> tuple[str, str]:
    """Whether the keeper committed to the same side the ball went.

    Returns ``(key, human_label)``.
    """
    ball_dir = ball_entry_x - gk_kick_x
    gk_dir = gk_goal_x - gk_kick_x
    if abs(gk_dir) < min_move_m:
        return "stayed_central", "Stayed central (no committed dive)"
    if abs(ball_dir) < min_move_m:
        return "ball_central", "Ball came at the keeper's body"
    if math.copysign(1.0, ball_dir) == math.copysign(1.0, gk_dir):
        return "correct_side", "Dived to the correct side"
    return "wrong_side", "Dived to the wrong side"


def save_verdict(reach: dict, envelope: ReachEnvelope | None) -> tuple[str, str]:
    """Three-way saveability verdict from the reach/time analysis.

    Returns ``(key, human_label)``.
    """
    if envelope is None:
        return "unknown", "Not assessed (goalkeeper anthropometrics missing)"
    if reach.get("reachable_standing"):
        return "reachable_standing", "Within standing reach, no dive required"

    required = reach.get("required_dive_speed_ms", float("nan"))
    elite = reach.get("elite_dive_speed_ms", ELITE_DIVE_SPEED_MS)
    if not np.isfinite(required) or required > elite:
        return "unsaveable", "Unsaveable: beyond an elite dive within the flight time"

    margin = reach.get("time_margin_s", float("-inf"))
    if not np.isfinite(margin) or margin < 0:
        return "saveable_late", "Saveable, but the keeper arrived late"
    return "saveable", "Saveable: reach and timing were both sufficient"


def compute_penalty_metrics(
    *,
    calib_pixels,
    kick_frame: int,
    goal_frame: int,
    gk_move_frame: int,
    kick_ball_px,
    kick_gk_px,
    goal_ball_px,
    goal_gk_px,
    fps: float,
    goal: GoalGeometry | None = None,
    anthro: Anthropometrics | None = None,
    elite_dive_speed_ms: float = ELITE_DIVE_SPEED_MS,
    shot_outcome: str | None = None,
    gk_body_points_px=None,
    gk_body_landmark_names: list[str] | None = None,
    defending_landmark_hint: str | None = None,
) -> dict:
    """Full metric set for one penalty, from the marked pixel coordinates.

    Returns a flat dictionary. The legacy keys used by the pygame overlay and
    the original ``results.csv`` (``dist``, ``vel_ms``, ``vel_kmh``, ``delta``,
    ``coord_x``, ``coord_z``, ``gk_response_time``, ``gk_dist``, ``gk_vel_ms``,
    ``gk_vel_kmh``) are preserved.

    When ``gk_body_points_px`` is provided (pose landmarks at the goal-line
    frame), the gap to the ball uses the nearest body point on the goal plane,
    preferring a defending-hand hint when given. Otherwise the marked keeper
    centre at contact is used.
    """
    goal = goal or GoalGeometry()
    raw_anthro = anthro or Anthropometrics()
    anthro = raw_anthro.resolved(use_defaults=True)

    calib = np.asarray(calib_pixels, dtype=float).reshape(-1, 2)
    if calib.shape[0] != 4:
        raise ValueError(f"calibration needs exactly 4 corners, got {calib.shape[0]}")

    dlt = dlt2d(goal.corner_coords(), calib)
    goal_ball_real = rec2d(dlt, [goal_ball_px])[0]
    gk_kick_real = rec2d(dlt, [kick_gk_px])[0]
    gk_goal_real = rec2d(dlt, [goal_gk_px])[0]
    kick_ball_real = rec2d(dlt, [kick_ball_px])[0] if kick_ball_px is not None else None

    entry_x, entry_z = float(goal_ball_real[0]), float(goal_ball_real[1])

    # --- Ball ---
    delta_frames = int(goal_frame) - int(kick_frame)
    flight_time = delta_frames / fps if fps > 0 else 0.0
    dist = ball_flight_distance(entry_x, entry_z, goal)
    vel_ms = dist / flight_time if flight_time > 0 else 0.0
    vel_kmh = vel_ms * 3.6

    row, col, zone_idx, zone_label = goal_zone(entry_x, entry_z, goal)
    flight = fit_ball_flight_3d(entry_x, entry_z, flight_time, goal=goal)
    inside = is_inside_goal(entry_x, entry_z, goal)
    outcome_key, outcome_label = classify_shot_outcome(shot_outcome, ball_inside_goal=inside)

    # --- Goalkeeper ---
    gk_response_frames = int(gk_move_frame) - int(kick_frame)
    gk_response_time = gk_response_frames / fps if fps > 0 else 0.0
    gk_dist = float(np.linalg.norm(gk_goal_real - gk_kick_real))
    delta_jump_frames = int(goal_frame) - int(gk_move_frame)
    gk_vel_ms = gk_dist / delta_jump_frames * fps if delta_jump_frames > 0 and fps > 0 else 0.0
    gk_vel_kmh = gk_vel_ms * 3.6

    gap_ref = gk_kick_real
    gap_source = "marked_centre"
    gap_landmark = ""
    if gk_body_points_px is not None:
        body_px = np.asarray(gk_body_points_px, dtype=float).reshape(-1, 2)
        try:
            body_real = rec2d(dlt, body_px)
            names = list(gk_body_landmark_names or [])
            # Prefer a defending limb when the caller names one (wrist/hand).
            preferred_idx = None
            if defending_landmark_hint and names:
                hint = defending_landmark_hint.lower()
                for i, name in enumerate(names):
                    if hint in name.lower():
                        preferred_idx = i
                        break
            if preferred_idx is not None and np.isfinite(body_real[preferred_idx]).all():
                gap_ref = body_real[preferred_idx]
                gap_source = "defending_part"
                gap_landmark = names[preferred_idx] if preferred_idx < len(names) else str(preferred_idx)
            else:
                gap_ref, _gap, idx = nearest_body_point_to_ball(body_real, (entry_x, entry_z))
                gap_source = "nearest_body_point"
                gap_landmark = names[idx] if idx < len(names) else str(idx)
        except Exception:
            gap_ref = gk_kick_real
            gap_source = "marked_centre"

    envelope = gk_reach_envelope(anthro)
    reach = gk_reach_time(
        gk_kick_real,
        (entry_x, entry_z),
        envelope,
        flight_time,
        gk_response_time,
        gk_vel_ms,
        elite_dive_speed_ms,
        gk_reference_xz=gap_ref,
        gap_source=gap_source,
    )
    reaction_key, reaction_label = reaction_classification(gk_response_time)
    side_key, side_label = dive_side(float(gk_kick_real[0]), float(gk_goal_real[0]), entry_x)
    verdict_key, verdict_label = save_verdict(reach, envelope)

    out: dict = {
        # legacy keys
        "dist": dist,
        "vel_ms": vel_ms,
        "vel_kmh": vel_kmh,
        "delta": delta_frames,
        "coord_x": entry_x,
        "coord_z": entry_z,
        "gk_response_time": gk_response_time,
        "gk_dist": gk_dist,
        "gk_vel_ms": gk_vel_ms,
        "gk_vel_kmh": gk_vel_kmh,
        # frames and timing
        "fps": float(fps),
        "kick_frame": int(kick_frame),
        "goal_frame": int(goal_frame),
        "gk_move_frame": int(gk_move_frame),
        "flight_frames": delta_frames,
        "flight_time_s": flight_time,
        "gk_response_frames": gk_response_frames,
        "gk_dive_frames": delta_jump_frames,
        "gk_dive_time_s": delta_jump_frames / fps if fps > 0 else 0.0,
        # ball placement + outcome
        "ball_entry_x_m": entry_x,
        "ball_entry_z_m": entry_z,
        "ball_entry_x_from_centre_m": entry_x - goal.mid_x,
        "ball_inside_goal": inside,
        "shot_outcome": outcome_key,
        "shot_outcome_label": outcome_label,
        "zone_row": row,
        "zone_col": col,
        "zone_index": zone_idx,
        "zone_label": zone_label,
        "placement_index": placement_index(entry_x, entry_z, goal),
        "dist_to_nearest_post_m": min(entry_x, goal.width - entry_x),
        "dist_to_crossbar_m": goal.height - entry_z,
        "dist_to_ground_m": entry_z,
        # flight model
        "launch_elevation_deg": flight["launch_elevation_deg"],
        "launch_azimuth_deg": flight["launch_azimuth_deg"],
        "apex_height_m": flight["apex_z_m"],
        "apex_time_s": flight["apex_time_s"],
        # goalkeeper interpretation
        "gk_start_x_m": float(gk_kick_real[0]),
        "gk_start_z_m": float(gk_kick_real[1]),
        "gk_end_x_m": float(gk_goal_real[0]),
        "gk_end_z_m": float(gk_goal_real[1]),
        "reaction_class": reaction_key,
        "reaction_label": reaction_label,
        "dive_side_class": side_key,
        "dive_side_label": side_label,
        "verdict_class": verdict_key,
        "verdict_label": verdict_label,
        "gap_landmark": gap_landmark,
        "anthro_source": "default" if anthro.used_defaults else "measured",
        # geometry and calibration provenance
        "goal_width_m": goal.width,
        "goal_height_m": goal.height,
        "penalty_distance_m": goal.penalty_distance,
        "ball_radius_m": goal.ball_radius,
        "calibration_residual_px": calibration_residual_px(calib, goal),
        "calibration_convex": is_convex_quadrilateral(calib),
    }
    out.update(reach)

    if envelope:
        out.update(
            {
                "gk_height_m": envelope.height_m,
                "gk_arm_span_m": envelope.arm_span_m,
                "gk_standing_reach_m": envelope.standing_reach_m,
                "gk_dive_extension_m": envelope.dive_extension_m,
            }
        )
    if anthro.kicker_height_m:
        out["kicker_height_m"] = anthro.kicker_height_m
    if kick_ball_real is not None:
        out["kick_ball_plane_x_m"] = float(kick_ball_real[0])
        out["kick_ball_plane_z_m"] = float(kick_ball_real[1])

    out["_flight_path"] = flight["path"]
    return out
