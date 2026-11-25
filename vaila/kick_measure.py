"""
================================================================================
Script: kick_measure.py - measure a FreeKiki kick analysis frame by frame
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.19
Created: 23 November 2026
Update Date: 24 November 2026

Description:
    Pure logic (no GUI) behind the "Kick Measure" tools of getpixelvideo/quickmeasure: open a
    ``kick_analysis_*`` folder of FreeKiki and turn any pixel of its reference video into metres
    of the pitch, in any frame, not only in the frames the pipeline could calibrate.

    Pitch frame (FreeKiki): ``x`` along the pitch, ``y`` across, ``z`` up, metres, origin at the
    centre spot; one frame of the video is one time step of ``1 / fps`` seconds (0-based index).

    * :func:`load_kick_measure_run` -> :class:`KickMeasureRun` (video, fps, cameras, ball track in
      pixels, people per frame);
    * :class:`CameraTrack` - one 3x4 projection matrix per frame. A frame with a FreeKiki DLT3D
      keeps it untouched (``source == "dlt3d"``). A frame without one is solved with a **fixed
      optical centre** (median of the 10 calibrated frames nearest in time; a single median of
      all of them failed on a ~7000 px lens, where the DLT3D centre drifts metres along the
      viewing axis): only rotation and zoom are free per
      frame, fitted to the raw pitch keypoints of ``field_kps_raw.csv`` (>= 3 points, confidence
      >= 0.5, robust loss) and propagated frame to frame (``source == "solved"``). A frame
      with too few points copies the nearest good frame and is flagged ``approximate=True``,
      unless it can be **tracked** (``source == "tracked"``): the static scene (crowd, boards)
      is followed from frame to frame with Lucas-Kanade (:func:`track_features`, people boxes,
      ball and broadcast overlays masked), the rotation/zoom of every frame is fitted to those
      matches, and the pitch is carried from the known cameras on both sides of the gap
      (:func:`track_approximate_frames`; :func:`validate_tracking` is its hold-out test);
      The fixed centre is a broadcast-camera hypothesis (camera on a tripod): a cut of camera
      is not detected here, so a cut needs its own run;
    * per-frame calibration files next to the SAM 3 outputs: :func:`write_frame_calibration`
      writes ``<stem>_frames.dlt3d`` / ``.dlt2d`` (one row per video frame, the layout
      ``QuickMeasureCalibration.from_dlt2d_sequence`` reads) and ``<stem>_frames_cameras.csv`` /
      ``.json`` (source, residual, anchor of every frame); :func:`read_frame_calibration`,
      :func:`find_frame_calibration`, :func:`ground_homography` and
      :meth:`CameraTrack.homography` read them back and :func:`load_kick_measure_run` prefers
      them when they exist;
    * operations on one frame: :func:`ground_point`, :func:`height_base_top`, :class:`LinePlane`;
    * time: :func:`event_time`, :func:`mean_speed`, :func:`spin_rate`;
    * paths: :func:`path_length`, :func:`fit_arc`, :func:`fit_parabola`;
    * kinematics: :func:`velocity_series`, :func:`acceleration_series`, :func:`speed_series`;
    * people and layers of the Kick Measure panel: :func:`load_people_info` (role kicker /
      keeper / referee / other, team A / B, frames, which poses exist), :func:`load_pose_2d`
      (``(frame, id) -> (70, 2)`` px), :func:`pose_connections` (MHR70 bones) and
      :func:`field_segments_px` (pitch lines + centre circle projected on a frame);
    * metrics per frame on a person: :func:`metric_series` -> :class:`MetricSeries` for the kinds
      in :data:`METRIC_KINDS` (``speed``, ``distance_ball``, ``distance_person``, ``height_kpt``,
      ``jump``, ``kpt_speed``) on top of :func:`person_ground_series`, :func:`ball_ground_series`
      and :func:`body_point_series` (3D points of :data:`BODY_POINTS`);
    * export of a scene: :func:`export_scene` writes a C3D (+ CSV) for the mkvis3d viewer
      (:func:`find_mkvis3d`, :func:`mkvis3d_argv`), a BVH per person with a ball path, pitch lines
      and a plain ``bpy`` companion script for Blender, and the metrics table.

    Command line::

        uv run --no-sync python -m vaila.kick_measure --run <kick_analysis_dir> [--info]
        uv run --no-sync python -m vaila.kick_measure --run <dir> --export cameras.json
        uv run --no-sync python -m vaila.kick_measure --run <dir> --write-cameras
        uv run --no-sync python -m vaila.kick_measure --run <dir> --validate-tracking 5,10,20,30
        uv run --no-sync python -m vaila.kick_measure --run <dir> --export-scene OUT \
            --people 21,23 --metric speed:21 --metric height_kpt:23:right_wrist \
            --metric distance_person:21:23 [--no-blender] [--no-mkvis3d]

    One camera sees the ball along a ray, so a height always depends on the plane chosen to cut
    that ray. Every height function documents its plane; :meth:`LinePlane.lateral_sensitivity`
    gives how much the height moves when that plane is shifted sideways.
================================================================================
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import os
import re
import sys
import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import least_squares
from scipy.signal import savgol_filter
from scipy.sparse import coo_matrix
from scipy.spatial.transform import Rotation

__all__ = [
    "BALL_RADIUS_M",
    "BODY_POINTS",
    "METRIC_KINDS",
    "ArcFit",
    "CameraTrack",
    "FrameCamera",
    "KickMeasureRun",
    "LinePlane",
    "MetricSeries",
    "ParabolaFit",
    "Person",
    "PersonInfo",
    "SpinResult",
    "acceleration_series",
    "ball_ground_series",
    "body_point_series",
    "FRAME_CAL_SUFFIX",
    "OVERLAY_MARGINS",
    "SOURCE_TRACKED",
    "TRACKING_MODEL_VERSION",
    "FeatureTracks",
    "camera_centre",
    "event_time",
    "event_time_uncertainty",
    "export_scene",
    "field_segments_px",
    "find_frame_calibration",
    "find_mkvis3d",
    "format_validation",
    "fit_arc",
    "fit_parabola",
    "ground_homography",
    "ground_point",
    "height_base_top",
    "load_field_template",
    "load_kick_measure_run",
    "load_people_info",
    "load_person_poses",
    "load_pose_2d",
    "mean_speed",
    "metric_series",
    "mkvis3d_argv",
    "path_length",
    "person_ground_series",
    "person_ground_source",
    "pose_connections",
    "project_points",
    "ray_from_pixel",
    "read_frame_calibration",
    "sam_output_dir",
    "spin_rate",
    "speed_series",
    "track_approximate_frames",
    "track_features",
    "validate_tracking",
    "velocity_series",
    "write_frame_calibration",
    "write_run_frame_calibration",
]

BALL_RADIUS_M = 0.11  # ball centre height when it rests on the grass (FreeKiki)
KICK_STATUS = "pipeline_status.json"
SOURCE_DLT3D = "dlt3d"
SOURCE_SOLVED = "solved"
SOURCE_APPROX = "approximate"
SOURCE_GIVEN = "given"

# ----------------------------------------------------------------------------------------------
# Projection helpers (pure NumPy, one 3x4 matrix P: pixel ~ P @ [X, Y, Z, 1])
# ----------------------------------------------------------------------------------------------


def project_points(P: np.ndarray, X: np.ndarray) -> np.ndarray:
    """Pixels ``(u, v)`` of the pitch points ``X`` seen by the 3x4 matrix ``P``.

    Signature: ``project_points(P (3, 4), X (n, 3) or (3,)) -> (n, 2)`` in pixels (metres in).
    """
    pts = np.atleast_2d(np.asarray(X, dtype=float))
    h = (np.asarray(P, dtype=float) @ np.c_[pts, np.ones(len(pts))].T).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return h[:, :2] / h[:, 2:3]


def camera_centre(P: np.ndarray) -> np.ndarray:
    """Optical centre ``C = -M^-1 p4`` (metres, pitch frame) of the 3x4 matrix ``P``.

    Signature: ``camera_centre(P (3, 4)) -> (3,)``. Raises ``ValueError`` when ``M = P[:, :3]``
    is singular (a planar DLT3D has no centre).
    """
    P = np.asarray(P, dtype=float)
    M = P[:, :3]
    s = np.linalg.svd(M, compute_uv=False)
    if s[0] == 0.0 or s[-1] / s[0] < 1e-9:
        raise ValueError("singular projection matrix: no camera centre")
    return -np.linalg.solve(M, P[:, 3])


def ray_from_pixel(P: np.ndarray, uv) -> tuple[np.ndarray, np.ndarray]:
    """Ray of pixel ``uv`` for the matrix ``P``: ``(centre (3,), unit direction (3,))``.

    The direction points **in front of** the camera whatever the sign convention of ``P``.
    Signature: ``ray_from_pixel(P (3, 4), uv (2,)) -> (C, d)``; ``C`` in metres.
    """
    P = np.asarray(P, dtype=float)
    M = P[:, :3]
    C = camera_centre(P)
    d = np.linalg.solve(M, np.array([float(uv[0]), float(uv[1]), 1.0]))
    d = d * (1.0 if np.linalg.det(M) > 0 else -1.0)  # w = t * sign(det M): t > 0 is in front
    return C, d / np.linalg.norm(d)


def ground_point(P: np.ndarray, u: float, v: float, z: float = 0.0) -> np.ndarray:
    """Point where the ray of pixel ``(u, v)`` meets the horizontal plane at height ``z``.

    Signature: ``ground_point(P (3, 4), u, v, z=0.0) -> (3,)`` = ``(x, y, z)`` in metres of the
    pitch frame. ``(nan, nan, z)`` when the ray never reaches the plane in front of the camera
    (above the horizon). For a frame of a track use :meth:`CameraTrack.ground_point`.
    """
    try:
        C, d = ray_from_pixel(P, (u, v))
    except ValueError:
        return np.array([math.nan, math.nan, float(z)])
    if abs(d[2]) < 1e-12:
        return np.array([math.nan, math.nan, float(z)])
    t = (float(z) - C[2]) / d[2]
    if t <= 0.0:
        return np.array([math.nan, math.nan, float(z)])
    X = C + t * d
    return np.array([X[0], X[1], float(z)])


def _ray_vertical_plane(
    P: np.ndarray, uv, point: np.ndarray, normal: np.ndarray
) -> np.ndarray | None:
    """Intersection of the ray of ``uv`` with the plane ``normal . (X - point) = 0`` or None."""
    C, d = ray_from_pixel(P, uv)
    den = float(np.dot(normal, d))
    if abs(den) < 1e-12:
        return None
    t = float(np.dot(normal, point - C)) / den
    if t <= 0.0:
        return None
    return C + t * d


def height_base_top(P: np.ndarray, base_uv, top_uv, base_z: float = 0.0) -> float:
    """Height (m) of ``top_uv`` above ``base_uv``, by the vertical plane through the base.

    The base pixel is put on the horizontal plane ``z = base_z`` (the grass by default); the
    plane is vertical, contains that point and is perpendicular to the **horizontal direction
    of the base ray**, so a vertical pole stands in it. The ray of the top pixel is cut by it.

    Signature: ``height_base_top(P (3, 4), base_uv (2,), top_uv (2,), base_z=0.0) -> float``,
    metres, ``top z - base_z``; ``nan`` when a ray misses. The error grows with the lateral
    error of the base on the ground (a base 1 m off sideways barely moves a pole height).
    """
    B = ground_point(P, base_uv[0], base_uv[1], base_z)
    if not np.all(np.isfinite(B)):
        return math.nan
    C = camera_centre(P)
    h = B[:2] - C[:2]
    norm = float(np.hypot(h[0], h[1]))
    if norm < 1e-9:
        return math.nan
    normal = np.array([h[0] / norm, h[1] / norm, 0.0])
    X = _ray_vertical_plane(P, top_uv, B, normal)
    return math.nan if X is None else float(X[2] - base_z)


# ----------------------------------------------------------------------------------------------
# Camera track
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class FrameCamera:
    """The camera of one frame.

    ``matrix`` 3x4 (pixel ~ P @ [X Y Z 1]); ``source`` one of ``"dlt3d"`` (FreeKiki DLT3D, kept
    as is), ``"solved"`` (fixed centre, rotation + zoom fitted to the raw keypoints),
    ``"approximate"`` (copy of the nearest good frame, ``copied_from``) or ``"given"``;
    ``rms_px`` / ``n_points`` the fit of a solved frame (nan / 0 otherwise); ``focal_px`` the zoom;
    ``rvec`` the world-to-camera rotation vector (rad) when known.
    """

    frame: int
    matrix: np.ndarray
    source: str
    rms_px: float = math.nan
    n_points: int = 0
    focal_px: float = math.nan
    rvec: tuple[float, float, float] | None = None
    copied_from: int | None = None

    @property
    def approximate(self) -> bool:
        return self.source == SOURCE_APPROX

    @property
    def from_frame(self) -> int | None:
        """The frame an approximate / tracked camera starts from (``copied_from``)."""
        return self.copied_from


def _build_matrix(rvec, focal: float, centre, pp) -> np.ndarray:
    R = Rotation.from_rotvec(np.asarray(rvec, dtype=float)).as_matrix()
    K = np.array([[focal, 0.0, pp[0]], [0.0, focal, pp[1]], [0.0, 0.0, 1.0]])
    return K @ np.c_[R, -R @ np.asarray(centre, dtype=float)]


def _rq3(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition ``M = K @ R`` with ``K`` upper triangular, positive diagonal."""
    flip = np.flipud(np.eye(3))
    q, r = np.linalg.qr((flip @ M).T)
    upper = flip @ r.T @ flip
    ortho = flip @ q.T
    signs = np.diag(np.sign(np.diag(upper)))
    return upper @ signs, signs @ ortho


def _decompose(P: np.ndarray) -> tuple[float, np.ndarray]:
    """``(focal_px, R)`` of a 3x4 matrix: R is a proper rotation world -> camera."""
    K, R = _rq3(np.asarray(P, dtype=float)[:, :3])
    if np.linalg.det(R) < 0:
        R = -R
    return float(K[0, 0] / K[2, 2]), R


def _seed_from_matrix(P: np.ndarray, centre: np.ndarray, pp, grid: np.ndarray) -> np.ndarray | None:
    """``[rvec(3), focal]`` of the fixed-centre camera that best reproduces ``P`` on ``grid``."""
    uv = project_points(P, grid)
    sign = 1.0 if np.linalg.det(P[:, :3]) > 0 else -1.0
    w = sign * (np.asarray(P) @ np.c_[grid, np.ones(len(grid))].T)[2]
    ok = (
        np.isfinite(uv).all(axis=1)
        & (w > 0)
        & (np.abs(uv[:, 0] - pp[0]) < 4000)
        & (np.abs(uv[:, 1] - pp[1]) < 4000)
    )
    if ok.sum() < 6:
        return None
    focal, R = _decompose(P)
    x0 = np.r_[Rotation.from_matrix(R).as_rotvec(), max(focal, 1.0)]
    res = least_squares(
        lambda p: (
            project_points(_build_matrix(p[:3], p[3], centre, pp), grid[ok]) - uv[ok]
        ).ravel(),
        x0,
    )
    return res.x


def _fit_frame(X, uv, C, pp, x0, sigma_px: float):
    """Robust fit of ``[rvec, focal]`` (fixed centre ``C``) to the keypoints ``uv`` of ``X``."""

    def residual(p):
        return ((project_points(_build_matrix(p[:3], p[3], C, pp), X) - uv) / sigma_px).ravel()

    return least_squares(residual, x0, loss="soft_l1", f_scale=1.0)


def _template_grid(template: np.ndarray) -> np.ndarray:
    lo = template[:, :2].min(axis=0)
    hi = template[:, :2].max(axis=0)
    xs = np.linspace(lo[0], hi[0], 11)
    ys = np.linspace(lo[1], hi[1], 7)
    return np.array([[x, y, 0.0] for x in xs for y in ys])


class CameraTrack:
    """A 3x4 projection matrix for each frame of the video.

    Build it with :meth:`from_calibration` (FreeKiki DLT3D + raw keypoints) or
    :meth:`from_matrices`. Frames outside ``[0, n_frames)`` use the nearest frame.
    """

    def __init__(
        self,
        frames: dict[int, FrameCamera],
        n_frames: int,
        centre: np.ndarray | None = None,
        principal_point: tuple[float, float] = (960.0, 540.0),
    ) -> None:
        if not frames:
            raise ValueError("a CameraTrack needs at least one frame")
        self.frames = dict(frames)
        self.n_frames = int(n_frames)
        self.centre = None if centre is None else np.asarray(centre, dtype=float)
        self.principal_point = (float(principal_point[0]), float(principal_point[1]))
        self._keys = sorted(self.frames)

    # -- construction ---------------------------------------------------------------------
    @classmethod
    def from_matrices(
        cls, matrices: dict[int, np.ndarray], n_frames: int | None = None
    ) -> CameraTrack:
        """Track from given matrices (``source == "given"``); missing frames copy the nearest.

        Signature: ``from_matrices({frame: P (3, 4)}, n_frames=None) -> CameraTrack``.
        """
        frames = {
            int(f): FrameCamera(int(f), np.asarray(P, dtype=float), SOURCE_GIVEN)
            for f, P in matrices.items()
        }
        n = int(n_frames) if n_frames is not None else max(frames) + 1
        return cls(frames, n)

    @classmethod
    def from_calibration(
        cls,
        dlt: dict[int, np.ndarray],
        kps: dict[int, dict[int, tuple[float, float, float]]],
        template: np.ndarray,
        n_frames: int,
        principal_point: tuple[float, float] = (960.0, 540.0),
        centre: np.ndarray | None = None,
        min_conf: float = 0.5,
        min_points: int = 3,
        max_rms_px: float = 6.0,
        sigma_px: float = 3.0,
        centre_window: int = 10,
    ) -> CameraTrack:
        """Per-frame cameras: the DLT3D where there is one, a fixed-centre solve elsewhere.

        Signature: ``from_calibration(dlt, kps, template, n_frames, principal_point=(960, 540),
        centre=None, min_conf=0.5, min_points=3, max_rms_px=6.0, sigma_px=3.0,
        centre_window=10)``.

        ``dlt`` ``{frame: P (3, 4)}`` (frames of the FreeKiki ``.dlt3d``); ``kps`` ``{frame:
        {point index: (u, v, confidence)}}`` (raw network keypoints); ``template`` ``(n, 3)``
        pitch coordinates (m) of the keypoints, indexed by point index.

        The optical centre of a frame to solve is the median centre of the ``centre_window``
        calibrated frames nearest in time (``0`` = median of all of them; ``centre`` forces one
        centre for every frame). The windowed default is needed on a long lens: the centre of a
        DLT3D is poorly fixed along the viewing axis (a few metres of drift over a clip at
        ~7000 px of focal length), so a single global median made the fit fail on real data.
        Unsolved frames copy the nearest calibrated/solved frame, flagged ``approximate``.
        """
        template = np.asarray(template, dtype=float)
        pp = (float(principal_point[0]), float(principal_point[1]))
        frames: dict[int, FrameCamera] = {}
        centres = []
        for f, P in sorted(dlt.items()):
            P = np.asarray(P, dtype=float)
            try:
                centres.append(camera_centre(P))
            except ValueError:
                continue  # planar DLT3D: no centre; the frame still keeps its matrix
            focal, R = _decompose(P)
            frames[int(f)] = FrameCamera(
                int(f),
                P,
                SOURCE_DLT3D,
                focal_px=focal,
                rvec=tuple(Rotation.from_matrix(R).as_rotvec().tolist()),
            )
        for f, P in dlt.items():
            if int(f) not in frames:
                frames[int(f)] = FrameCamera(int(f), np.asarray(P, dtype=float), SOURCE_DLT3D)
        if centre is None and not centres:
            raise ValueError("no calibrated frame with a usable camera centre")
        cal_frames = np.array([f for f, fc in sorted(frames.items()) if fc.rvec is not None])
        cal_centres = np.array(centres)
        if centre is not None:
            C_global = np.asarray(centre, dtype=float)
        else:
            C_global = np.median(cal_centres, axis=0)

        def centre_at(f: int) -> np.ndarray:
            if centre is not None or centre_window <= 0 or len(cal_frames) <= centre_window:
                return C_global
            near = np.argsort(np.abs(cal_frames - f), kind="stable")[:centre_window]
            return np.median(cal_centres[near], axis=0)

        grid = _template_grid(template)
        resolved_keys = sorted(f for f, fc in frames.items() if fc.rvec is not None)
        if not resolved_keys:
            raise ValueError("no calibrated frame can seed the fixed-centre solver")

        def usable(f: int) -> tuple[np.ndarray, np.ndarray, list[int]] | None:
            pts = kps.get(f, {})
            idx = sorted(
                i
                for i, (u, v, c) in pts.items()
                if 0 <= i < len(template) and c >= min_conf and np.isfinite(u) and np.isfinite(v)
            )
            if len(idx) < min_points:
                return None
            X = template[idx]
            if np.linalg.svd(X - X.mean(axis=0), compute_uv=False)[1] < 1.0:
                return None  # collinear points do not fix rotation + zoom
            uv = np.array([[pts[i][0], pts[i][1]] for i in idx])
            return X, uv, idx

        calibrated = np.array(sorted(frames))
        targets = [f for f in range(int(n_frames)) if f not in frames and usable(f) is not None]
        targets.sort(key=lambda f: (int(np.abs(calibrated - f).min()), f))

        def nearest(f: int, exclude_side: int | None = None) -> int | None:
            k = bisect.bisect_left(resolved_keys, f)
            cands = []
            if k > 0:
                cands.append(resolved_keys[k - 1])
            if k < len(resolved_keys):
                cands.append(resolved_keys[k])
            cands = [c for c in cands if c != exclude_side]
            return min(cands, key=lambda c: (abs(c - f), c)) if cands else None

        for f in targets:
            data = usable(f)
            if data is None:
                continue
            X, uv, idx = data
            C = centre_at(f)
            seeds = []
            first = nearest(f)
            if first is not None:
                seeds.append(first)
                other = nearest(f, exclude_side=first)
                if other is not None and (other < f) != (first < f):
                    seeds.append(other)
            for s in seeds:
                x0 = _seed_from_matrix(frames[s].matrix, C, pp, grid)  # neighbour, this centre
                if x0 is None:
                    continue
                res = _fit_frame(X, uv, C, pp, x0, sigma_px)
                err = np.hypot(
                    *(project_points(_build_matrix(res.x[:3], res.x[3], C, pp), X) - uv).T
                )
                rms = float(np.sqrt(np.mean(err**2)))
                if rms <= max_rms_px and res.x[3] > 0:
                    bisect.insort(resolved_keys, f)
                    frames[f] = FrameCamera(
                        f,
                        _build_matrix(res.x[:3], res.x[3], C, pp),
                        SOURCE_SOLVED,
                        rms_px=rms,
                        n_points=len(idx),
                        focal_px=float(res.x[3]),
                        rvec=tuple(res.x[:3].tolist()),
                    )
                    break

        good = sorted(frames)
        for f in range(int(n_frames)):
            if f in frames:
                continue
            k = bisect.bisect_left(good, f)
            cands = [
                g
                for g in (good[k - 1] if k > 0 else None, good[k] if k < len(good) else None)
                if g is not None
            ]
            src = min(cands, key=lambda g: (abs(g - f), g))
            base = frames[src]
            frames[f] = FrameCamera(
                f,
                base.matrix,
                SOURCE_APPROX,
                focal_px=base.focal_px,
                rvec=base.rvec,
                copied_from=src,
            )
        return cls(frames, n_frames, centre=C_global, principal_point=pp)

    # -- access ---------------------------------------------------------------------------
    def info(self, frame: int) -> FrameCamera:
        """:class:`FrameCamera` of ``frame`` (nearest known frame outside the range)."""
        f = int(frame)
        if f in self.frames:
            return self.frames[f]
        k = bisect.bisect_left(self._keys, f)
        cands = [self._keys[i] for i in (k - 1, k) if 0 <= i < len(self._keys)]
        src = min(cands, key=lambda g: (abs(g - f), g))
        base = self.frames[src]
        return FrameCamera(f, base.matrix, SOURCE_APPROX, copied_from=src,
                           focal_px=base.focal_px, rvec=base.rvec)  # fmt: skip

    def matrix(self, frame: int) -> np.ndarray:
        """3x4 projection matrix of ``frame``. Signature: ``matrix(frame) -> (3, 4)``."""
        return self.info(frame).matrix

    def source(self, frame: int) -> str:
        """``"dlt3d"``, ``"solved"``, ``"approximate"`` or ``"given"`` for ``frame``."""
        return self.info(frame).source

    def is_approximate(self, frame: int) -> bool:
        """True when the camera of ``frame`` is a copy of a neighbour (the HUD must warn)."""
        return self.info(frame).approximate

    def project(self, frame: int, X) -> np.ndarray:
        """Pixels of pitch points. Signature: ``project(frame, X (n, 3) m) -> (n, 2)`` px."""
        return project_points(self.matrix(frame), X)

    def ground_point(self, frame: int, u: float, v: float, z: float = 0.0) -> np.ndarray:
        """Pitch point under pixel ``(u, v)`` of ``frame`` on the plane height ``z``.

        Signature: ``ground_point(frame, u, v, z=0.0) -> (3,)`` = ``(x, y, z)`` metres (nan when
        the ray misses the plane). See :func:`ground_point`.
        """
        return ground_point(self.matrix(frame), u, v, z)

    def homography(self, frame: int) -> np.ndarray:
        """Ground-plane homography of ``frame``: pixel ~ ``H @ [x, y, 1]``, ``H[2, 2] = 1``.

        Signature: ``homography(frame) -> (3, 3)``. See :func:`ground_homography`.
        """
        return ground_homography(self.matrix(frame))

    def height_base_top(self, frame: int, base_uv, top_uv, base_z: float = 0.0) -> float:
        """Height (m) of ``top_uv`` over ``base_uv`` in ``frame``. See :func:`height_base_top`.

        Signature: ``height_base_top(frame, base_uv, top_uv, base_z=0.0) -> float``.
        """
        return height_base_top(self.matrix(frame), base_uv, top_uv, base_z)

    def counts(self) -> dict[str, int]:
        """Frames per source: ``{"dlt3d": n, "solved": n, "approximate": n, ...}``."""
        out: dict[str, int] = {}
        for f in range(self.n_frames):
            s = self.info(f).source
            out[s] = out.get(s, 0) + 1
        return out

    def approximate_frames(self) -> list[int]:
        """Sorted frames whose camera is an approximation."""
        return [f for f in range(self.n_frames) if self.info(f).approximate]

    def to_dict(self) -> dict:
        """JSON-ready description: centre, principal point and one entry per frame."""
        return {
            "n_frames": self.n_frames,
            "centre_m": None if self.centre is None else self.centre.tolist(),
            "principal_point_px": list(self.principal_point),
            "counts": self.counts(),
            "frames": [
                {
                    "frame": f,
                    "source": (fc := self.info(f)).source,
                    "approximate": fc.approximate,
                    "rms_px": None if math.isnan(fc.rms_px) else fc.rms_px,
                    "n_points": fc.n_points,
                    "focal_px": None if math.isnan(fc.focal_px) else fc.focal_px,
                    "copied_from": fc.copied_from,
                    "matrix": fc.matrix.ravel().tolist(),
                }
                for f in range(self.n_frames)
            ],
        }


# ----------------------------------------------------------------------------------------------
# Frame-to-frame tracking of the frames the fixed-centre solver could not fit
# ----------------------------------------------------------------------------------------------

SOURCE_TRACKED = "tracked"
#: Version of the tracking method; the pipeline stage that writes the per-frame cameras folds it
#: into its inputs so that a change of method redoes the stage.
TRACKING_MODEL_VERSION = "crowd-lk-idw-1"

#: Broadcast overlays that move with the screen, not with the pitch, as fractions of the frame
#: ``(x0, y0, x1, y1)``: the scoreboard with its banner (top-left), the channel logo (top-right)
#: and the "ge" bug (bottom-left) of the Fla x CAP clip (measured on 1920x1080:
#: 0-260 x 0-190, 1620-1830 x 85-155 and 110-250 x 900-1000 px, plus a safety margin). A
#: static overlay would otherwise vote for "the camera did not move". Features inside these
#: boxes are never tracked. Any other fixed overlay that survives is thrown away by the
#: robust fit as soon as the camera moves more than a few pixels.
OVERLAY_MARGINS: tuple[tuple[float, float, float, float], ...] = (
    (0.0, 0.0, 0.16, 0.21),
    (0.82, 0.05, 0.98, 0.17),
    (0.04, 0.80, 0.15, 0.95),
)
BOX_PAD = 0.10  # SAM person boxes are grown by this fraction of their size on every side
BALL_MASK_RADIUS_PX = 30
TRACK_STRIDES = (1, 2, 4, 8, 16)  # frame distances whose matches enter the joint fit
TRACK_MIN_INLIERS = 60  # inlier feature pairs touching a frame for it to count as tracked
TRACK_MAX_CROWD_RMS_PX = (
    4.0  # inlier feature transfer error (px) above which a frame stays approximate
)
_LK_WIN = (31, 31)
_LK_LEVELS = 5
_MAX_TRACKED_POINTS = 2500


def _single_thread():
    """Context manager: one BLAS thread. The fits are many tiny matrices; a threaded BLAS on a
    busy many-core machine made them 10 times slower (user time 8 min for 27 s)."""
    try:
        from threadpoolctl import threadpool_limits

        return threadpool_limits(limits=1)
    except ImportError:
        import contextlib

        return contextlib.nullcontext()


def _limited(fn):
    """Decorator: run ``fn`` with one BLAS thread (see :func:`_single_thread`)."""
    import functools

    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        with _single_thread():
            return fn(*args, **kwargs)

    return wrapper


class FeatureTracks:
    """Static-scene feature tracks of a video: ``obs[frame] = (ids (n,), uv (n, 2) px)``.

    Built by :func:`track_features`. The same feature id in two frames is the same physical point
    of the scene, so the pair ``(uv_i, uv_j)`` is a match between frames ``i`` and ``j``.
    """

    def __init__(self, width: int, height: int, obs: dict[int, tuple[np.ndarray, np.ndarray]]):
        self.width = int(width)
        self.height = int(height)
        self.obs = obs

    def frames(self) -> list[int]:
        return sorted(self.obs)

    def matches(self, i: int, j: int, limit: int = 0) -> tuple[np.ndarray, np.ndarray]:
        """Matched pixels ``(uv_i (m, 2), uv_j (m, 2))`` of frames ``i`` and ``j`` (at most
        ``limit`` of them when ``limit > 0``, an even deterministic subsample)."""
        a, b = self.obs.get(int(i)), self.obs.get(int(j))
        if a is None or b is None:
            return np.zeros((0, 2)), np.zeros((0, 2))
        _, ia, ib = np.intersect1d(a[0], b[0], assume_unique=True, return_indices=True)
        if limit > 0 and len(ia) > limit:
            sel = np.linspace(0, len(ia) - 1, limit).round().astype(int)
            ia, ib = ia[sel], ib[sel]
        return a[1][ia].astype(float), b[1][ib].astype(float)


def _static_mask(
    height: int, width: int, boxes, ball, overlay=OVERLAY_MARGINS, pad: float = BOX_PAD
) -> np.ndarray:
    """uint8 mask (255 = usable pixel): no overlay, no SAM person box (grown by ``pad``), no ball."""
    import cv2

    mask = np.full((height, width), 255, dtype=np.uint8)
    for x0, y0, x1, y1 in overlay:
        mask[
            int(y0 * height) : int(math.ceil(y1 * height)),
            int(x0 * width) : int(math.ceil(x1 * width)),
        ] = 0
    for x, y, w, h in boxes or ():
        px, py = pad * w, pad * h
        xa, ya = int(max(0, math.floor(x - px))), int(max(0, math.floor(y - py)))
        xb, yb = int(min(width, math.ceil(x + w + px))), int(min(height, math.ceil(y + h + py)))
        if xb > xa and yb > ya:
            mask[ya:yb, xa:xb] = 0
    if ball is not None and np.all(np.isfinite(ball)):
        cv2.circle(mask, (int(round(ball[0])), int(round(ball[1]))), BALL_MASK_RADIUS_PX, 0, -1)
    return mask


def track_features(
    video,
    segments,
    *,
    boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    ball_uv: dict[int, tuple[float, float]] | None = None,
    max_points: int = _MAX_TRACKED_POINTS,
    min_distance: int = 12,
    fb_tol_px: float = 0.7,
) -> FeatureTracks:
    """Follow static features (Shi-Tomasi corners + pyramidal Lucas-Kanade) through the frames.

    Signature: ``track_features(video, segments, boxes=None, ball_uv=None, max_points=2500,
    min_distance=12, fb_tol_px=0.7) -> FeatureTracks``. ``segments`` is a list of ``(first, last)``
    frame ranges (inclusive); tracks never cross from one segment into the next. ``boxes``
    ``{frame: [(x, y, w, h), ...]}`` (SAM boxes, grown by :data:`BOX_PAD`) and ``ball_uv``
    exclude moving things; :data:`OVERLAY_MARGINS` excludes the broadcast overlays. A point is kept
    only when tracking it forward and back lands within ``fb_tol_px`` of where it started.
    Deterministic: no random numbers.
    """
    import cv2

    cv2.setNumThreads(1)
    wanted: set[int] = set()
    for lo, hi in segments:
        wanted.update(range(int(lo), int(hi) + 1))
    if not wanted:
        return FeatureTracks(0, 0, {})
    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open video {video}")
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
    obs: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    prev_gray = None
    prev_frame = -10
    ids = np.zeros(0, dtype=np.int64)
    pts = np.zeros((0, 2), dtype=np.float32)
    next_id = 0
    try:
        for f in range(max(wanted) + 1):
            if f not in wanted:
                cap.grab()
                continue
            ok, bgr = cap.read()
            if not ok:
                break
            gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
            mask = _static_mask(height, width, (boxes or {}).get(f), (ball_uv or {}).get(f))
            if prev_gray is not None and f == prev_frame + 1 and len(pts):
                p0 = pts.reshape(-1, 1, 2)
                nxt, st, _ = cv2.calcOpticalFlowPyrLK(
                    prev_gray, gray, p0, np.zeros_like(p0),
                    winSize=_LK_WIN, maxLevel=_LK_LEVELS, criteria=criteria,
                )  # fmt: skip
                back, st2, _ = cv2.calcOpticalFlowPyrLK(
                    gray, prev_gray, nxt, np.zeros_like(nxt),
                    winSize=_LK_WIN, maxLevel=_LK_LEVELS, criteria=criteria,
                )  # fmt: skip
                nxt = nxt.reshape(-1, 2)
                fb = np.hypot(*(back.reshape(-1, 2) - pts).T)
                xi = np.clip(np.round(nxt[:, 0]).astype(int), 0, width - 1)
                yi = np.clip(np.round(nxt[:, 1]).astype(int), 0, height - 1)
                keep = (
                    (st.ravel() == 1)
                    & (st2.ravel() == 1)
                    & (fb < fb_tol_px)
                    & (nxt[:, 0] >= 0)
                    & (nxt[:, 0] < width)
                    & (nxt[:, 1] >= 0)
                    & (nxt[:, 1] < height)
                    & (mask[yi, xi] > 0)
                )
                ids, pts = ids[keep], nxt[keep].astype(np.float32)
            else:
                ids = np.zeros(0, dtype=np.int64)
                pts = np.zeros((0, 2), dtype=np.float32)
            need = max_points - len(pts)
            if need > 0:
                free = mask.copy()
                for x, y in pts:
                    cv2.circle(free, (int(x), int(y)), min_distance, 0, -1)
                new = cv2.goodFeaturesToTrack(
                    gray, maxCorners=need, qualityLevel=0.004, minDistance=min_distance,
                    mask=free, blockSize=5,
                )  # fmt: skip
                if new is not None and len(new):
                    new = new.reshape(-1, 2).astype(np.float32)
                    ids = np.r_[ids, np.arange(next_id, next_id + len(new), dtype=np.int64)]
                    pts = np.r_[pts, new]
                    next_id += len(new)
            obs[f] = (ids.copy(), pts.copy())
            prev_gray, prev_frame = gray, f
    finally:
        cap.release()
    return FeatureTracks(width, height, obs)


def _frame_rotation_focal(
    P: np.ndarray, centre: np.ndarray, pp, grid: np.ndarray
) -> tuple[np.ndarray, float] | None:
    """``(rvec, focal)`` of the fixed-centre camera nearest to ``P`` (None when it cannot be)."""
    x0 = _seed_from_matrix(P, centre, pp, grid)
    return None if x0 is None else (x0[:3].copy(), float(x0[3]))


def _rotation_mats(rvec: np.ndarray, focal: np.ndarray, pp) -> tuple[np.ndarray, np.ndarray]:
    """Stacks ``A = K R`` and ``A^-1 = R^T K^-1`` for rotation vectors ``(n, 3)`` and focals ``(n,)``."""
    R = Rotation.from_rotvec(np.atleast_2d(rvec)).as_matrix()
    f = np.atleast_1d(focal)
    K = np.zeros((len(f), 3, 3))
    K[:, 0, 0] = K[:, 1, 1] = f
    K[:, 0, 2], K[:, 1, 2], K[:, 2, 2] = pp[0], pp[1], 1.0
    Kinv = np.zeros_like(K)
    Kinv[:, 0, 0] = Kinv[:, 1, 1] = 1.0 / f
    Kinv[:, 0, 2], Kinv[:, 1, 2], Kinv[:, 2, 2] = -pp[0] / f, -pp[1] / f, 1.0
    return K @ R, np.transpose(R, (0, 2, 1)) @ Kinv


def _transfer(A_j: np.ndarray, Ainv_i: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Pixels of ``uv`` (n, 2) of frame i seen in frame j: ``K_j R_j R_i^T K_i^-1`` per row."""
    h = np.einsum("nab,nb->na", A_j @ Ainv_i, np.c_[uv, np.ones(len(uv))])
    return h[:, :2] / h[:, 2:3]


def _pair_init(uv_a, uv_b, rvec_a, f_a, pp):
    """``(rvec_b, f_b)`` from the matches of two consecutive frames given ``(rvec_a, f_a)``."""
    import cv2

    if len(uv_a) < 12:
        return None
    H, inl = cv2.findHomography(uv_a, uv_b, cv2.RANSAC, 3.0)
    if H is None or inl is None or int(inl.sum()) < 12:
        return None
    ua, ub = uv_a[inl.ravel() > 0], uv_b[inl.ravel() > 0]
    Ra = Rotation.from_rotvec(rvec_a)
    A_a, Ainv_a = _rotation_mats(rvec_a, np.array([f_a]), pp)

    def residual(p):
        Rb = Rotation.from_rotvec(p[:3]) * Ra
        A_b, Ainv_b = _rotation_mats(Rb.as_rotvec(), np.array([math.exp(p[3])]), pp)
        n = len(ua)
        fwd = _transfer(np.repeat(A_b, n, 0), np.repeat(Ainv_a, n, 0), ua) - ub
        bwd = _transfer(np.repeat(A_a, n, 0), np.repeat(Ainv_b, n, 0), ub) - ua
        return np.r_[fwd.ravel(), bwd.ravel()]

    res = least_squares(residual, np.r_[np.zeros(3), math.log(f_a)], loss="soft_l1", f_scale=1.0)
    rb = (Rotation.from_rotvec(res.x[:3]) * Ra).as_rotvec()
    return rb, float(math.exp(res.x[3]))


def _crowd_bundle(
    feat: FeatureTracks,
    frames: list[int],
    f_init: float,
    pp,
    *,
    strides=TRACK_STRIDES,
    per_pair: int = 60,
    zoom_sigma: float = 0.002,
    max_nfev: int = 50,
) -> dict[int, dict]:
    """Rotation and zoom of every frame of ``frames`` from the feature matches alone.

    The scene features (crowd, boards) obey ``x_j ~ K_j R_j R_i^T K_i^-1 x_i`` whatever their
    depth, with the lens of the broadcast camera. That is not the same ``(R, f)`` as the
    fixed-centre camera the pitch fits give (the optical centre they assume is only constrained to
    some metres along the viewing axis and the focal length absorbs the difference; on the Fla x
    CAP clip the features want ``f`` near 3000 px where the pitch fits give 6000-7000), so the two
    are never mixed here: this function knows nothing about the pitch. The gauge is fixed by
    ``R = I`` on the first frame; ``zoom_sigma`` is the smoothness weight on the second difference
    of ``log f``. The cost per match is symmetric (forward and backward transfer), soft-L1, then a
    second pass without the matches above ``max(2 px, 4 sigma)``; the Jacobian is built by finite
    differences frame by frame (a frame only enters the matches that touch it).
    Returns ``{frame: {"rvec", "focal", "rms_px", "n_inliers", "n_adjacent"}}``.
    """
    frames = sorted(frames)
    fidx = {f: k for k, f in enumerate(frames)}
    n_f = len(frames)
    known: dict[int, tuple[np.ndarray, float]] = {frames[0]: (np.zeros(3), float(f_init))}
    for k, f in enumerate(frames[1:], 1):
        ua, ub = feat.matches(f - 1, f)
        init = _pair_init(ua, ub, known[f - 1][0], known[f - 1][1], pp) if f - 1 in known else None
        known[f] = init if init is not None else known[frames[k - 1]]
    x0 = np.array([[*known[f][0], math.log(known[f][1])] for f in frames]).ravel()
    oi, oj, ui, uj, ostr = [], [], [], [], []
    for i in frames:
        for s in strides:
            j = i + s
            if j not in fidx:
                continue
            ua, ub = feat.matches(i, j, per_pair)
            if len(ua) < 8:
                continue
            oi.append(np.full(len(ua), fidx[i]))
            oj.append(np.full(len(ua), fidx[j]))
            ui.append(ua)
            uj.append(ub)
            ostr.append(np.full(len(ua), s))
    if not oi:
        return {}
    gi, gj, ui, uj, ostr = (np.concatenate(a) for a in (oi, oj, ui, uj, ostr))
    tri = np.array([(a, a + 1, a + 2) for a in range(n_f - 2) if frames[a + 2] - frames[a] == 2])

    def cameras(x):
        X = x.reshape(n_f, 4)
        A, Ainv = _rotation_mats(X[:, :3], np.exp(X[:, 3]), pp)
        return X, A, Ainv

    def solve(sel, x_start):
        oi_, oj_ = gi[sel], gj[sel]
        u_i, u_j = ui[sel], uj[sel]
        m = len(oi_)
        touch = [np.flatnonzero((oi_ == k) | (oj_ == k)) for k in range(n_f)]
        n_rows = 4 * m + 3 + len(tri)

        def obs_residual(x, idx):
            _, A, Ainv = cameras(x)
            fwd = _transfer(A[oj_[idx]], Ainv[oi_[idx]], u_i[idx]) - u_j[idx]
            bwd = _transfer(A[oi_[idx]], Ainv[oj_[idx]], u_j[idx]) - u_i[idx]
            return np.concatenate([fwd.ravel(), bwd.ravel()])

        def fun(x):
            X = x.reshape(n_f, 4)
            parts = [obs_residual(x, np.arange(m)), X[0, :3] / 1e-4]
            if len(tri):
                lf = X[:, 3]
                parts.append((lf[tri[:, 0]] - 2 * lf[tri[:, 1]] + lf[tri[:, 2]]) / zoom_sigma)
            return np.concatenate(parts)

        def jac(x, h=1e-6):
            rows, cols, vals = [], [], []
            for k in range(n_f):
                idx = touch[k]
                if not len(idx):
                    continue
                base = obs_residual(x, idx)
                r_all = np.concatenate(
                    [
                        (2 * idx[:, None] + np.arange(2)).ravel(),
                        (2 * m + 2 * idx[:, None] + np.arange(2)).ravel(),
                    ]
                )
                for q in range(4):
                    xp = x.copy()
                    xp[4 * k + q] += h
                    rows.append(r_all)
                    cols.append(np.full(len(r_all), 4 * k + q))
                    vals.append((obs_residual(xp, idx) - base) / h)
            for q in range(3):
                rows.append(np.array([4 * m + q]))
                cols.append(np.array([q]))
                vals.append(np.array([1.0 / 1e-4]))
            for r, t3 in enumerate(tri):
                for g, c in zip(t3, (1.0, -2.0, 1.0), strict=True):
                    rows.append(np.array([4 * m + 3 + r]))
                    cols.append(np.array([4 * g + 3]))
                    vals.append(np.array([c / zoom_sigma]))
            return coo_matrix(
                (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))),
                shape=(n_rows, len(x)),
            ).tocsr()

        return least_squares(
            fun, x_start, jac=jac, loss="soft_l1", f_scale=1.0, x_scale=1e-2, max_nfev=max_nfev,
        )  # fmt: skip

    def transfer_err(x, sel):
        _, A, Ainv = cameras(x)
        fwd = _transfer(A[gj[sel]], Ainv[gi[sel]], ui[sel]) - uj[sel]
        bwd = _transfer(A[gi[sel]], Ainv[gj[sel]], uj[sel]) - ui[sel]
        return 0.5 * (np.hypot(*fwd.T) + np.hypot(*bwd.T))

    every = np.ones(len(gi), dtype=bool)
    res = solve(every, x0)
    err = transfer_err(res.x, every)
    thr = max(2.0, 4.0 * 1.4826 * float(np.median(err)))
    keep = err <= thr
    if keep.sum() >= 20 * n_f:
        res = solve(keep, res.x)
        err = transfer_err(res.x, every)
    inl = err <= thr
    X = res.x.reshape(n_f, 4)
    out: dict[int, dict] = {}
    for f in frames:
        k = fidx[f]
        touch = inl & ((gi == k) | (gj == k))
        e = err[touch]
        out[f] = {
            "rvec": X[k, :3].copy(),
            "focal": float(math.exp(X[k, 3])),
            "rms_px": float(np.sqrt(np.mean(e**2))) if len(e) else math.inf,
            "n_inliers": int(touch.sum()),
            "n_adjacent": int((touch & (ostr == 1)).sum()),
        }
    return out


TRACK_ANCHORS = 4  # good frames per side that anchor a gap
TRACK_MAX_RMS_PX = 60.0  # uncertainty (px, on the pitch: anchor disagreement + fit) -> approximate
_PITCH_GRID = np.array(
    [[x, y, 0.0] for x in np.linspace(-50.0, 50.0, 21) for y in np.linspace(-32.0, 32.0, 13)]
)


def _centre_near(track: CameraTrack, frame: int, window: int = 10) -> np.ndarray | None:
    """Median optical centre of the ``window`` DLT3D frames nearest to ``frame`` (as the solver)."""
    cands = []
    for f, fc in track.frames.items():
        if fc.source == SOURCE_DLT3D:
            try:
                cands.append((abs(f - frame), f, camera_centre(fc.matrix)))
            except ValueError:
                continue
    if not cands:
        return track.centre
    cands.sort(key=lambda c: (c[0], c[1]))
    return np.median([c[2] for c in cands[:window]], axis=0)


def _estimate_frames(
    feat: FeatureTracks,
    track: CameraTrack,
    anchors: list[int],
    targets: list[int],
    window: tuple[int, int],
    centre: np.ndarray,
    *,
    per_pair: int = 60,
) -> dict[int, dict]:
    """Cameras of ``targets`` from the known cameras of ``anchors`` and the feature matches.

    One gap, all frames ``window[0]..window[1]`` take part:

    1. :func:`_crowd_bundle` gives, from the static features alone, the rotation and zoom of every
       frame of the window relative to each other (sub-pixel between neighbours).
    2. Each anchor ``g`` carries the **pitch** to frame ``t``: the image map ``K_t R_t R_g^T K_g^-1``
       is applied to the pixels of a grid of pitch points as the anchor's own camera sees them.
       Each anchor gives one prediction of where the pitch is in ``t``; their disagreement (px) is
       the honest uncertainty, because an anchor carries a few pixels of jitter of its own.
       The scene features and the pitch are *not* forced into one model: a joint fit of both (with
       a free optical centre, with the zoom tied to the anchors') was tried and fits worse, the
       crowd and the pitch only agree to a few pixels.
    3. The predictions of one side are averaged (nearer anchors count more) and the two sides
       blended with inverse-distance weights (a linear blend between the two anchors, so a drift
       that grows with the distance travelled cancels in the middle of a gap).
    4. A fixed-centre camera (rotation + zoom, as :meth:`CameraTrack.from_calibration`, centre
       ``centre``) is fitted to the blended pitch pixels: that is the camera of ``t``.

    Returns ``{frame: {"matrix", "rvec", "focal", "spread_px" (disagreement of the anchors, px on
    the pitch), "fit_px" (misfit of the fixed-centre family to the blended pixels), "rms_px" (the
    two together: the uncertainty of the frame), "n_inliers" (feature matches touching the
    frame), "n_adjacent", "crowd_rms_px", "from_frame"}}`` for the frames it could estimate.
    """
    pp = track.principal_point
    lo, hi = window
    est = _crowd_bundle(feat, list(range(lo, hi + 1)), 3000.0, pp, per_pair=per_pair)
    if not est:
        return {}
    grid_seed = _template_grid(_PITCH_GRID)
    uv_anchor = {g: project_points(track.matrix(g), _PITCH_GRID) for g in anchors if g in est}
    out: dict[int, dict] = {}
    w_img, h_img = feat.width, feat.height
    for t in targets:
        if t not in est or not uv_anchor:
            continue
        M_t, _ = _rotation_mats(est[t]["rvec"], np.array([est[t]["focal"]]), pp)
        preds, dist = [], []
        for g, uv in uv_anchor.items():
            _, Minv_g = _rotation_mats(est[g]["rvec"], np.array([est[g]["focal"]]), pp)
            h = (M_t[0] @ Minv_g[0] @ np.c_[uv, np.ones(len(uv))].T).T
            preds.append(h[:, :2] / h[:, 2:3])
            dist.append(abs(t - g))
        preds_a, dist_a = np.array(preds), np.array(dist, dtype=float)
        gl = np.array(list(uv_anchor))
        w = np.zeros(len(dist_a))
        for side in (gl < t, gl > t):
            if side.any():
                near = (
                    1.0 / (dist_a[side] + 0.5) ** 2
                )  # within a side the nearer anchor counts more
                w[side] = (1.0 / (float(dist_a[side].min()) + 0.5)) * near / near.sum()
        if not w.sum() > 0:
            continue
        w = w / w.sum()
        uvhat = np.einsum("a,anc->nc", w, preds_a)
        ok = (
            np.isfinite(uvhat).all(axis=1)
            & (uvhat[:, 0] > 0.0)
            & (uvhat[:, 0] < w_img)
            & (uvhat[:, 1] > 0.2 * h_img)
            & (uvhat[:, 1] < h_img)
        )
        if ok.sum() < 8:
            continue
        spread = np.sqrt(
            np.average(np.sum((preds_a[:, ok] - uvhat[ok]) ** 2, axis=2), axis=0, weights=w)
        )
        nearest = int(gl[int(np.argmin(dist_a))])
        seed = _seed_from_matrix(track.matrix(nearest), centre, pp, grid_seed)
        if seed is None:
            continue
        X = _PITCH_GRID[ok]
        target = uvhat[ok]
        fit = least_squares(
            lambda p, X=X, target=target: (
                project_points(_build_matrix(p[:3], p[3], centre, pp), X) - target
            ).ravel(),
            seed, loss="soft_l1", f_scale=2.0, x_scale=[1e-2] * 3 + [100.0],
        )  # fmt: skip
        P_fit = _build_matrix(fit.x[:3], fit.x[3], centre, pp)
        resid = np.hypot(*(project_points(P_fit, X) - uvhat[ok]).T)
        out[t] = {
            "matrix": P_fit,
            "rvec": fit.x[:3].copy(),
            "focal": float(fit.x[3]),
            "spread_px": float(np.sqrt(np.mean(spread**2))),
            "fit_px": float(np.sqrt(np.mean(resid**2))),
            "rms_px": float(np.hypot(np.sqrt(np.mean(spread**2)), np.sqrt(np.mean(resid**2)))),
            "n_inliers": est[t]["n_inliers"],
            "n_adjacent": est[t]["n_adjacent"],
            "crowd_rms_px": est[t]["rms_px"],
            "from_frame": nearest,
        }
    return out


def _sam_boxes(run_dir, stem: str) -> dict[int, list[tuple[float, float, float, float]]]:
    """SAM person boxes ``{frame: [(x, y, w, h), ...]}`` of ``sam_tracks.csv`` (empty if absent)."""
    sam = _find_sam_tracks(Path(run_dir), stem)
    if sam is None:
        return {}
    d = pd.read_csv(sam)
    boxes: dict[int, list[tuple[float, float, float, float]]] = {}
    for r in d.to_dict("records"):
        v = [float(r.get(c, math.nan)) for c in ("x_px", "y_px", "w_px", "h_px")]
        if all(np.isfinite(v)):
            boxes.setdefault(int(r["frame"]), []).append((v[0], v[1], v[2], v[3]))
    return boxes


def _accepted(r: dict, min_inliers: int, max_crowd_rms_px: float, max_rms_px: float) -> bool:
    """The gates that turn an estimate of :func:`_estimate_frames` into a ``tracked`` frame."""
    return bool(
        r["n_adjacent"] >= min_inliers // 2
        and r["n_inliers"] >= min_inliers
        and r["crowd_rms_px"] <= max_crowd_rms_px
        and r["rms_px"] <= max_rms_px
        and np.isfinite(r["matrix"]).all()
    )


@_limited
def track_approximate_frames(
    track: CameraTrack,
    video,
    *,
    boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
    ball_uv: dict[int, tuple[float, float]] | None = None,
    features: FeatureTracks | None = None,
    anchors_per_side: int = TRACK_ANCHORS,
    min_inliers: int = TRACK_MIN_INLIERS,
    max_crowd_rms_px: float = TRACK_MAX_CROWD_RMS_PX,
    max_rms_px: float = TRACK_MAX_RMS_PX,
) -> CameraTrack:
    """Replace the ``approximate`` frames of ``track`` by tracked cameras where possible.

    Signature: ``track_approximate_frames(track, video, *, boxes=None, ball_uv=None,
    features=None, anchors_per_side=4, min_inliers=60, max_crowd_rms_px=4.0,
    max_rms_px=60.0) -> CameraTrack``.

    A frame the fixed-centre solver could not fit (too few pitch keypoints) copied a neighbour.
    The broadcast camera pans and zooms, so the picture moves tens of pixels per frame while that
    copy stands still. Here the static scene (crowd, boards, line markings: not the SAM person
    ``boxes`` grown by 10 %, not the ``ball_uv``, not the broadcast overlays of
    :data:`OVERLAY_MARGINS`) is followed from frame to frame (:func:`track_features`) and the pitch
    is carried from the known cameras on both sides of each gap (:func:`_estimate_frames`).
    A tracked frame has ``source == "tracked"``, ``rms_px`` its uncertainty (px, on the pitch: the
    disagreement of the anchors that carried it and the misfit of the fixed-centre family, added
    in quadrature), ``n_points`` the feature matches that touch it and ``from_frame``
    (``copied_from``) its nearest anchor. A frame stays ``approximate`` when it has fewer than
    ``min_inliers`` matches, a feature-fit error above ``max_crowd_rms_px``, an uncertainty above
    ``max_rms_px``, or no anchor in reach (a failed fit shows as hundreds of px). Nothing is hidden: the
    caller can count both kinds. Anchors are the frames that are not approximate (``dlt3d``,
    ``solved``, ``given``) within ``anchors_per_side`` frames of a gap (the good frames between
    two close gaps included). The optical centre is the solver's (median of the 10 DLT3D frames
    nearest to the gap). ``features`` re-uses tracks already computed for the whole video.
    """
    approx = track.approximate_frames()
    if not approx:
        return track
    good = [f for f in range(track.n_frames) if not track.is_approximate(f)]
    if not good:
        return track
    # one window per cluster of approximate frames closer than the anchors reach
    reach = max(1, anchors_per_side)
    clusters: list[list[int]] = []
    for f in approx:
        if clusters and f - clusters[-1][-1] <= 2 * reach:
            clusters[-1].append(f)
        else:
            clusters.append([f])
    windows = []
    for c in clusters:
        lo = max(0, c[0] - reach)
        hi = min(track.n_frames - 1, c[-1] + reach)
        windows.append((lo, hi, c))
    if features is None:
        features = track_features(
            video, [(lo, hi) for lo, hi, _ in windows], boxes=boxes, ball_uv=ball_uv
        )
    new_frames = dict(track.frames)
    for lo, hi, c in windows:
        anchors = [g for g in good if lo <= g <= hi]  # includes good frames between two gaps
        if not anchors:
            continue
        centre = _centre_near(track, (c[0] + c[-1]) // 2)
        if centre is None:
            continue
        targets = [f for f in range(lo, hi + 1) if f in set(c)]
        res = _estimate_frames(features, track, anchors, targets, (lo, hi), centre)
        for f, r in res.items():
            if not _accepted(r, min_inliers, max_crowd_rms_px, max_rms_px):
                continue
            P = r["matrix"]
            nz = _normalised_dlt(P)
            new_frames[f] = FrameCamera(
                f,
                P if nz is None else nz,
                SOURCE_TRACKED,
                rms_px=r["rms_px"],
                n_points=int(r["n_inliers"]),
                focal_px=r["focal"],
                rvec=tuple(np.asarray(r["rvec"]).tolist()),
                copied_from=int(r["from_frame"]),
            )
    return CameraTrack(
        new_frames, track.n_frames, centre=track.centre, principal_point=track.principal_point
    )


def _raw_field_errors(
    P: np.ndarray,
    pts: dict[int, tuple[float, float, float]],
    template,
    min_conf: float,
    bounds: tuple[int, int] | None = None,
    margin_px: float = 12.0,
):
    """Reprojection error (px) of the confident raw keypoints that sit inside the picture.

    Keypoints within ``margin_px`` of the border are dropped when ``bounds = (width, height)`` is
    given: the network clips the points it places outside the picture onto the border, and they
    are not detections of anything.
    """
    sel = [
        i
        for i, (u, v, c) in pts.items()
        if c >= min_conf
        and 0 <= i < len(template)
        and (
            bounds is None
            or (margin_px <= u <= bounds[0] - margin_px and margin_px <= v <= bounds[1] - margin_px)
        )
    ]
    if len(sel) < 3:
        return None
    uv = np.array([[pts[i][0], pts[i][1]] for i in sel])
    return np.hypot(*(project_points(P, np.asarray(template)[sel]) - uv).T)


def _ground_grid_errors(P_a: np.ndarray, P_b: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pitch distance (m) between the ground points of 9 image pixels seen by two cameras."""
    err = []
    for u in (0.25, 0.5, 0.75):
        for v in (0.42, 0.60, 0.79):
            a = ground_point(P_a, u * width, v * height)
            b = ground_point(P_b, u * width, v * height)
            if np.all(np.isfinite(a)) and np.all(np.isfinite(b)):
                err.append(float(np.hypot(a[0] - b[0], a[1] - b[1])))
    return np.array(err)


def _smooth_reference(track: CameraTrack, t: int, centre: np.ndarray, half: int = 3):
    """Camera of frame ``t`` smoothed over the good frames ``t-half..t+half`` (or None).

    The cameras fitted to network keypoints jitter by a few pixels from frame to frame while the
    real camera moves smoothly, so a smoothed reference is the fairer ruler for a tracker that is
    smooth by construction. Rotation and ``log f`` of the fixed-centre family are filtered with a
    quadratic Savitzky-Golay window of ``2 * half + 1`` frames (``t`` itself included).
    """
    frames = list(range(t - half, t + half + 1))
    good = (SOURCE_DLT3D, SOURCE_SOLVED)
    if any(f < 0 or f >= track.n_frames or track.source(f) not in good for f in frames):
        return None
    pp = track.principal_point
    grid = _template_grid(_PITCH_GRID)
    seeds = [_seed_from_matrix(track.matrix(f), centre, pp, grid) for f in frames]
    if any(s is None for s in seeds):
        return None
    arr = np.array([[*s[:3], math.log(s[3])] for s in seeds if s is not None])
    sm = savgol_filter(arr, 2 * half + 1, 2, axis=0)[half]
    return _build_matrix(sm[:3], math.exp(sm[3]), centre, pp)


def _pitch_pixel_diff(P_a: np.ndarray, P_b: np.ndarray, width: int, height: int) -> np.ndarray:
    """Pixel distance between two cameras on the pitch points that ``P_b`` shows in the picture."""
    ua, ub = project_points(P_a, _PITCH_GRID), project_points(P_b, _PITCH_GRID)
    ok = (
        np.isfinite(ua).all(axis=1)
        & np.isfinite(ub).all(axis=1)
        & (ub[:, 0] > 0.0)
        & (ub[:, 0] < width)
        & (ub[:, 1] > 0.2 * height)
        & (ub[:, 1] < height)
    )
    return np.hypot(*(ua[ok] - ub[ok]).T)


@_limited
def validate_tracking(
    track: CameraTrack,
    features: FeatureTracks,
    kps: dict[int, dict[int, tuple[float, float, float]]],
    template,
    *,
    ks=(5, 10, 20, 30),
    step: int = 2,
    anchors_per_side: int = TRACK_ANCHORS,
    min_conf: float = 0.5,
    min_inliers: int = TRACK_MIN_INLIERS,
    max_crowd_rms_px: float = TRACK_MAX_CROWD_RMS_PX,
    max_rms_px: float = TRACK_MAX_RMS_PX,
) -> dict:
    """Hold-out test of the tracker on the frames whose camera is known.

    Signature: ``validate_tracking(track, features, kps, template, *, ks=(5, 10, 20, 30), step=2,
    anchors_per_side=4, min_conf=0.5, min_inliers=60, max_crowd_rms_px=4.0, max_rms_px=60.0)
    -> {"one_sided": {k: stats}, "two_sided": {k: stats}}``.

    A frame ``t`` whose camera is ``dlt3d`` / ``solved`` is pretended unknown. *One-sided*: only
    the good frames up to ``t - k`` anchor the track (it is carried ``k`` frames forward through
    the feature matches). *Two-sided*: the anchors are the good frames up to ``t - k`` and from
    ``t + k`` on (a gap of ``2k - 1`` frames, as a real one). Every other frame of the window is
    unknown too. The tracked camera is compared with (a) the raw network keypoints of ``kps``
    (confidence >= ``min_conf``, not clipped onto the picture border) of that frame, reprojected, and (b) the real camera on 9 image
    pixels (``ground`` error, metres). ``stats``: ``n_cases``, ``field_px_median`` / ``_max``
    (pooled over points), ``field_px_case_median`` (median of the per-frame medians),
    ``ground_m_median`` / ``_max``, ``floor_px_median`` (the real camera against the same
    keypoints: the noise floor of the reference), ``copy_px_median`` (the ``approximate``
    baseline: the nearest anchor copied), ``smooth_px_median`` / ``_max`` (against the real camera
    smoothed over +-3 good frames, on the pitch points in view: the jitter of the keypoint fits
    removed), ``failed`` (frames the tracker could not estimate or whose estimate fails the gates of
    :func:`track_approximate_frames`, so they would stay ``approximate``: not in the other numbers), ``cases`` (one dict per frame:
    errors, anchor spread, feature rms). ``step`` subsamples ``t``.
    """
    good = {f for f in range(track.n_frames) if track.source(f) in (SOURCE_DLT3D, SOURCE_SOLVED)}
    w_img, h_img = features.width, features.height
    result: dict = {"one_sided": {}, "two_sided": {}}
    for mode in ("one_sided", "two_sided"):
        for k in ks:
            field, per_case, ground, floor, cases, copied, smooth = [], [], [], [], [], [], []
            n_cases = failed = 0
            for t in range(0, track.n_frames, max(1, step)):
                if t not in good or t not in features.obs:
                    continue
                a = t - k
                if mode == "one_sided":
                    anchors = [g for g in range(a - anchors_per_side + 1, a + 1) if g in good]
                    window = (a - anchors_per_side + 1, t)
                    ok_edges = a in good
                else:
                    b = t + k
                    anchors = [g for g in range(a - anchors_per_side + 1, a + 1) if g in good] + [
                        g for g in range(b, b + anchors_per_side) if g in good
                    ]
                    window = (a - anchors_per_side + 1, b + anchors_per_side - 1)
                    ok_edges = a in good and b in good
                if window[0] < 0 or window[1] >= track.n_frames or not ok_edges or len(anchors) < 2:
                    continue
                if any(f not in features.obs for f in range(window[0], window[1] + 1)):
                    continue
                truth = _raw_field_errors(
                    track.matrix(t), kps.get(t, {}), template, min_conf, (w_img, h_img)
                )
                if truth is None:
                    continue
                centre = _centre_near(track, t)
                if centre is None:
                    continue
                res = _estimate_frames(features, track, anchors, [t], window, centre)
                if t not in res or not _accepted(res[t], min_inliers, max_crowd_rms_px, max_rms_px):
                    failed += 1  # the tracker would leave this frame approximate
                    continue
                P = res[t]["matrix"]
                e = _raw_field_errors(P, kps.get(t, {}), template, min_conf, (w_img, h_img))
                if e is None:
                    continue
                n_cases += 1
                field.append(e)
                per_case.append(float(np.median(e)))
                floor.append(truth)
                g_err = _ground_grid_errors(P, track.matrix(t), w_img, h_img)
                ground.append(g_err)
                ref_s = _smooth_reference(track, t, centre)
                if ref_s is not None:
                    smooth.append(_pitch_pixel_diff(P, ref_s, w_img, h_img))
                nearest = (
                    anchors[-1] if mode == "one_sided" else min(anchors, key=lambda g: abs(g - t))
                )
                ref = _raw_field_errors(
                    track.matrix(nearest), kps.get(t, {}), template, min_conf, (w_img, h_img)
                )
                if ref is not None:
                    copied.append(ref)  # the approximate baseline: copy the nearest anchor
                cases.append(
                    {
                        "frame": t,
                        "field_px_median": per_case[-1],
                        "ground_m_median": float(np.median(g_err)) if len(g_err) else math.nan,
                        "anchor_spread_px": res[t]["spread_px"],
                        "rms_px": res[t]["rms_px"],
                        "crowd_rms_px": res[t]["crowd_rms_px"],
                        "fit_px": res[t]["fit_px"],
                        "n_inliers": res[t]["n_inliers"],
                    }
                )

            def pooled(parts):
                return np.concatenate(parts) if parts else np.zeros(0)

            fa, ga, fl, cp, sm = (pooled(x) for x in (field, ground, floor, copied, smooth))

            def med(a):
                return float(np.median(a)) if len(a) else math.nan

            def mx(a):
                return float(a.max()) if len(a) else math.nan

            result[mode][int(k)] = {
                "n_cases": n_cases,
                "failed": failed,
                "field_px_median": med(fa),
                "field_px_max": mx(fa),
                "field_px_case_median": med(np.array(per_case)),
                "ground_m_median": med(ga),
                "ground_m_max": mx(ga),
                "floor_px_median": med(fl),
                "copy_px_median": med(cp),
                "smooth_px_median": med(sm),
                "smooth_px_max": mx(sm),
                "cases": cases,
            }
    return result


def format_validation(result: dict) -> str:
    """The hold-out table of :func:`validate_tracking` as text."""
    lines = []
    titles = (
        ("one_sided", "one-sided (anchors before)"),
        ("two_sided", "two-sided (anchors both sides)"),
    )
    for mode, title in titles:
        lines.append(f"hold-out {title}: tracked vs raw keypoints / vs real camera")
        lines.append(
            "   k  cases  fail | field px median  max  case-med | ground m median  max | "
            "floor px  copy px | vs smoothed ref px med  max"
        )
        for k, s in sorted(result.get(mode, {}).items()):
            lines.append(
                f"{k:4d} {s['n_cases']:6d} {s['failed']:5d} | {s['field_px_median']:16.2f} "
                f"{s['field_px_max']:6.1f} {s['field_px_case_median']:9.2f} | "
                f"{s['ground_m_median']:14.3f} {s['ground_m_max']:6.2f} | "
                f"{s['floor_px_median']:7.2f} {s['copy_px_median']:8.1f} | "
                f"{s['smooth_px_median']:19.2f} {s['smooth_px_max']:5.1f}"
            )
    return "\n".join(lines)


# ----------------------------------------------------------------------------------------------
# Per-frame calibration files (written next to the SAM 3 outputs)
# ----------------------------------------------------------------------------------------------

FRAME_CAL_SUFFIX = "_frames"  # <stem>_frames.dlt3d / .dlt2d / _cameras.csv / _cameras.json
_SAM_DIR_RE = "processed_sam3dinov3_*"


def ground_homography(P: np.ndarray) -> np.ndarray:
    """Ground-plane (``z = 0``) homography of the 3x4 matrix ``P``: pixel ~ ``H @ [x, y, 1]``.

    Signature: ``ground_homography(P (3, 4)) -> (3, 3)``, the columns 1, 2 and 4 of ``P``
    normalised to ``H[2, 2] = 1`` (the layout of a vaila ``.dlt2d`` row). Raises ``ValueError``
    when ``H[2, 2]`` is zero (the pitch origin lies on the camera's principal plane).
    """
    P = np.asarray(P, dtype=float)
    H = P[:, [0, 1, 3]].copy()
    if not np.isfinite(H).all() or abs(H[2, 2]) < 1e-12 * max(1.0, float(np.abs(H).max())):
        raise ValueError("no finite ground homography (H[2, 2] is zero)")
    return H / H[2, 2]


def _normalised_dlt(P: np.ndarray) -> np.ndarray | None:
    """``P / P[2, 3]`` (the ``.dlt3d`` layout with ``L12 = 1``) or None when not possible."""
    P = np.asarray(P, dtype=float)
    if not np.isfinite(P).all() or abs(P[2, 3]) < 1e-12 * max(1.0, float(np.abs(P).max())):
        return None
    return P / P[2, 3]


def _num(v: float) -> str:
    return format(float(v), ".17g")


def write_frame_calibration(
    track: CameraTrack,
    out_dir,
    stem: str,
    *,
    video=None,
    fps: float | None = None,
    width: int | None = None,
    height: int | None = None,
) -> dict[str, Path]:
    """Write the per-frame calibration of every frame next to the SAM 3 outputs.

    Signature: ``write_frame_calibration(track, out_dir, stem, *, video=None, fps=None,
    width=None, height=None) -> {"dlt3d", "dlt2d", "cameras", "summary": Path}``.

    Files (``out_dir`` is created), one row per video frame ``0..n_frames-1``, all finite:

    * ``<stem>_frames.dlt3d``: ``frame,L1..L11`` (``L12 = 1``), the 3x4 matrix of the frame;
    * ``<stem>_frames.dlt2d``: ``frame,L1..L8``, the ground-plane homography (pixel ~ ``H @
      [x, y, 1]``, ``H[2, 2] = 1``): what ``QuickMeasureCalibration.from_dlt2d_sequence`` reads;
    * ``<stem>_frames_cameras.csv``: ``frame,source,rms_px,n_points,focal_px,from_frame`` with
      ``source`` ``dlt3d | solved | tracked | approximate`` and ``from_frame`` the frame an
      approximate / tracked camera starts from (blank otherwise);
    * ``<stem>_frames_cameras.json``: ``{video, n_frames, width, height, fps, counts, centre,
      principal_point_px, made_by, created}``.

    Units: metres of the FreeKiki pitch frame (x along, y across, z up, origin centre spot). A
    frame whose matrix cannot be normalised (``P[2, 3] = 0``) copies the nearest frame that can
    and is written as ``approximate`` (never hidden). Numbers are written with 17 digits, so a
    read gives back the same matrices.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    n = int(track.n_frames)
    rows: list[tuple[FrameCamera, np.ndarray, np.ndarray]] = []
    usable: list[int] = []
    for f in range(n):
        fc = track.info(f)
        P = _normalised_dlt(fc.matrix)
        H = None
        if P is not None:
            try:
                H = ground_homography(P)
            except ValueError:
                H = None
        if P is not None and H is not None:
            usable.append(f)
        rows.append((fc, P if P is not None else fc.matrix, H if H is not None else np.eye(3)))
    if not usable:
        raise ValueError("no frame has a finite, normalisable camera matrix")
    fixed: list[tuple[FrameCamera, np.ndarray, np.ndarray]] = []
    for f, (fc, P, H) in enumerate(rows):
        if f in usable:
            fixed.append((fc, P, H))
            continue
        src = min(usable, key=lambda g: (abs(g - f), g))
        base_fc, base_P, base_H = rows[src]
        fixed.append(
            (
                FrameCamera(f, base_P, SOURCE_APPROX, copied_from=src, focal_px=base_fc.focal_px),
                base_P,
                base_H,
            )
        )
    p3 = out / f"{stem}{FRAME_CAL_SUFFIX}.dlt3d"
    p2 = out / f"{stem}{FRAME_CAL_SUFFIX}.dlt2d"
    pc = out / f"{stem}{FRAME_CAL_SUFFIX}_cameras.csv"
    pj = out / f"{stem}{FRAME_CAL_SUFFIX}_cameras.json"
    l3 = ["frame", *[f"L{i}" for i in range(1, 12)]]
    l2 = ["frame", *[f"L{i}" for i in range(1, 9)]]
    lines3, lines2, linesc = (
        [",".join(l3)],
        [",".join(l2)],
        ["frame,source,rms_px,n_points,focal_px,from_frame"],
    )
    counts: dict[str, int] = {}
    for f, (fc, P, H) in enumerate(fixed):
        v3 = [
            P[0, 0],
            P[0, 1],
            P[0, 2],
            P[0, 3],
            P[1, 0],
            P[1, 1],
            P[1, 2],
            P[1, 3],
            P[2, 0],
            P[2, 1],
            P[2, 2],
        ]
        v2 = [H[0, 0], H[0, 1], H[0, 2], H[1, 0], H[1, 1], H[1, 2], H[2, 0], H[2, 1]]
        lines3.append(",".join([str(f), *[_num(v) for v in v3]]))
        lines2.append(",".join([str(f), *[_num(v) for v in v2]]))
        counts[fc.source] = counts.get(fc.source, 0) + 1
        linesc.append(
            ",".join(
                [
                    str(f),
                    fc.source,
                    "" if math.isnan(fc.rms_px) else _num(fc.rms_px),
                    str(int(fc.n_points)),
                    "" if math.isnan(fc.focal_px) else _num(fc.focal_px),
                    "" if fc.copied_from is None else str(int(fc.copied_from)),
                ]
            )
        )
    p3.write_text("\n".join(lines3) + "\n", encoding="utf-8")
    p2.write_text("\n".join(lines2) + "\n", encoding="utf-8")
    pc.write_text("\n".join(linesc) + "\n", encoding="utf-8")
    from datetime import datetime

    summary = {
        "video": None if video is None else str(video),
        "n_frames": n,
        "width": None if width is None else int(width),
        "height": None if height is None else int(height),
        "fps": None if fps is None else float(fps),
        "counts": counts,
        "centre": None if track.centre is None else [float(x) for x in track.centre],
        "principal_point_px": list(track.principal_point),
        "made_by": "vaila.kick_measure 0.4.19 (DLT3D kept, fixed-centre solve, feature tracking)",
        "tracking_version": TRACKING_MODEL_VERSION,
        "created": datetime.now().isoformat(timespec="seconds"),
    }
    pj.write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return {"dlt3d": p3, "dlt2d": p2, "cameras": pc, "summary": pj}


def read_frame_calibration(path_dlt3d) -> CameraTrack:
    """Read ``<stem>_frames.dlt3d`` (+ the sibling cameras csv / json) back into a track.

    Signature: ``read_frame_calibration(path_dlt3d) -> CameraTrack``. The matrices are the
    normalised ones of the file (``P[2, 3] = 1``); ``source`` / ``rms_px`` / ``n_points`` /
    ``focal_px`` / ``from_frame`` (``FrameCamera.copied_from``) come from
    ``<stem>_frames_cameras.csv`` (without it every frame is ``"given"``); the centre and the
    principal point from the json when present. Raises ``ValueError`` on a file without rows or
    whose frames are not ``0..n-1``.
    """
    path = Path(path_dlt3d)
    d = pd.read_csv(path)
    if d.empty or d.shape[1] < 12:
        raise ValueError(f"{path.name} has no per-frame DLT3D rows")
    meta: dict[int, dict] = {}
    pc = path.with_name(f"{path.stem}_cameras.csv")
    if pc.is_file():
        for r in pd.read_csv(pc).to_dict("records"):
            meta[int(r["frame"])] = r
    info: dict = {}
    pj = path.with_name(f"{path.stem}_cameras.json")
    if pj.is_file():
        try:
            info = json.loads(pj.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            info = {}
    frames: dict[int, FrameCamera] = {}
    for row in d.to_numpy(dtype=float):
        f = int(row[0])
        L = [float(v) for v in row[1:12]]
        P = np.array([L[0:4], L[4:8], [L[8], L[9], L[10], 1.0]])
        m = meta.get(f, {})
        src = str(m.get("source", SOURCE_GIVEN)) if m else SOURCE_GIVEN
        rms = m.get("rms_px")
        npts = m.get("n_points")
        focal = m.get("focal_px")
        frm = m.get("from_frame")
        try:
            fl, R = _decompose(P)
            rvec = tuple(Rotation.from_matrix(R).as_rotvec().tolist())
        except (ValueError, np.linalg.LinAlgError):
            fl, rvec = math.nan, None
        frames[f] = FrameCamera(
            f,
            P,
            src,
            rms_px=math.nan if rms is None or pd.isna(rms) else float(rms),
            n_points=0 if npts is None or pd.isna(npts) else int(npts),
            focal_px=float(focal) if focal is not None and not pd.isna(focal) else fl,
            rvec=rvec,
            copied_from=None if frm is None or pd.isna(frm) else int(frm),
        )
    if sorted(frames) != list(range(len(frames))):
        raise ValueError(f"{path.name}: the frames are not 0..n-1, one row per video frame")
    pp = info.get("principal_point_px") or (960.0, 540.0)
    centre = info.get("centre")
    return CameraTrack(
        frames,
        len(frames),
        centre=None if centre is None else np.asarray(centre, dtype=float),
        principal_point=(float(pp[0]), float(pp[1])),
    )


def find_frame_calibration(work_dir_or_sam_dir, stem: str) -> Path | None:
    """Newest ``<stem>_frames.dlt3d`` of a run, or None.

    Signature: ``find_frame_calibration(work_dir_or_sam_dir, stem) -> Path | None``. Looks, under
    the given folder, for ``**/sam3dinov3/processed_sam3dinov3_*/<stem>/<stem>_frames.dlt3d`` (also
    when the folder is the ``work/<stem>`` folder, the ``sam3dinov3`` folder, one
    ``processed_sam3dinov3_*`` folder or the ``<stem>`` folder itself); the newest by modification
    time wins.
    """
    base = Path(work_dir_or_sam_dir)
    name = f"{stem}{FRAME_CAL_SUFFIX}.dlt3d"
    found: list[Path] = []
    direct = base / name
    if direct.is_file():
        found.append(direct)
    found.extend(base.glob(f"{stem}/{name}"))
    found.extend(base.glob(f"{_SAM_DIR_RE}/{stem}/{name}"))
    found.extend(base.glob(f"**/{_SAM_DIR_RE}/{stem}/{name}"))
    unique = {p.resolve(): p for p in found if p.is_file()}
    if not unique:
        return None
    return max(unique.values(), key=lambda p: (p.stat().st_mtime, str(p)))


def sam_output_dir(run_dir, stem: str) -> Path | None:
    """The newest ``work/<stem>/sam3dinov3/processed_sam3dinov3_*/<stem>`` folder, or None.

    Signature: ``sam_output_dir(kick_dir, stem) -> Path | None``: where the SAM 3 + SAM 3D Body
    outputs of the camera live and so where :func:`write_frame_calibration` writes.
    """
    base = Path(run_dir) / "work" / stem / "sam3dinov3"
    cands = [p for p in base.glob(f"{_SAM_DIR_RE}/{stem}") if p.is_dir()]
    if not cands:
        return None
    return max(cands, key=lambda p: (p.parent.name, p.stat().st_mtime))


def write_run_frame_calibration(run: KickMeasureRun, out_dir=None) -> dict[str, Path]:
    """Write the per-frame calibration of an opened run next to its SAM 3 outputs.

    Signature: ``write_run_frame_calibration(run, out_dir=None) -> {"dlt3d", "dlt2d", "cameras",
    "summary": Path}``. ``out_dir`` defaults to :func:`sam_output_dir` of the run; raises
    ``ValueError`` when the run has no SAM 3 output folder yet (and none is given).
    """
    target = Path(out_dir) if out_dir is not None else sam_output_dir(run.run_dir, run.stem)
    if target is None:
        raise ValueError(
            f"{run.run_dir.name} has no SAM 3 output folder (work/{run.stem}/sam3dinov3/"
            f"processed_sam3dinov3_*/{run.stem}) yet: run the sam stage first or pass out_dir"
        )
    return write_frame_calibration(
        run.cameras,
        target,
        run.stem,
        video=run.video,
        fps=run.fps,
        width=run.width,
        height=run.height,
    )


# ----------------------------------------------------------------------------------------------
# Plane of a line (kick -> arrival): distance along the line and height
# ----------------------------------------------------------------------------------------------


class LinePlane:
    """The vertical plane through a line on the ground, to read ``(s, z)`` of a ball ray.

    ``s`` is the distance (m) along the line from ``ground_a`` towards ``ground_b``; ``z`` the
    height (m) above the grass. One camera cannot tell where the ball is across the line, so
    the ball is taken **in this plane**; :meth:`lateral_sensitivity` tells how much ``z`` moves
    when the plane is shifted sideways (about 0.1 m of height per metre here, typically).

    Signature: ``LinePlane(camera, frame_a, ground_a, ground_b)`` with ``camera`` a
    :class:`CameraTrack` (anything with ``matrix(frame)``), ``frame_a`` the frame in which the
    line was marked (information), ``ground_a``/``ground_b`` pitch points ``(x, y[, z])`` in
    metres (only ``x, y`` are used). Build it from two clicks with :meth:`from_pixels`.
    """

    def __init__(self, camera, frame_a: int, ground_a, ground_b) -> None:
        a = np.asarray(ground_a, dtype=float)[:2]
        b = np.asarray(ground_b, dtype=float)[:2]
        length = float(np.hypot(*(b - a)))
        if not np.isfinite(length) or length < 1e-9:
            raise ValueError("the two ground points of a LinePlane must differ")
        self.camera = camera
        self.frame_a = int(frame_a)
        self.ground_a = np.array([a[0], a[1], 0.0])
        self.ground_b = np.array([b[0], b[1], 0.0])
        self.length = length
        self.direction = np.array([(b[0] - a[0]) / length, (b[1] - a[1]) / length, 0.0])
        self.normal = np.array([-self.direction[1], self.direction[0], 0.0])

    @classmethod
    def from_pixels(cls, camera, frame_a: int, uv_a, uv_b) -> LinePlane:
        """Line from two pixels of ``frame_a`` clicked on the grass.

        Signature: ``from_pixels(camera, frame_a, uv_a (2,), uv_b (2,)) -> LinePlane``.
        """
        P = camera.matrix(frame_a)
        a = ground_point(P, uv_a[0], uv_a[1], 0.0)
        b = ground_point(P, uv_b[0], uv_b[1], 0.0)
        if not (np.all(np.isfinite(a)) and np.all(np.isfinite(b))):
            raise ValueError("a clicked point does not reach the ground")
        return cls(camera, frame_a, a, b)

    def locate_xyz(self, frame: int, uv, lateral: float = 0.0) -> np.ndarray:
        """3D point ``(x, y, z)`` m where the ray of ``uv`` in ``frame`` cuts the plane.

        Signature: ``locate_xyz(frame, uv (2,), lateral=0.0) -> (3,)``; ``lateral`` shifts the
        plane sideways by that many metres (positive towards ``normal``). nan on a miss.
        """
        P = self.camera.matrix(frame)
        X = _ray_vertical_plane(P, uv, self.ground_a + lateral * self.normal, self.normal)
        return np.full(3, math.nan) if X is None else X

    def locate(self, frame: int, uv) -> tuple[float, float]:
        """``(s, z)`` of pixel ``uv`` in ``frame``: metres along the line and metres of height.

        Signature: ``locate(frame, uv (2,)) -> (s, z)``; ``(nan, nan)`` when the ray misses.
        """
        X = self.locate_xyz(frame, uv)
        if not np.all(np.isfinite(X)):
            return math.nan, math.nan
        return float(np.dot(X[:2] - self.ground_a[:2], self.direction[:2])), float(X[2])

    def lateral_sensitivity(self, frame: int, uv, lateral: float = 2.0) -> float:
        """Change of the height ``z`` (m) when the plane is shifted ``lateral`` m sideways.

        Signature: ``lateral_sensitivity(frame, uv, lateral=2.0) -> float`` = ``z(shifted) -
        z(plane)``; a measure of the uncertainty of the height of this ray.
        """
        a = self.locate_xyz(frame, uv)
        b = self.locate_xyz(frame, uv, lateral)
        return float(b[2] - a[2])

    def point(self, s: float, z: float = 0.0) -> np.ndarray:
        """Pitch point ``(x, y, z)`` m at distance ``s`` along the line and height ``z``."""
        return self.ground_a + s * self.direction + np.array([0.0, 0.0, float(z)])

    def height_line(
        self, frame: int, z: float, s0: float = 0.0, s1: float | None = None
    ) -> np.ndarray:
        """Pixels of the horizontal guide at height ``z`` from ``s0`` to ``s1`` in ``frame``.

        Signature: ``height_line(frame, z, s0=0.0, s1=None) -> (2, 2)`` rows ``(u, v)`` px
        (``s1`` defaults to the line length); to be drawn over the video.
        """
        end = self.length if s1 is None else float(s1)
        return self.camera.project(frame, np.array([self.point(s0, z), self.point(end, z)]))


# ----------------------------------------------------------------------------------------------
# Time, speed, spin
# ----------------------------------------------------------------------------------------------


def event_time(frame_start: int, frame_end: int, fps: float) -> float:
    """Duration (s) of an event from ``frame_start`` to ``frame_end``: ``(end - start) / fps``.

    Signature: ``event_time(frame_start, frame_end, fps) -> float`` seconds (negative when the
    end comes first). The resolution is one frame: see :func:`event_time_uncertainty`.
    """
    if fps <= 0:
        raise ValueError("fps must be positive")
    return (int(frame_end) - int(frame_start)) / float(fps)


def event_time_uncertainty(fps: float) -> float:
    """Uncertainty (s) of a duration read in frames: one frame, ``1 / fps``."""
    if fps <= 0:
        raise ValueError("fps must be positive")
    return 1.0 / float(fps)


def mean_speed(point_a, point_b, frame_start: int, frame_end: int, fps: float) -> float:
    """Mean speed (m/s) between two ground points over an event.

    Signature: ``mean_speed(point_a (2|3,), point_b (2|3,), frame_start, frame_end, fps) ->
    float`` = ``|b - a| / event_time``; the points are pitch positions in metres. Raises
    ``ValueError`` when the two frames are the same.
    """
    dt = abs(event_time(frame_start, frame_end, fps))
    if dt == 0.0:
        raise ValueError("start and end frame are the same: no duration")
    d = np.asarray(point_b, dtype=float) - np.asarray(point_a, dtype=float)
    return float(np.linalg.norm(d)) / dt


@dataclass(frozen=True)
class SpinResult:
    """Spin count: ``turns`` whole turns in ``duration_s``; ``rps`` and ``rpm`` follow.

    ``uncertainty_s`` is one frame (``1 / fps``) on the duration; ``rps_uncertainty`` and
    ``rpm_uncertainty`` follow from it (``rps * (1/fps) / duration``).
    """

    turns: int
    duration_s: float
    rps: float
    rpm: float
    uncertainty_s: float
    rps_uncertainty: float
    rpm_uncertainty: float


def spin_rate(mark_frames, fps: float) -> SpinResult:
    """Spin of the ball from the frames in which the **same mark** is seen again.

    Signature: ``spin_rate(mark_frames (k,) frame indices, fps) -> SpinResult``. The first mark
    opens the count: ``k`` marks mean ``k - 1`` turns between the first and the last frame, so
    ``rps = (k - 1) / ((last - first) / fps)`` and ``rpm = 60 rps``. Needs ``k >= 2``. The
    duration is known to +-1/fps (``uncertainty_s``). Above ``fps / 2`` turns per second a
    mark is seen less than twice per turn and the count aliases: this function cannot know.
    """
    marks = np.sort(np.asarray(list(mark_frames), dtype=float))
    if marks.size < 2:
        raise ValueError("at least two frames of the same mark are needed")
    if fps <= 0:
        raise ValueError("fps must be positive")
    duration = float(marks[-1] - marks[0]) / float(fps)
    if duration <= 0:
        raise ValueError("the marks must be in different frames")
    turns = int(marks.size - 1)
    rps = turns / duration
    unc = 1.0 / float(fps)
    return SpinResult(turns, duration, rps, 60.0 * rps, unc, rps * unc / duration,
                      60.0 * rps * unc / duration)  # fmt: skip


# ----------------------------------------------------------------------------------------------
# Paths: length, arc, parabola
# ----------------------------------------------------------------------------------------------


def path_length(points) -> float:
    """Length (same unit as ``points``) of the polyline through ``points``.

    Signature: ``path_length(points (n, d)) -> float``; rows with nan are skipped.
    """
    p = np.atleast_2d(np.asarray(points, dtype=float))
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 2:
        return 0.0
    return float(np.linalg.norm(np.diff(p, axis=0), axis=1).sum())


@dataclass(frozen=True)
class ArcFit:
    """Circle through a curve: ``centre`` (2,), ``radius``, ``angle_rad`` covered from the first
    to the last point, ``arc_length``, ``chord`` (first -> last), ``sagitta`` (height of the arc
    over the chord) and ``rms`` residual distance to the circle. A straight line gives
    ``radius = inf``, ``angle_rad = 0``, ``sagitta = 0``. Units: those of the points."""

    centre: tuple[float, float]
    radius: float
    angle_rad: float
    arc_length: float
    chord: float
    sagitta: float
    rms: float

    @property
    def angle_deg(self) -> float:
        return math.degrees(self.angle_rad)


def fit_arc(points) -> ArcFit:
    """Least-squares circle (algebraic start, geometric refinement) through ``points``.

    Signature: ``fit_arc(points (n >= 3, 2)) -> ArcFit``; ``(x, y)`` ground metres or
    ``(s, z)`` of a :class:`LinePlane`. Raises ``ValueError`` with fewer than 3 finite points.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        raise ValueError("an arc needs at least 3 points")
    mean = p.mean(axis=0)
    q = p - mean
    sv = np.linalg.svd(q, compute_uv=False)
    chord = float(np.hypot(*(p[-1] - p[0])))
    if sv[0] == 0.0 or sv[1] / sv[0] < 1e-9:
        return ArcFit((math.nan, math.nan), math.inf, 0.0, path_length(p), chord, 0.0, 0.0)
    A = np.c_[2 * q[:, 0], 2 * q[:, 1], np.ones(len(q))]
    sol, *_ = np.linalg.lstsq(A, (q**2).sum(axis=1), rcond=None)
    c0 = sol[:2]
    r0 = math.sqrt(max(sol[2] + float(c0 @ c0), 1e-18))
    res = least_squares(lambda t: np.hypot(q[:, 0] - t[0], q[:, 1] - t[1]) - t[2], np.r_[c0, r0])
    cx, cy, radius = float(res.x[0]), float(res.x[1]), abs(float(res.x[2]))
    ang = np.unwrap(np.arctan2(q[:, 1] - cy, q[:, 0] - cx))
    angle = float(abs(ang[-1] - ang[0]))
    rms = float(np.sqrt(np.mean(res.fun**2)))
    return ArcFit(
        (cx + float(mean[0]), cy + float(mean[1])),
        radius,
        angle,
        radius * angle,
        chord,
        radius * (1.0 - math.cos(angle / 2.0)),
        rms,
    )


@dataclass(frozen=True)
class ParabolaFit:
    """``z = a s^2 + b s + c`` with its apex ``(apex_s, apex_z)`` (nan when ``a == 0``) and the
    ``rms`` residual of ``z``. Units: those of the points."""

    a: float
    b: float
    c: float
    apex_s: float
    apex_z: float
    rms: float

    def z_at(self, s) -> np.ndarray:
        """Height of the parabola at ``s``."""
        s = np.asarray(s, dtype=float)
        return self.a * s**2 + self.b * s + self.c


def fit_parabola(points) -> ParabolaFit:
    """Least-squares parabola ``z(s)`` through ``points``.

    Signature: ``fit_parabola(points (n >= 3, 2)) -> ParabolaFit`` with rows ``(s, z)``, for
    example the profile given by :meth:`LinePlane.locate`.
    """
    p = np.asarray(points, dtype=float)
    if p.ndim != 2 or p.shape[1] != 2:
        raise ValueError("points must have shape (n, 2)")
    p = p[np.isfinite(p).all(axis=1)]
    if len(p) < 3:
        raise ValueError("a parabola needs at least 3 points")
    a, b, c = (float(v) for v in np.polyfit(p[:, 0], p[:, 1], 2))
    rms = float(np.sqrt(np.mean((np.polyval([a, b, c], p[:, 0]) - p[:, 1]) ** 2)))
    if a == 0.0:
        return ParabolaFit(a, b, c, math.nan, math.nan, rms)
    s_apex = -b / (2 * a)
    return ParabolaFit(a, b, c, s_apex, a * s_apex**2 + b * s_apex + c, rms)


# ----------------------------------------------------------------------------------------------
# Velocity / acceleration series
# ----------------------------------------------------------------------------------------------


def _derivative(positions, fps: float, order: int, smooth_window: int) -> np.ndarray:
    pos = np.asarray(positions, dtype=float)
    one_d = pos.ndim == 1
    pos2 = pos.reshape(-1, 1) if one_d else pos
    n = pos2.shape[0]
    out = np.full_like(pos2, math.nan)
    min_n = 2 if order == 1 else 3
    if fps <= 0:
        raise ValueError("fps must be positive")
    if n >= min_n:
        dt = 1.0 / float(fps)
        for j in range(pos2.shape[1]):
            col = pos2[:, j]
            ok = np.isfinite(col)
            if ok.sum() < min_n:
                continue
            idx = np.arange(n)
            filled = np.interp(idx, idx[ok], col[ok])  # holes: linear, then masked back
            win = int(smooth_window)
            if win >= 3:
                win = min(win, n if n % 2 else n - 1)
            if win >= 3 and win > 2:
                d = savgol_filter(filled, win, 2, deriv=order, delta=dt)
            elif order == 1:
                d = np.gradient(filled, dt, edge_order=2 if n >= 3 else 1)
            else:
                d = np.empty(n)
                d[1:-1] = (filled[:-2] - 2 * filled[1:-1] + filled[2:]) / dt**2
                d[0], d[-1] = d[1], d[-2]
            d = np.asarray(d, dtype=float)
            d[~ok] = math.nan
            out[:, j] = d
    return out[:, 0] if one_d else out


def velocity_series(positions, fps: float, smooth_window: int = 0) -> np.ndarray:
    """Velocity per frame from positions per frame (units of position per second).

    Signature: ``velocity_series(positions (n,) or (n, d), fps, smooth_window=0) -> same
    shape``. Central differences (second-order at the ends). ``smooth_window`` (odd, >= 3)
    switches to a Savitzky-Golay filter of order 2 over that many frames. nan rows stay nan.
    Positions are one per consecutive frame (a missing frame is a nan row).
    """
    return _derivative(positions, fps, 1, smooth_window)


def acceleration_series(positions, fps: float, smooth_window: int = 0) -> np.ndarray:
    """Acceleration per frame (units of position per second squared).

    Signature: ``acceleration_series(positions (n,) or (n, d), fps, smooth_window=0) -> same
    shape``. Second central difference (the end values repeat their neighbour), or a
    Savitzky-Golay second derivative when ``smooth_window`` is odd and >= 3. Differences of
    positions amplify noise: smooth a hand-clicked track before reading an acceleration.
    """
    return _derivative(positions, fps, 2, smooth_window)


def speed_series(positions, fps: float, smooth_window: int = 0) -> np.ndarray:
    """Speed (norm of the velocity) per frame, ``(n,)``. Same arguments as
    :func:`velocity_series`."""
    v = velocity_series(positions, fps, smooth_window)
    return np.abs(v) if v.ndim == 1 else np.linalg.norm(v, axis=1)


# ----------------------------------------------------------------------------------------------
# Opening a kick_analysis folder
# ----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Person:
    """One SAM person in one frame: ``id`` (SAM ``obj_id``), ``bbox`` ``(x, y, w, h)`` px with
    ``(x, y)`` the top-left corner, ``foot_uv`` pixel under the box (bottom centre), ``foot_xy``
    the pitch position in metres (None when unknown) and ``foot_source``: ``"ground_tracks"``
    (FreeKiki track) or ``"camera"`` (the box foot on the grass seen by that frame's camera)."""

    id: int
    bbox: tuple[float, float, float, float]
    foot_uv: tuple[float, float]
    foot_xy: tuple[float, float] | None = None
    foot_source: str = ""


@dataclass
class KickMeasureRun:
    """An opened kick analysis, ready to measure.

    ``run_dir`` the ``kick_analysis_*`` folder; ``video`` / ``stem`` the camera chosen
    (``camera_name`` ``"reference"`` or ``"other"``); ``threed`` its 3D folder, ``predict`` its
    detect run; ``fps``, ``n_frames``, ``width``, ``height`` of the video; ``cameras`` the
    :class:`CameraTrack`; ``ball_uv`` ``{frame: (u, v)}`` the ball in pixels and ``ball_source``
    where it came from; ``people`` ``{frame: [Person, ...]}``; ``warnings`` what was missing.
    """

    run_dir: Path
    camera_name: str
    video: Path
    stem: str
    threed: Path
    predict: Path
    fps: float
    n_frames: int
    width: int
    height: int
    cameras: CameraTrack
    ball_uv: dict[int, tuple[float, float]] = field(default_factory=dict)
    ball_source: str = ""
    people: dict[int, list[Person]] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    def people_at(self, frame: int) -> list[Person]:
        """People of ``frame`` (empty list when none)."""
        return self.people.get(int(frame), [])

    def n_people(self) -> int:
        """Distinct SAM ids over the whole video."""
        return len({p.id for ps in self.people.values() for p in ps})

    def summary(self) -> dict:
        """Counts for ``--info``: cameras per source, people, ball frames."""
        c = self.cameras.counts()
        return {
            "run": str(self.run_dir),
            "camera": self.camera_name,
            "video": str(self.video),
            "fps": self.fps,
            "n_frames": self.n_frames,
            "size_px": [self.width, self.height],
            "cameras": c,
            "approximate_frames": self.cameras.approximate_frames(),
            "tracked_frames": [
                f for f in range(self.n_frames) if self.cameras.source(f) == SOURCE_TRACKED
            ],
            "centre_m": None if self.cameras.centre is None else self.cameras.centre.tolist(),
            "ball_frames": len(self.ball_uv),
            "ball_source": self.ball_source,
            "people_ids": self.n_people(),
            "people_boxes": sum(len(v) for v in self.people.values()),
            "warnings": list(self.warnings),
        }

    def to_dict(self) -> dict:
        """JSON-ready: :meth:`summary` plus the per-frame cameras."""
        d = self.summary()
        d["camera_track"] = self.cameras.to_dict()
        d["ball_uv"] = {str(k): list(v) for k, v in sorted(self.ball_uv.items())}
        d["people_per_frame"] = {str(k): len(v) for k, v in sorted(self.people.items())}
        return d


def _freekiki():
    """The freekiki module (lazy), or None when it cannot be imported."""
    try:
        try:
            from . import freekiki
        except ImportError:
            import freekiki  # ty: ignore[unresolved-import]
        return freekiki
    except Exception:  # noqa: BLE001 - optional reuse; the local fallback below is complete
        return None


def _find_kick_dir(path) -> Path:
    fk = _freekiki()
    if fk is not None:
        try:
            found = fk._kick_dir(path)
            if found is not None:
                return Path(found)
        except Exception:  # noqa: BLE001
            pass
    folder = Path(path).expanduser()
    folder = folder if folder.is_dir() else folder.parent
    for candidate in [folder, *list(folder.resolve().parents)[:6]]:
        if (candidate / KICK_STATUS).is_file():
            return candidate.resolve()
    raise ValueError(
        f"{path} is not a kick_analysis folder (no {KICK_STATUS} in it or its parents)"
    )


def _kick_camera(run: Path, camera: str) -> dict:
    """``{video, stem, threed, predict}`` of one camera (freekiki reuse, local fallback)."""
    if camera not in ("reference", "other"):
        raise ValueError(f"camera must be reference or other, got {camera!r}")
    fk = _freekiki()
    if fk is not None:
        try:
            return dict(fk.kick_run_camera(run, camera))
        except ValueError:
            raise
        except Exception:  # noqa: BLE001 - fall through to the local reading
            pass
    status = json.loads((run / KICK_STATUS).read_text(encoding="utf-8"))
    videos = [Path(v) for v in (status.get("options") or {}).get("videos") or []]
    if not videos:
        videos = [
            Path(s["inputs"]["video"]["path"])
            for k, s in status.get("stages", {}).items()
            if k.startswith("detect:") and "video" in s.get("inputs", {})
        ]
    index = 0 if camera == "reference" else 1
    if index >= len(videos):
        raise ValueError(f"{run.name} has no {camera} camera")
    video = videos[index]
    stem = video.stem
    stage = status.get("stages", {}).get(f"detect:{stem}") or {}
    predict = (stage.get("result") or {}).get("predict")
    if not predict or not Path(predict).is_dir():
        found = sorted((run / "work" / stem).glob(f"freekiki_predict_{stem}_*"))
        predict = found[-1] if found else None
    return {
        "run": run,
        "video": video,
        "stem": stem,
        "threed": run / f"{stem}_3d_{status.get('ts', '')}",
        "predict": None if predict is None else Path(predict),
    }


def load_field_template(threed: Path | None = None) -> np.ndarray:
    """The pitch keypoints ``(n, 3)`` metres, row ``i`` = point ``i`` of the 49-point template.

    Signature: ``load_field_template(threed=None) -> (n, 3)``. Read from
    ``vaila/models/soccerfield_kiki.csv``; when missing, from ``field_kps_3d.csv`` of ``threed``.
    """
    here = Path(__file__).resolve().parent / "models" / "soccerfield_kiki.csv"
    if here.is_file():
        t = pd.read_csv(here).sort_values("point_number")
        return t[["x", "y", "z"]].to_numpy(dtype=float)
    if threed is not None and (Path(threed) / "field_kps_3d.csv").is_file():
        k = pd.read_csv(Path(threed) / "field_kps_3d.csv")
        t = k[k.frame == k.frame.min()].sort_values("point")
        return t[["x_m", "y_m", "z_m"]].to_numpy(dtype=float)
    raise FileNotFoundError("no pitch template (soccerfield_kiki.csv / field_kps_3d.csv)")


def _read_dlt3d(path: Path) -> dict[int, np.ndarray]:
    d = pd.read_csv(path).dropna()
    out: dict[int, np.ndarray] = {}
    for r in d.to_numpy(dtype=float):
        L = [float(v) for v in r[1:12]]
        out[int(r[0])] = np.array([L[0:4], L[4:8], [L[8], L[9], L[10], 1.0]])
    return out


def _read_raw_keypoints(path: Path) -> dict[int, dict[int, tuple[float, float, float]]]:
    r = pd.read_csv(path)
    n = sum(1 for c in r.columns if c.startswith("p") and c.endswith("_conf"))
    out: dict[int, dict[int, tuple[float, float, float]]] = {}
    for row in r.to_dict("records"):
        pts: dict[int, tuple[float, float, float]] = {}
        for i in range(n):
            c = row.get(f"p{i}_conf")
            u, v = row.get(f"p{i}_x"), row.get(f"p{i}_y")
            if c is None or u is None or v is None:
                continue
            if not (np.isfinite(c) and np.isfinite(u) and np.isfinite(v)) or not (u > 0 or v > 0):
                continue
            pts[i] = (float(u), float(v), float(c))
        out[int(row["frame"])] = pts
    return out


def _video_info(video: Path) -> tuple[float | None, int | None, int | None, int | None]:
    """``(fps, n_frames, width, height)`` of the video, None for what cannot be read."""
    try:
        import cv2
    except ImportError:
        return None, None, None, None
    if not video.is_file():
        return None, None, None, None
    cap = cv2.VideoCapture(str(video))
    try:
        if not cap.isOpened():
            return None, None, None, None
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0) or None
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0) or None
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0) or None
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0) or None
        return fps, n, w, h
    finally:
        cap.release()


def _readme_fps(run: Path) -> float | None:
    readme = run / "README.txt"
    if not readme.is_file():
        return None
    for line in readme.read_text(encoding="utf-8", errors="replace").splitlines():
        if line.startswith("fps:"):
            try:
                return float(line.split(":", 1)[1].split("|")[0])
            except ValueError:
                return None
    return None


def _load_ball(cam: dict, source: str) -> tuple[dict[int, tuple[float, float]], str]:
    markers = Path(cam["video"]).with_name(f"{cam['stem']}_markers.csv")
    detect = Path(cam["threed"]) / "ball.csv"
    if source in ("auto", "markers") and markers.is_file():
        m = pd.read_csv(markers)
        if {"frame", "p0_x", "p0_y"} <= set(m.columns):
            pts = {
                int(r["frame"]): (float(r["p0_x"]), float(r["p0_y"]))
                for r in m.to_dict("records")
                if np.isfinite(r["p0_x"]) and np.isfinite(r["p0_y"])
            }
            if pts:
                return pts, str(markers)
    if source in ("auto", "detect") and detect.is_file():
        b = pd.read_csv(detect)
        pts = {
            int(r["frame"]): (float(r["cx"]), float(r["cy"]))
            for r in b.to_dict("records")
            if np.isfinite(r["cx"]) and np.isfinite(r["cy"])
        }
        return pts, str(detect)
    return {}, ""


def _find_sam_tracks(run: Path, stem: str) -> Path | None:
    found = sorted((run / "work" / stem).glob("**/sam3/sam_tracks.csv"))
    return found[-1] if found else None


def _load_people(
    run: Path, cam: dict, cameras: CameraTrack
) -> tuple[dict[int, list[Person]], Path | None]:
    sam = _find_sam_tracks(run, cam["stem"])
    if sam is None:
        return {}, None
    tracks = pd.read_csv(sam)
    ground: dict[tuple[int, int], tuple[float, float]] = {}
    gt = Path(cam["threed"]) / "ground_tracks.csv"
    if gt.is_file():
        g = pd.read_csv(gt)
        for r in g.to_dict("records"):
            if np.isfinite(r["x_m"]) and np.isfinite(r["y_m"]):
                ground[(int(r["frame"]), int(r["person_id"]))] = (float(r["x_m"]), float(r["y_m"]))
    people: dict[int, list[Person]] = {}
    for r in tracks.to_dict("records"):
        frame, pid = int(r["frame"]), int(r["obj_id"])
        x, y, w, h = float(r["x_px"]), float(r["y_px"]), float(r["w_px"]), float(r["h_px"])
        if not all(np.isfinite([x, y, w, h])):
            continue
        foot = (x + w / 2.0, y + h)
        xy = ground.get((frame, pid))
        if xy is not None:
            source = "ground_tracks"
        else:
            g3 = cameras.ground_point(frame, foot[0], foot[1], 0.0)
            xy = (float(g3[0]), float(g3[1])) if np.all(np.isfinite(g3)) else None
            source = "camera" if xy is not None else ""
        people.setdefault(frame, []).append(Person(pid, (x, y, w, h), foot, xy, source))
    return people, sam


def load_person_poses(run: KickMeasureRun) -> dict[tuple[int, int], np.ndarray]:
    """3D poses on the pitch ``{(frame, person id): (n_keypoints, 3) metres}``.

    Signature: ``load_person_poses(run) -> dict``; read on demand from ``poses_3d_on_field.csv``
    of the 3D folder (empty dict when absent). Rows are ordered by ``kpt_idx``.
    """
    path = run.threed / "poses_3d_on_field.csv"
    if not path.is_file():
        return {}
    d = pd.read_csv(path)
    out: dict[tuple[int, int], np.ndarray] = {}
    for key, grp in d.groupby(["frame", "person_id"]):
        if not isinstance(key, tuple):
            continue
        xyz = grp.sort_values("kpt_idx")[["x_m", "y_m", "z_m"]].to_numpy(float)
        out[(int(key[0]), int(key[1]))] = xyz
    return out


def load_kick_measure_run(
    path,
    camera: str = "reference",
    ball_source: str = "auto",
    with_people: bool = True,
    track: bool = True,
    use_saved: bool = True,
) -> KickMeasureRun:
    """Open a ``kick_analysis_*`` folder for measuring.

    Signature: ``load_kick_measure_run(path, camera="reference", ball_source="auto",
    with_people=True, track=True, use_saved=True) -> KickMeasureRun``. ``path`` is the kick folder
    itself, its ``*_3d_*`` folder or the detect run inside ``work/``. ``camera``: ``"reference"``
    or ``"other"``. ``ball_source``: ``"auto"`` (the ball clicked by hand, ``<stem>_markers.csv``
    next to the video, else the detection ``ball.csv``), ``"markers"`` or ``"detect"``.

    Cameras: when ``use_saved`` and the SAM 3 folder holds a ``<stem>_frames.dlt3d`` that is not
    older than the detect run (:func:`write_frame_calibration`), it is read (fast). Otherwise the
    track is built from the DLT3D + raw keypoints and, with ``track`` and a readable video, the
    ``approximate`` frames are tracked from frame to frame (:func:`track_approximate_frames`;
    frames it cannot follow stay ``approximate``).

    Raises ``ValueError`` when ``path`` is not a kick analysis folder, has no detect run, or
    no calibrated frame (``<stem>.dlt3d`` with at least one complete row).
    """
    run = _find_kick_dir(path)
    cam = _kick_camera(run, camera)
    predict = cam.get("predict")
    if predict is None or not Path(predict).is_dir():
        raise ValueError(f"{run.name} has no detect run for {cam['stem']}")
    predict = Path(predict)
    warnings: list[str] = []

    dlt_path = predict / f"{cam['stem']}.dlt3d"
    if not dlt_path.is_file():
        raise ValueError(f"no {dlt_path.name} in {predict}")
    dlt = _read_dlt3d(dlt_path)
    if not dlt:
        raise ValueError(f"{dlt_path.name} has no calibrated frame")
    raw_path = predict / "field_kps_raw.csv"
    if raw_path.is_file():
        kps = _read_raw_keypoints(raw_path)
    else:
        kps = {}
        warnings.append("no field_kps_raw.csv: frames without a DLT3D stay approximate")

    fps, n_video, width, height = _video_info(Path(cam["video"]))
    if fps is None:
        fps = _readme_fps(run)
    if fps is None:
        fps = 30.0
        warnings.append("fps unknown (video not found): 30 assumed")
    seen = [*dlt, *kps]
    n_frames = n_video or (max(seen) + 1)
    n_frames = max(n_frames, max(dlt) + 1)
    if width is None or height is None:
        width, height = 1920, 1080
        warnings.append("video size unknown: 1920x1080 assumed (principal point 960, 540)")
    template = load_field_template(Path(cam["threed"]))
    ball, ball_src = _load_ball(cam, ball_source)
    cameras: CameraTrack | None = None
    saved = (
        find_frame_calibration(run / "work" / str(cam["stem"]), str(cam["stem"]))
        if use_saved
        else None
    )
    if saved is not None:
        newest_input = max(p.stat().st_mtime for p in (dlt_path, raw_path) if p.is_file())
        try:
            if saved.stat().st_mtime >= newest_input:
                cameras = read_frame_calibration(saved)
                if cameras.n_frames != n_frames:
                    warnings.append(
                        f"{saved.name} has {cameras.n_frames} frames, video {n_frames}: rebuilt"
                    )
                    cameras = None
                else:
                    warnings.append(f"cameras read from {saved}")
        except (OSError, ValueError, KeyError) as exc:
            warnings.append(f"cannot read {saved.name} ({exc}): rebuilt")
            cameras = None
    if cameras is None:
        cameras = CameraTrack.from_calibration(
            dlt, kps, template, n_frames, principal_point=(width / 2.0, height / 2.0)
        )
        if track and cameras.approximate_frames():
            if not Path(cam["video"]).is_file():
                warnings.append("video not found: approximate frames cannot be tracked")
            else:
                try:
                    cameras = track_approximate_frames(
                        cameras, cam["video"], boxes=_sam_boxes(run, str(cam["stem"])), ball_uv=ball
                    )
                except Exception as exc:  # noqa: BLE001 - tracking is an improvement, not a need
                    warnings.append(f"tracking of the approximate frames failed ({exc})")
    people: dict[int, list[Person]] = {}
    if with_people:
        people, sam = _load_people(run, cam, cameras)
        if sam is None:
            warnings.append("no sam_tracks.csv: no people")
    return KickMeasureRun(
        run_dir=run,
        camera_name=camera,
        video=Path(cam["video"]),
        stem=str(cam["stem"]),
        threed=Path(cam["threed"]),
        predict=predict,
        fps=float(fps),
        n_frames=int(n_frames),
        width=int(width),
        height=int(height),
        cameras=cameras,
        ball_uv=ball,
        ball_source=ball_src,
        people=people,
        warnings=warnings,
    )


# ----------------------------------------------------------------------------------------------
# People, 2D poses, pitch lines (what the Kick Measure panel draws and lists)
# ----------------------------------------------------------------------------------------------

METRIC_KINDS = ("speed", "distance_ball", "distance_person", "height_kpt", "jump", "kpt_speed")
BODY_POINTS: dict[str, tuple[str, ...]] = {  # name -> MHR70 keypoint names (averaged)
    "pelvis": ("left-hip", "right-hip"),
    "head": ("nose",),
    "neck": ("neck",),
    "left_wrist": ("left-wrist",),
    "right_wrist": ("right-wrist",),
    "left_ankle": ("left-ankle",),
    "right_ankle": ("right-ankle",),
}
_GROUND_KPT = 999  # ``poses_3d_on_field.csv`` row ``ground``: the person's point on the grass
_CENTRE_SPOT = 48  # point of the 49-point pitch template
_CENTRE_CIRCLE_M = 9.15
_CENTRE_CIRCLE_CHORDS = 48
_JUMP_BASELINE_FRAMES = 10
_SKELETONS = Path(__file__).resolve().parent / "skeletons"
_POSE_2D_RE = re.compile(r"_id_(\d+)_markers\.csv$")
_REFEREE_FILE_RE = re.compile(r"^P(\d+)_referee", re.IGNORECASE)
_TEAM_LETTER = {"A": "A", "TEAM_A": "A", "B": "B", "TEAM_B": "B"}

_CSV_CACHE: dict[str, tuple[tuple[int, int], pd.DataFrame]] = {}


def _read_csv_cached(path: Path) -> pd.DataFrame:
    """``pd.read_csv`` memoised on (mtime, size): the panel asks for the same file often."""
    st = path.stat()
    stamp = (st.st_mtime_ns, st.st_size)
    hit = _CSV_CACHE.get(str(path))
    if hit is not None and hit[0] == stamp:
        return hit[1]
    df = pd.read_csv(path)
    if len(_CSV_CACHE) >= 8:
        _CSV_CACHE.pop(next(iter(_CSV_CACHE)))
    _CSV_CACHE[str(path)] = (stamp, df)
    return df


def _safe_csv(path: Path, columns: set[str]) -> pd.DataFrame | None:
    """The CSV at ``path`` when it exists, can be read and has ``columns``; else None."""
    if not path.is_file():
        return None
    try:
        df = _read_csv_cached(path)
    except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
        return None
    return df if columns <= set(df.columns) else None


@dataclass(frozen=True)
class PersonInfo:
    """One SAM person of a run: ``id`` (SAM ``obj_id``, the ``person_id`` of the 3D files and the
    ``NN`` of ``<stem>_id_NN_markers.csv``), ``role`` (``"kicker"``, ``"keeper"``, ``"referee"``
    or ``"other"``), ``team`` (``"A"``, ``"B"`` or ``""``), first/last frame and number of frames
    with a SAM box, and whether a 3D / 2D pose exists."""

    id: int
    role: str
    team: str
    first_frame: int
    last_frame: int
    n_frames: int
    has_pose_3d: bool
    has_pose_2d: bool


def _player_map(run: KickMeasureRun) -> dict[int, dict]:
    """``{sam_id: row}`` of ``player_map.csv`` (empty when absent)."""
    df = _safe_csv(run.threed / "player_map.csv", {"player_id", "sam_id"})
    out: dict[int, dict] = {}
    if df is None:
        return out
    for r in df.to_dict("records"):
        if pd.isna(r["sam_id"]) or pd.isna(r["player_id"]):
            continue
        out[int(r["sam_id"])] = r
    return out


def _referee_ids(run: KickMeasureRun, by_player: dict[int, int]) -> set[int]:
    """SAM ids of referees: ``referees/P<player_id>_referee*.c3d`` files and ``team_map.csv`` rows
    whose role starts with ``referee``. ``P<n>`` is a *player* id of ``player_map.csv`` (checked on
    real data: ``P23_referee_main.c3d`` holds SAM id 25); a number that is no player id is taken
    as a SAM id."""
    ids: set[int] = set()
    folder = run.threed / "referees"
    if folder.is_dir():
        for f in sorted(folder.glob("P*_referee*.c3d")):
            m = _REFEREE_FILE_RE.match(f.name)
            if m:
                n = int(m.group(1))
                ids.add(by_player.get(n, n))
    tm = _safe_csv(run.threed / "data" / "team_map.csv", {"player_id", "role"})
    if tm is not None:
        for r in tm.to_dict("records"):
            if str(r["role"]).lower().startswith("referee") and not pd.isna(r["player_id"]):
                n = int(r["player_id"])
                ids.add(by_player.get(n, n))
    return ids


def _team_by_sam(run: KickMeasureRun, by_player: dict[int, int]) -> dict[int, str]:
    tm = _safe_csv(run.threed / "data" / "team_map.csv", {"player_id", "team"})
    out: dict[int, str] = {}
    if tm is None:
        return out
    for r in tm.to_dict("records"):
        if pd.isna(r["player_id"]) or pd.isna(r["team"]):
            continue
        letter = _TEAM_LETTER.get(str(r["team"]).strip().upper(), "")
        if letter:
            n = int(r["player_id"])
            out[by_player.get(n, n)] = letter
    return out


def _pose_2d_files(run: KickMeasureRun) -> dict[int, Path]:
    """``{id: <stem>_id_NN_markers.csv}`` of the newest SAM3-DINOv3 run inside ``work/<stem>``."""
    base = run.run_dir / "work" / run.stem
    if not base.is_dir():
        return {}
    for folder in sorted(base.glob("**/processed_sam3dinov3_*"), reverse=True):
        found: dict[int, Path] = {}
        for f in sorted((folder / run.stem).glob(f"{run.stem}_id_*_markers.csv")):
            m = _POSE_2D_RE.search(f.name)
            if m:
                found[int(m.group(1))] = f
        if found:
            return found
    return {}


def _pose_3d_table(run: KickMeasureRun) -> pd.DataFrame | None:
    return _safe_csv(
        run.threed / "poses_3d_on_field.csv",
        {"frame", "person_id", "kpt_idx", "kpt_name", "x_m", "y_m", "z_m"},
    )


def _roster_labels(run: KickMeasureRun) -> dict[int, tuple[str, str]]:
    """``{sam_id: (role, team)}`` the user gave in ``roster/<stem>/roster.csv`` of the run.

    Only a roster with at least one team counts, and only the people with a team (or OUT) are in
    the answer; a person left blank is not. ``role`` is ``kicker``, ``keeper``, ``referee`` or
    ``other``; ``team`` is ``"A"``, ``"B"`` or ``""`` (referees and OUT have none). ``{}`` when
    there is no usable roster (never an error: a broken roster file is the pipeline's to report).
    """
    try:
        try:
            from . import freekiki_roster as ro
        except ImportError:
            import freekiki_roster as ro  # ty: ignore[unresolved-import]
        path = ro.find_roster(run.run_dir, run.stem)
        roster = ro.read_roster(path) if path is not None else None
    except (ImportError, OSError, ValueError):  # RosterError is a ValueError
        return {}
    if roster is None or not roster.filled:
        return {}
    out: dict[int, tuple[str, str]] = {}
    for sam, e in roster.entries.items():
        if e.team in ("A", "B"):
            out[sam] = (e.role if e.role in ("kicker", "keeper") else "other", e.team)
        elif e.team == "REF":
            out[sam] = ("referee", "")
        elif e.team == "OUT":
            out[sam] = ("other", "")
    return out


def load_people_info(run: KickMeasureRun) -> dict[int, PersonInfo]:
    """The people of a run, by SAM id: role, team, frames and which poses exist.

    Signature: ``load_people_info(run) -> {id: PersonInfo}``. Frames (first/last/count) are the
    frames with a SAM box in ``run.people`` (the 3D pose frames for an id without a box). Role:
    ``player_map.csv`` (``kicker`` / ``keeper`` / ``other``, looked up by ``sam_id``), ``referee``
    when a ``referees/P<n>_referee*.c3d`` exists or ``data/team_map.csv`` says so; team:
    ``data/team_map.csv`` (``TEAM_A``/``TEAM_B`` -> ``"A"``/``"B"``, joined by ``player_id``).
    Missing files give ``"other"`` / ``""``, never an error.

    The user's roster (``roster/<stem>/roster.csv`` of the run, see :mod:`vaila.freekiki_roster`)
    comes first: when it has at least one team, each person it labels (A, B, REF or OUT) gets the
    team and role of the label, whatever the files above say; a person it leaves blank keeps the
    files' answer.
    """
    frames: dict[int, list[int]] = {}
    for f, ps in run.people.items():
        for p in ps:
            frames.setdefault(p.id, []).append(int(f))
    table = _pose_3d_table(run)
    ids3d: set[int] = set()
    if table is not None:
        ids3d = {int(v) for v in table["person_id"].dropna().unique()}
        for pid in sorted(ids3d - set(frames)):
            sub = table[table["person_id"] == pid]
            frames[pid] = sorted({int(v) for v in sub["frame"].dropna()})
    files2d = _pose_2d_files(run)
    pmap = _player_map(run)
    by_player = {int(r["player_id"]): sam for sam, r in pmap.items()}
    referees = _referee_ids(run, by_player)
    teams = _team_by_sam(run, by_player)
    labels = _roster_labels(run)
    out: dict[int, PersonInfo] = {}
    for pid in sorted(frames):
        fr = sorted(set(frames[pid]))
        role = str(pmap.get(pid, {}).get("role", "other") or "other")
        if role not in ("kicker", "keeper"):
            role = "other"
        if pid in referees:
            role = "referee"
        team = teams.get(pid, "")
        if pid in labels:  # the user's decision
            role, team = labels[pid]
        out[pid] = PersonInfo(
            id=pid,
            role=role,
            team=team,
            first_frame=fr[0],
            last_frame=fr[-1],
            n_frames=len(fr),
            has_pose_3d=pid in ids3d,
            has_pose_2d=pid in files2d,
        )
    return out


def load_pose_2d(run: KickMeasureRun) -> dict[tuple[int, int], np.ndarray]:
    """2D poses of the video ``{(frame, id): (70, 2) px}`` (NaN where a point is missing).

    Signature: ``load_pose_2d(run) -> dict``. Read from ``<stem>_id_NN_markers.csv`` (columns
    ``frame, p0_x, p0_y, ...``) of the newest ``processed_sam3dinov3_*`` folder of
    ``work/<stem>``; ``NN`` is the SAM id. ``{}`` when absent. Frames with no finite point are
    left out.
    """
    out: dict[tuple[int, int], np.ndarray] = {}
    for pid, path in _pose_2d_files(run).items():
        try:
            df = pd.read_csv(path)
        except (OSError, ValueError, pd.errors.ParserError, pd.errors.EmptyDataError):
            continue
        if "frame" not in df.columns:
            continue
        n = 0
        while f"p{n}_x" in df.columns and f"p{n}_y" in df.columns:
            n += 1
        if n == 0:
            continue
        cols = [c for i in range(n) for c in (f"p{i}_x", f"p{i}_y")]
        arr = df[cols].to_numpy(dtype=float).reshape(len(df), n, 2)
        for frame, xy in zip(df["frame"].to_numpy(dtype=int), arr, strict=True):
            if np.isfinite(xy).any():
                full = np.full((max(n, 70), 2), math.nan)
                full[:n] = xy
                out[(int(frame), pid)] = full
    return out


def _skeleton_connections(name: str) -> list[tuple[int, int]]:
    try:
        data = json.loads((_SKELETONS / name).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return []
    out: list[tuple[int, int]] = []
    for a, b in data.get("connections", []):
        try:
            out.append((int(str(a).lstrip("p")), int(str(b).lstrip("p"))))
        except ValueError:
            continue
    return out


def pose_connections() -> list[tuple[int, int]]:
    """Bones ``(i, j)`` of the MHR70 body skeleton (``sam3dinov3_mhr70_body.json``), 48 of them."""
    return _skeleton_connections("sam3dinov3_mhr70_body.json")


def _pitch_lines_m() -> list[tuple[np.ndarray, np.ndarray]]:
    """Pitch segments in metres: the ``soccerfield_kiki49`` lines plus the centre circle (48 chords)."""
    try:
        template = load_field_template()
    except FileNotFoundError:
        return []
    segs = [
        (template[a], template[b])
        for a, b in _skeleton_connections("soccerfield_kiki49.json")
        if a < len(template) and b < len(template)
    ]
    if len(template) > _CENTRE_SPOT:
        c = template[_CENTRE_SPOT]
        ang = np.linspace(0.0, 2.0 * math.pi, _CENTRE_CIRCLE_CHORDS + 1)
        ring = np.c_[
            c[0] + _CENTRE_CIRCLE_M * np.cos(ang),
            c[1] + _CENTRE_CIRCLE_M * np.sin(ang),
            np.full(len(ang), c[2]),
        ]
        segs += [(ring[i], ring[i + 1]) for i in range(_CENTRE_CIRCLE_CHORDS)]
    return segs


def field_segments_px(
    run: KickMeasureRun, frame: int
) -> list[tuple[tuple[float, float], tuple[float, float]]]:
    """Pitch lines of ``frame`` in video pixels, to draw over the video (the calibration check).

    Signature: ``field_segments_px(run, frame) -> [((u0, v0), (u1, v1)), ...]``. The lines are the
    ``soccerfield_kiki49`` connections plus the centre circle (radius 9.15 m, 48 chords),
    projected with the camera of ``frame``. A segment is dropped when an end is behind the camera
    or not finite; one that is partly off-image is kept (clip it when drawing).
    """
    segs = _pitch_lines_m()
    if not segs:
        return []
    P = run.cameras.matrix(frame)
    pts = np.array([p for s in segs for p in s], dtype=float)
    uv = project_points(P, pts)
    sign = 1.0 if np.linalg.det(P[:, :3]) > 0 else -1.0
    w = sign * (P @ np.c_[pts, np.ones(len(pts))].T)[2]
    ok = np.isfinite(uv).all(axis=1) & (w > 0) & (np.abs(uv) < 1e6).all(axis=1)
    out: list[tuple[tuple[float, float], tuple[float, float]]] = []
    for i in range(len(segs)):
        a, b = 2 * i, 2 * i + 1
        if ok[a] and ok[b]:
            out.append(((float(uv[a, 0]), float(uv[a, 1])), (float(uv[b, 0]), float(uv[b, 1]))))
    return out


# ----------------------------------------------------------------------------------------------
# Metrics per frame on a person (speed, distances, heights, jump)
# ----------------------------------------------------------------------------------------------


@dataclass
class MetricSeries:
    """A metric over the frames of the video.

    ``kind`` one of :data:`METRIC_KINDS`; ``person`` the SAM id; ``other`` the second person of
    ``distance_person``; ``point`` the body point of ``height_kpt`` / ``kpt_speed`` /
    ``jump``; ``label`` a readable name; ``unit`` ``"m"`` or ``"m/s"``; ``values`` ``(n_frames,)``
    with NaN where the metric cannot be computed.
    """

    kind: str
    person: int
    other: int | None
    point: str | None
    label: str
    unit: str
    values: np.ndarray
    source: str = ""

    def at(self, frame: int) -> float:
        """Value in ``frame`` (NaN outside the video or where missing)."""
        f = int(frame)
        if 0 <= f < len(self.values):
            return float(self.values[f])
        return math.nan

    def summary(self) -> dict:
        """``{"max", "min", "mean", "frame_of_max", "n_valid", "unit", "label", "source"}`` (NaN / None when
        no frame is valid). For ``jump`` the max is the jump height."""
        v = np.asarray(self.values, dtype=float)
        ok = np.isfinite(v)
        n = int(ok.sum())
        if n == 0:
            return {"max": math.nan, "min": math.nan, "mean": math.nan, "frame_of_max": None,
                    "n_valid": 0, "unit": self.unit, "label": self.label,
                    "source": self.source}  # fmt: skip
        return {
            "max": float(v[ok].max()),
            "min": float(v[ok].min()),
            "mean": float(v[ok].mean()),
            "frame_of_max": int(np.nanargmax(np.where(ok, v, -np.inf))),
            "n_valid": n,
            "unit": self.unit,
            "label": self.label,
            "source": self.source,
        }


GROUND_SOURCES = ("ground_tracks", "pose_ground", "ankles", "box_foot")
# The bottom of a SAM box jitters by tens of pixels (about 1 m of grass): a speed read on it needs a
# wider Savitzky-Golay window than the 5 frames that suit a ground track (real data: spikes of 20-78
# m/s with 5, at most 9 m/s with 9).
_BOX_FOOT_MIN_WINDOW = 9


def _ground_candidates(run: KickMeasureRun, pid: int) -> dict[str, np.ndarray]:
    """Every ground-position series of a person ``{source: (n_frames, 2)}``, in
    :data:`GROUND_SOURCES` order; a source with no valid frame is left out."""
    n = int(run.n_frames)
    found: dict[str, np.ndarray] = {}

    def keep(name: str, frames, xs, ys) -> None:
        out = np.full((n, 2), math.nan)
        for f, x, y in zip(frames, xs, ys, strict=True):
            if 0 <= f < n and np.isfinite(x) and np.isfinite(y):
                out[f] = (x, y)
        if np.isfinite(out).all(axis=1).any():
            found[name] = out

    gt = _safe_csv(run.threed / "ground_tracks.csv", {"frame", "person_id", "x_m", "y_m"})
    if gt is not None:
        sub = gt[gt["person_id"] == pid]
        keep(
            "ground_tracks",
            sub["frame"].to_numpy(int),
            sub["x_m"].to_numpy(float),
            sub["y_m"].to_numpy(float),
        )
    table = _pose_3d_table(run)
    if table is not None:
        mine = table[table["person_id"] == pid]
        g = mine[mine["kpt_idx"] == _GROUND_KPT]
        keep(
            "pose_ground",
            g["frame"].to_numpy(int),
            g["x_m"].to_numpy(float),
            g["y_m"].to_numpy(float),
        )
    try:
        ankles = np.stack(
            [body_point_series(run, pid, "left_ankle"), body_point_series(run, pid, "right_ankle")]
        )[:, :, :2]
    except ValueError:
        ankles = None
    if ankles is not None and np.isfinite(ankles).any():
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)  # a frame with no ankle at all
            mean = np.asarray(np.nanmean(ankles, axis=0), dtype=float)
        keep("ankles", np.arange(n), mean[:, 0], mean[:, 1])
    frames_b: list[int] = []
    xs_b: list[float] = []
    ys_b: list[float] = []
    for f, ps in run.people.items():
        for p in ps:
            if p.id == pid and not run.cameras.is_approximate(f):
                g3 = run.cameras.ground_point(f, p.foot_uv[0], p.foot_uv[1], 0.0)
                frames_b.append(int(f))
                xs_b.append(float(g3[0]))
                ys_b.append(float(g3[1]))
    keep("box_foot", frames_b, xs_b, ys_b)
    return found


def person_ground_source(run: KickMeasureRun, pid: int) -> tuple[np.ndarray, str]:
    """``(series, source)`` of :func:`person_ground_series`; ``source`` is ``""`` when no source
    has a valid frame (the series is then all NaN)."""
    cands = _ground_candidates(run, pid)
    if not cands:
        return np.full((int(run.n_frames), 2), math.nan), ""
    best = max(
        cands,
        key=lambda k: (int(np.isfinite(cands[k]).all(axis=1).sum()), -GROUND_SOURCES.index(k)),
    )
    return cands[best], best


def person_ground_series(run: KickMeasureRun, pid: int) -> np.ndarray:
    """Position of the person on the grass ``(n_frames, 2)`` metres, NaN where unknown.

    Signature: ``person_ground_series(run, pid) -> (n_frames, 2)``. One source per series (never
    mixed frame by frame): the one with the most valid frames among ``ground_tracks.csv``
    (FreeKiki, ``x_m, y_m``), the ``ground`` point of the 3D pose, the mean of the two ankles and
    the foot of the SAM box seen by the camera of each frame (``box_foot``, only on frames whose
    camera is calibrated or solved: an ``approximate`` camera is a copy of a neighbour that does
    not follow the pan, and it moved the foot by up to 9 m in one frame); a tie keeps the earlier
    one in that order. :func:`person_ground_source` also returns which one was used.
    """
    return person_ground_source(run, pid)[0]


def ball_ground_series(run: KickMeasureRun) -> np.ndarray:
    """Ball position on the pitch ``(n_frames, 2)`` metres from the on-field rows of
    ``ball_3d.csv``; NaN elsewhere (all NaN when the file is absent)."""
    n = int(run.n_frames)
    out = np.full((n, 2), math.nan)
    b = _safe_csv(run.threed / "ball_3d.csv", {"frame", "x_m", "y_m", "on_field"})
    if b is None:
        return out
    on = b["on_field"].astype(str).str.strip().str.lower().isin(("true", "1", "1.0"))
    for f, x, y in zip(
        b["frame"].to_numpy(int)[on.to_numpy()],
        b["x_m"].to_numpy(float)[on.to_numpy()],
        b["y_m"].to_numpy(float)[on.to_numpy()],
        strict=True,
    ):
        if 0 <= f < n and np.isfinite(x) and np.isfinite(y):
            out[f] = (x, y)
    return out


def _person_pose_array(run: KickMeasureRun, pid: int) -> tuple[np.ndarray, list[str]] | None:
    """``((n_frames, K, 3) metres NaN-filled, [keypoint names])`` of one person or None."""
    table = _pose_3d_table(run)
    if table is None:
        return None
    mine = table[(table["person_id"] == pid) & (table["kpt_idx"] != _GROUND_KPT)]
    if mine.empty:
        return None
    all_pts = table[table["kpt_idx"] != _GROUND_KPT]
    k = int(all_pts["kpt_idx"].max()) + 1
    names = [""] * k
    for idx, name in zip(all_pts["kpt_idx"], all_pts["kpt_name"], strict=True):
        if 0 <= int(idx) < k and not names[int(idx)]:
            names[int(idx)] = str(name)
    names = [nm or f"kpt{i}" for i, nm in enumerate(names)]
    n = int(run.n_frames)
    arr = np.full((n, k, 3), math.nan)
    fr = mine["frame"].to_numpy(int)
    ix = mine["kpt_idx"].to_numpy(int)
    inside = (fr >= 0) & (fr < n) & (ix >= 0) & (ix < k)
    arr[fr[inside], ix[inside]] = mine[["x_m", "y_m", "z_m"]].to_numpy(float)[inside]
    return arr, names


def body_point_series(run: KickMeasureRun, pid: int, point: str) -> np.ndarray:
    """3D position of a body point ``(n_frames, 3)`` metres (NaN where missing).

    Signature: ``body_point_series(run, pid, point) -> (n_frames, 3)``. ``point`` is a key of
    :data:`BODY_POINTS` (mean of its keypoints) or any keypoint name of the file (``left-knee``).
    Raises ``ValueError`` (readable) when the person has no 3D pose or the point is unknown.
    """
    got = _person_pose_array(run, pid)
    if got is None:
        raise ValueError(f"person {pid} has no 3D pose (poses_3d_on_field.csv)")
    arr, names = got
    wanted = BODY_POINTS.get(point, (point,))
    cols = [names.index(w) for w in wanted if w in names]
    if len(cols) != len(wanted):
        known = ", ".join(sorted(BODY_POINTS))
        raise ValueError(f"unknown body point {point!r}; use one of: {known}")
    sel = arr[:, cols, :]
    ok = np.asarray(np.isfinite(sel).all(axis=2))  # (n, c)
    cnt = ok.sum(axis=1)
    total = np.where(ok[..., None], sel, 0.0).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = total / cnt[:, None]
    out[cnt == 0] = math.nan
    return out


def _point_label(point: str) -> str:
    return point.replace("_", " ")


def metric_series(
    run: KickMeasureRun,
    kind: str,
    person: int,
    *,
    other: int | None = None,
    point: str | None = "pelvis",
    smooth_window: int = 5,
) -> MetricSeries:
    """One metric of one person over the video.

    Signature: ``metric_series(run, kind, person, *, other=None, point="pelvis",
    smooth_window=5) -> MetricSeries``. ``kind``:

    * ``speed`` - speed on the ground, ``|d foot / dt|`` in m/s (see :func:`person_ground_series`;
      Savitzky-Golay over ``smooth_window`` frames, see :func:`velocity_series`; at least 9 frames
      when the positions come from the SAM box foot, which is noisier than a ground track);
    * ``distance_ball`` - distance on the ground between the person and the ball, m;
    * ``distance_person`` - distance on the ground to the person ``other``, m;
    * ``height_kpt`` - height ``z`` of ``point``, m;
    * ``jump`` - height of the pelvis above its standing level (median of its first 10 valid
      frames), m; the summary max is the jump height;
    * ``kpt_speed`` - 3D speed of ``point``, m/s.

    Raises ``ValueError`` with a readable message when data is missing (it is shown in the GUI).
    """
    if kind not in METRIC_KINDS:
        raise ValueError(f"unknown metric {kind!r}; use one of: {', '.join(METRIC_KINDS)}")
    person = int(person)
    n = int(run.n_frames)
    if kind in ("height_kpt", "kpt_speed") and not point:
        raise ValueError(f"{kind} needs a body point ({', '.join(BODY_POINTS)})")
    other_id: int | None = None
    pt: str | None = None
    source = "pose_3d"  # the ground-based kinds below replace it
    if kind == "speed":
        xy, source = person_ground_source(run, person)
        if not source:
            raise ValueError(f"person {person} has no position on the pitch (ground track / pose)")
        win = (
            max(smooth_window, _BOX_FOOT_MIN_WINDOW)
            if smooth_window and source == "box_foot"
            else smooth_window
        )
        vals = speed_series(xy, run.fps, win)
        label, unit = f"speed P{person}", "m/s"
    elif kind == "distance_ball":
        xy, source = person_ground_source(run, person)
        if not source:
            raise ValueError(f"person {person} has no position on the pitch (ground track / pose)")
        ball = ball_ground_series(run)
        if not np.isfinite(ball).any():
            raise ValueError("this run has no ball on the pitch (ball_3d.csv on-field rows)")
        vals = np.linalg.norm(xy - ball, axis=1)
        label, unit = f"distance ball P{person}", "m"
    elif kind == "distance_person":
        if other is None:
            raise ValueError("distance between people needs a second person")
        other_id = int(other)
        if other_id == person:
            raise ValueError("choose two different people for the distance")
        xy, src1 = person_ground_source(run, person)
        xy2, src2 = person_ground_source(run, other_id)
        for pid, src in ((person, src1), (other_id, src2)):
            if not src:
                raise ValueError(f"person {pid} has no position on the pitch (ground track / pose)")
        source = src1 if src1 == src2 else f"{src1}+{src2}"
        vals = np.linalg.norm(xy - xy2, axis=1)
        label, unit = f"distance P{person}-P{other_id}", "m"
    elif kind == "height_kpt":
        pt = str(point)
        vals = body_point_series(run, person, pt)[:, 2]
        label, unit = f"height {_point_label(pt)} P{person}", "m"
    elif kind == "jump":
        pt = "pelvis"
        z = body_point_series(run, person, pt)[:, 2]
        valid = z[np.isfinite(z)]
        if len(valid) < 2:
            raise ValueError(f"person {person} has too few frames with a pelvis to measure a jump")
        vals = z - float(np.median(valid[:_JUMP_BASELINE_FRAMES]))
        label, unit = f"jump P{person}", "m"
    else:  # kpt_speed
        pt = str(point)
        xyz = body_point_series(run, person, pt)
        vals = speed_series(xyz, run.fps, smooth_window)
        label, unit = f"speed {_point_label(pt)} P{person}", "m/s"
    vals = np.asarray(vals, dtype=float)
    if vals.shape[0] != n:
        vals = np.resize(vals, n)
    if not np.isfinite(vals).any():
        raise ValueError(f"{label}: not enough valid frames to measure")
    return MetricSeries(kind, person, other_id, pt, label, unit, vals, source)


# ----------------------------------------------------------------------------------------------
# Export of the scene: C3D for mkvis3d, BVH + script for Blender, metrics table
# ----------------------------------------------------------------------------------------------


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def find_mkvis3d() -> Path | None:
    """The ``mkvis3d.py`` viewer: ``$VAILA_MKVIS3D`` when it is a file, else
    ``<repo root>/../mkvis3d/mkvis3d.py``; None when neither exists."""
    env = os.environ.get("VAILA_MKVIS3D", "").strip()
    if env:
        p = Path(env).expanduser()
        if p.is_file():
            return p
    cand = _repo_root().parent / "mkvis3d" / "mkvis3d.py"
    return cand if cand.is_file() else None


def mkvis3d_argv(c3d) -> list[str] | None:
    """Command line that opens ``c3d`` in mkvis3d: ``[python, mkvis3d.py, c3d]`` or None."""
    script = find_mkvis3d()
    if script is None:
        return None
    return [sys.executable, str(script), str(c3d)]


_BLENDER_SCRIPT = """# Auto-generated by vaila kick_measure.export_scene.
# Open Blender and run this file (Text Editor > Run Script), or:
#   blender --python kick_scene_blender.py
# Pitch metres, Z up, x along the pitch, origin at the centre spot.
import json
import math
import os

import bpy

HERE = os.path.dirname(os.path.abspath(__file__))
FPS = @FPS@
N_FRAMES = @N_FRAMES@
BALL_RADIUS = @BALL_RADIUS@
PEOPLE = @PEOPLE@  # [{"id": 21, "role": "kicker", "team": "", "bvh": "person_21.bvh"}, ...]
ROLE_COLOURS = {
    "kicker": (1.0, 0.45, 0.05, 1.0),
    "keeper": (0.1, 0.8, 0.2, 1.0),
    "referee": (0.9, 0.9, 0.1, 1.0),
    "other": (0.2, 0.55, 1.0, 1.0),
}


def _material(name, rgba):
    mat = bpy.data.materials.get(name) or bpy.data.materials.new(name)
    mat.diffuse_color = rgba
    return mat


def _clear():
    for obj in list(bpy.data.objects):
        bpy.data.objects.remove(obj, do_unlink=True)


def _import_people():
    sphere = None
    for person in PEOPLE:
        path = os.path.join(HERE, person["bvh"])
        if not os.path.isfile(path):
            print("missing BVH", path)
            continue
        before = set(bpy.data.objects.keys())
        bpy.ops.import_anim.bvh(
            filepath=path,
            axis_forward="-Z",
            axis_up="Y",
            global_scale=1.0,
            frame_start=1,
            update_scene_fps=False,
            update_scene_duration=False,
        )
        new = [bpy.data.objects[n] for n in bpy.data.objects.keys() if n not in before]
        prefix = "P%d" % person["id"]
        arm = None
        for obj in new:
            obj.name = "%s_%s" % (prefix, obj.name)
            if obj.type == "ARMATURE" and arm is None:
                arm = obj
        if arm is None:
            continue
        if sphere is None:
            bpy.ops.mesh.primitive_uv_sphere_add(radius=0.04, segments=8, ring_count=6)
            sphere = bpy.context.active_object
            sphere.name = "joint_template"
            sphere.hide_viewport = True
            sphere.hide_render = True
        mat = _material("role_" + person["role"], ROLE_COLOURS.get(person["role"], ROLE_COLOURS["other"]))
        for bone in arm.pose.bones:  # one small sphere follows each joint
            joint = bpy.data.objects.new("%s_%s_joint" % (prefix, bone.name), sphere.data)
            joint.data.materials.clear()
            joint.data.materials.append(mat)
            bpy.context.collection.objects.link(joint)
            con = joint.constraints.new("COPY_LOCATION")
            con.target = arm
            con.subtarget = bone.name


def _pitch_lines():
    path = os.path.join(HERE, "pitch_lines.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        segments = json.load(fh)["segments"]
    curve = bpy.data.curves.new("pitch_lines", type="CURVE")
    curve.dimensions = "3D"
    curve.bevel_depth = 0.04
    for a, b in segments:
        spline = curve.splines.new("POLY")
        spline.points.add(1)
        spline.points[0].co = (a[0], a[1], a[2] + 0.01, 1.0)
        spline.points[1].co = (b[0], b[1], b[2] + 0.01, 1.0)
    obj = bpy.data.objects.new("pitch_lines", curve)
    obj.data.materials.append(_material("pitch_line_white", (1.0, 1.0, 1.0, 1.0)))
    bpy.context.collection.objects.link(obj)


def _ball():
    path = os.path.join(HERE, "ball_path.json")
    if not os.path.isfile(path):
        return
    with open(path, encoding="utf-8") as fh:
        data = json.load(fh)
    bpy.ops.mesh.primitive_uv_sphere_add(radius=BALL_RADIUS, location=(0.0, 0.0, 0.0))
    ball = bpy.context.active_object
    ball.name = "ball"
    ball.data.materials.append(_material("ball_mat", (1.0, 0.95, 0.25, 1.0)))
    for frame, p in zip(data["frames"], data["xyz"]):
        if p is None or not all(math.isfinite(c) for c in p):
            continue
        ball.location = (p[0], p[1], p[2])
        ball.keyframe_insert(data_path="location", frame=int(frame) + 1)


def _world():
    sun = bpy.data.lights.new("sun", type="SUN")
    sun.energy = 3.0
    sun_obj = bpy.data.objects.new("sun", sun)
    sun_obj.rotation_euler = (math.radians(45), math.radians(-15), math.radians(30))
    bpy.context.collection.objects.link(sun_obj)


def build():
    _clear()
    _import_people()
    _pitch_lines()
    _ball()
    _world()
    scene = bpy.context.scene
    # The BVH import may change the scene rate: set the video rate and range LAST.
    fps_i = max(1, int(round(FPS)))
    scene.render.fps = fps_i
    scene.render.fps_base = fps_i / float(FPS)
    scene.frame_start = 1
    scene.frame_end = max(1, N_FRAMES)
    scene.frame_set(1)
    print("kick_scene: ready,", N_FRAMES, "frames at", FPS, "fps,", len(PEOPLE), "people")


if __name__ == "__main__":
    build()
"""


def _blender_script_text(fps: float, n_frames: int, people: list[dict]) -> str:
    """Text of the Blender companion script (plain ``bpy`` code; vaila never imports ``bpy``)."""
    return (
        _BLENDER_SCRIPT.replace("@FPS@", repr(float(fps)))
        .replace("@N_FRAMES@", repr(int(n_frames)))
        .replace("@BALL_RADIUS@", repr(BALL_RADIUS_M))
        .replace("@PEOPLE@", repr(people))
    )


def _clean(value: float):
    return None if not math.isfinite(value) else float(value)


def _finite_json(obj):
    """Replace NaN / inf by None recursively, for a strict JSON file."""
    if isinstance(obj, float):
        return _clean(obj)
    if isinstance(obj, dict):
        return {k: _finite_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_finite_json(v) for v in obj]
    return obj


def _ball_xyz_array(run: KickMeasureRun, ball_xyz) -> np.ndarray:
    """``(n_frames, 3)`` ball path: ``ball_xyz`` ``{frame: (x, y, z)}`` or the on-field rows of
    ``ball_3d.csv``; NaN elsewhere."""
    n = int(run.n_frames)
    out = np.full((n, 3), math.nan)
    if ball_xyz:
        for f, p in ball_xyz.items():
            if 0 <= int(f) < n:
                out[int(f)] = np.asarray(p, dtype=float)[:3]
        return out
    b = _safe_csv(run.threed / "ball_3d.csv", {"frame", "x_m", "y_m", "z_m", "on_field"})
    if b is None:
        return out
    on = b["on_field"].astype(str).str.strip().str.lower().isin(("true", "1", "1.0")).to_numpy()
    for f, x, y, z in zip(
        b["frame"].to_numpy(int)[on],
        b["x_m"].to_numpy(float)[on],
        b["y_m"].to_numpy(float)[on],
        b["z_m"].to_numpy(float)[on],
        strict=True,
    ):
        if 0 <= f < n:
            out[f] = (x, y, z)
    return out


def _export_modules():
    """``(write_c3d, write_c3d_csv, write_bvh)`` of the FreeKiki exporters (lazy imports)."""
    try:
        from .freekiki_blender import write_bvh
        from .freekiki_export import write_c3d, write_c3d_csv
    except ImportError:
        from freekiki_blender import write_bvh  # ty: ignore[unresolved-import]
        from freekiki_export import (  # ty: ignore[unresolved-import]
            write_c3d,
            write_c3d_csv,
        )
    return write_c3d, write_c3d_csv, write_bvh


def _resolve_metrics(run: KickMeasureRun, metrics) -> tuple[list[MetricSeries], list[str]]:
    """Metric series from ``MetricSeries`` or request objects (``kind``, ``person``, ``other``,
    ``point``); a request that cannot be computed is reported in the second list, not raised."""
    series: list[MetricSeries] = []
    errors: list[str] = []
    for m in metrics:
        if isinstance(m, MetricSeries):
            series.append(m)
            continue
        try:
            series.append(
                metric_series(
                    run,
                    m.kind,
                    m.person,
                    other=getattr(m, "other", None),
                    point=getattr(m, "point", None) or "pelvis",
                )
            )
        except ValueError as exc:
            errors.append(f"{getattr(m, 'kind', '?')} P{getattr(m, 'person', '?')}: {exc}")
    return series, errors


def export_scene(
    run: KickMeasureRun,
    out_dir,
    people,
    *,
    metrics=(),
    ball_xyz=None,
    mkvis3d: bool = True,
    blender: bool = True,
) -> dict[str, Path]:
    """Write the 3D scene of the chosen people to ``out_dir`` (created; the run is never touched).

    Signature: ``export_scene(run, out_dir, people, *, metrics=(), ball_xyz=None, mkvis3d=True,
    blender=True) -> {key: Path}``. ``people`` SAM ids; ``metrics`` :class:`MetricSeries` or request
    objects with ``kind``, ``person``, ``other``, ``point``; ``ball_xyz`` ``{frame: (x, y, z)}``
    overrides the ball of ``ball_3d.csv`` (on-field rows). Files and keys:

    * ``mkvis3d``: ``c3d`` = ``kick_scene.c3d`` and ``c3d_csv`` = ``kick_scene_c3d.csv`` with the
      markers ``P<id>_<keypoint>`` (3D poses), ``BALL`` and ``FKP_<name>`` (the pitch template,
      static); open with ``python mkvis3d.py kick_scene.c3d``;
    * ``blender``: ``bvh_<id>`` = ``blender/person_<id>.bvh`` (a person with no 3D pose is
      skipped), ``ball_json`` = ``blender/ball_path.json``, ``pitch_json`` =
      ``blender/pitch_lines.json``, ``blender_script`` = ``blender/kick_scene_blender.py``
      (run it inside Blender: it imports the BVHs, keys a ball sphere, draws the pitch lines and
      sets the scene rate and range last);
    * ``metrics`` given: ``metrics_csv`` (``frame, time_s`` and one column per metric) and
      ``metrics_json`` (the summary of each, plus the ones that failed under ``errors``);
    * always ``readme`` = ``README.txt`` with the commands to open the files.

    Raises ``ValueError`` when nothing can be exported (no 3D pose among the people and no ball).
    """
    write_c3d, write_c3d_csv, write_bvh = _export_modules()
    out = Path(out_dir).expanduser()
    out.mkdir(parents=True, exist_ok=True)
    info = load_people_info(run)
    fps, n = float(run.fps), int(run.n_frames)
    written: dict[str, Path] = {}

    chosen: list[int] = []
    for pid in people:
        if int(pid) not in chosen:
            chosen.append(int(pid))
    poses: dict[int, tuple[np.ndarray, list[str]]] = {}
    for pid in chosen:
        got = _person_pose_array(run, pid)
        if got is not None:
            poses[pid] = got
    ball = _ball_xyz_array(run, ball_xyz)
    have_ball = bool(np.isfinite(ball).any())
    if not poses and not have_ball:
        raise ValueError("no 3D pose among the chosen people and no ball track: nothing to export")

    if mkvis3d:
        labels: list[str] = []
        cols: list[np.ndarray] = []
        for pid, (arr, names) in poses.items():
            for j, name in enumerate(names):
                labels.append(f"P{pid}_{name}")
                cols.append(arr[:, j, :])
        if have_ball:
            labels.append("BALL")
            cols.append(ball)
        try:
            template = load_field_template(run.threed)
            tmpl_names = (
                pd.read_csv(Path(__file__).resolve().parent / "models" / "soccerfield_kiki.csv")
                .sort_values("point_number")["point_name"]
                .tolist()
            )
        except (FileNotFoundError, OSError, KeyError):
            template, tmpl_names = np.zeros((0, 3)), []
        for i, name in enumerate(tmpl_names[: len(template)]):
            labels.append(f"FKP_{name}")
            cols.append(np.tile(template[i], (n, 1)))
        points = np.stack(cols, axis=1)  # (frames, markers, 3)
        written["c3d"] = write_c3d(out / "kick_scene.c3d", labels, points, rate_hz=fps)
        written["c3d_csv"] = write_c3d_csv(out / "kick_scene_c3d.csv", labels, points, rate_hz=fps)

    if blender:
        bdir = out / "blender"
        bdir.mkdir(parents=True, exist_ok=True)
        entries: list[dict] = []
        for pid, (arr, names) in poses.items():
            keep = [j for j in range(arr.shape[1]) if np.isfinite(arr[:, j, :]).any()]
            if not keep:
                continue
            bvh = write_bvh(
                bdir / f"person_{pid}.bvh", [names[j] for j in keep], arr[:, keep, :], fps=fps
            )
            written[f"bvh_{pid}"] = bvh
            pi = info.get(pid)
            entries.append(
                {
                    "id": pid,
                    "role": pi.role if pi else "other",
                    "team": pi.team if pi else "",
                    "bvh": bvh.name,
                }
            )
        ball_json = bdir / "ball_path.json"
        ball_json.write_text(
            json.dumps(
                {
                    "fps": fps,
                    "n_frames": n,
                    "frames": list(range(n)),
                    "xyz": [
                        None if not np.isfinite(p).all() else [float(c) for c in p] for p in ball
                    ],
                },
                indent=1,
            ),
            encoding="utf-8",
        )
        written["ball_json"] = ball_json
        pitch_json = bdir / "pitch_lines.json"
        pitch_json.write_text(
            json.dumps(
                {"segments": [[a.tolist(), b.tolist()] for a, b in _pitch_lines_m()]}, indent=1
            ),
            encoding="utf-8",
        )
        written["pitch_json"] = pitch_json
        script = bdir / "kick_scene_blender.py"
        script.write_text(_blender_script_text(fps, n, entries), encoding="utf-8")
        written["blender_script"] = script

    series, errors = _resolve_metrics(run, metrics)
    if series or errors:
        table = {"frame": np.arange(n), "time_s": np.arange(n) / fps}
        names_used: set[str] = set()
        for s in series:
            col = f"{s.label} ({s.unit})"
            k = 2
            while col in names_used:
                col = f"{s.label} #{k} ({s.unit})"
                k += 1
            names_used.add(col)
            table[col] = np.asarray(s.values, dtype=float)
        csv_path = out / "metrics.csv"
        pd.DataFrame(table).to_csv(csv_path, index=False, float_format="%.6f")
        written["metrics_csv"] = csv_path
        json_path = out / "metrics_summary.json"
        json_path.write_text(
            json.dumps(
                _finite_json(
                    {
                        "run": str(run.run_dir),
                        "fps": fps,
                        "metrics": [
                            {
                                "kind": s.kind,
                                "person": s.person,
                                "other": s.other,
                                "point": s.point,
                                **s.summary(),
                            }  # fmt: skip
                            for s in series
                        ],
                        "errors": errors,
                    }
                ),
                indent=1,
            ),
            encoding="utf-8",
        )
        written["metrics_json"] = json_path

    readme = out / "README.txt"
    readme.write_text(_scene_readme(run, written, chosen, poses, info, errors), encoding="utf-8")
    written["readme"] = readme
    return written


def _scene_readme(run, written, chosen, poses, info, errors) -> str:
    lines = [
        "vaila Kick Measure scene",
        f"run   : {run.run_dir}",
        f"video : {run.video.name}  ({run.fps:g} fps, {run.n_frames} frames)",
        "axes  : pitch metres, x along the pitch, y across, z up, origin at the centre spot",
        "people: "
        + (
            ", ".join(
                f"P{p} ({info[p].role}{', team ' + info[p].team if p in info and info[p].team else ''})"
                if p in info
                else f"P{p}"
                for p in poses
            )
            or "none"
        ),
    ]
    skipped = [p for p in chosen if p not in poses]
    if skipped:
        lines.append("no 3D pose, skipped: " + ", ".join(f"P{p}" for p in skipped))
    if "c3d" in written:
        viewer = find_mkvis3d()
        lines += [
            "",
            "mkvis3d (3D viewer):",
            f"  python {viewer if viewer else 'mkvis3d.py'} {written['c3d'].name}",
            f"  python {viewer if viewer else 'mkvis3d.py'} info {written['c3d'].name}",
        ]
    if "blender_script" in written:
        lines += [
            "",
            "Blender (BVH per person, ball, pitch lines):",
            "  blender --python blender/kick_scene_blender.py",
            "  or open Blender, Text Editor > Open the script > Run Script",
            "  (a BVH has no gaps: a joint that is missing in a frame keeps its nearest valid position)",
        ]
    if "metrics_csv" in written:
        lines += ["", "metrics: metrics.csv (per frame) and metrics_summary.json"]
    for e in errors:
        lines.append(f"metric skipped: {e}")
    return "\n".join(lines) + "\n"


# ----------------------------------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vaila.kick_measure",
        description="Open a FreeKiki kick_analysis folder and report/export its per-frame "
        "cameras, ball track and people (what getpixelvideo --kick-run measures with).",
    )
    p.add_argument("--run", required=True, help="kick_analysis_* folder (or its 3D / detect run)")
    p.add_argument("--camera", choices=("reference", "other"), default="reference")
    p.add_argument("--info", action="store_true", help="print cameras, people and ball (default)")
    p.add_argument("--export", metavar="JSON", help="write cameras, ball and people counts here")
    p.add_argument(
        "--export-scene",
        metavar="OUT",
        help="write the 3D scene (C3D for mkvis3d, BVH + script for Blender, metrics) to OUT",
    )
    p.add_argument(
        "--people",
        metavar="IDS",
        help="SAM ids to export, e.g. 21,23 (default: every person with a 3D pose)",
    )
    p.add_argument(
        "--metric",
        metavar="KIND:PERSON[:OTHER_OR_POINT]",
        action="append",
        default=[],
        help=f"metric to compute and export (repeatable); KIND one of {', '.join(METRIC_KINDS)}; "
        "third part: the other person (distance_person) or the body point (height_kpt, kpt_speed)",
    )
    p.add_argument(
        "--write-cameras",
        action="store_true",
        help="write <stem>_frames.dlt3d/.dlt2d/_cameras.csv/.json (one row per video frame) "
        "into the SAM 3 output folder of the run",
    )
    p.add_argument(
        "--cameras-dir",
        metavar="DIR",
        help="with --write-cameras: write here instead of the SAM 3 folder",
    )
    p.add_argument(
        "--no-track",
        action="store_true",
        help="do not track the approximate frames (and ignore saved per-frame cameras)",
    )
    p.add_argument(
        "--validate-tracking",
        metavar="K,K,...",
        nargs="?",
        const="5,10,20,30",
        help="hold-out test of the tracker (frames k away from a known camera); default 5,10,20,30",
    )
    p.add_argument("--no-blender", action="store_true", help="with --export-scene: no BVH/script")
    p.add_argument("--no-mkvis3d", action="store_true", help="with --export-scene: no C3D/CSV")
    return p


def _parse_metric_spec(spec: str) -> tuple[str, int, int | None, str | None]:
    """``"kind:person[:other_or_point]"`` -> ``(kind, person, other, point)``; ValueError."""
    parts = [x.strip() for x in spec.split(":")]
    if len(parts) < 2 or len(parts) > 3 or not parts[0]:
        raise ValueError(f"bad --metric {spec!r}: use KIND:PERSON[:OTHER_OR_POINT]")
    kind = parts[0]
    try:
        person = int(parts[1])
    except ValueError:
        raise ValueError(f"bad --metric {spec!r}: PERSON must be a number") from None
    third = parts[2] if len(parts) == 3 and parts[2] else None
    other: int | None = None
    point: str | None = None
    if kind == "distance_person":
        if third is None:
            raise ValueError(f"bad --metric {spec!r}: distance_person needs the other person")
        try:
            other = int(third)
        except ValueError:
            raise ValueError(f"bad --metric {spec!r}: the other person must be a number") from None
    elif third is not None:
        point = third
    return kind, person, other, point


def _ranges(frames: list[int]) -> str:
    """``[3, 4, 5, 9]`` -> ``"3-5, 9"``."""
    out: list[str] = []
    start = prev = None
    for f in [*frames, None]:
        if start is None:
            start = prev = f
        elif f is not None and prev is not None and f == prev + 1:
            prev = f
        else:
            out.append(str(start) if start == prev else f"{start}-{prev}")
            start = prev = f
    return ", ".join(out)


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    try:
        run = load_kick_measure_run(
            args.run, camera=args.camera, track=not args.no_track, use_saved=not args.no_track
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    s = run.summary()
    action = args.export or args.export_scene or args.write_cameras or args.validate_tracking
    if args.info or not action:
        c = s["cameras"]
        print(f"run      : {s['run']}")
        print(f"video    : {s['video']}  ({s['size_px'][0]}x{s['size_px'][1]}, "
              f"{s['fps']:g} fps, {s['n_frames']} frames)")  # fmt: skip
        print(
            f"cameras  : {c.get(SOURCE_DLT3D, 0)} DLT3D (kept), {c.get(SOURCE_SOLVED, 0)} solved "
            f"(fixed centre), {c.get(SOURCE_TRACKED, 0)} tracked (frame to frame), "
            f"{c.get(SOURCE_APPROX, 0)} approximate"
        )
        if s["centre_m"] is not None:
            cx, cy, cz = s["centre_m"]
            print(
                f"centre   : ({cx:.2f}, {cy:.2f}, {cz:.2f}) m (median of the DLT3D frames; each solved frame uses its 10 nearest)"
            )
        if s["tracked_frames"]:
            print(f"tracked  : frames {_ranges(s['tracked_frames'])}")
        if s["approximate_frames"]:
            print(f"approx   : frames {_ranges(s['approximate_frames'])}")
        print(f"ball     : {s['ball_frames']} frames ({s['ball_source'] or 'none'})")
        print(f"people   : {s['people_ids']} ids, {s['people_boxes']} boxes")
        for w in s["warnings"]:
            print(f"warning  : {w}")
    if args.validate_tracking:
        code = _cli_validate_tracking(run, args.validate_tracking)
        if code:
            return code
    if args.write_cameras:
        try:
            written = write_run_frame_calibration(run, args.cameras_dir)
        except (ValueError, OSError) as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        counts = run.cameras.counts()
        for key, path in written.items():
            print(f"wrote    : {key}: {path}")
        print(
            f"cameras  : {run.n_frames} rows; "
            + ", ".join(f"{n} {s}" for s, n in sorted(counts.items()))
        )
    if args.export:
        out = Path(args.export).expanduser()
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(run.to_dict(), indent=1), encoding="utf-8")
        print(f"wrote {out}")
    if args.export_scene:
        return _cli_export_scene(run, args)
    return 0


def _cli_validate_tracking(run: KickMeasureRun, spec: str) -> int:
    try:
        ks = tuple(int(x) for x in spec.replace(" ", "").split(",") if x)
    except ValueError:
        print(
            f"error: --validate-tracking must be numbers separated by commas, got {spec!r}",
            file=sys.stderr,
        )
        return 2
    if not run.video.is_file():
        print(f"error: video {run.video} not found", file=sys.stderr)
        return 2
    raw = run.predict / "field_kps_raw.csv"
    if not raw.is_file():
        print(f"error: no {raw.name} in {run.predict}", file=sys.stderr)
        return 2
    boxes = {f: [p.bbox for p in ps] for f, ps in run.people.items()}
    feats = track_features(run.video, [(0, run.n_frames - 1)], boxes=boxes, ball_uv=run.ball_uv)
    result = validate_tracking(
        run.cameras, feats, _read_raw_keypoints(raw), load_field_template(run.threed), ks=ks, step=2
    )
    print(format_validation(result))
    return 0


def _cli_export_scene(run: KickMeasureRun, args) -> int:
    try:
        if args.people:
            people = [int(x) for x in args.people.replace(" ", "").split(",") if x]
        else:
            people = [i for i, pi in load_people_info(run).items() if pi.has_pose_3d]
    except ValueError:
        print(f"error: --people must be numbers separated by commas, got {args.people!r}",
              file=sys.stderr)  # fmt: skip
        return 2
    series: list[MetricSeries] = []
    try:
        for spec in args.metric:
            kind, person, other, point = _parse_metric_spec(spec)
            series.append(metric_series(run, kind, person, other=other, point=point or "pelvis"))
        written = export_scene(
            run,
            args.export_scene,
            people,
            metrics=series,
            mkvis3d=not args.no_mkvis3d,
            blender=not args.no_blender,
        )
    except (ValueError, OSError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2
    for s in series:
        sm = s.summary()
        print(f"metric   : {s.label}: max {sm['max']:.3f} {s.unit} (frame {sm['frame_of_max']}), "
              f"mean {sm['mean']:.3f}, {sm['n_valid']} valid frames")  # fmt: skip
    for key, path in written.items():
        print(f"wrote    : {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
