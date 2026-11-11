"""
================================================================================
Script: yolo_depth3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.4.15
Created: 10 November 2026
Update Date: 11 November 2026

================================================================================
Description
================================================================================

Monocular 3D from ONE video with the YOLO26 depth task: every 2D keypoint gets
the metric depth the network estimates at its pixel and is back-projected to 3D.

Three levels, each usable alone, each improving the previous one:

1. **No calibration.** 2D keypoints (YOLO26-pose, or any vailá pixel CSV given
   with ``--pixels``) + the YOLO26 depth map -> 3D in the CAMERA frame (OpenCV:
   +X right, +Y down, +Z forward, metres). The focal length is ``--focal-px``,
   ``--fov-deg`` or, by default, a guess (the image diagonal), and the outputs
   say which one was used.
2. **With a calibration.** ``--dlt3d`` puts every keypoint on its exact pixel
   ray of the DLT camera, at the camera depth of the map, so the result is in
   the LAB frame. ``--intrinsics-toml`` (Chessboard Calib) gives the real lens
   and removes the distortion of the pixels. ``--scale`` then corrects the
   network's absolute scale (4 to 46 % off, per the Ultralytics page):
   ``control`` compares the map with the true depth of the calibration control
   points; ``floor`` uses the foot standing on the plane Z = 0.
3. **Whole-trial fit** (``--fit``, on by default with a DLT3D). Every point
   stays on its ray and only its distance along the ray is solved, for the
   whole trial at once: the segments keep ONE length, the standing foot rests
   on the floor, the trunk stays above the feet and the depth is smooth in
   time. The depth map says which end of a segment is nearer (and, with a
   MediaPipe ``*_mp_norm.csv``, so does the pose model's own z); the segment
   length says by how much. ``--segment-lengths`` gives the lengths of the
   same person measured elsewhere (a two-camera reconstruction).

What the depth is, and is not
-----------------------------
The map is the depth of the VISIBLE SURFACE, in metres along the optical axis.
A joint centre is a few centimetres behind the skin, and an occluded keypoint
gets the depth of whatever is in front of it. Monocular metric depth is an
estimate, not a measurement: the quality numbers in ``yolodepth3d_summary.json``
(segment-length variation, foot height, reprojection in a second camera, error
against a reference reconstruction) are there to judge each run.

================================================================================
Usage
================================================================================

    uv run --no-sync python -m vaila.yolo_depth3d            # window
    uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4
    uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4 \\
        --pixels video_pixel_vaila.csv --dlt3d cam.dlt3d
    uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4 \\
        --pixels video_pixel_vaila.csv --dlt3d cam.dlt3d \\
        --segment-lengths rec3d_two_cameras.csv

Outputs go to a new ``yolodepth3d_<timestamp>/`` folder beside the video.

================================================================================
License
================================================================================

GNU Affero General Public License v3.0 - https://www.gnu.org/licenses/agpl-3.0.html
"""

from __future__ import annotations

import argparse
import datetime
import json
import shlex
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

try:
    from .monocular_dlt_align import (
        decompose_dlt3d,
        dlt_project,
        dlt_projection_matrix,
        load_dlt3d_file,
    )
except ImportError:  # standalone: python vaila/yolo_depth3d.py
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        decompose_dlt3d,
        dlt_project,
        dlt_projection_matrix,
        load_dlt3d_file,
    )

VERSION = "0.4.15"
DEPTH_MODELS = (
    "yolo26n-depth.pt",
    "yolo26s-depth.pt",
    "yolo26m-depth.pt",
    "yolo26l-depth.pt",
    "yolo26x-depth.pt",
)
DEFAULT_DEPTH_MODEL = "yolo26x-depth.pt"
DEFAULT_JOINT_DEPTH = 0.25  # how much of the depth difference between joints is kept
DEFAULT_POSE_MODEL = "yolo26m-pose.pt"
DEFAULT_SEG_MODEL = "yolo26n-seg.pt"
DEPTH_IMGSZ = 768  # the size the released depth weights were trained at
SCALE_MODES = ("none", "control", "floor", "floor-frame")
CROP_MARGIN = 0.25  # the person crop is its keypoint box plus this much on every side
NEAR_PERCENTILE = 20.0  # the window's foreground is found from its nearest fifth
NEAR_TOLERANCE = 0.06  # ... plus 6 % of that depth (about one limb thickness at 3 m)
MIN_LOGAFFINE_POINTS = 6  # below this, two parameters are not identifiable: one scale
VIDEO_SUFFIXES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")


# ---------------------------------------------------------------------------
# Marker layouts
# ---------------------------------------------------------------------------


ANKLE_HEIGHT_M = 0.12  # MediaPipe ankle of a shod standing foot above the floor
HEEL_HEIGHT_M = 0.08  # MediaPipe heel (back of the shoe)
# The YOLO26-pose (COCO) ankle is lower than MediaPipe's: on the two-camera test clip it sat 10
# to 15 px below it, 2 to 3 cm at that distance (malleolus about 7 cm + the sole).
COCO_ANKLE_HEIGHT_M = 0.10


@dataclass(frozen=True)
class Layout:
    """The skeleton of a marker set: segments, foot markers and their height when standing."""

    name: str
    bones: list[tuple[int, int]] = field(default_factory=list)
    feet: tuple[tuple[int, ...], ...] = ()  # one group of markers per foot
    foot_heights: dict[int, float] = field(default_factory=dict)  # standing height above Z = 0
    trunk: tuple[int, ...] = ()  # markers whose median depth is "the person's depth"
    # for the whole-trial fit: ``bones`` plus the trunk diagonals and the rigid head/hands/feet
    fit_bones: list[tuple[int, int]] = field(default_factory=list)
    mirror: dict[int, int] = field(default_factory=dict)  # left marker <-> right marker
    # (trunk markers, foot markers): the first group stays above the second one
    balance: tuple[tuple[int, ...], tuple[int, ...]] = ((), ())

    @property
    def foot_markers(self) -> list[int]:
        return [m for foot in self.feet for m in foot]


def _both_ways(pairs: dict[int, int]) -> dict[int, int]:
    return {**pairs, **{b: a for a, b in pairs.items()}}


_COCO_BONES = [(5, 6), (5, 7), (7, 9), (6, 8), (8, 10), (5, 11), (6, 12), (11, 12)] + [
    (11, 13),
    (13, 15),
    (12, 14),
    (14, 16),
]
_MP_BONES = (
    [(11, 12), (11, 13), (13, 15), (12, 14), (14, 16), (11, 23), (12, 24), (23, 24)]
    + [(23, 25), (25, 27), (24, 26), (26, 28), (27, 29), (28, 30), (29, 31), (30, 32)]
    + [(27, 31), (28, 32)]
)
_MP_HEAD = (
    [(0, 7), (0, 8), (7, 8), (0, 2), (0, 5), (2, 5), (2, 7), (5, 8), (0, 9), (0, 10), (9, 10)]
    + [(1, 2), (2, 3), (4, 5), (5, 6), (0, 1), (0, 4), (3, 7), (6, 8), (7, 9), (8, 10)]
    + [(7, 11), (8, 12), (0, 11), (0, 12)]
)
_MP_HANDS = [(15, 17), (15, 19), (15, 21), (17, 19), (16, 18), (16, 20), (16, 22), (18, 20)]

COCO17 = Layout(
    "coco17",
    _COCO_BONES,
    feet=((15,), (16,)),  # the ankles
    foot_heights={15: COCO_ANKLE_HEIGHT_M, 16: COCO_ANKLE_HEIGHT_M},
    trunk=(5, 6, 11, 12, 13, 14, 15, 16),
    fit_bones=_COCO_BONES
    + [(5, 12), (6, 11)]
    + [
        (0, 1),
        (0, 2),
        (1, 2),
        (1, 3),
        (2, 4),
        (3, 4),
        (0, 3),
        (0, 4),
        (3, 5),
        (4, 6),
        (0, 5),
        (0, 6),
    ],
    mirror=_both_ways({1: 2, 3: 4, 5: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16}),
    balance=((5, 6, 11, 12), (15, 16)),
)
MEDIAPIPE33 = Layout(
    "mediapipe33",
    _MP_BONES,
    # ankle and heel of each foot. The toes are left out: the depth read at a toe mixes with the
    # floor in front of it (measured: their scale comes out 5 to 8 % lower than the body's).
    feet=((27, 29), (28, 30)),
    foot_heights={27: ANKLE_HEIGHT_M, 28: ANKLE_HEIGHT_M, 29: HEEL_HEIGHT_M, 30: HEEL_HEIGHT_M},
    trunk=(11, 12, 23, 24, 25, 26, 27, 28),
    fit_bones=_MP_BONES + [(11, 24), (12, 23)] + _MP_HEAD + _MP_HANDS,
    mirror=_both_ways(
        {1: 4, 2: 5, 3: 6, 7: 8, 9: 10, 11: 12, 13: 14, 15: 16, 17: 18, 19: 20, 21: 22}
        | {23: 24, 25: 26, 27: 28, 29: 30, 31: 32}
    ),
    balance=((11, 12, 23, 24), (27, 28, 29, 30, 31, 32)),
)


def layout_for(n_markers: int) -> Layout:
    """The layout with ``n_markers`` points; an unknown count has no skeleton and still lifts."""
    return {17: COCO17, 33: MEDIAPIPE33}.get(int(n_markers), Layout(f"markers{int(n_markers)}"))


# ---------------------------------------------------------------------------
# Intrinsics and back-projection
# ---------------------------------------------------------------------------


def intrinsics_from_focal(focal_px: float, width: int, height: int) -> np.ndarray:
    """A pinhole K with square pixels and the principal point at the image centre."""
    return np.array(
        [[focal_px, 0.0, width / 2.0], [0.0, focal_px, height / 2.0], [0.0, 0.0, 1.0]], np.float64
    )


def intrinsics_from_fov(fov_deg: float, width: int, height: int) -> np.ndarray:
    """K from the HORIZONTAL field of view, in degrees."""
    focal = (width / 2.0) / np.tan(np.radians(fov_deg) / 2.0)
    return intrinsics_from_focal(float(focal), width, height)


def default_focal_px(width: int, height: int) -> float:
    """The guess used with no calibration: the image diagonal (a 53 degree diagonal view)."""
    return float(np.hypot(width, height))


def backproject_pinhole(uv: np.ndarray, z: np.ndarray, K: np.ndarray) -> np.ndarray:
    """Pixels (N, 2) and camera depths (N,) to camera-frame points (N, 3), OpenCV axes."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    z = np.asarray(z, np.float64).reshape(-1)
    hom = np.column_stack([uv, np.ones(len(uv))])
    rays = np.linalg.solve(np.asarray(K, np.float64), hom.T).T  # z component is 1
    return rays * z[:, None]


def undistort_pixels(uv: np.ndarray, K: np.ndarray, dist: np.ndarray) -> np.ndarray:
    """Remove the lens distortion of pixel coordinates (NaN rows stay NaN)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    out = np.full_like(uv, np.nan)
    ok = np.isfinite(uv).all(axis=1)
    if ok.any():
        pts = cv2.undistortPoints(uv[ok].reshape(-1, 1, 2), K, np.asarray(dist, np.float64), P=K)
        out[ok] = pts.reshape(-1, 2)
    return out


# ---------------------------------------------------------------------------
# DLT3D rays
# ---------------------------------------------------------------------------


def _coefficients(camera: Any) -> np.ndarray:
    """The 11 DLT coefficients of ``camera`` (a DltCamera or the coefficients themselves)."""
    return np.asarray(getattr(camera, "L", camera), np.float64).reshape(11)


def dlt_rays(L: Any, uv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The camera centre and, per pixel, the world displacement for ONE metre of camera depth.

    A world point on the ray of pixel ``uv`` at camera depth ``z`` is ``centre + z * step``.
    With P = [M | p4], ``M (X - C) = w [u, v, 1]`` and the camera depth is
    ``sign(det M) * w / |m3|``, so no intrinsic matrix is needed (nor its skew).
    """
    P = dlt_projection_matrix(_coefficients(L))
    M = P[:, :3]
    centre = -np.linalg.solve(M, P[:, 3])
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    hom = np.column_stack([uv, np.ones(len(uv))])
    direction = np.linalg.solve(M, hom.T).T  # world displacement per unit of w
    w_per_metre = np.sign(np.linalg.det(M)) * np.linalg.norm(M[2])
    return centre, direction * w_per_metre


def dlt_point_at_depth(L: Any, uv: np.ndarray, z: np.ndarray) -> np.ndarray:
    """World points (N, 3) on the DLT rays of ``uv`` at camera depths ``z`` (metres)."""
    centre, step = dlt_rays(L, uv)
    return centre + step * np.asarray(z, np.float64).reshape(-1, 1)


def camera_depth(camera: Any, xyz: np.ndarray) -> np.ndarray:
    """Depth along the optical axis (metres) of world points seen by a DLT camera."""
    P = dlt_projection_matrix(_coefficients(camera))
    M = P[:, :3]
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    w = xyz @ M[2] + P[2, 3]
    return np.sign(np.linalg.det(M)) * w / np.linalg.norm(M[2])


def world_to_camera(L: Any, xyz: np.ndarray) -> np.ndarray:
    """World points to the OpenCV camera frame of the decomposed DLT camera."""
    cam = decompose_dlt3d(_coefficients(L))
    xyz = np.asarray(xyz, np.float64).reshape(-1, 3)
    return (cam.R @ (xyz - cam.C).T).T


# ---------------------------------------------------------------------------
# Depth at a keypoint
# ---------------------------------------------------------------------------


def sample_depth(
    depth: np.ndarray,
    uv: np.ndarray,
    radius: int = 4,
    mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth of the foreground of a small window around each pixel.

    A plain window median answers the background behind a thin limb. Here the window's
    foreground is the pixels within :data:`NEAR_TOLERANCE` of its nearest fifth, and the
    answer is their median. ``mask`` (bool, same shape as ``depth``) keeps only the person.

    Returns ``(z, n)``: the depth (NaN when the pixel is outside the image, not finite or the
    window has no valid depth) and the number of pixels behind it.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    h, w = depth.shape[:2]
    z = np.full(len(uv), np.nan)
    n = np.zeros(len(uv), np.int64)
    radius = max(int(radius), 0)
    for i, (u, v) in enumerate(uv):
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        cu, cv = int(round(u)), int(round(v))
        if not (0 <= cu < w and 0 <= cv < h):
            continue
        r0, r1 = max(cv - radius, 0), min(cv + radius + 1, h)
        c0, c1 = max(cu - radius, 0), min(cu + radius + 1, w)
        patch = depth[r0:r1, c0:c1]
        ok = np.isfinite(patch) & (patch > 0)
        if mask is not None:
            ok &= mask[r0:r1, c0:c1]
        values = patch[ok]
        if values.size == 0:
            continue
        near = np.percentile(values, NEAR_PERCENTILE)
        kept = values[values <= near * (1.0 + NEAR_TOLERANCE)]
        z[i] = float(np.median(kept))
        n[i] = int(kept.size)
    return z, n


def window_radius(uv: np.ndarray, fraction: float = 0.012, lo: int = 2, hi: int = 12) -> int:
    """Window radius in pixels from the size of the person (the extent of its keypoints)."""
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    ok = np.isfinite(uv).all(axis=1)
    if ok.sum() < 2:
        return lo
    extent = float(np.max(np.ptp(uv[ok], axis=0)))
    return int(np.clip(round(fraction * extent), lo, hi))


# ---------------------------------------------------------------------------
# Scale of the depth
# ---------------------------------------------------------------------------


def fit_depth_scale(
    z_pred: np.ndarray, z_true: np.ndarray, mode: str = "scale"
) -> tuple[float, float]:
    """Fit ``log z_true = a * log z_pred + b``. ``mode='scale'`` keeps ``a = 1``.

    The scale is the median of the log ratio (robust to a marker read on the background).
    With fewer than :data:`MIN_LOGAFFINE_POINTS` valid points the log-affine fit falls back
    to one scale.
    """
    z_pred = np.asarray(z_pred, np.float64).reshape(-1)
    z_true = np.asarray(z_true, np.float64).reshape(-1)
    ok = np.isfinite(z_pred) & np.isfinite(z_true) & (z_pred > 0) & (z_true > 0)
    if not ok.any():
        raise ValueError("no valid depth pair to fit the scale")
    lp, lt = np.log(z_pred[ok]), np.log(z_true[ok])
    if mode == "logaffine" and ok.sum() >= MIN_LOGAFFINE_POINTS and np.ptp(lp) > 1e-6:
        a, b = np.polyfit(lp, lt, 1)
        return float(a), float(b)
    return 1.0, float(np.median(lt - lp))


def apply_depth_scale(z: np.ndarray, a: float, b: float) -> np.ndarray:
    """``exp(a * log z + b)`` (NaN and non-positive depths stay NaN)."""
    z = np.asarray(z, np.float64)
    out = np.full(z.shape, np.nan)
    ok = np.isfinite(z) & (z > 0)
    out[ok] = np.exp(a * np.log(z[ok]) + b)
    return out


def floor_depth_ratio(
    L: Any, uv: np.ndarray, z_pred: np.ndarray, foot_height_m: float = 0.0
) -> np.ndarray:
    """True/predicted depth of foot markers assumed at ``foot_height_m`` above the plane Z = 0.

    The ray of a foot ON the floor meets that plane at the foot's true depth. A lifted foot's
    ray meets it further away, so over the feet of a frame the smallest ratio is the scale.
    """
    centre, step = dlt_rays(L, uv)
    z_pred = np.asarray(z_pred, np.float64).reshape(-1)
    with np.errstate(divide="ignore", invalid="ignore"):
        z_floor = (foot_height_m - centre[2]) / step[:, 2]
        ratio = z_floor / z_pred
    ratio[~(np.isfinite(ratio) & (z_floor > 0) & (z_pred > 0))] = np.nan
    return ratio


def floor_scale_of_frame(
    L: Any,
    uv: np.ndarray,
    z_pred: np.ndarray,
    layout: Layout,
    foot_height_m: float | None = None,
    height_offset_m: float = 0.0,
) -> float:
    """Depth scale of one frame from the standing foot.

    Each foot marker gives a ratio at its own standing height (``foot_height_m`` replaces them
    all). A foot's ratio is the median of its markers; the frame's scale is the SMALLER of the
    feet, because a lifted foot's ray meets the floor further away. NaN when no foot has depth.
    """
    per_foot = []
    for foot in layout.feet:
        ratios = []
        for m in foot:
            height = foot_height_m if foot_height_m is not None else layout.foot_heights.get(m, 0.0)
            ratios.append(floor_depth_ratio(L, uv[[m]], z_pred[[m]], height + height_offset_m)[0])
        values = np.array(ratios)
        if np.isfinite(values).any():
            per_foot.append(float(np.nanmedian(values)))
    return min(per_foot) if per_foot else float("nan")


def shrink_to_person(z: np.ndarray, trunk: tuple[int, ...], keep: float) -> np.ndarray:
    """Pull every marker's depth towards the person's depth of its frame.

    ``z_out = person + keep * (z - person)``, where ``person`` is the median depth of the
    ``trunk`` markers (all markers when ``trunk`` is empty). ``keep = 1`` leaves the depth read
    at each joint, ``keep = 0`` puts the whole person at one depth. The map is good at WHERE the
    person is and poor at the depth between joints: on the two-camera test clip the best value
    was 0.11 to 0.26, and the raw joint depth (1.0) was worse than one depth for all (0.0).
    """
    import warnings

    z = np.asarray(z, np.float64)
    cols = list(trunk) if trunk else list(range(z.shape[1]))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        person = np.nanmedian(z[:, cols], axis=1, keepdims=True)
        fallback = np.nanmedian(z, axis=1, keepdims=True)
    person = np.where(np.isfinite(person), person, fallback)
    return person + float(keep) * (z - person)


# ---------------------------------------------------------------------------
# Segment lengths
# ---------------------------------------------------------------------------


def _moving_median(values: np.ndarray, window: int) -> np.ndarray:
    """NaN-aware moving median along axis 0."""
    values = np.asarray(values, np.float64)
    if window <= 1 or len(values) < 2:
        return values.copy()
    half = window // 2
    out = np.full_like(values, np.nan)
    for i in range(len(values)):
        chunk = values[max(i - half, 0) : i + half + 1]
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                out[i] = np.nanmedian(chunk, axis=0)
    return out


def segment_lengths(xyz: np.ndarray, bones: list[tuple[int, int]]) -> np.ndarray:
    """Length of each segment per frame, shape (frames, len(bones))."""
    xyz = np.asarray(xyz, np.float64)
    if not bones:
        return np.zeros((len(xyz), 0))
    i, j = np.array(bones).T
    return np.linalg.norm(xyz[:, i] - xyz[:, j], axis=2)


def segment_variation(xyz: np.ndarray, bones: list[tuple[int, int]]) -> dict[str, float]:
    """Coefficient of variation (std / mean) of each segment's length over the trial."""
    lengths = segment_lengths(xyz, bones)
    out: dict[str, float] = {}
    for k, (i, j) in enumerate(bones):
        col = lengths[:, k]
        col = col[np.isfinite(col)]
        if col.size >= 2 and col.mean() > 0:
            out[f"{i}-{j}"] = float(col.std() / col.mean())
    return out


def bone_fix_rays(
    origin: np.ndarray,
    steps: np.ndarray,
    z: np.ndarray,
    bones: list[tuple[int, int]],
    weight: float = 4.0,
    lengths: np.ndarray | None = None,
    median_window: int = 9,
) -> np.ndarray:
    """Adjust each point's depth along its own ray so the segments keep their length.

    ``X = origin + z * step`` (``steps`` shape (frames, markers, 3)). Per frame, minimise
    ``sum (z - z0)^2 + weight^2 * sum (|X_i - X_j| - length_ij)^2`` over the depths: the
    points never leave their rays. The target lengths are the trial median of the lengths
    computed after a moving median of the depths, because frame-to-frame depth noise only
    ever makes a segment look LONGER, and a target that is too long forces a false depth
    difference on a segment seen side-on.
    """
    from scipy.optimize import least_squares

    z = np.asarray(z, np.float64)
    steps = np.asarray(steps, np.float64)
    origin = np.asarray(origin, np.float64).reshape(3)
    out = z.copy()
    if not bones:
        return out
    bi, bj = np.array(bones).T
    if lengths is None:
        calm = _moving_median(z, median_window)
        calm_xyz = origin + steps * calm[..., None]
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lengths = np.nanmedian(segment_lengths(calm_xyz, bones), axis=0)
    lengths = np.asarray(lengths, np.float64)

    for f in range(len(z)):
        z0 = z[f]
        ok_pt = np.isfinite(z0) & np.isfinite(steps[f]).all(axis=1)
        use = ok_pt[bi] & ok_pt[bj] & np.isfinite(lengths) & (lengths > 0)
        if not use.any():
            continue
        idx = np.flatnonzero(ok_pt)
        pos = {int(m): k for k, m in enumerate(idx)}
        pi = np.array([pos[int(m)] for m in bi[use]])
        pj = np.array([pos[int(m)] for m in bj[use]])
        st = steps[f][idx]
        target = lengths[use]
        start = z0[idx]

        def residual(zz, st=st, pi=pi, pj=pj, target=target, start=start):
            pts = st * zz[:, None]
            seg = np.linalg.norm(pts[pi] - pts[pj], axis=1)
            return np.concatenate([zz - start, weight * (seg - target)])

        sol = least_squares(residual, start, method="trf")
        out[f, idx] = sol.x
    return out


def bone_fix_depths(
    L: Any,
    uv: np.ndarray,
    z: np.ndarray,
    bones: list[tuple[int, int]],
    weight: float = 4.0,
    lengths: np.ndarray | None = None,
) -> np.ndarray:
    """:func:`bone_fix_rays` on the rays of a DLT3D camera. ``uv`` is (frames, markers, 2)."""
    uv = np.asarray(uv, np.float64)
    centre, _ = dlt_rays(L, uv[0])
    steps = np.stack([dlt_rays(L, frame_uv)[1] for frame_uv in uv])
    return bone_fix_rays(centre, steps, z, bones, weight=weight, lengths=lengths)


# ---------------------------------------------------------------------------
# Whole-trial fit along the rays
# ---------------------------------------------------------------------------

# Tolerances of the fit, in metres (each term is divided by its tolerance, so they say how much
# every piece of evidence is trusted). They are physical guesses, not values tuned on a clip.
FIT_BONE_STAGES = (0.05, 0.02, 0.01)  # a segment length, tightened stage by stage
FIT_CUE_SIGMA = 0.05  # depth-map difference along a segment ...
FIT_CUE_SOFT = 0.10  # ... beyond this it counts less and less (wall behind a limb, blur)
FIT_CUE_GATE = 0.60  # ... and beyond this it is not a body part at all
FIT_ORDER_SIGMA = 0.03  # being on the wrong side of the order the pose model gives
FIT_ORDER_FULL = 0.10  # a pose-model difference this large (m) is a confident order
FIT_FLOOR_SIGMA = 0.02  # a standing foot marker above its standing height
FIT_FLOOR_SOFT = 0.05  # ... a lifted foot is allowed
FIT_FLOOR_UNDER = 3.0  # a marker UNDER its standing height costs this many times more
FIT_BALANCE_SIGMA = 0.05  # horizontal distance between the trunk and the feet
FIT_BALANCE_SOFT = 0.10  # ... a lean or a step is allowed
FIT_ACCELERATION = 9.0  # m/s^2 along the line of sight: how fast a depth may change its speed
FIT_INIT_KEEP = 0.5  # part of the map's joint depth kept in the starting guess


def _soft(r: np.ndarray, c: float) -> np.ndarray:
    """A residual whose SQUARE is the Cauchy loss of ``r`` (scale ``c``): a large r counts little."""
    return np.sign(r) * c * np.sqrt(np.log1p((r / c) ** 2))


def _dsoft(r: np.ndarray, c: float) -> np.ndarray:
    """Derivative of :func:`_soft` with respect to ``r``."""
    q = (r / c) ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        d = (np.abs(r) / c) / ((1.0 + q) * np.sqrt(np.log1p(q)))
    return np.where(np.abs(r) < 1e-9 * c, 1.0, d)


def _fit_terms(
    z: np.ndarray,
    origin: np.ndarray,
    steps: np.ndarray,
    weight: np.ndarray,
    bones: np.ndarray,
    lengths: np.ndarray,
    bone_sigma: float | None,
    cue: np.ndarray | None,
    order: np.ndarray | None,
    feet: tuple[np.ndarray, np.ndarray],
    balance: tuple[np.ndarray, np.ndarray] | None,
    acc_sigma: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Residuals of :func:`fit_depths` and their sparse Jacobian (rows, columns, values)."""
    n_frames, n_markers = z.shape
    idx = np.arange(n_frames * n_markers).reshape(n_frames, n_markers)
    P = origin + steps * z[:, :, None]
    res: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    cols: list[np.ndarray] = []
    vals: list[np.ndarray] = []
    count = 0

    def add(r: np.ndarray, entries: list[tuple[np.ndarray, np.ndarray]]) -> None:
        nonlocal count
        r = np.asarray(r, np.float64).ravel()
        for columns, values in entries:
            rows.append(count + np.arange(r.size))
            cols.append(np.asarray(columns).ravel())
            vals.append(np.asarray(values, np.float64).ravel())
        res.append(r)
        count += r.size

    bi, bj = bones[:, 0], bones[:, 1]
    both = weight[:, bi] * weight[:, bj]
    dz = z[:, bj] - z[:, bi]
    if bone_sigma is not None:  # every segment keeps its length
        d = P[:, bj] - P[:, bi]
        norm = np.linalg.norm(d, axis=2)
        unit = d / np.maximum(norm, 1e-9)[:, :, None]
        k = both / bone_sigma
        gi = -np.einsum("fbk,fbk->fb", unit, steps[:, bi])
        gj = np.einsum("fbk,fbk->fb", unit, steps[:, bj])
        add(k * (norm - lengths), [(idx[:, bi], k * gi), (idx[:, bj], k * gj)])
    if cue is not None:  # the depth-map difference along the segment, robust
        ok = np.isfinite(cue) * both
        r = (dz - np.nan_to_num(cue)) / FIT_CUE_SIGMA
        c = FIT_CUE_SOFT / FIT_CUE_SIGMA
        g = _dsoft(r, c) / FIT_CUE_SIGMA * ok
        add(_soft(r, c) * ok, [(idx[:, bi], -g), (idx[:, bj], g)])
    if order is not None:  # which end is nearer: only the wrong side costs
        sign = np.sign(np.nan_to_num(order))
        k = np.minimum(1.0, np.abs(np.nan_to_num(order)) / FIT_ORDER_FULL) * both / FIT_ORDER_SIGMA
        wrong = -sign * dz
        on = (wrong > 0) & (k > 0)
        add(
            np.where(on, wrong * k, 0.0),
            [(idx[:, bi], np.where(on, sign * k, 0.0)), (idx[:, bj], np.where(on, -sign * k, 0.0))],
        )
    markers, heights = feet
    if len(markers):  # a foot is never under its standing height and is softly held on it
        r = (P[:, markers, 2] - heights) / FIT_FLOOR_SIGMA
        under = r < 0
        c = FIT_FLOOR_SOFT / FIT_FLOOR_SIGMA
        w = weight[:, markers]
        value = np.where(under, FIT_FLOOR_UNDER * r, _soft(r, c)) * w
        slope = np.where(under, FIT_FLOOR_UNDER, _dsoft(r, c)) / FIT_FLOOR_SIGMA * w
        add(value, [(idx[:, markers], slope * steps[:, markers, 2])])
    if balance is not None:  # the trunk stays above the feet (horizontal offset, robust)
        up, down = balance
        n_up, n_down = weight[:, up].sum(axis=1), weight[:, down].sum(axis=1)
        ok = (n_up > 0) & (n_down > 0)
        w_up = weight[:, up] / np.maximum(n_up, 1.0)[:, None]
        w_down = weight[:, down] / np.maximum(n_down, 1.0)[:, None]
        c = FIT_BALANCE_SOFT / FIT_BALANCE_SIGMA
        for axis in (0, 1):
            off = (P[:, up, axis] * w_up).sum(axis=1) - (P[:, down, axis] * w_down).sum(axis=1)
            r = off / FIT_BALANCE_SIGMA
            g = _dsoft(r, c) / FIT_BALANCE_SIGMA * ok
            add(
                _soft(r, c) * ok,
                [(idx[:, m], g * w_up[:, k] * steps[:, m, axis]) for k, m in enumerate(up)]
                + [(idx[:, m], -g * w_down[:, k] * steps[:, m, axis]) for k, m in enumerate(down)],
            )
    if n_frames > 2:  # the depth changes smoothly in time
        k = np.full((n_frames - 2, n_markers), 1.0 / acc_sigma)
        add(
            (z[:-2] - 2.0 * z[1:-1] + z[2:]) / acc_sigma,
            [(idx[:-2], k), (idx[1:-1], -2.0 * k), (idx[2:], k)],
        )
    return np.concatenate(res), np.concatenate(rows), np.concatenate(cols), np.concatenate(vals)


def fit_depths(
    origin: np.ndarray,
    steps: np.ndarray,
    z0: np.ndarray,
    bones: list[tuple[int, int]],
    lengths: np.ndarray | None = None,
    bone_sigma: float | None = None,
    cue: np.ndarray | None = None,
    order: np.ndarray | None = None,
    feet: dict[int, float] | None = None,
    balance: tuple[tuple[int, ...], tuple[int, ...]] | None = None,
    fps: float = 30.0,
    max_nfev: int = 100,
) -> np.ndarray:
    """Camera depths (frames, markers) of a whole trial, each point staying on its own ray.

    One least-squares problem over every frame at once. A point is ``origin + z * steps``
    (``origin`` (3,) or (frames, 3); ``steps`` (frames, markers, 3), metres per metre of camera
    depth). The terms, all optional but the last:

    - ``lengths`` + ``bone_sigma``: every segment of ``bones`` keeps its length;
    - ``cue`` (frames, bones): the depth-map difference ``z_end - z_start`` along each segment,
      with a loss that stops counting the gross ones;
    - ``order`` (frames, bones): a signed hint of which end is nearer (the pose model's own z).
      Only its SIGN is used: being on the wrong side costs, the amount comes from the lengths;
    - ``feet`` ``{marker: standing height}``: a foot is never under the floor plane Z = 0 plus
      its height, and is softly held on it (a lifted foot is allowed);
    - ``balance`` ``(trunk markers, foot markers)``: the trunk stays above the feet;
    - the depth of each marker changes smoothly in time.

    A marker with no ray in a frame (NaN in ``steps``) takes no part and comes back NaN; a NaN in
    ``z0`` only means "no starting guess".
    """
    from scipy.optimize import least_squares
    from scipy.sparse import coo_matrix

    steps = np.asarray(steps, np.float64)
    n_frames, n_markers = steps.shape[:2]
    origin = np.asarray(origin, np.float64)
    origin = np.broadcast_to(origin.reshape(-1, 1, 3) if origin.ndim == 2 else origin, steps.shape)
    valid = np.asarray(np.isfinite(steps).all(axis=2))
    weight = np.asarray(valid, np.float64)
    steps = np.where(valid[:, :, None], steps, 0.0)
    start = _fill_depths(np.where(valid, np.asarray(z0, np.float64), np.nan))
    pairs = np.asarray(bones, int).reshape(-1, 2)
    if lengths is None:
        bone_sigma = None
        lengths = np.zeros(len(pairs))
    foot_markers = np.array(sorted(feet or {}), int)
    foot_heights = np.array([(feet or {})[int(m)] for m in foot_markers], np.float64)
    groups = None
    if balance is not None and len(balance[0]) and len(balance[1]):
        groups = (np.asarray(balance[0], int), np.asarray(balance[1], int))
    acc_sigma = FIT_ACCELERATION / float(fps) ** 2

    def terms(x: np.ndarray):
        return _fit_terms(
            x.reshape(n_frames, n_markers),
            origin,
            steps,
            weight,
            pairs,
            np.asarray(lengths, np.float64),
            bone_sigma,
            cue,
            order,
            (foot_markers, foot_heights),
            groups,
            acc_sigma,
        )

    def residuals(x: np.ndarray) -> np.ndarray:
        return terms(x)[0]

    def jacobian(x: np.ndarray):
        r, rows, cols, vals = terms(x)
        return coo_matrix((vals, (rows, cols)), shape=(r.size, x.size)).tocsr()

    def solve():
        return least_squares(
            residuals,
            start.ravel(),
            jac=jacobian,
            method="trf",
            tr_solver="lsmr",
            max_nfev=max_nfev,
        )

    try:  # many BLAS threads make these small sparse products slower, not faster
        from threadpoolctl import threadpool_limits

        with threadpool_limits(limits=1):
            sol = solve()
    except ImportError:
        sol = solve()
    z = sol.x.reshape(n_frames, n_markers)
    z[~valid] = np.nan
    return z


def _fill_depths(z: np.ndarray) -> np.ndarray:
    """A starting depth for every marker and frame: gaps take the frame's (then the trial's) median."""
    import warnings

    z = np.asarray(z, np.float64).copy()
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        per_frame = np.nanmedian(z, axis=1)
    ok = np.isfinite(per_frame)
    if not ok.any():
        raise ValueError("no depth at all: nothing to fit")
    per_frame = np.interp(np.arange(len(per_frame)), np.flatnonzero(ok), per_frame[ok])
    gaps = ~np.isfinite(z)
    z[gaps] = np.broadcast_to(per_frame[:, None], z.shape)[gaps]
    return z


def estimate_segment_lengths(
    xyz: np.ndarray, bones: list[tuple[int, int]], mirror: dict[int, int] | None = None
) -> np.ndarray:
    """Median length of each segment over the trial; left and right segments share their mean."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        lengths = np.nanmedian(segment_lengths(xyz, bones), axis=0)
    if mirror:
        index = {tuple(sorted(b)): k for k, b in enumerate(bones)}
        out = lengths.copy()
        for k, (a, b) in enumerate(bones):
            twin = index.get(tuple(sorted((mirror.get(a, a), mirror.get(b, b)))))
            if twin is not None and twin != k:
                pair = np.array([lengths[k], lengths[twin]])
                if np.isfinite(pair).any():
                    out[k] = float(np.nanmean(pair))
        lengths = out
    return lengths


def kinematic_fit(
    origin: np.ndarray,
    steps: np.ndarray,
    z_map: np.ndarray,
    layout: Layout,
    fps: float = 30.0,
    pose_z: np.ndarray | None = None,
    lengths: np.ndarray | None = None,
    foot_heights: dict[int, float] | None = None,
    balance: bool = True,
) -> dict[str, Any]:
    """The whole-trial fit: the map says which end is nearer, the segment length says how much.

    ``z_map`` (frames, markers) is the depth read at each keypoint, already scaled to metres
    (``--scale floor-frame``). ``pose_z`` (frames, markers, metres, any origin) is the pose
    model's own depth; only the ORDER of each segment's ends is taken from it.

    1. First pass with no segment lengths: the map's difference along every segment (robust),
       the order of ``pose_z``, the feet on the floor, the trunk above the feet, smooth depth.
    2. The segment lengths are the medians of that pass (left and right share their mean), unless
       ``lengths`` gives them (NaN entries are estimated).
    3. Passes with the lengths, tighter each time. The map is not used any more: with the length
       known, a segment has two possible depths and the first pass has already chosen the side.

    Returns ``{"z", "lengths", "lengths_given", "first_pass_z"}``.
    """
    bones = layout.fit_bones
    if not bones:
        raise ValueError(f"the {layout.name} layout has no segments to fit")
    pairs = np.asarray(bones, int)
    feet = foot_heights if foot_heights is not None else dict(layout.foot_heights)
    groups = layout.balance if balance else None
    z_map = np.asarray(z_map, np.float64)
    cue = z_map[:, pairs[:, 1]] - z_map[:, pairs[:, 0]]
    cue[np.abs(cue) > FIT_CUE_GATE] = np.nan
    order = None
    if pose_z is not None:
        pose_z = np.asarray(pose_z, np.float64)
        order = pose_z[:, pairs[:, 1]] - pose_z[:, pairs[:, 0]]
    start = shrink_to_person(z_map, layout.trunk, FIT_INIT_KEEP)
    common: dict[str, Any] = {"order": order, "feet": feet, "balance": groups, "fps": fps}
    z = fit_depths(origin, steps, start, bones, cue=cue, **common)
    first = z.copy()
    origin3 = np.asarray(origin, np.float64)
    origin3 = origin3.reshape(-1, 1, 3) if origin3.ndim == 2 else origin3
    estimated = estimate_segment_lengths(origin3 + steps * z[:, :, None], bones, layout.mirror)
    given = np.zeros(len(bones), bool)
    used = estimated
    if lengths is not None:
        lengths = np.asarray(lengths, np.float64)
        given = np.isfinite(lengths)
        used = np.where(given, lengths, estimated)
    if not np.isfinite(used).all():
        missing = [bones[k] for k in np.flatnonzero(~np.isfinite(used))]
        raise ValueError(f"no length could be estimated for the segments {missing}")
    for sigma in FIT_BONE_STAGES:
        z = fit_depths(origin, steps, z, bones, lengths=used, bone_sigma=sigma, **common)
    return {"z": z, "lengths": used, "lengths_given": given, "first_pass_z": first}


def write_segment_lengths(
    path: Path | str, bones: list[tuple[int, int]], lengths: np.ndarray, given: np.ndarray
) -> None:
    """The segment lengths of a run: ``marker_a, marker_b, length_m, source`` (0-based markers)."""
    pd.DataFrame(
        {
            "marker_a": [a for a, _b in bones],
            "marker_b": [b for _a, b in bones],
            "length_m": np.asarray(lengths, np.float64),
            "source": ["given" if g else "estimated" for g in given],
        }
    ).to_csv(path, index=False, float_format="%.5f")


def read_segment_lengths(
    path: Path | str, bones: list[tuple[int, int]], mirror: dict[int, int] | None = None
) -> np.ndarray:
    """Segment lengths for ``bones`` from a file; NaN where the file has no such segment.

    Two kinds of file: the ``*_segment_lengths.csv`` of a previous run, or a 3D CSV of the same
    person and markers (``frame,p0_x,p0_y,p0_z,...``, for example a two-camera Rec3D), whose
    median segment lengths are then used.
    """
    header = pd.read_csv(path, nrows=0).columns
    if {"marker_a", "marker_b", "length_m"} <= set(header):
        table = pd.read_csv(path)
        known = {
            tuple(sorted((int(a), int(b)))): float(v)
            for a, b, v in zip(table["marker_a"], table["marker_b"], table["length_m"], strict=True)
        }
        return np.array([known.get(tuple(sorted(b)), np.nan) for b in bones], np.float64)
    _frames, xyz = read_wide_xyz(path)
    n_needed = max(max(b) for b in bones) + 1
    if xyz.shape[1] < n_needed:
        raise ValueError(
            f"{Path(path).name} has {xyz.shape[1]} markers; the segments need {n_needed}"
        )
    return estimate_segment_lengths(xyz, bones, mirror)


# ---------------------------------------------------------------------------
# Depth of a crop around the person, and the pose model's own z
# ---------------------------------------------------------------------------


def crop_box(uv: np.ndarray, margin: float = CROP_MARGIN) -> tuple[int, int, int] | None:
    """Square box ``(x0, y0, side)`` around the keypoints plus ``margin`` of its size per side.

    It may leave the image (the crop is then padded). None with fewer than two keypoints.
    """
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    ok = np.isfinite(uv).all(axis=1)
    if ok.sum() < 2:
        return None
    lo, hi = uv[ok].min(axis=0), uv[ok].max(axis=0)
    side = int(round(max(float(np.max(hi - lo)) * (1.0 + 2.0 * margin), 32.0)))
    centre = (lo + hi) / 2.0
    return int(round(centre[0] - side / 2.0)), int(round(centre[1] - side / 2.0)), side


def crop_image(image: np.ndarray, box: tuple[int, int, int], fill: str = "edge") -> np.ndarray:
    """The square ``box`` of ``image``; what falls outside repeats the border (or is zero)."""
    x0, y0, side = box
    h, w = image.shape[:2]
    pad = max(0, -x0, -y0, x0 + side - w, y0 + side - h)
    if pad:
        border = cv2.BORDER_REPLICATE if fill == "edge" else cv2.BORDER_CONSTANT
        image = cv2.copyMakeBorder(image, pad, pad, pad, pad, border, value=0)
    return image[y0 + pad : y0 + pad + side, x0 + pad : x0 + pad + side]


def read_pose_z(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Frames and z (frames, markers) of a MediaPipe ``*_mp_norm.csv``.

    That file has ``frame_index`` and then ``x, y, z`` per landmark; z is the depth relative to
    the hips in IMAGE WIDTHS (negative = nearer the camera).
    """
    data = pd.read_csv(path).to_numpy(np.float64)
    if data.shape[1] < 4 or (data.shape[1] - 1) % 3:
        raise ValueError(f"{Path(path).name}: expected frame_index and x, y, z per landmark")
    return data[:, 0].astype(int), data[:, 3::3]


def find_pose_z(pixels: Path | str | None, video: Path | str) -> Path | None:
    """The MediaPipe ``*_mp_norm.csv`` that belongs to a pixel CSV, when vailá wrote one.

    Looked for beside the pixel CSV, in its ``<video stem>/`` folder, and up to two folders
    below the video and below the pixel CSV's parent.
    """
    video = Path(video)
    stems = [video.stem]
    roots = [video.parent]
    if pixels:
        pixels = Path(pixels)
        stems.insert(0, pixels.name.removesuffix(".csv").removesuffix("_pixel_vaila"))
        roots = [pixels.parent, pixels.parent.parent, video.parent]
    seen: set[Path] = set()
    for root in roots:
        for stem in dict.fromkeys(stems):
            name = f"{stem}_mp_norm.csv"
            for pattern in (name, f"{stem}/{name}", f"*/{name}", f"*/*/{name}"):
                for found in sorted(root.glob(pattern)):
                    if found not in seen and found.is_file():
                        return found
                    seen.add(found)
    return None


def pixel_size_at_one_metre(steps: np.ndarray, uv: np.ndarray) -> float:
    """Metres that one pixel spans at one metre of camera depth (about 1 / focal length)."""
    steps = np.asarray(steps, np.float64).reshape(-1, 3)
    uv = np.asarray(uv, np.float64).reshape(-1, 2)
    ok = np.isfinite(steps).all(axis=1) & np.isfinite(uv).all(axis=1)
    if ok.sum() < 2:
        return float("nan")
    steps, uv = steps[ok], uv[ok]
    far = int(np.argmax(np.linalg.norm(uv - uv[0], axis=1)))
    span = float(np.linalg.norm(uv[far] - uv[0]))
    return float(np.linalg.norm(steps[far] - steps[0]) / span) if span > 0 else float("nan")


# ---------------------------------------------------------------------------
# Comparison with a reference reconstruction
# ---------------------------------------------------------------------------


def _kabsch(src: np.ndarray, dst: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotation and translation (no scale) that best move ``src`` onto ``dst``."""
    cs, cd = src.mean(axis=0), dst.mean(axis=0)
    u, _s, vt = np.linalg.svd((src - cs).T @ (dst - cd))
    d = np.sign(np.linalg.det(vt.T @ u.T))
    rot = vt.T @ np.diag([1.0, 1.0, d]) @ u.T
    return rot, cd - rot @ cs


def compare_with_reference(
    xyz: np.ndarray, reference: np.ndarray, rigid_align: bool = False
) -> dict[str, Any]:
    """Distance to a reference reconstruction, per point, per axis and per marker.

    ``rigid_align`` first moves ``xyz`` onto the reference with ONE rotation + translation for
    the whole trial (no scale): for a camera-frame result, that leaves only the shape error.
    """
    xyz = np.asarray(xyz, np.float64)
    reference = np.asarray(reference, np.float64)
    n = min(len(xyz), len(reference))
    xyz, reference = xyz[:n], reference[:n]
    ok = np.asarray(np.isfinite(xyz).all(axis=2) & np.isfinite(reference).all(axis=2))
    if not ok.any():
        return {"n_points": 0}
    if rigid_align:
        rot, shift = _kabsch(xyz[ok], reference[ok])
        xyz = xyz @ rot.T + shift
    diff = xyz - reference
    dist = np.linalg.norm(diff, axis=2)
    per_marker = [
        float(dist[:, m][ok[:, m]].mean()) if ok[:, m].any() else float("nan")
        for m in range(xyz.shape[1])
    ]
    return {
        "n_points": int(ok.sum()),
        "rigid_align": bool(rigid_align),
        "mpjpe_m": float(dist[ok].mean()),
        "median_m": float(np.median(dist[ok])),
        "p95_m": float(np.percentile(dist[ok], 95)),
        "bias_m": [float(v) for v in diff[ok].mean(axis=0)],
        "per_marker_m": per_marker,
    }


# ---------------------------------------------------------------------------
# Files
# ---------------------------------------------------------------------------


def write_wide_xyz(path: Path | str, frames: np.ndarray, xyz: np.ndarray) -> None:
    """Write the Rec3D layout ``frame,p0_x,p0_y,p0_z,...``; a missing point stays empty."""
    xyz = np.asarray(xyz, np.float64)
    cols = ["frame"] + [f"p{m}_{a}" for m in range(xyz.shape[1]) for a in "xyz"]
    data = np.column_stack([np.asarray(frames), xyz.reshape(len(xyz), -1)])
    df = pd.DataFrame(data, columns=pd.Index(cols))
    df["frame"] = df["frame"].astype(int)
    df.to_csv(path, index=False, float_format="%.6f")


def read_wide_xyz(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read ``frame,x1,y1,z1,...`` by column order: (frames, xyz (rows, markers, 3))."""
    values = pd.read_csv(path).to_numpy(np.float64)
    if (values.shape[1] - 1) % 3 != 0:
        raise ValueError(f"{path}: expected frame + triples of x,y,z columns")
    return values[:, 0].astype(int), values[:, 1:].reshape(len(values), -1, 3)


def read_wide_xy(path: Path | str) -> tuple[np.ndarray, np.ndarray]:
    """Read a pixel CSV ``frame,x1,y1,...`` by column order: (frames, xy (rows, markers, 2))."""
    values = pd.read_csv(path).to_numpy(np.float64)
    if (values.shape[1] - 1) % 2 != 0:
        raise ValueError(f"{path}: expected frame + pairs of x,y columns")
    frames = values[:, 0].astype(int) if len(values) > 1 else np.array([0])
    return frames, values[:, 1:].reshape(len(values), -1, 2)


def read_control_points(path: Path | str) -> np.ndarray:
    """Control points (N, 3) from a ``.ref3d`` (one wide row) or a plain ``x,y,z`` text file."""
    path = Path(path)
    try:
        plain = np.loadtxt(path, delimiter=",", ndmin=2)
        if plain.shape[1] == 3:
            return plain.astype(np.float64)
    except ValueError:
        pass
    row = pd.read_csv(path).to_numpy(np.float64)[0, 1:]
    return row.reshape(-1, 3)


def write_ply(path: Path | str, points: np.ndarray, colors: np.ndarray) -> None:
    """A small ASCII PLY point cloud (points (N, 3) metres, colours (N, 3) RGB 0-255)."""
    points = np.asarray(points, np.float64).reshape(-1, 3)
    colors = np.asarray(colors).reshape(-1, 3).astype(int)
    ok = np.isfinite(points).all(axis=1)
    points, colors = points[ok], colors[ok]
    header = (
        "ply\nformat ascii 1.0\n"
        f"element vertex {len(points)}\n"
        "property float x\nproperty float y\nproperty float z\n"
        "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n"
    )
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(header)
        for (x, y, z), (r, g, b) in zip(points, colors, strict=True):
            fh.write(f"{x:.4f} {y:.4f} {z:.4f} {r} {g} {b}\n")


def ensure_weights(name: str, models_dir: Path) -> Path:
    """``models_dir/name``, downloaded from the Ultralytics release on first use."""
    path = Path(name)
    if path.is_file():
        return path
    path = Path(models_dir) / Path(name).name
    if path.is_file():
        return path
    models_dir.mkdir(parents=True, exist_ok=True)
    print(f"Downloading {path.name} to {models_dir} (first use only)...")
    from ultralytics.utils.downloads import attempt_download_asset

    got = Path(attempt_download_asset(str(path)))
    if not got.is_file():
        raise FileNotFoundError(
            f"{path.name} could not be downloaded. Put the file in {models_dir} "
            "(https://docs.ultralytics.com/tasks/depth) or check the network."
        )
    return got


def dlt_for_frame(dlt_frames: np.ndarray, coeffs: np.ndarray, frame: int) -> np.ndarray:
    """The DLT of ``frame``: the only row of a fixed camera, else the row of that frame."""
    if len(coeffs) == 1:
        return coeffs[0]
    hit = np.flatnonzero(dlt_frames.astype(int) == int(frame))
    return coeffs[hit[0]] if len(hit) else coeffs[int(np.argmin(np.abs(dlt_frames - frame)))]


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


def _device(requested: str | None) -> str:
    if requested:
        return requested
    try:
        import torch

        return "0" if torch.cuda.is_available() else "cpu"
    except ImportError:
        return "cpu"


def predict_depth(
    model: Any, frame: np.ndarray, device: str, imgsz: int = DEPTH_IMGSZ
) -> np.ndarray:
    """The metric depth map (H, W) of one BGR frame, resized to the frame if needed."""
    result = model.predict(frame, imgsz=imgsz, device=device, verbose=False)[0]
    depth = result.depth.data
    depth = depth.detach().float().cpu().numpy() if hasattr(depth, "detach") else np.asarray(depth)
    depth = np.squeeze(depth).astype(np.float32)
    if depth.shape != frame.shape[:2]:
        depth = cv2.resize(depth, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_LINEAR)
    return depth


def person_mask(model: Any, frame: np.ndarray, device: str) -> np.ndarray | None:
    """Union of the person masks of a segmentation model (None when no person is found)."""
    result = model.predict(frame, device=device, classes=[0], verbose=False)[0]
    if result.masks is None or len(result.masks) == 0:
        return None
    masks = result.masks.data.detach().cpu().numpy() > 0.5
    union = masks.any(axis=0).astype(np.uint8)
    if union.shape != frame.shape[:2]:
        union = cv2.resize(union, (frame.shape[1], frame.shape[0]), interpolation=cv2.INTER_NEAREST)
    return union.astype(bool)


def pose_keypoints(
    model: Any, frame: np.ndarray, device: str, conf: float
) -> dict[int, np.ndarray]:
    """Tracked people of one frame: ``{track id: (17, 3) x, y, confidence}``."""
    result = model.track(frame, persist=True, device=device, verbose=False, conf=conf)[0]
    if result.keypoints is None or result.boxes is None or len(result.boxes) == 0:
        return {}
    data = result.keypoints.data.detach().cpu().numpy()
    ids = result.boxes.id
    ids = ids.detach().cpu().numpy().astype(int) if ids is not None else np.arange(len(data))
    return {int(i): kp for i, kp in zip(ids, data, strict=False)}


# ---------------------------------------------------------------------------
# One video
# ---------------------------------------------------------------------------


def _smooth_depth(z: np.ndarray, fps: float, cutoff_hz: float) -> np.ndarray:
    """Zero-lag low-pass of each marker's depth; gaps are bridged for the filter and restored."""
    try:
        from .filter_utils import butter_filter
    except ImportError:
        from filter_utils import butter_filter  # ty: ignore[unresolved-import]

    out = z.copy()
    for m in range(z.shape[1]):
        col = z[:, m]
        ok = np.isfinite(col)
        if ok.sum() < 12 or cutoff_hz >= fps / 2:
            continue
        filled = np.interp(np.arange(len(col)), np.flatnonzero(ok), col[ok])
        smooth = np.asarray(butter_filter(filled, fps, filter_type="low", cutoff=cutoff_hz))
        out[ok, m] = smooth[ok]
    return out


def control_scale(
    depth_model: Any,
    device: str,
    video: Path,
    frame_index: int,
    control_pixels: Path,
    ref3d: Path,
    L: np.ndarray,
    mode: str,
    radius: int = 6,
) -> dict[str, Any]:
    """Scale of the depth from the calibration control points seen in ``video``."""
    cap = cv2.VideoCapture(str(video))
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
    ok, frame = cap.read()
    cap.release()
    if not ok:
        raise ValueError(f"cannot read frame {frame_index} of {video}")
    depth = predict_depth(depth_model, frame, device)
    _frames, px = read_wide_xy(control_pixels)
    world = read_control_points(ref3d)
    if px.shape[1] != len(world):
        raise ValueError(
            f"{control_pixels.name} has {px.shape[1]} points and {ref3d.name} has {len(world)}"
        )
    z_pred, _n = sample_depth(depth, px[0], radius=radius)
    z_true = camera_depth(L, world)
    a, b = fit_depth_scale(z_pred, z_true, mode=mode)
    fitted = apply_depth_scale(z_pred, a, b)
    with np.errstate(invalid="ignore"):
        ratio = z_true / z_pred
    return {
        "a": a,
        "b": b,
        "scale": float(np.exp(b)) if a == 1.0 else None,
        "n_points": int(np.isfinite(ratio).sum()),
        "ratio_per_point": [float(r) for r in ratio],
        "ratio_spread": _log_spread(ratio),
        "residual_m_after_fit": [float(r) for r in (fitted - z_true)],
        "z_true_m": [float(v) for v in z_true],
        "z_pred_m": [float(v) for v in z_pred],
    }


def _log_spread(ratio: np.ndarray) -> float:
    """Robust spread (1.4826 * MAD) of the log of the true/predicted depth ratios."""
    r = np.asarray(ratio, np.float64)
    r = np.log(r[np.isfinite(r) & (r > 0)])
    return float(1.4826 * np.median(np.abs(r - np.median(r)))) if r.size else float("nan")


CONTROL_SPREAD_WARNING = 0.10  # above this the control points disagree by more than 10 %


def _colorize(depth: np.ndarray) -> np.ndarray:
    """A BGR picture of a depth map (near = warm), for the preview video."""
    ok = np.isfinite(depth) & (depth > 0)
    if not ok.any():
        return np.zeros((*depth.shape, 3), np.uint8)
    inv = np.zeros_like(depth)
    inv[ok] = 1.0 / depth[ok]
    lo, hi = np.percentile(inv[ok], [2, 98])
    norm = np.clip((inv - lo) / max(hi - lo, 1e-9), 0, 1)
    return cv2.applyColorMap((norm * 255).astype(np.uint8), cv2.COLORMAP_TURBO)


def lift_video(video: Path, args: argparse.Namespace, out_dir: Path) -> dict[str, Any]:
    """Run the whole lift on one video and write its outputs. Returns the summary."""
    from ultralytics import YOLO

    device = _device(args.device)
    models_dir = (
        Path(args.models_dir) if args.models_dir else Path(__file__).resolve().parent / "models"
    )
    depth_model = YOLO(str(ensure_weights(args.depth_model, models_dir)))
    pose_model = seg_model = None
    pixels: dict[int, np.ndarray] | None = None
    if args.pixels:
        px_frames, px = read_wide_xy(Path(args.pixels))
        pixels = {int(f): xy for f, xy in zip(px_frames, px, strict=True)}
    else:
        pose_model = YOLO(str(ensure_weights(args.pose_model, models_dir)))
    if args.mask:
        seg_model = YOLO(str(ensure_weights(args.seg_model, models_dir)))
    pose_z: dict[int, np.ndarray] | None = None
    pose_z_path: Path | None = None
    if args.pose_z and args.pose_z.lower() != "none":
        pose_z_path = (
            # "auto" belongs to a pixel CSV: the z must be of the same markers
            (find_pose_z(args.pixels, video) if args.pixels else None)
            if args.pose_z.lower() == "auto"
            else Path(args.pose_z)
        )
        if pose_z_path is not None:
            z_frames, z_values = read_pose_z(pose_z_path)
            pose_z = {int(f): row for f, row in zip(z_frames, z_values, strict=True)}
            print(f"  pose z (order of each segment's ends): {pose_z_path}")

    cap = cv2.VideoCapture(str(video))
    if not cap.isOpened():
        raise ValueError(f"cannot open {video}")
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    n_total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    stem = video.stem

    writer = None
    if args.preview:
        fourcc = cv2.VideoWriter.fourcc(*"mp4v")
        writer = cv2.VideoWriter(
            str(out_dir / f"{stem}_depth_preview.mp4"), fourcc, fps, (width, height)
        )

    # per person: frame -> (uv (N, 2), conf (N,), z (N,), n (N,))
    people: dict[int, dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]] = {}
    t0 = time.time()
    frame_idx = -1
    n_done = 0
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frame_idx += 1
        if args.max_frames and n_done >= args.max_frames:
            break
        if pixels is not None:
            if frame_idx not in pixels:
                continue
            xy = pixels[frame_idx]
            found = {0: np.column_stack([xy, np.ones(len(xy))])}
        else:
            found = pose_keypoints(pose_model, frame, device, args.pose_conf)
        depth = predict_depth(depth_model, frame, device, args.imgsz)
        mask = person_mask(seg_model, frame, device) if seg_model is not None else None
        for pid, kp in found.items():
            uv = kp[:, :2].astype(np.float64).copy()
            conf = kp[:, 2].astype(np.float64)
            uv[~(conf >= args.kp_conf)] = np.nan
            radius = args.radius if args.radius else window_radius(uv)
            z, n = sample_depth(depth, uv, radius=radius, mask=mask)
            box = crop_box(uv, args.crop_margin) if args.crop else None
            if box is not None:
                z, n = _crop_depth_at(depth_model, frame, box, uv, z, radius, mask, device, args)
            people.setdefault(pid, {})[frame_idx] = (uv, conf, z, n)
        if writer is not None:
            view = cv2.addWeighted(frame, 0.35, _colorize(depth), 0.65, 0)
            for _pid, kp in found.items():
                for u, v in kp[:, :2]:
                    if np.isfinite(u) and np.isfinite(v):
                        cv2.circle(view, (int(round(u)), int(round(v))), 5, (255, 255, 255), -1)
                        cv2.circle(view, (int(round(u)), int(round(v))), 5, (0, 0, 0), 1)
            writer.write(view)
        if args.pointcloud_every and n_done % args.pointcloud_every == 0:
            _write_cloud(
                out_dir / f"{stem}_cloud_f{frame_idx:06d}.ply", frame, depth, args, width, height
            )
        n_done += 1
        if n_done % 50 == 0:
            print(
                f"  {video.name}: {n_done} frames ({(time.time() - t0) / n_done * 1000:.0f} ms/frame)"
            )
    cap.release()
    if writer is not None:
        writer.release()
    elapsed = time.time() - t0
    if not people:
        raise ValueError(f"{video.name}: no person / no pixel row matched the video frames")

    summary: dict[str, Any] = {
        "video": str(video),
        "fps": fps,
        "image_size": [width, height],
        "frames_in_video": n_total,
        "frames_processed": n_done,
        "ms_per_frame": elapsed / max(n_done, 1) * 1000.0,
        "depth_model": args.depth_model,
        "depth_imgsz": args.imgsz,
        "depth_crop": {"margin": args.crop_margin} if args.crop else None,
        "pose_z": str(pose_z_path) if pose_z_path is not None else None,
        "keypoints_from": str(args.pixels) if args.pixels else args.pose_model,
        "device": device,
        "people": {},
    }

    scale_fit = None
    dlt_frames = dlt_coeffs = None
    if args.dlt3d:
        dlt_frames, dlt_coeffs = load_dlt3d_file(args.dlt3d)
    if args.scale == "control":
        if dlt_coeffs is None or not (args.control_video and args.control_pixels and args.ref3d):
            raise ValueError(
                "--scale control needs --dlt3d, --control-video, --control-pixels and --ref3d"
            )
        scale_fit = control_scale(
            depth_model,
            device,
            Path(args.control_video),
            args.control_frame,
            Path(args.control_pixels),
            Path(args.ref3d),
            dlt_coeffs[0],
            args.scale_fit,
        )
        summary["scale_control"] = scale_fit
        if scale_fit["ratio_spread"] > CONTROL_SPREAD_WARNING:
            print(
                "  WARNING: the control points disagree on the depth scale "
                f"(spread {scale_fit['ratio_spread'] * 100:.0f} %). Thin poles or small markers are "
                "not resolved by the depth map, which then reads the wall behind them. "
                "Prefer --scale floor-frame."
            )
    if args.scale in ("floor", "floor-frame") and dlt_coeffs is None:
        raise ValueError(
            f"--scale {args.scale} needs --dlt3d (the floor is the plane Z = 0 of the lab)"
        )

    keep = sorted(people, key=lambda p: -len(people[p]))
    if pixels is None and not args.all_people:
        keep = keep[:1]
    for pid in keep:
        if len(people[pid]) < args.min_frames:
            continue
        tag = "" if pixels is not None or (not args.all_people) else f"_id{pid}"
        summary["people"][str(pid)] = _finish_person(
            people[pid],
            f"{stem}{tag}",
            args,
            out_dir,
            fps,
            width,
            height,
            dlt_frames,
            dlt_coeffs,
            scale_fit,
            pose_z=pose_z,
        )
    return summary


def _crop_depth_at(
    model: Any,
    frame: np.ndarray,
    box: tuple[int, int, int],
    uv: np.ndarray,
    z_full: np.ndarray,
    radius: int,
    mask: np.ndarray | None,
    device: str,
    args: Any,
) -> tuple[np.ndarray, np.ndarray]:
    """Depth at the keypoints from a square crop around the person.

    The crop puts several times more pixels on the body, so the depth BETWEEN joints is better
    resolved (on the two-camera test clip its correlation with the truth rose from 0.55 to
    0.79). A crop changes the network's absolute scale, so the crop only gives the relief: its
    level is set to the full-frame depth of the same keypoints (ratio of the medians).
    """
    import warnings

    crop = crop_image(frame, box)
    depth = predict_depth(model, crop, device, args.imgsz)
    crop_mask = (
        crop_image(mask.astype(np.uint8), box, fill="zero") > 0 if mask is not None else None
    )
    z, n = sample_depth(depth, uv - np.array(box[:2], np.float64), radius=radius, mask=crop_mask)
    both = np.isfinite(z) & np.isfinite(z_full)
    if not both.any():
        return z_full, n
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        level = float(np.median(z_full[both]) / np.median(z[both]))
    return z * level, n


def _write_cloud(
    path: Path, frame: np.ndarray, depth: np.ndarray, args: Any, width: int, height: int
) -> None:
    """A decimated point cloud of one frame in the camera frame (or the lab frame with a DLT)."""
    stride = 6
    vv, uu = np.mgrid[0:height:stride, 0:width:stride]
    uv = np.column_stack([uu.ravel(), vv.ravel()]).astype(np.float64)
    z = depth[vv.ravel(), uu.ravel()].astype(np.float64)
    if args.dlt3d:
        _f, coeffs = load_dlt3d_file(args.dlt3d)
        pts = dlt_point_at_depth(coeffs[0], uv, z)
    else:
        pts = backproject_pinhole(uv, z, _pinhole(args, width, height)[0])
    colors = frame[vv.ravel(), uu.ravel()][:, ::-1]
    write_ply(path, pts, colors)


def _pinhole(args: Any, width: int, height: int) -> tuple[np.ndarray, np.ndarray | None, str]:
    """K, distortion and where the focal came from, for the uncalibrated / intrinsics modes."""
    if args.intrinsics_toml:
        try:
            from .chessboard_calib import load_intrinsics_toml
        except ImportError:
            from chessboard_calib import load_intrinsics_toml  # ty: ignore[unresolved-import]

        intr = load_intrinsics_toml(Path(args.intrinsics_toml))
        return intr["K"], intr["dist"], f"intrinsics TOML ({Path(args.intrinsics_toml).name})"
    if args.focal_px:
        return intrinsics_from_focal(args.focal_px, width, height), None, "--focal-px"
    if args.fov_deg:
        return intrinsics_from_fov(args.fov_deg, width, height), None, "--fov-deg (horizontal)"
    focal = default_focal_px(width, height)
    return (
        intrinsics_from_focal(focal, width, height),
        None,
        "GUESS: image diagonal (no calibration)",
    )


def _finish_person(
    rows: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]],
    base: str,
    args: Any,
    out_dir: Path,
    fps: float,
    width: int,
    height: int,
    dlt_frames: np.ndarray | None,
    dlt_coeffs: np.ndarray | None,
    scale_fit: dict[str, Any] | None,
    pose_z: dict[int, np.ndarray] | None = None,
) -> dict[str, Any]:
    """Scale, lift, refine, check and write one person's trial."""
    frames = np.array(sorted(rows))
    uv = np.stack([rows[f][0] for f in frames])
    conf = np.stack([rows[f][1] for f in frames])
    z_raw = np.stack([rows[f][2] for f in frames])
    n_px = np.stack([rows[f][3] for f in frames])
    n_markers = uv.shape[1]
    layout = layout_for(n_markers)
    info: dict[str, Any] = {
        "layout": layout.name,
        "frames": int(len(frames)),
        "markers": int(n_markers),
        "points_with_depth": float(np.isfinite(z_raw).mean()),
    }

    K, dist, focal_source = _pinhole(args, width, height)
    uv_used = uv
    if dist is not None and np.any(dist != 0):
        uv_used = np.stack([undistort_pixels(f_uv, K, dist) for f_uv in uv])
        info["undistorted"] = True

    def dlt_of(k: int) -> np.ndarray:
        assert dlt_frames is not None and dlt_coeffs is not None
        return dlt_for_frame(dlt_frames, dlt_coeffs, int(frames[k]))

    # --- the whole-trial fit needs the lab frame, a skeleton and feet -----------------
    can_fit = dlt_coeffs is not None and bool(layout.fit_bones) and bool(layout.feet)
    if args.fit == "on" and not can_fit:
        raise ValueError(
            "--fit on needs --dlt3d and a known marker layout "
            f"(17 COCO or 33 MediaPipe points; this run has {n_markers})"
        )
    use_fit = can_fit and args.fit != "off"
    scale_mode = args.scale
    if use_fit and scale_mode == "none":
        scale_mode = "floor-frame"  # the fit starts from the floor-scaled map

    # --- scale of the depth -------------------------------------------------
    z = z_raw.copy()
    info["scale_mode"] = scale_mode
    if scale_mode == "control" and scale_fit is not None:
        z = apply_depth_scale(z, scale_fit["a"], scale_fit["b"])
        info["scale_applied"] = {"a": scale_fit["a"], "b": scale_fit["b"]}
    elif scale_mode in ("floor", "floor-frame") and layout.feet:
        per_frame = np.array(
            [
                floor_scale_of_frame(dlt_of(k), uv_used[k], z[k], layout, args.foot_height)
                for k in range(len(frames))
            ]
        )
        if not np.isfinite(per_frame).any():
            raise ValueError("--scale floor: no foot marker with depth was found")
        global_scale = float(np.nanmedian(per_frame))
        if scale_mode == "floor":
            z = z * global_scale
            info["scale_applied"] = {"scale": global_scale}
        else:
            ok = np.isfinite(per_frame)
            filled = np.interp(np.arange(len(per_frame)), np.flatnonzero(ok), per_frame[ok])
            calm = _moving_median(filled, max(int(round(fps / 3)) | 1, 3))
            z = z * calm[:, None]
            info["scale_applied"] = {
                "scale_median": global_scale,
                "scale_min": float(calm.min()),
                "scale_max": float(calm.max()),
            }
        info["foot_heights_m"] = (
            {str(m): args.foot_height for m in layout.foot_markers}
            if args.foot_height is not None
            else {str(m): h for m, h in layout.foot_heights.items()}
        )
        info["floor_ratio_p10_p90"] = [float(v) for v in np.nanpercentile(per_frame, [10, 90])]
        # how much the scale moves when the assumed foot height is 1 cm off (large for a low camera)
        k_mid = len(frames) // 2
        up = floor_scale_of_frame(
            dlt_of(k_mid), uv_used[k_mid], z_raw[k_mid], layout, args.foot_height, 0.01
        )
        now = floor_scale_of_frame(
            dlt_of(k_mid), uv_used[k_mid], z_raw[k_mid], layout, args.foot_height
        )
        if np.isfinite(up) and np.isfinite(now) and now > 0:
            info["scale_change_per_cm_of_foot_height"] = float(up / now - 1.0)
    elif scale_mode in ("floor", "floor-frame"):
        raise ValueError(f"--scale {scale_mode}: the {layout.name} layout has no foot markers")

    if not use_fit:
        if args.smooth_hz and args.smooth_hz > 0:
            z = _smooth_depth(z, fps, args.smooth_hz)
            info["smooth_hz"] = args.smooth_hz
        # depth between joints: keep only part of it
        info["joint_depth"] = args.joint_depth
        if args.joint_depth < 1.0:
            z = shrink_to_person(z, layout.trunk, args.joint_depth)

    # --- rays ----------------------------------------------------------------
    if dlt_coeffs is not None:
        rays = [dlt_rays(dlt_of(k), uv_used[k]) for k in range(len(frames))]
        origin = np.stack([r[0] for r in rays])[:, None, :]  # one camera centre per frame
        steps = np.stack([r[1] for r in rays])
        cam0 = decompose_dlt3d(dlt_of(0))
        info["camera"] = {
            "source": f"DLT3D ({Path(args.dlt3d).name})",
            "focal_px": [float(cam0.K[0, 0]), float(cam0.K[1, 1])],
            "principal_point_px": [float(cam0.K[0, 2]), float(cam0.K[1, 2])],
            "centre_m": [float(v) for v in cam0.C],
        }
    else:
        origin = np.zeros((len(frames), 1, 3))
        steps = np.stack([backproject_pinhole(f_uv, np.ones(n_markers), K) for f_uv in uv_used])
        info["camera"] = {
            "source": focal_source,
            "focal_px": [float(K[0, 0]), float(K[1, 1])],
            "principal_point_px": [float(K[0, 2]), float(K[1, 2])],
        }

    if use_fit:
        heights = (
            dict.fromkeys(layout.foot_markers, args.foot_height)
            if args.foot_height is not None
            else dict(layout.foot_heights)
        )
        z_order = None
        if pose_z is not None:
            # image widths -> metres: the width in pixels, the pixel size and the person's depth
            rows_z = [pose_z[int(f)] for f in frames if int(f) in pose_z]
            if len(rows_z) == len(frames) and all(len(r) == n_markers for r in rows_z):
                metres = np.array(
                    [pixel_size_at_one_metre(steps[k], uv_used[k]) for k in range(len(frames))]
                )
                person = np.nanmedian(_fill_depths(z), axis=1)
                z_order = np.stack(rows_z) * (width * metres * person)[:, None]
            else:
                info["pose_z_error"] = "the pose z file does not have these frames and markers"
        given = None
        if args.segment_lengths:
            given = read_segment_lengths(args.segment_lengths, layout.fit_bones, layout.mirror)
        fitted = kinematic_fit(
            origin[:, 0, :],
            steps,
            z,
            layout,
            fps=fps,
            pose_z=z_order,
            lengths=given,
            foot_heights=heights,
            balance=args.balance,
        )
        before = segment_variation(origin + steps * z[..., None], layout.bones)
        z = fitted["z"]
        write_segment_lengths(
            out_dir / f"{base}_segment_lengths.csv",
            layout.fit_bones,
            fitted["lengths"],
            fitted["lengths_given"],
        )
        info["fit"] = {
            "segments": len(layout.fit_bones),
            "segment_lengths_given": int(fitted["lengths_given"].sum()),
            "segment_lengths_from": str(args.segment_lengths) if args.segment_lengths else None,
            "segment_lengths_csv": f"{base}_segment_lengths.csv",
            "pose_z_order": z_order is not None,
            "balance": bool(args.balance),
            "foot_heights_m": {str(m): float(h) for m, h in heights.items()},
            "variation_before": before,
        }
    elif args.bone_fix and layout.bones:
        before = segment_variation(origin + steps * z[..., None], layout.bones)
        z = bone_fix_rays(origin[0, 0], steps, z, layout.bones, weight=args.bone_weight)
        info["bone_fix"] = {"weight": args.bone_weight, "variation_before": before}

    xyz = origin + steps * z[..., None]  # lab frame with a DLT, camera frame without
    if dlt_coeffs is not None:
        world = xyz
        cam_xyz = np.stack([world_to_camera(dlt_of(k), world[k]) for k in range(len(frames))])
        write_wide_xyz(out_dir / f"{base}_yolodepth3d_world.csv", frames, world)
        info["world_csv"] = f"{base}_yolodepth3d_world.csv"
    else:
        world = None
        cam_xyz = xyz
    write_wide_xyz(out_dir / f"{base}_yolodepth3d_cam.csv", frames, cam_xyz)
    info["cam_csv"] = f"{base}_yolodepth3d_cam.csv"

    # --- per-point depth table ------------------------------------------------
    table = pd.DataFrame(
        {
            "frame": np.repeat(frames, n_markers),
            "marker": np.tile(np.arange(n_markers), len(frames)),
            "u_px": uv[..., 0].ravel(),
            "v_px": uv[..., 1].ravel(),
            "confidence": conf.ravel(),
            "depth_raw_m": z_raw.ravel(),
            "depth_used_m": z.ravel(),
            "window_pixels": n_px.ravel(),
        }
    )
    table.to_csv(out_dir / f"{base}_depth_at_points.csv", index=False, float_format="%.5f")

    # --- C3D -------------------------------------------------------------------
    try:
        try:
            from .readcsv_export import auto_create_c3d_from_csv
        except ImportError:
            from readcsv_export import auto_create_c3d_from_csv  # ty: ignore[unresolved-import]

        best = world if world is not None else cam_xyz
        # the C3D writer keeps only columns that end in _X, _Y, _Z
        cols = ["frame"] + [f"p{m}_{a}" for m in range(n_markers) for a in "XYZ"]
        data = np.column_stack([frames, best.reshape(len(frames), -1)])
        df = pd.DataFrame(data, columns=pd.Index(cols))
        auto_create_c3d_from_csv(
            df,
            str(out_dir / f"{base}_yolodepth3d_m.c3d"),
            point_rate=round(fps),
            conversion_factor=1,
        )
        info["c3d"] = f"{base}_yolodepth3d_m.c3d"
    except Exception as exc:  # the C3D is a convenience; the CSV is the result
        info["c3d_error"] = str(exc)

    # --- quality ----------------------------------------------------------------
    quality: dict[str, Any] = {}
    best = world if world is not None else cam_xyz
    if layout.bones:
        variation = segment_variation(best, layout.bones)
        quality["segment_variation"] = variation
        if variation:
            quality["segment_variation_median"] = float(np.median(list(variation.values())))
    if world is not None and layout.feet:
        with np.errstate(all="ignore"):
            import warnings

            with warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                lowest = np.nanmin(world[:, layout.foot_markers, 2], axis=1)
        if np.isfinite(lowest).any():
            quality["lowest_foot_z_m"] = {
                "median": float(np.nanmedian(lowest)),
                "p05": float(np.nanpercentile(lowest, 5)),
                "p95": float(np.nanpercentile(lowest, 95)),
            }
    if world is not None and args.ref3d:
        ctrl = read_control_points(Path(args.ref3d))
        lo, hi = ctrl.min(axis=0) - args.volume_margin, ctrl.max(axis=0) + args.volume_margin
        ok = np.asarray(np.isfinite(world).all(axis=2))
        inside = np.asarray(((world >= lo) & (world <= hi)).all(axis=2))
        quality["inside_calibrated_volume"] = float(inside[ok].mean()) if ok.any() else None
        quality["volume_margin_m"] = args.volume_margin
    if world is not None and args.check_dlt3d and args.check_pixels:
        _cf, check_coeffs = load_dlt3d_file(args.check_dlt3d)
        c_frames, c_px = read_wide_xy(Path(args.check_pixels))
        by_frame = {int(f): xy for f, xy in zip(c_frames, c_px, strict=True)}
        errs = []
        for k, f in enumerate(frames):
            if int(f) in by_frame and by_frame[int(f)].shape[0] == n_markers:
                proj = dlt_project(check_coeffs[0], world[k])
                errs.append(np.linalg.norm(proj - by_frame[int(f)], axis=1))
        if errs:
            e = np.concatenate(errs)
            e = e[np.isfinite(e)]
            if e.size:
                quality["second_camera_reprojection_px"] = {
                    "mean": float(e.mean()),
                    "median": float(np.median(e)),
                    "p95": float(np.percentile(e, 95)),
                    "n": int(e.size),
                }
    if args.reference_3d:
        r_frames, ref = read_wide_xyz(Path(args.reference_3d))
        if ref.shape[1] == n_markers:
            by_frame = {int(f): k for k, f in enumerate(r_frames)}
            pick = [k for k, f in enumerate(frames) if int(f) in by_frame]
            ref_rows = np.array([by_frame[int(frames[k])] for k in pick], int)
            if pick:
                quality["reference"] = compare_with_reference(
                    best[pick], ref[ref_rows], rigid_align=world is None
                )
        else:
            quality["reference_error"] = (
                f"the reference has {ref.shape[1]} markers and this run has {n_markers}"
            )
    info["quality"] = quality
    return info


# ---------------------------------------------------------------------------
# CLI and window
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="yolo_depth3d",
        description="Monocular 3D from one video: 2D keypoints + YOLO26 metric depth (+ DLT3D).",
    )
    p.add_argument("-i", "--input", required=True, help="a video, or a folder of videos")
    p.add_argument(
        "-o", "--output-dir", help="where to create yolodepth3d_<time>/ (default: beside the video)"
    )
    p.add_argument("--pixels", help="existing pixel CSV (frame,x1,y1,...) instead of YOLO26-pose")
    p.add_argument("--depth-model", default=DEFAULT_DEPTH_MODEL, help="yolo26{n,s,m,l,x}-depth.pt")
    p.add_argument("--pose-model", default=DEFAULT_POSE_MODEL)
    p.add_argument("--seg-model", default=DEFAULT_SEG_MODEL)
    p.add_argument("--models-dir", help="folder of the weights (default: vaila/models)")
    p.add_argument("--imgsz", type=int, default=DEPTH_IMGSZ, help="depth inference size")
    p.add_argument("--device", help="'0', 'cpu', ... (default: GPU when available)")
    p.add_argument(
        "--mask", action="store_true", help="sample the depth only inside the person mask"
    )
    p.add_argument(
        "--radius", type=int, default=0, help="depth window radius in px (0 = from the person size)"
    )
    p.add_argument(
        "--crop",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="read the depth of each person from a square crop around it (more pixels on the body)",
    )
    p.add_argument(
        "--crop-margin", type=float, default=CROP_MARGIN, help="margin of the crop, per side"
    )
    p.add_argument("--pose-conf", type=float, default=0.25, help="person confidence of YOLO26-pose")
    p.add_argument("--kp-conf", type=float, default=0.3, help="keypoints below this get no depth")
    p.add_argument(
        "--all-people",
        action="store_true",
        help="write every tracked person, not only the longest track",
    )
    p.add_argument("--min-frames", type=int, default=5, help="skip a person seen in fewer frames")
    p.add_argument(
        "--max-frames", type=int, default=0, help="stop after N processed frames (0 = all)"
    )
    cal = p.add_argument_group("calibration")
    cal.add_argument("--focal-px", type=float, help="focal length in pixels (no calibration file)")
    cal.add_argument("--fov-deg", type=float, help="horizontal field of view in degrees")
    cal.add_argument(
        "--intrinsics-toml", help="Chessboard Calib <camera>_intrinsics.toml (lens + distortion)"
    )
    cal.add_argument("--dlt3d", help="the camera's .dlt3d: the result is in the lab frame")
    cal.add_argument(
        "--ref3d", help="control points (.ref3d or x,y,z text): volume check and --scale control"
    )
    sc = p.add_argument_group("scale of the depth")
    sc.add_argument("--scale", choices=SCALE_MODES, default="none")
    sc.add_argument(
        "--scale-fit",
        choices=("scale", "logaffine"),
        default="scale",
        help="fit used by --scale control",
    )
    sc.add_argument("--control-video", help="video that shows the calibration object")
    sc.add_argument(
        "--control-frame", type=int, default=0, help="frame of --control-video (0-based)"
    )
    sc.add_argument("--control-pixels", help="pixel CSV of the control points in that frame")
    sc.add_argument(
        "--foot-height", type=float, help="height of a standing foot marker above Z = 0 (m)"
    )
    fit = p.add_argument_group("whole-trial fit (needs --dlt3d)")
    fit.add_argument(
        "--fit",
        choices=("auto", "on", "off"),
        default="auto",
        help="solve the depths of the whole trial with constant segment lengths, the feet on "
        "the floor and the trunk above the feet (auto = on with --dlt3d and 17 or 33 markers)",
    )
    fit.add_argument(
        "--pose-z",
        default="auto",
        help="MediaPipe *_mp_norm.csv: its z gives the ORDER of each segment's ends "
        "(auto = look for it near the pixel CSV; none = do not use it)",
    )
    fit.add_argument(
        "--segment-lengths",
        help="segment lengths of the same person: the *_segment_lengths.csv of a previous run, "
        "or a 3D CSV with the same markers (a two-camera reconstruction)",
    )
    fit.add_argument(
        "--no-balance",
        dest="balance",
        action="store_false",
        help="do not assume that the trunk stays above the feet",
    )
    ref = p.add_argument_group("refinement and checks (with --fit off)")
    ref.add_argument(
        "--smooth-hz", type=float, default=0.0, help="zero-lag low-pass of the depth (0 = off)"
    )
    ref.add_argument(
        "--joint-depth",
        type=float,
        default=DEFAULT_JOINT_DEPTH,
        help="part of the depth between joints that is kept: 1 = as read, 0 = one depth per person",
    )
    ref.add_argument(
        "--bone-fix",
        action="store_true",
        help="keep the segment lengths (points stay on their rays)",
    )
    ref.add_argument("--bone-weight", type=float, default=4.0)
    ref.add_argument(
        "--volume-margin", type=float, default=0.5, help="margin around the control points (m)"
    )
    ref.add_argument(
        "--check-dlt3d", help="a SECOND camera's .dlt3d, for an independent reprojection check"
    )
    ref.add_argument("--check-pixels", help="that camera's pixel CSV (same markers, synced frames)")
    ref.add_argument(
        "--reference-3d", help="a reference 3D CSV (frame,x,y,z per marker) to compare with"
    )
    out = p.add_argument_group("extra outputs")
    out.add_argument(
        "--preview", action="store_true", help="video of the depth map with the keypoints"
    )
    out.add_argument(
        "--pointcloud-every", type=int, default=0, help="write a .ply every N frames (0 = off)"
    )
    return p


def _videos(path: Path) -> list[Path]:
    if path.is_dir():
        return sorted(p for p in path.iterdir() if p.suffix.lower() in VIDEO_SUFFIXES)
    return [path]


def run(args: argparse.Namespace) -> list[Path]:
    """Lift every video of ``args.input``. Returns the output folders."""
    videos = _videos(Path(args.input))
    if not videos:
        raise ValueError(f"no video found in {args.input}")
    if args.pixels and len(videos) > 1:
        raise ValueError("--pixels belongs to one video: give a video file, not a folder")
    stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    outs = []
    for video in videos:
        parent = Path(args.output_dir) if args.output_dir else video.parent
        out_dir = parent / f"yolodepth3d_{stamp}"
        out_dir.mkdir(parents=True, exist_ok=True)
        print(f"YOLO26 depth 3D: {video.name}")
        summary = lift_video(video, args, out_dir)
        summary["version"] = VERSION
        summary["created"] = datetime.datetime.now().isoformat(timespec="seconds")
        (out_dir / f"{video.stem}_yolodepth3d_summary.json").write_text(
            json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8"
        )
        _write_config(out_dir / "configuration_used.toml", args)
        _print_summary(summary, out_dir)
        hint = preview_hint(args)
        if hint:
            print(hint)
        outs.append(out_dir)
    return outs


def _write_config(path: Path, args: argparse.Namespace) -> None:
    lines = ["# yolo_depth3d: the options of this run", f'version = "{VERSION}"']
    for key, value in sorted(vars(args).items()):
        if value is None:
            continue
        if isinstance(value, bool):
            lines.append(f"{key} = {str(value).lower()}")
        elif isinstance(value, (int, float)):
            lines.append(f"{key} = {value}")
        else:
            lines.append(f"{key} = {json.dumps(str(value), ensure_ascii=False)}")
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def preview_hint(args: argparse.Namespace) -> str | None:
    """A line for the terminal when the depth preview video was not asked for (CLI only)."""
    if args.preview:
        return None
    return "  no depth preview video: add --preview to write <video>_depth_preview.mp4"


def _print_summary(summary: dict[str, Any], out_dir: Path) -> None:
    print(f"  {summary['frames_processed']} frames, {summary['ms_per_frame']:.0f} ms/frame")
    for pid, info in summary["people"].items():
        cam = info["camera"]
        print(f"  person {pid}: {info['frames']} frames, {info['layout']}, camera: {cam['source']}")
        q = info.get("quality", {})
        if "fit" in info:
            fit = info["fit"]
            lengths = (
                f"{fit['segment_lengths_given']} of {fit['segments']} given"
                if fit["segment_lengths_given"]
                else "estimated from this trial"
            )
            order = "depth map + pose z" if fit["pose_z_order"] else "depth map only"
            print(f"    whole-trial fit: segment lengths {lengths}; order of the ends: {order}")
        if "pose_z_error" in info:
            print(f"    pose z not used: {info['pose_z_error']}")
        if "segment_variation_median" in q:
            print(
                f"    segment length variation (median): {q['segment_variation_median'] * 100:.1f} %"
            )
        if "lowest_foot_z_m" in q:
            print(f"    lowest foot height (median): {q['lowest_foot_z_m']['median']:+.3f} m")
        if "second_camera_reprojection_px" in q:
            print(
                f"    reprojection in the second camera: {q['second_camera_reprojection_px']['mean']:.1f} px"
            )
        if "reference" in q and q["reference"].get("n_points"):
            r = q["reference"]
            tag = " (rigidly aligned)" if r["rigid_align"] else ""
            print(f"    error against the reference{tag}: {r['mpjpe_m'] * 100:.1f} cm")
    print(f"  outputs: {out_dir}")


def cli_command(args: argparse.Namespace) -> str:
    """The terminal command equivalent to ``args`` (printed by the window)."""
    parts = ["uv", "run", "--no-sync", "python", "-m", "vaila.yolo_depth3d"]
    defaults = build_parser().parse_args(["-i", "x"])
    for key, value in vars(args).items():
        flag = "--" + key.replace("_", "-")
        if key == "input":
            parts += ["-i", str(value)]
        elif value is None or value == getattr(defaults, key):
            continue
        elif isinstance(value, bool):
            parts.append(flag if value else "--no-" + key.replace("_", "-"))
        else:
            parts += [flag, str(value)]
    return " ".join(shlex.quote(p) for p in parts)


def run_yolo_depth3d() -> None:
    """Window: choose the video and the optional calibration, then run."""
    import tkinter as tk
    from tkinter import filedialog, messagebox, ttk

    root = getattr(tk, "_default_root", None)
    owns_root = root is None
    win = tk.Tk() if owns_root else tk.Toplevel(root)
    win.title("YOLO26 Depth 3D (monocular)")
    win.geometry("900x640")
    values: dict[str, tk.Variable] = {}
    frame = ttk.Frame(win, padding=10)
    frame.pack(fill="both", expand=True)
    frame.columnconfigure(1, weight=1)

    def pick(key: str, title: str, types: list[tuple[str, str]], folder: bool = False):
        def go():
            got = (
                filedialog.askdirectory(title=title, parent=win)
                if folder
                else filedialog.askopenfilename(title=title, filetypes=types, parent=win)
            )
            if got:
                values[key].set(got)

        return go

    rows = [
        ("input", "Video (required)", [("Video", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("All", "*.*")]),
        ("pixels", "Pixel CSV (optional; empty = YOLO26-pose)", [("CSV", "*.csv")]),
        ("dlt3d", "DLT3D of this camera (optional)", [("DLT3D", "*.dlt3d"), ("All", "*.*")]),
        ("intrinsics_toml", "Chessboard intrinsics TOML (optional)", [("TOML", "*.toml")]),
        (
            "ref3d",
            "Control points .ref3d (optional)",
            [("REF3D", "*.ref3d *.txt *.csv"), ("All", "*.*")],
        ),
        (
            "control_video",
            "Video of the calibration object (scale = control)",
            [("Video", "*.mp4 *.mov *.avi *.mkv")],
        ),
        ("control_pixels", "Pixels of the control points (scale = control)", [("CSV", "*.csv")]),
        (
            "segment_lengths",
            "Segment lengths or 3D CSV of the same person (optional)",
            [("CSV", "*.csv")],
        ),
        ("check_dlt3d", "Second camera DLT3D (check, optional)", [("DLT3D", "*.dlt3d")]),
        ("check_pixels", "Second camera pixel CSV (check, optional)", [("CSV", "*.csv")]),
        ("reference_3d", "Reference 3D CSV (check, optional)", [("CSV", "*.csv")]),
    ]
    for r, (key, label, types) in enumerate(rows):
        values[key] = tk.StringVar()
        ttk.Label(frame, text=label).grid(row=r, column=0, sticky="w", pady=2)
        ttk.Entry(frame, textvariable=values[key]).grid(row=r, column=1, sticky="ew", padx=6)
        ttk.Button(frame, text="Browse", command=pick(key, label, types)).grid(row=r, column=2)
    r = len(rows)
    values["depth_model"] = tk.StringVar(value=DEFAULT_DEPTH_MODEL)
    values["scale"] = tk.StringVar(value="none")
    values["fit"] = tk.BooleanVar(value=True)
    values["preview"] = tk.BooleanVar(value=True)
    values["mask"] = tk.BooleanVar(value=False)
    ttk.Label(frame, text="Depth model").grid(row=r, column=0, sticky="w", pady=4)
    ttk.Combobox(
        frame, textvariable=values["depth_model"], values=DEPTH_MODELS, state="readonly"
    ).grid(row=r, column=1, sticky="w", padx=6)
    ttk.Label(frame, text="Depth scale").grid(row=r + 1, column=0, sticky="w", pady=4)
    ttk.Combobox(frame, textvariable=values["scale"], values=SCALE_MODES, state="readonly").grid(
        row=r + 1, column=1, sticky="w", padx=6
    )
    checks = ttk.Frame(frame)
    checks.grid(row=r + 2, column=0, columnspan=3, sticky="w", pady=6)
    ttk.Checkbutton(checks, text="Whole-trial fit (with a DLT3D)", variable=values["fit"]).pack(
        side="left"
    )
    ttk.Checkbutton(checks, text="Person mask", variable=values["mask"]).pack(side="left", padx=12)
    ttk.Checkbutton(checks, text="Depth preview video", variable=values["preview"]).pack(
        side="left"
    )
    ttk.Label(
        frame,
        text=(
            "The depth is the visible surface and its absolute scale is an estimate.\n"
            "With a DLT3D the result is in the lab frame and the whole trial is fitted: constant\n"
            "segment lengths, feet on the floor (Z = 0), trunk above the feet. A MediaPipe\n"
            "*_mp_norm.csv near the pixel CSV is found and used for the order of the joints."
        ),
        foreground="#555",
    ).grid(row=r + 3, column=0, columnspan=3, sticky="w", pady=4)

    def start():
        argv = ["-i", values["input"].get().strip()]
        if not argv[1]:
            messagebox.showerror("YOLO26 Depth 3D", "Choose a video.", parent=win)
            return
        for key, _label, _types in rows[1:]:
            got = values[key].get().strip()
            if got:
                argv += ["--" + key.replace("_", "-"), got]
        argv += ["--depth-model", values["depth_model"].get(), "--scale", values["scale"].get()]
        for key in ("mask", "preview"):
            if values[key].get():
                argv.append("--" + key.replace("_", "-"))
        if not values["fit"].get():
            argv += ["--fit", "off"]
        args = build_parser().parse_args(argv)
        print(">> YOLO26 Depth 3D\n" + cli_command(args))
        win.destroy()
        try:
            outs = run(args)
        except Exception as exc:
            print(f"Error: {exc}")
            messagebox.showerror("YOLO26 Depth 3D", str(exc))
            return
        messagebox.showinfo("YOLO26 Depth 3D", "Done:\n" + "\n".join(str(o) for o in outs))

    ttk.Button(frame, text="Run", command=start).grid(row=r + 4, column=1, sticky="e", pady=10)
    ttk.Button(frame, text="Close", command=win.destroy).grid(row=r + 4, column=2, pady=10)
    if owns_root:
        win.mainloop()


def main(argv: list[str] | None = None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        run_yolo_depth3d()
        return 0
    args = build_parser().parse_args(argv)
    print(">> " + cli_command(args))
    try:
        run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
