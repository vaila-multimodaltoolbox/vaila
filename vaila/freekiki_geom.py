"""
================================================================================
Script: freekiki_geom.py - FreeKiki field geometry (camera from keypoints)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.9
Created: 04 October 2026
Update Date: 30 October 2026

Description:
    The pitch is a known 3D object (``vaila/models/soccerfield_kiki.csv``,
    metres, z up). Once some of the 49 keypoints are found in a frame, a camera
    is fitted to them and predicts where every other keypoint must be:

    * planar homography (world z = 0 -> image), RANSAC + Hartley-normalized SVD,
      from ``freekiki_diag.fit_field_homography`` and ``fit_dlt2d_hartley_svd``;
    * full camera for the points off the ground (flag tops z = 1.5 m, goal
      post tops z = 2.44 m):
        - DLT3D (11 parameters via Hartley-normalized SVD,
          ``fit_dlt3d_hartley_svd``) when >= 7 points including >= 2 off the
          ground are available, pruned of points that disagree and checked with
          ``decompose_dlt3d`` (square pixels, no skew, camera above the ground);
        - otherwise a pinhole camera recovered from the homography: principal
          point at the image centre, square pixels, focal from Zhang's two
          constraints (h1'w h2 = 0, h1'w h1 = h2'w h2 with w = diag(1/f^2,
          1/f^2, 1)), then R = [r1 r2 r1 x r2] and t from K^-1 H.

    ``dlt_coefficients`` turns each frame camera into the vailá DLT2D (8) and
    DLT3D (11) coefficients that ``rec2d.py`` / ``rec3d.py`` read, so the
    field calibration can drive per-frame reconstruction of player tracks.

    ``project_rigid_field`` forward-projects all metric coordinates from
    ``soccerfield_kiki.csv`` (49 pts) or ``soccerfield_ref3d.csv`` (37 pts) as
    an invariant rigid body into image space (including occluded and
    out-of-FOV points) and assigns per-keypoint visibility flags
    (``visible_in_fov``, ``occluded``, ``out_of_bounds``).

    ``refine_keypoints`` uses that camera on the network output:
      fill  a point the network did not accept (conf < kp_conf, or a
            same-pixel duplicate) is filled when its projection lies inside
            the image and the network still half-sees it (raw conf >=
            min_conf); it keeps the network position when that agrees with
            the projection, else takes the projection (code G);
      fix   fill + an accepted point farther than fix_px from its projection
            is replaced by the projection (Gx); in fill mode it is only
            flagged (Dg);
      off   unchanged.
    A geometry point says where the keypoint must be; it may be occluded.
    Distances are px@1920 (pixels rescaled to a 1920-px-wide image).

License:
    GNU Affero General Public License v3.0 (AGPLv3).
================================================================================
"""

from __future__ import annotations

import csv
import math
from collections import Counter
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

try:
    from . import freekiki_diag as diag
    from .dlt3d import calculate_dlt3d_params
    from .monocular_dlt_align import decompose_dlt3d, dlt_projection_matrix
except ImportError:
    import freekiki_diag as diag  # ty: ignore[unresolved-import]
    from dlt3d import calculate_dlt3d_params  # ty: ignore[unresolved-import]
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        decompose_dlt3d,
        dlt_projection_matrix,
    )

NKP = diag.NKP
REF3D_CSV = Path(__file__).resolve().parent / "models" / "soccerfield_ref3d.csv"
REF3D_CENTER_SHIFT = np.array([-52.5, -34.0, 0.0], dtype=float)
VISIBILITY_FLAGS = ("visible_in_fov", "occluded", "out_of_bounds")
GEOM_MODES = ("fill", "fix", "off")
GEOM_CODES = {
    "G": "filled by field geometry (network conf < kp_conf but >= min_conf; inside the image)",
    "Gx": "replaced by field geometry (accepted point farther than fix_px from the camera)",
    "Dg": "accepted, but farther than fix_px from the field geometry (kept in fill mode)",
}
GEOM_DEFAULTS: dict[str, Any] = {
    "mode": "fill",
    "min_conf": 0.05,  # network conf a point needs to be filled (not invented)
    "agree_px": 15.0,  # keep the network xy within this of the projection (px@1920)
    "fix_px": 20.0,  # accepted point farther than this disagrees with the camera
    "fit_px": 15.0,  # RANSAC / DLT3D pruning threshold (px@1920)
    "max_rmse_px": 8.0,  # camera fit RMSE on its own points (px@1920)
    "min_dlt_points": 7,  # DLT3D: 11 unknowns, 7 points keep redundancy
    "min_offplane": 2,  # DLT3D: points off the ground among them
    "min_focal_ratio": 0.2,  # focal / image width
    "max_focal_ratio": 50.0,
    "min_cam_height_m": 1.0,
    "max_cam_height_m": 300.0,
    "max_aspect_dev": 0.25,  # |fy/fx - 1| of a DLT3D camera
    "max_skew": 0.15,  # |K01| / focal of a DLT3D camera
    "margin": 0.01,  # of the image width kept from every border
    "smooth_window": 5,  # temporal rolling window for camera regularization (frames)
    "pnl_alpha": 0.7,  # PnLCalib joint points + lines weight alpha in [0, 1]
}
FILLABLE = ("Rk", "Rd")  # network codes geometry may fill


@lru_cache(maxsize=1)
def _field() -> tuple[np.ndarray, np.ndarray]:
    """``(xyz (49, 3) metres, planar mask)``; read once, never modified."""
    _, _, xyz = diag.load_field_points()
    xyz.setflags(write=False)
    planar = xyz[:, 2] == 0
    planar.setflags(write=False)
    return xyz, planar


@lru_cache(maxsize=1)
def load_ref3d_points() -> tuple[list[str], np.ndarray]:
    """``(names (37,), xyz (37, 3) metres)`` from ``soccerfield_ref3d.csv``.

    Shifted by ``(-52.5, -34.0, 0.0)`` so ``center_field`` sits at ``(0, 0, 0)``
    in the same rigid metric frame as ``soccerfield_kiki.csv`` while preserving
    all pairwise Euclidean distances.
    """
    names: list[str] = []
    coords: list[list[float]] = []
    with REF3D_CSV.open(encoding="utf-8") as f:
        for row in csv.DictReader(f):
            names.append(row["point_name"].strip())
            coords.append([float(row["x"]), float(row["y"]), float(row["z"])])
    xyz = np.asarray(coords, dtype=float) + REF3D_CENTER_SHIFT
    xyz.setflags(write=False)
    return names, xyz


@lru_cache(maxsize=1)
def _ref3d_to_kiki_map() -> tuple[tuple[int, ...], ...]:
    """Map each of the 37 ``soccerfield_ref3d.csv`` points to its closest kiki49 index(es)."""
    kiki_xyz, _ = _field()
    _, ref_xyz = load_ref3d_points()
    mapping: list[tuple[int, ...]] = []
    for j, pt in enumerate(ref_xyz):
        dists = np.linalg.norm(kiki_xyz - pt, axis=1)
        best = int(np.argmin(dists))
        if dists[best] <= 0.15:
            mapping.append((best,))
        elif j == 22:  # left_penalty_arc_top -> adjacent arc crossings p10, p11
            mapping.append((10, 11))
        elif j == 34:  # right_penalty_arc_top -> adjacent arc crossings p18, p19
            mapping.append((18, 19))
        else:
            mapping.append((best,))
    return tuple(mapping)


def hartley_normalize_2d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Hartley normalization of 2D points (centroid at 0, mean distance sqrt(2))."""
    pts = np.asarray(pts, dtype=float)[:, :2]
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = math.sqrt(2.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [[scale, 0.0, -scale * mean[0]], [0.0, scale, -scale * mean[1]], [0.0, 0.0, 1.0]],
        dtype=float,
    )
    pts_h = np.c_[pts, np.ones(len(pts))] @ T.T
    return pts_h[:, :2], T


def hartley_normalize_3d(pts: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Isotropic Hartley normalization of 3D points (centroid at 0, mean distance sqrt(3))."""
    pts = np.asarray(pts, dtype=float)[:, :3]
    mean = np.mean(pts, axis=0)
    centered = pts - mean
    mean_dist = float(np.mean(np.linalg.norm(centered, axis=1)))
    scale = math.sqrt(3.0) / mean_dist if mean_dist > 1e-12 else 1.0
    T = np.array(
        [
            [scale, 0.0, 0.0, -scale * mean[0]],
            [0.0, scale, 0.0, -scale * mean[1]],
            [0.0, 0.0, scale, -scale * mean[2]],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=float,
    )
    pts_h = np.c_[pts, np.ones(len(pts))] @ T.T
    return pts_h[:, :3], T


def fit_dlt2d_hartley_svd(xy: np.ndarray, xy_world: np.ndarray) -> np.ndarray | None:
    """Planar homography H (H[2,2] = 1) from world (X, Y) -> image (u, v) via Hartley + SVD."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    xy_w = np.asarray(xy_world, dtype=float)[:, :2]
    if len(xy) < 4 or len(xy) != len(xy_w):
        return None
    uv_n, T_img = hartley_normalize_2d(xy)
    XY_n, T_w = hartley_normalize_2d(xy_w)
    n = len(xy)
    A = np.zeros((2 * n, 9), dtype=float)
    X, Y = XY_n[:, 0], XY_n[:, 1]
    u, v = uv_n[:, 0], uv_n[:, 1]
    A[0::2, 0] = -X
    A[0::2, 1] = -Y
    A[0::2, 2] = -1.0
    A[0::2, 6] = u * X
    A[0::2, 7] = u * Y
    A[0::2, 8] = u
    A[1::2, 3] = -X
    A[1::2, 4] = -Y
    A[1::2, 5] = -1.0
    A[1::2, 6] = v * X
    A[1::2, 7] = v * Y
    A[1::2, 8] = v
    try:
        _, _, vt = np.linalg.svd(A)
        H_tilde = vt[-1].reshape(3, 3)
        H = np.linalg.inv(T_img) @ H_tilde @ T_w
    except np.linalg.LinAlgError:
        return None
    if abs(float(H[2, 2])) <= 1e-12:
        return None
    return H / float(H[2, 2])


def fit_dlt3d_hartley_svd(xy: np.ndarray, xyz_world: np.ndarray) -> np.ndarray | None:
    """DLT3D 11 parameters (L1..L11, P[2,3] = 1) from world (X, Y, Z) -> image (u, v) via Hartley + SVD."""
    xy = np.asarray(xy, dtype=float)[:, :2]
    xyz_w = np.asarray(xyz_world, dtype=float)[:, :3]
    if len(xy) < 6 or len(xy) != len(xyz_w):
        return None
    uv_n, T_img = hartley_normalize_2d(xy)
    XYZ_n, T_w = hartley_normalize_3d(xyz_w)
    n = len(xy)
    A = np.zeros((2 * n, 12), dtype=float)
    X, Y, Z = XYZ_n[:, 0], XYZ_n[:, 1], XYZ_n[:, 2]
    u, v = uv_n[:, 0], uv_n[:, 1]
    A[0::2, 0] = -X
    A[0::2, 1] = -Y
    A[0::2, 2] = -Z
    A[0::2, 3] = -1.0
    A[0::2, 8] = u * X
    A[0::2, 9] = u * Y
    A[0::2, 10] = u * Z
    A[0::2, 11] = u
    A[1::2, 4] = -X
    A[1::2, 5] = -Y
    A[1::2, 6] = -Z
    A[1::2, 7] = -1.0
    A[1::2, 8] = v * X
    A[1::2, 9] = v * Y
    A[1::2, 10] = v * Z
    A[1::2, 11] = v
    try:
        _, _, vt = np.linalg.svd(A)
        P_tilde = vt[-1].reshape(3, 4)
        P = np.linalg.inv(T_img) @ P_tilde @ T_w
    except np.linalg.LinAlgError:
        return None
    if abs(float(P[2, 3])) <= 1e-12:
        return None
    P = P / float(P[2, 3])
    return P.ravel()[:11]


def _settings(settings: dict | None) -> dict:
    return GEOM_DEFAULTS | dict(settings or {})


def _project(P: np.ndarray, xyz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Pixels and the in-front mask of world points through a 3x4 camera."""
    uvw = np.c_[xyz, np.ones(len(xyz))] @ P.T
    with np.errstate(invalid="ignore", divide="ignore"):
        uv = uvw[:, :2] / uvw[:, 2:3]
    return uv, uvw[:, 2] > 0


@lru_cache(maxsize=1)
def _painted_line_models() -> tuple[tuple[str, np.ndarray, np.ndarray, tuple[int, ...]], ...]:
    """Each painted line as ``(name, p1_3d, p2_3d, member_indices)``."""
    try:
        from . import freekiki_snap as fs
    except ImportError:
        import freekiki_snap as fs  # ty: ignore[unresolved-import]

    xyz, _ = _field()
    models: list[tuple[str, np.ndarray, np.ndarray, tuple[int, ...]]] = []
    for name, members in fs.PAINTED_LINES.items():
        pts = xyz[list(members)]
        ia, ib = max(
            ((a, b) for a in range(len(pts)) for b in range(a + 1, len(pts))),
            key=lambda ab: float(np.linalg.norm(pts[ab[0]] - pts[ab[1]])),
        )
        models.append((name, pts[ia].copy(), pts[ib].copy(), tuple(int(m) for m in members)))
    return tuple(models)


def _clip_line_in_front(
    P: np.ndarray, p1_3d: np.ndarray, p2_3d: np.ndarray, eps: float = 1e-3
) -> tuple[np.ndarray, np.ndarray] | None:
    """Project a 3D line segment through ``P`` (3x4), clipping against the camera frustum plane.

    Adapts PnLCalib's 3D line-plane intersection so segments partially behind
    the camera (``w < eps``) are clipped at ``w = eps`` instead of inverting
    through the vanishing line.
    """
    X1 = np.array([p1_3d[0], p1_3d[1], p1_3d[2], 1.0], dtype=float)
    X2 = np.array([p2_3d[0], p2_3d[1], p2_3d[2], 1.0], dtype=float)
    uvw1 = P @ X1
    uvw2 = P @ X2
    w1, w2 = float(uvw1[2]), float(uvw2[2])
    if w1 < eps and w2 < eps:
        return None
    if w1 < eps:
        dw = w2 - w1
        if abs(dw) <= 1e-12:
            return None
        t_clip = (eps - w1) / dw
        X1 = X1 + t_clip * (X2 - X1)
        uvw1 = P @ X1
    elif w2 < eps:
        dw = w1 - w2
        if abs(dw) <= 1e-12:
            return None
        t_clip = (eps - w2) / dw
        X2 = X2 + t_clip * (X1 - X2)
        uvw2 = P @ X2
    if abs(float(uvw1[2])) <= 1e-12 or abs(float(uvw2[2])) <= 1e-12:
        return None
    uv1 = uvw1[:2] / float(uvw1[2])
    uv2 = uvw2[:2] / float(uvw2[2])
    if not (np.isfinite(uv1).all() and np.isfinite(uv2).all()):
        return None
    return uv1, uv2


def _active_painted_lines(
    inliers_mask: np.ndarray,
) -> list[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Active painted lines with >= 2 inlier keypoints: ``(p1_3d, p2_3d, inlier_indices)``."""
    active: list[tuple[np.ndarray, np.ndarray, np.ndarray]] = []
    for _, p1, p2, members in _painted_line_models():
        idx = np.array([m for m in members if inliers_mask[m]], dtype=int)
        if len(idx) >= 2:
            active.append((p1, p2, idx))
    return active


def _pnl_line_residuals(
    P: np.ndarray,
    xy: np.ndarray,
    active_lines: list[tuple[np.ndarray, np.ndarray, np.ndarray]],
) -> np.ndarray:
    """Signed perpendicular point-to-projected-line residuals (pixels) across active lines."""
    res_list: list[np.ndarray] = []
    for p1_3d, p2_3d, idx in active_lines:
        clipped = _clip_line_in_front(P, p1_3d, p2_3d)
        if clipped is None:
            res_list.append(np.zeros(len(idx), dtype=float))
            continue
        uv1, uv2 = clipped
        d = uv2 - uv1
        norm = float(math.hypot(float(d[0]), float(d[1])))
        if norm <= 1e-6:
            res_list.append(np.zeros(len(idx), dtype=float))
            continue
        nx, ny = -float(d[1]) / norm, float(d[0]) / norm
        pts = xy[idx]
        perp = (pts[:, 0] - float(uv1[0])) * nx + (pts[:, 1] - float(uv1[1])) * ny
        res_list.append(perp)
    if not res_list:
        return np.zeros(0, dtype=float)
    return np.concatenate(res_list)


def _refine_homography_pnl(
    H: np.ndarray,
    xy: np.ndarray,
    xyz: np.ndarray,
    inliers: np.ndarray,
    width: float,
    s: dict,
) -> np.ndarray:
    """PnLCalib joint Points-and-Lines (PnL) homography refinement."""
    alpha = float(s.get("pnl_alpha", 0.7))
    if alpha <= 0.0 or int(inliers.sum()) < 4:
        return H
    active_lines = _active_painted_lines(inliers)
    if not active_lines:
        return H
    from scipy.optimize import least_squares

    H0 = np.asarray(H, dtype=float)
    if abs(float(H0[2, 2])) <= 1e-12:
        return H
    H0 = H0 / float(H0[2, 2])
    inl_xy = xy[inliers]
    inl_XY = xyz[inliers, :2]
    n_inl = len(inl_xy)
    XY1 = np.c_[inl_XY, np.ones(n_inl)]
    w_sign = float(np.sign(np.median((XY1 @ H0.T)[:, 2])) or 1.0)
    w_pt = max(1e-3, 1.0 - alpha)
    w_ln = alpha

    def _fun(h8: np.ndarray) -> np.ndarray:
        Hc = np.array(
            [[h8[0], h8[1], h8[2]], [h8[3], h8[4], h8[5]], [h8[6], h8[7], 1.0]],
            dtype=float,
        )
        ph = XY1 @ Hc.T
        denom = np.where(np.abs(ph[:, 2:3]) > 1e-9, ph[:, 2:3], 1e-9)
        uv = ph[:, :2] / denom
        r_pts = w_pt * (uv - inl_xy).ravel()
        # Lift planar Hc to 3x4 P (with sign so z=0 points have positive depth)
        P_planar = w_sign * np.array(
            [
                [Hc[0, 0], Hc[0, 1], 0.0, Hc[0, 2]],
                [Hc[1, 0], Hc[1, 1], 0.0, Hc[1, 2]],
                [Hc[2, 0], Hc[2, 1], 0.0, Hc[2, 2]],
            ],
            dtype=float,
        )
        r_lines = w_ln * _pnl_line_residuals(P_planar, xy, active_lines)
        return np.concatenate([r_pts, r_lines])

    try:
        sol = least_squares(_fun, H0.ravel()[:8], method="trf", loss="soft_l1", max_nfev=25)
    except Exception:
        return H
    H_opt = np.array(
        [
            [sol.x[0], sol.x[1], sol.x[2]],
            [sol.x[3], sol.x[4], sol.x[5]],
            [sol.x[6], sol.x[7], 1.0],
        ],
        dtype=float,
    )
    scale = diag.REF_WIDTH / max(1.0, width)
    ph_opt = XY1 @ H_opt.T
    if not np.all(np.abs(ph_opt[:, 2]) > 1e-9):
        return H
    uv_opt = ph_opt[:, :2] / ph_opt[:, 2:3]
    rmse_opt = float(np.sqrt(np.mean(np.sum((uv_opt - inl_xy) ** 2, axis=1)))) * scale
    if rmse_opt <= s["max_rmse_px"] and diag._orientation(H_opt, inl_XY) > 0:
        return H_opt
    return H


def camera_from_homography(
    H,
    width: float,
    height: float,
    s: dict | None = None,
    obj=None,
    img=None,
    *,
    full_xy: np.ndarray | None = None,
    inliers_mask: np.ndarray | None = None,
) -> dict | None:
    """Pinhole camera (square pixels, centred principal point) behind a field homography.

    The closed form (Zhang) starts a non-linear refinement on the planar
    points ``obj`` (metres) / ``img`` (pixels) when given (>= 4): the closed
    form alone is exact for perfect data but sensitive to label noise.
    Returns ``{"P", "focal_px", "C"}`` or None when the focal or the camera
    position is not physical (camera below the pitch, focal out of range).
    """
    s = _settings(s)
    cx, cy = width / 2.0, height / 2.0
    T = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]])
    G = T @ np.asarray(H, dtype=float)
    G = G / np.linalg.norm(G)
    h1, h2, h3 = G[:, 0], G[:, 1], G[:, 2]
    # Zhang with K = diag(f, f, 1): unknown x = 1/f^2, two linear equations.
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = -np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    den = float(a @ a)
    if not den > 0:
        return None
    x = float(a @ b) / den
    if not x > 0:
        return None
    f = 1.0 / math.sqrt(x)
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    Kinv = np.diag([1.0 / f, 1.0 / f, 1.0])
    m1, m2, m3 = Kinv @ h1, Kinv @ h2, Kinv @ h3
    lam = 2.0 / (np.linalg.norm(m1) + np.linalg.norm(m2))
    if (lam * m3)[2] < 0:  # the pitch centre is in front of the camera
        lam = -lam
    r1, r2, t = lam * m1, lam * m2, lam * m3
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    if obj is not None and img is not None and len(obj) >= 4:
        R, t, f, K = _refine_pinhole(
            obj,
            img,
            width,
            height,
            K,
            R,
            t,
            full_xy=full_xy,
            inliers_mask=inliers_mask,
            pnl_alpha=float(s.get("pnl_alpha", 0.7)),
        )
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    C = -R.T @ t
    if not s["min_cam_height_m"] <= C[2] <= s["max_cam_height_m"]:
        return None
    P = K @ np.c_[R, t]
    return {"P": P / P[2, 3], "focal_px": f, "C": C}


def _refine_pinhole(
    obj,
    img,
    width,
    height,
    K,
    R,
    t,
    *,
    full_xy: np.ndarray | None = None,
    inliers_mask: np.ndarray | None = None,
    pnl_alpha: float = 0.7,
):
    """One-view calibration: focal + pose, principal point / aspect fixed, no distortion.

    When ``full_xy`` and ``inliers_mask`` are provided and ``pnl_alpha > 0``,
    further refines ``(rvec, tvec, f)`` with PnLCalib's joint points + lines
    objective ``(1 - alpha) * err_pts + alpha * err_lines``.
    """
    import cv2

    flags = (
        cv2.CALIB_USE_INTRINSIC_GUESS
        | cv2.CALIB_FIX_PRINCIPAL_POINT
        | cv2.CALIB_FIX_ASPECT_RATIO
        | cv2.CALIB_ZERO_TANGENT_DIST
        | cv2.CALIB_FIX_K1
        | cv2.CALIB_FIX_K2
        | cv2.CALIB_FIX_K3
    )
    try:
        _, K2, _, rvecs, tvecs = cv2.calibrateCamera(
            [np.asarray(obj, dtype=np.float32)],
            [np.asarray(img, dtype=np.float32)],
            (int(width), int(height)),
            K.copy(),
            np.zeros(5),
            flags=flags,
        )
        R2, _ = cv2.Rodrigues(rvecs[0])
        t2 = tvecs[0].ravel()
        f2 = float(K2[0, 0])
    except cv2.error:
        R2, t2, f2, K2 = R, t, float(K[0, 0]), K

    if full_xy is not None and inliers_mask is not None and pnl_alpha > 0.0:
        active_lines = _active_painted_lines(inliers_mask)
        if active_lines:
            from scipy.optimize import least_squares

            obj_arr = np.asarray(obj, dtype=float)[:, :3]
            img_arr = np.asarray(img, dtype=float)[:, :2]
            cx, cy = float(K2[0, 2]), float(K2[1, 2])
            rvec0, _ = cv2.Rodrigues(R2)
            p0 = np.array([*rvec0.ravel(), *t2.ravel(), f2], dtype=float)
            w_pt = max(1e-3, 1.0 - pnl_alpha)
            w_ln = pnl_alpha

            def _pnl_cam_res(params: np.ndarray) -> np.ndarray:
                rv = params[:3]
                tv = params[3:6]
                fc = float(params[6])
                if fc <= 10.0:
                    return np.full(2 * len(obj_arr), 1e4, dtype=float)
                Rm, _ = cv2.Rodrigues(rv)
                Km = np.array([[fc, 0.0, cx], [0.0, fc, cy], [0.0, 0.0, 1.0]], dtype=float)
                Pm = Km @ np.c_[Rm, tv]
                uv_p, front_p = _project(Pm, obj_arr)
                if not front_p.all():
                    return np.full(2 * len(obj_arr), 1e4, dtype=float)
                r_pts = w_pt * (uv_p - img_arr).ravel()
                r_lines = w_ln * _pnl_line_residuals(Pm, full_xy, active_lines)
                return np.concatenate([r_pts, r_lines])

            try:
                r0 = _pnl_cam_res(p0)
                if np.isfinite(r0).all() and float(r0 @ r0) < 1e7:
                    sol = least_squares(_pnl_cam_res, p0, method="trf", loss="soft_l1", max_nfev=20)
                    r1_vec = _pnl_cam_res(sol.x)
                    if np.isfinite(r1_vec).all() and float(r1_vec @ r1_vec) <= float(r0 @ r0):
                        R_opt, _ = cv2.Rodrigues(sol.x[:3])
                        t_opt = sol.x[3:6].ravel()
                        f_opt = float(sol.x[6])
                        K_opt = np.array(
                            [[f_opt, 0.0, cx], [0.0, f_opt, cy], [0.0, 0.0, 1.0]], dtype=float
                        )
                        return R_opt, t_opt, f_opt, K_opt
            except Exception:
                pass

    return R2, t2, f2, K2


def _fit_dlt3d(xy, xyz, use, offplane, width: float, s: dict) -> dict | None:
    """Pruned DLT3D on ``use`` points (needs points off the ground), or None."""
    use = use.copy()
    scale = diag.REF_WIDTH / max(1.0, width)
    res = np.zeros(0)
    L = None
    for _ in range(4):
        if use.sum() < s["min_dlt_points"] or (use & offplane).sum() < s["min_offplane"]:
            return None
        L = fit_dlt3d_hartley_svd(xy[use], xyz[use])
        if L is None:
            L = calculate_dlt3d_params(xy[use], xyz[use])
        uv, front = _project(dlt_projection_matrix(L), xyz[use])
        res = np.linalg.norm(uv - xy[use], axis=1) * scale
        if not front.all():
            return None
        worst = int(np.argmax(res))
        if res[worst] <= s["fit_px"]:
            break
        use[np.flatnonzero(use)[worst]] = False
    else:
        return None
    rmse = float(np.sqrt(np.mean(res**2)))
    if L is None or rmse > s["max_rmse_px"]:
        return None
    try:
        cam = decompose_dlt3d(L)
    except (ValueError, np.linalg.LinAlgError):
        return None
    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
    if not (fx > 0 and fy > 0) or abs(fy / fx - 1.0) > s["max_aspect_dev"]:
        return None
    if abs(float(cam.K[0, 1])) / ((fx + fy) / 2.0) > s["max_skew"]:
        return None
    f = (fx + fy) / 2.0
    if not s["min_focal_ratio"] * width <= f <= s["max_focal_ratio"] * width:
        return None
    if not s["min_cam_height_m"] <= float(cam.C[2]) <= s["max_cam_height_m"]:
        return None
    return {"P": dlt_projection_matrix(L), "focal_px": f, "C": cam.C, "rmse_px": rmse}


def fit_field_camera(xy, ok, width: float, height: float, settings: dict | None = None) -> dict:
    """Camera of one frame from its trusted keypoints ``xy[ok]`` (pixels).

    ``source``: dlt3d (points off the ground used) | homography (pinhole
    camera recovered from the planar homography) | planar (homography only:
    z != 0 points cannot be projected) | none. ``status`` is ok or the reason.
    """
    s = _settings(settings)
    xyz, planar = _field()
    xy = np.asarray(xy, dtype=float).reshape(NKP, 2)
    ok = np.asarray(ok, dtype=bool) & np.isfinite(xy).all(axis=1)
    out: dict = {
        "status": "few_points",
        "source": "none",
        "P": None,
        "H": None,
        "h_sign": 1.0,
        "focal_px": None,
        "cam_height_m": None,
        "rmse_px": None,
        "n_points": int(ok.sum()),
        "inliers": np.zeros(NKP, dtype=bool),
    }
    use = ok & planar
    fit = diag.fit_field_homography(
        xy[use], xyz[use, :2], width=width, ransac_px=s["fit_px"], max_rmse_px=s["max_rmse_px"]
    )
    if fit["status"] != "ok":
        out["status"] = fit["status"]
        return out
    H = fit["H"]
    inliers = np.zeros(NKP, dtype=bool)
    inliers[np.flatnonzero(use)[fit["inliers"]]] = True
    scale = diag.REF_WIDTH / max(1.0, width)
    H_svd = fit_dlt2d_hartley_svd(xy[inliers], xyz[inliers, :2])
    if H_svd is not None:
        ph_svd = np.c_[xyz[inliers, :2], np.ones(int(inliers.sum()))] @ H_svd.T
        if np.all(np.abs(ph_svd[:, 2]) > 1e-12):
            uv_svd = ph_svd[:, :2] / ph_svd[:, 2:3]
            rmse_svd = float(np.sqrt(np.mean(np.sum((uv_svd - xy[inliers]) ** 2, axis=1)))) * scale
            if rmse_svd <= s["max_rmse_px"] and diag._orientation(H_svd, xyz[inliers, :2]) > 0:
                H = H_svd
                fit = fit | {"H": H, "rmse_px": round(rmse_svd, 3)}
    H_pnl = _refine_homography_pnl(H, xy, xyz, inliers, width, s)
    if H_pnl is not H:
        ph_pnl = np.c_[xyz[inliers, :2], np.ones(int(inliers.sum()))] @ H_pnl.T
        uv_pnl = ph_pnl[:, :2] / ph_pnl[:, 2:3]
        rmse_pnl = float(np.sqrt(np.mean(np.sum((uv_pnl - xy[inliers]) ** 2, axis=1)))) * scale
        H = H_pnl
        fit = fit | {"H": H, "rmse_px": round(rmse_pnl, 3)}
    w = (np.c_[xyz[inliers, :2], np.ones(int(inliers.sum()))] @ H.T)[:, 2]
    out.update(
        status="ok",
        source="planar",
        H=H,
        h_sign=float(np.sign(np.median(w)) or 1.0),
        inliers=inliers,
    )
    out["rmse_px"] = fit["rmse_px"]
    offplane = ok & ~planar
    cam = None
    if offplane.sum() >= s["min_offplane"]:
        cam = _fit_dlt3d(xy, np.asarray(xyz), inliers | offplane, offplane, width, s)
        if cam is not None:
            out["source"] = "dlt3d"
    if cam is None:
        cam = camera_from_homography(
            H,
            width,
            height,
            s,
            xyz[inliers],
            xy[inliers],
            full_xy=xy,
            inliers_mask=inliers,
        )
        if cam is not None:
            uv, _ = _project(cam["P"], xyz[inliers])
            rmse = float(np.sqrt(np.mean(np.sum((uv - xy[inliers]) ** 2, axis=1)))) * scale
            if rmse <= s["max_rmse_px"]:
                out["source"] = "homography"
                cam["rmse_px"] = rmse
            else:
                cam = None
    if cam is not None:
        out.update(
            P=cam["P"],
            focal_px=round(float(cam["focal_px"]), 1),
            cam_height_m=round(float(cam["C"][2]), 2),
            rmse_px=round(float(cam["rmse_px"]), 3),
        )
    return out


def project_field(model: dict, width: float, height: float, margin: float = 0.01):
    """``(xy (49, 2), inside (49,))``: every keypoint through the frame's camera.

    ``inside``: projected in front of the camera and inside the image (minus
    ``margin`` of the width). Without a full camera only z = 0 points project.
    """
    xyz, planar = _field()
    xy = np.full((NKP, 2), np.nan)
    front = np.zeros(NKP, dtype=bool)
    if model.get("P") is not None:
        xy, front = _project(model["P"], xyz)
    elif model.get("H") is not None:
        ph = np.c_[xyz[:, :2], np.ones(NKP)] @ np.asarray(model["H"]).T
        with np.errstate(invalid="ignore", divide="ignore"):
            xy = np.where(planar[:, None], ph[:, :2] / ph[:, 2:3], np.nan)
        front = planar & (np.sign(ph[:, 2]) == model["h_sign"])
    m = margin * width
    with np.errstate(invalid="ignore"):
        inside = (
            front
            & np.isfinite(xy).all(axis=1)
            & (xy[:, 0] >= m)
            & (xy[:, 0] <= width - m)
            & (xy[:, 1] >= m)
            & (xy[:, 1] <= height - m)
        )
    return xy, inside


def project_rigid_field(
    model: dict | None,
    width: float,
    height: float,
    *,
    raw_xy=None,
    kc=None,
    codes: list[str] | None = None,
    kp_conf: float = 0.5,
    margin: float = 0.0,
    schema: str = "kiki49",
    settings: dict | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Forward-project the full rigid-body pitch model into image space ``(u, v)``.

    Projects all points of ``schema`` (``"kiki49"`` = 49 points from
    ``soccerfield_kiki.csv``, or ``"ref3d"`` = 37 planar points from
    ``soccerfield_ref3d.csv``) using the frame's camera ``P`` (or planar
    homography ``H`` when ``P`` is unavailable). Keeps finite ``(u, v)``
    coordinates for all points in front of the camera (``w > 0``), including
    occluded and out-of-FOV points, and assigns per-keypoint visibility flags:
      * ``"visible_in_fov"``: inside the image FOV and seen/confirmed (or when
        no detection array is supplied);
      * ``"occluded"``: inside the image FOV on the pitch, but missed/rejected
        or displaced by an obstacle in the detector output;
      * ``"out_of_bounds"``: outside the image FOV or behind the camera.
    """
    if schema == "kiki49":
        xyz, planar = _field()
    elif schema == "ref3d":
        _, xyz = load_ref3d_points()
        planar = np.ones(len(xyz), dtype=bool)
    else:
        raise ValueError(f"Unknown field schema: {schema!r} (expected 'kiki49' or 'ref3d')")
    n_pts = len(xyz)
    if not model or model.get("status") != "ok":
        return np.full((n_pts, 2), np.nan), ["out_of_bounds"] * n_pts

    uv = np.full((n_pts, 2), np.nan)
    front = np.zeros(n_pts, dtype=bool)
    if model.get("P") is not None:
        p_uv, p_front = _project(np.asarray(model["P"], dtype=float), xyz)
        front = p_front & np.isfinite(p_uv).all(axis=1)
        uv = np.where(front[:, None], p_uv, np.nan)
    elif model.get("H") is not None:
        H = np.asarray(model["H"], dtype=float)
        ph = np.c_[xyz[:, :2], np.ones(n_pts)] @ H.T
        h_sign = float(model.get("h_sign", 1.0))
        front = planar & (np.sign(ph[:, 2]) == h_sign) & (np.abs(ph[:, 2]) > 1e-12)
        with np.errstate(invalid="ignore", divide="ignore"):
            p_uv = ph[:, :2] / ph[:, 2:3]
        front &= np.isfinite(p_uv).all(axis=1)
        uv = np.where(front[:, None], p_uv, np.nan)
    else:
        return uv, ["out_of_bounds"] * n_pts

    m = margin * width
    with np.errstate(invalid="ignore"):
        in_fov = (
            front
            & (uv[:, 0] >= m)
            & (uv[:, 0] <= width - m)
            & (uv[:, 1] >= m)
            & (uv[:, 1] <= height - m)
        )

    if kc is None and codes is None:
        flags = ["visible_in_fov" if in_fov[i] else "out_of_bounds" for i in range(n_pts)]
        return uv, flags

    s = _settings(settings)
    scale = diag.REF_WIDTH / max(1.0, width)
    kiki_uv, _ = (
        (uv, front) if schema == "kiki49" else project_field(model, width, height, margin=margin)
    )
    raw_arr = np.asarray(raw_xy, dtype=float).reshape(-1, 2) if raw_xy is not None else None
    kc_arr = np.asarray(kc, dtype=float).ravel() if kc is not None else None

    def _point_seen(idx: int, proj_pt: np.ndarray) -> bool:
        if codes is not None and idx < len(codes):
            c = codes[idx]
            if c == "G":
                return True
            if c == "D":
                if raw_arr is not None and idx < len(raw_arr) and np.isfinite(raw_arr[idx]).all():
                    err = float(np.linalg.norm(raw_arr[idx] - proj_pt)) * scale
                    return err <= s["fix_px"]
                return True
            return False
        if kc_arr is not None and idx < len(kc_arr):
            if not (np.isfinite(kc_arr[idx]) and kc_arr[idx] >= kp_conf):
                return False
            if raw_arr is not None and idx < len(raw_arr) and np.isfinite(raw_arr[idx]).all():
                err = float(np.linalg.norm(raw_arr[idx] - proj_pt)) * scale
                return err <= s["fix_px"]
            return True
        return True

    flags: list[str] = []
    if schema == "kiki49":
        for i in range(n_pts):
            if not in_fov[i]:
                flags.append("out_of_bounds")
            elif _point_seen(i, kiki_uv[i]):
                flags.append("visible_in_fov")
            else:
                flags.append("occluded")
    else:
        ref_map = _ref3d_to_kiki_map()
        direct = (codes is not None and len(codes) == n_pts) or (
            kc_arr is not None and len(kc_arr) == n_pts
        )
        for j in range(n_pts):
            if not in_fov[j]:
                flags.append("out_of_bounds")
            elif direct:
                flags.append("visible_in_fov" if _point_seen(j, uv[j]) else "occluded")
            else:
                seen = any(_point_seen(k_idx, kiki_uv[k_idx]) for k_idx in ref_map[j])
                flags.append("visible_in_fov" if seen else "occluded")
    return uv, flags


def regularize_camera_sequence(
    models: list[dict | None],
    cuts: list[bool] | np.ndarray,
    width: float,
    height: float,
    *,
    xy_seq: list[np.ndarray | None] | None = None,
    window: int | None = None,
    settings: dict | None = None,
) -> list[dict | None]:
    """Shot-aware temporal regularization of per-frame camera matrices ``H`` and ``P``.

    Applies a symmetric triangular rolling filter over normalized ``H``
    (``H[2,2] = 1``) and ``P`` (``P[2,3] = 1``) within each contiguous shot
    (never crossing ``cuts[t] == True``) to suppress high-frequency inter-frame
    jitter while guarding each frame's inlier reprojection RMSE <= ``max_rmse_px``.
    """
    s = _settings(settings)
    win = int(s["smooth_window"] if window is None else window)
    n = len(models)
    if win <= 1 or n <= 1:
        return [dict(m) if isinstance(m, dict) else m for m in models]

    cuts_arr = np.zeros(n, dtype=bool) if cuts is None else np.asarray(cuts, dtype=bool)
    xyz, planar = _field()
    scale = diag.REF_WIDTH / max(1.0, width)
    radius = max(1, win // 2)

    # Partition into contiguous shot ranges [s0, s1)
    shots: list[tuple[int, int]] = []
    start = 0
    for t in range(1, n):
        if cuts_arr[t]:
            shots.append((start, t))
            start = t
    shots.append((start, n))

    out: list[dict | None] = [dict(m) if isinstance(m, dict) else m for m in models]
    for s0, s1 in shots:
        if s1 - s0 <= 1:
            continue
        for t in range(s0, s1):
            m_t = models[t]
            if not m_t or m_t.get("status") != "ok":
                continue
            lo, hi = max(s0, t - radius), min(s1, t + radius + 1)
            inl = m_t.get("inliers")
            xy_t = (
                np.asarray(xy_seq[t], dtype=float).reshape(NKP, 2)
                if xy_seq is not None and t < len(xy_seq) and xy_seq[t] is not None
                else None
            )
            if inl is not None and xy_t is not None:
                inl = np.asarray(inl, dtype=bool) & np.isfinite(xy_t).all(axis=1)
            else:
                inl = None

            # Smooth H
            h_list, h_weights = [], []
            for k in range(lo, hi):
                mk = models[k]
                if mk and mk.get("status") == "ok" and mk.get("H") is not None:
                    Hk = np.asarray(mk["H"], dtype=float)
                    if abs(float(Hk[2, 2])) > 1e-12:
                        h_list.append(Hk / float(Hk[2, 2]))
                        h_weights.append(float(radius + 1 - abs(k - t)))
            H_new = m_t.get("H")
            if len(h_list) > 1 and H_new is not None:
                H_orig = np.asarray(H_new, dtype=float) / float(H_new[2, 2])
                w_arr = np.asarray(h_weights, dtype=float)
                H_bar = np.tensordot(w_arr, np.stack(h_list, axis=0), axes=(0, 0)) / w_arr.sum()
                if abs(float(H_bar[2, 2])) > 1e-12:
                    H_bar = H_bar / float(H_bar[2, 2])
                    inl_p = (inl & planar) if inl is not None else None
                    if inl_p is not None and inl_p.any() and xy_t is not None:
                        for alpha in (1.0, 0.5, 0.25, 0.0):
                            H_cand = alpha * H_bar + (1.0 - alpha) * H_orig
                            H_cand = H_cand / float(H_cand[2, 2])
                            ph = np.c_[xyz[inl_p, :2], np.ones(int(inl_p.sum()))] @ H_cand.T
                            if np.all(np.abs(ph[:, 2]) > 1e-12):
                                uv_h = ph[:, :2] / ph[:, 2:3]
                                rmse_h = (
                                    float(
                                        np.sqrt(np.mean(np.sum((uv_h - xy_t[inl_p]) ** 2, axis=1)))
                                    )
                                    * scale
                                )
                                if rmse_h <= s["max_rmse_px"]:
                                    H_new = H_cand
                                    break
                    else:
                        H_new = H_bar

            # Smooth P
            p_list, p_weights = [], []
            for k in range(lo, hi):
                mk = models[k]
                if mk and mk.get("status") == "ok" and mk.get("P") is not None:
                    Pk = np.asarray(mk["P"], dtype=float)
                    if abs(float(Pk[2, 3])) > 1e-12:
                        p_list.append(Pk / float(Pk[2, 3]))
                        p_weights.append(float(radius + 1 - abs(k - t)))
            P_new = m_t.get("P")
            rmse_new = m_t.get("rmse_px")
            if len(p_list) > 1 and P_new is not None:
                P_orig = np.asarray(P_new, dtype=float) / float(P_new[2, 3])
                w_arr = np.asarray(p_weights, dtype=float)
                P_bar = np.tensordot(w_arr, np.stack(p_list, axis=0), axes=(0, 0)) / w_arr.sum()
                if abs(float(P_bar[2, 3])) > 1e-12:
                    P_bar = P_bar / float(P_bar[2, 3])
                    if inl is not None and inl.any() and xy_t is not None:
                        for alpha in (1.0, 0.5, 0.25, 0.0):
                            P_cand = alpha * P_bar + (1.0 - alpha) * P_orig
                            P_cand = P_cand / float(P_cand[2, 3])
                            uv_p, front_p = _project(P_cand, xyz[inl])
                            if front_p.all():
                                rmse_p = (
                                    float(np.sqrt(np.mean(np.sum((uv_p - xy_t[inl]) ** 2, axis=1))))
                                    * scale
                                )
                                if rmse_p <= s["max_rmse_px"]:
                                    P_new = P_cand
                                    rmse_new = round(rmse_p, 3)
                                    break
                    else:
                        P_new = P_bar

            updated = dict(m_t)
            if H_new is not None:
                updated["H"] = H_new
                if inl is not None and (inl & planar).any():
                    inl_p = inl & planar
                    w_h = (np.c_[xyz[inl_p, :2], np.ones(int(inl_p.sum()))] @ H_new.T)[:, 2]
                    updated["h_sign"] = float(np.sign(np.median(w_h)) or 1.0)
            if P_new is not None:
                updated["P"] = P_new
                if rmse_new is not None:
                    updated["rmse_px"] = rmse_new
                try:
                    cam = decompose_dlt3d(P_new.ravel()[:11])
                    fx, fy = float(cam.K[0, 0]), float(cam.K[1, 1])
                    if fx > 0 and fy > 0:
                        updated["focal_px"] = round((fx + fy) / 2.0, 1)
                        updated["cam_height_m"] = round(float(cam.C[2]), 2)
                except (ValueError, np.linalg.LinAlgError):
                    pass
            out[t] = updated
    return out


@lru_cache(maxsize=1)
def _painted_line_directions() -> dict[int, list[np.ndarray]]:
    """For each planar keypoint on painted lines, the unit 2D world direction(s) of its line(s)."""
    try:
        from . import freekiki_snap as fs
    except ImportError:
        import freekiki_snap as fs  # ty: ignore[unresolved-import]

    xyz, _ = _field()
    line_dirs: dict[str, np.ndarray] = {}
    for name, members in fs.PAINTED_LINES.items():
        pts = xyz[list(members), :2]
        # Take the two members farthest apart to define the exact line direction
        ia, ib = max(
            ((a, b) for a in range(len(pts)) for b in range(a + 1, len(pts))),
            key=lambda ab: float(np.hypot(*(pts[ab[0]] - pts[ab[1]]))),
        )
        d = pts[ib] - pts[ia]
        norm = float(np.hypot(*d))
        if norm > 1e-9:
            line_dirs[name] = d / norm
    per_kp: dict[int, list[np.ndarray]] = {}
    for k in range(NKP):
        dirs = [line_dirs[n] for n in fs.lines_of(k) if n in line_dirs]
        if dirs:
            per_kp[k] = dirs
    return per_kp


def compute_frame_residuals(
    model: dict | None,
    raw_xy: np.ndarray | None,
    snap_xy: np.ndarray | None,
    acc: np.ndarray | list[bool] | None,
    width: float,
    height: float,
) -> dict[str, float | None]:
    """Per-frame reprojection RMSE breaking down raw network vs. white-line snapped points.

    Returns:
      * ``raw_reproj_rmse_px``: raw detected points vs. projected ground-truth model (px@1920)
      * ``raw_reproj_rmse_m``: back-projected raw planar points vs. metric ground truth (m)
      * ``snap_reproj_rmse_px``: white-line snapped points vs. projected ground-truth model (px@1920)
      * ``snap_reproj_rmse_m``: back-projected snapped planar points vs. metric ground truth (m)
      * ``snap_line_rmse_px``: perpendicular distance of snapped points to projected planar field lines (px@1920)
    """
    res: dict[str, float | None] = {
        "raw_reproj_rmse_px": None,
        "raw_reproj_rmse_m": None,
        "snap_reproj_rmse_px": None,
        "snap_reproj_rmse_m": None,
        "snap_line_rmse_px": None,
    }
    if not model or model.get("status") != "ok" or acc is None:
        return res
    acc_mask = np.asarray(acc, dtype=bool).ravel()[:NKP]
    if not acc_mask.any():
        return res

    xyz, planar = _field()
    scale = diag.REF_WIDTH / max(1.0, width)
    proj_uv, _ = project_rigid_field(model, width, height, schema="kiki49")
    proj_ok = np.isfinite(proj_uv).all(axis=1)

    H = model.get("H")
    H_inv = None
    if H is not None:
        try:
            H_inv = np.linalg.inv(np.asarray(H, dtype=float))
        except np.linalg.LinAlgError:
            H_inv = None

    def _backproject_rmse_m(pts_2d: np.ndarray, mask: np.ndarray) -> float | None:
        mask_p = mask & planar
        if H_inv is None or not mask_p.any():
            return None
        ph = np.c_[pts_2d[mask_p], np.ones(int(mask_p.sum()))] @ H_inv.T
        valid_w = np.abs(ph[:, 2]) > 1e-12
        if not valid_w.any():
            return None
        xy_m = ph[valid_w, :2] / ph[valid_w, 2:3]
        gt_m = xyz[mask_p, :2][valid_w]
        err_m = np.linalg.norm(xy_m - gt_m, axis=1)
        return round(float(np.sqrt(np.mean(err_m**2))), 4)

    if raw_xy is not None:
        raw_arr = np.asarray(raw_xy, dtype=float).reshape(NKP, 2)
        v_raw = acc_mask & np.isfinite(raw_arr).all(axis=1) & proj_ok
        if v_raw.any():
            err_px = np.linalg.norm(raw_arr[v_raw] - proj_uv[v_raw], axis=1) * scale
            res["raw_reproj_rmse_px"] = round(float(np.sqrt(np.mean(err_px**2))), 3)
            res["raw_reproj_rmse_m"] = _backproject_rmse_m(raw_arr, v_raw)

    if snap_xy is not None:
        snap_arr = np.asarray(snap_xy, dtype=float).reshape(NKP, 2)
        v_snap = acc_mask & np.isfinite(snap_arr).all(axis=1) & proj_ok
        if v_snap.any():
            err_px = np.linalg.norm(snap_arr[v_snap] - proj_uv[v_snap], axis=1) * scale
            res["snap_reproj_rmse_px"] = round(float(np.sqrt(np.mean(err_px**2))), 3)
            res["snap_reproj_rmse_m"] = _backproject_rmse_m(snap_arr, v_snap)

            # Perpendicular distance to projected planar field lines
            kp_dirs = _painted_line_directions()
            line_errs: list[float] = []
            P = model.get("P")
            for k in np.flatnonzero(v_snap & planar):
                k = int(k)
                dirs = kp_dirs.get(k)
                if not dirs:
                    continue
                p0 = proj_uv[k]
                for u_dir in dirs:
                    pt_step = np.array([[xyz[k, 0] + u_dir[0], xyz[k, 1] + u_dir[1], 0.0]])
                    if P is not None:
                        p1_arr, f1 = _project(np.asarray(P, dtype=float), pt_step)
                        if not f1[0] or not np.isfinite(p1_arr[0]).all():
                            continue
                        p1 = p1_arr[0]
                    elif H is not None:
                        ph1 = (
                            np.array([pt_step[0, 0], pt_step[0, 1], 1.0])
                            @ np.asarray(H, dtype=float).T
                        )
                        if abs(float(ph1[2])) <= 1e-12:
                            continue
                        p1 = ph1[:2] / ph1[2]
                    else:
                        continue
                    d_vec = p1 - p0
                    d_len = float(np.hypot(d_vec[0], d_vec[1]))
                    if d_len <= 1e-9:
                        continue
                    n_vec = np.array([-d_vec[1] / d_len, d_vec[0] / d_len])
                    perp_px = abs(float((snap_arr[k] - p0) @ n_vec)) * scale
                    line_errs.append(perp_px)
            if line_errs:
                res["snap_line_rmse_px"] = round(float(np.sqrt(np.mean(np.square(line_errs)))), 3)
    return res


def dlt_coefficients(model: dict | None) -> tuple[np.ndarray | None, np.ndarray | None]:
    """vailá DLT2D and DLT3D coefficients of a frame camera (``rec2d`` / ``rec3d`` layout).

    DLT2D (8): ground plane (X, Y, Z = 0, metres) -> image homography with
    H[2,2] = 1, ``[H00 H01 H02 H10 H11 H12 H20 H21]`` (same as fifa_to_dlt).
    DLT3D (11): 3x4 camera with P[2,3] = 1, ``L1..L11`` row by row. Only the
    camera sources that model z != 0 (dlt3d, homography) give a DLT3D. World
    frame: origin at the centre spot, x towards the right goal, y towards the
    "top" touchline, z up (soccerfield_kiki.csv).
    """
    if not model or model.get("status") != "ok":
        return None, None
    d2 = d3 = None
    H = model.get("H")
    if H is not None and abs(float(H[2, 2])) > 1e-12:
        d2 = (np.asarray(H, dtype=float) / float(H[2, 2])).ravel()[:8]
    P = model.get("P")
    if P is not None and abs(float(P[2, 3])) > 1e-12:
        d3 = (np.asarray(P, dtype=float) / float(P[2, 3])).ravel()[:11]
    return d2, d3


def refine_keypoints(
    raw_xy,
    raw_kc,
    codes,
    width: float,
    height: float,
    *,
    mode: str,
    kp_conf: float,
    settings=None,
    model: dict | None = None,
):
    """Fill / fix one frame's network keypoints with the field camera.

    ``codes`` are detect's per-point codes (``D`` accepted). Returns
    ``(xy, kc, gcodes, model)``; ``gcodes`` adds G / Gx / Dg (``GEOM_CODES``).
    Nothing changes without a valid camera, or with ``mode='off'``.
    """
    gcodes = list(codes)
    if mode not in GEOM_MODES:
        raise ValueError(f"geometry mode must be one of {GEOM_MODES}")
    if raw_xy is None or raw_kc is None or mode == "off":
        return raw_xy, raw_kc, gcodes, None
    s = _settings(settings)
    xy = np.array(raw_xy, dtype=float).reshape(NKP, 2)
    kc = np.array(raw_kc, dtype=float).reshape(NKP)
    ok = np.array([c == "D" for c in codes])
    if model is None:
        model = fit_field_camera(xy, ok, width, height, s)
    if model["status"] != "ok":
        return xy, kc, gcodes, model
    proj, inside = project_field(model, width, height, s["margin"])
    scale = diag.REF_WIDTH / max(1.0, width)
    with np.errstate(invalid="ignore"):
        dist = np.linalg.norm(xy - proj, axis=1) * scale
    for i in range(NKP):
        if codes[i] == "D":
            if np.isfinite(dist[i]) and dist[i] > s["fix_px"]:
                if mode == "fix" and inside[i]:
                    xy[i] = proj[i]
                    gcodes[i] = "Gx"
                else:
                    gcodes[i] = "Dg"
        elif codes[i] in FILLABLE and inside[i] and np.nan_to_num(kc[i], nan=-1.0) >= s["min_conf"]:
            if not dist[i] <= s["agree_px"]:
                xy[i] = proj[i]
            kc[i] = max(float(kc[i]), kp_conf)
            gcodes[i] = "G"
    return xy, kc, gcodes, model


def refine_predictions(
    pred: dict, *, det_conf: float, kp_conf: float, mode: str, settings=None
) -> tuple[dict, dict]:
    """``refine_keypoints`` on every image of an ``evaluate`` prediction set.

    Returns ``(pred with refined pred_xy / pred_kc, stats)``; images without
    a box above ``det_conf`` are left as they are.
    """
    xy_all = np.array(pred["pred_xy"], dtype=float)
    kc_all = np.array(pred["pred_kc"], dtype=float)
    box = np.nan_to_num(np.asarray(pred["box_conf"], dtype=float), nan=-1.0)
    sources: Counter = Counter()
    fills: Counter = Counter()
    fixes: Counter = Counter()
    for n in range(len(xy_all)):
        if box[n] < det_conf:
            continue
        w, h = float(pred["width"][n]), float(pred["height"][n])
        xy, kc = xy_all[n], kc_all[n]
        codes = [
            "D"
            if np.nan_to_num(kc[i], nan=-1.0) >= kp_conf and 0 <= xy[i, 0] < w and 0 <= xy[i, 1] < h
            else "Rk"
            for i in range(NKP)
        ]
        new_xy, new_kc, gcodes, model = refine_keypoints(
            xy, kc, codes, w, h, mode=mode, kp_conf=kp_conf, settings=settings
        )
        sources[model["source"] if model else "none"] += 1
        fills.update(f"p{i}" for i, c in enumerate(gcodes) if c == "G")
        fixes.update(f"p{i}" for i, c in enumerate(gcodes) if c == "Gx")
        xy_all[n], kc_all[n] = new_xy, new_kc
    n_box = int((box >= det_conf).sum())
    stats = {
        "mode": mode,
        "images_with_box": n_box,
        "camera_sources": dict(sorted(sources.items())),
        "camera_ok_rate": round(
            sum(v for k, v in sources.items() if k != "none") / max(1, n_box), 4
        ),
        "fills": dict(fills.most_common()),
        "fixes": dict(fixes.most_common()),
    }
    return pred | {"pred_xy": xy_all, "pred_kc": kc_all}, stats


def nonplanar_label_outliers(px, vis, width: float, height: float, *, tol_px: float = 15.0):
    """Labelled flag / post-top points far from the camera of the other labels.

    Leave-one-out: for each labelled z != 0 point, the camera is fitted
    without it. Returns ``[(index, residual_px@1920)]`` above ``tol_px``.
    """
    xyz, planar = _field()
    px = np.asarray(px, dtype=float).reshape(NKP, 2)
    vis = np.asarray(vis, dtype=bool)
    scale = diag.REF_WIDTH / max(1.0, width)
    out = []
    for i in np.flatnonzero(vis & ~planar):
        keep = vis.copy()
        keep[i] = False
        model = fit_field_camera(px, keep, width, height)
        if model["status"] != "ok" or model.get("P") is None:
            continue
        uv, front = _project(model["P"], xyz[i : i + 1])
        if front[0]:
            err = float(np.linalg.norm(uv[0] - px[i])) * scale
            if err > tol_px:
                out.append((int(i), round(err, 1)))
    return out
