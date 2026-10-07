"""
================================================================================
Script: camera.py - kiki49_build camera model (pinhole, homography, refinement)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Ported from mkvis3d ``openbiomech/soccer_field`` (the builder of the kiki49
dataset) so FreeKiki can convert external field datasets by itself.

Description:
    Pinhole cameras for the pitch: homography fit, camera-from-homography, projection.

    Projection follows the FIFA Skeletal Tracking starter kit
    (``lib/camera_tracker.py::_project_pitch_points``): ``cv2.projectPoints`` with
    OpenCV distortion ``(k1, k2, p1, p2, k3)``; x_cam = R X + t.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

F_MIN_REL = 0.3  # plausible focal range, in image widths
F_MAX_REL = 20.0


@dataclass(frozen=True)
class Camera:
    K: np.ndarray  # (3, 3)
    R: np.ndarray  # (3, 3) world -> camera
    t: np.ndarray  # (3,)
    dist: np.ndarray | None = None  # OpenCV (k1, k2, p1, p2, k3) or None

    def to_json(self) -> dict[str, list]:
        out = {"K": self.K.tolist(), "R": self.R.tolist(), "t": self.t.tolist()}
        if self.dist is not None:
            out["dist"] = self.dist.tolist()
        return out


def apply_h(H: np.ndarray, xy: np.ndarray) -> np.ndarray:
    """Map (N, 2) points through a 3x3 homography (NaN where w ~ 0)."""
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    w = p[:, 2:3]
    with np.errstate(divide="ignore", invalid="ignore"):
        out = p[:, :2] / w
    out[np.abs(w[:, 0]) < 1e-12] = np.nan
    return out


def fit_homography(
    world_xy: np.ndarray, px: np.ndarray, ransac_px: float | None = 3.0
) -> tuple[np.ndarray | None, np.ndarray]:
    """Ground-plane homography world (x, y) -> pixels. Returns (H, inlier mask).

    RANSAC with ``ransac_px`` reprojection threshold; ``None`` fits all points
    by least squares.
    """
    n = len(world_xy)
    if n < 4:
        return None, np.zeros(n, dtype=bool)
    robust = ransac_px is not None and n > 4
    H, mask = cv2.findHomography(
        world_xy.astype(np.float64),
        px.astype(np.float64),
        cv2.RANSAC if robust else 0,
        ransac_px if robust else 3.0,
    )
    if H is None or not np.all(np.isfinite(H)):
        return None, np.zeros(n, dtype=bool)
    inliers = mask.ravel().astype(bool) if mask is not None else np.ones(n, dtype=bool)
    return H / H[2, 2], inliers


def homography_from_camera(cam: Camera) -> np.ndarray:
    """Ground-plane (z=0) homography of an undistorted camera: K [r1 r2 t]."""
    H = cam.K @ np.column_stack([cam.R[:, 0], cam.R[:, 1], cam.t])
    return H / H[2, 2]


def camera_from_homography(
    H: np.ndarray, width: int, height: int, front_xy: np.ndarray
) -> Camera | None:
    """Recover K (square pixels, centred principal point), R, t from a plane homography.

    ``f`` solves Zhang's two orthogonality constraints on ``K^-1 H`` in least
    squares; R is the nearest rotation to [r1 r2 r1 x r2]. ``front_xy`` are
    ground points known to be in front of the camera (they fix the sign).
    Returns None when the focal is not finite or outside a plausible range.
    """
    cx, cy = width / 2.0, height / 2.0
    Hn = np.array([[1.0, 0.0, -cx], [0.0, 1.0, -cy], [0.0, 0.0, 1.0]]) @ H
    Hn = Hn / np.linalg.norm(Hn)
    h1, h2 = Hn[:, 0], Hn[:, 1]
    # (h1x h2x + h1y h2y) u + h1z h2z = 0 ;  (|h1xy|^2 - |h2xy|^2) u + h1z^2 - h2z^2 = 0 ; u = 1/f^2
    a = np.array([h1[0] * h2[0] + h1[1] * h2[1], h1[0] ** 2 + h1[1] ** 2 - h2[0] ** 2 - h2[1] ** 2])
    b = np.array([h1[2] * h2[2], h1[2] ** 2 - h2[2] ** 2])
    denom = float(a @ a)
    if denom <= 0.0:
        return None
    u = -float(a @ b) / denom
    if not np.isfinite(u) or u <= 0.0:
        return None
    f = 1.0 / np.sqrt(u)
    if not F_MIN_REL * width <= f <= F_MAX_REL * width:
        return None
    K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
    A = np.linalg.solve(K, H)
    lam = 2.0 / (np.linalg.norm(A[:, 0]) + np.linalg.norm(A[:, 1]))
    depth = (np.column_stack([front_xy, np.ones(len(front_xy))]) @ A[2]) * lam
    if np.median(depth) < 0:
        lam = -lam
    r1, r2, t = lam * A[:, 0], lam * A[:, 1], lam * A[:, 2]
    U, _, Vt = np.linalg.svd(np.column_stack([r1, r2, np.cross(r1, r2)]))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        return None
    return Camera(K=K, R=R, t=t)


def refine_camera(cam: Camera, world_xyz: np.ndarray, px: np.ndarray) -> Camera:
    """Least-squares refinement of (f, rotation, t) on 3D-2D correspondences (no distortion)."""
    rv0 = Rotation.from_matrix(cam.R).as_rotvec()
    cx, cy = cam.K[0, 2], cam.K[1, 2]

    def unpack(p: np.ndarray) -> Camera:
        K = np.array([[p[0], 0.0, cx], [0.0, p[0], cy], [0.0, 0.0, 1.0]])
        return Camera(K=K, R=Rotation.from_rotvec(p[1:4]).as_matrix(), t=p[4:7])

    def residual(p: np.ndarray) -> np.ndarray:
        uv, _ = project(world_xyz, unpack(p))
        return (uv - px).ravel()

    p0 = np.concatenate([[cam.K[0, 0]], rv0, cam.t])
    sol = least_squares(residual, p0, loss="soft_l1", f_scale=2.0, x_scale="jac")
    return unpack(sol.x)


def project(X: np.ndarray, cam: Camera) -> tuple[np.ndarray, np.ndarray]:
    """Project (N, 3) world points. Returns (uv (N, 2), camera depth (N,))."""
    X = np.asarray(X, dtype=np.float64)
    depth = X @ cam.R[2] + cam.t[2]
    if cam.dist is None:
        Xc = X @ cam.R.T + cam.t
        with np.errstate(divide="ignore", invalid="ignore"):
            uv = (Xc @ cam.K.T)[:, :2] / Xc[:, 2:3]
        return uv, depth
    rvec, _ = cv2.Rodrigues(cam.R)
    uv, _ = cv2.projectPoints(X.reshape(-1, 1, 3), rvec, cam.t.astype(np.float64), cam.K, cam.dist)
    return uv.reshape(-1, 2), depth


def in_view(uv: np.ndarray, depth: np.ndarray, width: int, height: int) -> np.ndarray:
    ok = np.isfinite(uv).all(axis=1) & (depth > 0.0)
    return ok & (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
