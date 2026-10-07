"""
================================================================================
Script: line_support.py - kiki49_build line-support quality gate
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
    Image evidence for a projected pitch model: fraction of projected lines on white paint.

    One QA score for every projected source: sample the painted markings
    (``kiki49.ground_lines``), project them, and count samples within ``tol_px``
    of a bright thin structure (morphological top-hat of the grey image).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence

import cv2
import numpy as np
from scipy.ndimage import map_coordinates
from scipy.optimize import least_squares
from scipy.spatial.transform import Rotation

from .camera import Camera, in_view, project
from .kiki49 import ground_lines

REF_WIDTH = 1280.0
MIN_SAMPLES = 40


def grass_mask(image_bgr: np.ndarray) -> np.ndarray:
    """Pitch region: green pixels, closed so the painted lines inside it are included."""
    hsv = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2HSV)
    green = cv2.inRange(hsv, (30, 40, 30), (90, 255, 255))
    k = max(3, int(round(31 * image_bgr.shape[1] / REF_WIDTH)) | 1)
    closed = cv2.morphologyEx(
        green, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    return closed > 0


def line_distance_map(image_bgr: np.ndarray, grass_only: bool = False) -> np.ndarray:
    """Distance (px) from every pixel to the nearest white-line pixel.

    ``grass_only`` ignores bright structures off the pitch (advertising boards,
    crowd); it needs a colour image.
    """
    gray = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2GRAY) if image_bgr.ndim == 3 else image_bgr
    scale = gray.shape[1] / REF_WIDTH
    k = max(3, int(round(15 * scale)) | 1)
    tophat = cv2.morphologyEx(
        gray, cv2.MORPH_TOPHAT, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
    )
    mask = (tophat > 25).astype(np.uint8)
    if grass_only:
        mask &= grass_mask(image_bgr).astype(np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((2, 2), np.uint8))
    return cv2.distanceTransform(1 - mask, cv2.DIST_L2, 3)


def sample_ground_lines(
    project_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    width: int,
    height: int,
    step_px: float = 4.0,
    lines: Sequence[np.ndarray] | None = None,
) -> np.ndarray:
    """Projected marking samples inside the image, thinned to ~``step_px`` spacing.

    ``lines`` defaults to every painted marking (``kiki49.ground_lines``).
    """
    pts = []
    for poly in ground_lines() if lines is None else lines:
        uv, depth = project_fn(poly)
        ok = np.isfinite(uv).all(axis=1) & (depth > 0)
        ok &= (uv[:, 0] >= 0) & (uv[:, 0] < width) & (uv[:, 1] >= 0) & (uv[:, 1] < height)
        uv = uv[ok]
        if len(uv) == 0:
            continue
        keep = [0]
        for i in range(1, len(uv)):
            if np.hypot(*(uv[i] - uv[keep[-1]])) >= step_px:
                keep.append(i)
        pts.append(uv[keep])
    return np.concatenate(pts) if pts else np.zeros((0, 2))


def support_score(
    dist_map: np.ndarray,
    project_fn: Callable[[np.ndarray], tuple[np.ndarray, np.ndarray]],
    tol_px: float = 3.0,
    lines: Sequence[np.ndarray] | None = None,
) -> float:
    """Fraction of projected marking samples within ``tol_px`` (scaled to width 1280) of paint.

    Returns NaN when fewer than ``MIN_SAMPLES`` samples fall inside the image.
    """
    h, w = dist_map.shape
    uv = sample_ground_lines(project_fn, w, h, step_px=4.0 * w / REF_WIDTH, lines=lines)
    if len(uv) < MIN_SAMPLES:
        return float("nan")
    d = dist_map[uv[:, 1].astype(int), uv[:, 0].astype(int)]
    return float(np.mean(d <= tol_px * w / REF_WIDTH))


def refine_camera_on_lines(
    cam: Camera,
    dist_map: np.ndarray,
    world_xyz: np.ndarray,
    px: np.ndarray,
    click_weight: float = 1.0,
    iterations: int = 2,
) -> Camera:
    """Fit (f, rotation, t) of an undistorted camera to the painted lines and clicks.

    Chamfer alignment: the projected marking samples (``ground_lines``) should
    sit on white paint, i.e. at distance ~0 in ``dist_map``; ``world_xyz`` ->
    ``px`` are clicked correspondences weighted by ``click_weight``. Robust
    loss, so occluded or unpainted markings do not pull the fit. The in-view
    marking set is re-selected ``iterations`` times as the camera moves.
    """
    h, w = dist_map.shape
    scale = w / REF_WIDTH
    lines = np.concatenate(ground_lines(0.5))
    cx, cy = cam.K[0, 2], cam.K[1, 2]
    px = np.asarray(px, dtype=np.float64)

    def unpack(p: np.ndarray) -> Camera:
        K = np.array([[p[0], 0.0, cx], [0.0, p[0], cy], [0.0, 0.0, 1.0]])
        return Camera(K=K, R=Rotation.from_rotvec(p[1:4]).as_matrix(), t=p[4:7])

    for _ in range(iterations):
        uv, depth = project(lines, cam)
        X = lines[in_view(uv, depth, w, h)]
        if len(X) < MIN_SAMPLES:
            return cam

        def residual(p: np.ndarray, X: np.ndarray = X) -> np.ndarray:
            c = unpack(p)
            uv, _ = project(X, c)
            uv = np.nan_to_num(uv, nan=-1e4)
            inside = np.clip(uv, 0, [w - 1, h - 1])
            d = map_coordinates(dist_map, [inside[:, 1], inside[:, 0]], order=1)
            d = d + np.hypot(*(uv - inside).T)
            r_clicks = (project(world_xyz, c)[0] - px).ravel() * click_weight
            return np.concatenate([d / np.sqrt(len(X) / max(len(px), 1)), r_clicks])

        p0 = np.concatenate([[cam.K[0, 0]], Rotation.from_matrix(cam.R).as_rotvec(), cam.t])
        sol = least_squares(residual, p0, loss="soft_l1", f_scale=3.0 * scale, x_scale="jac")
        cam = unpack(sol.x)
    return cam
