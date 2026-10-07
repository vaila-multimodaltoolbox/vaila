"""
================================================================================
Script: sample.py - kiki49_build labelled sample and 49-point completion
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
    One labelled image and the rule that completes its 49 keypoints.

    Every source produces annotated points (clicked, or intersected from line
    annotations) plus a model of the pitch in that image: a metric ground-plane
    homography ``H`` and/or a full ``Camera``. ``complete_keypoints`` keeps the
    annotated points and fills every other kiki point that projects inside the
    image, so a keypoint missing from a label always means "not in view" (the
    YOLO-Pose ``kobj`` loss treats v=0 as invisible, which matters for tracking).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from .camera import Camera, apply_h, camera_from_homography, in_view, project, refine_camera
from .kiki49 import AUX3D_POINTS, N_KPT, ground_lines, load_kiki49

# Per-keypoint provenance written to the manifest/report.
ORIGIN_NONE = 0
ORIGIN_ANNOTATED = 1  # human click or human line-annotation intersection
ORIGIN_PLANE = 2  # projected through the ground-plane homography
ORIGIN_CAMERA = 3  # projected through a 3D camera

REF_WIDTH = 1280.0  # pixel tolerances in this package are quoted at this width


@dataclass
class Sample:
    source: str
    uid: str  # unique within the source; becomes the file stem
    split: str
    group: str  # match / clip id: never shared between splits
    width: int
    height: int
    kps: np.ndarray  # (49, 3) pixels + visibility (0 or 2)
    origin: np.ndarray  # (49,) ORIGIN_* codes
    image: Path | None = None
    H: np.ndarray | None = None  # metric ground plane (x, y) -> pixels
    camera: Camera | None = None
    aux3d: bool = False  # camera trusted for the z > 0 / net points
    qa: float = float("nan")
    stats: dict[str, float] = field(default_factory=dict)


@dataclass
class Reject:
    source: str
    uid: str
    reason: str
    stats: dict[str, float] = field(default_factory=dict)


def px_scale(width: int) -> float:
    """Multiply a tolerance quoted at 1280 px by this to get pixels at ``width``."""
    return width / REF_WIDTH


def plane_front_sign(H: np.ndarray, width: int, height: int) -> float:
    """Sign that makes the homogeneous w of ``H @ [x, y, 1]`` a positive depth.

    The pixel at the bottom centre of a broadcast frame is below the horizon,
    so its ground point is in front of the camera; its w fixes the sign.
    """
    ref = apply_h(np.linalg.inv(H), np.array([[width / 2.0, 0.9 * height]]))[0]
    w = float(H[2] @ np.array([ref[0], ref[1], 1.0]))
    return 1.0 if w >= 0 else -1.0


def ground_front_point(H: np.ndarray, width: int, height: int) -> np.ndarray:
    """(1, 2) world point in front of the camera (for ``camera_from_homography``)."""
    return apply_h(np.linalg.inv(H), np.array([[width / 2.0, 0.9 * height]]))


def project_plane(
    H: np.ndarray, xy: np.ndarray, width: int, height: int
) -> tuple[np.ndarray, np.ndarray]:
    """Project ground points through ``H``. Returns (uv, signed depth proxy)."""
    p = np.column_stack([xy, np.ones(len(xy))]) @ H.T
    w = p[:, 2] * plane_front_sign(H, width, height)
    with np.errstate(divide="ignore", invalid="ignore"):
        uv = p[:, :2] / p[:, 2:3]
    return uv, w


def camera_from_plane(H: np.ndarray, width: int, height: int) -> tuple[Camera | None, float]:
    """Full camera agreeing with a ground-plane homography.

    Seeds with ``camera_from_homography`` (centred principal point, square
    pixels) and refines (f, R, t) on the in-view pitch markings mapped through
    ``H``. Returns (camera, median disagreement with ``H`` at 1280 px); the
    disagreement measures how well the pinhole model explains ``H``.
    """
    front = ground_front_point(H, width, height)
    cam = camera_from_homography(H, width, height, front)
    if cam is None:
        return None, float("nan")
    X = np.concatenate(ground_lines(1.0))
    uv, depth = project_plane(H, X[:, :2], width, height)
    ok = in_view(uv, depth, width, height)
    if ok.sum() < 20:
        return None, float("nan")
    X, uv = X[ok], uv[ok]
    try:
        cam = refine_camera(cam, X, uv)
    except ValueError:
        return None, float("nan")
    got, d = project(X, cam)
    if np.any(d <= 0):
        return None, float("nan")
    return cam, float(np.median(np.hypot(*(got - uv).T))) / px_scale(width)


def complete_keypoints(
    annotated: np.ndarray,
    width: int,
    height: int,
    H: np.ndarray | None,
    camera: Camera | None,
    aux3d: bool,
) -> tuple[np.ndarray, np.ndarray]:
    """Annotated points first, then planar points from the camera (if it models
    lens distortion) or ``H``, then z > 0 / net points from a trusted camera.

    ``annotated`` is (49, 3) pixels with v > 0 where a human placed the point.
    Only points inside the image are kept. Returns (kps (49, 3), origin (49,)).
    """
    xyz = load_kiki49().xyz
    kps = np.zeros((N_KPT, 3))
    origin = np.full(N_KPT, ORIGIN_NONE, dtype=np.int8)
    aux = np.zeros(N_KPT, dtype=bool)
    aux[list(AUX3D_POINTS)] = True

    if camera is not None and (camera.dist is not None or H is None):
        uv, depth = project(xyz, camera)
        ok = in_view(uv, depth, width, height) & ~aux
        kps[ok, :2], kps[ok, 2], origin[ok] = uv[ok], 2, ORIGIN_CAMERA
    elif H is not None:
        uv, depth = project_plane(H, xyz[:, :2], width, height)
        ok = in_view(uv, depth, width, height) & ~aux
        kps[ok, :2], kps[ok, 2], origin[ok] = uv[ok], 2, ORIGIN_PLANE
    if aux3d and camera is not None:
        uv, depth = project(xyz, camera)
        ok = in_view(uv, depth, width, height) & aux
        kps[ok, :2], kps[ok, 2], origin[ok] = uv[ok], 2, ORIGIN_CAMERA

    ann = annotated[:, 2] > 0
    inside = ann & in_view(annotated[:, :2], np.ones(N_KPT), width, height)
    kps[inside, :2], kps[inside, 2], origin[inside] = annotated[inside, :2], 2, ORIGIN_ANNOTATED
    # An annotated point outside the image is authoritative about "not in view".
    drop = ann & ~inside
    kps[drop], origin[drop] = 0.0, ORIGIN_NONE
    return kps, origin
