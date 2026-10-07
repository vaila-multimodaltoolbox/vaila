"""
================================================================================
Script: clicks.py - kiki49_build human clicks upgraded to 49 points
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
    Human pitch32 clicks (kiki points 0-31, same order) upgraded to kiki49.

    Clicks are kept as annotated points; the rest of the in-view points are
    filled from a pitch model: the tracked 3D camera when one exists and agrees
    with the clicks, otherwise a homography fitted to the clicks themselves.

    Points 10/11/18/19 changed meaning between pitch32 builds (goal-area y vs
    penalty-arc intersection). A click is kept only if it lands on the kiki arc
    point: within ``ARC_TOL_M`` on the pitch (homography path) or nearer to the
    arc point's projection than to the legacy point's (camera path).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import Options
from .camera import Camera, apply_h, fit_homography, project
from .kiki49 import ARC_POINTS, AUX3D_POINTS, LEGACY_GOAL_Y_POINTS_XY, N_KPT, load_kiki49
from .label_io import MIN_VISIBLE
from .line_support import line_distance_map, support_score
from .sample import Reject, Sample, camera_from_plane, complete_keypoints, project_plane, px_scale

ARC_TOL_M = 1.0
RANSAC_PX = 3.0  # px at 1280
MAX_INLIER_MEDIAN = 4.0  # px at 1280: clicks vs their own homography
MIN_FIT_POINTS = 5  # four points fit any homography exactly; demand redundancy
MAX_AGREEMENT = 6.0  # px at 1280: tracked camera vs clicks (median)
MAX_CLICK_OUTLIER = 20.0  # px at 1280: single click vs an agreeing camera
MAX_CAMERA_DISAGREEMENT = 2.0  # px at 1280: pinhole camera vs click homography

_LEGACY = np.array([[*LEGACY_GOAL_Y_POINTS_XY[k], 0.0] for k in ARC_POINTS])


def as_annotated(clicks: np.ndarray) -> np.ndarray:
    """(32, 3) pitch32 or (49, 3) already-mapped clicks -> (49, 3) with v in {0, 2}."""
    ann = np.zeros((N_KPT, 3))
    ann[: len(clicks)] = clicks
    ann[ann[:, 2] > 0, 2] = 2
    ann[ann[:, 2] <= 0] = 0.0
    return ann


def camera_agreement(ann: np.ndarray, cam: Camera, width: int) -> tuple[float, np.ndarray, int]:
    """Median click error of a camera (px at 1280) after the arc check.

    Returns (agreement, annotated keypoints with rejected clicks removed,
    number of clicks on the legacy goal-area-y points). NaN agreement when
    fewer than two clicks remain.
    """
    ann = ann.copy()
    s = px_scale(width)
    uv, _ = project(load_kiki49().xyz, cam)
    legacy_uv, _ = project(_LEGACY, cam)
    n_legacy = 0
    for k, luv in zip(ARC_POINTS, legacy_uv, strict=True):
        if ann[k, 2] > 0 and np.hypot(*(ann[k, :2] - luv)) < np.hypot(*(ann[k, :2] - uv[k])):
            ann[k] = 0.0
            n_legacy += 1
    idx = np.flatnonzero(ann[:, 2] > 0)
    if len(idx) < 2:
        return float("nan"), ann, n_legacy
    err = np.hypot(*(uv[idx] - ann[idx, :2]).T) / s
    agreement = float(np.median(err))
    if agreement < MAX_AGREEMENT:
        ann[idx[err > MAX_CLICK_OUTLIER]] = 0.0
    return agreement, ann, n_legacy


def clicks_to_sample(
    source: str,
    uid: str,
    split: str,
    group: str,
    clicks32: np.ndarray,
    img: np.ndarray,
    image_path: Path,
    opts: Options,
    camera: Camera | None = None,
) -> Sample | Reject:
    height, width = img.shape[:2]
    s = px_scale(width)
    ann = as_annotated(clicks32)
    stats: dict[str, float] = {"n_clicked": float((ann[:, 2] > 0).sum())}
    dist_map = line_distance_map(img)

    def accept(kps, origin, H, cam, aux3d, qa) -> Sample | Reject:
        if int((kps[:, 2] > 0).sum()) < MIN_VISIBLE:
            return Reject(source, uid, "few_visible", stats)
        return Sample(
            source,
            uid,
            split,
            group,
            width,
            height,
            kps,
            origin,
            image_path,
            H,
            cam,
            aux3d,
            qa,
            stats,
        )

    if camera is not None:
        agreement, cam_ann, n_legacy = camera_agreement(ann, camera, width)
        stats["click_agreement"] = agreement
        stats["legacy_arc"] = float(n_legacy)
        if agreement < MAX_AGREEMENT:
            qa = support_score(dist_map, lambda X: project(X, camera))
            stats["support"] = qa
            if qa >= opts.tau:
                kps, origin = complete_keypoints(cam_ann, width, height, None, camera, opts.aux3d)
                return accept(kps, origin, None, camera, opts.aux3d, qa)

    xyz = load_kiki49().xyz
    idx = np.flatnonzero(ann[:, 2] > 0)
    # Only ground points fit the plane; flags / post tops / net ground need a camera.
    aux_idx = np.array([k for k in idx if k in AUX3D_POINTS], dtype=int)
    idx = np.array([k for k in idx if k not in AUX3D_POINTS], dtype=int)
    if len(idx) < 4:
        return Reject(source, uid, "few_clicks", stats)
    base = np.array([k for k in idx if k not in ARC_POINTS], dtype=int)
    fit_idx = base if len(base) >= 4 else idx
    H, inliers = fit_homography(xyz[fit_idx, :2], ann[fit_idx, :2], RANSAC_PX * s)
    if H is None:
        return Reject(source, uid, "no_homography", stats)
    keep = set(fit_idx[inliers].tolist())
    n_legacy = 0
    Hinv = np.linalg.inv(H)
    for j, k in enumerate(ARC_POINTS):
        if ann[k, 2] <= 0:
            continue
        world = apply_h(Hinv, ann[k : k + 1, :2])[0]
        if np.hypot(*(world - xyz[k, :2])) <= ARC_TOL_M:
            keep.add(k)
        else:
            keep.discard(k)
            n_legacy += int(np.hypot(*(world - _LEGACY[j, :2])) <= ARC_TOL_M)
    stats["legacy_arc"] = float(n_legacy)
    keep_idx = np.array(sorted(keep), dtype=int)
    if len(keep_idx) < MIN_FIT_POINTS:
        return Reject(source, uid, "few_clicks", stats)
    H, _ = fit_homography(xyz[keep_idx, :2], ann[keep_idx, :2], None)
    if H is None:
        return Reject(source, uid, "no_homography", stats)
    res = np.hypot(*(apply_h(H, xyz[keep_idx, :2]) - ann[keep_idx, :2]).T) / s
    stats["click_residual"] = float(np.median(res))
    if stats["click_residual"] > MAX_INLIER_MEDIAN:
        return Reject(source, uid, "residual", stats)
    qa = support_score(dist_map, lambda X: project_plane(H, X[:, :2], width, height))
    stats["support"] = qa
    if not qa >= opts.tau:
        return Reject(source, uid, "line_support", stats)
    cam, disagreement = camera_from_plane(H, width, height)
    stats["cam_disagreement"] = disagreement
    aux3d = opts.aux3d and cam is not None and disagreement <= MAX_CAMERA_DISAGREEMENT
    dropped = np.setdiff1d(idx, keep_idx)
    ann[dropped] = 0.0
    if len(aux_idx):
        # Off-ground clicks survive only when a trusted camera confirms them.
        ok_aux = np.zeros(len(aux_idx), dtype=bool)
        if aux3d and cam is not None:
            uv, _ = project(xyz[aux_idx], cam)
            ok_aux = np.hypot(*(uv - ann[aux_idx, :2]).T) / s <= MAX_CLICK_OUTLIER
        ann[aux_idx[~ok_aux]] = 0.0
    kps, origin = complete_keypoints(ann, width, height, H, cam, aux3d)
    return accept(kps, origin, H, cam, aux3d, qa)
