"""
================================================================================
Script: freekiki_tracking.py - frame-to-frame propagation of the 49 field keypoints
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 25 October 2026
Update Date: 26 October 2026

Description:
    Each frame of ``detect`` is predicted on its own, so a keypoint can jump
    between frames or vanish for a frame when the network is unsure. This module
    keeps the accepted point of the previous frame as a prior:

    * a new detection replaces the prior only when it is confident and lies
      within ``band_px`` of it (the same keypoint, not a neighbour);
    * otherwise the prior is kept and the point is marked ``propagated``;
    * a point with neither a detection nor a prior stays ``none``.

    :func:`propagate_keypoints` is pure NumPy, so it is tested without a model.
"""

from __future__ import annotations

import numpy as np

SOURCE_DETECTED = "detected"
SOURCE_PROPAGATED = "propagated"
SOURCE_NONE = "none"


def propagate_keypoints(
    prev_xy: np.ndarray,
    prev_ok: np.ndarray,
    xy: np.ndarray,
    kc: np.ndarray,
    *,
    kp_conf: float,
    band_px: float,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """Merge the new detection with the previous frame's accepted points.

    ``prev_xy`` / ``xy`` are ``(K, 2)`` pixel coordinates, ``prev_ok`` is a
    ``(K,)`` bool mask of accepted previous points, ``kc`` the ``(K,)`` keypoint
    confidences of the new frame. Returns ``(out_xy, out_ok, sources)`` where
    ``sources[i]`` is ``detected``, ``propagated`` or ``none``.
    """
    prev_xy = np.asarray(prev_xy, dtype=np.float64).reshape(-1, 2)
    xy = np.asarray(xy, dtype=np.float64).reshape(-1, 2)
    kc = np.nan_to_num(np.asarray(kc, dtype=np.float64).reshape(-1), nan=-1.0)
    prev_ok = np.asarray(prev_ok, dtype=bool).reshape(-1)
    k = xy.shape[0]
    if prev_xy.shape[0] != k or prev_ok.shape[0] != k or kc.shape[0] != k:
        raise ValueError("prev_xy, prev_ok, xy and kc must describe the same keypoints")

    out_xy = np.full((k, 2), np.nan, dtype=np.float64)
    out_ok = np.zeros(k, dtype=bool)
    sources: list[str] = []
    for i in range(k):
        finite = bool(np.isfinite(xy[i]).all())
        detected = finite and kc[i] >= kp_conf
        near_prior = (
            detected
            and bool(prev_ok[i])
            and bool(np.isfinite(prev_xy[i]).all())
            and float(np.linalg.norm(xy[i] - prev_xy[i])) <= band_px
        )
        if detected and (not prev_ok[i] or near_prior):
            out_xy[i] = xy[i]
            out_ok[i] = True
            sources.append(SOURCE_DETECTED)
        elif prev_ok[i] and np.isfinite(prev_xy[i]).all():
            out_xy[i] = prev_xy[i]
            out_ok[i] = True
            sources.append(SOURCE_PROPAGATED)
        else:
            sources.append(SOURCE_NONE)
    return out_xy, out_ok, sources


def track_sequence(
    frames_xy: list[np.ndarray],
    frames_kc: list[np.ndarray],
    *,
    kp_conf: float,
    band_px: float,
) -> list[tuple[np.ndarray, np.ndarray, list[str]]]:
    """Apply :func:`propagate_keypoints` frame by frame, starting with no prior."""
    if len(frames_xy) != len(frames_kc):
        raise ValueError("frames_xy and frames_kc must have the same length")
    results: list[tuple[np.ndarray, np.ndarray, list[str]]] = []
    prev_xy = np.full((0, 2), np.nan)
    prev_ok = np.zeros(0, dtype=bool)
    for xy, kc in zip(frames_xy, frames_kc, strict=True):
        n = np.asarray(xy).reshape(-1, 2).shape[0]
        if prev_xy.shape[0] != n:
            prev_xy = np.full((n, 2), np.nan)
            prev_ok = np.zeros(n, dtype=bool)
        out_xy, out_ok, sources = propagate_keypoints(
            prev_xy, prev_ok, xy, kc, kp_conf=kp_conf, band_px=band_px
        )
        results.append((out_xy, out_ok, sources))
        prev_xy, prev_ok = out_xy, out_ok
    return results
