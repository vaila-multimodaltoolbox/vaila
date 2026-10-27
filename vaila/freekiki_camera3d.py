"""
================================================================================
Script: freekiki_camera3d.py - per-frame focal length from the FreeKiki DLT3D
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
    ``detect`` writes one DLT3D row per frame (``<video>.dlt3d``: frame, L1..L11).
    This module turns those rows into the per-frame pinhole intrinsics that
    ``sam3dinov3`` accepts through ``cam_int``.

    Only the **focal length** is taken from each DLT3D. The principal point is
    kept at the image centre and the skew at zero, because the full RQ
    decomposition of these DLT3D is not stable on broadcast footage: on the
    Paqueta clip the principal point jumps from (777, 939) to (482, 329) px and
    the skew is 11-73 px across four frames, while the focal stays between 7010
    and 7422 px. The focal is the quantity that varies smoothly with zoom, so it
    is the one that is trusted.

    * :func:`focal_from_dlt3d` - focal (px) of one DLT3D, or None if invalid;
    * :func:`focal_sequence` - focal per frame of a ``.dlt3d`` file (NaN = missing);
    * :func:`cam_int_sequence` - ``(n, 3, 3)`` intrinsics, NaN rows where missing;
    * :func:`write_cam_int_npy` - the sequence in the format ``sam3dinov3`` reads.

    Frames without a usable DLT3D (for example a camera from a homography only)
    stay NaN; ``sam3dinov3`` then uses its default intrinsics for those frames.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

try:
    from .monocular_dlt_align import decompose_dlt3d, load_dlt3d_file
except ImportError:
    from monocular_dlt_align import decompose_dlt3d, load_dlt3d_file  # ty: ignore[unresolved-import]

# A focal outside this band (in units of the image width) is a failed fit, not a lens.
MIN_FOCAL_RATIO = 0.3
MAX_FOCAL_RATIO = 10.0


def focal_from_dlt3d(L: np.ndarray, width: float) -> float | None:
    """Focal length in px of one DLT3D (11 coefficients), or None when it is not usable."""
    try:
        cam = decompose_dlt3d(np.asarray(L, dtype=np.float64))
    except (ValueError, np.linalg.LinAlgError):
        return None
    focal = 0.5 * float(cam.K[0, 0] + cam.K[1, 1])
    if not np.isfinite(focal) or focal <= 0:
        return None
    if not MIN_FOCAL_RATIO * width <= focal <= MAX_FOCAL_RATIO * width:
        return None
    return focal


def focal_sequence(dlt3d_csv: Path | str, n_frames: int, width: float) -> np.ndarray:
    """Focal per frame index ``0..n_frames-1`` from a ``.dlt3d`` file (NaN where missing)."""
    frames, coeffs = load_dlt3d_file(dlt3d_csv)
    out = np.full(int(n_frames), np.nan, dtype=np.float64)
    for frame, L in zip(frames, coeffs, strict=True):
        idx = int(frame)
        if not 0 <= idx < n_frames:
            continue
        focal = focal_from_dlt3d(L, width)
        if focal is not None:
            out[idx] = focal
    return out


def cam_int_sequence(focals: np.ndarray, width: int, height: int) -> np.ndarray:
    """``(n, 3, 3)`` pinhole intrinsics with the principal point at the image centre.

    Rows whose focal is NaN are left NaN, so the consumer can tell them apart.
    """
    focals = np.asarray(focals, dtype=np.float64).reshape(-1)
    out = np.full((focals.size, 3, 3), np.nan, dtype=np.float32)
    ok = np.isfinite(focals)
    out[ok] = 0.0
    out[ok, 0, 0] = focals[ok]
    out[ok, 1, 1] = focals[ok]
    out[ok, 0, 2] = width / 2.0
    out[ok, 1, 2] = height / 2.0
    out[ok, 2, 2] = 1.0
    return out


def write_cam_int_npy(path: Path | str, cam_int: np.ndarray) -> Path:
    """Save the sequence as ``.npy`` (``sam3dinov3 --cam-int-npy`` reads it back)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.save(path, np.asarray(cam_int, dtype=np.float32))
    return path


def focal_summary(focals: np.ndarray) -> dict:
    """Counts and range for the log line: how many frames got a usable focal."""
    valid = np.isfinite(focals)
    if not valid.any():
        return {"frames": int(focals.size), "valid": 0, "min_px": None, "max_px": None}
    return {
        "frames": int(focals.size),
        "valid": int(valid.sum()),
        "min_px": round(float(np.nanmin(focals)), 1),
        "max_px": round(float(np.nanmax(focals)), 1),
    }
