"""
================================================================================
Script: monocular_planar_align.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.4.5
Created: 16 September 2026
Last Updated: 24 September 2026

================================================================================
Description
================================================================================

Place a SINGLE-camera monocular 3D reconstruction (``sam3dinov3`` /
``sam3dinov3_visualize``) into a metric **ground-plane** frame using a
per-frame planar homography from ``planar_geometry_tracker``
(``homographies.npz``), when a full ``.dlt3d`` volume calibration is not
available.

Why this exists
---------------
``monocular_dlt_align.py`` needs ``.dlt3d`` (non-coplanar control points).
A tatame / floor calibration only yields DLT2D → ``H`` (world→pixel) and
``H_inv`` (pixel→floor XY at Z=0). This module uses those foot
correspondences to solve a similarity ``(s, R, T)`` that drops the body
onto the tatame frame.

Scale IS free here (unlike DLT3D align): the metric floor from ``H``
breaks the monocular depth/size degeneracy that a pure reprojection fit
cannot resolve when focal length was the network default FOV.

Pipeline
--------
1. Load camera-frame ``*_mhr70_rec3d.csv`` + matching ``*_markers.csv``.
2. Load ``homographies.npz`` (``H``, ``H_inv``, ``frame_ids``).
3. Per frame: map ankle/heel pixels through ``H_inv`` → ``(X, Y, 0)``.
4. Umeyama similarity from camera-frame feet → floor targets (permissive
   planarity guard; upright fallback if degenerate).
5. Apply ``X_w = s R (X_cam - origin) + T`` to all markers (+ optional mesh).
6. Smooth ``(R, T, log s)`` with zero-lag Butterworth (placement only).

Usage
-----
GUI: Frame B → Markerless 3D → Monocular → Planar world

CLI::

    uv run python -m vaila.monocular_planar_align \\
        --mono3d  id_00_mhr70_rec3d_butterworth.csv \\
        --pixels  id_00_markers_butterworth.csv \\
        --homographies processed_planar_geom_.../homographies.npz \\
        --fps 60 -o ./process_part5

License: AGPL-3.0
"""

from __future__ import annotations

import argparse
import os
import shutil
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

try:
    from .mesh_alignment import umeyama_alignment, write_obj_mesh, write_ply_mesh
    from .monocular_dlt_align import (
        DEFAULT_EXPORT_MESH,
        DEFAULT_PLACEMENT_ORIGIN_MARKERS,
        DEFAULT_SMOOTH_HZ,
        _find_sibling,
        _placement_origin,
        load_mesh_frame_camera,
        load_pixels_from_long_csv,
        load_wide_positional,
        smooth_placement,
    )
    from .planar_geometry_tracker import load_homographies_npz, project_points
    from .rec3d import (
        find_unreconstructed_markers,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )
except ImportError:  # standalone execution
    from mesh_alignment import (  # ty: ignore[unresolved-import]
        umeyama_alignment,
        write_obj_mesh,
        write_ply_mesh,
    )
    from monocular_dlt_align import (  # ty: ignore[unresolved-import]
        DEFAULT_EXPORT_MESH,
        DEFAULT_PLACEMENT_ORIGIN_MARKERS,
        DEFAULT_SMOOTH_HZ,
        _find_sibling,
        _placement_origin,
        load_mesh_frame_camera,
        load_pixels_from_long_csv,
        load_wide_positional,
        smooth_placement,
    )
    from planar_geometry_tracker import (  # ty: ignore[unresolved-import]
        load_homographies_npz,
        project_points,
    )
    from rec3d import (  # ty: ignore[unresolved-import]
        find_unreconstructed_markers,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )

try:
    from .dialogsuser import ask_output_directory
except ImportError:
    from dialogsuser import ask_output_directory  # ty: ignore[unresolved-import]

DEFAULT_FPS = 60.0
#: 0-based MHR70 ankles + heels (toes optional when finite).
DEFAULT_FOOT_MARKERS_0: tuple[int, ...] = (13, 14, 17, 20)
OPTIONAL_TOE_MARKERS_0: tuple[int, ...] = (15, 16, 18, 19)
MIN_FOOT_POINTS = 3
#: Printed only as a stature sanity check (does not drive the fit).
DEFAULT_EXPECTED_HEIGHT_M = 1.86


def pixels_to_floor_xy(h_inv: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Map (N, 2) pixels through H_inv → (N, 2) floor metres (Z=0 plane)."""
    uv = np.asarray(uv, dtype=np.float64).reshape(-1, 2)
    return project_points(np.asarray(h_inv, dtype=np.float64), uv)


def _rotation_camera_up_to_world_z() -> np.ndarray:
    """OpenCV camera +Y is DOWN; map camera-up (-Y) to world +Z."""
    cam_up = np.array([0.0, -1.0, 0.0])
    world_up = np.array([0.0, 0.0, 1.0])
    v = np.cross(cam_up, world_up)
    c = float(np.dot(cam_up, world_up))
    if float(np.linalg.norm(v)) < 1e-12:
        return np.eye(3) if c > 0 else np.diag([1.0, -1.0, -1.0])
    vx = np.array([[0.0, -v[2], v[1]], [v[2], 0.0, -v[0]], [-v[1], v[0], 0.0]])
    return np.eye(3) + vx + vx @ vx * (1.0 / (1.0 + c))


def _similarity_2d(src_xy: np.ndarray, tgt_xy: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Closed-form 2D similarity: tgt ~= s R src + t (R is 2x2 rotation)."""
    src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    tgt_xy = np.asarray(tgt_xy, dtype=np.float64).reshape(-1, 2)
    mu_s = src_xy.mean(axis=0)
    mu_t = tgt_xy.mean(axis=0)
    src_c = src_xy - mu_s
    tgt_c = tgt_xy - mu_t
    # Kabsch in 2D via complex / SVD.
    cov = tgt_c.T @ src_c / len(src_xy)
    u, _, vt = np.linalg.svd(cov)
    r2 = u @ vt
    if np.linalg.det(r2) < 0:
        u[:, -1] *= -1.0
        r2 = u @ vt
    var_s = float((src_c**2).sum() / len(src_xy))
    if var_s <= 1e-18:
        s = 1.0
    else:
        s = float(np.trace(r2.T @ cov) / var_s)
        if s <= 1e-12:
            s = 1.0
    t2 = mu_t - s * (r2 @ mu_s)
    return s, r2, t2


def _similarity_2d_fixed_scale(
    src_xy: np.ndarray, tgt_xy: np.ndarray, scale: float
) -> tuple[float, np.ndarray, np.ndarray]:
    """2D rigid+scale with ``scale`` held fixed: tgt ~= s R src + t."""
    src_xy = np.asarray(src_xy, dtype=np.float64).reshape(-1, 2)
    tgt_xy = np.asarray(tgt_xy, dtype=np.float64).reshape(-1, 2)
    s = float(scale)
    mu_s = src_xy.mean(axis=0)
    mu_t = tgt_xy.mean(axis=0)
    src_c = src_xy - mu_s
    tgt_c = tgt_xy - mu_t
    cov = tgt_c.T @ src_c / len(src_xy)
    u, _, vt = np.linalg.svd(cov)
    r2 = u @ vt
    if np.linalg.det(r2) < 0:
        u[:, -1] *= -1.0
        r2 = u @ vt
    t2 = mu_t - s * (r2 @ mu_s)
    return s, r2, t2


#: Floor ankle/heel pair must span at least this (m) to vote on global scale.
MIN_FLOOR_SPAN_M_FOR_SCALE = 0.30
MIN_CAM_SPAN_M_FOR_SCALE = 0.15
SCALE_ACCEPT_RANGE = (0.4, 2.5)


def solve_planar_similarity(
    source_cam: np.ndarray,
    target_floor: np.ndarray,
    *,
    min_points: int = MIN_FOOT_POINTS,
    fixed_scale: float | None = None,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Similarity ``target ~= s R source + t`` for foot correspondences.

    Floor targets are coplanar (Z=0), so full 3D Umeyama is ill-conditioned.
    Primary path: map camera-up → world +Z, then 2D similarity in XY, and set
    ``t_z`` so mean foot height lands on the floor. When ``fixed_scale`` is
    set, only yaw+translation are free (stops per-frame collapse when H maps
    both feet to nearly the same floor point).
    """
    source_cam = np.asarray(source_cam, dtype=np.float64).reshape(-1, 3)
    target_floor = np.asarray(target_floor, dtype=np.float64).reshape(-1, 3)
    ok = np.isfinite(source_cam).all(axis=1) & np.isfinite(target_floor).all(axis=1)
    src, tgt = source_cam[ok], target_floor[ok]
    if len(src) < min_points:
        return None

    tgt_z_std = float(np.nanstd(tgt[:, 2]))
    if tgt_z_std < 1e-3:
        r_up = _rotation_camera_up_to_world_z()
        src_up = (r_up @ src.T).T
        if fixed_scale is not None and fixed_scale > 1e-9:
            s, r2, t2 = _similarity_2d_fixed_scale(src_up[:, :2], tgt[:, :2], fixed_scale)
            path = "planar_2d_fixed_s"
        else:
            s, r2, t2 = _similarity_2d(src_up[:, :2], tgt[:, :2])
            path = "planar_2d"
        r_yaw = np.eye(3)
        r_yaw[:2, :2] = r2
        r_mat = r_yaw @ r_up
        src_aligned_z = s * src_up[:, 2]
        t = np.array(
            [t2[0], t2[1], float(np.nanmean(tgt[:, 2]) - np.nanmean(src_aligned_z))],
            dtype=np.float64,
        )
        # #region agent log
        try:
            import json as _json
            import time as _time

            _span_src = (
                float(np.linalg.norm(src_up[0, :2] - src_up[1, :2]))
                if len(src_up) >= 2
                else float("nan")
            )
            _span_tgt = (
                float(np.linalg.norm(tgt[0, :2] - tgt[1, :2])) if len(tgt) >= 2 else float("nan")
            )
            with open(
                "/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8"
            ) as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "78fc22",
                            "hypothesisId": "A,C,E",
                            "location": "monocular_planar_align.py:solve_planar_similarity",
                            "message": "planar_2d_path",
                            "data": {
                                "path": path,
                                "s": float(s),
                                "fixed_scale": fixed_scale,
                                "n": int(len(src)),
                                "span_src_xy": _span_src,
                                "span_tgt_xy": _span_tgt,
                                "var_src_xy": float(np.var(src_up[:, :2])),
                                "src_up_z_extent": float(
                                    np.nanmax(src_up[:, 2]) - np.nanmin(src_up[:, 2])
                                ),
                                "mean_src_up_z": float(np.nanmean(src_up[:, 2])),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        return r_mat, float(s), t

    if fixed_scale is not None and fixed_scale > 1e-9:
        # Planar targets expected; non-planar branch with fixed scale still uses upright+2D.
        r_up = _rotation_camera_up_to_world_z()
        src_up = (r_up @ src.T).T
        s, r2, t2 = _similarity_2d_fixed_scale(src_up[:, :2], tgt[:, :2], fixed_scale)
        r_yaw = np.eye(3)
        r_yaw[:2, :2] = r2
        r_mat = r_yaw @ r_up
        src_aligned_z = s * src_up[:, 2]
        t = np.array(
            [t2[0], t2[1], float(np.nanmean(tgt[:, 2]) - np.nanmean(src_aligned_z))],
            dtype=np.float64,
        )
        return r_mat, float(s), t

    result = umeyama_alignment(src, tgt, min_points=min_points, planarity_ratio_threshold=1e-8)
    if (
        not result.degenerate
        and result.R is not None
        and result.t is not None
        and result.s is not None
        and result.s > 1e-9
    ):
        # #region agent log
        try:
            import json as _json
            import time as _time

            with open(
                "/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8"
            ) as _f:
                _f.write(
                    _json.dumps(
                        {
                            "sessionId": "78fc22",
                            "hypothesisId": "A",
                            "location": "monocular_planar_align.py:solve_planar_similarity",
                            "message": "umeyama_3d_path",
                            "data": {
                                "path": "umeyama_3d",
                                "s": float(result.s),
                                "n": int(len(src)),
                            },
                            "timestamp": int(_time.time() * 1000),
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass
        # #endregion
        return result.R, float(result.s), result.t

    # Last resort: scale from distances + upright + centroid translation.
    r_up = _rotation_camera_up_to_world_z()
    d_src = np.linalg.norm(src[1:] - src[0], axis=1)
    d_tgt = np.linalg.norm(tgt[1:] - tgt[0], axis=1)
    valid = (d_src > 1e-6) & (d_tgt > 1e-6)
    if not np.any(valid):
        return None
    s = (
        float(fixed_scale)
        if fixed_scale is not None and fixed_scale > 1e-9
        else float(np.median(d_tgt[valid] / d_src[valid]))
    )
    mu_src = src.mean(axis=0)
    mu_tgt = tgt.mean(axis=0)
    t = mu_tgt - s * (r_up @ mu_src)
    # #region agent log
    try:
        import json as _json
        import time as _time

        with open("/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "78fc22",
                        "hypothesisId": "A",
                        "location": "monocular_planar_align.py:solve_planar_similarity",
                        "message": "fallback_distance_path",
                        "data": {"path": "fallback_dist", "s": float(s), "n": int(len(src))},
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion
    return r_up, s, t


def estimate_robust_global_scale(
    per_frame_s: np.ndarray,
    span_h: np.ndarray,
    span_cam: np.ndarray,
    *,
    expected_height_m: float | None = None,
    mono_stature_m: float | None = None,
) -> tuple[float, dict]:
    """Median scale from frames where floor foot span is trustworthy.

    Collapsed H projections (feet mapped nearly on top of each other) produce
    s→0 and shrink the whole body; those frames are excluded from the vote.
    """
    per_frame_s = np.asarray(per_frame_s, dtype=np.float64)
    span_h = np.asarray(span_h, dtype=np.float64)
    span_cam = np.asarray(span_cam, dtype=np.float64)
    lo, hi = SCALE_ACCEPT_RANGE
    good = (
        np.isfinite(per_frame_s)
        & np.isfinite(span_h)
        & np.isfinite(span_cam)
        & (span_h >= MIN_FLOOR_SPAN_M_FOR_SCALE)
        & (span_cam >= MIN_CAM_SPAN_M_FOR_SCALE)
        & (per_frame_s >= lo)
        & (per_frame_s <= hi)
    )
    info: dict = {
        "n_candidates": int(len(per_frame_s)),
        "n_good": int(good.sum()),
        "span_h_threshold_m": MIN_FLOOR_SPAN_M_FOR_SCALE,
    }
    if int(good.sum()) >= 5:
        s_global = float(np.median(per_frame_s[good]))
        info["source"] = "median_good_h_spans"
        info["s_global"] = s_global
        return s_global, info
    if (
        expected_height_m is not None
        and mono_stature_m is not None
        and mono_stature_m > 0.2
        and expected_height_m > 0.5
    ):
        s_global = float(expected_height_m / mono_stature_m)
        info["source"] = "expected_height_over_mono_stature"
        info["s_global"] = s_global
        info["expected_height_m"] = float(expected_height_m)
        info["mono_stature_m"] = float(mono_stature_m)
        return s_global, info
    finite = per_frame_s[np.isfinite(per_frame_s)]
    s_global = float(np.median(finite)) if finite.size else 1.0
    info["source"] = "median_all_fallback"
    info["s_global"] = s_global
    return s_global, info


def smooth_scales(scales: np.ndarray, cutoff_hz: float, fps: float, order: int = 2) -> np.ndarray:
    """Zero-lag Butterworth on log(s) so scale stays positive."""
    n = len(scales)
    nyquist = fps / 2.0
    if n <= 3 * (order + 1) or not (0 < cutoff_hz < nyquist):
        return scales
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    log_s = np.log(np.clip(scales, 1e-9, None))
    return np.exp(filtfilt(b, a, log_s))


def _collect_foot_indices(n_markers: int, use_toes: bool = True) -> list[int]:
    idx = [i for i in DEFAULT_FOOT_MARKERS_0 if i < n_markers]
    if use_toes:
        idx.extend(i for i in OPTIONAL_TOE_MARKERS_0 if i < n_markers)
    return idx


def align_monocular_to_planar_world(
    mono3d_path,
    homographies_path,
    output_directory,
    pixels_path=None,
    ref3d_path=None,
    point_rate=DEFAULT_FPS,
    smooth_hz=DEFAULT_SMOOTH_HZ,
    origin_markers=DEFAULT_PLACEMENT_ORIGIN_MARKERS,
    skeleton_json_path=None,
    mesh_source_dir=None,
    export_mesh=DEFAULT_EXPORT_MESH,
    expected_height_m=DEFAULT_EXPECTED_HEIGHT_M,
    smooth_config_path=None,
    gui=False,
):
    """Align camera-frame mono3D onto the planar (tatame) world frame.

    Returns:
        (output_dir, file_base) on success, or None on failure.
    """
    mono3d_path = Path(mono3d_path).expanduser().resolve()
    homographies_path = Path(homographies_path).expanduser().resolve()
    run_dir = mono3d_path.parent

    print(f"Running script: {Path(__file__).name}")
    print(f"Monocular 3D (camera frame): {mono3d_path}")
    print(f"Homographies (planar)      : {homographies_path}")

    frames_3d, cam_xyz = load_wide_positional(mono3d_path, 3)
    n_markers = cam_xyz.shape[1]
    print(f"Loaded {len(frames_3d)} frames x {n_markers} markers (camera frame, metres)")

    if pixels_path is None:
        auto = _find_sibling(run_dir, "*_markers*.csv")
        # Prefer exact *_markers.csv / *_markers_butterworth.csv over other matches.
        if auto is not None:
            pixels_path = auto
            print(f"Pixels auto-detected: {Path(auto).name}")
    if pixels_path is not None:
        frames_2d, uv_arr = load_wide_positional(pixels_path, 2)
        if uv_arr.shape[1] != n_markers:
            return _fail(
                f"Marker count mismatch: {mono3d_path.name} has {n_markers}, "
                f"{Path(pixels_path).name} has {uv_arr.shape[1]}",
                gui,
            )
        pixels_by_frame = {int(f): uv_arr[i] for i, f in enumerate(frames_2d)}
    else:
        long_csv = _find_sibling(run_dir, "*_keypoints2d.csv")
        if long_csv is None:
            return _fail(
                "No 2D pixel source found. Pass --pixels with the same person's *_markers.csv.",
                gui,
            )
        print(f"Pixels from long table: {long_csv.name}")
        pixels_by_frame = load_pixels_from_long_csv(long_csv)

    h_by_frame = load_homographies_npz(homographies_path)
    # Prefer H_inv from the npz when present (exact inverse used at write time).
    with np.load(homographies_path) as data:
        frame_ids = np.asarray(data["frame_ids"], dtype=np.int64)
        if "H_inv" in data.files:
            h_inv_stack = np.asarray(data["H_inv"], dtype=np.float64)
            h_inv_by_frame = {int(fid): h_inv_stack[i] for i, fid in enumerate(frame_ids)}
        else:
            h_inv_by_frame = {fid: np.linalg.inv(h) for fid, h in h_by_frame.items()}

    # Ankles+heels only for placement/scale (toes make H spans noisier when collapsed).
    foot_idx = _collect_foot_indices(n_markers, use_toes=False)
    print(f"Foot markers (0-based, ankles+heels): {foot_idx}")

    print("-" * 80)
    print("Pass 1: free per-frame scale from feet via H_inv (candidates for global s)...")

    frame_cache: list[dict] = []
    skipped: dict[str, int] = {}

    for row_idx, frame_value in enumerate(frames_3d):
        frame = int(frame_value)
        uv = pixels_by_frame.get(frame)
        if uv is None:
            skipped["no_pixels"] = skipped.get("no_pixels", 0) + 1
            continue
        h_inv = h_inv_by_frame.get(frame)
        if h_inv is None:
            skipped["no_h_for_frame"] = skipped.get("no_h_for_frame", 0) + 1
            continue

        points_cam = cam_xyz[row_idx]
        finite = np.isfinite(points_cam).all(axis=1)
        if int(finite.sum()) < MIN_FOOT_POINTS:
            skipped["too_few_3d"] = skipped.get("too_few_3d", 0) + 1
            continue

        origin = _placement_origin(points_cam, finite, origin_markers)
        shape = points_cam - origin

        src_list: list[np.ndarray] = []
        tgt_list: list[np.ndarray] = []
        for mi in foot_idx:
            if mi >= n_markers:
                continue
            if not np.isfinite(points_cam[mi]).all():
                continue
            if not np.isfinite(uv[mi]).all():
                continue
            xy = pixels_to_floor_xy(h_inv, uv[mi : mi + 1])[0]
            if not np.isfinite(xy).all():
                continue
            src_list.append(shape[mi])
            tgt_list.append(np.array([xy[0], xy[1], 0.0], dtype=np.float64))

        if len(src_list) < MIN_FOOT_POINTS:
            skipped["too_few_feet"] = skipped.get("too_few_feet", 0) + 1
            continue

        src_stack = np.stack(src_list)
        tgt_stack = np.stack(tgt_list)
        solved = solve_planar_similarity(src_stack, tgt_stack)
        if solved is None:
            skipped["umeyama_failed"] = skipped.get("umeyama_failed", 0) + 1
            continue
        _r_free, s_free, _t_free = solved
        span_cam = (
            float(np.linalg.norm(src_stack[0] - src_stack[1]))
            if len(src_stack) >= 2
            else float("nan")
        )
        span_h = (
            float(np.linalg.norm(tgt_stack[0, :2] - tgt_stack[1, :2]))
            if len(tgt_stack) >= 2
            else float("nan")
        )
        frame_cache.append(
            {
                "frame": frame,
                "shape": shape,
                "origin": origin,
                "src": src_stack,
                "tgt": tgt_stack,
                "s_free": float(s_free),
                "span_cam": span_cam,
                "span_h": span_h,
            }
        )

        # #region agent log
        if frame % 30 == 0 or s_free < 0.5 or s_free > 2.0:
            try:
                import json as _json
                import time as _time

                with open(
                    "/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8"
                ) as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "78fc22",
                                "runId": "post-fix",
                                "hypothesisId": "A,B",
                                "location": "monocular_planar_align.py:align_loop",
                                "message": "per_frame_scale",
                                "data": {
                                    "frame": int(frame),
                                    "s": float(s_free),
                                    "n_feet": int(len(src_stack)),
                                    "span_cam3d": span_cam,
                                    "span_h_xy": span_h,
                                    "span_ratio_h_over_cam": (span_h / span_cam)
                                    if span_cam > 1e-9
                                    else None,
                                },
                                "timestamp": int(_time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

    if not frame_cache:
        return _fail(f"No frame could be placed (reasons: {skipped})", gui)

    mono_stature = float("nan")
    if n_markers > 20:
        nose_y = cam_xyz[:, 0, 1]
        feet_y = np.nanmax(cam_xyz[:, [13, 14, 17, 20], 1], axis=1)
        mono_stature = float(np.nanmedian(feet_y - nose_y))

    s_free_arr = np.array([c["s_free"] for c in frame_cache], dtype=np.float64)
    span_h_arr = np.array([c["span_h"] for c in frame_cache], dtype=np.float64)
    span_cam_arr = np.array([c["span_cam"] for c in frame_cache], dtype=np.float64)
    s_global, scale_info = estimate_robust_global_scale(
        s_free_arr,
        span_h_arr,
        span_cam_arr,
        expected_height_m=float(expected_height_m) if expected_height_m else None,
        mono_stature_m=mono_stature if np.isfinite(mono_stature) else None,
    )
    print(
        f"Global scale s={s_global:.4f} from {scale_info['source']} "
        f"({scale_info['n_good']}/{scale_info['n_candidates']} good H spans "
        f">= {MIN_FLOOR_SPAN_M_FOR_SCALE:g} m)"
    )

    # #region agent log
    try:
        import json as _json
        import time as _time

        with open("/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "78fc22",
                        "runId": "post-fix",
                        "hypothesisId": "A",
                        "location": "monocular_planar_align.py:global_scale",
                        "message": "robust_global_scale",
                        "data": {**scale_info, "mono_stature_m": mono_stature},
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    print("Pass 2: fixed global scale; hold/slerp yaw when H foot span collapses...")
    from scipy.spatial.transform import Slerp

    provisional: list[dict] = []
    for entry in frame_cache:
        solved = solve_planar_similarity(entry["src"], entry["tgt"], fixed_scale=s_global)
        if solved is None:
            skipped["fixed_scale_failed"] = skipped.get("fixed_scale_failed", 0) + 1
            continue
        r_mat, s, t = solved
        provisional.append(
            {
                **entry,
                "R": r_mat,
                "s": float(s),
                "t": np.asarray(t, dtype=np.float64),
                "good_span": float(entry["span_h"]) >= MIN_FLOOR_SPAN_M_FOR_SCALE,
            }
        )

    if not provisional:
        return _fail(f"No frame could be placed after fixed-scale pass (reasons: {skipped})", gui)

    good_idx = [i for i, p in enumerate(provisional) if p["good_span"]]
    n_bad = len(provisional) - len(good_idx)
    print(
        f"Yaw trust: {len(good_idx)}/{len(provisional)} frames with span_h "
        f">= {MIN_FLOOR_SPAN_M_FOR_SCALE:g} m; slerp/hold on {n_bad} collapsed-span frames"
    )

    def _nearest_good(i: int) -> tuple[int | None, int | None]:
        prev_i = next((j for j in range(i - 1, -1, -1) if provisional[j]["good_span"]), None)
        next_i = next(
            (j for j in range(i + 1, len(provisional)) if provisional[j]["good_span"]), None
        )
        return prev_i, next_i

    solved_frames: list[int] = []
    shapes: list[np.ndarray] = []
    rotvecs: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    scales: list[float] = []
    origins: list[np.ndarray] = []
    foot_xy_err: list[float] = []

    for i, p in enumerate(provisional):
        if p["good_span"]:
            r_mat = p["R"]
            t = p["t"]
            s = p["s"]
            mode = "kabsch"
        else:
            prev_i, next_i = _nearest_good(i)
            if prev_i is not None and next_i is not None and next_i != prev_i:
                t_frac = (i - prev_i) / float(next_i - prev_i)
                r0 = Rotation.from_matrix(provisional[prev_i]["R"])
                r1 = Rotation.from_matrix(provisional[next_i]["R"])
                slerp = Slerp([0.0, 1.0], Rotation.concatenate([r0, r1]))
                r_mat = slerp([float(t_frac)]).as_matrix()[0]
                mode = "slerp"
            elif prev_i is not None:
                r_mat = provisional[prev_i]["R"]
                mode = "hold_prev"
            elif next_i is not None:
                r_mat = provisional[next_i]["R"]
                mode = "hold_next"
            else:
                r_mat = p["R"]
                mode = "kabsch_fallback"
            s = float(s_global)
            mu_src = p["src"].mean(axis=0)
            mu_tgt = p["tgt"].mean(axis=0)
            t = mu_tgt - s * (mu_src @ r_mat.T)

        # #region agent log
        if (not provisional[i]["good_span"]) or int(p["frame"]) in {234, 235, 236, 237, 240, 244}:
            try:
                import json as _json
                import time as _time

                with open(
                    "/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8"
                ) as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "78fc22",
                                "runId": "post-fix",
                                "hypothesisId": "F",
                                "location": "monocular_planar_align.py:pass2_yaw",
                                "message": "yaw_mode",
                                "data": {
                                    "frame": int(p["frame"]),
                                    "span_h": float(p["span_h"]),
                                    "good_span": bool(provisional[i]["good_span"]),
                                    "mode": mode,
                                },
                                "timestamp": int(_time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
        # #endregion

        placed_feet = (float(s) * (p["src"] @ r_mat.T)) + t
        err = np.linalg.norm(placed_feet[:, :2] - p["tgt"][:, :2], axis=1)
        foot_xy_err.append(float(np.nanmean(err)))
        solved_frames.append(int(p["frame"]))
        shapes.append(p["shape"])
        rotvecs.append(Rotation.from_matrix(r_mat).as_rotvec())
        translations.append(np.asarray(t, dtype=np.float64))
        scales.append(float(s))
        origins.append(p["origin"])

    if not solved_frames:
        return _fail(f"No frame could be placed after fixed-scale pass (reasons: {skipped})", gui)

    rotvecs_arr = np.stack(rotvecs)
    translations_arr = np.stack(translations)
    scales_arr = np.asarray(scales, dtype=np.float64)
    raw_world = _build_world_scaled(shapes, rotvecs_arr, translations_arr, scales_arr)

    # #region agent log
    try:
        import json as _json
        import time as _time

        _st_raw = raw_world[:, 0, 2] - np.nanmin(raw_world[:, [13, 14, 17, 20], 2], axis=1)
        with open("/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8") as _f:
            _f.write(
                _json.dumps(
                    {
                        "sessionId": "78fc22",
                        "runId": "post-fix",
                        "hypothesisId": "A,D",
                        "location": "monocular_planar_align.py:pre_smooth",
                        "message": "scale_stature_before_smooth",
                        "data": {
                            "s_min": float(np.nanmin(scales_arr)),
                            "s_p5": float(np.nanpercentile(scales_arr, 5)),
                            "s_median": float(np.nanmedian(scales_arr)),
                            "s_p95": float(np.nanpercentile(scales_arr, 95)),
                            "s_max": float(np.nanmax(scales_arr)),
                            "frac_s_lt_0_5": float(np.mean(scales_arr < 0.5)),
                            "frac_s_lt_0_7": float(np.mean(scales_arr < 0.7)),
                            "stature_raw_mean": float(np.nanmean(_st_raw)),
                            "stature_raw_p5": float(np.nanpercentile(_st_raw, 5)),
                            "n_frames": int(len(scales_arr)),
                            "s_global": float(s_global),
                        },
                        "timestamp": int(_time.time() * 1000),
                    }
                )
                + "\n"
            )
    except Exception:
        pass
    # #endregion

    smoothed = False
    if smooth_hz and smooth_hz > 0:
        rotvecs_s, translations_s = smooth_placement(
            rotvecs_arr, translations_arr, float(smooth_hz), float(point_rate)
        )
        # Scale is global/fixed — do not reintroduce per-frame log(s) jitter.
        scales_s = np.full_like(scales_arr, float(s_global))
        smoothed = not (
            np.array_equal(rotvecs_s, rotvecs_arr)
            and np.array_equal(translations_s, translations_arr)
        )
        if smoothed:
            # #region agent log
            try:
                import json as _json
                import time as _time

                with open(
                    "/home/preto/data/vaila/.cursor/debug-78fc22.log", "a", encoding="utf-8"
                ) as _f:
                    _f.write(
                        _json.dumps(
                            {
                                "sessionId": "78fc22",
                                "runId": "post-fix",
                                "hypothesisId": "D",
                                "location": "monocular_planar_align.py:post_smooth",
                                "message": "scale_after_smooth",
                                "data": {
                                    "s_min": float(np.nanmin(scales_s)),
                                    "s_p5": float(np.nanpercentile(scales_s, 5)),
                                    "s_median": float(np.nanmedian(scales_s)),
                                    "s_p95": float(np.nanpercentile(scales_s, 95)),
                                    "s_max": float(np.nanmax(scales_s)),
                                    "delta_median": float(
                                        np.nanmedian(scales_s) - np.nanmedian(scales_arr)
                                    ),
                                },
                                "timestamp": int(_time.time() * 1000),
                            }
                        )
                        + "\n"
                    )
            except Exception:
                pass
            # #endregion
            world = _build_world_scaled(shapes, rotvecs_s, translations_s, scales_s)
            rotvecs_arr, translations_arr, scales_arr = rotvecs_s, translations_s, scales_s
        else:
            world = raw_world
            print("[yellow]Smoothing skipped: too few frames or cutoff >= Nyquist.[/yellow]")
    else:
        world = raw_world

    stats = _planar_report(
        world,
        raw_world,
        solved_frames,
        float(point_rate),
        foot_xy_err,
        scales_arr,
        n_markers,
        expected_height_m=float(expected_height_m) if expected_height_m else None,
    )
    print("-" * 80)
    print("=== Planar Alignment Complete ===")
    print(
        f"Frames placed: {len(solved_frames)}/{len(frames_3d)}"
        + (f"  skipped: {skipped}" if skipped else "")
    )
    print(
        f"Foot XY vs H (m): mean={stats['foot_xy_err_m_mean']:.4f} "
        f"p95={stats['foot_xy_err_m_p95']:.4f} max={stats['foot_xy_err_m_max']:.4f}"
    )
    print(
        f"Lowest foot above floor (m): mean={stats['floor_contact_m_mean']:+.3f} "
        f"(fit targets Z=0 on feet)"
    )
    print(
        f"Scale s: mean={stats['scale_mean']:.4f}  "
        f"p5={stats['scale_p5']:.4f}  p95={stats['scale_p95']:.4f}"
    )
    if stats.get("stature_m_mean") is not None:
        print(
            f"Stature proxy (nose−lowest foot, m): mean={stats['stature_m_mean']:.3f}"
            + (f"  (expected ~{expected_height_m:.2f} m, sanity only)" if expected_height_m else "")
        )
    print(
        f"Horizontal speed (m/s): median={stats['speed_median']:.2f} "
        f"p95={stats['speed_p95']:.2f} max={stats['speed_max']:.2f}"
        + (f"  [placement smoothed at {smooth_hz:g} Hz]" if smoothed else "  [raw placement]")
    )

    return _write_outputs(
        world,
        solved_frames,
        rotvecs_arr,
        translations_arr,
        scales_arr,
        output_directory,
        point_rate=float(point_rate),
        smooth_hz=float(smooth_hz) if smoothed else 0.0,
        stats=stats,
        mono3d_path=mono3d_path,
        homographies_path=homographies_path,
        ref3d_path=Path(ref3d_path).expanduser().resolve() if ref3d_path else None,
        pixels_path=Path(pixels_path).expanduser().resolve() if pixels_path else None,
        skeleton_json_path=skeleton_json_path,
        origins=np.stack(origins),
        mesh_source_dir=Path(mesh_source_dir).expanduser().resolve()
        if mesh_source_dir
        else run_dir,
        export_mesh=export_mesh,
        expected_height_m=float(expected_height_m) if expected_height_m else None,
        smooth_config_path=Path(smooth_config_path).expanduser().resolve()
        if smooth_config_path
        else None,
        gui=gui,
    )


def _fail(message: str, gui: bool):
    print(f"[red]Error: {message}[/red]")
    if gui:
        try:
            from tkinter import messagebox

            messagebox.showerror("Monocular planar alignment", message)
        except Exception:  # noqa: BLE001
            pass
    return None


def _build_world_scaled(shapes, rotvecs, translations, scales) -> np.ndarray:
    return np.stack(
        [
            float(scales[i]) * (shapes[i] @ Rotation.from_rotvec(rotvecs[i]).as_matrix().T)
            + translations[i]
            for i in range(len(shapes))
        ]
    )


def _planar_report(
    world,
    raw_world,
    frames,
    fps,
    foot_xy_err,
    scales,
    n_markers,
    *,
    expected_height_m,
):
    err = np.asarray(foot_xy_err, dtype=np.float64)
    finite_err = err[np.isfinite(err)]
    stats: dict = {
        "frames_placed": len(frames),
        "foot_xy_err_m_mean": float(finite_err.mean()) if finite_err.size else float("nan"),
        "foot_xy_err_m_p95": float(np.percentile(finite_err, 95))
        if finite_err.size
        else float("nan"),
        "foot_xy_err_m_max": float(finite_err.max()) if finite_err.size else float("nan"),
        "per_frame_foot_xy_err_m": err,
        "scale_mean": float(np.nanmean(scales)),
        "scale_p5": float(np.nanpercentile(scales, 5)),
        "scale_p95": float(np.nanpercentile(scales, 95)),
        "per_frame_scale": np.asarray(scales, dtype=np.float64),
    }

    foot_idx = [i for i in (13, 14, 15, 16, 17, 18, 19, 20) if i < n_markers]
    lowest = (
        np.nanmin(world[:, foot_idx, 2], axis=1) if foot_idx else np.nanmin(world[:, :, 2], axis=1)
    )
    stats["floor_contact_m_mean"] = float(np.nanmean(lowest))
    stats["floor_contact_m_p5"] = float(np.nanpercentile(lowest, 5))
    stats["floor_contact_m_p95"] = float(np.nanpercentile(lowest, 95))

    if n_markers > 0:
        nose = world[:, 0, 2]
        stature = nose - lowest
        stature = stature[np.isfinite(stature)]
        if stature.size:
            stats["stature_m_mean"] = float(np.nanmean(stature))
            stats["stature_m_p5"] = float(np.nanpercentile(stature, 5))
            stats["stature_m_p95"] = float(np.nanpercentile(stature, 95))
            if expected_height_m:
                stats["stature_vs_expected_m"] = stats["stature_m_mean"] - float(expected_height_m)

    hip_idx = [i for i in (9, 10) if i < n_markers]
    pelvis = np.nanmean(world[:, hip_idx, :], axis=1) if hip_idx else np.nanmean(world, axis=1)
    speed = np.linalg.norm(np.diff(pelvis, axis=0)[:, :2], axis=1) * fps
    speed = speed[np.isfinite(speed)]
    stats["speed_median"] = float(np.median(speed)) if speed.size else float("nan")
    stats["speed_p95"] = float(np.percentile(speed, 95)) if speed.size else float("nan")
    stats["speed_max"] = float(speed.max()) if speed.size else float("nan")

    raw_pelvis = (
        np.nanmean(raw_world[:, hip_idx, :], axis=1) if hip_idx else np.nanmean(raw_world, axis=1)
    )
    raw_speed = np.linalg.norm(np.diff(raw_pelvis, axis=0)[:, :2], axis=1) * fps
    raw_speed = raw_speed[np.isfinite(raw_speed)]
    stats["raw_speed_max"] = float(raw_speed.max()) if raw_speed.size else float("nan")
    return stats


def export_planar_mesh_sequence(
    mesh_source_dir: str | Path,
    export_fmt: str,
    origins: np.ndarray,
    rotmats: np.ndarray,
    translations: np.ndarray,
    scales: np.ndarray,
    frames: list[int],
    output_dir: str | Path,
) -> dict | None:
    """Apply the same scaled similarity to per-frame body meshes."""
    if export_fmt == "none":
        return None
    if export_fmt not in ("obj", "ply"):
        raise ValueError(f"export_fmt must be 'none', 'obj' or 'ply', got {export_fmt!r}")

    mesh_source_dir = Path(mesh_source_dir)
    mesh_frames_dir = mesh_source_dir / "meshes"
    mesh_faces_path = mesh_source_dir / "mesh_faces.npy"
    if not mesh_frames_dir.is_dir() or not mesh_faces_path.is_file():
        print(
            f"[yellow]No meshes/ + mesh_faces.npy in {mesh_source_dir}; skipping mesh export.[/yellow]"
        )
        return None

    faces = np.load(mesh_faces_path)
    out_dir = Path(output_dir) / f"meshes_{export_fmt}"
    out_dir.mkdir(parents=True, exist_ok=True)
    writer = write_obj_mesh if export_fmt == "obj" else write_ply_mesh

    written = 0
    missing = 0
    for i, frame in enumerate(frames):
        src = mesh_frames_dir / f"frame_{int(frame):06d}.npz"
        mesh_cam = load_mesh_frame_camera(src) if src.is_file() else None
        if mesh_cam is None:
            missing += 1
            continue
        shape = mesh_cam - origins[i]
        mesh_world = float(scales[i]) * (shape @ rotmats[i].T) + translations[i]
        writer(out_dir / f"frame_{int(frame):06d}.{export_fmt}", mesh_world, faces)
        written += 1

    out_dir.joinpath("README_mesh_import.txt").write_text(
        "vaila monocular_planar_align — aligned mesh sequence\n"
        "====================================================\n\n"
        f"{written} frame(s), format .{export_fmt}, RAW (x, y, z) tatame/world frame.\n"
        "PREFERRED: run the *_blender_skeleton_viz.py script in the parent folder.\n",
        encoding="utf-8",
    )
    print(
        f"Mesh sequence written: {out_dir} ({written} frame(s)"
        + (f", {missing} missing)" if missing else ")")
    )
    return {"written": written, "missing": missing, "output_mesh_dir": str(out_dir)}


def _write_outputs(
    world,
    frames,
    rotvecs,
    translations,
    scales,
    output_directory,
    *,
    point_rate,
    smooth_hz,
    stats,
    mono3d_path,
    homographies_path,
    ref3d_path,
    pixels_path,
    skeleton_json_path,
    origins,
    mesh_source_dir,
    export_mesh,
    expected_height_m,
    smooth_config_path,
    gui,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = (
        Path(output_directory).expanduser().resolve() / f"processed_monocular_planar_{timestamp}"
    )
    new_dir.mkdir(parents=True, exist_ok=True)
    file_base = f"mono_planar_{timestamp}"

    n_markers = world.shape[1]
    header = ["frame"]
    for m in range(1, n_markers + 1):
        header.extend([f"p{m}_x", f"p{m}_y", f"p{m}_z"])
    rec3d_df = pd.DataFrame(
        np.column_stack([np.asarray(frames, dtype=np.float64), world.reshape(len(world), -1)]),
        columns=header,  # ty: ignore[invalid-argument-type]
    )
    rec3d_df["frame"] = rec3d_df["frame"].astype(int)

    csv_path = new_dir / f"{file_base}.csv"
    rec3d_df.to_csv(csv_path, index=False, float_format="%.6f")
    rec3d_df.to_csv(new_dir / f"{file_base}.3d", index=False, float_format="%.6f")
    print(f"World-frame 3D written: {csv_path}")

    try:
        import vaila.readcsv_export as readcsv_export
    except ImportError:
        import readcsv_export  # ty: ignore[unresolved-import]

    df_for_c3d = rec3d_df.copy()
    df_for_c3d.columns = [
        c if c == "frame" else f"{c.rsplit('_', 1)[0]}_{c.rsplit('_', 1)[1].upper()}"
        for c in df_for_c3d.columns
    ]
    for suffix, factor in (("_m", 1), ("_mm", 1000)):
        try:
            readcsv_export.auto_create_c3d_from_csv(
                df_for_c3d,
                str(new_dir / f"{file_base}{suffix}.c3d"),
                point_rate=point_rate,
                conversion_factor=factor,
            )
        except Exception as exc:  # noqa: BLE001
            print(f"[yellow]C3D ({suffix}) export failed: {exc}[/yellow]")

    save_rec3d_as_bvh(rec3d_df, str(new_dir), file_base, point_rate, gui=gui, swap_yz=True)

    mesh_summary = export_planar_mesh_sequence(
        mesh_source_dir,
        export_mesh,
        origins,
        Rotation.from_rotvec(rotvecs).as_matrix(),
        translations,
        scales,
        frames,
        new_dir,
    )

    report_df = pd.DataFrame(
        {
            "frame": frames,
            "foot_xy_err_m": stats["per_frame_foot_xy_err_m"],
            "scale": scales,
            "rotvec_x": rotvecs[:, 0],
            "rotvec_y": rotvecs[:, 1],
            "rotvec_z": rotvecs[:, 2],
            "translation_x_m": translations[:, 0],
            "translation_y_m": translations[:, 1],
            "translation_z_m": translations[:, 2],
        }
    )
    report_path = new_dir / f"{file_base}_alignment.csv"
    report_df.to_csv(report_path, index=False, float_format="%.6f")

    blender_script = generate_blender_companion_script(
        str(new_dir),
        file_base,
        skeleton_json_path,
        point_rate=point_rate,
        n_frames=len(rec3d_df),
        unreconstructed_markers=find_unreconstructed_markers(rec3d_df),
    )

    if smooth_config_path is not None and smooth_config_path.is_file():
        shutil.copy2(smooth_config_path, new_dir / "smooth_config.toml")

    stature_line = ""
    if stats.get("stature_m_mean") is not None:
        stature_line = f"stature proxy (nose−foot Z) mean {stats['stature_m_mean']:.3f} m" + (
            f"  (expected ~{expected_height_m:.2f} m; sanity only, not fitted)\n"
            if expected_height_m
            else "\n"
        )

    (new_dir / "README_monocular_planar_align.txt").write_text(
        "vaila monocular -> planar (tatame) world alignment\n"
        "==================================================\n\n"
        f"monocular source (camera frame): {mono3d_path}\n"
        f"2D pixels                      : {pixels_path}\n"
        f"homographies.npz               : {homographies_path}\n"
        f"ref3d (optional)               : {ref3d_path}\n"
        f"frames placed                  : {stats['frames_placed']}\n"
        f"point rate                     : {point_rate} Hz\n"
        f"placement smoothing            : "
        + (
            f"{smooth_hz:g} Hz zero-lag Butterworth on (R, T, log s)\n"
            if smooth_hz
            else "none (raw)\n"
        )
        + "\nWhat was done\n-------------\n"
        "Foot ankle/heel (and optional toe) pixels were mapped through H_inv onto\n"
        "the metric floor plane (Z=0). A similarity (s, R, T) placed the monocular\n"
        "camera-frame body so those feet match the planar geometry. Scale is FREE\n"
        "because the tatame H is metric — this corrects default-FOV monocular depth.\n\n"
        "Quality\n-------\n"
        f"foot XY vs H (m)       mean {stats['foot_xy_err_m_mean']:.4f}  "
        f"p95 {stats['foot_xy_err_m_p95']:.4f}  max {stats['foot_xy_err_m_max']:.4f}\n"
        f"lowest foot above floor  mean {stats['floor_contact_m_mean']:+.3f} m  "
        f"(p5 {stats['floor_contact_m_p5']:+.3f}, p95 {stats['floor_contact_m_p95']:+.3f})\n"
        f"scale s                mean {stats['scale_mean']:.4f}  "
        f"p5 {stats['scale_p5']:.4f}  p95 {stats['scale_p95']:.4f}\n"
        + stature_line
        + f"horizontal speed (m/s)  median {stats['speed_median']:.2f}  "
        f"p95 {stats['speed_p95']:.2f}  max {stats['speed_max']:.2f}"
        + (f"   (raw max {stats['raw_speed_max']:.2f})\n" if smooth_hz else "\n")
        + "\nCaveats\n-------\n"
        "* Planar H constrains the floor, not a full 3D volume. Body proportions\n"
        "  still come from the monocular network; only global (s, R, T) are solved.\n"
        "* Prefer monocular_dlt_align.py when a real .dlt3d (non-coplanar) exists.\n"
        "* Prefer rec3d_one_dlt3d.py when 2+ synchronised calibrated cameras exist.\n\n"
        "Files\n-----\n"
        f"{file_base}.csv / .3d              world-frame 3D (vaila rec3d convention)\n"
        f"{file_base}_m.c3d / _mm.c3d        C3D in metres / millimetres\n"
        f"{file_base}.bvh                    mocap for Blender (Y/Z swapped)\n"
        f"{file_base}_alignment.csv          per-frame foot XY error, scale, pose\n"
        + (f"{Path(blender_script).name}   run this in Blender\n" if blender_script else "")
        + (
            f"meshes_{export_mesh}/                  aligned per-frame mesh "
            f"({mesh_summary['written']} frame(s))\n"
            if mesh_summary
            else ""
        )
        + (
            "smooth_config.toml               copy of the upstream filter config\n"
            if smooth_config_path and smooth_config_path.is_file()
            else ""
        ),
        encoding="utf-8",
    )

    print("\n=== Processing Complete ===")
    print(f"Output directory: {new_dir}")
    print(f"  - {file_base}.csv / .3d (planar world-frame 3D)")
    print(f"  - {file_base}_m.c3d / _mm.c3d (C3D)")
    print(f"  - {file_base}.bvh (Blender mocap)")
    print(f"  - {report_path.name}")
    if blender_script:
        print(f"  - {os.path.basename(blender_script)}")
    if mesh_summary:
        print(f"  - meshes_{export_mesh}/ ({mesh_summary['written']} frame(s))")

    if gui:
        try:
            from tkinter import messagebox

            messagebox.showinfo(
                "Monocular planar alignment",
                f"Alignment complete.\n\n"
                f"Frames: {stats['frames_placed']}\n"
                f"Foot XY err: {stats['foot_xy_err_m_mean']:.4f} m mean\n"
                f"Scale s: {stats['scale_mean']:.4f}\n\n"
                f"{new_dir}",
            )
        except Exception:  # noqa: BLE001
            pass
    return str(new_dir), file_base


def run_monocular_planar_align_gui():
    """Prompt for inputs, then run planar alignment."""
    from tkinter import Tk, filedialog, messagebox, simpledialog

    root = Tk()
    root.withdraw()
    try:
        mono3d = filedialog.askopenfilename(
            title="Monocular 3D CSV in the CAMERA frame (*_mhr70_rec3d*.csv)",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not mono3d:
            return None
        pixels = filedialog.askopenfilename(
            title="Same person's 2D pixels (*_markers*.csv) — Cancel to auto-detect",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        homog = filedialog.askopenfilename(
            title="Planar homographies.npz (from Planar Geo)",
            filetypes=[("NPZ", "*.npz"), ("All files", "*")],
        )
        if not homog:
            return None
        ref3d = filedialog.askopenfilename(
            title="target_calibration.ref3d — optional, Cancel to skip",
            filetypes=[("REF3D", "*.ref3d"), ("CSV", "*.csv"), ("All files", "*")],
        )
        output = ask_output_directory(mono3d, title="Output directory")
        if not output:
            return None
        fps = simpledialog.askfloat(
            "Capture rate",
            "Point rate (Hz):",
            minvalue=0.0001,
            initialvalue=DEFAULT_FPS,
        )
        if fps is None:
            return None
        smooth = simpledialog.askfloat(
            "Placement smoothing",
            "Zero-lag Butterworth cutoff (Hz) on (R, T, log s).\nEnter 0 to disable:",
            minvalue=0.0,
            initialvalue=DEFAULT_SMOOTH_HZ,
        )
        if smooth is None:
            smooth = DEFAULT_SMOOTH_HZ
        export_mesh = DEFAULT_EXPORT_MESH
        mesh_dir = Path(mono3d).parent
        has_local_meshes = (mesh_dir / "meshes").is_dir()
        if not has_local_meshes or not messagebox.askyesno(
            "Mesh export",
            "meshes/ found next to the monocular 3D CSV.\n\n"
            "Also export an aligned per-frame mesh sequence (.obj)?",
        ):
            export_mesh = "none"
        if export_mesh != "none" and not (mesh_dir / "meshes").is_dir():
            picked = filedialog.askdirectory(
                title="Mesh source dir (contains meshes/ + mesh_faces.npy) — Cancel to skip"
            )
            if picked:
                mesh_dir = Path(picked)
            else:
                export_mesh = "none"
        elif export_mesh == "none" and not has_local_meshes:
            picked = filedialog.askdirectory(
                title="Mesh source dir (contains meshes/ + mesh_faces.npy) — Cancel to skip"
            )
            if picked:
                mesh_dir = Path(picked)
                export_mesh = DEFAULT_EXPORT_MESH
    finally:
        root.destroy()

    return align_monocular_to_planar_world(
        mono3d,
        homog,
        output,
        pixels_path=pixels or None,
        ref3d_path=ref3d or None,
        point_rate=float(fps),
        smooth_hz=float(smooth),
        mesh_source_dir=mesh_dir,
        export_mesh=export_mesh,
        gui=True,
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaila.monocular_planar_align",
        description=(
            "Place a monocular camera-frame 3D reconstruction onto a metric "
            "ground plane using planar_geometry_tracker homographies.npz."
        ),
    )
    parser.add_argument("--mono3d", type=Path, help="Camera-frame *_mhr70_rec3d*.csv")
    parser.add_argument("--pixels", type=Path, help="Matching *_markers*.csv (2D pixels)")
    parser.add_argument("--homographies", type=Path, help="homographies.npz from Planar Geo")
    parser.add_argument("--ref3d", type=Path, help="Optional target_calibration.ref3d")
    parser.add_argument("-o", "--output", type=Path, help="Output parent directory")
    parser.add_argument("--fps", type=float, default=DEFAULT_FPS, help="Point rate (Hz)")
    parser.add_argument(
        "--smooth-hz",
        type=float,
        default=DEFAULT_SMOOTH_HZ,
        help="Placement Butterworth cutoff (Hz); 0 disables",
    )
    parser.add_argument("--no-smooth", action="store_true", help="Disable placement smoothing")
    parser.add_argument(
        "--origin-markers",
        type=int,
        nargs="+",
        default=list(DEFAULT_PLACEMENT_ORIGIN_MARKERS),
        help="1-based markers for placement origin (default: MHR70 hips)",
    )
    parser.add_argument("--mesh-source-dir", type=Path, help="Dir with meshes/ + mesh_faces.npy")
    parser.add_argument(
        "--export-mesh",
        choices=("none", "obj", "ply"),
        default=DEFAULT_EXPORT_MESH,
        help="Aligned mesh export format",
    )
    parser.add_argument(
        "--expected-height-m",
        type=float,
        default=DEFAULT_EXPECTED_HEIGHT_M,
        help="Stature sanity check only (metres); not fitted",
    )
    parser.add_argument(
        "--smooth-config",
        type=Path,
        help="Optional smooth_config.toml to copy into the output folder",
    )
    parser.add_argument("--skeleton-json", type=Path, help="Optional Blender skeleton JSON")
    parser.add_argument("--gui", action="store_true", help="Force GUI even with some args")
    return parser


def main(argv: list[str] | None = None):
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    if args.gui or not any((args.mono3d, args.homographies, args.output)):
        return run_monocular_planar_align_gui()

    missing = [
        n
        for n, v in (
            ("--mono3d", args.mono3d),
            ("--homographies", args.homographies),
            ("--output", args.output),
        )
        if v is None
    ]
    if missing:
        parser.error(f"missing required arguments: {', '.join(missing)}")

    smooth_hz = 0.0 if args.no_smooth else float(args.smooth_hz)
    return align_monocular_to_planar_world(
        args.mono3d,
        args.homographies,
        args.output,
        pixels_path=args.pixels,
        ref3d_path=args.ref3d,
        point_rate=float(args.fps),
        smooth_hz=smooth_hz,
        origin_markers=tuple(args.origin_markers),
        skeleton_json_path=args.skeleton_json,
        mesh_source_dir=args.mesh_source_dir,
        export_mesh=args.export_mesh,
        expected_height_m=args.expected_height_m,
        smooth_config_path=args.smooth_config,
        gui=False,
    )


if __name__ == "__main__":
    main()
