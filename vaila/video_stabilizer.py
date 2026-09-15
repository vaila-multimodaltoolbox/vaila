"""Fixed-scene video stabilization from getpixelvideo marker coordinates.

Version: 0.4.2
Update Date: 15 September 2026
Author: Paulo R. P. Santiago
License: AGPL-3.0-or-later

Visual/hybrid modes use shape-preserving similarity by default. The metric
floor homography is never the default whole-frame visual warp: in hybrid mode a
trusted floor fit only reconstructs missing metric controls, which then enter
the visual solve as low-weight imputed correspondences. Transforms are
re-anchored to the reference frame after temporal smoothing, static markers
absent from the reference frame are recovered through a canonical reference
atlas, and the production video is encoded as H.264 through the repository
FFmpeg helpers and validated with ffprobe. No display is needed for the CLI.
Run without arguments for the Tkinter GUI.
"""

from __future__ import annotations

import argparse
import html
import json
import math
import os
import queue
import re
import shlex
import subprocess
import sys
import tempfile
import threading
import webbrowser
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

if __package__:
    from .ffmpeg_utils import (
        describe_video_encoder,
        encoders_with_cpu_fallback,
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )
    from .numberframes import get_precise_video_metadata
    from .planar_geometry_tracker import (
        FrameHomography,
        cv2_find_homography,
        has_non_collinear_quad,
        load_measurements_csv,
        load_target_geometry,
        project_points,
    )
else:
    from ffmpeg_utils import (
        describe_video_encoder,
        encoders_with_cpu_fallback,
        get_ffmpeg_path,
        get_ffmpeg_video_encoding_args,
        get_video_encode_ffmpeg_path,
        is_hardware_video_encoder,
    )
    from numberframes import get_precise_video_metadata
    from planar_geometry_tracker import (
        FrameHomography,
        cv2_find_homography,
        has_non_collinear_quad,
        load_measurements_csv,
        load_target_geometry,
        project_points,
    )

try:
    from .cli_highlight import print_gui_cli_mirror
except ImportError:
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]


@dataclass
class MarkerTable:
    frames: np.ndarray
    ids: list[int]
    xy: np.ndarray  # frame, marker, xy; all original observations, never imputed


def log(message, callback=None):
    print(f">> vaila/video_stabilizer: {message}", flush=True)
    if callback:
        callback(str(message))


def parse_id_spec(spec, available):
    """Parse nonnegative IDs, inclusive ranges and all, without contiguous-ID assumptions."""
    available = set(available)
    if spec is None or str(spec).strip().lower() == "all":
        return sorted(available)
    selected = set()
    for token in str(spec).split(","):
        token = token.strip()
        if not re.fullmatch(r"\d+(?:-\d+)?", token):
            raise ValueError(f"Invalid marker specification: {spec!r}. Use all or 0-7,10,12-14.")
        ends = [int(value) for value in token.split("-")]
        if len(ends) == 2 and ends[1] < ends[0]:
            raise ValueError(f"Descending marker range: {token}")
        selected.update(range(ends[0], ends[-1] + 1))
    missing = selected - available
    if missing:
        raise ValueError(f"Marker IDs not present in CSV: {sorted(missing)}")
    return sorted(selected)


def load_marker_csv(path, frame_count=None):
    """Reuse the planar CSV reader, validating frame IDs before dense timeline alignment."""
    df = pd.read_csv(path)
    if "frame" not in df:
        raise ValueError("Marker CSV must contain a zero-based frame column.")
    frames = pd.to_numeric(df.frame, errors="raise").to_numpy(dtype=float)
    if not len(frames) or not np.isfinite(frames).all() or np.any(frames < 0):
        raise ValueError("Frame IDs must be finite nonnegative integers.")
    if np.any(frames != np.floor(frames)) or df.frame.duplicated().any():
        raise ValueError("Frame IDs must be unique integers (wide CSV format).")
    ids = sorted(int(m[1]) for col in df for m in [re.fullmatch(r"p(\d+)_x", col)] if m)
    if not ids or any(f"p{i}_y" not in df for i in ids):
        raise ValueError("Expected paired pN_x/pN_y columns in a wide marker CSV.")
    n = int(frames.max()) + 1 if frame_count is None else frame_count
    if frames.max() >= n:
        raise ValueError(f"CSV frame {int(frames.max())} is outside video frames 0..{n - 1}.")
    observations = load_measurements_csv(path)
    xy = np.full((n, len(ids), 2), np.nan, dtype=np.float64)
    for frame, points in observations.items():
        for col, pid in enumerate(ids):
            if pid in points and np.isfinite(points[pid]).all():
                xy[frame, col] = points[pid]
    return MarkerTable(np.arange(n), ids, xy)


def _useful(points, model="similarity"):
    minimum = 3 if model == "affine" else 2
    if len(points) < minimum or not np.isfinite(points).all():
        return False
    centered = points - points.mean(axis=0)
    return np.linalg.matrix_rank(centered, tol=1e-7) >= (2 if model == "affine" else 1)


def choose_reference_frame(
    table, stabilization_ids, anchor_ids, reference="auto", model="similarity"
):
    cols = [table.ids.index(i) for i in stabilization_ids]
    pts = table.xy[:, cols]
    valid = np.isfinite(pts).all(axis=2)
    candidates = [i for i in range(len(pts)) if _useful(pts[i, valid[i]], model)]
    if not candidates:
        raise ValueError(
            "Insufficient correspondences: need at least two non-coincident static markers (three non-collinear for affine)."
        )
    if reference == "first":
        chosen = 0
    elif reference.startswith("frame:"):
        chosen = int(reference.split(":", 1)[1])
    elif reference == "auto":
        anchor_cols = [stabilization_ids.index(i) for i in anchor_ids]
        anchor_counts = valid[:, anchor_cols].sum(axis=1)
        best_anchor_count = max(anchor_counts[candidates])
        candidates = [i for i in candidates if anchor_counts[i] == best_anchor_count]
        max_count = max(valid[candidates].sum(axis=1))
        candidates = [i for i in candidates if valid[i].sum() == max_count]
        areas = {
            i: cv2.contourArea(cv2.convexHull(pts[i, valid[i]].astype(np.float32)))
            for i in candidates
        }
        max_area = max(areas.values())
        candidates = [i for i in candidates if areas[i] >= 0.25 * max_area]
        median = np.full(pts.shape[1:], np.nan)
        for col in range(pts.shape[1]):
            if valid[:, col].any():
                median[col] = np.median(pts[valid[:, col], col], axis=0)
        # Median is a distance target only. The reference is an actual observation.
        chosen = min(
            candidates,
            key=lambda i: np.mean(np.sum((pts[i, valid[i]] - median[valid[i]]) ** 2, axis=1)),
        )
    else:
        raise ValueError("Reference must be auto, first or frame:N (zero-based).")
    if chosen not in candidates and reference != "auto":
        raise ValueError(
            f"Reference frame {chosen} has insufficient useful correspondences or is outside the video."
        )
    return int(chosen)


def fit_weighted_transform(src, dst, weights=None, model="similarity"):
    """Centered float64 least squares; no normal-equation inverse or projective terms."""
    src, dst = np.asarray(src, dtype=np.float64), np.asarray(dst, dtype=np.float64)
    if not _useful(src, model) or not _useful(dst, model):
        return None
    weights = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=float)
    if not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("Correspondence weights must be positive and finite.")
    center_s = np.average(src, axis=0, weights=weights)
    center_d = np.average(dst, axis=0, weights=weights)
    x, y = src - center_s, dst - center_d
    if model == "similarity":
        denom = np.sum(weights * np.sum(x * x, axis=1))
        a = np.sum(weights * np.sum(x * y, axis=1)) / denom
        b = np.sum(weights * (x[:, 0] * y[:, 1] - x[:, 1] * y[:, 0])) / denom
        linear = np.array([[a, -b], [b, a]])
    else:
        w = np.sqrt(weights[:, None])
        linear = np.linalg.lstsq(x * w, y * w, rcond=None)[0].T
    matrix = np.eye(3)
    matrix[:2, :2] = linear
    matrix[:2, 2] = center_d - linear @ center_s
    if not np.isfinite(matrix).all() or np.linalg.det(linear) <= 1e-10:
        return None
    return matrix


def spatial_weights(points, width, height):
    normalized = points / np.array([width, height], dtype=float)
    distances2 = np.sum((normalized[:, None] - normalized[None, :]) ** 2, axis=2)
    density = np.exp(-distances2 / (2 * 0.15**2)).sum(axis=1)
    return 1.0 / density


def estimate_transform(src, dst, weights, model="similarity", estimator="robust-lsq"):
    if not _useful(src, model) or not _useful(dst, model):
        return None
    if estimator == "ransac":
        solver = cv2.estimateAffinePartial2D if model == "similarity" else cv2.estimateAffine2D
        matrix, mask = solver(src, dst, method=cv2.RANSAC, ransacReprojThreshold=4.0)
        if matrix is None:
            return None
        keep = mask.ravel().astype(bool)
        return fit_weighted_transform(src[keep], dst[keep], weights[keep], model)
    matrix = fit_weighted_transform(src, dst, weights, model)
    for _ in range(30):
        if matrix is None:
            break
        residual = np.linalg.norm(project_points(matrix, src) - dst, axis=1)
        median = np.median(residual)
        cutoff = max(2.0, median + 2.5 * 1.4826 * np.median(np.abs(residual - median)))
        robust = np.minimum(1.0, cutoff / np.maximum(residual, 1e-12))
        updated = fit_weighted_transform(src, dst, weights * robust, model)
        if updated is None or np.allclose(updated, matrix, atol=1e-8, rtol=1e-8):
            break
        matrix = updated
    return matrix


def matrix_to_params(matrix, model="similarity"):
    a, b = matrix[0, 0], matrix[1, 0]
    scale = math.hypot(a, b)
    theta = math.atan2(b, a)
    params = [matrix[0, 2], matrix[1, 2], theta, math.log(scale)]
    if model == "affine":
        rotation = np.array(
            [[math.cos(theta), -math.sin(theta)], [math.sin(theta), math.cos(theta)]]
        )
        triangular = rotation.T @ matrix[:2, :2]
        params.extend([math.log(triangular[1, 1]), triangular[0, 1]])
    return np.array(params)


def params_to_matrix(params, model="similarity"):
    tx, ty, theta, log_scale = params[:4]
    c, s = math.cos(theta), math.sin(theta)
    rotation = np.array([[c, -s], [s, c]])
    linear = np.eye(2) * math.exp(log_scale)
    if model == "affine":
        linear = np.array([[math.exp(log_scale), params[5]], [0, math.exp(params[4])]])
    matrix = np.eye(3)
    matrix[:2, :2] = rotation @ linear
    matrix[:2, 2] = (tx, ty)
    return matrix


def regularize_transforms(matrices, fps, smooth="savgol", model="similarity"):
    """Interpolate gaps in physical parameters; zero-phase smoothing, never matrix coefficients."""
    good = np.array([i for i, matrix in enumerate(matrices) if matrix is not None])
    if not len(good):
        raise ValueError(
            "No trustworthy transforms: insufficient correspondences with the reference."
        )
    params = np.array([matrix_to_params(matrices[i], model) for i in good])
    params[:, 2] = np.unwrap(params[:, 2])
    timeline = np.arange(len(matrices))
    raw = np.column_stack(
        [np.interp(timeline, good, params[:, col]) for col in range(params.shape[1])]
    )
    final = raw.copy()
    if smooth == "savgol" and len(raw) >= 5:
        from scipy.signal import savgol_filter

        window = min(7, len(raw) if len(raw) % 2 else len(raw) - 1)
        final = savgol_filter(raw, window, 2, axis=0, mode="interp")
    elif smooth == "lowpass" and len(raw) > 9:
        from scipy.signal import butter, sosfiltfilt

        sos = butter(2, min(6.0, fps * 0.2), fs=fps, output="sos")
        final = sosfiltfilt(sos, raw, axis=0)
    good_set = set(good)
    methods = [
        "direct"
        if i in good_set
        else "propagated"
        if i < good[0] or i > good[-1]
        else "interpolated"
        for i in timeline
    ]
    return raw, final, methods


def compute_canvas(matrices, width, height, mode="union"):
    corners = np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float)
    polygons = []
    for matrix in matrices:
        denominators = np.column_stack([corners, np.ones(4)]) @ matrix[2]
        if np.min(denominators) <= 0 <= np.max(denominators):
            raise ValueError("Planar warp crosses the projective horizon; choose visual mode.")
        polygons.append(project_points(matrix, corners))
    if mode == "original":
        left, top, out_w, out_h = 0, 0, width, height
    elif mode == "union":
        points = np.vstack(polygons)
        left, top = np.floor(points.min(axis=0))
        right, bottom = np.ceil(points.max(axis=0))
        out_w, out_h = int(right - left), int(bottom - top)
    else:
        intersection = polygons[0].astype(np.float32)
        for polygon in polygons[1:]:
            area, intersection = cv2.intersectConvexConvex(intersection, polygon.astype(np.float32))
            if area <= 0 or intersection is None:
                raise ValueError("No common crop exists across all frames; use canvas union.")
            intersection = intersection.reshape(-1, 2)
        center = intersection.mean(axis=0)
        lo, hi = 0.0, 1.0
        half = np.array([width, height]) / 2
        while all(
            cv2.pointPolygonTest(intersection, tuple(p), False) >= 0
            for p in center + (corners / [width, height] * 2 - 1) * half * hi
        ):
            hi *= 2
        for _ in range(50):
            mid = (lo + hi) / 2
            box = center + (corners / [width, height] * 2 - 1) * half * mid
            if all(cv2.pointPolygonTest(intersection, tuple(p), False) >= 0 for p in box):
                lo = mid
            else:
                hi = mid
        out_w, out_h = (np.floor(np.array([width, height]) * lo / 2) * 2).astype(int)
        left, top = center - np.array([out_w, out_h]) / 2
    if mode != "crop":
        out_w, out_h = int(out_w + out_w % 2), int(out_h + out_h % 2)
    if min(out_w, out_h) < 2 or out_w * out_h > 200_000_000:
        raise ValueError(
            f"Invalid or excessive canvas {out_w}x{out_h}; inspect marker correspondences."
        )
    translation = np.array([[1, 0, -left], [0, 1, -top], [0, 0, 1]], dtype=float)
    return translation, int(out_w), int(out_h)


def reanchor_transforms(matrices, reference_frame):
    """Force the reference frame back to identity after smoothing, before canvas bounds.

    Temporal smoothing moves the reference frame's own transform away from the
    identity, which biases every residual measured against the reference marker
    positions. Left-multiplying the whole timeline by ``inv(M_ref)`` restores the
    invariant without changing relative motion between frames.
    """
    matrices = np.asarray(matrices, dtype=np.float64)
    reference = matrices[int(reference_frame)]
    if not np.isfinite(reference).all() or abs(np.linalg.det(reference)) < 1e-12:
        raise ValueError("Reference transform is singular; cannot re-anchor the timeline.")
    result = np.linalg.inv(reference) @ matrices
    scale = result[:, 2, 2][:, None, None]
    if not np.all(np.abs(scale) > 1e-12):
        raise ValueError("Re-anchoring produced a degenerate projective scale.")
    result = result / scale
    if not np.allclose(result[int(reference_frame)], np.eye(3), atol=1e-9):
        raise ValueError("Re-anchoring failed to restore reference identity.")
    result[int(reference_frame)] = np.eye(3)
    return result


def _normalize_points(points):
    """Hartley isotropic normalization; returns the 3x3 transform and normalized points."""
    center = points.mean(axis=0)
    shifted = points - center
    mean_distance = float(np.mean(np.linalg.norm(shifted, axis=1)))
    scale = math.sqrt(2.0) / mean_distance if mean_distance > 1e-12 else 1.0
    transform = np.array(
        [[scale, 0.0, -scale * center[0]], [0.0, scale, -scale * center[1]], [0.0, 0.0, 1.0]]
    )
    return transform, shifted * scale


def _dlt_homography(src, dst, weights=None):
    """Weighted normalized DLT solved by SVD in float64; no normal-equation inverse."""
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if len(src) < 4 or not np.isfinite(src).all() or not np.isfinite(dst).all():
        return None
    weights = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    to_src, normalized_src = _normalize_points(src)
    to_dst, normalized_dst = _normalize_points(dst)
    rows = []
    for (x, y), (u, v), w in zip(
        normalized_src, normalized_dst, np.sqrt(np.maximum(weights, 1e-12)), strict=True
    ):
        rows.append(w * np.array([-x, -y, -1.0, 0.0, 0.0, 0.0, u * x, u * y, u]))
        rows.append(w * np.array([0.0, 0.0, 0.0, -x, -y, -1.0, v * x, v * y, v]))
    try:
        _, _, vt = np.linalg.svd(np.asarray(rows, dtype=np.float64))
    except np.linalg.LinAlgError:
        return None
    matrix = np.linalg.inv(to_dst) @ vt[-1].reshape(3, 3) @ to_src
    if not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
        return None
    return matrix / matrix[2, 2]


def fit_floor_homography(src, dst, estimator="robust-all", threshold=3.0):
    """Fit a planar homography from declared physical controls.

    ``robust-all`` keeps every declared correspondence and down-weights bad
    clicks with Huber IRLS, because manually declared controls are not outliers
    to be discarded. ``ransac`` keeps the OpenCV subset selection as an
    outlier-resistant alternative. Residuals are always reported over all
    observed controls by the caller.
    """
    src = np.asarray(src, dtype=np.float64)
    dst = np.asarray(dst, dtype=np.float64)
    if estimator == "ransac":
        matrix, inliers = cv2_find_homography(src, dst, threshold)
        if matrix is None or not np.isfinite(matrix).all() or abs(matrix[2, 2]) < 1e-12:
            return None, None, "unavailable"
        return matrix / matrix[2, 2], np.asarray(inliers).ravel().astype(bool), "ransac"
    matrix = _dlt_homography(src, dst)
    if matrix is None:
        return None, None, "unavailable"
    for _ in range(30):
        residual = np.linalg.norm(project_points(matrix, src) - dst, axis=1)
        median = float(np.median(residual))
        cutoff = max(threshold, median + 2.5 * 1.4826 * float(np.median(np.abs(residual - median))))
        updated = _dlt_homography(src, dst, np.minimum(1.0, cutoff / np.maximum(residual, 1e-12)))
        if updated is None or np.allclose(updated, matrix, atol=1e-10, rtol=1e-10):
            break
        matrix = updated
    residual = np.linalg.norm(project_points(matrix, src) - dst, axis=1)
    return matrix, residual <= max(threshold, 1e-9), "robust-all"


def floor_fit_quality(
    n_points, inlier_ratio, rmse_all_px, rmse_inliers_px, max_residual_px, condition, degenerate
):
    """Classify a floor fit as good/usable/poor/unavailable from all-point evidence.

    Only good and usable fits are allowed to contribute imputed visual
    correspondences. Thresholds are expressed in pixels of reprojection error
    over ALL observed controls, so a RANSAC fit that nails a four-point subset
    and misses the remaining physical controls cannot be rated good.
    """
    values = [rmse_all_px, max_residual_px, condition]
    if degenerate or n_points < 4 or not np.all(np.isfinite(values)):
        return "unavailable"
    if condition > 1e10 or rmse_all_px > 25.0 or inlier_ratio < 0.5:
        return "poor"
    if (
        n_points >= 6
        and rmse_all_px <= 4.0
        and inlier_ratio >= 0.8
        and max_residual_px <= 10.0
        and condition <= 1e8
    ):
        return "good"
    if rmse_all_px <= 12.0 and inlier_ratio >= 0.6 and max_residual_px <= 25.0:
        return "usable"
    return "poor"


def build_reference_atlas(
    table,
    ids,
    reference_frame,
    model="similarity",
    min_support=3,
    max_atlas_rmse_px=6.0,
    max_passes=None,
):
    """Extend the canonical target positions beyond the markers of the reference frame.

    The reference frame stays a real frame; no synthetic reference image is
    created. Markers visible in the reference frame seed the atlas, frames that
    share enough atlas markers are solved, and the remaining static markers seen
    in those frames are mapped into canonical coordinates and robustly
    aggregated. A new atlas point is accepted only with enough supporting frames
    and a small enough spread.
    """
    cols = [table.ids.index(i) for i in ids]
    observations = table.xy[:, cols]
    positions = np.full((len(ids), 2), np.nan, dtype=np.float64)
    sources = ["unavailable"] * len(ids)
    support = [0] * len(ids)
    spread = [np.nan] * len(ids)
    seeded = np.isfinite(observations[int(reference_frame)]).all(axis=1)
    positions[seeded] = observations[int(reference_frame)][seeded]
    for index in np.flatnonzero(seeded):
        sources[index] = "reference-frame"
        support[index] = 1
        spread[index] = 0.0
    passes = len(ids) + 1 if max_passes is None else int(max_passes)
    for _ in range(passes):
        known = np.isfinite(positions).all(axis=1)
        if known.all():
            break
        candidates = {index: [] for index in np.flatnonzero(~known)}
        for frame in range(len(observations)):
            present = np.isfinite(observations[frame]).all(axis=1)
            usable = present & known
            missing = present & ~known
            if not missing.any() or not _useful(positions[usable], model):
                continue
            matrix = estimate_transform(
                observations[frame, usable],
                positions[usable],
                np.ones(int(usable.sum())),
                model,
                "robust-lsq",
            )
            if matrix is None:
                continue
            projected = project_points(matrix, observations[frame, missing])
            for index, point in zip(np.flatnonzero(missing), projected, strict=True):
                if np.isfinite(point).all():
                    candidates[index].append(point)
        added = False
        for index, points in candidates.items():
            if len(points) < min_support:
                continue
            stacked = np.asarray(points, dtype=np.float64)
            estimate = np.median(stacked, axis=0)
            rmse = float(np.sqrt(np.mean(np.sum((stacked - estimate) ** 2, axis=1))))
            if rmse > max_atlas_rmse_px:
                continue
            positions[index] = estimate
            sources[index] = "atlas"
            support[index] = len(points)
            spread[index] = rmse
            added = True
        if not added:
            break
    atlas_df = pd.DataFrame(
        {
            "marker_id": list(ids),
            "atlas_source": sources,
            "atlas_support_frames": support,
            "atlas_rmse_px": spread,
        }
    )
    return positions, atlas_df


def impute_floor_markers(table, floor_matrices, quality, geometry, metric_ids, max_gap_frames=5):
    """Reconstruct missing metric controls from trusted floor fits only.

    Returns pixel positions for controls that were not observed, NaN elsewhere.
    Short remaining gaps are filled by interpolating the projected PIXEL
    positions over time; homography coefficients are never interpolated or
    smoothed.
    """
    world = np.array([geometry.world_xy(i) for i in metric_ids], dtype=np.float64)
    cols = [table.ids.index(i) for i in metric_ids]
    imputed = np.full((len(table.frames), len(metric_ids), 2), np.nan, dtype=np.float64)
    trusted = np.array([state in ("good", "usable") for state in quality], dtype=bool)
    for frame in range(len(table.frames)):
        if not trusted[frame] or not np.isfinite(floor_matrices[frame]).all():
            continue
        missing = ~np.isfinite(table.xy[frame, cols]).all(axis=1)
        if missing.any():
            imputed[frame, missing] = project_points(floor_matrices[frame], world[missing])
    if max_gap_frames > 0:
        timeline = np.arange(len(table.frames))
        for col in range(len(metric_ids)):
            available = np.isfinite(imputed[:, col]).all(axis=1)
            if available.sum() < 2:
                continue
            observed = np.isfinite(table.xy[:, cols[col]]).all(axis=1)
            known = np.flatnonzero(available)
            for frame in range(known[0] + 1, known[-1]):
                if available[frame] or observed[frame]:
                    continue
                before = known[known < frame].max()
                after = known[known > frame].min()
                if after - before - 1 > max_gap_frames:
                    continue
                imputed[frame, col] = [
                    np.interp(frame, timeline[[before, after]], imputed[[before, after], col, axis])
                    for axis in (0, 1)
                ]
    return imputed


def solve_floor(table, geometry, metric_ids, estimator="robust-all"):
    """Independent per-frame planar fits from declared metric controls only.

    Residuals are always reported over ALL observed controls, and separately
    over the inlier subset, so a fit that nails four points while missing the
    remaining physical controls cannot masquerade as a good fit.
    """
    matrices = np.full((len(table.frames), 3, 3), np.nan)
    rows = []
    cols = [table.ids.index(i) for i in metric_ids]
    world = np.array([geometry.world_xy(i) for i in metric_ids])
    for frame in table.frames:
        observed = table.xy[frame, cols]
        valid = np.isfinite(observed).all(axis=1)
        src, dst = world[valid], observed[valid]
        row = {
            "frame": int(frame),
            "n_floor_correspondences": int(valid.sum()),
            "n_inliers": 0,
            "floor_inlier_ratio": np.nan,
            "floor_rmse_all_px": np.nan,
            "floor_rmse_inliers_px": np.nan,
            "floor_median_all_px": np.nan,
            "floor_max_residual_px": np.nan,
            "condition": np.nan,
            "degenerate": True,
            "fit_quality": "unavailable",
            "method": "unavailable",
        }
        valid_ids = np.array(metric_ids)[valid].tolist()
        valid_quad = has_non_collinear_quad(
            valid_ids, dict(zip(valid_ids, src, strict=True))
        ) and has_non_collinear_quad(valid_ids, dict(zip(valid_ids, dst, strict=True)))
        if len(src) >= 4 and not valid_quad:
            row["method"] = "degenerate"
        if valid_quad:
            matrix, inliers, method = fit_floor_homography(src, dst, estimator, 3.0)
            if (
                matrix is not None
                and np.isfinite(matrix).all()
                and np.linalg.matrix_rank(matrix) == 3
                and abs(matrix[2, 2]) > 1e-12
            ):
                result = FrameHomography(int(frame), matrix / matrix[2, 2], method, len(src))
                residual = np.linalg.norm(project_points(result.homography, src) - dst, axis=1)
                inliers = np.asarray(inliers).ravel().astype(bool)
                condition = float(np.linalg.cond(matrix))
                degenerate = condition > 1e12
                inlier_residual = residual[inliers]
                quality = floor_fit_quality(
                    len(src),
                    float(inliers.mean()),
                    float(np.sqrt(np.mean(residual**2))),
                    float(np.sqrt(np.mean(inlier_residual**2))) if len(inlier_residual) else np.nan,
                    float(residual.max()),
                    condition,
                    degenerate,
                )
                if not degenerate:
                    matrices[frame] = result.homography
                row.update(
                    n_inliers=int(inliers.sum()),
                    floor_inlier_ratio=float(inliers.mean()),
                    floor_rmse_all_px=float(np.sqrt(np.mean(residual**2))),
                    floor_rmse_inliers_px=(
                        float(np.sqrt(np.mean(inlier_residual**2)))
                        if len(inlier_residual)
                        else np.nan
                    ),
                    floor_median_all_px=float(np.median(residual)),
                    floor_max_residual_px=float(residual.max()),
                    condition=condition,
                    degenerate=degenerate,
                    fit_quality="unavailable" if degenerate else quality,
                    method=result.method if not degenerate else "degenerate",
                )
        rows.append(row)
    floor_df = pd.DataFrame(rows)
    # Documented compatibility alias: the v0.4.1 column was already an all-point RMSE.
    floor_df["floor_rmse_px"] = floor_df["floor_rmse_all_px"]
    return matrices, floor_df


def transform_marker_table(table, matrices):
    result = np.full_like(table.xy, np.nan)
    for frame in table.frames:
        valid = np.isfinite(table.xy[frame]).all(axis=1)
        result[frame, valid] = project_points(matrices[frame], table.xy[frame, valid])
    data = {"frame": table.frames}
    for col, pid in enumerate(table.ids):
        data[f"p{pid}_x"] = result[:, col, 0]
        data[f"p{pid}_y"] = result[:, col, 1]
    return result, pd.DataFrame(data)


def motion_diagnostics(
    table,
    transformed,
    reference_frame,
    canvas_translation,
    stabilization_ids,
    anchor_ids,
    metric_ids,
    reference_positions=None,
    direct_transformed=None,
):
    """Displacement against the canonical targets, split by solver stage.

    ``reference_positions`` are the canonical atlas targets, which may cover
    markers absent from the real reference frame; without them the reference
    frame observations are used. ``direct_transformed`` carries the markers
    warped by the direct per-frame solve, so solver fit quality is reported
    apart from the smoothed and re-anchored output.
    """
    canonical = (
        table.xy[reference_frame]
        if reference_positions is None
        else np.asarray(reference_positions)
    )
    target = project_points(canvas_translation, canonical)
    before = np.linalg.norm(table.xy - canonical, axis=2)
    after = np.linalg.norm(transformed - target, axis=2)
    direct = (
        np.full(before.shape, np.nan)
        if direct_transformed is None
        else np.linalg.norm(direct_transformed - target, axis=2)
    )
    rows = []
    groups = [(f"p{i}", [i]) for i in table.ids]
    groups += [("all", stabilization_ids), ("anchors", anchor_ids), ("floor", metric_ids)]
    for name, ids in groups:
        cols = [table.ids.index(i) for i in ids]
        b, a = before[:, cols].ravel(), after[:, cols].ravel()
        d = direct[:, cols].ravel()
        valid = np.isfinite(b) & np.isfinite(a)
        b, a, d = b[valid], a[valid], d[valid]
        d = d[np.isfinite(d)]
        rows.append(
            {
                "marker": name,
                "n_observations": len(a),
                "rms_before_px": float(np.sqrt(np.mean(b * b))) if len(b) else np.nan,
                "rms_direct_px": float(np.sqrt(np.mean(d * d))) if len(d) else np.nan,
                "rms_after_px": float(np.sqrt(np.mean(a * a))) if len(a) else np.nan,
                "median_before_px": float(np.median(b)) if len(b) else np.nan,
                "median_after_px": float(np.median(a)) if len(a) else np.nan,
            }
        )
    return pd.DataFrame(rows)


def _check_cancel(cancel_event):
    if cancel_event is not None and cancel_event.is_set():
        raise InterruptedError("Cancellation requested; completed outputs are preserved.")


def _parse_rate(value):
    """Parse an FFprobe rational frame rate such as ``60000/1001``."""
    try:
        if isinstance(value, str) and "/" in value:
            numerator, denominator = value.split("/")
            return float(numerator) / float(denominator) if float(denominator) else float("nan")
        return float(value)
    except (TypeError, ValueError, ZeroDivisionError):
        return float("nan")


def detect_variable_frame_rate(metadata, tolerance=1e-3):
    """Report whether the source timebase is variable, from r_frame_rate vs avg_frame_rate."""
    nominal = _parse_rate(metadata.get("r_frame_rate"))
    average = _parse_rate(metadata.get("avg_frame_rate"))
    if not np.isfinite(nominal) or not np.isfinite(average) or average <= 0:
        return False, nominal, average
    return abs(nominal - average) / average > tolerance, nominal, average


def encode_frames_h264(path, size, fps, frame_source, callback=None):
    """Encode BGR frames to H.264 through the repository FFmpeg encoder selection.

    ``frame_source`` is a zero-argument callable returning a fresh iterator of
    frames, so a hardware encoder failure retries on libx264 by re-rendering
    instead of transcoding an intermediate file a second time. Encoder choice
    and encoding arguments come from ``ffmpeg_utils``; none are duplicated here.
    """
    width, height = size
    for encoder in encoders_with_cpu_fallback():
        command = [
            get_video_encode_ffmpeg_path(encoder),
            "-y",
            "-v",
            "error",
            "-f",
            "rawvideo",
            "-pix_fmt",
            "bgr24",
            "-s",
            f"{int(width)}x{int(height)}",
            "-r",
            f"{float(fps):.9f}",
            "-i",
            "pipe:0",
            *get_ffmpeg_video_encoding_args(encoder),
            "-movflags",
            "+faststart",
            str(path),
        ]
        log(f"Encoding H.264 with {describe_video_encoder(encoder)}", callback)
        process = subprocess.Popen(
            command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE
        )
        broken = False
        try:
            for frame in frame_source():
                try:
                    process.stdin.write(np.ascontiguousarray(frame, dtype=np.uint8).tobytes())
                except (BrokenPipeError, OSError):
                    broken = True
                    break
        except BaseException:
            process.kill()
            process.communicate()
            raise
        try:
            process.stdin.close()
        except (BrokenPipeError, OSError):
            broken = True
        stderr = process.communicate()[1]
        if process.returncode == 0 and not broken and Path(path).exists():
            return encoder
        message = (stderr or b"").decode("utf-8", "replace")[-500:]
        log(f"H.264 encode with {encoder} failed: {message}", callback)
        if not is_hardware_video_encoder(encoder):
            raise RuntimeError(f"FFmpeg H.264 encoding failed with {encoder}: {message}")
    raise RuntimeError("No usable FFmpeg H.264 encoder produced the production video.")


def validate_output_media(path, width, height, frames, fps, expect_audio):
    """Reopen the finished MP4 and prove it is structurally what the run promised."""
    metadata = get_precise_video_metadata(path)
    if not metadata or not metadata.get("fps"):
        raise RuntimeError(f"Production video is unreadable: {path}")
    streams = metadata.get("_raw_json", {}).get("streams", [])
    audio_streams = [stream for stream in streams if stream.get("codec_type") == "audio"]
    video_streams = [stream for stream in streams if stream.get("codec_type") == "video"]
    counted = int(metadata.get("nb_frames") or 0)
    measured_fps = float(metadata["fps"])
    duration_error = abs(float(metadata.get("duration") or 0.0) - frames / fps)
    capture = cv2.VideoCapture(str(path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, max(0, frames - 1))
        last_frame_ok = bool(capture.read()[0])
    finally:
        capture.release()
    result = {
        "video_opens": True,
        "video_codec": metadata.get("codec"),
        "width": int(metadata["width"]),
        "height": int(metadata["height"]),
        "size_ok": (int(metadata["width"]), int(metadata["height"])) == (int(width), int(height)),
        "frame_count": counted,
        "frame_count_ok": counted == int(frames),
        "fps": measured_fps,
        "fps_ok": abs(measured_fps - float(fps)) <= max(1e-3, 1e-4 * float(fps)),
        "duration_error_s": duration_error,
        "duration_ok": duration_error <= max(2.0 / float(fps), 0.05),
        "audio_present": bool(audio_streams),
        "audio_ok": bool(audio_streams) == bool(expect_audio),
        "last_frame_ok": last_frame_ok,
        "video_start_time_s": float(video_streams[0].get("start_time", 0.0))
        if video_streams
        else None,
        "audio_start_time_s": float(audio_streams[0].get("start_time", 0.0))
        if audio_streams
        else None,
    }
    result["codec_ok"] = str(result["video_codec"] or "").lower() in ("h264", "avc1")
    result["failures"] = [
        name
        for name in (
            "size_ok",
            "frame_count_ok",
            "fps_ok",
            "duration_ok",
            "audio_ok",
            "last_frame_ok",
        )
        if not result[name]
    ]
    result["valid"] = not result["failures"]
    return result


def _finish_video(silent, source, output, metadata, preserve_audio, callback):
    """Mux the source audio without re-encoding the freshly encoded H.264 video.

    The encoded video already holds exactly the rendered frames, so no ``-t``
    truncation and no blind ``-ss`` seek derived from the video stream's
    start time is applied to the audio input: an unrelated offset there would
    desynchronize rather than align. ``-shortest`` keeps a longer source audio
    track from extending the output past the rendered frames.
    """
    streams = metadata.get("_raw_json", {}).get("streams", [])
    has_audio = any(s.get("codec_type") == "audio" for s in streams)
    if not preserve_audio or not has_audio:
        silent.replace(output)
        return "disabled" if not preserve_audio else "source has no audio"
    video_stream = next((s for s in streams if s.get("codec_type") == "video"), {})
    audio_stream = next(s for s in streams if s.get("codec_type") == "audio")
    skew = float(audio_stream.get("start_time", 0.0)) - float(video_stream.get("start_time", 0.0))
    if abs(skew) > 0.5 / max(float(metadata.get("fps") or 1.0), 1e-9):
        log(
            f"Source audio starts {skew:+.4f} s from the video stream; copying timestamps as-is.",
            callback,
        )
    log("Preserving source audio (copy; AAC fallback if needed)...", callback)
    for codec in ("copy", "aac"):
        command = [
            get_ffmpeg_path(),
            "-y",
            "-v",
            "error",
            "-i",
            str(silent),
            "-i",
            str(source),
            "-map",
            "0:v:0",
            "-map",
            "1:a:0",
            "-c:v",
            "copy",
            "-c:a",
            codec,
            "-shortest",
            "-movflags",
            "+faststart",
            str(output),
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode == 0:
            return f"preserved ({codec})"
        log(f"Audio mux with {codec} failed: {result.stderr[-500:]}", callback)
    raise RuntimeError(
        "Could not preserve source audio. Inspect FFmpeg errors or explicitly use --no-audio."
    )


def _overlay(frame, source_frame, table, frame_id, matrix, target, method, floor_rmse):
    """Side-by-side source and stabilized coordinates, reference targets and residuals."""
    height, width = frame.shape[:2]
    panel = np.zeros_like(frame)
    source = source_frame.copy()
    for col, pid in enumerate(table.ids):
        point = table.xy[frame_id, col]
        if np.isfinite(point).all():
            xy = tuple(np.round(point).astype(int))
            cv2.circle(source, xy, 5, (0, 255, 0), 2)
            cv2.putText(source, f"p{pid}", xy, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            dst = tuple(np.round(project_points(matrix, point[None])[0]).astype(int))
            cv2.circle(frame, dst, 5, (0, 255, 0), 2)
            cv2.putText(frame, f"p{pid}", dst, cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
            if np.isfinite(target[col]).all():
                ref = tuple(np.round(target[col]).astype(int))
                cv2.drawMarker(frame, ref, (0, 255, 255), cv2.MARKER_CROSS, 14, 2)
                cv2.line(frame, dst, ref, (0, 0, 255), 2)
    ratio = min(width / source.shape[1], height / source.shape[0])
    small = cv2.resize(source, (int(source.shape[1] * ratio), int(source.shape[0] * ratio)))
    panel[: small.shape[0], : small.shape[1]] = small
    params = matrix_to_params(matrix)
    label = f"frame {frame_id} {method} rot={np.degrees(params[2]):.2f} scale={np.exp(params[3]):.4f} xy={params[0]:.1f},{params[1]:.1f} floor={floor_rmse:.2f}"
    combined = np.hstack([panel, frame])
    cv2.putText(combined, label, (12, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    return combined


def _write_report(path, summary, diagnostics):
    plotted = diagnostics[diagnostics.marker.isin(["all", "anchors", "floor"])].dropna()
    maximum = (
        max(plotted.rms_before_px.max(), plotted.rms_after_px.max(), 1.0) if len(plotted) else 1.0
    )
    bars = []
    for i, row in enumerate(plotted.itertuples()):
        y = 30 + i * 55
        bars.append(
            f'<text x="5" y="{y + 15}">{html.escape(row.marker)}</text><rect x="90" y="{y}" width="{400 * row.rms_before_px / maximum:.1f}" height="15" fill="#64748b"/><rect x="90" y="{y + 18}" width="{400 * row.rms_after_px / maximum:.1f}" height="15" fill="#15803d"/>'
        )
    rows = []
    for key, value in summary.items():
        if isinstance(value, dict):
            for inner, item in value.items():
                rows.append((f"{key}.{inner}", item))
        else:
            rows.append((key, value))
    entries = "".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{html.escape(str(v))}</td></tr>" for k, v in rows
    )
    note = summary.get("planar_residual_note")
    caveats = (
        "<h2>Limitations</h2><ul>"
        "<li>One global similarity cannot remove true multi-depth parallax from camera translation.</li>"
        "<li>The metric floor homography is exact only on the floor plane; it is never the default whole-frame visual warp.</li>"
        "<li>Black borders are genuinely uncaptured image area and are never filled with invented content.</li>"
        "<li>Variable-frame-rate sources are rendered at the average FPS; see <code>timestamp_mode</code>.</li>"
        "<li>Lens distortion is not corrected in v0.4.2; only diagnosed.</li>"
        "</ul>"
    )
    if note:
        caveats = f"<h2>Planar residual diagnostic</h2><p>{html.escape(str(note))}</p>" + caveats
    path.write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8"><title>vailá stabilization report</title><style>body{font:16px sans-serif;max-width:1100px;margin:30px auto;padding:20px}td,th{padding:8px;border:1px solid #ccc;text-align:left}table{border-collapse:collapse}svg{max-width:100%}</style><h1>Fixed-scene stabilization</h1><p>Gray: before RMS; green: after RMS (pixels). Parallax between floor and walls can remain under a shape-preserving global transform. Floor reprojection error alone does not prove visual stabilization.</p><svg width="560" height="210">'
        + "".join(bars)
        + "</svg><table>"
        + entries
        + "</table><h2>Marker and group displacement</h2>"
        + diagnostics.to_html(index=False, float_format=lambda v: f"{v:.3f}")
        + caveats
        + "</html>",
        encoding="utf-8",
    )


def run_video_stabilizer(
    video_path: Path,
    markers_csv: Path,
    output_dir: Path | None = None,
    *,
    mode="visual",
    model="similarity",
    stabilization_markers="all",
    anchor_markers=None,
    anchor_weight=2.0,
    metric_markers=None,
    geometry_config=None,
    reference="auto",
    smooth="savgol",
    canvas="union",
    preserve_audio=True,
    debug_overlay=False,
    estimator="robust-lsq",
    border="black",
    hybrid_imputed_weight=0.25,
    floor_estimator="robust-all",
    progress_callback=None,
    cancel_event=None,
) -> dict[str, Path]:
    """Estimate first, then render every source frame on a fixed canvas into fresh outputs."""
    for name, value, choices in (
        ("mode", mode, ("visual", "hybrid", "floor-lock")),
        ("model", model, ("similarity", "affine")),
        ("estimator", estimator, ("robust-lsq", "ransac")),
        ("smooth", smooth, ("none", "savgol", "lowpass")),
        ("canvas", canvas, ("union", "original", "crop")),
        ("border", border, ("black",)),
        ("floor-estimator", floor_estimator, ("robust-all", "ransac")),
    ):
        if value not in choices:
            raise ValueError(f"Invalid {name}: {value}")
    if not np.isfinite(anchor_weight) or anchor_weight <= 0:
        raise ValueError("Anchor weight must be positive and finite.")
    if not np.isfinite(hybrid_imputed_weight) or not 0 < hybrid_imputed_weight <= 1:
        raise ValueError("Hybrid imputed weight must lie in (0, 1]; imputed points never dominate.")
    video_path, markers_csv = (
        Path(video_path).expanduser().resolve(),
        Path(markers_csv).expanduser().resolve(),
    )
    if mode != "visual" and not geometry_config:
        raise ValueError(f"{mode} requires --geometry-config (visual mode does not).")
    metadata = get_precise_video_metadata(video_path)
    if not metadata or not metadata.get("nb_frames") or metadata["fps"] <= 0:
        raise ValueError(f"Cannot read video timeline metadata: {video_path}")
    n, fps = int(metadata["nb_frames"]), float(metadata["fps"])
    width, height = metadata["width"], metadata["height"]
    table = load_marker_csv(markers_csv, n)
    ids = parse_id_spec(stabilization_markers, table.ids)
    anchors = parse_id_spec(anchor_markers, table.ids) if anchor_markers else []
    if not set(anchors).issubset(ids):
        raise ValueError("Anchor markers must be included in stabilization markers.")
    ref = choose_reference_frame(table, ids, anchors, reference, model)
    log(
        f"Reference frame {ref}; visual markers {ids}; anchors {anchors}; {n} frames at {fps:.9f} FPS",
        progress_callback,
    )
    if model == "affine":
        log("Affine research mode allows shear and independent axis scales.", progress_callback)
    metric_ids, floor_matrices, floor_df = [], None, None
    if geometry_config:
        geometry = load_target_geometry(geometry_config)
        requested = parse_id_spec(metric_markers, table.ids) if metric_markers else table.ids
        metric_ids = sorted(set(requested) & set(geometry.sorted_ids))
        if len(metric_ids) < 4:
            raise ValueError(
                "Geometry diagnostics need at least four metric markers shared by CSV and TOML."
            )
        if any(
            not np.isfinite(
                [geometry.points[i].x, geometry.points[i].y, geometry.points[i].z]
            ).all()
            or abs(geometry.points[i].z) > 1e-9
            for i in metric_ids
        ):
            raise ValueError("Metric controls must be finite points on the Z=0 plane.")
        floor_matrices, floor_df = solve_floor(table, geometry, metric_ids, floor_estimator)
        quality_counts = floor_df.fit_quality.value_counts().to_dict()
        log(
            f"Floor-only metric IDs: {metric_ids}; {floor_estimator} fits by quality: {quality_counts}",
            progress_callback,
        )
    cols = [table.ids.index(i) for i in ids]
    atlas_positions, atlas_df = build_reference_atlas(table, ids, ref, model)
    extra = atlas_df.atlas_source.eq("atlas").sum()
    if extra:
        log(
            f"Reference atlas added {extra} static marker target(s) absent from frame {ref}.",
            progress_callback,
        )
    if atlas_df.atlas_source.eq("unavailable").any():
        missing = atlas_df.marker_id[atlas_df.atlas_source.eq("unavailable")].tolist()
        log(f"Markers without a canonical target (ignored): {missing}", progress_callback)
    canonical = table.xy[ref].copy()
    for col, pid in enumerate(ids):
        canonical[table.ids.index(pid)] = atlas_positions[col]
    imputed_xy, imputed_columns = None, {}
    if mode == "hybrid" and floor_matrices is not None:
        imputed_xy = impute_floor_markers(
            table, floor_matrices, floor_df.fit_quality.tolist(), geometry, metric_ids
        )
        imputed_columns = {ids.index(pid): k for k, pid in enumerate(metric_ids) if pid in ids}
        log(
            f"Hybrid imputation: {int(np.isfinite(imputed_xy).all(axis=2).sum())} reconstructed "
            f"floor observations at weight {hybrid_imputed_weight}",
            progress_callback,
        )
    matrices, counts, manual_counts, imputed_counts = [], [], [], []
    for frame in table.frames:
        _check_cancel(cancel_event)
        current, target = table.xy[frame, cols].copy(), atlas_positions
        usable_target = np.isfinite(target).all(axis=1)
        manual = np.isfinite(current).all(axis=1) & usable_target
        imputed = np.zeros(len(ids), dtype=bool)
        for col, source_col in imputed_columns.items():
            if (
                not manual[col]
                and usable_target[col]
                and np.isfinite(imputed_xy[frame, source_col]).all()
            ):
                current[col] = imputed_xy[frame, source_col]
                imputed[col] = True
        valid = manual | imputed
        src, dst = current[valid], target[valid]
        weights = spatial_weights(dst, width, height)
        for col, pid in enumerate(np.array(ids)[valid]):
            if pid in anchors:
                weights[col] *= anchor_weight
        # Manual observations always outweigh reconstructed ones.
        weights[imputed[valid]] *= hybrid_imputed_weight
        matrices.append(estimate_transform(src, dst, weights, model, estimator))
        counts.append(int(valid.sum()))
        manual_counts.append(int(manual.sum()))
        imputed_counts.append(int(imputed.sum()))
    raw, final_params, methods = regularize_transforms(matrices, fps, smooth, model)
    n_params = final_params.shape[1]
    direct_params = np.array(
        [
            matrix_to_params(matrix, model) if matrix is not None else np.full(n_params, np.nan)
            for matrix in matrices
        ]
    )
    if smooth != "none" and n < (5 if smooth == "savgol" else 10):
        log(
            f"Clip too short for {smooth}; keeping interpolated parameters unsmoothed.",
            progress_callback,
        )
    final = np.array([params_to_matrix(p, model) for p in final_params])
    if mode == "floor-lock":
        log(
            "WARNING: floor-lock is planar-only. Non-coplanar people, walls and columns can distort. Temporal visual smoothing is not applied.",
            progress_callback,
        )
        good = np.flatnonzero(
            np.isfinite(floor_matrices).all(axis=(1, 2)) & ~floor_df.degenerate.to_numpy()
        )
        if not len(good) or ref not in good:
            raise ValueError(
                "Floor-lock requires a valid reference floor homography. Choose a solved reference frame."
            )
        counts = floor_df.n_floor_correspondences.tolist()
        for frame in table.frames:
            nearest = int(good[np.argmin(abs(good - frame))])
            final[frame] = np.linalg.solve(floor_matrices[nearest].T, floor_matrices[ref].T).T
            final[frame] /= final[frame, 2, 2]
            methods[frame] = "floor-direct" if frame in good else "floor-propagated"
    for frame, method in enumerate(methods):
        if "propagated" in method or "interpolated" in method:
            log(f"Frame {frame}: {method}", progress_callback)
    reanchored = reanchor_transforms(final, ref)
    reanchored_params = np.array([matrix_to_params(m, model) for m in reanchored])
    translation, out_w, out_h = compute_canvas(reanchored, width, height, canvas)
    final = translation @ reanchored
    if not np.isfinite(final).all():
        raise ValueError("Non-finite final transforms; inspect observations.")
    canvas_offset = (float(translation[0, 2]), float(translation[1, 2]))
    output_dir = (
        Path(output_dir).expanduser().resolve()
        if output_dir
        else video_path.parent / ("vaila_stabilized_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    )
    if output_dir.exists() and any(output_dir.iterdir()):
        raise FileExistsError(
            f"Output directory must be empty; refusing to overwrite: {output_dir}"
        )
    output_dir.mkdir(parents=True, exist_ok=True)
    outputs = {
        "video": output_dir / f"{video_path.stem}_stabilized.mp4",
        "transforms": output_dir / "stabilization_transforms.csv",
        "markers": output_dir / "stabilized_markers.csv",
        "diagnostics": output_dir / "stabilization_diagnostics.csv",
        "report": output_dir / "stabilization_report.html",
    }
    transformed, wide = transform_marker_table(table, final)
    direct_output = np.array(
        [
            translation @ matrix if matrix is not None else np.full((3, 3), np.nan)
            for matrix in matrices
        ]
    )
    direct_transformed = transform_marker_table(table, direct_output)[0]
    diagnostics = motion_diagnostics(
        table,
        transformed,
        ref,
        translation,
        ids,
        anchors,
        metric_ids,
        reference_positions=canonical,
        direct_transformed=direct_transformed,
    )
    target = project_points(translation, canonical)
    residual = np.sum((transformed - target) ** 2, axis=2)
    rows = []
    for frame in table.frames:
        row = {
            "frame": int(frame),
            "reference_frame": ref,
            "solve_method": methods[frame],
            "n_markers": counts[frame],
            "n_manual_markers": manual_counts[frame],
            "n_imputed_markers": imputed_counts[frame],
            "direct_tx": direct_params[frame, 0],
            "direct_ty": direct_params[frame, 1],
            "direct_rotation_deg": np.degrees(direct_params[frame, 2]),
            "direct_scale": np.exp(direct_params[frame, 3]),
            "filled_tx": raw[frame, 0],
            "filled_ty": raw[frame, 1],
            "filled_rotation_deg": np.degrees(raw[frame, 2]),
            "filled_scale": np.exp(raw[frame, 3]),
            "smoothed_tx": final_params[frame, 0],
            "smoothed_ty": final_params[frame, 1],
            "smoothed_rotation_deg": np.degrees(final_params[frame, 2]),
            "smoothed_scale": np.exp(final_params[frame, 3]),
            "reanchored_tx": reanchored_params[frame, 0],
            "reanchored_ty": reanchored_params[frame, 1],
            "reanchored_rotation_deg": np.degrees(reanchored_params[frame, 2]),
            "reanchored_scale": np.exp(reanchored_params[frame, 3]),
            "canvas_offset_x": canvas_offset[0],
            "canvas_offset_y": canvas_offset[1],
            "is_interpolated": methods[frame].endswith("interpolated"),
            "is_propagated": methods[frame].endswith("propagated"),
        }
        for group, selected in (("marker", ids), ("anchor", anchors), ("floor_marker", metric_ids)):
            values = residual[frame, [table.ids.index(i) for i in selected]]
            values = values[np.isfinite(values)]
            row[group + "_rmse_px"] = float(np.sqrt(values.mean())) if len(values) else np.nan
        for i in range(3):
            for j in range(3):
                row[f"output_matrix_{i}{j}"] = final[frame, i, j]
        if model == "affine":
            row.update(
                direct_scale_y=np.exp(direct_params[frame, 4]),
                direct_shear=direct_params[frame, 5],
                filled_scale_y=np.exp(raw[frame, 4]),
                filled_shear=raw[frame, 5],
                smoothed_scale_y=np.exp(final_params[frame, 4]),
                smoothed_shear=final_params[frame, 5],
                reanchored_scale_y=np.exp(reanchored_params[frame, 4]),
                reanchored_shear=reanchored_params[frame, 5],
            )
        rows.append(row)
    transforms_df = pd.DataFrame(rows)
    if mode == "floor-lock":
        # Similarity parameters have no physical meaning for a projective warp.
        stages = [
            f"{stage}_{name}"
            for stage in ("direct", "filled", "smoothed", "reanchored")
            for name in ("tx", "ty", "rotation_deg", "scale")
        ]
        transforms_df[stages] = np.nan
    # Documented v0.4.1 compatibility aliases for the renamed stage columns.
    transforms_df["method"] = transforms_df.solve_method
    for old, new in (
        ("tx_raw", "filled_tx"),
        ("ty_raw", "filled_ty"),
        ("rotation_deg_raw", "filled_rotation_deg"),
        ("scale_raw", "filled_scale"),
        ("rotation_deg", "reanchored_rotation_deg"),
        ("scale", "reanchored_scale"),
    ):
        transforms_df[old] = transforms_df[new]
    transforms_df["tx"] = transforms_df.output_matrix_02
    transforms_df["ty"] = transforms_df.output_matrix_12
    for i in range(3):
        for j in range(3):
            transforms_df[f"matrix_{i}{j}"] = transforms_df[f"output_matrix_{i}{j}"]
    log(f"Rendering {n} frames on one {out_w}x{out_h} canvas", progress_callback)
    border_fractions = []
    with tempfile.TemporaryDirectory(prefix=".render_", dir=output_dir) as temporary:
        silent = Path(temporary) / "silent.mp4"
        capture = cv2.VideoCapture(str(video_path))
        writer = cv2.VideoWriter(str(silent), cv2.VideoWriter_fourcc(*"mp4v"), fps, (out_w, out_h))
        debug_writer = None
        try:
            if not writer.isOpened() or not capture.isOpened():
                raise RuntimeError("Could not open video decoder/encoder.")
            if debug_overlay:
                outputs["overlay"] = output_dir / "debug_overlay.mp4"
                debug_writer = cv2.VideoWriter(
                    str(outputs["overlay"]),
                    cv2.VideoWriter_fourcc(*"mp4v"),
                    fps,
                    (2 * out_w, out_h),
                )
                if not debug_writer.isOpened():
                    raise RuntimeError("Could not open debug overlay writer.")
            for frame_id in table.frames:
                _check_cancel(cancel_event)
                ok, frame = capture.read()
                if not ok:
                    raise RuntimeError(
                        f"Video ended early at frame {frame_id}; expected {n} frames."
                    )
                if frame.shape[:2] != (height, width):
                    raise RuntimeError(
                        "Decoded orientation differs from metadata; marker coordinates must use displayed video orientation."
                    )
                if mode == "floor-lock":
                    stable = cv2.warpPerspective(
                        frame, final[frame_id], (out_w, out_h), flags=cv2.INTER_CUBIC
                    )
                else:
                    stable = cv2.warpAffine(
                        frame, final[frame_id, :2], (out_w, out_h), flags=cv2.INTER_CUBIC
                    )
                writer.write(stable)
                # Geometric border estimate; dark image content is not mistaken for borders.
                polygon = project_points(
                    final[frame_id],
                    np.array([[0, 0], [width, 0], [width, height], [0, height]], dtype=float),
                ).astype(np.float32)
                area, _ = cv2.intersectConvexConvex(
                    polygon, np.array([[0, 0], [out_w, 0], [out_w, out_h], [0, out_h]], np.float32)
                )
                border_fractions.append(max(0.0, 1 - area / (out_w * out_h)))
                if debug_writer is not None:
                    rmse = (
                        float(floor_df.iloc[frame_id].floor_rmse_px)
                        if floor_df is not None
                        else np.nan
                    )
                    debug_writer.write(
                        _overlay(
                            stable.copy(),
                            frame,
                            table,
                            frame_id,
                            final[frame_id],
                            target,
                            methods[frame_id],
                            rmse,
                        )
                    )
                if frame_id % 25 == 0 or frame_id == n - 1:
                    log(f"Encoded {frame_id + 1}/{n} frames", progress_callback)
            if capture.read()[0]:
                raise RuntimeError(
                    "Video contains more frames than metadata; refusing a truncated output."
                )
        finally:
            capture.release()
            writer.release()
            if debug_writer is not None:
                debug_writer.release()
        _check_cancel(cancel_event)
        audio_status = _finish_video(
            silent, video_path, outputs["video"], metadata, preserve_audio, progress_callback
        )
    transforms_df.to_csv(outputs["transforms"], index=False)
    wide.to_csv(outputs["markers"], index=False)
    diagnostics.to_csv(outputs["diagnostics"], index=False)
    if floor_df is not None:
        outputs["floor_homographies"] = output_dir / "floor_homographies.npz"
        outputs["floor_diagnostics"] = output_dir / "floor_diagnostics.csv"
        np.savez_compressed(
            outputs["floor_homographies"],
            H=floor_matrices,
            frame_ids=table.frames,
            metric_ids=np.array(metric_ids),
        )
        floor_df.to_csv(outputs["floor_diagnostics"], index=False)
    summary = {
        "version": "0.4.1",
        "video": str(video_path),
        "markers": str(markers_csv),
        "mode": mode,
        "model": model,
        "estimator": estimator,
        "smooth": smooth,
        "reference_frame": ref,
        "stabilization_ids": ids,
        "anchor_ids": anchors,
        "metric_ids": metric_ids,
        "frames": n,
        "fps": fps,
        "canvas": f"{out_w}x{out_h}",
        "canvas_policy": canvas,
        "direct_percent": 100 * sum("direct" in m for m in methods) / n,
        "interpolated_percent": 100 * sum("direct" not in m for m in methods) / n,
        "max_scale_deviation": float(
            np.max(
                abs(np.exp(final_params[:, 3:5] if model == "affine" else final_params[:, 3:4]) - 1)
            )
        ),
        "max_rotation_deg": float(np.max(abs(np.degrees(final_params[:, 2])))),
        "mean_black_border_fraction": float(np.mean(border_fractions)),
        "audio": audio_status,
    }
    if mode == "floor-lock":
        summary.update(
            model="homography (floor-only)",
            requested_model=model,
            smooth="none (floor-lock)",
            max_scale_deviation="not applicable (projective)",
            max_rotation_deg="not applicable (projective)",
        )
    _write_report(outputs["report"], summary, diagnostics)
    outputs["summary"] = output_dir / "stabilization_summary.json"
    outputs["summary"].write_text(json.dumps(summary, indent=2), encoding="utf-8")
    for group in ("all", "anchors", "floor"):
        item = diagnostics[diagnostics.marker.eq(group)].iloc[0]
        log(
            f"{group}: RMS {item.rms_before_px:.3f} -> {item.rms_after_px:.3f} px",
            progress_callback,
        )
    log(f"Complete. Audio: {audio_status}. Report: {outputs['report']}", progress_callback)
    return outputs


def build_parser():
    parser = argparse.ArgumentParser(
        description="Stabilize a fixed scene using pixel markers. No metric geometry is required in visual mode."
    )
    parser.add_argument("--video", type=Path)
    parser.add_argument("--markers", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--mode", choices=("visual", "hybrid", "floor-lock"), default="visual")
    parser.add_argument("--model", choices=("similarity", "affine"), default="similarity")
    parser.add_argument("--estimator", choices=("robust-lsq", "ransac"), default="robust-lsq")
    parser.add_argument("--stabilization-markers", default="all")
    parser.add_argument("--anchor-markers")
    parser.add_argument("--anchor-weight", type=float, default=2.0)
    parser.add_argument("--metric-markers")
    parser.add_argument("--geometry-config", type=Path)
    parser.add_argument("--reference", default="auto", help="auto, first or frame:N (zero-based)")
    parser.add_argument("--smooth", choices=("none", "savgol", "lowpass"), default="savgol")
    parser.add_argument("--canvas", choices=("union", "original", "crop"), default="union")
    parser.add_argument("--border", choices=("black",), default="black")
    parser.add_argument("--debug-overlay", action="store_true")
    parser.add_argument("--no-audio", action="store_true")
    return parser


def _next_available_output_dir(output_dir):
    """Bump a non-empty existing output dir to `<dir>_v2`, `_v3`, ... instead of failing.

    CLI/GUI entry points call this before `run_video_stabilizer` so reruns with the
    same --output-dir don't require the user to pick a fresh empty folder each time.
    Direct callers of `run_video_stabilizer` keep the strict FileExistsError contract.
    """
    candidate = Path(output_dir).expanduser()
    if not candidate.exists() or not any(candidate.iterdir()):
        return str(candidate)
    version = 2
    while True:
        versioned = candidate.parent / f"{candidate.name}_v{version}"
        if not versioned.exists() or not any(versioned.iterdir()):
            return str(versioned)
        version += 1


def _run_args(args, **kwargs):
    options = vars(args).copy()
    options["video_path"] = options.pop("video")
    options["markers_csv"] = options.pop("markers")
    options["preserve_audio"] = not options.pop("no_audio")
    if options.get("output_dir"):
        options["output_dir"] = _next_available_output_dir(options["output_dir"])
    return run_video_stabilizer(**options, **kwargs)


def format_cli_command(argv):
    command = [sys.executable, "-m", "vaila.video_stabilizer", *map(str, argv)]
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


class StabilizerGUI:
    """One window, one worker; only the main thread reads or updates Tk state."""

    def __init__(self, parent=None):
        import tkinter as tk
        from tkinter import filedialog, ttk

        self.root = tk.Toplevel(parent) if parent is not None else tk.Tk()
        self.owns_root = parent is None
        self.root.title("vailá - Video Stabilizer")
        self.root.geometry("850x740")
        self.root.minsize(720, 640)
        if parent is not None:
            self.root.transient(parent)
        self.root.lift()
        self.events = queue.Queue()
        self.cancel = threading.Event()
        self.worker = None
        self.closing = False
        self.outputs = {}
        self.vars = {}
        form = ttk.Frame(self.root, padding=12)
        form.pack(fill="x")
        form.columnconfigure(1, weight=1)
        ttk.Label(
            form, text="Fixed scene → steady video. Metric coordinates are optional in visual mode."
        ).grid(row=0, column=0, columnspan=3, sticky="w", pady=6)
        repo_root = Path(__file__).resolve().parent.parent
        sample_defaults = {
            "video": repo_root / "tests/video_stabilizer/tatame.mp4",
            "markers": repo_root / "tests/video_stabilizer/tatame_markers.csv",
            "geometry-config": repo_root / "vaila/models/planar_targets/tatame_1x1m.toml",
        }
        for row, (key, label) in enumerate(
            (
                ("video", "Video"),
                ("markers", "Marker CSV"),
                ("output-dir", "Output directory (empty)"),
                ("geometry-config", "Geometry TOML (optional)"),
            ),
            1,
        ):
            default_path = sample_defaults.get(key)
            default_value = str(default_path) if default_path and default_path.is_file() else ""
            var = tk.StringVar(self.root, value=default_value)
            self.vars[key] = var
            ttk.Label(form, text=label).grid(row=row, column=0, sticky="w")
            entry = ttk.Entry(form, textvariable=var)
            entry.grid(row=row, column=1, sticky="ew", padx=6, pady=3)
            if key == "video":
                entry.focus_set()

            def browse(k=key, v=var):
                value = (
                    filedialog.askdirectory(parent=self.root)
                    if k == "output-dir"
                    else filedialog.askopenfilename(parent=self.root)
                )
                if value:
                    v.set(value)

            ttk.Button(form, text="Browse", command=browse).grid(row=row, column=2)
        options = ttk.Frame(self.root, padding=(12, 0))
        options.pack(fill="x")
        fields = [
            ("mode", "Mode", "visual", ("visual", "hybrid", "floor-lock")),
            ("model", "Model", "similarity", ("similarity", "affine")),
            ("estimator", "Estimator", "robust-lsq", ("robust-lsq", "ransac")),
            ("reference", "Reference", "auto", None),
            ("smooth", "Smoothing", "savgol", ("none", "savgol", "lowpass")),
            ("canvas", "Canvas", "union", ("union", "original", "crop")),
            ("stabilization-markers", "Static markers", "all", None),
            ("anchor-markers", "Priority anchors", "", None),
            ("anchor-weight", "Anchor weight", "2.0", None),
            ("metric-markers", "Metric markers", "", None),
        ]
        for index, (key, label, value, choices) in enumerate(fields):
            row, col = divmod(index, 2)
            self.vars[key] = tk.StringVar(self.root, value)
            ttk.Label(options, text=label).grid(row=row, column=col * 2, sticky="w", padx=5, pady=3)
            widget = (
                ttk.Combobox(options, textvariable=self.vars[key], values=choices, state="readonly")
                if choices
                else ttk.Entry(options, textvariable=self.vars[key])
            )
            widget.grid(row=row, column=col * 2 + 1, sticky="ew", padx=5)
            options.columnconfigure(col * 2 + 1, weight=1)
        ttk.Label(
            self.root,
            text="Marker IDs: all or 0-7,10. Reference: auto, first or frame:123.\nFloor-lock can distort people/walls. Union preserves captured content with black borders.",
            padding=12,
        ).pack(anchor="w")
        flags = ttk.Frame(self.root, padding=(12, 0))
        flags.pack(fill="x")
        for key, label in (
            ("no-audio", "Disable audio"),
            ("debug-overlay", "Separate debug overlay"),
        ):
            self.vars[key] = tk.BooleanVar(self.root, False)
            ttk.Checkbutton(flags, text=label, variable=self.vars[key]).pack(side="left", padx=8)
        actions = ttk.Frame(self.root, padding=12)
        actions.pack(fill="x")
        self.run_button = ttk.Button(actions, text="Stabilize", command=self.start)
        self.run_button.pack(side="left")
        self.cancel_button = ttk.Button(
            actions, text="Cancel", command=self.request_cancel, state="disabled"
        )
        self.cancel_button.pack(side="left", padx=8)
        ttk.Button(actions, text="Help", command=self.open_help).pack(side="right")
        self.report_button = ttk.Button(
            actions, text="Open report", command=self.open_report, state="disabled"
        )
        self.report_button.pack(side="right", padx=8)
        self.status = tk.StringVar(self.root, "Ready")
        ttk.Label(self.root, textvariable=self.status, padding=8).pack(fill="x")
        self.text = tk.Text(self.root, height=10, wrap="word", state="disabled")
        self.text.pack(fill="both", expand=True, padx=12, pady=(0, 12))
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self.root.bind("<Escape>", lambda _event: self.close())
        self.root.bind("<Control-Return>", lambda _event: self.start())
        self.root.after(75, self.drain)

    def start(self):
        if self.worker is not None and self.worker.is_alive():
            return
        argv = []
        for key, var in self.vars.items():
            value = var.get()
            if isinstance(value, bool):
                if value:
                    argv.append("--" + key)
            elif value.strip():
                argv.extend(["--" + key, value.strip()])
        try:
            args = build_parser().parse_args(argv)
            if not args.video or not args.markers:
                raise ValueError("Select a video and marker CSV.")
        except (SystemExit, ValueError) as exc:
            self.status.set(f"Invalid parameters: {exc}")
            return
        print_gui_cli_mirror(
            "vaila/video_stabilizer", ["uv", "run", "vaila/video_stabilizer.py", *map(str, argv)]
        )
        self.cancel.clear()
        self.run_button.configure(state="disabled")
        self.cancel_button.configure(state="normal")
        self.report_button.configure(state="disabled")
        self.status.set("Estimating transforms...")

        def work():
            try:
                outputs = _run_args(
                    args,
                    progress_callback=lambda message: self.events.put(("progress", message)),
                    cancel_event=self.cancel,
                )
                self.events.put(("done", outputs))
            except Exception as exc:
                log(str(exc))
                self.events.put(("error", str(exc)))

        self.worker = threading.Thread(target=work, daemon=False)
        self.worker.start()

    def request_cancel(self):
        self.cancel.set()
        self.status.set("Cancellation requested; waiting for the current encoding/mux step.")

    def close(self):
        if self.worker is not None and self.worker.is_alive():
            self.closing = True
            self.request_cancel()
        else:
            self.root.destroy()

    def drain(self):
        try:
            while True:
                kind, value = self.events.get_nowait()
                if kind == "progress":
                    self.text.configure(state="normal")
                    self.text.insert("end", value + "\n")
                    self.text.see("end")
                    self.text.configure(state="disabled")
                    if not self.cancel.is_set():
                        self.status.set(value)
                else:
                    self.run_button.configure(state="normal")
                    self.cancel_button.configure(state="disabled")
                    if kind == "done":
                        self.outputs = value
                        self.report_button.configure(state="normal")
                        self.status.set("Complete. Open the report for before/after measurements.")
                    else:
                        self.status.set(value)
        except queue.Empty:
            pass
        if self.closing and (self.worker is None or not self.worker.is_alive()):
            self.root.destroy()
            return
        self.root.after(75, self.drain)

    def open_help(self):
        webbrowser.open_new_tab(
            Path(__file__).with_name("help").joinpath("video_stabilizer.html").as_uri()
        )

    def open_report(self):
        if "report" in self.outputs:
            webbrowser.open_new_tab(self.outputs["report"].as_uri())


def run_video_stabilizer_gui(parent=None):
    app = StabilizerGUI(parent)
    if app.owns_root:
        app.root.mainloop()
    return app


def main(argv=None):
    argv = sys.argv[1:] if argv is None else argv
    if not argv:
        run_video_stabilizer_gui()
        return 0
    parser = build_parser()
    args = parser.parse_args(argv)
    if not args.video or not args.markers:
        parser.error("--video and --markers are required for CLI processing")
    try:
        _run_args(args)
        return 0
    except (Exception, KeyboardInterrupt) as exc:
        log(str(exc) or "Cancelled")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
