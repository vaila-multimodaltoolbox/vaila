"""
================================================================================
Script: planar_geometry_tracker.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.4.1
Created: 14 September 2026
Last Updated: 15 September 2026
================================================================================
Description:
    Standalone planar-geometry homography tracker, extrapolator, and gap-filler.

    Given a 2D pixel-space marker CSV produced by ``getpixelvideo.py`` and a
    metric target-geometry profile (TOML: EVA tatame mat, soccer pitch, court,
    etc.), this module fits a per-frame planar homography H_t between the
    metric target plane (Z = 0) and image space, using ``cv2.findHomography``
    + RANSAC when >= 4 non-collinear correspondences are available in a frame,
    and an inter-frame affine fallback (``cv2.estimateAffinePartial2D``)
    chained onto the nearest keyframe's homography otherwise. Every target
    landmark is then projected through H_t for every frame -- occluded points
    are imputed, and points beyond the camera field of view are extrapolated
    without clamping (raw negative / out-of-canvas pixel coordinates are kept
    so off-screen geometry stays correct).

    Also projects metric circles/arcs (e.g. a center circle, penalty arcs)
    into image space as projective wireframes, and can render a diagnostic
    "extended canvas" video showing the full projected geometry, including
    the parts that fall outside the original camera frame.

    100% standalone: never mutates ``getpixelvideo.py``'s runtime state. The
    GUI wires a "Geo Homog" button that shells out to this module via
    ``subprocess`` (see ``run_geometric_tracker_action`` there), matching the
    project's GUI-launches-CLI convention.

Usage:
    uv run python -m vaila.planar_geometry_tracker \\
        --config vaila/models/planar_targets/tatame_1x1m.toml \\
        --measurements-csv path/to/markers.csv \\
        --video-path path/to/video.mp4 \\
        --output-dir ./output/tatame_run_01 \\
        --ransac-thresh 3.0 \\
        --extended-canvas \\
        --debug-viz

Outputs (written to --output-dir):
    <stem>_imputed.csv           - wide CSV (frame,p0_x,p0_y,...), NaN-free,
                                    loadable straight back via getpixelvideo.py
    dense_projected_points_long.csv
                                  - long/relational per-point-per-frame table
    homographies.npz              - H, H_inv, frame_ids arrays
    target_calibration.ref3d      - target metric points, repo .ref3d CSV
                                     convention (see vaila/drawsportsfields.py)
    debug_projected_wireframe.mp4 - only with --debug-viz
================================================================================
"""

from __future__ import annotations

import argparse
import sys
import tomllib
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from numpy.typing import NDArray

# ------------------------------------------------------------------------- #
# Geometry configuration model
# ------------------------------------------------------------------------- #


@dataclass(frozen=True)
class TargetPoint:
    """One metric control point of the target geometry (Z = 0 plane)."""

    point_id: int
    name: str
    x: float
    y: float
    z: float = 0.0


@dataclass(frozen=True)
class TargetCircle:
    name: str
    center_point: int
    radius: float
    samples: int = 64


@dataclass(frozen=True)
class TargetArc:
    name: str
    center_point: int
    radius: float
    angle_start_deg: float
    angle_end_deg: float
    samples: int = 32


@dataclass(frozen=True)
class TargetGeometry:
    """Parsed ``[target]``/``[points]``/``[topology]``/circles/arcs TOML profile."""

    name: str
    description: str
    target_type: str
    points: dict[int, TargetPoint]
    perimeter: list[int] = field(default_factory=list)
    lines: list[tuple[int, int]] = field(default_factory=list)
    circles: list[TargetCircle] = field(default_factory=list)
    arcs: list[TargetArc] = field(default_factory=list)

    def world_xy(self, point_id: int) -> NDArray[np.float64]:
        p = self.points[point_id]
        return np.array([p.x, p.y], dtype=np.float64)

    @property
    def sorted_ids(self) -> list[int]:
        return sorted(self.points)


def load_target_geometry(config_path: str | Path) -> TargetGeometry:
    """Parse a target-geometry TOML profile (see ``vaila/models/planar_targets/``)."""
    config_path = Path(config_path)
    with config_path.open("rb") as fh:
        data = tomllib.load(fh)

    target = data.get("target", {})
    raw_points = data.get("points", {})
    points: dict[int, TargetPoint] = {}
    for key, val in raw_points.items():
        pid = int(key)
        points[pid] = TargetPoint(
            point_id=pid,
            name=str(val.get("name", f"p{pid}")),
            x=float(val["x"]),
            y=float(val["y"]),
            z=float(val.get("z", 0.0)),
        )

    topology = data.get("topology", {})
    perimeter = [int(i) for i in topology.get("perimeter", [])]
    lines = [(int(a), int(b)) for a, b in topology.get("lines", [])]

    circles = [
        TargetCircle(
            name=str(c.get("name", "circle")),
            center_point=int(c["center_point"]),
            radius=float(c["radius"]),
            samples=int(c.get("samples", 64)),
        )
        for c in data.get("circles", [])
    ]
    arcs = [
        TargetArc(
            name=str(a.get("name", "arc")),
            center_point=int(a["center_point"]),
            radius=float(a["radius"]),
            angle_start_deg=float(a["angle_start_deg"]),
            angle_end_deg=float(a["angle_end_deg"]),
            samples=int(a.get("samples", 32)),
        )
        for a in data.get("arcs", [])
    ]

    if not points:
        raise ValueError(f"Target geometry profile has no [points]: {config_path}")

    return TargetGeometry(
        name=str(target.get("name", config_path.stem)),
        description=str(target.get("description", "")),
        target_type=str(target.get("type", "polygon")),
        points=points,
        perimeter=perimeter,
        lines=lines,
        circles=circles,
        arcs=arcs,
    )


# ------------------------------------------------------------------------- #
# Measurements CSV ingestion (wide primary; long also supported)
# ------------------------------------------------------------------------- #


def load_measurements_csv(
    csv_path: str | Path,
) -> dict[int, dict[int, tuple[float, float]]]:
    """Load a marker CSV into ``{frame: {point_id: (u, v)}}``.

    Supports the wide format actually written by ``getpixelvideo.py``
    (``frame, p0_x, p0_y, p1_x, p1_y, ...``) as the primary path, and the
    long/relational format (``frame, point_id, u, v``) as a secondary path.
    Missing/NaN coordinates are simply omitted from the per-frame dict.
    """
    csv_path = Path(csv_path)
    df = pd.read_csv(csv_path)
    columns = set(df.columns)

    out: dict[int, dict[int, tuple[float, float]]] = {}

    if {"point_id", "u", "v"}.issubset(columns):
        # Long format.
        for row_d in df.to_dict("records"):
            frame = int(row_d["frame"])
            pid = int(row_d["point_id"])
            u = float(row_d["u"])
            v = float(row_d["v"])
            if np.isnan(u) or np.isnan(v):
                continue
            out.setdefault(frame, {})[pid] = (u, v)
        return out

    # Wide format: frame, p{i}_x, p{i}_y, ...
    if "frame" not in columns:
        raise ValueError(
            f"Measurements CSV {csv_path} has neither long (point_id,u,v) nor "
            "wide (frame,pN_x,pN_y,...) columns."
        )
    point_ids: set[int] = set()
    for col in df.columns:
        if col.startswith("p") and col.endswith("_x"):
            try:
                point_ids.add(int(col[1:-2]))
            except ValueError:
                continue

    for row_d in df.to_dict("records"):
        frame = int(row_d["frame"])
        per_frame: dict[int, tuple[float, float]] = {}
        for pid in point_ids:
            xk, yk = f"p{pid}_x", f"p{pid}_y"
            if xk not in row_d or yk not in row_d:
                continue
            u, v = row_d[xk], row_d[yk]
            if u is None or v is None:
                continue
            try:
                uf, vf = float(u), float(v)
            except (TypeError, ValueError):
                continue
            if np.isnan(uf) or np.isnan(vf):
                continue
            per_frame[pid] = (uf, vf)
        out[frame] = per_frame
    return out


# ------------------------------------------------------------------------- #
# Step 1: Non-collinearity checks
# ------------------------------------------------------------------------- #


def is_collinear_triplet(
    p1: NDArray[np.float64], p2: NDArray[np.float64], p3: NDArray[np.float64], tol: float = 1e-4
) -> bool:
    """Computes twice the triangle area to detect collinear points."""
    return (
        0.5 * abs(p1[0] * (p2[1] - p3[1]) + p2[0] * (p3[1] - p1[1]) + p3[0] * (p1[1] - p2[1])) < tol
    )


def has_non_collinear_quad(
    valid_indices: list[int], world_points: dict[int, NDArray[np.float64]]
) -> bool:
    """Returns True if there is at least one subset of 4 points containing no collinear triplets."""
    if len(valid_indices) < 4:
        return False
    for c in combinations(valid_indices, 4):
        p0, p1, p2, p3 = (world_points[i] for i in c)
        if not (
            is_collinear_triplet(p0, p1, p2)
            or is_collinear_triplet(p0, p1, p3)
            or is_collinear_triplet(p0, p2, p3)
            or is_collinear_triplet(p1, p2, p3)
        ):
            return True
    return False


# ------------------------------------------------------------------------- #
# Steps 2-3: Per-frame homography solve, affine fallback, PCHIP regularization
# ------------------------------------------------------------------------- #


@dataclass
class FrameHomography:
    frame: int
    homography: NDArray[np.float64]
    method: str  # "ransac" | "affine" | "translation" | "propagated" | "spline"
    n_correspondences: int


def _normalize_h(homography: NDArray[np.float64]) -> NDArray[np.float64]:
    if abs(homography[2, 2]) > 1e-12:
        return homography / homography[2, 2]
    return homography


def solve_frame_homographies(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
    ransac_thresh: float,
) -> dict[int, FrameHomography]:
    """Solve H_t per frame: RANSAC homography when possible, else inter-frame
    affine chained onto the nearest keyframe, else propagate the previous H.
    """
    world_points = {pid: geometry.world_xy(pid) for pid in geometry.sorted_ids}
    frames_sorted = sorted(measurements)

    solved: dict[int, FrameHomography] = {}
    keyframes: list[int] = []  # frames with a directly-RANSAC-estimated H

    for frame in frames_sorted:
        obs = measurements[frame]
        valid_ids = [pid for pid in obs if pid in world_points]
        if len(valid_ids) >= 4 and has_non_collinear_quad(valid_ids, world_points):
            world_pts = np.array([world_points[i] for i in valid_ids], dtype=np.float64)
            pixel_pts = np.array([obs[i] for i in valid_ids], dtype=np.float64)
            homography, _inliers = cv2_find_homography(world_pts, pixel_pts, ransac_thresh)
            if homography is not None:
                solved[frame] = FrameHomography(
                    frame=frame,
                    homography=_normalize_h(homography),
                    method="ransac",
                    n_correspondences=len(valid_ids),
                )
                keyframes.append(frame)

    # Second pass: affine fallback / propagation for every remaining frame,
    # walking in ascending frame order so "nearest keyframe" prefers the
    # closest already-solved frame (keyframe or fallback-solved) behind it.
    for frame in frames_sorted:
        if frame in solved:
            continue
        obs = measurements[frame]
        ref_frame = _nearest_solved_frame(frame, solved)
        if ref_frame is None:
            continue  # filled in the backward-propagation pass below
        ref = solved[ref_frame]
        common_ids = [pid for pid in obs if pid in world_points]
        # points visible in both this frame and the reference frame's own
        # measured set (not just any target point) -- true "mutually visible"
        ref_obs = measurements.get(ref_frame, {})
        common_ids = [pid for pid in common_ids if pid in ref_obs]

        if len(common_ids) >= 3:
            ref_pts = np.array([ref_obs[i] for i in common_ids], dtype=np.float64)
            curr_pts = np.array([obs[i] for i in common_ids], dtype=np.float64)
            transform, _inl = cv2_estimate_affine_partial(ref_pts, curr_pts)
            if transform is not None:
                t_t = np.eye(3, dtype=np.float64)
                t_t[:2, :] = transform
                solved[frame] = FrameHomography(
                    frame=frame,
                    homography=_normalize_h(t_t @ ref.homography),
                    method="affine",
                    n_correspondences=len(common_ids),
                )
                continue
        if len(common_ids) in (1, 2):
            ref_pts = np.array([ref_obs[i] for i in common_ids], dtype=np.float64)
            curr_pts = np.array([obs[i] for i in common_ids], dtype=np.float64)
            delta = np.mean(curr_pts - ref_pts, axis=0)
            t_t = np.eye(3, dtype=np.float64)
            t_t[0, 2] = delta[0]
            t_t[1, 2] = delta[1]
            solved[frame] = FrameHomography(
                frame=frame,
                homography=_normalize_h(t_t @ ref.homography),
                method="translation",
                n_correspondences=len(common_ids),
            )
            continue
        # 0 common points: propagate the reference homography unchanged.
        solved[frame] = FrameHomography(
            frame=frame,
            homography=ref.homography.copy(),
            method="propagated",
            n_correspondences=0,
        )

    if not solved:
        raise ValueError(
            "No frame yielded >= 4 non-collinear correspondences; cannot solve any homography."
        )

    # Propagate the earliest solved homography backwards to any leading
    # frames that had zero correspondences at all (per spec §8: "smoothly
    # propagate the first estimated homography backwards").
    first_solved_frame = min(solved)
    for frame in frames_sorted:
        if frame >= first_solved_frame:
            break
        solved[frame] = FrameHomography(
            frame=frame,
            homography=solved[first_solved_frame].homography.copy(),
            method="propagated",
            n_correspondences=0,
        )

    return _pchip_regularize(solved, frames_sorted)


def _nearest_solved_frame(frame: int, solved: dict[int, FrameHomography]) -> int | None:
    if not solved:
        return None
    candidates = [f for f in solved if f < frame]
    if candidates:
        return max(candidates)
    candidates = [f for f in solved if f > frame]
    if candidates:
        return min(candidates)
    return None


def _pchip_regularize(
    solved: dict[int, FrameHomography], frames_sorted: list[int]
) -> dict[int, FrameHomography]:
    """Global spline (PCHIP) regularization of the H_t sequence.

    RANSAC-solved keyframes are treated as trusted anchors; every homography
    element (H is normalized so H[2,2] = 1, leaving 8 free DOF) is smoothed
    across the full frame range with a monotone piecewise-cubic Hermite
    interpolant anchored at those trusted frames. This removes small
    frame-to-frame jitter from the affine-fallback/propagated frames without
    overshooting (PCHIP has no Gibbs-style ringing), while leaving frames
    that already have a directly RANSAC-estimated homography untouched.
    """
    from scipy.interpolate import PchipInterpolator

    trusted_frames = sorted(f for f, fh in solved.items() if fh.method == "ransac")
    if len(trusted_frames) < 2:
        return solved  # not enough anchors to regularize against

    trusted_x = np.array(trusted_frames, dtype=np.float64)

    # Stack the 8 free homography DOF (row-major, skipping H[2,2] == 1) per
    # trusted frame, interpolate each channel independently.
    dof_idx = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1)]
    trusted_vals = np.array(
        [[solved[f].homography[r, c] for (r, c) in dof_idx] for f in trusted_frames],
        dtype=np.float64,
    )
    interpolators = [
        PchipInterpolator(trusted_x, trusted_vals[:, k], extrapolate=False)
        for k in range(len(dof_idx))
    ]
    lo, hi = trusted_frames[0], trusted_frames[-1]

    out: dict[int, FrameHomography] = dict(solved)
    for frame in frames_sorted:
        fh = solved[frame]
        if fh.method == "ransac" or frame < lo or frame > hi:
            continue
        vals = [float(interp(frame)) for interp in interpolators]
        h_smoothed = np.eye(3, dtype=np.float64)
        for (r, c), v in zip(dof_idx, vals, strict=True):
            h_smoothed[r, c] = v
        out[frame] = FrameHomography(
            frame=frame,
            homography=h_smoothed,
            method="spline",
            n_correspondences=fh.n_correspondences,
        )
    return out


def cv2_find_homography(
    world_pts: NDArray[np.float64], pixel_pts: NDArray[np.float64], ransac_thresh: float
) -> tuple[NDArray[np.float64] | None, NDArray[Any] | None]:
    import cv2

    homography, inliers = cv2.findHomography(world_pts, pixel_pts, cv2.RANSAC, ransac_thresh)
    if homography is None:
        return None, None
    return homography.astype(np.float64), inliers


def cv2_estimate_affine_partial(
    src_pts: NDArray[np.float64], dst_pts: NDArray[np.float64]
) -> tuple[NDArray[np.float64] | None, NDArray[Any] | None]:
    import cv2

    transform, inliers = cv2.estimateAffinePartial2D(src_pts, dst_pts)
    if transform is None:
        return None, None
    return transform.astype(np.float64), inliers


# ------------------------------------------------------------------------- #
# Step 3 (projection) + Step 4 (circles/arcs)
# ------------------------------------------------------------------------- #


def project_points(
    homography: NDArray[np.float64], world_xy: NDArray[np.float64]
) -> NDArray[np.float64]:
    """Project Nx2 world (X, Y) points through H (unclamped, full-projective)."""
    n = world_xy.shape[0]
    homogeneous = np.hstack([world_xy, np.ones((n, 1), dtype=np.float64)])
    projected = (homography @ homogeneous.T).T
    w = projected[:, 2]
    w_safe = np.where(np.abs(w) < 1e-12, 1e-12, w)
    return projected[:, :2] / w_safe[:, None]


def sample_circle_world(circle: TargetCircle, geometry: TargetGeometry) -> NDArray[np.float64]:
    center = geometry.world_xy(circle.center_point)
    theta = np.linspace(0.0, 2.0 * np.pi, circle.samples, endpoint=True)
    x = center[0] + circle.radius * np.cos(theta)
    y = center[1] + circle.radius * np.sin(theta)
    return np.column_stack([x, y])


def sample_arc_world(arc: TargetArc, geometry: TargetGeometry) -> NDArray[np.float64]:
    center = geometry.world_xy(arc.center_point)
    theta = np.radians(np.linspace(arc.angle_start_deg, arc.angle_end_deg, arc.samples))
    x = center[0] + arc.radius * np.cos(theta)
    y = center[1] + arc.radius * np.sin(theta)
    return np.column_stack([x, y])


# ------------------------------------------------------------------------- #
# Dense per-frame diagnostic table + imputed wide CSV
# ------------------------------------------------------------------------- #


@dataclass
class DenseRow:
    frame: int
    point_id: int
    point_name: str
    u: float
    v: float
    is_measured: bool
    in_fov: bool
    reproj_error_px: float
    frame_rmse_px: float


def build_dense_table(
    geometry: TargetGeometry,
    measurements: dict[int, dict[int, tuple[float, float]]],
    solved: dict[int, FrameHomography],
    frame_size: tuple[int, int] | None,
) -> list[DenseRow]:
    rows: list[DenseRow] = []
    point_ids = geometry.sorted_ids
    world_xy = np.array([geometry.world_xy(pid) for pid in point_ids], dtype=np.float64)
    w_bound, h_bound = frame_size if frame_size is not None else (None, None)

    for frame in sorted(solved):
        fh = solved[frame]
        projected = project_points(fh.homography, world_xy)
        obs = measurements.get(frame, {})

        frame_errors: list[float] = []
        per_point: list[DenseRow] = []
        for idx, pid in enumerate(point_ids):
            u_proj, v_proj = projected[idx]
            measured = obs.get(pid)
            if measured is not None:
                u_meas, v_meas = measured
                err = float(np.hypot(u_meas - u_proj, v_meas - v_proj))
                frame_errors.append(err)
                u_out, v_out = u_meas, v_meas
                is_measured = True
                reproj_error = err
            else:
                u_out, v_out = float(u_proj), float(v_proj)
                is_measured = False
                reproj_error = float("nan")

            if w_bound is None or h_bound is None:
                in_fov = True
            else:
                in_fov = bool(0 <= u_out <= w_bound and 0 <= v_out <= h_bound)

            per_point.append(
                DenseRow(
                    frame=frame,
                    point_id=pid,
                    point_name=geometry.points[pid].name,
                    u=u_out,
                    v=v_out,
                    is_measured=is_measured,
                    in_fov=in_fov,
                    reproj_error_px=reproj_error,
                    frame_rmse_px=float("nan"),  # filled below
                )
            )

        frame_rmse = (
            float(np.sqrt(np.mean(np.square(frame_errors)))) if frame_errors else float("nan")
        )
        for row in per_point:
            row.frame_rmse_px = frame_rmse
        rows.extend(per_point)

    return rows


def dense_rows_to_wide_csv(rows: list[DenseRow], point_ids: list[int]) -> pd.DataFrame:
    """Reshape the long diagnostic rows into getpixelvideo.py's wide convention."""
    by_frame: dict[int, dict[int, tuple[float, float]]] = {}
    for row in rows:
        by_frame.setdefault(row.frame, {})[row.point_id] = (row.u, row.v)

    records = []
    for frame in sorted(by_frame):
        rec: dict[str, float | int] = {"frame": frame}
        for pid in point_ids:
            u, v = by_frame[frame].get(pid, (float("nan"), float("nan")))
            rec[f"p{pid}_x"] = u
            rec[f"p{pid}_y"] = v
        records.append(rec)
    return pd.DataFrame(records)


# ------------------------------------------------------------------------- #
# Outputs
# ------------------------------------------------------------------------- #


def write_outputs(
    output_dir: Path,
    stem: str,
    geometry: TargetGeometry,
    rows: list[DenseRow],
    solved: dict[int, FrameHomography],
) -> dict[str, Path]:
    # Geometry parsing/projection is also used by headless video stabilization.
    # Load the GUI-facing sports-field export layer only when exporting REF3D.
    if __package__:
        from .drawsportsfields import (
            FieldControlPoint,
            build_ref3d_dataframe,
            build_ref3d_map_dataframe,
        )
    else:
        from drawsportsfields import (
            FieldControlPoint,
            build_ref3d_dataframe,
            build_ref3d_map_dataframe,
        )

    output_dir.mkdir(parents=True, exist_ok=True)
    written: dict[str, Path] = {}

    point_ids = geometry.sorted_ids
    imputed_df = dense_rows_to_wide_csv(rows, point_ids)
    imputed_path = output_dir / f"{stem}_imputed.csv"
    imputed_df.to_csv(imputed_path, index=False)
    written["imputed_csv"] = imputed_path

    long_df = pd.DataFrame(
        [
            {
                "frame": r.frame,
                "point_id": r.point_id,
                "point_name": r.point_name,
                "u": r.u,
                "v": r.v,
                "is_measured": r.is_measured,
                "in_fov": r.in_fov,
                "reproj_error_px": r.reproj_error_px,
                "frame_rmse_px": r.frame_rmse_px,
            }
            for r in rows
        ]
    )
    long_path = output_dir / "dense_projected_points_long.csv"
    long_df.to_csv(long_path, index=False)
    written["long_csv"] = long_path

    frames_sorted = sorted(solved)
    h_stack = np.stack([solved[f].homography for f in frames_sorted])
    h_inv_stack = np.stack([np.linalg.inv(solved[f].homography) for f in frames_sorted])
    npz_path = output_dir / "homographies.npz"
    np.savez_compressed(
        npz_path,
        H=h_stack,
        H_inv=h_inv_stack,
        frame_ids=np.array(frames_sorted, dtype=np.int64),
    )
    written["homographies_npz"] = npz_path

    # target_calibration.ref3d: repo's real CSV .ref3d convention (see
    # vaila/drawsportsfields.py build_ref3d_dataframe/build_ref3d_map_dataframe),
    # not the spec's incorrect "Kinovea XML" claim -- kept consistent with
    # every other .ref3d producer/consumer in the repo.
    control_points = [
        FieldControlPoint(
            key=f"target:{pid}:{geometry.points[pid].name}",
            label=f"{pid:02d}  {geometry.points[pid].name}",
            x=geometry.points[pid].x,
            y=geometry.points[pid].y,
            z=geometry.points[pid].z,
            source_index=pid,
            source_name=geometry.points[pid].name,
        )
        for pid in point_ids
    ]
    ref3d_path = output_dir / "target_calibration.ref3d"
    build_ref3d_dataframe(control_points, index_base=1).to_csv(ref3d_path, index=False)
    map_path = output_dir / "target_calibration.ref3d_map.csv"
    build_ref3d_map_dataframe(control_points, index_base=1).to_csv(map_path, index=False)
    written["ref3d"] = ref3d_path
    written["ref3d_map"] = map_path

    return written


# ------------------------------------------------------------------------- #
# Extended-canvas / debug-viz renderer
# ------------------------------------------------------------------------- #

_COLOR_MEASURED = (0, 200, 0)  # solid green circle
_COLOR_IMPUTED_IN_FOV = (255, 255, 0)  # cyan triangle (BGR: cyan == (255,255,0))
_COLOR_EXTRAPOLATED = (255, 0, 255)  # magenta square
_COLOR_WIREFRAME = (0, 255, 255)  # yellow


def render_debug_video(
    video_path: Path,
    output_path: Path,
    geometry: TargetGeometry,
    solved: dict[int, FrameHomography],
    rows: list[DenseRow],
    *,
    extended_canvas: bool,
    canvas_scale: float,
) -> None:
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frame_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    if extended_canvas:
        canvas_w = int(round(canvas_scale * frame_w))
        canvas_h = int(round(canvas_scale * frame_h))
        x_offset = (canvas_scale - 1.0) * frame_w / 2.0
        y_offset = (canvas_scale - 1.0) * frame_h / 2.0
        t_canvas = np.array(
            [[1.0, 0.0, x_offset], [0.0, 1.0, y_offset], [0.0, 0.0, 1.0]], dtype=np.float64
        )
    else:
        canvas_w, canvas_h = frame_w, frame_h
        t_canvas = np.eye(3, dtype=np.float64)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # ty: ignore[unresolved-attribute]
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (canvas_w, canvas_h))

    rows_by_frame: dict[int, list[DenseRow]] = {}
    for r in rows:
        rows_by_frame.setdefault(r.frame, []).append(r)

    try:
        frame_idx = 0
        while True:
            ok, frame_img = cap.read()
            if not ok:
                break

            canvas = np.zeros((canvas_h, canvas_w, 3), dtype=np.uint8)
            if extended_canvas:
                x0, y0 = int(round(x_offset)), int(round(y_offset))
                canvas[y0 : y0 + frame_h, x0 : x0 + frame_w] = frame_img
            else:
                canvas[:, :] = frame_img

            if frame_idx in solved:
                h_canvas = t_canvas @ solved[frame_idx].homography
                _draw_wireframe(canvas, geometry, h_canvas)
                for r in rows_by_frame.get(frame_idx, []):
                    pt_canvas = project_points(t_canvas, np.array([[r.u, r.v]]))[0]
                    center = (int(round(pt_canvas[0])), int(round(pt_canvas[1])))
                    if r.is_measured:
                        cv2.circle(canvas, center, 5, _COLOR_MEASURED, -1)
                    elif r.in_fov:
                        _draw_triangle(canvas, center, _COLOR_IMPUTED_IN_FOV)
                    else:
                        _draw_square(canvas, center, _COLOR_EXTRAPOLATED)

            writer.write(canvas)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def _draw_wireframe(
    canvas: NDArray[np.uint8], geometry: TargetGeometry, homography: NDArray[np.float64]
) -> None:
    import cv2

    for a, b in geometry.lines:
        pts = project_points(homography, np.array([geometry.world_xy(a), geometry.world_xy(b)]))
        p1 = tuple(int(round(v)) for v in pts[0])
        p2 = tuple(int(round(v)) for v in pts[1])
        cv2.line(canvas, p1, p2, _COLOR_WIREFRAME, 1, cv2.LINE_AA)

    for circle in geometry.circles:
        world = sample_circle_world(circle, geometry)
        pts = project_points(homography, world)
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], True, _COLOR_WIREFRAME, 1, cv2.LINE_AA)

    for arc in geometry.arcs:
        world = sample_arc_world(arc, geometry)
        pts = project_points(homography, world)
        poly = pts.astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(canvas, [poly], False, _COLOR_WIREFRAME, 1, cv2.LINE_AA)


def _draw_triangle(
    canvas: NDArray[np.uint8], center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    import cv2

    cx, cy = center
    pts = np.array([[cx, cy - 6], [cx - 6, cy + 5], [cx + 6, cy + 5]], dtype=np.int32)
    cv2.fillPoly(canvas, [pts], color)


def _draw_square(
    canvas: NDArray[np.uint8], center: tuple[int, int], color: tuple[int, int, int]
) -> None:
    import cv2

    cx, cy = center
    cv2.rectangle(canvas, (cx - 5, cy - 5), (cx + 5, cy + 5), color, -1)


# ------------------------------------------------------------------------- #
# Video frame size probe (optional)
# ------------------------------------------------------------------------- #


def probe_video_frame_size(video_path: Path | None) -> tuple[int, int] | None:
    if video_path is None or not Path(video_path).is_file():
        return None
    import cv2

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
    try:
        w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    if w <= 0 or h <= 0:
        return None
    return w, h


# ------------------------------------------------------------------------- #
# Pipeline entry point
# ------------------------------------------------------------------------- #


def run_planar_geometry_tracker(
    config_path: Path,
    measurements_csv: Path,
    output_dir: Path,
    *,
    video_path: Path | None = None,
    ransac_thresh: float = 3.0,
    canvas_scale: float = 1.8,
    extended_canvas: bool = False,
    debug_viz: bool = False,
) -> dict[str, Path]:
    geometry = load_target_geometry(config_path)
    measurements = load_measurements_csv(measurements_csv)
    solved = solve_frame_homographies(geometry, measurements, ransac_thresh)
    frame_size = probe_video_frame_size(video_path)
    rows = build_dense_table(geometry, measurements, solved, frame_size)

    stem = Path(measurements_csv).stem
    written = write_outputs(output_dir, stem, geometry, rows, solved)

    if debug_viz and video_path is not None:
        debug_path = output_dir / "debug_projected_wireframe.mp4"
        render_debug_video(
            Path(video_path),
            debug_path,
            geometry,
            solved,
            rows,
            extended_canvas=extended_canvas,
            canvas_scale=canvas_scale,
        )
        written["debug_video"] = debug_path
    elif debug_viz and video_path is None:
        print(">> --debug-viz requested but no --video-path given; skipping video render.")

    return written


# ------------------------------------------------------------------------- #
# CLI
# ------------------------------------------------------------------------- #


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vaila.planar_geometry_tracker",
        description=(
            "Planar-geometry homography tracker/extrapolator/gap-filler: fits "
            "per-frame homographies against a metric target profile (TOML), "
            "imputes occluded markers and extrapolates points beyond the "
            "camera FOV without clamping."
        ),
    )
    parser.add_argument("--config", required=True, type=Path, help="Target-geometry TOML profile.")
    parser.add_argument(
        "--measurements-csv",
        required=True,
        type=Path,
        help="Marker CSV from getpixelvideo.py (wide p{i}_x/p{i}_y or long point_id/u/v).",
    )
    parser.add_argument("--video-path", type=Path, default=None, help="Reference video (optional).")
    parser.add_argument(
        "--output-dir", type=Path, default=Path("./vaila_tracker_output"), help="Output directory."
    )
    parser.add_argument(
        "--ransac-thresh",
        type=float,
        default=3.0,
        help="RANSAC inlier reprojection threshold in pixels (default 3.0).",
    )
    parser.add_argument(
        "--canvas-scale",
        type=float,
        default=1.8,
        help="Extended-canvas scale factor (default 1.8).",
    )
    parser.add_argument(
        "--extended-canvas",
        action="store_true",
        help="Render the debug video on an expanded virtual canvas.",
    )
    parser.add_argument(
        "--debug-viz",
        action="store_true",
        help="Write debug_projected_wireframe.mp4 (requires --video-path).",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)

    try:
        written = run_planar_geometry_tracker(
            args.config,
            args.measurements_csv,
            args.output_dir,
            video_path=args.video_path,
            ransac_thresh=args.ransac_thresh,
            canvas_scale=args.canvas_scale,
            extended_canvas=args.extended_canvas,
            debug_viz=args.debug_viz,
        )
    except Exception as exc:  # noqa: BLE001 - CLI top-level error reporting
        print(f">> planar_geometry_tracker error: {exc}", file=sys.stderr)
        return 1

    print(">> planar_geometry_tracker finished. Outputs:")
    for key, path in written.items():
        print(f"   {key}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
