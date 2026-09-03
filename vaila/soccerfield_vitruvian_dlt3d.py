"""
================================================================================
Script: soccerfield_vitruvian_dlt3d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago and contributors
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: vailá team
Version: 0.3.120
Created: 11 August 2026
Update Date: 03 September 2026
================================================================================
Description:
    Estimate a time-varying DLT3D camera from two complementary sources:

    1. football-field landmarks with known world coordinates on Z=0;
    2. non-coplanar vertical controls, normally the bottom/top centres of
       tracked player boxes and a weak player-height prior (the Vitruvian
       fallback), optionally augmented by measured goalpost/scene verticals.

    The field points first determine the eight coefficients visible on the
    ground plane.  Each player bottom centre is mapped through that plane to
    (X, Y, 0), while the corresponding top centre is assigned (X, Y, h).  The
    remaining three coefficients that multiply Z are then estimated by
    weighted least squares.  At least two spatially distinct vertical controls
    are required and the vertical design matrix must have rank three.

    This is camera calibration, not single-view triangulation of arbitrary
    joints.  The bbox vertical is an uncertain anthropometric camera control;
    it does not make every observed 2D point invertible without an additional
    surface, body, or depth constraint.

Outputs:
    <stem>.dlt3d                    frame + 11 DLT coefficients
    <stem>_report.csv               per-frame conditioning and reprojection
    <stem>_controls.csv             vertical controls used per frame
    <stem>_height_assignments.csv   track-height provenance

License: AGPL-3.0-or-later
================================================================================
"""

from __future__ import annotations

import argparse
import re
from contextlib import suppress
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from .dlt2d import dlt2d
    from .rec2d import rec2d
except ImportError:
    from dlt2d import dlt2d  # ty: ignore[unresolved-import]
    from rec2d import rec2d  # ty: ignore[unresolved-import]


DEFAULT_FIELD_REFERENCE = (
    Path(__file__).resolve().parent / "models" / "soccerfield_ref3d_fifa_dataset.csv"
)


@dataclass(frozen=True)
class HeightAssignment:
    """Metric height assigned to one tracked bbox slot."""

    track: str
    height_m: float
    method: str
    player_name: str = ""


@dataclass(frozen=True)
class CalibrationDiagnostics:
    """Auditable diagnostics for one frame calibration."""

    frame: int
    status: str
    n_field: int
    n_bbox_verticals: int
    n_known_verticals: int
    vertical_rank: int
    vertical_condition: float
    field_reprojection_rms_px: float
    bbox_bottom_reprojection_rms_px: float
    vertical_reprojection_rms_px: float
    error: str = ""


@dataclass(frozen=True)
class WidePointTable:
    """Frame-indexed wide point CSV in image pixels."""

    frames: np.ndarray
    point_names: tuple[str, ...]
    values: np.ndarray

    def points_at(self, frame: int) -> dict[str, np.ndarray]:
        matches = np.flatnonzero(self.frames == int(frame))
        if matches.size == 0:
            return {}
        row = self.values[int(matches[0])]
        return {
            name: row[idx]
            for idx, name in enumerate(self.point_names)
            if np.isfinite(row[idx]).all()
        }


def _normalise_track_name(value: object, *, zero_based: bool = False) -> str:
    text = str(value).strip().lower()
    match = re.fullmatch(r"p?(\d+)(?:\.0)?", text)
    if not match:
        return text
    number = int(match.group(1)) + (1 if zero_based else 0)
    return f"p{number}"


def _paired_columns(df: pd.DataFrame) -> tuple[str, list[tuple[str, str, str]]]:
    """Return frame column and (point-name, x-column, y-column) triples."""
    columns = [str(c) for c in df.columns]
    frame_col = next((c for c in columns if c.strip().lower() == "frame"), columns[0])
    by_lower = {c.lower(): c for c in columns}
    pairs: list[tuple[str, str, str]] = []
    seen: set[str] = set()

    for x_col in columns:
        lower = x_col.lower()
        point_name: str | None = None
        y_key: str | None = None
        match = re.fullmatch(r"(.+)_x", lower)
        if match:
            prefix = match.group(1)
            point_name = _normalise_track_name(prefix)
            y_key = f"{prefix}_y"
        else:
            match = re.fullmatch(r"x(\d+)", lower)
            if match:
                point_name = f"p{int(match.group(1))}"
                y_key = f"y{match.group(1)}"
        if point_name is None or y_key not in by_lower or point_name in seen:
            continue
        pairs.append((point_name, x_col, by_lower[y_key]))
        seen.add(point_name)

    if not pairs:
        raise ValueError(
            "No paired point columns found. Expected p1_x/p1_y or x1/y1 style columns."
        )
    return frame_col, pairs


def load_wide_points(csv_path: Path | str) -> WidePointTable:
    """Load vailá/getpixelvideo/SAM wide pixel coordinates."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    if df.empty:
        raise ValueError(f"point CSV is empty: {path}")
    frame_col, pairs = _paired_columns(df)
    frames = pd.to_numeric(df[frame_col], errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(frames).all():
        raise ValueError(f"frame column contains non-numeric values: {path}")
    values = np.full((len(df), len(pairs), 2), np.nan, dtype=float)
    for idx, (_, x_col, y_col) in enumerate(pairs):
        values[:, idx, 0] = pd.to_numeric(df[x_col], errors="coerce")
        values[:, idx, 1] = pd.to_numeric(df[y_col], errors="coerce")
    return WidePointTable(
        frames=frames.astype(int),
        point_names=tuple(name for name, _, _ in pairs),
        values=values,
    )


def load_field_reference(csv_path: Path | str) -> dict[str, np.ndarray]:
    """Load named and pN aliases for a planar field reference."""
    path = Path(csv_path)
    df = pd.read_csv(path)
    if not {"x", "y"}.issubset(df.columns):
        raise ValueError(f"field reference must contain x,y columns: {path}")
    result: dict[str, np.ndarray] = {}
    for row_index, row in df.iterrows():
        xy = np.array([float(row["x"]), float(row["y"])], dtype=float)
        result[f"p{int(row_index) + 1}"] = xy
        if "point_number" in df.columns:
            with suppress(Exception):
                result[f"p{int(row['point_number'])}"] = xy
        if "point_name" in df.columns:
            result[str(row["point_name"]).strip().lower()] = xy
    return result


def project_dlt2d(params: np.ndarray, world_xy: np.ndarray) -> np.ndarray:
    """Forward-project world XY points with vailá's eight-coefficient DLT2D."""
    a = np.asarray(params, dtype=float).reshape(8)
    xy = np.asarray(world_xy, dtype=float).reshape(-1, 2)
    den = xy[:, 0] * a[6] + xy[:, 1] * a[7] + 1.0
    return np.column_stack(
        (
            (xy[:, 0] * a[0] + xy[:, 1] * a[1] + a[2]) / den,
            (xy[:, 0] * a[3] + xy[:, 1] * a[4] + a[5]) / den,
        )
    )


def project_dlt3d(params: np.ndarray, world_xyz: np.ndarray) -> np.ndarray:
    """Forward-project world XYZ points with vailá's 11-coefficient DLT3D."""
    a = np.asarray(params, dtype=float).reshape(11)
    xyz = np.asarray(world_xyz, dtype=float).reshape(-1, 3)
    den = xyz[:, 0] * a[8] + xyz[:, 1] * a[9] + xyz[:, 2] * a[10] + 1.0
    return np.column_stack(
        (
            (xyz[:, 0] * a[0] + xyz[:, 1] * a[1] + xyz[:, 2] * a[2] + a[3]) / den,
            (xyz[:, 0] * a[4] + xyz[:, 1] * a[5] + xyz[:, 2] * a[6] + a[7]) / den,
        )
    )


def embed_ground_plane_dlt(dlt2d_params: np.ndarray) -> np.ndarray:
    """Embed the eight Z=0 coefficients into vailá's DLT3D ordering."""
    a = np.asarray(dlt2d_params, dtype=float).reshape(8)
    return np.array(
        [a[0], a[1], 0.0, a[2], a[3], a[4], 0.0, a[5], a[6], a[7], 0.0],
        dtype=float,
    )


def _vertical_design(
    dlt2d_params: np.ndarray,
    base_world_xy: np.ndarray,
    top_pixels: np.ndarray,
    heights_m: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    a = np.asarray(dlt2d_params, dtype=float).reshape(8)
    bases = np.asarray(base_world_xy, dtype=float).reshape(-1, 2)
    tops = np.asarray(top_pixels, dtype=float).reshape(-1, 2)
    heights = np.asarray(heights_m, dtype=float).reshape(-1)
    matrix = np.zeros((2 * len(bases), 3), dtype=float)
    rhs = np.zeros(2 * len(bases), dtype=float)

    for idx, ((x, y), (u, v), height) in enumerate(zip(bases, tops, heights, strict=True)):
        ground_den = a[6] * x + a[7] * y + 1.0
        ground_u_num = a[0] * x + a[1] * y + a[2]
        ground_v_num = a[3] * x + a[4] * y + a[5]
        matrix[2 * idx] = [height, 0.0, -u * height]
        matrix[2 * idx + 1] = [0.0, height, -v * height]
        rhs[2 * idx] = u * ground_den - ground_u_num
        rhs[2 * idx + 1] = v * ground_den - ground_v_num
    return matrix, rhs


def solve_vertical_dlt_column(
    dlt2d_params: np.ndarray,
    base_world_xy: np.ndarray,
    top_pixels: np.ndarray,
    heights_m: np.ndarray,
    *,
    weights: np.ndarray | None = None,
    min_verticals: int = 2,
    robust_iterations: int = 3,
) -> tuple[np.ndarray, int, float]:
    """Estimate L3, L7, L11 from metric vertical controls.

    One vertical contributes only two equations.  The function therefore
    rejects fewer than two controls and any rank-deficient arrangement.
    """
    bases = np.asarray(base_world_xy, dtype=float).reshape(-1, 2)
    tops = np.asarray(top_pixels, dtype=float).reshape(-1, 2)
    heights = np.asarray(heights_m, dtype=float).reshape(-1)
    if not (len(bases) == len(tops) == len(heights)):
        raise ValueError("base, top, and height arrays must have the same length")
    valid = (
        np.isfinite(bases).all(axis=1)
        & np.isfinite(tops).all(axis=1)
        & np.isfinite(heights)
        & (heights > 0)
    )
    bases, tops, heights = bases[valid], tops[valid], heights[valid]
    if len(bases) < min_verticals:
        raise ValueError(f"need at least {min_verticals} valid vertical controls, got {len(bases)}")
    if weights is None:
        base_weights = np.ones(len(bases), dtype=float)
    else:
        raw_weights = np.asarray(weights, dtype=float).reshape(-1)
        if len(raw_weights) != len(valid):
            raise ValueError("weights must match the unfiltered vertical-control count")
        base_weights = raw_weights[valid]
    if np.any(~np.isfinite(base_weights)) or np.any(base_weights <= 0):
        raise ValueError("vertical-control weights must be finite and positive")

    matrix, rhs = _vertical_design(dlt2d_params, bases, tops, heights)
    row_weights = np.repeat(base_weights, 2)
    weighted = matrix * np.sqrt(row_weights)[:, None]
    rank = int(np.linalg.matrix_rank(weighted))
    if rank < 3:
        raise ValueError(
            "vertical controls are rank deficient; use at least two spatially distinct verticals"
        )

    effective = base_weights.copy()
    theta = np.zeros(3, dtype=float)
    for _ in range(max(1, int(robust_iterations))):
        row_weights = np.repeat(effective, 2)
        sqrt_w = np.sqrt(row_weights)
        theta = np.linalg.lstsq(matrix * sqrt_w[:, None], rhs * sqrt_w, rcond=None)[0]
        dlt3d_params = embed_ground_plane_dlt(dlt2d_params)
        dlt3d_params[[2, 6, 10]] = theta
        xyz = np.column_stack((bases, heights))
        residual = np.linalg.norm(project_dlt3d(dlt3d_params, xyz) - tops, axis=1)
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        scale = max(1.4826 * mad, 1.0e-6)
        cutoff = 1.345 * scale
        robust = np.minimum(1.0, cutoff / np.maximum(residual, 1.0e-12))
        effective = base_weights * robust

    condition = float(np.linalg.cond(weighted))
    dlt3d_params = embed_ground_plane_dlt(dlt2d_params)
    dlt3d_params[[2, 6, 10]] = theta
    return dlt3d_params, rank, condition


def fit_vitruvian_frame(
    *,
    frame: int,
    field_world_xy: np.ndarray,
    field_pixels: np.ndarray,
    bbox_bottom_pixels: np.ndarray,
    bbox_top_pixels: np.ndarray,
    bbox_heights_m: np.ndarray,
    bbox_names: list[str] | None = None,
    known_base_world_xy: np.ndarray | None = None,
    known_top_pixels: np.ndarray | None = None,
    known_heights_m: np.ndarray | None = None,
    known_names: list[str] | None = None,
    bbox_weight: float = 0.25,
    known_weight: float = 1.0,
    min_field_points: int = 6,
    min_verticals: int = 2,
) -> tuple[np.ndarray, CalibrationDiagnostics, list[dict[str, object]]]:
    """Fit one ground-plane + Vitruvian DLT3D calibration."""
    field_world = np.asarray(field_world_xy, dtype=float).reshape(-1, 2)
    field_px = np.asarray(field_pixels, dtype=float).reshape(-1, 2)
    field_valid = np.isfinite(field_world).all(axis=1) & np.isfinite(field_px).all(axis=1)
    field_world, field_px = field_world[field_valid], field_px[field_valid]
    if len(field_world) < min_field_points:
        raise ValueError(
            f"need at least {min_field_points} valid field points, got {len(field_world)}"
        )
    planar = dlt2d(field_world, field_px)

    bottoms = np.asarray(bbox_bottom_pixels, dtype=float).reshape(-1, 2)
    tops = np.asarray(bbox_top_pixels, dtype=float).reshape(-1, 2)
    heights = np.asarray(bbox_heights_m, dtype=float).reshape(-1)
    if not (len(bottoms) == len(tops) == len(heights)):
        raise ValueError("bbox bottom, top, and height arrays must have the same length")
    bbox_valid = (
        np.isfinite(bottoms).all(axis=1)
        & np.isfinite(tops).all(axis=1)
        & np.isfinite(heights)
        & (heights > 0)
    )
    bottoms, tops, heights = bottoms[bbox_valid], tops[bbox_valid], heights[bbox_valid]
    bbox_labels = bbox_names or [f"bbox_{idx + 1}" for idx in range(len(bbox_valid))]
    bbox_labels = [name for name, keep in zip(bbox_labels, bbox_valid, strict=True) if keep]
    base_xy = rec2d(planar, bottoms) if len(bottoms) else np.empty((0, 2), dtype=float)

    if known_base_world_xy is None:
        known_bases = np.empty((0, 2), dtype=float)
        known_tops = np.empty((0, 2), dtype=float)
        known_heights = np.empty(0, dtype=float)
    else:
        known_bases = np.asarray(known_base_world_xy, dtype=float).reshape(-1, 2)
        known_tops = np.asarray(known_top_pixels, dtype=float).reshape(-1, 2)
        known_heights = np.asarray(known_heights_m, dtype=float).reshape(-1)
    known_labels = known_names or [f"known_{idx + 1}" for idx in range(len(known_bases))]

    all_bases = np.vstack((known_bases, base_xy))
    all_tops = np.vstack((known_tops, tops))
    all_heights = np.concatenate((known_heights, heights))
    weights = np.concatenate(
        (
            np.full(len(known_bases), known_weight, dtype=float),
            np.full(len(base_xy), bbox_weight, dtype=float),
        )
    )
    params, rank, condition = solve_vertical_dlt_column(
        planar,
        all_bases,
        all_tops,
        all_heights,
        weights=weights,
        min_verticals=min_verticals,
    )

    field_xyz = np.column_stack((field_world, np.zeros(len(field_world))))
    field_err = np.linalg.norm(project_dlt3d(params, field_xyz) - field_px, axis=1)
    bottom_xyz = np.column_stack((base_xy, np.zeros(len(base_xy))))
    bottom_err = (
        np.linalg.norm(project_dlt3d(params, bottom_xyz) - bottoms, axis=1)
        if len(bottoms)
        else np.empty(0)
    )
    top_xyz = np.column_stack((all_bases, all_heights))
    vertical_err = np.linalg.norm(project_dlt3d(params, top_xyz) - all_tops, axis=1)

    diagnostics = CalibrationDiagnostics(
        frame=int(frame),
        status="ok",
        n_field=int(len(field_world)),
        n_bbox_verticals=int(len(base_xy)),
        n_known_verticals=int(len(known_bases)),
        vertical_rank=rank,
        vertical_condition=condition,
        field_reprojection_rms_px=float(np.sqrt(np.mean(field_err**2))),
        bbox_bottom_reprojection_rms_px=(
            float(np.sqrt(np.mean(bottom_err**2))) if len(bottom_err) else float("nan")
        ),
        vertical_reprojection_rms_px=float(np.sqrt(np.mean(vertical_err**2))),
    )

    controls: list[dict[str, object]] = []
    labels = [*known_labels, *bbox_labels]
    sources = [*["known_vertical"] * len(known_bases), *["bbox_vitruvian"] * len(base_xy)]
    for name, source, base, top, height, weight, error in zip(
        labels,
        sources,
        all_bases,
        all_tops,
        all_heights,
        weights,
        vertical_err,
        strict=True,
    ):
        controls.append(
            {
                "frame": int(frame),
                "name": name,
                "source": source,
                "world_x": float(base[0]),
                "world_y": float(base[1]),
                "height_m": float(height),
                "top_u_px": float(top[0]),
                "top_v_px": float(top[1]),
                "weight": float(weight),
                "reprojection_error_px": float(error),
            }
        )
    return params, diagnostics, controls


def _normalise_column(text: object) -> str:
    value = str(text).strip().lower()
    value = re.sub(r"[^a-z0-9]+", "_", value)
    return value.strip("_")


def assign_track_heights(
    bottom: WidePointTable,
    top: WidePointTable,
    *,
    heights_csv: Path | str | None,
    default_height_m: float | None,
) -> dict[str, HeightAssignment]:
    """Assign heights by explicit track ID, or transparently by rank fallback."""
    tracks = [name for name in bottom.point_names if name in top.point_names]
    median_bbox_height: dict[str, float] = {}
    for track in tracks:
        bottom_idx = bottom.point_names.index(track)
        top_idx = top.point_names.index(track)
        common = np.intersect1d(bottom.frames, top.frames)
        values: list[float] = []
        for frame in common:
            bi = int(np.flatnonzero(bottom.frames == frame)[0])
            ti = int(np.flatnonzero(top.frames == frame)[0])
            height_px = bottom.values[bi, bottom_idx, 1] - top.values[ti, top_idx, 1]
            if np.isfinite(height_px) and height_px > 0:
                values.append(float(height_px))
        median_bbox_height[track] = float(np.median(values)) if values else float("nan")

    assignments: dict[str, HeightAssignment] = {}
    roster: list[tuple[float, str]] = []
    if heights_csv is not None:
        table = pd.read_csv(heights_csv)
        table = table.rename(columns={col: _normalise_column(col) for col in table.columns})
        height_values: pd.Series
        if "height_m" in table.columns:
            height_values = pd.to_numeric(table["height_m"], errors="coerce")
        elif "altura_m" in table.columns:
            height_values = pd.to_numeric(table["altura_m"], errors="coerce")
        elif "height_cm" in table.columns:
            height_values = pd.to_numeric(table["height_cm"], errors="coerce") / 100.0
        else:
            raise ValueError("height CSV must contain height_m, altura_m, or height_cm")

        track_col = next(
            (
                col
                for col in ("track", "track_id", "person_slot", "slot", "slot_index")
                if col in table.columns
            ),
            None,
        )
        name_col = next((col for col in ("player_name", "player", "nome") if col in table), None)
        if track_col is not None:
            for row_idx, row in table.iterrows():
                height = float(height_values.iloc[row_idx])
                if not np.isfinite(height) or height <= 0:
                    continue
                track = _normalise_track_name(
                    row[track_col], zero_based=(track_col == "slot_index")
                )
                if track not in tracks:
                    continue
                player = str(row[name_col]) if name_col is not None else ""
                assignments[track] = HeightAssignment(track, height, "explicit_track", player)
        else:
            for row_idx, row in table.iterrows():
                height = float(height_values.iloc[row_idx])
                if not np.isfinite(height) or height <= 0:
                    continue
                player = str(row[name_col]) if name_col is not None else ""
                roster.append((height, player))

    if roster:
        ranked_tracks = sorted(
            (track for track in tracks if track not in assignments),
            key=lambda track: median_bbox_height[track],
            reverse=True,
        )
        ranked_roster = sorted(roster, key=lambda item: item[0], reverse=True)
        for track, (height, player) in zip(ranked_tracks, ranked_roster, strict=False):
            assignments[track] = HeightAssignment(track, height, "bbox_height_rank", player)

    if default_height_m is not None:
        if not np.isfinite(default_height_m) or default_height_m <= 0:
            raise ValueError("default height must be finite and positive")
        for track in tracks:
            assignments.setdefault(
                track,
                HeightAssignment(track, float(default_height_m), "default_height", ""),
            )
    return assignments


def load_known_verticals(csv_path: Path | str | None) -> dict[int, list[dict[str, object]]]:
    """Load optional measured scene verticals in a simple long CSV layout."""
    if csv_path is None:
        return {}
    df = pd.read_csv(csv_path)
    required = {"frame", "world_x", "world_y", "height_m", "top_u_px", "top_v_px"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"known-vertical CSV is missing columns: {sorted(missing)}")
    result: dict[int, list[dict[str, object]]] = {}
    for row_index, row in df.iterrows():
        frame = int(row["frame"])
        result.setdefault(frame, []).append(
            {
                "name": str(row.get("name", f"known_{row_index + 1}")),
                "base": np.array([row["world_x"], row["world_y"]], dtype=float),
                "top": np.array([row["top_u_px"], row["top_v_px"]], dtype=float),
                "height": float(row["height_m"]),
            }
        )
    return result


def calibrate_time_varying(
    *,
    field_pixels_csv: Path | str,
    field_reference_csv: Path | str,
    bbox_bottom_csv: Path | str,
    bbox_top_csv: Path | str,
    heights_csv: Path | str | None = None,
    known_verticals_csv: Path | str | None = None,
    default_height_m: float | None = 1.80,
    bbox_weight: float = 0.25,
    known_weight: float = 1.0,
    min_field_points: int = 6,
    min_verticals: int = 2,
    selected_frames: set[int] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Calibrate every supported frame without interpolating raw DLT coefficients."""
    field = load_wide_points(field_pixels_csv)
    bottom = load_wide_points(bbox_bottom_csv)
    top = load_wide_points(bbox_top_csv)
    reference = load_field_reference(field_reference_csv)
    assignments = assign_track_heights(
        bottom,
        top,
        heights_csv=heights_csv,
        default_height_m=default_height_m,
    )
    known_by_frame = load_known_verticals(known_verticals_csv)

    common_frames = (
        set(field.frames.tolist()) & set(bottom.frames.tolist()) & set(top.frames.tolist())
    )
    if selected_frames is not None:
        common_frames &= {int(frame) for frame in selected_frames}

    dlt_rows: list[dict[str, object]] = []
    report_rows: list[dict[str, object]] = []
    control_rows: list[dict[str, object]] = []
    common_tracks = [
        track for track in bottom.point_names if track in top.point_names and track in assignments
    ]

    for frame in sorted(common_frames):
        field_points = field.points_at(frame)
        field_names = [
            name for name in field.point_names if name in field_points and name in reference
        ]
        field_world = np.asarray([reference[name] for name in field_names], dtype=float)
        field_px = np.asarray([field_points[name] for name in field_names], dtype=float)

        bottom_points = bottom.points_at(frame)
        top_points = top.points_at(frame)
        bbox_names = [
            track for track in common_tracks if track in bottom_points and track in top_points
        ]
        bbox_bottom = np.asarray([bottom_points[name] for name in bbox_names], dtype=float)
        bbox_top = np.asarray([top_points[name] for name in bbox_names], dtype=float)
        bbox_heights = np.asarray([assignments[name].height_m for name in bbox_names], dtype=float)

        known = known_by_frame.get(frame, [])
        known_bases = np.asarray([item["base"] for item in known], dtype=float).reshape(-1, 2)
        known_tops = np.asarray([item["top"] for item in known], dtype=float).reshape(-1, 2)
        known_heights = np.asarray([item["height"] for item in known], dtype=float)
        known_names = [str(item["name"]) for item in known]

        try:
            params, diagnostics, controls = fit_vitruvian_frame(
                frame=frame,
                field_world_xy=field_world,
                field_pixels=field_px,
                bbox_bottom_pixels=bbox_bottom,
                bbox_top_pixels=bbox_top,
                bbox_heights_m=bbox_heights,
                bbox_names=bbox_names,
                known_base_world_xy=known_bases,
                known_top_pixels=known_tops,
                known_heights_m=known_heights,
                known_names=known_names,
                bbox_weight=bbox_weight,
                known_weight=known_weight,
                min_field_points=min_field_points,
                min_verticals=min_verticals,
            )
        except (ValueError, np.linalg.LinAlgError) as exc:
            report_rows.append(
                asdict(
                    CalibrationDiagnostics(
                        frame=frame,
                        status="skipped",
                        n_field=len(field_world),
                        n_bbox_verticals=len(bbox_names),
                        n_known_verticals=len(known),
                        vertical_rank=0,
                        vertical_condition=float("nan"),
                        field_reprojection_rms_px=float("nan"),
                        bbox_bottom_reprojection_rms_px=float("nan"),
                        vertical_reprojection_rms_px=float("nan"),
                        error=str(exc),
                    )
                )
            )
            continue

        dlt_rows.append({"frame": frame, **{f"p{i + 1}": value for i, value in enumerate(params)}})
        report_rows.append(asdict(diagnostics))
        control_rows.extend(controls)

    height_rows = [asdict(assignments[track]) for track in sorted(assignments)]
    dlt_df = pd.DataFrame(dlt_rows, columns=["frame", *[f"p{i}" for i in range(1, 12)]])
    report_df = pd.DataFrame(report_rows)
    controls_df = pd.DataFrame(control_rows)
    heights_df = pd.DataFrame(height_rows)
    return dlt_df, report_df, controls_df, heights_df


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Time-varying football-camera DLT3D from planar field keypoints plus "
            "Vitruvian player-bbox verticals."
        )
    )
    parser.add_argument("--field-pixels", type=Path, required=True)
    parser.add_argument("--field-ref", type=Path, default=DEFAULT_FIELD_REFERENCE)
    parser.add_argument("--bbox-bottom", type=Path, required=True)
    parser.add_argument("--bbox-top", type=Path, required=True)
    parser.add_argument("--heights", type=Path, default=None, help="Height roster or track map CSV")
    parser.add_argument(
        "--known-verticals",
        type=Path,
        default=None,
        help="Optional goalpost/scene vertical controls in long CSV form",
    )
    parser.add_argument("--default-height-m", type=float, default=1.80)
    parser.add_argument("--bbox-weight", type=float, default=0.25)
    parser.add_argument("--known-weight", type=float, default=1.0)
    parser.add_argument("--min-field-points", type=int, default=6)
    parser.add_argument("--min-verticals", type=int, default=2)
    parser.add_argument(
        "--frame",
        type=int,
        action="append",
        default=None,
        help="Restrict calibration to this frame; repeat for multiple anchors",
    )
    parser.add_argument("--output", type=Path, required=True, help="Output directory")
    parser.add_argument("--stem", default="vitruvian_timevarying", help="Output filename stem")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_argparser().parse_args(argv)
    dlt_df, report_df, controls_df, heights_df = calibrate_time_varying(
        field_pixels_csv=args.field_pixels,
        field_reference_csv=args.field_ref,
        bbox_bottom_csv=args.bbox_bottom,
        bbox_top_csv=args.bbox_top,
        heights_csv=args.heights,
        known_verticals_csv=args.known_verticals,
        default_height_m=args.default_height_m,
        bbox_weight=args.bbox_weight,
        known_weight=args.known_weight,
        min_field_points=args.min_field_points,
        min_verticals=args.min_verticals,
        selected_frames=set(args.frame) if args.frame else None,
    )
    args.output.mkdir(parents=True, exist_ok=True)
    dlt_path = args.output / f"{args.stem}.dlt3d"
    report_path = args.output / f"{args.stem}_report.csv"
    controls_path = args.output / f"{args.stem}_controls.csv"
    heights_path = args.output / f"{args.stem}_height_assignments.csv"
    dlt_df.to_csv(dlt_path, index=False)
    report_df.to_csv(report_path, index=False)
    controls_df.to_csv(controls_path, index=False)
    heights_df.to_csv(heights_path, index=False)
    print(f">> DLT3D rows: {len(dlt_df)} -> {dlt_path}")
    print(f">> diagnostics: {report_path}")
    print(f">> controls: {controls_path}")
    print(f">> height assignments: {heights_path}")
    if dlt_df.empty:
        print(">> No frame passed the field/vertical rank checks; inspect the report.")
        return 2
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
