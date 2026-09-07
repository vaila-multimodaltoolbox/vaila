"""
================================================================================
Script: quickmeasure.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.127
Created: 06 September 2026
Last Updated: 07 September 2026
================================================================================
Description:
    Kinovea-style quick on-image measurements for `getpixelvideo.py`:
    calibrate first, then pick a live measure mode with digit keys ``1``–``0``
    and click on the video. Each completed set is drawn on the image with its
    value (distance / area / angle / velocity / …) and stored for CSV export.

    After ``Q`` turns Quick Measure on (and calibration is done):
      ``1`` distance   — 2 clicks → line + length label (repeat for more pairs)
      ``2`` area       — ≥3 clicks, ``Enter`` closes the polygon → area label
      ``3`` angle      — 3 clicks (vertex in the middle) → angle label
      ``4`` velocity   — 2 clicks on different frames (needs FPS)
      ``5`` acceleration — 3 clicks on distinct frames (needs FPS)
      ``6``–``0``      — reserved

    Calibration-first flow (mirrors Kinovea's "calibrate measure"):
      - ``line``  — 2 clicks on a segment of known length + the typed length.
      - ``plane`` — 4 clicks around a rectangle of known width/height.
      - ``ref3d`` — load a ``.ref3d`` (modes 1–3 / dlt3d formats 1–4), drop one
                    world axis for planar ``rec2d``, then pixel CSV or guide clicks.

    The first save in a session creates ``processed_quickmeasure_<timestamp>/``;
    calibration autosaves and later ``S`` saves update that same directory.
    It contains CSV data plus a didactic ``quickmeasure_report.html`` explaining
    every measurement, metric, result, and exported column. Results use a
    matrix-friendly layout: one result per row and scalar ``point_N_*`` columns,
    never packed coordinate/list strings. Velocity/acceleration require video FPS.

Units:
    - Uncalibrated session: pixel units ("px", "px/s", "px/s^2").
    - Calibrated session (DLT2D loaded): whatever unit the REF2D file used
      (assumed metres, "m"/"m/s"/"m/s^2", matching dlt2d.py/rec2d.py's own
      convention) — the caller may override `unit_label`.

License:
    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.

    This program is distributed in the hope that it will be useful, but
    WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
    General Public License for more details.

    You should have received a copy of the GNU GPLv3 (General Public
    License Version 3) along with this program. If not, see
    <https://www.gnu.org/licenses/>.
"""

from __future__ import annotations

import html
import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from .dlt2d import dlt2d as dlt2d_solve
    from .dlt2d import process_files as dlt2d_process_files
    from .dlt3d import detect_ref3d_format, normalize_ref3d_to_format1
    from .rec2d_one_dlt2d import rec2d
except ImportError:
    from dlt2d import dlt2d as dlt2d_solve  # ty: ignore[unresolved-import]
    from dlt2d import process_files as dlt2d_process_files  # ty: ignore[unresolved-import]
    from dlt3d import (  # ty: ignore[unresolved-import]
        detect_ref3d_format,
        normalize_ref3d_to_format1,
    )
    from rec2d_one_dlt2d import rec2d  # ty: ignore[unresolved-import]


MEASURE_TYPES: tuple[str, ...] = ("distance", "area", "angle", "velocity", "acceleration")

# Digit keys after ``Q`` select the live measure mode (see LIVE_MODE_KEYS).
LIVE_MODE_KEYS: dict[str, str] = {
    "1": "distance",
    "2": "area",
    "3": "angle",
    "4": "velocity",
    "5": "acceleration",
}
# Auto-finalize after this many draft clicks. ``None`` means press Enter (area).
LIVE_MODE_AUTO_POINTS: dict[str, int | None] = {
    "distance": 2,
    "area": None,
    "angle": 3,
    "velocity": 2,
    "acceleration": 3,
}
LIVE_MODE_NEEDS_FPS: frozenset[str] = frozenset({"velocity", "acceleration"})

# Click-based calibration modes and how many image clicks each one needs.
CALIBRATION_MODES: tuple[str, ...] = ("line", "plane")
CALIBRATION_CLICKS: dict[str, int] = {"line": 2, "plane": 4}
# Real-world measurements the user must type after clicking, per mode.
CALIBRATION_MEASURES: dict[str, tuple[str, ...]] = {
    "line": ("length",),
    "plane": ("width", "height"),
}

# Plane used for single-camera DLT2D from a 3D REF3D: drop one world axis.
PLANE_DROP_AXES: tuple[str, ...] = ("z", "y", "x")
PLANE_KEEP_AXES: dict[str, tuple[str, str]] = {
    "z": ("x", "y"),  # XY plane (floor / top-down)
    "y": ("x", "z"),  # XZ plane
    "x": ("y", "z"),  # YZ plane
}


class QuickMeasureError(ValueError):
    """Raised for invalid/insufficient quick-measurement input.

    A plain ValueError subclass so callers that already catch ValueError
    keep working, while UI code can special-case this type if it wants to.
    """


def _point_numbers_from_columns(columns) -> list[int]:
    numbers: set[int] = set()
    for col in columns:
        if isinstance(col, str) and col.startswith("p") and "_" in col:
            head = col.split("_", 1)[0][1:]
            if head.isdigit():
                numbers.add(int(head))
    return sorted(numbers)


def load_ref3d_format1(ref3d_file: str, *, min_points: int = 4) -> tuple[pd.DataFrame, int]:
    """Load a ``.ref3d`` (modes 1–3 / dlt3d formats 1–4) as format-1 + detect code."""
    if not os.path.isfile(ref3d_file):
        raise QuickMeasureError(f"REF3D file not found: {ref3d_file}")
    fmt = detect_ref3d_format(ref3d_file)
    if fmt is None:
        raise QuickMeasureError(
            f"Unrecognized REF3D layout (need mode1 wide, mode2 point/x/y/z, "
            f"or mode3 bare x,y,z): {ref3d_file}"
        )
    df = normalize_ref3d_to_format1(ref3d_file, min_points=min_points)
    if df is None or df.empty:
        raise QuickMeasureError(
            f"REF3D file could not be normalized (need >= {min_points} points): {ref3d_file}"
        )
    return df, int(fmt)


def ref3d_points_xyz(ref_df: pd.DataFrame) -> list[tuple[int, float, float, float]]:
    """Ordered ``(p_index, x, y, z)`` tuples from a format-1 REF3D DataFrame."""
    row = ref_df.iloc[0]
    points: list[tuple[int, float, float, float]] = []
    for idx in _point_numbers_from_columns(ref_df.columns):
        points.append(
            (
                idx,
                float(row[f"p{idx}_x"]),
                float(row[f"p{idx}_y"]),
                float(row[f"p{idx}_z"]),
            )
        )
    return points


def drop_axis_to_plane(
    points_xyz: Sequence[Sequence[float]],
    drop_axis: str = "z",
    *,
    dedupe: bool = True,
) -> tuple[list[tuple[float, float]], list[int], int]:
    """Project 3D REF points to 2D by dropping one axis.

    Returns ``(uv_points, kept_source_indices, n_duplicates_skipped)``.
    When ``dedupe`` is True, later points that collapse onto an earlier (u,v)
    are skipped — required for COD-style cages where posts share two axes.
    """
    axis = str(drop_axis).strip().lower()
    if axis not in PLANE_KEEP_AXES:
        raise QuickMeasureError(f"drop_axis must be one of {PLANE_DROP_AXES}, got {drop_axis!r}")
    keep = PLANE_KEEP_AXES[axis]
    axis_to_i = {"x": 0, "y": 1, "z": 2}
    i0, i1 = axis_to_i[keep[0]], axis_to_i[keep[1]]
    uv: list[tuple[float, float]] = []
    kept: list[int] = []
    seen: set[tuple[float, float]] = set()
    skipped = 0
    for src_i, pt in enumerate(points_xyz):
        if len(pt) < 3:
            raise QuickMeasureError("Each REF3D point needs x,y,z.")
        u, v = float(pt[i0]), float(pt[i1])
        key = (round(u, 9), round(v, 9))
        if dedupe and key in seen:
            skipped += 1
            continue
        seen.add(key)
        uv.append((u, v))
        kept.append(src_i)
    if len(uv) < 4:
        raise QuickMeasureError(
            f"After dropping {axis.upper()} only {len(uv)} unique planar points remain "
            f"(need >= 4). Try another drop axis or a planar subset of the REF3D."
        )
    return uv, kept, skipped


def ref3d_to_ref2d_dataframe(
    ref_df: pd.DataFrame, drop_axis: str = "z", *, dedupe: bool = True
) -> tuple[pd.DataFrame, list[int], int]:
    """Build a one-row REF2D-like DataFrame from format-1 REF3D by dropping an axis."""
    xyz = ref3d_points_xyz(ref_df)
    uv, kept_idx, skipped = drop_axis_to_plane(
        [(p[1], p[2], p[3]) for p in xyz], drop_axis, dedupe=dedupe
    )
    data: dict[str, list[float | int]] = {"frame": [0]}
    for out_i, src_i in enumerate(kept_idx, start=1):
        p_index = xyz[src_i][0]
        # Keep original pN labels so pixel CSVs from getpixelvideo still match.
        data[f"p{p_index}_x"] = [uv[out_i - 1][0]]
        data[f"p{p_index}_y"] = [uv[out_i - 1][1]]
    return pd.DataFrame(data), [xyz[i][0] for i in kept_idx], skipped


def read_pixel_calibration_points(
    pixel_file: str,
) -> list[tuple[int, float, float]]:
    """Read first-frame ``(p_index, x_px, y_px)`` from a getpixelvideo markers CSV."""
    if not os.path.isfile(pixel_file):
        raise QuickMeasureError(f"Calibration pixel file not found: {pixel_file}")
    df = pd.read_csv(pixel_file)
    if df.empty:
        raise QuickMeasureError(f"Calibration pixel file is empty: {pixel_file}")
    row = df.iloc[0]
    points: list[tuple[int, float, float]] = []
    for idx in _point_numbers_from_columns(df.columns):
        x = row.get(f"p{idx}_x")
        y = row.get(f"p{idx}_y")
        if pd.isna(x) or pd.isna(y):
            continue
        points.append((idx, float(x), float(y)))
    if len(points) < 4:
        raise QuickMeasureError(
            f"Pixel calibration CSV needs >= 4 valid points, found {len(points)}."
        )
    return points


def suggest_pixel_csv_for_ref3d(ref3d_file: str, video_path: str | None = None) -> str | None:
    """Guess a sibling getpixelvideo markers CSV near the REF3D / video."""
    candidates: list[str] = []
    ref_dir = os.path.dirname(os.path.abspath(ref3d_file))
    if video_path:
        stem = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(os.path.abspath(video_path))
        for name in (
            f"{stem}_markers_1_line.csv",
            f"{stem}_markers.csv",
            f"{stem}.csv",
        ):
            candidates.append(os.path.join(video_dir, name))
            candidates.append(os.path.join(ref_dir, name))
    for name in sorted(os.listdir(ref_dir)):
        lower = name.lower()
        if lower.endswith(".csv") and ("marker" in lower or "pixel" in lower or "calib" in lower):
            candidates.append(os.path.join(ref_dir, name))
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


@dataclass
class QuickMeasureCalibration:
    """A single fixed-camera calibration used to convert pixel clicks to
    real-world units.

    Three ways to build it, all producing the same interface:

    - ``from_line_clicks``  — Kinovea-style scale calibration (``kind="line"``).
    - ``from_plane_clicks`` — Kinovea-style perspective calibration
      (``kind="plane"``), solved as an 8-parameter DLT2D homography.
    - ``from_dlt2d_file`` / ``from_calibration_points`` — reuse an existing
      vailá DLT2D calibration (``kind="dlt2d"``).

    DLT math is reused from `dlt2d.py` (`dlt2d()`, `process_files()`) and
    applied with `rec2d_one_dlt2d.py`'s `rec2d()`; nothing is reimplemented
    here.
    """

    dlt_params: np.ndarray | None = None
    unit_label: str = "m"
    source: str = ""
    kind: str = "dlt2d"
    # ``line`` mode only: isotropic scale (unit per pixel) and image-space origin.
    scale: float | None = None
    origin_px: tuple[float, float] | None = None
    # Provenance of a click-built calibration (empty for file-based ones).
    calibration_pixels: list[tuple[float, float]] = field(default_factory=list)
    calibration_real: list[tuple[float, float]] = field(default_factory=list)
    real_measures: dict[str, float] = field(default_factory=dict)
    # REF3D → planar DLT2D provenance.
    drop_axis: str | None = None
    ref3d_format: int | None = None
    ref3d_path: str = ""
    pixel_csv_path: str = ""
    kept_point_indices: list[int] = field(default_factory=list)

    def pixel_to_real(self, x: float, y: float) -> tuple[float, float]:
        if self.kind == "line":
            if self.scale is None or self.origin_px is None:
                raise QuickMeasureError("Line calibration is missing its scale/origin.")
            x0, y0 = self.origin_px
            # +x right, +y up: image rows grow downwards, so y is inverted.
            return (float(x) - x0) * self.scale, (y0 - float(y)) * self.scale
        if self.dlt_params is None:
            raise QuickMeasureError("Calibration is missing its DLT2D parameters.")
        out = rec2d(self.dlt_params, np.array([[x, y]], dtype=float))
        return float(out[0, 0]), float(out[0, 1])

    def describe(self) -> str:
        """One-line human-readable summary for UI status messages."""
        if self.kind == "line" and self.scale is not None:
            length = self.real_measures.get("length", float("nan"))
            return (
                f"Line calibration: {length:g} {self.unit_label} "
                f"({self.scale:.6g} {self.unit_label}/px)"
            )
        if self.kind == "plane":
            width = self.real_measures.get("width", float("nan"))
            height = self.real_measures.get("height", float("nan"))
            return f"Plane calibration: {width:g} x {height:g} {self.unit_label} (DLT2D homography)"
        if self.kind == "ref3d":
            keep = PLANE_KEEP_AXES.get(self.drop_axis or "z", ("x", "y"))
            plane = "".join(a.upper() for a in keep)
            n = len(self.kept_point_indices) or len(self.calibration_real)
            return (
                f"REF3D→{plane} DLT2D ({self.unit_label}, drop "
                f"{(self.drop_axis or '?').upper()}, {n} pts) "
                f"from {os.path.basename(self.ref3d_path or self.source)}"
            )
        return f"DLT2D calibration ({self.unit_label}) from {os.path.basename(self.source)}"

    @classmethod
    def from_line_clicks(
        cls,
        p1: Sequence[float],
        p2: Sequence[float],
        real_length: float,
        unit_label: str = "m",
    ) -> QuickMeasureCalibration:
        """Kinovea-style line calibration: two clicks on a segment whose real
        length the user types.

        Assumes the measured plane is parallel to the sensor (isotropic
        scale). The origin is the FIRST click, ``+x`` points right and
        ``+y`` points up.
        """
        x1, y1 = float(p1[0]), float(p1[1])
        x2, y2 = float(p2[0]), float(p2[1])
        pixel_length = float(np.hypot(x2 - x1, y2 - y1))
        if pixel_length <= 0:
            raise QuickMeasureError(
                "Calibration line has zero pixel length — click two distinct points."
            )
        real_length = float(real_length)
        if not np.isfinite(real_length) or real_length <= 0:
            raise QuickMeasureError("Calibration length must be a finite number greater than 0.")
        scale = real_length / pixel_length
        return cls(
            unit_label=unit_label,
            kind="line",
            scale=scale,
            origin_px=(x1, y1),
            source="clicks:line",
            calibration_pixels=[(x1, y1), (x2, y2)],
            calibration_real=[(0.0, 0.0), (real_length, 0.0)],
            real_measures={"length": real_length},
        )

    @classmethod
    def from_plane_clicks(
        cls,
        pixel_points: Sequence[Sequence[float]],
        width: float,
        height: float,
        unit_label: str = "m",
    ) -> QuickMeasureCalibration:
        """Kinovea-style plane calibration: four clicks around a rectangle of
        known ``width`` x ``height``.

        Click order defines the frame: the first click is the origin
        ``(0, 0)``, the second sets ``+x`` at ``(width, 0)``, the third is the
        opposite corner ``(width, height)``, and the fourth is
        ``(0, height)``. Perspective is corrected by the DLT2D homography.
        """
        pts = [(float(p[0]), float(p[1])) for p in pixel_points]
        if len(pts) != 4:
            raise QuickMeasureError(
                f"Plane calibration needs exactly 4 clicked corners, got {len(pts)}."
            )
        width = float(width)
        height = float(height)
        if not np.isfinite(width) or width <= 0 or not np.isfinite(height) or height <= 0:
            raise QuickMeasureError(
                "Calibration width and height must be finite numbers greater than 0."
            )
        real = [(0.0, 0.0), (width, 0.0), (width, height), (0.0, height)]
        calib = cls.from_point_correspondences(pts, real, unit_label=unit_label)
        calib.kind = "plane"
        calib.source = "clicks:plane"
        calib.real_measures = {"width": width, "height": height}
        return calib

    @classmethod
    def from_point_correspondences(
        cls,
        pixel_points: Sequence[Sequence[float]],
        real_points: Sequence[Sequence[float]],
        unit_label: str = "m",
    ) -> QuickMeasureCalibration:
        """Solve DLT2D directly from >= 4 pixel/real point pairs, reusing
        `dlt2d.py`'s `dlt2d()` least-squares solver.
        """
        pixels = np.asarray([[float(p[0]), float(p[1])] for p in pixel_points], dtype=float)
        reals = np.asarray([[float(p[0]), float(p[1])] for p in real_points], dtype=float)
        if pixels.shape[0] != reals.shape[0]:
            raise QuickMeasureError(
                f"Calibration needs matching point counts "
                f"(got {pixels.shape[0]} pixel, {reals.shape[0]} real)."
            )
        if pixels.shape[0] < 4:
            raise QuickMeasureError("DLT2D calibration needs at least 4 point pairs.")
        try:
            params = np.asarray(dlt2d_solve(reals, pixels), dtype=float)
        except np.linalg.LinAlgError as exc:
            raise QuickMeasureError(
                f"Degenerate calibration geometry (collinear points?): {exc}"
            ) from exc
        if params.size != 8 or not np.isfinite(params).all():
            raise QuickMeasureError(
                "Calibration produced invalid DLT2D parameters — check the clicked points."
            )
        return cls(
            dlt_params=params,
            unit_label=unit_label,
            kind="dlt2d",
            source="clicks:points",
            calibration_pixels=[(float(p[0]), float(p[1])) for p in pixels],
            calibration_real=[(float(p[0]), float(p[1])) for p in reals],
        )

    @classmethod
    def from_dlt2d_file(cls, dlt2d_file: str, unit_label: str = "m") -> QuickMeasureCalibration:
        """Load an existing `.dlt2d` coefficients CSV (as saved by
        `dlt2d.py`'s `save_dlt_parameters`): columns `frame,
        dlt_param_1..8`. Uses the FIRST row — a quick-measure session is one
        fixed camera, matching `rec2d_one_dlt2d.py`'s own single-row
        convention.
        """
        if not os.path.isfile(dlt2d_file):
            raise QuickMeasureError(f"DLT2D file not found: {dlt2d_file}")
        df = pd.read_csv(dlt2d_file)
        if df.shape[0] < 1:
            raise QuickMeasureError(f"DLT2D file has no rows: {dlt2d_file}")
        params = df.iloc[0, 1:].to_numpy(dtype=float)
        if params.size != 8:
            raise QuickMeasureError(
                f"Expected 8 DLT2D parameters, found {params.size} in {dlt2d_file}"
            )
        if np.isnan(params).any():
            raise QuickMeasureError(f"DLT2D file contains NaN parameters: {dlt2d_file}")
        return cls(dlt_params=params, unit_label=unit_label, source=dlt2d_file)

    @classmethod
    def from_calibration_points(
        cls, pixel_file: str, ref_file: str, unit_label: str = "m"
    ) -> QuickMeasureCalibration:
        """Compute DLT2D parameters directly from a calibration pixel CSV
        plus a `.ref2d` real-world reference file, by calling `dlt2d.py`'s
        own `process_files()` (label-matched least squares) — no
        reimplementation of the calibration math here.
        """
        if not os.path.isfile(pixel_file):
            raise QuickMeasureError(f"Calibration pixel file not found: {pixel_file}")
        if not os.path.isfile(ref_file):
            raise QuickMeasureError(f"REF2D file not found: {ref_file}")
        dlt_params_by_frame = dlt2d_process_files(pixel_file, ref_file)
        if not dlt_params_by_frame:
            raise QuickMeasureError(
                "Could not compute DLT2D parameters from the given calibration files "
                "(no common labeled points, or frame-count mismatch)."
            )
        _frame, params = dlt_params_by_frame[0]
        params = np.asarray(params, dtype=float)
        if np.isnan(params).any():
            raise QuickMeasureError(
                "Computed DLT2D parameters contain NaN — check calibration points "
                "(need >= 4 valid common points)."
            )
        return cls(dlt_params=params, unit_label=unit_label, source=pixel_file)

    @classmethod
    def from_ref3d_point_pairs(
        cls,
        pixel_points: Sequence[Sequence[float]],
        real_points_2d: Sequence[Sequence[float]],
        *,
        unit_label: str = "m",
        drop_axis: str = "z",
        ref3d_path: str = "",
        ref3d_format: int | None = None,
        pixel_csv_path: str = "",
        kept_point_indices: Sequence[int] | None = None,
    ) -> QuickMeasureCalibration:
        """Solve DLT2D from pixel/real pairs already projected to a plane."""
        calib = cls.from_point_correspondences(pixel_points, real_points_2d, unit_label=unit_label)
        calib.kind = "ref3d"
        calib.drop_axis = str(drop_axis).strip().lower()
        calib.ref3d_path = ref3d_path
        calib.ref3d_format = ref3d_format
        calib.pixel_csv_path = pixel_csv_path
        calib.source = ref3d_path or calib.source
        calib.kept_point_indices = [int(i) for i in (kept_point_indices or [])]
        return calib

    @classmethod
    def from_ref3d_and_pixel_csv(
        cls,
        ref3d_file: str,
        pixel_file: str,
        drop_axis: str = "z",
        unit_label: str = "m",
        *,
        dedupe: bool = True,
    ) -> QuickMeasureCalibration:
        """Load ``.ref3d`` (any mode), drop one axis, pair with pixel CSV → DLT2D."""
        ref_df, fmt = load_ref3d_format1(ref3d_file, min_points=4)
        ref2d_df, kept_labels, skipped = ref3d_to_ref2d_dataframe(
            ref_df, drop_axis=drop_axis, dedupe=dedupe
        )
        pixel_pts = read_pixel_calibration_points(pixel_file)
        pixel_by_label = {idx: (x, y) for idx, x, y in pixel_pts}
        real_by_label: dict[int, tuple[float, float]] = {}
        row = ref2d_df.iloc[0]
        for label in kept_labels:
            if f"p{label}_x" in ref2d_df.columns:
                real_by_label[label] = (float(row[f"p{label}_x"]), float(row[f"p{label}_y"]))
        common = sorted(set(pixel_by_label) & set(real_by_label))
        if len(common) < 4:
            raise QuickMeasureError(
                f"REF3D/pixel CSV share only {len(common)} labeled points after "
                f"dropping {drop_axis.upper()} (need >= 4). "
                f"Pixel labels={sorted(pixel_by_label)}; REF labels={sorted(real_by_label)}."
            )
        pixels = [pixel_by_label[i] for i in common]
        reals = [real_by_label[i] for i in common]
        calib = cls.from_ref3d_point_pairs(
            pixels,
            reals,
            unit_label=unit_label,
            drop_axis=drop_axis,
            ref3d_path=ref3d_file,
            ref3d_format=fmt,
            pixel_csv_path=pixel_file,
            kept_point_indices=common,
        )
        if skipped:
            calib.real_measures["duplicates_skipped"] = float(skipped)
        return calib


@dataclass
class CalibrationDraft:
    """Collects the calibration clicks before the user types the real
    measurement(s). Owned by the UI layer, but all validation lives here so
    the flow is testable without pygame.
    """

    mode: str = "line"
    unit_label: str = "m"
    points: list[tuple[float, float]] = field(default_factory=list)

    def __post_init__(self) -> None:
        self.mode = str(self.mode).strip().lower()
        if self.mode not in CALIBRATION_MODES:
            raise QuickMeasureError(
                f"Unknown calibration mode {self.mode!r} (expected one of {CALIBRATION_MODES})."
            )

    @property
    def required_points(self) -> int:
        return CALIBRATION_CLICKS[self.mode]

    @property
    def required_measures(self) -> tuple[str, ...]:
        return CALIBRATION_MEASURES[self.mode]

    @property
    def is_complete(self) -> bool:
        return len(self.points) >= self.required_points

    @property
    def remaining(self) -> int:
        return max(0, self.required_points - len(self.points))

    def add_point(self, x: float, y: float) -> int:
        if self.is_complete:
            raise QuickMeasureError(f"Calibration already has its {self.required_points} points.")
        self.points.append((float(x), float(y)))
        return len(self.points)

    def undo_last(self) -> bool:
        if self.points:
            self.points.pop()
            return True
        return False

    def instructions(self) -> str:
        """Status line telling the user exactly what to click next."""
        if self.mode == "line":
            steps = ("click the START of a segment of known length", "click its END")
        else:
            steps = (
                "click corner 1 (origin) of a known rectangle",
                "click corner 2 (defines the width direction)",
                "click corner 3 (opposite corner)",
                "click corner 4 (closes the rectangle)",
            )
        if self.is_complete:
            wanted = " and ".join(self.required_measures)
            return f"CALIBRATION ({self.mode}): all points clicked — type the real {wanted}."
        idx = len(self.points)
        return (
            f"CALIBRATION ({self.mode}) {idx + 1}/{self.required_points}: {steps[idx]} "
            f"(right-click undo, Q leaves the mode)"
        )

    def build(self, measures: Sequence[float]) -> QuickMeasureCalibration:
        """Turn the collected clicks plus the typed real measurement(s) into
        a `QuickMeasureCalibration`.
        """
        if not self.is_complete:
            raise QuickMeasureError(
                f"Calibration needs {self.required_points} clicked points, got {len(self.points)}."
            )
        values = [float(v) for v in measures]
        if len(values) != len(self.required_measures):
            wanted = ", ".join(self.required_measures)
            raise QuickMeasureError(
                f"{self.mode.capitalize()} calibration needs {len(self.required_measures)} "
                f"real measurement(s): {wanted}."
            )
        if self.mode == "line":
            return QuickMeasureCalibration.from_line_clicks(
                self.points[0], self.points[1], values[0], unit_label=self.unit_label
            )
        return QuickMeasureCalibration.from_plane_clicks(
            self.points, values[0], values[1], unit_label=self.unit_label
        )


@dataclass
class Ref3dCalibrationDraft:
    """Guided click calibration against a loaded ``.ref3d`` (no pixel CSV yet).

    The scheme overlay highlights the next world point; the user clicks its
    image location until every kept planar point has a pixel correspondence.
    """

    ref3d_path: str
    drop_axis: str = "z"
    unit_label: str = "m"
    ref3d_format: int | None = None
    points_xyz: list[tuple[int, float, float, float]] = field(default_factory=list)
    kept_labels: list[int] = field(default_factory=list)
    real_uv: list[tuple[float, float]] = field(default_factory=list)
    pixel_points: list[tuple[float, float]] = field(default_factory=list)
    duplicates_skipped: int = 0

    @classmethod
    def from_ref3d_file(
        cls,
        ref3d_file: str,
        drop_axis: str = "z",
        unit_label: str = "m",
        *,
        dedupe: bool = True,
    ) -> Ref3dCalibrationDraft:
        ref_df, fmt = load_ref3d_format1(ref3d_file, min_points=4)
        xyz = ref3d_points_xyz(ref_df)
        uv, kept_src, skipped = drop_axis_to_plane(
            [(p[1], p[2], p[3]) for p in xyz], drop_axis, dedupe=dedupe
        )
        kept_labels = [xyz[i][0] for i in kept_src]
        return cls(
            ref3d_path=ref3d_file,
            drop_axis=str(drop_axis).strip().lower(),
            unit_label=unit_label,
            ref3d_format=fmt,
            points_xyz=xyz,
            kept_labels=kept_labels,
            real_uv=uv,
            duplicates_skipped=skipped,
        )

    @property
    def required_points(self) -> int:
        return len(self.real_uv)

    @property
    def is_complete(self) -> bool:
        return len(self.pixel_points) >= self.required_points

    @property
    def remaining(self) -> int:
        return max(0, self.required_points - len(self.pixel_points))

    @property
    def next_label(self) -> int | None:
        idx = len(self.pixel_points)
        if idx >= len(self.kept_labels):
            return None
        return self.kept_labels[idx]

    @property
    def next_xyz(self) -> tuple[float, float, float] | None:
        label = self.next_label
        if label is None:
            return None
        for p_idx, x, y, z in self.points_xyz:
            if p_idx == label:
                return (x, y, z)
        return None

    def add_point(self, x: float, y: float) -> int:
        if self.is_complete:
            raise QuickMeasureError(
                f"REF3D calibration already has its {self.required_points} image clicks."
            )
        self.pixel_points.append((float(x), float(y)))
        return len(self.pixel_points)

    def undo_last(self) -> bool:
        if self.pixel_points:
            self.pixel_points.pop()
            return True
        return False

    def instructions(self) -> str:
        keep = PLANE_KEEP_AXES[self.drop_axis]
        plane = "".join(a.upper() for a in keep)
        if self.is_complete:
            return (
                f"CALIBRATION (ref3d→{plane}): all {self.required_points} points clicked — "
                "press Enter to solve DLT2D."
            )
        label = self.next_label
        xyz = self.next_xyz
        xyz_txt = f"({xyz[0]:g}, {xyz[1]:g}, {xyz[2]:g})" if xyz is not None else "?"
        return (
            f"CALIBRATION (ref3d→{plane}) {len(self.pixel_points) + 1}/"
            f"{self.required_points}: click image location of p{label} "
            f"world {xyz_txt} (right-click undo)"
        )

    def build(self) -> QuickMeasureCalibration:
        if not self.is_complete:
            raise QuickMeasureError(
                f"REF3D guide needs {self.remaining} more click(s) "
                f"({len(self.pixel_points)}/{self.required_points})."
            )
        calib = QuickMeasureCalibration.from_ref3d_point_pairs(
            self.pixel_points,
            self.real_uv,
            unit_label=self.unit_label,
            drop_axis=self.drop_axis,
            ref3d_path=self.ref3d_path,
            ref3d_format=self.ref3d_format,
            kept_point_indices=self.kept_labels,
        )
        if self.duplicates_skipped:
            calib.real_measures["duplicates_skipped"] = float(self.duplicates_skipped)
        return calib


@dataclass
class QuickMeasurePoint:
    frame: int
    x: float
    y: float


@dataclass
class QuickMeasureSession:
    """Click-session state for one video. `getpixelvideo.py` calls
    `set_live_mode()` / `add_live_point()` for digit-key modes, or
    `add_point()` + `measure()` from the submenu; all geometry/kinematics
    math lives here so the host file stays a thin integration layer.
    """

    fps: float | None = None
    calibration: QuickMeasureCalibration | None = None
    points: list[QuickMeasurePoint] = field(default_factory=list)
    results: list[dict] = field(default_factory=list)
    # Set when points are already stored in real-world units (e.g. reloaded
    # from a calibrated points CSV), so no further conversion is applied.
    unit_override: str | None = None
    # True once the user explicitly declined the calibration-first prompt.
    calibration_skipped: bool = False
    # Set when menu ``R`` starts a guided REF3D click calibration; host picks it up.
    pending_ref3d_draft: Ref3dCalibrationDraft | None = None
    # Live digit-key mode (``1``–``5``): in-progress clicks + active type.
    active_mode: str | None = None
    draft_points: list[QuickMeasurePoint] = field(default_factory=list)
    # Created once on the first calibration/result save. Every later save from
    # this video session updates the same export instead of minting another
    # timestamped directory.
    export_dir: str | None = field(default=None, init=False, repr=False)

    @property
    def unit_label(self) -> str:
        if self.unit_override:
            return self.unit_override
        return self.calibration.unit_label if self.calibration else "px"

    @property
    def is_calibrated(self) -> bool:
        return self.calibration is not None or self.unit_override is not None

    def add_point(self, frame: int, x: float, y: float) -> int:
        self.points.append(QuickMeasurePoint(int(frame), float(x), float(y)))
        return len(self.points)

    def undo_last(self) -> bool:
        if self.draft_points:
            self.draft_points.pop()
            return True
        if self.points:
            self.points.pop()
            return True
        return False

    def clear(self) -> None:
        self.points.clear()
        self.draft_points.clear()

    def clear_results(self) -> None:
        self.results.clear()

    def live_mode_status(self) -> str:
        if self.active_mode is None:
            return (
                "QMeas: press 1=distance 2=area 3=angle 4=velocity 5=accel "
                "(6–0 reserved) then click"
            )
        n = len(self.draft_points)
        auto = LIVE_MODE_AUTO_POINTS.get(self.active_mode)
        if self.active_mode == "area":
            return (
                f"MODE area: {n} vertex(es) — click more, Enter closes polygon "
                f"(≥3), right-click undo"
            )
        if auto is None:
            return f"MODE {self.active_mode}: {n} point(s)"
        return f"MODE {self.active_mode}: {n}/{auto} point(s) (right-click undo)"

    def set_live_mode(self, key: str) -> str:
        """Select live measure mode from digit key ``0``–``9``. Clears draft."""
        digit = str(key).strip()
        if digit not in "0123456789":
            raise QuickMeasureError(f"Live mode key must be 0–9, got {key!r}")
        if digit not in LIVE_MODE_KEYS:
            self.active_mode = None
            self.draft_points.clear()
            return f"Key {digit}: reserved (no measure mode yet). Use 1–5."
        mode = LIVE_MODE_KEYS[digit]
        if mode in LIVE_MODE_NEEDS_FPS:
            self._require_fps()
        self.active_mode = mode
        self.draft_points.clear()
        return self.live_mode_status()

    def add_live_point(self, frame: int, x: float, y: float) -> tuple[str, dict | None]:
        """Add a click in the active live mode; auto-finalize when enough points.

        Returns ``(status_message, completed_result_or_None)``.
        """
        if self.active_mode is None:
            raise QuickMeasureError(
                "No live measure mode — press 1–5 first (distance/area/angle/velocity/accel)."
            )
        mode = self.active_mode
        if mode in LIVE_MODE_NEEDS_FPS:
            self._require_fps()
        self.draft_points.append(QuickMeasurePoint(int(frame), float(x), float(y)))
        auto = LIVE_MODE_AUTO_POINTS.get(mode)
        if auto is not None and len(self.draft_points) >= auto:
            result = self._finalize_draft(result_frame=int(frame))
            return format_result(result), result
        return self.live_mode_status(), None

    def finalize_live_area(self, result_frame: int) -> tuple[str, dict | None]:
        """Close the area polygon (Enter while in area mode)."""
        if self.active_mode != "area":
            return "Enter closes the polygon only in area mode (press 2).", None
        if len(self.draft_points) < 3:
            return (
                f"Area needs ≥3 vertices, have {len(self.draft_points)} — keep clicking.",
                None,
            )
        result = self._finalize_draft(result_frame=int(result_frame))
        return format_result(result), result

    def _finalize_draft(self, result_frame: int) -> dict:
        if self.active_mode is None:
            raise QuickMeasureError("No active live measure mode.")
        mode = self.active_mode
        draft = list(self.draft_points)
        if not draft:
            raise QuickMeasureError("No draft points to finalize.")
        # Append draft into the session point list, then measure from those pts.
        start_id = len(self.points) + 1
        self.points.extend(draft)
        saved = self.points
        # Temporarily expose only the draft as the measure point set.
        self.points = draft
        try:
            result = self._compute_measure(mode)
        finally:
            self.points = saved
        n_used = int(result["n_points"])
        used = draft[-n_used:] if n_used else draft
        result = {
            **result,
            "frames": [p.frame for p in used],
            "point_ids": list(range(start_id, start_id + len(used))),
            "fps": self.fps,
            "calibration": self.calibration.kind if self.calibration else "none",
            "result_frame": int(result_frame),
            "pixels": [(p.x, p.y) for p in used],
            "reals": [self._to_units(p) for p in used],
            "mode_key": next((k for k, v in LIVE_MODE_KEYS.items() if v == mode), ""),
        }
        self.results.append(result)
        self.draft_points.clear()
        return result

    def _to_units(self, p: QuickMeasurePoint) -> tuple[float, float]:
        if self.unit_override:
            # Points are already expressed in real-world units.
            return p.x, p.y
        if self.calibration:
            return self.calibration.pixel_to_real(p.x, p.y)
        return p.x, p.y

    def _require_fps(self) -> float:
        if not self.fps or self.fps <= 0:
            raise QuickMeasureError(
                "Velocity/Acceleration need video FPS — press I to set FPS, then retry."
            )
        return float(self.fps)

    def measure_distance(self) -> dict:
        if len(self.points) < 2:
            raise QuickMeasureError("Distance needs at least 2 clicked points.")
        p1, p2 = self.points[-2], self.points[-1]
        x1, y1 = self._to_units(p1)
        x2, y2 = self._to_units(p2)
        value = float(np.hypot(x2 - x1, y2 - y1))
        return {"type": "distance", "value": value, "unit": self.unit_label, "n_points": 2}

    def measure_area(self) -> dict:
        if len(self.points) < 3:
            raise QuickMeasureError("Area needs at least 3 clicked points (polygon vertices).")
        pts = [self._to_units(p) for p in self.points]
        xs = np.array([p[0] for p in pts], dtype=float)
        ys = np.array([p[1] for p in pts], dtype=float)
        # Shoelace formula.
        value = 0.5 * abs(float(np.dot(xs, np.roll(ys, -1)) - np.dot(ys, np.roll(xs, -1))))
        return {
            "type": "area",
            "value": value,
            "unit": f"{self.unit_label}^2",
            "n_points": len(pts),
        }

    def measure_velocity(self) -> dict:
        if len(self.points) < 2:
            raise QuickMeasureError("Velocity needs at least 2 clicked points on different frames.")
        fps = self._require_fps()
        p1, p2 = self.points[-2], self.points[-1]
        if p1.frame == p2.frame:
            raise QuickMeasureError("Velocity needs 2 points clicked on different frames.")
        x1, y1 = self._to_units(p1)
        x2, y2 = self._to_units(p2)
        dt = (p2.frame - p1.frame) / fps
        distance = float(np.hypot(x2 - x1, y2 - y1))
        value = distance / dt if dt != 0 else float("nan")
        return {
            "type": "velocity",
            "value": value,
            "unit": f"{self.unit_label}/s",
            "n_points": 2,
            "dt": dt,
        }

    def measure_acceleration(self) -> dict:
        if len(self.points) < 3:
            raise QuickMeasureError(
                "Acceleration needs at least 3 clicked points on 3 distinct frames."
            )
        fps = self._require_fps()
        p1, p2, p3 = self.points[-3], self.points[-2], self.points[-1]
        if len({p1.frame, p2.frame, p3.frame}) < 3:
            raise QuickMeasureError("Acceleration needs 3 points on 3 distinct frames.")
        x1, y1 = self._to_units(p1)
        x2, y2 = self._to_units(p2)
        x3, y3 = self._to_units(p3)
        dt1 = (p2.frame - p1.frame) / fps
        dt2 = (p3.frame - p2.frame) / fps
        v1 = np.hypot(x2 - x1, y2 - y1) / dt1 if dt1 != 0 else float("nan")
        v2 = np.hypot(x3 - x2, y3 - y2) / dt2 if dt2 != 0 else float("nan")
        dt_avg = (dt1 + dt2) / 2.0
        value = (v2 - v1) / dt_avg if dt_avg != 0 else float("nan")
        return {
            "type": "acceleration",
            "value": float(value),
            "unit": f"{self.unit_label}/s^2",
            "n_points": 3,
        }

    def measure_angle(self) -> dict:
        """Angle in degrees.

        - 3 points: angle at the middle point (p[-2] is the vertex).
        - 4+ points: angle between line(p[-4],p[-3]) and line(p[-2],p[-1]).
        """
        if len(self.points) < 3:
            raise QuickMeasureError(
                "Angle needs 3 points (vertex in the middle) or 4 points (two lines)."
            )
        if len(self.points) == 3:
            a, b, c = (self._to_units(p) for p in self.points[-3:])
            v1 = np.array([a[0] - b[0], a[1] - b[1]], dtype=float)
            v2 = np.array([c[0] - b[0], c[1] - b[1]], dtype=float)
            n_points = 3
        else:
            p1, p2, p3, p4 = (self._to_units(p) for p in self.points[-4:])
            v1 = np.array([p2[0] - p1[0], p2[1] - p1[1]], dtype=float)
            v2 = np.array([p4[0] - p3[0], p4[1] - p3[1]], dtype=float)
            n_points = 4
        n1 = float(np.linalg.norm(v1))
        n2 = float(np.linalg.norm(v2))
        if n1 <= 0 or n2 <= 0:
            raise QuickMeasureError("Angle needs two non-zero length segments.")
        cos_a = float(np.clip(np.dot(v1, v2) / (n1 * n2), -1.0, 1.0))
        value = float(np.degrees(np.arccos(cos_a)))
        return {"type": "angle", "value": value, "unit": "deg", "n_points": n_points}

    def _compute_measure(self, kind: str) -> dict:
        kind = kind.strip().lower()
        dispatch = {
            "distance": self.measure_distance,
            "area": self.measure_area,
            "angle": self.measure_angle,
            "velocity": self.measure_velocity,
            "acceleration": self.measure_acceleration,
        }
        if kind not in dispatch:
            raise QuickMeasureError(
                f"Unknown measurement type: {kind!r} (expected one of {MEASURE_TYPES})"
            )
        return dispatch[kind]()

    def measure(self, kind: str) -> dict:
        result = self._compute_measure(kind)
        n_used = int(result["n_points"])
        used = self.points[-n_used:] if n_used else []
        first_id = len(self.points) - n_used + 1
        result = {
            **result,
            "frames": [p.frame for p in used],
            "point_ids": list(range(first_id, first_id + n_used)),
            "fps": self.fps,
            "calibration": self.calibration.kind if self.calibration else "none",
            "result_frame": used[-1].frame if used else None,
            "pixels": [(p.x, p.y) for p in used],
            "reals": [self._to_units(p) for p in used],
        }
        self.results.append(result)
        return result

    # ------------------------------------------------------------------
    # Persistence: every clicked point is saved with pixel AND real-world
    # coordinates so the measurements can be recomputed from the file alone.
    # ------------------------------------------------------------------

    def points_dataframe(self) -> pd.DataFrame:
        rows = []
        for i, p in enumerate(self.points, start=1):
            x_real, y_real = self._to_units(p)
            rows.append(
                {
                    "point_id": i,
                    "frame": p.frame,
                    "x_px": p.x,
                    "y_px": p.y,
                    "x_real": x_real,
                    "y_real": y_real,
                    "unit": self.unit_label,
                    "calibration_kind": self.calibration.kind if self.calibration else "none",
                }
            )
        columns = [
            "point_id",
            "frame",
            "x_px",
            "y_px",
            "x_real",
            "y_real",
            "unit",
            "calibration_kind",
        ]
        return pd.DataFrame(rows, columns=pd.Index(columns))

    def calibration_dataframe(self) -> pd.DataFrame:
        columns = [
            "kind",
            "unit",
            "measure_name",
            "measure_value",
            "scale_unit_per_px",
            "origin_x_px",
            "origin_y_px",
            "point_index",
            "x_px",
            "y_px",
            "x_real",
            "y_real",
            "dlt_param_index",
            "dlt_param_value",
        ]
        calib = self.calibration
        if calib is None:
            return pd.DataFrame(columns=pd.Index(columns))
        base = {
            "kind": calib.kind,
            "unit": calib.unit_label,
            "scale_unit_per_px": calib.scale,
            "origin_x_px": calib.origin_px[0] if calib.origin_px else None,
            "origin_y_px": calib.origin_px[1] if calib.origin_px else None,
        }
        rows: list[dict] = []
        for name, value in calib.real_measures.items():
            rows.append({**base, "measure_name": name, "measure_value": value})
        for i, (px, real) in enumerate(
            zip(calib.calibration_pixels, calib.calibration_real, strict=False)
        ):
            rows.append(
                {
                    **base,
                    "point_index": i + 1,
                    "x_px": px[0],
                    "y_px": px[1],
                    "x_real": real[0],
                    "y_real": real[1],
                }
            )
        if calib.dlt_params is not None:
            for i, value in enumerate(np.asarray(calib.dlt_params, dtype=float).tolist()):
                rows.append({**base, "dlt_param_index": i + 1, "dlt_param_value": value})
        if not rows:
            rows.append(base)
        return pd.DataFrame(rows, columns=pd.Index(columns))

    def results_dataframe(self, max_points: int | None = None) -> pd.DataFrame:
        """Return one result per row with every source value in its own column.

        Point groups expand as ``point_1_*``, ``point_2_*``, etc. This avoids
        embedding lists, comma pairs, or space-delimited values inside CSV
        cells and keeps exports directly usable as rectangular data tables.
        """
        observed_points = max(
            (
                max(
                    int(result.get("n_points") or 0),
                    len(result.get("frames") or []),
                    len(result.get("point_ids") or []),
                    len(result.get("pixels") or []),
                    len(result.get("reals") or []),
                )
                for result in self.results
            ),
            default=0,
        )
        matrix_points = max(observed_points, int(max_points or 0))
        columns = [
            "result_id",
            "type",
            "value",
            "unit",
            "n_points",
            "result_frame",
            "fps",
            "mode_key",
            "calibration",
            "elapsed_time_s",
        ]
        point_fields = ("id", "frame", "x_px", "y_px", "x_real", "y_real")
        columns.extend(
            f"point_{point_index}_{field_name}"
            for point_index in range(1, matrix_points + 1)
            for field_name in point_fields
        )
        rows = []
        for i, r in enumerate(self.results, start=1):
            frames = list(r.get("frames") or [])
            point_ids = list(r.get("point_ids") or [])
            pixels = list(r.get("pixels") or [])
            reals = list(r.get("reals") or [])
            row = {
                "result_id": i,
                "type": r.get("type"),
                "value": r.get("value"),
                "unit": r.get("unit"),
                "n_points": r.get("n_points"),
                "result_frame": r.get("result_frame"),
                "fps": r.get("fps"),
                "mode_key": r.get("mode_key", ""),
                "calibration": r.get("calibration", "none"),
                "elapsed_time_s": r.get("dt"),
            }
            for point_index in range(matrix_points):
                prefix = f"point_{point_index + 1}"
                pixel = pixels[point_index] if point_index < len(pixels) else (None, None)
                real = reals[point_index] if point_index < len(reals) else (None, None)
                row[f"{prefix}_id"] = (
                    point_ids[point_index] if point_index < len(point_ids) else None
                )
                row[f"{prefix}_frame"] = frames[point_index] if point_index < len(frames) else None
                row[f"{prefix}_x_px"] = pixel[0]
                row[f"{prefix}_y_px"] = pixel[1]
                row[f"{prefix}_x_real"] = real[0]
                row[f"{prefix}_y_real"] = real[1]
            rows.append(row)
        return pd.DataFrame(rows, columns=pd.Index(columns))

    def _resolve_export_dir(self, output_dir: str) -> str:
        """Create this session's export directory once, then keep reusing it."""
        if self.export_dir is not None:
            os.makedirs(self.export_dir, exist_ok=True)
            return self.export_dir

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base = os.path.join(output_dir, f"processed_quickmeasure_{timestamp}")
        candidate = base
        suffix = 2
        while os.path.exists(candidate):
            candidate = f"{base}_{suffix}"
            suffix += 1
        os.makedirs(candidate)
        self.export_dir = candidate
        return candidate

    def _write_html_report(
        self,
        report_path: str,
        safe_stem: str,
        paths: dict[str, str],
    ) -> None:
        """Write a standalone, human-readable guide to this session's export."""

        def esc(value: object) -> str:
            return html.escape(str(value), quote=True)

        measurement_rows = [
            (
                "Distance",
                "1",
                "2 points",
                "Straight-line length between the points.",
                "sqrt((x2-x1)^2 + (y2-y1)^2)",
                self.unit_label,
            ),
            (
                "Area",
                "2",
                "3 or more polygon vertices; Enter closes",
                "Area enclosed by the clicked polygon.",
                "Shoelace polygon formula",
                f"{self.unit_label}^2",
            ),
            (
                "Angle",
                "3",
                "3 points; middle point is the vertex",
                "Smaller angle between the two rays.",
                "arccos of normalized vector dot product",
                "degrees (deg)",
            ),
            (
                "Velocity",
                "4",
                "2 points on different frames",
                "Displacement magnitude divided by elapsed time.",
                "distance / ((frame2-frame1)/fps)",
                f"{self.unit_label}/s",
            ),
            (
                "Acceleration",
                "5",
                "3 points on distinct frames",
                "Change in segment speed divided by average elapsed time.",
                "(velocity2-velocity1) / average(dt1,dt2)",
                f"{self.unit_label}/s^2",
            ),
        ]
        file_descriptions = {
            "points": "Every clicked point in image pixels and calibrated coordinates.",
            "calibration": "Calibration mode, known dimensions, clicked references, and parameters.",
            "dlt2d": "Eight DLT2D coefficients used for pixel-to-plane reconstruction.",
            "ref2d": "Planar reference coordinates produced from the selected REF3D plane.",
            "ref3d_meta": "REF3D source, format, dropped axis, retained points, and units.",
            "results": "All completed measurements as one rectangular row per result.",
            "results_distance": "Distance results only.",
            "results_area": "Area results only.",
            "results_angle": "Angle results only.",
            "results_velocity": "Velocity results only.",
            "results_acceleration": "Acceleration results only.",
            "report": "This guide and live summary of the export.",
            "readme": "Compact plain-text file list and recomputation command.",
        }
        result_columns = [
            ("result_id", "Sequential result number in this session."),
            ("type", "distance, area, angle, velocity, or acceleration."),
            ("value", "Computed numeric metric."),
            ("unit", "Unit attached to value; calibrated unit, pixels, degrees, or rate."),
            ("n_points", "Number of points used by this result."),
            ("result_frame", "Frame where the completed result is drawn."),
            ("fps", "Video frames per second used for time-based metrics."),
            ("mode_key", "Quick Measure digit key that selected this type."),
            ("calibration", "Calibration model used by this result."),
            ("elapsed_time_s", "Elapsed seconds for velocity; blank for other result types."),
            ("point_N_id", "Source point identifier N."),
            ("point_N_frame", "Source video frame for point N."),
            ("point_N_x_px / point_N_y_px", "Separate image coordinates for point N."),
            (
                "point_N_x_real / point_N_y_real",
                "Separate calibrated plane coordinates for point N.",
            ),
        ]
        point_columns = [
            ("point_id", "Sequential clicked-point identifier."),
            ("frame", "Zero-based video frame containing the click."),
            ("x_px / y_px", "Horizontal / vertical image coordinate in pixels."),
            ("x_real / y_real", "Coordinates after calibration; equal to pixels if uncalibrated."),
            ("unit", "Coordinate unit."),
            ("calibration_kind", "none, line, plane, ref2d, dlt2d, or ref3d."),
        ]

        def table(headers: Sequence[str], rows: Sequence[Sequence[object]]) -> str:
            head = "".join(f"<th>{esc(item)}</th>" for item in headers)
            body = "".join(
                "<tr>" + "".join(f"<td>{esc(item)}</td>" for item in row) + "</tr>" for row in rows
            )
            return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"

        file_rows = [
            (role, os.path.basename(path), file_descriptions.get(role, "Session export file."))
            for role, path in paths.items()
            if role != "dir"
        ]
        result_rows = [
            (
                index,
                result.get("type", ""),
                f"{float(result.get('value', float('nan'))):.6g}",
                result.get("unit", ""),
                result.get("result_frame", ""),
                ", ".join(str(frame) for frame in result.get("frames", [])),
                ", ".join(str(point_id) for point_id in result.get("point_ids", [])),
            )
            for index, result in enumerate(self.results, start=1)
        ]
        calibration = self.calibration.describe() if self.calibration else "None; pixel units"
        results_section = (
            table(
                ("ID", "Type", "Value", "Unit", "Result frame", "Source frames", "Point IDs"),
                result_rows,
            )
            if result_rows
            else '<p class="notice">No completed measurement yet. This report will update on the next save.</p>'
        )
        generated = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Quick Measure report — {esc(safe_stem)}</title>
<style>
body{{font-family:Arial,sans-serif;line-height:1.55;color:#17202a;background:#f4f6f7;margin:0}}
main{{max-width:1100px;margin:24px auto;background:#fff;padding:28px;border-radius:10px}}
h1,h2{{color:#154360}} h2{{margin-top:30px;border-bottom:2px solid #d6eaf8;padding-bottom:5px}}
table{{border-collapse:collapse;width:100%;margin:12px 0;display:block;overflow-x:auto}}
th,td{{border:1px solid #ccd1d1;padding:8px 10px;text-align:left;vertical-align:top}}
th{{background:#d6eaf8}} code{{background:#eef2f3;padding:2px 5px;border-radius:4px}}
.summary,.notice{{background:#eaf2f8;padding:12px 16px;border-left:5px solid #2e86c1}}
.warning{{background:#fef9e7;padding:12px 16px;border-left:5px solid #f1c40f}}
</style>
</head>
<body><main>
<h1><i>vailá</i> Quick Measure report</h1>
<p>This page documents the data in this directory. It is regenerated whenever the session is saved.</p>
<div class="summary">
<strong>Video/data stem:</strong> {esc(safe_stem)}<br>
<strong>Generated:</strong> {esc(generated)}<br>
<strong>Calibration:</strong> {esc(calibration)}<br>
<strong>Coordinate unit:</strong> {esc(self.unit_label)} &nbsp;
<strong>FPS:</strong> {esc(self.fps if self.fps is not None else "not set")}<br>
<strong>Saved points:</strong> {len(self.points)} &nbsp;
<strong>Completed results:</strong> {len(self.results)}
</div>
<h2>Results from this session</h2>
{results_section}
<h2>What each measurement means</h2>
{table(("Measurement", "Key", "Required input", "Meaning", "Calculation", "Output unit"), measurement_rows)}
<div class="warning"><strong>Interpretation:</strong> Pixel values are image measurements, not physical measurements. Physical distance, area, velocity, and acceleration require a valid calibration. Velocity and acceleration also require the correct video FPS.</div>
<h2>Files in this directory</h2>
{table(("Role", "File", "Purpose"), file_rows)}
<h2>Results CSV column dictionary</h2>
{table(("Column", "Meaning"), result_columns)}
<p>Matrix layout: one completed measurement per row and one scalar per cell. Point columns repeat from <code>point_1_*</code> through the largest point set in the session. No list, coordinate pair, or space-delimited sequence is stored inside one CSV cell.</p>
<h2>Points CSV column dictionary</h2>
{table(("Column", "Meaning"), point_columns)}
<h2>Calibration data</h2>
<p>The calibration CSV records the known real dimensions, clicked image/reference points, scale, origin, and DLT parameters when applicable. REF3D exports may also include a planar <code>.ref2d</code>, DLT coefficients, and source metadata.</p>
<h2>Recompute without the video</h2>
<p>Because the points file stores both image and calibrated coordinates, metrics can be recomputed later:</p>
<p><code>uv run python -m vaila.quickmeasure --points-csv {esc(os.path.basename(paths["points"]))} --measure distance</code></p>
</main></body></html>
"""
        with open(report_path, "w", encoding="utf-8") as handle:
            handle.write(document)

    def save_session(self, output_dir: str, stem: str = "quickmeasure") -> dict[str, str]:
        """Write or refresh this session's files in one timestamped directory."""
        if not self.points and not self.results and self.calibration is None:
            raise QuickMeasureError("Nothing to save — no calibration, points or results yet.")
        run_dir = self._resolve_export_dir(output_dir)
        safe_stem = os.path.splitext(os.path.basename(stem))[0] or "quickmeasure"

        paths: dict[str, str] = {"dir": run_dir}
        points_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_points.csv")
        self.points_dataframe().to_csv(points_path, index=False)
        paths["points"] = points_path

        if self.calibration is not None:
            calib_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_calibration.csv")
            self.calibration_dataframe().to_csv(calib_path, index=False)
            paths["calibration"] = calib_path
            if self.calibration.dlt_params is not None:
                dlt_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure.dlt2d")
                params = np.asarray(self.calibration.dlt_params, dtype=float).tolist()
                pd.DataFrame(
                    [{"frame": 1, **{f"dlt_param_{i + 1}": v for i, v in enumerate(params)}}]
                ).to_csv(dlt_path, index=False)
                paths["dlt2d"] = dlt_path
            if self.calibration.kind == "ref3d" and self.calibration.calibration_real:
                keep = PLANE_KEEP_AXES.get(self.calibration.drop_axis or "z", ("x", "y"))
                ref2d_path = os.path.join(
                    run_dir,
                    f"{safe_stem}_quickmeasure_drop{(self.calibration.drop_axis or 'z').upper()}.ref2d",
                )
                data: dict[str, list[float | int]] = {"frame": [0]}
                labels = self.calibration.kept_point_indices or list(
                    range(1, len(self.calibration.calibration_real) + 1)
                )
                for label, (u, v) in zip(labels, self.calibration.calibration_real, strict=False):
                    data[f"p{label}_x"] = [u]
                    data[f"p{label}_y"] = [v]
                pd.DataFrame(data).to_csv(ref2d_path, index=False)
                paths["ref2d"] = ref2d_path
                meta_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_ref3d_meta.csv")
                pd.DataFrame(
                    [
                        {
                            "ref3d_path": self.calibration.ref3d_path,
                            "ref3d_format": self.calibration.ref3d_format,
                            "drop_axis": self.calibration.drop_axis,
                            "kept_axes": "".join(keep),
                            "pixel_csv_path": self.calibration.pixel_csv_path,
                            "kept_point_indices": " ".join(str(i) for i in labels),
                            "unit": self.calibration.unit_label,
                        }
                    ]
                ).to_csv(meta_path, index=False)
                paths["ref3d_meta"] = meta_path

        if self.results:
            results_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_results.csv")
            matrix_points = max(int(result.get("n_points") or 0) for result in self.results)
            self.results_dataframe(max_points=matrix_points).to_csv(results_path, index=False)
            paths["results"] = results_path
            # One CSV per measure type that has data.
            by_type: dict[str, list[dict]] = {}
            for r in self.results:
                by_type.setdefault(str(r.get("type")), []).append(r)
            for kind, items in by_type.items():
                type_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_results_{kind}.csv")
                # Reuse dataframe builder via a temporary session slice.
                tmp = QuickMeasureSession(fps=self.fps, calibration=self.calibration)
                tmp.results = items
                tmp.results_dataframe(max_points=matrix_points).to_csv(type_path, index=False)
                paths[f"results_{kind}"] = type_path

        report = os.path.join(run_dir, "quickmeasure_report.html")
        readme = os.path.join(run_dir, "README_quickmeasure.txt")
        paths["report"] = report
        paths["readme"] = readme
        self._write_html_report(report, safe_stem, paths)
        with open(readme, "w", encoding="utf-8") as handle:
            handle.write("vailá Quick Measure export\n")
            handle.write(f"stem: {safe_stem}\n")
            handle.write(f"unit: {self.unit_label}\n")
            if self.calibration is not None:
                handle.write(f"calibration: {self.calibration.describe()}\n")
            handle.write(
                "results_schema: one result per row; point_N_* scalar columns; no packed lists\n"
            )
            handle.write("files:\n")
            for role, path in paths.items():
                if role != "dir":
                    handle.write(f"  - {role}: {os.path.basename(path)}\n")
            handle.write(
                "\nRecompute: uv run python -m vaila.quickmeasure "
                f"--points-csv {os.path.basename(points_path)} --measure distance\n"
            )
        return paths


def format_result(result: dict) -> str:
    label = result["type"].capitalize()
    return f"{label}: {result['value']:.4f} {result['unit']}"


def needs_calibration(session: QuickMeasureSession | None) -> bool:
    """True when entering Quick Measure must start with calibration.

    This is the calibration-first rule the host GUI applies when `Q` / the
    **QMeas** button turns the mode on: a brand-new session calibrates first,
    an already-calibrated one (or one where the user explicitly chose to stay
    in pixels) goes straight to free measuring.
    """
    if session is None:
        return True
    return session.calibration is None and not session.calibration_skipped


# -----------------------------------------------------------------------
# Recompute measurements from a saved calibrated points CSV (no video).
# -----------------------------------------------------------------------


def session_from_points_csv(points_csv: str, fps: float | None = None) -> QuickMeasureSession:
    """Rebuild a session from a CSV written by `QuickMeasureSession.save_session()`.

    Real-world columns are used when present (calibrated file); otherwise the
    pixel columns are used and the session stays in ``px``.
    """
    if not os.path.isfile(points_csv):
        raise QuickMeasureError(f"Points CSV not found: {points_csv}")
    df = pd.read_csv(points_csv)
    if df.empty:
        raise QuickMeasureError(f"Points CSV has no rows: {points_csv}")
    if "frame" not in df.columns:
        raise QuickMeasureError(f"Points CSV is missing the 'frame' column: {points_csv}")

    has_real = {"x_real", "y_real"}.issubset(df.columns) and not df["x_real"].isna().all()
    if has_real:
        x_col, y_col = "x_real", "y_real"
    elif {"x_px", "y_px"}.issubset(df.columns):
        x_col, y_col = "x_px", "y_px"
    else:
        raise QuickMeasureError(
            f"Points CSV needs x_real/y_real or x_px/y_px columns: {points_csv}"
        )

    unit = "px"
    if "unit" in df.columns and df["unit"].notna().any():
        unit = str(df["unit"].dropna().iloc[0])
    session = QuickMeasureSession(fps=fps)
    if has_real and unit != "px":
        session.unit_override = unit
    for _, row in df.iterrows():
        session.add_point(int(row["frame"]), float(row[x_col]), float(row[y_col]))
    return session


def measure_from_points_csv(
    points_csv: str, kind: str, fps: float | None = None, point_ids: Sequence[int] | None = None
) -> dict:
    """Compute one measurement from a saved calibrated points CSV.

    ``point_ids`` (1-based, as written in the CSV) selects a subset; by
    default the whole file is used with the documented point-set convention
    (last 2 for distance/velocity, last 3 for acceleration, all for area).
    """
    session = session_from_points_csv(points_csv, fps=fps)
    if point_ids:
        wanted = [int(i) for i in point_ids]
        total = len(session.points)
        for i in wanted:
            if i < 1 or i > total:
                raise QuickMeasureError(f"point_id {i} out of range (file has {total} points).")
        session.points = [session.points[i - 1] for i in wanted]
    return session.measure(kind)


def _build_cli_parser():
    import argparse

    parser = argparse.ArgumentParser(
        prog="python -m vaila.quickmeasure",
        description=(
            "Recompute Kinovea-style measurements from a calibrated quick-measure points CSV."
        ),
    )
    parser.add_argument("--points-csv", required=True, help="CSV written by Quick Measure save.")
    parser.add_argument(
        "--measure",
        default="distance",
        choices=list(MEASURE_TYPES),
        help="Measurement to compute (default: distance).",
    )
    parser.add_argument("--fps", type=float, default=None, help="Frame rate (velocity/accel).")
    parser.add_argument(
        "--point-ids",
        default=None,
        help="Optional 1-based point ids to use, comma separated (e.g. 1,4,5).",
    )
    parser.add_argument("--out", default=None, help="Optional CSV path to append the result to.")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point: measure from a saved points CSV, no video needed."""
    args = _build_cli_parser().parse_args(argv)
    ids = [int(v) for v in args.point_ids.split(",") if v.strip()] if args.point_ids else None
    try:
        result = measure_from_points_csv(args.points_csv, args.measure, fps=args.fps, point_ids=ids)
    except QuickMeasureError as exc:
        print(f">> vaila/quickmeasure: error: {exc}")
        return 2
    print(f">> vaila/quickmeasure: {format_result(result)}")
    if args.out:
        row = pd.DataFrame(
            [
                {
                    "points_csv": args.points_csv,
                    "type": result["type"],
                    "value": result["value"],
                    "unit": result["unit"],
                    "n_points": result["n_points"],
                    "fps": args.fps,
                }
            ]
        )
        header = not os.path.isfile(args.out)
        row.to_csv(args.out, mode="a", header=header, index=False)
        print(f">> vaila/quickmeasure: appended result to {args.out}")
    return 0


# -----------------------------------------------------------------------
# pygame UI helpers (imported lazily so the math/state above stays usable,
# and unit-testable, without a display or pygame installed).
# -----------------------------------------------------------------------


def draw_quickmeasure_overlay(
    screen, session: QuickMeasureSession, zoom_level, crop_x, crop_y, font
):
    """Draw completed measurements (with value labels) + in-progress draft."""
    import pygame

    def _to_screen(x: float, y: float) -> tuple[int, int]:
        return int((x * zoom_level) - crop_x), int((y * zoom_level) - crop_y)

    color_done = (255, 0, 255)
    color_draft = (255, 180, 0)
    color_label = (255, 255, 0)

    for r in session.results:
        pixels = r.get("pixels") or []
        if not pixels:
            continue
        screen_pts = [_to_screen(px, py) for px, py in pixels]
        rtype = str(r.get("type", ""))
        if rtype == "area" and len(screen_pts) >= 3:
            pygame.draw.polygon(screen, color_done, screen_pts, 1)
        elif len(screen_pts) >= 2:
            pygame.draw.lines(screen, color_done, False, screen_pts, 2)
        for sx, sy in screen_pts:
            pygame.draw.circle(screen, color_done, (sx, sy), 4, 1)
        # Label near the geometric mid / vertex.
        if rtype == "angle" and len(screen_pts) >= 3:
            lx, ly = screen_pts[1] if len(screen_pts) == 3 else screen_pts[0]
        elif screen_pts:
            lx = sum(p[0] for p in screen_pts) // len(screen_pts)
            ly = sum(p[1] for p in screen_pts) // len(screen_pts)
        else:
            continue
        label = f"{r.get('value', float('nan')):.3f} {r.get('unit', '')}"
        screen.blit(font.render(label, True, color_label), (lx + 8, ly - 18))

    # In-progress draft for the active live mode.
    if session.draft_points:
        draft_pts = [_to_screen(p.x, p.y) for p in session.draft_points]
        if len(draft_pts) >= 2:
            closed = session.active_mode == "area" and len(draft_pts) >= 3
            pygame.draw.lines(screen, color_draft, closed, draft_pts, 1)
        for i, (sx, sy) in enumerate(draft_pts):
            pygame.draw.circle(screen, color_draft, (sx, sy), 5, 2)
            screen.blit(font.render(str(i + 1), True, color_draft), (sx + 6, sy - 14))

    # Mode banner.
    banner_txt = session.live_mode_status()
    banner = font.render(banner_txt, True, (0, 0, 0))
    pad = 6
    box = pygame.Surface((banner.get_width() + 2 * pad, banner.get_height() + 2 * pad))
    box.fill((255, 180, 0) if session.active_mode else (200, 200, 200))
    box.blit(banner, (pad, pad))
    screen.blit(box, (10, screen.get_height() - box.get_height() - 10))


def draw_calibration_overlay(
    screen, draft: CalibrationDraft | Ref3dCalibrationDraft, zoom_level, crop_x, crop_y, font
) -> None:
    """Draw the calibration clicks collected so far plus the next-step
    instruction banner. Called every frame while calibration is pending.
    """
    import pygame

    color = (0, 255, 255)  # Cyan: distinct from magenta measure points.
    screen_pts = []
    points = draft.pixel_points if isinstance(draft, Ref3dCalibrationDraft) else draft.points
    for px, py in points:
        screen_pts.append((int((px * zoom_level) - crop_x), int((py * zoom_level) - crop_y)))
    if len(screen_pts) >= 2:
        closed = isinstance(draft, CalibrationDraft) and draft.mode == "plane" and draft.is_complete
        pygame.draw.lines(screen, color, closed, screen_pts, 2)
    for i, (sx, sy) in enumerate(screen_pts):
        pygame.draw.circle(screen, color, (sx, sy), 6, 2)
        label = (
            f"p{draft.kept_labels[i]}"
            if isinstance(draft, Ref3dCalibrationDraft) and i < len(draft.kept_labels)
            else f"C{i + 1}"
        )
        screen.blit(font.render(label, True, color), (sx + 8, sy - 16))

    if isinstance(draft, Ref3dCalibrationDraft):
        draw_ref3d_scheme_panel(screen, draft, font)

    banner = font.render(draft.instructions(), True, (0, 0, 0))
    pad = 6
    box = pygame.Surface((banner.get_width() + 2 * pad, banner.get_height() + 2 * pad))
    box.fill(color)
    box.blit(banner, (pad, pad))
    screen.blit(box, (10, 10))


def draw_ref3d_scheme_panel(
    screen, draft: Ref3dCalibrationDraft, font, *, margin: int = 12
) -> None:
    """Draw a small orthographic scheme of the REF3D points (kept plane axes)."""
    import pygame

    keep = PLANE_KEEP_AXES[draft.drop_axis]
    axis_i = {"x": 1, "y": 2, "z": 3}  # offset into (label,x,y,z)
    pts = []
    for label in draft.kept_labels:
        for p in draft.points_xyz:
            if p[0] == label:
                pts.append((label, float(p[axis_i[keep[0]]]), float(p[axis_i[keep[1]]])))
                break
    if not pts:
        return

    us = [p[1] for p in pts]
    vs = [p[2] for p in pts]
    u_min, u_max = min(us), max(us)
    v_min, v_max = min(vs), max(vs)
    span_u = max(u_max - u_min, 1e-9)
    span_v = max(v_max - v_min, 1e-9)

    panel_w, panel_h = 220, 180
    screen_w, screen_h = screen.get_size()
    ox = screen_w - panel_w - margin
    oy = margin
    panel = pygame.Surface((panel_w, panel_h))
    panel.fill((20, 20, 30))
    pygame.draw.rect(panel, (0, 255, 255), panel.get_rect(), 1)
    title = font.render(f"REF3D scheme ({''.join(a.upper() for a in keep)})", True, (0, 255, 255))
    panel.blit(title, (8, 6))

    plot_x0, plot_y0 = 24, 28
    plot_w, plot_h = panel_w - 40, panel_h - 48
    next_label = draft.next_label
    for label, u, v in pts:
        sx = plot_x0 + int((u - u_min) / span_u * plot_w)
        # Screen y grows down; keep world +v upward on the panel.
        sy = plot_y0 + plot_h - int((v - v_min) / span_v * plot_h)
        color = (255, 255, 0) if label == next_label else (180, 220, 255)
        pygame.draw.circle(panel, color, (sx, sy), 5 if label == next_label else 3)
        panel.blit(font.render(str(label), True, color), (sx + 6, sy - 8))
    screen.blit(panel, (ox, oy))


def finish_calibration_draft(
    draft: CalibrationDraft, ask_text
) -> tuple[QuickMeasureCalibration | None, str]:
    """Ask the user for the real measurement(s) and build the calibration.

    ``ask_text(prompt, default) -> str | None`` is injected by the caller
    (`getpixelvideo.py` passes its pygame ``show_input_dialog``), so this
    flow stays testable without a display. Returning ``None`` from
    ``ask_text`` cancels.
    """
    if not draft.is_complete:
        return None, (
            f"Calibration needs {draft.remaining} more click(s) before entering measurements."
        )
    values: list[float] = []
    for name in draft.required_measures:
        answer = ask_text(f"Real {name} of the clicked {draft.mode} (in {draft.unit_label})", "")
        if answer is None or not str(answer).strip():
            return None, "Calibration cancelled."
        try:
            values.append(float(str(answer).strip().replace(",", ".")))
        except ValueError:
            return None, f"Invalid {name}: {answer!r} is not a number."
    try:
        calib = draft.build(values)
    except QuickMeasureError as exc:
        return None, f"Calibration error: {exc}"
    return calib, calib.describe()


def finish_ref3d_calibration_draft(
    draft: Ref3dCalibrationDraft,
) -> tuple[QuickMeasureCalibration | None, str]:
    """Solve DLT2D once every guided REF3D image click is collected."""
    if not draft.is_complete:
        return None, draft.instructions()
    try:
        calib = draft.build()
    except QuickMeasureError as exc:
        return None, f"Calibration error: {exc}"
    return calib, calib.describe()


def _ask_drop_axis_via_dialog(parent=None) -> str | None:
    """Ask which world axis to drop for planar DLT2D. Returns ``x``/``y``/``z`` or None."""
    import tkinter as tk
    from tkinter import simpledialog

    root = parent
    owns_root = False
    if root is None:
        root = tk.Tk()
        root.withdraw()
        owns_root = True
    try:
        answer = simpledialog.askstring(
            "Quick Measure — REF3D plane",
            "Drop which world axis for rec2d / DLT2D?\n"
            "  z = keep XY (floor / top-down)\n"
            "  y = keep XZ\n"
            "  x = keep YZ\n"
            "Default: z",
            initialvalue="z",
            parent=root,
        )
    finally:
        if owns_root:
            root.destroy()
    if answer is None:
        return None
    axis = str(answer).strip().lower()
    if axis not in PLANE_DROP_AXES:
        return None
    return axis


def _ask_calibration_via_dialog(
    video_path: str | None = None,
) -> tuple[QuickMeasureCalibration | None, str]:
    """Tkinter file-picker flow for DLT2D / REF2D / REF3D calibration."""
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        choice = messagebox.askyesnocancel(
            "Quick Measure — Calibration",
            "Yes: load an existing .dlt2d coefficients file.\n"
            "No: load pixel CSV + reference (.ref2d or .ref3d).\n"
            "Cancel: abort.",
        )
        if choice is None:
            return None, "Calibration cancelled."
        if choice:
            dlt2d_file = filedialog.askopenfilename(
                title="Select .dlt2d coefficients file",
                filetypes=[("DLT2D files", "*.dlt2d"), ("CSV files", "*.csv")],
            )
            if not dlt2d_file:
                return None, "Calibration cancelled."
            calib = QuickMeasureCalibration.from_dlt2d_file(dlt2d_file)
            return calib, f"Calibration loaded: {os.path.basename(dlt2d_file)}"

        pixel_file = filedialog.askopenfilename(
            title="Select PIXEL calibration CSV (getpixelvideo markers)",
            filetypes=[("CSV files", "*.csv")],
        )
        if not pixel_file:
            return None, "Calibration cancelled."
        ref_file = filedialog.askopenfilename(
            title="Select REF2D or REF3D real-world coordinates",
            filetypes=[
                ("REF3D / REF2D", "*.ref3d *.ref2d"),
                ("REF3D files", "*.ref3d"),
                ("REF2D files", "*.ref2d"),
                ("CSV files", "*.csv"),
            ],
        )
        if not ref_file:
            return None, "Calibration cancelled."

        lower = ref_file.lower()
        if lower.endswith(".ref3d"):
            drop = _ask_drop_axis_via_dialog(parent=root)
            if drop is None:
                return None, "Calibration cancelled."
            calib = QuickMeasureCalibration.from_ref3d_and_pixel_csv(
                ref_file, pixel_file, drop_axis=drop, unit_label="m"
            )
            return calib, calib.describe()

        calib = QuickMeasureCalibration.from_calibration_points(pixel_file, ref_file)
        return calib, f"Calibration computed from: {os.path.basename(pixel_file)}"
    except QuickMeasureError as e:
        messagebox.showerror("Quick Measure — Calibration error", str(e))
        return None, f"Calibration error: {e}"
    finally:
        root.destroy()


def ask_ref3d_calibration_files(
    video_path: str | None = None,
    unit_label: str = "m",
) -> tuple[QuickMeasureCalibration | Ref3dCalibrationDraft | None, str]:
    """Pick a ``.ref3d``, choose drop axis, then load pixel CSV or return a guide draft.

    Returns either a ready ``QuickMeasureCalibration`` (CSV found/chosen) or a
    ``Ref3dCalibrationDraft`` the host must fill by guided clicks.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        ref_file = filedialog.askopenfilename(
            title="Select REF3D file (mode1 / mode2 / mode3)",
            filetypes=[("REF3D files", "*.ref3d"), ("CSV files", "*.csv"), ("All", "*.*")],
        )
        if not ref_file:
            return None, "REF3D calibration cancelled."
        drop = _ask_drop_axis_via_dialog(parent=root)
        if drop is None:
            return None, "REF3D calibration cancelled."

        suggested = suggest_pixel_csv_for_ref3d(ref_file, video_path)
        use_csv = messagebox.askyesno(
            "Quick Measure — pixel coordinates",
            "Load a getpixelvideo pixel CSV for these REF3D points?\n\n"
            f"Suggested: {os.path.basename(suggested) if suggested else '(none found)'}\n\n"
            "Yes: pick / confirm the CSV and solve DLT2D now.\n"
            "No: guide — click each REF3D point on the video (scheme shown).",
        )
        if use_csv:
            initial = suggested or ""
            pixel_file = filedialog.askopenfilename(
                title="Select PIXEL calibration CSV",
                initialdir=os.path.dirname(initial) if initial else None,
                initialfile=os.path.basename(initial) if initial else None,
                filetypes=[("CSV files", "*.csv")],
            )
            if not pixel_file:
                return None, "REF3D calibration cancelled."
            calib = QuickMeasureCalibration.from_ref3d_and_pixel_csv(
                ref_file, pixel_file, drop_axis=drop, unit_label=unit_label
            )
            return calib, calib.describe()

        draft = Ref3dCalibrationDraft.from_ref3d_file(
            ref_file, drop_axis=drop, unit_label=unit_label
        )
        return draft, draft.instructions()
    except QuickMeasureError as e:
        messagebox.showerror("Quick Measure — REF3D error", str(e))
        return None, f"Calibration error: {e}"
    finally:
        root.destroy()


def show_quickmeasure_menu(
    screen,
    session: QuickMeasureSession,
    window_width,
    window_height,
    save_dir: str | None = None,
    save_stem: str = "quickmeasure",
) -> str:
    """Blocking modal submenu (own pygame event loop, mirrors
    `show_help_dialog`'s pattern so it never touches the host file's main
    loop state machine). Returns a one-line status message for the caller to
    show via its existing `save_message_text` mechanism.
    """
    import pygame

    font = pygame.font.Font(None, 26)
    small_font = pygame.font.Font(None, 20)

    n = len(session.points)
    calib_line = (
        session.calibration.describe()
        if session.calibration
        else "no calibration — values are in pixels"
    )
    lines = [
        "QUICK MEASURE — save / calibrate / classify",
        "",
        f"Points: {n}   Results: {len(session.results)}   Unit: {session.unit_label}",
        f"Live mode: {session.active_mode or '(press 1-5 on video)'}   FPS: {session.fps}",
        f"Calibration: {calib_line}",
        "",
        "On the VIDEO (after Q):",
        "  1 distance  2 area  3 angle  4 velocity  5 accel",
        "  6-0 reserved — value drawn on-image per completed set",
        "  Enter closes area polygon; velocity/accel need FPS (I)",
        "",
        "In this menu:",
        "1-5: classify current free points (legacy)",
        "S: Save points / calibration / results CSV",
        "C: Load DLT2D / REF2D / REF3D calibration...",
        "R: Load REF3D calibration (plane drop + CSV or guide)",
        "X: Clear all points (+ draft)",
        "",
        "Esc: Close menu",
    ]

    overlay = pygame.Surface((window_width, window_height))
    result_message: str | None = None
    waiting = True
    while waiting:
        overlay.fill((0, 0, 0))
        for i, line in enumerate(lines):
            f = font if i == 0 else small_font
            color = (255, 255, 0) if i == 0 else (255, 255, 255)
            overlay.blit(f.render(line, True, color), (20, 20 + i * 26))
        if result_message:
            overlay.blit(
                small_font.render(result_message, True, (0, 255, 0)),
                (20, 20 + len(lines) * 26 + 10),
            )
        overlay.set_alpha(235)
        screen.blit(overlay, (0, 0))
        pygame.display.flip()

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE:
                    waiting = False
                elif event.key in (pygame.K_1, pygame.K_KP1):
                    try:
                        result_message = format_result(session.measure("distance"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key in (pygame.K_2, pygame.K_KP2):
                    try:
                        result_message = format_result(session.measure("area"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key in (pygame.K_3, pygame.K_KP3):
                    try:
                        result_message = format_result(session.measure("angle"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    try:
                        result_message = format_result(session.measure("velocity"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key in (pygame.K_5, pygame.K_KP5):
                    try:
                        result_message = format_result(session.measure("acceleration"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key == pygame.K_c:
                    video_guess = (
                        os.path.join(save_dir, save_stem) if save_dir and save_stem else None
                    )
                    calib, msg = _ask_calibration_via_dialog(video_guess)
                    if calib is not None:
                        session.calibration = calib
                    result_message = msg
                elif event.key == pygame.K_r:
                    video_guess = (
                        os.path.join(save_dir, save_stem) if save_dir and save_stem else None
                    )
                    result, msg = ask_ref3d_calibration_files(video_guess)
                    if isinstance(result, QuickMeasureCalibration):
                        session.calibration = result
                        result_message = msg
                    elif isinstance(result, Ref3dCalibrationDraft):
                        session.pending_ref3d_draft = result
                        result_message = (
                            f"REF3D guide ready — close menu (Esc) then click points. {msg}"
                        )
                    else:
                        result_message = msg
                elif event.key == pygame.K_s:
                    try:
                        paths = session.save_session(save_dir or os.getcwd(), save_stem)
                        result_message = f"Saved: {os.path.basename(paths['dir'])}"
                        print(f">> vaila/quickmeasure: saved CSVs in {paths['dir']}")
                        for role, path in paths.items():
                            if role != "dir":
                                print(f">> vaila/quickmeasure:   {role}: {path}")
                        print(
                            ">> vaila/quickmeasure: Equivalent CLI\n"
                            f"uv run python -m vaila.quickmeasure "
                            f"--points-csv {paths['points']} --measure distance"
                        )
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key == pygame.K_x:
                    session.clear()
                    session.clear_results()
                    session.active_mode = None
                    result_message = "Points, draft and results cleared."

    return result_message or "Quick Measure menu closed (no measurement taken)."


if __name__ == "__main__":
    raise SystemExit(main())
