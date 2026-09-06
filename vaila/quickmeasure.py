"""
================================================================================
Script: quickmeasure.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.124
Created: 06 September 2026
Last Updated: 06 September 2026
================================================================================
Description:
    Kinovea-style quick on-image measurements for `getpixelvideo.py`:
    calibrate first by clicking on the image and typing the real-world
    measurement, then click points freely and classify the current point set
    as Distance, Area, Velocity, or Acceleration.

    Calibration-first flow (mirrors Kinovea's "calibrate measure"):
      - ``line``  — 2 clicks on a segment of known length + the typed length.
                    Isotropic scale, origin at the first click, +x right,
                    +y up (image y inverted). Valid for a plane parallel to
                    the sensor.
      - ``plane`` — 4 clicks around a rectangle of known width/height +
                    the typed width and height. Solves the 8-parameter DLT2D
                    homography, so perspective is corrected.
    Both are built from clicks alone (`CalibrationDraft`), no file picking.
    Loading an existing ``.dlt2d`` (or pixel CSV + ``.ref2d``) still works.

    Every clicked point can be saved to CSV with pixel AND calibrated
    real-world coordinates; `measure_from_points_csv()` (and the
    ``python -m vaila.quickmeasure`` CLI) recompute Distance/Area/Velocity/
    Acceleration from that saved file alone, without the video.

    This module owns all quick-measurement STATE and MATH (and, for the
    pygame-based UI pieces, the modal submenu + overlay rendering). It does
    NOT reimplement calibration math — DLT2D parameters are computed by
    reusing `dlt2d.py` (`dlt2d()`, `process_files()`) exactly the way
    `dlt2d.py`'s own CLI does, and pixel->real-world conversion reuses
    `rec2d_one_dlt2d.py`'s `rec2d()`. `getpixelvideo.py` only owns the
    integration glue (state var, hotkeys, one click handler, one draw call).

    Single-video sessions can only ever be calibrated with DLT2D (one image
    plane = 2D). Stereo DLT3D triangulation (`rec3d_multicam()` from
    `rec3d.py`) requires two synchronized cameras/videos and is intentionally
    out of scope for this module's live single-video click session; it stays
    a batch/CLI workflow via `rec3d_one_dlt3d.py`.

Point-set convention (documented, not configurable, so results are
reproducible from clicks alone):
    - Distance / Velocity: use the LAST 2 clicked points.
    - Area: use ALL clicked points, in click order (shoelace polygon).
    - Acceleration: use the LAST 3 clicked points.

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

import os
from collections.abc import Sequence
from dataclasses import dataclass, field
from datetime import datetime

import numpy as np
import pandas as pd

try:
    from .dlt2d import dlt2d as dlt2d_solve
    from .dlt2d import process_files as dlt2d_process_files
    from .rec2d_one_dlt2d import rec2d
except ImportError:
    from dlt2d import dlt2d as dlt2d_solve  # ty: ignore[unresolved-import]
    from dlt2d import process_files as dlt2d_process_files  # ty: ignore[unresolved-import]
    from rec2d_one_dlt2d import rec2d  # ty: ignore[unresolved-import]


MEASURE_TYPES: tuple[str, ...] = ("distance", "area", "velocity", "acceleration")

# Click-based calibration modes and how many image clicks each one needs.
CALIBRATION_MODES: tuple[str, ...] = ("line", "plane")
CALIBRATION_CLICKS: dict[str, int] = {"line": 2, "plane": 4}
# Real-world measurements the user must type after clicking, per mode.
CALIBRATION_MEASURES: dict[str, tuple[str, ...]] = {
    "line": ("length",),
    "plane": ("width", "height"),
}


class QuickMeasureError(ValueError):
    """Raised for invalid/insufficient quick-measurement input.

    A plain ValueError subclass so callers that already catch ValueError
    keep working, while UI code can special-case this type if it wants to.
    """


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
class QuickMeasurePoint:
    frame: int
    x: float
    y: float


@dataclass
class QuickMeasureSession:
    """Click-session state for one video. `getpixelvideo.py` calls
    `add_point()` on left-click and `measure()` from its submenu; all
    geometry/kinematics math lives here so the host file stays a thin
    integration layer.
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
        if self.points:
            self.points.pop()
            return True
        return False

    def clear(self) -> None:
        self.points.clear()

    def _to_units(self, p: QuickMeasurePoint) -> tuple[float, float]:
        if self.unit_override:
            # Points are already expressed in real-world units.
            return p.x, p.y
        if self.calibration:
            return self.calibration.pixel_to_real(p.x, p.y)
        return p.x, p.y

    def _require_fps(self) -> float:
        if not self.fps or self.fps <= 0:
            raise QuickMeasureError("Velocity/Acceleration require a valid fps (frame rate).")
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

    def measure(self, kind: str) -> dict:
        kind = kind.strip().lower()
        dispatch = {
            "distance": self.measure_distance,
            "area": self.measure_area,
            "velocity": self.measure_velocity,
            "acceleration": self.measure_acceleration,
        }
        if kind not in dispatch:
            raise QuickMeasureError(
                f"Unknown measurement type: {kind!r} (expected one of {MEASURE_TYPES})"
            )
        result = dispatch[kind]()
        n_used = int(result["n_points"])
        used = self.points[-n_used:] if n_used else []
        first_id = len(self.points) - n_used + 1
        result = {
            **result,
            "frames": [p.frame for p in used],
            "point_ids": list(range(first_id, first_id + n_used)),
            "fps": self.fps,
            "calibration": self.calibration.kind if self.calibration else "none",
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

    def results_dataframe(self) -> pd.DataFrame:
        columns = ["result_id", "type", "value", "unit", "n_points", "frames", "point_ids", "fps"]
        rows = []
        for i, r in enumerate(self.results, start=1):
            rows.append(
                {
                    "result_id": i,
                    "type": r.get("type"),
                    "value": r.get("value"),
                    "unit": r.get("unit"),
                    "n_points": r.get("n_points"),
                    "frames": " ".join(str(f) for f in r.get("frames", [])),
                    "point_ids": " ".join(str(p) for p in r.get("point_ids", [])),
                    "fps": r.get("fps"),
                }
            )
        return pd.DataFrame(rows, columns=pd.Index(columns))

    def save_session(self, output_dir: str, stem: str = "quickmeasure") -> dict[str, str]:
        """Write the point, calibration and result CSVs into a timestamped
        folder under ``output_dir``. Returns the written paths by role.
        """
        if not self.points and not self.results and self.calibration is None:
            raise QuickMeasureError("Nothing to save — no calibration, points or results yet.")
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        run_dir = os.path.join(output_dir, f"processed_quickmeasure_{timestamp}")
        os.makedirs(run_dir, exist_ok=True)
        safe_stem = os.path.splitext(os.path.basename(stem))[0] or "quickmeasure"

        paths: dict[str, str] = {"dir": run_dir}
        points_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_points.csv")
        self.points_dataframe().to_csv(points_path, index=False)
        paths["points"] = points_path

        if self.calibration is not None:
            calib_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_calibration.csv")
            self.calibration_dataframe().to_csv(calib_path, index=False)
            paths["calibration"] = calib_path

        if self.results:
            results_path = os.path.join(run_dir, f"{safe_stem}_quickmeasure_results.csv")
            self.results_dataframe().to_csv(results_path, index=False)
            paths["results"] = results_path
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
    """Draw clicked quick-measure points + connecting lines on `screen` (in
    video pixel -> screen coordinate space, same convention `getpixelvideo.py`
    uses for markers/boxes). No-op when there are no points.
    """
    if not session.points:
        return
    import pygame

    color = (255, 0, 255)  # Magenta: visually distinct from marker/bbox colors.
    screen_pts = []
    for p in session.points:
        sx = int((p.x * zoom_level) - crop_x)
        sy = int((p.y * zoom_level) - crop_y)
        screen_pts.append((sx, sy))

    if len(screen_pts) >= 2:
        pygame.draw.lines(screen, color, False, screen_pts, 1)
    for i, (sx, sy) in enumerate(screen_pts):
        pygame.draw.circle(screen, color, (sx, sy), 4, 1)
        label_surface = font.render(str(i + 1), True, color)
        screen.blit(label_surface, (sx + 6, sy - 14))


def draw_calibration_overlay(
    screen, draft: CalibrationDraft, zoom_level, crop_x, crop_y, font
) -> None:
    """Draw the calibration clicks collected so far plus the next-step
    instruction banner. Called every frame while calibration is pending.
    """
    import pygame

    color = (0, 255, 255)  # Cyan: distinct from magenta measure points.
    screen_pts = []
    for px, py in draft.points:
        screen_pts.append((int((px * zoom_level) - crop_x), int((py * zoom_level) - crop_y)))
    if len(screen_pts) >= 2:
        closed = draft.mode == "plane" and draft.is_complete
        pygame.draw.lines(screen, color, closed, screen_pts, 2)
    for i, (sx, sy) in enumerate(screen_pts):
        pygame.draw.circle(screen, color, (sx, sy), 6, 2)
        screen.blit(font.render(f"C{i + 1}", True, color), (sx + 8, sy - 16))

    banner = font.render(draft.instructions(), True, (0, 0, 0))
    pad = 6
    box = pygame.Surface((banner.get_width() + 2 * pad, banner.get_height() + 2 * pad))
    box.fill(color)
    box.blit(banner, (pad, pad))
    screen.blit(box, (10, 10))


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


def _ask_calibration_via_dialog() -> tuple[QuickMeasureCalibration | None, str]:
    """Tkinter file-picker flow to load a DLT2D calibration: either an
    existing `.dlt2d` coefficients file, or a pixel-calibration CSV +
    `.ref2d` reference pair (computed on the fly via `dlt2d.py`). Runs its
    own short-lived Tk root, same pattern as the rest of `getpixelvideo.py`'s
    Tk-based pickers.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    try:
        use_existing = messagebox.askyesno(
            "Quick Measure — Calibration",
            "Load an existing .dlt2d coefficients file?\n\n"
            "Yes: pick a .dlt2d file (from dlt2d.py).\n"
            "No: pick a pixel calibration CSV + a .ref2d file instead.",
        )
        if use_existing:
            dlt2d_file = filedialog.askopenfilename(
                title="Select .dlt2d coefficients file",
                filetypes=[("DLT2D files", "*.dlt2d"), ("CSV files", "*.csv")],
            )
            if not dlt2d_file:
                return None, "Calibration cancelled."
            calib = QuickMeasureCalibration.from_dlt2d_file(dlt2d_file)
            return calib, f"Calibration loaded: {os.path.basename(dlt2d_file)}"

        pixel_file = filedialog.askopenfilename(
            title="Select PIXEL calibration CSV",
            filetypes=[("CSV files", "*.csv")],
        )
        if not pixel_file:
            return None, "Calibration cancelled."
        ref_file = filedialog.askopenfilename(
            title="Select REF2D real-world coordinates file",
            filetypes=[("REF2D files", "*.ref2d"), ("CSV files", "*.csv")],
        )
        if not ref_file:
            return None, "Calibration cancelled."
        calib = QuickMeasureCalibration.from_calibration_points(pixel_file, ref_file)
        return calib, f"Calibration computed from: {os.path.basename(pixel_file)}"
    except QuickMeasureError as e:
        messagebox.showerror("Quick Measure — Calibration error", str(e))
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
        "QUICK MEASURE — classify current points",
        "",
        f"Points clicked: {n}   Unit: {session.unit_label}",
        f"Calibration: {calib_line}",
        "",
        "1: Distance     (last 2 points)",
        "2: Area         (all points, polygon)",
        "3: Velocity     (last 2 points, needs fps + 2 frames)",
        "4: Acceleration (last 3 points, needs fps + 3 frames)",
        "",
        "S: Save points / calibration / results CSV",
        "C: Load DLT2D calibration from file...",
        "X: Clear all points",
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
                        result_message = format_result(session.measure("velocity"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key in (pygame.K_4, pygame.K_KP4):
                    try:
                        result_message = format_result(session.measure("acceleration"))
                    except QuickMeasureError as e:
                        result_message = f"Error: {e}"
                elif event.key == pygame.K_c:
                    calib, msg = _ask_calibration_via_dialog()
                    if calib is not None:
                        session.calibration = calib
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
                    result_message = "Points cleared."

    return result_message or "Quick Measure menu closed (no measurement taken)."


if __name__ == "__main__":
    raise SystemExit(main())
