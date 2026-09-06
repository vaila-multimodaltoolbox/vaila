"""
================================================================================
Script: quickmeasure.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.122
Created: 06 September 2026
Last Updated: 06 September 2026
================================================================================
Description:
    Kinovea-style quick on-image measurements for `getpixelvideo.py`: click
    points on the video, then classify the current point set as Distance,
    Area, Velocity, or Acceleration.

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
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

try:
    from .dlt2d import process_files as dlt2d_process_files
    from .rec2d_one_dlt2d import rec2d
except ImportError:
    from dlt2d import process_files as dlt2d_process_files  # ty: ignore[unresolved-import]
    from rec2d_one_dlt2d import rec2d  # ty: ignore[unresolved-import]


MEASURE_TYPES: tuple[str, ...] = ("distance", "area", "velocity", "acceleration")


class QuickMeasureError(ValueError):
    """Raised for invalid/insufficient quick-measurement input.

    A plain ValueError subclass so callers that already catch ValueError
    keep working, while UI code can special-case this type if it wants to.
    """


@dataclass
class QuickMeasureCalibration:
    """A single fixed-camera DLT2D calibration used to convert pixel clicks
    to real-world units. Reuses `dlt2d.py` (to compute) and
    `rec2d_one_dlt2d.py`'s `rec2d()` (to apply) rather than reimplementing
    either.
    """

    dlt_params: np.ndarray
    unit_label: str = "m"
    source: str = ""

    def pixel_to_real(self, x: float, y: float) -> tuple[float, float]:
        out = rec2d(self.dlt_params, np.array([[x, y]], dtype=float))
        return float(out[0, 0]), float(out[0, 1])

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

    @property
    def unit_label(self) -> str:
        return self.calibration.unit_label if self.calibration else "px"

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
        return dispatch[kind]()


def format_result(result: dict) -> str:
    label = result["type"].capitalize()
    return f"{label}: {result['value']:.4f} {result['unit']}"


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
    screen, session: QuickMeasureSession, window_width, window_height
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
    lines = [
        "QUICK MEASURE — classify current points",
        "",
        f"Points clicked: {n}   Unit: {session.unit_label}"
        + ("  (calibrated)" if session.calibration else "  (pixels — no calibration)"),
        "",
        "1: Distance     (last 2 points)",
        "2: Area         (all points, polygon)",
        "3: Velocity     (last 2 points, needs fps + 2 frames)",
        "4: Acceleration (last 3 points, needs fps + 3 frames)",
        "",
        "C: Load DLT2D calibration...",
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
                elif event.key == pygame.K_x:
                    session.clear()
                    result_message = "Points cleared."

    return result_message or "Quick Measure menu closed (no measurement taken)."
