"""
Project: vailá Multimodal Toolbox
Script: pynalty.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 19 December 2025
Update Date: 08 November 2026
Version: 0.4.15

Description:
Guided analysis of a penalty kick from a single broadcast or handheld camera.
The pygame workflow uses a short 4-step macro flow (kick, goal/calibration,
response, save) with a large on-video action banner and one-click saving.
Optional AI tools (trajectory, pose, body measures) stay opt-in.
The interface supports Portuguese (default) and English independently of reports.
Reusable calibration is validated and previewed on the arrival frame.

Usage:
- GUI mode: click "Pynalty" inside the "Soccer Tools" launcher in the vailá
  main window (Frame B), or run ``uv run vaila/pynalty.py`` with no flags to
  open a file-picker dialog for the input video.
- CLI mode: ``uv run vaila/pynalty.py -i video.mp4 -o out_dir -c data.toml``
  skips the file dialog and preloads the output directory and saved marks.
- Headless re-render: ``uv run vaila/pynalty.py -i video.mp4 -c data.toml
  --report-only`` regenerates every artefact from a previous session without
  opening a window.

Outputs (inside ``<video_stem>_results/``):
- ``report.html`` / ``report_pt.html``  self-contained reports
- ``results.csv``                       tidy per-variable table
- ``pynalty_summary.csv``               one wide row for this penalty
- ``pynalty_database.csv``              appendable multi-penalty database
- ``ball_path_pixel.csv`` / ``ball_path_3d.csv``
- ``pose_kicker_pixel.csv`` / ``pose_gk_pixel.csv``
- ``pynalty_ball_path.mp4`` / ``pynalty_pose_overlay.mp4``
- ``data.toml``                         full reproducible state
- event snapshots as PNG

Requirements:
- Python 3.12, OpenCV, pygame, numpy, Tkinter
- Optional: ultralytics (automatic ball tracking), mediapipe (pose estimation)

License:
    This project is licensed under the terms of AGPLv3.0.
"""

from __future__ import annotations

import argparse
import contextlib
import copy
import os
import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog

import cv2
import numpy as np
import pygame

try:
    from .cli_highlight import print_gui_cli_mirror  # package import
    from .pynalty_analysis import (
        DEFAULT_GK_HEIGHT_M,
        Anthropometrics,
        BallPath,
        BallPathPoint,
        GoalGeometry,
        ball_path_speed_series,
        classify_shot_outcome,
        compute_penalty_metrics,
        distvelball_penalti,
        dlt2d,
        flight_speed_at,
        interpolate_ball_path,
        is_convex_quadrilateral,
        order_goal_corners,
        rec2d,
    )
    from .pynalty_report import (
        DATABASE_CSV,
        ReportContext,
        append_database,
        write_ball_path_csvs,
        write_html_reports,
        write_pose_csv,
        write_results_csv,
        write_summary_csv,
    )
except ImportError:  # standalone fallback
    from cli_highlight import print_gui_cli_mirror
    from pynalty_analysis import (
        DEFAULT_GK_HEIGHT_M,
        Anthropometrics,
        BallPath,
        BallPathPoint,
        GoalGeometry,
        ball_path_speed_series,
        classify_shot_outcome,
        compute_penalty_metrics,
        distvelball_penalti,
        dlt2d,
        flight_speed_at,
        interpolate_ball_path,
        is_convex_quadrilateral,
        order_goal_corners,
        rec2d,
    )
    from pynalty_report import (
        DATABASE_CSV,
        ReportContext,
        append_database,
        write_ball_path_csvs,
        write_html_reports,
        write_pose_csv,
        write_results_csv,
        write_summary_csv,
    )

try:
    import toml
except ImportError:
    try:
        import tomli as toml
    except ImportError:
        print(
            "Warning: 'toml' library not found. Saving/Loading might fail or require installation."
        )
        toml = None

__all__ = [
    "PynaltyApp",
    "PynaltyEvent",
    "build_parser",
    "distvelball_penalti",
    "dlt2d",
    "main",
    "rec2d",
]

# ==============================================================================
# Appearance
# ==============================================================================

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 80, 80)
GREEN = (100, 220, 120)
BLUE = (80, 180, 255)
YELLOW = (255, 220, 90)
GRAY = (150, 160, 175)
DARK_GRAY = (38, 44, 52)
PANEL_BG = (24, 28, 34)
PANEL_LINE = (58, 68, 80)
AMBER = (255, 183, 77)
VIOLET = (224, 64, 251)
CYAN = (0, 220, 220)

PANEL_W = 320
BOTTOM_H = 200

STEP_SPECS = (
    {
        "key": "kick",
        "name": "1. Ball contact",
        "why": "Starts the flight clock and fixes where both athletes were at contact.",
        "how": "Confirm foot-ball contact with ENTER, then click Ball and Keeper.",
        "required": True,
    },
    {
        "key": "goal",
        "name": "2. Ball at goal line",
        "why": "Ends the flight clock; same frame used for goal calibration and outcome.",
        "how": "Usually the last frame. Confirm, calibrate corners, mark Ball + hand, then G/D/M/W.",
        "required": True,
    },
    {
        "key": "calibration",
        "name": "2b. Goal calibration",
        "why": "Converts pixels into metres on the plane of the goal mouth.",
        "how": "On the arrival frame: click bottom-left, top-left, top-right, bottom-right.",
        "required": True,
    },
    {
        "key": "gk_move",
        "name": "3. Keeper starts moving",
        "why": "Anchors reaction time. Negative means the keeper guessed early.",
        "how": "Confirm the first clear dive/prepare motion with ENTER.",
        "required": True,
    },
    {
        "key": "ball_path",
        "name": "4. Ball path (optional)",
        "why": "Feeds the trajectory overlay and the animated replay in the report.",
        "how": "Optional. Press A for YOLO, or click the ball frame by frame.",
        "required": False,
    },
    {
        "key": "boxes",
        "name": "5. Pose boxes (optional)",
        "why": "Crops athletes so MediaPipe finds landmarks a full frame misses.",
        "how": "Optional. Drag KICKER then KEEPER boxes and press P.",
        "required": False,
    },
    {
        "key": "anthro",
        "name": "6. Body measures (optional)",
        "why": "Sets the keeper's reach. If skipped, elite defaults (1.88 m stature) are used.",
        "how": "Optional. Press B to type stature / arm span / standing reach.",
        "required": False,
    },
)

REQUIRED_STEP_KEYS = frozenset(s["key"] for s in STEP_SPECS if s["required"])

# Macro UI order: Kick → Goal/Calib → Response → Save
MACRO_STEPS = (
    {
        "id": 0,
        "name_pt": "1. Chute ⚽",
        "name_en": "1. Kick ⚽",
        "sub_pt": "Contato + Bola + Goleiro",
        "sub_en": "Contact + Ball + Keeper",
        "phases": (0, 1),
    },
    {
        "id": 1,
        "name_pt": "2. Gol 🥅",
        "name_en": "2. Goal 🥅",
        "sub_pt": "Calibração + Bola + Mão + Resultado",
        "sub_en": "Calib + Ball + Hand + Outcome",
        "phases": (2, 3, 4),
    },
    {
        "id": 2,
        "name_pt": "3. Resposta 🧤",
        "name_en": "3. Response 🧤",
        "sub_pt": "Início de movimento",
        "sub_en": "First movement",
        "phases": (5,),
    },
    {
        "id": 3,
        "name_pt": "4. Salvar 💾",
        "name_en": "4. Save 💾",
        "sub_pt": "Revisão e exportação",
        "sub_en": "Review and export",
        "phases": (6, 7, 8, 9),
    },
)

# Phase machine indices (keep optional tools before review).
PHASE_KICK_FRAME = 0
PHASE_KICK_POINTS = 1
PHASE_GOAL_FRAME = 2
PHASE_GOAL_POINTS = 3
PHASE_OUTCOME = 4
PHASE_RESPONSE = 5
PHASE_BALL_PATH = 6
PHASE_BOXES = 7
PHASE_ANTHRO = 8
PHASE_REVIEW = 9


def _banner(title: str, detail: str = "") -> None:
    """Boxed terminal banner. The ``>>`` prefix survives absl logging."""
    line = "=" * 72
    print()
    print(line)
    print(f">> vaila/pynalty: {title}")
    if detail:
        print(f">>   {detail}")
    print(line)


def _flush_message(screen, font, text: str) -> None:
    """Paint a status banner and flip the display before a slow operation."""
    if screen is None or font is None:
        return
    surf = font.render(text, True, BLACK)
    pad = 18
    box = pygame.Surface((surf.get_width() + pad * 2, surf.get_height() + pad * 2))
    box.fill(YELLOW)
    box.blit(surf, (pad, pad))
    cx = screen.get_width() // 2 - box.get_width() // 2
    cy = screen.get_height() // 2 - box.get_height() // 2
    screen.blit(box, (cx, cy))
    pygame.display.flip()
    pygame.event.pump()


def _detect_fps(video_path: str) -> float | None:
    """Best-effort precise FPS via the same ffprobe/OpenCV metadata logic as numberframes.py.

    Returns ``None`` on failure so callers can fall back to a cheaper source
    (e.g. the ``cv2.VideoCapture`` already open) without a spurious 30.0.
    """
    try:
        from .numberframes import get_video_info  # package import
    except ImportError:
        try:
            from numberframes import get_video_info  # standalone fallback
        except ImportError:
            return None
    try:
        info = get_video_info(video_path)
        fps = info.get("recommended_sampling_hz") or info.get("display_fps") or info.get("avg_fps")
        return float(fps) if fps and fps > 0 else None
    except Exception as exc:
        print(f"Warning: FPS auto-detection (ffprobe) failed: {exc}")
        return None


def _wrap(text: str, font, width: int) -> list[str]:
    """Greedy word wrap to a pixel width."""
    words, lines, current = text.split(), [], ""
    for w in words:
        probe = f"{current} {w}".strip()
        if font.size(probe)[0] <= width or not current:
            current = probe
        else:
            lines.append(current)
            current = w
    if current:
        lines.append(current)
    return lines


class _Button:
    """A flat clickable rectangle drawn on the pygame surface."""

    def __init__(self, key: str, label: str, primary: bool = False):
        self.key = key
        self.label = label
        self.primary = primary
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, screen, font, hover: bool, enabled: bool = True) -> None:
        if not enabled:
            bg, fg = (44, 50, 58), (110, 118, 128)
            border_col = PANEL_LINE
            border_w = 1
        elif self.primary:
            bg, fg = ((45, 140, 75), WHITE) if hover else ((32, 115, 60), WHITE)
            border_col = GREEN
            border_w = 2
        elif hover:
            bg, fg = (76, 130, 96), WHITE
            border_col = WHITE
            border_w = 1
        else:
            bg, fg = (52, 62, 74), WHITE
            border_col = PANEL_LINE
            border_w = 1
        pygame.draw.rect(screen, bg, self.rect, border_radius=5)
        pygame.draw.rect(screen, border_col, self.rect, border_w, border_radius=5)
        surf = font.render(self.label, True, fg)
        screen.blit(
            surf,
            (
                self.rect.centerx - surf.get_width() // 2,
                self.rect.centery - surf.get_height() // 2,
            ),
        )


class PynaltyEvent:
    """One step of the marking workflow, with its captured frame and points."""

    def __init__(self, name, instructions, key: str = "", why: str = ""):
        self.name = name
        self.instructions = instructions
        self.key = key
        self.why = why
        self.frame_idx = -1
        self.points: dict = {}
        self.is_done = False

    def reset(self):
        self.frame_idx = -1
        self.points = {}
        self.is_done = False


class PynaltyApp:
    """Interactive penalty marking session and its analysis outputs."""

    def __init__(
        self,
        video_path=None,
        *,
        goal: GoalGeometry | None = None,
        anthro: Anthropometrics | None = None,
        show_wizard: bool = True,
        lang: str = "both",
        database: str | None = None,
        ui_lang: str = "pt",
        calibration: str | None = None,
    ):
        self.video_path = video_path
        self.cap = None
        self.total_frames = 0
        self.fps = 30.0
        self.width = 800
        self.height = 600

        self.goal = goal or GoalGeometry()
        self.anthro = anthro or Anthropometrics()
        self.lang = lang
        self.database_path = database
        self.ui_lang = ui_lang
        self.calibration_path = calibration
        self.explicit_calibration_path = calibration
        self.phase = PHASE_KICK_FRAME
        self.calibration_stage = "idle"
        self.calibration_preview_edited = False
        self.calibration_draft = []
        self.calibration_record = None
        self.calibration_return = None
        self.calibration_backup = None
        self.output_dir_override: str | None = None
        self.fps_confirmed = False
        self.pending_calibration = None

        self.events: list[PynaltyEvent] = []
        self._init_events()
        self.current_event_idx = 0

        self.current_frame_idx = 0
        self.frame_img = None
        self.zoom = 1.0
        self.offset_x = PANEL_W
        self.offset_y = 0
        self.is_dragging = False
        self.last_mouse_pos = (0, 0)

        self.screen = None
        self.font = None
        self.font_big = None
        self.font_small = None
        self.display_size = (1280, 800)
        self.buttons: list[_Button] = []
        self.macro_step_rects: list[tuple[pygame.Rect, int]] = []
        self.quick_save_rect: pygame.Rect | None = None
        self.optional_tool_rects: list[tuple[pygame.Rect, int]] = []

        self.show_help = False
        self.show_wizard = False  # The active calibration prompt is the welcome screen.
        self.start_drag_slider = False
        self.playing = False
        self.box_drag_start: tuple[float, float] | None = None
        self.box_drag_current: tuple[float, float] | None = None

        self.feedback_msg = ""
        self.feedback_timer = 0
        self.last_results: dict = {}
        self.pose_sequences: dict = {}
        self.pose_metrics: dict = {}
        self.shot_outcome: str | None = None  # goal | save | miss | woodwork

        if self.video_path:
            self.load_video(self.video_path)

    PHASE_KEYS = (
        "kick",  # 0 frame
        "kick",  # 1 points
        "goal",  # 2 frame (+ calibration stages)
        "goal",  # 3 points Ball + hand
        "goal",  # 4 outcome
        "gk_move",  # 5 response
        "ball_path",  # 6 optional
        "boxes",  # 7 optional
        "anthro",  # 8 optional
        "review",  # 9 save
    )

    def tr(self, pt: str, en: str) -> str:
        return pt if self.ui_lang == "pt" else en

    @property
    def is_calibrating(self) -> bool:
        return self.calibration_stage in ("frame", "corners", "preview")

    @property
    def navigation_locked(self) -> bool:
        # Only lock scrubbing while placing goal corners / previewing TOML.
        return self.is_calibrating and self.calibration_stage != "frame"

    def set_phase(self, phase: int):
        self.phase = max(0, min(phase, len(self.PHASE_KEYS) - 1))
        key = self.PHASE_KEYS[self.phase]
        self.current_event_idx = next((i for i, e in enumerate(self.events) if e.key == key), -1)
        self.playing = False
        self.start_drag_slider = False
        if self.phase == PHASE_GOAL_FRAME and self.step("goal").frame_idx < 0 and self.total_frames:
            # Default arrival/calib frame is the last frame of the clip.
            self.seek(max(0, self.total_frames - 1), force=True)
        elif (
            (
                self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS, PHASE_OUTCOME)
                or self.phase in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE)
            )
            and self.current_event
            and self.current_event.frame_idx >= 0
        ):
            self.seek(self.current_event.frame_idx, force=True)

    def is_macro_step_done(self, idx: int) -> bool:
        if idx == 0:
            k = self.step("kick")
            return k.frame_idx != -1 and all(n in k.points for n in ("Ball", "GK"))
        if idx == 1:
            g = self.step("goal")
            return (
                g.frame_idx != -1
                and len(self.calib_points()) == 4
                and all(n in g.points for n in ("Ball", "GK"))
                and bool(self.shot_outcome)
            )
        if idx == 2:
            return self.step("gk_move").frame_idx != -1
        if idx == 3:
            return self.core_steps_done()
        return False

    def current_macro_step(self) -> int:
        if self.phase in (PHASE_KICK_FRAME, PHASE_KICK_POINTS):
            return 0
        if (
            self.phase in (PHASE_GOAL_FRAME, PHASE_GOAL_POINTS, PHASE_OUTCOME)
            or self.is_calibrating
        ):
            return 1
        if self.phase == PHASE_RESPONSE:
            return 2
        return 3

    def goto_macro_step(self, idx: int):
        """Navigate directly to the chosen macro step, resuming appropriate phase."""
        if idx == 0:
            k = self.step("kick")
            if k.frame_idx >= 0 and not all(n in k.points for n in ("Ball", "GK")):
                self.set_phase(PHASE_KICK_POINTS)
            elif k.frame_idx >= 0 and self.phase == PHASE_KICK_POINTS:
                self.set_phase(PHASE_KICK_FRAME)
            else:
                self.set_phase(PHASE_KICK_POINTS if k.frame_idx >= 0 else PHASE_KICK_FRAME)
        elif idx == 1:
            # Review goal/calib — never wipe corners just by clicking the card.
            g = self.step("goal")
            if g.frame_idx < 0:
                self.calibration_stage = "idle"
                self.set_phase(PHASE_GOAL_FRAME)
            elif len(self.calib_points()) != 4 and not getattr(self, "pending_calibration", None):
                self._begin_goal_calibration()
            elif not all(n in g.points for n in ("Ball", "GK")):
                self.calibration_stage = "idle"
                self.set_phase(PHASE_GOAL_POINTS)
            elif not self.shot_outcome:
                self.calibration_stage = "idle"
                self.set_phase(PHASE_OUTCOME)
            else:
                self.calibration_stage = "idle"
                self.set_phase(PHASE_GOAL_POINTS)
        elif idx == 2:
            self.calibration_stage = "idle"
            self.set_phase(PHASE_RESPONSE)
        elif idx == 3:
            self.calibration_stage = "idle"
            self.set_phase(PHASE_REVIEW)
            self.compute_metrics()

    def resume_workflow(self, optional_phase=None):
        """Resume at the first incomplete macro step in the new UI order."""
        self.calibration_stage = "idle"
        k = self.step("kick")
        if not 0 <= k.frame_idx < max(1, self.total_frames):
            self.set_phase(PHASE_KICK_FRAME)
            return
        if not all(n in k.points for n in ("Ball", "GK")):
            self.set_phase(PHASE_KICK_POINTS)
            return

        g = self.step("goal")
        if not 0 <= g.frame_idx < max(1, self.total_frames):
            self.set_phase(PHASE_GOAL_FRAME)
            return
        if g.frame_idx <= k.frame_idx:
            self.set_phase(PHASE_GOAL_FRAME)
            return
        if len(self.calib_points()) != 4:
            if getattr(self, "pending_calibration", None):
                self.calibration_stage = "preview"
                self.set_phase(PHASE_GOAL_FRAME)
                self.seek(g.frame_idx, force=True)
            else:
                self._begin_goal_calibration()
            return
        if not all(n in g.points for n in ("Ball", "GK")):
            self.set_phase(PHASE_GOAL_POINTS)
            return
        if not self.shot_outcome:
            self.set_phase(PHASE_OUTCOME)
            return

        if not 0 <= self.step("gk_move").frame_idx < max(1, self.total_frames):
            self.set_phase(PHASE_RESPONSE)
            return

        # Core done → land on Save (never auto-enter optional AI phases).
        if isinstance(optional_phase, int) and PHASE_BALL_PATH <= optional_phase <= PHASE_REVIEW:
            self.set_phase(optional_phase)
        else:
            self.set_phase(PHASE_REVIEW)

    def calibration_data(self):
        return {
            "format_version": 1,
            "width": self.width,
            "height": self.height,
            "points": copy.deepcopy(self.calibration_draft),
            "source_video": str(self.video_path or ""),
            "source_frame": self.current_frame_idx,
            "geometry": {
                "width": self.goal.width,
                "height": self.goal.height,
                "penalty_distance": self.goal.penalty_distance,
                "ball_radius": self.goal.ball_radius,
            },
        }

    def validate_calibration(self, data):
        if data.get("format_version") != 1:
            raise ValueError(
                self.tr("Versão de calibração desconhecida", "Unknown calibration version")
            )
        if (data.get("width"), data.get("height")) != (self.width, self.height):
            raise ValueError(
                self.tr(
                    "Resolução diferente; refaça a calibração", "Resolution differs; recalibrate"
                )
            )
        pts = np.asarray(data.get("points"), dtype=float)
        if (
            pts.shape != (4, 2)
            or not np.isfinite(pts).all()
            or (pts < 0).any()
            or (pts[:, 0] >= self.width).any()
            or (pts[:, 1] >= self.height).any()
            or not is_convex_quadrilateral(pts)
        ):
            raise ValueError(
                self.tr(
                    "Cantos inválidos: use quatro pontos convexos dentro da imagem",
                    "Invalid corners: use four convex points inside the image",
                )
            )
        geometry = GoalGeometry(**data["geometry"])
        values = [geometry.width, geometry.height, geometry.penalty_distance, geometry.ball_radius]
        if not np.isfinite(values).all() or min(values) <= 0:
            raise ValueError(self.tr("Geometria inválida", "Invalid geometry"))
        coeff = dlt2d(geometry.corner_coords(), pts)
        if not np.isfinite(coeff).all() or not np.allclose(
            rec2d(coeff, pts), geometry.corner_coords(), atol=1e-5
        ):
            raise ValueError(self.tr("Transformação inválida", "Invalid transformation"))
        return pts.tolist(), geometry

    def prepare_calibration(self, *, report_only=False):
        """Explicit > session > directory; report regeneration never discovers files."""
        path = self.explicit_calibration_path
        if not path and self.calib_points():
            data = self.calibration_record or self.calibration_data()
            data = copy.deepcopy(data)
            data["points"] = self.calib_points()
            try:
                self.validate_calibration(data)
            except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
                self.flash(str(exc), 240)
                self.step("calibration").reset()
                self.calibration_record = None
                self.calibration_draft = []
                self.calibration_stage = "idle"
                return False
            return True
        if not path and not report_only and self.video_path:
            candidate = Path(self.video_path).parent / "pynalty_calibration.toml"
            if candidate.exists():
                path = str(candidate)
        if not path:
            return True
        try:
            with open(path, encoding="utf-8") as fh:
                data = toml.load(fh)
            pts, geometry = self.validate_calibration(data)
        except (
            OSError,
            ValueError,
            TypeError,
            KeyError,
            AttributeError,
            np.linalg.LinAlgError,
        ) as exc:
            self.flash(self.tr("Calibração não aceita: ", "Calibration rejected: ") + str(exc), 240)
            print(self.feedback_msg)
            self.step("calibration").reset()
            self.calibration_record = None
            self.calibration_draft = []
            self.calibration_stage = "idle"
            return False
        self.calibration_path = str(path)
        if report_only:
            self.step("calibration").points = {"points": pts}
            self.step("calibration").frame_idx = -1
            self.goal, self.calibration_record = geometry, data
            self.compute_metrics()
        else:
            # Stage preview for the goal macro; do not steal focus from Kick.
            self.calibration_draft = pts
            self.calibration_preview_edited = False
            self.pending_calibration = data
            self.calibration_stage = "idle"
        return True

    def _begin_goal_calibration(self):
        """Start corner marking on the arrival frame (default: last frame)."""
        g = self.step("goal")
        frame = g.frame_idx if g.frame_idx >= 0 else max(0, self.total_frames - 1)
        if g.frame_idx < 0:
            g.frame_idx = frame
        self.set_phase(PHASE_GOAL_FRAME)
        self.seek(frame, force=True)
        if getattr(self, "pending_calibration", None) and self.calibration_draft:
            self.calibration_stage = "preview"
        else:
            self.calibration_draft = []
            self.calibration_preview_edited = False
            self.calibration_stage = "corners"

    def redo_calibration(self):
        if not self.is_calibrating:
            self.calibration_return = self.phase
            self.calibration_backup = (
                copy.deepcopy(self.step("calibration").points),
                self.step("calibration").frame_idx,
            )
        self.calibration_draft = []
        self.pending_calibration = None
        self.calibration_preview_edited = False
        g = self.step("goal")
        frame = g.frame_idx if g.frame_idx >= 0 else max(0, self.total_frames - 1)
        self.set_phase(PHASE_GOAL_FRAME)
        self.seek(frame, force=True)
        self.calibration_stage = "corners"

    def cancel_calibration(self):
        if self.calibration_return is not None:
            self.step("calibration").points, self.step("calibration").frame_idx = (
                self.calibration_backup
            )
            phase = self.calibration_return
            self.calibration_return = self.calibration_backup = None
            self.calibration_stage = "idle"
            self.set_phase(phase)
            self.flash(self.tr("Calibração anterior mantida", "Previous calibration retained"))

    def persist_calibration(self, path=None):
        destination = Path(
            path
            or self.calibration_path
            or Path(self.video_path or ".").resolve().parent / "pynalty_calibration.toml"
        )
        # Atomic replacement leaves an existing calibration intact on failure.
        import tempfile

        temporary = None
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", encoding="utf-8", dir=destination.parent, suffix=".toml", delete=False
            ) as fh:
                temporary = fh.name
                toml.dump(self.calibration_record, fh)
            os.replace(temporary, destination)
            self.calibration_path = str(destination)
            return True
        except (OSError, AttributeError) as exc:
            self.flash(
                self.tr(
                    "Calibração mantida na sessão. Escolha outro destino: ",
                    "Calibration kept in session. Choose another destination: ",
                )
                + str(exc),
                240,
            )
            return False
        finally:
            if temporary and os.path.exists(temporary):
                with contextlib.suppress(OSError):
                    os.unlink(temporary)

    def confirm_calibration(self):
        if self.calibration_stage == "frame":
            self.playing = False
            self.calibration_stage = "corners"
            return
        if len(self.calibration_draft) != 4:
            return
        pending = getattr(self, "pending_calibration", None)
        data = pending or self.calibration_data()
        data = copy.deepcopy(data)
        data["points"] = copy.deepcopy(self.calibration_draft)
        reused = bool(pending) and not self.calibration_preview_edited
        # Keep provenance when reusing an untouched TOML; otherwise stamp arrival frame.
        if not reused:
            data["source_frame"] = self.current_frame_idx
        try:
            pts, geometry = self.validate_calibration(data)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            self.flash(str(exc), 180)
            return
        self.step("calibration").points = {"points": pts}
        self.step("calibration").frame_idx = -1 if reused else self.current_frame_idx
        if self.step("goal").frame_idx < 0:
            self.step("goal").frame_idx = self.current_frame_idx
        self.goal, self.calibration_record = geometry, data
        self.pending_calibration = None
        self.calibration_stage = "idle"
        self.refresh_pose_metrics()
        if not reused and not self.persist_calibration():
            root = tk.Tk()
            root.withdraw()
            try:
                path = filedialog.asksaveasfilename(
                    title=self.tr("Salvar calibração em outro local", "Save calibration elsewhere"),
                    initialfile="pynalty_calibration.toml",
                    defaultextension=".toml",
                    parent=root,
                )
            finally:
                root.destroy()
            if path:
                self.persist_calibration(path)
        phase = self.calibration_return
        self.calibration_return = self.calibration_backup = None
        if phase is None:
            self.set_phase(PHASE_GOAL_POINTS)
        else:
            self.set_phase(phase)
        self.compute_metrics()
        self.flash(self.tr("Calibração confirmada", "Calibration confirmed"))

    def confirm_frame(self):
        if self.phase not in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE):
            return
        if self.is_calibrating and self.phase == PHASE_GOAL_FRAME:
            # Enter during calib preview/corners confirms calibration, not a new frame.
            if self.calibration_stage in ("corners", "preview"):
                self.confirm_calibration()
            return
        evt = self.current_event
        frame = self.current_frame_idx
        self.playing = False
        if self.phase == PHASE_GOAL_FRAME and frame <= self.step("kick").frame_idx:
            self.flash(
                self.tr("A chegada deve ser posterior ao chute", "Arrival must follow contact"), 120
            )
            return
        if evt.frame_idx != frame:
            evt.points = {}
            self.last_results = {}
            self.pose_sequences = {}
            self.pose_metrics = {}
            if evt.key in ("kick", "goal"):
                self.shot_outcome = None
                self.step("ball_path").reset()
                self.step("boxes").reset()
            if evt.key == "kick" and self.step("goal").frame_idx <= frame:
                self.step("goal").reset()
                self.step("calibration").reset()
                self.calibration_record = None
        evt.frame_idx = frame
        self.flash(self.tr(f"Frame {frame} confirmado", f"Frame {frame} confirmed"))
        if self.phase == PHASE_KICK_FRAME:
            self.set_phase(PHASE_KICK_POINTS)
        elif self.phase == PHASE_GOAL_FRAME:
            if len(self.calib_points()) == 4:
                self.calibration_stage = "idle"
                self.set_phase(PHASE_GOAL_POINTS)
            elif getattr(self, "pending_calibration", None):
                self.calibration_draft = list(
                    self.pending_calibration.get("points") or self.calibration_draft
                )
                self.calibration_stage = "preview"
            else:
                self.calibration_draft = []
                self.calibration_stage = "corners"
        elif self.phase == PHASE_RESPONSE:
            self.resume_workflow()

    # ------------------------------------------------------------------ setup

    def _init_events(self):
        self.events = [
            PynaltyEvent(
                spec["name"].split(". ", 1)[-1],
                spec["how"],
                key=spec["key"],
                why=spec["why"],
            )
            for i, spec in enumerate(STEP_SPECS)
        ]

    def step(self, key: str) -> PynaltyEvent:
        for evt in self.events:
            if evt.key == key:
                return evt
        raise KeyError(key)

    @property
    def current_event(self):
        if 0 <= self.current_event_idx < len(self.events):
            return self.events[self.current_event_idx]
        return None

    @property
    def current_key(self) -> str:
        evt = self.current_event
        return evt.key if evt else ""

    def load_video(self, path):
        """Open the clip and read its metadata. Does not touch pygame."""
        self.cap = cv2.VideoCapture(path)
        if not self.cap.isOpened():
            print(f"Error opening video: {path}")
            return False

        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        precise_fps = _detect_fps(path)
        cv_fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = precise_fps or (cv_fps if cv_fps and cv_fps > 0 else 30.0)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return True

    def _init_pygame(self):
        pygame.init()
        self.font = pygame.font.SysFont("Arial", 15)
        self.font_big = pygame.font.SysFont("Arial", 22)
        self.font_small = pygame.font.SysFont("Arial", 12)

        dis_w = min(self.width + PANEL_W, 1600)
        dis_h = min(self.height + BOTTOM_H, 900)
        self.display_size = (max(dis_w, 900), max(dis_h, 620))
        self.screen = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
        pygame.display.set_caption(f"vailá - Pynalty - {os.path.basename(self.video_path or '')}")
        self.fit_view()
        self.update_frame()

    # ------------------------------------------------------------------- view

    def content_rect(self) -> pygame.Rect:
        w, h = self.screen.get_size()
        return pygame.Rect(PANEL_W, 0, max(1, w - PANEL_W), max(1, h - BOTTOM_H))

    def fit_view(self):
        """Scale and centre the frame inside the video area."""
        if not self.screen or not self.width or not self.height:
            return
        area = self.content_rect()
        self.zoom = min(area.width / self.width, area.height / self.height)
        self.offset_x = area.x + (area.width - self.width * self.zoom) / 2
        self.offset_y = area.y + (area.height - self.height * self.zoom) / 2

    def zoom_at(self, factor: float, mx: int, my: int):
        """Zoom keeping the image point under the cursor fixed on screen."""
        old = self.zoom
        new = float(np.clip(old * factor, 0.05, 20.0))
        if new == old:
            return
        ix = (mx - self.offset_x) / old
        iy = (my - self.offset_y) / old
        self.zoom = new
        self.offset_x = mx - ix * new
        self.offset_y = my - iy * new

    def update_frame(self):
        if not self.cap:
            return False
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return False
        self._set_frame_image(frame)
        return True

    def _set_frame_image(self, frame: np.ndarray) -> None:
        """Convert one decoded OpenCV frame into the current pygame image."""
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_img = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

    def advance_playback(self) -> bool:
        """Decode the next frame sequentially, avoiding a costly seek per frame."""
        if not self.cap or self.current_frame_idx >= self.total_frames - 1:
            self.playing = False
            return False

        expected_frame = self.current_frame_idx + 1
        capture_position = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES)))
        if capture_position != expected_frame:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, expected_frame)

        ret, frame = self.cap.read()
        if not ret:
            self.playing = False
            return False

        decoded_frame = int(round(self.cap.get(cv2.CAP_PROP_POS_FRAMES))) - 1
        self.current_frame_idx = int(
            np.clip(decoded_frame, expected_frame, max(expected_frame, self.total_frames - 1))
        )
        self._set_frame_image(frame)
        return True

    def seek(self, frame_idx: int, *, force: bool = False):
        if self.navigation_locked and not force:
            return
        self.current_frame_idx = int(np.clip(frame_idx, 0, max(0, self.total_frames - 1)))
        self.update_frame()

    def screen_to_image_coords(self, sx, sy):
        return (sx - self.offset_x) / self.zoom, (sy - self.offset_y) / self.zoom

    def image_to_screen_coords(self, ix, iy):
        return int(ix * self.zoom + self.offset_x), int(iy * self.zoom + self.offset_y)

    def flash(self, message: str, frames: int = 45):
        self.feedback_msg = message
        self.feedback_timer = frames

    # ------------------------------------------------------------ step status

    def ball_path(self) -> BallPath:
        """The marked ball path as a :class:`BallPath`.

        Reads the parallel-array layout written by :meth:`set_ball_path` and
        also accepts a hand-written ``path = [[frame, x, y], ...]`` list.
        """
        evt = self.step("ball_path")
        frames = evt.points.get("path_frames") or []
        xs = evt.points.get("path_x") or []
        ys = evt.points.get("path_y") or []
        sources = evt.points.get("path_sources") or []

        if not frames:
            for i, item in enumerate(evt.points.get("path") or []):
                if len(item) < 3:
                    continue
                frames.append(item[0])
                xs.append(item[1])
                ys.append(item[2])
                if i >= len(sources):
                    sources.append("manual")

        pts = []
        for i, frame in enumerate(frames):
            if i >= len(xs) or i >= len(ys):
                break
            pts.append(
                BallPathPoint(
                    frame=int(frame),
                    x_px=float(xs[i]),
                    y_px=float(ys[i]),
                    source=sources[i] if i < len(sources) else "manual",
                )
            )
        return BallPath(points=pts)

    def set_ball_path(self, path: BallPath):
        """Persist the ball path as TOML-safe homogeneous parallel arrays."""
        evt = self.step("ball_path")
        ordered = path.sorted_points()
        evt.points.pop("path", None)
        evt.points["path_frames"] = [int(p.frame) for p in ordered]
        evt.points["path_x"] = [float(p.x_px) for p in ordered]
        evt.points["path_y"] = [float(p.y_px) for p in ordered]
        evt.points["path_sources"] = [str(p.source) for p in ordered]

    def calib_points(self) -> list:
        return self.step("calibration").points.get("points", []) or []

    def step_status(self, idx: int) -> tuple[bool, str]:
        """``(done, short readback)`` for the step at ``idx``."""
        evt = self.events[idx]
        key = evt.key
        if key == "gk_move":
            return evt.frame_idx != -1, (
                f"frame {evt.frame_idx}" if evt.frame_idx != -1 else "set frame (ENTER)"
            )
        if key == "kick":
            have = [n for n in ("Ball", "GK") if n in evt.points]
            done = evt.frame_idx != -1 and len(have) == 2
            if "Ball" not in evt.points:
                detail = "next: click BALL centre"
            elif "GK" not in evt.points:
                detail = f"frame {evt.frame_idx}, next: click KEEPER"
            else:
                detail = f"frame {evt.frame_idx}, ball+keeper"
            return done, detail
        if key == "goal":
            have = [n for n in ("Ball", "GK") if n in evt.points]
            outcome_ok = bool(self.shot_outcome)
            done = evt.frame_idx != -1 and len(have) == 2 and outcome_ok
            if "Ball" not in evt.points:
                detail = "next: click BALL centre"
            elif "GK" not in evt.points:
                detail = f"frame {evt.frame_idx}, next: click KEEPER"
            elif not outcome_ok:
                detail = "next: G=goal D=save M=miss W=woodwork"
            else:
                detail = f"frame {evt.frame_idx}, {self.shot_outcome}"
            return done, detail
        if key == "calibration":
            n = len(self.calib_points())
            return n == 4, f"{n}/4 corners"
        if key == "ball_path":
            n = len(self.ball_path().points)
            # Optional: empty counts as skippable/done for progress; filled needs 3+.
            if n == 0:
                return True, "skipped (optional)"
            return n >= 3, f"{n} points"
        if key == "boxes":
            have = [n for n in ("Kicker", "GK") if n in evt.points]
            if not have:
                return True, "skipped (optional)"
            return len(have) == 2, f"{'+'.join(have)}"
        if key == "anthro":
            if self.anthro.is_complete():
                r = self.anthro.resolved(use_defaults=False)
                return True, f"{r.gk_height_m:.2f} m (measured)"
            return True, f"defaults {DEFAULT_GK_HEIGHT_M:.2f} m"
        return evt.is_done, ""

    def step_is_required(self, idx: int) -> bool:
        return self.events[idx].key in REQUIRED_STEP_KEYS

    def can_leave_step(self, idx: int) -> bool:
        """Forward navigation is blocked until a required step is finished."""
        done, _ = self.step_status(idx)
        return done or not self.step_is_required(idx)

    def micro_prompt(self) -> str:
        if self.is_calibrating:
            if self.calibration_stage == "frame":
                return self.tr(
                    "Frame do gol/defesa (em geral o último). Enter inicia os 4 cantos.",
                    "Goal/save frame (usually the last). Enter starts the 4 corners.",
                )
            n = len(self.calibration_draft)
            if n < 4:
                corners = self.tr(
                    "inferior esquerdo|superior esquerdo|superior direito|inferior direito",
                    "bottom left|top left|top right|bottom right",
                ).split("|")
                return self.tr(
                    f"Clique no canto {n + 1}: {corners[n]}. Botão direito desfaz.",
                    f"Click corner {n + 1}: {corners[n]}. Right click undoes.",
                )
            return self.tr(
                "Confira enquadramento/zoom/câmera. Enter confirma; C refaz.",
                "Check framing/zoom/camera. Enter confirms; C restarts.",
            )
        prompts = {
            PHASE_KICK_FRAME: (
                "1. Contato pé-bola. Navegue e confirme com Enter.",
                "1. Foot-ball contact. Scrub and confirm with Enter.",
            ),
            PHASE_RESPONSE: (
                "3. Primeiro movimento do goleiro (tempo de resposta). Enter confirma.",
                "3. Keeper first movement (response time). Enter confirms.",
            ),
            PHASE_GOAL_FRAME: (
                "2. Frame da bola na linha / defesa (padrão: último). Enter confirma.",
                "2. Ball-at-line / save frame (default: last). Enter confirms.",
            ),
            PHASE_OUTCOME: (
                "Resultado: Gol (G), Defesa (D), Fora (M) ou Trave (W).",
                "Outcome: Goal (G), Save (D), Miss (M) or Woodwork (W).",
            ),
            PHASE_BALL_PATH: (
                "Opcional: A detecta a bola; clique para trajetória manual.",
                "Optional: A detects the ball; click for a manual trajectory.",
            ),
            PHASE_BOXES: (
                "Opcional: arraste caixas cobrador/goleiro; P executa pose.",
                "Optional: drag kicker/keeper boxes; P runs pose.",
            ),
            PHASE_ANTHRO: (
                "Opcional: B informa medidas corporais (ou use padrões).",
                "Optional: B enters body measurements (or use defaults).",
            ),
            PHASE_REVIEW: (
                "Revise e salve com S. Trajetória/pose/medidas são opcionais.",
                "Review and save with S. Trajectory/pose/body are optional.",
            ),
        }
        if self.phase == PHASE_KICK_POINTS:
            if all(k in self.current_event.points for k in ("Ball", "GK")):
                return self.tr(
                    "Pontos do chute ok. Continuar avança para o gol.",
                    "Kick points done. Continue advances to the goal.",
                )
            ball = "Ball" not in self.current_event.points
            return self.tr(
                "Clique no centro da BOLA." if ball else "Clique no centro do GOLEIRO.",
                "Click the BALL centre." if ball else "Click the KEEPER centre.",
            )
        if self.phase == PHASE_GOAL_POINTS:
            if all(k in self.current_event.points for k in ("Ball", "GK")):
                return self.tr(
                    "Pontos da chegada ok. Continuar escolhe o resultado.",
                    "Arrival points done. Continue chooses the outcome.",
                )
            ball = "Ball" not in self.current_event.points
            return self.tr(
                "Clique na BOLA na linha." if ball else "Clique na MÃO do goleiro.",
                "Click the BALL on the line." if ball else "Click the keeper HAND.",
            )
        return self.tr(*prompts[self.phase])

    def action_banner_text(self) -> str:
        """Large on-video instruction for the current action."""
        return self.micro_prompt()

    def core_steps_done(self) -> bool:
        """The required marking steps needed before metrics and save."""
        return all(
            self.step_status(i)[0]
            for i, evt in enumerate(self.events)
            if evt.key in REQUIRED_STEP_KEYS
        )

    def all_events_done(self) -> bool:
        return self.core_steps_done()

    def validation_warnings(self) -> list[str]:
        """Human-readable problems with the current marks."""
        warn: list[str] = []
        kick = self.step("kick")
        goal = self.step("goal")
        gk = self.step("gk_move")

        if kick.frame_idx != -1 and goal.frame_idx != -1 and goal.frame_idx <= kick.frame_idx:
            warn.append(
                self.tr(
                    "O frame da chegada deve ser posterior ao chute.",
                    "Goal frame must come after the contact frame.",
                )
            )
        if gk.frame_idx != -1 and kick.frame_idx != -1:
            dt = (gk.frame_idx - kick.frame_idx) / self.fps
            if dt < -1.0:
                warn.append(
                    self.tr(
                        f"Goleiro se move {abs(dt):.2f} s antes do chute: confira a Resposta.",
                        f"Keeper moves {abs(dt):.2f} s before contact - check Response.",
                    )
                )
        calib = self.calib_points()
        if len(calib) == 4 and not is_convex_quadrilateral(calib):
            warn.append(
                self.tr(
                    "Cantos não convexos. Pressione C para refazer a calibração.",
                    "Calibration corners are not convex. Press C to recalibrate.",
                )
            )
        if len(self.ball_path().points) in (1, 2):
            warn.append(
                self.tr(
                    "A trajetória precisa de 3 ou mais pontos. Use A na fase de trajetória.",
                    "Ball path needs 3+ points for a curve. Press A to auto-detect.",
                )
            )
        if not self.anthro.is_complete():
            warn.append(
                self.tr(
                    f"Estatura padrão do goleiro: {DEFAULT_GK_HEIGHT_M:.2f} m; edite na fase Medidas.",
                    f"Using default keeper stature {DEFAULT_GK_HEIGHT_M:.2f} m (press B to override).",
                )
            )
        if self.step("goal").frame_idx != -1 and not self.shot_outcome:
            warn.append(
                self.tr(
                    "Confirme o resultado: G=gol, D=defesa, M=fora, W=trave.",
                    "Set the shot result: G=goal, D=save, M=miss, W=woodwork.",
                )
            )
        return warn

    # --------------------------------------------------------------- drawing

    def draw_content(self):
        self.screen.fill(DARK_GRAY)
        area = self.content_rect()

        if self.frame_img:
            w = max(1, int(self.frame_img.get_width() * self.zoom))
            h = max(1, int(self.frame_img.get_height() * self.zoom))
            scaler = pygame.transform.scale if self.playing else pygame.transform.smoothscale
            scaled = scaler(self.frame_img, (w, h))
            self.screen.set_clip(area)
            self.screen.blit(scaled, (self.offset_x, self.offset_y))
            self.screen.set_clip(None)

        self.screen.set_clip(area)
        self.draw_markers()
        if not self.is_calibrating:
            self.draw_ball_path_overlay()
            self.draw_boxes()
            self.draw_goal_axes()
            self.draw_vectors()
        self.draw_action_banner()
        self.screen.set_clip(None)

        self.draw_step_panel()
        self.draw_bottom_bar()

        if self.feedback_timer > 0:
            self.feedback_timer -= 1
            self.draw_feedback()

        if self.show_wizard:
            self.draw_wizard()
        elif self.show_help:
            self.draw_help_overlay()

        if pygame.display.get_surface() is self.screen:
            pygame.display.flip()

    def draw_action_banner(self):
        """Prominent instruction strip over the video area."""
        area = self.content_rect()
        text = self.action_banner_text()
        if not text:
            return
        lines = _wrap(text, self.font_big, area.width - 48)
        box_h = len(lines) * 26 + 20
        box = pygame.Rect(area.x + 12, area.y + 12, area.width - 24, box_h)
        veil = pygame.Surface((box.width, box.height), pygame.SRCALPHA)
        veil.fill((10, 14, 20, 200))
        self.screen.blit(veil, box.topleft)
        pygame.draw.rect(self.screen, YELLOW, box, 2, border_radius=6)
        for i, line in enumerate(lines):
            surf = self.font_big.render(line, True, WHITE)
            self.screen.blit(surf, (box.x + 14, box.y + 10 + i * 26))

    def draw_marker(self, pos, color, label=""):
        sx, sy = self.image_to_screen_coords(pos[0], pos[1])
        size = 9
        pygame.draw.line(self.screen, color, (sx - size, sy), (sx + size, sy), 2)
        pygame.draw.line(self.screen, color, (sx, sy - size), (sx, sy + size), 2)
        pygame.draw.circle(self.screen, color, (sx, sy), 7, 2)
        if label:
            self.screen.blit(self.font_small.render(label, True, BLACK), (sx + 11, sy - 9))
            self.screen.blit(self.font_small.render(label, True, color), (sx + 10, sy - 10))

    def draw_markers(self):
        for key, suffix, color in (
            ("kick", self.tr("chute", "contact"), GREEN),
            ("goal", self.tr("chegada", "arrival"), RED),
        ):
            evt = self.step(key)
            if self.is_calibrating or evt.frame_idx != self.current_frame_idx:
                continue
            gk_label = self.tr("Mão", "Hand") if key == "goal" else self.tr("Goleiro", "Keeper")
            for name, label in (
                ("Ball", self.tr("Bola", "Ball")),
                ("GK", gk_label),
            ):
                if name in evt.points:
                    self.draw_marker(evt.points[name], color, f"{label}: {suffix}")
        points = self.calibration_draft if self.is_calibrating else self.calib_points()
        for i, point in enumerate(points):
            self.draw_marker(point, BLUE, f"C{i + 1}")

    def draw_ball_path_overlay(self):
        pts = self.ball_path().sorted_points()
        if not pts:
            return
        screen_pts = [self.image_to_screen_coords(p.x_px, p.y_px) for p in pts]
        if len(screen_pts) > 1:
            pygame.draw.lines(self.screen, CYAN, False, screen_pts, 2)
        for p, sp in zip(pts, screen_pts, strict=True):
            color = YELLOW if p.source == "manual" else CYAN
            radius = 5 if p.frame == self.current_frame_idx else 3
            pygame.draw.circle(self.screen, color, sp, radius)
            if p.frame == self.current_frame_idx:
                pygame.draw.circle(self.screen, WHITE, sp, radius + 4, 2)

    def draw_boxes(self):
        boxes = self.step("boxes").points
        for name, color in (("Kicker", GREEN), ("GK", AMBER)):
            box = boxes.get(name)
            if not box or len(box) != 4:
                continue
            x1, y1 = self.image_to_screen_coords(box[0], box[1])
            x2, y2 = self.image_to_screen_coords(box[2], box[3])
            rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            pygame.draw.rect(self.screen, color, rect, 2)
            self.screen.blit(self.font_small.render(name, True, color), (rect.x, rect.y - 15))

        if self.box_drag_start and self.box_drag_current:
            x1, y1 = self.image_to_screen_coords(*self.box_drag_start)
            x2, y2 = self.image_to_screen_coords(*self.box_drag_current)
            rect = pygame.Rect(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1))
            pygame.draw.rect(self.screen, WHITE, rect, 1)

        # Pose skeletons for the current frame, when they have been computed.
        for name, color in (("Kicker", GREEN), ("GK", AMBER)):
            seq = self.pose_sequences.get(name)
            if not seq or self.current_frame_idx not in getattr(seq, "landmarks", {}):
                continue
            pts = seq.landmarks[self.current_frame_idx]
            for k in range(pts.shape[0]):
                if np.isnan(pts[k, 0]) or np.isnan(pts[k, 1]):
                    continue
                pygame.draw.circle(
                    self.screen, color, self.image_to_screen_coords(pts[k, 0], pts[k, 1]), 2
                )

    def draw_vectors(self):
        kick, goal = self.step("kick"), self.step("goal")
        pairs = (
            ("Ball", CYAN, "dist"),
            ("GK", VIOLET, "gk_dist"),
        )
        for name, color, metric_key in pairs:
            a, b = kick.points.get(name), goal.points.get(name)
            if not a or not b:
                continue
            pa = self.image_to_screen_coords(a[0], a[1])
            pb = self.image_to_screen_coords(b[0], b[1])
            pygame.draw.line(self.screen, color, pa, pb, 2)
            if self.last_results:
                value = self.last_results.get(metric_key)
                if value is not None:
                    mid = ((pa[0] + pb[0]) // 2, (pa[1] + pb[1]) // 2)
                    lbl = self.font_small.render(f"{value:.2f} m", True, BLACK, color)
                    self.screen.blit(lbl, mid)

    def draw_goal_axes(self):
        pts = self.calib_points()
        if len(pts) != 4:
            return
        try:
            dlt_inv = dlt2d(np.asarray(pts, dtype=float), self.goal.corner_coords())
            projected = rec2d(dlt_inv, [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
            origin = self.image_to_screen_coords(projected[0, 0], projected[0, 1])
            ax = self.image_to_screen_coords(projected[1, 0], projected[1, 1])
            az = self.image_to_screen_coords(projected[2, 0], projected[2, 1])
            pygame.draw.line(self.screen, (255, 70, 70), origin, ax, 4)
            pygame.draw.line(self.screen, (90, 140, 255), origin, az, 4)
            self.screen.blit(self.font_small.render("X 1 m", True, (255, 70, 70)), ax)
            self.screen.blit(self.font_small.render("Z 1 m", True, (90, 140, 255)), az)
            pygame.draw.circle(self.screen, YELLOW, origin, 4)

            outline = [self.image_to_screen_coords(p[0], p[1]) for p in pts]
            pygame.draw.lines(self.screen, BLUE, True, outline, 1)
        except Exception:
            pass

    def draw_step_panel(self):
        h = self.screen.get_height() - BOTTOM_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, PANEL_W, h))
        pygame.draw.line(self.screen, PANEL_LINE, (PANEL_W - 1, 0), (PANEL_W - 1, h), 1)
        self.screen.set_clip(pygame.Rect(0, 0, PANEL_W, h))

        y = 10
        self.screen.blit(self.font_big.render("Pynalty", True, GREEN), (16, y))
        sub_title = self.tr("Análise de Pênalti", "Penalty Analysis")
        self.screen.blit(self.font_small.render(sub_title, True, GRAY), (110, y + 8))
        y += 34

        # Macro-Steps Stepper
        self.macro_step_rects = []
        curr_macro = self.current_macro_step()
        mouse = pygame.mouse.get_pos()

        for step in MACRO_STEPS:
            idx = step["id"]
            title = self.tr(step["name_pt"], step["name_en"])
            sub = self.tr(step["sub_pt"], step["sub_en"])
            is_curr = idx == curr_macro
            is_done = self.is_macro_step_done(idx)

            card_rect = pygame.Rect(10, y, PANEL_W - 20, 36)
            self.macro_step_rects.append((card_rect, idx))
            hover = card_rect.collidepoint(mouse)

            if is_curr:
                bg = (40, 52, 70) if not hover else (48, 62, 84)
                border_color = YELLOW
                border_w = 2
            elif is_done:
                bg = (28, 42, 34) if not hover else (34, 52, 42)
                border_color = (65, 140, 85)
                border_w = 1
            else:
                bg = (30, 35, 42) if not hover else (38, 44, 54)
                border_color = (52, 60, 72)
                border_w = 1

            pygame.draw.rect(self.screen, bg, card_rect, border_radius=5)
            pygame.draw.rect(self.screen, border_color, card_rect, border_w, border_radius=5)

            mark = "●" if is_curr else ("✓" if is_done else "○")
            mark_color = YELLOW if is_curr else (GREEN if is_done else GRAY)
            self.screen.blit(
                self.font_small.render(mark, True, mark_color),
                (card_rect.x + 8, card_rect.y + 11),
            )

            title_color = WHITE if (is_curr or is_done) else GRAY
            self.screen.blit(
                self.font.render(title, True, title_color),
                (card_rect.x + 24, card_rect.y + 4),
            )

            detail = sub
            if idx == 0:
                k = self.step("kick")
                if k.frame_idx >= 0:
                    pts = (
                        "+".join(k.points.keys())
                        if k.points
                        else self.tr("sem pontos", "no points")
                    )
                    detail = f"Frame {k.frame_idx} ({pts})"
            elif idx == 1:
                g = self.step("goal")
                if len(self.calib_points()) == 4 and g.frame_idx >= 0:
                    out = self.shot_outcome or self.tr("pendente", "pending")
                    detail = f"Frame {g.frame_idx} ({out})"
                elif len(self.calib_points()) == 4:
                    detail = self.tr("Calibrado (4 cantos)", "Calibrated (4 corners)")
                elif g.frame_idx >= 0:
                    detail = f"Frame {g.frame_idx}"
            elif idx == 2:
                f = self.step("gk_move").frame_idx
                if f >= 0:
                    detail = f"Frame {f}"
            elif idx == 3:
                detail = self.tr(
                    "Pronto para salvar" if self.core_steps_done() else "Aguardando passos",
                    "Ready to save" if self.core_steps_done() else "Pending steps",
                )

            self.screen.blit(
                self.font_small.render(detail, True, AMBER if is_curr else GRAY),
                (card_rect.x + 24, card_rect.y + 20),
            )
            y += 40

        y += 4
        pygame.draw.line(self.screen, PANEL_LINE, (16, y), (PANEL_W - 16, y), 1)
        y += 8

        # Active Phase Micro-Prompt Box
        prompt_lines = _wrap(self.micro_prompt(), self.font_small, PANEL_W - 36)
        prompt_box_h = max(38, len(prompt_lines) * 16 + 12)
        prompt_box = pygame.Rect(10, y, PANEL_W - 20, prompt_box_h)
        pygame.draw.rect(self.screen, (20, 24, 30), prompt_box, border_radius=4)
        pygame.draw.rect(self.screen, (45, 55, 68), prompt_box, 1, border_radius=4)
        for i, line in enumerate(prompt_lines):
            self.screen.blit(
                self.font_small.render(line, True, WHITE),
                (prompt_box.x + 8, prompt_box.y + 6 + i * 16),
            )
        y += prompt_box_h + 8

        # Live Metrics Card
        metrics = self.last_results or self.compute_metrics() or {}
        vel = metrics.get("vel_kmh")
        flight = metrics.get("flight_time_s")
        reaction = metrics.get("gk_response_time")
        outcome = self.shot_outcome

        card_h = 76
        m_box = pygame.Rect(10, y, PANEL_W - 20, card_h)
        pygame.draw.rect(self.screen, (18, 22, 28), m_box, border_radius=5)
        pygame.draw.rect(self.screen, (40, 52, 65), m_box, 1, border_radius=5)

        m_title = self.tr("MÉTRICAS EM TEMPO REAL", "LIVE METRICS")
        self.screen.blit(self.font_small.render(m_title, True, CYAN), (m_box.x + 8, m_box.y + 5))

        vel_txt = f"{vel:.1f} km/h" if isinstance(vel, int | float) else "— km/h"
        flight_txt = f"{flight:.3f} s" if isinstance(flight, int | float) else "— s"
        self.screen.blit(
            self.font_small.render(
                f"⚽ {self.tr('Bola', 'Ball')}: {vel_txt}  |  ⏱️ {self.tr('Voo', 'Flight')}: {flight_txt}",
                True,
                WHITE,
            ),
            (m_box.x + 8, m_box.y + 24),
        )

        react_txt = f"{reaction:+.3f} s" if isinstance(reaction, int | float) else "— s"
        out_txt = outcome.upper() if outcome else self.tr("Pendente", "Pending")
        out_col = (
            GREEN
            if outcome == "goal"
            else (
                BLUE
                if outcome == "save"
                else (RED if outcome == "miss" else (AMBER if outcome == "woodwork" else GRAY))
            )
        )
        self.screen.blit(
            self.font_small.render(
                f"🧤 {self.tr('Reação', 'Reaction')}: {react_txt}  |  ", True, WHITE
            ),
            (m_box.x + 8, m_box.y + 44),
        )
        r_w = self.font_small.size(f"🧤 {self.tr('Reação', 'Reaction')}: {react_txt}  |  ")[0]
        self.screen.blit(
            self.font_small.render(out_txt, True, out_col), (m_box.x + 8 + r_w, m_box.y + 44)
        )
        y += card_h + 8

        # Quick Save Button
        if self.core_steps_done():
            self.quick_save_rect = pygame.Rect(10, y, PANEL_W - 20, 32)
            qs_hover = self.quick_save_rect.collidepoint(mouse)
            qs_bg = (40, 130, 70) if not qs_hover else (55, 160, 85)
            pygame.draw.rect(self.screen, qs_bg, self.quick_save_rect, border_radius=5)
            pygame.draw.rect(self.screen, GREEN, self.quick_save_rect, 1, border_radius=5)
            qs_lbl = self.tr("💾 SALVAR RESULTADOS (S)", "💾 SAVE RESULTS (S)")
            qs_surf = self.font.render(qs_lbl, True, WHITE)
            self.screen.blit(
                qs_surf,
                (
                    self.quick_save_rect.centerx - qs_surf.get_width() // 2,
                    self.quick_save_rect.centery - qs_surf.get_height() // 2,
                ),
            )
            y += 38
        else:
            self.quick_save_rect = None

        # Optional Tools Row
        self.optional_tool_rects = []
        opt_w = (PANEL_W - 20 - 12) // 3
        tools = [
            (PHASE_BALL_PATH, self.tr("Traj A", "Path A")),
            (PHASE_BOXES, self.tr("Pose P", "Pose P")),
            (PHASE_ANTHRO, self.tr("Med B", "Body B")),
        ]
        for t_i, (t_phase, t_label) in enumerate(tools):
            t_rect = pygame.Rect(10 + t_i * (opt_w + 6), y, opt_w, 24)
            self.optional_tool_rects.append((t_rect, t_phase))
            t_hover = t_rect.collidepoint(mouse)
            t_active = self.phase == t_phase
            t_bg = (45, 60, 80) if t_active else ((40, 48, 58) if t_hover else (30, 35, 42))
            t_border = YELLOW if t_active else (PANEL_LINE if not t_hover else WHITE)
            pygame.draw.rect(self.screen, t_bg, t_rect, border_radius=4)
            pygame.draw.rect(self.screen, t_border, t_rect, 1, border_radius=4)
            t_surf = self.font_small.render(t_label, True, WHITE if (t_active or t_hover) else GRAY)
            self.screen.blit(
                t_surf,
                (
                    t_rect.centerx - t_surf.get_width() // 2,
                    t_rect.centery - t_surf.get_height() // 2,
                ),
            )
        y += 28

        # Goal calibration schematic while placing corners
        if self.is_calibrating:
            corners = [(48, y + 54), (48, y + 14), (250, y + 14), (250, y + 54)]
            pygame.draw.lines(self.screen, GRAY, True, corners, 2)
            for i, pt in enumerate(corners):
                color = YELLOW if i == len(self.calibration_draft) else BLUE
                pygame.draw.circle(self.screen, color, pt, 10)
                self.screen.blit(
                    self.font_small.render(str(i + 1), True, BLACK), (pt[0] - 4, pt[1] - 7)
                )
            y += 68

        # Validation warnings
        for warning in self.validation_warnings():
            for line in _wrap(warning, self.font_small, PANEL_W - 32):
                if y + 16 < h:
                    self.screen.blit(self.font_small.render(line, True, AMBER), (16, y))
                    y += 16

        self.screen.set_clip(None)

    def draw_bottom_bar(self):
        w, h = self.screen.get_size()
        top = h - BOTTOM_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, top, w, BOTTOM_H))
        pygame.draw.line(self.screen, PANEL_LINE, (0, top), (w, top), 1)

        margin = 20
        timeline_w = w - margin * 2
        event_y = top + 10
        event_h = 7
        slider_y = top + 27
        slider_h = 10
        progress_ratio = self.current_frame_idx / max(1, self.total_frames - 1)

        # Event timeline: confirmed keeper movement, kick and goal-line arrival.
        pygame.draw.rect(
            self.screen,
            (45, 52, 61),
            (margin, event_y, timeline_w, event_h),
            border_radius=3,
        )
        playhead = margin + int(progress_ratio * timeline_w)
        pygame.draw.line(
            self.screen,
            (235, 205, 70),
            (playhead, event_y - 2),
            (playhead, event_y + event_h + 2),
            1,
        )

        pygame.draw.rect(
            self.screen,
            (60, 70, 82),
            (margin, slider_y, timeline_w, slider_h),
            border_radius=5,
        )
        if playhead > margin:
            pygame.draw.rect(
                self.screen,
                (55, 135, 190),
                (margin, slider_y, playhead - margin, slider_h),
                border_radius=5,
            )

        if self.total_frames > 1:
            for evt, color in (
                (self.step("gk_move"), VIOLET),
                (self.step("kick"), GREEN),
                (self.step("goal"), RED),
            ):
                if evt.frame_idx != -1:
                    x = margin + int(evt.frame_idx / (self.total_frames - 1) * timeline_w)
                    pygame.draw.rect(self.screen, color, (x - 2, event_y - 2, 4, event_h + 4))
            pygame.draw.circle(self.screen, WHITE, (playhead, slider_y + slider_h // 2), 8)

        elapsed = self.current_frame_idx / max(self.fps, 1.0)
        duration = max(0, self.total_frames - 1) / max(self.fps, 1.0)
        info = (
            f"Frame {self.current_frame_idx}/{max(0, self.total_frames - 1)}  |  "
            f"{elapsed:.2f}/{duration:.2f} s  |  {self.fps:.2f} fps  |  "
            f"zoom {self.zoom:.2f}x"
        )
        self.screen.blit(self.font_small.render(info, True, GRAY), (margin, slider_y + 15))

        specs = [
            ("prev", self.tr("Voltar", "Back")),
            ("lang", "PT / EN"),
            ("calibrate", self.tr("Refazer calibração C", "Recalibrate C")),
            ("load", self.tr("Carregar L", "Load L")),
            ("help", self.tr("Ajuda H", "Help H")),
        ]
        primary_key = "confirm"
        if self.is_calibrating:
            label = self.tr("Confirmar frame Enter", "Confirm frame Enter")
            if self.calibration_stage == "corners" and len(self.calibration_draft) < 4:
                label = self.tr("Marque os 4 cantos", "Mark the 4 corners")
            elif len(self.calibration_draft) == 4:
                label = (
                    self.tr("Usar calibração Enter", "Use calibration Enter")
                    if getattr(self, "pending_calibration", None)
                    and not self.calibration_preview_edited
                    else self.tr("Confirmar e salvar Enter", "Confirm and save Enter")
                )
            specs.insert(0, ("confirm", label))
            if self.calibration_return is not None:
                specs.append(("cancel", self.tr("Cancelar Esc", "Cancel Esc")))
        elif self.phase in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE):
            specs.insert(0, ("confirm", self.tr("Confirmar frame Enter", "Confirm frame Enter")))
        elif self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS):
            specs.insert(0, ("edit", self.tr("Editar frame E", "Edit frame E")))
            if all(k in self.current_event.points for k in ("Ball", "GK")):
                primary_key = "next"
                specs.insert(0, ("next", self.tr("Continuar Enter", "Continue Enter")))
            else:
                primary_key = "edit"
        elif self.phase == PHASE_OUTCOME:
            primary_key = "goal"
            specs = [
                (k, self.tr(pt, en))
                for k, pt, en in (
                    ("goal", "Gol G", "Goal G"),
                    ("save_outcome", "Defesa D", "Save D"),
                    ("miss", "Fora M", "Miss M"),
                    ("woodwork", "Trave W", "Woodwork W"),
                )
            ] + specs
        elif self.phase in (PHASE_BALL_PATH, PHASE_BOXES, PHASE_ANTHRO):
            action = {
                PHASE_BALL_PATH: ("auto_ball", self.tr("Detectar bola A", "Detect ball A")),
                PHASE_BOXES: ("pose", self.tr("Executar pose P", "Run pose P")),
                PHASE_ANTHRO: ("anthro", self.tr("Medidas B", "Body B")),
            }[self.phase]
            primary_key = "next"
            specs = [
                ("next", self.tr("Voltar ao Salvar", "Back to Save")),
                action,
            ] + specs
        else:
            primary_key = "save"
            specs.insert(0, ("save", self.tr("Salvar resultados S", "Save results S")))
        if self.core_steps_done() and not any(k == "save" for k, _ in specs):
            specs.insert(0, ("save", self.tr("Salvar resultados S", "Save results S")))
            if primary_key not in ("goal", "confirm", "next"):
                primary_key = "save"
        cols = max(2, (w - 40) // 190)
        bw, bh, gap = (w - 40) // cols - 6, 29, 6
        self.buttons = [_Button(k, label, primary=(k == primary_key)) for k, label in specs]
        mouse = pygame.mouse.get_pos()
        for i, btn in enumerate(self.buttons):
            btn.rect = pygame.Rect(
                margin + (i % cols) * (bw + gap), top + 56 + (i // cols) * 34, bw, bh
            )
            btn.draw(self.screen, self.font_small, btn.rect.collidepoint(mouse))

    def draw_feedback(self):
        area = self.content_rect()
        lines = _wrap(self.feedback_msg, self.font_small, area.width - 32)
        box = pygame.Rect(area.x + 8, 8, area.width - 16, len(lines) * 18 + 16)
        pygame.draw.rect(self.screen, PANEL_BG, box, border_radius=5)
        for i, line in enumerate(lines):
            self.screen.blit(
                self.font_small.render(line, True, YELLOW), (box.x + 8, box.y + 8 + i * 18)
            )

    def draw_wizard(self):
        self.draw_help_overlay()

    def draw_help_overlay(self):
        w, h = self.screen.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 235))
        self.screen.blit(veil, (0, 0))
        text = self.tr(
            "Pynalty: 4 etapas (Chute → Gol/Calibração → Resposta → Salvar).\n1 Chute: confirme o contato e marque Bola + Goleiro.\n2 Gol: use o frame da chegada/defesa (em geral o último): 4 cantos + Bola + Mão + G/D/M/W.\n3 Resposta: frame do início de movimento do goleiro.\n4 Salvar: S exporta. A/P/B são opcionais e nunca bloqueiam.\nEnter confirma a ação do banner. Setas: ±1 / ±10 frames; Espaço: reproduzir/pausar.\nC: refazer calibração. Esc: cancelar recalibração ou sair. E: editar frame dos pontos.\nClique esquerdo: marcar; direito: desfazer. Timeline: marcadores coloridos pulam ao evento.\nF: FPS. F2 / PT-EN: idioma. H fecha a ajuda.",
            "Pynalty: 4 steps (Kick → Goal/Calibration → Response → Save).\n1 Kick: confirm contact and mark Ball + Keeper.\n2 Goal: arrival/save frame (usually last): 4 corners + Ball + Hand + G/D/M/W.\n3 Response: keeper first-movement frame.\n4 Save: S exports. A/P/B are optional and never block.\nEnter confirms the banner action. Arrows: ±1 / ±10 frames; Space: play/pause.\nC: recalibrate. Esc: cancel recalibration or quit. E: edit point frame.\nLeft click: mark; right: undo. Timeline colored markers snap to events.\nF: FPS. F2 / PT-EN: language. H closes help.",
        )
        y = 32
        for paragraph in text.split("\n"):
            for line in _wrap(paragraph, self.font, w - 64):
                self.screen.blit(self.font.render(line, True, WHITE), (32, y))
                y += 23
            y += 10

    def _text_input(self, prompt: str, initial: str = "") -> str | None:
        """Modal single-line text prompt drawn on the pygame surface."""
        text = initial
        clock = pygame.time.Clock()
        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return None
                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_RETURN, pygame.K_KP_ENTER):
                        return text
                    if event.key == pygame.K_ESCAPE:
                        return None
                    if event.key == pygame.K_BACKSPACE:
                        text = text[:-1]
                    elif event.unicode and event.unicode.isprintable():
                        text += event.unicode

            w, h = self.screen.get_size()
            veil = pygame.Surface((w, h), pygame.SRCALPHA)
            veil.fill((0, 0, 0, 200))
            self.screen.blit(veil, (0, 0))

            box = pygame.Rect(w // 2 - 300, h // 2 - 70, 600, 140)
            pygame.draw.rect(self.screen, PANEL_BG, box, border_radius=8)
            pygame.draw.rect(self.screen, GREEN, box, 2, border_radius=8)

            y = box.y + 18
            for line in _wrap(prompt, self.font, box.width - 36):
                self.screen.blit(self.font.render(line, True, WHITE), (box.x + 18, y))
                y += 20
            field = pygame.Rect(box.x + 18, box.bottom - 58, box.width - 36, 30)
            pygame.draw.rect(self.screen, (12, 15, 19), field, border_radius=4)
            self.screen.blit(self.font.render(f"{text}_", True, YELLOW), (field.x + 8, field.y + 6))
            self.screen.blit(
                self.font_small.render(
                    self.tr(
                        "Enter confirma, Esc cancela, vazio = pular",
                        "ENTER confirms, ESC cancels, empty = skip",
                    ),
                    True,
                    GRAY,
                ),
                (box.x + 18, box.bottom - 24),
            )
            pygame.display.flip()
            clock.tick(60)

    def ask_anthro(self):
        """Collect the body measurements that drive the reach model."""
        current = self.anthro.resolved()

        def ask(prompt: str, value) -> float | None:
            raw = self._text_input(prompt, f"{value:.2f}" if value else "")
            if raw is None:
                return None
            raw = raw.strip().replace(",", ".")
            if not raw:
                return 0.0
            try:
                return float(raw)
            except ValueError:
                self.flash(self.tr(f"Número inválido: {raw}", f"Not a number: {raw}"))
                return 0.0

        height = ask(
            self.tr(
                "Estatura do goleiro em metros (ex.: 1,88)",
                "Goalkeeper stature in metres (e.g. 1.88)",
            ),
            current.gk_height_m,
        )
        if height is None:
            return
        span = ask(
            self.tr(
                "Envergadura do goleiro em metros (vazio: 1,02 × estatura)",
                "Goalkeeper arm span in metres (leave empty for 1.02 x stature)",
            ),
            current.gk_arm_span_m,
        )
        if span is None:
            return
        reach = ask(
            self.tr(
                "Alcance vertical do goleiro em metros (vazio: 1,25 × estatura)",
                "Goalkeeper standing overhead reach in metres (leave empty for 1.25 x stature)",
            ),
            current.gk_standing_reach_m,
        )
        if reach is None:
            return
        kicker = ask(
            self.tr(
                "Estatura do cobrador em metros (opcional)", "Kicker stature in metres (optional)"
            ),
            current.kicker_height_m,
        )
        if kicker is None:
            return

        self.anthro = Anthropometrics(
            gk_height_m=height or None,
            gk_arm_span_m=span or None,
            gk_standing_reach_m=reach or None,
            kicker_height_m=kicker or None,
            dive_extension_factor=self.anthro.dive_extension_factor,
        )
        resolved = self.anthro.resolved()
        if resolved.gk_height_m:
            self.flash(
                self.tr(
                    f"Goleiro {resolved.gk_height_m:.2f} m, envergadura {resolved.gk_arm_span_m:.2f} m",
                    f"Keeper {resolved.gk_height_m:.2f} m, span {resolved.gk_arm_span_m:.2f} m",
                )
            )
        self.compute_metrics()

    def ask_fps(self):
        raw = self._text_input(
            self.tr("Taxa de quadros em Hz", "Frame rate in Hz"), f"{self.fps:.3f}"
        )
        if raw is None:
            return
        try:
            value = float(raw.strip().replace(",", "."))
        except ValueError:
            self.flash(self.tr("Número inválido", "Not a number"))
            return
        if value > 0:
            self.fps = value
            self.flash(
                self.tr(f"Taxa de quadros: {value:.3f} Hz", f"Frame rate set to {value:.3f} Hz")
            )
            self.compute_metrics()

    def confirm_fps_on_start(self):
        """Accept auto-detected FPS with a flash; press F anytime to edit.

        Velocity and time depend on FPS. ``self.fps`` already holds the
        ffprobe-based detection from ``load_video`` (same logic as
        numberframes.py). A blocking modal before the layout is avoided so the
        marking UI is immediately usable.
        """
        self.fps_confirmed = True
        self.flash(
            self.tr(
                f"FPS {self.fps:.3f} Hz — Enter inicia o chute; F altera",
                f"FPS {self.fps:.3f} Hz — Enter starts kick; F to change",
            ),
            180,
        )

    def flight_window(self, pad: int = 4) -> tuple[int, int]:
        """Frame range covering the whole event, with a little padding."""
        frames = [
            self.step(key).frame_idx
            for key in ("gk_move", "kick", "goal")
            if self.step(key).frame_idx >= 0
        ]
        if not frames:
            return 0, max(0, self.total_frames - 1)
        lo = max(0, min(frames) - pad)
        hi = min(max(0, self.total_frames - 1), max(frames) + pad)
        return lo, hi

    def auto_detect_ball(self):
        """Run YOLO over the flight window and merge with the manual marks."""
        kick, goal = self.step("kick"), self.step("goal")
        if kick.frame_idx == -1 or goal.frame_idx == -1:
            self.flash(
                self.tr(
                    "Confirme primeiro os frames de chute e chegada",
                    "Set the contact and goal frames first",
                )
            )
            return

        lo, hi = min(kick.frame_idx, goal.frame_idx), max(kick.frame_idx, goal.frame_idx)
        lo, hi = max(0, lo - 2), min(max(0, self.total_frames - 1), hi + 2)
        _banner("Automatic ball detection", f"frames {lo}-{hi} of {self.video_path}")
        _flush_message(
            self.screen,
            self.font_big,
            self.tr("Detectando a bola com YOLO ...", "Detecting the ball with YOLO ..."),
        )

        try:
            from .pynalty_vision import detect_ball_path
        except ImportError:
            from pynalty_vision import detect_ball_path

        detected = detect_ball_path(
            self.video_path,
            lo,
            hi,
            seed_px=tuple(kick.points["Ball"]) if "Ball" in kick.points else None,
            seed_frame=kick.frame_idx if "Ball" in kick.points else None,
        )
        if not detected.points:
            self.flash(
                self.tr(
                    "Bola não detectada: marque manualmente",
                    "No ball detections - mark the ball by hand",
                )
            )
            return

        # Manual marks always win over an automatic detection on the same frame.
        merged = {p.frame: p for p in detected.points}
        for p in self.ball_path().points:
            if p.source == "manual":
                merged[p.frame] = p
        self.set_ball_path(BallPath(points=list(merged.values())))
        self.flash(self.tr(f"Trajetória: {len(merged)} pontos", f"Ball path: {len(merged)} points"))

    def run_pose(self):
        """Run MediaPipe inside the drawn boxes over the flight window."""
        boxes = self.step("boxes").points
        if not boxes:
            self.flash(
                self.tr(
                    "Arraste primeiro uma caixa ao redor do atleta",
                    "Draw a bounding box in step 6 first",
                )
            )
            return
        kick, goal = self.step("kick"), self.step("goal")
        if kick.frame_idx == -1 or goal.frame_idx == -1:
            self.flash(
                self.tr(
                    "Confirme primeiro os frames de chute e chegada",
                    "Set the contact and goal frames first",
                )
            )
            return

        lo, hi = self.flight_window(pad=6)
        _banner("Pose estimation", f"frames {lo}-{hi}, boxes: {', '.join(boxes)}")

        try:
            from .pynalty_vision import pose_from_bbox
        except ImportError:
            from pynalty_vision import pose_from_bbox

        self.pose_sequences = {}
        for name in ("Kicker", "GK"):
            box = boxes.get(name)
            if not box or len(box) != 4:
                continue
            _flush_message(
                self.screen,
                self.font_big,
                self.tr(
                    f"Executando MediaPipe: {name} ...", f"Running MediaPipe on the {name} ..."
                ),
            )
            seq = pose_from_bbox(
                self.video_path,
                (box[0], box[1], box[2], box[3]),
                lo,
                hi,
                label=name,
            )
            if not seq.is_empty():
                self.pose_sequences[name] = seq

        self.refresh_pose_metrics()
        if self.pose_sequences:
            self.flash(self.tr("Pose calculada", "Pose ready"))
        else:
            self.flash(
                self.tr(
                    "Pose não detectada: confira as caixas",
                    "No pose landmarks found - check the boxes",
                )
            )

    def refresh_pose_metrics(self):
        self.pose_metrics = {}
        if not self.pose_sequences:
            return
        try:
            from .pynalty_vision import gk_kinematics, kicker_kinematics
        except ImportError:
            from pynalty_vision import gk_kinematics, kicker_kinematics
        kick, goal = self.step("kick"), self.step("goal")
        metrics: dict = {}
        if "Kicker" in self.pose_sequences:
            metrics.update(
                kicker_kinematics(self.pose_sequences["Kicker"], kick.frame_idx, self.fps)
            )
        if "GK" in self.pose_sequences:
            calib = self.calib_points()
            metrics.update(
                gk_kinematics(
                    self.pose_sequences["GK"],
                    kick.frame_idx,
                    goal.frame_idx,
                    self.fps,
                    calib_pixels=calib if len(calib) == 4 else None,
                    goal=self.goal,
                )
            )
        self.pose_metrics = metrics

    def reorder_calibration(self):
        pts = self.calib_points()
        if len(pts) != 4:
            self.flash(self.tr("Marque os quatro cantos primeiro", "Mark all four corners first"))
            return
        ordered = order_goal_corners(pts)
        self.step("calibration").points["points"] = [[float(p[0]), float(p[1])] for p in ordered]
        self.flash(
            self.tr(
                "Cantos ordenados: inferior esquerdo, superior esquerdo, superior direito, inferior direito",
                "Corners reordered: bottom-left, top-left, top-right, bottom-right",
            )
        )
        self.compute_metrics()

    # ---------------------------------------------------------------- metrics

    def compute_metrics(self) -> dict:
        """Recompute the metric set, returning it (empty when marks are missing)."""
        if not self.core_steps_done():
            # Allow metrics once Ball+GK+calib+gk_move exist even before outcome,
            # so the panel can show speeds while waiting for G/D/M/W.
            kick, goal, gk_move = self.step("kick"), self.step("goal"), self.step("gk_move")
            soft_ok = (
                gk_move.frame_idx != -1
                and kick.frame_idx != -1
                and goal.frame_idx != -1
                and "Ball" in kick.points
                and "GK" in kick.points
                and "Ball" in goal.points
                and "GK" in goal.points
                and len(self.calib_points()) == 4
            )
            if not soft_ok:
                self.last_results = {}
                return {}

        kick, goal, gk_move = self.step("kick"), self.step("goal"), self.step("gk_move")
        calib = self.calib_points()

        body_pts = None
        body_names = None
        defend_hint = None
        seq = self.pose_sequences.get("GK")
        if seq is not None and "Ball" in goal.points:
            try:
                from .pynalty_vision import LANDMARK_NAMES, nearest_landmark_to_ball
            except ImportError:
                from pynalty_vision import LANDMARK_NAMES, nearest_landmark_to_ball

            near = nearest_landmark_to_ball(seq, goal.frame_idx, goal.points["Ball"])
            if near:
                body_pts = near["all_xy"]
                body_names = list(LANDMARK_NAMES)
                defend_hint = near.get("landmark_name")

        try:
            results = compute_penalty_metrics(
                calib_pixels=calib,
                kick_frame=kick.frame_idx,
                goal_frame=goal.frame_idx,
                gk_move_frame=gk_move.frame_idx,
                kick_ball_px=kick.points.get("Ball"),
                kick_gk_px=kick.points["GK"],
                goal_ball_px=goal.points["Ball"],
                goal_gk_px=goal.points["GK"],
                fps=self.fps,
                goal=self.goal,
                anthro=self.anthro,
                shot_outcome=self.shot_outcome,
                gk_body_points_px=body_pts,
                gk_body_landmark_names=body_names,
                defending_landmark_hint=defend_hint,
            )
        except Exception as exc:
            print(f">> vaila/pynalty: could not compute metrics: {exc}")
            self.last_results = {}
            return {}

        results["kick_ball_px"] = kick.points.get("Ball")
        results["kick_gk_px"] = kick.points.get("GK")
        results["goal_ball_px"] = goal.points.get("Ball")
        results["goal_gk_px"] = goal.points.get("GK")
        results["calib_pixels"] = calib
        self.last_results = results
        return results

    def calculate_and_show_results(self, y_start=100):
        """Backward-compatible hook: recompute and surface any failure on screen."""
        results = self.compute_metrics()
        if not results and self.screen is not None:
            self.screen.blit(
                self.font.render(
                    self.tr(
                        "Medidas indisponíveis: confira calibração, frames e pontos",
                        "Metrics unavailable: check steps 1-4",
                    ),
                    True,
                    RED,
                ),
                (PANEL_W + 20, y_start),
            )
        return results

    # ------------------------------------------------------------ persistence

    def to_data(self) -> dict:
        """Serialise the whole session for TOML."""
        resolved = self.anthro
        data: dict = {
            "video": self.video_path,
            "fps": self.fps,
            "shot_outcome": self.shot_outcome or "",
            "goal": {
                "width_m": self.goal.width,
                "height_m": self.goal.height,
                "penalty_distance_m": self.goal.penalty_distance,
                "ball_radius_m": self.goal.ball_radius,
            },
            "anthropometrics": {
                k: v
                for k, v in (
                    ("gk_height_m", resolved.gk_height_m),
                    ("gk_arm_span_m", resolved.gk_arm_span_m),
                    ("gk_standing_reach_m", resolved.gk_standing_reach_m),
                    ("kicker_height_m", resolved.kicker_height_m),
                    ("dive_extension_factor", resolved.dive_extension_factor),
                )
                if v is not None
            },
            "events": [
                {"name": e.name, "key": e.key, "frame_idx": e.frame_idx, "points": e.points}
                for e in self.events
            ],
        }
        if self.calibration_record:
            data["calibration_record"] = self.calibration_record
        data["workflow_phase"] = self.phase
        if self.last_results:
            data["results"] = {
                k: v
                for k, v in self.last_results.items()
                if not k.startswith("_") and isinstance(v, int | float | str | bool)
            }
        return data

    def load_from_data(self, data: dict) -> bool:
        """Restore a session from a parsed TOML mapping.

        Handles the current format (a list of ``[[events]]`` tables, matched by
        ``key`` when present so a reordered workflow still loads), and the
        legacy flat keys written by the first Pynalty release.
        """
        if not isinstance(data, dict):
            return False

        self._init_events()
        self.shot_outcome = None
        self.pose_sequences = {}
        self.pose_metrics = {}
        self.last_results = {}
        self.calibration_stage = "idle"
        self.calibration_preview_edited = False
        self.calibration_draft = []
        self.pending_calibration = None
        self.calibration_return = self.calibration_backup = None
        if data.get("fps"):
            with contextlib.suppress(TypeError, ValueError):
                self.fps = float(data["fps"])

        raw_outcome = data.get("shot_outcome") or (data.get("results") or {}).get("shot_outcome")
        if raw_outcome:
            self.shot_outcome = str(raw_outcome).strip().lower() or None

        g = data.get("goal") or {}
        if g:
            self.goal = GoalGeometry(
                width=float(g.get("width_m", self.goal.width)),
                height=float(g.get("height_m", self.goal.height)),
                penalty_distance=float(g.get("penalty_distance_m", self.goal.penalty_distance)),
                ball_radius=float(g.get("ball_radius_m", self.goal.ball_radius)),
            )

        a = data.get("anthropometrics") or {}
        if a:
            self.anthro = Anthropometrics(
                gk_height_m=a.get("gk_height_m"),
                gk_arm_span_m=a.get("gk_arm_span_m"),
                gk_standing_reach_m=a.get("gk_standing_reach_m"),
                kicker_height_m=a.get("kicker_height_m"),
                dive_extension_factor=float(
                    a.get("dive_extension_factor", self.anthro.dive_extension_factor)
                ),
            )

        events_data = data.get("events") or []
        if events_data:
            by_key = {e.key: e for e in self.events}
            for i, entry in enumerate(events_data):
                key = entry.get("key")
                target = by_key.get(key) if key else None
                if target is None and not key and i < len(self.events):
                    target = self.events[i]
                if target is None:
                    continue
                target.frame_idx = int(entry.get("frame_idx", -1))
                target.points = copy.deepcopy(entry.get("points") or {})
            print(">> vaila/pynalty: loaded marks (current format)")
        else:
            print(">> vaila/pynalty: loaded marks (legacy format)")
            self.step("gk_move").frame_idx = int(data.get("gk_move_frame", -1))
            kick = self.step("kick")
            kick.frame_idx = int(data.get("kick_frame", -1))
            if data.get("kick_ball_pixel"):
                kick.points["Ball"] = list(data["kick_ball_pixel"])
            if data.get("kick_gk_pixel"):
                kick.points["GK"] = list(data["kick_gk_pixel"])
            goal = self.step("goal")
            goal.frame_idx = int(data.get("goal_frame", -1))
            if data.get("goal_ball_pixel"):
                goal.points["Ball"] = list(data["goal_ball_pixel"])
            if data.get("goal_gk_pixel"):
                goal.points["GK"] = list(data["goal_gk_pixel"])
            if data.get("calibration_pixels"):
                self.step("calibration").points["points"] = [
                    list(p) for p in data["calibration_pixels"]
                ]

        self.calibration_record = data.get("calibration_record")
        self.resume_workflow(optional_phase=data.get("workflow_phase"))
        self.compute_metrics()
        if self.screen is not None:
            self.update_frame()
        return True

    def load_toml(self):
        """Pick a TOML file and restore the session from it."""
        if not toml:
            print("TOML library not available.")
            return
        root = tk.Tk()
        root.withdraw()
        file_path = filedialog.askopenfilename(
            title=self.tr("Carregar sessão", "Load session"), filetypes=[("TOML", "*.toml")]
        )
        root.destroy()
        if not file_path:
            return
        try:
            with open(file_path) as fh:
                data = toml.load(fh)
            self.load_from_data(data)
            self.flash(
                self.tr(
                    f"Carregado: {os.path.basename(file_path)}",
                    f"Loaded {os.path.basename(file_path)}",
                )
            )
            return True
        except Exception as exc:
            print(f"Error loading: {exc}")
            self.flash(self.tr("Falha ao carregar", "Load failed"))

    # ----------------------------------------------------------------- output

    def get_results_dir(self):
        """Timestamp-free per-video results folder, created on demand."""
        if not self.video_path:
            return None
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        dir_name = f"{base_name}_results"
        parent = self.output_dir_override or os.path.dirname(self.video_path)
        target_dir = os.path.join(parent, dir_name)
        os.makedirs(target_dir, exist_ok=True)
        return target_dir

    def save_snapshot(self, frame_idx, filename, out_dir, overlay_func=None):
        if not self.cap or frame_idx is None or frame_idx < 0:
            return None
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return None
        if overlay_func:
            frame = overlay_func(frame)
        path = os.path.join(out_dir, filename)
        cv2.imwrite(path, frame)
        return path

    def _snapshot_overlays(self):
        """OpenCV overlay callables for the event snapshots."""
        kick, goal, calib = self.step("kick"), self.step("goal"), self.step("calibration")

        def dot(img, p, color, label):
            cv2.circle(img, (int(p[0]), int(p[1])), 8, color, -1)
            cv2.putText(
                img,
                label,
                (int(p[0]) + 12, int(p[1])),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                color,
                2,
            )

        def kick_overlay(img):
            if "Ball" in kick.points:
                dot(img, kick.points["Ball"], (0, 255, 0), "Ball @ contact")
            if "GK" in kick.points:
                dot(img, kick.points["GK"], (255, 255, 0), "GK @ contact")
            for name, color in (("Kicker", (0, 255, 0)), ("GK", (0, 180, 255))):
                box = self.step("boxes").points.get(name)
                if box and len(box) == 4:
                    cv2.rectangle(
                        img,
                        (int(box[0]), int(box[1])),
                        (int(box[2]), int(box[3])),
                        color,
                        2,
                    )
                    cv2.putText(
                        img,
                        f"{name} region",
                        (int(box[0]), max(16, int(box[1]) - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        color,
                        2,
                    )
            return img

        def goal_overlay(img):
            if "Ball" in goal.points:
                dot(img, goal.points["Ball"], (0, 0, 255), "Ball @ line")
            if "GK" in goal.points:
                dot(img, goal.points["GK"], (255, 0, 255), "Hand @ line")
            k_ball, g_ball = kick.points.get("Ball"), goal.points.get("Ball")
            if k_ball and g_ball:
                pt1 = (int(k_ball[0]), int(k_ball[1]))
                pt2 = (int(g_ball[0]), int(g_ball[1]))
                cv2.line(img, pt1, pt2, (255, 255, 0), 2)
                if self.last_results:
                    mid = ((pt1[0] + pt2[0]) // 2, (pt1[1] + pt2[1]) // 2)
                    cv2.putText(
                        img,
                        f"{self.last_results.get('dist', 0):.2f} m",
                        mid,
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 255),
                        2,
                    )
            k_gk, g_gk = kick.points.get("GK"), goal.points.get("GK")
            if k_gk and g_gk:
                cv2.line(
                    img,
                    (int(k_gk[0]), int(k_gk[1])),
                    (int(g_gk[0]), int(g_gk[1])),
                    (255, 0, 255),
                    2,
                )
            return img

        def calib_overlay(img):
            pts = calib.points.get("points", []) or []
            for i, p in enumerate(pts):
                cv2.circle(img, (int(p[0]), int(p[1])), 6, (255, 200, 100), -1)
                cv2.putText(
                    img,
                    f"C{i + 1}",
                    (int(p[0]) + 10, int(p[1])),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 200, 100),
                    2,
                )
            if len(pts) == 4:
                cnt = np.array(pts, dtype=np.int32).reshape((-1, 1, 2))
                cv2.polylines(img, [cnt], True, (255, 200, 100), 2)
                cv2.putText(
                    img,
                    "Goal region",
                    (int(min(p[0] for p in pts)), max(16, int(min(p[1] for p in pts)) - 10)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 200, 100),
                    2,
                )
            return img

        return kick_overlay, goal_overlay, calib_overlay

    def _overlay_regions(self):
        """Labelled static regions for the overlay video and composite image."""
        regions = []
        pts = self.calib_points()
        if len(pts) == 4:
            xs = [p[0] for p in pts]
            ys = [p[1] for p in pts]
            regions.append(((min(xs), min(ys), max(xs), max(ys)), "Goal region", (255, 200, 100)))
        box = self.step("boxes").points.get("Kicker")
        if box and len(box) == 4:
            regions.append(((box[0], box[1], box[2], box[3]), "Kick region", (0, 255, 0)))
        return regions

    def _attach_model_speed(self, ball_rows: list[dict], metrics: dict) -> None:
        """Add the flight-model speed to each ball-path row, in place.

        The pixel path cannot be differentiated into metres per second off the
        goal plane, so the speed shown alongside the trajectory comes from the
        fitted 3D flight instead.
        """
        flight = metrics.get("_flight_path")
        kick_frame = metrics.get("kick_frame")
        if flight is None or kick_frame is None or self.fps <= 0 or not ball_rows:
            return
        times = [(r["frame"] - kick_frame) / self.fps for r in ball_rows]
        speeds = flight_speed_at(flight, times)
        for row, t_rel, speed in zip(ball_rows, times, speeds, strict=True):
            row["flight_time_s"] = t_rel
            row["model_speed_ms"] = float(speed)

    def save_results_package(self) -> bool:
        """Write every artefact for this penalty. Returns False on failure."""
        out_dir = self.get_results_dir()
        if not out_dir:
            print(">> vaila/pynalty: no output directory (is a video loaded?)")
            return False

        metrics = self.compute_metrics()
        if not metrics:
            print(">> vaila/pynalty: steps 1-4 must be complete before saving")
            return False
        if not self.shot_outcome:
            # Infer once from geometry so headless --report-only still works.
            inside = bool(metrics.get("ball_inside_goal"))
            self.shot_outcome, _ = classify_shot_outcome(None, ball_inside_goal=inside)
            metrics = self.compute_metrics() or metrics
            print(f">> vaila/pynalty: shot_outcome inferred as {self.shot_outcome}")

        _banner("Saving results", out_dir)
        _flush_message(
            self.screen,
            self.font_big,
            self.tr("Salvando imagens e relatórios ...", "Saving snapshots and reports ..."),
        )

        kick, goal, gk_move, calib = (
            self.step("kick"),
            self.step("goal"),
            self.step("gk_move"),
            self.step("calibration"),
        )
        kick_overlay, goal_overlay, calib_overlay = self._snapshot_overlays()
        regions = self._overlay_regions()

        snapshots: dict[str, str] = {}
        for key, frame_idx, filename, overlay in (
            ("snapshot_gk_move", gk_move.frame_idx, "snapshot_gk_move.png", None),
            ("snapshot_kick", kick.frame_idx, "snapshot_kick.png", kick_overlay),
            ("snapshot_goal", goal.frame_idx, "snapshot_goal.png", goal_overlay),
            (
                "snapshot_calibration",
                calib.frame_idx if calib.frame_idx != -1 else goal.frame_idx,
                "snapshot_calibration.png",
                calib_overlay,
            ),
        ):
            path = self.save_snapshot(frame_idx, filename, out_dir, overlay)
            if path:
                snapshots[key] = path

        # --- ball path -----------------------------------------------------
        ball_rows: list[dict] = []
        raw_path = self.ball_path()
        if raw_path.points:
            filled = interpolate_ball_path(raw_path, *self.flight_window(pad=0))
            calib_pixels = self.calib_points() if len(self.calib_points()) == 4 else None
            ball_rows = ball_path_speed_series(filled, calib_pixels, self.fps, self.goal)
            self._attach_model_speed(ball_rows, metrics)

        videos: dict[str, str] = {}
        if ball_rows or self.pose_sequences:
            try:
                from .pynalty_vision import write_ball_path_composite, write_overlay_video
            except ImportError:
                from pynalty_vision import write_ball_path_composite, write_overlay_video

            lo, hi = self.flight_window(pad=4)
            if ball_rows:
                _flush_message(
                    self.screen,
                    self.font_big,
                    self.tr("Gerando vídeo da trajetória ...", "Rendering the ball path video ..."),
                )
                path = write_overlay_video(
                    self.video_path,
                    os.path.join(out_dir, "pynalty_ball_path.mp4"),
                    lo,
                    hi,
                    ball_rows=ball_rows,
                    regions=regions,
                    fps=self.fps,
                )
                if path:
                    videos["ball_path"] = path
                composite = write_ball_path_composite(
                    self.video_path,
                    os.path.join(out_dir, "ball_path_composite.png"),
                    goal.frame_idx,
                    ball_rows,
                    regions=regions,
                )
                if composite:
                    snapshots["ball_path_composite"] = composite

            if self.pose_sequences:
                _flush_message(
                    self.screen,
                    self.font_big,
                    self.tr("Gerando vídeo com pose ...", "Rendering the pose overlay video ..."),
                )
                path = write_overlay_video(
                    self.video_path,
                    os.path.join(out_dir, "pynalty_pose_overlay.mp4"),
                    lo,
                    hi,
                    poses=list(self.pose_sequences.values()),
                    regions=regions,
                    fps=self.fps,
                )
                if path:
                    videos["pose_overlay"] = path

        # --- pose CSVs -----------------------------------------------------
        pose_files: dict[str, str] = {}
        for name, filename in (("Kicker", "pose_kicker_pixel.csv"), ("GK", "pose_gk_pixel.csv")):
            seq = self.pose_sequences.get(name)
            path = write_pose_csv(out_dir, seq, filename) if seq else None
            if path:
                pose_files[name] = path

        # --- state ---------------------------------------------------------
        toml_path = os.path.join(out_dir, "data.toml")
        if toml:
            try:
                with open(toml_path, "w") as fh:
                    toml.dump(self.to_data(), fh)
            except Exception as exc:
                print(f">> vaila/pynalty: could not write data.toml: {exc}")
        else:
            print(">> vaila/pynalty: toml not installed, skipping data.toml")

        # --- tables and reports -------------------------------------------
        ctx = ReportContext(
            video_path=self.video_path,
            fps=self.fps,
            metrics=metrics,
            goal=self.goal,
            snapshots=snapshots,
            ball_rows=ball_rows,
            flight_path=metrics.get("_flight_path"),
            pose_metrics=self.pose_metrics,
            pose_files=pose_files,
            videos=videos,
            anthro=self.to_data()["anthropometrics"],
        )

        write_results_csv(out_dir, ctx)
        write_summary_csv(out_dir, ctx)
        write_ball_path_csvs(out_dir, ball_rows, ctx.flight_path)

        db_path = self.database_path or os.path.join(
            self.output_dir_override or os.path.dirname(self.video_path) or ".", DATABASE_CSV
        )
        append_database(db_path, ctx)

        reports = write_html_reports(out_dir, ctx, self.lang)

        print(f">> vaila/pynalty: results in {out_dir}")
        for path in reports.values():
            print(f">>   report: {path}")
        print(f">>   database: {db_path}")
        return True

    # ------------------------------------------------------------- interaction

    def _on_button(self, key: str):
        if key == "lang":
            self.ui_lang = "en" if self.ui_lang == "pt" else "pt"
            self.feedback_timer = 0
        elif key == "calibrate":
            self.redo_calibration()
        elif key == "cancel":
            self.cancel_calibration()
        elif key == "confirm":
            if self.is_calibrating:
                self.confirm_calibration()
            else:
                self.confirm_frame()
        elif key == "edit" and self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS, PHASE_OUTCOME):
            self.calibration_stage = "idle"
            self.set_phase(
                PHASE_KICK_FRAME if self.phase == PHASE_KICK_POINTS else PHASE_GOAL_FRAME
            )
        elif key == "prev":
            self.calibration_stage = "idle"
            if self.phase > PHASE_KICK_FRAME:
                self.set_phase(self.phase - 1)
        elif key == "next":
            if self.phase in (PHASE_BALL_PATH, PHASE_BOXES, PHASE_ANTHRO):
                self.set_phase(PHASE_REVIEW)
                self.compute_metrics()
            elif self.phase in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE):
                self.confirm_frame()
            elif self.is_calibrating:
                self.confirm_calibration()
            elif self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS) and all(
                k in self.current_event.points for k in ("Ball", "GK")
            ):
                if self.phase == PHASE_KICK_POINTS:
                    self.set_phase(PHASE_GOAL_FRAME)
                else:
                    self.set_phase(PHASE_OUTCOME)
            elif self.phase == PHASE_OUTCOME and self.shot_outcome:
                self.resume_workflow()
        elif key in ("goal", "save_outcome", "miss", "woodwork"):
            self._set_shot_outcome("save" if key == "save_outcome" else key)
        elif key == "auto_ball":
            self.set_phase(PHASE_BALL_PATH)
            self.auto_detect_ball()
        elif key == "pose":
            self.set_phase(PHASE_BOXES)
            self.run_pose()
        elif key == "anthro":
            self.set_phase(PHASE_ANTHRO)
            self.ask_anthro()
        elif key == "save":
            if self.core_steps_done():
                if self.phase != PHASE_REVIEW:
                    self.set_phase(PHASE_REVIEW)
                self.flash(
                    self.tr("Resultados salvos", "Results saved")
                    if self.save_results_package()
                    else self.tr("Falha ao salvar", "Save failed"),
                    120,
                )
            else:
                self.flash(
                    self.tr(
                        "Complete Chute, Gol/Calibração e Resposta antes de salvar",
                        "Complete Kick, Goal/Calibration and Response before saving",
                    ),
                    150,
                )
        elif key == "load":
            if self.load_toml():
                self.prepare_calibration()
        elif key == "help":
            self.show_help = not self.show_help

    def goto_step(self, idx: int, *, force: bool = False):
        """Compatibility hook for event indices; UI navigation uses macro steps."""
        if not 0 <= idx < len(self.events):
            return
        key = self.events[idx].key
        if key == "calibration":
            self.redo_calibration()
            return
        macro = {"kick": 0, "goal": 1, "gk_move": 2, "ball_path": 3, "boxes": 3, "anthro": 3}.get(
            key, 3
        )
        self.goto_macro_step(macro)

    def _maybe_advance(self):
        if self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS) and all(
            k in self.current_event.points for k in ("Ball", "GK")
        ):
            if self.phase == PHASE_KICK_POINTS:
                self.set_phase(PHASE_GOAL_FRAME)
            else:
                self.set_phase(PHASE_OUTCOME)

    def _set_shot_outcome(self, key: str):
        if self.step("goal").frame_idx < 0:
            self.flash(
                self.tr(
                    "Defina o frame de chegada antes do resultado",
                    "Set arrival frame before outcome",
                )
            )
            return
        outcome, label = classify_shot_outcome(key, ball_inside_goal=True)
        # Re-classify with real geometry when metrics exist.
        inside = bool(self.last_results.get("ball_inside_goal", True))
        outcome, label = classify_shot_outcome(key, ball_inside_goal=inside)
        self.shot_outcome = outcome
        self.flash(self.tr("Resultado confirmado", f"Result: {label}"))
        self.compute_metrics()
        if self.phase == PHASE_OUTCOME:
            # After outcome go to Response (or Save if response already set).
            self.resume_workflow()

    def _mark_point(self, ix: float, iy: float):
        """Place the next point for the current step."""
        evt = self.current_event
        key = self.current_key

        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return
        if (
            self.phase
            in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE, PHASE_OUTCOME, PHASE_REVIEW)
            and not self.is_calibrating
        ):
            return
        if self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS):
            # Keep marks on the confirmed event frame; snap back if user scrubbed away.
            if evt.frame_idx >= 0 and self.current_frame_idx != evt.frame_idx:
                self.seek(evt.frame_idx, force=True)
                self.flash(
                    self.tr(
                        "Volte ao frame do evento para marcar (ou E para editar)",
                        "Return to the event frame to mark (or E to edit)",
                    )
                )
                return
            for name in ("Ball", "GK"):
                if name not in evt.points:
                    evt.points[name] = [ix, iy]
                    label = (
                        self.tr("Bola", "Ball")
                        if name == "Ball"
                        else (
                            self.tr("Mão", "Hand")
                            if self.phase == PHASE_GOAL_POINTS
                            else self.tr("Goleiro", "Keeper")
                        )
                    )
                    self.flash(self.tr(f"{label} marcada", f"{label} marked"))
                    break
            self.compute_metrics()
            self._maybe_advance()
            return
        if self.is_calibrating:
            if self.calibration_stage == "frame" or len(self.calibration_draft) >= 4:
                return
            self.calibration_draft.append([ix, iy])
            if len(self.calibration_draft) == 4:
                self.calibration_stage = "preview"
            return

        if key == "ball_path":
            path = self.ball_path()
            merged = {p.frame: p for p in path.points}
            merged[self.current_frame_idx] = BallPathPoint(
                frame=self.current_frame_idx, x_px=ix, y_px=iy, source="manual"
            )
            self.set_ball_path(BallPath(points=list(merged.values())))
            self.flash(
                self.tr(
                    f"Bola marcada no frame {self.current_frame_idx} ({len(merged)} pontos)",
                    f"Ball marked on frame {self.current_frame_idx} ({len(merged)} points)",
                )
            )
            self.compute_metrics()
            return

        if key == "anthro":
            self.flash(
                self.tr(
                    "B informa medidas; volte ao Salvar se usar padrões",
                    "Press B to type measurements, or return to Save for defaults",
                )
            )
            return

        self.compute_metrics()

    def _undo_point(self):
        evt = self.current_event
        key = self.current_key
        if (
            self.phase in (PHASE_KICK_FRAME, PHASE_GOAL_FRAME, PHASE_RESPONSE, PHASE_REVIEW)
            and not self.is_calibrating
        ):
            return
        if self.is_calibrating:
            pending = self.pending_calibration
            if pending is not None:
                pending = copy.deepcopy(pending)
                pending["source_video"] = str(self.video_path or "")
                pending["source_frame"] = self.current_frame_idx
                self.pending_calibration = pending
                self.calibration_preview_edited = True
            if self.calibration_draft:
                self.calibration_draft.pop()
                self.calibration_stage = "corners"
            return
        elif self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS, PHASE_OUTCOME):
            if self.phase == PHASE_OUTCOME:
                self.set_phase(PHASE_GOAL_POINTS)
            self.shot_outcome = None
            for name in ("GK", "Ball"):
                if name in self.current_event.points:
                    del self.current_event.points[name]
                    break
        elif key == "ball_path":
            path = self.ball_path()
            merged = {p.frame: p for p in path.points}
            if self.current_frame_idx in merged:
                del merged[self.current_frame_idx]
                self.flash(
                    self.tr(
                        f"Ponto da bola removido no frame {self.current_frame_idx}",
                        f"Ball point on frame {self.current_frame_idx} removed",
                    )
                )
            elif merged:
                last = max(merged)
                del merged[last]
                self.flash(
                    self.tr(
                        f"Ponto da bola removido no frame {last}",
                        f"Ball point on frame {last} removed",
                    )
                )
            self.set_ball_path(BallPath(points=list(merged.values())))
        elif key == "boxes":
            for name in ("GK", "Kicker"):
                if name in evt.points:
                    del evt.points[name]
                    self.flash(self.tr(f"Caixa {name} removida", f"{name} box removed"))
                    break
        self.compute_metrics()

    def _handle_keydown(self, event) -> bool:
        key = event.key
        if key == pygame.K_F2:
            self._on_button("lang")
            return True
        if self.show_wizard:
            self.show_wizard = False
            return True
        if self.show_help:
            if key in (pygame.K_h, pygame.K_ESCAPE):
                self.show_help = False
            return True
        jumps = {pygame.K_RIGHT: 1, pygame.K_LEFT: -1, pygame.K_UP: 10, pygame.K_DOWN: -10}
        if key in jumps:
            self.seek(self.current_frame_idx + jumps[key])
        elif key == pygame.K_HOME:
            self.seek(0)
        elif key == pygame.K_END:
            self.seek(self.total_frames - 1)
        elif key in (pygame.K_1, pygame.K_KP1):
            # Fast key: Kick macro
            if self.phase not in (PHASE_KICK_FRAME, PHASE_KICK_POINTS):
                self.goto_macro_step(0)
            elif self.phase == PHASE_KICK_FRAME:
                self.confirm_frame()
            else:
                self.set_phase(PHASE_KICK_FRAME)
        elif key in (pygame.K_2, pygame.K_KP2):
            # Fast key: Goal/Calib macro
            if self.phase not in (PHASE_GOAL_FRAME, PHASE_GOAL_POINTS, PHASE_OUTCOME) and not (
                self.is_calibrating
            ):
                self.goto_macro_step(1)
            elif self.phase == PHASE_GOAL_FRAME and not self.is_calibrating:
                self.confirm_frame()
            elif self.phase in (PHASE_GOAL_POINTS, PHASE_OUTCOME) or self.is_calibrating:
                self.calibration_stage = "idle"
                self.set_phase(PHASE_GOAL_FRAME)
        elif key in (pygame.K_3, pygame.K_KP3):
            # Fast key: Response macro
            if self.phase != PHASE_RESPONSE:
                self.goto_macro_step(2)
            else:
                self.confirm_frame()
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            if self.is_calibrating:
                self._on_button("confirm")
            elif self.phase in (PHASE_KICK_POINTS, PHASE_GOAL_POINTS) and all(
                k in self.current_event.points for k in ("Ball", "GK")
            ):
                self._on_button("next")
            elif self.phase == PHASE_REVIEW and self.core_steps_done():
                self._on_button("save")
            else:
                self._on_button("confirm")
        elif key == pygame.K_TAB:
            self._on_button("prev" if pygame.key.get_mods() & pygame.KMOD_SHIFT else "next")
        elif key == pygame.K_SPACE and not self.navigation_locked:
            self.playing = not self.playing
        elif key in (
            pygame.K_PLUS,
            pygame.K_KP_PLUS,
            pygame.K_EQUALS,
            pygame.K_MINUS,
            pygame.K_KP_MINUS,
        ):
            area = self.content_rect()
            self.zoom_at(
                1 / 1.1 if key in (pygame.K_MINUS, pygame.K_KP_MINUS) else 1.1,
                area.centerx,
                area.centery,
            )
        elif key == pygame.K_0:
            self.fit_view()
        elif key == pygame.K_f:
            self.ask_fps()
        elif key == pygame.K_ESCAPE:
            if self.is_calibrating and self.calibration_return is not None:
                self.cancel_calibration()
            else:
                return False
        else:
            actions = {
                pygame.K_c: "calibrate",
                pygame.K_e: "edit",
                pygame.K_g: "goal",
                pygame.K_d: "save_outcome",
                pygame.K_m: "miss",
                pygame.K_w: "woodwork",
                pygame.K_a: "auto_ball",
                pygame.K_p: "pose",
                pygame.K_b: "anthro",
                pygame.K_s: "save",
                pygame.K_l: "load",
                pygame.K_h: "help",
            }
            if key in actions:
                self._on_button(actions[key])
        return True

    def _handle_mousedown(self, event):
        mx, my = event.pos
        if self.show_wizard:
            self.show_wizard = False
            return

        if self.show_help:
            return
        for btn in self.buttons:
            if btn.rect.collidepoint(mx, my):
                if event.button == 1:
                    self._on_button(btn.key)
                return

        # Direct navigation via sidebar macro-steps
        for rect, step_idx in getattr(self, "macro_step_rects", []):
            if rect.collidepoint(mx, my):
                if event.button == 1:
                    self.goto_macro_step(step_idx)
                return

        # Direct quick save
        quick_save = getattr(self, "quick_save_rect", None)
        if quick_save and quick_save.collidepoint(mx, my):
            if event.button == 1:
                self._on_button("save")
            return

        # Optional tools direct access
        for rect, opt_phase in getattr(self, "optional_tool_rects", []):
            if rect.collidepoint(mx, my):
                if event.button == 1:
                    self.set_phase(opt_phase)
                return

        w, h = self.screen.get_size()
        slider_zone = pygame.Rect(0, h - BOTTOM_H, w, 44)

        if event.button == 1:
            if slider_zone.collidepoint(mx, my) and not self.navigation_locked:
                self.playing = False
                self.start_drag_slider = True
                snapped = False
                if self.total_frames > 1:
                    margin = 20
                    timeline_w = w - margin * 2
                    for evt in (self.step("gk_move"), self.step("kick"), self.step("goal")):
                        if evt.frame_idx != -1:
                            evt_x = margin + int(
                                evt.frame_idx / (self.total_frames - 1) * timeline_w
                            )
                            if abs(mx - evt_x) <= 8:
                                self.seek(evt.frame_idx)
                                snapped = True
                                break
                if not snapped:
                    self._slider_seek(mx)
            elif self.content_rect().collidepoint(mx, my):
                ix, iy = self.screen_to_image_coords(mx, my)
                if not (0 <= ix < self.width and 0 <= iy < self.height):
                    return
                if self.current_key == "boxes":
                    self.box_drag_start = (ix, iy)
                    self.box_drag_current = (ix, iy)
                else:
                    self._mark_point(ix, iy)
        elif event.button == 3 and self.content_rect().collidepoint(mx, my):
            ix, iy = self.screen_to_image_coords(mx, my)
            if 0 <= ix < self.width and 0 <= iy < self.height:
                self._undo_point()
        elif event.button == 2:
            self.is_dragging = True
            self.last_mouse_pos = (mx, my)

    def _handle_mouseup(self, event):
        self.start_drag_slider = False
        if event.button == 2:
            self.is_dragging = False
        elif event.button == 1 and self.box_drag_start and self.box_drag_current:
            x1, y1 = self.box_drag_start
            x2, y2 = self.box_drag_current
            self.box_drag_start = self.box_drag_current = None
            if abs(x2 - x1) < 12 or abs(y2 - y1) < 12:
                self.flash(
                    self.tr(
                        "Caixa muito pequena: arraste um retângulo maior",
                        "Box too small - drag a larger rectangle",
                    )
                )
                return
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            points = self.step("boxes").points
            name = "Kicker" if "Kicker" not in points else "GK"
            points[name] = box
            self.flash(
                self.tr(
                    f"Caixa {name} marcada: P executa pose", f"{name} box set - press P to run pose"
                )
            )

    def _slider_seek(self, mx: int):
        w = self.screen.get_width()
        margin = 20
        ratio = float(np.clip((mx - margin) / max(1, w - 2 * margin), 0.0, 1.0))
        self.seek(int(ratio * max(0, self.total_frames - 1)))

    def run(self):
        """Open the window and run the marking loop until the user quits."""
        self._init_pygame()
        self.confirm_fps_on_start()
        _banner(
            "Interactive marking",
            f"{os.path.basename(self.video_path or '')} | {self.total_frames} frames @ {self.fps:.2f} fps",
        )
        running = True
        clock = pygame.time.Clock()

        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.VIDEORESIZE:
                    self.display_size = (max(760, event.w), max(600, event.h))
                    self.screen = pygame.display.set_mode(self.display_size, pygame.RESIZABLE)
                elif event.type == pygame.KEYDOWN:
                    running = self._handle_keydown(event)
                elif event.type == pygame.MOUSEWHEEL:
                    mx, my = pygame.mouse.get_pos()
                    if self.content_rect().collidepoint(mx, my):
                        self.zoom_at(1.1 if event.y > 0 else 1 / 1.1, mx, my)
                elif event.type == pygame.MOUSEBUTTONDOWN:
                    self._handle_mousedown(event)
                elif event.type == pygame.MOUSEBUTTONUP:
                    self._handle_mouseup(event)
                elif event.type == pygame.MOUSEMOTION:
                    mx, my = event.pos
                    if self.start_drag_slider:
                        self._slider_seek(mx)
                    elif self.box_drag_start is not None:
                        self.box_drag_current = self.screen_to_image_coords(mx, my)
                    elif self.is_dragging:
                        self.offset_x += mx - self.last_mouse_pos[0]
                        self.offset_y += my - self.last_mouse_pos[1]
                        self.last_mouse_pos = (mx, my)

            if self.playing:
                self.advance_playback()

            self.draw_content()
            clock.tick(max(1.0, self.fps) if self.playing else 60)

        pygame.quit()
        if self.cap:
            self.cap.release()


def load_video_file_dialog(ui_lang="pt"):
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Selecionar vídeo" if ui_lang == "pt" else "Select video",
        filetypes=[("Vídeo" if ui_lang == "pt" else "Video", "*.mp4 *.avi *.mov *.mkv")],
    )
    root.destroy()
    return file_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pynalty - penalty kick analysis")
    parser.add_argument("-i", "--input", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output directory (optional)")
    parser.add_argument("-c", "--config", help="Path to a data.toml with saved marks")
    parser.add_argument("--calibration", help="Reusable goal calibration TOML")
    parser.add_argument("--ui-lang", choices=("pt", "en"), default="pt", help="Interface language")
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Force the file-picker dialog even if -i is given",
    )
    parser.add_argument(
        "--database",
        help=f"Path to the accumulating penalty database (default: {DATABASE_CSV} beside the output)",
    )
    parser.add_argument(
        "--no-wizard", action="store_true", help="Skip the welcome overlay on launch"
    )
    parser.add_argument(
        "--auto-ball",
        action="store_true",
        help="Run YOLO ball detection right after loading a config",
    )
    parser.add_argument(
        "--pose",
        action="store_true",
        help="Run MediaPipe pose in the saved bounding boxes after loading a config",
    )
    parser.add_argument(
        "--report-only",
        action="store_true",
        help="Regenerate every output from -c without opening the marking window",
    )
    parser.add_argument(
        "--penalty-distance",
        type=float,
        default=None,
        help="Shot distance in metres (default 11.0)",
    )
    parser.add_argument(
        "--goal-width", type=float, default=None, help="Goal width in metres (default 7.32)"
    )
    parser.add_argument(
        "--goal-height", type=float, default=None, help="Goal height in metres (default 2.44)"
    )
    parser.add_argument(
        "--lang",
        choices=("en", "pt", "both"),
        default="both",
        help="Report language (default: both)",
    )
    return parser


def _goal_from_args(args) -> GoalGeometry:
    base = GoalGeometry()
    return GoalGeometry(
        width=args.goal_width if args.goal_width else base.width,
        height=args.goal_height if args.goal_height else base.height,
        penalty_distance=(
            args.penalty_distance if args.penalty_distance else base.penalty_distance
        ),
        ball_radius=base.ball_radius,
    )


def _mirror_cli(args, vid_path: str) -> None:
    """Print the copy-pasteable command that reproduces this run."""
    cli_argv = ["-i", vid_path, "--ui-lang", args.ui_lang]
    if args.calibration:
        cli_argv += ["--calibration", args.calibration]
    if args.output:
        cli_argv += ["-o", args.output]
    if args.config:
        cli_argv += ["-c", args.config]
    if args.database:
        cli_argv += ["--database", args.database]
    if args.no_wizard:
        cli_argv.append("--no-wizard")
    if args.auto_ball:
        cli_argv.append("--auto-ball")
    if args.pose:
        cli_argv.append("--pose")
    if args.report_only:
        cli_argv.append("--report-only")
    if args.goal_width:
        cli_argv += ["--goal-width", str(args.goal_width)]
    if args.goal_height:
        cli_argv += ["--goal-height", str(args.goal_height)]
    if args.penalty_distance:
        cli_argv += ["--penalty-distance", str(args.penalty_distance)]
    if args.lang != "both":
        cli_argv += ["--lang", args.lang]
    print_gui_cli_mirror("vaila/pynalty", ["uv", "run", "vaila/pynalty.py", *cli_argv])


def main(argv: list[str] | None = None) -> int:
    """CLI and GUI entry point.

    GUI mode: no flags needed, a file-picker dialog opens.
    CLI mode: ``-i video.mp4 -o out_dir -c data.toml`` skips the dialog.
    ``--report-only`` regenerates every artefact from ``-c`` headlessly.
    """
    args = build_parser().parse_args(argv)

    if args.report_only and (not args.config or not args.input):
        print("Error: --report-only needs -i and -c; no file dialogs are opened")
        return 1

    vid_path = None
    if not args.gui or args.report_only:
        if args.input:
            vid_path = args.input
        elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Fallback for a bare positional path, e.g. `pynalty.py video.mp4`
            vid_path = sys.argv[1]

    if not vid_path:
        vid_path = load_video_file_dialog(args.ui_lang)

    if not vid_path:
        print("No video selected.")
        return 1

    if not os.path.exists(vid_path):
        print(f"Error: Video file not found: {vid_path}")
        return 1

    if args.report_only and not args.config:
        print("Error: --report-only needs -c with a saved data.toml")
        return 1

    _mirror_cli(args, vid_path)

    app = PynaltyApp(
        vid_path,
        goal=_goal_from_args(args),
        show_wizard=not args.no_wizard,
        lang=args.lang,
        database=args.database,
        ui_lang=args.ui_lang,
        calibration=args.calibration,
    )

    if args.output:
        os.makedirs(args.output, exist_ok=True)
        app.output_dir_override = args.output

    if args.config:
        if not os.path.exists(args.config):
            print(f"Config file not found: {args.config}")
            if args.report_only:
                return 1
        elif toml is None:
            print("Cannot load the config: the toml library is not installed")
            if args.report_only:
                return 1
        else:
            print(f"Loading marks from {args.config} ...")
            try:
                with open(args.config) as fh:
                    data = toml.load(fh)
                app.load_from_data(data)
            except Exception as exc:
                print(f"Failed to load config: {exc}")
                if args.report_only:
                    return 1

    calibration_ok = app.prepare_calibration(report_only=args.report_only)
    if not calibration_ok and args.report_only:
        if app.cap:
            app.cap.release()
        return 1

    if args.auto_ball:
        app.auto_detect_ball()
    if args.pose:
        app.run_pose()

    if args.report_only:
        ok = app.save_results_package()
        if app.cap:
            app.cap.release()
        return 0 if ok else 1

    app.run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
