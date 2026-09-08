"""
Project: vailá Multimodal Toolbox
Script: pynalty.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 19 December 2025
Update Date: 08 September 2026
Version: 0.3.130

Description:
Guided analysis of a penalty kick from a single broadcast or handheld camera.
The pygame workflow calibrates the goal, confirms all three event frames,
then collects ball/keeper points on paused contact and arrival frames. Optional
trajectory, pose and body measurements precede review and result export.
The interface supports Portuguese (default) and English independently of reports.
Reusable calibration is validated and previewed before acceptance.

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
        "key": "gk_move",
        "name": "1. Keeper starts moving",
        "why": "Anchors reaction time. Negative means the keeper guessed early.",
        "how": "Confirm the first clear dive/prepare motion with ENTER.",
        "required": True,
    },
    {
        "key": "kick",
        "name": "2. Ball contact",
        "why": "Starts the flight clock and fixes where both athletes were at contact.",
        "how": "Confirm foot-ball contact with ENTER. Points are requested after all three frames.",
        "required": True,
    },
    {
        "key": "goal",
        "name": "3. Ball at goal line",
        "why": "Ends the flight clock and gives the entry point. Outcome (goal/save/miss) is required.",
        "how": "Confirm goal-line arrival with ENTER. Points and outcome follow the three frames.",
        "required": True,
    },
    {
        "key": "calibration",
        "name": "4. Goal calibration",
        "why": "Converts pixels into metres on the plane of the goal mouth.",
        "how": "Pause with ENTER, click bottom-left, top-left, top-right, bottom-right, then confirm.",
        "required": True,
    },
    {
        "key": "ball_path",
        "name": "5. Ball path (optional)",
        "why": "Feeds the trajectory overlay and the animated replay in the report.",
        "how": "Optional. Press A for YOLO, or click the ball frame by frame. Skip with Step > if you do not need it.",
        "required": False,
    },
    {
        "key": "boxes",
        "name": "6. Pose boxes (optional)",
        "why": "Crops athletes so MediaPipe finds landmarks a full frame misses. Improves gap-to-ball.",
        "how": "Optional. Drag KICKER then KEEPER boxes and press P. Skip with Step > if you do not need pose.",
        "required": False,
    },
    {
        "key": "anthro",
        "name": "7. Body measures (optional)",
        "why": "Sets the keeper's reach. If skipped, elite defaults (1.88 m stature) are used.",
        "how": "Optional. Press B to type stature / arm span / standing reach, or skip to use defaults.",
        "required": False,
    },
)

REQUIRED_STEP_KEYS = frozenset(s["key"] for s in STEP_SPECS if s["required"])


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

    def __init__(self, key: str, label: str):
        self.key = key
        self.label = label
        self.rect = pygame.Rect(0, 0, 0, 0)

    def draw(self, screen, font, hover: bool, enabled: bool = True) -> None:
        if not enabled:
            bg, fg = (44, 50, 58), (110, 118, 128)
        elif hover:
            bg, fg = (76, 130, 96), WHITE
        else:
            bg, fg = (52, 62, 74), WHITE
        pygame.draw.rect(screen, bg, self.rect, border_radius=5)
        pygame.draw.rect(screen, PANEL_LINE, self.rect, 1, border_radius=5)
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
        self.phase = 0
        self.calibration_stage = "frame"
        self.calibration_preview_edited = False
        self.calibration_draft = []
        self.calibration_record = None
        self.calibration_return = None
        self.calibration_backup = None
        self.output_dir_override: str | None = None

        self.events: list[PynaltyEvent] = []
        self._init_events()
        self.current_event_idx = 3

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
        "calibration",
        "gk_move",
        "kick",
        "goal",
        "kick",
        "goal",
        "goal",
        "ball_path",
        "boxes",
        "anthro",
        "review",
    )

    def tr(self, pt: str, en: str) -> str:
        return pt if self.ui_lang == "pt" else en

    @property
    def navigation_locked(self) -> bool:
        return self.phase in (4, 5, 6) or (self.phase == 0 and self.calibration_stage != "frame")

    def set_phase(self, phase: int):
        self.phase = max(0, min(phase, len(self.PHASE_KEYS) - 1))
        key = self.PHASE_KEYS[self.phase]
        self.current_event_idx = next((i for i, e in enumerate(self.events) if e.key == key), -1)
        self.playing = False
        self.start_drag_slider = False
        if self.phase in (4, 5, 6) or self.phase in (1, 2, 3) and self.current_event.frame_idx >= 0:
            self.seek(self.current_event.frame_idx, force=True)

    def resume_workflow(self, optional_phase=None):
        if len(self.calib_points()) != 4:
            self.set_phase(0)
            return
        for phase, key in enumerate(("gk_move", "kick", "goal"), 1):
            if not 0 <= self.step(key).frame_idx < self.total_frames:
                self.set_phase(phase)
                return
        if self.step("goal").frame_idx <= self.step("kick").frame_idx:
            self.set_phase(3)
            return
        for phase, key in ((4, "kick"), (5, "goal")):
            if not all(k in self.step(key).points for k in ("Ball", "GK")):
                self.set_phase(phase)
                return
        if not self.shot_outcome:
            self.set_phase(6)
            return
        self.set_phase(
            optional_phase if isinstance(optional_phase, int) and 7 <= optional_phase <= 10 else 7
        )

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
                self.calibration_stage = "frame"
                self.set_phase(0)
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
            self.calibration_stage = "frame"
            self.set_phase(0)
            return False
        self.calibration_path = str(path)
        if report_only:
            self.step("calibration").points = {"points": pts}
            self.step("calibration").frame_idx = -1
            self.goal, self.calibration_record = geometry, data
            self.compute_metrics()
        else:
            self.calibration_draft = pts
            self.calibration_preview_edited = False
            self.pending_calibration = data
            self.calibration_stage = "preview"
            self.set_phase(0)
        return True

    def redo_calibration(self):
        if self.phase != 0:
            self.calibration_return = self.phase
            self.calibration_backup = (
                copy.deepcopy(self.step("calibration").points),
                self.step("calibration").frame_idx,
            )
        self.calibration_draft = []
        self.pending_calibration = None
        self.calibration_preview_edited = False
        self.calibration_stage = "frame"
        self.set_phase(0)

    def cancel_calibration(self):
        if self.calibration_return is not None:
            self.step("calibration").points, self.step("calibration").frame_idx = (
                self.calibration_backup
            )
            phase = self.calibration_return
            self.calibration_return = self.calibration_backup = None
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
        data = getattr(self, "pending_calibration", None) or self.calibration_data()
        data = copy.deepcopy(data)
        data["points"] = copy.deepcopy(self.calibration_draft)
        try:
            pts, geometry = self.validate_calibration(data)
        except (ValueError, TypeError, KeyError, np.linalg.LinAlgError) as exc:
            self.flash(str(exc), 180)
            return
        reused = (
            bool(getattr(self, "pending_calibration", None)) and not self.calibration_preview_edited
        )
        self.step("calibration").points = {"points": pts}
        self.step("calibration").frame_idx = -1 if reused else self.current_frame_idx
        self.goal, self.calibration_record = geometry, data
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
            self.resume_workflow()
        else:
            self.set_phase(phase)
        self.compute_metrics()
        self.flash(self.tr("Calibração confirmada", "Calibration confirmed"))

    def confirm_frame(self):
        if self.phase not in (1, 2, 3):
            return
        evt = self.current_event
        frame = self.current_frame_idx
        self.playing = False
        if self.phase == 3 and frame <= self.step("kick").frame_idx:
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
        evt.frame_idx = frame
        self.flash(self.tr(f"Frame {frame} confirmado", f"Frame {frame} confirmed"))
        if self.phase < 3:
            self.set_phase(self.phase + 1)
        else:
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
        if self.phase == 0:
            if self.calibration_stage == "frame":
                return self.tr(
                    "Escolha um frame com o gol visível. Enter pausa e inicia os cantos.",
                    "Choose a frame with the goal visible. Enter pauses and starts the corners.",
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
                "Confira enquadramento, zoom e posição da câmera. Enter confirma; C refaz.",
                "Check framing, zoom and camera position. Enter confirms; C restarts.",
            )
        prompts = {
            1: (
                "Primeiro movimento claro de preparação/mergulho do goleiro. Confirme com Enter.",
                "First clear keeper preparation/dive movement. Confirm with Enter.",
            ),
            2: (
                "Instante do contato do pé com a bola. Confirme com Enter.",
                "Instant the foot contacts the ball. Confirm with Enter.",
            ),
            3: (
                "Instante em que a bola alcança a linha do gol. Confirme com Enter.",
                "Instant the ball reaches the goal line. Confirm with Enter.",
            ),
            6: (
                "Qual foi o resultado? Escolha abaixo ou use G / D / M / W.",
                "What was the outcome? Choose below or use G / D / M / W.",
            ),
            7: (
                "Opcional: A detecta a bola; clique para trajetória manual, ou Pular.",
                "Optional: A detects the ball; click for a manual trajectory, or Skip.",
            ),
            8: (
                "Opcional: arraste a caixa do cobrador, depois do goleiro; P executa pose, ou Pular.",
                "Optional: drag kicker then keeper boxes; P runs pose, or Skip.",
            ),
            9: (
                "Opcional: B informa medidas corporais, ou Pular usa os valores padrão.",
                "Optional: B enters body measurements, or Skip uses defaults.",
            ),
            10: (
                "Revise os frames e as marcações. Salvar resultados gera o pacote completo.",
                "Review frames and marks. Save results generates the full package.",
            ),
        }
        if self.phase in (4, 5):
            if all(k in self.current_event.points for k in ("Ball", "GK")):
                return self.tr(
                    "Pontos confirmados. Continuar avança; botão direito desfaz.",
                    "Points confirmed. Continue advances; right click undoes.",
                )
            ball = "Ball" not in self.current_event.points
            return self.tr(
                "Clique no centro da bola." if ball else "Clique no centro do goleiro.",
                "Click the ball centre." if ball else "Click the keeper centre.",
            )
        return self.tr(*prompts[self.phase])

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
                        f"Goleiro se move {abs(dt):.2f} s antes do chute: confira o frame.",
                        f"Keeper moves {abs(dt):.2f} s before contact - check step 1.",
                    )
                )
        calib = self.calib_points()
        if len(calib) == 4 and not is_convex_quadrilateral(calib):
            warn.append(
                self.tr(
                    "Cantos não convexos. Pressione C para refazer a calibração.",
                    "Calibration corners are not convex. Press O to reorder them.",
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
        if self.phase != 0:
            self.draw_ball_path_overlay()
            self.draw_boxes()
            self.draw_goal_axes()
            self.draw_vectors()
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

        pygame.display.flip()

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
            if self.phase == 0 or evt.frame_idx != self.current_frame_idx:
                continue
            for name, label in (
                ("Ball", self.tr("Bola", "Ball")),
                ("GK", self.tr("Goleiro", "Keeper")),
            ):
                if name in evt.points:
                    self.draw_marker(evt.points[name], color, f"{label}: {suffix}")
        points = self.calibration_draft if self.phase == 0 else self.calib_points()
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
        self.screen.set_clip(pygame.Rect(0, 0, PANEL_W, h))
        y = 12
        titles = self.tr(
            "Calibração|Movimento do goleiro|Contato com a bola|Bola na linha do gol|Pontos no chute|Pontos na chegada|Resultado|Trajetória (opcional)|Pose (opcional)|Medidas (opcional)|Revisão",
            "Calibration|Keeper movement|Ball contact|Ball at goal line|Contact points|Arrival points|Outcome|Trajectory (optional)|Pose (optional)|Body (optional)|Review",
        ).split("|")
        self.screen.blit(self.font_big.render("Pynalty", True, GREEN), (16, y))
        y += 34
        for line in _wrap(f"{self.phase + 1}/11  {titles[self.phase]}", self.font, PANEL_W - 32):
            self.screen.blit(self.font.render(line, True, YELLOW), (16, y))
            y += 21
        y += 8
        for line in _wrap(self.micro_prompt(), self.font, PANEL_W - 32):
            self.screen.blit(self.font.render(line, True, WHITE), (16, y))
            y += 21
        y += 12
        controls = (
            self.tr(
                "Setas: ±1 / ±10 frames. Espaço: reproduzir/pausar.",
                "Arrows: ±1 / ±10 frames. Space: play/pause.",
            )
            if not self.navigation_locked
            else self.tr(
                "Frame pausado e travado. Roda: zoom. Arraste com botão do meio: mover imagem.",
                "Frame paused and locked. Wheel: zoom. Middle drag: pan.",
            )
        )
        for line in _wrap(controls, self.font_small, PANEL_W - 32):
            self.screen.blit(self.font_small.render(line, True, GRAY), (16, y))
            y += 16
        y += 12
        for key, label in zip(("gk_move", "kick", "goal"), titles[1:4], strict=True):
            frame = self.step(key).frame_idx
            self.screen.blit(
                self.font_small.render(
                    f"{label}: {frame if frame >= 0 else '—'}", True, GREEN if frame >= 0 else GRAY
                ),
                (16, y),
            )
            y += 20
        if self.phase == 0:
            # Image order: 2--3 above 1--4. Highlight the next required corner.
            corners = [(48, y + 64), (48, y + 14), (250, y + 14), (250, y + 64)]
            pygame.draw.lines(self.screen, GRAY, True, corners, 2)
            for i, pt in enumerate(corners):
                color = YELLOW if i == len(self.calibration_draft) else BLUE
                pygame.draw.circle(self.screen, color, pt, 11)
                self.screen.blit(
                    self.font_small.render(str(i + 1), True, BLACK), (pt[0] - 4, pt[1] - 7)
                )
        elif self.phase == 10:
            for warning in self.validation_warnings():
                for line in _wrap(warning, self.font_small, PANEL_W - 32):
                    self.screen.blit(self.font_small.render(line, True, AMBER), (16, y))
                    y += 16
            for label, key, unit in (
                (self.tr("Bola", "Ball"), "vel_kmh", "km/h"),
                (self.tr("Voo", "Flight"), "flight_time_s", "s"),
                (self.tr("Reação", "Reaction"), "gk_response_time", "s"),
            ):
                value = self.last_results.get(key)
                if isinstance(value, int | float):
                    self.screen.blit(
                        self.font_small.render(f"{label}: {value:.3f} {unit}", True, WHITE), (16, y)
                    )
                    y += 20
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
        if self.phase == 0:
            label = self.tr("Confirmar frame Enter", "Confirm frame Enter")
            if len(self.calibration_draft) == 4:
                label = (
                    self.tr("Usar calibração Enter", "Use calibration Enter")
                    if getattr(self, "pending_calibration", None)
                    else self.tr("Confirmar e salvar", "Confirm and save")
                )
            specs.insert(0, ("confirm", label))
            if self.calibration_return is not None:
                specs.append(("cancel", self.tr("Cancelar Esc", "Cancel Esc")))
        elif self.phase in (1, 2, 3):
            specs.insert(0, ("confirm", self.tr("Confirmar frame Enter", "Confirm frame Enter")))
        elif self.phase in (4, 5, 6):
            specs.insert(0, ("edit", self.tr("Editar frame E", "Edit frame E")))
            if self.phase in (4, 5) and all(k in self.current_event.points for k in ("Ball", "GK")):
                specs.insert(0, ("next", self.tr("Continuar", "Continue")))
            if self.phase == 6:
                specs = [
                    (k, self.tr(pt, en))
                    for k, pt, en in (
                        ("goal", "Gol G", "Goal G"),
                        ("save_outcome", "Defesa D", "Save D"),
                        ("miss", "Fora M", "Miss M"),
                        ("woodwork", "Trave W", "Woodwork W"),
                    )
                ] + specs
        elif self.phase in (7, 8, 9):
            action = {
                7: ("auto_ball", self.tr("Detectar bola A", "Detect ball A")),
                8: ("pose", self.tr("Executar pose P", "Run pose P")),
                9: ("anthro", self.tr("Medidas B", "Body B")),
            }[self.phase]
            specs = [("next", self.tr("Pular / Continuar", "Skip / Continue")), action] + specs
        else:
            specs.insert(0, ("save", self.tr("Salvar resultados S", "Save results S")))
        cols = max(2, (w - 40) // 190)
        bw, bh, gap = (w - 40) // cols - 6, 29, 6
        self.buttons = [_Button(k, label) for k, label in specs]
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
            "Pynalty: calibrar → confirmar três frames → clicar nos pontos → opcionais → salvar.\nEnter confirma somente a ação indicada. Setas: ±1 / ±10 frames; Espaço: reproduzir/pausar.\nC: refazer calibração. Esc: cancelar recalibração ou sair. E: editar frame dos pontos.\nRoda / +/-: zoom; botão do meio: mover imagem; 0: ajustar à janela.\nClique esquerdo: próximo ponto; direito: desfazer. Não há cliques na fase dos frames.\nTab / Shift+Tab: avançar / voltar. G / D / M / W: gol / defesa / fora / trave.\nA: trajetória automática; P: pose; B: medidas; S: salvar na revisão; L: carregar sessão.\nPT / EN ou F2 muda o idioma sem perder marcações. F: ajustar fps.\nH fecha a ajuda. Enter inicia.",
            "Pynalty: calibrate → confirm three frames → click points → optional steps → save.\nEnter confirms only the indicated action. Arrows: ±1 / ±10 frames; Space: play/pause.\nC: recalibrate. Esc: cancel recalibration or quit. E: edit point frame.\nWheel / +/-: zoom; middle drag: pan; 0: fit window.\nLeft click: next point; right click: undo. Clicks do nothing during frame selection.\nTab / Shift+Tab: next / back. G / D / M / W: goal / save / miss / woodwork.\nA: automatic trajectory; P: pose; B: body; S: save at review; L: load session.\nPT / EN or F2 changes language without losing marks. F: adjust fps.\nH closes help. Enter starts.",
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
        """First box shown when the window opens, ahead of calibration.

        Velocity and time depend on FPS, so it must be settled before anything
        else. ``self.fps`` already holds the ffprobe-based auto-detection from
        ``load_video`` (same logic as numberframes.py); this just lets the user
        confirm it or type the correct value over it (e.g. when ffprobe/OpenCV
        misread a variable-frame-rate or phone-captured clip).
        """
        raw = self._text_input(
            self.tr(
                f"FPS detectado: {self.fps:.3f} Hz. Confirme ou corrija (necessário p/ velocidade e tempo)",
                f"Detected FPS: {self.fps:.3f} Hz. Confirm or correct (needed for velocity and time)",
            ),
            f"{self.fps:.3f}",
        )
        if raw is None or not raw.strip():
            return
        try:
            value = float(raw.strip().replace(",", "."))
        except ValueError:
            self.flash(
                self.tr(
                    "Número inválido; mantendo FPS detectado", "Not a number; keeping detected FPS"
                )
            )
            return
        if value > 0:
            self.fps = value
            self.flash(self.tr(f"FPS confirmado: {value:.3f} Hz", f"FPS confirmed: {value:.3f} Hz"))

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
        self.calibration_stage = "frame"
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
                dot(img, goal.points["GK"], (255, 0, 255), "GK @ line")
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
            if self.phase == 0:
                self.confirm_calibration()
            else:
                self.confirm_frame()
        elif key == "edit" and self.phase in (4, 5, 6):
            self.set_phase(2 if self.phase == 4 else 3)
        elif key == "prev":
            if self.phase > 1:
                self.set_phase(self.phase - 1)
            elif self.phase == 1:
                self.redo_calibration()
        elif key == "next":
            if self.phase in (7, 8, 9):
                self.set_phase(self.phase + 1)
                self.compute_metrics()
            elif self.phase in (1, 2, 3):
                self.confirm_frame()
            elif self.phase == 0:
                self.confirm_calibration()
            elif self.phase in (4, 5) and all(
                k in self.current_event.points for k in ("Ball", "GK")
            ):
                self.set_phase(self.phase + 1)
            elif self.phase == 6 and self.shot_outcome:
                self.set_phase(7)
        elif key in ("goal", "save_outcome", "miss", "woodwork"):
            self._set_shot_outcome("save" if key == "save_outcome" else key)
        elif key == "auto_ball" and self.phase == 7:
            self.auto_detect_ball()
        elif key == "pose" and self.phase == 8:
            self.run_pose()
        elif key == "anthro" and self.phase == 9:
            self.ask_anthro()
        elif key == "save" and self.phase == 10:
            self.flash(
                self.tr("Resultados salvos", "Results saved")
                if self.save_results_package()
                else self.tr("Falha ao salvar", "Save failed"),
                120,
            )
        elif key == "load":
            if self.load_toml():
                self.prepare_calibration()
        elif key == "help":
            self.show_help = not self.show_help

    def goto_step(self, idx: int, *, force: bool = False):
        """Compatibility hook for event indices; UI navigation uses explicit phases."""
        if not 0 <= idx < len(self.events):
            return
        key = self.events[idx].key
        phase = {
            "calibration": 0,
            "gk_move": 1,
            "kick": 2,
            "goal": 3,
            "ball_path": 7,
            "boxes": 8,
            "anthro": 9,
        }[key]
        if phase == 0:
            self.redo_calibration()
        elif phase <= self.phase:
            self.set_phase(phase)
        else:
            self._on_button("next")

    def _maybe_advance(self):
        if self.phase in (4, 5) and all(k in self.current_event.points for k in ("Ball", "GK")):
            self.set_phase(self.phase + 1)

    def _set_shot_outcome(self, key: str):
        if self.phase != 6:
            return
        outcome, label = classify_shot_outcome(key, ball_inside_goal=True)
        # Re-classify with real geometry when metrics exist.
        inside = bool(self.last_results.get("ball_inside_goal", True))
        outcome, label = classify_shot_outcome(key, ball_inside_goal=inside)
        self.shot_outcome = outcome
        self.flash(self.tr("Resultado confirmado", f"Result: {label}"))
        self.compute_metrics()
        self.set_phase(7)

    def _mark_point(self, ix: float, iy: float):
        """Place the next point for the current step."""
        evt = self.current_event
        key = self.current_key

        if not (0 <= ix < self.width and 0 <= iy < self.height):
            return
        if self.phase in (1, 2, 3, 6, 10):
            return
        if self.phase in (4, 5):
            for name in ("Ball", "GK"):
                if name not in evt.points:
                    evt.points[name] = [ix, iy]
                    self.flash(self.tr("Ponto confirmado", "Point confirmed"))
                    break
            self.compute_metrics()
            self._maybe_advance()
            return
        if self.phase == 0:
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
                    "B informa medidas; Pular usa valores padrão",
                    "Press B to type measurements, or Step > to use defaults",
                )
            )
            return

        self.compute_metrics()

    def _undo_point(self):
        evt = self.current_event
        key = self.current_key
        if self.phase in (1, 2, 3, 10):
            return
        if self.phase == 0:
            if getattr(self, "pending_calibration", None):
                self.pending_calibration = copy.deepcopy(self.pending_calibration)
                self.pending_calibration["source_video"] = str(self.video_path or "")
                self.pending_calibration["source_frame"] = self.current_frame_idx
                self.calibration_preview_edited = True
            if self.calibration_draft:
                self.calibration_draft.pop()
                self.calibration_stage = "corners"
            return
        elif self.phase in (4, 5, 6):
            if self.phase == 6:
                self.set_phase(5)
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
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
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
            if self.phase == 0 and self.calibration_return is not None:
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

        w, h = self.screen.get_size()
        slider_zone = pygame.Rect(0, h - BOTTOM_H, w, 44)

        if event.button == 1:
            if slider_zone.collidepoint(mx, my) and not self.navigation_locked:
                self.playing = False
                self.start_drag_slider = True
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
