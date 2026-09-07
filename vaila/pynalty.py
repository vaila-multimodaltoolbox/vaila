"""
Project: vailá Multimodal Toolbox
Script: pynalty.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 19 December 2025
Update Date: 07 September 2026
Version: 0.3.129

Description:
Guided analysis of a penalty kick from a single broadcast or handheld camera.
A seven-step pygame workflow collects the event frames, the goal calibration,
the ball trajectory, the athlete bounding boxes and the goalkeeper's body
measurements. From those marks the module reports the shot kinematics, the
goalkeeper's reaction and dive, and how far and how fast the keeper had to move
to touch the ball, then writes a self-contained HTML report in English and
Portuguese plus CSV files for a growing penalty database.

Workflow steps
    1. Goalkeeper starts moving   - scrub and press ENTER (or click) to set frame
    2. Ball contact               - click ball centre (frame auto-set), then keeper centre
    3. Ball at goal line          - click ball, then keeper; then G/D/M/W for outcome
    4. Goal calibration           - four goal corners
    5. Ball path (optional)       - YOLO or manual trail
    6. Bounding boxes (optional)  - MediaPipe pose crops
    7. Anthropometrics (optional) - defaults used when skipped

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
import os
import sys
import tkinter as tk
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
BOTTOM_H = 104

STEP_SPECS = (
    {
        "key": "gk_move",
        "name": "1. Keeper starts moving",
        "why": "Anchors reaction time. Negative means the keeper guessed early.",
        "how": "Scrub to the first clear dive/prepare motion, then press ENTER (or click the frame).",
        "required": True,
    },
    {
        "key": "kick",
        "name": "2. Ball contact",
        "why": "Starts the flight clock and fixes where both athletes were at contact.",
        "how": "Scrub to contact, click the BALL centre (frame is set automatically), then click the KEEPER centre.",
        "required": True,
    },
    {
        "key": "goal",
        "name": "3. Ball at goal line",
        "why": "Ends the flight clock and gives the entry point. Outcome (goal/save/miss) is required.",
        "how": "Click BALL centre (frame auto-set), then KEEPER centre, then press G=goal, D=save, M=miss, W=woodwork.",
        "required": True,
    },
    {
        "key": "calibration",
        "name": "4. Goal calibration",
        "why": "Converts pixels into metres on the plane of the goal mouth.",
        "how": "Click the 4 corners in order: bottom-left, top-left, top-right, bottom-right.",
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

BUTTON_SPECS = (
    ("prev", "< Step"),
    ("next", "Step >"),
    ("auto_ball", "Auto Ball (A)"),
    ("pose", "Run Pose (P)"),
    ("anthro", "Body (B)"),
    ("save", "Save (S)"),
    ("load", "Load (L)"),
    ("help", "Help (H)"),
)


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
        self.output_dir_override: str | None = None

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
        self.buttons = [_Button(k, label) for k, label in BUTTON_SPECS]

        self.show_help = False
        self.show_wizard = show_wizard
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

    # ------------------------------------------------------------------ setup

    def _init_events(self):
        self.events = [
            PynaltyEvent(
                f"{i + 1}. {spec['name']}",
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
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps = fps if fps and fps > 0 else 30.0
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
        pygame.display.set_caption(
            f"vailá - Pynalty Analysis - {os.path.basename(self.video_path or '')}"
        )
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
            return
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame_idx)
        ret, frame = self.cap.read()
        if not ret:
            return
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        self.frame_img = pygame.image.frombuffer(frame.tobytes(), frame.shape[1::-1], "RGB")

    def seek(self, frame_idx: int):
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
        """One-line instruction for the active micro-step."""
        key = self.current_key
        evt = self.current_event
        if evt is None:
            return ""
        if key == "gk_move":
            return "Scrub, then ENTER (or click) to lock this frame."
        if key == "kick":
            if "Ball" not in evt.points:
                return "Click the BALL centre — the current frame is locked automatically."
            if "GK" not in evt.points:
                return "Click the KEEPER centre at contact."
            return "Step complete — press Step > to continue."
        if key == "goal":
            if "Ball" not in evt.points:
                return "Click the BALL centre at the goal line — frame locks automatically."
            if "GK" not in evt.points:
                return "Click the KEEPER centre at the goal line."
            if not self.shot_outcome:
                return "Result? G=goal  D=save (defesa)  M=miss  W=woodwork"
            return f"Outcome: {self.shot_outcome} — press Step > to continue."
        if key == "calibration":
            n = len(self.calib_points())
            names = ("bottom-left", "top-left", "top-right", "bottom-right")
            if n < 4:
                return f"Click corner {n + 1}/4 ({names[n]})."
            return "Calibration complete — press Step >."
        if key == "ball_path":
            return "Optional: A=YOLO or click ball. Step > to skip."
        if key == "boxes":
            return "Optional: drag Kicker then GK boxes, P=pose. Step > to skip."
        if key == "anthro":
            return f"Optional: B=body measures, or skip (default {DEFAULT_GK_HEIGHT_M:.2f} m)."
        return evt.instructions

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
            warn.append("Goal frame must come after the contact frame.")
        if gk.frame_idx != -1 and kick.frame_idx != -1:
            dt = (gk.frame_idx - kick.frame_idx) / self.fps
            if dt < -1.0:
                warn.append(f"Keeper moves {abs(dt):.2f} s before contact - check step 1.")
        calib = self.calib_points()
        if len(calib) == 4 and not is_convex_quadrilateral(calib):
            warn.append("Calibration corners are not convex. Press O to reorder them.")
        if len(self.ball_path().points) in (1, 2):
            warn.append("Ball path needs 3+ points for a curve. Press A to auto-detect.")
        if not self.anthro.is_complete():
            warn.append(
                f"Using default keeper stature {DEFAULT_GK_HEIGHT_M:.2f} m (press B to override)."
            )
        if self.step("goal").frame_idx != -1 and not self.shot_outcome:
            warn.append("Set the shot result: G=goal, D=save, M=miss, W=woodwork.")
        return warn

    # --------------------------------------------------------------- drawing

    def draw_content(self):
        self.screen.fill(DARK_GRAY)
        area = self.content_rect()

        if self.frame_img:
            w = max(1, int(self.frame_img.get_width() * self.zoom))
            h = max(1, int(self.frame_img.get_height() * self.zoom))
            scaled = pygame.transform.smoothscale(self.frame_img, (w, h))
            self.screen.set_clip(area)
            self.screen.blit(scaled, (self.offset_x, self.offset_y))
            self.screen.set_clip(None)

        self.screen.set_clip(area)
        self.draw_markers()
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
        kick, goal, calib = self.step("kick"), self.step("goal"), self.step("calibration")
        if "Ball" in kick.points:
            self.draw_marker(kick.points["Ball"], GREEN, "Ball @ contact")
        if "GK" in kick.points:
            self.draw_marker(kick.points["GK"], CYAN, "GK @ contact")
        if "Ball" in goal.points:
            self.draw_marker(goal.points["Ball"], RED, "Ball @ line")
        if "GK" in goal.points:
            self.draw_marker(goal.points["GK"], VIOLET, "GK @ line")
        for i, p in enumerate(calib.points.get("points", []) or []):
            self.draw_marker(p, BLUE, f"C{i + 1}")

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
        h = self.screen.get_height()
        pygame.draw.rect(self.screen, PANEL_BG, (0, 0, PANEL_W, h))
        pygame.draw.line(self.screen, PANEL_LINE, (PANEL_W, 0), (PANEL_W, h), 1)

        y = 14
        self.screen.blit(self.font_big.render("Pynalty", True, GREEN), (16, y))
        y += 30

        done_count = sum(1 for i in range(len(self.events)) if self.step_status(i)[0])
        ratio = done_count / len(self.events)
        pygame.draw.rect(self.screen, (48, 56, 66), (16, y, PANEL_W - 32, 10), border_radius=5)
        pygame.draw.rect(
            self.screen, GREEN, (16, y, int((PANEL_W - 32) * ratio), 10), border_radius=5
        )
        y += 16
        self.screen.blit(
            self.font_small.render(f"{done_count}/{len(self.events)} steps complete", True, GRAY),
            (16, y),
        )
        y += 22

        for idx, evt in enumerate(self.events):
            done, detail = self.step_status(idx)
            active = idx == self.current_event_idx
            if active:
                pygame.draw.rect(
                    self.screen, (44, 66, 50), (8, y - 3, PANEL_W - 16, 38), border_radius=5
                )
            mark = "x" if done else " "
            color = GREEN if done else (WHITE if active else GRAY)
            self.screen.blit(self.font.render(f"[{mark}] {evt.name}", True, color), (16, y))
            self.screen.blit(self.font_small.render(detail, True, GRAY), (34, y + 18))
            y += 38

        y += 8
        pygame.draw.line(self.screen, PANEL_LINE, (12, y), (PANEL_W - 12, y), 1)
        y += 12

        evt = self.current_event
        if evt:
            self.screen.blit(self.font.render("What to do now", True, YELLOW), (16, y))
            y += 22
            for line in _wrap(self.micro_prompt(), self.font_small, PANEL_W - 32):
                self.screen.blit(self.font_small.render(line, True, WHITE), (16, y))
                y += 16
            y += 6
            for line in _wrap(evt.instructions, self.font_small, PANEL_W - 32):
                self.screen.blit(self.font_small.render(line, True, GRAY), (16, y))
                y += 15
            y += 8
            self.screen.blit(self.font.render("Why it matters", True, YELLOW), (16, y))
            y += 22
            for line in _wrap(evt.why, self.font_small, PANEL_W - 32):
                self.screen.blit(self.font_small.render(line, True, GRAY), (16, y))
                y += 16

        warnings = self.validation_warnings()
        if warnings:
            y += 10
            self.screen.blit(self.font.render("Check", True, AMBER), (16, y))
            y += 20
            for w in warnings[:4]:
                for line in _wrap(f"- {w}", self.font_small, PANEL_W - 32):
                    self.screen.blit(self.font_small.render(line, True, AMBER), (16, y))
                    y += 15

        if self.last_results:
            y = min(y + 12, h - 150)
            pygame.draw.line(self.screen, PANEL_LINE, (12, y), (PANEL_W - 12, y), 1)
            y += 10
            r = self.last_results
            lines = [
                f"Ball {r.get('vel_kmh', 0):.1f} km/h ({r.get('vel_ms', 0):.1f} m/s)",
                f"Flight {r.get('flight_time_s', 0):.3f} s over {r.get('dist', 0):.2f} m",
                f"Zone {r.get('zone_label', '-')}",
                f"Reaction {r.get('gk_response_time', 0):+.3f} s",
                f"Dive {r.get('gk_dist', 0):.2f} m at {r.get('gk_vel_ms', 0):.2f} m/s",
                f"Gap to ball {r.get('gap_m', float('nan')):.2f} m ({r.get('gap_source', '')})",
                f"Result: {r.get('shot_outcome_label', r.get('shot_outcome', '-'))}",
                r.get("verdict_label", ""),
            ]
            for line in lines:
                if not line:
                    continue
                for chunk in _wrap(line, self.font_small, PANEL_W - 32):
                    self.screen.blit(self.font_small.render(chunk, True, WHITE), (16, y))
                    y += 15

    def draw_bottom_bar(self):
        w, h = self.screen.get_size()
        top = h - BOTTOM_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, top, w, BOTTOM_H))
        pygame.draw.line(self.screen, PANEL_LINE, (0, top), (w, top), 1)

        margin = 20
        slider_w = w - margin * 2
        slider_y = top + 18
        pygame.draw.rect(
            self.screen, (60, 70, 82), (margin, slider_y, slider_w, 6), border_radius=3
        )

        if self.total_frames > 1:
            for evt, color in (
                (self.step("gk_move"), VIOLET),
                (self.step("kick"), GREEN),
                (self.step("goal"), RED),
            ):
                if evt.frame_idx != -1:
                    x = margin + int(evt.frame_idx / (self.total_frames - 1) * slider_w)
                    pygame.draw.rect(self.screen, color, (x - 1, slider_y - 6, 3, 18))
            handle = margin + int(self.current_frame_idx / max(1, self.total_frames - 1) * slider_w)
            pygame.draw.circle(self.screen, WHITE, (handle, slider_y + 3), 8)

        info = (
            f"Frame {self.current_frame_idx}/{max(0, self.total_frames - 1)}  |  "
            f"{self.fps:.2f} fps  |  zoom {self.zoom:.2f}x"
        )
        self.screen.blit(self.font_small.render(info, True, GRAY), (margin, slider_y + 16))

        bw, bh, gap = 118, 30, 8
        bx = margin
        by = top + BOTTOM_H - bh - 12
        mouse = pygame.mouse.get_pos()
        for btn in self.buttons:
            btn.rect = pygame.Rect(bx, by, bw, bh)
            btn.draw(self.screen, self.font_small, btn.rect.collidepoint(mouse))
            bx += bw + gap
            if bx + bw > w - margin:
                break

    def draw_feedback(self):
        surf = self.font_big.render(self.feedback_msg, True, YELLOW)
        pad = 18
        w, h = surf.get_width() + pad * 2, surf.get_height() + pad * 2
        cx, cy = self.screen.get_width() // 2, self.screen.get_height() // 2
        box = pygame.Surface((w, h), pygame.SRCALPHA)
        box.fill((0, 0, 0, 190))
        self.screen.blit(box, (cx - w // 2, cy - h // 2))
        self.screen.blit(surf, (cx - surf.get_width() // 2, cy - surf.get_height() // 2))

    def draw_wizard(self):
        w, h = self.screen.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 225))
        self.screen.blit(veil, (0, 0))

        x, y = 60, 40
        self.screen.blit(
            self.font_big.render("Pynalty - guided penalty analysis", True, GREEN), (x, y)
        )
        y += 34
        intro = (
            "Work through the seven steps below. Steps 1-4 are required for the metrics; "
            "steps 5-7 add the trajectory replay, the pose kinematics and the saveability verdict."
        )
        for line in _wrap(intro, self.font, w - 2 * x):
            self.screen.blit(self.font.render(line, True, GRAY), (x, y))
            y += 20
        y += 14

        for i, spec in enumerate(STEP_SPECS):
            self.screen.blit(self.font.render(f"{i + 1}. {spec['name']}", True, YELLOW), (x, y))
            y += 20
            for line in _wrap(spec["how"], self.font_small, w - 2 * x - 20):
                self.screen.blit(self.font_small.render(line, True, WHITE), (x + 20, y))
                y += 15
            for line in _wrap(spec["why"], self.font_small, w - 2 * x - 20):
                self.screen.blit(self.font_small.render(line, True, GRAY), (x + 20, y))
                y += 15
            y += 6

        y = min(y + 10, h - 40)
        self.screen.blit(
            self.font.render(
                "TAB moves between steps  |  H opens the shortcut list  |  press any key to start",
                True,
                GREEN,
            ),
            (x, y),
        )

    def draw_help_overlay(self):
        w, h = self.screen.get_size()
        veil = pygame.Surface((w, h), pygame.SRCALPHA)
        veil.fill((0, 0, 0, 220))
        self.screen.blit(veil, (0, 0))

        left = [
            "PYNALTY SHORTCUTS",
            "",
            "Workflow",
            "  TAB / Shift+TAB   next / previous step",
            "  1 . . 7           jump (forward blocked if incomplete)",
            "  ENTER             lock frame (step 1) / body dialog (7)",
            "  G D M W           goal / save / miss / woodwork",
            "  O                 reorder calibration corners",
            "",
            "Video",
            "  SPACE             play / pause",
            "  Left / Right      one frame",
            "  Up / Down         ten frames",
            "  Home / End        first / last frame",
            "  Wheel             zoom at the cursor",
            "  Middle drag       pan",
            "  0                 fit the frame to the window",
        ]
        right = [
            "",
            "",
            "Marking",
            "  Left click        next point (ball click locks frame)",
            "  Right click       remove the last point",
            "  Drag (step 6)     bounding box (optional)",
            "",
            "Actions",
            "  A                 auto-detect ball (optional)",
            "  P                 MediaPipe pose (optional)",
            "  B                 body measures (optional; defaults if skip)",
            "  S                 save the full results package",
            "  L                 load marks from a TOML file",
            "  F                 override the frame rate",
            "  H                 close this help",
            "  ESC               quit",
        ]
        for col, lines in ((70, left), (w // 2 + 20, right)):
            y = 50
            for line in lines:
                color = (
                    YELLOW
                    if line.isupper() and line
                    else (GREEN if line and not line.startswith("  ") else WHITE)
                )
                font = self.font_big if line.isupper() and line else self.font
                self.screen.blit(font.render(line, True, color), (col, y))
                y += 26

    # ---------------------------------------------------------------- actions

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
                self.font_small.render("ENTER confirms, ESC cancels, empty = skip", True, GRAY),
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
                self.flash(f"Not a number: {raw}")
                return 0.0

        height = ask("Goalkeeper stature in metres (e.g. 1.88)", current.gk_height_m)
        if height is None:
            return
        span = ask(
            "Goalkeeper arm span in metres (leave empty for 1.02 x stature)",
            current.gk_arm_span_m,
        )
        if span is None:
            return
        reach = ask(
            "Goalkeeper standing overhead reach in metres (leave empty for 1.25 x stature)",
            current.gk_standing_reach_m,
        )
        if reach is None:
            return
        kicker = ask("Kicker stature in metres (optional)", current.kicker_height_m)
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
            self.flash(f"Keeper {resolved.gk_height_m:.2f} m, span {resolved.gk_arm_span_m:.2f} m")
        self.compute_metrics()

    def ask_fps(self):
        raw = self._text_input("Frame rate in Hz", f"{self.fps:.3f}")
        if raw is None:
            return
        try:
            value = float(raw.strip().replace(",", "."))
        except ValueError:
            self.flash("Not a number")
            return
        if value > 0:
            self.fps = value
            self.flash(f"Frame rate set to {value:.3f} Hz")
            self.compute_metrics()

    def flight_window(self, pad: int = 4) -> tuple[int, int]:
        """Frame range covering the whole event, with a little padding."""
        frames = [e.frame_idx for e in self.events if e.frame_idx is not None and e.frame_idx >= 0]
        if not frames:
            return 0, max(0, self.total_frames - 1)
        lo = max(0, min(frames) - pad)
        hi = min(max(0, self.total_frames - 1), max(frames) + pad)
        return lo, hi

    def auto_detect_ball(self):
        """Run YOLO over the flight window and merge with the manual marks."""
        kick, goal = self.step("kick"), self.step("goal")
        if kick.frame_idx == -1 or goal.frame_idx == -1:
            self.flash("Set the contact and goal frames first")
            return

        lo, hi = min(kick.frame_idx, goal.frame_idx), max(kick.frame_idx, goal.frame_idx)
        lo, hi = max(0, lo - 2), min(max(0, self.total_frames - 1), hi + 2)
        _banner("Automatic ball detection", f"frames {lo}-{hi} of {self.video_path}")
        _flush_message(self.screen, self.font_big, "Detecting the ball with YOLO ...")

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
            self.flash("No ball detections - mark the ball by hand")
            return

        # Manual marks always win over an automatic detection on the same frame.
        merged = {p.frame: p for p in detected.points}
        for p in self.ball_path().points:
            if p.source == "manual":
                merged[p.frame] = p
        self.set_ball_path(BallPath(points=list(merged.values())))
        self.flash(f"Ball path: {len(merged)} points")

    def run_pose(self):
        """Run MediaPipe inside the drawn boxes over the flight window."""
        boxes = self.step("boxes").points
        if not boxes:
            self.flash("Draw a bounding box in step 6 first")
            return
        kick, goal = self.step("kick"), self.step("goal")
        if kick.frame_idx == -1 or goal.frame_idx == -1:
            self.flash("Set the contact and goal frames first")
            return

        lo, hi = self.flight_window(pad=6)
        _banner("Pose estimation", f"frames {lo}-{hi}, boxes: {', '.join(boxes)}")

        try:
            from .pynalty_vision import gk_kinematics, kicker_kinematics, pose_from_bbox
        except ImportError:
            from pynalty_vision import gk_kinematics, kicker_kinematics, pose_from_bbox

        self.pose_sequences = {}
        for name in ("Kicker", "GK"):
            box = boxes.get(name)
            if not box or len(box) != 4:
                continue
            _flush_message(self.screen, self.font_big, f"Running MediaPipe on the {name} ...")
            seq = pose_from_bbox(
                self.video_path,
                (box[0], box[1], box[2], box[3]),
                lo,
                hi,
                label=name,
            )
            if not seq.is_empty():
                self.pose_sequences[name] = seq

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
        if self.pose_sequences:
            self.flash(f"Pose ready for {', '.join(self.pose_sequences)}")
        else:
            self.flash("No pose landmarks found - check the boxes")

    def reorder_calibration(self):
        pts = self.calib_points()
        if len(pts) != 4:
            self.flash("Mark all four corners first")
            return
        ordered = order_goal_corners(pts)
        self.step("calibration").points["points"] = [[float(p[0]), float(p[1])] for p in ordered]
        self.flash("Corners reordered: bottom-left, top-left, top-right, bottom-right")
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
                self.font.render("Metrics unavailable: check steps 1-4", True, RED),
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
                if target is None and i < len(self.events):
                    target = self.events[i]
                if target is None:
                    continue
                target.frame_idx = int(entry.get("frame_idx", -1))
                target.points = dict(entry.get("points") or {})
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

        self.current_event_idx = 0
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
        file_path = filedialog.askopenfilename(filetypes=[("TOML files", "*.toml")])
        root.destroy()
        if not file_path:
            return
        try:
            with open(file_path) as fh:
                data = toml.load(fh)
            self.load_from_data(data)
            self.flash(f"Loaded {os.path.basename(file_path)}")
        except Exception as exc:
            print(f"Error loading: {exc}")
            self.flash("Load failed")

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
        _flush_message(self.screen, self.font_big, "Saving snapshots and reports ...")

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
                _flush_message(self.screen, self.font_big, "Rendering the ball path video ...")
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
                _flush_message(self.screen, self.font_big, "Rendering the pose overlay video ...")
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
        if key == "prev":
            self.goto_step(self.current_event_idx - 1, force=True)
        elif key == "next":
            if not self.can_leave_step(self.current_event_idx):
                self.flash("Finish this step before advancing")
            else:
                self.goto_step(self.current_event_idx + 1, force=True)
        elif key == "auto_ball":
            self.auto_detect_ball()
        elif key == "pose":
            self.run_pose()
        elif key == "anthro":
            self.ask_anthro()
        elif key == "save":
            self.flash("All results saved" if self.save_results_package() else "Save failed", 60)
        elif key == "load":
            self.load_toml()
        elif key == "help":
            self.show_help = not self.show_help

    def goto_step(self, idx: int, *, force: bool = False):
        """Move to another step. Forward moves require the current step finished."""
        n = len(self.events)
        target = idx % n
        if not force and target > self.current_event_idx and not self.can_leave_step(
            self.current_event_idx
        ):
            self.flash("Finish this step before advancing")
            return
        self.current_event_idx = target
        evt = self.current_event
        if evt and evt.frame_idx != -1:
            self.seek(evt.frame_idx)

    def _maybe_advance(self):
        """Auto-advance when the current required step just became complete."""
        done, _ = self.step_status(self.current_event_idx)
        if (
            done
            and self.current_event_idx < len(self.events) - 1
            and self.step_is_required(self.current_event_idx)
        ):
            self.flash("Step complete → next")
            self.goto_step(self.current_event_idx + 1, force=True)

    def _set_shot_outcome(self, key: str):
        outcome, label = classify_shot_outcome(key, ball_inside_goal=True)
        # Re-classify with real geometry when metrics exist.
        inside = bool(self.last_results.get("ball_inside_goal", True))
        outcome, label = classify_shot_outcome(key, ball_inside_goal=inside)
        self.shot_outcome = outcome
        self.flash(f"Result: {label}")
        self.compute_metrics()
        self._maybe_advance()

    def _mark_point(self, ix: float, iy: float):
        """Place the next point for the current step."""
        evt = self.current_event
        key = self.current_key

        if key == "gk_move":
            evt.frame_idx = self.current_frame_idx
            self.flash(f"Keeper move frame {self.current_frame_idx}")
            self.compute_metrics()
            self._maybe_advance()
            return

        if key in {"kick", "goal"}:
            which = "contact" if key == "kick" else "line"
            if "Ball" not in evt.points:
                evt.points["Ball"] = [ix, iy]
                evt.frame_idx = self.current_frame_idx  # frame locks with first click
                self.flash(f"Ball @ {which} — frame {evt.frame_idx} locked")
            elif "GK" not in evt.points:
                evt.points["GK"] = [ix, iy]
                if evt.frame_idx == -1:
                    evt.frame_idx = self.current_frame_idx
                self.flash(f"Keeper @ {which} marked")
                if key == "goal" and not self.shot_outcome:
                    self.flash("Now press G=goal, D=save, M=miss or W=woodwork", 90)
            else:
                # Restart marks on this step; keep asking for outcome on goal.
                evt.points = {"Ball": [ix, iy]}
                evt.frame_idx = self.current_frame_idx
                if key == "goal":
                    self.shot_outcome = None
                self.flash(f"Restarted: ball @ {which}, frame {evt.frame_idx}")
            self.compute_metrics()
            if key == "kick" or (key == "goal" and self.shot_outcome):
                self._maybe_advance()
            return

        if key == "calibration":
            pts = evt.points.get("points", []) or []
            if len(pts) < 4:
                pts.append([ix, iy])
                evt.points["points"] = pts
                names = ("bottom-left", "top-left", "top-right", "bottom-right")
                self.flash(f"Corner {len(pts)}/4 ({names[len(pts) - 1]})")
                if len(pts) == 4:
                    evt.frame_idx = self.current_frame_idx
            else:
                self.flash("All four corners set - right-click to remove one")
            self.compute_metrics()
            self._maybe_advance()
            return

        if key == "ball_path":
            path = self.ball_path()
            merged = {p.frame: p for p in path.points}
            merged[self.current_frame_idx] = BallPathPoint(
                frame=self.current_frame_idx, x_px=ix, y_px=iy, source="manual"
            )
            self.set_ball_path(BallPath(points=list(merged.values())))
            self.flash(f"Ball marked on frame {self.current_frame_idx} ({len(merged)} points)")
            self.compute_metrics()
            return

        if key == "anthro":
            self.flash("Press B to type measurements, or Step > to use defaults")
            return

        self.compute_metrics()

    def _undo_point(self):
        evt = self.current_event
        key = self.current_key
        if key == "calibration":
            pts = evt.points.get("points", []) or []
            if pts:
                pts.pop()
                evt.points["points"] = pts
                self.flash(f"Corner removed ({len(pts)}/4)")
        elif key in {"kick", "goal"}:
            for name in ("GK", "Ball"):
                if name in evt.points:
                    del evt.points[name]
                    self.flash(f"{name} mark removed")
                    break
        elif key == "ball_path":
            path = self.ball_path()
            merged = {p.frame: p for p in path.points}
            if self.current_frame_idx in merged:
                del merged[self.current_frame_idx]
                self.flash(f"Ball point on frame {self.current_frame_idx} removed")
            elif merged:
                last = max(merged)
                del merged[last]
                self.flash(f"Ball point on frame {last} removed")
            self.set_ball_path(BallPath(points=list(merged.values())))
        elif key == "boxes":
            for name in ("GK", "Kicker"):
                if name in evt.points:
                    del evt.points[name]
                    self.flash(f"{name} box removed")
                    break
        self.compute_metrics()

    def _handle_keydown(self, event) -> bool:
        """Process a key press. Returns False to quit the loop."""
        if self.show_wizard:
            self.show_wizard = False
            return True

        key = event.key
        if key == pygame.K_RIGHT:
            self.seek(self.current_frame_idx + 1)
        elif key == pygame.K_LEFT:
            self.seek(self.current_frame_idx - 1)
        elif key == pygame.K_UP:
            self.seek(self.current_frame_idx + 10)
        elif key == pygame.K_DOWN:
            self.seek(self.current_frame_idx - 10)
        elif key == pygame.K_HOME:
            self.seek(0)
        elif key == pygame.K_END:
            self.seek(self.total_frames - 1)
        elif key in (pygame.K_RETURN, pygame.K_KP_ENTER):
            evt = self.current_event
            if self.current_key == "anthro":
                self.ask_anthro()
            elif self.current_key == "gk_move" and evt is not None:
                evt.frame_idx = self.current_frame_idx
                self.flash(f"Frame {self.current_frame_idx} set for keeper move")
                self.compute_metrics()
                self._maybe_advance()
            elif evt is not None and self.current_key in {"kick", "goal"}:
                # ENTER still allowed to re-lock the frame before clicks.
                evt.frame_idx = self.current_frame_idx
                self.flash(f"Frame {self.current_frame_idx} locked for this step")
                self.compute_metrics()
            elif evt is not None:
                evt.frame_idx = self.current_frame_idx
                self.flash(
                    f"Frame {self.current_frame_idx} set for step {self.current_event_idx + 1}"
                )
                self.compute_metrics()
        elif key == pygame.K_g:
            self._set_shot_outcome("goal")
        elif key == pygame.K_d:
            self._set_shot_outcome("save")
        elif key == pygame.K_m:
            self._set_shot_outcome("miss")
        elif key == pygame.K_w:
            self._set_shot_outcome("woodwork")
        elif key == pygame.K_TAB:
            direction = -1 if (pygame.key.get_mods() & pygame.KMOD_SHIFT) else 1
            if direction > 0 and not self.can_leave_step(self.current_event_idx):
                self.flash("Finish this step before advancing")
            else:
                self.goto_step(self.current_event_idx + direction, force=direction < 0)
        elif pygame.K_1 <= key <= pygame.K_7:
            target = key - pygame.K_1
            if target > self.current_event_idx and not self.can_leave_step(self.current_event_idx):
                self.flash("Finish this step before jumping ahead")
            else:
                self.goto_step(target, force=target <= self.current_event_idx)
        elif key == pygame.K_SPACE:
            self.playing = not self.playing
        elif key in (pygame.K_PLUS, pygame.K_KP_PLUS, pygame.K_EQUALS):
            area = self.content_rect()
            self.zoom_at(1.1, area.centerx, area.centery)
        elif key in (pygame.K_MINUS, pygame.K_KP_MINUS):
            area = self.content_rect()
            self.zoom_at(1 / 1.1, area.centerx, area.centery)
        elif key == pygame.K_0:
            self.fit_view()
            self.flash("View reset")
        elif key == pygame.K_a:
            self.auto_detect_ball()
        elif key == pygame.K_p:
            self.run_pose()
        elif key == pygame.K_b:
            self.ask_anthro()
        elif key == pygame.K_o:
            self.reorder_calibration()
        elif key == pygame.K_s:
            self.flash("All results saved" if self.save_results_package() else "Save failed", 60)
        elif key == pygame.K_l:
            self.load_toml()
        elif key == pygame.K_h:
            self.show_help = not self.show_help
        elif key == pygame.K_f:
            self.ask_fps()
        elif key == pygame.K_ESCAPE:
            return False
        return True

    def _handle_mousedown(self, event):
        mx, my = event.pos
        if self.show_wizard:
            self.show_wizard = False
            return

        for btn in self.buttons:
            if btn.rect.collidepoint(mx, my):
                if event.button == 1:
                    self._on_button(btn.key)
                return

        w, h = self.screen.get_size()
        slider_zone = pygame.Rect(0, h - BOTTOM_H, w, 44)

        if event.button == 1:
            if slider_zone.collidepoint(mx, my):
                self.start_drag_slider = True
                self._slider_seek(mx)
            elif self.content_rect().collidepoint(mx, my):
                ix, iy = self.screen_to_image_coords(mx, my)
                if self.current_key == "boxes":
                    self.box_drag_start = (ix, iy)
                    self.box_drag_current = (ix, iy)
                else:
                    self._mark_point(ix, iy)
        elif event.button == 3:
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
                self.flash("Box too small - drag a larger rectangle")
                return
            box = [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]
            points = self.step("boxes").points
            name = "Kicker" if "Kicker" not in points else "GK"
            points[name] = box
            self.flash(f"{name} box set - press P to run pose")

    def _slider_seek(self, mx: int):
        w = self.screen.get_width()
        margin = 20
        ratio = float(np.clip((mx - margin) / max(1, w - 2 * margin), 0.0, 1.0))
        self.seek(int(ratio * max(0, self.total_frames - 1)))

    def run(self):
        """Open the window and run the marking loop until the user quits."""
        self._init_pygame()
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
                    self.display_size = event.size
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

            if self.playing and self.current_frame_idx < self.total_frames - 1:
                self.seek(self.current_frame_idx + 1)
                pygame.time.delay(int(1000 / max(1.0, self.fps)))

            self.draw_content()
            clock.tick(60)

        pygame.quit()
        if self.cap:
            self.cap.release()


def load_video_file_dialog():
    root = tk.Tk()
    root.withdraw()
    file_path = filedialog.askopenfilename(
        title="Select Video File", filetypes=[("Video files", "*.mp4 *.avi *.mov *.mkv")]
    )
    root.destroy()
    return file_path


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Pynalty - penalty kick analysis")
    parser.add_argument("-i", "--input", help="Path to input video file")
    parser.add_argument("-o", "--output", help="Path to output directory (optional)")
    parser.add_argument("-c", "--config", help="Path to a data.toml with saved marks")
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
    cli_argv = ["-i", vid_path]
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

    vid_path = None
    if not args.gui:
        if args.input:
            vid_path = args.input
        elif len(sys.argv) > 1 and not sys.argv[1].startswith("-"):
            # Fallback for a bare positional path, e.g. `pynalty.py video.mp4`
            vid_path = sys.argv[1]

    if not vid_path:
        vid_path = load_video_file_dialog()

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
