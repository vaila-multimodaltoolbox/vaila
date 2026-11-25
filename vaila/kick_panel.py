"""
Project: vailá Multimodal Toolbox
Script: kick_panel.py - Step-by-step panel for a FreeKiki kick_analysis folder

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 23 November 2026
Update Date: 24 November 2026
Version: 0.4.19

Description:
    Docked pygame panel (right column of the getpixelvideo window) that walks the
    user through a loaded ``kick_analysis`` run in four steps instead of drawing
    everything at once:

    1. Calibration - how many frames have a DLT3D / solved / approximate camera,
       which source the current frame uses, and the pitch-lines toggle.
    2. Layers - switch on only what is needed (boxes, pose, ball, labels), filter
       people by role / team, tick the people to measure; the ball can be copied
       into marker p0 for editing / AI Track.
    3. Metrics - choose the active person, a metric kind (and a body point), add it
       to the list, or jump to a hand-measure mode (Quick Measure).
    4. Export - mkvis3d (.c3d), Blender (.bvh) and metrics CSV, then open the result
       in the 3D viewer or in Blender.

    The module holds only the panel state and the pygame drawing; it never imports
    getpixelvideo or kick_measure. The host calls :meth:`KickPanel.draw` every
    frame, routes mouse clicks to :meth:`KickPanel.handle_click` and mouse wheel
    events to :meth:`KickPanel.handle_wheel`, and reacts to the returned action
    tuples:

        ("consumed",)                    click handled, nothing else to do
        ("layers",)                      a layer toggle changed (see ``layers``)
        ("selection",)                   selected people / role filter changed
        ("metrics",)                     the metric list changed
        ("measure", "<key>")             start a hand-measure mode (key of MANUAL_MODES)
        ("ball_to_p0",)                  copy the ball track into marker p0
        ("export", "mkvis3d"|"blender"|"metrics"|"all")
        ("open", "mkvis3d"|"blender")

    ``step`` is 1-based (1 = calibration ... 4 = export), matching the tab numbers.
    Text is English or Portuguese according to ``lang`` ("en" / "pt").

Usage:
    panel = KickPanel(people, {"dlt3d": 60, "solved": 64, "approximate": 45}, 169)
    rect = panel.panel_rect(screen.get_width(), video_height)
    panel.draw(screen, rect, font, frame=42, frame_source="solved", metric_text={})
    action = panel.handle_click(event.pos)

License:
    This program is free software: you can redistribute it and/or modify it under
    the terms of the GNU Affero General Public License as published by the Free
    Software Foundation, either version 3 of the License, or (at your option) any
    later version.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import pygame

PANEL_WIDTH = 320
TAB_WIDTH = 28  # width of the strip that reopens a closed panel
STEPS = ("calibration", "layers", "metrics", "export")
LAYERS = ("field", "boxes", "pose", "ball", "labels")  # default: field on, the rest off
METRIC_KINDS = ("speed", "distance_ball", "distance_person", "height_kpt", "jump", "kpt_speed")
BODY_POINTS = (
    "pelvis",
    "head",
    "neck",
    "left_wrist",
    "right_wrist",
    "left_ankle",
    "right_ankle",
)
MANUAL_MODES = (
    ("1", "distance"),
    ("3", "angle"),
    ("4", "velocity"),
    ("7", "height base+top"),
    ("8", "line plane"),
    ("9", "event time"),
    ("0", "spin"),
    (",", "arc"),
    (".", "height line"),
)
ROLE_FILTERS = ("all", "kicker", "keeper", "team A", "team B", "referee", "selected")

_POINT_KINDS = ("height_kpt", "kpt_speed")  # kinds that need a body point
_FIXED_POINT = {"jump": "pelvis"}

# --- colours ---------------------------------------------------------------
_BG = (16, 20, 26, 228)
_BORDER = (78, 90, 104)
_TEXT = (232, 236, 240)
_DIM = (148, 158, 170)
_ACCENT = (36, 118, 118)
_ACCENT_ON = (70, 180, 176)
_BTN = (46, 54, 64)
_BTN_OFF = (30, 34, 40)
_OFF_TEXT = (92, 98, 106)
_GREEN = (84, 204, 104)
_YELLOW = (242, 212, 72)
_ORANGE = (248, 152, 52)
_RED = (240, 110, 110)
_CYAN = (110, 190, 240)
# Camera sources of kick_measure.CameraTrack, in the order the step-1 list shows them.
_SOURCES = ("dlt3d", "solved", "tracked", "approximate")
_SOURCE_COLOURS = {"dlt3d": _GREEN, "solved": _YELLOW, "tracked": _CYAN, "approximate": _ORANGE}
_SOURCE_NAMES = {
    "en": {"dlt3d": "DLT3D", "solved": "solved", "tracked": "tracked", "approximate": "approx"},
    "pt": {
        "dlt3d": "DLT3D",
        "solved": "resolvida",
        "tracked": "rastreada",
        "approximate": "aproximada",
    },
}

# --- texts -----------------------------------------------------------------
_TEXTS: dict[str, dict[str, str]] = {
    "en": {
        "tab1": "1 Calib",
        "tab2": "2 Layers",
        "tab3": "3 Metrics",
        "tab4": "4 Export",
        "hint1": "Check the calibration: the pitch lines should sit on the field. Then go to step 2.",
        "hint2": "Switch on only what you need and tick the people to measure.",
        "hint3": "Pick a person and a metric, then press + Add. Or measure by hand below.",
        "hint3_none": "Select people in step 2 first, then come back here.",
        "hint4": "Export the selected people and metrics, then open them in 3D.",
        "frames_by_source": "Frames by calibration source",
        "frame_now": "Frame {n}",
        "source_help": (
            "solved = refit from field points; tracked = carried by the image motion from "
            "the nearest good frames; approx = copied from another frame, measure with care."
        ),
        "layers": "Layers",
        "layer_field": "Field lines",
        "layer_boxes": "Boxes",
        "layer_pose": "Pose",
        "layer_ball": "Ball",
        "layer_labels": "Labels",
        "show": "Show",
        "ball_to_p0": "Ball -> marker p0 (edit / AI Track)",
        "people": "People {shown}/{total} - selected {sel}",
        "no_people": "(no people for this filter)",
        "no_pose": "no pose",
        "person": "Person",
        "person_none": "(none selected)",
        "metric": "Metric",
        "point": "Body point",
        "point_fixed": "Body point: pelvis (fixed)",
        "add": "+ Add metric",
        "added": "Added metrics ({n})",
        "none_added": "(no metrics yet)",
        "hand": "Measure by hand",
        "export_mkvis3d": "Export mkvis3d (.c3d)",
        "export_blender": "Export Blender (.bvh)",
        "export_metrics": "Export metrics CSV",
        "export_all": "Export ALL",
        "open_mkvis3d": "Open mkvis3d",
        "open_blender": "Open Blender",
        "summary4": "{sel} people selected, {met} metrics",
        "export_all_people": "No one selected: exports every person with a 3D pose",
        "last_export": "Last export",
        "no_export": "No export yet.",
        "tab_closed": "Kick",
        "collapse": "Hide panel",
        "err_none": "Select a person first (step 2).",
        "err_second": "Select a second person.",
    },
    "pt": {
        "tab1": "1 Calib",
        "tab2": "2 Camad.",
        "tab3": "3 Métr.",
        "tab4": "4 Export",
        "hint1": "Confira a calibração: as linhas do campo devem ficar sobre o gramado. Depois vá ao passo 2.",
        "hint2": "Ligue só o que precisa e marque as pessoas a medir.",
        "hint3": "Escolha a pessoa e a métrica e aperte + Adicionar. Ou meça à mão abaixo.",
        "hint3_none": "Selecione pessoas no passo 2 e volte aqui.",
        "hint4": "Exporte as pessoas e métricas escolhidas e abra em 3D.",
        "frames_by_source": "Quadros por fonte de calibração",
        "frame_now": "Quadro {n}",
        "source_help": (
            "resolvida = refeita com pontos do campo; rastreada = levada pelo movimento da imagem "
            "a partir dos quadros bons vizinhos; aproximada = copiada de outro quadro, meça com "
            "cautela."
        ),
        "layers": "Camadas",
        "layer_field": "Linhas do campo",
        "layer_boxes": "Caixas",
        "layer_pose": "Pose",
        "layer_ball": "Bola",
        "layer_labels": "Rótulos",
        "show": "Mostrar",
        "ball_to_p0": "Bola -> marcador p0 (editar / AI Track)",
        "people": "Pessoas {shown}/{total} - marcadas {sel}",
        "no_people": "(nenhuma pessoa neste filtro)",
        "no_pose": "sem pose",
        "person": "Pessoa",
        "person_none": "(nenhuma marcada)",
        "metric": "Métrica",
        "point": "Ponto do corpo",
        "point_fixed": "Ponto do corpo: pelve (fixo)",
        "add": "+ Adicionar métrica",
        "added": "Métricas adicionadas ({n})",
        "none_added": "(nenhuma métrica ainda)",
        "hand": "Medir à mão",
        "export_mkvis3d": "Exportar mkvis3d (.c3d)",
        "export_blender": "Exportar Blender (.bvh)",
        "export_metrics": "Exportar métricas CSV",
        "export_all": "Exportar TUDO",
        "open_mkvis3d": "Abrir mkvis3d",
        "open_blender": "Abrir Blender",
        "summary4": "{sel} pessoas marcadas, {met} métricas",
        "export_all_people": "Ninguém marcado: exporta todas as pessoas com pose 3D",
        "last_export": "Última exportação",
        "no_export": "Nenhuma exportação ainda.",
        "tab_closed": "Kick",
        "collapse": "Ocultar painel",
        "err_none": "Marque uma pessoa primeiro (passo 2).",
        "err_second": "Marque uma segunda pessoa.",
    },
}

_KIND_LABELS = {
    "en": {
        "speed": "Speed (m/s)",
        "distance_ball": "Distance to ball (m)",
        "distance_person": "Distance to person (m)",
        "height_kpt": "Height of body point (m)",
        "jump": "Jump height (m)",
        "kpt_speed": "Speed of body point (m/s)",
    },
    "pt": {
        "speed": "Velocidade (m/s)",
        "distance_ball": "Distância até a bola (m)",
        "distance_person": "Distância até a pessoa (m)",
        "height_kpt": "Altura de ponto do corpo (m)",
        "jump": "Altura do salto (m)",
        "kpt_speed": "Velocidade de ponto do corpo (m/s)",
    },
}
_POINT_LABELS = {
    "en": {
        "pelvis": "pelvis",
        "head": "head",
        "neck": "neck",
        "left_wrist": "L wrist",
        "right_wrist": "R wrist",
        "left_ankle": "L ankle",
        "right_ankle": "R ankle",
    },
    "pt": {
        "pelvis": "pelve",
        "head": "cabeça",
        "neck": "pescoço",
        "left_wrist": "punho esq.",
        "right_wrist": "punho dir.",
        "left_ankle": "tornozelo esq.",
        "right_ankle": "tornozelo dir.",
    },
}
_METRIC_NAMES = {
    "en": {
        "speed": "Speed",
        "distance_ball": "Distance to ball",
        "distance_person": "Distance",
        "height_kpt": "Height",
        "jump": "Jump",
        "kpt_speed": "Point speed",
    },
    "pt": {
        "speed": "Velocidade",
        "distance_ball": "Distância até a bola",
        "distance_person": "Distância",
        "height_kpt": "Altura",
        "jump": "Salto",
        "kpt_speed": "Velocidade do ponto",
    },
}
_ROLE_LABELS = {
    "en": {
        "all": "all",
        "kicker": "kicker",
        "keeper": "keeper",
        "team A": "team A",
        "team B": "team B",
        "referee": "referee",
        "selected": "selected",
        "other": "other",
    },
    "pt": {
        "all": "todos",
        "kicker": "batedor",
        "keeper": "goleiro",
        "team A": "time A",
        "team B": "time B",
        "referee": "árbitro",
        "selected": "marcados",
        "other": "outro",
    },
}
_MANUAL_LABELS = {
    "en": {
        "distance": "distance",
        "angle": "angle",
        "velocity": "velocity",
        "height base+top": "height b+t",
        "line plane": "line plane",
        "event time": "event time",
        "spin": "spin",
        "arc": "arc",
        "height line": "height line",
    },
    "pt": {
        "distance": "distância",
        "angle": "ângulo",
        "velocity": "velocidade",
        "height base+top": "altura b+t",
        "line plane": "plano linha",
        "event time": "tempo evento",
        "spin": "giro",
        "arc": "arco",
        "height line": "linha altura",
    },
}


@dataclass
class PanelPerson:
    """One person of the run as the panel lists it."""

    id: int
    role: str = "other"
    team: str = ""
    first: int = 0
    last: int = 0
    n: int = 0
    has_pose: bool = False


@dataclass(frozen=True)
class MetricRequest:
    """A metric the user added: what to measure and on whom."""

    kind: str
    person: int
    other: int | None = None
    point: str | None = None

    def key(self) -> str:
        """Stable id, e.g. ``speed:21``, ``height_kpt:21:right_wrist``, ``distance_person:21:23``."""
        parts = [self.kind, str(self.person)]
        if self.other is not None:
            parts.append(str(self.other))
        if self.point:
            parts.append(self.point)
        return ":".join(parts)


class KickPanel:
    """State + pygame drawing of the kick_analysis step panel (see module docstring)."""

    PAD = 8

    def __init__(
        self,
        people: list[PanelPerson] | tuple[PanelPerson, ...],
        calib_counts: dict[str, int],
        n_frames: int,
        *,
        lang: str = "en",
        has_pose: bool = True,
        has_ball: bool = True,
    ) -> None:
        self.people: list[PanelPerson] = sorted(people, key=lambda p: p.id)
        self.calib_counts: dict[str, int] = dict(calib_counts)
        self.n_frames = int(n_frames)
        self.lang = "pt" if lang == "pt" else "en"
        self.has_pose = bool(has_pose)
        self.has_ball = bool(has_ball)

        self.is_open: bool = True
        self.step: int = 1  # 1-based: 1 calibration, 2 layers, 3 metrics, 4 export
        self.layers: dict[str, bool] = {name: name == "field" for name in LAYERS}
        self.role_filter: str = "all"
        self.selected: set[int] = set()
        self.active_person: int | None = None
        self.pending_kind: str = METRIC_KINDS[0]
        self.pending_point: str = "pelvis"
        self.metrics: list[MetricRequest] = []
        self.last_export: str = ""
        self.message: str = ""

        # Filled by draw(): public so tests and the host can find the controls.
        self.hit_rects: dict[str, pygame.Rect] = {}
        self.scroll_areas: dict[str, pygame.Rect] = {}
        self.scroll: dict[str, int] = {"people": 0, "kinds": 0, "metrics": 0}
        self.disabled: set[str] = set()
        self._rect: pygame.Rect | None = None
        self._ensure_kind = True  # scroll the metric-kind list to the pending kind once
        self._scroll_info: dict[str, tuple[int, int]] = {}  # name -> (content_h, row_h)

        # Per-frame drawing context and caches (bounded).
        self._scr: pygame.Surface | None = None
        self._font: pygame.font.Font | None = None
        self._lh = 16
        self._bottom = 0
        self._bg_cache: pygame.Surface | None = None
        self._text_cache: dict[tuple, pygame.Surface] = {}
        self._fit_cache: dict[tuple, str] = {}
        self._wrap_cache: dict[tuple, list[str]] = {}

    # ------------------------------------------------------------------ state
    def person_visible(self, pid: int) -> bool:
        """True when the person passes the role filter (and the "selected" filter)."""
        person = next((p for p in self.people if p.id == pid), None)
        if person is None:
            return False
        flt = self.role_filter
        if flt == "all":
            return True
        if flt == "selected":
            return pid in self.selected
        if flt == "kicker":
            return person.role == "kicker"
        if flt == "keeper":
            return person.role == "keeper"
        if flt == "referee":
            return person.role == "referee"
        if flt == "team A":
            return person.team == "A"
        if flt == "team B":
            return person.team == "B"
        return True

    def toggle_person(self, pid: int) -> None:
        """Select / unselect a person; the first one selected becomes the active person."""
        if pid in self.selected:
            self.selected.discard(pid)
            if self.active_person == pid:
                self.active_person = min(self.selected) if self.selected else None
        else:
            self.selected.add(pid)
            if self.active_person is None or self.active_person not in self.selected:
                self.active_person = pid

    def add_metric(self) -> MetricRequest:
        """Add the pending metric for the active person; raises ValueError when impossible."""
        if self.active_person is None:
            raise ValueError(self._t("err_none"))
        kind = self.pending_kind
        if kind not in METRIC_KINDS:
            raise ValueError(f"unknown metric kind: {kind}")
        other: int | None = None
        point: str | None = None
        if kind == "distance_person":
            others = sorted(self.selected - {self.active_person})
            if not others:
                raise ValueError(self._t("err_second"))
            other = others[0]
        if kind in _POINT_KINDS:
            if self.pending_point not in BODY_POINTS:
                raise ValueError(f"unknown body point: {self.pending_point}")
            point = self.pending_point
        elif kind in _FIXED_POINT:
            point = _FIXED_POINT[kind]
        req = MetricRequest(kind, self.active_person, other, point)
        if all(m.key() != req.key() for m in self.metrics):
            self.metrics.append(req)
        return req

    def metric_label(self, req: MetricRequest) -> str:
        """Readable label of a metric in the panel language, e.g. ``Height · R wrist · ID 21``."""
        name = _METRIC_NAMES[self.lang].get(req.kind, req.kind)
        if req.kind == "distance_person" and req.other is not None:
            return f"{name} ID {req.person} <-> ID {req.other}"
        parts = [name]
        if req.point and req.kind in _POINT_KINDS:
            parts.append(_POINT_LABELS[self.lang].get(req.point, req.point))
        parts.append(f"ID {req.person}")
        return " · ".join(parts)

    def remove_metric(self, key: str) -> None:
        self.metrics = [m for m in self.metrics if m.key() != key]

    # --------------------------------------------------------------- geometry
    def panel_rect(self, screen_w: int, video_h: int) -> pygame.Rect:
        """Right column when open; a 28 px reopen tab at the right edge when closed."""
        if self.is_open:
            w = min(PANEL_WIDTH, int(screen_w))
            rect = pygame.Rect(int(screen_w) - w, 0, w, int(video_h))
        else:
            rect = pygame.Rect(int(screen_w) - TAB_WIDTH, 8, TAB_WIDTH, min(120, int(video_h) - 8))
        self._rect = rect
        return rect

    # ------------------------------------------------------------ text helpers
    def _t(self, key: str, **kw) -> str:
        text = _TEXTS[self.lang].get(key, key)
        return text.format(**kw) if kw else text

    def _txt(self, s: str, color: tuple, rotate: int = 0) -> pygame.Surface:
        assert self._font is not None
        key = (id(self._font), s, color, rotate)
        surf = self._text_cache.get(key)
        if surf is None:
            if len(self._text_cache) > 800:
                self._text_cache.clear()
            surf = self._font.render(s, True, color)
            if rotate:
                surf = pygame.transform.rotate(surf, rotate)
            self._text_cache[key] = surf
        return surf

    def _fit(self, s: str, width: int, *, keep_end: bool = False) -> str:
        """Truncate ``s`` with an ellipsis so it is at most ``width`` px wide."""
        assert self._font is not None
        key = (id(self._font), s, width, keep_end)
        got = self._fit_cache.get(key)
        if got is not None:
            return got
        font = self._font
        if font.size(s)[0] <= width:
            out = s
        else:
            ell = "..."
            lo, hi = 0, len(s)
            while lo < hi:  # longest piece that fits with the ellipsis
                mid = (lo + hi + 1) // 2
                piece = (ell + s[-mid:]) if keep_end else (s[:mid] + ell)
                if font.size(piece)[0] <= width:
                    lo = mid
                else:
                    hi = mid - 1
            out = ((ell + s[-lo:]) if keep_end else (s[:lo] + ell)) if lo else ell
        if len(self._fit_cache) > 800:
            self._fit_cache.clear()
        self._fit_cache[key] = out
        return out

    def _wrap(self, s: str, width: int, max_lines: int) -> list[str]:
        assert self._font is not None
        key = (id(self._font), s, width, max_lines)
        got = self._wrap_cache.get(key)
        if got is not None:
            return got
        font = self._font
        lines: list[str] = []
        cur = ""
        for word in s.split():
            trial = f"{cur} {word}".strip()
            if font.size(trial)[0] <= width or not cur:
                cur = trial
            else:
                lines.append(cur)
                cur = word
        if cur:
            lines.append(cur)
        if len(lines) > max_lines:
            lines = lines[:max_lines]
            lines[-1] = self._fit(lines[-1] + " ...", width)
        lines = [self._fit(ln, width) for ln in lines]
        if len(self._wrap_cache) > 400:
            self._wrap_cache.clear()
        self._wrap_cache[key] = lines
        return lines

    def _label(self, x: int, y: int, s: str, color: tuple, width: int) -> int:
        assert self._scr is not None
        self._scr.blit(self._txt(self._fit(s, width), color), (x, y))
        return y + self._lh + 2

    def _para(self, x: int, y: int, width: int, s: str, color: tuple, max_lines: int = 3) -> int:
        assert self._scr is not None
        for line in self._wrap(s, width, max_lines):
            self._scr.blit(self._txt(line, color), (x, y))
            y += self._lh + 1
        return y + 3

    # ------------------------------------------------------------- primitives
    def _reg(self, key: str, r: pygame.Rect, disabled: bool, clip: pygame.Rect | None) -> None:
        r = r.clip(clip) if clip is not None else r.copy()
        if r.w <= 0 or r.h <= 0:
            return
        self.hit_rects[key] = r
        if disabled:
            self.disabled.add(key)

    def _button(
        self,
        key: str,
        r: pygame.Rect,
        label: str,
        *,
        active: bool = False,
        disabled: bool = False,
        clip: pygame.Rect | None = None,
        colour: tuple | None = None,
    ) -> None:
        assert self._scr is not None
        scr = self._scr
        if disabled:
            bg, fg, edge = _BTN_OFF, _OFF_TEXT, _BTN
        elif active:
            bg, fg, edge = colour or _ACCENT, _TEXT, _ACCENT_ON
        else:
            bg, fg, edge = _BTN, _TEXT, _BORDER
        pygame.draw.rect(scr, bg, r, border_radius=3)
        pygame.draw.rect(scr, edge, r, 1, border_radius=3)
        surf = self._txt(self._fit(label, r.w - 8), fg)
        scr.blit(surf, surf.get_rect(center=r.center))
        self._reg(key, r, disabled, clip)

    def _checkbox(
        self,
        key: str,
        r: pygame.Rect,
        label: str,
        checked: bool,
        *,
        disabled: bool = False,
        clip: pygame.Rect | None = None,
        text_color: tuple | None = None,
    ) -> None:
        assert self._scr is not None
        scr = self._scr
        side = min(14, r.h - 4)
        box = pygame.Rect(r.x + 2, r.y + (r.h - side) // 2, side, side)
        edge = _OFF_TEXT if disabled else _DIM
        pygame.draw.rect(scr, _BTN_OFF if disabled else _BTN, box)
        pygame.draw.rect(scr, edge, box, 1)
        if checked:
            col = _OFF_TEXT if disabled else _GREEN
            pts = [
                (box.x + 3, box.centery),
                (box.x + side // 2 - 1, box.bottom - 4),
                (box.right - 3, box.y + 3),
            ]
            pygame.draw.lines(scr, col, False, pts, 2)
        fg = _OFF_TEXT if disabled else (text_color or _TEXT)
        tx = box.right + 6
        surf = self._txt(self._fit(label, max(8, r.right - tx - 2)), fg)
        scr.blit(surf, (tx, r.y + (r.h - surf.get_height()) // 2))
        self._reg(key, r, disabled, clip)

    def _radio(
        self,
        key: str,
        r: pygame.Rect,
        label: str,
        on: bool,
        *,
        clip: pygame.Rect | None = None,
    ) -> None:
        assert self._scr is not None
        scr = self._scr
        c = (r.x + 9, r.centery)
        pygame.draw.circle(scr, _BTN, c, 6)
        pygame.draw.circle(scr, _DIM, c, 6, 1)
        if on:
            pygame.draw.circle(scr, _ACCENT_ON, c, 3)
        surf = self._txt(self._fit(label, r.w - 24), _TEXT if on else _DIM)
        scr.blit(surf, (r.x + 20, r.y + (r.h - surf.get_height()) // 2))
        self._reg(key, r, False, clip)

    def _chips(
        self,
        y: int,
        x0: int,
        width: int,
        items: list[tuple[str, str, bool, bool]],
    ) -> int:
        """Flow of chips (key, label, active, disabled); returns the y below the last row."""
        row_h = self._lh + 6
        x = x0
        for key, label, active, disabled in items:
            tw = min(width, self._font_width(label) + 14)
            if x + tw > x0 + width and x > x0:
                x = x0
                y += row_h + 3
            self._button(key, pygame.Rect(x, y, tw, row_h), label, active=active, disabled=disabled)
            x += tw + 4
        return y + row_h + 6

    def _font_width(self, s: str) -> int:
        assert self._font is not None
        return self._font.size(s)[0]

    def _scroll_list(
        self,
        name: str,
        area: pygame.Rect,
        row_h: int,
        n_items: int,
        draw_row: Callable[[int, pygame.Rect], None],
    ) -> None:
        """Draw rows [i] of a vertically scrolling list inside ``area`` (clipped, with a bar)."""
        assert self._scr is not None
        scr = self._scr
        content = n_items * row_h
        if content > area.h:  # whole rows only: no half row at the top or the bottom
            area = pygame.Rect(area.x, area.y, area.w, max(row_h, area.h // row_h * row_h))
        limit = max(0, content - area.h)
        off = max(0, min(self.scroll.get(name, 0), limit)) // row_h * row_h
        self.scroll[name] = off
        self.scroll_areas[name] = area.copy()
        self._scroll_info[name] = (content, row_h)
        prev = scr.get_clip()
        scr.set_clip(area.clip(prev))
        bar_w = 5 if content > area.h else 0
        first = off // row_h
        last = min(n_items, (off + area.h) // row_h + 1)
        for i in range(first, last):
            r = pygame.Rect(area.x, area.y + i * row_h - off, area.w - bar_w - 2, row_h)
            draw_row(i, r)
        scr.set_clip(prev)
        if bar_w:
            track = pygame.Rect(area.right - bar_w, area.y, bar_w, area.h)
            pygame.draw.rect(scr, _BTN_OFF, track)
            th = max(14, int(area.h * area.h / content))
            ty = area.y + int((area.h - th) * off / max(1, content - area.h))
            pygame.draw.rect(scr, _DIM, pygame.Rect(track.x, ty, bar_w, th), border_radius=2)

    # ------------------------------------------------------------------- draw
    def draw(
        self,
        screen: pygame.Surface,
        rect: pygame.Rect,
        font: pygame.font.Font,
        *,
        frame: int,
        frame_source: str | None,
        metric_text: dict[str, str],
    ) -> None:
        """Draw the panel (or the reopen tab) inside ``rect``; nothing is drawn outside it."""
        self._rect = rect
        self.hit_rects = {}
        self.scroll_areas = {}
        self._scroll_info = {}
        self.disabled = set()
        self._scr = screen
        self._font = font
        self._lh = font.get_linesize()
        prev_clip = screen.get_clip()
        screen.set_clip(rect.clip(prev_clip))
        try:
            if not self.is_open:
                self._draw_closed(rect)
            else:
                self._draw_open(rect, frame, frame_source, metric_text)
        finally:
            screen.set_clip(prev_clip)

    def _draw_closed(self, rect: pygame.Rect) -> None:
        assert self._scr is not None
        scr = self._scr
        pygame.draw.rect(scr, _ACCENT, rect, border_radius=4)
        pygame.draw.rect(scr, _ACCENT_ON, rect, 1, border_radius=4)
        cx = rect.centerx
        top = rect.y + 8
        pygame.draw.polygon(scr, _TEXT, [(cx + 4, top), (cx + 4, top + 10), (cx - 4, top + 5)])
        label = self._txt(self._fit(self._t("tab_closed"), rect.h - 30), _TEXT, rotate=90)
        scr.blit(label, label.get_rect(midtop=(cx, top + 16)))
        self._reg("reopen", rect, False, None)

    def _background(self, rect: pygame.Rect) -> None:
        assert self._scr is not None
        if self._bg_cache is None or self._bg_cache.get_size() != rect.size:
            self._bg_cache = pygame.Surface(rect.size, pygame.SRCALPHA)
            self._bg_cache.fill(_BG)
            pygame.draw.rect(self._bg_cache, _BORDER, self._bg_cache.get_rect(), 1)
        self._scr.blit(self._bg_cache, rect.topleft)

    def _draw_open(
        self,
        rect: pygame.Rect,
        frame: int,
        frame_source: str | None,
        metric_text: dict[str, str],
    ) -> None:
        assert self._scr is not None
        scr = self._scr
        pad = self.PAD
        self._background(rect)

        # Tabs + collapse control.
        tab_h = self._lh + 8
        coll_w = 22
        tabs_w = rect.w - 2 * pad - coll_w - 4
        names = [self._t(f"tab{i + 1}") for i in range(4)]
        want = [self._font_width(n) + 8 for n in names]
        spare = tabs_w - sum(want)
        widths = (
            [wd + spare // 4 for wd in want]
            if spare >= 0
            else [int(tabs_w * wd / sum(want)) for wd in want]
        )
        tx = rect.x + pad
        for i in range(4):
            r = pygame.Rect(tx, rect.y + pad, widths[i] - 2, tab_h)
            self._button(f"tab:{i + 1}", r, names[i], active=self.step == i + 1)
            tx += widths[i]
        cr = pygame.Rect(rect.right - pad - coll_w, rect.y + pad, coll_w, tab_h)
        pygame.draw.rect(scr, _BTN, cr, border_radius=3)
        pygame.draw.rect(scr, _BORDER, cr, 1, border_radius=3)
        c = cr.center
        pygame.draw.polygon(
            scr, _TEXT, [(c[0] - 3, c[1] - 5), (c[0] - 3, c[1] + 5), (c[0] + 4, c[1])]
        )
        self._reg("collapse", cr, False, None)

        x0 = rect.x + pad
        w = rect.w - 2 * pad
        y = rect.y + pad + tab_h + pad

        # Message strip at the bottom (errors of add_metric etc.).
        self._bottom = rect.bottom - pad
        if self.message:
            lines = self._wrap(self.message, w, 2)
            strip = len(lines) * (self._lh + 1) + 2
            self._bottom -= strip
            yy = self._bottom + 2
            for line in lines:
                scr.blit(self._txt(line, _RED), (x0, yy))
                yy += self._lh + 1

        if self.step == 2:
            self._page_layers(x0, y, w)
        elif self.step == 3:
            self._page_metrics(x0, y, w, metric_text)
        elif self.step == 4:
            self._page_export(x0, y, w)
        else:
            self._page_calibration(x0, y, w, frame, frame_source)

    # ------------------------------------------------------------------ pages
    def _page_calibration(
        self, x0: int, y: int, w: int, frame: int = 0, frame_source: str | None = None
    ) -> None:
        assert self._scr is not None
        scr = self._scr
        y = self._para(x0, y, w, self._t("hint1"), _TEXT, 4)
        y = self._label(x0, y + 2, self._t("frames_by_source"), _DIM, w)
        total = max(1, self.n_frames)
        for src in _SOURCES:
            n = int(self.calib_counts.get(src, 0))
            col = _SOURCE_COLOURS[src]
            pygame.draw.rect(scr, col, pygame.Rect(x0 + 2, y + (self._lh - 10) // 2, 10, 10))
            text = f"{_SOURCE_NAMES[self.lang][src]}: {n}/{self.n_frames}"
            scr.blit(self._txt(self._fit(text, w - 20), _TEXT), (x0 + 18, y))
            y += self._lh + 3
        # Proportion bar.
        bar = pygame.Rect(x0, y + 2, w, 8)
        pygame.draw.rect(scr, _BTN_OFF, bar)
        bx = bar.x
        for src in _SOURCES:
            seg = int(round(bar.w * int(self.calib_counts.get(src, 0)) / total))
            seg = max(0, min(seg, bar.right - bx))
            if seg:
                pygame.draw.rect(scr, _SOURCE_COLOURS[src], pygame.Rect(bx, bar.y, seg, bar.h))
                bx += seg
        pygame.draw.rect(scr, _BORDER, bar, 1)
        y = bar.bottom + 10

        src_name = _SOURCE_NAMES[self.lang].get(frame_source or "", frame_source or "-")
        src_col = _SOURCE_COLOURS.get(frame_source or "", _DIM)
        head = self._t("frame_now", n=frame) + ": "
        scr.blit(self._txt(head, _TEXT), (x0, y))
        hw = self._font_width(head)
        scr.blit(self._txt(self._fit(src_name, max(8, w - hw)), src_col), (x0 + hw, y))
        y += self._lh + 6
        y = self._para(x0, y, w, self._t("source_help"), _DIM, 3)
        y += 4
        self._checkbox(
            "layer:field",
            pygame.Rect(x0, y, w, self._lh + 8),
            self._t("layer_field"),
            self.layers["field"],
        )

    def _role_text(self, role: str) -> str:
        return _ROLE_LABELS[self.lang].get(role, role)

    def _person_label(self, p: PanelPerson) -> str:
        parts = [f"ID {p.id}", self._role_text(p.role)]
        if p.team:
            parts.append(p.team)
        parts.append(f"{p.first}–{p.last}")
        return " · ".join(parts)

    def _page_layers(self, x0: int, y: int, w: int) -> None:
        y = self._para(x0, y, w, self._t("hint2"), _TEXT, 3)
        half = w // 2
        row_h = self._lh + 8
        grid = (
            ("boxes", True),
            ("pose", self.has_pose),
            ("ball", self.has_ball),
            ("labels", True),
        )
        for i, (name, enabled) in enumerate(grid):
            r = pygame.Rect(x0 + (i % 2) * half, y + (i // 2) * row_h, half - 2, row_h)
            self._checkbox(
                f"layer:{name}",
                r,
                self._t(f"layer_{name}"),
                self.layers[name],
                disabled=not enabled,
            )
        y += 2 * row_h + 4
        items = [
            (f"role:{r}", _ROLE_LABELS[self.lang][r], self.role_filter == r, False)
            for r in ROLE_FILTERS
        ]
        y = self._chips(y, x0, w, items)
        self._button(
            "ball_to_p0",
            pygame.Rect(x0, y, w, self._lh + 8),
            self._t("ball_to_p0"),
            disabled=not self.has_ball,
        )
        y += self._lh + 8 + 8
        shown = [p for p in self.people if self.person_visible(p.id)]
        y = self._label(
            x0,
            y,
            self._t("people", shown=len(shown), total=len(self.people), sel=len(self.selected)),
            _DIM,
            w,
        )
        area = pygame.Rect(x0, y, w, max(self._lh + 4, self._bottom - y))
        if not shown:
            self._label(x0, y, self._t("no_people"), _DIM, w)
            return
        row = self._lh + 6

        def draw_row(i: int, r: pygame.Rect) -> None:
            p = shown[i]
            col = _ACCENT_ON if p.id == self.active_person else (_TEXT if p.has_pose else _DIM)
            self._checkbox(
                f"person:{p.id}",
                r,
                self._person_label(p),
                p.id in self.selected,
                clip=self.scroll_areas["people"],
                text_color=col,
            )

        self._scroll_list("people", area, row, len(shown), draw_row)

    def _page_metrics(self, x0: int, y: int, w: int, metric_text: dict[str, str]) -> None:
        assert self._scr is not None
        scr = self._scr
        sel = sorted(self.selected)
        y = self._para(x0, y, w, self._t("hint3" if sel else "hint3_none"), _TEXT, 3)
        btn_h = self._lh + 8
        # Active person chooser.
        if sel:
            act = self.active_person if self.active_person in self.selected else sel[0]
            person = next((p for p in self.people if p.id == act), None)
            label = f"{self._t('person')}: ID {act}"
            if person is not None:
                label += f" · {self._role_text(person.role)}"
        else:
            label = f"{self._t('person')}: {self._t('person_none')}"
        two = len(sel) > 1
        self._button("actor_prev", pygame.Rect(x0, y, 24, btn_h), "<", disabled=not two)
        self._button("actor_next", pygame.Rect(x0 + w - 24, y, 24, btn_h), ">", disabled=not two)
        text = self._txt(self._fit(label, w - 60), _TEXT if sel else _DIM)
        scr.blit(text, text.get_rect(center=(x0 + w // 2, y + btn_h // 2)))
        y += btn_h + 6

        # Bottom block anchored to the panel bottom: "Measure by hand".
        cols = 3
        bw = (w - (cols - 1) * 4) // cols
        n_rows = (len(MANUAL_MODES) + cols - 1) // cols
        grid_h = n_rows * (btn_h + 3)
        manual_top = self._bottom - grid_h - (self._lh + 4)
        # Fixed pieces between the list of kinds and the manual block.
        chips_h = 0
        points_text = ""
        if self.pending_kind in _POINT_KINDS:
            chips_h = self._lh + 2
            chip_rows = self._chip_rows(w)
            chips_h += chip_rows * (self._lh + 9) + 6
        elif self.pending_kind in _FIXED_POINT:
            points_text = self._t("point_fixed")
            chips_h = self._lh + 4
        add_h = btn_h + 6
        row = self._lh + 5
        mrow = 2 * self._lh + 6
        label_h = self._lh + 3
        avail = manual_top - y - chips_h - add_h - 2 * label_h - 6
        if self.metrics:
            m_min = min(len(self.metrics), 2) * mrow
            m_full = len(self.metrics) * mrow
        else:
            m_min = m_full = self._lh + 4
        k_full = len(METRIC_KINDS) * row
        k_h = max(2 * row, min(k_full, avail - m_min))
        if k_h < k_full:
            k_h = k_h // row * row
        m_h = max(self._lh + 4, min(m_full, avail - k_h))

        y = self._label(x0, y, self._t("metric"), _DIM, w)
        k_area = pygame.Rect(x0, y, w, k_h)
        labels = _KIND_LABELS[self.lang]

        def draw_kind(i: int, r: pygame.Rect) -> None:
            kind = METRIC_KINDS[i]
            self._radio(
                f"kind:{kind}",
                r,
                labels[kind],
                kind == self.pending_kind,
                clip=self.scroll_areas["kinds"],
            )

        if self._ensure_kind:
            self._ensure_kind = False
            idx = METRIC_KINDS.index(self.pending_kind) if self.pending_kind in METRIC_KINDS else 0
            cur = self.scroll["kinds"]
            vis = max(1, k_h // row)
            first = cur // row
            if idx < first:
                self.scroll["kinds"] = idx * row
            elif idx >= first + vis:
                self.scroll["kinds"] = (idx - vis + 1) * row
        self._scroll_list("kinds", k_area, row, len(METRIC_KINDS), draw_kind)
        y += k_h + 4

        if self.pending_kind in _POINT_KINDS:
            y = self._label(x0, y, self._t("point"), _DIM, w)
            pl = _POINT_LABELS[self.lang]
            y = self._chips(
                y,
                x0,
                w,
                [(f"point:{b}", pl[b], b == self.pending_point, False) for b in BODY_POINTS],
            )
        elif points_text:
            y = self._label(x0, y, points_text, _DIM, w)

        self._button(
            "add_metric", pygame.Rect(x0, y, w, btn_h), self._t("add"), colour=_ACCENT, active=True
        )
        y += add_h

        y = self._label(x0, y, self._t("added", n=len(self.metrics)), _DIM, w)
        m_area = pygame.Rect(x0, y, w, m_h)
        if not self.metrics:
            self._label(x0, y, self._t("none_added"), _DIM, w)
        else:
            reqs = self.metrics

            def draw_metric(i: int, r: pygame.Rect) -> None:
                req = reqs[i]
                key = req.key()
                inner = r.inflate(-2, -2)
                pygame.draw.rect(scr, _BTN_OFF, inner, border_radius=3)
                xr = pygame.Rect(inner.right - 20, inner.y + 2, 18, 18)
                tw = inner.w - 28
                scr.blit(
                    self._txt(self._fit(self.metric_label(req), tw), _TEXT),
                    (inner.x + 4, inner.y + 2),
                )
                val = metric_text.get(key, "")
                scr.blit(
                    self._txt(self._fit(val, inner.w - 8), _GREEN if val else _DIM),
                    (inner.x + 4, inner.y + 2 + self._lh),
                )
                pygame.draw.rect(scr, _BTN, xr, border_radius=3)
                pygame.draw.line(scr, _RED, (xr.x + 5, xr.y + 5), (xr.right - 6, xr.bottom - 6), 2)
                pygame.draw.line(scr, _RED, (xr.x + 5, xr.bottom - 6), (xr.right - 6, xr.y + 5), 2)
                self._reg(f"remove:{key}", xr, False, self.scroll_areas["metrics"])

            self._scroll_list("metrics", m_area, mrow, len(reqs), draw_metric)

        # Measure by hand (bottom-anchored).
        gy = self._label(x0, manual_top, self._t("hand"), _DIM, w)
        names = _MANUAL_LABELS[self.lang]
        for i, (key, name) in enumerate(MANUAL_MODES):
            r = pygame.Rect(x0 + (i % cols) * (bw + 4), gy + (i // cols) * (btn_h + 3), bw, btn_h)
            self._button(f"measure:{key}", r, f"{key} {names.get(name, name)}")

    def _chip_rows(self, width: int) -> int:
        """Rows the body-point chips need (same flow as :meth:`_chips`)."""
        labels = _POINT_LABELS[self.lang]
        rows, x = 1, 0
        for b in BODY_POINTS:
            tw = min(width, self._font_width(labels[b]) + 14)
            if x + tw > width and x > 0:
                rows += 1
                x = 0
            x += tw + 4
        return rows

    def _page_export(self, x0: int, y: int, w: int) -> None:
        y = self._para(x0, y, w, self._t("hint4"), _TEXT, 3)
        y = self._label(
            x0,
            y + 2,
            self._t("summary4", sel=len(self.selected), met=len(self.metrics)),
            _DIM,
            w,
        )
        if not self.selected:
            y = self._para(x0, y, w, self._t("export_all_people"), _YELLOW, 3)
        btn_h = self._lh + 10
        for key in ("mkvis3d", "blender", "metrics", "all"):
            self._button(
                f"export:{key}",
                pygame.Rect(x0, y, w, btn_h),
                self._t(f"export_{key}"),
                colour=_ACCENT,
                active=(key == "all"),
            )
            y += btn_h + 4
        y += 6
        has = bool(self.last_export)
        half = (w - 4) // 2
        self._button(
            "open:mkvis3d",
            pygame.Rect(x0, y, half, btn_h),
            self._t("open_mkvis3d"),
            disabled=not has,
        )
        self._button(
            "open:blender",
            pygame.Rect(x0 + half + 4, y, w - half - 4, btn_h),
            self._t("open_blender"),
            disabled=not has,
        )
        y += btn_h + 10
        y = self._label(x0, y, self._t("last_export"), _DIM, w)
        if has:
            lines = self._wrap_path(
                self.last_export, w, max(1, (self._bottom - y) // (self._lh + 1))
            )
            assert self._scr is not None
            for line in lines:
                self._scr.blit(self._txt(line, _TEXT), (x0, y))
                y += self._lh + 1
        else:
            self._label(x0, y, self._t("no_export"), _DIM, w)

    def _wrap_path(self, path: str, width: int, max_lines: int) -> list[str]:
        """Break a long path into at most ``max_lines`` pieces; the last keeps the file name."""
        max_lines = max(1, min(max_lines, 4))
        lines: list[str] = []
        rest = path
        while rest and len(lines) < max_lines - 1 and self._font_width(rest) > width:
            n = len(rest)
            while n > 1 and self._font_width(rest[:n]) > width:
                n -= 1
            lines.append(rest[:n])
            rest = rest[n:]
        lines.append(self._fit(rest, width, keep_end=True))
        return lines

    # ----------------------------------------------------------------- events
    def handle_click(self, pos: tuple[int, int]) -> tuple[str, ...] | None:
        """Return None when the click is outside the panel, else an action tuple."""
        rect = self._rect
        if rect is None or not rect.collidepoint(pos):
            return None
        hit = None
        for key, r in self.hit_rects.items():
            if r.collidepoint(pos):
                hit = key
                break
        if hit is None:
            return ("consumed",)
        if hit in self.disabled:
            return ("consumed",)
        kind, _, arg = hit.partition(":")
        if kind != "add_metric":
            self.message = ""
        if kind == "reopen":
            self.is_open = True
            return ("consumed",)
        if kind == "collapse":
            self.is_open = False
            return ("consumed",)
        if kind == "tab":
            self.step = int(arg)
            self._ensure_kind = True
            return ("consumed",)
        if kind == "layer":
            self.layers[arg] = not self.layers.get(arg, False)
            return ("layers",)
        if kind == "role":
            self.role_filter = arg
            self.scroll["people"] = 0
            return ("selection",)
        if kind == "person":
            self.toggle_person(int(arg))
            return ("selection",)
        if kind == "actor_prev" or kind == "actor_next":
            self._cycle_active(-1 if kind == "actor_prev" else 1)
            return ("consumed",)
        if kind == "kind":
            self.pending_kind = arg
            self._ensure_kind = True
            return ("consumed",)
        if kind == "point":
            self.pending_point = arg
            return ("consumed",)
        if kind == "add_metric":
            try:
                self.add_metric()
            except ValueError as exc:
                self.message = str(exc)
                return ("consumed",)
            self.message = ""
            return ("metrics",)
        if kind == "remove":
            self.remove_metric(arg)
            return ("metrics",)
        if kind == "measure":
            return ("measure", arg)
        if kind == "ball_to_p0":
            return ("ball_to_p0",)
        if kind == "export":
            return ("export", arg)
        if kind == "open":
            return ("open", arg)
        return ("consumed",)

    def _cycle_active(self, direction: int) -> None:
        sel = sorted(self.selected)
        if not sel:
            return
        if self.active_person in sel:
            idx = (sel.index(self.active_person) + direction) % len(sel)
        else:
            idx = 0
        self.active_person = sel[idx]

    def handle_wheel(self, pos: tuple[int, int], dy: int) -> bool:
        """Scroll the list under ``pos``; True when the event belongs to the panel."""
        rect = self._rect
        if rect is None or not rect.collidepoint(pos):
            return False
        for name, area in self.scroll_areas.items():
            if area.collidepoint(pos):
                content, row_h = self._scroll_info.get(name, (0, 1))
                limit = max(0, content - area.h)
                self.scroll[name] = max(0, min(limit, self.scroll.get(name, 0) - dy * row_h * 2))
                break
        return True
