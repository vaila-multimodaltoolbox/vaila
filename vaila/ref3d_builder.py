"""
================================================================================
REF3D builder - ref3d_builder.py
================================================================================
*vailá* – Multimodal Toolbox
Authors: Prof. Dr. Paulo R. P. Santiago and Rafael L. M. Monteiro
https://github.com/vaila-multimodaltoolbox/vaila
Date: 16 November 2026
Update: 18 November 2026
Version: 0.4.19
Python Version: 3.12.16

Description:
------------
Build the 3D control points of a DLT3D calibration (the ``.ref3d`` file) from the
measures of the calibration object, instead of typing every coordinate:

* **box**   - a volume X x Y x Z; optional marks along each axis add the points of
  the edges at those distances. Order: Z, then Y, then X (bottom layer first).
* **pole**  - a topographic pole / rod with marks at known heights (``0, 0.5, 1``),
  stood at several floor positions (the corners of a rectangle, or a list). Order:
  for each position, the marks from the bottom up.
* **triad** - a trihedron: three arms from one origin, X sideways, Y forward, Z up,
  each with its marks (``X 0.25,0.5 | Y 0.5,1 | Z 0.5,1``). Order: for each position,
  the origin, then the X marks, the Y marks and the Z marks. Like the pole, it can be
  moved over the corners of a rectangle.

The points are numbered p0, p1, ... in that order: click them in the same order in
getpixelvideo (button **REF3D**, which shows each point while you click it).

``fit_and_check`` solves the DLT3D from the clicked pixels and says whether it is a
real camera (square pixels, small reprojection error); a wrong measure in the
template shows up there at once.

The measures can also come from a template TOML filled in by hand
(``ref3d_template.toml``; REF3D N writes one, T loads it):

    uv run --no-sync python -m vaila.ref3d_builder --new-template ref3d_template.toml
    uv run --no-sync python -m vaila.ref3d_builder --template-file ref3d_template.toml \\
        -o calib.ref3d

Usage (no window):
    uv run --no-sync python -m vaila.ref3d_builder --template pole --marks 0,1 \\
        --positions "rect 0.55 x 5" -o calib.ref3d
    uv run --no-sync python -m vaila.ref3d_builder --ref3d calib.ref3d \\
        --pixels calib_markers_1_line.csv -o calib.dlt3d
================================================================================
"""

from __future__ import annotations

import argparse
import math
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

TEMPLATES = ("box", "pole", "triad")
MIN_DLT3D_POINTS = 6


class Ref3dError(ValueError):
    """A template or measure that cannot be turned into control points."""


@dataclass(frozen=True)
class Ref3dPoint:
    index: int
    x: float
    y: float
    z: float
    label: str

    @property
    def xyz(self) -> tuple[float, float, float]:
        return (self.x, self.y, self.z)


@dataclass
class Ref3dTemplate:
    """The measures a template is built from; saved next to the files to redo them."""

    kind: str
    size: tuple[float, float, float] = (0.0, 0.0, 0.0)  # box
    div_x: tuple[float, ...] = ()
    div_y: tuple[float, ...] = ()
    div_z: tuple[float, ...] = ()
    marks: tuple[float, ...] = ()  # pole
    arms_x: tuple[float, ...] = ()  # triad
    arms_y: tuple[float, ...] = ()
    arms_z: tuple[float, ...] = ()
    positions: tuple[tuple[float, float], ...] = field(default_factory=tuple)
    # [videos]: file name of each video -> the floor position the object stood at in it
    video_positions: tuple[tuple[str, tuple[float, float]], ...] = ()

    def points(self) -> list[Ref3dPoint]:
        if self.kind == "box":
            return box_points(*self.size, self.div_x, self.div_y, self.div_z)
        if self.kind == "pole":
            return pole_points(self.marks, self.positions or ((0.0, 0.0),))
        if self.kind == "triad":
            return triad_points(
                self.arms_x, self.arms_y, self.arms_z, self.positions or ((0.0, 0.0),)
            )
        raise Ref3dError(f"unknown template {self.kind!r} (use {', '.join(TEMPLATES)})")

    def describe(self) -> str:
        if self.kind == "box":
            sx, sy, sz = self.size
            return f"box {sx:g} x {sy:g} x {sz:g} m"
        n_pos = len(self.positions) or 1
        videos = f", {len(self.video_positions)} videos" if self.video_positions else ""
        if self.kind == "pole":
            return f"pole, marks {_fmt(self.marks)} m, {n_pos} positions{videos}"
        return (
            f"triad X {_fmt(self.arms_x)} | Y {_fmt(self.arms_y)} | Z {_fmt(self.arms_z)} m, "
            f"{n_pos} positions{videos}"
        )


def _fmt(values) -> str:
    return ", ".join(f"{v:g}" for v in values) or "-"


# --- parsing ------------------------------------------------------------------------


def parse_lengths(text: str, *, name: str = "values") -> tuple[float, ...]:
    """``"0.25, 0.5; 1"`` -> (0.25, 0.5, 1.0). Commas, semicolons or spaces separate."""
    raw = [tok for tok in re.split(r"[,;\s]+", str(text).strip()) if tok]
    try:
        values = tuple(float(tok) for tok in raw)
    except ValueError as exc:
        raise Ref3dError(f"{name}: numbers in metres, like 0.25, 0.5, 1 ({exc})") from exc
    if any(not math.isfinite(v) for v in values):
        raise Ref3dError(f"{name}: only finite numbers")
    return values


def parse_size(text: str) -> tuple[float, float, float]:
    """``"0.55 x 5 x 1"`` or ``"0.55, 5, 1"`` -> (X, Y, Z)."""
    values = parse_lengths(re.sub(r"[xX*]", " ", str(text)), name="size X, Y, Z")
    if len(values) != 3 or min(values) <= 0:
        raise Ref3dError("size: three positive numbers X, Y, Z in metres, like 0.55 x 5 x 1")
    return values


def rectangle_positions(width_x: float, length_y: float) -> tuple[tuple[float, float], ...]:
    """The 4 corners of a W (X) x L (Y) rectangle: near edge left->right, then far edge."""
    if width_x <= 0 or length_y <= 0:
        raise Ref3dError("rectangle: width X and length Y must be positive")
    return ((0.0, 0.0), (width_x, 0.0), (0.0, length_y), (width_x, length_y))


def parse_positions(text: str) -> tuple[tuple[float, float], ...]:
    """Floor positions of the pole / triad.

    ``"rect 0.55 x 5"`` (or ``"0.55 x 5"``) -> the 4 corners of a rectangle X x Y;
    ``"0,0; 0.55,0; 0,5"`` -> those positions, in that order; empty -> one, at 0,0.
    """
    body = str(text).strip()
    if not body:
        return ((0.0, 0.0),)
    rect = re.fullmatch(
        r"(?:rect(?:angle)?\s*)?([-+0-9.eE]+)\s*[xX*]\s*([-+0-9.eE]+)", body, flags=re.I
    )
    if rect:
        return rectangle_positions(float(rect.group(1)), float(rect.group(2)))
    positions = []
    for chunk in (c for c in body.split(";") if c.strip()):
        pair = parse_lengths(chunk, name="position x, y")
        if len(pair) != 2:
            raise Ref3dError(f"position {chunk.strip()!r}: two numbers x, y (like 0.55, 5)")
        positions.append((pair[0], pair[1]))
    return tuple(positions)


# --- templates ----------------------------------------------------------------------


def _axis_stops(size: float, marks: tuple[float, ...]) -> list[float]:
    stops = {0.0, float(size)}
    for m in marks:
        if not 0.0 < m < size:
            raise Ref3dError(f"mark {m:g} is outside 0..{size:g}")
        stops.add(float(m))
    return sorted(stops)


def box_points(
    sx: float,
    sy: float,
    sz: float,
    div_x: tuple[float, ...] = (),
    div_y: tuple[float, ...] = (),
    div_z: tuple[float, ...] = (),
) -> list[Ref3dPoint]:
    """Corners of the volume, plus the points of its edges at the given marks."""
    if min(sx, sy, sz) <= 0:
        raise Ref3dError("box: X, Y and Z must be positive")
    xs, ys, zs = _axis_stops(sx, div_x), _axis_stops(sy, div_y), _axis_stops(sz, div_z)
    points: list[Ref3dPoint] = []
    for z in zs:
        for y in ys:
            for x in xs:
                on_edge = (x in (0.0, sx)) + (y in (0.0, sy)) + (z in (0.0, sz))
                if on_edge >= 2:  # a corner (3) or a point of an edge (2)
                    kind = "corner" if on_edge == 3 else "edge"
                    points.append(Ref3dPoint(len(points), x, y, z, f"{kind} ({x:g}, {y:g}, {z:g})"))
    return points


def pole_points(
    marks: tuple[float, ...], positions: tuple[tuple[float, float], ...]
) -> list[Ref3dPoint]:
    """For each floor position, the pole marks from the bottom up."""
    if not marks:
        raise Ref3dError("pole: give the heights of the marks, like 0, 1")
    if any(m < 0 for m in marks):
        raise Ref3dError("pole: mark heights are measured up from the floor (>= 0)")
    points: list[Ref3dPoint] = []
    for k, (x, y) in enumerate(positions, start=1):
        for z in sorted(marks):
            points.append(Ref3dPoint(len(points), x, y, z, f"pos{k} · mark {z:g} m"))
    return points


def triad_points(
    arms_x: tuple[float, ...],
    arms_y: tuple[float, ...],
    arms_z: tuple[float, ...],
    positions: tuple[tuple[float, float], ...],
) -> list[Ref3dPoint]:
    """For each position: origin, X marks, Y marks, Z marks (distances from the origin)."""
    if not (arms_x or arms_y or arms_z):
        raise Ref3dError("triad: give the marks of at least one arm")
    for name, arm in (("X", arms_x), ("Y", arms_y), ("Z", arms_z)):
        if any(a <= 0 for a in arm):
            raise Ref3dError(f"triad: the {name} marks are distances from the origin (> 0)")
    points: list[Ref3dPoint] = []
    for k, (x0, y0) in enumerate(positions, start=1):
        points.append(Ref3dPoint(len(points), x0, y0, 0.0, f"pos{k} · origin"))
        for a in sorted(arms_x):
            points.append(Ref3dPoint(len(points), x0 + a, y0, 0.0, f"pos{k} · X {a:g} m"))
        for a in sorted(arms_y):
            points.append(Ref3dPoint(len(points), x0, y0 + a, 0.0, f"pos{k} · Y {a:g} m"))
        for a in sorted(arms_z):
            points.append(Ref3dPoint(len(points), x0, y0, a, f"pos{k} · Z {a:g} m"))
    return points


def points_are_coplanar(points: list[Ref3dPoint], rtol: float = 1e-6) -> bool:
    """True when every point lies on one plane (no 3D calibration possible)."""
    xyz = np.array([p.xyz for p in points], dtype=float)
    if len(xyz) < 4:
        return True
    s = np.linalg.svd(xyz - xyz.mean(axis=0), compute_uv=False)
    return bool(s[0] == 0.0 or s[-1] / s[0] < rtol)


# --- sketch -----------------------------------------------------------------------

# Oblique view of a right-handed world (X x Y = Z), seen from the front and above:
# X to the right, Y away from the viewer (up and to the right on the screen), Z up.
SKETCH_AXES = {
    "x": (1.0, 0.0),
    "y": (0.55, -0.40),
    "z": (0.0, -1.0),
}


def sketch_projection(xyz) -> np.ndarray:
    """2D screen coordinates (y down) of world points for the panel's 3D sketch."""
    xyz = np.asarray(xyz, dtype=float).reshape(-1, 3)
    basis = np.array([SKETCH_AXES["x"], SKETCH_AXES["y"], SKETCH_AXES["z"]])
    return xyz @ basis


# --- schematic SVG for the illustrated guide (vaila/help/ref3d_guide.html) ----------

SVG_AXIS_COLOURS = {"x": "#d6453d", "y": "#2f9e44", "z": "#2f6fd6"}
_SVG_INK, _SVG_MUTED, _SVG_PAPER, _SVG_LINE = "#1e293b", "#64748b", "#ffffff", "#cbd5e1"
_SVG_NEXT = "#f59f00"


def _svg_text(x: float, y: float, text: str, *, size: int = 13, fill: str = _SVG_INK, **kw) -> str:
    extra = "".join(f' {k.replace("_", "-")}="{v}"' for k, v in kw.items())
    safe = text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")
    halo = f' stroke="{_SVG_PAPER}" stroke-width="3" paint-order="stroke"'  # legible over lines
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-size="{size}" fill="{fill}"{halo}{extra}>{safe}</text>'
    )


def _svg_arrow_defs() -> str:
    marks = [
        f'<marker id="arrow-{name}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" '
        f'markerHeight="7" orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" '
        f'fill="{colour}"/></marker>'
        for name, colour in (*SVG_AXIS_COLOURS.items(), ("seq", _SVG_MUTED))
    ]
    return "<defs>" + "".join(marks) + "</defs>"


def _template_segments(
    template: Ref3dTemplate, points: list[Ref3dPoint]
) -> tuple[list[tuple[tuple, tuple, str]], list[tuple]]:
    """Object lines (start, end, colour) and the floor positions in the order of use."""
    lines: list[tuple[tuple, tuple, str]] = []
    if template.kind == "box":
        # Edges: consecutive points along one axis that share the other two coordinates.
        xyz = [p.xyz for p in points]
        for axis in range(3):
            others = [a for a in range(3) if a != axis]
            groups: dict[tuple, list[tuple]] = {}
            for q in xyz:
                groups.setdefault(tuple(q[a] for a in others), []).append(q)
            for group in groups.values():
                group.sort(key=lambda q, axis=axis: q[axis])
                lines += [(a, b, _SVG_INK) for a, b in zip(group, group[1:], strict=False)]
        return lines, []
    positions = list(template.positions or ((0.0, 0.0),))
    for x0, y0 in positions:
        if template.kind == "pole":
            lines.append(((x0, y0, 0.0), (x0, y0, max(template.marks)), _SVG_INK))
            continue
        for axis, arm in (("x", template.arms_x), ("y", template.arms_y), ("z", template.arms_z)):
            if arm:
                end = {"x": (x0 + max(arm), y0, 0.0), "y": (x0, y0 + max(arm), 0.0)}.get(
                    axis, (x0, y0, max(arm))
                )
                lines.append(((x0, y0, 0.0), end, SVG_AXIS_COLOURS[axis]))
    return lines, [(x, y, 0.0) for x, y in positions]


def template_svg(
    template: Ref3dTemplate,
    *,
    title: str = "",
    width: int = 560,
    height: int = 380,
    highlight: int | None = None,
    coords: bool = True,
) -> str:
    """A schematic drawing of a template: its lines, every point pN in click order.

    The view is :func:`sketch_projection` (right-handed, X right, Y away, Z up), the
    same as the REF3D panel of getpixelvideo. Pole and triad positions are joined by
    dashed arrows in the order they are used (pos1 -> pos2 -> ...).
    """
    points = template.points()
    lines, floor = _template_segments(template, points)
    xyz = np.array([p.xyz for p in points], dtype=float)
    scr = sketch_projection(np.vstack([xyz, np.zeros((1, 3))]))
    top = 46 if title else 16
    pad_x, pad_bottom = 80.0, 80.0
    lo, hi = scr.min(axis=0), scr.max(axis=0)
    span = np.maximum(hi - lo, 1e-6)
    scale = min((width - 2 * pad_x) / span[0], (height - top - 30 - pad_bottom) / span[1])
    off = np.array(
        [
            (width - span[0] * scale) / 2 - lo[0] * scale,
            top + 30 + (height - top - 30 - pad_bottom - span[1] * scale) / 2 - lo[1] * scale,
        ]
    )

    def at(q) -> tuple[float, float]:
        u, v = sketch_projection(q)[0] * scale + off
        return float(u), float(v)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Segoe UI, system-ui, sans-serif" '
        f'role="img" aria-label="{title or template.describe()}">',
        _svg_arrow_defs(),
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
        f'fill="{_SVG_PAPER}" stroke="{_SVG_LINE}"/>',
    ]
    if title:
        out.append(_svg_text(16, 28, title, size=16, font_weight="600"))
    # Floor outline (pole / triad): dashed arrows in the order of the positions.
    for a, b in zip(floor, floor[1:], strict=False):
        (x1, y1), (x2, y2) = at(a), at(b)
        gap = 12.0 / max(math.hypot(x2 - x1, y2 - y1), 1e-6)  # stop short of the points
        x1, y1, x2, y2 = (
            x1 + (x2 - x1) * gap,
            y1 + (y2 - y1) * gap,
            x2 - (x2 - x1) * gap,
            y2 - (y2 - y1) * gap,
        )
        out.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{_SVG_MUTED}" stroke-width="1.5" stroke-dasharray="6 5" '
            f'marker-end="url(#arrow-seq)"/>'
        )
    for k, q in enumerate(floor, start=1):
        u, v = at(q)
        out.append(_svg_text(u - 14, v + 46, f"pos{k}", size=12, fill=_SVG_MUTED))
    for a, b, colour in lines:
        (x1, y1), (x2, y2) = at(a), at(b)
        width_px = 4 if template.kind == "pole" else 2.5
        out.append(
            f'<line x1="{x1:.1f}" y1="{y1:.1f}" x2="{x2:.1f}" y2="{y2:.1f}" '
            f'stroke="{colour}" stroke-width="{width_px}" stroke-linecap="round"/>'
        )
    centre = np.mean([at(p.xyz) for p in points], axis=0)
    for p in points:
        u, v = at(p.xyz)
        colour = _SVG_NEXT if p.index == highlight else _SVG_INK
        out.append(
            f'<circle cx="{u:.1f}" cy="{v:.1f}" r="{6 if p.index == highlight else 4.5}" '
            f'fill="{colour}" stroke="{_SVG_PAPER}" stroke-width="1.5"/>'
        )
        # Labels on the outer side of the drawing, away from its lines.
        left = u < centre[0] - 1
        anchor, tx = ("end", u - 9) if left else ("start", u + 9)
        ty = v - 8 if v <= centre[1] else v + 16
        out.append(
            _svg_text(
                tx, ty, f"p{p.index}", size=14, fill=colour, font_weight="700", text_anchor=anchor
            )
        )
        if coords:
            out.append(
                _svg_text(
                    tx,
                    ty + (-14 if v <= centre[1] else 13),
                    f"({p.x:g}, {p.y:g}, {p.z:g})",
                    size=11,
                    fill=_SVG_MUTED,
                    text_anchor=anchor,
                )
            )
    out.append(_svg_axes_gizmo(width - 170, height - 22, 34))
    out.append("</svg>")
    return "\n".join(out) + "\n"


def _svg_axes_gizmo(x0: float, y0: float, length: float) -> str:
    """Small X / Y / Z arrows (right-handed) with the X x Y = Z caption."""
    parts = []
    for axis, colour in SVG_AXIS_COLOURS.items():
        dx, dy = SKETCH_AXES[axis]
        norm = math.hypot(dx, dy)
        x1, y1 = x0 + dx / norm * length, y0 + dy / norm * length
        parts.append(
            f'<line x1="{x0:.1f}" y1="{y0:.1f}" x2="{x1:.1f}" y2="{y1:.1f}" stroke="{colour}" '
            f'stroke-width="2.5" marker-end="url(#arrow-{axis})"/>'
        )
        parts.append(
            _svg_text(x1 + 4, y1 + 4, axis.upper(), size=13, fill=colour, font_weight="700")
        )
    parts.append(_svg_text(x0 + length + 26, y0 + 4, "X × Y = Z", size=12, fill=_SVG_MUTED))
    return "\n".join(parts)


_AXES_WORDS = {
    "en": {
        "x": "X  sideways",
        "y": "Y  forward (away from the camera)",
        "z": "Z  up",
        "floor": "floor  Z = 0",
        "origin": "origin (0, 0, 0)",
        "hand1": "Right hand: fingers turn X → Y,",
        "hand2": "thumb = Z (X × Y = Z)",
    },
    "pt": {
        "x": "X  lateral",
        "y": "Y  para frente (longe da câmera)",
        "z": "Z  para cima",
        "floor": "chão  Z = 0",
        "origin": "origem (0, 0, 0)",
        "hand1": "Mão direita: dedos giram X → Y,",
        "hand2": "polegar = Z (X × Y = Z)",
    },
}


def axes_svg(width: int = 560, height: int = 340, title: str = "", lang: str = "en") -> str:
    """The world axes of the calibration: right-handed, origin at a floor corner."""
    words = _AXES_WORDS["pt" if lang == "pt" else "en"]
    o = np.array([width * 0.22, height * 0.72])
    length = min(width, height) * 0.42
    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {width} {height}" '
        f'width="{width}" height="{height}" font-family="Segoe UI, system-ui, sans-serif" '
        f'role="img" aria-label="{title or "X, Y, Z axes, right-handed"}">',
        _svg_arrow_defs(),
        f'<rect x="0.5" y="0.5" width="{width - 1}" height="{height - 1}" rx="10" '
        f'fill="{_SVG_PAPER}" stroke="{_SVG_LINE}"/>',
    ]
    if title:
        out.append(_svg_text(16, 28, title, size=16, font_weight="600"))
    # The floor (Z = 0) as a light parallelogram spanned by X and Y.
    ex = np.array(SKETCH_AXES["x"]) * length * 0.8
    ey = np.array(SKETCH_AXES["y"]) / math.hypot(*SKETCH_AXES["y"]) * length * 0.8
    floor = [o, o + ex, o + ex + ey, o + ey]
    out.append(
        '<polygon points="'
        + " ".join(f"{u:.1f},{v:.1f}" for u, v in floor)
        + f'" fill="#f1f5f9" stroke="{_SVG_LINE}" stroke-dasharray="5 4"/>'
    )
    out.append(_svg_text(*(o + ex * 0.45 + ey * 0.55), words["floor"], size=12, fill=_SVG_MUTED))
    for axis, colour in SVG_AXIS_COLOURS.items():
        d = np.array(SKETCH_AXES[axis]) / math.hypot(*SKETCH_AXES[axis])
        end = o + d * length
        out.append(
            f'<line x1="{o[0]:.1f}" y1="{o[1]:.1f}" x2="{end[0]:.1f}" y2="{end[1]:.1f}" '
            f'stroke="{colour}" stroke-width="4" marker-end="url(#arrow-{axis})"/>'
        )
        tx, ty, anchor = {
            "x": (end[0] + 10, end[1] - 10, "start"),
            "y": (end[0] + 10, end[1] - 4, "start"),
            "z": (end[0] - 40, end[1] - 10, "start"),
        }[axis]
        out.append(
            _svg_text(
                tx, ty, words[axis], size=14, fill=colour, font_weight="700", text_anchor=anchor
            )
        )
    # Turn from X to Y about Z (counter-clockwise seen from above): right-hand rule.
    r = length * 0.30
    a0 = math.atan2(-SKETCH_AXES["x"][1], SKETCH_AXES["x"][0])
    a1 = math.atan2(-SKETCH_AXES["y"][1], SKETCH_AXES["y"][0])
    p0 = o + r * np.array([math.cos(a0), -math.sin(a0)])
    p1 = o + r * np.array([math.cos(a1), -math.sin(a1)])
    out.append(
        f'<path d="M{p0[0]:.1f},{p0[1]:.1f} A{r:.1f},{r:.1f} 0 0 0 {p1[0]:.1f},{p1[1]:.1f}" '
        f'fill="none" stroke="{_SVG_MUTED}" stroke-width="1.8" marker-end="url(#arrow-seq)"/>'
    )
    out.append(_svg_text(o[0] - 14, o[1] + 24, words["origin"], size=12, fill=_SVG_INK))
    out.append(
        _svg_text(
            width - 210,
            height - 46,
            words["hand1"],
            size=12,
            fill=_SVG_MUTED,
        )
    )
    out.append(_svg_text(width - 210, height - 28, words["hand2"], size=12, fill=_SVG_MUTED))
    out.append("</svg>")
    return "\n".join(out) + "\n"


# --- files --------------------------------------------------------------------------


def write_ref3d(points: list[Ref3dPoint], path: str | Path) -> Path:
    """Write the wide REF3D (``frame,p0_x,p0_y,p0_z,...``, one row) read by dlt3d.py."""
    path = Path(path)
    cols, row = ["frame"], [0.0]
    for p in points:
        cols += [f"p{p.index}_x", f"p{p.index}_y", f"p{p.index}_z"]
        row += [p.x, p.y, p.z]
    frame = pd.DataFrame([row], columns=pd.Index(cols))
    frame["frame"] = frame["frame"].astype(int)
    frame.to_csv(path, index=False, float_format="%.6g")
    return path


def read_ref3d_points(path: str | Path) -> list[Ref3dPoint]:
    """Read any REF3D variant dlt3d.py accepts, as points p0..pN."""
    try:
        from .dlt3d import normalize_ref3d_to_format1
    except ImportError:
        from dlt3d import normalize_ref3d_to_format1  # ty: ignore[unresolved-import]

    df = normalize_ref3d_to_format1(str(path), min_points=1)
    if df is None or df.empty:
        raise Ref3dError(f"not a REF3D file dlt3d.py can read: {path}")
    row = df.iloc[0]
    points = []
    for col in df.columns:
        m = re.fullmatch(r"p(\d+)_x", str(col))
        if m is None:
            continue
        i = int(m.group(1))
        xyz = (row[f"p{i}_x"], row[f"p{i}_y"], row[f"p{i}_z"])
        if all(pd.notna(v) for v in xyz):
            points.append(Ref3dPoint(i, *(float(v) for v in xyz), f"p{i}"))
    return sorted(points, key=lambda p: p.index)


TEMPLATE_FILE_NAME = "ref3d_template.toml"

# Keys of each section of a template file, with the comment written next to them.
_SECTION_KEYS: dict[str, tuple[str, ...]] = {
    "pole": ("marks", "rect", "positions"),
    "box": ("size", "div_x", "div_y", "div_z"),
    "triad": ("arms_x", "arms_y", "arms_z", "rect", "positions"),
}
_EXAMPLES: dict[str, Ref3dTemplate] = {
    "pole": Ref3dTemplate(
        "pole", marks=(0.0, 0.25, 0.5, 0.75, 1.0), positions=rectangle_positions(1.4, 5.0)
    ),
    "box": Ref3dTemplate("box", size=(1.0, 5.0, 1.0)),
    "triad": Ref3dTemplate("triad", arms_x=(0.5,), arms_y=(0.5, 1.0), arms_z=(0.5, 1.0)),
}

_HEADER = """\
# ==============================================================================
# vailá REF3D template: the measures of your calibration object, in METRES
# ==============================================================================
#
# WHAT IT IS FOR
#   getpixelvideo must know the 3D place of every point you click. This file
#   holds those measures. Measure the object once, write the numbers here,
#   save, and load the file: no more typing measures in dialogs.
#
# HOW TO USE IT
#   1. Write your measures in the section of your object: [pole], [box], [triad].
#   2. Set `kind` to the name of that section.
#   3. In getpixelvideo press REF3D, then Enter (T loads this file).
#      Click the points in the order REF3D shows: p0, p1, p2, ...
#
# THE AXES (right-handed). The origin (0, 0, 0) is a floor point.
#   X = sideways, to the right     Y = forward, away from the camera
#   Z = up, above the floor
#
#        Y
#        ^
#     5  +-----------+
#        |           |      rect = [1.4, 5]:  X = 1.4 m (width)
#        |           |                        Y = 5 m   (length)
#     0  +-----------+--> X
#        0          1.4
#
# UNITS: every number is in metres. 0.25 means 25 cm. Use a dot, not a comma.
# ==============================================================================
"""


def _toml_list(values) -> str:
    return "[" + ", ".join(f"{float(v):g}" for v in values) + "]"


def _rect_of(positions) -> tuple[float, float] | None:
    """(X, Y) when the positions are the 4 corners of rectangle_positions(X, Y)."""
    if len(positions) != 4:
        return None
    width, length = positions[3]
    if width <= 0 or length <= 0:
        return None
    return (width, length) if tuple(positions) == rectangle_positions(width, length) else None


def _positions_lines(positions, *, empty_means: str) -> list[str]:
    rect = _rect_of(positions)
    if rect is not None:
        x, y = rect
        return [
            f"rect = {_toml_list(rect)}      # X = width (sideways), Y = length (forward)",
            "# The object stands at the 4 corners of that rectangle on the floor:",
            f"#   (0, 0), ({x:g}, 0), (0, {y:g}), ({x:g}, {y:g})",
            "# To use your own floor points instead, delete the rect line and uncomment:",
            f"# positions = [[0, 0], [{x:g}, 0], [0, {y:g}]]   # (x, y) in click order",
        ]
    if not positions or tuple(positions) == ((0.0, 0.0),):
        return [
            "rect = []                 # empty = the object stands only at (0, 0)",
            "# Or list the floor points yourself, in click order, and remove rect:",
            "# positions = [[0, 0], [1.4, 0], [0, 5]]   # (x, y) of each floor position",
        ]
    pairs = ", ".join(f"[{x:g}, {y:g}]" for x, y in positions)
    return [
        "# The object stands at these floor points (x, y), in click order:",
        f"positions = [{pairs}]",
    ]


def _section_lines(template: Ref3dTemplate) -> list[str]:
    if template.kind == "pole":
        return [
            "[pole]   # a rod with marks, stood on the floor at several positions",
            "# Heights of the marks on the rod, from the floor (0) upwards, in metres:",
            f"marks = {_toml_list(template.marks)}   # 0 = at the floor, 1 = 1 m above it",
            *_positions_lines(template.positions, empty_means="only at (0, 0)"),
        ]
    if template.kind == "box":
        return [
            "[box]   # a solid box X x Y x Z; the marks are lines drawn on its faces",
            f"size = {_toml_list(template.size)}   # X (width), Y (length), Z (height)",
            "# Extra marks between the corners (the corners themselves are automatic).",
            f"div_x = {_toml_list(template.div_x)}   # marks along X",
            f"div_y = {_toml_list(template.div_y)}   # marks along Y",
            f"div_z = {_toml_list(template.div_z)}   # marks along Z",
        ]
    return [
        "[triad]   # three arms that meet at the origin (0, 0, 0)",
        "# Marks on each arm, in metres, counted from the origin:",
        f"arms_x = {_toml_list(template.arms_x)}   # the arm pointing sideways (X)",
        f"arms_y = {_toml_list(template.arms_y)}   # the arm pointing forward (Y)",
        f"arms_z = {_toml_list(template.arms_z)}   # the arm pointing up (Z)",
        *_positions_lines(template.positions, empty_means="only at (0, 0)"),
    ]


def template_toml_text(template: Ref3dTemplate | None = None, *, all_kinds: bool = True) -> str:
    """The commented TOML a person fills in: ``kind`` picks the section that is used.

    ``template`` gives the values of its own section (the others get examples);
    ``all_kinds=False`` writes only that section.
    """
    active = template if template is not None and template.kind in TEMPLATES else _EXAMPLES["pole"]
    kinds = TEMPLATES if all_kinds else (active.kind,)
    lines = [
        _HEADER,
        f'kind = "{active.kind}"   # the object you measured: "pole", "box" or "triad"',
    ]
    for kind in ("pole", "box", "triad"):
        if kind not in kinds:
            continue
        lines.append("")
        lines.extend(_section_lines(active if kind == active.kind else _EXAMPLES[kind]))
    if active.kind != "box" or active.video_positions:
        lines.extend(_videos_lines(active))
    return "\n".join(_align_comments(lines)) + "\n"


def _videos_lines(template: Ref3dTemplate) -> list[str]:
    """The optional [videos] table; commented out as an example when there is none."""
    intro = [
        "",
        "# Optional: where the object stood in each video. One line per video:",
        "# its file name = [x, y], one of the floor positions above. REF3D then sends",
        "# the clicks of that video to its own points, in the order of the list.",
    ]
    if template.video_positions:
        return [
            *intro[:4],
            "[videos]",
            *(f'"{name}" = [{x:g}, {y:g}]' for name, (x, y) in template.video_positions),
        ]
    return [
        *intro[:4],
        "# [videos]",
        '# "calib_pos1_frame_1_to_3.mp4" = [0, 0]',
    ]


def _align_comments(lines: list[str], column: int = 34) -> list[str]:
    """Put the ``# comment`` of every ``key = value`` line in one column."""
    out = []
    for line in lines:
        code, sep, comment = line.partition(" #")
        if sep and code.strip() and not code.lstrip().startswith("#"):
            line = f"{code.rstrip():<{column - 1}} # {comment.strip()}"
        out.append(line)
    return out


def write_blank_template(
    path: str | Path, template: Ref3dTemplate | None = None, *, kind: str | None = None
) -> Path:
    """Write a template to fill in (all three sections); never overwrites a file."""
    path = Path(path)
    if path.exists():
        raise Ref3dError(f"{path} exists: edit it, or choose another name")
    if template is None and kind is not None:
        if kind not in TEMPLATES:
            raise Ref3dError(f"kind {kind!r}: use {', '.join(TEMPLATES)}")
        template = _EXAMPLES[kind]
    path.write_text(template_toml_text(template), encoding="utf-8")
    return path


def write_template(template: Ref3dTemplate, path: str | Path) -> Path:
    """Save the measures of the template (the same TOML a person fills in, one section)."""
    path = Path(path)
    path.write_text(template_toml_text(template, all_kinds=False), encoding="utf-8")
    return path


def _floats(where: str, key: str, value) -> tuple[float, ...]:
    if isinstance(value, int | float) and not isinstance(value, bool):
        value = [value]
    if not isinstance(value, list):
        raise Ref3dError(f"{where} {key}: a list of numbers in metres, like [0, 1]")
    try:
        out = tuple(float(v) for v in value)
    except (TypeError, ValueError) as exc:
        raise Ref3dError(f"{where} {key}: a list of numbers in metres, like [0, 1]") from exc
    if any(not math.isfinite(v) for v in out):
        raise Ref3dError(f"{where} {key}: only finite numbers")
    return out


def _positions_from(where: str, section: dict) -> tuple[tuple[float, float], ...]:
    rect = _floats(where, "rect", section.get("rect", []))
    raw = section.get("positions", [])
    if rect and raw:
        raise Ref3dError(f"{where}: use rect or positions, not both")
    if rect:
        if len(rect) != 2:
            raise Ref3dError(f"{where} rect: two numbers [X, Y], like [1.38, 5]")
        return rectangle_positions(*rect)
    if not isinstance(raw, list):
        raise Ref3dError(f"{where} positions: a list of [x, y], like [[0, 0], [1, 0]]")
    positions = []
    for item in raw:
        pair = _floats(where, "positions", item)
        if len(pair) != 2:
            raise Ref3dError(f"{where} positions: each one is [x, y], like [1.38, 0]")
        positions.append((pair[0], pair[1]))
    return tuple(positions)


def template_from_dict(data: dict) -> Ref3dTemplate:
    """A template from the TOML data: sections ``[pole]``/``[box]``/``[triad]``, or the
    flat keys of the files saved before them."""
    kind = str(data.get("kind", "")).strip().lower()
    if kind not in TEMPLATES:
        raise Ref3dError(f"kind = {data.get('kind')!r}: use {', '.join(TEMPLATES)}")
    if kind not in data:  # older flat file: every key at the top
        flat = {k: v for k, v in data.items() if k != "kind"}
        section = {k: v for k, v in flat.items() if k in _SECTION_KEYS[kind] or k in ("size",)}
        where = "template"
    else:
        section = data[kind]
        if not isinstance(section, dict):
            raise Ref3dError(f"[{kind}]: a section with its measures")
        where = f"[{kind}]"
        allowed = _SECTION_KEYS[kind]
        for key in section:
            if key not in allowed:
                raise Ref3dError(f"{where} unknown key {key!r} (use {', '.join(allowed)})")
    if kind == "box":
        size = _floats(where, "size", section.get("size", []))
        if len(size) != 3 or min(size) <= 0:
            raise Ref3dError(f"{where} size: three positive numbers [X, Y, Z], like [1.38, 5, 1]")
        template = Ref3dTemplate(
            "box",
            size=(size[0], size[1], size[2]),
            div_x=_floats(where, "div_x", section.get("div_x", [])),
            div_y=_floats(where, "div_y", section.get("div_y", [])),
            div_z=_floats(where, "div_z", section.get("div_z", [])),
        )
    elif kind == "pole":
        template = Ref3dTemplate(
            "pole",
            marks=_floats(where, "marks", section.get("marks", [])),
            positions=_positions_from(where, section),
        )
    else:
        template = Ref3dTemplate(
            "triad",
            arms_x=_floats(where, "arms_x", section.get("arms_x", [])),
            arms_y=_floats(where, "arms_y", section.get("arms_y", [])),
            arms_z=_floats(where, "arms_z", section.get("arms_z", [])),
            positions=_positions_from(where, section),
        )
    template.points()  # raises Ref3dError when the measures cannot make points
    template.video_positions = _videos_from(data, kind, template.positions or ((0.0, 0.0),))
    return template


def _videos_from(
    data: dict, kind: str, positions: tuple[tuple[float, float], ...]
) -> tuple[tuple[str, tuple[float, float]], ...]:
    """The optional ``[videos]`` table: ``"file.mp4" = [x, y]``, a floor position of the object."""
    raw = data.get("videos", {})
    if not raw:
        return ()
    if kind == "box":
        raise Ref3dError("[videos]: only for pole and triad, which stand at floor positions")
    if not isinstance(raw, dict):
        raise Ref3dError('[videos]: one line per video: "file.mp4" = [x, y]')
    out = []
    for name, value in raw.items():
        xy = _floats("[videos]", str(name), value)
        if len(xy) != 2:
            raise Ref3dError(f'[videos] "{name}": the floor position [x, y], like [1.4, 0]')
        if not any(abs(xy[0] - x) < 1e-9 and abs(xy[1] - y) < 1e-9 for x, y in positions):
            raise Ref3dError(
                f'[videos] "{name}": [{xy[0]:g}, {xy[1]:g}] is not one of the floor positions '
                "of the object above"
            )
        out.append((str(name), (xy[0], xy[1])))
    return tuple(out)


def read_template(path: str | Path) -> Ref3dTemplate:
    """Read a template TOML (the one filled in by hand, or the one saved next to a video)."""
    try:
        with open(path, "rb") as fh:
            data = tomllib.load(fh)
    except tomllib.TOMLDecodeError as exc:
        raise Ref3dError(f"{Path(path).name}: not valid TOML ({exc})") from exc
    return template_from_dict(data)


def write_dlt3d(L: np.ndarray, path: str | Path) -> Path:
    """Write the 11 coefficients in dlt3d.py's format (frame 0).

    Same format and equations as the file Make DLT3D (dlt3d.py) writes, so there is no need
    to run it again. On the same pixel values the two files are identical; Save solves from
    the exact clicks while the pixel CSV (what dlt3d.py reads) keeps whole pixels, so a later
    Make DLT3D can differ by under 1 px in projection.
    """
    path = Path(path)
    coeffs = ",".join(f"{v:.6f}" for v in np.asarray(L, dtype=float).reshape(11))
    path.write_text("frame,L1,L2,L3,L4,L5,L6,L7,L8,L9,L10,L11\n" + f"0,{coeffs}\n")
    return path


def read_pixels_row(path: str | Path) -> dict[int, tuple[float, float]]:
    """The first row of a pixel CSV (``frame,p0_x,p0_y,...``) as {point: (u, v)}."""
    df = pd.read_csv(path)
    row = df.iloc[0]
    pixels = {}
    for col in df.columns:
        m = re.fullmatch(r"p(\d+)_?x", str(col))
        if m is None:
            continue
        y_col = str(col)[:-1] + "y"
        if y_col in df.columns and pd.notna(row[col]) and pd.notna(row[y_col]):
            pixels[int(m.group(1))] = (float(row[col]), float(row[y_col]))
    return pixels


# --- DLT3D fit ----------------------------------------------------------------------


@dataclass
class DltCheck:
    """A DLT3D solved from the clicked points, and how believable the camera is."""

    L: np.ndarray
    used: list[int]  # point numbers in the fit
    error_px: dict[int, float]  # reprojection error of each used point
    projected: dict[int, tuple[float, float]]  # every template point through the camera
    fx: float
    fy: float
    problems: list[str]
    warnings: list[str]

    @property
    def mean_px(self) -> float:
        return float(np.mean(list(self.error_px.values())))

    @property
    def max_px(self) -> float:
        return float(np.max(list(self.error_px.values())))

    @property
    def ok(self) -> bool:
        return not self.problems

    def summary(self) -> str:
        head = (
            f"DLT3D from {len(self.used)} points: error mean {self.mean_px:.1f} px, "
            f"max {self.max_px:.1f} px; fx {self.fx:.0f} px, fy {self.fy:.0f} px"
        )
        if self.problems:
            return head + " - NOT A REAL CAMERA: " + "; ".join(self.problems)
        return head + " - camera OK"

    def save_note(self) -> str:
        """What to tell the user once the .dlt3d is on disk: the file is final (same format
        and equations as Make DLT3D / dlt3d.py) only for a real camera."""
        if self.ok:
            return "ready for Rec3D (skip Make DLT3D)"
        return "NOT A REAL CAMERA - check the measures"

    def banner(self) -> str:
        """The one-line message of the window after Save (no wrapping: keep it short)."""
        stats = f"error {self.mean_px:.1f} px, fx {self.fx:.0f} / fy {self.fy:.0f}"
        if self.ok:
            return f"Saved .ref3d + pixels + .dlt3d, {self.save_note()}: {stats}"
        return f"Saved .ref3d + pixels + .dlt3d: {stats} - {self.save_note()}"

    def terminal_note(self) -> str:
        """A line for the terminal after the .dlt3d path; empty when the camera is not real."""
        if not self.ok:
            return ""
        return (
            "same format and equations as Make DLT3D (dlt3d.py), solved from your exact clicks "
            "(the CSV keeps whole pixels): use it in Rec3D, no need to run it again"
        )


def solve_dlt3d(xyz: np.ndarray, uv: np.ndarray) -> np.ndarray:
    """Least-squares 11-parameter DLT (same equations as dlt3d.calculate_dlt3d_params)."""
    n = len(xyz)
    A = np.zeros((2 * n, 11))
    B = np.zeros(2 * n)
    for i, ((X, Y, Z), (u, v)) in enumerate(zip(xyz, uv, strict=True)):
        A[2 * i, 0:4] = [X, Y, Z, 1]
        A[2 * i, 8:11] = [-u * X, -u * Y, -u * Z]
        B[2 * i] = u
        A[2 * i + 1, 4:8] = [X, Y, Z, 1]
        A[2 * i + 1, 8:11] = [-v * X, -v * Y, -v * Z]
        B[2 * i + 1] = v
    return np.linalg.lstsq(A, B, rcond=None)[0]


def fit_and_check(
    points: list[Ref3dPoint], pixels: dict[int, tuple[float, float]]
) -> DltCheck | str:
    """Solve the DLT3D from the clicked points and judge the camera.

    Returns the check, or a sentence saying why it cannot be solved yet (fewer than
    6 points, or every clicked point on one plane).
    """
    try:
        from .monocular_dlt_align import assess_dlt_camera, decompose_dlt3d, dlt_project
    except ImportError:
        from monocular_dlt_align import (  # ty: ignore[unresolved-import]
            assess_dlt_camera,
            decompose_dlt3d,
            dlt_project,
        )

    by_index = {p.index: p for p in points}
    used = [i for i in sorted(pixels) if i in by_index]
    if len(used) < MIN_DLT3D_POINTS:
        return f"DLT3D needs {MIN_DLT3D_POINTS} points, {len(used)} clicked"
    if points_are_coplanar([by_index[i] for i in used]):
        return "DLT3D: the clicked points are on one plane; click points at other heights"
    xyz = np.array([by_index[i].xyz for i in used], dtype=float)
    uv = np.array([pixels[i] for i in used], dtype=float)
    L = solve_dlt3d(xyz, uv)
    err = np.linalg.norm(dlt_project(L, xyz) - uv, axis=1)
    all_xyz = np.array([p.xyz for p in points], dtype=float)
    proj = dlt_project(L, all_xyz)
    projected = {
        p.index: (float(u), float(v))
        for p, (u, v) in zip(points, proj, strict=True)
        if np.isfinite(u) and np.isfinite(v)
    }
    try:
        camera = decompose_dlt3d(L)
    except (ValueError, np.linalg.LinAlgError) as exc:
        return f"DLT3D: no camera can be recovered ({exc})"
    validation = {
        "n_control_points": len(used),
        "calibration_reprojection_px_mean": float(err.mean()),
        "calibration_reprojection_px_per_point": err.tolist(),
    }
    problems, warnings = assess_dlt_camera(camera, validation)
    # assess_dlt_camera numbers the points by position in the fit; name them pN.
    problems = [
        re.sub(r"\bp(\d+) ", lambda m: f"p{used[int(m.group(1))]} ", msg) for msg in problems
    ]
    return DltCheck(
        L=L,
        used=used,
        error_px={i: float(e) for i, e in zip(used, err, strict=True)},
        projected=projected,
        fx=float(camera.K[0, 0]),
        fy=float(camera.K[1, 1]),
        problems=problems,
        warnings=warnings,
    )


# --- CLI ----------------------------------------------------------------------------


def template_from_args(args: argparse.Namespace) -> Ref3dTemplate:
    positions = parse_positions(args.positions or "")
    if args.template == "box":
        if not args.size:
            raise Ref3dError("--size X,Y,Z is needed for a box")
        return Ref3dTemplate(
            "box",
            size=parse_size(args.size),
            div_x=parse_lengths(args.div_x or ""),
            div_y=parse_lengths(args.div_y or ""),
            div_z=parse_lengths(args.div_z or ""),
        )
    if args.template == "pole":
        return Ref3dTemplate("pole", marks=parse_lengths(args.marks or ""), positions=positions)
    return Ref3dTemplate(
        "triad",
        arms_x=parse_lengths(args.arms_x or ""),
        arms_y=parse_lengths(args.arms_y or ""),
        arms_z=parse_lengths(args.arms_z or ""),
        positions=positions,
    )


def _template_of(args: argparse.Namespace) -> Ref3dTemplate:
    if args.template_file:
        return read_template(args.template_file)
    return template_from_args(args)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m vaila.ref3d_builder",
        description="Build a .ref3d from the measures of a box, pole or triad, "
        "or solve and check a DLT3D from a .ref3d and its pixel CSV.",
    )
    p.add_argument("--template", choices=TEMPLATES, help="what the calibration object is")
    p.add_argument(
        "--template-file",
        help="the measures from a template TOML (instead of --template, --marks, ...)",
    )
    p.add_argument(
        "--new-template",
        metavar="FILE.toml",
        help="write a template TOML to fill in (kind from --template, default pole)",
    )
    p.add_argument("--size", help="box: X x Y x Z in metres, e.g. '0.55 x 5 x 1'")
    p.add_argument("--div-x", help="box: marks along X, e.g. '0.25'")
    p.add_argument("--div-y", help="box: marks along Y, e.g. '1,2,3,4'")
    p.add_argument("--div-z", help="box: marks along Z, e.g. '0.5'")
    p.add_argument("--marks", help="pole: heights of the marks, e.g. '0, 0.5, 1'")
    p.add_argument("--arms-x", help="triad: marks on the X arm, e.g. '0.25, 0.5'")
    p.add_argument("--arms-y", help="triad: marks on the Y arm")
    p.add_argument("--arms-z", help="triad: marks on the Z arm")
    p.add_argument(
        "--positions",
        help="pole/triad: 'rect 0.55 x 5' (4 corners, X x Y) or '0,0; 0.55,0; 0,5'",
    )
    p.add_argument("--ref3d", help="existing .ref3d (with --pixels: solve the DLT3D)")
    p.add_argument("--pixels", help="pixel CSV (frame,p0_x,p0_y,...) clicked in that order")
    p.add_argument("-o", "--output", help="output .ref3d (template) or .dlt3d (--pixels)")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        if args.new_template:
            out = write_blank_template(args.new_template, kind=args.template or "pole")
            print(f"Template written to {out}: fill it in, then use --template-file {out}")
            return 0
        if args.pixels:
            if not (args.ref3d or args.template or args.template_file):
                raise Ref3dError("--pixels needs --ref3d (or --template / --template-file)")
            points = read_ref3d_points(args.ref3d) if args.ref3d else _template_of(args).points()
            check = fit_and_check(points, read_pixels_row(args.pixels))
            if isinstance(check, str):
                print(check)
                return 1
            print(check.summary())
            for i in check.used:
                print(f"  p{i}: {check.error_px[i]:.2f} px")
            for w in check.warnings:
                print(f"  warning: {w}")
            out = Path(args.output) if args.output else Path(args.pixels).with_suffix(".dlt3d")
            if out.exists():
                print(f"Not written: {out} exists (choose another -o)")
                return 1
            write_dlt3d(check.L, out)
            print(f"DLT3D saved to {out}")
            return 0 if check.ok else 2
        if not (args.template or args.template_file):
            build_parser().print_help()
            return 1
        template = _template_of(args)
        points = template.points()
        for p in points:
            print(f"  p{p.index}: ({p.x:g}, {p.y:g}, {p.z:g})  {p.label}")
        if points_are_coplanar(points):
            print("warning: every point is on one plane - a DLT3D needs points off it")
        if args.output:
            out = Path(args.output)
            if out.exists():
                print(f"Not written: {out} exists (choose another -o)")
                return 1
            write_ref3d(points, out)
            write_template(template, out.with_suffix(".template.toml"))
            print(f"{len(points)} points ({template.describe()}) saved to {out}")
        return 0
    except Ref3dError as exc:
        print(f"Error: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
