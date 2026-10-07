"""
================================================================================
Script: kiki49.py - kiki49_build 49-point soccer-field geometry
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Ported from mkvis3d ``openbiomech/soccer_field`` (the builder of the kiki49
dataset) so FreeKiki can convert external field datasets by itself.

Description:
    Kiki 49-point soccer-field geometry.

    Points come from ``vaila/models/soccerfield_kiki.csv``. Frame: metres, origin at the centre
    spot, +x towards the right goal, +y towards the far ("top") touchline, +z up.
    Points 0-31 keep the pitch32 order; 10/11/18/19 are the penalty-arc x
    penalty-box-line intersections (not the legacy goal-area-y points).
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from functools import cache
from pathlib import Path

import numpy as np

# Single source of truth for the 49 points (also used by freekiki / freekiki_diag).
CSV_PATH = Path(__file__).resolve().parent.parent / "models" / "soccerfield_kiki.csv"
N_KPT = 49

HALF_LENGTH = 52.45
HALF_WIDTH = 33.95
BOX_DEPTH = 16.5
BOX_HALF_WIDTH = 20.16
GOAL_AREA_DEPTH = 5.5
GOAL_AREA_HALF_WIDTH = 9.16
PENALTY_SPOT = 11.0
CIRCLE_RADIUS = 9.15
GOAL_HALF_WIDTH = 3.66
GOAL_HEIGHT = 2.44

# Legacy pitch32 points whose meaning changed in kiki49: the pitch32 builder
# placed them on the box line at the goal-area y (+-9.16); kiki49 uses the
# penalty-arc intersections (+-7.312489). Human clickers already used arcs.
ARC_POINTS = (10, 11, 18, 19)
LEGACY_GOAL_Y_POINTS_XY = {
    10: (-35.95, 9.16),
    11: (-35.95, -9.16),
    18: (35.95, 9.16),
    19: (35.95, -9.16),
}
# Points above the ground plane plus the assumed-depth net ground points:
# only a real 3D camera may label them.
AUX3D_POINTS = (34, 35, 36, 37, 38, 39, 42, 43, 44, 45, 46, 47)


@dataclass(frozen=True)
class Kiki49:
    names: tuple[str, ...]
    flip_idx: tuple[int, ...]
    xyz: np.ndarray  # (49, 3) float64, metres

    @property
    def planar(self) -> np.ndarray:
        return self.xyz[:, 2] == 0.0


@cache
def load_kiki49() -> Kiki49:
    with CSV_PATH.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if [int(r["point_number"]) for r in rows] != list(range(N_KPT)):
        raise ValueError(f"{CSV_PATH} must list points 0..{N_KPT - 1} in order")
    xyz = np.array([[float(r["x"]), float(r["y"]), float(r["z"])] for r in rows], dtype=np.float64)
    xyz.setflags(write=False)
    return Kiki49(
        names=tuple(r["point_name"] for r in rows),
        flip_idx=tuple(int(r["flip_idx"]) for r in rows),
        xyz=xyz,
    )


def _segment(a: tuple[float, float], b: tuple[float, float], step: float) -> np.ndarray:
    n = max(2, int(np.ceil(np.hypot(b[0] - a[0], b[1] - a[1]) / step)) + 1)
    s = np.linspace(0.0, 1.0, n)[:, None]
    return (1.0 - s) * np.array(a) + s * np.array(b)


def _arc(cx: float, r: float, t0: float, t1: float, step: float) -> np.ndarray:
    n = max(8, int(np.ceil(r * abs(t1 - t0) / step)) + 1)
    t = np.linspace(t0, t1, n)
    return np.column_stack([cx + r * np.cos(t), r * np.sin(t)])


@cache
def ground_lines(step: float = 0.25) -> tuple[np.ndarray, ...]:
    """Painted pitch markings as (M, 3) polylines on z=0, sampled every ``step`` m."""
    L, W = HALF_LENGTH, HALF_WIDTH
    polys: list[np.ndarray] = [
        _segment((-L, W), (L, W), step),
        _segment((-L, -W), (L, -W), step),
        _segment((-L, -W), (-L, W), step),
        _segment((L, -W), (L, W), step),
        _segment((0.0, -W), (0.0, W), step),
        _arc(0.0, CIRCLE_RADIUS, 0.0, 2.0 * np.pi, step),
    ]
    half_arc = float(np.arccos((BOX_DEPTH - PENALTY_SPOT) / CIRCLE_RADIUS))
    for sx in (-1.0, 1.0):
        gx = sx * L
        bx = sx * (L - BOX_DEPTH)
        ax = sx * (L - GOAL_AREA_DEPTH)
        polys += [
            _segment((gx, BOX_HALF_WIDTH), (bx, BOX_HALF_WIDTH), step),
            _segment((bx, BOX_HALF_WIDTH), (bx, -BOX_HALF_WIDTH), step),
            _segment((bx, -BOX_HALF_WIDTH), (gx, -BOX_HALF_WIDTH), step),
            _segment((gx, GOAL_AREA_HALF_WIDTH), (ax, GOAL_AREA_HALF_WIDTH), step),
            _segment((ax, GOAL_AREA_HALF_WIDTH), (ax, -GOAL_AREA_HALF_WIDTH), step),
            _segment((ax, -GOAL_AREA_HALF_WIDTH), (gx, -GOAL_AREA_HALF_WIDTH), step),
        ]
        spot = sx * (L - PENALTY_SPOT)
        # Arc outside the box: centred on the spot, facing midfield.
        if sx < 0:
            polys.append(_arc(spot, CIRCLE_RADIUS, -half_arc, half_arc, step))
        else:
            polys.append(_arc(spot, CIRCLE_RADIUS, np.pi - half_arc, np.pi + half_arc, step))
    return tuple(np.column_stack([p, np.zeros(len(p))]) for p in polys)


@cache
def goal_frames(step: float = 0.1) -> tuple[np.ndarray, ...]:
    """Goal posts and crossbars as (M, 3) polylines (for previews only)."""
    out = []
    for sx in (-1.0, 1.0):
        x = sx * HALF_LENGTH
        g, h = GOAL_HALF_WIDTH, GOAL_HEIGHT
        for a, b in (((-g, 0.0), (-g, h)), ((-g, h), (g, h)), ((g, h), (g, 0.0))):
            yz = _segment(a, b, step)
            out.append(np.column_stack([np.full(len(yz), x), yz]))
    return tuple(out)
