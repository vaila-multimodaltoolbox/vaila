"""
================================================================================
Script: freekiki_report_players.py - people on the top view of the press report
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 27 October 2026
Update Date: 27 October 2026

Description:
    The top view of the report (figure and animation minimap) shows, besides the ball, the
    goal area, the kicker, the two players of the wall nearest the ball at the wall frame, and
    the goalkeeper with his path. Each person is the mean of the ankles in
    ``poses_3d_on_field.csv`` (pitch metres, on the ground). The roles are found by the nearest
    person; ``kicker_id``, ``wall_ids`` and ``keeper_id`` replace them. Nothing here changes the
    fit: this module only reads the poses and draws.
"""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .freekiki_report_i18n import tr
except ImportError:
    from freekiki_report_i18n import tr  # ty: ignore[unresolved-import]

KICKER = "#1664a0"
WALL = "#14934b"
KEEPER = "#a85c13"
INK = "#14213d"
GRID = "#9aa3b2"
GOAL_X = 52.45  # goal line, |x| from the centre spot (FIFA pitch)
GOAL_HALF = 3.66  # half the goal mouth
PENALTY_DEPTH = 16.5
PENALTY_HALF = 20.16
SIX_YARD_DEPTH = 5.5
SIX_YARD_HALF = 9.16
# the largest distance (m) from the ball/entry at which a person is taken for each role
ROLE_RADIUS_M = {"kicker": 2.0, "wall": 2.0, "keeper": 3.0}


@dataclass
class People:
    """Ground tracks of the people (id -> (n, 2) metres, NaN where absent) and the roles."""

    tracks: dict[int, np.ndarray]
    kicker: int | None
    wall: list[int]
    keeper: int | None
    manual: bool = False  # True when any role was given by the user instead of found


def load_ground_tracks(run_dir, n: int) -> dict[int, np.ndarray]:
    """Ground point of each person per frame: the mean of the ankles. {} if there is no file."""
    path = Path(run_dir) / "poses_3d_on_field.csv"
    if not path.is_file():
        return {}
    sums: dict[tuple[int, int], list[float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if "ankle" not in row["kpt_name"] or row["x_m"] == "" or row["y_m"] == "":
                continue
            key = (int(row["person_id"]), int(row["frame"]))
            acc = sums.setdefault(key, [0.0, 0.0, 0.0])
            acc[0] += float(row["x_m"])
            acc[1] += float(row["y_m"])
            acc[2] += 1.0
    tracks: dict[int, np.ndarray] = {}
    for (pid, f), (sx, sy, k) in sums.items():
        if 0 <= f < n:
            arr = tracks.setdefault(pid, np.full((n, 2), np.nan))
            arr[f] = [sx / k, sy / k]
    return tracks


def _at(xy: np.ndarray, frame: int) -> np.ndarray:
    """The point of ``frame``; NaN when the frame is outside the run (no person is near it)."""
    return xy[frame] if 0 <= frame < len(xy) else np.full(2, np.nan)


def _nearest(
    tracks: dict[int, np.ndarray], frame: int, point, radius: float, exclude=()
) -> list[tuple[float, int]]:
    found = []
    for pid, arr in tracks.items():
        if pid in exclude or not 0 <= frame < len(arr) or not np.isfinite(arr[frame]).all():
            continue
        d = float(np.linalg.norm(arr[frame] - np.asarray(point, dtype=float)))
        if d <= radius:
            found.append((d, pid))
    return sorted(found)


def find_people(
    tracks: dict[int, np.ndarray],
    ball_xy: np.ndarray,
    kick_f: int,
    barrier_f: int,
    end_f: int,
    keeper_xy,
    *,
    kicker_id: int | None = None,
    wall_ids: list[int] | None = None,
    keeper_id: int | None = None,
) -> People:
    """The roles: the kicker is the person nearest the ball at the kick, the wall the two
    persons nearest the ball at the wall frame (not the kicker), the keeper the person nearest
    the entry point at the last frame. The arguments given by the user replace the search."""
    manual = kicker_id is not None or wall_ids is not None or keeper_id is not None
    if kicker_id is None and tracks:
        near = _nearest(tracks, kick_f, _at(ball_xy, kick_f), ROLE_RADIUS_M["kicker"])
        kicker_id = near[0][1] if near else None
    if wall_ids is None:
        near = _nearest(
            tracks,
            barrier_f,
            _at(ball_xy, barrier_f),
            ROLE_RADIUS_M["wall"],
            exclude={kicker_id} if kicker_id is not None else (),
        )
        wall_ids = [pid for _, pid in near[:2]]
    if keeper_id is None:
        exclude = {p for p in [kicker_id, *wall_ids] if p is not None}
        near = _nearest(tracks, end_f, keeper_xy, ROLE_RADIUS_M["keeper"], exclude=exclude)
        keeper_id = near[0][1] if near else None
    return People(tracks, kicker_id, list(wall_ids), keeper_id, manual)


def people_note(people: People | None, lang: str) -> str:
    """One line for the caption: which ids were drawn and how they were found."""
    if people is None or (people.kicker is None and not people.wall and people.keeper is None):
        return tr(lang, "top_none")
    wall = ", ".join(str(p) for p in people.wall) or "—"
    how = tr(lang, "top_how_manual" if people.manual else "top_how_auto")
    return tr(
        lang,
        "top_note",
        kicker="—" if people.kicker is None else str(people.kicker),
        wall=wall,
        keeper="—" if people.keeper is None else str(people.keeper),
        how=how,
    )


def draw_goal_area(ax, entry_xy) -> None:
    """The goal line, the goal mouth, the penalty area and the six-yard box at the goal the
    kick is aimed at (the side of the entry point)."""
    gx = float(np.sign(entry_xy[0])) * GOAL_X
    s = -float(np.sign(gx))  # direction from the goal line into the pitch
    for half, depth in ((PENALTY_HALF, PENALTY_DEPTH), (SIX_YARD_HALF, SIX_YARD_DEPTH)):
        ax.plot([gx, gx + s * depth, gx + s * depth, gx, gx], [-half, -half, half, half, -half],
                color=GRID, lw=1.1, zorder=1)  # fmt: skip
    ax.plot([gx, gx], [-GOAL_HALF, GOAL_HALF], color=INK, lw=3.2, zorder=2)


def _mark(ax, xy, color, label, lang, *, labels: bool, dx=8, dy=-12) -> None:
    ax.plot(xy[0], xy[1], "o", ms=9, color=color, mec="white", mew=1.2, zorder=7)
    if labels and label:
        ax.annotate(label, xy, xytext=(dx, dy), textcoords="offset points", color=color,
                    fontsize=9, fontweight="bold", zorder=8)  # fmt: skip


def draw_people(
    ax,
    people: People | None,
    *,
    kick_f: int,
    barrier_f: int,
    end_f: int,
    frame: int,
    lang: str,
    labels: bool = True,
) -> None:
    """The kicker at the kick, the wall at the wall frame (with the segment between the two
    players), the keeper with his path up to ``frame`` (or the end)."""
    if people is None:
        return
    tracks = people.tracks
    if people.keeper is not None and people.keeper in tracks:
        arr = tracks[people.keeper]
        seg = arr[kick_f : end_f + 1]
        seg = seg[np.isfinite(seg).all(axis=1)]
        if len(seg) > 1:
            ax.plot(
                seg[:, 0], seg[:, 1], color=KEEPER, lw=2.2, ls=(0, (3, 2)), alpha=0.85, zorder=4
            )
        now = min(max(frame, kick_f), end_f)
        if np.isfinite(arr[now]).all():
            _mark(ax, arr[now], KEEPER, tr(lang, "top_keeper", id=people.keeper), lang,
                  labels=labels, dx=10, dy=-16)  # fmt: skip
    if people.kicker is not None and people.kicker in tracks:
        arr = tracks[people.kicker]
        if np.isfinite(arr[kick_f]).all():
            _mark(ax, arr[kick_f], KICKER, tr(lang, "top_kicker", id=people.kicker), lang,
                  labels=labels, dx=-40, dy=-18)  # fmt: skip
    wall_pts = [tracks[p][barrier_f] for p in people.wall if p in tracks]
    wall_pts = [w for w in wall_pts if np.isfinite(w).all()]
    if len(wall_pts) == 2:
        ax.plot([wall_pts[0][0], wall_pts[1][0]], [wall_pts[0][1], wall_pts[1][1]],
                color=WALL, lw=2.6, zorder=5)  # fmt: skip
    for xy in wall_pts:
        _mark(ax, xy, WALL, "", lang, labels=labels)
    if wall_pts and labels:
        mid = np.mean(np.array(wall_pts), axis=0)
        text = (tr(lang, "top_wall", a=people.wall[0], b=people.wall[1])
                if len(wall_pts) == 2 else tr(lang, "top_wall_one", a=people.wall[0]))  # fmt: skip
        ax.annotate(text, mid, xytext=(14, 10), textcoords="offset points", color=WALL,
                    fontsize=9, fontweight="bold", zorder=8)  # fmt: skip


def people_extent(people: People | None, kick_f: int, end_f: int) -> np.ndarray:
    """All the ground points of the roles between the kick and the end (for the axis limits)."""
    if people is None:
        return np.empty((0, 2))
    ids = [p for p in [people.kicker, people.keeper, *people.wall] if p is not None]
    pts = [people.tracks[p][kick_f : end_f + 1] for p in ids if p in people.tracks]
    if not pts:
        return np.empty((0, 2))
    arr = np.vstack(pts)
    return arr[np.isfinite(arr).all(axis=1)]
