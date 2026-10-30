"""
================================================================================
Script: freekiki_setpiece.py - what kind of set piece the kick is (penalty or free kick)
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 28 October 2026
Update Date: 29 October 2026

Description:
    The kind of the set piece, from where the ball rests and who stands around it (before the kick):

    * ``penalty``: the ball is 11 m from the goal line (the penalty mark, FIFA Laws), within 1.5 m, near
      the middle of the goal (|y| <= 3.0 m), and fewer than two people stand in the wall ring;
    * ``free_kick_wall``: two or more people with their feet 8.5 to 10.0 m from the ball, between the
      ball and the goal (the wall stands 9.15 m away, FIFA Laws);
    * ``free_kick``: any other ball at rest.

    The distances and the ring are operational choices, not validated thresholds. The attacked goal is
    the one nearer to the ball. The result goes to ``data/setpiece.json`` of each analysis.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

try:
    from .freekiki_layout import data_dir as analysis_data_dir
except ImportError:
    from freekiki_layout import data_dir as analysis_data_dir  # ty: ignore[unresolved-import]

SETPIECE_NAME = "setpiece.json"
PROVISIONAL_NAME = "setpiece_provisional.json"  # the 3D folder's own guess, before the one-click
GOAL_LINE_X_M = 52.45  # the goal line, |x| from the centre spot
PENALTY_DISTANCE_M = 11.0  # the penalty mark (FIFA Laws)
PENALTY_TOLERANCE_M = 1.5  # operational: how far from the penalty mark a penalty may be
PENALTY_LATERAL_M = 3.0  # operational: the penalty is taken near the middle of the goal
WALL_RING_M = (8.5, 10.0)  # operational: the wall stands 9.15 m away (FIFA Laws), with a margin
WALL_MIN_PEOPLE = 2  # operational: two people in the ring make a wall
FOOT_LABELS = ("left-ankle", "right-ankle", "left-heel", "right-heel")


def classify(ball_xy: tuple[float, float], feet: dict[int, tuple[float, float]]) -> dict:
    """The kind of the set piece from the ball at rest (metres) and the feet of the people (metres).

    ``feet`` maps a person id to the centre of their feet on the ground. Returns the kind, the goal
    attacked, the distances and the people in the wall ring, with the reason in words.
    """
    bx, by = float(ball_xy[0]), float(ball_xy[1])
    goal_x = GOAL_LINE_X_M if bx >= 0 else -GOAL_LINE_X_M
    distance = abs(goal_x - bx)
    wall = []
    for pid, (x, y) in sorted(feet.items()):
        d = math.hypot(x - bx, y - by)
        between = (x - bx) * (goal_x - bx) > 0
        if WALL_RING_M[0] <= d <= WALL_RING_M[1] and between:
            wall.append(int(pid))
    at_penalty_mark = abs(distance - PENALTY_DISTANCE_M) <= PENALTY_TOLERANCE_M
    central = abs(by) <= PENALTY_LATERAL_M
    if len(wall) >= WALL_MIN_PEOPLE:
        kind = "free_kick_wall"
        reason = (
            f"{len(wall)} people with the feet {WALL_RING_M[0]} to {WALL_RING_M[1]} m from the ball"
        )
    elif at_penalty_mark and central:
        kind = "penalty"
        reason = (
            f"the ball is {distance:.2f} m from the goal line (penalty mark {PENALTY_DISTANCE_M} m) "
            f"and {abs(by):.2f} m from the middle; fewer than {WALL_MIN_PEOPLE} people in the wall ring"
        )
    else:
        kind = "free_kick"
        why = (f"{abs(by):.2f} m from the middle of the goal" if at_penalty_mark
               else "not on the penalty mark")  # fmt: skip
        reason = (
            f"the ball is {distance:.2f} m from the goal line, {why}, and "
            f"fewer than {WALL_MIN_PEOPLE} people in the wall ring"
        )
    return {
        "kind": kind,
        "reason": reason,
        "ball_xy_m": [round(bx, 3), round(by, 3)],
        "goal_x_m": goal_x,
        "distance_to_goal_m": round(distance, 3),
        "lateral_m": round(by, 3),
        "wall_ids": wall,
        "wall_count": len(wall),
    }


def ball_rest(ball_csv: Path | str) -> tuple[float, float] | None:
    """The ball's (x, y) in metres at the first frame in which it is on the field."""
    with Path(ball_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("x_m", "") != "" and row.get("on_field", "True") == "True":
                return float(row["x_m"]), float(row["y_m"])
    return None


def feet_at(poses_csv: Path | str, frame: int) -> dict[int, tuple[float, float]]:
    """Each person's feet centre (metres) at ``frame``; the frame nearest with feet when it has none."""
    rows: dict[int, dict[int, list[tuple[float, float]]]] = {}
    with Path(poses_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["kpt_name"] not in FOOT_LABELS or row["x_m"] == "":
                continue
            f = int(row["frame"])
            rows.setdefault(f, {}).setdefault(int(row["person_id"]), []).append(
                (float(row["x_m"]), float(row["y_m"]))
            )  # fmt: skip
    if not rows:
        return {}
    chosen = min(rows, key=lambda f: (abs(f - frame), f))
    return {pid: (sum(p[0] for p in pts) / len(pts), sum(p[1] for p in pts) / len(pts))
            for pid, pts in rows[chosen].items()}  # fmt: skip


def classify_run(folder: Path | str) -> dict:
    """The set piece of one 3D folder (its ``ball_3d_on_field.csv`` and ``poses_3d_on_field.csv``).

    A folder without the ball file, or with no ball on the field, is ``unknown``: the run goes on with
    the free-kick analysis and no penalty routing.
    """
    folder = Path(folder)
    rest = (
        ball_rest(folder / "ball_3d_on_field.csv")
        if (folder / "ball_3d_on_field.csv").is_file()
        else None
    )
    if rest is None:
        return {
            "kind": "unknown",
            "reason": "no ball on the field",
            "wall_count": 0,
            "wall_ids": [],
        }
    first = 0
    with (folder / "ball_3d_on_field.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("x_m", "") != "":
                first = int(row["frame"])
                break
    poses = folder / "poses_3d_on_field.csv"
    return classify(rest, feet_at(poses, first) if poses.is_file() else {})


def with_player_ids(result: dict, sam_to_player: dict[int, int]) -> dict:
    """The wall in the player ids of the run (keeper 0, kicker 1, the others 2, 3, ...).

    ``wall_ids`` are SAM track ids (the ``person_id`` of the poses), so they are converted with the
    run's ``player_map.csv``. A wall track that the player map does not hold stays in ``wall_sam_ids``
    only, and the count is the number of people mapped to a player.
    """
    sam = list(result.get("wall_ids", []))
    mapped = [sam_to_player[s] for s in sam if s in sam_to_player]
    out = dict(result)
    out["wall_sam_ids"] = sam
    out["wall_ids"] = mapped
    out["wall_count"] = len(mapped)
    return out


def write(data_dir: Path | str, result: dict, name: str = SETPIECE_NAME) -> Path:
    path = Path(data_dir) / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    return path


SET_PIECE_CHOICES = ("auto", "penalty", "free_kick")  # the GUI box and ``--set-piece``


def apply_choice(result: dict, choice: str) -> dict:
    """The set piece with the kind the user chose (``auto`` keeps the rule's answer).

    The rule's own kind stays in the result as ``auto_kind``, with its reason, so the README shows what
    the rule said next to the choice. A forced penalty keeps the goal side of the rule when there is a
    ball at rest; without one the penalty is reported as unavailable by the pipeline.
    """
    if choice not in SET_PIECE_CHOICES:
        raise ValueError(f"set piece must be one of {SET_PIECE_CHOICES}, got {choice!r}")
    out = dict(result, choice=choice, auto_kind=result.get("kind", "unknown"))
    if choice == "auto":
        return out
    out["kind"] = choice
    out["reason"] = f"chosen by the user ({choice}); the ball rule said {out['auto_kind']}" + (
        f": {result['reason']}" if result.get("reason") else ""
    )
    return out


def write_provisional(folder: Path | str, ball_mode: str) -> dict:
    """The set piece of a 3D folder, judged from the ball of its own run (a guess until the one-click).

    ``ball_mode`` is the ball of that run (``off``, ``net`` or ``sam3``). A SAM or net track is not
    corrected by hand, so the one-click analysis, which reads the corrected ball, has the last word.
    The result goes to ``data/setpiece_provisional.json`` of the folder.
    """
    result = classify_run(folder) | {"provisional": True, "ball_mode": ball_mode}
    write(analysis_data_dir(folder), result, name=PROVISIONAL_NAME)
    return result
