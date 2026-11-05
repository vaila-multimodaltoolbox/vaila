"""
================================================================================
Script: freekiki_setpiece.py - what kind of kick / open-play analysis it is
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.14
Created: 28 October 2026
Update Date: 02 November 2026

Description:
    The kind of the kick / open-play analysis, from how the ball comes to it and who stands around it.

    Goal-bound (ball to goal → flight TV report):

    * ``kick``: chute a gol in open play — ball was not at rest (``pre_contact`` rolling / unseen).
      Auto-classify returns this; no wall.
    * ``penalty``: ball ~11 m from the goal line, central, fewer than two people in the wall ring.
    * ``free_kick_wall`` / ``free_kick``: ball at rest; wall ring or not.

    Trajectory-only (no shot at goal → planar tracks / data only):

    * ``open_play``: user choice — data only (no goal TV report, no dribble/pass analysis).
    * ``dribble`` / ``pass``: user choices for planar 1vN / pass reports (never from ``classify``).

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


def classify(
    ball_xy: tuple[float, float],
    feet: dict[int, tuple[float, float]],
    pre_contact: str | None = None,
) -> dict:
    """The kind of the kick from the ball at the kick (metres) and the feet of the people (metres).

    ``feet`` maps a person id to the centre of their feet on the ground. ``pre_contact`` is how the
    ball came to the kick (``still``, ``rolling`` or ``unseen``; None when unknown): a ball that was
    not at rest is classified as ``kick`` (chute a gol in play, no wall). Returns the kind, the goal
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
    if pre_contact in ("rolling", "unseen"):
        kind = "kick"
        how = (
            "rolled into the kick"
            if pre_contact == "rolling"
            else "already flies in the first frame"
        )
        reason = (
            f"the ball {how}, so it was not at rest (chute a gol / kick); it is kicked "
            f"{distance:.2f} m from the goal line"
        )
    elif len(wall) >= WALL_MIN_PEOPLE:
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


def _flight_contact(folder: Path) -> dict | None:
    try:
        from .freekiki_auto_flight import flight_contact
    except ImportError:
        from freekiki_auto_flight import flight_contact  # ty: ignore[unresolved-import]
    return flight_contact(folder)


def classify_run(folder: Path | str) -> dict:
    """The kind of the kick of one 3D folder (``ball_3d_on_field.csv``, ``poses_3d_on_field.csv``
    and the contact the automatic flight wrote in ``ball_flight.json``).

    A ball that was not at rest (the flight recorded it rolling or already flying) is taken at the
    kick contact, with the people at that frame. A still ball, or an older folder, keeps the rule
    as it was: the first ball on the field. A folder without the ball file, or with no
    ball on the field, is ``unknown``: the run goes on with the free-kick analysis and no penalty
    routing.
    """
    folder = Path(folder)
    contact = _flight_contact(folder)
    if (
        contact is not None
        and contact.get("pre_contact") in ("rolling", "unseen")
        and contact.get("kick_xy") is not None
    ):  # a still ball keeps the rule below as it was
        poses = folder / "poses_3d_on_field.csv"
        feet = feet_at(poses, int(contact["frame"])) if poses.is_file() else {}
        result = classify(tuple(contact["kick_xy"]), feet, contact.get("pre_contact"))
        result["pre_contact"] = contact.get("pre_contact")
        result["contact_frame"] = int(contact["frame"])
        return result
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


SET_PIECE_CHOICES = (
    "auto",
    "penalty",
    "free_kick",
    "kick",
    "open_play",
    "dribble",
    "pass",
)  # GUI box, ``--set-piece``
WALL_KINDS = ("free_kick", "free_kick_wall", "unknown")  # the kinds whose report draws a wall
GOAL_KINDS = ("penalty", "free_kick", "free_kick_wall", "kick")  # ball to goal → flight TV report
OPENPLAY_KINDS = ("open_play", "dribble", "pass")  # no wall; no shot-at-goal report


def has_wall(kind: str | None) -> bool:
    """True for a free kick (the 9.15 m wall belongs to the report); False for penalty / kick / open play."""
    return (kind or "unknown") in WALL_KINDS


def is_goal_kick(kind: str | None) -> bool:
    """True when the analysis treats the ball as shot toward the goal (flight TV report)."""
    return (kind or "") in GOAL_KINDS


def is_openplay_analysis(kind: str | None) -> bool:
    """True for the dedicated dribble / pass planar reports (not plain open_play data-only)."""
    return (kind or "") in ("dribble", "pass")


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
