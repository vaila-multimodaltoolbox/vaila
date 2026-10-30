"""
================================================================================
Script: freekiki_teams.py - roles, teams and real stature of the people of a two-camera run
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    Who is who on the pitch, from the colour inside each person's box only (the grass is outside the
    box, so it does not take part):

    * the torso of each SAM box (20 to 55 % of its height, 25 to 75 % of its width) is classified
      pixel by pixel in HSV (green, blue, orange, yellow). TEAM_A is the colour named by the user
      (``team_a_colour``, green in the 28 Oct clip); TEAM_B is the other colour that most field players
      wear; the referees wear the colour left over, and the goalkeeper has his own colour;
    * the colour of a player is the one that covers the most torso pixels over the frames of the
      track, counted in both cameras when the player is paired; it needs at least 15 % of the pixels;
    * roles come from ``player_map.csv``: ``0`` the goalkeeper and ``1`` the kicker. The goalkeeper
      is in the team opposite to the kicker (the one defending the goal the kick attacks), because
      his shirt colour is not a team colour. The referees are the orange boxes, ordered by distance to
      the ball at the first frame: ``REF_MAIN``, ``REF_AUX1``, ``REF_AUX2``;
    * no more than 11 players per team (goalkeeper included) and 3 referees: the extra ones are
      ``unassigned``, with the reason;
    * stature: the height of each paired person in metres, from the triangulated skeleton of
      :mod:`freekiki_pose_binocular` (metric, by the DLT of both cameras, with no stature prior).
      The top of the head is the nose plus ``NOSE_TO_TOP_M`` (an assumption); the feet are on the
      ground (z = 0). Only upright frames count. A height outside ``STATURE_RANGE_M`` is not reported.

    The labels ``ID <player> <role> <team> <height> m`` replace the SAM camera translation, which is
    not a height. ``render_labelled`` draws them on the SAM 3 + DINOv3 overlay of each camera.
"""

from __future__ import annotations

import csv
import gzip
import json
import math
from pathlib import Path

import numpy as np

GREEN, BLUE, ORANGE, YELLOW = "green", "blue", "orange", "yellow"
REF_TEAM = "REF"
MAX_PER_TEAM = 11  # goalkeeper included
MAX_REFEREES = 3
MIN_PURITY = 0.15  # the colour must cover at least this share of the torso pixels
TORSO = (0.25, 0.20, 0.75, 0.55)  # x0, y0, x1, y1 as fractions of the SAM box
UPRIGHT_COS = math.cos(math.radians(20.0))  # the trunk within 20 degrees of the vertical
NOSE_TO_TOP_M = 0.10  # assumption: the nose sits about 10 cm below the top of the head
STATURE_RANGE_M = (1.40, 2.10)  # a stature outside this range is not reported
MIN_UPRIGHT_FRAMES = 5
ROLE_TAG = {
    "keeper": "GK",
    "kicker": "KICKER",
    "referee_main": "REF_MAIN",
    "referee_aux1": "REF_AUX1",
    "referee_aux2": "REF_AUX2",
}
REF_ROLES = ("referee_main", "referee_aux1", "referee_aux2")


def colour_counts(bgr: np.ndarray) -> tuple[dict[str, int], int]:
    """Pixels of each shirt colour in a BGR crop, and the number of pixels in the crop."""
    import cv2

    hsv = cv2.cvtColor(bgr, cv2.COLOR_BGR2HSV)
    h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    masks = {
        ORANGE: ((h <= 14) | (h >= 165)) & (s >= 120) & (v >= 120),
        YELLOW: (h >= 18) & (h <= 34) & (s >= 120) & (v >= 140),
        BLUE: (h >= 95) & (h <= 125) & (s >= 50) & (v >= 100),
        GREEN: (h >= 48) & (h <= 95) & (s >= 50) & (v >= 40),
    }
    return {name: int(mask.sum()) for name, mask in masks.items()}, int(h.size)


def torso_crop(image: np.ndarray, box) -> np.ndarray | None:
    """The upper body of a SAM box (the shirt), clipped to the image; None when empty."""
    height, width = image.shape[:2]
    x1, y1, x2, y2 = (float(v) for v in box[:4])
    w, h = x2 - x1, y2 - y1
    a = int(max(0, x1 + TORSO[0] * w))
    b = int(max(0, y1 + TORSO[1] * h))
    c = int(min(width, x1 + TORSO[2] * w))
    e = int(min(height, y1 + TORSO[3] * h))
    if e <= b or c <= a:
        return None
    return image[b:e, a:c]


def read_predictions(path: Path | str) -> tuple[dict[int, list[dict]], list[str]]:
    """Frame -> SAM instances, and the keypoint names, from a SAM 3 + DINOv3 predictions file."""
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        data = json.load(fh)
    frames = {int(f["frame"]): f["instances"] for f in data["frames"]}
    for instances in frames.values():
        for inst in instances:
            inst["keypoints_2d"] = inst["keypoints_2d_px"]  # the names the overlay draws from
            inst["bbox"] = inst["bbox_xyxy"]
    return frames, list(data["keypoint_names"])


def colour_of_tracks(video: Path | str, frames: dict[int, list[dict]]) -> dict[int, dict]:
    """SAM object id -> ``{"counts": {colour: pixels}, "pixels": torso pixels, "frames": n}``."""
    import cv2

    tracks: dict[int, dict] = {}
    cap = cv2.VideoCapture(str(video))
    try:
        index = 0
        while True:
            ok, image = cap.read()
            if not ok:
                break
            for inst in frames.get(index, []):
                crop = torso_crop(image, inst["bbox_xyxy"])
                if crop is None or crop.size == 0:
                    continue
                counts, n = colour_counts(crop)
                entry = tracks.setdefault(
                    int(inst["sam_obj_id"]),
                    {"counts": dict.fromkeys(counts, 0), "pixels": 0, "frames": 0},
                )
                for colour, value in counts.items():
                    entry["counts"][colour] += value
                entry["pixels"] += n
                entry["frames"] += 1
            index += 1
    finally:
        cap.release()
    return tracks


def add_tracks(total: dict, extra: dict) -> dict:
    """Sums the counts of one player seen in both cameras."""
    if not total:
        return {
            "counts": dict(extra["counts"]),
            "pixels": extra["pixels"],
            "frames": extra["frames"],
        }
    for colour, value in extra["counts"].items():
        total["counts"][colour] = total["counts"].get(colour, 0) + value
    total["pixels"] += extra["pixels"]
    total["frames"] += extra["frames"]
    return total


def dominant(entry: dict | None) -> tuple[str | None, float]:
    """The colour that covers the most torso pixels, and its share (None below ``MIN_PURITY``)."""
    if not entry or entry["pixels"] <= 0:
        return None, 0.0
    colour, value = max(entry["counts"].items(), key=lambda kv: kv[1])
    purity = value / entry["pixels"]
    return (colour, purity) if purity >= MIN_PURITY else (None, purity)


def read_player_map(path: Path | str) -> dict[str, dict]:
    with Path(path).open(newline="", encoding="utf-8") as fh:
        return {row["player_id"]: row for row in csv.DictReader(fh)}


def stature_by_player(poses_csv: Path | str) -> dict[int, dict]:
    """Height (m) of each paired person from the upright frames of the triangulated skeleton."""
    people: dict[int, dict[int, dict[str, np.ndarray]]] = {}
    with Path(poses_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["valid"] != "True" or row["kpt_name"] == "":
                continue
            frame = int(row["frame"])
            xyz = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            people.setdefault(int(row["player_id"]), {}).setdefault(frame, {})[row["kpt_name"]] = (
                xyz
            )
    out: dict[int, dict] = {}
    for pid, frames in people.items():
        tops: list[float] = []
        heels: list[float] = []
        for pts in frames.values():
            need = ("nose", "left-shoulder", "right-shoulder", "left-hip", "right-hip")
            if not all(name in pts for name in need):
                continue
            mid_sh = (pts["left-shoulder"] + pts["right-shoulder"]) / 2
            mid_hip = (pts["left-hip"] + pts["right-hip"]) / 2
            trunk = mid_sh - mid_hip
            norm = float(np.linalg.norm(trunk))
            if norm <= 0 or trunk[2] / norm < UPRIGHT_COS:
                continue
            tops.append(float(pts["nose"][2]) + NOSE_TO_TOP_M)
            feet = [pts[n][2] for n in ("left-heel", "right-heel") if n in pts]
            if feet:
                heels.append(min(feet))
        if len(tops) < MIN_UPRIGHT_FRAMES or not heels:
            out[pid] = {
                "height_m": None,
                "frames": len(tops),
                "reason": "too few upright frames with the head and the heels",
            }
            continue
        height = float(np.median(tops) - np.median(heels))
        if not STATURE_RANGE_M[0] <= height <= STATURE_RANGE_M[1]:
            out[pid] = {
                "height_m": None,
                "frames": len(tops),
                "reason": f"{height:.2f} m is outside {STATURE_RANGE_M} (not reported)",
            }
            continue
        out[pid] = {"height_m": round(height, 2), "frames": len(tops), "reason": ""}
    return out


def height_from_points(L: np.ndarray, foot_px, top_px) -> float | None:
    """Height (m) of the point at ``top_px`` above the ground point under ``foot_px``.

    The foot is on the ground (z = 0): its position comes from the plane homography of the DLT. The top
    point lies on the vertical through that foot; its height is the one that projects onto its pixel
    (least squares over the two image coordinates). The calibration of the field is metric, and the
    goal posts (2.44 m) and the corner flags (1.5 m) fix the vertical scale of the camera.
    """
    try:
        from .monocular_dlt_align import dlt_projection_matrix
    except ImportError:
        from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]

    P = dlt_projection_matrix(np.asarray(L, dtype=float))
    fu, fv = float(foot_px[0]), float(foot_px[1])
    tu, tv = float(top_px[0]), float(top_px[1])
    ground = np.linalg.solve(P[:, [0, 1, 3]], np.array([fu, fv, 1.0]))
    if abs(ground[2]) < 1e-12:
        return None
    X, Y = ground[0] / ground[2], ground[1] / ground[2]
    a = P[:, :2] @ np.array([X, Y]) + P[:, 3]  # the ground point, homogeneous, at height 0
    b = P[:, 2]  # the vertical direction, homogeneous
    M = np.array([b[0] - tu * b[2], b[1] - tv * b[2]])
    rhs = np.array([tu * a[2] - a[0], tv * a[2] - a[1]])
    denom = float(M @ M)
    if denom <= 0:
        return None
    return float(M @ rhs) / denom


def stature_from_keypoints(
    L_by_frame: dict[int, np.ndarray], frames: dict[int, list[dict]], sam_id: int, names: list[str]
) -> dict:
    """Median stature (m) of one track from the SAM keypoints of its upright frames, one camera.

    Head: the nose plus ``NOSE_TO_TOP_M``. Feet: the mean of the visible heels, on the ground. Only the
    frames where the trunk is near the vertical in the image count (the shoulders above the hips).
    """
    idx = {name: i for i, name in enumerate(names)}
    need = ("nose", "left-shoulder", "right-shoulder", "left-hip", "right-hip")
    if not all(n in idx for n in need) or not any(n.endswith("heel") for n in idx):
        return {"height_m": None, "frames": 0, "reason": "keypoints missing in this model"}
    values = []
    for f, instances in frames.items():
        L = L_by_frame.get(f)
        if L is None:
            continue
        for inst in instances:
            if int(inst["sam_obj_id"]) != sam_id:
                continue
            kp = np.asarray(inst.get("keypoints_2d_px") or [], dtype=float)
            if kp.shape[0] <= max(idx.values()):
                continue
            pts = {n: kp[idx[n]] for n in need + ("left-heel", "right-heel") if n in idx}
            if not all(np.isfinite(pts[n]).all() for n in need):
                continue
            trunk = (pts["left-shoulder"] + pts["right-shoulder"]) / 2 - (
                pts["left-hip"] + pts["right-hip"]
            ) / 2
            if abs(trunk[0]) > 0.35 * abs(trunk[1]) or trunk[1] >= 0:
                continue  # the shoulders must lie above the hips, nearly on the vertical
            heels = [
                pts[n]
                for n in ("left-heel", "right-heel")
                if n in pts and np.isfinite(pts[n]).all()
            ]
            if not heels:
                continue
            foot = np.mean(heels, axis=0)
            h = height_from_points(L, foot, pts["nose"])
            if h is not None and np.isfinite(h):
                values.append(h + NOSE_TO_TOP_M)
    if len(values) < MIN_UPRIGHT_FRAMES:
        return {"height_m": None, "frames": len(values), "reason": "too few upright frames"}
    value = float(np.median(values))
    if not STATURE_RANGE_M[0] <= value <= STATURE_RANGE_M[1]:
        return {"height_m": None, "frames": len(values),
                "reason": f"{value:.2f} m is outside {STATURE_RANGE_M} (not reported)"}  # fmt: skip
    return {"height_m": round(value, 2), "frames": len(values), "reason": ""}


def _ball_rest(ball_csv: Path | str) -> tuple[float, float] | None:
    with Path(ball_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx"):
                return float(row["cx"]), float(row["cy"])
    return None


def _foot_pixel(box) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3])


def choose_team_colours(
    colours: dict[str, dict], roles: dict[str, str], team_a_colour: str
) -> dict[str, str | None]:
    """The colours of TEAM_A, TEAM_B and the referees, from the people on the pitch.

    TEAM_A is the colour the user names (``team_a_colour``). TEAM_B is the other colour that most
    field players wear, counted by players and then by frames; the goalkeeper is left out. The referees
    wear the most frequent colour that is neither a team colour nor the goalkeeper's.
    """
    colour_of = {pid: dominant(colours.get(pid))[0] for pid in roles}
    keeper = next((pid for pid, role in roles.items() if role == "keeper"), None)
    keeper_colour = colour_of.get(keeper) if keeper is not None else None

    def tally(exclude: set) -> dict[str, tuple[int, int]]:
        count: dict[str, tuple[int, int]] = {}
        for pid, colour in colour_of.items():
            if pid == keeper or colour is None or colour in exclude:
                continue
            players, frames = count.get(colour, (0, 0))
            count[colour] = (players + 1, frames + colours.get(pid, {}).get("frames", 0))
        return count

    field = tally({team_a_colour})
    team_b = max(field, key=lambda c: field[c]) if field else None
    rest = tally({team_a_colour, team_b, keeper_colour})
    referee = max(rest, key=lambda c: rest[c]) if rest else None
    return {"TEAM_A": team_a_colour, "TEAM_B": team_b, REF_TEAM: referee}


def assign(
    colours: dict[str, dict],
    roles: dict[str, str],
    referee_order: list[str],
    team_colours: dict[str, str | None] | None = None,
    referee_colour: str | None = ORANGE,
) -> dict[str, dict]:
    """Team and role of each player id. ``colours`` maps player id -> summed colour entry.

    ``team_colours`` (from :func:`choose_team_colours`) says which colour is each team; the default is
    the green and blue of this clip. ``referee_order`` lists the people with the referee colour, nearest
    to the ball first.
    """
    if team_colours is None:
        team_colours = {"TEAM_A": GREEN, "TEAM_B": BLUE}
    team_of = {colour: team for team, colour in team_colours.items()
               if team in ("TEAM_A", "TEAM_B") and colour}  # fmt: skip
    opposite = {"TEAM_A": "TEAM_B", "TEAM_B": "TEAM_A"}
    colour_of = {pid: dominant(colours.get(pid)) for pid in roles}
    kicker = next((pid for pid, role in roles.items() if role == "kicker"), None)
    keeper = next((pid for pid, role in roles.items() if role == "keeper"), None)
    rows: dict[str, dict] = {}
    for pid in roles:
        colour, purity = colour_of[pid]
        rows[pid] = {"role": roles[pid], "colour": colour or "", "purity": round(purity, 3),
                     "team": "", "reason": ""}  # fmt: skip

    kicker_team = team_of.get(colour_of[kicker][0]) if kicker is not None else None
    if kicker is not None and kicker_team is not None:
        rows[kicker]["team"] = kicker_team
    if keeper is not None:
        if kicker_team is not None:
            rows[keeper]["team"] = opposite[kicker_team]
            rows[keeper]["reason"] = "goalkeeper: opposite team of the kicker"
        else:
            rows[keeper]["reason"] = "goalkeeper: the kicker has no team colour"

    for pid in roles:
        if rows[pid]["team"] or pid in (kicker, keeper):
            continue
        colour = colour_of[pid][0]
        if colour is None or colour == referee_colour:
            continue  # referees are handled below
        if colour in team_of:
            rows[pid]["team"] = team_of[colour]

    for pid in roles:
        if roles[pid] in REF_ROLES or referee_colour is None or colour_of[pid][0] != referee_colour:
            continue
        rows[pid]["team"] = REF_TEAM
    for pid, role in zip(referee_order[:MAX_REFEREES], REF_ROLES, strict=False):
        rows[pid]["role"] = role
        rows[pid]["team"] = REF_TEAM
    for pid in referee_order[MAX_REFEREES:]:
        rows[pid]["team"] = ""
        rows[pid]["reason"] = f"more than {MAX_REFEREES} referee-coloured people: not a referee"

    for team in ("TEAM_A", "TEAM_B"):
        members = [pid for pid, r in rows.items() if r["team"] == team]
        members.sort(key=lambda pid: -(colours.get(pid, {}).get("frames", 0) * rows[pid]["purity"]))
        for pid in members[MAX_PER_TEAM:]:
            rows[pid]["team"] = ""
            rows[pid]["reason"] = f"more than {MAX_PER_TEAM} players in {team}: not assigned"
    return rows


def label_text(pid: str, role: str, team: str, height: float | None) -> str:
    """``ID 0 GK TEAM_B 1.84 m`` (the role and the team are left out when unknown)."""
    parts = [f"ID {pid}"]
    if role in ROLE_TAG:
        parts.append(ROLE_TAG[role])
    if team and team != REF_TEAM:
        parts.append(team)
    elif team == REF_TEAM and role not in ROLE_TAG:
        parts.append(REF_TEAM)
    if height is not None:
        parts.append(f"{height:.2f} m")
    return " ".join(parts)


def render_labelled(
    predictions: Path | str,
    video: Path | str,
    labels: dict[int, str],
    out_mp4: Path | str,
    *,
    only: set[int] | None = None,
) -> Path:
    """The SAM 3 + DINOv3 skeleton overlay of a camera with ``labels`` (SAM object id -> text)."""
    import cv2

    try:
        from .sam3dinov3 import _draw_pose_overlay, skeleton_edges
    except ImportError:
        from sam3dinov3 import (  # ty: ignore[unresolved-import]
            _draw_pose_overlay,
            skeleton_edges,
        )

    frames, names = read_predictions(predictions)
    edges = skeleton_edges(names)
    cap = cv2.VideoCapture(str(video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width, height = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    out = Path(out_mp4)
    out.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out), cv2.VideoWriter.fourcc(*"mp4v"), fps, (width, height))
    index = 0
    try:
        while True:
            ok, image = cap.read()
            if not ok:
                break
            instances = frames.get(index, [])
            if only is not None:
                instances = [i for i in instances if int(i["sam_obj_id"]) in only]
            if instances:
                image = _draw_pose_overlay(
                    image, instances, edges, names, draw_ids=True, label_texts=labels
                )
            writer.write(image)
            index += 1
    finally:
        cap.release()
        writer.release()
    return out
