"""
================================================================================
Script: freekiki_team_analysis.py - the teams of one analysis (binocular or one camera)
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
    The people of one analysis, labelled and written to folders. An analysis is the binocular run
    (two cameras) or the monocular run of one camera; the same code serves both.

    * Only people on the pitch are used: the ``keep`` flag of ``players_on_field.csv`` (the feet on the
      pitch in most of the checked frames). A person on the pitch without a 3D track is listed in
      ``data/not_on_pitch.csv`` with the reason.
    * Teams and roles come from :mod:`freekiki_teams` (the torso colour inside each box, the goalkeeper
      opposite the kicker, the referees nearest the ball). In a two-camera analysis the other camera's
      track follows the ``partner`` column of the reference camera's ``player_map.csv``.
    * ``team_A/``, ``team_B/`` and ``referees/`` each hold one ``.c3d`` per player of that team and one
      ``overlay_<camera>.mp4`` per camera that draws only the players of that team.
    * ``data/`` holds the tables: ``team_map.csv``, ``team_colours.json``, ``labels_<camera>.json``,
      ``not_on_pitch.csv`` and ``analysis.json``.
    * Heights come from the reference camera's keypoints only when that camera's vertical scale is
      checked: the detected field points above the ground must reproduce the DLT model (the goal posts
      at 2.44 m) within ``VERTICAL_TOLERANCE_PX``. The tolerance is an operational choice.
"""

from __future__ import annotations

import csv
import glob
import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .freekiki_export import write_c3d
    from .freekiki_pose_binocular import KPT_COUNT
    from .freekiki_teams import (
        GREEN,
        REF_TEAM,
        add_tracks,
        assign,
        choose_team_colours,
        colour_of_tracks,
        dominant,
        label_text,
        read_player_map,
        read_predictions,
        render_labelled,
        stature_from_keypoints,
    )
except ImportError:
    from freekiki_export import write_c3d  # ty: ignore[unresolved-import]
    from freekiki_pose_binocular import KPT_COUNT  # ty: ignore[unresolved-import]
    from freekiki_teams import (  # ty: ignore[unresolved-import]
        GREEN,
        REF_TEAM,
        add_tracks,
        assign,
        choose_team_colours,
        colour_of_tracks,
        dominant,
        label_text,
        read_player_map,
        read_predictions,
        render_labelled,
        stature_from_keypoints,
    )

VERTICAL_TOLERANCE_PX = 1.5  # operational: the field points above the ground, in pixels
TEAM_FOLDERS = {"TEAM_A": "team_A", "TEAM_B": "team_B", REF_TEAM: "referees"}


@dataclass
class Source:
    """One camera of the analysis: its 3D folder, its cut video and its work folder."""

    name: str  # "ref" or "other": the names of the overlay files
    folder: Path  # the camera's 3D folder (player_map.csv, players_on_field.csv, ball.csv)
    video: Path  # the cut video of the camera
    work: Path  # work/<stem>: the SAM 3 + DINOv3 predictions and the field keypoints
    dlt: dict[int, np.ndarray] | None = None  # the camera's DLT per frame (for the stature)


def on_pitch(folder: Path) -> set[int]:
    """The SAM ids of the people on the pitch (``keep`` of ``players_on_field.csv``)."""
    with (Path(folder) / "players_on_field.csv").open(newline="", encoding="utf-8") as fh:
        return {int(r["person_id"]) for r in csv.DictReader(fh) if r["keep"] == "True"}


def _predictions(work: Path) -> Path:
    found = sorted(Path(work).glob("sam3dinov3/**/*_sam3dinov3_predictions.json.gz"))
    if not found:
        raise FileNotFoundError(f"no SAM 3 + DINOv3 predictions under {work}")
    return found[-1]


def vertical_residual_px(folder: Path, work: Path) -> float:
    """Median distance (px) between the detected field points and the DLT model, above the ground.

    The model points are the ones of ``field_kps_3d.csv`` with z > 0 (the goal posts at 2.44 m, the
    corner flags at 1.5 m); the detected points are ``field_kps_getpixelvideo.csv``. NaN when there is
    no such point, so the check cannot pass.
    """
    model: dict[tuple[int, int], tuple[float, float, float]] = {}
    if not (Path(folder) / "field_kps_3d.csv").is_file():
        return math.nan
    with (Path(folder) / "field_kps_3d.csv").open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if float(row["z_m"]) > 0 and row["u_px"] and row["v_px"]:
                model[(int(row["frame"]), int(row["point"]))] = (
                    float(row["u_px"]), float(row["v_px"]), float(row["z_m"])
                )  # fmt: skip
    found = sorted(
        glob.glob(str(Path(work) / "freekiki_predict_*" / "field_kps_getpixelvideo.csv"))
    )
    if not found or not model:
        return math.nan
    devs: list[float] = []
    with Path(found[-1]).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            frame = int(row["frame"])
            for (f, point), (u, v, _z) in model.items():
                if f != frame or not row.get(f"p{point}_x"):
                    continue
                devs.append(
                    math.hypot(float(row[f"p{point}_x"]) - u, float(row[f"p{point}_y"]) - v)
                )
    return float(np.median(devs)) if devs else math.nan


def read_person_points(poses_csv: Path) -> tuple[dict[int, np.ndarray], list[str]]:
    """``player_id -> (frames, KPT_COUNT, 3)`` metres of the valid keypoints, and the keypoint names.

    The binocular poses carry a ``valid`` column; the monocular ones do not, and every row with a
    coordinate is used. A missing keypoint stays NaN.
    """
    names = [""] * KPT_COUNT
    rows: list[dict] = []
    with Path(poses_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("valid", "True") != "True" or row["x_m"] == "":
                continue
            rows.append(row)
            k = int(row["kpt_idx"])
            if 0 <= k < KPT_COUNT and row["kpt_name"]:
                names[k] = row["kpt_name"]
    if not rows:
        return {}, [name or f"kpt_{i:03d}" for i, name in enumerate(names)]
    n = max(int(r["frame"]) for r in rows) + 1
    people: dict[int, np.ndarray] = {}
    for row in rows:
        pid, k, f = int(row["player_id"]), int(row["kpt_idx"]), int(row["frame"])
        if not 0 <= k < KPT_COUNT:
            continue
        arr = people.setdefault(pid, np.full((n, KPT_COUNT, 3), np.nan))
        arr[f, k] = (float(row["x_m"]), float(row["y_m"]), float(row["z_m"]))
    return people, [name or f"kpt_{i:03d}" for i, name in enumerate(names)]


def _partner_sam(row: dict) -> int | None:
    """The other camera's SAM id of a paired row (``partner`` is ``<3D folder>:<sam id>``)."""
    text = row.get("partner", "")
    if not text:
        return None
    return int(text.rsplit(":", 1)[1])


def _ball_rest(ball_csv: Path) -> tuple[float, float] | None:
    with Path(ball_csv).open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row.get("cx"):
                return float(row["cx"]), float(row["cy"])
    return None


def _foot_pixel(box) -> tuple[float, float]:
    return (float(box[0]) + float(box[2])) / 2.0, float(box[3])


def run_team_analysis(
    sources: list[Source],
    out_dir: Path | str,
    *,
    poses_csv: Path | str,
    ball_csv: Path | str,
    kind: str,
    rate_hz: float,
    team_a_colour: str = GREEN,
) -> dict:
    """Teams, roles and folders of one analysis. ``sources[0]`` is the reference camera.

    ``kind`` is ``binocular`` (two sources) or ``mono-ref`` / ``mono-other`` (one source), and goes
    into ``data/analysis.json``. Returns the counts and the colours found.
    """
    if not 1 <= len(sources) <= 2:
        raise ValueError("an analysis has one camera (monocular) or two (binocular)")
    out = Path(out_dir)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    keeps = [on_pitch(s.folder) for s in sources]
    maps = [read_player_map(s.folder / "player_map.csv") for s in sources]

    pid_of: list[dict[int, str]] = [{} for _ in sources]  # SAM id -> player id, per camera
    role_of: dict[str, str] = {}
    not_on_pitch: list[dict] = []
    for pid, row in maps[0].items():
        sam = int(row["sam_id"])
        if sam in keeps[0]:
            pid_of[0][sam] = pid
            role_of[pid] = row["role"]
    if len(sources) == 2:
        for pid, row in maps[0].items():
            other = _partner_sam(row)
            if other is not None and other in keeps[1] and pid in role_of:
                pid_of[1][other] = pid
        for row in maps[1].values():
            sam = int(row["sam_id"])
            pid = row["player_id"]
            if sam in keeps[1] and sam not in pid_of[1] and pid not in role_of:
                pid_of[1][sam] = pid
                role_of[pid] = row["role"]

    predictions: list[tuple[Path, dict, list[str]]] = []
    colours: dict[str, dict] = {}
    for i, source in enumerate(sources):
        path = _predictions(source.work)
        frames, names = read_predictions(path)
        predictions.append((path, frames, names))
        for sam, entry in colour_of_tracks(source.video, frames).items():
            if sam in pid_of[i]:
                pid = pid_of[i][sam]
                colours[pid] = add_tracks(colours.get(pid, {}), entry)
        seen = set(frames_sams(frames))
        for sam in sorted(seen):
            if sam not in keeps[i]:
                reason = "not on the pitch"
            elif sam not in pid_of[i]:
                reason = "on the pitch but without a 3D track in this camera"
            else:
                continue
            not_on_pitch.append({"camera": source.name, "sam_id": sam, "reason": reason})

    pids = sorted({pid for mapping in pid_of for pid in mapping.values()}, key=int)
    roles = {pid: role_of[pid] for pid in pids}
    ball = _ball_rest(Path(ball_csv))
    ref_frames, ref_names = predictions[0][1], predictions[0][2]
    first_box: dict[str, list] = {}
    for f in sorted(ref_frames):
        for inst in ref_frames[f]:
            pid = pid_of[0].get(int(inst["sam_obj_id"]))
            if pid is not None and pid not in first_box:
                first_box[pid] = inst["bbox_xyxy"]

    def distance(pid: str) -> float:
        if ball is None or pid not in first_box:
            return math.inf
        fx, fy = _foot_pixel(first_box[pid])
        return math.hypot(fx - ball[0], fy - ball[1])

    team_colours = choose_team_colours(colours, roles, team_a_colour)
    referee_colour = team_colours[REF_TEAM]
    referees = [
        pid for pid in pids if referee_colour and dominant(colours.get(pid))[0] == referee_colour
    ]
    rows = assign(colours, roles, sorted(referees, key=distance), team_colours, referee_colour)

    residual = vertical_residual_px(sources[0].folder, sources[0].work)
    validated = bool(residual == residual and residual <= VERTICAL_TOLERANCE_PX)
    sam_of_ref = {pid: sam for sam, pid in pid_of[0].items()}
    heights: dict[str, dict] = {}
    for pid in pids:
        if not validated:
            heights[pid] = {"height_m": None, "frames": 0,
                            "reason": f"camera vertical scale not checked (goal posts off by "
                            f"{residual:.1f} px, tolerance {VERTICAL_TOLERANCE_PX} px)"}  # fmt: skip
        elif pid not in sam_of_ref or sources[0].dlt is None:
            heights[pid] = {
                "height_m": None,
                "frames": 0,
                "reason": "no 3D track in the reference camera",
            }
        else:
            heights[pid] = stature_from_keypoints(
                sources[0].dlt, ref_frames, sam_of_ref[pid], ref_names
            )

    fields = ["player_id", "role", "team", "colour", "purity", "colour_frames", "height_m",
              "height_frames", "reason"]  # fmt: skip
    with (data / "team_map.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for pid in pids:
            r = rows[pid]
            h = heights.get(pid, {})
            writer.writerow({
                "player_id": pid, "role": r["role"], "team": r["team"], "colour": r["colour"],
                "purity": r["purity"], "colour_frames": colours.get(pid, {}).get("frames", 0),
                "height_m": "" if h.get("height_m") is None else h["height_m"],
                "height_frames": h.get("frames", 0),
                "reason": "; ".join(x for x in (r["reason"], h.get("reason", "")) if x),
            })  # fmt: skip
    with (data / "not_on_pitch.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["camera", "sam_id", "reason"])
        writer.writeheader()
        writer.writerows(not_on_pitch)
    (data / "team_colours.json").write_text(json.dumps(team_colours, indent=2), encoding="utf-8")

    labels: list[dict[int, str]] = []
    for i, source in enumerate(sources):
        mapping = {
            sam: label_text(pid, rows[pid]["role"], rows[pid]["team"], heights.get(pid, {}).get("height_m"))
            for sam, pid in pid_of[i].items()
        }  # fmt: skip
        labels.append(mapping)
        (data / f"labels_{source.name}.json").write_text(
            json.dumps({str(k): v for k, v in mapping.items()}, indent=2), encoding="utf-8"
        )

    people, point_names = read_person_points(Path(poses_csv))
    written: dict[str, list[str]] = {}
    for team, folder_name in TEAM_FOLDERS.items():
        folder = out / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        written[folder_name] = []
        for pid in pids:
            if rows[pid]["team"] != team or int(pid) not in people:
                continue
            name = f"P{int(pid):02d}_{rows[pid]['role']}.c3d"
            write_c3d(folder / name, point_names, people[int(pid)], rate_hz=rate_hz)
            written[folder_name].append(name)
        for i, source in enumerate(sources):
            sams = {sam for sam, pid in pid_of[i].items() if rows[pid]["team"] == team}
            if sams:
                render_labelled(predictions[i][0], source.video, labels[i], folder / f"overlay_{source.name}.mp4",
                                only=sams)  # fmt: skip

    counts = {team: sum(1 for pid in pids if rows[pid]["team"] == team) for team in TEAM_FOLDERS}
    summary = {
        "kind": kind,
        "cameras": [s.name for s in sources],
        "team_colours": team_colours,
        "counts": counts,
        "vertical_residual_px": None if residual != residual else round(residual, 3),
        "heights_validated": validated,
        "c3d": written,
    }
    (data / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def frames_sams(frames: dict[int, list[dict]]) -> list[int]:
    """Every SAM object id seen in the predictions."""
    return sorted({int(inst["sam_obj_id"]) for instances in frames.values() for inst in instances})
