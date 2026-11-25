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
Version: 0.4.19
Created: 28 October 2026
Update Date: 24 November 2026

Description:
    The people of one analysis, labelled and written to folders. An analysis is the binocular run
    (two cameras) or the monocular run of one camera; the same code serves both.

    * Only people on the pitch are used: the ``keep`` flag of ``players_on_field.csv`` (the feet on the
      pitch in most of the checked frames). A person on the pitch without a 3D track is listed in
      ``data/not_on_pitch.csv`` with the reason.
    * Teams and roles come from :mod:`freekiki_teams` (the torso colour inside each box, the goalkeeper
      opposite the kicker, the referees nearest the ball). In a two-camera analysis the other camera's
      track follows the ``partner`` column of the reference camera's ``player_map.csv``.
    * With a filled roster (:mod:`freekiki_roster`, the user's labels of the reference camera's SAM
      ids) the teams and the referees come from the labels, never from the colour: A -> ``TEAM_A``,
      B -> ``TEAM_B``, REF -> ``REF``; a person the user left blank has no team ("not labelled in
      the roster"). The colour is still measured and written, for information only
      (``analysis.json`` says ``"teams_source": "roster"``, ``team_colours.json`` ``"source"``).
    * ``team_A/``, ``team_B/`` and ``referees/`` each hold one ``.c3d`` per player of that team and one
      ``overlay_<camera>.mp4`` per camera that draws only the players of that team.
    * ``data/`` holds the tables: ``team_map.csv``, ``team_colours.json``, ``labels_<camera>.json``,
      ``not_on_pitch.csv`` and ``analysis.json``.
    * Heights come from the reference camera's keypoints and the camera of every frame
      (:func:`frame_cameras`: the per-frame calibration the pipeline writes next to the SAM 3 outputs,
      the ``approximate`` frames left out, else the FreeKiki DLT3D): ``stature_from_chain`` of
      :mod:`freekiki_teams`, with ``height_frames`` and ``height_iqr_m`` in ``team_map.csv``. The
      scale is the calibrated ground plane of each frame, so a clip where the goal posts are not
      visible still gets heights; the posts check (the detected field points above the ground must
      reproduce the DLT model within ``VERTICAL_TOLERANCE_PX``, an operational choice) stays as the
      flag ``heights_validated`` of ``analysis.json``. Without any camera the heights are blank, with
      the posts check as the reason when it failed.
"""

from __future__ import annotations

import csv
import glob
import json
import math
import re
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np

try:
    from .freekiki_export import write_c3d
    from .freekiki_pose_binocular import KPT_COUNT
    from .freekiki_teams import (
        GREEN,
        REF_ROLES,
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
        stature_from_chain,
    )
except ImportError:
    from freekiki_export import write_c3d  # ty: ignore[unresolved-import]
    from freekiki_pose_binocular import KPT_COUNT  # ty: ignore[unresolved-import]
    from freekiki_teams import (  # ty: ignore[unresolved-import]
        GREEN,
        REF_ROLES,
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
        stature_from_chain,
    )

VERTICAL_TOLERANCE_PX = 1.5  # operational: the field points above the ground, in pixels
TEAM_FOLDERS = {"TEAM_A": "team_A", "TEAM_B": "team_B", REF_TEAM: "referees"}
ROSTER_TEAM = {"A": "TEAM_A", "B": "TEAM_B", "REF": REF_TEAM}  # the roster's team codes
NOT_LABELLED = "not labelled in the roster"
_EARLIER = re.compile(r"^(P\d+_.+\.c3d|overlay_.+\.mp4)$")  # what a team folder is written with


@dataclass
class Source:
    """One camera of the analysis: its 3D folder, its cut video and its work folder."""

    name: str  # "ref" or "other": the names of the overlay files
    folder: Path  # the camera's 3D folder (player_map.csv, players_on_field.csv, ball.csv)
    video: Path  # the cut video of the camera
    work: Path  # work/<stem>: the SAM 3 + DINOv3 predictions and the field keypoints
    dlt: dict[int, np.ndarray] | None = None  # the camera's DLT per frame (for the stature)


def frame_cameras(source: Source) -> tuple[dict[int, np.ndarray] | None, str]:
    """``(frame -> camera, where it comes from)`` for the statures of one camera; usable frames only.

    The per-frame calibration the pipeline writes next to the SAM 3 outputs
    (``kick_measure.find_frame_calibration`` / ``read_frame_calibration``: every frame of the video,
    the ``approximate`` ones left out) is preferred; otherwise ``source.dlt`` (the FreeKiki DLT3D of
    the frames that have one). ``kick_measure`` is imported here, lazily, and any failure to read the
    file falls back, so the analysis works when it is missing. ``(None, "")`` without any camera.
    """
    try:
        try:
            from . import kick_measure as km
        except ImportError:
            import kick_measure as km  # ty: ignore[unresolved-import]
        finder = getattr(km, "find_frame_calibration", None)
        reader = getattr(km, "read_frame_calibration", None)
        found = finder(source.work, Path(source.work).name) if finder and reader else None
        if found is not None and reader is not None:
            track = reader(found)
            cameras = {
                f: np.asarray(track.matrix(f), dtype=float)
                for f in range(int(track.n_frames))
                if not track.is_approximate(f)
            }
            if cameras:
                return cameras, Path(found).name
    except Exception:  # noqa: BLE001 - a bad or missing file is a reason to fall back, not to stop
        pass
    if source.dlt:
        return dict(source.dlt), "freekiki dlt3d"
    return None, ""


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


def _roster_rows(
    pids: list[str],
    roles: dict[str, str],
    colours: dict[str, dict],
    sam_of_ref: dict[str, int],
    roster,
    distance,
) -> tuple[dict[str, dict], dict[str, str]]:
    """Team, role and reason of each player id from the roster (labels of the reference SAM ids).

    Returns ``(rows, names)``. The colour and its purity are measured as usual and only reported.
    ``roles`` holds the roles of ``player_map.csv`` (the keeper and the kicker are the ones the
    players stage took from the same roster); the roster's own role wins when it names one.
    Referees: the ``referee_main`` label is the main referee; the others, and the ones with no
    role, are ranked by their distance to the ball (``referee_main`` when no one was labelled so,
    then ``referee_aux1``, ``referee_aux2``).
    """
    rows: dict[str, dict] = {}
    names: dict[str, str] = {}
    for pid in pids:
        colour, purity = dominant(colours.get(pid))
        row = {"role": roles[pid], "colour": colour or "", "purity": round(purity, 3),
               "team": "", "reason": ""}  # fmt: skip
        sam = sam_of_ref.get(pid)
        entry = roster.entries.get(sam) if sam is not None else None
        if entry is None or not entry.team:
            row["reason"] = (
                NOT_LABELLED
                if sam is not None
                else (f"{NOT_LABELLED} (no track in the reference camera)")
            )
        elif entry.team == "OUT":
            row["reason"] = "roster: OUT (not used)"
        else:
            row["team"] = ROSTER_TEAM[entry.team]
            row["reason"] = f"roster: SAM {sam} labelled {entry.team}"
            if entry.role in ("keeper", "kicker"):
                row["role"] = entry.role
            if entry.name:
                names[pid] = entry.name
        if entry is not None and entry.team == "OUT" and entry.name:
            names[pid] = entry.name
        rows[pid] = row
    refs = [pid for pid in pids if rows[pid]["team"] == REF_TEAM]

    def label(pid: str) -> str:
        return roster.entries[sam_of_ref[pid]].role

    main = next((pid for pid in refs if label(pid) == "referee_main"), None)
    if main is None:  # no one labelled so: the nearest of the people with no role takes it
        blank = sorted((pid for pid in refs if label(pid) == ""), key=distance)
        main = blank[0] if blank else None
    if main is not None:
        rows[main]["role"] = "referee_main"
    aux = sorted((pid for pid in refs if pid != main), key=distance)
    for pid, role in zip(aux, REF_ROLES[1:], strict=False):
        rows[pid]["role"] = role
    return rows, names


def _retire_earlier_files(folder: Path, dest: Path) -> None:
    """Move the C3D and overlay files an earlier analysis left in a team folder to ``dest``.

    A rerun with other teams (new roster, new colours) must not leave the old ``P<id>_<role>.c3d``
    beside the new ones: a person would appear in two teams, or as a referee who is not one. Nothing
    is deleted; the files go to ``<analysis>/superseded/<time>/<folder>/``.
    """
    old = [p for p in folder.iterdir() if p.is_file() and (_EARLIER.match(p.name))]
    if not old:
        return
    dest.mkdir(parents=True, exist_ok=True)
    for path in old:
        shutil.move(str(path), str(dest / path.name))


def _roster_reason(team: str, reason: str) -> str:
    """Why a person the user labelled is not in the analysis (``team``: the roster's code)."""
    if team == "OUT":
        return "roster: OUT (not used)"
    return f"labelled {team} in the roster, but {reason}"


def _team_colours_of(rows: dict[str, dict], colours: dict[str, dict]) -> dict[str, str | None]:
    """The colour each team wears (information only): the commonest torso colour of its members.

    The goalkeeper is left out of a team, as his shirt is not the team's.
    """
    found: dict[str, str | None] = {}
    for team in TEAM_FOLDERS:
        count: dict[str, int] = {}
        for pid, r in rows.items():
            colour = dominant(colours.get(pid))[0]
            if r["team"] == team and r["role"] != "keeper" and colour:
                count[colour] = count.get(colour, 0) + 1
        found[team] = max(count, key=lambda c: count[c]) if count else None
    return found


def run_team_analysis(
    sources: list[Source],
    out_dir: Path | str,
    *,
    poses_csv: Path | str,
    ball_csv: Path | str,
    kind: str,
    rate_hz: float,
    team_a_colour: str = GREEN,
    roster=None,
) -> dict:
    """Teams, roles and folders of one analysis. ``sources[0]`` is the reference camera.

    ``kind`` is ``binocular`` (two sources) or ``mono-ref`` / ``mono-other`` (one source), and goes
    into ``data/analysis.json``. Returns the counts and the colours found.

    ``roster`` (:class:`vaila.freekiki_roster.Roster`, labels by SAM id of the reference camera):
    when it is filled the teams and the referees come from it and not from the colour (the colour
    is still measured and reported); ``analysis.json`` gets ``"teams_source": "roster"`` (else
    ``"colour"``). People labelled but not on the pitch or without 3D go to ``data/not_on_pitch.csv``
    with the reason; people left blank have no team.
    """
    if not 1 <= len(sources) <= 2:
        raise ValueError("an analysis has one camera (monocular) or two (binocular)")
    out = Path(out_dir)
    data = out / "data"
    data.mkdir(parents=True, exist_ok=True)
    use_roster = roster is not None and bool(roster.filled)
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
            if use_roster and i == 0 and roster.team_of(sam):
                reason = _roster_reason(roster.team_of(sam), reason)
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

    sam_of_ref = {pid: sam for sam, pid in pid_of[0].items()}
    names: dict[str, str] = {}
    team_colours: dict[str, str | None]
    if use_roster:
        rows, names = _roster_rows(pids, roles, colours, sam_of_ref, roster, distance)
        team_colours = _team_colours_of(rows, colours)
    else:
        team_colours = choose_team_colours(colours, roles, team_a_colour)
        referee_colour = team_colours[REF_TEAM]
        referees = [
            pid
            for pid in pids
            if referee_colour and dominant(colours.get(pid))[0] == referee_colour
        ]
        rows = assign(colours, roles, sorted(referees, key=distance), team_colours, referee_colour)

    residual = vertical_residual_px(sources[0].folder, sources[0].work)
    validated = bool(residual == residual and residual <= VERTICAL_TOLERANCE_PX)
    cameras, camera_origin = frame_cameras(sources[0])
    heights: dict[str, dict] = {}
    for pid in pids:
        if cameras is None and not validated:
            heights[pid] = {"height_m": None, "frames": 0, "iqr_m": None,
                            "reason": f"camera vertical scale not checked (goal posts off by "
                            f"{residual:.1f} px, tolerance {VERTICAL_TOLERANCE_PX} px)"}  # fmt: skip
        elif pid not in sam_of_ref or cameras is None:
            heights[pid] = {
                "height_m": None,
                "frames": 0,
                "iqr_m": None,
                "reason": "no 3D track in the reference camera",
            }
        else:
            # the chain method takes its scale from the calibrated ground plane of each frame, not
            # from the goal posts, so a clip where the posts are not visible still gives a height
            # (``heights_validated`` stays in analysis.json as a flag).
            heights[pid] = stature_from_chain(cameras, ref_frames, sam_of_ref[pid], ref_names)

    fields = ["player_id", "role", "team", "colour", "purity", "colour_frames", "height_m",
              "height_frames", "height_iqr_m", "reason", "name"]  # fmt: skip
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
                "height_iqr_m": "" if h.get("iqr_m") is None else h["iqr_m"],
                "reason": "; ".join(x for x in (r["reason"], h.get("reason", "")) if x),
                "name": names.get(pid, ""),
            })  # fmt: skip
    with (data / "not_on_pitch.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["camera", "sam_id", "reason"])
        writer.writeheader()
        writer.writerows(not_on_pitch)
    colours_file = {**team_colours, "source": "roster"} if use_roster else team_colours
    (data / "team_colours.json").write_text(json.dumps(colours_file, indent=2), encoding="utf-8")

    labels: list[dict[int, str]] = []
    for i, source in enumerate(sources):
        mapping = {
            sam: label_text(pid, rows[pid]["role"], rows[pid]["team"],
                            heights.get(pid, {}).get("height_m"), names.get(pid, ""))
            for sam, pid in pid_of[i].items()
        }  # fmt: skip
        labels.append(mapping)
        (data / f"labels_{source.name}.json").write_text(
            json.dumps({str(k): v for k, v in mapping.items()}, indent=2), encoding="utf-8"
        )

    people, point_names = read_person_points(Path(poses_csv))
    written: dict[str, list[str]] = {}
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    for team, folder_name in TEAM_FOLDERS.items():
        folder = out / folder_name
        folder.mkdir(parents=True, exist_ok=True)
        _retire_earlier_files(folder, out / "superseded" / stamp / folder_name)
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
        "team_colours": colours_file,
        "teams_source": "roster" if use_roster else "colour",
        "counts": counts,
        "vertical_residual_px": None if residual != residual else round(residual, 3),
        "heights_validated": validated,
        "height_method": "chain-p80" if cameras is not None else None,
        "height_cameras": camera_origin or None,
        "c3d": written,
    }
    (data / "analysis.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def frames_sams(frames: dict[int, list[dict]]) -> list[int]:
    """Every SAM object id seen in the predictions."""
    return sorted({int(inst["sam_obj_id"]) for instances in frames.values() for inst in instances})
