"""
================================================================================
Script: freekiki_report.py - TV-facing report (HTML + PDF) of a finished 3D ball run
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.13
Created: 26 October 2026
Update Date: 29 October 2026

Description:
    ``freekiki.py report-tv --run <3D folder>`` writes ``report_tv/`` inside the finished run,
    with one folder per language (``pt/`` and ``en/``): an HTML and a PDF that explain the kick
    for a journalist (numbers, physics, references, limits), the figures, and the slow-motion
    animation of :mod:`vaila.freekiki_report_video`. Every number comes from the run's own files
    (``ball_flight.json``, ``ball_3d_on_field.csv``, ``ball.csv``, the DLT3D of the detect run).

    The HTML also carries an editor: the anchors and the per-frame projections are embedded, and
    :file:`freekiki_report_app.js` (a port of the physics module) recomputes the fitted flight in
    the reader's browser when an anchor is moved.

    * :func:`load_run` - reads the run and the per-frame cameras;
    * :func:`compute_metrics` - distances, times, speeds, Re, Sp, Cd, the forces per frame,
      and the confidence tier from the anchor residuals;
    * :func:`make_figures` - the PNG figures, embedded in the HTML as base64;
    * :func:`editor_payload` - the data of the in-browser editor;
    * :func:`build_report` - fills the template of each language and prints the PDF with
      Chrome (or Chromium) headless.

    A penalty kick (``data/setpiece.json`` of the run says ``"kind": "penalty"``) has no wall: the
    report then drops the wall (anchor, rectangle, time to the wall, texts) and shows the
    goalkeeper's hands (the mean of the wrists of the keeper of the player map) at the kick and
    in the last frame, and their distance to the ball.

    The confidence tier decides the warning at the top: ``alta`` (anchors within 0.3 m),
    ``média`` (within 1 m) or ``baixa`` (1 m or more, the numbers of the track are an estimate).
"""

from __future__ import annotations

import base64
import copy
import csv
import json
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path

import numpy as np

try:
    from .freekiki_layout import resolve
except ImportError:
    from freekiki_layout import resolve  # ty: ignore[unresolved-import]
try:
    from .freekiki_detect3d import _video_from_readme
    from .freekiki_physics import (
        AIR_DENSITY,
        BALL_AREA,
        BALL_MASS,
        BALL_RADIUS,
        GRAVITY,
        PIXEL_SIGMA,
        drag_coefficient,
        lift_coefficient,
        reynolds,
        spin_parameter,
    )
    from .freekiki_report_i18n import (
        LANGS,
        editor_labels,
        fmt_num,
        localize_anchor_label,
        tr,
    )
    from .freekiki_report_players import (
        KEEPER,
        People,
        draw_goal_area,
        draw_hands,
        draw_people,
        find_people,
        load_camera_roles,
        load_ground_tracks,
        people_extent,
        people_note,
    )
    from .freekiki_report_video import write_lance_video
    from .freekiki_stereo import find_dlt3d
    from .monocular_dlt_align import dlt_projection_matrix
except ImportError:
    from freekiki_detect3d import _video_from_readme  # ty: ignore[unresolved-import]
    from freekiki_physics import (  # ty: ignore[unresolved-import]
        AIR_DENSITY,
        BALL_AREA,
        BALL_MASS,
        BALL_RADIUS,
        GRAVITY,
        PIXEL_SIGMA,
        drag_coefficient,
        lift_coefficient,
        reynolds,
        spin_parameter,
    )
    from freekiki_report_i18n import (  # ty: ignore[unresolved-import]
        LANGS,
        editor_labels,
        fmt_num,
        localize_anchor_label,
        tr,
    )
    from freekiki_report_players import (  # ty: ignore[unresolved-import]
        KEEPER,
        People,
        draw_goal_area,
        draw_hands,
        draw_people,
        find_people,
        load_camera_roles,
        load_ground_tracks,
        people_extent,
        people_note,
    )
    from freekiki_report_video import write_lance_video  # ty: ignore[unresolved-import]
    from freekiki_stereo import find_dlt3d  # ty: ignore[unresolved-import]
    from monocular_dlt_align import dlt_projection_matrix  # ty: ignore[unresolved-import]

HERE = Path(__file__).resolve().parent
TEMPLATES = {
    "pt": HERE / "freekiki_report_template.html",
    "en": HERE / "freekiki_report_template_en.html",
}
APP_JS = HERE / "freekiki_report_app.js"
# file names of each language: HTML, PDF and animation
FILE_NAMES = {
    "pt": ("relatorio_bola.html", "relatorio_bola.pdf", "animacao_lance.mp4"),
    "en": ("ball_report.html", "ball_report.pdf", "kick_animation.mp4"),
}
BARRIER_M = 9.15  # the distance of the wall from the kick (FIFA rule)
CROSSBAR_M = 2.44  # goal height
DRAG_K = 0.5 * AIR_DENSITY * BALL_AREA / BALL_MASS  # 1/m, a = DRAG_K * C * v^2
DEFAULT_ANCHOR_SIGMA = 0.3  # m, for anchor points written without a sigma
# the names of the anchors of the first version (no label written), with their text key
LEGACY_ANCHOR_KEYS = {
    "kick_foot": "anchor_start",
    "barrier_target": "anchor_barrier",
    "barrier_shoulder": "anchor_shoulder",
    "keeper_hands": "anchor_end",
}
HANDS_NEAR_M = 0.5  # the hands this close to the ball are within the error of the measurement
HANDS_REACH_FRAMES = 2  # a hand missing on the frame is taken from a frame this close
WRIST_NAMES = ("left-wrist", "right-wrist")  # the MHR70 has no hand point: the hands are the wrists
INK = "#14213d"
MUTED = "#5b6577"
BALL = "#d62246"
WALL = "#14934b"
ALT = "#f4a259"


@dataclass
class ReportRun:
    run_dir: Path
    video: Path
    fps: float
    n: int
    flight: dict
    track: np.ndarray  # (n, 3) metres, NaN where the ball is not on the pitch
    pixels: dict[int, tuple[float, float]]
    projections: dict[int, np.ndarray]  # frame -> 3x4 pixel projection (the DLT3D of that frame)
    anchors: dict  # the anchor points of the fit (anchor_points of ball_flight.json)
    people_dir: Path  # the folder with the poses and the player map (see _people_dir)
    penalty: bool = False  # data/setpiece.json says "penalty": no wall, the keeper's hands instead
    keeper_hands: np.ndarray | None = None  # (n, 3) mean of the keeper's wrists, penalty only
    goal_x: float | None = None  # the goal line of the kick (m), from data/setpiece.json


def _read_csv(path: Path) -> list[dict]:
    with path.open(newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _people_dir(run_dir: Path) -> Path:
    """The folder whose ``poses_3d_on_field.csv`` and ``player_map.csv`` the report reads.

    A binocular folder (``rec3d-binocular``) holds the flight but no poses: its ``summary.json``
    names the reference camera's 3D folder, which has them. Any other folder is its own.
    """
    if (run_dir / "poses_3d_on_field.csv").is_file():
        return run_dir
    summary = resolve(run_dir, "summary.json")
    if summary.is_file():
        ref = json.loads(summary.read_text(encoding="utf-8")).get("reference_run")
        if ref and (Path(ref) / "poses_3d_on_field.csv").is_file():
            return Path(ref)
    return run_dir


def _setpiece(people_dir: Path) -> dict:
    """``data/setpiece.json`` of the run (the kind of set piece the pipeline decided); {} if none."""
    path = Path(people_dir) / "data" / "setpiece.json"
    if not path.is_file():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def keeper_wrists(people_dir, person_id: int, n: int) -> np.ndarray | None:
    """The keeper's hands per frame, ``(n, 3)`` in pitch metres: the mean of the two wrists of the
    person ``person_id`` in ``poses_3d_on_field.csv``. NaN where no wrist was found; None when the
    file has no wrist of that person."""
    path = Path(people_dir) / "poses_3d_on_field.csv"
    if not path.is_file():
        return None
    sums = np.zeros((n, 3))
    count = np.zeros(n)
    with path.open(newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["kpt_name"] not in WRIST_NAMES or int(row["person_id"]) != person_id:
                continue
            f = int(row["frame"])
            if not 0 <= f < n or "" in (row["x_m"], row["y_m"], row["z_m"]):
                continue
            xyz = np.array([float(row["x_m"]), float(row["y_m"]), float(row["z_m"])])
            if np.isfinite(xyz).all():
                sums[f] += xyz
                count[f] += 1
    if not count.any():
        return None
    out = np.full((n, 3), np.nan)
    seen = count > 0
    out[seen] = sums[seen] / count[seen, None]
    return out


def _hands_at(hands: np.ndarray | None, frame: int) -> np.ndarray | None:
    """The hands on ``frame``, or on the nearest frame within ``HANDS_REACH_FRAMES`` that has them."""
    if hands is None:
        return None
    for d in range(HANDS_REACH_FRAMES + 1):
        for f in (frame - d, frame + d):
            if 0 <= f < len(hands) and np.isfinite(hands[f]).all():
                return np.asarray(hands[f], dtype=float).copy()
    return None


def load_run(run_dir, video=None) -> ReportRun:
    """Read a finished 3D folder (the one ``ball-rebuild`` or ``detect-soccer-3d`` wrote)."""
    import cv2

    run_dir = Path(run_dir)
    flight_path = resolve(run_dir, "ball_flight.json")
    if not flight_path.is_file():
        raise FileNotFoundError(
            f"{flight_path} not found: the report needs a fitted flight. In the GUI open "
            "Ball anchors... (≥3 points), Save, then Rebuild ball; or pass --ball-anchors "
            'to ball-rebuild (player ids or {"points": [...]})'
        )
    flight = json.loads(flight_path.read_text(encoding="utf-8"))
    video_path = Path(video) if video else _video_from_readme(run_dir)
    if video_path is None or not video_path.is_file():
        raise FileNotFoundError("The video is not in the README of the run; pass --video")
    cap = cv2.VideoCapture(str(video_path))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
    n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    if n <= 0:
        raise ValueError(f"Cannot read the frame count of {video_path}")

    track = np.full((n, 3), np.nan)
    for row in _read_csv(resolve(run_dir, "ball_3d_on_field.csv")):
        f = int(row["frame"])
        if 0 <= f < n and row["x_m"] != "":
            track[f] = [float(row["x_m"]), float(row["y_m"]), float(row["z_m"])]
    pixels = {
        int(r["frame"]): (float(r["cx"]), float(r["cy"]))
        for r in _read_csv(resolve(run_dir, "ball.csv"))
        if r["cx"] != ""
    }

    projections = {f: dlt_projection_matrix(L) for f, L in find_dlt3d(run_dir, video_path).items()}
    people_dir = _people_dir(run_dir)
    setpiece = _setpiece(people_dir)
    penalty = setpiece.get("kind") == "penalty"
    keeper = load_camera_roles(people_dir).get("keeper_id") if penalty else None
    goal_x = setpiece.get("goal_x_m")
    return ReportRun(
        run_dir=run_dir,
        video=video_path,
        fps=fps,
        n=n,
        flight=flight,
        track=track,
        pixels=pixels,
        projections=projections,
        anchors=flight.get("anchor_points", {}),
        people_dir=people_dir,
        penalty=penalty,
        keeper_hands=None if keeper is None else keeper_wrists(people_dir, keeper, n),
        goal_x=None if goal_x is None else float(goal_x),
    )


def confidence(flight: dict) -> tuple[str, float | None]:
    """``(tier, worst anchor error in metres)`` from the residuals of the fit.

    The tier keys are the Portuguese words (``alta``, ``média``, ``baixa``); the text of each
    language comes from :mod:`vaila.freekiki_report_i18n`.
    """
    values = [abs(v) for v in flight.get("anchor_residuals", {}).values() if v is not None]
    if not values:
        return "baixa", None
    worst = max(values)
    if worst < 0.3:
        return "alta", worst
    if worst < 1.0:
        return "média", worst
    return "baixa", worst


def compute_metrics(run: ReportRun) -> dict:
    """Every number of the report, from the run's files (no text, no language)."""
    flight = run.flight
    track = run.track
    finite = np.flatnonzero(np.isfinite(track).all(axis=1))
    if finite.size < 2:
        raise ValueError("The ball track has fewer than two frames on the pitch")
    kick_f = int(round(float(flight["kick_time_s"]) * run.fps))
    kick_f = min(max(kick_f, int(finite[0])), int(finite[-1]) - 1)
    end_f = int(finite[-1])
    speed = np.asarray(flight["speed_mps"], dtype=float)  # m/s per frame (0 at frame 0)
    omega = np.asarray(flight["theta"][6:9], dtype=float)
    omega_norm = float(np.linalg.norm(omega))

    seg = np.linalg.norm(np.diff(track[kick_f : end_f + 1], axis=0), axis=1)
    flight_s = (end_f - kick_f) / run.fps
    horizontal = float(np.hypot(*(track[end_f, :2] - track[kick_f, :2])))
    vx, vy, vz = (float(v) for v in flight["theta"][2:5])
    zmax_f = int(np.nanargmax(track[:, 2]))

    frames = np.arange(run.n)
    sp = np.array([spin_parameter(v, omega_norm) if v > 0 else 0.0 for v in speed])
    re = np.array([reynolds(v) if v > 0 else 0.0 for v in speed])
    cd = np.array(
        [drag_coefficient(v, s) if v > 0 else np.nan for v, s in zip(speed, sp, strict=True)]
    )
    cl = np.array([lift_coefficient(s) for s in sp])
    a_drag = np.where(speed > 0, DRAG_K * cd * speed**2, np.nan)
    a_magnus = np.where(speed > 0, DRAG_K * cl * speed**2, np.nan)

    penalty = run.penalty
    barrier_f, keeper_f, keeper_hands = event_frames(run.anchors, end_f, penalty=penalty)
    hands_start = _hands_at(run.keeper_hands, kick_f) if penalty else None
    hands_end = _hands_at(run.keeper_hands, end_f) if penalty else None
    if hands_end is not None:
        keeper_hands = hands_end
    radial = np.linalg.norm(track[:, :2] - track[kick_f, :2], axis=1)
    entry = track[end_f]
    tier, worst = confidence(flight)
    dev = _deviation_from_chord(track[kick_f : end_f + 1, :2])
    wall_ok = not penalty and barrier_f < run.n and np.isfinite(track[barrier_f, 2])
    nan = float("nan")
    return {
        "penalty": penalty,
        "anchors": display_anchors(run.anchors, penalty, kick_f, end_f, hands_start, hands_end),
        "hands_start": hands_start,
        "hands_end": hands_end,
        "hands_travel_m": float(np.linalg.norm(hands_end - hands_start))
        if hands_start is not None and hands_end is not None
        else None,
        "kick_frame": kick_f,
        "end_frame": end_f,
        "frames": frames,
        "distance_m": horizontal,
        "flight_s": flight_s,
        "path_m": float(seg.sum()),
        "mean_kmh": float(seg.sum() / flight_s * 3.6) if flight_s > 0 else float("nan"),
        "launch_kmh": float(np.hypot(np.hypot(vx, vy), vz) * 3.6),
        "launch_deg": float(np.degrees(np.arctan2(vz, np.hypot(vx, vy)))),
        "zmax_m": float(track[zmax_f, 2]),
        "zmax_t": float((zmax_f - kick_f) / run.fps),
        "spin_rps": float(flight["spin_rps"]),
        "spin_axis": flight["spin_axis"],
        "spin_vertical": abs(float(flight["spin_axis"][2])) > 0.8,
        "peak_kmh": float(flight["peak_speed_mps"] * 3.6),
        "end_kmh": float(speed[end_f] * 3.6),
        "speed_kmh": speed * 3.6,
        "re": re,
        "sp": sp,
        "cd": cd,
        "cl": cl,
        "a_drag": a_drag,
        "a_magnus": a_magnus,
        "a_gravity": GRAVITY,
        "barrier_f": barrier_f,
        "barrier_t": nan if penalty else float((barrier_f - kick_f) / run.fps),
        "barrier_kmh": float(speed[barrier_f] * 3.6) if wall_ok else nan,
        "barrier_to_goal_m": float(np.linalg.norm(entry - track[barrier_f])) if wall_ok else nan,
        "after_wall_s": nan if penalty else float(flight_s - (barrier_f - kick_f) / run.fps),
        "wall_z": float(track[barrier_f, 2]) if wall_ok else nan,
        "wall_clear_cm": float((track[barrier_f, 2] - 1.8) * 100) if wall_ok else nan,
        "re_min": float(np.nanmin(re[kick_f + 1 : end_f + 1])),
        "re_max": float(np.nanmax(re[kick_f + 1 : end_f + 1])),
        "keeper_f": keeper_f,
        "keeper_gap_m": float(np.linalg.norm(keeper_hands - entry))
        if np.isfinite(keeper_hands).all()
        else None,
        "radial": radial,
        "dev_m": dev["dev_max_m"],
        "dev_at_m": dev["dev_at_m"],
        "chord_m": dev["chord_m"],
        "rms_px": float(flight["rms_px"]),
        "max_px": float(flight["max_px"]),
        "per_frame_px": {int(k): float(v) for k, v in flight["per_frame_px"].items()},
        "anchor_residuals": flight["anchor_residuals"],
        "tier": tier,
        "method": str(flight.get("method", "anchors")),
        "worst_anchor_m": worst,
        "above_fit_range": bool(flight.get("above_fit_range", False)),
        "fit_success": bool(flight.get("success", True)),
    }


WALL_ANCHOR_KEYS = ("p_wall", "barrier_target", "barrier_shoulder")
WALL_WORDS = ("barreira", "barrier", "plano a 9,15", "wall", "ombro", "shoulder")
KEEPER_ANCHOR_KEYS = ("keeper_hands", "p_arrival")
KEEPER_WORDS = ("goleiro", "keeper", "mãos", "hands")


def _label(spec: dict) -> str:
    return str(spec.get("label", "")).lower()


def _is_wall_anchor(key: str, spec: dict) -> bool:
    return key in WALL_ANCHOR_KEYS or any(w in _label(spec) for w in WALL_WORDS)


def _is_keeper_anchor(key: str, spec: dict) -> bool:
    return key in KEEPER_ANCHOR_KEYS or any(w in _label(spec) for w in KEEPER_WORDS)


def display_anchors(
    anchors: dict,
    penalty: bool,
    kick_f: int,
    end_f: int,
    hands_start: np.ndarray | None,
    hands_end: np.ndarray | None,
) -> dict:
    """The anchor points the animation draws.

    A free kick draws them all. A penalty kick has no wall, so the wall anchors are dropped; the
    keeper's hands measured from the poses (at the kick and in the last frame) replace the
    keeper anchor of the fit, which is the same hand taken once. ``show_from`` is the first frame
    the point is drawn at and ``i18n`` the text key of its label.
    """
    if not penalty:
        return dict(anchors)
    shown = {
        k: v
        for k, v in anchors.items()
        if isinstance(v, dict) and "xyz" in v and not _is_wall_anchor(k, v)
    }
    if hands_start is None and hands_end is None:
        return shown
    shown = {k: v for k, v in shown.items() if not _is_keeper_anchor(k, v)}
    if hands_start is not None:
        shown["keeper_hands_start"] = {
            "frame": kick_f, "xyz": [float(c) for c in hands_start], "show_from": kick_f,
            "i18n": "vid_anchor_hands_start", "label_offset": [14, 22],
        }  # fmt: skip
    if hands_end is not None:
        shown["keeper_hands"] = {
            "frame": end_f, "xyz": [float(c) for c in hands_end], "show_from": end_f,
            "i18n": "vid_anchor_hands_end", "label_offset": [14, -30],
        }  # fmt: skip
    return shown


def event_frames(
    anchors: dict, end_f: int, *, penalty: bool = False
) -> tuple[int, int, np.ndarray]:
    """``(wall frame, keeper frame, keeper point)`` from the anchor points of a run.

    The wall is the anchor whose label says "barreira" or "barrier" (or the legacy
    ``barrier_target``, or else the middle anchor by frame); the keeper point is the legacy
    ``keeper_hands`` or else the last anchor by frame. A penalty has no wall: its "wall frame" is
    the last frame, and the keeper point is the anchor that names the goalkeeper's hand (NaN if none).
    """
    items = [v for v in anchors.values() if isinstance(v, dict) and "frame" in v and "xyz" in v]
    items.sort(key=lambda v: int(v["frame"]))

    def labelled(*words: str) -> dict | None:
        return next(
            (v for v in items if any(w in str(v.get("label", "")).lower() for w in words)),
            None,
        )

    if penalty:
        keeper = anchors.get("keeper_hands") or anchors.get("p_arrival") or labelled(*KEEPER_WORDS)
        xyz = np.array(keeper["xyz"], dtype=float) if keeper is not None else np.full(3, np.nan)
        return end_f, end_f, xyz

    wall = labelled("barreira", "barrier") or anchors.get("barrier_target")
    if wall is None and items:
        wall = items[len(items) // 2]
    barrier_f = int(wall["frame"]) if wall is not None else 9
    # the last anchor is the keeper only when it comes after the wall (a binocular run has
    # the wall alone: the wall is not the keeper)
    last = items[-1] if items else None
    keeper_after_wall = last is not None and (wall is None or int(last["frame"]) > barrier_f)
    keeper = anchors.get("keeper_hands") or (last if keeper_after_wall else None)
    keeper_f = int(keeper["frame"]) if keeper is not None else end_f
    keeper_xyz = np.array(keeper["xyz"], dtype=float) if keeper is not None else np.full(3, np.nan)
    return barrier_f, keeper_f, keeper_xyz


def _deviation_from_chord(xy: np.ndarray) -> dict:
    """Largest sideways distance of the path from the straight line kick -> entry (top view)."""
    chord = xy[-1] - xy[0]
    length = float(np.linalg.norm(chord))
    if length <= 0:
        return {"dev_max_m": 0.0, "dev_at_m": 0.0, "chord_m": 0.0}
    u = chord / length
    normal = np.array([-u[1], u[0]])
    along = (xy - xy[0]) @ u
    side = (xy - xy[0]) @ normal
    i = int(np.argmax(np.abs(side)))
    return {"dev_max_m": float(abs(side[i])), "dev_at_m": float(along[i]), "chord_m": length}


# --------------------------------------------------------------------------- #
# Figures
# --------------------------------------------------------------------------- #
def _style(ax) -> None:
    ax.grid(color="#d9dee7", lw=0.8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def make_figures(
    run: ReportRun, m: dict, out_dir: Path, lang: str = "pt", people: People | None = None
) -> dict[str, Path]:
    """The figures of the report as PNG files in ``out_dir``, labelled in ``lang``."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    def num(x, nd=1):
        return fmt_num(x, nd, lang)

    out_dir.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    kf, ef = m["kick_frame"], m["end_frame"]
    penalty = bool(m.get("penalty"))
    hands = [m.get("hands_start"), m.get("hands_end")]
    track = run.track[kf : ef + 1]
    speed = m["speed_kmh"][kf : ef + 1]
    t = (np.arange(kf, ef + 1) - kf) / run.fps

    # 1. Top view coloured by speed.
    fig, ax = plt.subplots(figsize=(8, 5))
    draw_goal_area(ax, track[-1, :2])
    sc = ax.scatter(track[:, 0], track[:, 1], c=speed, cmap="autumn_r", s=34, zorder=3)
    ax.plot(track[:, 0], track[:, 1], color=BALL, lw=1.6, zorder=2)
    draw_people(ax, people, kick_f=kf, barrier_f=m["barrier_f"], end_f=ef, frame=ef,
                lang=lang)  # fmt: skip
    draw_hands(ax, hands[0], hands[1], lang)
    ax.plot(*track[0, :2], "o", color="white", mec=BALL, ms=11, zorder=4)
    ax.annotate(
        tr(lang, "fig_start"),
        tuple(track[0, :2]),
        xytext=(8, -14),
        textcoords="offset points",
        color=INK,
        fontweight="bold",
    )
    ax.annotate(
        tr(lang, "fig_entry_pk" if penalty else "fig_entry"),
        tuple(track[-1, :2]),
        xytext=(8, -16) if penalty else (8, 6),  # a penalty ends beside the keeper's label
        textcoords="offset points",
        color=INK,
        fontweight="bold",
    )
    cb = fig.colorbar(sc, ax=ax, shrink=0.8)
    cb.set_label(tr(lang, "fig_top_cbar"))
    extent = np.vstack(
        [
            track[:, :2],
            people_extent(people, kf, ef),
            *[h[:2][None] for h in hands if h is not None],
        ]
    )
    ax.set_xlim(extent[:, 0].min() - 3, extent[:, 0].max() + 3)
    ax.set_ylim(extent[:, 1].min() - 3, extent[:, 1].max() + 3)
    ax.set_aspect("equal")
    ax.set_xlabel(tr(lang, "fig_top_x"))
    ax.set_ylabel(tr(lang, "fig_top_y"))
    _style(ax)
    ax.set_title(
        tr(lang, "fig_top_title", d=num(m["distance_m"]), t=num(m["flight_s"], 2)),
        color=INK,
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout()
    paths["vista_de_cima"] = out_dir / "fig_vista_de_cima.png"
    fig.savefig(paths["vista_de_cima"], dpi=150)
    plt.close(fig)

    # 2. Side profile: height against the horizontal distance, with the wall (a free kick) or the
    #    goalkeeper's hands (a penalty) and the crossbar.
    fig, ax = plt.subplots(figsize=(8, 4.2))
    r = m["radial"][kf : ef + 1]
    ax.plot(r, track[:, 2], color=BALL, lw=2.6, label=tr(lang, "fig_traj"))
    goal_r = float(r[-1])
    if penalty:
        if run.goal_x is not None:
            goal_r = float(abs(run.goal_x - run.track[kf, 0]))
        for h, filled, key, dy in (
            (hands[0], False, "fig_hands_start", 8),
            (hands[1], True, "fig_hands_end", -14),
        ):
            if h is None:
                continue
            hr = float(np.linalg.norm(h[:2] - run.track[kf, :2]))
            ax.plot(
                hr,
                h[2],
                "o",
                ms=9,
                mfc=KEEPER if filled else "white",
                mec=KEEPER,
                mew=2.0,
                zorder=5,
            )
            ax.annotate(tr(lang, key), (hr, h[2]), xytext=(8, dy), textcoords="offset points",
                        color=KEEPER, fontsize=9, fontweight="bold")  # fmt: skip
    else:
        ax.add_patch(plt.Rectangle((BARRIER_M - 0.25, 0), 0.5, 1.8, color=WALL, alpha=0.85))
        ax.annotate(
            tr(lang, "fig_wall_label"),
            (BARRIER_M + 0.4, 1.2),
            color=WALL,
            fontsize=9,
            fontweight="bold",
        )
    ax.plot([goal_r, goal_r], [0, CROSSBAR_M], color=INK, lw=3)
    ax.annotate(tr(lang, "fig_crossbar"), (goal_r - 4.5, CROSSBAR_M + 0.1), color=INK, fontsize=9)
    ax.axhline(0, color=MUTED, lw=1)
    ax.set_xlabel(tr(lang, "fig_profile_x"))
    ax.set_ylabel(tr(lang, "fig_profile_y"))
    ax.set_ylim(0, max(3.0, m["zmax_m"] + 0.4))
    _style(ax)
    ax.set_title(
        tr(lang, "fig_profile_title", z=num(m["zmax_m"], 2)),
        color=INK,
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout()
    paths["perfil"] = out_dir / "fig_perfil.png"
    fig.savefig(paths["perfil"], dpi=150)
    plt.close(fig)

    # 3. Timeline: ball speed, drag coefficient and Reynolds number.
    fig, (ax, ax2) = plt.subplots(
        2, 1, figsize=(8, 5.2), sharex=True, gridspec_kw={"height_ratios": [1.3, 1]}
    )
    ax.plot(t, speed, color=BALL, lw=2.6)
    if not penalty:
        ax.axvline(m["barrier_t"], color=WALL, ls=":", lw=1.4)
    ax.set_ylabel(tr(lang, "fig_time_speed"))
    _style(ax)
    ax2.plot(t, m["cd"][kf : ef + 1], color=INK, lw=2, label=tr(lang, "fig_time_cd"))
    ax2.set_ylabel("Cd")
    ax2.set_xlabel(tr(lang, "fig_time_x"))
    ax2b = ax2.twinx()
    ax2b.plot(t, m["re"][kf : ef + 1] / 1e5, color=WALL, lw=1.8, ls="--",
              label=tr(lang, "fig_time_re"))  # fmt: skip
    ax2b.set_ylabel(tr(lang, "fig_time_re"), color=WALL)
    _style(ax2)
    ax.set_title(tr(lang, "fig_time_title"), color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    paths["tempo"] = out_dir / "fig_tempo.png"
    fig.savefig(paths["tempo"], dpi=150)
    plt.close(fig)

    # 4. Sideways deviation from the straight line kick -> entry.
    fig, ax = plt.subplots(figsize=(8, 3.6))
    xy = track[:, :2] - track[0, :2]
    chord = track[-1, :2] - track[0, :2]
    length = float(np.linalg.norm(chord))
    if length > 0:
        u = chord / length
        normal = np.array([-u[1], u[0]])
        along, side = xy @ u, xy @ normal
        ax.plot(along, side, color=BALL, lw=2.8, label=tr(lang, "fig_traj"))
        ax.axhline(
            0,
            color=INK,
            lw=1.4,
            ls="--",
            label=tr(lang, "fig_curve_line_pk" if penalty else "fig_curve_line"),
        )
        ax.fill_between(along, 0, side, color=BALL, alpha=0.08)
    ax.set_xlabel(tr(lang, "fig_curve_x"))
    ax.set_ylabel(tr(lang, "fig_curve_y"))
    _style(ax)
    ax.legend(fontsize=8.5, loc="upper right")
    ax.set_title(
        tr(lang, "fig_curve_title", dev=num(m["dev_m"], 2)),
        color=INK,
        loc="left",
        fontweight="bold",
    )
    fig.tight_layout()
    paths["curva"] = out_dir / "fig_curva.png"
    fig.savefig(paths["curva"], dpi=150)
    plt.close(fig)

    # 5. Forces in three instants: gravity, drag and Magnus accelerations (m/s^2).
    middle = (
        (tr(lang, "fig_when_mid"), (kf + ef) // 2)
        if penalty
        else (tr(lang, "fig_when_wall"), m["barrier_f"])
    )
    picks = [
        (tr(lang, "fig_when_after"), kf + 1),
        middle,
        (tr(lang, "fig_when_end"), max(ef - 1, kf + 1)),
    ]
    picks = [(lab, f) for lab, f in picks if f < run.n and np.isfinite(m["a_drag"][f])]
    if picks:
        fig, ax = plt.subplots(figsize=(8, 3.8))
        width = 0.26
        names = [lab for lab, _ in picks]
        xs = np.arange(len(picks))
        ax.bar(xs - width, [GRAVITY] * len(picks), width, color=INK,
               label=tr(lang, "fig_force_grav"))  # fmt: skip
        ax.bar(xs, [m["a_drag"][f] for _, f in picks], width, color=BALL,
               label=tr(lang, "fig_force_drag"))  # fmt: skip
        ax.bar(xs + width, [m["a_magnus"][f] for _, f in picks], width, color=WALL,
               label=tr(lang, "fig_force_magnus"))  # fmt: skip
        ax.set_xticks(xs, names)
        ax.set_ylabel(tr(lang, "fig_force_y"))
        ax.legend(fontsize=8.5)
        _style(ax)
        ax.set_title(tr(lang, "fig_forces_title"), color=INK, loc="left", fontweight="bold")
        fig.tight_layout()
        paths["forcas"] = out_dir / "fig_forcas.png"
        fig.savefig(paths["forcas"], dpi=150)
        plt.close(fig)

    # 6. Reprojection error per frame (how well the fitted flight matches the marked pixels).
    fig, ax = plt.subplots(figsize=(8, 3.0))
    frames = sorted(m["per_frame_px"])
    ax.bar(frames, [m["per_frame_px"][f] for f in frames], color="#7b61b5")
    ax.axhline(m["rms_px"], color=INK, ls="--", lw=1.2,
               label=tr(lang, "fig_res_mean", rms=num(m["rms_px"])))  # fmt: skip
    ax.set_xlabel(tr(lang, "fig_res_x"))
    ax.set_ylabel(tr(lang, "fig_res_y"))
    ax.legend(fontsize=8.5)
    _style(ax)
    ax.set_title(tr(lang, "fig_res_title"), color=INK, loc="left", fontweight="bold")
    fig.tight_layout()
    paths["residuos"] = out_dir / "fig_residuos.png"
    fig.savefig(paths["residuos"], dpi=150)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------- #
# Editor data and HTML
# --------------------------------------------------------------------------- #
def _anchor_points(run: ReportRun, lang: str) -> list[dict]:
    """The anchor points of the fit with their sigma: from ``anchor_spec`` when the run has it,
    else from the anchors of the first version (sigma 0.3 m), labelled in ``lang``."""
    out: list[dict] = []
    spec = (run.flight.get("anchor_spec") or {}).get("points") or []
    if spec:
        for p in spec:
            out.append(
                {
                    "label": localize_anchor_label(lang, str(p.get("label", ""))),
                    "frame": int(p["frame"]),
                    "xyz": [float(v) for v in p["xyz"]],
                    "sigma": float(p["sigma"]),
                }
            )
        return out
    for key, v in run.anchors.items():
        if not isinstance(v, dict) or "xyz" not in v:
            continue
        label = (
            str(v["label"])
            if v.get("label")
            else tr(lang, LEGACY_ANCHOR_KEYS.get(key, "anchor_end"))
        )
        out.append(
            {
                "label": localize_anchor_label(lang, label),
                "frame": int(v["frame"]),
                "xyz": [float(c) for c in v["xyz"]],
                "sigma": DEFAULT_ANCHOR_SIGMA,
            }
        )
    return out


def editor_payload(run: ReportRun, m: dict, lang: str) -> dict:
    """What the in-browser editor needs: the pixel observations with their cameras, the anchor
    points (with their sigma), the fitted parameters and the texts of the editor.

    The editor refits one camera; the two-camera fit is not reproduced in the browser, and the
    automatic fit (its anchors come from the detection) is not editable yet: both switch the
    editor off.
    """
    key = _method_key(m)
    if key in ("binocular", "auto"):
        labels = editor_labels(lang)
        if key == "auto":
            labels["ed_disabled"] = tr(lang, "ed_disabled_auto")
        return {"disabled": True, "lang": lang, "labels": labels}
    points = _anchor_points(run, lang)
    frames = [
        {"frame": int(f), "u": float(uv[0]), "v": float(uv[1]),
         "P": [float(x) for x in run.projections[f].ravel()]}
        for f, uv in sorted(run.pixels.items())
        if f in run.projections
    ]  # fmt: skip
    return {
        "lang": lang,
        "fps": float(run.fps),
        "n": int(run.n),
        "rest_height": float(BALL_RADIUS),
        "sigma_px": float(PIXEL_SIGMA),
        "frames": frames,
        "points": points,
        "original_points": copy.deepcopy(points),
        "theta": [float(v) for v in run.flight["theta"]],
        "labels": editor_labels(lang),
        "kick_frame": int(m["kick_frame"]),
    }


def _json_for_html(obj: dict) -> str:
    """JSON that can sit inside a <script> element (no '</' that would close it)."""
    return json.dumps(obj, ensure_ascii=False, separators=(",", ":")).replace("</", "<\\/")


# --------------------------------------------------------------------------- #
# HTML and PDF
# --------------------------------------------------------------------------- #
def _b64(path: Path) -> str:
    return "data:image/png;base64," + base64.b64encode(path.read_bytes()).decode()


def _method_key(m: dict) -> str:
    """The suffix of the method texts: ``binocular`` for the two-camera fit, ``auto`` for the
    monocular fit with the automatic start and arrival, else ``anchors``."""
    method = m.get("method")
    if method == "binocular":
        return "binocular"
    return "auto" if method == "monocular_auto" else "anchors"


def _warn_block(m: dict, lang: str) -> str:
    if m["tier"] == "alta":
        return ""
    key = "tier_" + m["tier"]
    return (
        '<div class="warn">'
        + tr(
            lang,
            "warn",
            tier=tr(lang, key),
            tier_text=tr(
                lang,
                "penalty_tier_text_baixa"
                if m.get("penalty") and m["tier"] == "baixa"
                else "tier_text_" + m["tier"],
            ),
            worst=fmt_num(m["worst_anchor_m"], 1, lang),
            reference=tr(lang, "ref_" + _method_key(m)),
        )
        + "</div>"
    )


def _quotes(m: dict, lang: str) -> str:
    def num(x, nd=1):
        return fmt_num(x, nd, lang)

    sfx = "_pk" if m.get("penalty") else ""
    items = [
        tr(lang, "quote_distance", dist=num(m["distance_m"]), flight=num(m["flight_s"], 2)),
        tr(lang, "quote_mean" + sfx, mean=num(m["mean_kmh"], 0)),
        tr(lang, "quote_height" + sfx, zmax=num(m["zmax_m"], 1)),
        tr(lang, "quote_measured"),
    ]
    if m["tier"] != "baixa":
        items.append(tr(lang, "quote_curve", dev=num(m["dev_m"], 1)))
    return "\n".join(f"<li>{s}</li>" for s in items)


def _hands_texts(m: dict, lang: str) -> tuple[str, str]:
    """``(lead sentence, paragraph)`` about the goalkeeper's hands of a penalty; "" when unknown."""

    def num(x, nd=1):
        return fmt_num(x, nd, lang)

    gap, travel = m.get("keeper_gap_m"), m.get("hands_travel_m")
    moved = tr(lang, "hands_moved", travel=num(travel)) if travel is not None else ""
    if gap is None:
        tail = "" if travel is not None else tr(lang, "hands_missing")
        lead = ""
    else:
        tail = tr(lang, "hands_near" if gap <= HANDS_NEAR_M else "hands_far", gap=num(gap))
        lead = tr(lang, "hands_lead", gap=num(gap))
    return lead, " ".join(t for t in (moved, tail) if t)


def _select_blocks(html: str, penalty: bool) -> str:
    """Keep the ``<!--pk-->..<!--/pk-->`` blocks of a penalty or the ``<!--fk-->`` blocks of a free
    kick, and drop the others: the template carries the text of both set pieces."""
    keep, drop = ("pk", "fk") if penalty else ("fk", "pk")
    html = re.sub(rf"<!--{drop}-->.*?<!--/{drop}-->", "", html, flags=re.S)
    return html.replace(f"<!--{keep}-->", "").replace(f"<!--/{keep}-->", "")


def fill_template(m: dict, figs: dict[str, Path], meta: dict, lang: str, payload: dict) -> str:
    """The HTML: the template of ``lang`` with every placeholder replaced."""
    html = _select_blocks(TEMPLATES[lang].read_text(encoding="utf-8"), bool(m.get("penalty")))
    hands_lead, hands_text = _hands_texts(m, lang)

    def num(x, nd=1):
        return fmt_num(x, nd, lang)

    values = {
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "authors": meta["authors"],
        "dist": num(m["distance_m"]),
        "flight": num(m["flight_s"], 2),
        "mean": num(m["mean_kmh"], 0),
        "launch": num(m["launch_kmh"], 0),
        "launch_deg": num(m["launch_deg"], 0),
        "zmax": num(m["zmax_m"], 2),
        "zmax_t": num(m["zmax_t"], 2),
        "spin": num(m["spin_rps"], 1),
        "peak": num(m["peak_kmh"], 0),
        "end_kmh": num(m["end_kmh"], 0),
        "wall_t": num(m["barrier_t"], 2),
        "wall_kmh": num(m["barrier_kmh"], 0),
        "after_wall": num(m["after_wall_s"], 2),
        "wall_z": num(m["wall_z"], 2),
        "wall_clear": num(abs(m["wall_clear_cm"]), 0),
        "wall_side": tr(lang, "wall_above" if m["wall_clear_cm"] >= 0 else "wall_below"),
        "spin_axis": tr(lang, "spin_vertical" if m["spin_vertical"] else "spin_tilted"),
        "kicker_phrase": meta["kicker_phrase"],
        "re_min": num(m["re_min"] / 1e5, 1),
        "re_max": num(m["re_max"] / 1e5, 1),
        "wall_goal": num(m["barrier_to_goal_m"], 1),
        "keeper_gap": num(m["keeper_gap_m"], 1),
        "hands_lead": hands_lead,
        "hands_text": hands_text,
        "hands_limit": tr(lang, "hands_limit"),
        "dev": num(m["dev_m"], 2),
        "dev_at": num(m["dev_at_m"], 1),
        "rms": num(m["rms_px"], 1),
        "worst": num(m["worst_anchor_m"], 1),
        "tier": tr(lang, "tier_" + m["tier"]),
        "warn_block": _warn_block(m, lang),
        "method_text": tr(lang, "method_" + _method_key(m), rms=num(m["rms_px"], 1)),
        "refs_text": tr(lang, "refs_" + _method_key(m), worst=num(m["worst_anchor_m"], 1)),
        "wall_text": tr(lang, "wall_" + _method_key(m)),
        "quotes": _quotes(m, lang),
        "generated": meta["generated"],
        "players_note": meta["players_note"],
        "animation": meta["animation"],
        "editor_payload": _json_for_html(payload),
        "app_js": APP_JS.read_text(encoding="utf-8").replace("</script", "<\\/script"),
    }
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", str(value))
    for key, path in figs.items():
        html = html.replace("{{img_" + key + "}}", _b64(path))
    while "{{img_" in html:  # a figure that was not drawn (e.g. no forces) leaves no broken image
        start = html.index("{{img_")
        end = html.index("}}", start) + 2
        html = html[:start] + html[end:]
    return html


def find_chrome() -> str | None:
    for name in ("google-chrome", "chromium", "chromium-browser"):
        path = shutil.which(name)
        if path:
            return path
    return None


def print_pdf(html_path: Path, pdf_path: Path) -> Path | None:
    """HTML -> PDF with headless Chrome; None when no Chrome is installed."""
    chrome = find_chrome()
    if chrome is None:
        return None
    subprocess.run(
        [chrome, "--headless=new", "--disable-gpu", "--no-sandbox", "--no-pdf-header-footer",
         f"--print-to-pdf={pdf_path}", html_path.as_uri()],
        check=False, capture_output=True, timeout=300,
    )  # fmt: skip
    return pdf_path if pdf_path.is_file() else None


def _languages(lang: str) -> list[str]:
    if lang == "both":
        return list(LANGS)
    if lang not in LANGS:
        raise ValueError(f"lang must be one of {', '.join(LANGS)} or both, not {lang!r}")
    return [lang]


def _people_for(run: ReportRun, m: dict, players: dict | None) -> People:
    """The people of the top view: their ground tracks and the roles.

    The roles come from the run's player map when it exists (the goalkeeper and the kicker of
    the same clip, the same ids in every camera); otherwise they are the nearest persons. A role
    given by the user (``players``) replaces both.
    """
    _, _, keeper_xyz = event_frames(run.anchors, m["end_frame"], penalty=run.penalty)
    given = players or {}
    mapped = load_camera_roles(run.people_dir)
    penalty = run.penalty
    people = find_people(
        load_ground_tracks(run.people_dir, run.n),
        run.track[:, :2],
        m["kick_frame"],
        m["barrier_f"],
        m["end_frame"],
        keeper_xyz[:2],
        kicker_id=given.get("kicker_id", mapped.get("kicker_id")),
        wall_ids=given.get("wall_ids"),
        keeper_id=given.get("keeper_id", mapped.get("keeper_id")),
        penalty=penalty,
    )
    people.names = mapped.get("names", {})
    people.penalty = penalty
    people.manual = any(given.get(k) is not None for k in ("kicker_id", "wall_ids", "keeper_id"))
    people.source = "manual" if people.manual else ("map" if mapped else "nearest")
    return people


def build_report(
    run_dir,
    *,
    video=None,
    title: str = "",
    subtitle: str = "",
    authors: str = "",
    names: str = "",
    animation: bool = True,
    lang: str = "both",
    players: dict | None = None,
) -> dict:
    """Write ``<run>/report_tv/<lang>/`` for each language (HTML, PDF, figures, MP4).

    ``title`` and ``subtitle`` replace the default ones of every language; ``names`` is the
    kicker's name, used in the text of both languages. ``players`` may hold ``kicker_id``,
    ``wall_ids`` and ``keeper_id`` to replace the people found by the nearest point.
    """
    from datetime import date

    langs = _languages(lang)
    run = load_run(run_dir, video)
    given_keeper = (players or {}).get("keeper_id")
    if run.penalty and given_keeper is not None:
        run.keeper_hands = keeper_wrists(run.people_dir, int(given_keeper), run.n)
    m = compute_metrics(run)
    people = _people_for(run, m, players)
    root = Path(run.run_dir) / "report_tv"
    result: dict = {"dir": str(root), "tier": m["tier"], "languages": {}}
    today = date.today()
    sfx = "_pk" if run.penalty else ""
    for code in langs:
        out = root / code
        html_name, pdf_name, mp4_name = FILE_NAMES[code]
        figs = make_figures(run, m, out / "figuras", code, people)
        video_path = None
        if animation:
            video_path = write_lance_video(
                run, out / mp4_name, m, title=title or tr(code, "default_title" + sfx), lang=code,
                people=people,
            )  # fmt: skip
        meta = {
            "title": title or tr(code, "default_title" + sfx),
            "subtitle": subtitle or tr(code, "default_subtitle"),
            "authors": authors,
            "kicker_phrase": tr(code, "kicker_of", name=names)
            if names
            else tr(code, "kicker_default" + sfx),
            "generated": tr(code, "date_format", d=today.day, m=today.month, y=today.year),
            "animation": mp4_name if video_path else "",
            "players_note": people_note(people, code),
        }
        html = fill_template(m, figs, meta, code, editor_payload(run, m, code))
        html_path = out / html_name
        html_path.write_text(html, encoding="utf-8")
        pdf_path = print_pdf(html_path, out / pdf_name)
        result["languages"][code] = {
            "dir": str(out),
            "html": str(html_path),
            "pdf": str(pdf_path) if pdf_path else None,
            "animation": str(video_path) if video_path else None,
        }
    return result
