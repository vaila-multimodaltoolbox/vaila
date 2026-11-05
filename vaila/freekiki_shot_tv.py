"""
================================================================================
Script: freekiki_shot_tv.py - TV delivery package for a FreeKiki shot at goal
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.14
Created: 02 November 2026
Update Date: 02 November 2026

Description:
    Builds the TV folder of a shot at goal analysed by FreeKiki (``kick-analysis``), with the
    same deliverables as the pass and dribble packages: a 60 s explainer video with burnt-in
    captions, the shot at its original rate with the SAM 3D Body skeletons of the kicker and
    the keeper, a Portuguese report (HTML + PDF), captions / script / teleprompter lines,
    visual checks, data and checksums.

    The keeper's response is measured, not assumed:

    * :func:`keeper_response` — the keeper's nearest hand point to the ball per frame (SAM 3D
      Body finger tips and wrists, pitch metres), the hand reaction onset (operational: the
      reaching hand rises 0.30 m above its first-three-frame height; an early bound at
      0.15 m), the smallest hand–ball gap and when it happens, and the feet–ball distance;
    * :func:`goal_crossing` — where the reconstructed ball meets the goal line, against the
      posts, which shows the size of the single-camera sideways error when the outcome is
      known from the footage.

    CLI::

        python -m vaila.freekiki_shot_tv --run <kick 3D folder> --kicker 23 --keeper 14 \\
            --outcome goal --pass-package <pass TV folder> --out <delivery folder>
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import html
import json
import math
import shlex
import shutil
import sys
from datetime import date
from fractions import Fraction
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image, ImageDraw

try:
    from .freekiki_openplay_video import _srt_time, _text, encode_frames
    from .freekiki_pass import speed_headline_kmh
    from .freekiki_pass_tv import (
        _ball_pixels,
        _keypoints_file,
        _load_poses,
        _probe,
        _read_frames,
        _readme_video,
        _sha256,
        _video_rate,
    )
    from .freekiki_report import compute_metrics, load_run, print_pdf
    from .freekiki_report_i18n import fmt_num
except ImportError:
    from freekiki_openplay_video import (  # ty: ignore[unresolved-import]
        _srt_time,
        _text,
        encode_frames,
    )
    from freekiki_pass import speed_headline_kmh  # ty: ignore[unresolved-import]
    from freekiki_pass_tv import (  # ty: ignore[unresolved-import]
        _ball_pixels,
        _keypoints_file,
        _load_poses,
        _probe,
        _read_frames,
        _readme_video,
        _sha256,
        _video_rate,
    )
    from freekiki_report import (  # ty: ignore[unresolved-import]
        compute_metrics,
        load_run,
        print_pdf,
    )
    from freekiki_report_i18n import fmt_num  # ty: ignore[unresolved-import]

BALL_RADIUS_M = 0.11
POST_HALF_WIDTH_M = 3.66
CROSSBAR_M = 2.44
ONSET_RISE_M = 0.30  # the reaching hand is clearly raised
ONSET_EARLY_M = 0.15  # first visible rise (early bound)
W, H = 1920, 1080
MAIN = (1280, 720)
BG = "#0b1728"
KICKER_C = "#ffe25c"
KEEPER_C = "#5dd6ff"
BALL_C = "#ff5a5a"
YELLOW = "#ffe25c"
MUTED = "#8da9c2"
TV_SECONDS, TV_RATE = 60, 30
HAND_TIPS = ("thumb", "index", "middle", "ring", "pinky")

REFERENCES = [
    (
        "Spratford, W., Mellifont, R., & Burkett, B. (2009). The influence of dive direction on "
        "the movement characteristics for elite football goalkeepers. Sports Biomechanics, 8(3), "
        "235–244.",
        "https://doi.org/10.1080/14763140903229526",
    ),
    (
        "Woods, D. L., Wyma, J. M., Yund, E. W., Herron, T. J., & Reed, B. (2015). Factors "
        "influencing the latency of simple reaction time. Frontiers in Human Neuroscience, 9, 131.",
        "https://doi.org/10.3389/fnhum.2015.00131",
    ),
    (
        "Santiago / Jornal da USP (2022). O futebol de um contra um é fonte para análises "
        "científicas.",
        "https://jornal.usp.br/radio-usp/o-futebol-de-um-para-um-e-fonte-para-analises-cientificas/",
    ),
]


def _n(x, nd: int = 1) -> str:
    return "—" if x is None else fmt_num(float(x), nd, "pt")


def _gap_cm(gap_m: float) -> int:
    """Hand–ball gap for TV, rounded to 10 cm (the single-camera error is larger than that)."""
    return int(10 * round(float(gap_m) * 100 / 10))


def _f(x) -> float | None:
    if x is None:
        return None
    x = float(x)
    return x if math.isfinite(x) else None


# ------------------------------------------------------------------------------- measurements


def read_poses_3d(path: Path, person_id: int) -> dict[int, dict[str, np.ndarray]]:
    """Frame -> keypoint name -> xyz (pitch metres) of one person from ``poses_3d_on_field``."""
    out: dict[int, dict[str, np.ndarray]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("person_id") != str(person_id):
                continue
            try:
                p = np.array([float(r["x_m"]), float(r["y_m"]), float(r["z_m"])])
            except (KeyError, ValueError):
                continue
            if np.isfinite(p).all():
                out.setdefault(int(r["frame"]), {})[r["kpt_name"]] = p
    return out


def read_ball_xyz(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            f = int(r["frame"])
            while len(rows) <= f:
                rows.append([np.nan, np.nan, np.nan])
            with contextlib.suppress(KeyError, ValueError):
                rows[f] = [float(r["x_m"]), float(r["y_m"]), float(r["z_m"])]
    return np.asarray(rows, dtype=float).reshape(-1, 3)


def read_ground(path: Path, person_id: int) -> dict[int, tuple[float, float]]:
    out: dict[int, tuple[float, float]] = {}
    with path.open(newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("person_id") == str(person_id):
                try:
                    out[int(r["frame"])] = (float(r["x_m"]), float(r["y_m"]))
                except (KeyError, ValueError):
                    continue
    return out


def _hand_points(kp: dict[str, np.ndarray], side: str) -> list[np.ndarray]:
    names = [f"{side}-wrist"] + [f"{side}-{t}-tip" for t in HAND_TIPS]
    return [kp[n] for n in names if n in kp]


def keeper_response(
    poses: dict[int, dict[str, np.ndarray]],
    ball: np.ndarray,
    fps: float,
    *,
    rise_m: float = ONSET_RISE_M,
    early_m: float = ONSET_EARLY_M,
) -> dict:
    """The keeper's measured response to the shot (frame 0 = kick).

    The reaching hand is the side whose points get closest to the ball. Its height is the
    mean of its wrist and finger tips; the onset is the first frame at which that height is
    ``rise_m`` above its mean over frames 0–2 (``early`` uses ``early_m``). The gap is the 3D
    distance from the ball's centre to the nearest hand point minus the ball radius.
    """
    frames = sorted(f for f in poses if f < len(ball) and np.isfinite(ball[f]).all())
    if len(frames) < 4:
        raise ValueError("the keeper has fewer than four posed frames with the ball")
    best_side, best = "right", math.inf
    for side in ("right", "left"):
        for f in frames:
            for p in _hand_points(poses[f], side):
                d = float(np.linalg.norm(p - ball[f]))
                if d < best:
                    best, best_side = d, side
    rows: list[dict[str, Any]] = []
    for f in frames:
        kp = poses[f]
        pts = _hand_points(kp, "right") + _hand_points(kp, "left")
        reach = _hand_points(kp, best_side)
        gap = min(float(np.linalg.norm(p - ball[f])) for p in pts) - BALL_RADIUS_M if pts else None
        hand_z = float(np.mean([p[2] for p in reach])) if reach else None
        feet = [kp[n] for n in ("left-ankle", "right-ankle") if n in kp]
        feet_xy = np.mean(feet, axis=0)[:2] if feet else None
        hips = [kp[n] for n in ("left-hip", "right-hip") if n in kp]
        pelvis = np.mean(hips, axis=0) if hips else None
        rows.append(
            {
                "frame": f,
                "t_s": f / fps,
                "hand_gap_m": gap,
                "hand_z_m": hand_z,
                "feet_ball_horiz_m": None
                if feet_xy is None
                else float(np.linalg.norm(feet_xy - ball[f][:2])),
                "pelvis_xyz_m": None if pelvis is None else [float(v) for v in pelvis],
                "ball_z_m": float(ball[f][2]),
            }
        )
    z = np.array([np.nan if r["hand_z_m"] is None else r["hand_z_m"] for r in rows])
    base = float(np.nanmean(z[:3]))
    onset = next((r for r, v in zip(rows, z, strict=True) if v - base >= rise_m), None)
    early = next((r for r, v in zip(rows, z, strict=True) if v - base >= early_m), None)
    gaps = [r for r in rows if r["hand_gap_m"] is not None]
    closest = min(gaps, key=lambda r: r["hand_gap_m"])
    p0 = rows[0]["pelvis_xyz_m"]
    pc = closest["pelvis_xyz_m"]
    return {
        "reaching_hand": best_side,
        "hand_baseline_z_m": base,
        "onset_frame": None if onset is None else onset["frame"],
        "onset_s": None if onset is None else onset["t_s"],
        "onset_early_s": None if early is None else early["t_s"],
        "onset_resolution_s": 1.0 / fps,
        "closest_frame": closest["frame"],
        "closest_s": closest["t_s"],
        "closest_gap_m": closest["hand_gap_m"],
        "closest_hand_z_m": closest["hand_z_m"],
        "closest_ball_z_m": closest["ball_z_m"],
        "closest_feet_ball_horiz_m": closest["feet_ball_horiz_m"],
        "pelvis_displacement_at_closest_m": None
        if p0 is None or pc is None
        else float(np.linalg.norm(np.asarray(pc[:2]) - np.asarray(p0[:2]))),
        "motor_time_after_onset_s": None if onset is None else closest["t_s"] - onset["t_s"],
        "rule": (
            f"onset: reaching-hand height ≥ {rise_m:.2f} m above its frames 0–2 mean "
            f"(early bound {early_m:.2f} m); gap: ball centre to nearest hand point minus "
            f"{BALL_RADIUS_M:.2f} m"
        ),
        "series": rows,
    }


def goal_crossing(
    ball: np.ndarray, goal_x: float, *, half_width: float = POST_HALF_WIDTH_M
) -> dict:
    """Where the reconstructed ball meets the goal line (straight extrapolation of the last step)."""
    ok = np.flatnonzero(np.isfinite(ball).all(axis=1))
    if ok.size < 2:
        raise ValueError("fewer than two ball frames")
    a, b = ball[ok[-2]], ball[ok[-1]]
    step = b - a
    gap_x = goal_x - b[0]
    if abs(step[0]) < 1e-9 or np.sign(step[0]) != np.sign(gap_x):
        return {"reaches_line": False, "last_frame": int(ok[-1]), "last_gap_m": abs(float(gap_x))}
    k = gap_x / step[0]
    at_line = b + k * step
    y = float(at_line[1])
    outside = abs(y) - half_width
    return {
        "reaches_line": True,
        "last_frame": int(ok[-1]),
        "last_gap_m": abs(float(gap_x)),
        "y_at_line_m": y,
        "z_at_line_m": float(at_line[2]),
        "post_y_m": float(np.sign(y) * half_width),
        "outside_post_m": float(outside),
        "inside_goal": bool(outside <= 0 and at_line[2] <= CROSSBAR_M),
    }


def monotone_flight_speed(
    raw_mps, fps: float, kick_time_s: float | None, launch_mps: float
) -> list[float | None]:
    """Per-frame ball speed that only falls after the kick (drag), for display and figures.

    The fitted flight gives the launch speed at the kick instant; its per-frame values rise over
    the first frame step only because the kick falls inside it. Frames before the kick are
    ``None``; frames between the kick and the first full frame interpolate from the launch speed
    to that frame; later frames keep the fitted values, capped so they never rise.
    """
    raw = np.asarray(raw_mps, dtype=float)
    t = np.arange(len(raw)) / fps
    tk = max(0.0, float(kick_time_s or 0.0))
    ok = np.flatnonzero(np.isfinite(raw) & (raw > 0))
    if not ok.size:
        return [None] * len(raw)
    k = int(ok[np.argmax(raw[ok])])  # first full frame after the kick
    start = max(float(launch_mps), float(raw[k]))
    out: list[float | None] = []
    cap = start
    for i in range(len(raw)):
        if t[i] < tk:
            out.append(None)
            continue
        if i < k:
            u = (t[i] - tk) / max(t[k] - tk, 1e-9)
            v = start + u * (float(raw[k]) - start)
        elif np.isfinite(raw[i]) and raw[i] > 0:
            v = float(raw[i])
        else:
            out.append(None)
            continue
        cap = min(cap, v)
        out.append(cap)
    return out


# ------------------------------------------------------------------------------- metrics


def _pass_link(pass_package: Path | None) -> dict | None:
    if pass_package is None:
        return None
    path = Path(pass_package) / "data" / "pass_analysis.json"
    if not path.is_file():
        return None
    a = json.loads(path.read_text(encoding="utf-8"))
    shot = a.get("shot") or {}
    t = a.get("timing") or {}
    return {
        "package": str(Path(pass_package)),
        "passer_id": a.get("passer_id"),
        "receiver_id": a.get("receiver_id"),
        "passer_to_arrival_m": t.get("passer_to_arrival_m"),
        "flight_s": t.get("flight_s"),
        "arrival_to_shot_m": shot.get("arrival_to_shot_m"),
        "passer_to_shot_m": shot.get("passer_to_shot_m"),
        "gap_estimate_s": shot.get("gap_estimate_s"),
    }


def build_metrics(
    run: Path,
    *,
    kicker: int,
    keeper: int,
    outcome: str,
    outcome_note: str,
    pass_package: Path | None = None,
) -> dict:
    """All numbers of the package from the FreeKiki run (``metricas_materia.json`` v2)."""
    report_run = load_run(run)
    m = compute_metrics(report_run)
    fps = float(report_run.fps)
    ball = read_ball_xyz(run / "ball_3d_on_field.csv")
    setpiece = json.loads((run / "data" / "setpiece.json").read_text(encoding="utf-8"))
    goal_x = float(setpiece["goal_x_m"])
    flight = json.loads((run / "ball_flight.json").read_text(encoding="utf-8"))
    keeper_kp = read_poses_3d(run / "poses_3d_on_field.csv", keeper)
    response = keeper_response(keeper_kp, ball, fps)
    crossing = goal_crossing(ball, goal_x)
    g_keeper = read_ground(run / "ground_tracks.csv", keeper)
    g_kicker = read_ground(run / "ground_tracks.csv", kicker)
    by_frame = {r["frame"]: r for r in response["series"]}
    series = []
    for f in range(len(ball)):
        b = ball[f]
        kxy = g_keeper.get(f)
        r = by_frame.get(f, {})
        series.append(
            {
                "frame": f,
                "t_s": f / fps,
                "ball_xyz_m": [float(v) for v in b],
                "keeper_xy_m": list(kxy) if kxy else None,
                "kicker_xy_m": list(g_kicker[f]) if f in g_kicker else None,
                "ball_keeper_horiz_m": None
                if kxy is None
                else float(np.hypot(b[0] - kxy[0], b[1] - kxy[1])),
                "ball_goal_line_m": float(abs(goal_x - b[0])),
                "hand_gap_m": r.get("hand_gap_m"),
                "hand_z_m": r.get("hand_z_m"),
            }
        )
    raw_speed = [float(v) for v in m["speed_kmh"] / 3.6]
    launch_kmh = float(m["launch_kmh"])
    speed = monotone_flight_speed(raw_speed, fps, flight.get("kick_time_s"), launch_kmh / 3.6)
    head = speed_headline_kmh(launch_kmh, float(m["mean_kmh"]))
    horiz = [s["ball_keeper_horiz_m"] for s in series if s["ball_keeper_horiz_m"] is not None]
    feet_min = min(
        (s for s in series if s["ball_keeper_horiz_m"] is not None),
        key=lambda s: s["ball_keeper_horiz_m"],
    )
    gap_cm = _gap_cm(response["closest_gap_m"])
    verdict_head = "Faltou alcance, não reflexo"
    verdict = (
        f"Foi gol. O goleiro reagiu a tempo (~{_n(response['onset_s'], 2)} s depois do chute), "
        f"mas a mão ficou a ~{gap_cm} cm da bola, que passava a "
        f"~{_n(response['closest_ball_z_m'])} m de altura e a cerca de {head} km/h. "
        "Faltou alcance, não reflexo."
    )
    return {
        "schema": "lance-decifrado-materia-chute-v2",
        "run_dir": str(run),
        "roles": {"kicker_sam_id": kicker, "keeper_sam_id": keeper},
        "camera": {
            "kind": "mono",
            "fps": fps,
            "n_frames": int(len(ball)),
            "contact_frame": int(m["kick_frame"]),
            "rms_px": _f(flight.get("rms_px")),
            "quality": flight.get("quality", "ok"),
            "arrival_status": flight.get("arrival_status"),
            "flight_model": flight.get("flight_model"),
            "kick_time_s": _f(flight.get("kick_time_s")),
        },
        "kick_location": {
            "ball_xy_m": setpiece.get("ball_xy_m"),
            "goal_x_m": goal_x,
            "distance_to_goal_line_m": _f(setpiece.get("distance_to_goal_m")),
        },
        "speeds": {
            "launch_kmh": launch_kmh,
            "headline_kmh": head,
            "mean_path_kmh": float(m["mean_kmh"]),
            "end_kmh": float(m["end_kmh"]),
            "speed_mps_series": speed,
            "speed_fit_raw_mps": raw_speed,
            "definitions": {
                "launch_kmh": "speed at the kick from the fitted initial velocity (the largest: "
                "after the kick the ball only slows)",
                "headline_kmh": "launch speed rounded to 5 km/h (TV figure)",
                "mean_path_kmh": "path length / flight time, kick to last frame",
                "speed_mps_series": "fitted speed per frame, non-increasing from the kick "
                "(frames inside the kick step interpolated from the launch speed)",
                "speed_fit_raw_mps": "fitted speed per frame as written by FreeKiki",
            },
        },
        "flight": {
            "path_length_m": float(m["path_m"]),
            "duration_s": float(m["flight_s"]),
            "max_height_m": float(m["zmax_m"]),
            "spin_rps": float(m["spin_rps"]),
        },
        "keeper_window": {
            "ball_keeper_at_kick_m": horiz[0] if horiz else None,
            "feet_closest": {
                "frame": feet_min["frame"],
                "t_s": feet_min["t_s"],
                "distance_m": feet_min["ball_keeper_horiz_m"],
            },
        },
        "keeper_response": response,
        "goal_crossing": crossing,
        "reaction_time_context_s": {
            "simple_visual_rt_adults_typical": [0.2, 0.25],
            "note": "laboratory simple reaction time (light → button), order of magnitude; "
            "a save is a choice task with a whole-body movement",
        },
        "series": series,
        "outcome": {"result": outcome, "note": outcome_note},
        "link_pass": _pass_link(pass_package),
        "editorial_verdict": {
            "headline": verdict_head,
            "summary_pt": verdict,
            "defendable": "unlikely: reaction in time, reach short",
        },
    }


# ------------------------------------------------------------------------------- figures


def make_figures(metrics: dict, out: Path) -> dict[str, Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out.mkdir(parents=True, exist_ok=True)
    s = metrics["series"]
    t = np.array([r["t_s"] for r in s])
    resp = metrics["keeper_response"]
    figs: dict[str, Path] = {}

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    sp = metrics["speeds"]
    v = np.array([np.nan if x is None else x for x in sp["speed_mps_series"]]) * 3.6
    kick_t = max(0.0, float(metrics["camera"].get("kick_time_s") or 0.0))
    ok = np.isfinite(v)
    tt = np.r_[kick_t, t[: len(v)][ok]]
    vv = np.r_[sp["launch_kmh"], v[ok]]
    ax.plot(tt, vv, color="#d62246", lw=2, label="velocidade da bola (modelo de voo)")
    ax.scatter([kick_t], [sp["launch_kmh"]], color="#14213d", zorder=5, label="saída do pé")
    ax.axhline(
        sp["mean_path_kmh"],
        color="#5b6577",
        ls="--",
        lw=1,
        label=f"média no caminho: {_n(sp['mean_path_kmh'], 0)} km/h",
    )
    ax.legend(fontsize=8, loc="lower left")
    ax.set_xlabel("tempo desde o chute (s)")
    ax.set_ylabel("velocidade da bola (km/h)")
    ax.set_title(
        f"Saída do pé {_n(sp['launch_kmh'], 0)} km/h → último quadro {_n(sp['end_kmh'], 0)} km/h "
        f"(na TV: cerca de {sp['headline_kmh']} km/h)"
    )
    ax.grid(alpha=0.25)
    figs["speed_kmh"] = _save(fig, out / "speed_kmh.png")

    fig, ax = plt.subplots(figsize=(7.4, 3.8))
    gap = np.array([np.nan if r["hand_gap_m"] is None else r["hand_gap_m"] for r in s])
    feet = np.array(
        [np.nan if r["ball_keeper_horiz_m"] is None else r["ball_keeper_horiz_m"] for r in s]
    )
    ax.plot(t, feet, color="#5b6577", lw=1.6, label="bola → pés do goleiro (no chão)")
    ax.plot(t, gap, color="#0b7fa8", lw=2.2, label="bola → mão mais próxima (3D)")
    ax.scatter([resp["closest_s"]], [resp["closest_gap_m"]], color="#d62246", zorder=5)
    ax.annotate(
        f"{_n(resp['closest_gap_m'], 2)} m aos {_n(resp['closest_s'], 2)} s",
        (resp["closest_s"], resp["closest_gap_m"]),
        xytext=(-120, -4),
        textcoords="offset points",
        fontsize=9,
        color="#d62246",
        arrowprops={"arrowstyle": "->", "color": "#d62246"},
    )
    ax.set_xlabel("tempo desde o chute (s)")
    ax.set_ylabel("distância (m)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25)
    figs["dist"] = _save(fig, out / "dist.png")

    fig, ax = plt.subplots(figsize=(7.4, 3.6))
    t_line = metrics["flight"]["duration_s"]
    ax.axvspan(
        0.2, 0.25, color="#bbbbbb", alpha=0.35, label="tempo de reação simples (laboratório)"
    )
    if resp["onset_early_s"] is not None and resp["onset_s"] is not None:
        ax.axvspan(
            resp["onset_early_s"],
            resp["onset_s"],
            color="#5dd6ff",
            alpha=0.5,
            label="goleiro levanta a mão (medido)",
        )
    ax.axvline(resp["closest_s"], color="#d62246", lw=2, label="bola passa mais perto da mão")
    ax.axvline(t_line, color="#14213d", lw=1.2, ls="--", label="último quadro (bola junto à linha)")
    ax.axvline(0, color="#14934b", lw=2, label="chute")
    ax.set_xlim(-0.02, t_line + 0.05)
    ax.set_yticks([])
    ax.set_xlabel("tempo desde o chute (s)")
    ax.legend(fontsize=7.5, loc="upper center", bbox_to_anchor=(0.5, -0.28), ncol=2, frameon=False)
    figs["reacao"] = _save(fig, out / "reacao.png")
    return figs


def _save(fig, path: Path) -> Path:
    import matplotlib.pyplot as plt

    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    return path


# ------------------------------------------------------------------------------- video


def build_scenes(m: dict) -> list[tuple[int, int, str, str, str, str]]:
    r = m["keeper_response"]
    sp = m["speeds"]
    link = m.get("link_pass")
    who = (
        f"O mesmo jogador que recebeu o passe (id {link['receiver_id']}) finaliza."
        if link and link.get("receiver_id") is not None
        else "Chute em jogo."
    )
    gap_cm = _gap_cm(r["closest_gap_m"])
    return [
        (
            0,
            6,
            "O LANCE",
            "Foi gol · dez metros",
            f"Chute a {_n(m['kick_location']['distance_to_goal_line_m'])} m da linha. {who}",
            "natural",
        ),
        (
            6,
            16,
            "REPLAY",
            "Corpo e bola",
            "Esqueletos estimados pelo SAM 3D Body: chutador e goleiro. A bola leva o rastro "
            "vermelho.",
            "replay",
        ),
        (
            16,
            28,
            "VELOCIDADE",
            f"Cerca de {sp['headline_kmh']} km/h",
            f"Sai do pé a cerca de {sp['headline_kmh']} km/h e só perde velocidade: "
            f"{_n(m['flight']['path_length_m'])} m em {_n(m['flight']['duration_s'], 2)} s, "
            f"média de {_n(sp['mean_path_kmh'], 0)} km/h.",
            "replay",
        ),
        (
            28,
            40,
            "ESPAÇO",
            "Do pé ao goleiro",
            f"No chute, {_n(m['keeper_window']['ball_keeper_at_kick_m'])} m até o goleiro. Em "
            f"{_n(r['closest_s'], 2)} s a bola passa por ele, a {_n(r['closest_ball_z_m'])} m de "
            "altura.",
            "replay",
        ),
        (
            40,
            52,
            "REAÇÃO",
            "Ele reagiu a tempo",
            f"O goleiro levanta a mão cerca de {_n(r['onset_s'], 2)} s depois do chute — tempo de "
            "reação de atleta.",
            "reaction",
        ),
        (
            52,
            58,
            "VEREDITO",
            "Faltou alcance, não reflexo",
            f"Foi gol. A mão ficou a cerca de {gap_cm} cm da bola, alta e a cerca de "
            f"{sp['headline_kmh']} km/h.",
            "hold",
        ),
        (
            58,
            60,
            "FECHO",
            "GOL",
            "Medidas FreeKiki / vailá, uma câmera. Resultado confirmado no vídeo.",
            "end",
        ),
    ]


class ShotTV:
    """Pictures of the package: the shot camera with the kicker and keeper, map and numbers."""

    def __init__(self, m: dict, run: Path) -> None:
        self.m = m
        self.run = run
        self.kicker = int(m["roles"]["kicker_sam_id"])
        self.keeper = int(m["roles"]["keeper_sam_id"])
        self.video = _readme_video(run)
        self.rate = _video_rate(self.video)
        self.frames, self.dims = _read_frames(self.video, MAIN)
        self.kp_file = _keypoints_file(run, self.video.stem)
        self.poses = _load_poses(self.kp_file, {self.kicker, self.keeper})
        self.ball_px = _ball_pixels(run / "ball.csv")
        self.s = m["series"]
        self.n = min(len(self.frames), len(self.s))
        try:
            from .sam3dinov3 import SKELETON_EDGE_NAMES
        except ImportError:
            from sam3dinov3 import SKELETON_EDGE_NAMES  # ty: ignore[unresolved-import]
        self.edges = SKELETON_EDGE_NAMES
        pts = [r["ball_xyz_m"][:2] for r in self.s]
        pts += [r["keeper_xy_m"] for r in self.s if r["keeper_xy_m"]]
        pts += [r["kicker_xy_m"] for r in self.s if r["kicker_xy_m"]]
        arr = np.asarray(pts, dtype=float)
        arr = arr[np.isfinite(arr).all(axis=1)]
        self.lo, self.hi = arr.min(axis=0) - 2.0, arr.max(axis=0) + 2.0
        self.goal_x = float(m["kick_location"]["goal_x_m"])
        self.lo[0] = min(self.lo[0], self.goal_x - 1.0)
        self.hi[0] = max(self.hi[0], self.goal_x + 1.0)

    def view(self, f: int) -> Image.Image:
        f = max(0, min(f, self.n - 1))
        image = cv2.resize(self.frames[f], MAIN)
        sx, sy = MAIN[0] / self.dims[0], MAIN[1] / self.dims[1]
        trail = [self.ball_px[i] for i in range(max(0, f - 12), f + 1) if i in self.ball_px]
        for a, b in zip(trail[:-1], trail[1:], strict=False):
            cv2.line(
                image,
                tuple(np.rint(a * [sx, sy]).astype(int)),
                tuple(np.rint(b * [sx, sy]).astype(int)),
                (80, 80, 255),
                3,
                cv2.LINE_AA,
            )
        if f in self.ball_px:
            c = tuple(np.rint(self.ball_px[f] * [sx, sy]).astype(int))
            cv2.circle(image, c, 10, (40, 40, 255), 2, cv2.LINE_AA)
        for pid, color, name in (
            (self.kicker, KICKER_C, "Chutador"),
            (self.keeper, KEEPER_C, "Goleiro"),
        ):
            kp = self.poses.get((f, pid))
            if not kp:
                continue
            bgr = tuple(int(color[i : i + 2], 16) for i in (5, 3, 1))
            for a, b in self.edges:
                if a in kp and b in kp:
                    cv2.line(
                        image,
                        tuple(np.rint(kp[a] * [sx, sy]).astype(int)),
                        tuple(np.rint(kp[b] * [sx, sy]).astype(int)),
                        bgr,
                        2,
                        cv2.LINE_AA,
                    )
            anchor = kp.get("nose", next(iter(kp.values()))) * [sx, sy]
            x, y = np.rint(anchor + [0, -18]).astype(int)
            x, y = int(max(0, min(1100, x))), int(max(22, min(700, y)))
            cv2.putText(
                image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (10, 20, 35), 4, cv2.LINE_AA
            )
            cv2.putText(image, name, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, bgr, 2, cv2.LINE_AA)
        return Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))

    def map(self, size: tuple[int, int], f: int) -> Image.Image:
        image = Image.new("RGB", size, "#142e35")
        draw = ImageDraw.Draw(image)
        w, h = size
        span = self.hi - self.lo
        scale = min((w - 40) / span[0], (h - 90) / span[1])
        ox = (w - span[0] * scale) / 2
        oy = 40 + (h - 90 - span[1] * scale) / 2

        def px(p):
            return (ox + (p[0] - self.lo[0]) * scale, oy + (self.hi[1] - p[1]) * scale)

        s = 1.0 if self.goal_x > 0 else -1.0
        line = "#3d6a70"
        draw.line([px([self.goal_x, -12]), px([self.goal_x, 12])], fill=line, width=2)
        for depth, half in ((16.5, 20.16), (5.5, 9.16)):
            x2 = self.goal_x - s * depth
            draw.line(
                [
                    px([self.goal_x, -half]),
                    px([x2, -half]),
                    px([x2, half]),
                    px([self.goal_x, half]),
                ],
                fill=line,
                width=2,
            )
        for y in (-POST_HALF_WIDTH_M, POST_HALF_WIDTH_M):
            x, yy = px([self.goal_x, y])
            draw.ellipse((x - 4, yy - 4, x + 4, yy + 4), fill="#f0f5fb")
        _text(draw, "ZONA DO GOL · VISTA DE CIMA", (18, 12), 16, "#dfebef", bold=True)
        path = [r["ball_xyz_m"][:2] for r in self.s[: f + 1]]
        for a, b in zip(path[:-1], path[1:], strict=False):
            draw.line([px(a), px(b)], fill=BALL_C, width=3)
        if path:
            x, y = px(path[-1])
            draw.ellipse((x - 7, y - 7, x + 7, y + 7), fill=BALL_C)
        for key, color, name in (
            ("keeper_xy_m", KEEPER_C, "Goleiro"),
            ("kicker_xy_m", KICKER_C, "Chutador"),
        ):
            pts = [r[key] for r in self.s[: f + 1] if r[key]]
            for a, b in zip(pts[:-1], pts[1:], strict=False):
                draw.line([px(a), px(b)], fill=color, width=2)
            if pts:
                x, y = px(pts[-1])
                draw.ellipse((x - 6, y - 6, x + 6, y + 6), fill=color)
                _text(draw, name, (x + 10, y - 12), 14, color, bold=True)
        bar = 2.0 * scale
        draw.line([(20, h - 30), (20 + bar, h - 30)], fill="#dfebef", width=3)
        _text(draw, "2 m · plano do campo", (20, h - 24), 13, "#dfebef")
        return image

    def reaction_strip(self, size: tuple[int, int], f: int) -> Image.Image:
        image = Image.new("RGB", size, "#101c2c")
        draw = ImageDraw.Draw(image)
        w, h = size
        r = self.m["keeper_response"]
        tmax = max(0.62, float(self.m["flight"]["duration_s"]) + 0.02)

        def x(t):
            return 10 + t / tmax * (w - 20)

        y0, y1 = 30, h - 20
        draw.rectangle((x(0.20), y0, x(0.25), y1), fill="#3a4658")
        if r["onset_early_s"] is not None and r["onset_s"] is not None:
            draw.rectangle((x(r["onset_early_s"]), y0, x(r["onset_s"]), y1), fill="#1f6f8f")
        draw.line([(x(r["closest_s"]), y0 - 4), (x(r["closest_s"]), y1 + 4)], fill=BALL_C, width=3)
        now = self.s[f]["t_s"]
        draw.line([(x(now), y0 - 4), (x(now), y1 + 4)], fill=YELLOW, width=3)
        _text(draw, "REAÇÃO DO GOLEIRO (s)", (12, 6), 13, MUTED, bold=True)
        _text(draw, f"agora {_n(now, 2)} s", (w - 130, 6), 13, YELLOW, bold=True)
        return image

    def render(self, f: int, scene=None, elapsed: float = 0.0) -> np.ndarray:
        f = max(0, min(f, self.n - 1))
        canvas = Image.new("RGB", (W, H), BG)
        draw = ImageDraw.Draw(canvas)
        chapter = scene[2] if scene else "SAM 3D BODY · VELOCIDADE NATURAL"
        title = scene[3] if scene else "O chute e a reação do goleiro"
        _text(draw, "LANCE DECIFRADO  /  FREEKIKI + VAILÁ", (55, 26), 20, MUTED, bold=True)
        _text(draw, title, (55, 64), 40, bold=True)
        _text(draw, chapter, (1375, 40), 20, YELLOW, bold=True, width=490)
        _text(draw, f"Desde o chute: {_n(self.s[f]['t_s'], 2)} s", (1375, 100), 22)
        canvas.paste(self.view(f), (55, 160))
        canvas.paste(self.map((500, 320), f), (1370, 160))
        canvas.paste(self.reaction_strip((500, 90), f), (1370, 500))
        draw = ImageDraw.Draw(canvas)
        v = self.m["speeds"]["speed_mps_series"]
        if f < len(v) and v[f] is not None:
            _text(draw, f"Bola: {v[f] * 3.6:.0f} km/h", (1370, 610), 24, YELLOW, bold=True)
        r = self.s[f]
        gap = r.get("hand_gap_m")
        _text(draw, f"Mão do goleiro → bola: {_n(gap, 2)} m", (1370, 650), 21)
        _text(draw, f"Altura da bola: {_n(r['ball_xyz_m'][2], 2)} m", (1370, 684), 19)
        draw.rectangle((55, 900, 1870, 1018), fill="#152b43")
        caption = (
            scene[4]
            if scene
            else ("SAM 3D Body: poses estimadas no vídeo original. Uma câmera; bola marcada à mão.")
        )
        _text(draw, caption, (78, 920), 27, width=1750)
        for i, (label, color) in enumerate(
            (("Chutador", KICKER_C), ("Goleiro", KEEPER_C), ("Bola", BALL_C))
        ):
            _text(draw, label, (60 + i * 220, 1034), 18, color, bold=True)
        label = "30 fps · SEM NARRAÇÃO" if scene else f"{float(self.rate):.3f} fps · ORIGINAL"
        _text(draw, label, (1440, 1034), 18, MUTED)
        progress = elapsed / TV_SECONDS if scene else (f + 1) / self.n
        draw.rectangle((0, H - 6, int(W * progress), H - 1), fill=YELLOW)
        return np.asarray(canvas)


def write_videos(tv: ShotTV, out: Path, scenes: list) -> list[str]:
    encode_frames(out / "lance_kick_ui.mp4", tv.rate, (tv.render(f) for f in range(tv.n)))
    r = tv.m["keeper_response"]
    closest = int(r["closest_frame"])
    onset = int(r["onset_frame"]) if r["onset_frame"] is not None else closest

    def frames():
        for k in range(TV_SECONDS * TV_RATE):
            sec = k / TV_RATE
            scene = next(s for s in scenes if s[0] <= sec < s[1])
            start, end, mode = scene[0], scene[1], scene[5]
            u = (sec - start) / (end - start)
            if mode == "natural":
                f = min(tv.n - 1, int((sec - start) * float(tv.rate)))
            elif mode == "reaction":  # slow over the keeper's response
                lo = max(0, onset - 3)
                f = lo + min(int(u * (closest + 2 - lo)), closest + 1 - lo)
            elif mode == "hold":
                f = closest
            elif mode == "end":
                f = tv.n - 1
            else:
                f = min(tv.n - 1, int(u * tv.n))
            yield tv.render(f, scene, sec)

    encode_frames(out / "explicacao_tv.mp4", Fraction(TV_RATE), frames())
    return ["explicacao_tv.mp4", "lance_kick_ui.mp4"]


def write_script(out: Path, scenes: list, m: dict) -> None:
    rows, srt = [], []
    for i, (a, b, chapter, title, caption, _mode) in enumerate(scenes, 1):
        rows.append(
            {"start_s": a, "end_s": b, "chapter": chapter, "title": title, "caption": caption}
        )
        srt.append(f"{i}\n{_srt_time(a)} --> {_srt_time(b)}\n{title}\n{caption}\n")
    with (out / "roteiro_legendas.csv").open("w", newline="", encoding="utf-8") as fh:
        wr = csv.DictWriter(fh, fieldnames=list(rows[0]))
        wr.writeheader()
        wr.writerows(rows)
    (out / "explicacao_tv.srt").write_text("\n".join(srt), encoding="utf-8")
    lines = teleprompter(m)
    (out / "roteiro_tv.md").write_text(
        "# Roteiro de legendas — sem narração\n\n"
        + "\n\n".join(
            f"**{r['start_s']}–{r['end_s']} s · {r['chapter']}**\n\n{r['title']}\n\n{r['caption']}"
            for r in rows
        )
        + "\n\n## Frases para teleprompter\n\n"
        + "\n".join(f"- {line}" for line in lines)
        + "\n",
        encoding="utf-8",
    )


def teleprompter(m: dict) -> list[str]:
    r = m["keeper_response"]
    sp = m["speeds"]
    gap_cm = _gap_cm(r["closest_gap_m"])
    return [
        f"Foi gol. Dez metros da linha, cerca de {sp['headline_kmh']} quilômetros por hora.",
        f"O goleiro reagiu: levanta a mão um quarto de segundo depois do chute "
        f"(~{_n(r['onset_s'], 2)} s).",
        f"Mas a bola passa a cerca de {gap_cm} centímetros da mão, quase a dois metros de altura.",
        "Faltou alcance, não reflexo.",
    ]


# ------------------------------------------------------------------------------- report


def _card(value: str, unit: str, label: str, tag: str, cls: str = "") -> str:
    t = (
        "<span class='tag solid'>medido</span>"
        if tag == "m"
        else "<span class='tag est'>estimativa</span>"
    )
    u = f" <small>{unit}</small>" if unit else ""
    return f"<div class='card {cls}'><div class='n'>{value}{u}</div><div class='l'>{label} {t}</div></div>"


def build_html(m: dict, out: Path, videos: list[str], extra: list[str]) -> Path:
    r = m["keeper_response"]
    sp = m["speeds"]
    fl = m["flight"]
    gc = m["goal_crossing"]
    kw = m["keeper_window"]
    link = m.get("link_pass")
    gap_cm = _gap_cm(r["closest_gap_m"])
    rows = []
    s = m["series"]
    marks = [("Chute", 0)]
    if r["onset_frame"] is not None:
        marks.append(("Goleiro levanta a mão", int(r["onset_frame"])))
    marks += [("Bola mais perto da mão", int(r["closest_frame"])), ("Último quadro", len(s) - 1)]
    for name, f in marks:
        x = s[f]
        rows.append(
            f"<tr><td>{name}</td><td>{f}</td><td>{_n(x['t_s'], 2)}</td>"
            f"<td>{_n(x['hand_gap_m'], 2)}</td><td>{_n(x['ball_keeper_horiz_m'], 2)}</td>"
            f"<td>{_n(x['ball_xyz_m'][2], 2)}</td><td>{_n(x['hand_z_m'], 2)}</td>"
            f"<td>{_n(x['ball_goal_line_m'], 2)}</td></tr>"
        )
    vids = "".join(
        f"<figure><video controls preload='metadata' src='{v}'></video>"
        f"<figcaption><a href='{v}'>{v}</a></figcaption></figure>"
        for v in videos + extra
    )
    link_html = ""
    if link:
        gap = link.get("gap_estimate_s") or []
        link_html = (
            "<h2>2. Do passe ao chute</h2><p>O chute é do jogador que recebeu o passe do id "
            f"{link['passer_id']} (id {link['receiver_id']} no vídeo do passe; id "
            f"{m['roles']['kicker_sam_id']} neste vídeo). O passe percorreu "
            f"{_n(link['passer_to_arrival_m'])} m em {_n(link['flight_s'], 2)} s; do 1º toque ao "
            f"ponto do chute são {_n(link['arrival_to_shot_m'])} m, e do passador ao ponto do "
            f"chute {_n(link['passer_to_shot_m'])} m. O chute foi filmado por outra câmera: o "
            "tempo entre a recepção e o chute não é medido"
            + (f" (estimativa: {_n(gap[0])} a {_n(gap[-1])} s)" if gap else "")
            + ". Pacote do passe: <code>passe/enviar/resultados_passe_tv_20261102</code>.</p>"
        )
    side = "fora" if gc.get("outside_post_m", 0) > 0 else "dentro"
    crossing = (
        f"A reconstrução termina a {_n(gc['last_gap_m'], 2)} m da linha e, prolongada até a "
        f"linha, passa a {_n(abs(gc.get('outside_post_m', 0)), 2)} m {side} da trave. O vídeo "
        "mostra gol junto à trave: essa diferença é o erro lateral de uma câmera só neste "
        "lance, e vale também para a distância da mão."
        if gc.get("reaches_line")
        else f"A reconstrução termina a {_n(gc['last_gap_m'], 2)} m da linha."
    )
    refs = "".join(f"<li><a href='{u}'>{html.escape(t)}</a></li>" for t, u in REFERENCES)
    checks = "".join(
        f"<figure><img src='{p.name}'><figcaption>{cap}</figcaption></figure>"
        for p, cap in (
            (out / "conferencia_inicio.jpg", "t = 0 — o chute."),
            (out / "conferencia_reacao.jpg", "O goleiro começa a levantar a mão."),
            (out / "conferencia_passagem.jpg", "A bola passa mais perto da mão do goleiro."),
            (out / "conferencia_fim.jpg", "Último quadro analisado."),
        )
        if p.is_file()
    )
    tele = "".join(f"<li>{html.escape(t)}</li>" for t in teleprompter(m))
    markup = f"""<!doctype html><html lang="pt-BR"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Foi gol: faltou alcance, não reflexo</title>
<style>
:root{{--ink:#14213d;--muted:#5b6577;--line:#d9dee7;--soft:#f3f5f9;--ball:#d62246;--wall:#14934b}}
@page{{size:A4;margin:13mm 14mm}}*{{box-sizing:border-box}}
html{{-webkit-print-color-adjust:exact;print-color-adjust:exact}}
body{{margin:0;color:var(--ink);background:#fff;font-family:Inter,"Segoe UI",Roboto,system-ui,sans-serif;font-size:15px;line-height:1.55}}
main{{max-width:900px;margin:0 auto;padding:32px 16px 48px}}
.kicker{{text-transform:uppercase;letter-spacing:.08em;font-size:12px;color:var(--ball);font-weight:700}}
h1{{font-size:32px;line-height:1.1;margin:6px 0 8px}}h2{{font-size:21px;margin:30px 0 8px;break-after:avoid}}
.banner{{background:#14213d;color:#fff;border-left:8px solid var(--ball);padding:14px 18px;border-radius:6px;margin:14px 0}}
.banner b{{color:#ffe25c}}.lead{{font-size:17px;background:var(--soft);border-left:5px solid var(--ball);padding:14px 16px;border-radius:4px}}
.cards{{display:grid;grid-template-columns:repeat(3,1fr);gap:10px;margin:14px 0 6px}}
.card{{border:1px solid var(--line);border-radius:8px;padding:12px 14px;break-inside:avoid}}
.card .n{{font-size:26px;font-weight:800;line-height:1.1}}.card .n small{{font-size:15px;color:var(--muted)}}
.card .l{{font-size:13px;color:var(--muted)}}.card.hi{{border-color:var(--ball);background:#fff5f7}}
.tag{{display:inline-block;font-size:11px;font-weight:700;padding:1px 7px;border-radius:10px}}
.tag.solid{{background:#e3f4ea;color:#0d6b36}}.tag.est{{background:#fdeccf;color:#8a5300}}
.verdict{{border:2px solid var(--ball);border-radius:8px;padding:14px 18px;background:#fff5f7;break-inside:avoid}}
figure{{margin:12px 0;break-inside:avoid}}figure img,video{{width:100%;height:auto;border:1px solid var(--line);border-radius:6px}}
figcaption{{font-size:12.5px;color:var(--muted)}}table{{width:100%;border-collapse:collapse;font-size:13px}}
td,th{{border-bottom:1px solid var(--line);padding:6px;text-align:left}}.note{{border-left:3px solid var(--wall);padding:4px 12px;color:var(--muted);font-size:14px}}
ol.refs{{font-size:13px;color:var(--muted)}}footer{{margin-top:30px;border-top:1px solid var(--line);font-size:12px;color:var(--muted);padding-top:10px}}
@media(max-width:640px){{.cards{{grid-template-columns:repeat(2,1fr)}}h1{{font-size:26px}}}}
@media print{{video{{display:none}}figure img{{max-height:85mm;width:auto;max-width:100%}}main{{padding:0}}}}
</style></head><body><main>
<div class="kicker">Lance decifrado · FreeKiki / <i>vailá</i> · {date.today().strftime("%d/%m/%Y")}</div>
<h1>Foi gol: faltou alcance, não reflexo</h1>
<div class="banner"><b>RESULTADO · GOL</b> — {html.escape(m["outcome"]["note"])}</div>
<p class="lead">{html.escape(m["editorial_verdict"]["summary_pt"])}</p>
<div class="cards">
{_card(_n(m["kick_location"]["distance_to_goal_line_m"]), "m", "do chute até a linha do gol", "m", "hi")}
{_card("~" + str(sp["headline_kmh"]), "km/h", "velocidade na saída do pé (arredondada)", "e")}
{_card(_n(r["closest_s"], 2), "s", "do chute até a bola passar pelo goleiro", "m")}
</div><div class="cards">
{_card("~" + _n(r["onset_s"], 2), "s", "o goleiro levanta a mão (reação)", "e", "hi")}
{_card(f"~{gap_cm}", "cm", "menor distância mão → bola", "e")}
{_card(_n(r["closest_ball_z_m"]), "m", "altura da bola nessa passagem", "e")}
</div>
<h2>1. Os vídeos</h2>{vids}
<p>Legendas já desenhadas no vídeo de 60 s; também em <a href="explicacao_tv.srt">SRT</a>,
<a href="roteiro_tv.md">roteiro</a> e <a href="roteiro_legendas.csv">CSV</a>. Sem voz.
A câmera lenta repete quadros observados; nenhum quadro é interpolado.</p>
{link_html}
<h2>3. Onde foi o chute</h2>
<p>Chute em jogo, com a bola já em voo no primeiro quadro, a {_n(m["kick_location"]["distance_to_goal_line_m"])} m da linha.
A bola percorreu {_n(fl["path_length_m"])} m em {_n(fl["duration_s"], 2)} s e subiu até {_n(fl["max_height_m"])} m (o travessão está a 2,44 m).</p>
<figure><img src="figuras/trajetoria.png"><figcaption>Trajetória ajustada vista de cima.</figcaption></figure>
<figure><img src="figuras/perfil.png"><figcaption>Altura da bola ao longo do caminho.</figcaption></figure>
<h2>4. Velocidade</h2>
<p>Na TV: <b>cerca de {sp["headline_kmh"]} km/h na saída do pé</b>. Depois do chute a bola só perde velocidade (o ar a freia):
saiu a {_n(sp["launch_kmh"], 0)} km/h pelo modelo de voo, fez o caminho a {_n(sp["mean_path_kmh"], 0)} km/h em média e,
no último quadro, ainda andava a {_n(sp["end_kmh"], 0)} km/h. Com uma câmera, a velocidade é estimativa.</p>
<figure><img src="figuras/speed_kmh.png"><figcaption>Velocidade da bola desde a saída do pé (modelo de voo): só diminui.</figcaption></figure>
<figure><img src="figuras/forcas.png"><figcaption>Forças do modelo de voo: gravidade, arrasto e Magnus.</figcaption></figure>
<h2>5. O goleiro</h2>
<p>No chute, a bola estava a {_n(kw["ball_keeper_at_kick_m"])} m do goleiro (no chão). O goleiro levanta a mão
{"direita" if r["reaching_hand"] == "right" else "esquerda"}: ela começa a subir aos ~{_n(r["onset_early_s"], 2)} s e já está claramente
erguida aos ~{_n(r["onset_s"], 2)} s. Aos {_n(r["closest_s"], 2)} s a bola passa a <b>~{gap_cm} cm da mão</b>, a {_n(r["closest_ball_z_m"])} m de altura;
o corpo dele se deslocou {_n(r["pelvis_displacement_at_closest_m"], 2)} m até ali.</p>
<table><tr><th>Momento</th><th>Quadro</th><th>t (s)</th><th>mão → bola (m)</th><th>pés → bola (m)</th>
<th>altura da bola (m)</th><th>altura da mão (m)</th><th>bola → linha (m)</th></tr>{"".join(rows)}</table>
<figure><img src="figuras/dist.png"><figcaption>Distância da bola à mão mais próxima do goleiro (3D) e aos pés (no chão).</figcaption></figure>
<h2>6. Deu tempo de defender?</h2>
<div class="verdict"><b>{html.escape(m["editorial_verdict"]["headline"])}.</b> {html.escape(m["editorial_verdict"]["summary_pt"])}</div>
<p>O goleiro reagiu em cerca de {_n(r["onset_s"], 2)} s, na ordem do tempo de reação simples medido em laboratório
(uma luz e um botão, perto de 0,2 s em adultos). Depois disso sobraram ~{_n(r["motor_time_after_onset_s"], 2)} s até a bola passar por ele:
o tempo existia; o que faltou foi alcançar uma bola alta, rápida e a ~{gap_cm} cm da mão. Um mergulho de goleiro começa pelo
quadril e depende do lado e da altura da bola, o que custa tempo extra no salto.</p>
<figure><img src="figuras/reacao.png"><figcaption>Linha do tempo: chute, reação medida do goleiro, passagem da bola e fim do recorte.</figcaption></figure>
<h2>7. O chutador</h2>
<figure><img src="figuras/sticks_side.png"><figcaption>Chutador no contato (vista lateral).</figcaption></figure>
<figure><img src="figuras/sticks_front_goal.png"><figcaption>Chutador no contato, olhando do gol.</figcaption></figure>
<h2>8. Conferência visual</h2>{checks}
<h2>9. Qualidade e limites</h2>
<p class="note">Uma câmera (16 mm), {m["camera"]["n_frames"]} quadros a {_n(m["camera"]["fps"], 3)} fps; bola marcada à mão;
modelo de voo {m["camera"]["flight_model"]} com erro de {_n(m["camera"]["rms_px"], 1)} px. {crossing}
As poses são do SAM 3D Body com altura padrão; a distância da mão tem incerteza de ±0,3–0,5 m e o instante da reação,
de ±1 quadro ({_n(r["onset_resolution_s"], 3)} s). A reação é medida pela subida da mão
({html.escape(r["rule"])}). O resultado (gol) vem do vídeo da transmissão, não da reconstrução.</p>
<h2>10. Frases para teleprompter</h2><ul>{tele}</ul>
<h2>11. Leitura científica e referências</h2>
<p>O tempo de reação simples de laboratório é um piso cognitivo: uma defesa exige escolher o lado e mover o corpo inteiro.
Aqui, a reação foi medida no próprio lance; as referências dão contexto, não validam a reconstrução.</p><ol class="refs">{refs}</ol>
<h2>12. Auditoria e reprodução</h2>
<p><a href="data/metricas_materia.json">Métricas</a> · <a href="inputs_sha256.json">Entradas</a> ·
<a href="outputs_sha256.json">Saídas</a> · <a href="validacao_entrega.json">Validação</a> ·
<a href="README_reproducao.md">Comando de reprodução</a>.</p>
<footer>Lance Decifrado · FreeKiki / <i>vailá</i> · AGPLv3</footer>
</main></body></html>"""
    path = out / "relatorio_chute.html"
    path.write_text(markup, encoding="utf-8")
    return path


# ------------------------------------------------------------------------------- package


def _copy(src: Path, dst: Path, used: list[Path]) -> None:
    if src.is_file():
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        used.append(src)


def build_shot_tv_package(
    run: Path | str,
    out: Path | str,
    *,
    kicker: int,
    keeper: int,
    outcome: str = "goal",
    outcome_note: str = "gol confirmado no vídeo da transmissão (operador)",
    pass_package: Path | str | None = None,
    animation: bool = True,
    command: list[str] | None = None,
) -> dict:
    run = Path(run).resolve()
    out = Path(out).resolve()
    if out == run:
        raise ValueError("the package must go to a separate folder")
    out.mkdir(parents=True, exist_ok=True)
    m = build_metrics(
        run,
        kicker=kicker,
        keeper=keeper,
        outcome=outcome,
        outcome_note=outcome_note,
        pass_package=Path(pass_package) if pass_package else None,
    )
    data = out / "data"
    data.mkdir(exist_ok=True)
    (data / "metricas_materia.json").write_text(
        json.dumps(m, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    used: list[Path] = []
    figs = out / "figuras"
    make_figures(m, figs)
    rt = run / "report_tv" / "pt" / "figuras"
    for src, name in (
        ("fig_vista_de_cima.png", "trajetoria.png"),
        ("fig_perfil.png", "perfil.png"),
        ("fig_forcas.png", "forcas.png"),
        ("fig_curva.png", "curva.png"),
    ):
        _copy(rt / src, figs / name, used)
    for view in ("side", "front_goal", "top"):
        _copy(run / "data" / f"kicker_sticks_{view}_pt.png", figs / f"sticks_{view}.png", used)
    tv = ShotTV(m, run)
    r = m["keeper_response"]
    stills = [("inicio", 0), ("passagem", int(r["closest_frame"])), ("fim", tv.n - 1)]
    if r["onset_frame"] is not None:
        stills.insert(1, ("reacao", int(r["onset_frame"])))
    for label, f in stills:
        Image.fromarray(tv.render(f)).save(out / f"conferencia_{label}.jpg", quality=92)
    scenes = build_scenes(m)
    write_script(out, scenes, m)
    videos = write_videos(tv, out, scenes) if animation else []
    extra = []
    for src, name in (
        (next(iter(sorted(run.glob("*_3d_overlay.mp4"))), None), "lance_overlay.mp4"),
        (next(iter(sorted(run.glob("*_topview.mp4"))), None), "lance_topview.mp4"),
        (run / "report_tv" / "pt" / "animacao_lance.mp4", "animacao_lance.mp4"),
    ):
        if src is not None and Path(src).is_file():
            _copy(Path(src), out / name, used)
            extra.append(name)
    html_path = build_html(m, out, videos, extra)
    pdf = print_pdf(html_path, out / "relatorio_chute.pdf")
    for name in (
        "ball_3d_on_field.csv",
        "ball_flight.json",
        "ball.csv",
        "ground_tracks.csv",
        "player_map.csv",
        "README.txt",
    ):
        _copy(run / name, data / name, used)
    for name in ("analysis.json", "kicker_metrics.json", "setpiece.json"):
        _copy(run / "data" / name, data / name, used)
    for pid, role in ((kicker, "kicker"), (keeper, "keeper")):
        _copy(run / f"person_{pid:03d}.c3d", data / f"person_{pid:03d}_{role}.c3d", used)
        _copy(run / f"person_{pid:03d}_c3d.csv", data / f"person_{pid:03d}_{role}_c3d.csv", used)
    if tv.kp_file is not None:
        with tv.kp_file.open(newline="", encoding="utf-8") as fh:
            rd = csv.DictReader(fh)
            fields = list(rd.fieldnames or [])
            rows = [x for x in rd if x.get("person_id") in (str(kicker), str(keeper))]
        with (data / f"keypoints2d_ids_{kicker}_{keeper}.csv").open(
            "w", newline="", encoding="utf-8"
        ) as fh:
            wr = csv.DictWriter(fh, fieldnames=fields)
            wr.writeheader()
            wr.writerows(rows)
        used.append(tv.kp_file)
    _copy(tv.video, data / tv.video.name, used)
    inputs = []
    for p in dict.fromkeys(used):
        inputs.append({"path": str(p), "bytes": p.stat().st_size, "sha256": _sha256(p)})
    (out / "inputs_sha256.json").write_text(json.dumps(inputs, indent=2), encoding="utf-8")
    cmd = command or [
        sys.executable,
        "-m",
        "vaila.freekiki_shot_tv",
        "--run",
        str(run),
        "--kicker",
        str(kicker),
        "--keeper",
        str(keeper),
        "--outcome",
        outcome,
        "--out",
        str(out),
    ]
    (out / "README_reproducao.md").write_text(
        "# Chute a gol para TV — reprodução\n\n"
        + shlex.join(cmd)
        + "\n\nO comando lê o run FreeKiki (somente leitura) e reconstrói este pacote inteiro: "
        "métricas, figuras, vídeos, relatório, roteiro e hashes.\n\n"
        "## Decisões\n\n"
        f"- Chutador SAM {kicker}, goleiro SAM {keeper} (conferido na imagem).\n"
        f"- Resultado: {outcome} — {outcome_note}.\n"
        f"- Velocidade na TV: cerca de {m['speeds']['headline_kmh']} km/h "
        f"(saída do pé {m['speeds']['launch_kmh']:.1f} km/h, arredondada a 5; média no caminho "
        f"{m['speeds']['mean_path_kmh']:.1f} km/h). Depois do chute a velocidade mostrada só cai.\n"
        f"- Reação do goleiro: {r['rule']}.\n",
        encoding="utf-8",
    )
    probes = {v: _probe(out / v) for v in videos + extra if (out / v).is_file()}
    checks = {
        "tv_60s_1800_frames": probes.get("explicacao_tv.mp4", {}).get("nb_read_frames")
        == str(TV_SECONDS * TV_RATE)
        if animation
        else None,
        "natural_all_frames": probes.get("lance_kick_ui.mp4", {}).get("nb_read_frames") == str(tv.n)
        if animation
        else None,
        "pdf": pdf is not None,
        "keeper_onset_found": r["onset_frame"] is not None,
    }
    (out / "validacao_entrega.json").write_text(
        json.dumps(
            {
                "created": date.today().isoformat(),
                "videos": probes,
                "checks": checks,
                "goal_crossing": m["goal_crossing"],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    (out / "report_summary.json").write_text(
        json.dumps(
            {
                "verdict": m["editorial_verdict"],
                "speeds": {k: v for k, v in m["speeds"].items() if k != "speed_mps_series"},
                "keeper": {k: v for k, v in r.items() if k != "series"},
                "videos": videos + extra,
                "report": str(html_path),
                "pdf": str(pdf) if pdf else None,
                "command": cmd,
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    outputs = {
        str(p.relative_to(out)): {"sha256": _sha256(p), "bytes": p.stat().st_size}
        for p in sorted(out.rglob("*"))
        if p.is_file() and p.name != "outputs_sha256.json"
    }
    (out / "outputs_sha256.json").write_text(json.dumps(outputs, indent=2), encoding="utf-8")
    return {"dir": str(out), "checks": checks, "metrics": m}


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="FreeKiki shot TV package (vailá)")
    ap.add_argument("--run", required=True, type=Path, help="kick-analysis 3D folder")
    ap.add_argument("--out", required=True, type=Path, help="delivery folder")
    ap.add_argument("--kicker", required=True, type=int, help="SAM id of the kicker")
    ap.add_argument("--keeper", required=True, type=int, help="SAM id of the keeper")
    ap.add_argument("--outcome", default="goal", help="result seen in the footage (goal, save…)")
    ap.add_argument("--outcome-note", default="gol confirmado no vídeo da transmissão (operador)")
    ap.add_argument("--pass-package", type=Path, help="pass TV folder of the same play")
    ap.add_argument("--no-animation", action="store_true", help="skip the two videos")
    a = ap.parse_args(argv)
    res = build_shot_tv_package(
        a.run,
        a.out,
        kicker=a.kicker,
        keeper=a.keeper,
        outcome=a.outcome,
        outcome_note=a.outcome_note,
        pass_package=a.pass_package,
        animation=not a.no_animation,
        command=[sys.executable, "-m", "vaila.freekiki_shot_tv", *(argv or sys.argv[1:])],
    )
    print(f"[freekiki_shot_tv] package: {res['dir']} | checks: {res['checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
