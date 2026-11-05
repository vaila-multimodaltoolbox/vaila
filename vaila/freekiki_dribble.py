"""
================================================================================
Script: freekiki_dribble.py - 1vN dribble analysis on FreeKiki open-play runs
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
    Analyse a filmed dribble sequence from planar ball and person tracks:

    * ball path length, speed, accel / decel on the pitch plane;
    * dribbler path, COD events, mean / peak curvature (rad/m), lateral deviation;
    * each marker: same kinematics plus interpersonal distance / relative speed to the
      dribbler (min approach).

    Person ids are SAM / ``person_id`` values typed by the user (not remapped player_id 0/1).
    Writes ``data/dribble_analysis.json``.

    References (framing only): Wilson et al. 2018 J Sports Sci (PMID 30470166);
    Wilson–Santiago predator–prey COD; Corrêa et al. interpersonal distance / relative velocity.
"""

from __future__ import annotations

from pathlib import Path

try:
    from .freekiki_openplay_kinematics import (
        detect_fps,
        load_ball_xy,
        load_xy_series,
        pad_to,
        pair_approach,
        track_metrics,
        write_json,
    )
except ImportError:
    from freekiki_openplay_kinematics import (  # ty: ignore[unresolved-import]
        detect_fps,
        load_ball_xy,
        load_xy_series,
        pad_to,
        pair_approach,
        track_metrics,
        write_json,
    )

ANALYSIS_NAME = "dribble_analysis.json"


def _strip_series(metrics: dict, *, keep_series: bool) -> dict:
    """Drop bulky series when a compact summary is enough."""
    if keep_series:
        return metrics
    skip = {"speed_mps", "accel_mps2", "heading_rad", "curvature_rad_per_m", "xy_m", "cod_events"}
    out = {k: v for k, v in metrics.items() if k not in skip}
    out["cod_events"] = metrics.get("cod_events") or []
    return out


def analyze_dribble(
    folder: Path | str,
    dribbler_id: int,
    marker_ids: list[int],
    *,
    fps: float | None = None,
    keep_series: bool = True,
) -> dict:
    """Planar dribble metrics for ``dribbler_id`` vs ``marker_ids`` (SAM person ids)."""
    folder = Path(folder)
    if not marker_ids:
        raise ValueError("at least one marker_id is required for a dribble analysis")
    markers = [int(m) for m in marker_ids if m is not None and str(m).strip() != ""]
    if not markers:
        raise ValueError("at least one marker_id is required for a dribble analysis")
    rate = float(fps) if fps and fps > 0 else detect_fps(folder)
    ball = load_ball_xy(folder)
    drib = load_xy_series(folder, int(dribbler_id))
    n = max(len(ball), len(drib), *(len(load_xy_series(folder, m)) for m in markers))
    if n <= 0:
        raise FileNotFoundError(
            f"no ball or person tracks in {folder}: need ball_3d_on_field.csv and ground_tracks"
        )
    ball = pad_to(ball, n)
    drib = pad_to(drib, n)
    ball_m = track_metrics(ball, rate)
    drib_m = track_metrics(drib, rate)
    marker_rows = []
    for mid in markers:
        xy = pad_to(load_xy_series(folder, mid), n)
        person = track_metrics(xy, rate)
        approach = pair_approach(drib, xy, rate)
        marker_rows.append(
            {
                "person_id": int(mid),
                "role": "marker",
                "metrics": _strip_series(person, keep_series=keep_series),
                "vs_dribbler": {
                    "min_distance_m": approach["min_distance_m"],
                    "min_frame": approach["min_frame"],
                    "min_t_s": approach["min_t_s"],
                    "rel_speed_at_min_mps": approach["rel_speed_at_min_mps"],
                    **(
                        {
                            "distance_m": approach["distance_m"],
                            "rel_speed_mps": approach["rel_speed_mps"],
                        }
                        if keep_series
                        else {}
                    ),
                },
            }
        )
    result = {
        "kind": "dribble",
        "fps": rate,
        "n_frames": n,
        "dribbler_id": int(dribbler_id),
        "marker_ids": markers,
        "ball": _strip_series(ball_m, keep_series=keep_series),
        "dribbler": _strip_series(drib_m, keep_series=keep_series),
        "markers": marker_rows,
        "notes": (
            "Planar metrics from FreeKiki tracks. Curvature in rad/m follows the Wilson curved-path "
            "framing; COD events are heading changes ≥ 25°. Interpersonal distance / relative "
            "velocity follow the Corrêa–Duarte 1v1 operational reading. Not a closed-skill lab test."
        ),
    }
    if keep_series:
        result["ball"] = ball_m
        result["dribbler"] = drib_m
    return result


def run_dribble(
    folder: Path | str,
    dribbler_id: int,
    marker_ids: list[int],
    *,
    fps: float | None = None,
    keep_series: bool = True,
) -> dict:
    """Analyse and write ``data/dribble_analysis.json``."""
    result = analyze_dribble(folder, dribbler_id, marker_ids, fps=fps, keep_series=keep_series)
    path = write_json(folder, ANALYSIS_NAME, result)
    result["path"] = str(path)
    return result
