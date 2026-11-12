"""
================================================================================
Script: pynalty_freekiki.py - Pynalty penalty analysis bridged with FreeKiki 3D
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.17
Created: 09 November 2026
Update Date: 12 November 2026

Description:
    Bridges FreeKiki 3D penalty runs (per-frame DLT3D, ball flight and 3D poses)
    with Pynalty (:mod:`vaila.pynalty_analysis` and :mod:`vaila.pynalty_report`).
    Provides programmatic and CLI access to compute Pynalty metrics and reports
    directly from a finished FreeKiki 3D or ``kick-analysis`` folder.

License:
    This project is licensed under the terms of AGPLv3.0.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

try:
    from .freekiki_kick_pipeline import _penalty
    from .freekiki_setpiece import classify_run
except ImportError:
    from freekiki_kick_pipeline import _penalty  # ty: ignore[unresolved-import]
    from freekiki_setpiece import classify_run  # ty: ignore[unresolved-import]


def run_pynalty_from_freekiki(
    ref3d: Path | str,
    video: Path | str,
    work_dir: Path | str,
    out_dir: Path | str | None = None,
    *,
    goal_x: float | None = None,
    save_frame: int | None = None,
    shot_outcome: str = "auto",
    gk_height_m: float | None = None,
    gk_arm_span_m: float | None = None,
    gk_mass_kg: float | None = None,
    kicker_height_m: float | None = None,
    gk_onset_frame: int | None = None,
    save_sigma_along: float | None = None,
) -> dict:
    """Compute Pynalty + FreeKiki penalty metrics and bilingual HTML report for a 3D run."""
    ref3d_p = Path(ref3d)
    video_p = Path(video)
    work_p = Path(work_dir)
    out_p = Path(out_dir) if out_dir is not None else ref3d_p
    gx = (
        float(goal_x)
        if goal_x is not None
        else float((classify_run(ref3d_p) or {}).get("goal_x_m") or 52.45)
    )
    statures: dict[str, float] = {}
    if gk_height_m is not None:
        statures["keeper"] = float(gk_height_m)
    if kicker_height_m is not None:
        statures["kicker"] = float(kicker_height_m)
    opts = {
        "save_frame": save_frame,
        "shot_outcome": shot_outcome,
        "statures": statures,
        "gk_arm_span": gk_arm_span_m,
        "gk_mass": gk_mass_kg,
        "gk_onset_frame": gk_onset_frame,
        "save_sigma_along": save_sigma_along,
    }
    return _penalty(ref3d_p, video_p, work_p, out_p, goal_x=gx, opts=opts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="pynalty_freekiki.py",
        description="Run Pynalty penalty analysis on a FreeKiki 3D folder.",
    )
    parser.add_argument("--ref3d", required=True, type=Path, help="FreeKiki 3D folder")
    parser.add_argument("--video", required=True, type=Path, help="Reference video file")
    parser.add_argument("--work", required=True, type=Path, help="FreeKiki work folder")
    parser.add_argument("--out", default=None, type=Path, help="Output analysis folder")
    parser.add_argument("--goal-x", type=float, default=None, help="Attacked goal-line x (m)")
    parser.add_argument("--save-frame", type=int, default=None, help="0-based save frame")
    parser.add_argument("--shot-outcome", default="auto", help="auto|goal|save|miss|woodwork")
    parser.add_argument("--gk-height", type=float, default=None, help="Goalkeeper stature (m)")
    parser.add_argument("--gk-arm-span", type=float, default=None, help="Goalkeeper arm span (m)")
    parser.add_argument("--gk-mass", type=float, default=None, help="Goalkeeper body mass (kg)")
    parser.add_argument("--kicker-height", type=float, default=None, help="Kicker stature (m)")
    parser.add_argument("--gk-onset-frame", type=int, default=None, help="0-based GK onset frame")
    parser.add_argument(
        "--save-sigma-along", type=float, default=None, help="Save depth uncertainty (m)"
    )
    args = parser.parse_args(argv)
    res = run_pynalty_from_freekiki(
        args.ref3d,
        args.video,
        args.work,
        args.out,
        goal_x=args.goal_x,
        save_frame=args.save_frame,
        shot_outcome=args.shot_outcome,
        gk_height_m=args.gk_height,
        gk_arm_span_m=args.gk_arm_span,
        gk_mass_kg=args.gk_mass,
        kicker_height_m=args.kicker_height,
        gk_onset_frame=args.gk_onset_frame,
        save_sigma_along=args.save_sigma_along,
    )
    print(json.dumps({"available": res.get("available"), "report": res.get("report")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
