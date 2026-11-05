"""
================================================================================
Script: freekiki_openplay_report.py - TV report for FreeKiki dribble / pass analyses
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
    Builds ``report_tv/<lang>/`` HTML (+ optional PDF) for planar open-play analyses:

    * ``dribble`` — ball + dribbler + markers (COD, curvature, interpersonal distance);
    * ``pass`` — passer + receiver + ball (distances and release→receive window).

    Uses the analysis JSON from :mod:`freekiki_dribble` / :mod:`freekiki_pass` and the same
    press-report visual language as the free-kick TV report (cards, top view, series plots).
"""

from __future__ import annotations

import base64
import re
from datetime import date
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np

try:
    from .freekiki_dribble import run_dribble
    from .freekiki_layout import data_dir as analysis_data_dir
    from .freekiki_pass import run_pass
    from .freekiki_report import print_pdf
    from .freekiki_report_i18n import LANGS, fmt_num, tr
    from .freekiki_setpiece import apply_choice
    from .freekiki_setpiece import write as write_setpiece
except ImportError:
    from freekiki_dribble import run_dribble  # ty: ignore[unresolved-import]
    from freekiki_layout import data_dir as analysis_data_dir  # ty: ignore[unresolved-import]
    from freekiki_pass import run_pass  # ty: ignore[unresolved-import]
    from freekiki_report import print_pdf  # ty: ignore[unresolved-import]
    from freekiki_report_i18n import LANGS, fmt_num, tr  # ty: ignore[unresolved-import]
    from freekiki_setpiece import apply_choice  # ty: ignore[unresolved-import]
    from freekiki_setpiece import write as write_setpiece  # ty: ignore[unresolved-import]

HERE = Path(__file__).resolve().parent
TEMPLATES = {
    "pt": HERE / "freekiki_openplay_report_template.html",
    "en": HERE / "freekiki_openplay_report_template_en.html",
}
FILE_NAMES = {
    "pt": ("relatorio_openplay.html", "relatorio_openplay.pdf"),
    "en": ("openplay_report.html", "openplay_report.pdf"),
}
BALL_C = "#d62246"
DRIB_C = "#14213d"
MARK_CS = ("#14934b", "#3d5a80", "#f4a259", "#7b2cbf")
PASSER_C = "#14213d"
RECV_C = "#14934b"


def _num(x, nd: int = 1, lang: str = "pt") -> str:
    if x is None:
        return "—"
    return fmt_num(float(x), nd, lang)


def _kmh(mps, lang: str = "pt") -> str:
    if mps is None:
        return "—"
    return _num(float(mps) * 3.6, 0, lang)


def _xy(series) -> np.ndarray:
    if not series:
        return np.zeros((0, 2))
    rows = []
    for row in series:
        if row is None or row[0] is None or row[1] is None:
            rows.append([np.nan, np.nan])
        else:
            rows.append([float(row[0]), float(row[1])])
    return np.asarray(rows, dtype=float)


def _series(values) -> np.ndarray:
    if not values:
        return np.zeros(0)
    return np.asarray([np.nan if v is None else float(v) for v in values], dtype=float)


def _b64(path: Path) -> str:
    data = path.read_bytes()
    return "data:image/png;base64," + base64.b64encode(data).decode("ascii")


def _select_blocks(html: str, kind: str) -> str:
    keep = {"dribble": ("dr",), "pass": ("pa",)}.get(kind, ())
    for tag in ("dr", "pa"):
        if tag in keep:
            html = html.replace(f"<!--{tag}-->", "").replace(f"<!--/{tag}-->", "")
        else:
            html = re.sub(rf"<!--{tag}-->.*?<!--/{tag}-->", "", html, flags=re.S)
    return html


def _save_fig(fig, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    return path


def make_figures(analysis: dict, out_dir: Path, lang: str) -> dict[str, Path]:
    """Top view, speed series and interpersonal / pass-distance series."""
    out_dir.mkdir(parents=True, exist_ok=True)
    kind = analysis["kind"]
    if kind == "pass" and analysis.get("timing"):
        return _pass_figures(analysis, out_dir, lang)
    fps = float(analysis["fps"])
    figs: dict[str, Path] = {}
    palette = {p["id"]: p["color"] for p in analysis.get("participants", [])}
    drib_color = palette.get(analysis.get("dribbler_id"), DRIB_C)

    # --- top view ---
    fig, ax = plt.subplots(figsize=(7.2, 5.2))
    ball = _xy(analysis["ball"].get("xy_m"))
    if len(ball):
        ax.plot(ball[:, 0], ball[:, 1], color=BALL_C, lw=2.0, label=tr(lang, "op_legend_ball"))
    if kind == "dribble":
        drib = _xy(analysis["dribbler"].get("xy_m"))
        if len(drib):
            ax.plot(
                drib[:, 0],
                drib[:, 1],
                color=drib_color,
                lw=2.2,
                label=tr(lang, "op_legend_dribbler"),
            )
        for i, row in enumerate(analysis.get("markers") or []):
            xy = _xy(row["metrics"].get("xy_m"))
            if not len(xy):
                continue
            c = palette.get(row["person_id"], MARK_CS[i % len(MARK_CS)])
            ax.plot(
                xy[:, 0],
                xy[:, 1],
                color=c,
                lw=1.6,
                label=row.get("name") or tr(lang, "op_legend_marker", id=row["person_id"]),
            )
            vs = row.get("vs_dribbler") or {}
            mf = vs.get("min_frame")
            if (
                mf is not None
                and 0 <= mf < len(drib)
                and 0 <= mf < len(xy)
                and np.isfinite(drib[mf]).all()
                and np.isfinite(xy[mf]).all()
            ):
                ax.plot(
                    [drib[mf, 0], xy[mf, 0]],
                    [drib[mf, 1], xy[mf, 1]],
                    color=c,
                    ls="--",
                    lw=1.0,
                )
    else:
        passer = _xy(analysis["passer"].get("xy_m"))
        receiver = _xy(analysis["receiver"].get("xy_m"))
        if len(passer):
            ax.plot(
                passer[:, 0],
                passer[:, 1],
                color=PASSER_C,
                lw=2.0,
                label=tr(lang, "op_legend_passer"),
            )
        if len(receiver):
            ax.plot(
                receiver[:, 0],
                receiver[:, 1],
                color=RECV_C,
                lw=2.0,
                label=tr(lang, "op_legend_receiver"),
            )
        win = analysis.get("window") or {}
        for key, color in (("release_frame", PASSER_C), ("receive_frame", RECV_C)):
            f = win.get(key)
            if f is not None and 0 <= f < len(ball) and np.isfinite(ball[f]).all():
                ax.scatter([ball[f, 0]], [ball[f, 1]], s=40, color=color, zorder=5)
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_xlabel(tr(lang, "op_fig_x"))
    ax.set_ylabel(tr(lang, "op_fig_y"))
    ax.set_title(tr(lang, "op_fig_top_title"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    figs["top"] = _save_fig(fig, out_dir / "top.png")

    # --- speed ---
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    t = np.arange(analysis["n_frames"]) / fps if fps > 0 else np.arange(analysis["n_frames"])
    v_ball = _series(analysis["ball"].get("speed_mps")) * 3.6
    if len(v_ball):
        ax.plot(t[: len(v_ball)], v_ball, color=BALL_C, lw=1.8, label=tr(lang, "op_legend_ball"))
    if kind == "dribble":
        v_d = _series(analysis["dribbler"].get("speed_mps")) * 3.6
        if len(v_d):
            ax.plot(
                t[: len(v_d)], v_d, color=drib_color, lw=1.6, label=tr(lang, "op_legend_dribbler")
            )
    else:
        for key, color, lab in (
            ("passer", PASSER_C, "op_legend_passer"),
            ("receiver", RECV_C, "op_legend_receiver"),
        ):
            vv = _series(analysis[key].get("speed_mps")) * 3.6
            if len(vv):
                ax.plot(t[: len(vv)], vv, color=color, lw=1.5, label=tr(lang, lab))
    ax.set_xlabel(tr(lang, "op_fig_t"))
    ax.set_ylabel(tr(lang, "op_fig_speed"))
    ax.set_title(tr(lang, "op_fig_speed_title"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    figs["speed"] = _save_fig(fig, out_dir / "speed.png")

    # --- distances ---
    fig, ax = plt.subplots(figsize=(7.2, 3.8))
    if kind == "dribble":
        ball_distance = _series(analysis["ball"].get("distance_to_dribbler_m"))
        if len(ball_distance):
            ax.plot(
                t[: len(ball_distance)],
                ball_distance,
                color=BALL_C,
                ls="--",
                lw=1.5,
                label="Bola–driblador" if lang == "pt" else "Ball–dribbler",
            )
        for i, row in enumerate(analysis.get("markers") or []):
            dist = _series((row.get("vs_dribbler") or {}).get("distance_m"))
            if not len(dist):
                continue
            ax.plot(
                t[: len(dist)],
                dist,
                color=palette.get(row["person_id"], MARK_CS[i % len(MARK_CS)]),
                lw=1.6,
                label=row.get("name") or tr(lang, "op_legend_marker", id=row["person_id"]),
            )
        ax.set_title(tr(lang, "op_fig_dist_drib_title"))
    else:
        win = analysis.get("window") or {}
        for key, color, lab in (
            ("distance_passer_ball_m", PASSER_C, "op_legend_passer_ball"),
            ("distance_receiver_ball_m", RECV_C, "op_legend_receiver_ball"),
            ("distance_passer_receiver_m", "#7b2cbf", "op_legend_passer_receiver"),
        ):
            dist = _series(win.get(key))
            if len(dist):
                ax.plot(t[: len(dist)], dist, color=color, lw=1.6, label=tr(lang, lab))
        for key, color in (("release_frame", PASSER_C), ("receive_frame", RECV_C)):
            f = win.get(key)
            if f is not None:
                ax.axvline(f / fps if fps > 0 else f, color=color, ls="--", lw=1.0, alpha=0.7)
        ax.set_title(tr(lang, "op_fig_dist_pass_title"))
    ax.set_xlabel(tr(lang, "op_fig_t"))
    ax.set_ylabel(tr(lang, "op_fig_dist"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    figs["dist"] = _save_fig(fig, out_dir / "dist.png")
    return figs


# --- pass report (timing, receiver run, shot link) ---------------------------------------------

PITCH_HALF_L = 52.5
PENALTY_DEPTH, PENALTY_HALF_W = 16.5, 20.16
GOAL_AREA_DEPTH, GOAL_AREA_HALF_W = 5.5, 9.16
SHOT_C = "#d62246"


def _at(series, f: int | None, *, forward: int = 1) -> float | None:
    """``series[f]`` or the next finite value within ``forward`` frames (speeds start at 1)."""
    if f is None or not series:
        return None
    for k in range(f, min(len(series), f + forward + 1)):
        v = series[k]
        if v is not None and np.isfinite(v):
            return float(v)
    return None


def _draw_box(ax, goal_x: float) -> None:
    """Goal line, penalty area, goal area and penalty spot of the attacked goal."""
    s = 1.0 if goal_x > 0 else -1.0
    gx = goal_x
    kw = {"color": "#9aa5b8", "lw": 1.0, "zorder": 0}
    ax.plot([gx, gx], [-34, 34], **kw)
    for depth, half in ((PENALTY_DEPTH, PENALTY_HALF_W), (GOAL_AREA_DEPTH, GOAL_AREA_HALF_W)):
        x2 = gx - s * depth
        ax.plot([gx, x2, x2, gx], [-half, -half, half, half], **kw)
    ax.plot([gx, gx + s * 1.5, gx + s * 1.5, gx], [-3.66, -3.66, 3.66, 3.66], **kw)
    ax.scatter([gx - s * 11.0], [0.0], s=8, color="#9aa5b8", zorder=0)


def _pass_figures(analysis: dict, out_dir: Path, lang: str) -> dict[str, Path]:
    """Top view with release / first-touch positions, and speed / distance on t = 0 at release."""
    out_dir.mkdir(parents=True, exist_ok=True)
    timing = analysis["timing"]
    shot = analysis.get("shot") or {}
    pid, rid = analysis["passer_id"], analysis["receiver_id"]
    figs: dict[str, Path] = {}
    ball = _xy(analysis["ball"].get("xy_m"))
    passer = _xy(analysis["passer"].get("xy_m"))
    receiver = _xy(analysis["receiver"].get("xy_m"))
    rf = int(timing.get("release_frame") or 0)
    af = timing.get("arrival_frame")
    t = np.asarray(timing["t_rel_s"], dtype=float)

    # --- top view ---
    fig, ax = plt.subplots(figsize=(7.6, 5.8))
    pts = [a[np.isfinite(a).all(axis=1)] for a in (ball, passer, receiver) if len(a)]
    if shot.get("shot_xy"):
        pts.append(np.asarray([shot["shot_xy"], shot.get("shot_end_xy") or shot["shot_xy"]]))
    allp = np.vstack(pts) if pts else np.zeros((0, 2))
    goal_x = shot.get("goal_x_m")
    if goal_x is None and len(allp):
        goal_x = PITCH_HALF_L if float(np.nanmean(allp[:, 0])) > 0 else -PITCH_HALF_L
    if goal_x is not None:
        _draw_box(ax, float(goal_x))
    last = af if af is not None else len(ball) - 1
    if len(ball):
        ax.plot(
            ball[: last + 1, 0],
            ball[: last + 1, 1],
            color=BALL_C,
            lw=2.2,
            label=tr(lang, "op_legend_ball"),
        )
        if last + 1 < len(ball):
            ax.plot(ball[last:, 0], ball[last:, 1], color=BALL_C, lw=1.2, ls=":", alpha=0.7)
    for arr, color, lab in (
        (passer, PASSER_C, tr(lang, "op_legend_passer_id", id=pid)),
        (receiver, RECV_C, tr(lang, "op_legend_receiver_id", id=rid)),
    ):
        if not len(arr):
            continue
        ax.plot(arr[:, 0], arr[:, 1], color=color, lw=1.8, label=lab)
        if rf < len(arr) and np.isfinite(arr[rf]).all():
            ax.scatter([arr[rf, 0]], [arr[rf, 1]], marker="s", s=60, color=color, zorder=5)
        if af is not None and af < len(arr) and np.isfinite(arr[af]).all():
            ax.scatter(
                [arr[af, 0]],
                [arr[af, 1]],
                marker="o",
                s=60,
                color=color,
                zorder=5,
                edgecolor="white",
            )
    p0, arr_pt = timing.get("passer_at_release_xy"), timing.get("arrival_xy")
    if p0 and arr_pt:
        ax.plot([p0[0], arr_pt[0]], [p0[1], arr_pt[1]], color="black", ls="--", lw=1.2)
        mid = (np.asarray(p0) + np.asarray(arr_pt)) / 2.0
        ax.annotate(
            f"{_num(timing.get('passer_to_arrival_m'), 1, lang)} m",
            mid,
            xytext=(8, 0),
            textcoords="offset points",
            fontsize=11,
            fontweight="bold",
        )
        ax.scatter(
            [arr_pt[0]],
            [arr_pt[1]],
            marker="*",
            s=220,
            color="#f4a259",
            edgecolor="black",
            zorder=6,
            label=tr(lang, "op_legend_arrival_pt"),
        )
    if shot.get("shot_xy"):
        sx, sy = shot["shot_xy"]
        ax.scatter(
            [sx], [sy], marker="^", s=90, color=SHOT_C, zorder=6, label=tr(lang, "op_legend_shot")
        )
        end = shot.get("shot_end_xy")
        if end:
            ax.annotate(
                "",
                xy=end,
                xytext=(sx, sy),
                arrowprops={"arrowstyle": "->", "color": SHOT_C, "lw": 1.8},
            )
        if arr_pt:
            ax.plot([arr_pt[0], sx], [arr_pt[1], sy], color=SHOT_C, ls=":", lw=1.2)
    ax.set_aspect("equal", adjustable="box")
    if len(allp):
        lo, hi = allp.min(axis=0) - 3.0, allp.max(axis=0) + 3.0
        ax.set_xlim(lo[0], hi[0])
        ax.set_ylim(lo[1], hi[1])
    ax.set_xlabel(tr(lang, "op_fig_x"))
    ax.set_ylabel(tr(lang, "op_fig_y"))
    ax.set_title(tr(lang, "op_fig_top_title"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.2)
    figs["top"] = _save_fig(fig, out_dir / "top.png")

    series = timing["series"]
    t_arr = timing.get("arrival_t_s")

    def _marks(ax) -> None:
        ax.axvline(0.0, color="#5b6577", ls="--", lw=1.0, label=tr(lang, "op_legend_release"))
        if t_arr is not None:
            ax.axvline(
                t_arr, color=RECV_C, ls="-", lw=1.2, alpha=0.8, label=tr(lang, "op_legend_arrival")
            )

    # --- speed ---
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for key, color, lab in (
        ("ball_speed_mps", BALL_C, tr(lang, "op_legend_ball")),
        ("passer_speed_mps", PASSER_C, tr(lang, "op_legend_passer_id", id=pid)),
        ("receiver_speed_mps", RECV_C, tr(lang, "op_legend_receiver_id", id=rid)),
    ):
        vv = _series(series.get(key)) * 3.6
        if len(vv):
            ax.plot(t[: len(vv)], vv, color=color, lw=1.8, label=lab)
    _marks(ax)
    run = timing.get("receiver_run") or {}
    if run.get("lead_lower_bound_s") is not None and run.get("t_first_s") is not None:
        v0 = float(run["v_first_mps"]) * 3.6
        ax.annotate(
            tr(lang, "op_fig_already_running", id=rid, v=_num(v0, 0, lang)),
            (run["t_first_s"], v0),
            xytext=(40, -38),
            textcoords="offset points",
            fontsize=9,
            color=RECV_C,
            arrowprops={"arrowstyle": "->", "color": RECV_C, "lw": 1.0},
        )
    ax.set_xlim(min(-0.1, float(np.nanmin(t)) - 0.05), float(np.nanmax(t)) + 0.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(tr(lang, "op_fig_t_rel"))
    ax.set_ylabel(tr(lang, "op_fig_speed"))
    ax.set_title(tr(lang, "op_fig_speed_title"))
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, alpha=0.25)
    figs["speed"] = _save_fig(fig, out_dir / "speed.png")

    # --- distances ---
    fig, ax = plt.subplots(figsize=(7.4, 3.9))
    for key, color, lab in (
        ("distance_passer_ball_m", PASSER_C, "op_legend_passer_ball"),
        ("distance_receiver_ball_m", RECV_C, "op_legend_receiver_ball"),
        ("distance_passer_receiver_m", "#7b2cbf", "op_legend_passer_receiver"),
    ):
        dist = _series(series.get(key))
        if len(dist):
            ax.plot(t[: len(dist)], dist, color=color, lw=1.8, label=tr(lang, lab))
    _marks(ax)
    ax.set_xlim(min(-0.1, float(np.nanmin(t)) - 0.05), float(np.nanmax(t)) + 0.05)
    ax.set_ylim(bottom=0)
    ax.set_xlabel(tr(lang, "op_fig_t_rel"))
    ax.set_ylabel(tr(lang, "op_fig_dist"))
    ax.set_title(tr(lang, "op_fig_dist_pass_title"))
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.25)
    figs["dist"] = _save_fig(fig, out_dir / "dist.png")
    return figs


def _card(value: str, unit: str, label: str, tag: str | None, lang: str, cls: str = "") -> str:
    tag_html = ""
    if tag == "m":
        tag_html = f" <span class='tag solid'>{tr(lang, 'op_tag_measured')}</span>"
    elif tag == "e":
        tag_html = f" <span class='tag est'>{tr(lang, 'op_tag_estimate')}</span>"
    unit_html = f" <small>{unit}</small>" if unit else ""
    return (
        f"<div class='card {cls}'><div class='n'>{value}{unit_html}</div>"
        f"<div class='l'>{label}{tag_html}</div></div>"
    )


def _cards(*cards: str) -> str:
    return "<div class='cards'>" + "".join(cards) + "</div>"


def _xy_text(arr: np.ndarray, f: int | None, lang: str) -> str:
    if f is None or f >= len(arr) or not np.isfinite(arr[f]).all():
        return "—"
    return f"{_num(arr[f, 0], 1, lang)}; {_num(arr[f, 1], 1, lang)}"


def _pass_table(analysis: dict, lang: str) -> str:
    timing = analysis["timing"]
    series = timing["series"]
    shot = analysis.get("shot") or {}
    pid, rid = analysis["passer_id"], analysis["receiver_id"]
    ball = _xy(analysis["ball"].get("xy_m"))
    passer = _xy(analysis["passer"].get("xy_m"))
    receiver = _xy(analysis["receiver"].get("xy_m"))
    rf = int(timing.get("release_frame") or 0)
    af = timing.get("arrival_frame")
    first_after = float(timing["t_rel_s"][rf]) > 0.005  # release before the first frame
    rows = [("op_row_first" if first_after else "op_row_release", rf)]
    if af is not None:
        rows += [("op_row_mid", int(round((rf + af) / 2))), ("op_row_arrival", int(af))]
    head = (
        "<div class='tablewrap'><table class='metrics key'><thead><tr>"
        f"<th>{tr(lang, 'op_th_instant')}</th><th>{tr(lang, 'op_th_t')}</th>"
        f"<th>{tr(lang, 'op_th_ball_xy')}</th><th>{tr(lang, 'op_th_vball')}</th>"
        f"<th>{tr(lang, 'op_th_player_xy', id=pid)}</th><th>{tr(lang, 'op_th_vplayer', id=pid)}</th>"
        f"<th>{tr(lang, 'op_th_player_xy', id=rid)}</th><th>{tr(lang, 'op_th_vplayer', id=rid)}</th>"
        "</tr></thead><tbody>"
    )
    body = []
    for key, f in rows:
        body.append(
            "<tr>"
            f"<td>{tr(lang, key)}</td>"
            f"<td>{_num(timing['t_rel_s'][f], 2, lang)}</td>"
            f"<td>{_xy_text(ball, f, lang)}</td>"
            f"<td>{_kmh(_at(series['ball_speed_mps'], f), lang)}</td>"
            f"<td>{_xy_text(passer, f, lang)}</td>"
            f"<td>{_kmh(_at(series['passer_speed_mps'], f), lang)}</td>"
            f"<td>{_xy_text(receiver, f, lang)}</td>"
            f"<td>{_kmh(_at(series['receiver_speed_mps'], f), lang)}</td>"
            "</tr>"
        )
    if shot.get("shot_xy"):
        sx, sy = shot["shot_xy"]
        body.append(
            "<tr>"
            f"<td>{tr(lang, 'op_row_shot')}</td><td>—</td>"
            f"<td>{_num(sx, 1, lang)}; {_num(sy, 1, lang)}</td>"
            f"<td>~{_num(shot.get('speed_headline_kmh'), 0, lang)}</td>"
            "<td>—</td><td>—</td><td>—</td><td>—</td></tr>"
        )
    note = f"<p class='note'>{tr(lang, 'op_table_note')}</p>"
    return head + "".join(body) + "</tbody></table></div>" + note


def _pass_body(analysis: dict, lang: str) -> str:
    timing = analysis["timing"]
    b = timing.get("ball") or {}
    run = timing.get("receiver_run") or {}
    pas = timing.get("passer") or {}
    shot = analysis.get("shot") or {}
    parts = [
        f"<h2>{tr(lang, 'op_h_pass_ball')}</h2>",
        _cards(
            _card(
                _num(timing.get("passer_to_arrival_m"), 1, lang),
                "m",
                tr(lang, "op_card_pass_dist"),
                "m",
                lang,
                "hi",
            ),
            _card(
                _num(timing.get("flight_s"), 2, lang), "s", tr(lang, "op_card_pass_time"), "m", lang
            ),
            _card(
                _kmh(b.get("v_release_mps"), lang), "km/h", tr(lang, "op_card_pass_v0"), "m", lang
            ),
        ),
        _cards(
            _card(
                _kmh(b.get("v_arrival_mps"), lang), "km/h", tr(lang, "op_card_pass_v1"), "m", lang
            ),
            _card(
                _kmh(b.get("v_mean_mps"), lang), "km/h", tr(lang, "op_card_pass_vmean"), "m", lang
            ),
            _card(
                _num(timing.get("passer_receiver_at_release_m"), 1, lang),
                "m",
                tr(lang, "op_card_pair_release"),
                "m",
                lang,
            ),
        ),
        f"<h2>{tr(lang, 'op_h_pass_receiver')}</h2>",
    ]
    if run.get("lead_lower_bound_s") is not None:
        timing_card = _card(
            "≥ " + _num(run["lead_lower_bound_s"], 1, lang),
            "s",
            tr(lang, "op_card_recv_lead"),
            "e",
            lang,
            "recv",
        )
    elif run.get("onset_t_s") is not None:
        timing_card = _card(
            _num(run["onset_t_s"], 2, lang), "s", tr(lang, "op_card_recv_onset"), "m", lang, "recv"
        )
    else:
        timing_card = _card("—", "", tr(lang, "op_card_recv_onset"), None, lang, "recv")
    parts += [
        _cards(
            _card(
                _kmh(run.get("v_first_mps"), lang),
                "km/h",
                tr(lang, "op_card_recv_v0"),
                "m",
                lang,
                "recv",
            ),
            timing_card,
            _card(
                _num(run.get("displacement_m"), 1, lang),
                "m",
                tr(lang, "op_card_recv_disp"),
                "m",
                lang,
            ),
        ),
        _cards(
            _card(
                _kmh(run.get("v_peak_mps"), lang), "km/h", tr(lang, "op_card_recv_peak"), "m", lang
            ),
            _card(
                _kmh(run.get("v_arrival_mps"), lang), "km/h", tr(lang, "op_card_recv_v1"), "m", lang
            ),
            _card(
                _kmh(pas.get("v_release_mps"), lang),
                "km/h",
                tr(lang, "op_card_passer_v0"),
                "m",
                lang,
            ),
        ),
        f"<h2>{tr(lang, 'op_h_table')}</h2>",
        _pass_table(analysis, lang),
    ]
    if shot.get("shot_xy"):
        parts += [
            f"<h2>{tr(lang, 'op_h_shot')}</h2>",
            _cards(
                _card(
                    "~" + _num(shot.get("speed_headline_kmh"), 0, lang),
                    "km/h",
                    tr(lang, "op_card_shot_launch"),
                    "e",
                    lang,
                    "hi",
                ),
                _card(
                    _num(shot.get("distance_to_goal_m"), 1, lang),
                    "m",
                    tr(lang, "op_card_shot_goal"),
                    "m",
                    lang,
                ),
                _card(
                    _num(shot.get("arrival_to_shot_m"), 1, lang),
                    "m",
                    tr(lang, "op_card_shot_from_arrival"),
                    "m",
                    lang,
                ),
            ),
            _cards(
                _card(
                    _num(shot.get("mean_kmh"), 0, lang),
                    "km/h",
                    tr(lang, "op_card_shot_mean"),
                    "m",
                    lang,
                ),
                _card(
                    _num(shot.get("passer_to_shot_m"), 1, lang),
                    "m",
                    tr(lang, "op_card_shot_from_passer"),
                    "m",
                    lang,
                ),
                _card(
                    _num(shot.get("zmax_m"), 2, lang),
                    "m",
                    tr(lang, "op_card_shot_height"),
                    "e",
                    lang,
                ),
            ),
        ]
        gap = shot.get("gap_estimate_s") or []
        if gap:
            parts.append(
                "<p>"
                + tr(
                    lang,
                    "op_shot_text",
                    d=_num(shot.get("arrival_to_shot_m"), 1, lang),
                    g0=_num(gap[0], 1, lang),
                    g1=_num(gap[-1], 1, lang),
                    path=_num(shot.get("distance_m"), 1, lang),
                    fl=_num(shot.get("flight_s"), 2, lang),
                    mean=_num(shot.get("mean_kmh"), 0, lang),
                    launch=_num(shot.get("launch_kmh"), 0, lang),
                    head=_num(shot.get("speed_headline_kmh"), 0, lang),
                )
                + "</p>"
            )
    return "\n".join(parts)


def _pass_lead(analysis: dict, lang: str) -> str:
    timing = analysis["timing"]
    b = timing.get("ball") or {}
    run = timing.get("receiver_run") or {}
    shot = analysis.get("shot") or {}
    rid = analysis["receiver_id"]
    text = tr(
        lang,
        "op_lead_pass_main",
        passer=analysis["passer_id"],
        receiver=rid,
        v0=_kmh(b.get("v_release_mps"), lang),
        flight=_num(timing.get("flight_s"), 2, lang),
        dist=_num(timing.get("passer_to_arrival_m"), 1, lang),
    )
    if run.get("lead_lower_bound_s") is not None:
        text += tr(
            lang,
            "op_lead_pass_run_before",
            receiver=rid,
            vr=_kmh(run.get("v_first_mps"), lang),
            lead=_num(run["lead_lower_bound_s"], 1, lang),
        )
    elif run.get("onset_t_s") is not None:
        lead = float(run["onset_t_s"])
        text += tr(
            lang,
            "op_lead_pass_run_onset",
            receiver=rid,
            lead=_num(abs(lead), 2, lang),
            when=tr(lang, "op_when_before" if lead < 0 else "op_when_after"),
        )
    if shot.get("shot_xy") and shot.get("distance_to_goal_m") is not None:
        text += tr(
            lang,
            "op_lead_shot",
            receiver=rid,
            goal=_num(shot.get("distance_to_goal_m"), 1, lang),
            launch=_num(shot.get("speed_headline_kmh"), 0, lang),
        )
    return text


def _pass_limits(analysis: dict, lang: str) -> str:
    timing = analysis["timing"]
    run = timing.get("receiver_run") or {}
    fps = float(analysis["fps"])
    parts = [tr(lang, "op_limit_pass_fps", fps=_num(fps, 3, lang), dt=_num(1000.0 / fps, 0, lang))]
    if timing.get("release_source") == "flight_fit":
        parts.append(
            tr(
                lang,
                "op_limit_pass_release_fit",
                rel=_num(-float(timing["release_t_s"]), 3, lang),
                back=_num(
                    None
                    if timing.get("release_t_backextrap_s") is None
                    else -float(timing["release_t_backextrap_s"]),
                    3,
                    lang,
                ),
            )
        )
    else:
        parts.append(tr(lang, "op_limit_pass_release_heur"))
    if timing.get("arrival_frame") is not None:
        key = (
            "op_limit_pass_arrival_user"
            if timing.get("arrival_source") == "user"
            else "op_limit_pass_arrival_auto"
        )
        parts.append(tr(lang, key, f=timing["arrival_frame"]))
    parts.append(tr(lang, "op_limit_pass_tracks"))
    rng = run.get("lead_lower_bound_range_s")
    if run.get("lead_lower_bound_s") is not None and rng:
        parts.append(
            tr(
                lang,
                "op_limit_pass_run_bound",
                a=_num(run.get("a_max_mps2"), 0, lang),
                l0=_num(rng[0], 1, lang),
                l1=_num(rng[1], 1, lang),
                ahi=_num(7.0, 0, lang),
                alo=_num(4.0, 0, lang),
            )
        )
    if analysis.get("shot"):
        parts.append(tr(lang, "op_limit_pass_shot"))
    return " ".join(parts)


def _markers_table(analysis: dict, lang: str) -> str:
    rows = analysis.get("markers") or []
    if not rows:
        return f"<p>{tr(lang, 'op_no_markers')}</p>"
    head = (
        f"<table class='metrics'><thead><tr>"
        f"<th>{tr(lang, 'op_th_id')}</th>"
        f"<th>{tr(lang, 'op_th_min_d')}</th>"
        f"<th>{tr(lang, 'op_th_rel_v')}</th>"
        f"<th>{tr(lang, 'op_th_cod')}</th>"
        f"<th>{tr(lang, 'op_th_path')}</th>"
        f"</tr></thead><tbody>"
    )
    body = []
    for row in rows:
        vs = row.get("vs_dribbler") or {}
        m = row.get("metrics") or {}
        body.append(
            "<tr>"
            f"<td>{row['person_id']}</td>"
            f"<td>{_num(vs.get('min_distance_m'), 2, lang)}</td>"
            f"<td>{_num(vs.get('rel_speed_at_min_mps'), 2, lang)}</td>"
            f"<td>{m.get('cod_count', '—')}</td>"
            f"<td>{_num(m.get('path_length_m'), 1, lang)}</td>"
            "</tr>"
        )
    return head + "".join(body) + "</tbody></table>"


def _fill(analysis: dict, figs: dict[str, Path], meta: dict, lang: str) -> str:
    kind = analysis["kind"]
    ball = analysis["ball"]
    template = TEMPLATES[lang].read_text(encoding="utf-8")
    html = _select_blocks(template, kind)
    values = {
        "kicker": tr(lang, "op_kicker"),
        "title": meta["title"],
        "subtitle": meta["subtitle"],
        "authors": meta.get("authors") or "",
        "generated": meta["generated"],
        "ball_path": _num(ball.get("path_length_m"), 1, lang),
        "ball_vmax": _kmh(ball.get("v_max_mps"), lang),
        "ball_amean": _num(ball.get("a_peak_mps2"), 1, lang),
        "card_ball_path": tr(lang, "op_card_ball_path"),
        "card_ball_vmax": tr(lang, "op_card_ball_vmax"),
        "card_ball_accel": tr(lang, "op_card_ball_accel"),
        "h_top": tr(lang, "op_h_top"),
        "h_speed": tr(lang, "op_h_speed"),
        "h_dist": tr(lang, "op_h_dist"),
        "h_science": tr(lang, "op_h_science"),
        "h_markers": tr(lang, "op_h_markers"),
        "cap_top": tr(lang, "op_cap_top"),
        "cap_speed": tr(lang, "op_cap_speed"),
        "cap_dist": tr(lang, "op_cap_dist"),
        "science_text": tr(lang, "op_science_" + kind),
        "limit_text": tr(lang, "op_limit"),
        "ref1": tr(lang, "op_ref1"),
        "ref2": tr(lang, "op_ref2"),
        "ref3": tr(lang, "op_ref3"),
        "markers_table": "",
        "drib_path": "—",
        "drib_vmax": "—",
        "drib_cod": "—",
        "drib_curv": "—",
        "drib_accel": "—",
        "drib_decel": "—",
        "card_drib_path": tr(lang, "op_card_drib_path"),
        "card_drib_vmax": tr(lang, "op_card_drib_vmax"),
        "card_drib_cod": tr(lang, "op_card_drib_cod"),
        "card_drib_curv": tr(lang, "op_card_drib_curv"),
        "card_drib_accel": tr(lang, "op_card_drib_accel"),
        "card_drib_decel": tr(lang, "op_card_drib_decel"),
        "pass_len": "—",
        "pass_flight": "—",
        "pass_pair": "—",
        "card_pass_len": tr(lang, "op_card_pass_len"),
        "card_pass_flight": tr(lang, "op_card_pass_flight"),
        "card_pass_pair": tr(lang, "op_card_pass_pair"),
        "pass_body": "",
        "img_top": _b64(figs["top"]) if "top" in figs else "",
        "img_speed": _b64(figs["speed"]) if "speed" in figs else "",
        "img_dist": _b64(figs["dist"]) if "dist" in figs else "",
    }
    if kind == "dribble":
        d = analysis["dribbler"]
        values.update(
            {
                "lead": tr(
                    lang,
                    "op_lead_dribble",
                    dribbler=analysis["dribbler_id"],
                    n=len(analysis.get("marker_ids") or []),
                    path=_num(d.get("path_length_m"), 1, lang),
                    cod=d.get("cod_count", 0),
                    curv=_num(d.get("curvature_mean_rad_per_m"), 2, lang),
                ),
                "drib_path": _num(d.get("path_length_m"), 1, lang),
                "drib_vmax": _kmh(d.get("v_max_mps"), lang),
                "drib_cod": str(d.get("cod_count", 0)),
                "drib_curv": _num(d.get("curvature_mean_rad_per_m"), 2, lang),
                "drib_accel": _num(d.get("a_peak_mps2"), 1, lang),
                "drib_decel": _num(d.get("decel_peak_mps2"), 1, lang),
                "markers_table": _markers_table(analysis, lang),
            }
        )
    else:
        win = analysis.get("window") or {}
        values.update(
            {
                "lead": tr(
                    lang,
                    "op_lead_pass",
                    passer=analysis["passer_id"],
                    receiver=analysis["receiver_id"],
                    length=_num(
                        win.get("ball_path_release_to_receive_m") or win.get("pass_chord_m"),
                        1,
                        lang,
                    ),
                    flight=_num(win.get("flight_s"), 2, lang),
                ),
                "pass_len": _num(
                    win.get("ball_path_release_to_receive_m") or win.get("pass_chord_m"), 1, lang
                ),
                "pass_flight": _num(win.get("flight_s"), 2, lang),
                "pass_pair": _num(win.get("passer_receiver_at_release_m"), 1, lang),
            }
        )
        if analysis.get("timing"):
            values.update(
                {
                    "lead": _pass_lead(analysis, lang),
                    "pass_body": _pass_body(analysis, lang),
                    "cap_top": tr(lang, "op_cap_top_pass")
                    + (tr(lang, "op_cap_top_shot") if analysis.get("shot") else ""),
                    "cap_speed": tr(lang, "op_cap_speed_pass"),
                    "cap_dist": tr(lang, "op_cap_dist_pass"),
                    "science_text": tr(lang, "op_science_pass_timed")
                    + tr(lang, "op_science_pass_run"),
                    "limit_text": _pass_limits(analysis, lang),
                }
            )
    for key, value in values.items():
        html = html.replace("{{" + key + "}}", str(value))
    return html


def build_openplay_report(
    folder,
    *,
    kind: str,
    dribbler_id: int | None = None,
    marker_ids: list[int] | None = None,
    passer_id: int | None = None,
    receiver_id: int | None = None,
    shot_folder=None,
    arrival_frame: int | None = None,
    ball_source: str = "auto",
    fps: float | None = None,
    title: str = "",
    subtitle: str = "",
    authors: str = "",
    lang: str = "both",
    write_kind: bool = True,
) -> dict:
    """Run analysis, write figures + HTML/PDF under ``report_tv/<lang>/``."""
    folder = Path(folder)
    if kind not in ("dribble", "pass"):
        raise ValueError(f"kind must be 'dribble' or 'pass', got {kind!r}")
    if write_kind:
        write_setpiece(
            analysis_data_dir(folder),
            apply_choice(
                {"kind": "kick", "reason": "planar open-play report (no shot at goal)"},
                kind,
            ),
        )
    if kind == "dribble":
        if dribbler_id is None or not marker_ids:
            raise ValueError("dribble report needs --dribbler and at least one --markers id")
        analysis = run_dribble(folder, int(dribbler_id), list(marker_ids), fps=fps)
    else:
        if passer_id is None or receiver_id is None:
            raise ValueError("pass report needs --passer and --receiver")
        analysis = run_pass(
            folder,
            int(passer_id),
            int(receiver_id),
            fps=fps,
            ball=ball_source,
            shot_folder=shot_folder,
            arrival_frame=arrival_frame,
        )

    langs = list(LANGS) if lang == "both" else [lang]
    if any(code not in LANGS for code in langs):
        raise ValueError(f"lang must be one of {', '.join(LANGS)} or both")
    root = folder / "report_tv"
    result: dict = {
        "dir": str(root),
        "kind": kind,
        "analysis": analysis.get("path"),
        "languages": {},
    }
    today = date.today()
    for code in langs:
        out = root / code
        figs = make_figures(analysis, out / "figuras_openplay", code)
        meta = {
            "title": title or tr(code, "op_title_" + kind),
            "subtitle": subtitle or tr(code, "op_subtitle"),
            "authors": authors,
            "generated": tr(code, "date_format", d=today.day, m=today.month, y=today.year),
        }
        html = _fill(analysis, figs, meta, code)
        html_name, pdf_name = FILE_NAMES[code]
        html_path = out / html_name
        out.mkdir(parents=True, exist_ok=True)
        html_path.write_text(html, encoding="utf-8")
        pdf_path = print_pdf(html_path, out / pdf_name)
        result["languages"][code] = {
            "dir": str(out),
            "html": str(html_path),
            "pdf": str(pdf_path) if pdf_path else None,
        }
    return result
