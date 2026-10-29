"""
================================================================================
Script: freekiki_kicker_figures.py - stick figures, curves and overlays of the kicker
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.12
Created: 28 October 2026
Update Date: 28 October 2026

Description:
    The figures of the kicker, built from the ``context`` that :func:`freekiki_kicker.analyze_kicker`
    returns (triangulated keypoints, fitted ball, frame rate, contact frame):

    * ``kicker_sticks_side_{pt,en}.png``: stick figure of the side view (x = direction of the kick,
      z = height) at six instants around the contact;
    * ``kicker_sticks_top_{pt,en}.png``: the same instants from above (x, y);
    * ``kicker_curves_{pt,en}.png``: speeds, knee angle and extension, segment angular speeds, pelvis
      and limb angles, and the knee's share of the foot speed, all aligned to the contact, in the
      layout of Levanon & Dapena (1998);
    * ``kicker_overlay_{ref,other}_{pt,en}.png``: the triangulated keypoints reprojected onto the
      contact frame of each camera, with the residual of the keypoints of that camera. This is the
      visual check that the triangulation sits on the athlete;
    * ``kicker_interactive_{pt,en}.html``: a self-contained plotly page with a frame slider over the
      two stick views and the curves.

    A missing point is not drawn, and a missing segment is a gap, never a zero. Every label exists in
    Portuguese and English. The operational definitions of the angles are in :mod:`freekiki_kicker`.
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

LANGS = ("pt", "en")
SIX_INSTANTS = (-3, -2, -1, 0, 1, 2)  # frames from the contact: about -50 to +33 ms at 59.94 Hz
GROUP_COLORS = {"kick": "#c0392b", "support": "#1664a0", "trunk": "#555555"}


def _leg(side: str) -> tuple[tuple[str, str], ...]:
    return (
        (f"{side}-hip", f"{side}-knee"),
        (f"{side}-knee", f"{side}-ankle"),
        (f"{side}-ankle", f"{side}-big-toe-tip"),
        (f"{side}-ankle", f"{side}-heel"),
        (f"{side}-big-toe-tip", f"{side}-small-toe-tip"),
    )


#: The stick figure: neck, trunk with the pelvis line, both legs with the foot, both arms.
STICK_EDGES = (
    ("neck", "nose"),
    ("neck", "left-shoulder"),
    ("neck", "right-shoulder"),
    ("left-shoulder", "left-hip"),
    ("right-shoulder", "right-hip"),
    ("left-hip", "right-hip"),
    ("left-shoulder", "left-elbow"),
    ("left-elbow", "left-wrist"),
    ("right-shoulder", "right-elbow"),
    ("right-elbow", "right-wrist"),
    *_leg("left"),
    *_leg("right"),
)
JOINTS = tuple(dict.fromkeys(name for edge in STICK_EDGES for name in edge))

LABELS = {
    "pt": {
        "contact": "contato",
        "side_xlabel": "x: direção do chute (m)",
        "side_ylabel": "z: altura (m)",
        "top_xlabel": "x: direção do chute (m)",
        "top_ylabel": "y: lateral (m)",
        "side_title": "Vista lateral",
        "top_title": "Vista de cima",
        "sticks_suptitle": "Palito do chutador em instantes em relação ao contato",
        "curves_suptitle": "Chutador: velocidades, ângulos e joelho (0 = contato)",
        "speeds": "Velocidade (m/s)",
        "knee": "Joelho (graus)",
        "knee_ext_axis": "Extensão (graus/s)",
        "share_axis": "Fração (a.u.)",
        "angular": "Vel. angular (graus/s)",
        "angles": "Ângulos (graus)",
        "share": "Fração da velocidade do pé devida ao joelho",
        "foot": "pé de chute",
        "ball_fit": "bola (ajuste)",
        "lateral": "pé, componente lateral",
        "knee_angle": "ângulo do joelho",
        "knee_ext": "extensão do joelho (graus/s)",
        "thigh": "coxa",
        "shank": "perna",
        "foot_seg": "pé",
        "hip_angle": "quadril (ombro–quadril–joelho)",
        "ankle_angle": "tornozelo (joelho–tornozelo–dedo)",
        "pelvis": "pelve (linha dos quadris)",
        "plane": "planos coxa–perna e perna–pé",
        "time_s": "tempo a partir do contato (s)",
        "time_ms": "tempo (ms)",
        "window": "janela pré-contato",
        "kick_limb": "perna de chute",
        "support_limb": "perna de apoio",
        "trunk": "tronco e pelve",
        "ball": "bola",
        "ball_path": "trajetória da bola",
        "no_points": "sem pontos",
        "cam_ref": "câmera de referência",
        "cam_other": "outra câmera",
        "overlay_title": "Triangulação reprojetada: {cam}",
        "residual": "resíduo mediano {med:.1f} px, máximo {worst:.1f} px ({n} pontos)",
        "no_residual": "sem resíduo registrado neste quadro",
        "no_frame": "quadro do vídeo não disponível",
        "interactive_title": "Chutador: palito e curvas (interativo)",
    },
    "en": {
        "contact": "contact",
        "side_xlabel": "x: kick direction (m)",
        "side_ylabel": "z: height (m)",
        "top_xlabel": "x: kick direction (m)",
        "top_ylabel": "y: lateral (m)",
        "side_title": "Side view",
        "top_title": "Top view",
        "sticks_suptitle": "Kicker stick figure at instants relative to the contact",
        "curves_suptitle": "Kicker: speeds, angles and knee (0 = contact)",
        "speeds": "Speed (m/s)",
        "knee": "Knee (deg)",
        "knee_ext_axis": "Extension (deg/s)",
        "share_axis": "Fraction (a.u.)",
        "angular": "Angular speed (deg/s)",
        "angles": "Angles (deg)",
        "share": "Share of the foot speed made by the knee",
        "foot": "kicking foot",
        "ball_fit": "ball (fit)",
        "lateral": "foot, lateral component",
        "knee_angle": "knee angle",
        "knee_ext": "knee extension (deg/s)",
        "thigh": "thigh",
        "shank": "shank",
        "foot_seg": "foot",
        "hip_angle": "hip (shoulder–hip–knee)",
        "ankle_angle": "ankle (knee–ankle–toe)",
        "pelvis": "pelvis (hip line)",
        "plane": "thigh–shank and shank–foot planes",
        "time_s": "time from contact (s)",
        "time_ms": "time (ms)",
        "window": "pre-contact window",
        "kick_limb": "kicking leg",
        "support_limb": "support leg",
        "trunk": "trunk and pelvis",
        "ball": "ball",
        "ball_path": "ball path",
        "no_points": "no points",
        "cam_ref": "reference camera",
        "cam_other": "other camera",
        "overlay_title": "Reprojected triangulation: {cam}",
        "residual": "median residual {med:.1f} px, max {worst:.1f} px ({n} points)",
        "no_residual": "no residual recorded on this frame",
        "no_frame": "video frame not available",
        "interactive_title": "Kicker: stick figure and curves (interactive)",
    },
}


def _stereo():
    """The stereo and DLT helpers, imported here: they pull in the detection modules."""
    try:
        from . import freekiki_stereo as st
        from . import monocular_dlt_align as mda
    except ImportError:
        import freekiki_stereo as st  # ty: ignore[unresolved-import]
        import monocular_dlt_align as mda  # ty: ignore[unresolved-import]
    return st, mda


def _col(series: list[dict], key: str) -> np.ndarray:
    """A column of the series as floats (NaN where the value is empty)."""
    return np.array([float(r[key]) if r.get(key, "") != "" else math.nan for r in series])


def _point(points: dict[str, np.ndarray], name: str, f: int) -> np.ndarray | None:
    arr = points.get(name)
    if arr is None or not 0 <= f < len(arr):
        return None
    xyz = arr[f]
    return xyz if np.isfinite(xyz).all() else None


def _ball_at(ball: np.ndarray, f: int) -> np.ndarray | None:
    if not 0 <= f < len(ball) or not np.isfinite(ball[f]).all():
        return None
    return ball[f]


def _group(a: str, b: str, kick: str, support: str) -> str:
    for side, group in ((kick, "kick"), (support, "support")):
        if a.startswith(f"{side}-") and b.startswith(f"{side}-"):
            return group
    return "trunk"


def _segments(points, f: int, dims, kick: str, support: str) -> dict[str, list]:
    """The stick segments of one frame, grouped by limb, as ((x1, y1), (x2, y2)) on ``dims``."""
    out: dict[str, list] = {"kick": [], "support": [], "trunk": []}
    for a, b in STICK_EDGES:
        pa, pb = _point(points, a, f), _point(points, b, f)
        if pa is None or pb is None:
            continue
        seg = ((pa[dims[0]], pa[dims[1]]), (pb[dims[0]], pb[dims[1]]))
        out[_group(a, b, kick, support)].append(seg)
    return out


def _limits(points, ball: np.ndarray, frames, dims) -> tuple[float, float, float, float] | None:
    """A square box around every finite point of the frames (metres), or None when there is none."""
    rows = [arr[f] for arr in points.values() for f in frames if 0 <= f < len(arr)]
    rows += [ball[f] for f in frames if 0 <= f < len(ball)]
    good = [r for r in rows if np.isfinite(r).all()]
    if not good:
        return None
    xy = np.array(good)[:, list(dims)]
    lo, hi = xy.min(axis=0), xy.max(axis=0)
    half = max(float(np.max(hi - lo)), 1.0) / 2.0 + 0.4
    mid = (lo + hi) / 2.0
    return (mid[0] - half, mid[0] + half, mid[1] - half, mid[1] + half)


def _moment(k: int, fps: float, L: dict) -> str:
    return L["contact"] if k == 0 else f"{k / fps * 1000.0:+.0f} ms"


def _sticks_png(ctx: dict, path: Path, lang: str, dims: tuple[int, int], *, top: bool) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    contact, fps = ctx["contact"], ctx["fps"]
    kick, support, points, ball = ctx["kick_foot"], ctx["support_foot"], ctx["points"], ctx["ball"]
    frames = [contact + k for k in SIX_INSTANTS]
    box = _limits(points, ball, frames, dims)
    path_xy = [ball[f] for f in range(contact - 3, contact + 3) if 0 <= f < len(ball)]
    fig, axes = plt.subplots(1, len(frames), figsize=(2.6 * len(frames), 3.6), sharex=True,
                             sharey=True)  # fmt: skip
    for ax, k, f in zip(axes, SIX_INSTANTS, frames, strict=True):
        segs = _segments(points, f, dims, kick, support)
        for group, color in GROUP_COLORS.items():
            for p, q in segs[group]:
                ax.plot([p[0], q[0]], [p[1], q[1]], color=color, lw=2.2, solid_capstyle="round")
        joints = [_point(points, name, f) for name in JOINTS]
        joints = [j for j in joints if j is not None]
        if joints:
            ax.scatter([j[dims[0]] for j in joints], [j[dims[1]] for j in joints], s=8, color="#222222",
                       zorder=3)  # fmt: skip
        if path_xy:
            ax.plot([p[dims[0]] for p in path_xy], [p[dims[1]] for p in path_xy], ls=":",
                    color="#999999", lw=1)  # fmt: skip
        bp = _ball_at(ball, f)
        if bp is not None:
            ax.scatter([bp[dims[0]]], [bp[dims[1]]], s=50, color="#e67e22", zorder=4)
        if not any(segs.values()):
            ax.text(0.5, 0.5, L["no_points"], ha="center", va="center", transform=ax.transAxes)
        ax.set_title(_moment(k, fps, L), fontsize=9)
        ax.grid(alpha=0.3)
        ax.set_aspect("equal")
        if box is not None:
            ax.set_xlim(box[0], box[1])
            ax.set_ylim(box[2], box[3])
    fig.supxlabel(L["top_xlabel"] if top else L["side_xlabel"])
    fig.supylabel(L["top_ylabel"] if top else L["side_ylabel"])
    fig.suptitle(
        f"{L['top_title'] if top else L['side_title']}: {L['sticks_suptitle']}", fontsize=10
    )
    fig.tight_layout(rect=(0.04, 0.05, 1.0, 0.93))  # room on the left for the axis label
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _curves_png(ctx: dict, series: list[dict], path: Path, lang: str) -> Path:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    L = LABELS[lang]
    t = _col(series, "time_rel_s")
    fig, axes = plt.subplots(5, 1, figsize=(8, 14), sharex=True)
    ax_speed, ax_knee, ax_w, ax_ang, ax_share = axes

    ax_speed.plot(t, _col(series, "foot_speed_mps"), color="#1664a0", label=L["foot"])
    ax_speed.plot(t, _col(series, "ball_speed_mps"), color="#c0392b", label=L["ball_fit"])
    ax_speed.plot(t, _col(series, "lateral_foot_mps"), color="#7f8c8d", ls="--", label=L["lateral"])
    ax_speed.axvspan(-ctx["window_s"], 0.0, color="#f5b041", alpha=0.18, label=L["window"])
    ax_speed.set_ylabel(L["speeds"])

    ax_knee.plot(t, _col(series, "knee_angle_deg"), color="#14934b", label=L["knee_angle"])
    ax_knee.set_ylabel(L["knee"])
    twin = ax_knee.twinx()
    twin.plot(t, _col(series, "knee_ext_deg_s"), color="#a85c13", ls="--", label=L["knee_ext"])
    twin.set_ylabel(L["knee_ext_axis"])

    for key, label, color in (("thigh_w_deg_s", L["thigh"], "#1664a0"),
                              ("shank_w_deg_s", L["shank"], "#14934b"),
                              ("foot_w_deg_s", L["foot_seg"], "#a85c13")):  # fmt: skip
        ax_w.plot(t, _col(series, key), color=color, label=label)
    ax_w.set_ylabel(L["angular"])

    for key, label, color in (("pelvis_deg", L["pelvis"], "#8e44ad"),
                              ("plane_deg", L["plane"], "#2c3e50"),
                              ("hip_angle_deg", L["hip_angle"], "#16a085"),
                              ("ankle_angle_deg", L["ankle_angle"], "#d35400")):  # fmt: skip
        ax_ang.plot(t, _col(series, key), color=color, label=label)
    ax_ang.set_ylabel(L["angles"])

    ax_share.plot(t, _col(series, "knee_share"), color="#14934b", label=L["share"])
    ax_share.axhline(0.0, color="#999999", lw=0.8, ls=":")
    ax_share.axhline(1.0, color="#999999", lw=0.8, ls=":")
    ax_share.set_ylabel(L["share_axis"])
    ax_share.set_xlabel(L["time_s"])

    for ax in axes:
        ax.axvline(0.0, color="#333333", lw=0.8)
        ax.grid(alpha=0.3)
        handles, labels = ax.get_legend_handles_labels()
        if ax is ax_knee:
            h2, l2 = twin.get_legend_handles_labels()
            handles, labels = handles + h2, labels + l2
        ax.legend(handles, labels, loc="upper left", fontsize=7)
    fig.suptitle(L["curves_suptitle"], fontsize=11)
    fig.align_ylabels(axes)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _read_frame(video: Path, index: int):
    import cv2

    cap = cv2.VideoCapture(str(video))
    try:
        cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = cap.read()
    finally:
        cap.release()
    return img[:, :, ::-1] if ok else None


def _residual(rows: list[dict], frame: int, kicker_id: int, key: str) -> tuple[float, float, int]:
    """Median and maximum residual (px) of one camera on the kicker's keypoints of one frame."""
    vals = []
    for row in rows or []:
        if int(row.get("player_id", -1)) != kicker_id or int(row.get("frame", -1)) != frame:
            continue
        try:
            vals.append(float(row[key]))
        except (KeyError, TypeError, ValueError):
            continue
    if not vals:
        return math.nan, math.nan, 0
    return float(np.median(vals)), float(np.max(vals)), len(vals)


def _overlay_png(ctx: dict, ref_view, entry: dict, path: Path, lang: str) -> Path | None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    st, mda = _stereo()
    L = LABELS[lang]
    contact, kick, support = ctx["contact"], ctx["kick_foot"], ctx["support_foot"]
    cut = st.cut_frame_at(entry["view"], st.capture_time(ref_view, contact))
    if cut is None:
        return None
    cf = int(round(cut))
    dlt = entry["view"].dlt.get(cf)
    img = _read_frame(entry["video"], cf)
    if dlt is None or img is None:
        return None
    proj = mda.dlt_projection_matrix(dlt)

    def px(name: str):
        xyz = _point(ctx["points"], name, contact)
        return None if xyz is None else st.project(proj, xyz)

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.imshow(img)
    ax.set_axis_off()
    found = {name: px(name) for name in JOINTS}
    found = {name: p for name, p in found.items() if p is not None}
    for a, b in STICK_EDGES:
        if a not in found or b not in found:
            continue
        color = GROUP_COLORS[_group(a, b, kick, support)]
        ax.plot([found[a][0], found[b][0]], [found[a][1], found[b][1]], color=color, lw=2.5)
    for p in found.values():
        ax.scatter([p[0]], [p[1]], s=22, color="#ffffff", edgecolor="#000000", zorder=3)
    if found:  # zoom on the kicker: on the whole frame he is only a few pixels tall
        xy = np.array(list(found.values()))
        h, w = img.shape[:2]
        pad = max(60.0, 0.8 * float(np.max(np.ptp(xy, axis=0))))
        ax.set_xlim(max(0.0, xy[:, 0].min() - pad), min(float(w), xy[:, 0].max() + pad))
        ax.set_ylim(min(float(h), xy[:, 1].max() + pad), max(0.0, xy[:, 1].min() - pad))
    cam = L["cam_ref"] if entry["name"] == "ref" else L["cam_other"]
    med, worst, n = _residual(ctx["rows"], contact, ctx["kicker_id"], f"resid_{entry['name']}_px")
    note = L["residual"].format(med=med, worst=worst, n=n) if n else L["no_residual"]
    ax.set_title(f"{L['overlay_title'].format(cam=cam)} ({L['contact']}, {cf})\n{note}", fontsize=9)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)
    return path


def _interactive_html(ctx: dict, series: list[dict], path: Path, lang: str) -> Path:
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    L = LABELS[lang]
    contact, points, ball = ctx["contact"], ctx["points"], ctx["ball"]
    kick, support = ctx["kick_foot"], ctx["support_foot"]
    frames = [int(r["frame"]) for r in series]
    t = _col(series, "time_rel_s")
    side_dims, top_dims = (0, 2), (0, 1)
    side_box = _limits(points, ball, frames, side_dims) or (-1, 1, -1, 1)
    top_box = _limits(points, ball, frames, top_dims) or (-1, 1, -1, 1)

    def stick(f: int, dims) -> list:
        segs = _segments(points, f, dims, kick, support)
        out = []
        for group, color in GROUP_COLORS.items():
            xs, ys = [], []
            for p, q in segs[group]:
                xs += [p[0], q[0], None]
                ys += [p[1], q[1], None]
            out.append(go.Scatter(x=xs, y=ys, mode="lines", line={"color": color, "width": 4},
                                  showlegend=False))  # fmt: skip
        joints = [j for j in (_point(points, name, f) for name in JOINTS) if j is not None]
        out.append(go.Scatter(x=[j[dims[0]] for j in joints], y=[j[dims[1]] for j in joints],
                              mode="markers", marker={"size": 6, "color": "#222222"},
                              showlegend=False))  # fmt: skip
        bp = _ball_at(ball, f)
        out.append(go.Scatter(x=[] if bp is None else [bp[dims[0]]],
                              y=[] if bp is None else [bp[dims[1]]], mode="markers",
                              marker={"size": 12, "color": "#e67e22"}, showlegend=False))  # fmt: skip
        return out

    fig = make_subplots(
        rows=5,
        cols=2,
        specs=[
            [{}, {}],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
            [{"colspan": 2}, None],
        ],  # fmt: skip
        subplot_titles=(
            L["side_title"],
            L["top_title"],
            L["speeds"],
            L["angles"],
            L["angular"],
            L["share"],
        ),  # fmt: skip
        vertical_spacing=0.05,
    )
    f0 = contact if contact in frames else frames[0]
    dynamic: list[int] = []
    for dims, col in ((side_dims, 1), (top_dims, 2)):
        for trace in stick(f0, dims):
            fig.add_trace(trace, row=1, col=col)
            dynamic.append(len(fig.data) - 1)
    bp_all = [ball[f] for f in frames if 0 <= f < len(ball) and np.isfinite(ball[f]).all()]
    if bp_all:
        arr = np.array(bp_all)
        fig.add_trace(go.Scatter(x=arr[:, 0], y=arr[:, 2], mode="lines", showlegend=False,
                                 line={"color": "#999999", "dash": "dot"}), row=1, col=1)  # fmt: skip
        fig.add_trace(go.Scatter(x=arr[:, 0], y=arr[:, 1], mode="lines", showlegend=False,
                                 line={"color": "#999999", "dash": "dot"}), row=1, col=2)  # fmt: skip

    curve_rows = {
        2: [
            ("foot_speed_mps", L["foot"]),
            ("ball_speed_mps", L["ball_fit"]),
            ("lateral_foot_mps", L["lateral"]),
        ],  # fmt: skip
        3: [
            ("knee_angle_deg", L["knee_angle"]),
            ("hip_angle_deg", L["hip_angle"]),
            ("ankle_angle_deg", L["ankle_angle"]),
            ("pelvis_deg", L["pelvis"]),
            ("plane_deg", L["plane"]),
        ],  # fmt: skip
        4: [
            ("thigh_w_deg_s", L["thigh"]),
            ("shank_w_deg_s", L["shank"]),
            ("foot_w_deg_s", L["foot_seg"]),
            ("knee_ext_deg_s", L["knee_ext"]),
        ],  # fmt: skip
        5: [("knee_share", L["share"])],
    }
    cursors: dict[int, int] = {}
    for row, items in curve_rows.items():
        values = []
        for key, label in items:
            y = _col(series, key)
            values.append(y)
            fig.add_trace(go.Scatter(x=t, y=y, mode="lines", name=label), row=row, col=1)
        stacked = np.concatenate(values) if values else np.array([])
        finite = stacked[np.isfinite(stacked)]
        lo, hi = (float(finite.min()), float(finite.max())) if finite.size else (-1.0, 1.0)
        fig.add_trace(go.Scatter(x=[0.0, 0.0], y=[lo, hi], mode="lines", showlegend=False,
                                 line={"color": "#333333", "dash": "dot"}), row=row, col=1)  # fmt: skip
        cursors[row] = len(fig.data) - 1
        fig.update_yaxes(range=[lo, hi], row=row, col=1)

    dynamic_all = dynamic + list(cursors.values())
    frame_objs = []
    for i, f in enumerate(frames):
        data = stick(f, side_dims) + stick(f, top_dims)
        for idx in cursors.values():
            data.append(go.Scatter(x=[t[i], t[i]], y=[fig.data[idx].y[0], fig.data[idx].y[1]],
                                   mode="lines"))  # fmt: skip
        frame_objs.append(go.Frame(name=str(i), data=data, traces=dynamic_all))
    fig.frames = frame_objs

    steps = [
        {
            "method": "animate",
            "label": f"{tt * 1000:+.0f}",
            "args": [[str(i)], {"frame": {"duration": 0, "redraw": True}, "mode": "immediate"}],
        }
        for i, tt in enumerate(t)
    ]
    fig.update_layout(
        title=L["interactive_title"],
        template="plotly_white",
        height=1500,
        sliders=[
            {
                "active": frames.index(f0) if f0 in frames else 0,
                "steps": steps,
                "currentvalue": {"prefix": L["time_ms"] + ": "},
                "pad": {"t": 40},
            }
        ],  # fmt: skip
        updatemenus=[
            {
                "type": "buttons",
                "showactive": False,
                "x": 0,
                "y": 1.12,
                "xanchor": "left",
                "yanchor": "top",
                "buttons": [
                    {
                        "label": "▶",
                        "method": "animate",
                        "args": [
                            None,
                            {"frame": {"duration": 120, "redraw": True}, "fromcurrent": True},
                        ],
                    }
                ],
            }
        ],  # fmt: skip
    )
    fig.update_xaxes(range=[side_box[0], side_box[1]], row=1, col=1)
    fig.update_yaxes(range=[side_box[2], side_box[3]], scaleanchor="x", row=1, col=1)
    fig.update_xaxes(range=[top_box[0], top_box[1]], row=1, col=2)
    fig.update_yaxes(range=[top_box[2], top_box[3]], scaleanchor="x2", row=1, col=2)
    fig.update_xaxes(title_text=L["side_xlabel"], row=1, col=1)
    fig.update_yaxes(title_text=L["side_ylabel"], row=1, col=1)
    fig.update_xaxes(title_text=L["top_xlabel"], row=1, col=2)
    fig.update_yaxes(title_text=L["top_ylabel"], row=1, col=2)
    fig.update_xaxes(title_text=L["time_s"], row=5, col=1)
    fig.write_html(str(path), include_plotlyjs=True, full_html=True)
    return path


def write_figures(
    result: dict, out_dir: Path | str, *, views: list[dict] | None = None
) -> dict[str, Path]:
    """Every figure of one kicker. ``views`` (reference camera first) adds the overlays.

    Each view is ``{"name": "ref" | "other", "view": CamView, "video": Path}``. Returns the paths by
    key; ``figure`` is the Portuguese curves page (the one the metrics file has always pointed to).
    """
    ctx = result["context"]
    series = result["series"]
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths: dict[str, Path] = {}
    for lang in LANGS:
        suffix = f"_{lang}"
        paths[f"sticks_side{suffix}"] = _sticks_png(
            ctx, out / f"kicker_sticks_side{suffix}.png", lang, (0, 2), top=False
        )
        paths[f"sticks_top{suffix}"] = _sticks_png(
            ctx, out / f"kicker_sticks_top{suffix}.png", lang, (0, 1), top=True
        )
        paths[f"curves{suffix}"] = _curves_png(
            ctx, series, out / f"kicker_curves{suffix}.png", lang
        )
        paths[f"interactive{suffix}"] = _interactive_html(
            ctx, series, out / f"kicker_interactive{suffix}.html", lang
        )
        if (
            views
        ):  # the first view is the reference camera: the frames of the others are mapped to it
            for entry in views:
                overlay = _overlay_png(
                    ctx, views[0]["view"], entry, out / f"kicker_overlay_{entry['name']}{suffix}.png", lang
                )  # fmt: skip
                if overlay is not None:
                    paths[f"overlay_{entry['name']}{suffix}"] = overlay
    paths["figure"] = paths["curves_pt"]
    return paths
