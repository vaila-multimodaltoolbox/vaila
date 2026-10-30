"""
================================================================================
Script: freekiki_report_video.py - slow-motion didactic animation of a finished kick
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
    The broadcast video of the kick, every frame slowed down about five times, with the
    reconstruction drawn over it: the fitted flight (solid up to now, dashed after), its
    shadow on the grass, the corrected marker, the physical anchors, and the three forces on
    the ball (gravity, drag against the motion, Magnus along omega x v) with their values.
    A panel shows time, speed, Reynolds number, Cd, spin and height; a banner names the
    event; a small top view of the pitch sits in the corner. The texts follow ``lang``
    (``pt`` or ``en``), from :mod:`vaila.freekiki_report_i18n`.

    Each frame of the broadcast is drawn with matplotlib and repeated; the key moments
    (the kick, the wall, the end) are held longer. The crop is computed once from every
    projected point, so the ball and the players stay in the picture all the time.

    A penalty kick has no wall: the banner reads preparation / kick / ball in flight / last frame,
    the wall is not held, and the anchors are the ones of ``m["anchors"]`` (the keeper's hands at
    the kick and in the last frame instead of the wall).
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

try:
    from .freekiki_detect3d import _project_point
    from .freekiki_report_i18n import fmt_num, localize_anchor_label, tr
    from .freekiki_report_players import draw_goal_area, draw_hands, draw_people, people_extent
except ImportError:
    from freekiki_detect3d import _project_point  # ty: ignore[unresolved-import]
    from freekiki_report_i18n import (  # ty: ignore[unresolved-import]
        fmt_num,
        localize_anchor_label,
        tr,
    )
    from freekiki_report_players import (  # ty: ignore[unresolved-import]
        draw_goal_area,
        draw_hands,
        draw_people,
        people_extent,
    )

OUT_FPS = 30
SLOW = 5  # every broadcast frame lasts SLOW output frames (about 5x slow motion at 29.97 fps)
HOLD_S = {"kick": 1.0, "wall": 2.0, "end": 2.5}  # seconds held on the key moments
MARGIN_PX = 220


def _crop_box(
    run: Any, anchors_px: list[tuple[float, float]], width: int, height: int
) -> tuple[int, int, int, int]:
    """A 16:9 window that contains every projected point of the run, clamped to the image."""
    pts = [np.asarray(p, dtype=float) for p in anchors_px]
    for f in np.flatnonzero(np.isfinite(run.track).all(axis=1)):
        P = run.projections.get(int(f))
        if P is None:
            continue
        uv = _project_point(P, run.track[f])
        if uv is not None:
            pts.append(np.asarray(uv))
    arr = np.array(pts, dtype=float)
    x0, y0 = arr.min(axis=0) - MARGIN_PX
    x1, y1 = arr.max(axis=0) + MARGIN_PX
    w, h = x1 - x0, y1 - y0
    if w / h < 16 / 9:
        w = h * 16 / 9
    else:
        h = w * 9 / 16
    cx, cy = (x0 + x1) / 2, (y0 + y1) / 2
    w, h = min(w, width), min(h, height)
    x0 = min(max(cx - w / 2, 0), width - w)
    y0 = min(max(cy - h / 2, 0), height - h)
    return int(x0) // 2 * 2, int(y0) // 2 * 2, int(w) // 2 * 2, int(h) // 2 * 2


def _event(f: int, m: dict, lang: str) -> tuple[str, str]:
    kf, bf, ef = m["kick_frame"], m["barrier_f"], m["end_frame"]
    if f < kf:
        return tr(lang, "vid_prep"), "#333333"
    if f == kf:
        return tr(lang, "vid_kick"), "#1664a0"
    if m.get("penalty"):  # no wall: the ball flies until the last frame
        if f < ef:
            return tr(lang, "vid_flight"), "#5b6577"
        return tr(lang, "vid_last"), "#d62246"
    if f < bf:
        return tr(lang, "vid_rising"), "#5b6577"
    if f < ef:
        return tr(lang, "vid_wall"), "#14934b"
    return tr(lang, "vid_goal"), "#d62246"


def overlay_geometry(run: Any, m: dict, f: int) -> dict:
    """The overlays of frame ``f`` in pixels of its own image: every 3D point (the fitted flight,
    its shadow on the grass, the anchors, the ball) is projected with the camera of frame ``f``.

    The grass moves with the camera, so a point taken with the camera of another frame lands on
    the wrong spot of this image. Returns ``{"trail", "rest", "shadow", "anchors", "ball"}``;
    ``trail`` is the path up to ``f``, ``rest`` the path after it.
    """
    kf, ef = m["kick_frame"], m["end_frame"]
    track = run.track
    P = run.projections.get(f)

    def proj(xyz):
        if P is None or not np.isfinite(np.asarray(xyz, dtype=float)).all():
            return None
        uv = _project_point(P, xyz)
        return None if uv is None else (float(uv[0]), float(uv[1]))

    trail, rest, shadow = [], [], []
    for k in range(kf, ef + 1):
        p = proj(track[k])
        if p is not None:
            (trail if k <= f else rest).append(p)
        if k <= f and np.isfinite(track[k, :2]).all():
            s = proj([track[k, 0], track[k, 1], 0.0])
            if s is not None:
                shadow.append(s)
    anchors = {
        key: proj(spec["xyz"]) if f >= int(spec.get("show_from", 0)) else None
        for key, spec in m.get("anchors", run.anchors).items()
        if isinstance(spec, dict) and "xyz" in spec
    }
    return {"trail": trail, "rest": rest, "shadow": shadow, "anchors": anchors,
            "ball": proj(track[f])}  # fmt: skip


def _render(
    run: Any,
    m: dict,
    img_rgb: np.ndarray,
    f: int,
    box: tuple[int, int, int, int],
    title: str,
    lang: str,
    people: Any = None,
) -> np.ndarray:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    x0, y0, w, h = box
    fig = plt.figure(figsize=(w / 100, h / 100), dpi=100)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.imshow(img_rgb)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    def pix(P, xyz):
        uv = _project_point(P, xyz) if P is not None else None
        return None if uv is None else (uv[0] - x0, uv[1] - y0)

    track = run.track
    P = run.projections.get(f)
    geo = overlay_geometry(run, m, f)

    def shifted(p):
        return None if p is None else (p[0] - x0, p[1] - y0)

    # Fitted flight: solid up to the current frame, dashed after it, shadow on the grass.
    trail = [shifted(p) for p in geo["trail"]]
    rest = [shifted(p) for p in geo["rest"]]
    if len(trail) > 1:
        ax.plot(*np.array(trail).T, color="#d62246", lw=3.2, zorder=6)
    if len(rest) > 1:
        ax.plot(*np.array(rest).T, color="#d62246", lw=2.2, ls=(0, (2, 1.6)), alpha=0.8, zorder=5)
    shadow = [shifted(p) for p in geo["shadow"]]
    if len(shadow) > 1:
        ax.plot(*np.array(shadow).T, color="black", lw=2, alpha=0.35, zorder=3)

    # Corrected marker of this frame and the fitted ball on it.
    if f in run.pixels:
        u, v = run.pixels[f]
        ax.plot(u - x0, v - y0, "o", ms=18, mfc="none", mec="#ffe066", mew=2.2, zorder=7)
    if np.isfinite(track[f]).all():
        ball = shifted(geo["ball"])
        if ball is not None:
            ax.plot(*ball, "o", ms=9, color="#d62246", mec="white", mew=1.2, zorder=9)
            _forces(ax, run, m, f, pix, track[f], P, lang)

    # Physical anchors, drawn at their own frames: the legacy names keep their colours and offsets,
    # any other anchor point gets its own label and a colour from the palette.
    legacy = {
        "kick_foot": (tr(lang, "vid_anchor_kicker"), "#00e5ff", 14, -18),
        "barrier_shoulder": (tr(lang, "vid_anchor_shoulder"), "#ff4fd8", -120, -30),
        "barrier_target": (tr(lang, "vid_anchor_wall"), "#ff9f1a", 14, -26),
        "keeper_hands": (tr(lang, "vid_anchor_keeper"), "#3cff6b", 14, 18),
        "keeper_hands_start": (tr(lang, "vid_anchor_hands_start"), "#ffe066", 14, -18),
    }
    palette = ["#ffe066", "#ff4fd8", "#ff9f1a", "#3cff6b", "#ffffff"]
    for j, (key, spec) in enumerate(m.get("anchors", run.anchors).items()):
        if not isinstance(spec, dict) or "xyz" not in spec:
            continue
        label, color, dx, dy = legacy.get(
            key,
            (
                localize_anchor_label(lang, str(spec.get("label", key))),
                palette[j % len(palette)],
                14,
                -18,
            ),
        )
        if spec.get("i18n"):
            label = tr(lang, spec["i18n"])
        if spec.get("label_offset"):
            dx, dy = spec["label_offset"]
        p = shifted(geo["anchors"].get(key))
        if p is None:
            continue
        ax.plot(*p, "+", ms=20, mew=2.4, color=color, zorder=8)
        ax.text(p[0] + dx, p[1] + dy, label, color=color, fontsize=10, fontweight="bold", zorder=9,
                bbox={"boxstyle": "round,pad=0.25", "fc": "black", "alpha": 0.45, "ec": "none"})  # fmt: skip

    _panel(ax, run, m, f, w, h, lang)
    event, col = _event(f, m, lang)
    ax.text(14, 14, event, color="white", fontsize=15, fontweight="bold", va="top", zorder=12,
            bbox={"boxstyle": "round,pad=0.45", "fc": col, "alpha": 0.92, "ec": "white", "lw": 1.2})  # fmt: skip
    caption = tr(lang, "vid_caption", title=title, slow=SLOW)
    ax.text(14, h - 12, caption, color="white", fontsize=9, va="bottom", zorder=12,
            bbox={"boxstyle": "round,pad=0.25", "fc": "black", "alpha": 0.5, "ec": "none"})  # fmt: skip
    _minimap(fig, run, m, f, lang, people)

    fig.canvas.draw()
    frame = np.asarray(fig.canvas.buffer_rgba())[:, :, :3].copy()  # ty: ignore[unresolved-attribute]
    plt.close(fig)
    return frame


def _velocity_dir(track: np.ndarray, f: int, fps: float) -> np.ndarray | None:
    lo, hi = max(f - 1, 0), min(f + 1, len(track) - 1)
    if hi == lo or not np.isfinite(track[[lo, hi]]).all():
        return None
    v = (track[hi] - track[lo]) * fps / (hi - lo)
    n = np.linalg.norm(v)
    return v / n if n > 0 else None


def _screen_arrow(pix, P, pos: np.ndarray, direction: np.ndarray, length_px: float):
    """Base and tip, in pixels, of a 3D direction drawn with a fixed length on the screen."""
    base = pix(P, pos)
    probe = pix(P, pos + direction * 0.02)
    if base is None or probe is None:
        return None
    d = np.array(probe) - np.array(base)
    n = float(np.linalg.norm(d))
    if n < 1e-6:
        return None
    return np.array(base), np.array(base) + d / n * length_px


def _forces(ax, run: Any, m: dict, f: int, pix, pos: np.ndarray, P, lang: str) -> None:
    """Gravity, drag and Magnus on the ball: arrows whose screen length is proportional to the force."""
    if P is None:
        return
    v_hat = _velocity_dir(run.track, f, run.fps)
    if v_hat is None:
        return
    omega = np.asarray(run.flight["theta"][6:9], dtype=float)
    spin_dir = np.cross(omega, v_hat) if np.linalg.norm(omega) > 0 else None
    arrows = [
        (
            tr(lang, "vid_force_gravity"),
            np.array([0.0, 0.0, -1.0]),
            m["a_gravity"],
            "#ffffff",
            (10, 18),
        ),
        (
            tr(lang, "vid_force_drag"),
            -v_hat,
            float(m["a_drag"][f]) if np.isfinite(m["a_drag"][f]) else 0.0,
            "#ff9f1a",
            (-10, -18),
        ),
    ]
    if spin_dir is not None and np.isfinite(m["a_magnus"][f]):
        n = np.linalg.norm(spin_dir)
        if n > 0:
            arrows.append(
                (
                    tr(lang, "vid_force_magnus"),
                    spin_dir / n,
                    float(m["a_magnus"][f]),
                    "#00e5ff",
                    (14, -24),
                )
            )
    for name, direction, magnitude, color, (dx, dy) in arrows:
        if magnitude <= 0:
            continue
        arrow = _screen_arrow(pix, P, pos, direction, min(8.0 * magnitude, 160.0))
        if arrow is None:
            continue
        base, tip = arrow
        ax.annotate("", xy=tuple(tip), xytext=tuple(base),
                    arrowprops={"arrowstyle": "-|>", "color": color, "lw": 2.6}, zorder=10)  # fmt: skip
        text = f"{name} {fmt_num(magnitude, 1, lang)} m/s²"
        ax.text(tip[0] + dx, tip[1] + dy, text, color=color, fontsize=8.5,
                fontweight="bold", zorder=11,
                bbox={"boxstyle": "round,pad=0.2", "fc": "black", "alpha": 0.55, "ec": "none"})  # fmt: skip


def _panel(ax, run: Any, m: dict, f: int, w: int, h: int, lang: str) -> None:
    kf = m["kick_frame"]

    def num(x, nd):
        return fmt_num(x, nd, lang)

    t = (f - kf) / run.fps if f >= kf else None
    speed = m["speed_kmh"][f] if f < len(m["speed_kmh"]) else np.nan
    re = m["re"][f] if f < len(m["re"]) else np.nan
    cd = m["cd"][f] if f < len(m["cd"]) else np.nan
    z = run.track[f, 2] if np.isfinite(run.track[f, 2]) else np.nan
    rows = [
        (tr(lang, "vid_after"), f"{num(t, 2)} s" if t is not None else "—"),
        (
            tr(lang, "vid_speed"),
            f"{num(speed, 0)} km/h" if np.isfinite(speed) and speed > 0 else "—",
        ),
        (tr(lang, "vid_height"), f"{num(z, 2)} m" if np.isfinite(z) else "—"),
        (
            tr(lang, "vid_re"),
            f"{num(re / 1e5, 1)} × 10⁵" if np.isfinite(re) and re > 0 else "—",
        ),
        (tr(lang, "vid_cd"), num(cd, 2) if np.isfinite(cd) else "—"),
        (tr(lang, "vid_spin"), f"{num(m['spin_rps'], 1)} rps"),
    ]
    txt = "\n".join(f"{a}: {b}" for a, b in rows)
    ax.text(w - 14, 14, txt, color="white", fontsize=10.5, fontweight="bold", ha="right", va="top",
            linespacing=1.5, zorder=12, bbox={"boxstyle": "round,pad=0.5", "fc": "black", "alpha": 0.62, "ec": "none"})  # fmt: skip


def _minimap(fig, run: Any, m: dict, f: int, lang: str, people: Any = None) -> None:
    """The top view in the corner: the ball path, the goal area, the kicker, the wall and the
    keeper (with his path), drawn at the same metres as the figure of the report."""
    kf, ef = m["kick_frame"], m["end_frame"]
    xy = run.track[kf : ef + 1, :2]
    if not np.isfinite(xy).all():
        return
    # Bottom-right corner. set_aspect("equal") shrinks the axes; set_anchor("SE") keeps
    # the shrunk box pinned low instead of floating up in the middle of the slot.
    ax = fig.add_axes((0.755, 0.015, 0.23, 0.20))
    ax.set_facecolor("white")
    draw_goal_area(ax, run.track[ef, :2])
    ax.plot(xy[:, 0], xy[:, 1], color="#d62246", lw=2, zorder=3)
    if np.isfinite(run.track[f, :2]).all() and kf <= f <= ef:
        ax.plot(*run.track[f, :2], "o", color="#d62246", mec="white", ms=7, zorder=6)
    ax.plot(*run.track[kf, :2], "o", color="#14213d", ms=5, zorder=6)
    draw_people(ax, people, kick_f=kf, barrier_f=m["barrier_f"], end_f=ef, frame=f,
                lang=lang, labels=False)  # fmt: skip
    hands = [m.get("hands_start"), m.get("hands_end")]
    draw_hands(ax, hands[0], hands[1], lang, labels=False)
    extent = np.vstack(
        [xy, people_extent(people, kf, ef), *[h[:2][None] for h in hands if h is not None]]
    )
    pad = 2.5
    ax.set_xlim(extent[:, 0].min() - pad, extent[:, 0].max() + pad)
    ax.set_ylim(extent[:, 1].min() - pad, extent[:, 1].max() + pad)
    ax.set_aspect("equal")
    ax.set_anchor("SE")
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_title(tr(lang, "vid_minimap"), fontsize=8)


def write_lance_video(
    run: Any, out_path, m: dict, *, title: str = "", lang: str = "pt", people: Any = None
) -> Path:
    """Write the slow-motion MP4 of the kick to ``out_path``, with the texts of ``lang``."""
    import cv2

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cap = cv2.VideoCapture(str(run.video))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    anchor_px = []  # every anchor, in the camera of every frame (the camera moves)
    for spec in m.get("anchors", run.anchors).values():
        for P in run.projections.values():
            uv = _project_point(P, spec["xyz"])
            if uv is not None:
                anchor_px.append(uv)
    box = _crop_box(run, anchor_px + list(run.pixels.values()), width, height)
    x0, y0, w, h = box
    last = min(m["end_frame"], run.n - 1)
    holds = {m["kick_frame"]: int(HOLD_S["kick"] * OUT_FPS)}
    if not m.get("penalty"):  # a penalty has no wall to hold on
        holds[m["barrier_f"]] = int(HOLD_S["wall"] * OUT_FPS)
    holds[last] = int(HOLD_S["end"] * OUT_FPS)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), OUT_FPS, (w, h))  # ty: ignore[unresolved-attribute]
    written = 0
    try:
        for f in range(last + 1):
            ok, bgr = cap.read()
            if not ok:
                break
            img = bgr[y0 : y0 + h, x0 : x0 + w, ::-1].copy()
            frame = _render(run, m, img, f, box, title, lang, people)
            frame = cv2.resize(frame, (w, h)) if frame.shape[:2] != (h, w) else frame
            bgr_out = frame[:, :, ::-1]
            for _ in range(SLOW + holds.get(f, 0)):
                writer.write(bgr_out)
                written += 1
    finally:
        writer.release()
        cap.release()
    if written == 0:
        raise RuntimeError(f"no frame was written to {out_path}")
    return out_path
