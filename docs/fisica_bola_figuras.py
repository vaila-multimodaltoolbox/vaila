"""
================================================================================
Script: fisica_bola_figuras.py - figures of the ball-flight physics document
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.11
Created: 26 October 2026
Update Date: 26 October 2026

Description:
    Draws the figures for docs/fisica_bola_freekiki_{en,pt}.html from the code itself and
    from a 3D folder written by ``freekiki.py ball-rebuild --ball-anchors``. Nothing is
    refitted: the figures read ``ball_flight.json`` (the fit), ``ball_3d.csv`` (the raw
    ground points) and ``ball_3d_on_field.csv`` (the fitted track). Index:
    ``docs/fisica_bola_freekiki.html``.

    Usage: python docs/fisica_bola_figuras.py <3D folder>  (writes docs/images/*.png)
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

try:
    from vaila.freekiki_physics import BRAZUCA_CD, drag_coefficient
except ImportError:  # run as a script from the repository
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "vaila"))
    from freekiki_physics import BRAZUCA_CD, drag_coefficient  # ty: ignore[unresolved-import]

HERE = Path(__file__).resolve().parent
IMAGES = HERE / "images"


def fig_drag() -> Path:
    v = np.linspace(5.0, 40.0, 300)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(v, [drag_coefficient(x, 0.0) for x in v], label="Sp = 0 (Brazuca, Kiratidis Eq. 5)")
    ax.plot(v, [drag_coefficient(x, 0.15) for x in v], "--", label="Sp = 0.15 (spin term, Eq. 6)")
    ax.axvline(BRAZUCA_CD["v_max"], color="grey", ls=":", label="end of the fit (35 m/s)")
    ax.axvspan(12.0, 15.0, color="orange", alpha=0.15, label="drag crisis (near 13-15 m/s)")
    ax.set(xlabel="speed v (m/s)", ylabel="drag coefficient Cd", ylim=(0, 0.6))
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = IMAGES / "fisica_cd_vs_velocidade.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def _read(path: Path, cols: list[str]) -> dict[str, np.ndarray]:
    with path.open(newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    return {c: np.array([float(r[c]) for r in rows if r[c] != ""]) for c in cols}


def fig_track(folder: Path) -> Path:
    raw = _read(folder / "ball_3d.csv", ["frame", "x_m", "y_m"])
    fit = _read(folder / "ball_3d_on_field.csv", ["frame", "x_m", "y_m", "z_m"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(
        raw["x_m"], raw["y_m"], "o", ms=4, color="grey", label="ground projection of each pixel"
    )
    ax.plot(fit["x_m"], fit["y_m"], "-", color="C0", label="fitted flight (top view)")
    for f in (0, 9, 30):
        i = int(np.flatnonzero(fit["frame"] == f)[0])
        ax.annotate(
            f"frame {f}",
            (fit["x_m"][i], fit["y_m"][i]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=8,
        )
    ax.set(xlabel="x (m, towards the right goal)", ylabel="y (m)", aspect="equal")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    out = IMAGES / "fisica_trajetoria_topo.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_height_speed(folder: Path, flight: dict) -> Path:
    fit = _read(folder / "ball_3d_on_field.csv", ["frame", "z_m"])
    t = fit["frame"] / 29.97
    speed = np.array(flight["speed_mps"])
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(7, 5.5), sharex=True)
    a1.plot(t, fit["z_m"], "-o", ms=3, color="C0")
    a1.axhline(1.8, color="C3", ls="--", label="barrier passage: 1.8 m at frame 9")
    a1.set(ylabel="height z (m)")
    a1.legend(fontsize=8)
    a1.grid(alpha=0.3)
    a2.plot(t, speed, "-o", ms=3, color="C2")
    a2.axhline(BRAZUCA_CD["v_max"], color="grey", ls=":", label="end of the Cd fit (35 m/s)")
    a2.set(xlabel="time (s)", ylabel="speed (m/s)")
    a2.legend(fontsize=8)
    a2.grid(alpha=0.3)
    out = IMAGES / "fisica_altura_velocidade.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def fig_residuals(flight: dict) -> Path:
    frames = sorted(int(f) for f in flight["per_frame_px"])
    err = [flight["per_frame_px"][str(f)] for f in frames]
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.bar(frames, err, color="C4")
    ax.axhline(flight["rms_px"], color="k", ls="--", label=f"rms {flight['rms_px']} px")
    ax.set(xlabel="frame", ylabel="reprojection error (px)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, axis="y")
    out = IMAGES / "fisica_residuos_pixels.png"
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def main(argv: list[str]) -> int:
    if len(argv) != 2:
        print(__doc__)
        return 2
    folder = Path(argv[1])
    IMAGES.mkdir(parents=True, exist_ok=True)
    flight = json.loads((folder / "ball_flight.json").read_text(encoding="utf-8"))
    for path in (
        fig_drag(),
        fig_track(folder),
        fig_height_speed(folder, flight),
        fig_residuals(flight),
    ):
        print("wrote", path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
