"""
================================================================================
Script: label_io.py - kiki49_build YOLO-Pose label lines
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
Version: 0.4.7
Created: 06 October 2026
Update Date: 06 October 2026

Ported from mkvis3d ``openbiomech/soccer_field`` (the builder of the kiki49
dataset) so FreeKiki can convert external field datasets by itself.

Description:
    Ultralytics YOLO-Pose label lines and dataset descriptor for kiki49.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .kiki49 import N_KPT, load_kiki49

MIN_VISIBLE = 4


def empty_keypoints() -> np.ndarray:
    """(49, 3) array of pixel x, pixel y, visibility (0 = not labelled)."""
    return np.zeros((N_KPT, 3), dtype=np.float64)


def format_label(kps_px: np.ndarray, width: int, height: int, cls: int = 0) -> str:
    """One YOLO-Pose line: cls cx cy w h + 49 x (x y v), normalized; invisible -> 0 0 0."""
    vis = kps_px[:, 2] > 0
    if int(vis.sum()) < 1:
        raise ValueError("label needs at least one visible keypoint")
    norm = np.zeros((N_KPT, 3))
    norm[vis, 0] = kps_px[vis, 0] / width
    norm[vis, 1] = kps_px[vis, 1] / height
    norm[vis, 2] = 2
    x0, y0 = norm[vis, 0].min(), norm[vis, 1].min()
    x1, y1 = norm[vis, 0].max(), norm[vis, 1].max()
    bw, bh = max(x1 - x0, 1e-3), max(y1 - y0, 1e-3)
    head = [(x0 + x1) / 2, (y0 + y1) / 2, bw, bh]
    body = [f"{v:.6f}" for v in head]
    for x, y, v in norm:
        body += [f"{x:.6f}", f"{y:.6f}", str(int(v))]
    return f"{cls} " + " ".join(body)


def parse_label(line: str, width: int, height: int) -> np.ndarray:
    """Inverse of :func:`format_label` -> (49, 3) pixel keypoints."""
    vals = line.split()
    if len(vals) != 5 + 3 * N_KPT:
        raise ValueError(f"expected {5 + 3 * N_KPT} fields, got {len(vals)}")
    kp = np.array(vals[5:], dtype=np.float64).reshape(N_KPT, 3)
    kp[:, 0] *= width
    kp[:, 1] *= height
    kp[kp[:, 2] <= 0] = 0.0
    return kp


def write_data_yaml(root: Path) -> Path:
    """Ultralytics descriptor; YAML flow values are written as JSON (a YAML subset)."""
    geo = load_kiki49()
    lines = [
        f"path: {json.dumps(str(root.resolve()))}",
        "train: images/train",
        "val: images/val",
        "test: images/test",
        f"kpt_shape: [{N_KPT}, 3]",
        f"flip_idx: {json.dumps(list(geo.flip_idx))}",
        "names:",
        "  0: football_pitch",
        "kpt_names:",
        f"  0: {json.dumps(list(geo.names))}",
    ]
    out = root / "data.yaml"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return out
