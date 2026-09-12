"""
Offline bench: AI Track infill from sparse manual anchors vs GT on JJ_Kabuto.

JJ_Kabuto is the real-world failure clip reported by the user: fast marker
acceleration + motion blur + white->black background transition, where the
pre-Kalman tracker lost the marker and drifted laterally. Unlike Biotronica
(which ships a separate sparse "manual" anchor CSV plus a dense GT CSV),
JJ_Kabuto only has one fully-dense, hand-corrected marker CSV
(`JJ_Kabuto_markers.csv`, 331/331 frames non-NaN for p0). We build sparse
"manual anchors" by uniformly subsampling that dense GT (every 8th frame),
mirroring how a human would sparsely click-correct a clip, then feed those
anchors into `infill_and_smooth` and score the reconstructed trajectory
against the full dense GT -- same methodology as
`test_ai_tracker_biotronica.py`.

Skips automatically when the local JJ_Kabuto video/CSV are absent.

Author: Prof. Dr. Paulo R. P. Santiago
Update Date: 11 September 2026
Version: 0.3.137
"""

from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest

from vaila.tracking.ai_tracker import AITrackerParameters, infill_and_smooth

_JJKABUTO_DIR = Path("/home/preto/data/jjkabuto")
_VIDEO = _JJKABUTO_DIR / "JJ_Kabuto.mp4"
_GT = _JJKABUTO_DIR / "JJ_Kabuto_markers.csv"

# Sparse-anchor sampling stride: every Nth dense-GT frame becomes a "manual click"
# anchor fed to infill_and_smooth; the rest are scored against GT as reconstructed.
_ANCHOR_STRIDE = 8


def _load_marker_csv(path: Path) -> dict[int, tuple[float, float]]:
    df = pd.read_csv(path)
    frame_col = "frame" if "frame" in df.columns else df.columns[0]
    x_col = "p0_x" if "p0_x" in df.columns else df.columns[1]
    y_col = "p0_y" if "p0_y" in df.columns else df.columns[2]
    out: dict[int, tuple[float, float]] = {}
    for _, row in df.iterrows():
        if pd.isna(row[x_col]) or pd.isna(row[y_col]):
            continue
        out[int(row[frame_col])] = (float(row[x_col]), float(row[y_col]))
    return out


@pytest.mark.skipif(
    not (_VIDEO.is_file() and _GT.is_file()),
    reason="JJ_Kabuto sample video/CSV not present on this machine",
)
def test_jjkabuto_ai_track_bench_vs_gt() -> None:
    """Run batch AI Track from sparse anchors; report error vs dense hand-corrected GT."""
    gt = _load_marker_csv(_GT)
    assert len(gt) >= 300

    known = {f: xy for f, xy in gt.items() if f % _ANCHOR_STRIDE == 0}
    assert len(known) >= 30

    cap = cv2.VideoCapture(str(_VIDEO))
    assert cap.isOpened()
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(cap.get(cv2.CAP_PROP_FPS) or 60.0)

    params = AITrackerParameters(
        search_window=(160, 160),
        block_window=(36, 36),
        similarity_threshold=0.45,
        template_update_threshold=0.70,
        spatial_sigma=28.0,
        template_learning_rate=0.10,
        use_mask=True,
        use_deep_features=False,  # CPU-fast for CI / workstation without GPU lock
        tracking_shape="point",
    )

    result = infill_and_smooth(
        cap=cap,
        total_frames=total_frames,
        known_points=known,
        fps=fps,
        parameters=params,
    )
    cap.release()

    traj = result.trajectory
    errs: list[float] = []
    for f, (gx, gy) in gt.items():
        if 0 <= f < len(traj):
            px, py = float(traj[f, 0]), float(traj[f, 1])
            if np.isnan(px) or np.isnan(py):
                continue
            errs.append(float(np.hypot(px - gx, py - gy)))

    assert errs
    err_arr = np.asarray(errs, dtype=np.float64)
    mean_e = float(err_arr.mean())
    median_e = float(np.median(err_arr))
    p95_e = float(np.percentile(err_arr, 95))
    print(
        f">> JJ_Kabuto AI Track bench: n={len(err_arr)} "
        f"mean={mean_e:.2f}px median={median_e:.2f}px p95={p95_e:.2f}px "
        f"(anchors={len(known)})"
    )

    near: list[float] = []
    known_frames = np.array(sorted(known.keys()))
    for f, (gx, gy) in gt.items():
        if 0 <= f < len(traj):
            dist_anchor = float(np.min(np.abs(known_frames - f)))
            if dist_anchor <= 5:
                px, py = float(traj[f, 0]), float(traj[f, 1])
                near.append(float(np.hypot(px - gx, py - gy)))
    if near:
        near_mean = float(np.mean(near))
        print(f">> Near-anchor (≤5f) mean err={near_mean:.2f}px")
        assert near_mean < 8.0

    # No lateral-drift blowup: reconstructed trajectory must track the marker
    # through the acceleration+blur+background-flip segment, not run off to the
    # side. A drifted track produces large p95/mean; gate loosely here (this is
    # the first-ever run -- baseline, not yet frozen against a prior number).
    assert mean_e < 60.0, f"Expected mean err < 60px, got {mean_e:.2f}"
    assert median_e < 30.0, f"Expected median err < 30px, got {median_e:.2f}"
