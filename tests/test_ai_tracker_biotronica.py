"""
Offline bench: AI Track infill from sparse manual anchors vs GT on Biotronica_Lift.

Skips automatically when the local Biotronica video/CSVs are absent.

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

_BIOTRONICA_DIR = Path(
    "/home/preto/data/Biotronica_Lift/videos/20260828_162054_vailacut_20260904_151532"
)
_VIDEO = _BIOTRONICA_DIR / "20260828_162054_frame_404_to_2104.mp4"
_MANUAL = (
    _BIOTRONICA_DIR / "20260828_162054_frame_404_to_2104_markers_ai_and_human_manual.csv"
)
_GT = _BIOTRONICA_DIR / "20260828_162054_frame_404_to_2104_markers.csv"


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
    not (_VIDEO.is_file() and _MANUAL.is_file() and _GT.is_file()),
    reason="Biotronica_Lift sample video/CSVs not present on this machine",
)
def test_biotronica_ai_track_bench_vs_gt() -> None:
    """Run batch AI Track from manual anchors; report error vs GT; mean must beat baseline ~46px."""
    known = _load_marker_csv(_MANUAL)
    gt = _load_marker_csv(_GT)
    assert len(known) >= 50
    assert len(gt) >= 1000

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
        f">> Biotronica AI Track bench: n={len(err_arr)} "
        f"mean={mean_e:.2f}px median={median_e:.2f}px p95={p95_e:.2f}px "
        f"(anchors={len(known)})"
    )

    # Near-anchor frames should stay tight
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

    # Full-trajectory mean must improve vs prior ~46 px AI dump
    assert mean_e < 35.0, f"Expected mean err < 35px, got {mean_e:.2f}"
    assert median_e < 20.0, f"Expected median err < 20px, got {median_e:.2f}"
