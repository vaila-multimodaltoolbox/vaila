"""Sparse FreeKiki ``<stem>.dlt3d`` / ``.dlt2d`` (NaN rows) in ``quickmeasure.load_frames_calibration``.

Update Date: 24 November 2026
Version: 0.4.19

The FreeKiki detect run writes ``<stem>.dlt3d`` / ``<stem>.dlt2d`` with one row per video frame but
only the calibrated frames hold numbers (the others are NaN). Loading it as a per-frame calibration
gave ``(nan, nan)`` on every click of a NaN frame. The loader now swaps it for the per-frame
``<stem>_frames.dlt3d`` / ``.dlt2d`` next to the SAM 3 outputs of the same kick folder, or refuses.
Everything is synthetic (one known camera) and written under ``tmp_path``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila import kick_measure as km
from vaila import quickmeasure as qm

W, H = 320, 240
N_FRAMES = 8
STEM = "main_clip"
CENTRE = np.array([0.0, -40.0, 15.0])
GOOD = (1, 2, 5, 6)  # the frames the sparse FreeKiki file calibrated
BAD = tuple(f for f in range(N_FRAMES) if f not in GOOD)
GROUND = np.array([5.0, -4.0, 0.0])


def _camera(frame: int) -> np.ndarray:
    zc = -CENTRE / np.linalg.norm(CENTRE)
    xc = np.cross(zc, [0.0, 0.0, 1.0])
    xc /= np.linalg.norm(xc)
    yc = np.cross(zc, xc)
    R = np.array([xc, yc, zc])
    f = 300.0 + 10.0 * frame
    K = np.array([[f, 0, W / 2], [0, f, H / 2], [0, 0, 1.0]])
    return K @ np.c_[R, -R @ CENTRE]


def _rows3(frames=range(N_FRAMES), nan_frames=()) -> pd.DataFrame:
    rows = []
    for f in frames:
        if f in nan_frames:
            rows.append([f] + [np.nan] * 11)
            continue
        P = _camera(f)
        P3 = P / P[2, 3]
        rows.append([f, *P3[0], *P3[1], *P3[2, :3]])
    return pd.DataFrame(rows, columns=pd.Index(["frame"] + [f"L{i}" for i in range(1, 12)]))


def _rows2(nan_frames=()) -> pd.DataFrame:
    rows = []
    for f in range(N_FRAMES):
        if f in nan_frames:
            rows.append([f] + [np.nan] * 8)
            continue
        Hm = _camera(f)[:, [0, 1, 3]]
        Hm = Hm / Hm[2, 2]
        rows.append([f, *Hm[0], *Hm[1], Hm[2, 0], Hm[2, 1]])
    return pd.DataFrame(rows, columns=pd.Index(["frame"] + [f"L{i}" for i in range(1, 9)]))


def _layout(
    tmp_path: Path, *, with_frames_file: bool = True, kind: str = "dlt3d"
) -> tuple[Path, Path]:
    """``kick_analysis_x/work/<stem>/...``; returns (sparse file, kick folder)."""
    kick = tmp_path / "kick_analysis_x"
    detect = kick / "work" / STEM / f"freekiki_predict_{STEM}_x"
    detect.mkdir(parents=True)
    sam = kick / "work" / STEM / "sam3dinov3" / "processed_sam3dinov3_x" / STEM
    sam.mkdir(parents=True)
    if kind == "dlt3d":
        _rows3(nan_frames=BAD).to_csv(detect / f"{STEM}.dlt3d", index=False)
    else:
        _rows2(nan_frames=BAD).to_csv(detect / f"{STEM}.dlt2d", index=False)
    if with_frames_file:
        _rows3().to_csv(sam / f"{STEM}_frames.dlt3d", index=False)
        _rows2().to_csv(sam / f"{STEM}_frames.dlt2d", index=False)
    return detect / f"{STEM}.{kind}", kick


def test_sparse_dlt3d_is_replaced_by_the_per_frame_file(tmp_path: Path, capsys) -> None:
    sparse, _ = _layout(tmp_path)
    calib = qm.load_frames_calibration(sparse, total_frames=N_FRAMES)
    assert isinstance(calib, qm.QuickMeasureCalibration) and calib.kind == "camera3d"
    assert calib.source.endswith(f"{STEM}_frames.dlt3d")
    assert "processed_sam3dinov3_x" in calib.source
    for f in range(N_FRAMES):  # the former NaN frames now give the right ground point
        u, v = km.project_points(_camera(f), GROUND)[0]
        x, y = calib.pixel_to_real_at(f, float(u), float(v))
        assert np.isfinite([x, y]).all()
        assert (x, y) == pytest.approx((GROUND[0], GROUND[1]), abs=1e-3)
    out = capsys.readouterr().out
    assert f"{STEM}.dlt3d has no camera on {len(BAD)} of {N_FRAMES} frames" in out
    assert "sparse FreeKiki file" in out and "loaded the per-frame calibration" in out


def test_sparse_dlt3d_without_a_per_frame_file_is_refused(tmp_path: Path) -> None:
    sparse, _ = _layout(tmp_path, with_frames_file=False)
    with pytest.raises(qm.QuickMeasureError) as err:
        qm.load_frames_calibration(sparse, total_frames=N_FRAMES)
    msg = str(err.value)
    assert f"{len(BAD)} of {N_FRAMES}" in msg
    assert str(BAD[0]) in msg
    assert "kick-cameras" in msg and "--kick-run" in msg


def test_sparse_dlt3d_with_an_incomplete_per_frame_file_is_refused(tmp_path: Path) -> None:
    sparse, kick = _layout(tmp_path)
    frames = next(kick.glob("work/*/sam3dinov3/*/*/*_frames.dlt3d"))
    _rows3(nan_frames=(3,)).to_csv(frames, index=False)  # a camera is missing there too
    with pytest.raises(qm.QuickMeasureError, match="kick-cameras"):
        qm.load_frames_calibration(sparse, total_frames=N_FRAMES)


def test_sparse_dlt3d_outside_a_kick_folder_is_refused(tmp_path: Path) -> None:
    lone = tmp_path / "elsewhere" / f"{STEM}.dlt3d"
    lone.parent.mkdir()
    _rows3(nan_frames=BAD).to_csv(lone, index=False)
    with pytest.raises(qm.QuickMeasureError, match="kick-cameras"):
        qm.load_frames_calibration(lone, total_frames=N_FRAMES)


def test_complete_dlt3d_loads_unchanged(tmp_path: Path, capsys) -> None:
    complete = tmp_path / "x" / f"{STEM}_frames.dlt3d"
    complete.parent.mkdir()
    _rows3().to_csv(complete, index=False)
    calib = qm.load_frames_calibration(complete, total_frames=N_FRAMES)
    assert isinstance(calib, qm.QuickMeasureCalibration) and calib.kind == "camera3d"
    assert calib.source == str(complete)
    assert "sparse FreeKiki file" not in capsys.readouterr().out


def test_complete_dlt3d_in_a_kick_folder_is_not_substituted(tmp_path: Path) -> None:
    sparse, kick = _layout(tmp_path)
    full = sparse.with_name("full.dlt3d")
    _rows3().to_csv(full, index=False)
    calib = qm.load_frames_calibration(full, total_frames=N_FRAMES)
    assert isinstance(calib, qm.QuickMeasureCalibration)
    assert calib.source == str(full)


def test_sparse_dlt2d_is_replaced_by_the_per_frame_file(tmp_path: Path, capsys) -> None:
    sparse, _ = _layout(tmp_path, kind="dlt2d")
    calibs = qm.load_frames_calibration(sparse, total_frames=N_FRAMES)
    assert isinstance(calibs, list) and len(calibs) == N_FRAMES
    assert all(c.source.endswith(f"{STEM}_frames.dlt2d") for c in calibs)
    for f, c in enumerate(calibs):  # ground homography of that frame: pixel -> pitch metres
        u, v = km.project_points(_camera(f), GROUND)[0]
        x, y = c.pixel_to_real(float(u), float(v))
        assert (x, y) == pytest.approx((GROUND[0], GROUND[1]), abs=1e-3)
    out = capsys.readouterr().out
    assert f"{STEM}.dlt2d has no camera on {len(BAD)} of {N_FRAMES} frames" in out


def test_sparse_dlt2d_without_a_per_frame_file_is_refused(tmp_path: Path) -> None:
    sparse, _ = _layout(tmp_path, with_frames_file=False, kind="dlt2d")
    with pytest.raises(qm.QuickMeasureError) as err:
        qm.load_frames_calibration(sparse, total_frames=N_FRAMES)
    msg = str(err.value)
    assert f"{len(BAD)} of {N_FRAMES}" in msg and "kick-cameras" in msg


def test_complete_dlt2d_loads_unchanged(tmp_path: Path, capsys) -> None:
    complete = tmp_path / "x" / f"{STEM}_frames.dlt2d"
    complete.parent.mkdir()
    _rows2().to_csv(complete, index=False)
    calibs = qm.load_frames_calibration(complete, total_frames=N_FRAMES)
    assert isinstance(calibs, list) and len(calibs) == N_FRAMES
    assert calibs[0].source == str(complete)
    assert "sparse FreeKiki file" not in capsys.readouterr().out
