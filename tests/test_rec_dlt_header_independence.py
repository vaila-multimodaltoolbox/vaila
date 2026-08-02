"""Regression tests for the 2026-08 DLT/REC fixes:

- rec3d_one_dlt3d.py, rec3d.py, rec2d.py, rec2d_one_dlt2d.py: pixel CSV column
  labels are no longer inspected — only column ORDER matters (frame column,
  then x,y pairs). Proven here with deliberately non-standard headers.
- rec3d.py: the multi-camera pixel correlation bug (each camera's pixel data
  must come from ITS OWN file, matched by frame, not the same row reused for
  every camera) is proven with an exact numeric ground truth, including
  per-frame-varying DLT3D parameters (the "DLT matrix").
- dlt2d.py / dlt3d.py: mismatched frame/point counts must fail gracefully
  (clean error, exit code 0, no output file) instead of crashing.
- rec3d_one_dlt3d.py/rec3d.py/rec2d.py: --fps/--rate now accept fractional Hz
  (e.g. 119.88012001, a real NTSC-derived capture rate) instead of only int,
  needed for an accurate kinematic timeline.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

import pytest

VAILA_DIR = Path(__file__).parent.parent / "vaila"


def _write_csv(path: Path, header: list[str], rows: list[list]) -> None:
    with path.open("w", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(header)
        writer.writerows(rows)


def _run(args: list[str]) -> subprocess.CompletedProcess:
    return subprocess.run(
        [sys.executable, *args],
        capture_output=True,
        text=True,
        cwd=str(VAILA_DIR.parent),
    )


def _read_csv_by_row(path: Path) -> list[dict]:
    with path.open(newline="") as fh:
        return list(csv.DictReader(fh))


# --------------------------------------------------------------------------- #
# rec3d_one_dlt3d.py — header independence (fixed DLT per camera)
# --------------------------------------------------------------------------- #
def test_rec3d_one_dlt3d_header_independent(tmp_path):
    work = tmp_path / "work"
    work.mkdir()

    # Deliberately NON-standard headers: only column order should matter.
    _write_csv(
        work / "camA_pixels.csv",
        ["t", "ax", "ay"],
        [[0, 5, 10], [1, 2, 4]],
    )
    _write_csv(
        work / "camB_pixels.csv",
        ["time", "bx", "by"],
        [[0, 10, 15], [1, 4, 6]],
    )
    # Cam A: u=X, v=Y ; Cam B: u=Y, v=Z (same convention as test_rec3d_multicam_basic)
    _write_csv(
        work / "camA.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    )
    _write_csv(
        work / "camB.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    )

    out_dir = work / "out"
    out_dir.mkdir()

    result = _run(
        [
            "vaila/rec3d_one_dlt3d.py",
            "--dlt3d",
            str(work / "camA.dlt3d"),
            str(work / "camB.dlt3d"),
            "--pixels",
            str(work / "camA_pixels.csv"),
            str(work / "camB_pixels.csv"),
            "--fps",
            "100",
            "--output",
            str(out_dir),
        ]
    )
    assert result.returncode == 0, result.stderr

    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec3d" in f.name]
    assert len(subfolders) == 1
    csv_files = list(subfolders[0].glob("*.csv"))
    assert len(csv_files) == 1

    rows = _read_csv_by_row(csv_files[0])
    by_frame = {int(float(r["frame"])): r for r in rows}
    assert pytest.approx(float(by_frame[0]["p1_x"]), abs=1e-4) == 5
    assert pytest.approx(float(by_frame[0]["p1_y"]), abs=1e-4) == 10
    assert pytest.approx(float(by_frame[0]["p1_z"]), abs=1e-4) == 15
    assert pytest.approx(float(by_frame[1]["p1_x"]), abs=1e-4) == 2
    assert pytest.approx(float(by_frame[1]["p1_y"]), abs=1e-4) == 4
    assert pytest.approx(float(by_frame[1]["p1_z"]), abs=1e-4) == 6


# --------------------------------------------------------------------------- #
# rec3d.py — multi-camera correlation bug fix + per-frame DLT matrix
# --------------------------------------------------------------------------- #
def test_rec3d_multicam_correlation_with_per_frame_dlt(tmp_path):
    """
    Two cameras, two markers, two frames, with DIFFERENT DLT3D parameters per
    frame per camera (a real "DLT matrix", not a single fixed row). Each
    camera's pixel file has DISTINCT, non-interchangeable values, and
    deliberately non-standard headers.

    If pixel data from one camera's file were reused for the other camera (the
    bug this test guards against), or if the wrong frame's DLT row were
    applied, the reconstructed points would NOT match the ground truth used to
    generate the synthetic pixel observations below.
    """
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "cams"
    input_dir.mkdir()

    # Ground truth (frame -> marker -> (X, Y, Z)):
    #   frame 0: p1=(5,10,15)  p2=(1,2,3)
    #   frame 1: p1=(2,4,6)    p2=(7,8,9)
    #
    # Cam A model (frame 0, scale 1): u=X,  v=Y   -> a=[1,0,0,0, 0,1,0,0, 0,0,0]
    # Cam A model (frame 1, scale 2): u=2X, v=2Y  -> a=[2,0,0,0, 0,2,0,0, 0,0,0]
    # Cam B model (frame 0, scale 1): u=Y,  v=Z   -> a=[0,1,0,0, 0,0,1,0, 0,0,0]
    # Cam B model (frame 1, scale 3): u=3Y, v=3Z  -> a=[0,3,0,0, 0,0,3,0, 0,0,0]
    _write_csv(
        input_dir / "camA_pixels.csv",
        ["t", "m1x", "m1y", "m2x", "m2y"],
        [
            [0, 5, 10, 1, 2],  # frame 0, scale 1: (X,Y)=(5,10) and (1,2)
            [1, 4, 8, 14, 16],  # frame 1, scale 2: 2*(2,4)=(4,8), 2*(7,8)=(14,16)
        ],
    )
    _write_csv(
        input_dir / "camB_pixels.csv",
        ["time", "n1x", "n1y", "n2x", "n2y"],
        [
            [0, 10, 15, 2, 3],  # frame 0, scale 1: (Y,Z)=(10,15) and (2,3)
            [1, 12, 18, 24, 27],  # frame 1, scale 3: 3*(4,6)=(12,18), 3*(8,9)=(24,27)
        ],
    )

    dlt_a = work / "camA.dlt3d"
    dlt_b = work / "camB.dlt3d"
    _write_csv(
        dlt_a,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [
            [0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],
            [1, 2, 0, 0, 0, 0, 2, 0, 0, 0, 0, 0],
        ],
    )
    _write_csv(
        dlt_b,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [
            [0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0],
            [1, 0, 3, 0, 0, 0, 0, 3, 0, 0, 0, 0],
        ],
    )

    out_dir = work / "out"
    out_dir.mkdir()

    result = _run(
        [
            "vaila/rec3d.py",
            "--dlt-files",
            str(dlt_a),
            str(dlt_b),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ]
    )
    assert result.returncode == 0, result.stderr

    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert len(subfolders) == 1
    csv_files = list(subfolders[0].glob("*.csv"))
    assert len(csv_files) == 1

    rows = _read_csv_by_row(csv_files[0])
    by_frame = {int(float(r["frame"])): r for r in rows}

    expected = {
        0: {"p1": (5, 10, 15), "p2": (1, 2, 3)},
        1: {"p1": (2, 4, 6), "p2": (7, 8, 9)},
    }
    for frame, markers in expected.items():
        row = by_frame[frame]
        for marker, (x, y, z) in markers.items():
            assert pytest.approx(float(row[f"{marker}_x"]), abs=1e-4) == x
            assert pytest.approx(float(row[f"{marker}_y"]), abs=1e-4) == y
            assert pytest.approx(float(row[f"{marker}_z"]), abs=1e-4) == z


def test_rec3d_rejects_camera_count_mismatch(tmp_path):
    """--input-dir must contain exactly one pixel CSV per --dlt-files camera."""
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "cams"
    input_dir.mkdir()

    _write_csv(input_dir / "only_one_camera.csv", ["frame", "p1_x", "p1_y"], [[0, 1, 2]])
    dlt_a = work / "camA.dlt3d"
    _write_csv(
        dlt_a,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    )
    dlt_b = work / "camB.dlt3d"
    _write_csv(
        dlt_b,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    )
    out_dir = work / "out"
    out_dir.mkdir()

    result = _run(
        [
            "vaila/rec3d.py",
            "--dlt-files",
            str(dlt_a),
            str(dlt_b),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ]
    )
    assert result.returncode == 0, result.stderr
    # No reconstruction subfolder should be created when the camera count
    # doesn't match (only a messagebox.showerror is triggered, headlessly).
    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert subfolders == []


# --------------------------------------------------------------------------- #
# rec2d_one_dlt2d.py — header independence (fixed DLT2D)
# --------------------------------------------------------------------------- #
def test_rec2d_one_dlt2d_header_independent(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "in"
    input_dir.mkdir()

    # DLT2D params from test_rec2d_basic: u=100X+100, v=100Y+100
    dlt_file = work / "calib.dlt2d"
    _write_csv(
        dlt_file,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
        [[0, 100, 0, 100, 0, 100, 100, 0, 0]],
    )
    # Non-standard pixel header: "idx,mx,my" instead of "frame,p1_x,p1_y".
    _write_csv(input_dir / "trial.csv", ["idx", "mx", "my"], [[0, 150, 150]])

    out_dir = work / "out"
    out_dir.mkdir()
    result = _run(
        [
            "vaila/rec2d_one_dlt2d.py",
            "--dlt-file",
            str(dlt_file),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
        ]
    )
    assert result.returncode == 0, result.stderr

    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec2d" in f.name]
    assert len(subfolders) == 1
    csv_files = list(subfolders[0].glob("*.csv"))
    assert len(csv_files) == 1
    rows = _read_csv_by_row(csv_files[0])
    assert "Frame" in rows[0]  # first column always normalized to "Frame"
    assert pytest.approx(float(rows[0]["mx"]), abs=1e-4) == 0.5
    assert pytest.approx(float(rows[0]["my"]), abs=1e-4) == 0.5


# --------------------------------------------------------------------------- #
# rec2d.py — header independence + per-frame DLT2D matrix
# --------------------------------------------------------------------------- #
def test_rec2d_header_independent_per_frame(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "in"
    input_dir.mkdir()

    dlt_file = work / "calib.dlt2d"
    _write_csv(
        dlt_file,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
        [
            [0, 100, 0, 100, 0, 100, 100, 0, 0],
            [1, 100, 0, 100, 0, 100, 100, 0, 0],
        ],
    )
    _write_csv(
        input_dir / "trial.csv",
        ["t", "px", "py"],
        [[0, 150, 150], [1, 200, 200]],
    )

    out_dir = work / "out"
    out_dir.mkdir()
    result = _run(
        [
            "vaila/rec2d.py",
            "--dlt-file",
            str(dlt_file),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ]
    )
    assert result.returncode == 0, result.stderr

    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec2d" in f.name]
    assert len(subfolders) == 1
    csv_files = list(subfolders[0].glob("*.csv"))
    assert len(csv_files) == 1
    rows = _read_csv_by_row(csv_files[0])
    by_frame = {int(float(r["Frame"])): r for r in rows}
    assert pytest.approx(float(by_frame[0]["px"]), abs=1e-4) == 0.5
    assert pytest.approx(float(by_frame[0]["py"]), abs=1e-4) == 0.5
    assert pytest.approx(float(by_frame[1]["px"]), abs=1e-4) == 1.0
    assert pytest.approx(float(by_frame[1]["py"]), abs=1e-4) == 1.0


# --------------------------------------------------------------------------- #
# dlt2d.py — mismatched frame counts must fail gracefully
# --------------------------------------------------------------------------- #
def test_dlt2d_frame_count_mismatch_no_crash(tmp_path):
    work = tmp_path / "work"
    work.mkdir()

    pixel_file = work / "pixels.csv"
    _write_csv(
        pixel_file,
        ["frame", "p1_x", "p1_y"],
        [[0, 100, 100], [1, 101, 101], [2, 102, 102]],
    )
    real_file = work / "real.ref2d"
    # Deliberately 2 rows (not 1, not matching pixel's 3 rows).
    _write_csv(real_file, ["frame", "p1_x", "p1_y"], [[0, 0, 0], [1, 1, 1]])

    result = _run(
        [
            "vaila/dlt2d.py",
            "--pixel",
            str(pixel_file),
            "--real",
            str(real_file),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert not (work / "pixels.dlt2d").exists()


# --------------------------------------------------------------------------- #
# dlt3d.py — mismatched point counts must fail gracefully (no raw KeyError)
# --------------------------------------------------------------------------- #
def test_dlt3d_point_count_mismatch_no_crash(tmp_path):
    work = tmp_path / "work"
    work.mkdir()

    # Pixel file tracks 6 markers (dlt3d's own minimum), but the REF3D
    # calibration file only defines 3 of them -> common_points has only 3,
    # below the 6-point minimum for an 11-parameter DLT3D solve.
    pixel_header = ["frame"]
    for i in range(1, 7):
        pixel_header.extend([f"p{i}_x", f"p{i}_y"])
    pixel_row = [0] + list(range(100, 100 + 12))
    pixel_file = work / "pixels.csv"
    _write_csv(pixel_file, pixel_header, [pixel_row])

    ref_header = ["frame"]
    for i in range(1, 4):
        ref_header.extend([f"p{i}_x", f"p{i}_y", f"p{i}_z"])
    ref_row = [0] + list(range(1, 10))
    ref_file = work / "real.ref3d"
    _write_csv(ref_file, ref_header, [ref_row])

    result = _run(
        [
            "vaila/dlt3d.py",
            "--pixel",
            str(pixel_file),
            "--real",
            str(ref_file),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert not (work / "pixels.dlt3d").exists()


# --------------------------------------------------------------------------- #
# --fps / --rate must accept fractional Hz (e.g. real NTSC-derived rates)
# --------------------------------------------------------------------------- #
FRACTIONAL_FPS = 119.88012001


def test_rec3d_one_dlt3d_accepts_fractional_fps(tmp_path):
    import ezc3d

    work = tmp_path / "work"
    work.mkdir()

    _write_csv(work / "camA_pixels.csv", ["frame", "p1_x", "p1_y"], [[0, 5, 10]])
    _write_csv(work / "camB_pixels.csv", ["frame", "p1_x", "p1_y"], [[0, 10, 15]])
    _write_csv(
        work / "camA.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    )
    _write_csv(
        work / "camB.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    )
    out_dir = work / "out"
    out_dir.mkdir()

    result = _run(
        [
            "vaila/rec3d_one_dlt3d.py",
            "--dlt3d",
            str(work / "camA.dlt3d"),
            str(work / "camB.dlt3d"),
            "--pixels",
            str(work / "camA_pixels.csv"),
            str(work / "camB_pixels.csv"),
            "--fps",
            str(FRACTIONAL_FPS),
            "--output",
            str(out_dir),
        ]
    )
    assert result.returncode == 0, result.stderr

    subfolder = next(f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec3d" in f.name)

    bvh_file = next(subfolder.glob("*.bvh"))
    frame_time_line = next(
        line for line in bvh_file.read_text().splitlines() if line.startswith("Frame Time:")
    )
    got_frame_time = float(frame_time_line.split(":")[1])
    assert got_frame_time == pytest.approx(1.0 / FRACTIONAL_FPS, abs=1e-6)

    c3d_file = next(subfolder.glob("*_m.c3d"))
    c3d = ezc3d.c3d(str(c3d_file))
    got_rate = c3d["parameters"]["POINT"]["RATE"]["value"][0]
    assert got_rate == pytest.approx(FRACTIONAL_FPS, abs=1e-3)


def test_rec3d_accepts_fractional_rate(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "cams"
    input_dir.mkdir()

    _write_csv(input_dir / "camA_pixels.csv", ["frame", "p1_x", "p1_y"], [[0, 5, 10]])
    _write_csv(input_dir / "camB_pixels.csv", ["frame", "p1_x", "p1_y"], [[0, 10, 15]])
    _write_csv(
        work / "camA.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0]],
    )
    _write_csv(
        work / "camB.dlt3d",
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8", "L9", "L10", "L11"],
        [[0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0]],
    )
    out_dir = work / "out"
    out_dir.mkdir()

    result = _run(
        [
            "vaila/rec3d.py",
            "--dlt-files",
            str(work / "camA.dlt3d"),
            str(work / "camB.dlt3d"),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            str(FRACTIONAL_FPS),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert f"Data rate used: {FRACTIONAL_FPS} Hz" in result.stdout


def test_rec2d_accepts_fractional_rate(tmp_path):
    work = tmp_path / "work"
    work.mkdir()
    input_dir = work / "in"
    input_dir.mkdir()

    dlt_file = work / "calib.dlt2d"
    _write_csv(
        dlt_file,
        ["frame", "L1", "L2", "L3", "L4", "L5", "L6", "L7", "L8"],
        [[0, 100, 0, 100, 0, 100, 100, 0, 0]],
    )
    _write_csv(input_dir / "trial.csv", ["frame", "p1_x", "p1_y"], [[0, 150, 150]])

    out_dir = work / "out"
    out_dir.mkdir()
    result = _run(
        [
            "vaila/rec2d.py",
            "--dlt-file",
            str(dlt_file),
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            str(FRACTIONAL_FPS),
        ]
    )
    assert result.returncode == 0, result.stderr
    assert f"Data rate used: {FRACTIONAL_FPS} Hz" in result.stdout
