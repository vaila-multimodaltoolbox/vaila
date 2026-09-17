import shutil
import subprocess
import sys
from pathlib import Path

import pandas as pd
import pytest


@pytest.fixture
def test_data(tmp_path):
    # Setup paths
    base_dir = Path(__file__).parent.parent
    dlt3d_dir = base_dir / "tests" / "DLT3D_and_Rec3d"
    animal_dir = base_dir / "tests" / "Animal_Open_Field"

    # Create temp workspace
    workspace = tmp_path / "workspace"
    workspace.mkdir()

    # Create lightweight 2D calibration pixels (5 frames) from zone1 or synthetic
    zone1_path = animal_dir / "zone1_20241216_165950.csv"
    if zone1_path.exists():
        df_z1 = pd.read_csv(zone1_path, nrows=5)
        df_z1.to_csv(workspace / "pixels2d.csv", index=False)
    else:
        # Fallback 4 points square across 5 frames
        rows = []
        for f in range(5):
            rows.append(
                {
                    "frame": f,
                    "p1_x": 100.0 + f,
                    "p1_y": 100.0 + f,
                    "p2_x": 200.0 + f,
                    "p2_y": 100.0 + f,
                    "p3_x": 200.0 + f,
                    "p3_y": 200.0 + f,
                    "p4_x": 100.0 + f,
                    "p4_y": 200.0 + f,
                }
            )
        pd.DataFrame(rows).to_csv(workspace / "pixels2d.csv", index=False)

    shutil.copy(animal_dir / "ref_real_openfield.ref2d", workspace / "real2d.ref2d")

    shutil.copy(dlt3d_dir / "pixelcorrds" / "c01_markers_1_line.csv", workspace / "pixels3d.csv")
    shutil.copy(
        dlt3d_dir / "ref3d_realworld" / "ref3d_realworld_format1.ref3d",
        workspace / "real3d.ref3d",
    )

    # DLT files and pixels for multi-camera 3D
    shutil.copy(dlt3d_dir / "dlt3d" / "c01_markers_1_line.dlt3d", workspace / "cam1.dlt3d")
    shutil.copy(dlt3d_dir / "dlt3d" / "c02_markers_1_line.dlt3d", workspace / "cam2.dlt3d")
    shutil.copy(dlt3d_dir / "pixelcorrds" / "c01_markers_1_line.csv", workspace / "cam1_pix.csv")
    shutil.copy(dlt3d_dir / "pixelcorrds" / "c02_markers_1_line.csv", workspace / "cam2_pix.csv")

    return workspace


def test_dlt2d_integration(test_data):
    pixel_file = str(test_data / "pixels2d.csv")
    real_file = str(test_data / "real2d.ref2d")

    # Use subprocess to test CLI parsing
    result = subprocess.run(
        [sys.executable, "vaila/dlt2d.py", "--pixel", pixel_file, "--real", real_file],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    dlt_output = test_data / "pixels2d.dlt2d"
    assert dlt_output.exists()
    dlt_df = pd.read_csv(dlt_output)
    assert len(dlt_df) == 5


def test_dlt3d_integration(test_data):
    pixel_file = str(test_data / "pixels3d.csv")
    real_file = str(test_data / "real3d.ref3d")

    result = subprocess.run(
        [sys.executable, "vaila/dlt3d.py", "--pixel", pixel_file, "--real", real_file],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    dlt_output = test_data / "pixels3d.dlt3d"
    assert dlt_output.exists()


def test_rec2d_integration(test_data):
    # First generate the .dlt2d
    subprocess.run(
        [
            sys.executable,
            "vaila/dlt2d.py",
            "--pixel",
            str(test_data / "pixels2d.csv"),
            "--real",
            str(test_data / "real2d.ref2d"),
        ],
        check=True,
    )

    dlt_file = str(test_data / "pixels2d.dlt2d")
    # Place pixels in a subdir for batch processing
    input_dir = test_data / "input2d"
    input_dir.mkdir()
    shutil.copy(test_data / "pixels2d.csv", input_dir / "data.csv")

    output_dir = test_data / "output2d"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec2d_one_dlt2d.py",
            "--dlt-file",
            dlt_file,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Check if a reconstructed folder was created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert len(subfolders) >= 1
    assert any(f.suffix == ".2d" for f in subfolders[0].iterdir())


def test_rec2d_multiline_and_partial_nan(test_data):
    """Test rec2d.py with multi-line DLT2D and partial marker occlusion."""
    # Generate DLT2D from 5-frame pixel data
    subprocess.run(
        [
            sys.executable,
            "vaila/dlt2d.py",
            "--pixel",
            str(test_data / "pixels2d.csv"),
            "--real",
            str(test_data / "real2d.ref2d"),
        ],
        check=True,
    )
    dlt_file = str(test_data / "pixels2d.dlt2d")

    # Create tracking data where frame 0 has p2 as NaN, but p1 is valid
    track_df = pd.DataFrame(
        {
            "Frame": [0, 1, 2],
            "p1_x": [0.017, 0.017, 0.017],
            "p1_y": [0.524, 0.524, 0.524],
            "p2_x": [float("nan"), 0.030, 0.030],
            "p2_y": [float("nan"), 0.536, 0.536],
        }
    )
    in_dir = test_data / "in_multiline_2d"
    in_dir.mkdir()
    track_df.to_csv(in_dir / "track.csv", index=False)

    out_dir = test_data / "out_multiline_2d"
    out_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec2d.py",
            "--dlt-file",
            dlt_file,
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0

    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec2d" in f.name]
    assert len(subfolders) == 1
    csv_file = next(f for f in subfolders[0].iterdir() if f.suffix == ".csv")
    res_df = pd.read_csv(csv_file)

    # Frame 0: p1 should NOT be NaN even though p2 was NaN
    assert not pd.isna(res_df.loc[0, "p1_x"])
    assert not pd.isna(res_df.loc[0, "p1_y"])
    # Frame 0: p2 SHOULD be NaN
    assert pd.isna(res_df.loc[0, "p2_x"])
    assert pd.isna(res_df.loc[0, "p2_y"])


def test_rec3d_integration(test_data):
    dlt_file = str(test_data / "cam1.dlt3d")
    input_dir = test_data / "input3d"
    input_dir.mkdir()
    shutil.copy(test_data / "pixels3d.csv", input_dir / "data.csv")

    output_dir = test_data / "output3d"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec3d.py",
            "--dlt-files",
            dlt_file,
            "--input-dir",
            str(input_dir),
            "--output-dir",
            str(output_dir),
            "--rate",
            "100",
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Check if a reconstructed folder was created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec" in f.name]
    assert len(subfolders) >= 1
    csv_file = next(f for f in subfolders[0].iterdir() if f.suffix == ".csv")
    first_row = csv_file.read_text(encoding="utf-8").splitlines()[1]
    assert first_row.startswith("0,") or first_row.startswith("1,")


def test_rec3d_one_dlt3d_integration(test_data):
    # rec3d_one_dlt3d.py uses --dlt3d, --pixels, --fps, --output
    dlt1 = str(test_data / "cam1.dlt3d")
    dlt2 = str(test_data / "cam2.dlt3d")
    pix1 = str(test_data / "cam1_pix.csv")
    pix2 = str(test_data / "cam2_pix.csv")

    output_dir = test_data / "output3d_one"
    output_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec3d_one_dlt3d.py",
            "--dlt3d",
            dlt1,
            dlt2,
            "--pixels",
            pix1,
            pix2,
            "--fps",
            "100",
            "--output",
            str(output_dir),
        ],
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0
    # Check if reconstruction files were created
    subfolders = [f for f in output_dir.iterdir() if f.is_dir() and "vaila_rec3d" in f.name]
    assert len(subfolders) == 1
    # It creates .csv and .3d files
    files = list(subfolders[0].iterdir())
    assert any(f.suffix == ".csv" for f in files)
    assert any(f.suffix == ".3d" for f in files)
    csv_file = next(f for f in files if f.suffix == ".csv" and not f.name.endswith("_labels.csv"))
    first_row = csv_file.read_text(encoding="utf-8").splitlines()[1]
    assert first_row.startswith("0,") or first_row.startswith("1,")


def test_rec2d_single_row_dlt_fallback(test_data):
    """Ensure rec2d.py applies a single-row DLT to all frames in multi-frame pixel CSV."""
    dlt_single = pd.DataFrame(
        {
            "frame": [0],
            "dlt_param_1": [100.0],
            "dlt_param_2": [0.0],
            "dlt_param_3": [100.0],
            "dlt_param_4": [0.0],
            "dlt_param_5": [100.0],
            "dlt_param_6": [100.0],
            "dlt_param_7": [0.0],
            "dlt_param_8": [0.0],
        }
    )
    dlt_file = test_data / "single_row.dlt2d"
    dlt_single.to_csv(dlt_file, index=False)

    pixel_multi = pd.DataFrame(
        {
            "Frame": [0, 1, 2, 3],
            "p1_x": [150.0, 150.0, 150.0, 150.0],
            "p1_y": [150.0, 150.0, 150.0, 150.0],
        }
    )
    in_dir = test_data / "in_single_fallback_2d"
    in_dir.mkdir()
    pixel_multi.to_csv(in_dir / "data.csv", index=False)

    out_dir = test_data / "out_single_fallback_2d"
    out_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec2d.py",
            "--dlt-file",
            str(dlt_file),
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec2d" in f.name]
    assert len(subfolders) == 1
    res_df = pd.read_csv(next(f for f in subfolders[0].iterdir() if f.suffix == ".csv"))
    # All 4 frames must be reconstructed to 0.5, 0.5 without NaNs
    assert len(res_df) == 4
    assert not res_df["p1_x"].isna().any()
    assert (res_df["p1_x"] == 0.5).all()


def test_rec3d_single_row_dlt_fallback(test_data):
    """Ensure rec3d.py applies single-row DLT3D parameters to all frames in multi-frame pixel CSV."""
    dlt1 = pd.DataFrame(
        {
            "frame": [0],
            "L1": [100.0],
            "L2": [0.0],
            "L3": [0.0],
            "L4": [100.0],
            "L5": [0.0],
            "L6": [100.0],
            "L7": [0.0],
            "L8": [100.0],
            "L9": [0.0],
            "L10": [0.0],
            "L11": [0.0],
        }
    )
    dlt2 = pd.DataFrame(
        {
            "frame": [0],
            "L1": [0.0],
            "L2": [100.0],
            "L3": [0.0],
            "L4": [200.0],
            "L5": [0.0],
            "L6": [0.0],
            "L7": [100.0],
            "L8": [200.0],
            "L9": [0.0],
            "L10": [0.0],
            "L11": [0.0],
        }
    )
    dlt_path1 = test_data / "cam1_1row.dlt3d"
    dlt_path2 = test_data / "cam2_1row.dlt3d"
    dlt1.to_csv(dlt_path1, index=False)
    dlt2.to_csv(dlt_path2, index=False)

    # 4 frames of tracking
    in_dir = test_data / "in_single_fallback_3d"
    in_dir.mkdir()
    c1_df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "p0_x": [130.0, 130.0, 130.0, 130.0],
            "p0_y": [140.0, 140.0, 140.0, 140.0],
        }
    )
    c2_df = pd.DataFrame(
        {
            "frame": [0, 1, 2, 3],
            "p0_x": [240.0, 240.0, 240.0, 240.0],
            "p0_y": [270.0, 270.0, 270.0, 270.0],
        }
    )
    c1_df.to_csv(in_dir / "c1_data.csv", index=False)
    c2_df.to_csv(in_dir / "c2_data.csv", index=False)

    out_dir = test_data / "out_single_fallback_3d"
    out_dir.mkdir()

    result = subprocess.run(
        [
            sys.executable,
            "vaila/rec3d.py",
            "--dlt-files",
            str(dlt_path1),
            str(dlt_path2),
            "--input-dir",
            str(in_dir),
            "--output-dir",
            str(out_dir),
            "--rate",
            "100",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0
    subfolders = [f for f in out_dir.iterdir() if f.is_dir() and "vaila_rec3d" in f.name]
    assert len(subfolders) == 1
    res_df = pd.read_csv(next(f for f in subfolders[0].iterdir() if f.suffix == ".csv"))
    assert len(res_df) == 4
    # All frames reconstructed. rec3d.py labels its output from p0 (0.4.3),
    # matching getpixelvideo.py's pixel columns.
    assert not res_df["p0_x"].isna().any()
    assert (res_df["p0_x"] == 0.3).all()
