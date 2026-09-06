"""Tests for 3D visualization and C3D/CSV integration of soccer field calibration models.

Validates:
- Exporting calibration models to C3D from drawsportsfields.py.
- Reading calibration models in showc3d.py and viewc3d_pyvista.py.
- Reading calibration models in readcsv.py.
- Units detection (m vs mm) and scale adaptation.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pytest

from vaila.drawsportsfields import (
    load_calibration_model_csv,
    save_calibration_model_c3d,
    save_calibration_model_csv,
)
from vaila.readcsv import read_csv_generic
from vaila.showc3d import draw_soccer_field_features, load_c3d_file
from vaila.viewc3d_pyvista import _load_c3d_arrays

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "vaila" / "models"


def test_save_calibration_model_c3d_in_meters(tmp_path: Path) -> None:
    """Test saving soccerfield_kiki calibration points to C3D in meters."""
    kiki_csv = MODELS_DIR / "soccerfield_kiki.csv"
    assert kiki_csv.exists()

    points = load_calibration_model_csv(kiki_csv)
    assert len(points) == 49

    out_c3d = tmp_path / "kiki_test_meters.c3d"
    saved_path = save_calibration_model_c3d(points, out_c3d, conversion_factor=1.0)
    assert saved_path.exists()
    assert saved_path.stat().st_size > 0

    # Read back with showc3d loader
    pts, fpath, fps, labels = load_c3d_file(saved_path)
    assert pts.shape == (1, 49, 3)
    assert len(labels) == 49
    assert labels[0] == points[0]["point_name"]
    assert fps == 1.0

    # Coordinates in meters: max span should be around 105m length
    x_coords = pts[0, :, 0]
    span_x = x_coords.max() - x_coords.min()
    assert 90.0 < span_x < 120.0

    # Read back with viewc3d_pyvista loader
    payload = _load_c3d_arrays(saved_path)
    assert payload["units"] == "m"
    assert payload["n_markers"] == 49
    assert payload["n_frames"] == 1
    assert payload["points_data"].shape == (1, 49, 3)


def test_save_calibration_model_c3d_in_millimeters(tmp_path: Path) -> None:
    """Test saving calibration points to C3D in millimeters and auto-converting back to meters."""
    kiki_csv = MODELS_DIR / "soccerfield_kiki.csv"
    points = load_calibration_model_csv(kiki_csv)

    out_c3d = tmp_path / "kiki_test_mm.c3d"
    saved_path = save_calibration_model_c3d(points, out_c3d, conversion_factor=1000.0)
    assert saved_path.exists()

    # Both loaders should convert from mm back to meters
    pts, _, _, labels = load_c3d_file(saved_path)
    assert pts.shape == (1, 49, 3)
    # Even though saved in mm (e.g. 52450 mm), loader scales to meters (52.45 m)
    assert np.nanmax(np.abs(pts)) < 150.0

    payload = _load_c3d_arrays(saved_path)
    assert payload["units"] == "mm"
    assert np.nanmax(np.abs(payload["points_data"])) < 150.0


def test_readcsv_generic_identifies_calibration_model() -> None:
    """Test read_csv_generic detects soccerfield_kiki.csv as a calibration model."""
    kiki_csv = MODELS_DIR / "soccerfield_kiki.csv"
    index_vector, marker_data, valid_markers, delimiter = read_csv_generic(str(kiki_csv))

    assert len(valid_markers) == 49
    assert len(marker_data) == 49
    assert delimiter == ","
    assert len(index_vector) == 1

    # Check keypoint coordinates
    first_marker = list(valid_markers.keys())[0]
    coords = marker_data[first_marker]
    assert coords.shape == (1, 3)
    assert not np.isnan(coords).any()


def test_readcsv_generic_custom_calibration_model(tmp_path: Path) -> None:
    """Test read_csv_generic on a custom exported calibration model."""
    custom_pts = [
        {"point_name": "corner_lb", "point_number": 0, "x": -52.45, "y": -33.95, "z": 0.0},
        {"point_name": "corner_rb", "point_number": 1, "x": 52.45, "y": -33.95, "z": 0.0},
        {"point_name": "goal_post_top", "point_number": 2, "x": -52.45, "y": 0.0, "z": 2.44},
    ]
    out_csv = tmp_path / "test_custom_model.csv"
    save_calibration_model_csv(custom_pts, out_csv)

    index_vector, marker_data, valid_markers, delimiter = read_csv_generic(str(out_csv))
    assert len(valid_markers) == 3
    assert "corner_lb" in valid_markers
    assert "goal_post_top" in valid_markers
    assert marker_data["goal_post_top"][0, 2] == pytest.approx(2.44)


def test_showc3d_draw_soccer_field_features() -> None:
    """Test drawing soccer field features without GUI display."""
    import matplotlib.pyplot as plt

    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    draw_soccer_field_features(ax, -52.45, 52.45, -33.95, 33.95, z_ground=0.0)

    # Check lines added to axes
    lines = ax.get_lines()
    assert len(lines) >= 3  # pitch boundary, halfway line, center circle
    plt.close(fig)
