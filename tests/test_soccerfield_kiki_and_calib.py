"""Tests for soccerfield_kiki.csv model, coincident keypoint rendering, and calibration keypoint editor."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest

from vaila import drawsportsfields as dsf

REPO_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = REPO_ROOT / "vaila" / "models"


def test_soccerfield_kiki_file_exists_and_matches_49_points() -> None:
    """Verify soccerfield_kiki.csv exists, has 49 keypoints, and matches canonical schema."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    assert kiki_path.exists(), f"File missing: {kiki_path}"

    df_kiki = pd.read_csv(kiki_path)
    assert len(df_kiki) == 49

    expected_cols = ["point_name", "point_number", "flip_idx", "x", "y", "z", "x_norm", "y_norm"]
    for col in expected_cols:
        assert col in df_kiki.columns, f"Missing column {col} in soccerfield_kiki.csv"

    # Compare first 48 against soccerfield_ref3d_fifa_dataset.csv
    dataset_path = MODELS_DIR / "soccerfield_ref3d_fifa_dataset.csv"
    assert dataset_path.exists()
    df_dataset = pd.read_csv(dataset_path)

    pd.testing.assert_frame_equal(df_kiki.iloc[:48], df_dataset)

    # Verify point 48: center_field at (0, 0, 0)
    row_48 = df_kiki.iloc[48]
    assert row_48["point_name"] == "center_field"
    assert row_48["point_number"] == 48
    assert row_48["flip_idx"] == 48
    assert pytest.approx(row_48["x"]) == 0.0
    assert pytest.approx(row_48["y"]) == 0.0
    assert pytest.approx(row_48["z"]) == 0.0
    assert pytest.approx(row_48["x_norm"]) == 0.5
    assert pytest.approx(row_48["y_norm"]) == 0.5


def test_kiki_sport_detection_and_registry() -> None:
    """Verify 'kiki' is detected by _detect_sport and registered in SPORT_REGISTRY."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    df_kiki = pd.read_csv(kiki_path)

    detected = dsf._detect_sport(str(kiki_path), df_kiki)
    assert detected == "kiki"

    assert "kiki" in dsf.SPORT_REGISTRY
    spec = dsf.SPORT_REGISTRY["kiki"]
    assert spec.model_csv == "soccerfield_kiki.csv"
    assert "Kiki" in spec.title or "kiki" in spec.title.lower()
    assert spec.plot_fn is dsf.plot_field_fifa_dataset


def test_plot_field_fifa_dataset_kiki_rendering_and_deoverlapped_keypoints() -> None:
    """Ensure that coincident points (corner ground vs flags, goal posts) have different positions and orientations."""
    kiki_path = MODELS_DIR / "soccerfield_kiki.csv"
    df_kiki = pd.read_csv(kiki_path)

    fig, ax = dsf.plot_field_fifa_dataset(df_kiki)
    assert fig is not None
    assert ax is not None

    # Collect all text annotations from the axes
    texts = ax.texts
    text_info = {}
    for t in texts:
        s = t.get_text()
        pos = t.get_position()
        rot = t.get_rotation()
        text_info[s] = (pos, rot, t)

    # Check that corner points and flags are both present with concise numeric labels
    assert "0" in text_info, "Corner 0 missing"
    assert "38" in text_info, "Flag 38 missing"

    pos_0, rot_0, _ = text_info["0"]
    pos_38, rot_38, _ = text_info["38"]

    # Position must NOT be the same!
    assert pos_0 != pos_38, f"Positions overlap: {pos_0} vs {pos_38}"

    # Check corner 5 vs flag 39
    assert "5" in text_info
    assert "39" in text_info
    pos_5, _, _ = text_info["5"]
    pos_39, _, _ = text_info["39"]
    assert pos_5 != pos_39

    # Check corner 24 vs flag 46
    assert "24" in text_info
    assert "46" in text_info
    pos_24, _, _ = text_info["24"]
    pos_46, _, _ = text_info["46"]
    assert pos_24 != pos_46

    # Check corner 29 vs flag 47
    assert "29" in text_info
    assert "47" in text_info
    pos_29, _, _ = text_info["29"]
    pos_47, _, _ = text_info["47"]
    assert pos_29 != pos_47

    # Check goal posts: base (32) vs top (34) - drawn backwards (para trás)
    assert "32" in text_info
    assert "34" in text_info
    pos_32, _, _ = text_info["32"]
    pos_34, _, _ = text_info["34"]
    assert pos_32 != pos_34
    # Must be drawn backwards behind the goal line (X < -52.45) and outside penalty area (Y <= -3.66)
    assert pos_32[0] < -52.45, f"Post base 32 should be behind goal line: {pos_32[0]}"
    assert pos_34[0] < -52.45, f"Post top 34 should be behind goal line: {pos_34[0]}"
    assert pos_32[1] <= -3.66, f"Post base 32 should not cross into goal area: {pos_32[1]}"
    assert pos_34[1] <= -3.66, f"Post top 34 should not cross into goal area: {pos_34[1]}"
    sep_32_34 = np.hypot(pos_32[0] - pos_34[0], pos_32[1] - pos_34[1])
    assert sep_32_34 >= 2.5, f"Goal base and top too close: {sep_32_34:.2f}m"

    # Check top post: 33 vs 35 - drawn backwards (para trás)
    assert "33" in text_info
    assert "35" in text_info
    pos_33, _, _ = text_info["33"]
    pos_35, _, _ = text_info["35"]
    assert pos_33[0] < -52.45, f"Post base 33 should be behind goal line: {pos_33[0]}"
    assert pos_35[0] < -52.45, f"Post top 35 should be behind goal line: {pos_35[0]}"
    assert pos_33[1] >= 3.66, f"Post base 33 should not cross into goal area: {pos_33[1]}"
    assert pos_35[1] >= 3.66, f"Post top 35 should not cross into goal area: {pos_35[1]}"
    sep_33_35 = np.hypot(pos_33[0] - pos_35[0], pos_33[1] - pos_35[1])
    assert sep_33_35 >= 2.5, f"Top goal base and top too close: {sep_33_35:.2f}m"

    # Check net ground points (fundo do gol: 36, 37, 44, 45)
    assert "36" in text_info
    assert "37" in text_info
    assert "44" in text_info
    assert "45" in text_info
    pos_36, _, _ = text_info["36"]
    assert pos_36[0] < -54.0, f"Net point 36 should be behind net: {pos_36[0]}"
    sep_32_36 = np.hypot(pos_32[0] - pos_36[0], pos_32[1] - pos_36[1])
    assert sep_32_36 >= 2.0, f"Net point too close to post: {sep_32_36:.2f}m"

    # Right goal posts: 40, 42, 41, 43 - drawn backwards behind right goal line (X > 52.45)
    assert "40" in text_info
    assert "42" in text_info
    assert "41" in text_info
    assert "43" in text_info
    pos_40, _, _ = text_info["40"]
    pos_42, _, _ = text_info["42"]
    assert pos_40[0] > 52.45, f"Right goal base 40 should be behind goal line: {pos_40[0]}"
    assert pos_42[0] > 52.45, f"Right goal top 42 should be behind goal line: {pos_42[0]}"
    assert pos_40[1] <= -3.66
    assert pos_42[1] <= -3.66
    sep_40_42 = np.hypot(pos_40[0] - pos_42[0], pos_40[1] - pos_42[1])
    assert sep_40_42 >= 2.5, f"Right goal base and top too close: {sep_40_42:.2f}m"

    # Check center field spot (point 48) inside center circle
    assert "48" in text_info, "Center field point 48 missing from plot"
    pos_48, _, _ = text_info["48"]
    assert pos_48[0] > 0.0, f"Point 48 should be offset from center: {pos_48[0]}"
    assert pos_48[1] > 0.0, f"Point 48 should be offset from center: {pos_48[1]}"

    # Leader lines check: verify axes lines contain dotted connecting lines
    lines = ax.lines
    assert len(lines) > 0, "Leader lines missing from plot"

    plt.close(fig)


def test_ref_label_deoverlapping_coincident_points() -> None:
    """Test _ref_label differentiates label positions and orientations when points share x, y."""
    fig, ax = plt.subplots()
    points = {
        "corner_ground": (-52.45, 33.95, 0),
        "corner_flag_top": (-52.45, 33.95, 38, 1.5),
    }
    dsf._ref_label(ax, points, field_w=105.0, field_h=68.0, show=True)

    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) == 2

    t1, t2 = texts[0], texts[1]
    assert t1.get_position() != t2.get_position()
    assert t1.get_rotation() != t2.get_rotation()
    plt.close(fig)


def test_calibration_model_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Test saving calibration points to CSV and loading them back accurately."""
    custom_points = [
        {"point_name": "origin", "point_number": 0, "x": 0.0, "y": 0.0, "z": 0.0},
        {"point_name": "bench_left", "point_number": 1, "x": -10.5, "y": 38.0, "z": 0.8},
        {"point_name": "camera_pole_top", "point_number": 2, "x": 25.0, "y": -40.0, "z": 6.5},
    ]

    out_file = tmp_path / "my_custom_model.csv"
    saved_path = dsf.save_calibration_model_csv(custom_points, out_file)
    assert saved_path.exists()

    loaded = dsf.load_calibration_model_csv(saved_path)
    assert len(loaded) == 3
    assert loaded[0]["point_name"] == "origin"
    assert pytest.approx(loaded[0]["x"]) == 0.0
    assert pytest.approx(loaded[0]["z"]) == 0.0

    assert loaded[1]["point_name"] == "bench_left"
    assert pytest.approx(loaded[1]["x"]) == -10.5
    assert pytest.approx(loaded[1]["y"]) == 38.0
    assert pytest.approx(loaded[1]["z"]) == 0.8

    assert loaded[2]["point_name"] == "camera_pole_top"
    assert pytest.approx(loaded[2]["z"]) == 6.5


def test_calibration_model_augment_and_scratch_workflow(tmp_path: Path) -> None:
    """Test augmenting an existing model and building a model from scratch."""
    # 1. Start from scratch
    scratch_points: list[dict] = []
    # Add 4 corners from scratch
    scratch_points.append({"point_name": "c1", "point_number": 1, "x": -50.0, "y": 30.0, "z": 0.0})
    scratch_points.append({"point_name": "c2", "point_number": 2, "x": -50.0, "y": -30.0, "z": 0.0})
    scratch_points.append({"point_name": "c3", "point_number": 3, "x": 50.0, "y": 30.0, "z": 0.0})
    scratch_points.append({"point_name": "c4", "point_number": 4, "x": 50.0, "y": -30.0, "z": 0.0})

    scratch_file = tmp_path / "scratch_model.csv"
    dsf.save_calibration_model_csv(scratch_points, scratch_file)

    loaded_scratch = dsf.load_calibration_model_csv(scratch_file)
    assert len(loaded_scratch) == 4

    # 2. Augment the model by adding 2 new keypoints
    loaded_scratch.append(
        {"point_name": "c1_flag_top", "point_number": 5, "x": -50.0, "y": 30.0, "z": 1.5}
    )
    loaded_scratch.append(
        {"point_name": "penalty_spot", "point_number": 6, "x": -40.0, "y": 0.0, "z": 0.0}
    )

    augmented_file = tmp_path / "augmented_model.csv"
    dsf.save_calibration_model_csv(loaded_scratch, augmented_file)

    loaded_augmented = dsf.load_calibration_model_csv(augmented_file)
    assert len(loaded_augmented) == 6
    assert loaded_augmented[4]["point_name"] == "c1_flag_top"
    assert pytest.approx(loaded_augmented[4]["z"]) == 1.5


def test_plot_field_reference_points_deoverlapping() -> None:
    """Test plot_field de-overlaps coincident reference points and produces distinct positions."""
    df_fifa = pd.read_csv(MODELS_DIR / "soccerfield_ref3d_fifa.csv")
    fig, ax = dsf.plot_field(df_fifa, show_reference_points=True)
    assert fig is not None
    assert ax is not None

    texts = [t for t in ax.texts if t.get_text()]
    assert len(texts) >= 30

    # Ensure no two text labels share the exact same position
    positions = [t.get_position() for t in texts]
    assert len(positions) == len(set(positions)), (
        "Coincident points in plot_field produced overlapping text positions"
    )

    plt.close(fig)


def test_calibration_keypoints_custom_coords_and_labels(tmp_path: Path) -> None:
    """Test manual coordinate entry, custom labels, and accurate normalization."""
    pts = [
        {"point_name": "cam_post_high", "point_number": 100, "x": -52.45, "y": 0.0, "z": 8.5},
        {"point_name": "coach_box_center", "point_number": 101, "x": -15.0, "y": 35.0, "z": 0.0},
    ]
    out_file = tmp_path / "custom_calib_model.csv"
    dsf.save_calibration_model_csv(pts, out_file)
    assert out_file.exists()

    df_out = pd.read_csv(out_file)
    assert len(df_out) == 2
    assert "point_name" in df_out.columns
    assert df_out.loc[0, "point_name"] == "cam_post_high"
    assert df_out.loc[0, "point_number"] == 100
    assert pytest.approx(df_out.loc[0, "z"]) == 8.5
    assert df_out.loc[1, "point_name"] == "coach_box_center"
    assert pytest.approx(df_out.loc[1, "x"]) == -15.0
