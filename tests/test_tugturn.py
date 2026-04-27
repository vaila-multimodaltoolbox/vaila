"""
Unit tests for vaila/tugturn.py.

These tests are deliberately fast and isolated: they exercise the smallest
testable units of the TUG/Turn pipeline (helpers, math primitives, and the
``TUGAnalyzer`` building blocks) using synthetic data only — no file I/O on
the real MediaPipe CSV. The slower integration / regression / E2E tests live
in ``tests/test_tugturn_integration.py``.

Reference (Chinaglia et al., 2026): https://arxiv.org/abs/2602.21425
Software repos:
    - https://github.com/vaila-multimodaltoolbox/vaila
    - https://github.com/paulopreto/tugturn_GP
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from vaila.tugturn import (
    DEFAULT_SKELETON_CONNECTIONS_JSON,
    FALLBACK_SKELETON_CONNECTIONS,
    LEFT_BODY_POINTS,
    MEDIAPIPE_LANDMARK_NAMES,
    PHASE_PLOT_ORDER,
    RIGHT_BODY_POINTS,
    TUGAnalyzer,
    _format_html_value,
    _parse_pose_connections,
    build_side_by_side_rows,
    calculate_absolute_inclination_3d,
    calculate_angle_3d,
    calculate_axial_vector_coding,
    calculate_limb_vector_coding_y,
    canonical_phase_name,
    get_connection_color,
    load_mediapipe_pose_connections,
    ordered_phase_ranges,
    phase_display_name,
    sample_frames,
    write_single_row_csv,
)

# ---------------------------------------------------------------------------
# Helpers / fixtures
# ---------------------------------------------------------------------------


def _synthetic_pose_dataframe(n_frames: int = 300, fps: float = 60.0) -> pd.DataFrame:
    """Build a 33-landmark MediaPipe CSV-like DataFrame with a simple
    deterministic kinematic pattern that is sufficient to exercise the
    analyzer's column extraction and basic math without requiring the real
    test trial.

    Layout: standing posture along Y (forward), Z up. The subject walks
    forward at ~0.5 m/s (X≈0), with mild vertical CoM modulation so the
    sit-to-stand thresholds activate predictably.
    """
    t = np.arange(n_frames) / fps
    forward = 0.5 * t  # 0.5 m/s along +Y
    z_modulation = 0.05 * np.sin(2 * np.pi * 1.0 * t)  # 1 Hz, ±5 cm

    cols: dict[str, np.ndarray] = {"frame": np.arange(n_frames, dtype=int)}

    z_ref = {
        0: 1.65,
        11: 1.45,
        12: 1.45,
        13: 1.20,
        14: 1.20,
        15: 0.95,
        16: 0.95,
        17: 0.92,
        18: 0.92,
        19: 0.92,
        20: 0.92,
        21: 0.92,
        22: 0.92,
        23: 0.95,
        24: 0.95,
        25: 0.55,
        26: 0.55,
        27: 0.10,
        28: 0.10,
        29: 0.07,
        30: 0.07,
        31: 0.05,
        32: 0.05,
    }
    x_offset = {
        0: 0.0,
        1: -0.03,
        2: -0.05,
        3: -0.07,
        4: 0.03,
        5: 0.05,
        6: 0.07,
        7: -0.10,
        8: 0.10,
        9: -0.02,
        10: 0.02,
        11: 0.18,
        13: 0.18,
        15: 0.18,
        17: 0.18,
        19: 0.18,
        21: 0.18,
        12: -0.18,
        14: -0.18,
        16: -0.18,
        18: -0.18,
        20: -0.18,
        22: -0.18,
        23: 0.10,
        25: 0.10,
        27: 0.10,
        29: 0.10,
        31: 0.10,
        24: -0.10,
        26: -0.10,
        28: -0.10,
        30: -0.10,
        32: -0.10,
    }

    for idx in range(33):
        z = z_ref.get(idx, 0.5) + z_modulation
        x = np.full(n_frames, x_offset.get(idx, 0.0))
        y = forward + (0.0 if idx not in {29, 30, 31, 32} else 0.05)
        # 1-based MediaPipe naming (vaila default): p1..p33
        p = idx + 1
        cols[f"p{p}_x"] = x
        cols[f"p{p}_y"] = y
        cols[f"p{p}_z"] = z

    return pd.DataFrame(cols)


@pytest.fixture(scope="module")
def synthetic_df() -> pd.DataFrame:
    return _synthetic_pose_dataframe()


@pytest.fixture(scope="module")
def synthetic_analyzer(synthetic_df: pd.DataFrame) -> TUGAnalyzer:
    a = TUGAnalyzer(synthetic_df, fs=60.0)
    a.calculate_com_3d()
    a.extract_kinematics()
    return a


# ---------------------------------------------------------------------------
# 1. Module-level constants / invariants
# ---------------------------------------------------------------------------


def test_module_constants_invariants():
    assert RIGHT_BODY_POINTS.isdisjoint(LEFT_BODY_POINTS)
    assert RIGHT_BODY_POINTS and LEFT_BODY_POINTS

    assert PHASE_PLOT_ORDER == [
        "stand",
        "first_gait",
        "stop_5s",
        "turn180",
        "second_gait",
        "sit",
    ]
    assert len(set(PHASE_PLOT_ORDER)) == len(PHASE_PLOT_ORDER)
    assert len(MEDIAPIPE_LANDMARK_NAMES) == 33
    assert DEFAULT_SKELETON_CONNECTIONS_JSON["schema"] == "mediapipe_pose_33_pn"
    assert isinstance(DEFAULT_SKELETON_CONNECTIONS_JSON["connections"], list)
    assert FALLBACK_SKELETON_CONNECTIONS


# ---------------------------------------------------------------------------
# 2. String / phase helpers
# ---------------------------------------------------------------------------


def test_canonical_phase_name():
    assert canonical_phase_name("turn180") == "turn180"
    assert canonical_phase_name("first_gait") == "first_gait"
    assert canonical_phase_name(123) == "123"  # ty: ignore[invalid-argument-type]
    assert canonical_phase_name("") == ""


def test_phase_display_name():
    assert phase_display_name("turn180") == "Turn180"
    assert phase_display_name("first_gait") == "First Gait"
    assert phase_display_name("stop_5s") == "Stop 5S"
    assert phase_display_name("stand") == "Stand"


def test_ordered_phase_ranges_filters_and_orders():
    phases = {
        "sit": (10.0, 12.0),
        "first_gait": (1.0, 5.0),
        "stand": (0.0, 1.0),
        "Total_TUG_Time": 12.0,
        "garbage": "nope",
        "turn180": [5.0, 6.0],
    }
    out = list(ordered_phase_ranges(phases))
    names = [n for n, _ in out]
    assert names == ["stand", "first_gait", "turn180", "sit"]


# ---------------------------------------------------------------------------
# 3. Sampling helpers
# ---------------------------------------------------------------------------


def test_sample_frames():
    assert sample_frames([]) == []

    frames = [3, 1, 2, 4, 5, 1]
    assert sample_frames(frames, max_frames=10) == [1, 2, 3, 4, 5]

    frames = list(range(100))
    sampled = sample_frames(frames, max_frames=5)
    assert len(sampled) == 5
    assert sampled == [0, 24, 49, 74, 99]
    assert sampled[0] == 0
    assert sampled[-1] == 99
    assert sampled == sorted(set(sampled))


# ---------------------------------------------------------------------------
# 4. Connection / colour helpers
# ---------------------------------------------------------------------------


def test_get_connection_color():
    assert get_connection_color(12, 14) == "red"
    assert get_connection_color(11, 13) == "blue"
    assert get_connection_color(0, 1) == "black"
    assert get_connection_color(11, 12) == "black"


def test_parse_pose_connections_valid_and_invalid():
    conns = [
        ["p1", "p2"],
        ["p12", "p13"],
        ["nose", "p3"],
        ["p34", "p1"],
        ["p0", "p2"],
        "not-a-pair",
        ["p2"],
    ]
    parsed = _parse_pose_connections(conns)
    assert parsed == [(0, 1), (11, 12)]


def test_load_mediapipe_pose_connections_default_file():
    parsed = load_mediapipe_pose_connections()
    assert parsed
    for a, b in parsed:
        assert 0 <= a < 33 and 0 <= b < 33


def test_load_mediapipe_pose_connections_falls_back(tmp_path: Path):
    parsed = load_mediapipe_pose_connections(tmp_path / "does_not_exist.json")
    assert parsed
    assert all(isinstance(t, tuple) and len(t) == 2 for t in parsed)


def test_load_mediapipe_pose_connections_invalid_json(tmp_path: Path):
    bad = tmp_path / "bad_skeleton.json"
    bad.write_text("{not: valid: json}", encoding="utf-8")
    parsed = load_mediapipe_pose_connections(bad)
    assert parsed


def test_load_mediapipe_pose_connections_empty_connections(tmp_path: Path):
    f = tmp_path / "empty.json"
    f.write_text(json.dumps({"connections": []}), encoding="utf-8")
    parsed = load_mediapipe_pose_connections(f)
    assert parsed


# ---------------------------------------------------------------------------
# 5. CSV / HTML formatting helpers
# ---------------------------------------------------------------------------


def test_format_html_value():
    assert _format_html_value(1) == "1.000"
    assert _format_html_value(2.5) == "2.500"
    assert _format_html_value(None) == ""
    assert _format_html_value("abc") == "abc"


def test_build_side_by_side_rows_skips_per_step():
    left = {"Step_Length_m": 0.4, "per_step": {"x": 1}, "Stride_Length_m": 0.8}
    right = {"Step_Length_m": 0.42, "Stride_Length_m": 0.82}
    html = build_side_by_side_rows(left, right)
    assert "per_step" not in html
    assert "Step Length m" in html
    assert "0.400" in html and "0.420" in html
    assert "0.800" in html and "0.820" in html


def test_write_single_row_csv_roundtrip(tmp_path: Path):
    fp = tmp_path / "row.csv"
    data = {"File_ID": "demo", "Velocity_m_s": 1.234, "Cadence": 100}
    write_single_row_csv(fp, data)
    df = pd.read_csv(fp)
    assert len(df) == 1
    assert df.iloc[0]["File_ID"] == "demo"
    assert np.isclose(df.iloc[0]["Velocity_m_s"], 1.234)
    assert int(df.iloc[0]["Cadence"]) == 100


# ---------------------------------------------------------------------------
# 6. Geometry primitives
# ---------------------------------------------------------------------------


def test_calculate_angle_3d_basic_cases():
    p1 = np.array([[0, 0, 0]])
    p2 = np.array([[1, 0, 0]])
    p3 = np.array([[1, 1, 0]])
    assert np.isclose(calculate_angle_3d(p1, p2, p3)[0], 90.0)

    p1 = np.array([[-1, 0, 0]])
    p2 = np.array([[0, 0, 0]])
    p3 = np.array([[1, 0, 0]])
    assert np.isclose(calculate_angle_3d(p1, p2, p3)[0], 180.0)

    p1 = np.array([[1, 0, 0]])
    p2 = np.array([[0, 0, 0]])
    p3 = np.array([[1, 0, 0]])
    assert np.isclose(calculate_angle_3d(p1, p2, p3)[0], 0.0)


def test_calculate_angle_3d_vectorised():
    n = 50
    p1 = np.tile([0, 0, 0], (n, 1)).astype(float)
    p2 = np.tile([1, 0, 0], (n, 1)).astype(float)
    p3 = np.column_stack([np.ones(n), np.linspace(0.0, 10.0, n), np.zeros(n)])
    angles = calculate_angle_3d(p1, p2, p3)
    assert angles.shape == (n,)
    assert np.all(np.isfinite(angles))
    assert angles.min() >= 0.0 and angles.max() <= 180.0
    # First frame: p3==p2 → zero-length v2 unit vector, dot=0 → 90 deg
    # (graceful handling, not a NaN).
    assert np.isclose(angles[0], 90.0)
    assert np.isclose(angles[-1], 90.0)


def test_calculate_angle_3d_handles_zero_length_vectors():
    p1 = np.array([[0, 0, 0]])
    p2 = np.array([[0, 0, 0]])
    p3 = np.array([[1, 1, 0]])
    angles = calculate_angle_3d(p1, p2, p3)
    assert np.all(np.isfinite(angles))


def test_calculate_absolute_inclination_3d_basic():
    vertical = np.array([0, 0, 1])
    p_top = np.array([[0, 0, 2]])
    p_bot = np.array([[0, 0, 1]])
    assert np.isclose(calculate_absolute_inclination_3d(p_top, p_bot, vertical)[0], 0.0)

    p_top = np.array([[1, 0, 1]])
    p_bot = np.array([[0, 0, 1]])
    assert np.isclose(calculate_absolute_inclination_3d(p_top, p_bot, vertical)[0], 90.0)

    p_top = np.array([[0, 0, 0]])
    p_bot = np.array([[0, 0, 1]])
    assert np.isclose(calculate_absolute_inclination_3d(p_top, p_bot, vertical)[0], 180.0)


def test_calculate_absolute_inclination_3d_default_axis():
    p_top = np.array([[0, 0, 1.0]])
    p_bot = np.array([[0, 0, 0.0]])
    assert np.isclose(calculate_absolute_inclination_3d(p_top, p_bot)[0], 0.0)


# ---------------------------------------------------------------------------
# 7. TUGAnalyzer column-extraction (multi-schema robustness)
# ---------------------------------------------------------------------------


def _minimal_df_with_schema(naming: str, n: int = 5) -> pd.DataFrame:
    rng = np.random.default_rng(0)
    cols: dict[str, np.ndarray] = {}
    for idx in range(33):
        x = rng.normal(size=n)
        y = rng.normal(size=n)
        z = rng.normal(size=n)
        if naming == "p1_lower":
            p = idx + 1
            cols[f"p{p}_x"], cols[f"p{p}_y"], cols[f"p{p}_z"] = x, y, z
        elif naming == "p0_lower":
            cols[f"p{idx}_x"], cols[f"p{idx}_y"], cols[f"p{idx}_z"] = x, y, z
        elif naming == "name":
            nm = MEDIAPIPE_LANDMARK_NAMES[idx]
            cols[f"{nm}_x"], cols[f"{nm}_y"], cols[f"{nm}_z"] = x, y, z
        else:
            raise ValueError(naming)
    return pd.DataFrame(cols)


@pytest.mark.parametrize("naming", ["p1_lower", "p0_lower", "name"])
def test_get_point_3d_supports_multiple_schemas(naming):
    df = _minimal_df_with_schema(naming)
    a = TUGAnalyzer(df, fs=60.0)
    pt = a._get_point_3d(0)  # nose
    assert pt.shape == (5, 3)
    assert np.all(np.isfinite(pt))


def test_get_point_3d_xy_only_yields_zero_z():
    rng = np.random.default_rng(1)
    cols = {}
    for idx in range(33):
        p = idx + 1
        cols[f"p{p}_x"] = rng.normal(size=4)
        cols[f"p{p}_y"] = rng.normal(size=4)
    df = pd.DataFrame(cols)
    a = TUGAnalyzer(df, fs=60.0)
    pt = a._get_point_3d(11)
    assert pt.shape == (4, 3)
    assert np.all(pt[:, 2] == 0.0)


def test_get_point_3d_invalid_index_raises():
    df = _minimal_df_with_schema("p1_lower")
    a = TUGAnalyzer(df, fs=60.0)
    with pytest.raises(ValueError):
        a._get_point_3d(99)


def test_get_point_3d_missing_columns_raises():
    df = pd.DataFrame({"frame": np.arange(3)})
    a = TUGAnalyzer(df, fs=60.0)
    with pytest.raises(ValueError):
        a._get_point_3d(0)


# ---------------------------------------------------------------------------
# 8. TUGAnalyzer derived metrics on synthetic data
# ---------------------------------------------------------------------------


def test_calculate_com_3d_creates_columns(synthetic_analyzer):
    df = synthetic_analyzer.df
    for c in ("CoM_x", "CoM_y", "CoM_z"):
        assert c in df.columns
        assert np.all(np.isfinite(df[c].to_numpy()))
    assert 0.5 < df["CoM_z"].mean() < 1.5


def test_extract_kinematics_columns_present(synthetic_analyzer):
    df = synthetic_analyzer.df
    expected_cols = [
        "Knee_Angle_R",
        "Knee_Angle_L",
        "Hip_Angle_R",
        "Hip_Angle_L",
        "Ankle_Angle_R",
        "Ankle_Angle_L",
        "Trunk_Inclination",
        "Coupling_Angle_Hip_Knee_R",
        "Coupling_Angle_Hip_Knee_L",
        "CoM_vel_x",
        "CoM_acc_y",
        "Mid_Trunk_x",
        "Mid_Trunk_y",
        "Mid_Trunk_z",
        "Med_Foot_Right_y",
        "Med_Foot_Left_y",
        "XcoM_x",
        "XcoM_y",
    ]
    missing = [c for c in expected_cols if c not in df.columns]
    assert not missing, f"Missing kinematic columns: {missing}"


def test_extract_kinematics_trunk_inclination_near_zero(synthetic_analyzer):
    """Synthetic posture keeps shoulders directly above hips, so inclination
    relative to vertical must stay close to 0 deg."""
    incl = synthetic_analyzer.df["Trunk_Inclination"].to_numpy()
    assert np.nanmean(incl) < 5.0


def test_calculate_anatomical_frames_orthonormal(synthetic_analyzer):
    X, Y, Z = synthetic_analyzer.calculate_anatomical_frames()
    assert X is not None and Y is not None and Z is not None
    assert np.allclose(np.linalg.norm(X, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(Y, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.linalg.norm(Z, axis=1), 1.0, atol=1e-3)
    assert np.allclose(np.einsum("ij,ij->i", X, Y), 0.0, atol=1e-3)
    assert np.allclose(np.einsum("ij,ij->i", X, Z), 0.0, atol=1e-3)
    assert np.allclose(np.einsum("ij,ij->i", Y, Z), 0.0, atol=1e-3)


# ---------------------------------------------------------------------------
# 9. Vector Coding helpers (shape / bounds)
# ---------------------------------------------------------------------------


def _vc_result_invariants(res: dict, allow_empty: bool = False):
    keys = (
        "gamma_deg",
        "Movement_Percent",
        "Coordination_Pattern",
        "In_Phase_pct",
        "Anti_Phase_pct",
        "Proximal_Phase_pct",
        "Distal_Phase_pct",
        "CAV_deg",
        "Dominant_Pattern",
    )
    for k in keys:
        assert k in res, f"Missing VC key: {k}"

    if not allow_empty and res["gamma_deg"]:
        assert len(res["gamma_deg"]) == len(res["Movement_Percent"]) == 100
        assert len(res["Coordination_Pattern"]) == 100
        assert all(0.0 <= g < 360.0 for g in res["gamma_deg"])
        total_pct = (
            res["In_Phase_pct"]
            + res["Anti_Phase_pct"]
            + res["Proximal_Phase_pct"]
            + res["Distal_Phase_pct"]
        )
        assert abs(total_pct - 100.0) < 1e-3
        assert res["Dominant_Pattern"] in {
            "In_Phase",
            "Anti_Phase",
            "Proximal_Phase",
            "Distal_Phase",
        }
        assert res["CAV_deg"] >= 0.0


def test_axial_vector_coding_returns_valid_distribution(synthetic_analyzer):
    n = len(synthetic_analyzer.df)
    fps = synthetic_analyzer.fs
    res = calculate_axial_vector_coding(synthetic_analyzer, fps, 0.0, (n - 1) / fps)
    _vc_result_invariants(res)


def test_axial_vector_coding_short_phase_returns_defaults(synthetic_analyzer):
    res = calculate_axial_vector_coding(synthetic_analyzer, synthetic_analyzer.fs, 0.0, 0.0)
    _vc_result_invariants(res, allow_empty=True)
    assert res["gamma_deg"] == []
    assert res["Dominant_Pattern"] == "N/A"


def test_limb_vector_coding_returns_valid_distribution(synthetic_analyzer):
    n = len(synthetic_analyzer.df)
    fps = synthetic_analyzer.fs
    res = calculate_limb_vector_coding_y(synthetic_analyzer, fps, 0.0, (n - 1) / fps, 14, 26)
    _vc_result_invariants(res)


def test_limb_vector_coding_short_phase_returns_defaults(synthetic_analyzer):
    res = calculate_limb_vector_coding_y(
        synthetic_analyzer, synthetic_analyzer.fs, 0.0, 0.0, 14, 26
    )
    _vc_result_invariants(res, allow_empty=True)


# ---------------------------------------------------------------------------
# 10. Spatial-override metadata propagation
# ---------------------------------------------------------------------------


def test_meta_spatial_overrides_default_to_none(synthetic_df):
    a = TUGAnalyzer(synthetic_df, fs=60.0)
    assert a._meta_y_chair is None
    assert a._meta_y_turn is None
    assert a._meta_y_tol is None


def test_dt_zero_when_fs_zero(synthetic_df):
    a = TUGAnalyzer(synthetic_df, fs=0.0)
    assert a.dt == 0.0
