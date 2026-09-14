"""PODS regression proof with deterministic kinematics and no external downloads.

Version: 0.3.141
Update Date: 14 September 2026
"""

from pathlib import Path
from types import SimpleNamespace

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd
import pytest

from vaila import cmj_pods as pods
from vaila import vaila_and_jump as jump


def trial(fps=200):
    t = np.arange(int(1.8 * fps)) / fps
    y = np.zeros_like(t)
    down = (t > 0.3) & (t < 0.7)
    y[down] = -0.125 * (1 - np.cos(np.pi * (t[down] - 0.3) / 0.4))
    up = (t >= 0.7) & (t < 1.1)
    y[up] = -0.25 + 2.5 * (t[up] - 0.7) ** 2
    flight = t >= 1.1
    y[flight] = 0.15 + 2 * (t[flight] - 1.1) - 0.5 * pods.GRAVITY * (t[flight] - 1.1) ** 2
    velocity = np.gradient(y, 1 / fps)
    acceleration = np.gradient(velocity, 1 / fps)
    off, bottom, apex, landing = (round(x * fps) for x in (1.1, 0.7, 1.3, 1.51))
    acceleration[off] = -8.0  # accepted last-small-positive-force edge
    acceleration[off + 1 : landing] = -pods.GRAVITY
    mass = 80.0
    force = mass * (acceleration + pods.GRAVITY)
    force[off + 1 : landing] = 0
    data = pd.DataFrame(
        {
            "cg_y_m_filtered": y + 1,
            "cg_vy": velocity,
            "cg_ay": acceleration,
            "force_vertical": force,
            "power": force * velocity,
            "reference_cg_y": 1.0,
        }
    )
    legacy = {
        "propulsion_start_frame": bottom,
        "takeoff_frame": int(0.99 * fps),
        "propulsion_time_s": 0.29,
        "takeoff_frame_kinetic": off,
        "takeoff_frame_foot_contact": off + 1,
        "max_height_frame": apex,
        "landing_frame_foot_contact": landing,
        "landing_frame": landing + 5,
        "height_qc_recommended_m": 4 / (2 * pods.GRAVITY),
        "height_qc_recommended_status": "plausible",
        "height_qc_recommended_source": "com_takeoff_ref",
        "squat_depth_m": -0.25,
        "gravity_check_status": "ok",
    }
    return data, legacy, mass, fps


def test_formulas_phase_order_and_legacy_preservation():
    data, legacy, mass, fps = trial()
    original = legacy.copy()
    events, metrics, frames = pods.calculate_pods(data, legacy, mass, fps)
    assert legacy == original
    assert metrics["takeoff_source"] == "kinetic_estimated_grf"
    assert metrics["cmj_phase_qc_status"] == "ok"
    assert (
        events["movement_onset_frame"]
        < events["peak_downward_velocity_frame"]
        <= events["minimum_displacement_frame"]
        < events["takeoff_frame_selected"]
        < events["max_height_frame"]
        < events["landing_frame_selected"]
    )
    ttto = (events["takeoff_frame_selected"] - events["movement_onset_frame"]) / fps
    assert metrics["mrsi_AU"] == pytest.approx(legacy["height_qc_recommended_m"] / ttto)
    assert ttto != legacy["propulsion_time_s"]
    assert metrics["takeoff_velocity_height_derived_m_s"] == pytest.approx(2)
    assert metrics["jump_momentum_selected_kg_m_s"] == pytest.approx(160)
    assert metrics["jump_momentum_com_est_kg_m_s"] == pytest.approx(
        mass * data.cg_vy.iloc[events["takeoff_frame_selected"]]
    )
    assert metrics["countermovement_depth_m"] == 0.25
    assert metrics["takeoff_kinetic_vs_foot_diff_frames"] == -1
    assert metrics["takeoff_kinetic_vs_foot_diff_s"] == -1 / fps
    assert frames.force_vertical_est_N.equals(data.force_vertical.rename("force_vertical_est_N"))
    assert frames.force_vertical_est_N_per_kg.iloc[0] == pytest.approx(9.81)
    assert frames.power_est_W.equals(data.power.rename("power_est_W"))
    braking = data.force_vertical.iloc[
        events["peak_downward_velocity_frame"] : events["minimum_displacement_frame"]
    ]
    prop = data.force_vertical.iloc[
        events["minimum_displacement_frame"] : events["takeoff_frame_selected"]
    ]
    assert metrics["avg_braking_force_est_N"] == pytest.approx(braking.mean())
    assert metrics["avg_propulsive_force_est_N_per_kg"] == pytest.approx(prop.mean() / mass)
    assert metrics["peak_propulsive_force_est_N"] == pytest.approx(prop.max())
    assert (
        metrics["force_at_min_displacement_est_N"]
        == data.force_vertical.iloc[events["minimum_displacement_frame"]]
    )
    assert list(frames.cmj_phase.drop_duplicates()) == list(pods.PHASES)
    assert frames.cmj_phase.iloc[events["takeoff_frame_selected"]] == "flight"


def test_single_frame_spike_and_small_noise_do_not_move_onset():
    data, legacy, mass, fps = trial()
    clean, _, _ = pods.calculate_pods(data, legacy, mass, fps)
    data.loc[15, ["cg_y_m_filtered", "cg_vy"]] = [0.1, -20]
    data.loc[45, ["cg_y_m_filtered", "cg_vy"]] = [0.1, -20]
    rng = np.random.default_rng(9)
    data["cg_y_m_filtered"] += rng.normal(0, 0.0002, len(data))
    data["cg_vy"] += rng.normal(0, 0.002, len(data))
    noisy, _, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert abs(noisy["movement_onset_frame"] - clean["movement_onset_frame"]) <= 2
    assert abs(noisy["peak_downward_velocity_frame"] - clean["peak_downward_velocity_frame"]) <= 2


@pytest.mark.parametrize(
    "mode,source",
    [
        ("kinetic", "kinetic_estimated_grf"),
        ("foot", "foot_contact_marker"),
        ("fallback", "fallback_com_reference"),
    ],
)
def test_takeoff_selection(mode, source):
    data, legacy, mass, fps = trial()
    if mode != "kinetic":
        legacy["takeoff_frame_kinetic"] = None
    if mode == "fallback":
        legacy["takeoff_frame_foot_contact"] = None
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert metrics["takeoff_source"] == source
    if mode == "fallback":
        assert np.isnan(metrics["mrsi_AU"])
        assert np.isnan(metrics["time_to_takeoff_s"])
        assert np.isnan(metrics["avg_propulsive_force_est_N"])


def test_masked_flight_zeros_cannot_validate_kinetic_candidate():
    data, legacy, mass, fps = trial()
    data.loc[legacy["takeoff_frame_kinetic"] + 1 :, "cg_ay"] = 5
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert metrics["takeoff_source"] == "foot_contact_marker"


def test_invalid_foot_candidate_does_not_veto_valid_kinetic_takeoff():
    data, legacy, mass, fps = trial()
    legacy["takeoff_frame_foot_contact"] = 5
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert metrics["takeoff_source"] == "kinetic_estimated_grf"


def test_phase_average_excludes_known_airborne_samples():
    data, legacy, mass, fps = trial()
    legacy["takeoff_frame_kinetic"] = None
    end = legacy["takeoff_frame_foot_contact"]
    data["airborne"] = False
    data.loc[end - 2 : end - 1, "airborne"] = True
    data.loc[end - 2 : end - 1, ["force_vertical", "power"]] = 0
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    expected = data.force_vertical.iloc[legacy["propulsion_start_frame"] : end - 2].mean()
    assert metrics["avg_propulsive_force_est_N"] == pytest.approx(expected)


def test_takeoff_disagreement_is_reported_and_prefers_foot():
    data, legacy, mass, fps = trial()
    legacy["takeoff_frame_foot_contact"] += 12
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert metrics["takeoff_source"] == "foot_contact_marker"
    assert "disagree" in metrics["cmj_phase_qc_note"]


@pytest.mark.parametrize(
    "failure", ["calibration", "height", "order", "onset", "force", "gravity_missing"]
)
def test_qc_failure_does_not_create_valid_looking_metrics(failure):
    data, legacy, mass, fps = trial()
    if failure == "calibration":
        legacy["gravity_check_status"] = "suspect_fps_or_scale"
    elif failure == "gravity_missing":
        legacy["gravity_check_status"] = "insufficient_data"
    elif failure == "height":
        legacy["height_qc_recommended_m"] = None
    elif failure == "order":
        legacy["landing_frame_foot_contact"] = 30
    elif failure == "onset":
        data["cg_vy"] = 0
    elif failure == "force":
        data.loc[155, "cg_ay"] = 200
    _, metrics, frames = pods.calculate_pods(data, legacy, mass, fps)
    if failure != "height":
        assert np.isnan(metrics["avg_propulsive_force_est_N"])
    if failure not in {"force", "gravity_missing"}:
        assert np.isnan(metrics["mrsi_AU"])
    if failure == "order":
        assert not frames.cmj_phase.any()
    if failure == "force":
        assert metrics["kinetic_metrics_qc_status"] == "force_plausibility_failed"
        assert data.cg_ay.iloc[155] == 200


@pytest.mark.parametrize("fps", [30, 60])
def test_low_fps_warns_without_blanket_rejection(fps):
    data, legacy, mass, fps = trial(fps)
    _, metrics, _ = pods.calculate_pods(
        data, legacy, mass, fps, {"baseline_start_frame": 1, "baseline_end_frame": 6}
    )
    assert metrics["mrsi_qc_status"] == "limited_temporal_resolution"
    assert np.isfinite(metrics["mrsi_AU"])
    assert metrics["kinetic_metrics_qc_status"] == "limited_temporal_resolution"


@pytest.mark.parametrize("phase", ["braking", "propulsive"])
def test_too_short_phase_has_no_force_aggregate(phase):
    data, legacy, mass, fps = trial()
    if phase == "braking":
        data.loc[legacy["propulsion_start_frame"] - 1, "cg_vy"] = -2
    else:
        off = legacy["propulsion_start_frame"] + 2
        legacy["takeoff_frame_foot_contact"] = off
        legacy["takeoff_frame_kinetic"] = None
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert np.isnan(metrics[f"avg_{phase}_force_est_N"])
    assert metrics["kinetic_metrics_qc_status"] == "limited_temporal_resolution"


def test_empty_input_and_nonfinite_calibration():
    for mass, fps in ((80, 200), (np.nan, 200), (80, np.nan)):
        _, metrics, frames = pods.calculate_pods(pd.DataFrame(), {}, mass, fps)
        assert np.isnan(metrics["mrsi_AU"])
        assert frames.empty


def test_velocity_disagreement_is_explained_without_silent_source_change():
    data, legacy, mass, fps = trial()
    legacy["height_qc_recommended_m"] = 0.6
    _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
    assert "velocity disagree" in metrics["kinetic_metrics_qc_note"]
    assert metrics["jump_momentum_source"] == "height_derived"


def test_every_new_scalar_and_frame_alias_is_documented():
    data, legacy, mass, fps = trial()
    _, metrics, frames = pods.calculate_pods(data, legacy, mass, fps)
    documentation = (Path(__file__).parents[1] / "docs/cmj_pods.md").read_text()
    for key in (set(metrics) - set(legacy)) | set(frames):
        assert f"`{key}`" in documentation, key


def test_pods_report_event_table_and_plot(tmp_path):
    data, legacy, mass, fps = trial()
    events, metrics, frames = pods.calculate_pods(data, legacy, mass, fps)
    html = pods.pods_report_html({**legacy, **metrics})
    assert "CMJ Performance Profile — PODS" in html
    assert "not direct force-platform" in html
    assert "height-derived" in html.lower()
    assert "12 Hz" in html
    event_html = pods.event_rows_html(events, fps)
    assert "NOT actual foot-off" in event_html
    assert "Movement onset" in event_html
    assert "kinetic_estimated_grf" in event_html
    output = tmp_path / "diagnostic.png"
    assert pods.plot_pods_diagnostic(pd.concat([data, frames], axis=1), events, fps, output) == str(
        output
    )
    assert output.stat().st_size > 10000


def test_config_template_and_explicit_cli_phase_file(tmp_path, monkeypatch):
    config = tmp_path / "custom.toml"
    jump._save_jump_context_template(config, {"mass_kg": 80, "fps": 200, "shank_length_m": 0.4})
    config.write_text(
        config.read_text().replace(
            "movement_onset_min_duration_s = 0.03", "movement_onset_min_duration_s = 0.08"
        )
    )
    received = []
    monkeypatch.setattr(
        jump, "process_mediapipe_data", lambda *args, **kwargs: received.append(kwargs) or True
    )
    monkeypatch.setattr(jump, "_JUMP_CONTEXT", None)
    args = SimpleNamespace(
        config=str(config),
        input=str(tmp_path / "other" / "input.csv"),
        output=str(tmp_path / "out"),
    )
    assert jump._run_cli_mediapipe(args) == 0
    assert received[0]["phase_options"]["movement_onset_min_duration_s"] == 0.08
    with pytest.raises(ValueError):
        jump._jump_phase_options_from_cfg({"phase_min_duration_s": float("nan")})


def test_team_metrics_and_report_with_missing_values(tmp_path):
    rows = []
    for athlete, height in (("A", 0.3), ("B", 0.4), ("C", np.nan)):
        data, legacy, mass, fps = trial()
        legacy["height_qc_recommended_m"] = height
        _, metrics, _ = pods.calculate_pods(data, legacy, mass, fps)
        rows.append(
            {
                **legacy,
                **metrics,
                "athlete": athlete,
                "trial": "cmj",
                "height_qc_discrepancy_m": 0.01,
            }
        )
    assert set(pods.PODS_TEAM_METRICS).issubset(jump._TEAM_METRICS)
    result = jump.generate_team_report(rows, tmp_path, "test")
    html = Path(result).read_text()
    assert "PODS (team)" in html
    assert "not universally better" in html
    assert (tmp_path / "team_plots" / "team_mrsi_AU_test.png").exists()
    quality = pd.read_csv(tmp_path / "team_jump_quality_zscores_test.csv")
    assert quality.loc[quality.athlete == "C", "mrsi_AU_zscore"].isna().all()


def test_full_mediapipe_pipeline_exports_pods_and_keeps_legacy(tmp_path, monkeypatch):
    # Real landmark fixture; suppress only unrelated expensive rendering, not computation/export.
    fixture = Path(__file__).parent / "vaila_and_jump/vaila_mediapipe/salto_mp_norm_savgol.csv"
    monkeypatch.setattr(jump, "_JUMP_CONTEXT", {"mass_kg": 80, "fps": 240, "shank_length_m": 0.4})
    single_plots = (
        "generate_normalized_diagnostic_plot",
        "plot_jump_phases_analysis",
        "plot_jump_cg_feet_analysis",
    )
    for name in single_plots:
        monkeypatch.setattr(jump, name, lambda *a, **kw: "")
    for name in ("generate_jump_plots", "plot_valgus_event"):
        monkeypatch.setattr(jump, name, lambda *a, **kw: [])
    for name in (
        "generate_jump_animation_gif",
        "plot_jump_stickfigures_subplot",
        "plot_jump_stickfigures_with_cg",
        "plot_valgus_ratio",
        "plot_fppa_time_series",
    ):
        monkeypatch.setattr(jump, name, lambda *a, **kw: None)
    assert jump.process_mediapipe_data(str(fixture), str(tmp_path))
    scalar = pd.read_csv(next(tmp_path.glob("*_jump_results_*.csv")))
    frames = pd.read_csv(next(tmp_path.glob("*_calibrated_*.csv")))
    for key in (
        "mrsi_AU",
        "takeoff_frame_selected",
        "avg_braking_force_est_N",
        "kinetic_metrics_qc_status",
        "mass_kg",
        "propulsion_time_s",
        "takeoff_frame",
        "power",
    ):
        assert key in (frames if key == "power" else scalar)
    for key in (
        "force_vertical_est_N",
        "force_vertical_est_N_per_kg",
        "power_est_W",
        "power_est_W_per_kg",
        "cmj_phase",
        "cg_y_m_filtered",
        "cg_vy",
        "cg_ay",
    ):
        assert key in frames
    np.testing.assert_allclose(frames.force_vertical, frames.force_vertical_est_N)
    row = scalar.iloc[0]
    # Values captured from pre-PODS HEAD on this fixture/config, not derived from the new helper.
    golden = {
        "height_cg_method_m": 0.348161,
        "height_qc_recommended_m": 0.003066,
        "takeoff_frame": 99,
        "propulsion_start_frame": 85,
        "propulsion_time_s": 0.058333,
        "max_power_W": 273382.990416,
        "max_power_W_per_kg": 3417.287380,
        "power_avg_propulsion_W": 41.244043,
        "total_energy_J": 2.405903,
        "squat_depth_m": 0.232878,
    }
    for key, expected in golden.items():
        assert row[key] == pytest.approx(expected, abs=2e-6)
    assert row.propulsion_time_s == pytest.approx(
        (row.takeoff_frame - row.propulsion_start_frame) / 240, abs=1e-6
    )
    assert "CMJ Performance Profile — PODS" in next(tmp_path.glob("*_report_*.html")).read_text()
