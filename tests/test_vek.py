from pathlib import Path

import numpy as np
import pandas as pd

from vaila.vek import analyze_trial, build_force_model, load_config, process_vek_file


def _write_synthetic_vek_files(tmp_path: Path) -> tuple[Path, Path, Path, Path]:
    n = 80
    frames = np.arange(n)
    t = frames / 240.0
    kick = 1.0 / (1.0 + np.exp(-(frames - 35) / 4.0))

    hip_x = np.full(n, 0.0)
    hip_y = np.full(n, 0.9)
    knee_x = 0.12 + 0.06 * kick
    knee_y = 0.52 + 0.02 * np.sin(2 * np.pi * t * 3)
    ankle_x = 0.24 + 0.42 * kick
    ankle_y = 0.14 + 0.03 * np.sin(2 * np.pi * t * 4)
    heel_x = ankle_x - 0.04
    heel_y = ankle_y - 0.02
    foot_x = ankle_x + 0.10
    foot_y = ankle_y + 0.01

    pose = pd.DataFrame(
        {
            "frame_index": frames,
            "right_hip_x": hip_x,
            "right_hip_y": hip_y,
            "right_knee_x": knee_x,
            "right_knee_y": knee_y,
            "right_ankle_x": ankle_x,
            "right_ankle_y": ankle_y,
            "right_heel_x": heel_x,
            "right_heel_y": heel_y,
            "right_foot_index_x": foot_x,
            "right_foot_index_y": foot_y,
        }
    )
    pose_path = tmp_path / "pose.csv"
    pose.to_csv(pose_path, index=False)

    ball_x = np.where(frames < 40, 0.72, 0.72 + (frames - 40) * 0.012)
    ball_y = np.full(n, 0.16)
    ball = pd.DataFrame(
        {
            "frame_index": frames,
            "ball_x": ball_x,
            "ball_y": ball_y,
            "confidence": np.ones(n) * 0.9,
        }
    )
    ball_path = tmp_path / "ball.csv"
    ball.to_csv(ball_path, index=False)

    calibration = pd.DataFrame(
        {
            "length_m": [0.10, 0.20, 0.30, 0.40, 0.50, 0.70, 0.90],
            "force_n": [0.0, 20.0, 50.0, 90.0, 145.0, 285.0, 470.0],
        }
    )
    calibration_path = tmp_path / "calibration.csv"
    calibration.to_csv(calibration_path, index=False)

    config_path = tmp_path / "vek_config.toml"
    config_path.write_text(
        f"""
[athlete]
id = "test"
name = "Synthetic"
body_mass_kg = 70.0
shank_length_m = 0.43

[trial]
id = "01"
condition = "resisted"
dominant_limb = "right"
kicking_limb = "right"

[camera]
fps = 240.0
coordinate_mode = "meters"
invert_y = false

[band]
anchor_x_m = 0.0
anchor_y_m = 0.10
attachment_landmark = "ankle"
slack_length_m = 0.10
calibration_csv = "{calibration_path.name}"
model = "pchip"

[ball]
mass_kg = 0.43
initially_stationary = true

[filter]
human_cutoff_hz = 12.0
foot_cutoff_hz = 15.0
ball_cutoff_hz = 24.0
band_cutoff_hz = 12.0

[analysis]
contact_frame = 40
contact_threshold_m = 0.18
pre_contact_frames = 8
post_contact_frames = 12
contact_duration_frames = 3
""",
        encoding="utf-8",
    )
    return pose_path, ball_path, calibration_path, config_path


def test_vek_force_model_uses_calibration(tmp_path: Path) -> None:
    _, _, calibration_path, config_path = _write_synthetic_vek_files(tmp_path)
    cfg = load_config(config_path)
    cfg.band_calibration_csv = str(calibration_path)
    force_model, method = build_force_model(cfg)

    values = force_model(np.array([0.10, 0.35, 0.90]))

    assert method == "pchip_calibration"
    assert values[0] == 0.0
    assert values[1] > 50.0
    assert values[2] == 470.0


def test_vek_analyze_trial_and_outputs(tmp_path: Path) -> None:
    pose_path, ball_path, calibration_path, config_path = _write_synthetic_vek_files(tmp_path)
    cfg = load_config(config_path)

    timeseries, metrics = analyze_trial(pose_path, cfg, ball_path)

    assert metrics["contact_frame"] == 40
    assert metrics["contact_detection_method"] == "manual_configured"
    assert metrics["peak_band_tension_n"] > 0
    assert metrics["positive_band_work_j"] > 0
    assert metrics["ball_launch_velocity_m_s"] > 0
    assert {"band_force_n", "foot_speed_m_s", "ball_speed_m_s"}.issubset(timeseries.columns)

    result = process_vek_file(
        pose_path,
        config_path,
        tmp_path / "out",
        ball_csv=ball_path,
        band_calibration_csv=calibration_path,
    )

    assert result.report_html.exists()
    assert result.timeseries_csv.exists()
    assert result.summary_csv.exists()
    assert "vaila-ElasticKick" in result.report_html.read_text(encoding="utf-8")
