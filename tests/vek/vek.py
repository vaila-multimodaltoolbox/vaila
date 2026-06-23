"""
===============================================================================
vek.py — vailá-ElasticKick (VEK)
===============================================================================
Starter module for biomechanical analysis of an elastic-band resisted kick.

Main inputs
-----------
1. Markerless 2-D pose CSV produced by vailá/MediaPipe.
2. TOML configuration with athlete, camera, elastic-band and analysis data.
3. Optional ball trajectory CSV from the same video.

Main outputs
------------
- vek_timeseries_*.csv: synchronized kinematics, band force and power.
- vek_summary_*.csv: trial-level metrics.
- vek_report_*.html: first-pass technical report.
- PNG plots for quality control and reporting.

Important scientific scope
--------------------------
The force reported here is the estimated tension in the elastic band. It is not
an inverse-dynamics estimate of total muscular force or total kicking power.
For product use, validate the full measurement chain before creating normative
percentiles or performance classifications.

License: GNU GPL v3.0
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import os
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt

try:  # Python 3.11+
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]


VERSION = "0.1.0"
DEFAULT_CUTOFF_HZ = 10.0


@dataclass(slots=True)
class VEKConfig:
    athlete_id: str = "unknown"
    athlete_name: str = "Unknown athlete"
    trial_id: str = "01"
    condition: str = "resisted"
    kicking_side: str = "right"

    fps: float = 240.0
    cutoff_hz: float = DEFAULT_CUTOFF_HZ
    coordinate_mode: str = "normalized_shank"  # meters | pixels | normalized_shank
    invert_y: bool = True
    image_height_px: float | None = None
    pixels_per_meter: float | None = None
    shank_length_m: float = 0.43

    anchor_x_m: float = 0.0
    anchor_y_m: float = 0.0
    attachment_landmark: str = "ankle"  # ankle | heel | foot_index | foot_center
    band_slack_length_m: float = 1.0
    band_k_n_per_m: float | None = None
    band_calibration_csv: str | None = None

    ball_mass_kg: float = 0.43
    ball_initially_stationary: bool = True

    contact_frame: int | None = None
    contact_threshold_m: float = 0.18
    pre_contact_frames: int = 10
    post_contact_frames: int = 10
    contact_duration_frames: int | None = None


# -----------------------------------------------------------------------------
# Configuration and input validation
# -----------------------------------------------------------------------------


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def load_config(config_path: str | Path) -> VEKConfig:
    path = Path(config_path).expanduser().resolve()
    with path.open("rb") as f:
        raw = tomllib.load(f)

    athlete = raw.get("athlete", {})
    trial = raw.get("trial", {})
    camera = raw.get("camera", {})
    band = raw.get("band", {})
    ball = raw.get("ball", {})
    analysis = raw.get("analysis", {})

    calibration_csv = band.get("calibration_csv")
    if calibration_csv:
        calibration_path = Path(str(calibration_csv)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = path.parent / calibration_path
        calibration_csv = str(calibration_path.resolve())

    cfg = VEKConfig(
        athlete_id=str(athlete.get("id", "unknown")),
        athlete_name=str(athlete.get("name", "Unknown athlete")),
        trial_id=str(trial.get("id", "01")),
        condition=str(trial.get("condition", "resisted")),
        kicking_side=str(trial.get("kicking_side", "right")).lower(),
        fps=float(_as_float(camera.get("fps"), 240.0)),
        cutoff_hz=float(_as_float(analysis.get("cutoff_hz"), DEFAULT_CUTOFF_HZ)),
        coordinate_mode=str(camera.get("coordinate_mode", "normalized_shank")).lower(),
        invert_y=bool(camera.get("invert_y", True)),
        image_height_px=_as_float(camera.get("image_height_px")),
        pixels_per_meter=_as_float(camera.get("pixels_per_meter")),
        shank_length_m=float(_as_float(athlete.get("shank_length_m"), 0.43)),
        anchor_x_m=float(_as_float(band.get("anchor_x_m"), 0.0)),
        anchor_y_m=float(_as_float(band.get("anchor_y_m"), 0.0)),
        attachment_landmark=str(band.get("attachment_landmark", "ankle")).lower(),
        band_slack_length_m=float(_as_float(band.get("slack_length_m"), 1.0)),
        band_k_n_per_m=_as_float(band.get("k_n_per_m")),
        band_calibration_csv=calibration_csv,
        ball_mass_kg=float(_as_float(ball.get("mass_kg"), 0.43)),
        ball_initially_stationary=bool(ball.get("initially_stationary", True)),
        contact_frame=(
            int(analysis["contact_frame"])
            if analysis.get("contact_frame") is not None
            else None
        ),
        contact_threshold_m=float(_as_float(analysis.get("contact_threshold_m"), 0.18)),
        pre_contact_frames=int(analysis.get("pre_contact_frames", 10)),
        post_contact_frames=int(analysis.get("post_contact_frames", 10)),
        contact_duration_frames=(
            int(analysis["contact_duration_frames"])
            if analysis.get("contact_duration_frames") is not None
            else None
        ),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: VEKConfig) -> None:
    if cfg.kicking_side not in {"left", "right"}:
        raise ValueError("kicking_side must be 'left' or 'right'")
    if cfg.fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.cutoff_hz <= 0 or cfg.cutoff_hz >= cfg.fps / 2:
        raise ValueError("cutoff_hz must be between 0 and the Nyquist frequency")
    if cfg.coordinate_mode not in {"meters", "pixels", "normalized_shank"}:
        raise ValueError("coordinate_mode must be meters, pixels, or normalized_shank")
    if cfg.coordinate_mode == "pixels" and not cfg.pixels_per_meter:
        raise ValueError("pixels_per_meter is required for coordinate_mode='pixels'")
    if cfg.band_slack_length_m <= 0:
        raise ValueError("band slack length must be positive")
    if not cfg.band_calibration_csv and not cfg.band_k_n_per_m:
        raise ValueError("provide band.calibration_csv or band.k_n_per_m")


# -----------------------------------------------------------------------------
# Signal utilities
# -----------------------------------------------------------------------------


def lowpass(values: np.ndarray, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    if values.size < max(15, order * 3):
        return values.copy()
    series = pd.Series(values).interpolate(limit_direction="both").to_numpy()
    sos = butter(order, cutoff_hz, btype="low", fs=fps, output="sos")
    return sosfiltfilt(sos, series)


def derivative(values: np.ndarray, fps: float) -> np.ndarray:
    return np.gradient(np.asarray(values, dtype=float), 1.0 / fps)


def vector_speed(x: np.ndarray, y: np.ndarray, fps: float) -> np.ndarray:
    return np.hypot(derivative(x, fps), derivative(y, fps))


def angle_2d(ax: np.ndarray, ay: np.ndarray, bx: np.ndarray, by: np.ndarray) -> np.ndarray:
    """Angle between vectors a and b, in degrees."""
    dot = ax * bx + ay * by
    norm = np.hypot(ax, ay) * np.hypot(bx, by)
    cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


# -----------------------------------------------------------------------------
# Coordinate extraction and calibration
# -----------------------------------------------------------------------------


def find_frame_column(df: pd.DataFrame) -> str:
    for name in ("frame_index", "frame", "frames", "Frame"):
        if name in df.columns:
            return name
    raise ValueError("No frame column found (expected frame_index or frame)")


def xy_columns(df: pd.DataFrame, landmark: str) -> tuple[str, str]:
    candidates = [
        (f"{landmark}_x", f"{landmark}_y"),
        (f"{landmark}.x", f"{landmark}.y"),
    ]
    for x_col, y_col in candidates:
        if x_col in df.columns and y_col in df.columns:
            return x_col, y_col
    raise ValueError(f"Missing coordinates for landmark '{landmark}'")


def _raw_xy(df: pd.DataFrame, landmark: str) -> tuple[np.ndarray, np.ndarray]:
    x_col, y_col = xy_columns(df, landmark)
    return df[x_col].to_numpy(float), df[y_col].to_numpy(float)


def coordinate_scale(df: pd.DataFrame, cfg: VEKConfig) -> float:
    if cfg.coordinate_mode == "meters":
        return 1.0
    if cfg.coordinate_mode == "pixels":
        assert cfg.pixels_per_meter is not None
        return 1.0 / cfg.pixels_per_meter

    side = cfg.kicking_side
    knee_x, knee_y = _raw_xy(df, f"{side}_knee")
    ankle_x, ankle_y = _raw_xy(df, f"{side}_ankle")
    raw_shank = np.hypot(knee_x - ankle_x, knee_y - ankle_y)
    median_raw_shank = float(np.nanmedian(raw_shank))
    if not np.isfinite(median_raw_shank) or median_raw_shank <= 0:
        raise ValueError("Could not estimate normalized shank length")
    return cfg.shank_length_m / median_raw_shank


def calibrated_xy(
    df: pd.DataFrame,
    landmark: str,
    cfg: VEKConfig,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _raw_xy(df, landmark)
    if cfg.coordinate_mode == "pixels" and cfg.invert_y:
        if cfg.image_height_px is None:
            raise ValueError("image_height_px is required to invert pixel coordinates")
        y = cfg.image_height_px - y
    elif cfg.coordinate_mode == "normalized_shank" and cfg.invert_y:
        y = 1.0 - y
    elif cfg.coordinate_mode == "meters" and cfg.invert_y:
        y = -y
    return x * scale, y * scale


def foot_center(df: pd.DataFrame, cfg: VEKConfig, scale: float) -> tuple[np.ndarray, np.ndarray]:
    side = cfg.kicking_side
    heel_x, heel_y = calibrated_xy(df, f"{side}_heel", cfg, scale)
    toe_x, toe_y = calibrated_xy(df, f"{side}_foot_index", cfg, scale)
    return (heel_x + toe_x) / 2.0, (heel_y + toe_y) / 2.0


def attachment_xy(df: pd.DataFrame, cfg: VEKConfig, scale: float) -> tuple[np.ndarray, np.ndarray]:
    side = cfg.kicking_side
    if cfg.attachment_landmark == "foot_center":
        return foot_center(df, cfg, scale)
    return calibrated_xy(df, f"{side}_{cfg.attachment_landmark}", cfg, scale)


# -----------------------------------------------------------------------------
# Elastic-band model
# -----------------------------------------------------------------------------


def build_force_model(cfg: VEKConfig):
    """Return F(length_m) using a monotonic calibration curve or linear fallback."""
    if cfg.band_calibration_csv:
        calibration = pd.read_csv(cfg.band_calibration_csv)
        required = {"length_m", "force_n"}
        if not required.issubset(calibration.columns):
            raise ValueError("Band calibration CSV must contain length_m and force_n")
        calibration = (
            calibration.loc[:, ["length_m", "force_n"]]
            .dropna()
            .groupby("length_m", as_index=False)["force_n"]
            .mean()
            .sort_values("length_m")
        )
        if len(calibration) < 3:
            raise ValueError("Band calibration requires at least three force-length points")
        interpolator = PchipInterpolator(
            calibration["length_m"].to_numpy(float),
            calibration["force_n"].to_numpy(float),
            extrapolate=False,
        )
        min_length = float(calibration["length_m"].min())
        max_length = float(calibration["length_m"].max())

        def force(length_m: np.ndarray) -> np.ndarray:
            length_m = np.asarray(length_m, dtype=float)
            clipped = np.clip(length_m, min_length, max_length)
            values = np.asarray(interpolator(clipped), dtype=float)
            values[length_m <= cfg.band_slack_length_m] = 0.0
            return np.clip(values, 0.0, None)

        return force

    assert cfg.band_k_n_per_m is not None

    def linear_force(length_m: np.ndarray) -> np.ndarray:
        extension = np.clip(np.asarray(length_m) - cfg.band_slack_length_m, 0.0, None)
        return cfg.band_k_n_per_m * extension

    return linear_force


# -----------------------------------------------------------------------------
# Ball data and event detection
# -----------------------------------------------------------------------------


def load_ball_data(ball_csv: str | Path | None) -> pd.DataFrame | None:
    if not ball_csv:
        return None
    ball = pd.read_csv(ball_csv)
    frame_col = find_frame_column(ball)
    ball = ball.rename(columns={frame_col: "frame_index"})
    if {"ball_x_m", "ball_y_m"}.issubset(ball.columns):
        return ball[["frame_index", "ball_x_m", "ball_y_m"]].copy()
    if {"ball_x", "ball_y"}.issubset(ball.columns):
        # Skeleton assumption: the optional ball tracker exports metric coordinates.
        return ball.rename(columns={"ball_x": "ball_x_m", "ball_y": "ball_y_m"})[
            ["frame_index", "ball_x_m", "ball_y_m"]
        ]
    raise ValueError("Ball CSV must contain ball_x_m and ball_y_m")


def detect_contact_frame(df: pd.DataFrame, cfg: VEKConfig) -> tuple[int | None, str]:
    if cfg.contact_frame is not None:
        return cfg.contact_frame, "configured"
    required = {"foot_x_m", "foot_y_m", "ball_x_m", "ball_y_m"}
    if not required.issubset(df.columns):
        return None, "not_available"
    distance = np.hypot(df["foot_x_m"] - df["ball_x_m"], df["foot_y_m"] - df["ball_y_m"])
    valid = distance.dropna()
    if valid.empty:
        return None, "not_available"
    idx = int(valid.idxmin())
    frame = int(df.loc[idx, "frame_index"])
    method = "minimum_distance"
    if float(valid.loc[idx]) > cfg.contact_threshold_m:
        method = "minimum_distance_low_confidence"
    return frame, method


# -----------------------------------------------------------------------------
# Core analysis
# -----------------------------------------------------------------------------


def analyze_trial(pose_csv: str | Path, cfg: VEKConfig, ball_csv: str | Path | None = None):
    pose = pd.read_csv(pose_csv)
    frame_col = find_frame_column(pose)
    pose = pose.rename(columns={frame_col: "frame_index"}).copy()
    pose = pose.sort_values("frame_index").reset_index(drop=True)
    scale = coordinate_scale(pose, cfg)
    side = cfg.kicking_side

    hip_x, hip_y = calibrated_xy(pose, f"{side}_hip", cfg, scale)
    knee_x, knee_y = calibrated_xy(pose, f"{side}_knee", cfg, scale)
    ankle_x, ankle_y = calibrated_xy(pose, f"{side}_ankle", cfg, scale)
    foot_x, foot_y = foot_center(pose, cfg, scale)
    attach_x, attach_y = attachment_xy(pose, cfg, scale)

    trajectory_names = {
        "hip_x_m": hip_x,
        "hip_y_m": hip_y,
        "knee_x_m": knee_x,
        "knee_y_m": knee_y,
        "ankle_x_m": ankle_x,
        "ankle_y_m": ankle_y,
        "foot_x_m": foot_x,
        "foot_y_m": foot_y,
        "attachment_x_m": attach_x,
        "attachment_y_m": attach_y,
    }
    out = pd.DataFrame({"frame_index": pose["frame_index"].astype(int)})
    out["time_s"] = (out["frame_index"] - out["frame_index"].iloc[0]) / cfg.fps
    for name, values in trajectory_names.items():
        out[name] = lowpass(values, cfg.fps, cfg.cutoff_hz)

    out["hip_speed_m_s"] = vector_speed(out.hip_x_m, out.hip_y_m, cfg.fps)
    out["knee_speed_m_s"] = vector_speed(out.knee_x_m, out.knee_y_m, cfg.fps)
    out["ankle_speed_m_s"] = vector_speed(out.ankle_x_m, out.ankle_y_m, cfg.fps)
    out["foot_speed_m_s"] = vector_speed(out.foot_x_m, out.foot_y_m, cfg.fps)

    thigh_x = out.hip_x_m.to_numpy() - out.knee_x_m.to_numpy()
    thigh_y = out.hip_y_m.to_numpy() - out.knee_y_m.to_numpy()
    shank_x = out.ankle_x_m.to_numpy() - out.knee_x_m.to_numpy()
    shank_y = out.ankle_y_m.to_numpy() - out.knee_y_m.to_numpy()
    out["knee_angle_deg"] = angle_2d(thigh_x, thigh_y, shank_x, shank_y)
    out["knee_angular_velocity_deg_s"] = derivative(out.knee_angle_deg, cfg.fps)

    dx = out.attachment_x_m.to_numpy() - cfg.anchor_x_m
    dy = out.attachment_y_m.to_numpy() - cfg.anchor_y_m
    out["band_length_m"] = np.hypot(dx, dy)
    out["band_extension_m"] = np.clip(out.band_length_m - cfg.band_slack_length_m, 0.0, None)
    force_model = build_force_model(cfg)
    out["band_force_n"] = force_model(out.band_length_m.to_numpy())
    out["band_lengthening_velocity_m_s"] = derivative(out.band_length_m, cfg.fps)
    out["band_power_w_signed"] = out.band_force_n * out.band_lengthening_velocity_m_s
    out["band_power_w_positive"] = np.clip(out.band_power_w_signed, 0.0, None)

    ball = load_ball_data(ball_csv)
    if ball is not None:
        out = out.merge(ball, how="left", on="frame_index")
        out["ball_x_m"] = lowpass(out.ball_x_m.to_numpy(), cfg.fps, cfg.cutoff_hz)
        out["ball_y_m"] = lowpass(out.ball_y_m.to_numpy(), cfg.fps, cfg.cutoff_hz)
        out["ball_speed_m_s"] = vector_speed(out.ball_x_m, out.ball_y_m, cfg.fps)

    contact_frame, contact_method = detect_contact_frame(out, cfg)
    metrics = summarize_trial(out, cfg, contact_frame, contact_method)
    return out, metrics


def _window(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    return df[(df.frame_index >= start_frame) & (df.frame_index <= end_frame)]


def summarize_trial(
    df: pd.DataFrame,
    cfg: VEKConfig,
    contact_frame: int | None,
    contact_method: str,
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "vek_version": VERSION,
        "athlete_id": cfg.athlete_id,
        "athlete_name": cfg.athlete_name,
        "trial_id": cfg.trial_id,
        "condition": cfg.condition,
        "kicking_side": cfg.kicking_side,
        "fps": cfg.fps,
        "contact_frame": contact_frame,
        "contact_detection": contact_method,
        "peak_band_force_n": float(df.band_force_n.max()),
        "peak_positive_band_power_w": float(df.band_power_w_positive.max()),
        "band_work_positive_j": float(np.trapezoid(df.band_power_w_positive, df.time_s)),
        "max_band_extension_m": float(df.band_extension_m.max()),
        "peak_foot_speed_all_m_s": float(df.foot_speed_m_s.max()),
        "peak_knee_angular_velocity_deg_s": float(np.nanmax(np.abs(df.knee_angular_velocity_deg_s))),
    }

    if contact_frame is None:
        metrics["qc_status"] = "review_contact_event"
        return metrics

    pre = _window(df, contact_frame - cfg.pre_contact_frames, contact_frame)
    post = _window(df, contact_frame + 1, contact_frame + cfg.post_contact_frames)
    metrics["peak_foot_speed_pre_contact_m_s"] = float(pre.foot_speed_m_s.max()) if not pre.empty else np.nan
    metrics["band_force_at_contact_n"] = (
        float(df.loc[df.frame_index == contact_frame, "band_force_n"].iloc[0])
        if (df.frame_index == contact_frame).any()
        else np.nan
    )

    if "ball_speed_m_s" in df.columns and not post.empty:
        ball_out = float(post.ball_speed_m_s.median())
        metrics["ball_speed_out_m_s"] = ball_out
        metrics["ball_speed_out_km_h"] = ball_out * 3.6
        metrics["ball_kinetic_energy_j"] = 0.5 * cfg.ball_mass_kg * ball_out**2
        metrics["estimated_ball_impulse_n_s"] = cfg.ball_mass_kg * ball_out
        foot_speed = metrics.get("peak_foot_speed_pre_contact_m_s", np.nan)
        metrics["ball_to_foot_speed_ratio"] = (
            ball_out / foot_speed if np.isfinite(foot_speed) and foot_speed > 0 else np.nan
        )
        if cfg.contact_duration_frames:
            contact_time = cfg.contact_duration_frames / cfg.fps
            metrics["contact_time_s"] = contact_time
            metrics["estimated_mean_ball_force_n"] = metrics["estimated_ball_impulse_n_s"] / contact_time
            metrics["apparent_ball_power_w"] = metrics["ball_kinetic_energy_j"] / contact_time

    low_confidence = "low_confidence" in contact_method
    metrics["qc_status"] = "review" if low_confidence else "ok"
    return metrics


# -----------------------------------------------------------------------------
# Outputs
# -----------------------------------------------------------------------------


def save_plots(df: pd.DataFrame, output_dir: Path, stem: str, contact_frame: int | None) -> list[Path]:
    paths: list[Path] = []

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.time_s, df.band_force_n, label="Band force (N)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Force (N)")
    ax.grid(True, alpha=0.3)
    if contact_frame is not None and (df.frame_index == contact_frame).any():
        t_contact = float(df.loc[df.frame_index == contact_frame, "time_s"].iloc[0])
        ax.axvline(t_contact, linestyle="--", label="Foot-ball contact")
    ax.legend()
    path = output_dir / f"{stem}_band_force.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.time_s, df.foot_speed_m_s, label="Foot speed (m/s)")
    if "ball_speed_m_s" in df.columns:
        ax.plot(df.time_s, df.ball_speed_m_s, label="Ball speed (m/s)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Speed (m/s)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / f"{stem}_speeds.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(df.time_s, df.band_power_w_signed, label="Band power, signed (W)")
    ax.plot(df.time_s, df.band_power_w_positive, label="Positive power (W)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Power (W)")
    ax.grid(True, alpha=0.3)
    ax.legend()
    path = output_dir / f"{stem}_band_power.png"
    fig.tight_layout()
    fig.savefig(path, dpi=180)
    plt.close(fig)
    paths.append(path)

    return paths


def write_html_report(metrics: dict[str, Any], plot_paths: list[Path], report_path: Path) -> None:
    rows = []
    for key, value in metrics.items():
        if isinstance(value, float):
            value_text = "" if not np.isfinite(value) else f"{value:.4f}"
        else:
            value_text = str(value)
        rows.append(f"<tr><th>{html.escape(str(key))}</th><td>{html.escape(value_text)}</td></tr>")

    images = "\n".join(
        f'<figure><img src="{html.escape(path.name)}" alt="{html.escape(path.stem)}">'
        f"<figcaption>{html.escape(path.stem)}</figcaption></figure>"
        for path in plot_paths
    )
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>vailá-ElasticKick report</title>
<style>
body {{ font-family: Arial, sans-serif; max-width: 1100px; margin: 2rem auto; color: #172033; }}
h1 {{ margin-bottom: 0.2rem; }}
.notice {{ background: #fff6d8; border-left: 5px solid #d79d00; padding: 1rem; }}
table {{ border-collapse: collapse; width: 100%; margin: 1rem 0 2rem; }}
th, td {{ border-bottom: 1px solid #ddd; padding: 0.55rem; text-align: left; }}
th {{ width: 48%; }}
img {{ max-width: 100%; height: auto; }}
figure {{ margin: 2rem 0; }}
</style>
</head>
<body>
<h1>vailá-ElasticKick — VEK</h1>
<p>Version {VERSION}</p>
<div class="notice"><strong>Interpretation:</strong> Band force and band power are estimates based on the calibrated force–length relation. They are not direct measurements of muscular force or total kicking power.</div>
<h2>Trial summary</h2>
<table>{''.join(rows)}</table>
<h2>Quality-control plots</h2>
{images}
</body>
</html>"""
    report_path.write_text(document, encoding="utf-8")


def process_vek_file(
    pose_csv: str | Path,
    config_toml: str | Path,
    output_dir: str | Path | None = None,
    ball_csv: str | Path | None = None,
) -> Path:
    cfg = load_config(config_toml)
    pose_path = Path(pose_csv).expanduser().resolve()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    root = Path(output_dir).expanduser().resolve() if output_dir else pose_path.parent
    trial_dir = root / f"vaila_elastickick_{pose_path.stem}_{timestamp}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    timeseries, metrics = analyze_trial(pose_path, cfg, ball_csv)
    stem = f"vek_{cfg.athlete_id}_{cfg.trial_id}"
    timeseries_path = trial_dir / f"{stem}_timeseries.csv"
    summary_path = trial_dir / f"{stem}_summary.csv"
    metadata_path = trial_dir / f"{stem}_metadata.json"
    report_path = trial_dir / f"{stem}_report.html"

    timeseries.to_csv(timeseries_path, index=False)
    pd.DataFrame([metrics]).to_csv(summary_path, index=False)
    metadata_path.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=False), encoding="utf-8")
    plots = save_plots(timeseries, trial_dir, stem, metrics.get("contact_frame"))
    write_html_report(metrics, plots, report_path)
    return report_path


# -----------------------------------------------------------------------------
# Minimal GUI and CLI entry points
# -----------------------------------------------------------------------------


def main_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    pose = filedialog.askopenfilename(title="Select VEK pose CSV", filetypes=[("CSV", "*.csv")])
    if not pose:
        root.destroy()
        return
    config = filedialog.askopenfilename(title="Select VEK TOML config", filetypes=[("TOML", "*.toml")])
    if not config:
        root.destroy()
        return
    ball = filedialog.askopenfilename(
        title="Select optional ball trajectory CSV (Cancel to skip)",
        filetypes=[("CSV", "*.csv")],
    )
    output = filedialog.askdirectory(title="Select output directory") or str(Path(pose).parent)

    try:
        report = process_vek_file(pose, config, output, ball or None)
        messagebox.showinfo("VEK complete", f"Report created:\n{report}")
        with contextlib.suppress(Exception):
            webbrowser.open_new_tab(report.as_uri())
    except Exception as exc:
        messagebox.showerror("VEK error", str(exc))
    finally:
        root.destroy()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vailá-ElasticKick: elastic-band resisted kick biomechanics"
    )
    parser.add_argument("-i", "--input", help="MediaPipe/markerless pose CSV")
    parser.add_argument("-c", "--config", help="VEK TOML configuration")
    parser.add_argument("--ball", help="Optional ball trajectory CSV")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--gui", action="store_true", help="Open graphical file selectors")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    if args.gui or not args.input:
        main_gui()
        return 0
    if not args.config:
        raise SystemExit("--config is required in CLI mode")
    report = process_vek_file(args.input, args.config, args.output, args.ball)
    print(f"VEK report: {report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
