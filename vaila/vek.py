"""
===============================================================================
vek.py - vaila-ElasticKick (VEK)
===============================================================================
Author: Paulo Roberto Pereira Santiago / vaila contributors
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 16 June 2026
Update Date: 16 June 2026
Version: 0.3.56
Python Version: 3.12.13

Description:
------------
Biomechanical assessment of soccer/futsal kicking under elastic resistance.

VEK combines markerless lower-limb kinematics, elastic-band force-length
calibration, ball tracking and event detection to estimate:

- elastic-band length, extension, tension, power and mechanical work;
- hip, knee, ankle and foot linear velocities;
- knee, hip and ankle angles, angular velocities and ROM;
- ball launch speed, launch angle, kinetic energy and impulse;
- exploratory transfer efficiency: ball speed / foot speed.

Scientific scope:
-----------------
Band tension and band power are estimates from an external calibration curve.
They are not inverse-dynamics estimates of muscle force or total kicking power.
For validation work, prefer a manually confirmed impact frame and a laboratory
calibration of the full measurement chain.

Usage:
------
GUI:
    python -m vaila.vek --gui

Single-trial CLI:
    python -m vaila.vek --input pose.csv --ball ball.csv \
        --band-calibration calibration.csv --config vek_config.toml --output results

Batch CLI:
    python -m vaila.vek --batch team_dir --output results

License:
--------
GNU Affero General Public License v3.0.
===============================================================================
"""

from __future__ import annotations

import argparse
import contextlib
import html
import json
import math
import webbrowser
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator
from scipy.signal import butter, sosfiltfilt

try:
    import tomllib
except ModuleNotFoundError:  # pragma: no cover
    import tomli as tomllib  # type: ignore[no-redef]

VERSION = "0.3.56"
UPDATE_DATE = "16 June 2026"

DEFAULT_CONFIG_NAME = "vek_config.toml"
DEFAULT_CALIBRATION_NAME = "vek_band_calibration.csv"
DEFAULT_HUMAN_CUTOFF_HZ = 12.0
DEFAULT_FOOT_CUTOFF_HZ = 15.0
DEFAULT_BALL_CUTOFF_HZ = 24.0
DEFAULT_BAND_CUTOFF_HZ = 12.0


@dataclass(slots=True)
class VEKConfig:
    athlete_id: str = "unknown"
    athlete_name: str = "Unknown athlete"
    body_mass_kg: float | None = None
    shank_length_m: float = 0.43
    trial_id: str = "01"
    condition: str = "resisted"
    dominant_limb: str = "right"
    kicking_limb: str = "right"
    fps: float = 240.0
    coordinate_mode: str = "normalized_shank"
    invert_y: bool = True
    image_height_px: float | None = None
    pixels_per_meter: float | None = None
    ball_mass_kg: float = 0.43
    ball_initially_stationary: bool = True
    anchor_x_m: float = 0.0
    anchor_y_m: float = 0.0
    attachment_landmark: str = "ankle"
    slack_length_m: float = 1.0
    band_k_n_per_m: float | None = None
    band_calibration_csv: str | None = None
    calibration_model: str = "pchip"
    polynomial_degree: int = 2
    human_cutoff_hz: float = DEFAULT_HUMAN_CUTOFF_HZ
    foot_cutoff_hz: float = DEFAULT_FOOT_CUTOFF_HZ
    ball_cutoff_hz: float = DEFAULT_BALL_CUTOFF_HZ
    band_cutoff_hz: float = DEFAULT_BAND_CUTOFF_HZ
    contact_frame: int | None = None
    contact_method: str = "combined"
    contact_threshold_m: float = 0.18
    pre_contact_frames: int = 10
    post_contact_frames: int = 10
    contact_duration_frames: int | None = 3
    min_contact_confidence: float = 0.55


@dataclass(slots=True)
class VEKResult:
    output_dir: Path
    report_html: Path
    timeseries_csv: Path
    summary_csv: Path
    metrics: dict[str, Any]


def _as_float(value: Any, default: float | None = None) -> float | None:
    if value is None:
        return default
    if isinstance(value, str):
        value = value.strip().replace(",", ".")
        if not value:
            return default
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _as_int_or_none(value: Any) -> int | None:
    if value is None or value == "":
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _read_toml(path: Path) -> dict[str, Any]:
    with path.open("rb") as f:
        return tomllib.load(f)


def load_config(config_path: str | Path | None = None) -> VEKConfig:
    if config_path is None:
        cfg = VEKConfig()
        validate_config(cfg)
        return cfg

    path = Path(config_path).expanduser().resolve()
    raw = _read_toml(path)

    athlete = raw.get("athlete", {})
    trial = raw.get("trial", {})
    camera = raw.get("camera", {})
    ball = raw.get("ball", {})
    band = raw.get("band", {})
    analysis = raw.get("analysis", {})
    filtering = raw.get("filter", raw.get("filtering", {}))

    calibration_csv = band.get("calibration_csv") or analysis.get("band_calibration_csv")
    if calibration_csv:
        calibration_path = Path(str(calibration_csv)).expanduser()
        if not calibration_path.is_absolute():
            calibration_path = path.parent / calibration_path
        calibration_csv = str(calibration_path.resolve())

    single_cutoff = _as_float(analysis.get("cutoff_hz"), None)
    cfg = VEKConfig(
        athlete_id=str(athlete.get("id", athlete.get("athlete_id", "unknown"))),
        athlete_name=str(athlete.get("name", athlete.get("athlete_name", "Unknown athlete"))),
        body_mass_kg=_as_float(athlete.get("body_mass_kg", athlete.get("mass_kg"))),
        shank_length_m=float(_as_float(athlete.get("shank_length_m"), 0.43)),
        trial_id=str(trial.get("id", trial.get("trial_id", "01"))),
        condition=str(trial.get("condition", "resisted")),
        dominant_limb=str(trial.get("dominant_limb", "right")).lower(),
        kicking_limb=str(trial.get("kicking_limb", trial.get("kicking_side", "right"))).lower(),
        fps=float(_as_float(camera.get("fps", analysis.get("fps")), 240.0)),
        coordinate_mode=str(camera.get("coordinate_mode", "normalized_shank")).lower(),
        invert_y=bool(camera.get("invert_y", True)),
        image_height_px=_as_float(camera.get("image_height_px")),
        pixels_per_meter=_as_float(camera.get("pixels_per_meter")),
        ball_mass_kg=float(_as_float(ball.get("mass_kg"), 0.43)),
        ball_initially_stationary=bool(ball.get("initially_stationary", True)),
        anchor_x_m=float(_as_float(band.get("anchor_x_m"), 0.0)),
        anchor_y_m=float(_as_float(band.get("anchor_y_m"), 0.0)),
        attachment_landmark=str(band.get("attachment_landmark", "ankle")).lower(),
        slack_length_m=float(_as_float(band.get("slack_length_m"), 1.0)),
        band_k_n_per_m=_as_float(band.get("k_n_per_m")),
        band_calibration_csv=calibration_csv,
        calibration_model=str(band.get("model", band.get("calibration_model", "pchip"))).lower(),
        polynomial_degree=int(band.get("polynomial_degree", 2)),
        human_cutoff_hz=float(
            _as_float(filtering.get("human_cutoff_hz"), single_cutoff or DEFAULT_HUMAN_CUTOFF_HZ)
        ),
        foot_cutoff_hz=float(
            _as_float(filtering.get("foot_cutoff_hz"), single_cutoff or DEFAULT_FOOT_CUTOFF_HZ)
        ),
        ball_cutoff_hz=float(
            _as_float(filtering.get("ball_cutoff_hz"), single_cutoff or DEFAULT_BALL_CUTOFF_HZ)
        ),
        band_cutoff_hz=float(
            _as_float(filtering.get("band_cutoff_hz"), single_cutoff or DEFAULT_BAND_CUTOFF_HZ)
        ),
        contact_frame=_as_int_or_none(analysis.get("contact_frame")),
        contact_method=str(analysis.get("contact_method", "combined")).lower(),
        contact_threshold_m=float(_as_float(analysis.get("contact_threshold_m"), 0.18)),
        pre_contact_frames=int(analysis.get("pre_contact_frames", 10)),
        post_contact_frames=int(analysis.get("post_contact_frames", 10)),
        contact_duration_frames=_as_int_or_none(analysis.get("contact_duration_frames", 3)),
        min_contact_confidence=float(_as_float(analysis.get("min_contact_confidence"), 0.55)),
    )
    validate_config(cfg)
    return cfg


def validate_config(cfg: VEKConfig) -> None:
    for name in (cfg.dominant_limb, cfg.kicking_limb):
        if name not in {"left", "right"}:
            raise ValueError("dominant_limb and kicking_limb must be 'left' or 'right'")
    if cfg.fps <= 0:
        raise ValueError("fps must be positive")
    if cfg.coordinate_mode not in {"meters", "pixels", "normalized_shank"}:
        raise ValueError("coordinate_mode must be meters, pixels, or normalized_shank")
    if cfg.coordinate_mode == "pixels" and not cfg.pixels_per_meter:
        raise ValueError("pixels_per_meter is required for pixel coordinates")
    if cfg.coordinate_mode == "pixels" and cfg.invert_y and not cfg.image_height_px:
        raise ValueError("image_height_px is required to invert pixel y coordinates")
    if cfg.slack_length_m <= 0:
        raise ValueError("band slack_length_m must be positive")
    if not cfg.band_calibration_csv and not cfg.band_k_n_per_m:
        raise ValueError("Provide band.calibration_csv or band.k_n_per_m")
    if cfg.calibration_model not in {"pchip", "linear", "polynomial"}:
        raise ValueError("band model must be pchip, linear or polynomial")
    for cutoff_name in (
        "human_cutoff_hz",
        "foot_cutoff_hz",
        "ball_cutoff_hz",
        "band_cutoff_hz",
    ):
        cutoff = getattr(cfg, cutoff_name)
        if cutoff <= 0 or cutoff >= cfg.fps / 2:
            raise ValueError(f"{cutoff_name} must be between 0 and Nyquist")


def write_default_config(path: str | Path, calibration_csv: str | Path | None = None) -> Path:
    path = Path(path).expanduser().resolve()
    cal_text = str(calibration_csv or DEFAULT_CALIBRATION_NAME)
    text = f'''# vaila-ElasticKick (VEK) example configuration

[athlete]
id = "001"
name = "Athlete"
body_mass_kg = 75.0
shank_length_m = 0.43

[trial]
id = "01"
condition = "resisted"
dominant_limb = "right"
kicking_limb = "right"

[camera]
fps = 240.0
coordinate_mode = "normalized_shank"  # meters | pixels | normalized_shank
invert_y = true
# image_height_px = 1080
# pixels_per_meter = 520.0

[band]
anchor_x_m = 0.00
anchor_y_m = 0.25
attachment_landmark = "ankle"  # ankle | heel | foot_index | foot_center
slack_length_m = 1.00
calibration_csv = "{cal_text}"
model = "pchip"  # pchip | linear | polynomial
polynomial_degree = 2
# k_n_per_m = 850.0

[ball]
mass_kg = 0.43
initially_stationary = true

[filter]
human_cutoff_hz = 12.0
foot_cutoff_hz = 15.0
ball_cutoff_hz = 24.0
band_cutoff_hz = 12.0

[analysis]
contact_method = "combined"  # manual/configured | minimum_distance | ball_acceleration | combined
contact_threshold_m = 0.18
pre_contact_frames = 10
post_contact_frames = 10
contact_duration_frames = 3
# contact_frame = 418
'''
    path.write_text(text, encoding="utf-8")
    return path


def find_frame_column(df: pd.DataFrame) -> str:
    candidates = (
        "frame_index",
        "frame",
        "Frame",
        "frames",
        "Frames",
        "frame_number",
        "Frame Number",
    )
    for name in candidates:
        if name in df.columns:
            return name
    raise ValueError("No frame column found. Expected frame_index or frame.")


def _column_candidates(landmark: str, axis: str) -> tuple[str, ...]:
    return (
        f"{landmark}_{axis}",
        f"{landmark}.{axis}",
        f"{landmark.upper()}_{axis.upper()}",
        f"{landmark}_{axis.upper()}",
        f"{landmark.capitalize()}_{axis}",
        f"{landmark.capitalize()}_{axis.upper()}",
    )


def xy_columns(df: pd.DataFrame, landmark: str) -> tuple[str, str]:
    x_candidates = _column_candidates(landmark, "x")
    y_candidates = _column_candidates(landmark, "y")
    x_col = next((c for c in x_candidates if c in df.columns), None)
    y_col = next((c for c in y_candidates if c in df.columns), None)
    if x_col and y_col:
        return x_col, y_col
    raise ValueError(f"Missing x/y coordinates for landmark '{landmark}'")


def _confidence_column(df: pd.DataFrame, landmark: str) -> str | None:
    for name in (
        f"{landmark}_visibility",
        f"{landmark}_confidence",
        f"{landmark}_score",
        f"{landmark}.visibility",
        f"{landmark}.confidence",
    ):
        if name in df.columns:
            return name
    return None


def _raw_xy(df: pd.DataFrame, landmark: str) -> tuple[np.ndarray, np.ndarray]:
    x_col, y_col = xy_columns(df, landmark)
    return df[x_col].to_numpy(float), df[y_col].to_numpy(float)


def lowpass(values: Any, fps: float, cutoff_hz: float, order: int = 4) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return arr
    valid_count = int(np.isfinite(arr).sum())
    if valid_count < max(12, order * 3):
        return pd.Series(arr).interpolate(limit_direction="both").to_numpy(float)
    filled = pd.Series(arr).interpolate(limit_direction="both").to_numpy(float)
    cutoff = min(float(cutoff_hz), fps / 2.0 * 0.95)
    sos = butter(order, cutoff, btype="lowpass", fs=fps, output="sos")
    return sosfiltfilt(sos, filled)


def derivative(values: Any, fps: float) -> np.ndarray:
    arr = pd.Series(np.asarray(values, dtype=float)).interpolate(limit_direction="both").to_numpy()
    if arr.size <= 1:
        return np.zeros_like(arr)
    return np.gradient(arr, 1.0 / fps)


def vector_speed(x: Any, y: Any, fps: float) -> np.ndarray:
    return np.hypot(derivative(x, fps), derivative(y, fps))


def angle_2d(v1x: Any, v1y: Any, v2x: Any, v2y: Any) -> np.ndarray:
    a1 = np.asarray(v1x, dtype=float)
    a2 = np.asarray(v1y, dtype=float)
    b1 = np.asarray(v2x, dtype=float)
    b2 = np.asarray(v2y, dtype=float)
    dot = a1 * b1 + a2 * b2
    norm = np.hypot(a1, a2) * np.hypot(b1, b2)
    cosine = np.divide(dot, norm, out=np.full_like(dot, np.nan), where=norm > 1e-12)
    return np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0)))


def coordinate_scale(df: pd.DataFrame, cfg: VEKConfig) -> float:
    if cfg.coordinate_mode == "meters":
        return 1.0
    if cfg.coordinate_mode == "pixels":
        assert cfg.pixels_per_meter is not None
        return 1.0 / cfg.pixels_per_meter

    knee_x, knee_y = _raw_xy(df, f"{cfg.kicking_limb}_knee")
    ankle_x, ankle_y = _raw_xy(df, f"{cfg.kicking_limb}_ankle")
    raw_shank = np.hypot(knee_x - ankle_x, knee_y - ankle_y)
    median_raw = float(np.nanmedian(raw_shank))
    if not np.isfinite(median_raw) or median_raw <= 0:
        raise ValueError("Could not estimate shank scale from pose data")
    return cfg.shank_length_m / median_raw


def calibrated_xy(
    df: pd.DataFrame,
    landmark: str,
    cfg: VEKConfig,
    scale: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = _raw_xy(df, landmark)
    if cfg.coordinate_mode == "pixels" and cfg.invert_y:
        assert cfg.image_height_px is not None
        y = cfg.image_height_px - y
    elif cfg.coordinate_mode == "normalized_shank" and cfg.invert_y:
        y = 1.0 - y
    elif cfg.coordinate_mode == "meters" and cfg.invert_y:
        y = -y
    return x * scale, y * scale


def point_xy(
    df: pd.DataFrame,
    landmark: str,
    cfg: VEKConfig,
    scale: float,
    cutoff_hz: float,
) -> tuple[np.ndarray, np.ndarray]:
    x, y = calibrated_xy(df, landmark, cfg, scale)
    return lowpass(x, cfg.fps, cutoff_hz), lowpass(y, cfg.fps, cutoff_hz)


def foot_center_xy(df: pd.DataFrame, cfg: VEKConfig, scale: float) -> tuple[np.ndarray, np.ndarray]:
    side = cfg.kicking_limb
    try:
        heel_x, heel_y = calibrated_xy(df, f"{side}_heel", cfg, scale)
        toe_x, toe_y = calibrated_xy(df, f"{side}_foot_index", cfg, scale)
        return (heel_x + toe_x) / 2.0, (heel_y + toe_y) / 2.0
    except ValueError:
        return calibrated_xy(df, f"{side}_ankle", cfg, scale)


def attachment_xy(df: pd.DataFrame, cfg: VEKConfig, scale: float) -> tuple[np.ndarray, np.ndarray]:
    if cfg.attachment_landmark == "foot_center":
        return foot_center_xy(df, cfg, scale)
    return calibrated_xy(df, f"{cfg.kicking_limb}_{cfg.attachment_landmark}", cfg, scale)


def load_calibration_table(path: str | Path) -> pd.DataFrame:
    table = pd.read_csv(Path(path).expanduser())
    required = {"length_m", "force_n"}
    if not required.issubset(table.columns):
        raise ValueError("Band calibration CSV must contain length_m and force_n columns")
    table = (
        table.loc[:, ["length_m", "force_n"]]
        .dropna()
        .groupby("length_m", as_index=False)["force_n"]
        .mean()
        .sort_values("length_m")
    )
    if len(table) < 2:
        raise ValueError("Band calibration requires at least two force-length points")
    if (table["length_m"].diff().dropna() <= 0).any():
        raise ValueError("Band calibration length_m values must increase")
    if (table["force_n"] < 0).any():
        raise ValueError("Band calibration force_n values must be non-negative")
    return table


def build_force_model(cfg: VEKConfig):
    if cfg.band_calibration_csv:
        table = load_calibration_table(cfg.band_calibration_csv)
        x = table["length_m"].to_numpy(float)
        y = table["force_n"].to_numpy(float)
        min_x = float(np.min(x))
        max_x = float(np.max(x))

        if cfg.calibration_model == "linear":
            coefficients = np.polyfit(x, y, deg=1)

            def force(length_m: Any) -> np.ndarray:
                length = np.asarray(length_m, dtype=float)
                values = np.polyval(coefficients, np.clip(length, min_x, max_x))
                values[length <= cfg.slack_length_m] = 0.0
                return np.clip(values, 0.0, None)

            return force, "linear_calibration"

        if cfg.calibration_model == "polynomial":
            degree = max(1, min(int(cfg.polynomial_degree), len(x) - 1, 5))
            coefficients = np.polyfit(x, y, deg=degree)

            def force(length_m: Any) -> np.ndarray:
                length = np.asarray(length_m, dtype=float)
                values = np.polyval(coefficients, np.clip(length, min_x, max_x))
                values[length <= cfg.slack_length_m] = 0.0
                return np.clip(values, 0.0, None)

            return force, f"polynomial_degree_{degree}"

        interpolator = None if len(table) < 3 else PchipInterpolator(x, y, extrapolate=False)

        def force(length_m: Any) -> np.ndarray:
            length = np.asarray(length_m, dtype=float)
            clipped = np.clip(length, min_x, max_x)
            if interpolator is None:
                values = np.interp(clipped, x, y)
            else:
                values = np.asarray(interpolator(clipped), dtype=float)
            values[length <= cfg.slack_length_m] = 0.0
            return np.clip(values, 0.0, None)

        return force, "pchip_calibration"

    assert cfg.band_k_n_per_m is not None

    def linear_hooke(length_m: Any) -> np.ndarray:
        length = np.asarray(length_m, dtype=float)
        extension = np.clip(length - cfg.slack_length_m, 0.0, None)
        return cfg.band_k_n_per_m * extension

    return linear_hooke, "linear_stiffness_fallback"


def load_ball_data(ball_csv: str | Path | None, cfg: VEKConfig, scale: float) -> pd.DataFrame | None:
    if not ball_csv:
        return None
    ball = pd.read_csv(Path(ball_csv).expanduser())
    frame_col = find_frame_column(ball)
    ball = ball.rename(columns={frame_col: "frame_index"}).copy()
    if {"ball_x_m", "ball_y_m"}.issubset(ball.columns):
        x = ball["ball_x_m"].to_numpy(float)
        y = ball["ball_y_m"].to_numpy(float)
    elif {"ball_x", "ball_y"}.issubset(ball.columns):
        x = ball["ball_x"].to_numpy(float)
        y = ball["ball_y"].to_numpy(float)
        if cfg.coordinate_mode == "pixels" and cfg.invert_y:
            assert cfg.image_height_px is not None
            y = cfg.image_height_px - y
        elif cfg.coordinate_mode == "normalized_shank" and cfg.invert_y:
            y = 1.0 - y
        elif cfg.coordinate_mode == "meters" and cfg.invert_y:
            y = -y
        x = x * scale
        y = y * scale
    else:
        raise ValueError("Ball CSV must contain ball_x/ball_y or ball_x_m/ball_y_m")

    out = pd.DataFrame({"frame_index": ball["frame_index"].astype(int), "ball_x_m": x, "ball_y_m": y})
    if "ball_z" in ball.columns:
        out["ball_z_m"] = ball["ball_z"].to_numpy(float) * (1.0 if "ball_z_m" in ball.columns else scale)
    if "confidence" in ball.columns:
        out["ball_confidence"] = ball["confidence"].to_numpy(float)
    elif "ball_confidence" in ball.columns:
        out["ball_confidence"] = ball["ball_confidence"].to_numpy(float)
    return out.sort_values("frame_index").reset_index(drop=True)


def _normalise_score(values: np.ndarray, higher_is_better: bool = True) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    finite = np.isfinite(arr)
    out = np.zeros_like(arr, dtype=float)
    if finite.sum() == 0:
        return out
    lo = float(np.nanmin(arr[finite]))
    hi = float(np.nanmax(arr[finite]))
    if math.isclose(lo, hi):
        out[finite] = 1.0
        return out
    scaled = (arr - lo) / (hi - lo)
    if not higher_is_better:
        scaled = 1.0 - scaled
    out[finite] = np.clip(scaled[finite], 0.0, 1.0)
    return out


def detect_contact_event(df: pd.DataFrame, cfg: VEKConfig) -> tuple[int | None, str, float]:
    if cfg.contact_frame is not None:
        return cfg.contact_frame, "manual_configured", 1.0
    if not {"foot_x_m", "foot_y_m", "ball_x_m", "ball_y_m"}.issubset(df.columns):
        return None, "not_available", 0.0

    distance = np.hypot(df["foot_x_m"] - df["ball_x_m"], df["foot_y_m"] - df["ball_y_m"]).to_numpy(float)
    ball_speed = df.get("ball_speed_m_s", pd.Series(np.zeros(len(df)))).to_numpy(float)
    foot_speed = df.get("foot_speed_m_s", pd.Series(np.zeros(len(df)))).to_numpy(float)
    ball_acc = np.clip(derivative(ball_speed, cfg.fps), 0.0, None)
    foot_decel = np.clip(-derivative(foot_speed, cfg.fps), 0.0, None)
    tracking_conf = df.get("ball_confidence", pd.Series(np.ones(len(df)))).to_numpy(float)

    method = cfg.contact_method
    if method in {"manual", "configured"}:
        return None, "manual_not_configured", 0.0
    if method == "minimum_distance":
        idx = int(np.nanargmin(distance))
        confidence = 1.0 if distance[idx] <= cfg.contact_threshold_m else 0.45
        return int(df.iloc[idx]["frame_index"]), "minimum_distance", confidence
    if method == "ball_acceleration":
        idx = int(np.nanargmax(ball_acc))
        return int(df.iloc[idx]["frame_index"]), "ball_acceleration_onset", 0.65

    distance_score = _normalise_score(distance, higher_is_better=False)
    ball_acc_score = _normalise_score(ball_acc, higher_is_better=True)
    foot_decel_score = _normalise_score(foot_decel, higher_is_better=True)
    conf_score = np.clip(np.nan_to_num(tracking_conf, nan=0.5), 0.0, 1.0)
    combined = 0.45 * distance_score + 0.30 * ball_acc_score + 0.15 * foot_decel_score + 0.10 * conf_score
    idx = int(np.nanargmax(combined))
    confidence = float(np.clip(combined[idx], 0.0, 1.0))
    label = "combined_confidence"
    if distance[idx] > cfg.contact_threshold_m:
        label += "_distance_review"
        confidence = min(confidence, 0.55)
    return int(df.iloc[idx]["frame_index"]), label, confidence


def _rom(values: pd.Series) -> float:
    arr = values.to_numpy(float)
    if np.isfinite(arr).sum() == 0:
        return float("nan")
    return float(np.nanmax(arr) - np.nanmin(arr))


def _at_frame(df: pd.DataFrame, frame: int | None, col: str) -> float:
    if frame is None or col not in df.columns:
        return float("nan")
    row = df.loc[df["frame_index"] == frame, col]
    if row.empty:
        idx = int((df["frame_index"] - frame).abs().idxmin())
        return float(df.loc[idx, col])
    return float(row.iloc[0])


def _window(df: pd.DataFrame, start_frame: int, end_frame: int) -> pd.DataFrame:
    return df[(df["frame_index"] >= start_frame) & (df["frame_index"] <= end_frame)]


def compute_qc(pose: pd.DataFrame, df: pd.DataFrame, cfg: VEKConfig, contact_conf: float) -> dict[str, Any]:
    side = cfg.kicking_limb
    landmarks = [
        f"{side}_hip",
        f"{side}_knee",
        f"{side}_ankle",
        f"{side}_heel",
        f"{side}_foot_index",
    ]
    coordinate_cols: list[str] = []
    confidence_values: list[np.ndarray] = []
    for lm in landmarks:
        with contextlib.suppress(ValueError):
            x_col, y_col = xy_columns(pose, lm)
            coordinate_cols.extend([x_col, y_col])
        conf_col = _confidence_column(pose, lm)
        if conf_col:
            confidence_values.append(pose[conf_col].to_numpy(float))

    if coordinate_cols:
        missing_pct = float(pose[coordinate_cols].isna().mean().mean() * 100.0)
    else:
        missing_pct = float("nan")
    tracking_conf = (
        float(np.nanmean(np.concatenate(confidence_values))) if confidence_values else float("nan")
    )
    foot_acc = np.abs(derivative(df["foot_speed_m_s"], cfg.fps))
    ball_acc = np.abs(derivative(df["ball_speed_m_s"], cfg.fps)) if "ball_speed_m_s" in df else np.array([])
    velocity_discontinuity = float(np.nanpercentile(foot_acc, 99)) if foot_acc.size else float("nan")
    if ball_acc.size:
        velocity_discontinuity = max(velocity_discontinuity, float(np.nanpercentile(ball_acc, 99)))

    calibration_valid = bool(cfg.band_calibration_csv or cfg.band_k_n_per_m)
    fail_reasons = []
    if np.isfinite(missing_pct) and missing_pct > 20:
        fail_reasons.append("missing_pose_data")
    if np.isfinite(tracking_conf) and tracking_conf < 0.50:
        fail_reasons.append("low_pose_confidence")
    if contact_conf < cfg.min_contact_confidence:
        fail_reasons.append("contact_event_review")
    if not calibration_valid:
        fail_reasons.append("band_calibration_missing")

    return {
        "missing_data_pct": missing_pct,
        "interpolation_pct": missing_pct,
        "tracking_confidence_mean": tracking_conf,
        "contact_confidence": contact_conf,
        "calibration_valid": calibration_valid,
        "velocity_discontinuity_indicator": velocity_discontinuity,
        "qc_status": "fail" if fail_reasons else "pass",
        "qc_notes": ";".join(fail_reasons) if fail_reasons else "ok",
    }


def analyze_trial(
    pose_csv: str | Path,
    cfg: VEKConfig,
    ball_csv: str | Path | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    pose_path = Path(pose_csv).expanduser().resolve()
    pose = pd.read_csv(pose_path)
    frame_col = find_frame_column(pose)
    pose = pose.rename(columns={frame_col: "frame_index"}).sort_values("frame_index").reset_index(drop=True)
    pose["frame_index"] = pose["frame_index"].astype(int)

    scale = coordinate_scale(pose, cfg)
    side = cfg.kicking_limb
    out = pd.DataFrame({"frame_index": pose["frame_index"]})
    out["time_s"] = (out["frame_index"] - out["frame_index"].iloc[0]) / cfg.fps

    for landmark, prefix, cutoff in (
        (f"{side}_hip", "hip", cfg.human_cutoff_hz),
        (f"{side}_knee", "knee", cfg.human_cutoff_hz),
        (f"{side}_ankle", "ankle", cfg.human_cutoff_hz),
    ):
        x, y = point_xy(pose, landmark, cfg, scale, cutoff)
        out[f"{prefix}_x_m"] = x
        out[f"{prefix}_y_m"] = y

    foot_x, foot_y = foot_center_xy(pose, cfg, scale)
    attach_x, attach_y = attachment_xy(pose, cfg, scale)
    out["foot_x_m"] = lowpass(foot_x, cfg.fps, cfg.foot_cutoff_hz)
    out["foot_y_m"] = lowpass(foot_y, cfg.fps, cfg.foot_cutoff_hz)
    out["attachment_x_m"] = lowpass(attach_x, cfg.fps, cfg.band_cutoff_hz)
    out["attachment_y_m"] = lowpass(attach_y, cfg.fps, cfg.band_cutoff_hz)

    for prefix in ("hip", "knee", "ankle", "foot"):
        out[f"{prefix}_speed_m_s"] = vector_speed(out[f"{prefix}_x_m"], out[f"{prefix}_y_m"], cfg.fps)

    thigh_from_hip_x = out["hip_x_m"].to_numpy() - out["knee_x_m"].to_numpy()
    thigh_from_hip_y = out["hip_y_m"].to_numpy() - out["knee_y_m"].to_numpy()
    shank_to_ankle_x = out["ankle_x_m"].to_numpy() - out["knee_x_m"].to_numpy()
    shank_to_ankle_y = out["ankle_y_m"].to_numpy() - out["knee_y_m"].to_numpy()
    foot_vec_x = out["foot_x_m"].to_numpy() - out["ankle_x_m"].to_numpy()
    foot_vec_y = out["foot_y_m"].to_numpy() - out["ankle_y_m"].to_numpy()
    trunk_ref_x = np.zeros(len(out))
    trunk_ref_y = np.ones(len(out))

    out["knee_angle_deg"] = angle_2d(thigh_from_hip_x, thigh_from_hip_y, shank_to_ankle_x, shank_to_ankle_y)
    out["hip_angle_deg"] = angle_2d(-thigh_from_hip_x, -thigh_from_hip_y, trunk_ref_x, trunk_ref_y)
    out["ankle_angle_deg"] = angle_2d(-shank_to_ankle_x, -shank_to_ankle_y, foot_vec_x, foot_vec_y)
    out["trunk_inclination_deg"] = np.nan
    for joint in ("knee", "hip", "ankle"):
        out[f"{joint}_angular_velocity_deg_s"] = derivative(out[f"{joint}_angle_deg"], cfg.fps)

    dx = out["attachment_x_m"].to_numpy() - cfg.anchor_x_m
    dy = out["attachment_y_m"].to_numpy() - cfg.anchor_y_m
    out["band_length_m"] = lowpass(np.hypot(dx, dy), cfg.fps, cfg.band_cutoff_hz)
    out["band_extension_m"] = np.clip(out["band_length_m"] - cfg.slack_length_m, 0.0, None)
    force_model, calibration_method = build_force_model(cfg)
    out["band_force_n"] = force_model(out["band_length_m"].to_numpy())
    out["band_extension_velocity_m_s"] = derivative(out["band_length_m"], cfg.fps)
    out["band_power_w"] = out["band_force_n"] * out["band_extension_velocity_m_s"]
    out["band_power_positive_w"] = np.clip(out["band_power_w"], 0.0, None)
    out["band_power_negative_w"] = np.clip(out["band_power_w"], None, 0.0)
    out["band_force_rate_n_s"] = derivative(out["band_force_n"], cfg.fps)

    ball = load_ball_data(ball_csv, cfg, scale)
    if ball is not None:
        out = out.merge(ball, how="left", on="frame_index")
        out["ball_x_m"] = lowpass(out["ball_x_m"], cfg.fps, cfg.ball_cutoff_hz)
        out["ball_y_m"] = lowpass(out["ball_y_m"], cfg.fps, cfg.ball_cutoff_hz)
        out["ball_vx_m_s"] = derivative(out["ball_x_m"], cfg.fps)
        out["ball_vy_m_s"] = derivative(out["ball_y_m"], cfg.fps)
        out["ball_speed_m_s"] = np.hypot(out["ball_vx_m_s"], out["ball_vy_m_s"])
        out["ball_acceleration_m_s2"] = derivative(out["ball_speed_m_s"], cfg.fps)

    contact_frame, contact_method, contact_conf = detect_contact_event(out, cfg)
    out["is_contact_frame"] = out["frame_index"] == contact_frame if contact_frame is not None else False
    metrics = summarize_trial(
        out,
        pose,
        cfg,
        pose_path=pose_path,
        ball_csv=Path(ball_csv).expanduser().resolve() if ball_csv else None,
        contact_frame=contact_frame,
        contact_method=contact_method,
        contact_confidence=contact_conf,
        coordinate_scale=scale,
        calibration_method=calibration_method,
    )
    return out, metrics


def summarize_trial(
    df: pd.DataFrame,
    pose: pd.DataFrame,
    cfg: VEKConfig,
    *,
    pose_path: Path,
    ball_csv: Path | None,
    contact_frame: int | None,
    contact_method: str,
    contact_confidence: float,
    coordinate_scale: float,
    calibration_method: str,
) -> dict[str, Any]:
    positive_work = float(np.trapezoid(df["band_power_positive_w"], df["time_s"]))
    negative_work = float(np.trapezoid(df["band_power_negative_w"], df["time_s"]))
    metrics: dict[str, Any] = {
        "vek_version": VERSION,
        "update_date": UPDATE_DATE,
        "athlete_id": cfg.athlete_id,
        "athlete_name": cfg.athlete_name,
        "trial_id": cfg.trial_id,
        "condition": cfg.condition,
        "dominant_limb": cfg.dominant_limb,
        "kicking_limb": cfg.kicking_limb,
        "limb_is_dominant": cfg.dominant_limb == cfg.kicking_limb,
        "pose_csv": str(pose_path),
        "ball_csv": str(ball_csv) if ball_csv else "",
        "fps": cfg.fps,
        "coordinate_mode": cfg.coordinate_mode,
        "coordinate_scale_m_per_unit": coordinate_scale,
        "calibration_method": calibration_method,
        "contact_frame": contact_frame,
        "contact_detection_method": contact_method,
        "contact_confidence": contact_confidence,
        "peak_band_tension_n": float(np.nanmax(df["band_force_n"])),
        "tension_at_impact_n": _at_frame(df, contact_frame, "band_force_n"),
        "maximum_extension_m": float(np.nanmax(df["band_extension_m"])),
        "extension_at_impact_m": _at_frame(df, contact_frame, "band_extension_m"),
        "peak_band_power_w": float(np.nanmax(df["band_power_positive_w"])),
        "mean_positive_band_power_w": float(np.nanmean(df["band_power_positive_w"])),
        "positive_band_work_j": positive_work,
        "negative_band_work_j": negative_work,
        "loading_rate_peak_n_s": float(np.nanmax(df["band_force_rate_n_s"])),
        "unloading_rate_peak_n_s": float(np.nanmin(df["band_force_rate_n_s"])),
        "peak_hip_speed_m_s": float(np.nanmax(df["hip_speed_m_s"])),
        "peak_knee_speed_m_s": float(np.nanmax(df["knee_speed_m_s"])),
        "peak_ankle_speed_m_s": float(np.nanmax(df["ankle_speed_m_s"])),
        "peak_foot_velocity_m_s": float(np.nanmax(df["foot_speed_m_s"])),
        "foot_velocity_at_impact_m_s": _at_frame(df, contact_frame, "foot_speed_m_s"),
        "peak_knee_angular_velocity_deg_s": float(np.nanmax(np.abs(df["knee_angular_velocity_deg_s"]))),
        "peak_hip_angular_velocity_deg_s": float(np.nanmax(np.abs(df["hip_angular_velocity_deg_s"]))),
        "peak_ankle_angular_velocity_deg_s": float(np.nanmax(np.abs(df["ankle_angular_velocity_deg_s"]))),
        "knee_angle_at_impact_deg": _at_frame(df, contact_frame, "knee_angle_deg"),
        "hip_angle_at_impact_deg": _at_frame(df, contact_frame, "hip_angle_deg"),
        "ankle_angle_at_impact_deg": _at_frame(df, contact_frame, "ankle_angle_deg"),
        "knee_rom_deg": _rom(df["knee_angle_deg"]),
        "hip_rom_deg": _rom(df["hip_angle_deg"]),
        "ankle_rom_deg": _rom(df["ankle_angle_deg"]),
    }

    if contact_frame is not None:
        pre = _window(df, contact_frame - cfg.pre_contact_frames, contact_frame)
        post = _window(df, contact_frame + 1, contact_frame + cfg.post_contact_frames)
    else:
        pre = df
        post = df.iloc[0:0]

    metrics["peak_foot_velocity_pre_contact_m_s"] = (
        float(np.nanmax(pre["foot_speed_m_s"])) if not pre.empty else float("nan")
    )

    if "ball_speed_m_s" in df.columns:
        launch_source = post if not post.empty else df
        launch_idx = int(launch_source["ball_speed_m_s"].idxmax())
        launch_vx = float(df.loc[launch_idx, "ball_vx_m_s"])
        launch_vy = float(df.loc[launch_idx, "ball_vy_m_s"])
        launch_speed = float(df.loc[launch_idx, "ball_speed_m_s"])
        foot_speed = metrics["foot_velocity_at_impact_m_s"]
        if not np.isfinite(foot_speed) or foot_speed <= 0:
            foot_speed = metrics["peak_foot_velocity_pre_contact_m_s"]
        contact_time = (
            cfg.contact_duration_frames / cfg.fps
            if cfg.contact_duration_frames and cfg.contact_duration_frames > 0
            else float("nan")
        )
        ball_energy = 0.5 * cfg.ball_mass_kg * launch_speed**2
        ball_impulse = cfg.ball_mass_kg * launch_speed
        metrics.update(
            {
                "ball_peak_velocity_m_s": float(np.nanmax(df["ball_speed_m_s"])),
                "ball_launch_velocity_m_s": launch_speed,
                "ball_launch_velocity_km_h": launch_speed * 3.6,
                "ball_launch_angle_deg": math.degrees(math.atan2(launch_vy, launch_vx)),
                "ball_kinetic_energy_j": ball_energy,
                "ball_impulse_n_s": ball_impulse,
                "apparent_ball_power_w": ball_energy / contact_time
                if np.isfinite(contact_time) and contact_time > 0
                else float("nan"),
                "ball_to_foot_speed_ratio": launch_speed / foot_speed
                if np.isfinite(foot_speed) and foot_speed > 0
                else float("nan"),
                "contact_time_s": contact_time,
            }
        )

    metrics.update(compute_qc(pose, df, cfg, contact_confidence))
    return metrics


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _metric_text(value: Any, digits: int = 2, suffix: str = "") -> str:
    if value is None:
        return "NA"
    if isinstance(value, float):
        if not np.isfinite(value):
            return "NA"
        return f"{value:.{digits}f}{suffix}"
    return html.escape(str(value))


def _save_line_plot(
    df: pd.DataFrame,
    path: Path,
    y_columns: list[str],
    labels: list[str],
    ylabel: str,
    title: str,
    contact_frame: int | None,
) -> Path:
    fig, ax = plt.subplots(figsize=(10, 4.8))
    for col, label in zip(y_columns, labels, strict=False):
        if col in df.columns:
            ax.plot(df["time_s"], df[col], label=label, linewidth=1.8)
    if contact_frame is not None and (df["frame_index"] == contact_frame).any():
        t_contact = float(df.loc[df["frame_index"] == contact_frame, "time_s"].iloc[0])
        ax.axvline(t_contact, color="#d62728", linestyle="--", linewidth=1.3, label="impact")
    ax.set_title(title)
    ax.set_xlabel("Time (s)")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def save_plots(df: pd.DataFrame, output_dir: Path, stem: str, contact_frame: int | None) -> list[Path]:
    plots = [
        _save_line_plot(
            df,
            output_dir / f"{stem}_band_force.png",
            ["band_force_n"],
            ["Band tension"],
            "Force (N)",
            "Elastic-band tension",
            contact_frame,
        ),
        _save_line_plot(
            df,
            output_dir / f"{stem}_band_power.png",
            ["band_power_w", "band_power_positive_w"],
            ["Signed power", "Positive power"],
            "Power (W)",
            "Elastic-band power",
            contact_frame,
        ),
        _save_line_plot(
            df,
            output_dir / f"{stem}_band_length.png",
            ["band_length_m", "band_extension_m"],
            ["Band length", "Extension"],
            "Length (m)",
            "Elastic-band length and extension",
            contact_frame,
        ),
        _save_line_plot(
            df,
            output_dir / f"{stem}_foot_velocity.png",
            ["foot_speed_m_s"],
            ["Foot velocity"],
            "Velocity (m/s)",
            "Foot velocity",
            contact_frame,
        ),
        _save_line_plot(
            df,
            output_dir / f"{stem}_ball_velocity.png",
            ["ball_speed_m_s"],
            ["Ball velocity"],
            "Velocity (m/s)",
            "Ball velocity",
            contact_frame,
        ),
        _save_line_plot(
            df,
            output_dir / f"{stem}_knee_angle.png",
            ["knee_angle_deg"],
            ["Knee angle"],
            "Angle (deg)",
            "Knee angle",
            contact_frame,
        ),
    ]

    if {"foot_x_m", "foot_y_m", "ball_x_m", "ball_y_m"}.issubset(df.columns):
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.plot(df["foot_x_m"], df["foot_y_m"], label="foot", linewidth=1.5)
        ax.plot(df["ball_x_m"], df["ball_y_m"], label="ball", linewidth=1.5)
        if contact_frame is not None and (df["frame_index"] == contact_frame).any():
            row = df.loc[df["frame_index"] == contact_frame].iloc[0]
            ax.scatter([row["foot_x_m"]], [row["foot_y_m"]], s=60, label="impact foot")
            ax.scatter([row["ball_x_m"]], [row["ball_y_m"]], s=60, label="impact ball")
        ax.set_aspect("equal", adjustable="datalim")
        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_title("Contact detection trajectory")
        ax.grid(True, alpha=0.25)
        ax.legend()
        path = output_dir / f"{stem}_contact_detection.png"
        fig.tight_layout()
        fig.savefig(path, dpi=170)
        plt.close(fig)
        plots.append(path)
    return [p for p in plots if p.exists()]


def write_html_report(metrics: dict[str, Any], plots: list[Path], report_path: Path) -> None:
    kpis = [
        ("Peak Band Tension", "peak_band_tension_n", " N"),
        ("Peak Band Power", "peak_band_power_w", " W"),
        ("Foot Velocity", "foot_velocity_at_impact_m_s", " m/s"),
        ("Ball Velocity", "ball_launch_velocity_m_s", " m/s"),
        ("Positive Work", "positive_band_work_j", " J"),
        ("Ball Impulse", "ball_impulse_n_s", " N.s"),
    ]
    cards = "\n".join(
        f'<div class="card"><div class="label">{html.escape(label)}</div>'
        f'<div class="value">{_metric_text(metrics.get(key), 2, suffix)}</div></div>'
        for label, key, suffix in kpis
    )
    rows = "\n".join(
        f"<tr><th>{html.escape(str(k))}</th><td>{_metric_text(v, 4)}</td></tr>"
        for k, v in metrics.items()
    )
    images = "\n".join(
        f'<figure><img src="{html.escape(p.name)}" alt="{html.escape(p.stem)}">'
        f"<figcaption>{html.escape(p.stem)}</figcaption></figure>"
        for p in plots
    )
    qc_class = "pass" if metrics.get("qc_status") == "pass" else "review"
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>VEK Report - {html.escape(str(metrics.get("athlete_id", "")))}</title>
<style>
body {{ margin: 0; font-family: Arial, sans-serif; background: #f5f7fb; color: #172033; }}
header {{ background: #152238; color: #fff; padding: 28px 34px; }}
header h1 {{ margin: 0 0 6px; font-size: 30px; }}
main {{ max-width: 1180px; margin: 0 auto; padding: 24px; }}
.notice {{ background: #fff6d8; border-left: 5px solid #d79d00; padding: 12px 14px; margin: 16px 0; }}
.cards {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(165px, 1fr)); gap: 12px; }}
.card {{ background: #fff; border: 1px solid #dde3ef; border-radius: 8px; padding: 14px; }}
.label {{ color: #526070; font-size: 13px; text-transform: uppercase; letter-spacing: .04em; }}
.value {{ font-size: 25px; font-weight: 700; margin-top: 8px; }}
.qc {{ display: inline-block; padding: 5px 10px; border-radius: 999px; font-weight: 700; }}
.pass {{ background: #e9f8ef; color: #146c2e; }}
.review {{ background: #fff0e1; color: #98530a; }}
section {{ margin-top: 24px; }}
table {{ border-collapse: collapse; width: 100%; background: #fff; }}
th, td {{ border-bottom: 1px solid #e3e7ef; padding: 9px 10px; text-align: left; }}
th {{ width: 42%; color: #344054; }}
figure {{ background: #fff; border: 1px solid #dde3ef; border-radius: 8px; padding: 10px; margin: 14px 0; }}
img {{ max-width: 100%; height: auto; display: block; }}
figcaption {{ margin-top: 8px; color: #526070; font-size: 14px; }}
</style>
</head>
<body>
<header>
<h1>vaila-ElasticKick (VEK)</h1>
<div>{html.escape(str(metrics.get("athlete_name", "")))} | Trial {html.escape(str(metrics.get("trial_id", "")))} | {html.escape(str(metrics.get("condition", "")))}</div>
</header>
<main>
<div class="notice"><strong>Scientific scope:</strong> band force and power are estimated from calibration data; they are not direct muscle force or total kicking power.</div>
<p>Quality control: <span class="qc {qc_class}">{html.escape(str(metrics.get("qc_status", "review")).upper())}</span> {html.escape(str(metrics.get("qc_notes", "")))}</p>
<section class="cards">{cards}</section>
<section><h2>Trend Plots</h2>{images}</section>
<section><h2>Trial Metrics</h2><table>{rows}</table></section>
</main>
</body>
</html>
"""
    report_path.write_text(document, encoding="utf-8")


def process_vek_file(
    pose_csv: str | Path,
    config_toml: str | Path,
    output_dir: str | Path | None = None,
    ball_csv: str | Path | None = None,
    band_calibration_csv: str | Path | None = None,
) -> VEKResult:
    cfg = load_config(config_toml)
    if band_calibration_csv:
        cfg.band_calibration_csv = str(Path(band_calibration_csv).expanduser().resolve())
        validate_config(cfg)

    pose_path = Path(pose_csv).expanduser().resolve()
    root = Path(output_dir).expanduser().resolve() if output_dir else pose_path.parent
    trial_dir = root / f"vaila_vek_{pose_path.stem}_{_timestamp()}"
    trial_dir.mkdir(parents=True, exist_ok=True)

    timeseries, metrics = analyze_trial(pose_path, cfg, ball_csv)
    stem = f"vek_{cfg.athlete_id}_{cfg.trial_id}"
    timeseries_csv = trial_dir / f"{stem}_timeseries.csv"
    summary_csv = trial_dir / f"{stem}_summary.csv"
    report_html = trial_dir / f"{stem}_report.html"
    metadata_json = trial_dir / f"{stem}_metadata.json"

    timeseries.to_csv(timeseries_csv, index=False)
    pd.DataFrame([metrics]).to_csv(summary_csv, index=False)
    metadata_json.write_text(json.dumps(asdict(cfg), indent=2, ensure_ascii=True), encoding="utf-8")
    plots = save_plots(timeseries, trial_dir, stem, metrics.get("contact_frame"))
    write_html_report(metrics, plots, report_html)
    return VEKResult(trial_dir, report_html, timeseries_csv, summary_csv, metrics)


def _find_first(parent: Path, patterns: tuple[str, ...]) -> Path | None:
    for pattern in patterns:
        found = sorted(parent.glob(pattern))
        if found:
            return found[0]
    return None


def process_team_batch(input_dir: str | Path, output_dir: str | Path | None = None) -> Path:
    root = Path(input_dir).expanduser().resolve()
    if not root.exists():
        raise FileNotFoundError(root)
    out_root = (
        Path(output_dir).expanduser().resolve() if output_dir else root / f"vaila_vek_team_{_timestamp()}"
    )
    out_root.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for athlete_dir in sorted(p for p in root.iterdir() if p.is_dir()):
        cfg_path = _find_first(athlete_dir, (DEFAULT_CONFIG_NAME, "*vek*.toml", "*.toml"))
        pose_path = _find_first(athlete_dir, ("*pose*.csv", "*markerless*.csv", "*.csv"))
        ball_path = _find_first(athlete_dir, ("*ball*.csv",))
        cal_path = _find_first(athlete_dir, (DEFAULT_CALIBRATION_NAME, "*calibration*.csv"))
        if not cfg_path or not pose_path:
            continue
        with contextlib.suppress(Exception):
            result = process_vek_file(
                pose_path,
                cfg_path,
                out_root / athlete_dir.name,
                ball_csv=ball_path,
                band_calibration_csv=cal_path,
            )
            summaries.append(result.metrics)

    if not summaries:
        raise ValueError("No athlete folders with VEK config and pose CSV were processed")

    team = pd.DataFrame(summaries)
    team_csv = out_root / "vek_team_summary.csv"
    team.to_csv(team_csv, index=False)

    rankings = team.copy()
    rank_cols = [
        c
        for c in ("ball_launch_velocity_m_s", "peak_foot_velocity_m_s", "positive_band_work_j")
        if c in rankings.columns
    ]
    if rank_cols:
        rankings = rankings.sort_values(rank_cols[0], ascending=False)
    rankings.to_csv(out_root / "vek_team_rankings.csv", index=False)

    asym_cols = [
        "athlete_id",
        "athlete_name",
        "kicking_limb",
        "limb_is_dominant",
        "ball_launch_velocity_m_s",
        "peak_foot_velocity_m_s",
        "positive_band_work_j",
    ]
    team[[c for c in asym_cols if c in team.columns]].to_csv(out_root / "vek_limb_comparison.csv", index=False)

    report_path = out_root / "vek_team_report.html"
    write_team_report(team, report_path)
    return report_path


def _cv(values: pd.Series) -> float:
    arr = values.dropna().to_numpy(float)
    if arr.size < 2 or np.nanmean(arr) == 0:
        return float("nan")
    return float(np.nanstd(arr, ddof=1) / np.nanmean(arr) * 100.0)


def write_team_report(team: pd.DataFrame, report_path: Path) -> None:
    metrics = ["ball_launch_velocity_m_s", "peak_foot_velocity_m_s", "positive_band_work_j"]
    rows = []
    for metric in metrics:
        if metric in team.columns:
            rows.append(
                f"<tr><th>{metric}</th><td>{_metric_text(float(team[metric].max()), 3)}</td>"
                f"<td>{_metric_text(float(team[metric].mean()), 3)}</td>"
                f"<td>{_metric_text(_cv(team[metric]), 2, '%')}</td></tr>"
            )
    ranking_rows = "\n".join(
        f"<tr><td>{html.escape(str(row.get('athlete_id', '')))}</td>"
        f"<td>{html.escape(str(row.get('athlete_name', '')))}</td>"
        f"<td>{html.escape(str(row.get('kicking_limb', '')))}</td>"
        f"<td>{_metric_text(row.get('ball_launch_velocity_m_s'), 2)}</td></tr>"
        for _, row in team.head(30).iterrows()
    )
    report_path.write_text(
        f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><title>VEK Team Report</title>
<style>body{{font-family:Arial,sans-serif;max-width:1100px;margin:24px auto;color:#172033}}
table{{border-collapse:collapse;width:100%}}th,td{{border-bottom:1px solid #ddd;padding:8px;text-align:left}}</style>
</head><body><h1>VEK Team Report</h1>
<p>Trials processed: {len(team)}</p>
<h2>Trial Comparison</h2><table><tr><th>Metric</th><th>Best</th><th>Mean</th><th>CV</th></tr>{''.join(rows)}</table>
<h2>Rankings</h2><table><tr><th>Athlete ID</th><th>Name</th><th>Limb</th><th>Ball velocity (m/s)</th></tr>{ranking_rows}</table>
</body></html>""",
        encoding="utf-8",
    )


def preview_signals(pose_csv: str | Path, config_toml: str | Path, ball_csv: str | Path | None = None) -> Path:
    cfg = load_config(config_toml)
    df, metrics = analyze_trial(pose_csv, cfg, ball_csv)
    out_dir = Path(pose_csv).expanduser().resolve().parent / f"vaila_vek_preview_{_timestamp()}"
    out_dir.mkdir(parents=True, exist_ok=True)
    plots = save_plots(df, out_dir, "vek_preview", metrics.get("contact_frame"))
    return plots[0] if plots else out_dir


def main_gui() -> None:
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Tk()
    root.title("vaila-ElasticKick (VEK)")
    root.resizable(False, False)

    vars_: dict[str, tk.StringVar] = {
        "pose": tk.StringVar(),
        "ball": tk.StringVar(),
        "cal": tk.StringVar(),
        "config": tk.StringVar(),
        "output": tk.StringVar(),
    }

    def pick_file(key: str, title: str, patterns: list[tuple[str, str]]) -> None:
        value = filedialog.askopenfilename(title=title, filetypes=patterns, parent=root)
        if value:
            vars_[key].set(value)

    def pick_output() -> None:
        value = filedialog.askdirectory(title="Select VEK output directory", parent=root)
        if value:
            vars_["output"].set(value)

    def save_config() -> None:
        target = filedialog.asksaveasfilename(
            title="Save VEK configuration",
            defaultextension=".toml",
            filetypes=[("TOML", "*.toml")],
            parent=root,
        )
        if target:
            write_default_config(target, vars_["cal"].get() or DEFAULT_CALIBRATION_NAME)
            vars_["config"].set(target)

    def run_analysis(open_report: bool = True) -> None:
        if not vars_["pose"].get() or not vars_["config"].get():
            messagebox.showwarning("VEK", "Select pose CSV and TOML configuration.", parent=root)
            return
        try:
            result = process_vek_file(
                vars_["pose"].get(),
                vars_["config"].get(),
                vars_["output"].get() or None,
                ball_csv=vars_["ball"].get() or None,
                band_calibration_csv=vars_["cal"].get() or None,
            )
        except Exception as exc:
            messagebox.showerror("VEK error", str(exc), parent=root)
            return
        messagebox.showinfo("VEK complete", f"Report created:\n{result.report_html}", parent=root)
        if open_report:
            with contextlib.suppress(Exception):
                webbrowser.open_new_tab(result.report_html.as_uri())

    frm = tk.Frame(root, padx=12, pady=12)
    frm.pack(fill="both", expand=True)
    tk.Label(frm, text="vaila-ElasticKick (VEK)", font=("Arial", 14, "bold")).grid(
        row=0, column=0, columnspan=3, sticky="w", pady=(0, 8)
    )
    fields = [
        ("pose", "Pose CSV", "Select markerless pose CSV", [("CSV", "*.csv")]),
        ("ball", "Ball CSV", "Select optional ball tracking CSV", [("CSV", "*.csv")]),
        ("cal", "Band calibration", "Select elastic-band calibration CSV", [("CSV", "*.csv")]),
        ("config", "TOML config", "Select VEK TOML configuration", [("TOML", "*.toml")]),
    ]
    for row, (key, label, title, patterns) in enumerate(fields, start=1):
        tk.Label(frm, text=label).grid(row=row, column=0, sticky="w", pady=3)
        tk.Entry(frm, textvariable=vars_[key], width=58).grid(row=row, column=1, sticky="we", pady=3)
        tk.Button(frm, text="Browse", command=lambda k=key, t=title, p=patterns: pick_file(k, t, p)).grid(
            row=row, column=2, padx=(5, 0), pady=3
        )
    tk.Label(frm, text="Output dir").grid(row=5, column=0, sticky="w", pady=3)
    tk.Entry(frm, textvariable=vars_["output"], width=58).grid(row=5, column=1, sticky="we", pady=3)
    tk.Button(frm, text="Browse", command=pick_output).grid(row=5, column=2, padx=(5, 0), pady=3)

    buttons = tk.Frame(frm)
    buttons.grid(row=6, column=0, columnspan=3, sticky="we", pady=(12, 0))
    tk.Button(buttons, text="Run Analysis", command=run_analysis).pack(side="left", padx=3)
    tk.Button(
        buttons,
        text="Preview Signals",
        command=lambda: webbrowser.open_new_tab(
            preview_signals(vars_["pose"].get(), vars_["config"].get(), vars_["ball"].get() or None).as_uri()
        )
        if vars_["pose"].get() and vars_["config"].get()
        else messagebox.showwarning("VEK", "Select pose CSV and config first.", parent=root),
    ).pack(side="left", padx=3)
    tk.Button(buttons, text="Save Configuration", command=save_config).pack(side="left", padx=3)
    tk.Button(buttons, text="Close", command=root.destroy).pack(side="right", padx=3)
    root.mainloop()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="vaila-ElasticKick (VEK): elastic-band resisted kicking assessment"
    )
    parser.add_argument("-i", "--input", help="Markerless pose CSV")
    parser.add_argument("--ball", help="Ball tracking CSV")
    parser.add_argument("--band-calibration", help="Elastic-band force-length calibration CSV")
    parser.add_argument("-c", "--config", help="VEK TOML configuration")
    parser.add_argument("-o", "--output", help="Output directory")
    parser.add_argument("--batch", help="Team directory with athlete subfolders")
    parser.add_argument("--write-default-config", help="Write an example TOML configuration and exit")
    parser.add_argument("--gui", action="store_true", help="Open Tkinter GUI")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    if args.write_default_config:
        path = write_default_config(args.write_default_config, args.band_calibration)
        print(f"VEK default config written: {path}")
        return 0
    if args.gui or (not args.input and not args.batch):
        main_gui()
        return 0
    if args.batch:
        report = process_team_batch(args.batch, args.output)
        print(f"VEK team report: {report}")
        return 0
    if not args.input or not args.config:
        raise SystemExit("--input and --config are required for single-trial CLI mode")
    result = process_vek_file(
        args.input,
        args.config,
        args.output,
        ball_csv=args.ball,
        band_calibration_csv=args.band_calibration,
    )
    print(f"VEK report: {result.report_html}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
