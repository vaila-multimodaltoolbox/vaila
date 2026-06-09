"""
===============================================================================
vaila_deadlift.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 28 May 2026
Update Date: 09 June 2026
Version: 0.3.48

Companion module:
-----------------
:mod:`vaila.vaila_deadlift_imu` provides a dedicated IMU-only pipeline using
Madgwick / Mahony AHRS sensor fusion (proper orientation tracking, magnetometer
support, Earth-frame gravity removal). Wired to GUI button **B6_r7_c1 -
Deadlift IMU**.
Python Version: 3.12.13

Description:
------------
This script processes deadlift and RDL kinematic data from two input types:

1. **IMU data** (accelerometer + gyroscope CSV from barbell-mounted sensors):
   - Automatic repetition detection from vertical acceleration
   - Per-rep velocity estimation via numerical integration
   - Per-rep power (P = F × v) and work (W = ∫P dt) from barbell weight
   - Cadence metrics (reps/min, inter-rep intervals)
   - Center of mass (barbell displacement trajectory)
   - Rep-to-rep comparison statistics (max, min, range, CV%)

2. **MediaPipe pose estimation** (.csv with landmark coordinates):
   - All existing kinematic checks (stance, spine, shin, cervical, bar path)
   - Multi-repetition detection from averaged left/right wrist vertical displacement
     (bar-height proxy from weight-plate markers)
   - Whole-body center of mass estimation (De Leva 1996 segment proportions)
   - Foot spread (ankle-to-ankle), knee spread (knee-to-knee)
   - Hand-to-foot horizontal distance
   - Per-rep power & work from COM vertical velocity and body mass
   - Cadence and rep-to-rep comparison statistics

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to
transform from screen coordinates (where y increases downward) to biomechanical
coordinates (where y increases upward).

Features:
---------
- Auto-detects data type (IMU vs MediaPipe) from CSV column headers
- Repetition counting with phase segmentation (concentric/eccentric)
- Per-rep metrics: peak/mean velocity, peak/mean power, work, ROM, force
- Cadence: reps/min, mean/std rep duration, interval variability
- Rep-to-rep comparison: max, min, range, CV%, best/worst rep identification
- Center of mass trajectory (barbell for IMU, whole-body for MediaPipe)
- Body distance metrics (foot/knee spread, hand-foot horizontal offset)
- Generates time-series visualizations and a complete HTML evaluation report

Dependencies:
-------------
- Python 3.x, pandas, numpy, matplotlib, scipy, tkinter, math, datetime

Usage:
------
- GUI: Run with no arguments or --gui.
- CLI: Use -i <path_to.csv> -c <path_to_config.toml> -o <output_dir>
===============================================================================

"""

import contextlib
import datetime
import math
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import butter, filtfilt, find_peaks

try:
    import tomllib as _toml_reader
except Exception:
    _toml_reader = None

_DEADLIFT_CONTEXT = None

# De Leva (1996) segment mass fractions for 2D whole-body COM estimation.
# Keys map to MediaPipe landmark pairs: (proximal, distal) → midpoint.
_SEGMENT_MASS = {
    "head": (0.081, "left_ear", "right_ear"),
    "trunk": (0.430, "mid_shoulder", "mid_hip"),
    "upper_arm_l": (0.027, "left_shoulder", "left_elbow"),
    "upper_arm_r": (0.027, "right_shoulder", "right_elbow"),
    "forearm_hand_l": (0.022, "left_elbow", "left_wrist"),
    "forearm_hand_r": (0.022, "right_elbow", "right_wrist"),
    "thigh_l": (0.142, "left_hip", "left_knee"),
    "thigh_r": (0.142, "right_hip", "right_knee"),
    "shank_l": (0.043, "left_knee", "left_ankle"),
    "shank_r": (0.043, "right_knee", "right_ankle"),
    "foot_l": (0.014, "left_ankle", "left_foot_index"),
    "foot_r": (0.014, "right_ankle", "right_foot_index"),
}


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


def _parse_locale_float(value) -> float:
    if isinstance(value, bool):
        raise ValueError("Boolean is not a float")
    text = str(value).strip().replace(",", ".")
    if not text:
        raise ValueError("Empty float")
    return float(text)


def _load_context_from_toml(base_dir: Path | None = None) -> dict | None:
    search_paths = []
    if base_dir:
        search_paths.append(base_dir / "vaila_deadlift_config.toml")
    search_paths.append(Path(__file__).parent / "vaila_deadlift_config.toml")

    for p in search_paths:
        if p.exists():
            try:
                if _toml_reader is None:
                    import toml

                    data = toml.load(str(p))
                else:
                    with open(p, "rb") as f:
                        data = _toml_reader.load(f)
                cfg = data.get("deadlift_context", {})
                return {
                    "mass_kg": _parse_locale_float(cfg.get("mass_kg", 75.0)),
                    "fps": _parse_locale_float(cfg.get("fps", 30.0)),
                    "shank_length_m": _parse_locale_float(cfg.get("shank_length_m", 0.40)),
                    "weight_kg": _parse_locale_float(cfg.get("weight_kg", 20.0)),
                    "use_total_mass_for_power": bool(cfg.get("use_total_mass_for_power", True)),
                    "imu_fps": _parse_locale_float(cfg.get("imu_fps", 25.0)),
                }
            except Exception:
                pass
    return None


def _load_parameters_file(base_dir: Path) -> dict | None:
    """Load external parameters (deadlift_parameters.txt) if present."""
    search_dirs = [base_dir, *base_dir.parents[:3]]
    for directory in search_dirs:
        p = directory / "deadlift_parameters.txt"
        if not p.exists():
            continue
        try:
            params: dict = {}
            with open(p, encoding="utf-8") as f:
                for line in f:
                    if "," in line:
                        key, val = line.strip().split(",", 1)
                        params[key.strip()] = val.strip()
            return params
        except Exception:
            pass
    return None


def _load_text_parameters(path: Path) -> dict:
    """Load simple comma or colon separated text parameters."""
    params: dict[str, str] = {}
    if not path.exists():
        return params
    try:
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if not text or text.startswith("#") or set(text) <= {"="}:
                    continue
                if "," in text:
                    key, val = text.split(",", 1)
                elif ":" in text:
                    key, val = text.split(":", 1)
                else:
                    continue
                params[key.strip().lower()] = val.strip()
    except Exception:
        return {}
    return params


def _find_nearby_file(base_dir: Path, filename: str) -> Path | None:
    for directory in [base_dir, *base_dir.parents[:3]]:
        p = directory / filename
        if p.exists():
            return p
    return None


def _load_kinematics_parameters(base_dir: Path, explicit_path: str | None = None) -> dict:
    path = Path(explicit_path) if explicit_path else _find_nearby_file(base_dir, "kinematics_parameter.txt")
    if path is None:
        return {}
    raw = _load_text_parameters(path)
    params: dict[str, float | str] = {"kinematics_parameter_file": str(path)}
    fps_keys = ("recommended hz", "display fps", "avg fps", "capture fps", "fps")
    for key in fps_keys:
        if key in raw:
            try:
                params["fps"] = _parse_locale_float(raw[key])
                break
            except Exception:
                pass
    for key, out_key in (("frames", "frames"), ("duration", "duration_s")):
        if key in raw:
            with contextlib.suppress(Exception):
                params[out_key] = _parse_locale_float(raw[key].split()[0])
    if "resolution" in raw:
        params["resolution"] = raw["resolution"]
    return params


def _load_subject_parameters(base_dir: Path, explicit_path: str | None = None) -> dict:
    path = Path(explicit_path) if explicit_path else _find_nearby_file(base_dir, "subject_parameters.txt")
    if path is None:
        return {}
    params: dict[str, float | str] = {"subject_parameter_file": str(path)}
    try:
        df = pd.read_csv(path)
        if not df.empty:
            row = df.iloc[0]
            column_map = {
                "subjectmass_kg": "mass_kg",
                "mass_kg": "mass_kg",
                "body_mass_kg": "mass_kg",
                "shank_len_meter": "shank_length_m",
                "shank_length_m": "shank_length_m",
                "loadmass_kg": "weight_kg",
                "barbell_mass_kg": "weight_kg",
                "weight_kg": "weight_kg",
            }
            for src, dst in column_map.items():
                if src in row.index:
                    with contextlib.suppress(Exception):
                        params[dst] = _parse_locale_float(row[src])
        return params
    except Exception:
        pass

    raw = _load_text_parameters(path)
    key_map = {
        "subjectmass_kg": "mass_kg",
        "mass_kg": "mass_kg",
        "body_mass_kg": "mass_kg",
        "shank_len_meter": "shank_length_m",
        "shank_length_m": "shank_length_m",
        "loadmass_kg": "weight_kg",
        "barbell_mass_kg": "weight_kg",
        "weight_kg": "weight_kg",
    }
    for src, dst in key_map.items():
        if src in raw:
            with contextlib.suppress(Exception):
                params[dst] = _parse_locale_float(raw[src])
    return params


def _merge_deadlift_overrides(
    ctx: dict,
    base_dir: Path,
    overrides: dict | None = None,
) -> dict:
    merged = dict(ctx)
    overrides = overrides or {}

    kin_params = _load_kinematics_parameters(base_dir, overrides.get("kinematics_params"))
    subject_params = _load_subject_parameters(base_dir, overrides.get("subject_params"))
    for src in (kin_params, subject_params):
        for key, value in src.items():
            if key in {"fps", "mass_kg", "shank_length_m", "weight_kg"}:
                merged[key] = value
            else:
                merged[key] = value

    scalar_map = {
        "fps": "fps",
        "mass_kg": "mass_kg",
        "shank_length_m": "shank_length_m",
        "barbell_mass_kg": "weight_kg",
        "weight_kg": "weight_kg",
        "gif_duration_s": "gif_duration_s",
    }
    for src, dst in scalar_map.items():
        if overrides.get(src) is not None:
            merged[dst] = overrides[src]
    return merged


# ---------------------------------------------------------------------------
# Data type detection
# ---------------------------------------------------------------------------


def detect_data_type(df: pd.DataFrame) -> str:
    """Auto-detect CSV type from column headers."""
    imu_cols = {"acc_x", "acc_y", "acc_z", "gyr_x", "gyr_y", "gyr_z"}
    mp_cols = {"right_shoulder_x", "right_hip_x", "right_knee_x"}
    if imu_cols.issubset(set(df.columns)):
        return "imu"
    if mp_cols.issubset(set(df.columns)):
        return "mediapipe"
    return "unknown"


# ---------------------------------------------------------------------------
# General helpers
# ---------------------------------------------------------------------------


def calculate_joint_angle(A, B, C):
    """Calculates absolute angle at vertex B given points A, B, C."""
    rad = math.atan2(C[1] - B[1], C[0] - B[0]) - math.atan2(A[1] - B[1], A[0] - B[0])
    angle = abs(math.degrees(rad))
    if angle > 180.0:
        angle = 360.0 - angle
    return angle


def calc_conversion_factor(df, shank_length_real=0.40):
    lengths = []
    start_f, end_f = 5, min(30, len(df))
    for i in range(start_f, end_f):
        rkx, rky = df["right_knee_x"].iloc[i], df["right_knee_y"].iloc[i]
        rax, ray = df["right_ankle_x"].iloc[i], df["right_ankle_y"].iloc[i]
        lengths.append(np.sqrt((rkx - rax) ** 2 + (rky - ray) ** 2))
    return shank_length_real / np.median(lengths) if lengths else 0.40


# ---------------------------------------------------------------------------
# IMU Processing Pipeline
# ---------------------------------------------------------------------------


def _estimate_gravity(df: pd.DataFrame, n_static: int = 30):
    """Estimate gravity direction and magnitude from initial static samples."""
    acc = df[["acc_x", "acc_y", "acc_z"]].iloc[:n_static].values
    g_vec = acc.mean(axis=0)
    g_mag = float(np.linalg.norm(g_vec))
    if g_mag < 1e-6:
        g_mag = 9.81
        g_vec = np.array([0.0, 0.0, g_mag])
    return g_vec / g_mag, g_mag


def _compute_vertical_acc(df: pd.DataFrame, g_unit: np.ndarray, g_mag: float) -> np.ndarray:
    """Project 3-axis acceleration onto the gravity axis and remove gravity."""
    acc = df[["acc_x", "acc_y", "acc_z"]].values
    a_vert = acc @ g_unit
    return a_vert - g_mag


def detect_reps_imu(
    a_vert: np.ndarray,
    fps: float,
    min_rep_s: float = 2.5,
    prominence_frac: float = 0.05,
) -> tuple[list[dict], np.ndarray, np.ndarray, np.ndarray]:
    """
    Detect repetition boundaries from vertical acceleration.

    Uses filtered vertical acceleration peaks (rep lockouts) and valleys
    (rep bottoms) rather than double integration, avoiding drift issues.

    Returns (reps, velocity, displacement, a_filtered).
    Each rep dict has: rep_number, start_frame, peak_frame, end_frame, duration_s.
    """
    dt = 1.0 / fps
    nyq = fps / 2.0

    # Light filter for metrics (preserve dynamics)
    b_hi, a_hi = butter(2, min(5.0, nyq * 0.9) / nyq)
    a_filt = filtfilt(b_hi, a_hi, a_vert, padlen=min(50, len(a_vert) - 1))

    # Heavier filter for rep detection (smooth envelope, ~1.2 Hz)
    b_det, a_det = butter(2, min(1.2, nyq * 0.9) / nyq)
    a_detect = filtfilt(b_det, a_det, a_vert, padlen=min(50, len(a_vert) - 1))

    min_dist = max(int(min_rep_s * fps), 3)
    prom = np.ptp(a_detect) * prominence_frac

    peaks, _ = find_peaks(a_detect, distance=min_dist, prominence=max(prom, 0.1))
    valleys, _ = find_peaks(-a_detect, distance=min_dist, prominence=max(prom, 0.1))
    all_valleys = np.concatenate([[0], valleys, [len(a_detect) - 1]])

    reps: list[dict] = []
    for i, pk in enumerate(peaks):
        prev = all_valleys[all_valleys < pk]
        start = int(prev[-1]) if len(prev) > 0 else 0
        nxt = all_valleys[all_valleys > pk]
        end = int(nxt[0]) if len(nxt) > 0 else len(a_detect) - 1
        reps.append(
            {
                "rep_number": i + 1,
                "start_frame": start,
                "peak_frame": int(pk),
                "end_frame": end,
                "duration_s": (end - start) / fps,
            }
        )

    # Per-rep velocity estimation with drift correction
    n = len(a_filt)
    velocity = np.zeros(n)
    disp = np.zeros(n)

    for rep in reps:
        s, e = rep["start_frame"], min(rep["end_frame"], n - 1)
        a_seg = a_filt[s : e + 1]
        v_seg = np.cumsum(a_seg) * dt
        # Linear drift correction: velocity at start and end should be ~0
        drift = np.linspace(v_seg[0], v_seg[-1], len(v_seg))
        v_seg -= drift
        velocity[s : e + 1] = v_seg

        d_seg = np.cumsum(v_seg) * dt
        d_drift = np.linspace(d_seg[0], d_seg[-1], len(d_seg))
        d_seg -= d_drift
        disp[s : e + 1] = d_seg

    return reps, velocity, disp, a_filt


def compute_rep_imu_metrics(
    a_filt: np.ndarray,
    velocity: np.ndarray,
    disp: np.ndarray,
    fps: float,
    weight_kg: float,
    reps: list[dict],
) -> list[dict]:
    """Per-rep biomechanical metrics from barbell-mounted IMU."""
    dt = 1.0 / fps
    g = 9.81
    metrics: list[dict] = []

    for rep in reps:
        s, pk, e = rep["start_frame"], rep["peak_frame"], rep["end_frame"]
        v_conc = velocity[s : pk + 1]
        a_conc = a_filt[s : pk + 1]
        d_rep = disp[s : e + 1]

        force_conc = weight_kg * (g + a_conc)
        power_conc = force_conc * v_conc

        pos_v = v_conc[v_conc > 0]
        pos_p = power_conc[power_conc > 0]

        peak_vel = float(np.max(np.abs(v_conc))) if len(v_conc) > 0 else 0.0
        mean_vel = float(np.mean(pos_v)) if len(pos_v) > 0 else 0.0
        peak_power = float(np.max(power_conc)) if len(power_conc) > 0 else 0.0
        mean_power = float(np.mean(pos_p)) if len(pos_p) > 0 else 0.0
        work = float(np.trapezoid(pos_p, dx=dt)) if len(pos_p) > 0 else 0.0
        rom = float(np.ptp(d_rep)) if len(d_rep) > 0 else 0.0
        peak_force = float(np.max(force_conc)) if len(force_conc) > 0 else 0.0
        mean_force = float(np.mean(force_conc)) if len(force_conc) > 0 else 0.0

        conc_s = (pk - s) / fps
        ecc_s = (e - pk) / fps

        metrics.append(
            {
                "rep_number": rep["rep_number"],
                "start_frame": s,
                "peak_frame": pk,
                "end_frame": e,
                "duration_s": rep["duration_s"],
                "concentric_s": conc_s,
                "eccentric_s": ecc_s,
                "peak_velocity_ms": peak_vel,
                "mean_velocity_ms": mean_vel,
                "peak_power_W": peak_power,
                "mean_power_W": mean_power,
                "work_J": work,
                "rom_m": rom,
                "peak_force_N": peak_force,
                "mean_force_N": mean_force,
            }
        )

    return metrics


# ---------------------------------------------------------------------------
# MediaPipe Processing Pipeline (existing + enhancements)
# ---------------------------------------------------------------------------


def process_deadlift_kinematics(df, fps, factor):
    """
    Calculates frame-by-frame deadlift metrics, including setup checks.

    The averaged wrist landmark is used as a barbell/plate-height proxy because
    MediaPipe does not track the bar directly.
    """
    dt = 1.0 / fps

    df["bar_marker_x_m"] = (df["left_wrist_x_m"] + df["right_wrist_x_m"]) / 2.0
    df["bar_marker_y_m"] = (df["left_wrist_y_m"] + df["right_wrist_y_m"]) / 2.0
    df["bar_height_m"] = df["bar_marker_y_m"]
    df["wrist_avg_y_m"] = df["bar_marker_y_m"]

    # 1. Stance Width Ratio (Ankle Separation / Hip Separation)
    hip_dist = (
        np.sqrt(
            (df["left_hip_x"] - df["right_hip_x"]) ** 2
            + (df["left_hip_y"] - df["right_hip_y"]) ** 2
        )
        * factor
    )
    ankle_dist = (
        np.sqrt(
            (df["left_ankle_x"] - df["right_ankle_x"]) ** 2
            + (df["left_ankle_y"] - df["right_ankle_y"]) ** 2
        )
        * factor
    )

    # Foot spread (ankle-to-ankle in meters)
    foot_spread_m = np.sqrt(
        (df["left_ankle_x_m"] - df["right_ankle_x_m"]) ** 2
        + (df["left_ankle_y_m"] - df["right_ankle_y_m"]) ** 2
    )

    # Knee spread (knee-to-knee in meters)
    knee_spread_m = np.sqrt(
        (df["left_knee_x_m"] - df["right_knee_x_m"]) ** 2
        + (df["left_knee_y_m"] - df["right_knee_y_m"]) ** 2
    )

    # Hand-to-foot horizontal distance (wrist X vs ankle X, in meters)
    hand_foot_horiz_l = df["left_wrist_x_m"] - df["left_ankle_x_m"]
    hand_foot_horiz_r = df["right_wrist_x_m"] - df["right_ankle_x_m"]

    metric_updates = {
        "stance_width_ratio": ankle_dist / hip_dist,
        "foot_spread_m": foot_spread_m,
        "knee_spread_m": knee_spread_m,
        "hand_foot_horiz_l_m": hand_foot_horiz_l,
        "hand_foot_horiz_r_m": hand_foot_horiz_r,
    }

    # 2. Joint Angles (Knee and Shin-to-Ground)
    knee_angles_l = []
    knee_angles_r = []
    shin_angles_r = []
    spine_deviations = []
    bar_path_offsets = []
    cervical_angles = []
    hip_extension_angles_r = []
    arm_verticality_deltas = []
    bar_midfoot_errors = []

    for _idx, row in df.iterrows():
        lh = [row["left_hip_x_m"], row["left_hip_y_m"]]
        lk = [row["left_knee_x_m"], row["left_knee_y_m"]]
        la = [row["left_ankle_x_m"], row["left_ankle_y_m"]]

        rh = [row["right_hip_x_m"], row["right_hip_y_m"]]
        rk = [row["right_knee_x_m"], row["right_knee_y_m"]]
        ra = [row["right_ankle_x_m"], row["right_ankle_y_m"]]

        ls = [row["left_shoulder_x_m"], row["left_shoulder_y_m"]]
        rs = [row["right_shoulder_x_m"], row["right_shoulder_y_m"]]
        le = [row["left_ear_x_m"], row["left_ear_y_m"]]

        bar_marker = [row["bar_marker_x_m"], row["bar_marker_y_m"]]
        r_heel = [row["right_heel_x_m"], row["right_heel_y_m"]]
        r_toe = [row["right_foot_index_x_m"], row["right_foot_index_y_m"]]

        k_ang_l = calculate_joint_angle(lh, lk, la)
        k_ang_r = calculate_joint_angle(rh, rk, ra)
        knee_angles_l.append(180.0 - k_ang_l)
        knee_angles_r.append(180.0 - k_ang_r)
        hip_extension_angles_r.append(calculate_joint_angle(rs, rh, rk))

        shin_angles_r.append(calculate_joint_angle(rk, ra, [ra[0] + 0.1, ra[1]]))

        mid_hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
        mid_shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        spine_ang = calculate_joint_angle(mid_shoulder, mid_hip, [mid_hip[0] + 0.1, mid_hip[1]])
        spine_deviations.append(spine_ang)

        bar_path_offsets.append(abs(bar_marker[0] - rk[0]))
        arm_verticality_deltas.append(bar_marker[0] - mid_shoulder[0])
        midfoot_x = (r_heel[0] + r_toe[0]) / 2.0
        bar_midfoot_errors.append(bar_marker[0] - midfoot_x)

        cervical_ang = calculate_joint_angle(le, mid_shoulder, mid_hip)
        cervical_angles.append(abs(180.0 - cervical_ang))

    metric_updates.update(
        {
            "knee_flexion_l": knee_angles_l,
            "knee_flexion_r": knee_angles_r,
            "shin_angle_ground": shin_angles_r,
            "spine_deviation": np.abs(np.array(spine_deviations) - spine_deviations[0]),
            "bar_path_proximity_m": bar_path_offsets,
            "cervical_deviation": cervical_angles,
            "hip_extension_r": hip_extension_angles_r,
            "arm_verticality_delta_m": arm_verticality_deltas,
            "bar_midfoot_error_m": bar_midfoot_errors,
        }
    )
    metrics_df = pd.DataFrame(metric_updates, index=df.index)

    # 6. Hip Hinge Gradient
    metrics_df["hip_velocity_x"] = np.gradient(df["right_hip_x_m"], dt)
    metrics_df["shoulder_velocity_y"] = np.gradient(df["right_shoulder_y_m"], dt)
    metrics_df["knee_extension_velocity_deg_s"] = -np.gradient(metrics_df["knee_flexion_r"], dt)
    metrics_df["hip_extension_velocity_deg_s"] = np.gradient(metrics_df["hip_extension_r"], dt)
    metrics_df["pull_synchronism_ratio"] = metrics_df["knee_extension_velocity_deg_s"] / (
        metrics_df["hip_extension_velocity_deg_s"].abs() + 1e-6
    )

    return pd.concat([df, metrics_df], axis=1)


def compute_center_of_mass_2d(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """Estimate whole-body 2D COM from MediaPipe landmarks (De Leva 1996)."""
    n = len(df)
    com_x = np.zeros(n)
    com_y = np.zeros(n)
    total_mass = 0.0

    # Pre-compute mid-shoulder and mid-hip as virtual landmarks
    mid_shoulder_x = (df["left_shoulder_x_m"].values + df["right_shoulder_x_m"].values) / 2
    mid_shoulder_y = (df["left_shoulder_y_m"].values + df["right_shoulder_y_m"].values) / 2
    mid_hip_x = (df["left_hip_x_m"].values + df["right_hip_x_m"].values) / 2
    mid_hip_y = (df["left_hip_y_m"].values + df["right_hip_y_m"].values) / 2

    for _seg_name, (frac, prox, dist) in _SEGMENT_MASS.items():
        total_mass += frac
        if prox == "mid_shoulder":
            px, py = mid_shoulder_x, mid_shoulder_y
        elif prox == "mid_hip":
            px, py = mid_hip_x, mid_hip_y
        else:
            px = df[f"{prox}_x_m"].values
            py = df[f"{prox}_y_m"].values

        if dist == "mid_shoulder":
            dx, dy = mid_shoulder_x, mid_shoulder_y
        elif dist == "mid_hip":
            dx, dy = mid_hip_x, mid_hip_y
        else:
            dx = df[f"{dist}_x_m"].values
            dy = df[f"{dist}_y_m"].values

        seg_x = (px + dx) / 2.0
        seg_y = (py + dy) / 2.0
        com_x += frac * seg_x
        com_y += frac * seg_y

    if total_mass > 0:
        com_x /= total_mass
        com_y /= total_mass

    return com_x, com_y


def detect_reps_mediapipe(
    df: pd.DataFrame,
    fps: float,
    min_rep_s: float = 1.0,
    prominence_frac: float = 0.20,
    expected_reps: int | None = None,
) -> list[dict]:
    """Detect completed repetitions from averaged wrist/bar-height maxima.

    Two starts are handled:
    1. Low start: first maximum is rep 1.
    2. Standing start: initial standing maximum is setup; rep 1 is the next
       standing maximum after the bar reaches the low phase.
    """
    if "bar_height_m" in df.columns:
        signal = df["bar_height_m"].values
    else:
        signal = (df["right_wrist_y_m"].values + df["left_wrist_y_m"].values) / 2.0

    nyq = fps / 2.0
    cutoff = min(1.5, nyq * 0.9)
    b, a = butter(2, cutoff / nyq)
    sig_filt = filtfilt(b, a, signal, padlen=min(50, len(signal) - 1))

    sig_min = float(np.nanmin(sig_filt))
    sig_max = float(np.nanmax(sig_filt))
    sig_range = sig_max - sig_min
    if sig_range <= 1e-9:
        return []

    low_threshold = sig_min + 0.35 * sig_range
    high_threshold = sig_min + 0.65 * sig_range
    if sig_filt[0] <= low_threshold:
        start_phase = "low"
    elif sig_filt[0] >= high_threshold:
        start_phase = "standing"
    else:
        start_phase = (
            "low" if abs(sig_filt[0] - sig_min) <= abs(sig_filt[0] - sig_max) else "standing"
        )

    min_dist = max(int(min_rep_s * fps), 3)
    prom = sig_range * prominence_frac

    # Maxima = standing/count frames; minima = low phase/bar near ground.
    peaks, _ = find_peaks(sig_filt, distance=min_dist, prominence=max(prom * 0.5, 0.002))
    valleys, _ = find_peaks(-sig_filt, distance=min_dist, prominence=max(prom, 0.005))
    peaks = np.array([int(p) for p in peaks], dtype=int)
    valleys = np.array([int(v) for v in valleys], dtype=int)

    if start_phase == "standing" and len(valleys) > 0:
        first_low = int(valleys[0])
        peaks = peaks[peaks > first_low]

    if len(peaks) < 1:
        return []

    reps: list[dict] = []
    for i, peak in enumerate(peaks):
        start = int(peaks[i - 1]) if i > 0 else 0
        end = int(peak)
        cycle_valleys = valleys[(valleys >= start) & (valleys <= end)]
        if len(cycle_valleys) > 0:
            bottom = int(cycle_valleys[np.argmin(sig_filt[cycle_valleys])])
        else:
            bottom = start + int(np.argmin(sig_filt[start : end + 1]))
        reps.append(
            {
                "rep_number": i + 1,
                "start_frame": start,
                "bottom_frame": bottom,
                "end_frame": end,
                "peak_frame": end,
                "count_frame": end,
                "bottom_time_s": bottom / fps,
                "peak_time_s": end / fps,
                "count_time_s": end / fps,
                "duration_s": (end - start) / fps,
                "start_phase": start_phase if i == 0 else "standing",
            }
        )

    if expected_reps and expected_reps != len(reps):
        print(
            f"[MediaPipe] Metadata rep count {expected_reps} differs from "
            f"phase-based wrist count {len(reps)}; using phase-based count."
        )
    return reps


def _trim_reps_to_expected_count(reps: list[dict], expected_reps: int | None) -> list[dict]:
    """Use annotated rep count, when available, to drop lead-in/trailing cycles."""
    if expected_reps is None or expected_reps <= 0 or len(reps) <= expected_reps:
        return reps
    excess = len(reps) - expected_reps
    drop_start = excess // 2
    drop_end = excess - drop_start
    trimmed = reps[drop_start : len(reps) - drop_end]
    for i, rep in enumerate(trimmed, start=1):
        rep["rep_number"] = i
    return trimmed


def compute_mediapipe_rep_metrics(
    df: pd.DataFrame,
    com_x: np.ndarray,
    com_y: np.ndarray,
    fps: float,
    mass_kg: float,
    barbell_kg: float,
    reps: list[dict],
    use_total_mass_for_power: bool = True,
) -> list[dict]:
    """Per-rep power/work from COM vertical velocity and configured load mass."""
    dt = 1.0 / fps
    g = 9.81
    com_vy = np.gradient(com_y, dt)
    analysis_mass_kg = mass_kg + barbell_kg if use_total_mass_for_power else mass_kg
    metrics: list[dict] = []

    for rep in reps:
        s = rep["start_frame"]
        bot = rep.get("bottom_frame", rep.get("peak_frame", s))
        e = rep["end_frame"]

        # Concentric: bottom to end (lockout). COM supplies velocity/ROM; configured
        # barbell mass from the weight plates is included by default in total load.
        vy_conc = com_vy[bot : e + 1]
        force_conc = analysis_mass_kg * (g + np.gradient(com_vy[bot : e + 1], dt))
        power_conc = force_conc * vy_conc

        pos_v = vy_conc[vy_conc > 0]
        pos_p = power_conc[power_conc > 0]

        peak_vel = float(np.max(np.abs(vy_conc))) if len(vy_conc) > 0 else 0.0
        mean_vel = float(np.mean(pos_v)) if len(pos_v) > 0 else 0.0
        peak_power = float(np.max(power_conc)) if len(power_conc) > 0 else 0.0
        mean_power = float(np.mean(pos_p)) if len(pos_p) > 0 else 0.0
        work = float(np.trapezoid(pos_p, dx=dt)) if len(pos_p) > 0 else 0.0

        com_range = float(np.ptp(com_y[s : e + 1]))
        bar_range = (
            float(np.ptp(df["bar_height_m"].iloc[s : e + 1]))
            if "bar_height_m" in df.columns
            else 0.0
        )

        # Distance metrics at bottom frame
        foot_sp = float(df["foot_spread_m"].iloc[bot]) if "foot_spread_m" in df.columns else 0.0
        knee_sp = float(df["knee_spread_m"].iloc[bot]) if "knee_spread_m" in df.columns else 0.0
        hf_l = (
            float(df["hand_foot_horiz_l_m"].iloc[bot])
            if "hand_foot_horiz_l_m" in df.columns
            else 0.0
        )
        hf_r = (
            float(df["hand_foot_horiz_r_m"].iloc[bot])
            if "hand_foot_horiz_r_m" in df.columns
            else 0.0
        )

        conc_s = (e - bot) / fps
        ecc_s = (bot - s) / fps

        metrics.append(
            {
                "rep_number": rep["rep_number"],
                "start_frame": s,
                "bottom_frame": bot,
                "end_frame": e,
                "count_frame": rep.get("count_frame", e),
                "bottom_time_s": rep.get("bottom_time_s", bot / fps),
                "count_time_s": rep.get("count_time_s", e / fps),
                "start_phase": rep.get("start_phase", ""),
                "duration_s": rep["duration_s"],
                "concentric_s": conc_s,
                "eccentric_s": ecc_s,
                "peak_velocity_ms": peak_vel,
                "mean_velocity_ms": mean_vel,
                "peak_power_W": peak_power,
                "mean_power_W": mean_power,
                "work_J": work,
                "rom_m": com_range,
                "bar_rom_m": bar_range,
                "body_mass_kg": mass_kg,
                "barbell_mass_kg": barbell_kg,
                "analysis_mass_kg": analysis_mass_kg,
                "foot_spread_m": foot_sp,
                "knee_spread_m": knee_sp,
                "hand_foot_horiz_l_m": hf_l,
                "hand_foot_horiz_r_m": hf_r,
            }
        )

    return metrics


# ---------------------------------------------------------------------------
# Common analysis: cadence, rep comparison
# ---------------------------------------------------------------------------


def compute_cadence(rep_metrics: list[dict], fps: float) -> dict:
    """Cadence and timing statistics from rep metrics."""
    n = len(rep_metrics)
    if n == 0:
        return {"n_reps": 0}

    durations = [r["duration_s"] for r in rep_metrics]
    peak_key = "peak_frame" if "peak_frame" in rep_metrics[0] else "bottom_frame"
    peak_times = [r[peak_key] / fps for r in rep_metrics]

    result: dict = {
        "n_reps": n,
        "mean_rep_time_s": float(np.mean(durations)),
        "std_rep_time_s": float(np.std(durations)),
    }

    if n >= 2:
        intervals = np.diff(peak_times)
        total_time = peak_times[-1] - peak_times[0]
        result["cadence_rpm"] = float((n - 1) / (total_time / 60.0)) if total_time > 0 else 0.0
        result["mean_interval_s"] = float(np.mean(intervals))
        result["std_interval_s"] = float(np.std(intervals))

        conc = [r.get("concentric_s", 0) for r in rep_metrics]
        ecc = [r.get("eccentric_s", 0) for r in rep_metrics]
        result["mean_concentric_s"] = float(np.mean(conc))
        result["mean_eccentric_s"] = float(np.mean(ecc))
    else:
        result["cadence_rpm"] = 0.0

    return result


def compute_rep_comparison(rep_metrics: list[dict]) -> dict[str, dict]:
    """Max/min/range/CV% across repetitions for key variables."""
    if len(rep_metrics) < 2:
        return {}

    df_reps = pd.DataFrame(rep_metrics)
    compare_cols = [
        "peak_velocity_ms",
        "mean_velocity_ms",
        "peak_power_W",
        "mean_power_W",
        "work_J",
        "rom_m",
        "duration_s",
        "concentric_s",
        "eccentric_s",
    ]

    result: dict[str, dict] = {}
    for col in compare_cols:
        if col not in df_reps.columns or df_reps[col].isna().all():
            continue
        vals = df_reps[col].dropna()
        if len(vals) < 2:
            continue
        mn = float(vals.mean())
        result[col] = {
            "max": float(vals.max()),
            "min": float(vals.min()),
            "range": float(vals.max() - vals.min()),
            "mean": mn,
            "std": float(vals.std()),
            "cv_pct": float(vals.std() / mn * 100) if mn > 0 else 0.0,
            "best_rep": int(vals.idxmax()) + 1,
            "worst_rep": int(vals.idxmin()) + 1,
        }
    return result


# ---------------------------------------------------------------------------
# Legacy single-rep helpers (kept for backward compatibility)
# ---------------------------------------------------------------------------


def evaluate_initial_pull_synchronism(df, phases):
    """Detects early knee extension that is not matched by hip opening."""
    start_frame = int(phases["bottom_frame"])
    end_frame = int(phases["end_frame"])
    if end_frame <= start_frame:
        return "INFO: Concentric phase is too short to evaluate synchronism."

    window_end = start_frame + max(1, int((end_frame - start_frame) * 0.15))
    early_pull = df.iloc[start_frame : window_end + 1]
    if early_pull.empty:
        return "INFO: Initial pull window is empty."

    critical = early_pull[
        (early_pull["shoulder_velocity_y"] > 0)
        & (early_pull["knee_extension_velocity_deg_s"] > 0)
        & (
            early_pull["knee_extension_velocity_deg_s"]
            > 2.0 * early_pull["hip_extension_velocity_deg_s"].abs()
        )
    ]
    if not critical.empty:
        frame = int(critical.index[0])
        return (
            "CRITICAL: Knee extension starts too far ahead of hip opening "
            f"(good-morning pattern risk) at frame {frame}."
        )
    return "PASS: Hips and knees rise synchronously at the start of the pull."


def identify_deadlift_phases(df):
    """Segments the movement based on vertical displacement of the shoulder (single-rep)."""
    lowest_shoulder_frame = df["right_shoulder_y_m"].idxmin()

    ecc_range = df.loc[:lowest_shoulder_frame]
    start_frame = 0
    for idx, row in ecc_range.iterrows():
        if abs(row["shoulder_velocity_y"]) > 0.05:
            start_frame = idx
            break

    post_lowest = df.loc[lowest_shoulder_frame:]
    end_frame = len(df) - 1
    for idx, row in post_lowest.iterrows():
        if idx > lowest_shoulder_frame and abs(row["shoulder_velocity_y"]) < 0.03:
            end_frame = idx
            break

    return {
        "start_frame": start_frame,
        "bottom_frame": lowest_shoulder_frame,
        "end_frame": end_frame,
    }


def classify_variant_at_bottom(df, bottom_frame):
    row = df.iloc[bottom_frame]
    knee_flex = row["knee_flexion_r"]
    shin_ang = row["shin_angle_ground"]

    if knee_flex <= 7.0 and shin_ang >= 85.0:
        return "STIFF-LEGGED DEADLIFT"
    elif 10.0 <= knee_flex <= 20.0 and 80.0 <= shin_ang <= 90.0:
        return "ROMANIAN DEADLIFT (RDL)"
    elif knee_flex > 45.0:
        return "CONVENTIONAL DEADLIFT"
    else:
        return "UNDETERMINED / MIXED VARIANT"


# ---------------------------------------------------------------------------
# IMU Plotting & Report
# ---------------------------------------------------------------------------


def _generate_imu_plots(
    a_filt: np.ndarray,
    velocity: np.ndarray,
    disp: np.ndarray,
    fps: float,
    reps: list[dict],
    rep_metrics: list[dict],
    output_dir: str,
    base_name: str,
) -> list[str]:
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    time_axis = np.arange(len(a_filt)) / fps
    plot_files: list[str] = []

    # --- Plot 1: Acceleration, velocity, displacement time series ---
    fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
    axes[0].plot(time_axis, a_filt, color="steelblue", linewidth=0.8)
    axes[0].set_ylabel("Vert. Acc (m/s²)")
    axes[0].set_title(f"IMU Deadlift Time Series — {base_name}")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(time_axis, velocity, color="darkorange", linewidth=0.8)
    axes[1].set_ylabel("Velocity (m/s)")
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(time_axis, disp, color="seagreen", linewidth=0.8)
    axes[2].set_ylabel("Displacement (m)")
    axes[2].set_xlabel("Time (s)")
    axes[2].grid(True, alpha=0.3)

    for rep in reps:
        for ax in axes:
            ax.axvline(rep["peak_frame"] / fps, color="red", linestyle="--", alpha=0.5)
        axes[2].annotate(
            f"R{rep['rep_number']}",
            (rep["peak_frame"] / fps, disp[rep["peak_frame"]]),
            fontsize=8,
            ha="center",
            va="bottom",
            color="red",
        )

    fig.tight_layout()
    p1 = os.path.join(output_dir, f"{base_name}_imu_timeseries_{ts}.png")
    fig.savefig(p1, dpi=150)
    plt.close(fig)
    plot_files.append(p1)

    # --- Plot 2: Per-rep bar charts (velocity, power, work) ---
    if len(rep_metrics) >= 2:
        rep_nums = [r["rep_number"] for r in rep_metrics]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].bar(rep_nums, [r["peak_velocity_ms"] for r in rep_metrics], color="steelblue")
        axes[0, 0].set_ylabel("Peak Velocity (m/s)")
        axes[0, 0].set_title("Peak Velocity per Rep")

        axes[0, 1].bar(rep_nums, [r["peak_power_W"] for r in rep_metrics], color="darkorange")
        axes[0, 1].set_ylabel("Peak Power (W)")
        axes[0, 1].set_title("Peak Power per Rep")

        axes[1, 0].bar(rep_nums, [r["work_J"] for r in rep_metrics], color="seagreen")
        axes[1, 0].set_ylabel("Work (J)")
        axes[1, 0].set_title("Work per Rep")
        axes[1, 0].set_xlabel("Rep #")

        axes[1, 1].bar(rep_nums, [r["duration_s"] for r in rep_metrics], color="mediumpurple")
        axes[1, 1].set_ylabel("Duration (s)")
        axes[1, 1].set_title("Duration per Rep")
        axes[1, 1].set_xlabel("Rep #")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3, axis="y")

        fig.suptitle(f"Rep-by-Rep Metrics — {base_name}", fontsize=13, y=1.01)
        fig.tight_layout()
        p2 = os.path.join(output_dir, f"{base_name}_imu_rep_bars_{ts}.png")
        fig.savefig(p2, dpi=150)
        plt.close(fig)
        plot_files.append(p2)

    # --- Plot 3: Velocity profile overlay per rep ---
    if len(rep_metrics) >= 2:
        fig, ax = plt.subplots(figsize=(10, 5))
        cmap = plt.get_cmap("viridis")
        for i, rep in enumerate(reps):
            s, e = rep["start_frame"], rep["end_frame"]
            v_rep = velocity[s : e + 1]
            t_rep = np.arange(len(v_rep)) / fps
            color = cmap(i / max(len(reps) - 1, 1))
            ax.plot(t_rep, v_rep, color=color, label=f"Rep {rep['rep_number']}", alpha=0.8)
        ax.set_xlabel("Time within rep (s)")
        ax.set_ylabel("Velocity (m/s)")
        ax.set_title(f"Velocity Profile Overlay — {base_name}")
        ax.legend(fontsize=7, ncol=3)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        p3 = os.path.join(output_dir, f"{base_name}_imu_velocity_overlay_{ts}.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        plot_files.append(p3)

    return plot_files


def _generate_imu_html_report(
    rep_metrics: list[dict],
    cadence: dict,
    comparison: dict[str, dict],
    plot_files: list[str],
    output_dir: str,
    base_name: str,
    weight_kg: float,
    fps: float,
) -> str:
    report_path = os.path.join(output_dir, f"{base_name}_imu_biomechanical_report.html")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # --- Rep table rows ---
    rep_rows = ""
    for r in rep_metrics:
        rep_rows += f"""
        <tr>
            <td>{r["rep_number"]}</td>
            <td>{r["duration_s"]:.2f}</td>
            <td>{r["concentric_s"]:.2f}</td>
            <td>{r["eccentric_s"]:.2f}</td>
            <td>{r["peak_velocity_ms"]:.3f}</td>
            <td>{r["mean_velocity_ms"]:.3f}</td>
            <td>{r["peak_power_W"]:.1f}</td>
            <td>{r["mean_power_W"]:.1f}</td>
            <td>{r["work_J"]:.2f}</td>
            <td>{r["rom_m"]:.3f}</td>
            <td>{r["peak_force_N"]:.1f}</td>
        </tr>"""

    # --- Comparison table rows ---
    comp_rows = ""
    label_map = {
        "peak_velocity_ms": "Peak Velocity (m/s)",
        "mean_velocity_ms": "Mean Velocity (m/s)",
        "peak_power_W": "Peak Power (W)",
        "mean_power_W": "Mean Power (W)",
        "work_J": "Work (J)",
        "rom_m": "ROM (m)",
        "duration_s": "Duration (s)",
        "concentric_s": "Concentric (s)",
        "eccentric_s": "Eccentric (s)",
    }
    for key, stats in comparison.items():
        name = label_map.get(key, key)
        comp_rows += f"""
        <tr>
            <td>{name}</td>
            <td>{stats["max"]:.3f}</td>
            <td>{stats["min"]:.3f}</td>
            <td>{stats["range"]:.3f}</td>
            <td>{stats["mean"]:.3f}</td>
            <td>{stats["std"]:.3f}</td>
            <td>{stats["cv_pct"]:.1f}%</td>
            <td>#{stats["best_rep"]}</td>
            <td>#{stats["worst_rep"]}</td>
        </tr>"""

    # --- Images ---
    images_html = ""
    for pf in plot_files:
        images_html += f'<div class="img-container"><img src="{os.path.basename(pf)}"></div>\n'

    html = f"""<!DOCTYPE html>
<html>
<head>
    <title>vailá - IMU Deadlift Biomechanical Report</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 30px; background: #fafafa; color: #333; }}
        h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
        h2 {{ color: #2c5282; margin-top: 30px; }}
        table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }}
        th, td {{ border: 1px solid #e2e8f0; padding: 10px; text-align: center; }}
        th {{ background: #ebf8ff; color: #2b6cb0; }}
        .summary-box {{ background: #e2e8f0; padding: 15px; border-left: 6px solid #4a5568;
                        font-size: 1.1em; margin: 20px 0; }}
        .img-container {{ text-align: center; margin: 20px 0; }}
        img {{ max-width: 90%; height: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        .highlight {{ font-weight: bold; color: #2b6cb0; }}
    </style>
</head>
<body>
    <h1>IMU Deadlift Biomechanical Report</h1>
    <p><strong>File:</strong> {base_name}</p>
    <p><strong>Date:</strong> {now}</p>
    <p><strong>Barbell Weight:</strong> {weight_kg:.1f} kg &nbsp;|&nbsp;
       <strong>IMU Sampling Rate:</strong> {fps:.1f} Hz</p>

    <div class="summary-box">
        <strong>Repetitions Detected:</strong> {cadence["n_reps"]}
        &nbsp;|&nbsp; <strong>Cadence:</strong> {cadence.get("cadence_rpm", 0):.1f} reps/min
        &nbsp;|&nbsp; <strong>Mean Rep Time:</strong> {cadence["mean_rep_time_s"]:.2f} ± {cadence["std_rep_time_s"]:.2f} s
    </div>

    <h2>Per-Repetition Metrics</h2>
    <table>
        <tr>
            <th>Rep</th><th>Duration (s)</th><th>Conc. (s)</th><th>Ecc. (s)</th>
            <th>Peak Vel (m/s)</th><th>Mean Vel (m/s)</th>
            <th>Peak Power (W)</th><th>Mean Power (W)</th>
            <th>Work (J)</th><th>ROM (m)</th><th>Peak Force (N)</th>
        </tr>
        {rep_rows}
    </table>

    <h2>Rep-to-Rep Comparison</h2>
    <table>
        <tr>
            <th>Metric</th><th>Max</th><th>Min</th><th>Range</th>
            <th>Mean</th><th>Std</th><th>CV%</th>
            <th>Best Rep</th><th>Worst Rep</th>
        </tr>
        {comp_rows}
    </table>

    <h2>Time Series & Visualizations</h2>
    {images_html}

</body>
</html>"""

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html)
    return report_path


# ---------------------------------------------------------------------------
# MediaPipe Plotting & Report (enhanced)
# ---------------------------------------------------------------------------


def generate_deadlift_plots(
    df,
    phases,
    output_dir,
    base_name,
    reps=None,
    com_x=None,
    com_y=None,
    rep_metrics=None,
):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bot = int(phases["bottom_frame"])
    if "time_s" in df.columns:
        time_axis = df["time_s"].to_numpy(dtype=float)
    else:
        time_axis = np.arange(len(df), dtype=float)
    plot_files: list[str] = []

    # Plot 1: Knee & Shin Kinematics
    plt.figure(figsize=(10, 5))
    plt.plot(time_axis, df["knee_flexion_r"], label="Knee Flexion (deg)", color="red")
    plt.plot(time_axis, df["shin_angle_ground"], label="Shin-to-Ground Angle (deg)", color="blue")
    plt.axvline(time_axis[bot], color="green", linestyle="--", label="Bottom Turnaround")
    if reps and len(reps) > 1:
        for rep in reps:
            bf = int(rep.get("bottom_frame", rep.get("peak_frame", 0)))
            if 0 <= bf < len(time_axis):
                plt.axvline(time_axis[bf], color="gray", linestyle=":", alpha=0.5)
    plt.title(f"Lower Limb Kinematics - {base_name}")
    plt.xlabel("Time (s)")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(output_dir, f"{base_name}_lower_limb_{timestamp}.png")
    plt.savefig(p1, dpi=150)
    plt.close()
    plot_files.append(p1)

    # Plot 2: Spine Mechanics and Bar Proximity
    plt.figure(figsize=(10, 5))
    plt.plot(
        time_axis, df["spine_deviation"], label="Spine Deviation from Setup (deg)", color="purple"
    )
    plt.plot(
        time_axis,
        df["bar_path_proximity_m"] * 100,
        label="Bar horizontal offset from knee (cm)",
        color="orange",
    )
    plt.axvline(time_axis[bot], color="green", linestyle="--")
    plt.title("Spine and Bar Path Safety Metrics")
    plt.xlabel("Time (s)")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(output_dir, f"{base_name}_safety_metrics_{timestamp}.png")
    plt.savefig(p2, dpi=150)
    plt.close()
    plot_files.append(p2)

    # Plot 3: Body distances and vertical COM/bar path
    if "foot_spread_m" in df.columns:
        fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
        axes[0].plot(time_axis, df["foot_spread_m"] * 100, color="teal", label="Foot spread")
        axes[0].plot(time_axis, df["knee_spread_m"] * 100, color="coral", label="Knee spread")
        axes[0].set_ylabel("Distance (cm)")
        axes[0].set_title(f"Body Distances - {base_name}")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(
            time_axis,
            df["hand_foot_horiz_l_m"] * 100,
            color="steelblue",
            label="Hand-Foot L",
        )
        axes[1].plot(
            time_axis,
            df["hand_foot_horiz_r_m"] * 100,
            color="darkorange",
            label="Hand-Foot R",
        )
        axes[1].set_ylabel("Horiz. offset (cm)")
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        if com_y is not None:
            axes[2].plot(time_axis, com_y, color="seagreen", label="COM Y (m)")
        if "bar_height_m" in df.columns:
            axes[2].plot(
                time_axis,
                df["bar_height_m"],
                color="crimson",
                alpha=0.8,
                label="Bar height (wrist avg Y)",
            )
            for rep in reps or []:
                peak_frame = int(rep.get("count_frame", rep.get("peak_frame", rep.get("end_frame", 0))))
                bottom_frame = int(rep.get("bottom_frame", 0))
                if 0 <= peak_frame < len(df):
                    axes[2].plot(
                        time_axis[peak_frame],
                        df["bar_height_m"].iloc[peak_frame],
                        "^",
                        color="black",
                        markersize=5,
                        label="Peak/count" if rep.get("rep_number") == 1 else None,
                    )
                if 0 <= bottom_frame < len(df):
                    axes[2].plot(
                        time_axis[bottom_frame],
                        df["bar_height_m"].iloc[bottom_frame],
                        "v",
                        color="navy",
                        markersize=5,
                        label="Valley/low" if rep.get("rep_number") == 1 else None,
                    )
                    dt = float(np.nanmedian(np.diff(time_axis))) if len(time_axis) > 1 else 0.0
                    half_window = max(1, int(round(0.15 / dt))) if dt > 0 else 1
                    low_start = max(0, bottom_frame - half_window)
                    low_end = min(len(df) - 1, bottom_frame + half_window)
                    axes[2].axvspan(time_axis[low_start], time_axis[low_end], color="navy", alpha=0.08)
        axes[2].set_ylabel("Vertical Position [m]")
        axes[2].set_xlabel("Time (s)")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)

        for rep in reps or []:
            bf = int(rep.get("bottom_frame", rep.get("peak_frame", 0)))
            pf = int(rep.get("count_frame", rep.get("peak_frame", rep.get("end_frame", 0))))
            for ax in axes:
                if 0 <= bf < len(time_axis):
                    ax.axvline(time_axis[bf], color="gray", linestyle=":", alpha=0.35)
                if 0 <= pf < len(time_axis):
                    ax.axvline(time_axis[pf], color="black", linestyle="--", alpha=0.18)

        fig.tight_layout()
        p3 = os.path.join(output_dir, f"{base_name}_body_distances_{timestamp}.png")
        fig.savefig(p3, dpi=150)
        plt.close(fig)
        plot_files.append(p3)

    # Plot 4: Per-rep bar charts (if multi-rep)
    if rep_metrics and len(rep_metrics) >= 2:
        rep_nums = [r["rep_number"] for r in rep_metrics]
        fig, axes = plt.subplots(2, 2, figsize=(12, 8))

        axes[0, 0].bar(rep_nums, [r["peak_velocity_ms"] for r in rep_metrics], color="steelblue")
        axes[0, 0].set_ylabel("Peak Velocity (m/s)")
        axes[0, 0].set_title("Peak Velocity per Rep")

        axes[0, 1].bar(rep_nums, [r["peak_power_W"] for r in rep_metrics], color="darkorange")
        axes[0, 1].set_ylabel("Peak Power (W)")
        axes[0, 1].set_title("Peak Power per Rep")

        axes[1, 0].bar(rep_nums, [r["work_J"] for r in rep_metrics], color="seagreen")
        axes[1, 0].set_ylabel("Work (J)")
        axes[1, 0].set_title("Work per Rep")
        axes[1, 0].set_xlabel("Rep #")

        axes[1, 1].bar(rep_nums, [r["duration_s"] for r in rep_metrics], color="mediumpurple")
        axes[1, 1].set_ylabel("Duration (s)")
        axes[1, 1].set_title("Duration per Rep")
        axes[1, 1].set_xlabel("Rep #")

        for ax in axes.flat:
            ax.grid(True, alpha=0.3, axis="y")
        fig.suptitle(f"Rep-by-Rep Metrics - {base_name}", fontsize=13, y=1.01)
        fig.tight_layout()
        p4 = os.path.join(output_dir, f"{base_name}_rep_bars_{timestamp}.png")
        fig.savefig(p4, dpi=150)
        plt.close(fig)
        plot_files.append(p4)

    return plot_files

def _deadlift_body_segments() -> list[tuple[str, str]]:
    return [
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "left_shoulder"),
        ("right_hip", "right_shoulder"),
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]


def _get_point_m(row: pd.Series, name: str) -> tuple[float, float] | None:
    x_col = f"{name}_x_m"
    y_col = f"{name}_y_m"
    if x_col not in row.index or y_col not in row.index:
        return None
    x = row[x_col]
    y = row[y_col]
    if pd.isna(x) or pd.isna(y):
        return None
    return float(x), float(y)


def _deadlift_axis_limits(df: pd.DataFrame, frames: list[int]) -> tuple[float, float, float, float]:
    xs: list[float] = []
    ys: list[float] = []
    for frame in frames:
        row = df.iloc[int(frame)]
        for a, b in _deadlift_body_segments():
            pa = _get_point_m(row, a)
            pb = _get_point_m(row, b)
            if pa and pb:
                xs.extend([pa[0], pb[0]])
                ys.extend([pa[1], pb[1]])
        if "com_x_m" in row.index and "com_y_m" in row.index:
            xs.append(float(row["com_x_m"]))
            ys.append(float(row["com_y_m"]))
        if "bar_marker_x_m" in row.index and "bar_marker_y_m" in row.index:
            xs.append(float(row["bar_marker_x_m"]))
            ys.append(float(row["bar_marker_y_m"]))
    if not xs or not ys:
        return -1.0, 1.0, -1.0, 1.0
    x_min, x_max = min(xs), max(xs)
    y_min, y_max = min(ys), max(ys)
    pad = max(x_max - x_min, y_max - y_min, 0.5) * 0.15
    return x_min - pad, x_max + pad, y_min - pad, y_max + pad


def _draw_deadlift_stick_frame(ax, row: pd.Series, title: str | None = None) -> None:
    for a, b in _deadlift_body_segments():
        pa = _get_point_m(row, a)
        pb = _get_point_m(row, b)
        if pa and pb:
            ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="black", lw=2)

    ls = _get_point_m(row, "left_shoulder")
    rs = _get_point_m(row, "right_shoulder")
    nose = _get_point_m(row, "nose")
    if ls and rs and nose:
        mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
        ax.plot([mid[0], nose[0]], [mid[1], nose[1]], color="black", lw=2)

    if "com_x_m" in row.index and "com_y_m" in row.index:
        ax.plot(
            float(row["com_x_m"]),
            float(row["com_y_m"]),
            "o",
            color="orange",
            markersize=6,
            markeredgecolor="black",
            label="COM",
        )
    if "bar_marker_x_m" in row.index and "bar_marker_y_m" in row.index:
        ax.plot(
            float(row["bar_marker_x_m"]),
            float(row["bar_marker_y_m"]),
            "s",
            color="crimson",
            markersize=5,
            label="Bar marker",
        )
    if title:
        ax.set_title(title, fontsize=10)
    ax.set_aspect("equal")
    ax.grid(True, alpha=0.2)


def _deadlift_phase_keyframes(reps: list[dict], fps: float) -> list[dict]:
    events: list[dict] = []
    if not reps:
        return events

    first = reps[0]
    initial_phase = "Low phase" if first.get("start_phase") == "low" else "Standing/setup"
    events.append(
        {
            "frame": int(first["start_frame"]),
            "phase": initial_phase,
            "counter": 0,
            "rep_number": 0,
            "time_s": int(first["start_frame"]) / fps,
        }
    )

    for rep in reps:
        rep_number = int(rep["rep_number"])
        events.append(
            {
                "frame": int(rep["bottom_frame"]),
                "phase": f"Low before rep {rep_number}",
                "counter": rep_number - 1,
                "rep_number": rep_number,
                "time_s": int(rep["bottom_frame"]) / fps,
            }
        )
        events.append(
            {
                "frame": int(rep["count_frame"]),
                "phase": f"Standing rep {rep_number}",
                "counter": rep_number,
                "rep_number": rep_number,
                "time_s": int(rep["count_frame"]) / fps,
            }
        )

    cleaned: list[dict] = []
    for event in events:
        if (
            cleaned
            and cleaned[-1]["frame"] == event["frame"]
            and cleaned[-1]["counter"] == event["counter"]
        ):
            continue
        cleaned.append(event)
    return cleaned


def generate_deadlift_stickfigure_phases(
    df: pd.DataFrame,
    reps: list[dict],
    output_dir: str,
    base_name: str,
    fps: float,
) -> str | None:
    events = _deadlift_phase_keyframes(reps, fps)
    if not events:
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    frames = [max(0, min(len(df) - 1, int(event["frame"]))) for event in events]
    x_min, x_max, y_min, y_max = _deadlift_axis_limits(df, frames)

    n_cols = min(5, max(1, len(events)))
    n_rows = int(math.ceil(len(events) / n_cols))
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3.2 * n_cols, 3.4 * n_rows))
    axes_arr = np.ravel(np.atleast_1d(axes))

    for ax, event in zip(axes_arr, events, strict=False):
        frame = max(0, min(len(df) - 1, int(event["frame"])))
        title = (
            f"{event['phase']}\n"
            f"Counter {event['counter']} | Frame {frame}\n"
            f"Time {event['time_s']:.2f} s"
        )
        _draw_deadlift_stick_frame(ax, df.iloc[frame], title)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)

    for ax in axes_arr[len(events) :]:
        ax.axis("off")

    handles, labels = axes_arr[0].get_legend_handles_labels()
    if handles:
        fig.legend(handles, labels, loc="lower center", ncol=2)
    fig.suptitle(f"Deadlift Sequential Low/Standing Keyframes - {base_name}", fontsize=13)
    fig.tight_layout(rect=(0, 0.04, 1, 0.96))
    out = os.path.join(output_dir, f"{base_name}_stickfigures_phases_{timestamp}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    return out


def generate_deadlift_animation_gif(
    df: pd.DataFrame,
    reps: list[dict],
    output_dir: str,
    base_name: str,
    fps: float,
    frames_between: int = 2,
    figsize=(5, 5),
    duration_s: float = 1.2,
) -> str | None:
    try:
        import imageio
    except Exception:
        print("Warning: imageio not available; skipping GIF generation")
        return None

    events = _deadlift_phase_keyframes(reps, fps)
    if not events:
        return None

    frames = [max(0, min(len(df) - 1, int(event["frame"]))) for event in events]
    x_min, x_max, y_min, y_max = _deadlift_axis_limits(df, frames)

    images = []
    for event in events:
        frame = max(0, min(len(df) - 1, int(event["frame"])))
        fig, ax = plt.subplots(figsize=figsize)
        _draw_deadlift_stick_frame(ax, df.iloc[frame], None)
        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        label = (
            f"{event['phase']}\n"
            f"Counter: {event['counter']}\n"
            f"Frame: {frame}\n"
            f"Time: {event['time_s']:.2f} s"
        )
        ax.text(
            0.02,
            0.98,
            label,
            transform=ax.transAxes,
            ha="left",
            va="top",
            fontsize=11,
            bbox={"facecolor": "white", "edgecolor": "black", "alpha": 0.85},
        )
        ax.axis("off")
        try:
            from matplotlib.backends.backend_agg import FigureCanvasAgg

            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(h, w, 4)
            images.append(img[:, :, :3].copy())
        except Exception as e:
            print(f"GIF frame render failed at frame {frame}: {e}")
        plt.close(fig)

    if not images:
        return None
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"{base_name}_deadlift_anim_{timestamp}.gif")
    try:
        imageio.mimsave(gif_path, images, duration=duration_s, loop=0)
        print(f"Saved GIF animation: {gif_path}")
        return gif_path
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        return None


def generate_html_report(
    df,
    phases,
    variant,
    plot_files,
    output_dir,
    base_name,
    reps=None,
    rep_metrics=None,
    cadence=None,
    comparison=None,
    com_x=None,
    com_y=None,
):
    bot_row = df.iloc[phases["bottom_frame"]]
    setup_row = df.iloc[phases["start_frame"]]
    report_path = os.path.join(output_dir, f"{base_name}_biomechanical_report.html")
    now = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    # Form Quality Validations
    spine_warning = (
        "PASS" if bot_row["spine_deviation"] < 5.0 else "WARNING: Excessive Lumbar Flexion Risk"
    )
    bar_warning = (
        "PASS"
        if bot_row["bar_path_proximity_m"] <= 0.06
        else "WARNING: Mechanical Moment Arm Too Large"
    )
    shin_warning = (
        "PASS" if bot_row["shin_angle_ground"] >= 80.0 else "WARNING: Excessive Knee Forward Travel"
    )
    arm_delta = setup_row["arm_verticality_delta_m"]
    if arm_delta > 0.05:
        arm_status = "FAIL: Arm is angled forward. Hip position is too low."
    elif arm_delta < -0.05:
        arm_status = "FAIL: Arm is angled backward. Hip position is too high."
    else:
        arm_status = "PASS: Arm is vertical over the bar."

    midfoot_error = setup_row["bar_midfoot_error_m"]
    if abs(midfoot_error) > 0.03:
        midfoot_status = "WARNING: Position the bar over midfoot before pulling."
    else:
        midfoot_status = "PASS: Bar is aligned over midfoot at setup."
    synchronism_status = evaluate_initial_pull_synchronism(df, phases)

    # --- Summary section for multi-rep ---
    n_reps = cadence.get("n_reps", 1) if cadence else 1
    cadence_rpm = cadence.get("cadence_rpm", 0) if cadence else 0
    mean_rep_t = cadence.get("mean_rep_time_s", 0) if cadence else 0
    std_rep_t = cadence.get("std_rep_time_s", 0) if cadence else 0

    # Body distances at bottom frame
    foot_sp = float(bot_row.get("foot_spread_m", 0)) * 100
    knee_sp = float(bot_row.get("knee_spread_m", 0)) * 100
    hf_l = float(bot_row.get("hand_foot_horiz_l_m", 0)) * 100
    hf_r = float(bot_row.get("hand_foot_horiz_r_m", 0)) * 100

    # --- Rep table ---
    rep_table_html = ""
    if rep_metrics and len(rep_metrics) >= 1:
        rep_rows = ""
        for r in rep_metrics:
            rep_rows += f"""
            <tr>
                <td>{r["rep_number"]}</td>
                <td>{r["duration_s"]:.2f}</td>
                <td>{r["concentric_s"]:.2f}</td>
                <td>{r["eccentric_s"]:.2f}</td>
                <td>{r["peak_velocity_ms"]:.3f}</td>
                <td>{r["mean_power_W"]:.1f}</td>
                <td>{r["work_J"]:.2f}</td>
                <td>{r["rom_m"]:.3f}</td>
                <td>{r.get("bar_rom_m", 0):.3f}</td>
                <td>{r.get("foot_spread_m", 0) * 100:.1f}</td>
                <td>{r.get("knee_spread_m", 0) * 100:.1f}</td>
            </tr>"""
        rep_table_html = f"""
        <h2>Per-Repetition Metrics</h2>
        <table>
            <tr>
                <th>Rep</th><th>Duration (s)</th><th>Conc. (s)</th><th>Ecc. (s)</th>
                <th>Peak Vel (m/s)</th><th>Mean Power (W)</th>
                <th>Work (J)</th><th>COM ROM (m)</th><th>Bar ROM (m)</th>
                <th>Foot Sp. (cm)</th><th>Knee Sp. (cm)</th>
            </tr>
            {rep_rows}
        </table>"""

    # --- Comparison table ---
    comp_table_html = ""
    if comparison:
        label_map = {
            "peak_velocity_ms": "Peak Velocity (m/s)",
            "mean_velocity_ms": "Mean Velocity (m/s)",
            "peak_power_W": "Peak Power (W)",
            "mean_power_W": "Mean Power (W)",
            "work_J": "Work (J)",
            "rom_m": "ROM (m)",
            "duration_s": "Duration (s)",
            "concentric_s": "Concentric (s)",
            "eccentric_s": "Eccentric (s)",
        }
        comp_rows = ""
        for key, stats in comparison.items():
            name = label_map.get(key, key)
            comp_rows += f"""
            <tr>
                <td>{name}</td>
                <td>{stats["max"]:.3f}</td>
                <td>{stats["min"]:.3f}</td>
                <td>{stats["range"]:.3f}</td>
                <td>{stats["mean"]:.3f}</td>
                <td>{stats["cv_pct"]:.1f}%</td>
                <td>#{stats["best_rep"]}</td>
                <td>#{stats["worst_rep"]}</td>
            </tr>"""
        comp_table_html = f"""
        <h2>Rep-to-Rep Comparison</h2>
        <table>
            <tr>
                <th>Metric</th><th>Max</th><th>Min</th><th>Range</th>
                <th>Mean</th><th>CV%</th><th>Best Rep</th><th>Worst Rep</th>
            </tr>
            {comp_rows}
        </table>"""

    html_content = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>vailá - Deadlift Kinematic Evaluation Report</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 30px; background-color: #fafafa; color: #333; }}
            h1 {{ color: #1a365d; border-bottom: 3px solid #2b6cb0; padding-bottom: 10px; }}
            h2 {{ color: #2c5282; margin-top: 30px; }}
            table {{ width: 100%; border-collapse: collapse; margin-top: 15px; background: white; }}
            th, td {{ border: 1px solid #e2e8f0; padding: 12px; text-align: left; }}
            th {{ background-color: #ebf8ff; color: #2b6cb0; }}
            .status-pass {{ color: green; font-weight: bold; }}
            .status-warn {{ color: red; font-weight: bold; }}
            .variant-box {{ background-color: #e2e8f0; padding: 15px; border-left: 6px solid #4a5568; font-size: 1.2em; font-weight: bold; margin: 20px 0; }}
            .summary-box {{ background: #e2e8f0; padding: 15px; border-left: 6px solid #2b6cb0; font-size: 1.1em; margin: 20px 0; }}
            .img-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 90%; height: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>Deadlift Biomechanical Diagnosis</h1>
        <p><strong>Analysis Target:</strong> {base_name}</p>
        <p><strong>Execution Date:</strong> {now}</p>

        <div class="variant-box">
            Classified Variant at Peak Depth: {variant}
        </div>

        <div class="summary-box">
            <strong>Repetitions:</strong> {n_reps}
            &nbsp;|&nbsp; <strong>Cadence:</strong> {cadence_rpm:.1f} reps/min
            &nbsp;|&nbsp; <strong>Mean Rep Time:</strong> {mean_rep_t:.2f} ± {std_rep_t:.2f} s
        </div>

        <h2>Kinematic Metrics Summary (Turnaround Frame)</h2>
        <table>
            <tr><th>Evaluated Criterion</th><th>Measured Value</th><th>Safety Threshold</th><th>Status</th></tr>
            <tr>
                <td>Knee Flexion (Internal Angle)</td>
                <td>{bot_row["knee_flexion_r"]:.1f}°</td>
                <td>RDL: 10° - 15° | Stiff: 0° - 5°</td>
                <td>Informative</td>
            </tr>
            <tr>
                <td>Tibia Angle to Ground Plane</td>
                <td>{bot_row["shin_angle_ground"]:.1f}°</td>
                <td>90° Vertical (±10° Tolerance)</td>
                <td class="{"status-pass" if "PASS" in shin_warning else "status-warn"}">{shin_warning}</td>
            </tr>
            <tr>
                <td>Spine Angular Deviation from Setup</td>
                <td>{bot_row["spine_deviation"]:.1f}°</td>
                <td>&lt; 5.0° Flexion Extension Change</td>
                <td class="{"status-pass" if "PASS" in spine_warning else "status-warn"}">{spine_warning}</td>
            </tr>
            <tr>
                <td>Bar Path Distance from Tibia</td>
                <td>{bot_row["bar_path_proximity_m"] * 100:.1f} cm</td>
                <td>&le; 5.0 cm Horizontal Gap</td>
                <td class="{"status-pass" if "PASS" in bar_warning else "status-warn"}">{bar_warning}</td>
            </tr>
            <tr>
                <td>Foot Spread (Ankle-to-Ankle)</td>
                <td>{foot_sp:.1f} cm</td>
                <td>Individual-dependent</td>
                <td>Informative</td>
            </tr>
            <tr>
                <td>Knee Spread (Knee-to-Knee)</td>
                <td>{knee_sp:.1f} cm</td>
                <td>Track over ankle</td>
                <td>Informative</td>
            </tr>
            <tr>
                <td>Hand-Foot Horizontal Offset (L / R)</td>
                <td>{hf_l:.1f} / {hf_r:.1f} cm</td>
                <td>&le; 5.0 cm for vertical grip</td>
                <td>Informative</td>
            </tr>
            <tr>
                <td>Setup Arm Verticality</td>
                <td>{arm_delta * 100:.1f} cm shoulder-wrist horizontal delta</td>
                <td>&le; 5.0 cm absolute offset</td>
                <td class="{"status-pass" if arm_status.startswith("PASS") else "status-warn"}">{arm_status}</td>
            </tr>
            <tr>
                <td>Bar Over Midfoot Setup</td>
                <td>{midfoot_error * 100:.1f} cm wrist-midfoot horizontal error</td>
                <td>&le; 3.0 cm absolute offset</td>
                <td class="{"status-pass" if midfoot_status.startswith("PASS") else "status-warn"}">{midfoot_status}</td>
            </tr>
            <tr>
                <td>Initial Pull Synchronism</td>
                <td>First 15% of concentric phase</td>
                <td>Knee extension rate &le; 2x hip opening rate</td>
                <td class="{"status-pass" if synchronism_status.startswith("PASS") else "status-warn"}">{synchronism_status}</td>
            </tr>
        </table>

        {rep_table_html}
        {comp_table_html}

        <h2>Kinematic Curve Reconstructions</h2>
    """
    for pf in plot_files:
        html_content += f'<div class="img-container"><img src="{os.path.basename(pf)}"></div>'

    try:
        maybe_gifs = [p for p in os.listdir(output_dir) if p.lower().endswith(".gif")]
        for gif_name in sorted(maybe_gifs):
            html_content += (
                f'<div class="img-container"><img src="{gif_name}" '
                f'alt="Deadlift stick figure animation"><p><em>{gif_name}</em></p></div>'
            )
    except Exception:
        pass

    html_content += "</body></html>"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path


# ---------------------------------------------------------------------------
# Main processing pipelines
# ---------------------------------------------------------------------------


def process_imu_deadlift_data(input_file: str, output_dir: str) -> bool:
    """Full IMU-based deadlift analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    ctx = _load_context_from_toml(Path(input_file).parent) or {}
    params = _load_parameters_file(Path(input_file).parent)

    fps = ctx.get("imu_fps", 25.0)
    weight_kg = ctx.get("weight_kg", 20.0)

    if params:
        import contextlib

        with contextlib.suppress(ValueError, TypeError):
            weight_kg = float(params.get("weight", weight_kg))

    print(f"[IMU] Barbell weight: {weight_kg:.1f} kg | Sample rate: {fps:.1f} Hz")

    g_unit, g_mag = _estimate_gravity(df, n_static=min(30, len(df) // 4))
    a_vert = _compute_vertical_acc(df, g_unit, g_mag)

    reps, velocity, disp, a_filt = detect_reps_imu(a_vert, fps)
    print(f"[IMU] Detected {len(reps)} repetitions")

    rep_metrics = compute_rep_imu_metrics(a_filt, velocity, disp, fps, weight_kg, reps)
    cadence = compute_cadence(rep_metrics, fps)
    comparison = compute_rep_comparison(rep_metrics)

    # Build processed DataFrame
    ts_df = pd.DataFrame(
        {
            "time_s": np.arange(len(a_vert)) / fps,
            "vert_acc_ms2": a_filt,
            "velocity_ms": velocity,
            "displacement_m": disp,
        }
    )
    proc_df = pd.concat([df, ts_df], axis=1)

    plot_files = _generate_imu_plots(
        a_filt, velocity, disp, fps, reps, rep_metrics, output_dir, base_name
    )
    report_path = _generate_imu_html_report(
        rep_metrics, cadence, comparison, plot_files, output_dir, base_name, weight_kg, fps
    )

    # Save processed CSVs
    ts_stamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    proc_df.to_csv(
        os.path.join(output_dir, f"{base_name}_imu_processed_{ts_stamp}.csv"), index=False
    )

    rep_df = pd.DataFrame(rep_metrics)
    rep_df.to_csv(os.path.join(output_dir, f"{base_name}_rep_metrics_{ts_stamp}.csv"), index=False)

    print("\n[SUCCESS] IMU Deadlift Analysis Complete")
    print(f"  Reps: {cadence['n_reps']}")
    print(f"  Cadence: {cadence.get('cadence_rpm', 0):.1f} reps/min")
    if rep_metrics:
        best_vel = max(r["peak_velocity_ms"] for r in rep_metrics)
        best_pow = max(r["peak_power_W"] for r in rep_metrics)
        print(f"  Best peak velocity: {best_vel:.3f} m/s")
        print(f"  Best peak power: {best_pow:.1f} W")
    print(f"  Report: {report_path}")
    return True


def process_mediapipe_deadlift_data(input_file, output_dir, overrides: dict | None = None):
    """Full MediaPipe pose-based deadlift analysis pipeline."""
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    has_metric_pose = {"left_wrist_y_m", "right_wrist_y_m", "right_knee_y_m"}.issubset(df.columns)
    if not has_metric_pose:
        for col in [c for c in df.columns if c.endswith("_y")]:
            df[col] = 1.0 - df[col]

    ctx = _load_context_from_toml(Path(input_file).parent) or {
        "fps": 30.0,
        "shank_length_m": 0.40,
        "mass_kg": 75.0,
        "weight_kg": 20.0,
        "use_total_mass_for_power": True,
    }
    ctx = _merge_deadlift_overrides(ctx, Path(input_file).parent, overrides)
    params = _load_parameters_file(Path(input_file).parent)
    fps = ctx["fps"]
    mass_kg = ctx.get("mass_kg", 75.0)
    barbell_kg = ctx.get("weight_kg", 20.0)
    use_total_mass_for_power = ctx.get("use_total_mass_for_power", True)
    expected_reps = None

    if params:
        with contextlib.suppress(ValueError, TypeError):
            expected_reps = int(float(params.get("real_repetition_count", "")))
        if not (overrides and (overrides.get("subject_params") or overrides.get("barbell_mass_kg") is not None or overrides.get("weight_kg") is not None)) and not ctx.get("subject_parameter_file"):
            with contextlib.suppress(ValueError, TypeError):
                barbell_kg = float(params.get("weight", barbell_kg))

    factor = 1.0 if has_metric_pose else calc_conversion_factor(df, ctx.get("shank_length_m", 0.40))

    if not has_metric_pose:
        coord_cols = [col for col in df.columns if col.endswith(("_x", "_y", "_z"))]
        metric_coords = df[coord_cols].mul(factor).add_suffix("_m")
        df = pd.concat([df, metric_coords], axis=1)

    df = process_deadlift_kinematics(df, fps, factor)
    df["time_s"] = np.arange(len(df), dtype=float) / fps

    # Center of mass estimation
    com_x, com_y = compute_center_of_mass_2d(df)
    df["com_x_m"] = com_x
    df["com_y_m"] = com_y
    df["com_velocity_y"] = np.gradient(com_y, 1.0 / fps)
    df["body_mass_kg"] = mass_kg
    df["barbell_mass_kg"] = barbell_kg
    df["analysis_mass_kg"] = mass_kg + barbell_kg if use_total_mass_for_power else mass_kg
    df["calibration_factor_m_per_unit"] = factor
    if ctx.get("kinematics_parameter_file"):
        df["kinematics_parameter_file"] = str(ctx.get("kinematics_parameter_file"))
    if ctx.get("subject_parameter_file"):
        df["subject_parameter_file"] = str(ctx.get("subject_parameter_file"))

    # Legacy single-rep phase detection (for variant classification & form checks)
    phases = identify_deadlift_phases(df)
    variant = classify_variant_at_bottom(df, phases["bottom_frame"])

    # Multi-rep detection from averaged wrist/bar-height Y.
    reps = detect_reps_mediapipe(df, fps, expected_reps=expected_reps)
    print(f"[MediaPipe] Detected {len(reps)} repetitions")

    # Per-rep metrics (COM-based power/work, distances, bar ROM)
    rep_metrics = compute_mediapipe_rep_metrics(
        df,
        com_x,
        com_y,
        fps,
        mass_kg,
        barbell_kg,
        reps,
        use_total_mass_for_power=use_total_mass_for_power,
    )
    cadence_stats = compute_cadence(rep_metrics, fps)
    comparison = compute_rep_comparison(rep_metrics)

    plot_files = generate_deadlift_plots(
        df,
        phases,
        output_dir,
        base_name,
        reps=reps,
        com_x=com_x,
        com_y=com_y,
        rep_metrics=rep_metrics,
    )
    stickfig_path = generate_deadlift_stickfigure_phases(df, reps, output_dir, base_name, fps)
    if stickfig_path:
        plot_files.append(stickfig_path)
    generate_deadlift_animation_gif(
        df,
        reps,
        output_dir,
        base_name,
        fps,
        frames_between=2,
        figsize=(5, 5),
        duration_s=float(ctx.get("gif_duration_s", 1.2)),
    )

    report_path = generate_html_report(
        df,
        phases,
        variant,
        plot_files,
        output_dir,
        base_name,
        reps=reps,
        rep_metrics=rep_metrics,
        cadence=cadence_stats,
        comparison=comparison,
        com_x=com_x,
        com_y=com_y,
    )

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(
        os.path.join(output_dir, f"{base_name}_deadlift_kinematics_{timestamp}.csv"), index=False
    )

    calibrated_cols = [
        c
        for c in df.columns
        if c.endswith("_m")
        or c
        in {
            "frame_index",
            "time_s",
            "com_velocity_y",
            "body_mass_kg",
            "barbell_mass_kg",
            "analysis_mass_kg",
            "calibration_factor_m_per_unit",
            "stance_width_ratio",
            "knee_flexion_l",
            "knee_flexion_r",
            "shin_angle_ground",
            "spine_deviation",
            "cervical_deviation",
            "hip_extension_r",
            "pull_synchronism_ratio",
            "kinematics_parameter_file",
            "subject_parameter_file",
        }
    ]
    calibrated_cols = list(dict.fromkeys(calibrated_cols))
    df[calibrated_cols].to_csv(
        os.path.join(output_dir, f"{base_name}_calibrated_{timestamp}.csv"),
        index=False,
        float_format="%.6f",
    )

    if rep_metrics:
        rep_df = pd.DataFrame(rep_metrics)
        rep_df.to_csv(
            os.path.join(output_dir, f"{base_name}_rep_metrics_{timestamp}.csv"), index=False
        )

    print(f"\n[SUCCESS] Processed variant: {variant}")
    print(f"  Reps: {cadence_stats['n_reps']}")
    print(f"  Cadence: {cadence_stats.get('cadence_rpm', 0):.1f} reps/min")
    print(f"  Report: {report_path}")
    return True


def process_deadlift_file(input_file: str, output_dir: str, overrides: dict | None = None) -> bool:
    """Auto-detect data type and route to the appropriate pipeline."""
    df_peek = pd.read_csv(input_file, nrows=5)
    dtype = detect_data_type(df_peek)

    if dtype == "imu":
        print(f"[AUTO] Detected IMU data in {os.path.basename(input_file)}")
        return process_imu_deadlift_data(input_file, output_dir)
    elif dtype == "mediapipe":
        print(f"[AUTO] Detected MediaPipe pose data in {os.path.basename(input_file)}")
        return process_mediapipe_deadlift_data(input_file, output_dir, overrides=overrides)
    else:
        print(f"[WARN] Unknown data format in {os.path.basename(input_file)}. Trying IMU...")
        return process_imu_deadlift_data(input_file, output_dir)


# ---------------------------------------------------------------------------
# GUI and CLI entry points
# ---------------------------------------------------------------------------



def _ask_optional_float(title: str, prompt: str, initial: float | None = None) -> float | None:
    value = simpledialog.askstring(
        title,
        prompt,
        initialvalue=("" if initial is None else str(initial)),
    )
    if value is None or not value.strip():
        return None
    try:
        return _parse_locale_float(value)
    except Exception:
        messagebox.showwarning("Invalid value", f"Ignored invalid numeric value: {value}")
        return None


def _collect_gui_overrides(root: Tk) -> dict:
    overrides: dict = {}
    if messagebox.askyesno("Kinematics parameters", "Select a kinematics_parameter.txt file?"):
        path = filedialog.askopenfilename(
            title="Select kinematics_parameter.txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")],
        )
        if path:
            overrides["kinematics_params"] = path
    if messagebox.askyesno("Subject parameters", "Select a subject_parameters.txt file?"):
        path = filedialog.askopenfilename(
            title="Select subject_parameters.txt",
            filetypes=[("Text/CSV files", "*.txt *.csv"), ("All files", "*.*")],
        )
        if path:
            overrides["subject_params"] = path
    if messagebox.askyesno("Manual overrides", "Enter camera/subject/load values manually?"):
        value = _ask_optional_float("Camera FPS", "FPS / Hz:")
        if value is not None:
            overrides["fps"] = value
        value = _ask_optional_float("Subject mass", "Subject mass (kg):")
        if value is not None:
            overrides["mass_kg"] = value
        value = _ask_optional_float("Shank length", "Shank length (m):")
        if value is not None:
            overrides["shank_length_m"] = value
        value = _ask_optional_float("Bar load", "Deadlift bar + plates mass (kg):")
        if value is not None:
            overrides["barbell_mass_kg"] = value
        value = _ask_optional_float("GIF interval", "GIF viewing interval per keyframe (s):", 1.2)
        if value is not None:
            overrides["gif_duration_s"] = value
    return overrides

def main_gui():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    target_dir = filedialog.askdirectory(
        title="Select directory containing Deadlift CSV files (IMU or MediaPipe)"
    )
    if not target_dir:
        return

    csv_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]
    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in selected directory.")
        return

    overrides = _collect_gui_overrides(root)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_parent = os.path.join(target_dir, f"vaila_deadlift_analysis_{timestamp}")
    os.makedirs(output_parent, exist_ok=True)

    for f in csv_files:
        print(f"Analyzing File: {f}")
        file_base = os.path.splitext(os.path.basename(f))[0]
        per_file_dir = os.path.join(output_parent, file_base)
        os.makedirs(per_file_dir, exist_ok=True)
        process_deadlift_file(f, per_file_dir, overrides=overrides)

    messagebox.showinfo(
        "Analysis Complete",
        f"All files evaluated successfully!\nResults directory:\n{output_parent}",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="vaila_deadlift: Automated Biomechanical Deadlift Tracker (IMU + MediaPipe)"
    )
    parser.add_argument("-i", "--input", type=str, help="Path to input CSV file (IMU or MediaPipe)")
    parser.add_argument(
        "-o", "--output", type=str, help="Destination directory for analytics plots and reports"
    )
    parser.add_argument("--kinematics-params", type=str, help="Path to kinematics_parameter.txt with camera FPS/metadata")
    parser.add_argument("--subject-params", type=str, help="Path to subject_parameters.txt with subject mass, shank length, and load mass")
    parser.add_argument("--fps", type=float, help="Override camera sampling rate in Hz")
    parser.add_argument("--mass-kg", type=float, help="Override subject body mass in kg")
    parser.add_argument("--shank-length-m", type=float, help="Override shank length calibration in meters")
    parser.add_argument("--barbell-mass-kg", type=float, help="Override deadlift bar + plates mass in kg")
    parser.add_argument("--gif-duration-s", type=float, default=None, help="GIF viewing interval per keyframe in seconds (default: 1.2)")
    parser.add_argument(
        "--gui", action="store_true", help="Force graphical file manager interface layout"
    )
    args = parser.parse_args()

    if args.gui or not args.input:
        main_gui()
    else:
        out = args.output or os.path.dirname(os.path.abspath(args.input))
        overrides = {
            "kinematics_params": args.kinematics_params,
            "subject_params": args.subject_params,
            "fps": args.fps,
            "mass_kg": args.mass_kg,
            "shank_length_m": args.shank_length_m,
            "barbell_mass_kg": args.barbell_mass_kg,
            "gif_duration_s": args.gif_duration_s,
        }
        process_deadlift_file(args.input, out, overrides=overrides)
