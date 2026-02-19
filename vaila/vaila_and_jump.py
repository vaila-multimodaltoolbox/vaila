"""
===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 11 January 2026
Version: 0.1.2
Python Version: 3.12.12

Description:
------------
This script processes jump data from multiple .csv files in a specified directory,
performing biomechanical calculations based on either the time of flight or the
jump height. The results are saved in a new output directory with a timestamp
for each processed file.

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to
transform from screen coordinates (where y increases downward) to biomechanical
coordinates (where y increases upward). This allows proper visualization and
analysis of the jumping motion.

Features:
---------
- Supports two calculation modes:
  - Based on time of flight (calculates jump height)
  - Based on measured jump height (uses the height directly)
  - Processes MediaPipe (in vailá) pose estimation data:
    - Automatically inverts y-coordinates for proper biomechanical analysis
    - Converts normalized coordinates to meters using shank length as reference
    - Calculates center of gravity (CG) position for accurate jump height
- Calculates various metrics for each jump:
  - Force
  - Liftoff Force (Thrust)
  - Velocity (takeoff)
  - Potential Energy
  - Kinetic Energy
  - Average Power (if contact time is provided)
  - Relative Power (W/kg)
  - Jump Performance Index (JPI)
  - Total time (if time of flight and contact time are available)
- Processes all .csv files in the selected directory.
- Saves the results in a new output directory named "vaila_verticaljump_<timestamp>".
- Generates visualizations and HTML reports for data analysis.

Dependencies:
-------------
- Python 3.x
- pandas
- numpy
- matplotlib
- tkinter
- math
- datetime

Usage:
------
- Run the script, select the target directory containing .csv files, and specify
  whether the data is based on time of flight, jump height, or MediaPipe data.
- The script will process each .csv file, performing calculations and saving
  the results in a new directory.

Example:
--------
$ python vaila_and_jump.py

Input File Format:
-----------------
The input CSV files should have one of the following formats:

1. Time-of-flight based format:
   mass(kg), time_of_flight(s), contact_time(s)[optional]

   Example:
   ```
   mass_kg,time_of_flight_s,contact_time_s
   75.0,0.45,0.22
   80.2,0.42,0.25
   65.5,0.48,0.20
   ```

2. Jump-height based format:
   mass(kg), height(m), contact_time(s)[optional]

   Example:
   ```
   mass_kg,height_m,contact_time_s
   75.0,0.25,0.22
   80.2,0.22,0.25
   65.5,0.28,0.20
   ```

3. MediaPipe pose estimation format:
   CSV file with MediaPipe pose landmark coordinates
   (frame_index, nose_x, nose_y, nose_z, left_eye_inner_x, etc.)

Coordinate System:
-----------------
For MediaPipe data, coordinates are transformed to use the following system:
- Origin: Bottom-left corner
- X-axis: Increases from left to right
- Y-axis: Increases from bottom to top (inverted from screen coordinates)
- Z-axis: Increases from back to front

Output:
-------
The script generates various output files:
- CSV files with jump metrics
- Full processed data with all original and calculated values
- Calibrated data in meters with proper coordinate orientation
- Visualizations of the jump performance
- HTML report summarizing all analysis

Notes:
------
- Ensure that all necessary libraries are installed.
- For accurate results, the MediaPipe landmark detection should be of good quality.

License:
--------
This script is licensed under the GNU General Public License v3.0.
===============================================================================
"""

import base64
import contextlib
import math
import os
import webbrowser
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from rich import print

try:  # Python 3.11+
    import tomllib as _toml_reader  # type: ignore[attr-defined]
except Exception:  # pragma: no cover
    _toml_reader = None  # type: ignore[assignment]

# -----------------------
# Jump context management
# -----------------------
_JUMP_CONTEXT: dict[str, float] | None = None


def _open_jump_help() -> None:
    try:
        help_html = Path(__file__).parent / "help" / "vaila_and_jump_help.html"
        help_md = Path(__file__).parent / "help" / "vaila_and_jump_help.md"
        if help_html.exists():
            webbrowser.open_new_tab(help_html.as_uri())
        elif help_md.exists():
            webbrowser.open_new_tab(help_md.as_uri())
        else:
            messagebox.showinfo("Help", "Help file not found.")
    except Exception as e:
        try:
            messagebox.showerror("Help", f"Could not open help file: {e}")
        except Exception:
            print(f"Help open error: {e}")


def _load_jump_context_from_toml(
    base_dir: Path | None = None,
) -> dict[str, float] | None:
    """Try to load vaila_and_jump_config.toml (prefer the data directory)."""
    search_paths = []
    if base_dir is not None:
        search_paths.append(Path(base_dir) / "vaila_and_jump_config.toml")
    search_paths.extend(
        [
            Path(__file__).parent / "vaila_and_jump_config.toml",
            Path(__file__).parent / "models" / "vaila_and_jump_config.toml",
        ]
    )
    for p in search_paths:
        if p.exists():
            try:
                if _toml_reader is None:
                    import toml  # type: ignore

                    data = toml.load(str(p))
                else:
                    with open(p, "rb") as f:
                        data = _toml_reader.load(f)  # type: ignore[attr-defined]
                cfg = data.get("jump_context", {})
                mass = float(cfg.get("mass_kg", 0))
                fps = float(cfg.get("fps", 0))
                shank = float(cfg.get("shank_length_m", 0.0))
                if mass > 0 and fps > 0 and shank > 0:
                    return {"mass_kg": mass, "fps": float(fps), "shank_length_m": shank}
            except Exception:
                pass
    return None


def _save_jump_context_template(dest: Path, ctx: dict[str, float]) -> None:
    content = (
        "# ================================================\n"
        "# vaila_and_jump configuration (per-folder)\n"
        "# ================================================\n"
        "# Place this file alongside the CSV files you will analyze.\n"
        "# It will be loaded automatically so you are not prompted every run.\n"
        "#\n"
        "# Fields:\n"
        "# - mass_kg: Subject mass in kilograms (e.g., 75.0)\n"
        "# - fps: Video frame rate (Hz). Use CAPTURE FPS for slow-motion (e.g., 240)\n"
        "# - shank_length_m: Estimated shank length (m), used to scale to meters\n"
        "#\n"
        "# Optional (informative) fields you may add under [notes]:\n"
        "# [notes]\n"
        '# subject_id = "S01"\n'
        '# session = "pre"\n'
        '# comment = "CMJ test set"\n'
        "# ================================================\n"
        "[jump_context]\n"
        f"mass_kg = {ctx.get('mass_kg', 75.0):.3f}\n"
        f"fps = {float(ctx.get('fps', 240)):.3f}\n"
        f"shank_length_m = {ctx.get('shank_length_m', 0.40):.3f}\n"
    )
    dest.write_text(content, encoding="utf-8")


def _get_or_ask_jump_context(
    base_dir: Path | None = None,
) -> dict[str, float] | None:
    global _JUMP_CONTEXT
    if _JUMP_CONTEXT is not None:
        return _JUMP_CONTEXT
    # Try TOML first
    ctx = _load_jump_context_from_toml(base_dir=base_dir)
    if ctx:
        _JUMP_CONTEXT = ctx
        print(
            f"Loaded jump context from TOML: mass={ctx['mass_kg']} kg, fps={ctx['fps']}, shank={ctx['shank_length_m']} m"
        )
        return _JUMP_CONTEXT
    # Ask once via dialogs
    root = Tk()
    root.withdraw()
    with contextlib.suppress(Exception):
        root.attributes("-topmost", True)
    try:
        mass = simpledialog.askfloat(
            "Mass (kg)",
            "Enter subject mass (kg):",
            parent=root,
            minvalue=20.0,
            maxvalue=200.0,
        )
        if mass is None:
            return None
        fps = simpledialog.askfloat(
            "FPS", "Enter video FPS (frames/s):", parent=root, minvalue=1.0, maxvalue=1000.0
        )
        if fps is None:
            fps = 30
        shank = simpledialog.askfloat(
            "Shank length (m)",
            "Enter shank length in meters (e.g., 0.40):",
            parent=root,
            minvalue=0.1,
            maxvalue=1.0,
        )
        if shank is None:
            shank = 0.40
        _JUMP_CONTEXT = {
            "mass_kg": float(mass),
            "fps": float(fps),
            "shank_length_m": float(shank),
        }
        # Offer to save template
        try:
            if messagebox.askyesno(
                "Save Config",
                "Save these values to vaila_and_jump_config.toml in the data folder for batch runs?",
            ):
                dest_dir = Path(base_dir) if base_dir is not None else Path(__file__).parent
                dest = dest_dir / "vaila_and_jump_config.toml"
                _save_jump_context_template(dest, _JUMP_CONTEXT)
                messagebox.showinfo("Saved", f"Template saved at: {dest}")
        except Exception:
            pass
        return _JUMP_CONTEXT
    finally:
        with contextlib.suppress(Exception):
            root.destroy()


def calculate_force(mass, gravity=9.81):
    """
    Calculate the weight force based on the mass and gravity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The weight force in Newtons.
    """
    return mass * gravity


def calculate_jump_height(time_of_flight, gravity=9.81):
    """
    Calculate the jump height from the time of flight.

    Args:
        time_of_flight (float): The time of flight in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The jump height in meters.
    """
    return (gravity * time_of_flight**2) / 8


def calculate_power(force, height, contact_time):
    """
    Calculate the power output during the jump.

    Args:
        force (float): The force exerted during the jump in Newtons.
        height (float): The jump height in meters.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The power output in Watts.
    """
    work = force * height
    return work / contact_time


def calculate_velocity(height, gravity=9.81):
    """
    Calculate the takeoff velocity needed to reach the given height.

    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The takeoff velocity in meters per second.
    """
    return math.sqrt(2 * gravity * height)


def calculate_kinetic_energy(mass, velocity):
    """
    Calculate the kinetic energy based on mass and velocity.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The velocity in meters per second.

    Returns:
        float: The kinetic energy in Joules.
    """
    return 0.5 * mass * velocity**2


def calculate_potential_energy(mass, height, gravity=9.81):
    """
    Calculate the potential energy based on mass and height.

    Args:
        mass (float): The mass of the jumper in kilograms.
        height (float): The height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The potential energy in Joules.
    """
    return mass * gravity * height


def calculate_average_power(potential_energy, contact_time):
    """
    Calculate the average power output during the contact phase.

    Args:
        potential_energy (float): The potential energy in Joules.
        contact_time (float): The contact time in seconds.

    Returns:
        float: The average power in Watts.
    """
    return potential_energy / contact_time


def calculate_liftoff_force(mass, velocity, contact_time, gravity=9.81):
    """
    Calculate the total liftoff force (thrust) required during the contact phase.

    Args:
        mass (float): The mass of the jumper in kilograms.
        velocity (float): The takeoff velocity in m/s.
        contact_time (float): The contact time in seconds.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The liftoff force in Newtons.
    """
    # Calculate the weight force
    weight_force = mass * gravity

    # Calculate the force needed to accelerate to the takeoff velocity
    acceleration_force = (mass * velocity) / contact_time

    # Total liftoff force
    liftoff_force = weight_force + acceleration_force

    return liftoff_force


def calculate_time_of_flight(height, gravity=9.81):
    """
    Calculate the time of flight from the jump height.

    Args:
        height (float): The jump height in meters.
        gravity (float, optional): The gravitational acceleration. Default is 9.81 m/s^2.

    Returns:
        float: The time of flight in seconds.
    """
    return math.sqrt(8 * height / gravity)


def calculate_baseline(data, n_frames=10):
    """
    Calculate the baseline using the first n frames.

    Args:
        data (pd.DataFrame): DataFrame with MediaPipe data
        n_frames (int): Number of frames to calculate the baseline average

    Returns:
        tuple: (baseline_feet, baseline_cg)
    """
    # Calculate the average of the first n frames for the feet
    right_foot_baseline = data["right_foot_index_y"].iloc[:n_frames].mean()
    left_foot_baseline = data["left_foot_index_y"].iloc[:n_frames].mean()
    feet_baseline = (right_foot_baseline + left_foot_baseline) / 2

    # Calculate the average of the first n frames for the CG
    cg_y_baseline = data["cg_y"].iloc[:n_frames].mean()

    return feet_baseline, cg_y_baseline


def identify_jump_phases(data, feet_baseline, _cg_baseline, fps):
    """
    Improved identification of jump phases with three different height calculation methods.

    Methods:
    1. CG Method: From initial CG position to maximum CG height
    2. Flight Time Method: Based on CG flight time (time in air)
    3. Feet Method: Based on feet height (left and right foot_index)

    Args:
        data (pd.DataFrame): DataFrame with the data
        feet_baseline (float): Baseline for the feet (e.g., average foot position during initial frames 10-20)
        cg_baseline (float): Baseline for the CG (e.g., average CG Y position during initial frames 10-20, used for normalization)
        fps (int): Frames per second

    Returns:
        dict: Dictionary with comprehensive jump phase information including all three height methods
    """
    # METHOD 1: CG-based height (from initial position to maximum)
    max_cg_height = data["cg_y_normalized"].max()
    min_cg_height = data["cg_y_normalized"].min()  # For squat depth

    # Find max height frame
    max_height_frame = data["cg_y_normalized"].idxmax()

    # Find squat frame (minimum position) - This is the start of propulsion
    squat_frame = data["cg_y_normalized"].idxmin()

    # CORRECTED: Find takeoff between squat and peak, closest to baseline (0)
    takeoff_frame = squat_frame  # Default if no better found
    if squat_frame <= max_height_frame:
        squat_to_peak_range = data.loc[squat_frame:max_height_frame]
        if not squat_to_peak_range.empty:
            baseline_tolerance = 0.02  # 2cm tolerance
            baseline_candidates = squat_to_peak_range[
                abs(squat_to_peak_range["cg_y_normalized"]) <= baseline_tolerance
            ]

            if len(baseline_candidates) > 0:
                takeoff_frame = baseline_candidates.index[0]
            elif not squat_to_peak_range["cg_y_normalized"].empty:
                closest_to_baseline_idx = (squat_to_peak_range["cg_y_normalized"]).abs().idxmin()
                takeoff_frame = closest_to_baseline_idx

    # Improved landing detection: when CG returns close to baseline
    landing_frame = len(data) - 1 if len(data) > 0 else 0  # Default to last frame
    if max_height_frame < len(data) - 1:
        post_peak_data = data[data.index > max_height_frame]
        if not post_peak_data.empty:
            landing_candidates = post_peak_data[
                post_peak_data["cg_y_normalized"] <= 0.02
            ]  # 2cm tolerance
            if len(landing_candidates) > 0:
                landing_frame = landing_candidates.index[0]
            elif not post_peak_data["cg_y_normalized"].empty:  # Check if series is not empty
                closest_to_baseline_landing_idx = (post_peak_data["cg_y_normalized"]).abs().idxmin()
                landing_frame = closest_to_baseline_landing_idx

    # METHOD 2: Flight time-based height calculation
    flight_time = 0
    if takeoff_frame is not None and landing_frame is not None and landing_frame > takeoff_frame:
        flight_time = (landing_frame - takeoff_frame) / fps
    height_from_flight_time = (9.81 * flight_time**2) / 8 if flight_time > 0 else 0

    # METHOD 3: Feet-based height calculations (USING THE feet_baseline PARAMETER)
    left_foot_height = None
    right_foot_height = None
    avg_feet_height = None

    if "left_foot_index_y_m" in data.columns:
        left_foot_max = data["left_foot_index_y_m"].max()
        left_foot_height = left_foot_max - feet_baseline  # Use passed feet_baseline

    if "right_foot_index_y_m" in data.columns:
        right_foot_max = data["right_foot_index_y_m"].max()
        right_foot_height = right_foot_max - feet_baseline  # Use passed feet_baseline

    if left_foot_height is not None and right_foot_height is not None:
        avg_feet_height = (left_foot_height + right_foot_height) / 2
    elif left_foot_height is not None:
        avg_feet_height = left_foot_height
    elif right_foot_height is not None:
        avg_feet_height = right_foot_height

    # Individual foot takeoff detection (USING THE feet_baseline PARAMETER)
    left_takeoff_idx = None
    right_takeoff_idx = None

    if "left_foot_index_y_m" in data.columns:
        # Using feet_baseline + 0.02m threshold
        left_takeoff_candidates = data[data["left_foot_index_y_m"] > feet_baseline + 0.02]
        if not left_takeoff_candidates.empty:
            left_takeoff_idx = left_takeoff_candidates.index[0]

    if "right_foot_index_y_m" in data.columns:
        # Using feet_baseline + 0.02m threshold
        right_takeoff_candidates = data[data["right_foot_index_y_m"] > feet_baseline + 0.02]
        if not right_takeoff_candidates.empty:
            right_takeoff_idx = right_takeoff_candidates.index[0]

    # Individual foot landing detection (USING THE feet_baseline PARAMETER)
    left_landing_idx = None
    right_landing_idx = None

    post_peak_data_for_feet = (
        data[data.index > max_height_frame] if max_height_frame < len(data) - 1 else pd.DataFrame()
    )

    if "left_foot_index_y_m" in data.columns and not post_peak_data_for_feet.empty:
        # Using feet_baseline + 0.02m threshold
        left_landing_candidates = post_peak_data_for_feet[
            post_peak_data_for_feet["left_foot_index_y_m"] <= feet_baseline + 0.02
        ]
        if not left_landing_candidates.empty:
            left_landing_idx = left_landing_candidates.index[0]

    if "right_foot_index_y_m" in data.columns and not post_peak_data_for_feet.empty:
        # Using feet_baseline + 0.02m threshold
        right_landing_candidates = post_peak_data_for_feet[
            post_peak_data_for_feet["right_foot_index_y_m"] <= feet_baseline + 0.02
        ]
        if not right_landing_candidates.empty:
            right_landing_idx = right_landing_candidates.index[0]

    # CORRECTED Propulsion time calculation
    propulsion_time_value = 0
    if takeoff_frame is not None and squat_frame is not None and takeoff_frame > squat_frame:
        propulsion_time_value = (takeoff_frame - squat_frame) / fps

    ascent_time_value = 0
    if (
        max_height_frame is not None
        and takeoff_frame is not None
        and max_height_frame > takeoff_frame
    ):
        ascent_time_value = (max_height_frame - takeoff_frame) / fps

    descent_time_value = 0
    if (
        landing_frame is not None
        and max_height_frame is not None
        and landing_frame > max_height_frame
    ):
        descent_time_value = (landing_frame - max_height_frame) / fps

    return {
        "propulsion_start_frame": squat_frame,
        "takeoff_frame": takeoff_frame,
        "max_height_frame": max_height_frame,
        "landing_frame": landing_frame,
        "flight_time_s": flight_time,
        "max_height_m": max_cg_height,
        "propulsion_time_s": propulsion_time_value,
        "ascent_time_s": ascent_time_value,
        "descent_time_s": descent_time_value,
        "height_cg_method_m": max_cg_height,
        "squat_depth_m": abs(min_cg_height),
        "height_flight_time_method_m": height_from_flight_time,
        "height_left_foot_m": left_foot_height,
        "height_right_foot_m": right_foot_height,
        "height_avg_feet_m": avg_feet_height,
        "left_takeoff_frame": left_takeoff_idx,
        "right_takeoff_frame": right_takeoff_idx,
        "left_landing_frame": left_landing_idx,
        "right_landing_frame": right_landing_idx,
        "left_takeoff_time_s": (left_takeoff_idx / fps if left_takeoff_idx is not None else None),
        "right_takeoff_time_s": (
            right_takeoff_idx / fps if right_takeoff_idx is not None else None
        ),
        "left_landing_time_s": (left_landing_idx / fps if left_landing_idx is not None else None),
        "right_landing_time_s": (
            right_landing_idx / fps if right_landing_idx is not None else None
        ),
    }


def calculate_kinematics(data, results):
    """
    Calculate advanced kinematic metrics for injury screening (Valgus, Stability).

    Based on scientific literature (DOI: 10.1016/j.heliyon.2024 - Figure 4) for clinical validity
    of pre-season assessment.

    Metrics:
    1. Knee-Hip Separation Ratio (Valgus Ratio): dist_knees / dist_hips
    2. FPPA (Frontal Plane Projection Angle): Rigorous 2D vector-based calculation
       - Vector v1: HIP -> KNEE (Femur)
       - Vector v2: KNEE -> ANKLE (Tibia)
       - Angle calculated using atan2 for precise vector angle computation
    3. Landing Stability (Sway): Horizontal CG range in 0.4s post-landing window

    Args:
        data (pd.DataFrame): DataFrame with landmark data.
                             Tries to use '_m' columns if available, else normalized.
        results (dict): Dictionary containing jump phase frames.

    Returns:
        dict: Dictionary with new kinematic metrics added.
    """
    kinematics = {}
    fps = results.get("fps", 30)

    # Helper for Euclidean distance
    def dist_2d(p1_x, p1_y, p2_x, p2_y):
        return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

    def get_coords(row, prefix):
        # Prefer meters if available, else normalized
        if f"{prefix}_x_m" in row and f"{prefix}_y_m" in row:
            return row[f"{prefix}_x_m"], row[f"{prefix}_y_m"]
        return row.get(f"{prefix}_x", 0), row.get(f"{prefix}_y", 0)

    def calculate_fppa_vector_2d(hip_x, hip_y, knee_x, knee_y, ankle_x, ankle_y):
        """
        Calculate FPPA (Frontal Plane Projection Angle) using rigorous 2D vector method.

        Based on scientific literature: DOI 10.1016/j.heliyon.2024 - Figure 4

        FPPA is calculated as the internal angle at the knee joint between:
        - Vector v1: HIP -> KNEE (Femur vector)
        - Vector v2: KNEE -> ANKLE (Tibia vector)

        The FPPA represents the deviation from straight alignment (180°).

        Args:
            hip_x, hip_y: Hip coordinates
            knee_x, knee_y: Knee coordinates
            ankle_x, ankle_y: Ankle coordinates

        Returns:
            float: FPPA angle in degrees
            - 0° = straight alignment (180° internal angle)
            - Positive = Varus (Abduction - knee collapses laterally/outward away from midline)
            - Negative = Valgus (Adduction - knee collapses medially/inward toward midline)
        """
        # Vector v1: HIP -> KNEE (Femur vector, pointing from hip to knee)
        v1_x = knee_x - hip_x
        v1_y = knee_y - hip_y

        # Vector v2: KNEE -> ANKLE (Tibia vector, pointing from knee to ankle)
        v2_x = ankle_x - knee_x
        v2_y = ankle_y - knee_y

        # Check for zero vectors
        mag_v1 = math.sqrt(v1_x**2 + v1_y**2)
        mag_v2 = math.sqrt(v2_x**2 + v2_y**2)

        if mag_v1 < 1e-6 or mag_v2 < 1e-6:
            return 0.0

        # Calculate vectors pointing away from knee (for internal angle calculation)
        # Vector from knee to hip (reversed v1)
        ba_x = hip_x - knee_x
        ba_y = hip_y - knee_y

        # Vector from knee to ankle (v2)
        bc_x = ankle_x - knee_x
        bc_y = ankle_y - knee_y

        # Calculate internal angle using dot product method
        # cos(θ) = (BA · BC) / (|BA| * |BC|)
        mag_ba = math.sqrt(ba_x**2 + ba_y**2)
        mag_bc = math.sqrt(bc_x**2 + bc_y**2)

        if mag_ba < 1e-6 or mag_bc < 1e-6:
            return 0.0

        dot_product = ba_x * bc_x + ba_y * bc_y
        cos_angle = dot_product / (mag_ba * mag_bc)

        # Clamp to [-1, 1] to avoid numerical errors
        cos_angle = max(-1.0, min(1.0, cos_angle))

        # Calculate internal angle at the knee (in degrees)
        internal_angle_deg = math.degrees(math.acos(cos_angle))

        # Calculate cross product to determine valgus/varus direction
        # Cross product = ba_x * bc_y - ba_y * bc_x
        cross_product = ba_x * bc_y - ba_y * bc_x

        # FPPA is the deviation from 180° (straight alignment)
        # Internal angle of 180° = straight leg = 0° FPPA
        # Internal angle < 180° = valgus collapse (positive FPPA)
        # Internal angle > 180° = varus collapse (negative FPPA)
        deviation = 180.0 - internal_angle_deg

        # Use cross product sign to determine valgus/varus
        # In biomechanical coordinates (Y up, X right):
        # - For left leg: positive cross product → valgus (knee moves right/medial)
        # - For right leg: negative cross product → valgus (knee moves left/medial)
        # We use absolute deviation and apply sign based on cross product
        if abs(deviation) < 0.1:  # Essentially straight
            fppa_angle = 0.0
        else:
            # Apply sign: positive = varus (abduction/lateral), negative = valgus (adduction/medial)
            if cross_product > 0:
                fppa_angle = abs(deviation)  # Varus (positive) - Abduction/Lateral collapse
            else:
                fppa_angle = -abs(deviation)  # Valgus (negative) - Adduction/Medial collapse

        return fppa_angle

    propulsion_frame = results.get("propulsion_start_frame")
    landing_frame = results.get("landing_frame")

    # Time series for Valgus Ratio and FPPA (For plotting)
    valgus_ratio_series = []
    fppa_left_series = []
    fppa_right_series = []
    knee_sep_series = []
    ankle_sep_series = []

    # Calculate for all frames (for plotting)
    if "left_knee_x" in data.columns:  # Check if we have data
        for i in range(len(data)):
            row = data.iloc[i]
            lh_x, lh_y = get_coords(row, "left_hip")
            rh_x, rh_y = get_coords(row, "right_hip")
            lk_x, lk_y = get_coords(row, "left_knee")
            rk_x, rk_y = get_coords(row, "right_knee")
            la_x, la_y = get_coords(row, "left_ankle")
            ra_x, ra_y = get_coords(row, "right_ankle")

            hip_sep = dist_2d(lh_x, lh_y, rh_x, rh_y)
            knee_sep = dist_2d(lk_x, lk_y, rk_x, rk_y)
            ankle_sep = dist_2d(la_x, la_y, ra_x, ra_y)

            ratio = knee_sep / hip_sep if hip_sep > 0 else np.nan
            valgus_ratio_series.append(ratio)
            knee_sep_series.append(knee_sep)
            ankle_sep_series.append(ankle_sep)

            # Calculate FPPA for time series
            # For right side, we invert the sign to match left side convention
            # Both sides: positive = varus (abduction), negative = valgus (adduction)
            fppa_left = calculate_fppa_vector_2d(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
            fppa_right_raw = calculate_fppa_vector_2d(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)
            # Invert right side to match left side convention (valgus = positive)
            # This ensures both sides show valgus as positive values
            fppa_right = (
                -fppa_right_raw
                if fppa_right_raw is not None and not math.isnan(fppa_right_raw)
                else np.nan
            )
            fppa_left_series.append(
                fppa_left if fppa_left is not None and not math.isnan(fppa_left) else np.nan
            )
            fppa_right_series.append(fppa_right)

    kinematics["valgus_ratio_series"] = valgus_ratio_series
    kinematics["fppa_left_series"] = fppa_left_series
    kinematics["fppa_right_series"] = fppa_right_series
    kinematics["knee_sep_series"] = knee_sep_series
    kinematics["ankle_sep_series"] = ankle_sep_series

    # 1. Valgus Ratio & 2. FPPA & Knee Angle
    # Calculate for Initial Contact (landing) and Max Flexion (squat/propulsion)
    # Calculate for Initial Contact (landing) and Max Flexion (squat/propulsion)
    for phase_name, target_frame_idx in [("squat", propulsion_frame), ("landing", landing_frame)]:
        if target_frame_idx is None:
            continue

        # Search for valid frame (with all keypoints) nearby if the specific frame is bad
        # Search window of +/- 20 frames (approx 0.6s at 30fps) to find valid data
        frame_idx = None
        search_window = 20

        # Check target frame first
        frames_to_check = [target_frame_idx]
        # Add neighbors (alternating +1, -1, +2, -2...)
        for offset in range(1, search_window + 1):
            if target_frame_idx + offset < len(data):
                frames_to_check.append(target_frame_idx + offset)
            if target_frame_idx - offset >= 0:
                frames_to_check.append(target_frame_idx - offset)

        for f_idx in frames_to_check:
            if f_idx >= len(data):
                continue

            row = data.iloc[f_idx]

            # Check if all needed points are present
            valid_points = True
            needed_points = [
                "left_hip",
                "right_hip",
                "left_knee",
                "right_knee",
                "left_ankle",
                "right_ankle",
            ]
            for pt in needed_points:
                px, py = get_coords(row, pt)
                if px == 0 and py == 0:  # Assuming 0,0 is missing data return from get_coords
                    # Check if it was really missing in source (using normalized check as proxy if needed,
                    # but get_coords returns 0,0 for missing. Ideally we should check strict None but get_coords handles it using .get(...,0))
                    # However, get_coords returns (0,0) if keys are missing.
                    # Better to check if the columns exist and are not NaN?
                    # heuristic: if hip/knee/ankle is exactly 0,0 it's likely invalid/missing
                    if px == 0 and py == 0:
                        valid_points = False
                        break

            if valid_points:
                frame_idx = f_idx
                break

        if frame_idx is None:
            # If still no valid frame found, skip
            continue

        row = data.iloc[frame_idx]

        lh_x, lh_y = get_coords(row, "left_hip")
        rh_x, rh_y = get_coords(row, "right_hip")
        lk_x, lk_y = get_coords(row, "left_knee")
        rk_x, rk_y = get_coords(row, "right_knee")
        la_x, la_y = get_coords(row, "left_ankle")
        ra_x, ra_y = get_coords(row, "right_ankle")

        # Valgus Ratio
        hip_sep = dist_2d(lh_x, lh_y, rh_x, rh_y)
        knee_sep = dist_2d(lk_x, lk_y, rk_x, rk_y)

        if hip_sep > 0:
            kinematics[f"valgus_ratio_{phase_name}"] = knee_sep / hip_sep
            kinematics[f"hip_sep_{phase_name}_m"] = hip_sep
            kinematics[f"knee_sep_{phase_name}_m"] = knee_sep
        else:
            kinematics[f"valgus_ratio_{phase_name}"] = None

        # FPPA (Frontal Plane Projection Angle) - Rigorous 2D vector calculation
        # IMPORTANT: For right side, we invert the sign to match left side convention
        # This ensures both sides show varus (lateral collapse) as positive values
        # and valgus (medial collapse) as negative values, for consistent visualization
        fppa_left = calculate_fppa_vector_2d(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
        fppa_right_raw = calculate_fppa_vector_2d(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)
        fppa_right = -fppa_right_raw if fppa_right_raw is not None else None

        kinematics[f"fppa_left_{phase_name}_deg"] = fppa_left
        kinematics[f"fppa_right_{phase_name}_deg"] = fppa_right

        # Knee Relative Angle (Hip-Knee-Ankle) - same as FPPA for consistency
        kinematics[f"knee_angle_left_{phase_name}_deg"] = fppa_left
        kinematics[f"knee_angle_right_{phase_name}_deg"] = fppa_right

        # KASR (Knee-to-Ankle Separation Ratio) & KSD
        ankle_sep = dist_2d(la_x, la_y, ra_x, ra_y)

        if ankle_sep > 0:
            kinematics[f"kasr_{phase_name}"] = knee_sep / ankle_sep
        else:
            kinematics[f"kasr_{phase_name}"] = None

        kinematics[f"ksd_{phase_name}_m"] = knee_sep
        kinematics[f"ankle_sep_{phase_name}_m"] = ankle_sep

    # 3. Landing Phase Analysis (IC, +40ms, +100ms, Max Valgus)
    # Based on Mechanisms for Noncontact Anterior Cruciate Ligament Injury
    if landing_frame is not None and landing_frame < len(data):
        # Calculate FPPA at specific time points after landing
        # IC + 40ms
        frame_40ms = landing_frame + int(0.040 * fps)
        if frame_40ms < len(data):
            row_40ms = data.iloc[frame_40ms]
            lh_x, lh_y = get_coords(row_40ms, "left_hip")
            rh_x, rh_y = get_coords(row_40ms, "right_hip")
            lk_x, lk_y = get_coords(row_40ms, "left_knee")
            rk_x, rk_y = get_coords(row_40ms, "right_knee")
            la_x, la_y = get_coords(row_40ms, "left_ankle")
            ra_x, ra_y = get_coords(row_40ms, "right_ankle")

            fppa_left_40ms = calculate_fppa_vector_2d(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
            fppa_right_40ms_raw = calculate_fppa_vector_2d(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)
            # Invert right side to match left side convention (valgus = positive)
            fppa_right_40ms = -fppa_right_40ms_raw if fppa_right_40ms_raw is not None else None
            kinematics["fppa_left_landing_40ms_deg"] = fppa_left_40ms
            kinematics["fppa_right_landing_40ms_deg"] = fppa_right_40ms
            kinematics["landing_40ms_frame"] = frame_40ms

        # IC + 100ms
        frame_100ms = landing_frame + int(0.100 * fps)
        if frame_100ms < len(data):
            row_100ms = data.iloc[frame_100ms]
            lh_x, lh_y = get_coords(row_100ms, "left_hip")
            rh_x, rh_y = get_coords(row_100ms, "right_hip")
            lk_x, lk_y = get_coords(row_100ms, "left_knee")
            rk_x, rk_y = get_coords(row_100ms, "right_knee")
            la_x, la_y = get_coords(row_100ms, "left_ankle")
            ra_x, ra_y = get_coords(row_100ms, "right_ankle")

            fppa_left_100ms = calculate_fppa_vector_2d(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
            fppa_right_100ms_raw = calculate_fppa_vector_2d(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)
            # Invert right side to match left side convention (valgus = positive)
            fppa_right_100ms = -fppa_right_100ms_raw if fppa_right_100ms_raw is not None else None
            kinematics["fppa_left_landing_100ms_deg"] = fppa_left_100ms
            kinematics["fppa_right_landing_100ms_deg"] = fppa_right_100ms
            kinematics["landing_100ms_frame"] = frame_100ms

        # Max Valgus Calculation
        # Find maximum valgus angle between landing_frame and landing_frame + 0.2s
        window_frames = int(0.2 * fps)  # 0.2 seconds window
        end_window = min(len(data), landing_frame + window_frames)

        max_valgus_left = None
        max_valgus_right = None
        max_valgus_frame_left = None
        max_valgus_frame_right = None

        if end_window > landing_frame:
            for frame_idx in range(landing_frame, end_window):
                if frame_idx >= len(data):
                    break

                row = data.iloc[frame_idx]

                lh_x, lh_y = get_coords(row, "left_hip")
                rh_x, rh_y = get_coords(row, "right_hip")
                lk_x, lk_y = get_coords(row, "left_knee")
                rk_x, rk_y = get_coords(row, "right_knee")
                la_x, la_y = get_coords(row, "left_ankle")
                ra_x, ra_y = get_coords(row, "right_ankle")

                # Calculate FPPA for this frame
                fppa_left = calculate_fppa_vector_2d(lh_x, lh_y, lk_x, lk_y, la_x, la_y)
                fppa_right_raw = calculate_fppa_vector_2d(rh_x, rh_y, rk_x, rk_y, ra_x, ra_y)
                # Invert right side to match left side convention (valgus = positive)
                fppa_right = -fppa_right_raw if fppa_right_raw is not None else None

                # Track maximum valgus (positive values indicate valgus)
                if max_valgus_left is None or (
                    fppa_left is not None and fppa_left > max_valgus_left
                ):
                    max_valgus_left = fppa_left
                    max_valgus_frame_left = frame_idx

                if max_valgus_right is None or (
                    fppa_right is not None and fppa_right > max_valgus_right
                ):
                    max_valgus_right = fppa_right
                    max_valgus_frame_right = frame_idx

        kinematics["max_valgus_angle_left_deg"] = max_valgus_left
        kinematics["max_valgus_angle_right_deg"] = max_valgus_right
        kinematics["max_valgus_frame_left"] = max_valgus_frame_left
        kinematics["max_valgus_frame_right"] = max_valgus_frame_right

    # 4. Landing Stability
    if landing_frame is not None and landing_frame < len(data):
        window_frames = int(0.4 * fps)
        end_window = min(len(data), landing_frame + window_frames)

        if end_window > landing_frame:
            # Prefer meters
            col_to_use = "cg_x_m" if "cg_x_m" in data.columns else "cg_x"
            # Note: cg_x usually exists if cg_y exists
            if col_to_use in data.columns:
                sway_segment = data[col_to_use].iloc[landing_frame:end_window]
                sway = sway_segment.max() - sway_segment.min()
                kinematics["landing_stability_sway_x"] = sway
                kinematics["landing_stability_unit"] = "m" if "_m" in col_to_use else "norm"

    # Validation: Print calculated angles for debugging
    print("\n=== FPPA Validation (Vector-based calculation) ===")
    print(f"Squat (Propulsion Start) - Left FPPA: {kinematics.get('fppa_left_squat_deg', 'N/A')}")
    if isinstance(kinematics.get("fppa_left_squat_deg"), (int, float)):
        print(f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_left_squat_deg'))[0]}")
    print(f"Squat (Propulsion Start) - Right FPPA: {kinematics.get('fppa_right_squat_deg', 'N/A')}")
    if isinstance(kinematics.get("fppa_right_squat_deg"), (int, float)):
        print(f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_right_squat_deg'))[0]}")
    print(
        f"Initial Contact (Landing) - Left FPPA: {kinematics.get('fppa_left_landing_deg', 'N/A')}"
    )
    if isinstance(kinematics.get("fppa_left_landing_deg"), (int, float)):
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_left_landing_deg'))[0]}"
        )
    print(
        f"Initial Contact (Landing) - Right FPPA: {kinematics.get('fppa_right_landing_deg', 'N/A')}"
    )
    if isinstance(kinematics.get("fppa_right_landing_deg"), (int, float)):
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_right_landing_deg'))[0]}"
        )
    if kinematics.get("fppa_left_landing_40ms_deg") is not None:
        print(
            f"IC + 40ms - Left FPPA: {kinematics.get('fppa_left_landing_40ms_deg'):.2f}° (Frame {kinematics.get('landing_40ms_frame')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_left_landing_40ms_deg'))[0]}"
        )
    if kinematics.get("fppa_right_landing_40ms_deg") is not None:
        print(
            f"IC + 40ms - Right FPPA: {kinematics.get('fppa_right_landing_40ms_deg'):.2f}° (Frame {kinematics.get('landing_40ms_frame')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_right_landing_40ms_deg'))[0]}"
        )
    if kinematics.get("fppa_left_landing_100ms_deg") is not None:
        print(
            f"IC + 100ms - Left FPPA: {kinematics.get('fppa_left_landing_100ms_deg'):.2f}° (Frame {kinematics.get('landing_100ms_frame')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_left_landing_100ms_deg'))[0]}"
        )
    if kinematics.get("fppa_right_landing_100ms_deg") is not None:
        print(
            f"IC + 100ms - Right FPPA: {kinematics.get('fppa_right_landing_100ms_deg'):.2f}° (Frame {kinematics.get('landing_100ms_frame')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('fppa_right_landing_100ms_deg'))[0]}"
        )
    if kinematics.get("max_valgus_angle_left_deg") is not None:
        print(
            f"Max Valgus (0.2s post-landing) - Left: {kinematics.get('max_valgus_angle_left_deg'):.2f}° (Frame {kinematics.get('max_valgus_frame_left')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('max_valgus_angle_left_deg'))[0]}"
        )
    if kinematics.get("max_valgus_angle_right_deg") is not None:
        print(
            f"Max Valgus (0.2s post-landing) - Right: {kinematics.get('max_valgus_angle_right_deg'):.2f}° (Frame {kinematics.get('max_valgus_frame_right')})"
        )
        print(
            f"  Risk: {_get_fppa_risk_classification(kinematics.get('max_valgus_angle_right_deg'))[0]}"
        )
    print("=" * 60)

    return kinematics


def plot_valgus_ratio(data, results, output_dir, base_name):
    """
    Generate a plot of the Knee-Hip Separation Ratio over time.
    """
    if "valgus_ratio_series" not in results or not results["valgus_ratio_series"]:
        return None

    import matplotlib.pyplot as plt

    valgus_series = results["valgus_ratio_series"]
    fps = results.get("fps", 30)
    time_axis = [i / fps for i in range(len(valgus_series))]

    plt.figure(figsize=(10, 6))
    plt.plot(
        time_axis, valgus_series, label="Knee-Hip Separation Ratio", color="purple", linewidth=2
    )

    # Threshold line
    plt.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Neutral (1.0)")
    plt.axhline(y=0.8, color="red", linestyle="--", alpha=0.5, label="High Risk (< 0.8)")

    # Mark events
    propulsion_frame = results.get("propulsion_start_frame")
    landing_frame = results.get("landing_frame")

    if propulsion_frame is not None and propulsion_frame < len(time_axis):
        plt.axvline(
            x=propulsion_frame / fps, color="orange", linestyle=":", label="Squat (Deepest)"
        )
        val = valgus_series[propulsion_frame]
        if not math.isnan(val) and val is not None:
            plt.scatter(propulsion_frame / fps, val, color="orange", zorder=5)

    if landing_frame is not None and landing_frame < len(time_axis):
        plt.axvline(x=landing_frame / fps, color="green", linestyle=":", label="Landing (Initial)")
        val = valgus_series[landing_frame]
        if not math.isnan(val) and val is not None:
            plt.scatter(landing_frame / fps, val, color="green", zorder=5)

    plt.title("Knee-Hip Separation Ratio (Valgus Screening)")
    plt.xlabel("Time (s)")
    plt.ylabel("Ratio (Knee Dist / Hip Dist)")
    plt.legend()
    plt.grid(True, alpha=0.3)

    filename = f"{base_name}_valgus_ratio.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100)
    plt.close()
    return filepath


def plot_fppa_time_series(data, results, output_dir, base_name):
    """
    Generate a time series plot of FPPA (Frontal Plane Projection Angle) showing
    adduction/abduction (valgus) angles over time, similar to Figure 3 from scientific literature.

    Shows left and right FPPA with key events highlighted.
    """
    if "fppa_left_series" not in results or not results["fppa_left_series"]:
        return None

    import matplotlib.pyplot as plt

    fppa_left_series = results["fppa_left_series"]
    fppa_right_series = results.get("fppa_right_series", [])
    fps = results.get("fps", 30)

    # Convert to time in milliseconds relative to landing
    landing_frame = results.get("landing_frame")
    if landing_frame is None:
        return None

    # Create time axis in milliseconds relative to landing
    time_ms = [(i - landing_frame) * (1000.0 / fps) for i in range(len(fppa_left_series))]

    fig, ax = plt.subplots(figsize=(14, 8))

    # Plot FPPA time series
    # Convert to numpy arrays and handle NaN values
    fppa_left_array = np.array(fppa_left_series)
    fppa_right_array = np.array(fppa_right_series) if fppa_right_series else np.array([])

    ax.plot(
        time_ms,
        fppa_left_array,
        label="Left FPPA (Adduction/Abduction)",
        color="blue",
        linewidth=2,
        alpha=0.8,
    )
    if len(fppa_right_array) > 0:
        ax.plot(
            time_ms,
            fppa_right_array,
            label="Right FPPA (Adduction/Abduction)",
            color="red",
            linewidth=2,
            alpha=0.8,
        )

    # Reference line at 0° (straight alignment)
    ax.axhline(y=0, color="gray", linestyle="--", linewidth=1, alpha=0.5, label="Straight (0°)")

    # Risk threshold lines
    ax.axhline(
        y=5, color="orange", linestyle=":", linewidth=1, alpha=0.5, label="Moderate Risk (5°)"
    )
    ax.axhline(y=10, color="red", linestyle=":", linewidth=1, alpha=0.5, label="High Risk (10°)")
    ax.axhline(y=-5, color="orange", linestyle=":", linewidth=1, alpha=0.5)
    ax.axhline(y=-10, color="red", linestyle=":", linewidth=1, alpha=0.5)

    # Mark key events
    propulsion_frame = results.get("propulsion_start_frame")
    takeoff_frame = results.get("takeoff_frame")
    landing_40ms_frame = results.get("landing_40ms_frame")
    landing_100ms_frame = results.get("landing_100ms_frame")
    max_valgus_frame_left = results.get("max_valgus_frame_left")
    max_valgus_frame_right = results.get("max_valgus_frame_right")

    # Landing (IC) at 0ms
    ax.axvline(x=0, color="green", linestyle="-", linewidth=2, alpha=0.7, label="IC (Landing)")
    if landing_frame < len(fppa_left_array):
        fppa_l_ic = fppa_left_array[landing_frame]
        if not np.isnan(fppa_l_ic):
            ax.scatter(0, fppa_l_ic, color="green", s=100, zorder=5, marker="o")
        if len(fppa_right_array) > 0 and landing_frame < len(fppa_right_array):
            fppa_r_ic = fppa_right_array[landing_frame]
            if not np.isnan(fppa_r_ic):
                ax.scatter(0, fppa_r_ic, color="green", s=100, zorder=5, marker="s")

    # Squat (Propulsion Start)
    if propulsion_frame is not None:
        time_squat_ms = (propulsion_frame - landing_frame) * (1000.0 / fps)
        ax.axvline(
            x=time_squat_ms,
            color="orange",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Squat (Propulsion Start)",
        )
        if propulsion_frame < len(fppa_left_array):
            fppa_l_squat = fppa_left_array[propulsion_frame]
            if not np.isnan(fppa_l_squat):
                ax.scatter(time_squat_ms, fppa_l_squat, color="orange", s=100, zorder=5, marker="o")

    # Takeoff
    if takeoff_frame is not None:
        time_takeoff_ms = (takeoff_frame - landing_frame) * (1000.0 / fps)
        ax.axvline(
            x=time_takeoff_ms,
            color="purple",
            linestyle="--",
            linewidth=2,
            alpha=0.7,
            label="Takeoff",
        )
        if takeoff_frame < len(fppa_left_array):
            fppa_l_to = fppa_left_array[takeoff_frame]
            if not np.isnan(fppa_l_to):
                ax.scatter(time_takeoff_ms, fppa_l_to, color="purple", s=100, zorder=5, marker="o")

    # IC + 40ms
    if landing_40ms_frame is not None:
        time_40ms = (landing_40ms_frame - landing_frame) * (1000.0 / fps)
        ax.axvline(
            x=time_40ms, color="cyan", linestyle=":", linewidth=2, alpha=0.7, label="IC + 40ms"
        )
        if landing_40ms_frame < len(fppa_left_array):
            fppa_l_40 = fppa_left_array[landing_40ms_frame]
            if not np.isnan(fppa_l_40):
                ax.scatter(time_40ms, fppa_l_40, color="cyan", s=100, zorder=5, marker="o")
        if len(fppa_right_array) > 0 and landing_40ms_frame < len(fppa_right_array):
            fppa_r_40 = fppa_right_array[landing_40ms_frame]
            if not np.isnan(fppa_r_40):
                ax.scatter(time_40ms, fppa_r_40, color="cyan", s=100, zorder=5, marker="s")

    # IC + 100ms
    if landing_100ms_frame is not None:
        time_100ms = (landing_100ms_frame - landing_frame) * (1000.0 / fps)
        ax.axvline(
            x=time_100ms, color="magenta", linestyle=":", linewidth=2, alpha=0.7, label="IC + 100ms"
        )
        if landing_100ms_frame < len(fppa_left_array):
            fppa_l_100 = fppa_left_array[landing_100ms_frame]
            if not np.isnan(fppa_l_100):
                ax.scatter(time_100ms, fppa_l_100, color="magenta", s=100, zorder=5, marker="o")
        if len(fppa_right_array) > 0 and landing_100ms_frame < len(fppa_right_array):
            fppa_r_100 = fppa_right_array[landing_100ms_frame]
            if not np.isnan(fppa_r_100):
                ax.scatter(time_100ms, fppa_r_100, color="magenta", s=100, zorder=5, marker="s")

    # Max Valgus
    if max_valgus_frame_left is not None:
        time_max_valgus_ms = (max_valgus_frame_left - landing_frame) * (1000.0 / fps)
        if max_valgus_frame_left < len(fppa_left_array):
            fppa_l_max = fppa_left_array[max_valgus_frame_left]
            if not np.isnan(fppa_l_max):
                ax.scatter(
                    time_max_valgus_ms,
                    fppa_l_max,
                    color="red",
                    s=150,
                    zorder=6,
                    marker="*",
                    label="Max Valgus (Left)",
                    edgecolors="black",
                    linewidths=1,
                )

    if max_valgus_frame_right is not None and len(fppa_right_array) > 0:
        time_max_valgus_ms = (max_valgus_frame_right - landing_frame) * (1000.0 / fps)
        if max_valgus_frame_right < len(fppa_right_array):
            fppa_r_max = fppa_right_array[max_valgus_frame_right]
            if not np.isnan(fppa_r_max):
                ax.scatter(
                    time_max_valgus_ms,
                    fppa_r_max,
                    color="red",
                    s=150,
                    zorder=6,
                    marker="*",
                    label="Max Valgus (Right)",
                    edgecolors="black",
                    linewidths=1,
                )

    # Labels and formatting
    ax.set_xlabel("Time (ms) relative to Initial Contact (IC)", fontsize=12)
    ax.set_ylabel(
        "FPPA (degrees)\nPositive = Varus (Abduction/Lateral), Negative = Valgus (Adduction/Medial)",
        fontsize=12,
    )
    ax.set_title(
        "Time Sequence of Knee Adduction/Abduction Angles (FPPA)", fontsize=14, fontweight="bold"
    )
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=10, ncol=2)

    # Set x-axis to show key time points
    ax.set_xlim(min(time_ms), max(time_ms))

    filename = f"{base_name}_fppa_time_series.png"
    filepath = os.path.join(output_dir, filename)
    plt.savefig(filepath, dpi=100, bbox_inches="tight")
    plt.close()
    return filepath


def plot_valgus_event(data, results, output_dir, base_name):
    """
    Generate annotated stick figure plots for Squat and Landing events.
    Displays alignment and calculated metrics (Ratio, KASR, FPPA).
    Includes full body stick figure with head/neck and proper aspect ratio.
    """
    # #region agent log
    import json

    import matplotlib.pyplot as plt

    log_path = "/home/preto/Desktop/Preto/vaila/.cursor/debug.log"

    def log_debug(location, message, data_dict, hypothesis_id):
        try:
            with open(log_path, "a", encoding="utf-8") as f:
                f.write(
                    json.dumps(
                        {
                            "sessionId": "debug-session",
                            "runId": "run1",
                            "hypothesisId": hypothesis_id,
                            "location": location,
                            "message": message,
                            "data": data_dict,
                            "timestamp": datetime.now().timestamp() * 1000,
                        }
                    )
                    + "\n"
                )
        except Exception:
            pass

    # #endregion

    plot_paths = []

    propulsion_frame = results.get("propulsion_start_frame")
    landing_frame = results.get("landing_frame")
    fps = results.get("fps", 30)

    events = []
    if propulsion_frame is not None:
        events.append(("Squat", propulsion_frame, "squat"))
    # Landing will be handled separately with multiple moments
    # if landing_frame is not None:
    #     events.append(("Landing", landing_frame, "landing"))

    body_segments = [
        # Legs
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        # Feet
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
        # Pelvis & Trunk
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "left_shoulder"),
        ("right_hip", "right_shoulder"),
        # Arms
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    def get_pt(row, name):
        # Tries to get meter coords, then normalized
        if f"{name}_x_m" in row:
            return row[f"{name}_x_m"], row[f"{name}_y_m"]
        return row.get(f"{name}_x"), row.get(f"{name}_y")

    def dist_2d(p1_x, p1_y, p2_x, p2_y):
        """Calculate Euclidean distance between two 2D points."""
        if p1_x is None or p1_y is None or p2_x is None or p2_y is None:
            return None
        return math.sqrt((p1_x - p2_x) ** 2 + (p1_y - p2_y) ** 2)

    # #region agent log
    log_debug(
        f"vaila_and_jump.py:{903}",
        "plot_valgus_event started",
        {"num_events": len(events), "events": [e[0] for e in events]},
        "START",
    )
    # #endregion

    for title, frame_idx, phase_key in events:
        if frame_idx >= len(data):
            continue

        row = data.iloc[frame_idx]

        # #region agent log
        log_debug(
            f"vaila_and_jump.py:{904}",
            "Processing event",
            {"title": title, "frame_idx": int(frame_idx), "phase_key": phase_key},
            "START",
        )
        # #endregion

        # Check validity of key lower body points first
        check_points = [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
        ]
        valid_points = True
        for pt in check_points:
            p = get_pt(row, pt)
            if p[0] is None:
                valid_points = False
                break
        if not valid_points:
            continue

        # Calculate data range first to determine appropriate figsize
        temp_x = []
        temp_y = []
        for part_name in [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_shoulder",
            "right_shoulder",
            "nose",
            "left_foot_index",
            "right_foot_index",
        ]:
            pt = get_pt(row, part_name)
            if pt[0] is not None:
                temp_x.append(pt[0])
                temp_y.append(pt[1])

        # For equal aspect ratio plots, use a square or near-square figsize
        # When set_aspect('equal') is applied, we want 1:1 data unit scaling
        # Using a square figsize ensures the plot box can be adjusted correctly
        # The actual data will be scaled to fit within the limits we set
        figsize = (8, 8)  # Use square figure for equal aspect ratio

        fig, ax = plt.subplots(figsize=figsize)

        # #region agent log
        # Calculate values for logging
        log_data_x_range = 0
        log_data_y_range = 0
        log_data_aspect = 0
        if temp_x and temp_y:
            log_data_x_range = float(max(temp_x) - min(temp_x)) if temp_x else 0
            log_data_y_range = float(max(temp_y) - min(temp_y)) if temp_y else 0
            if log_data_x_range > 0 and log_data_y_range > 0:
                x_pad = max(log_data_x_range * 0.2, 0.1)
                y_pad = max(log_data_y_range * 0.2, 0.1)
                total_x = log_data_x_range + 2 * x_pad
                total_y = log_data_y_range + 2 * y_pad
                log_data_aspect = float(total_y / total_x) if total_x > 0 else 0
        log_debug(
            f"vaila_and_jump.py:{974}",
            "Figure created",
            {
                "phase": phase_key,
                "figsize": list(figsize),
                "data_x_range": log_data_x_range,
                "data_y_range": log_data_y_range,
                "calculated_aspect": log_data_aspect,
            },
            "FIG",
        )
        # #endregion

        # FIRST: Collect ALL points that will be plotted (including separation lines)
        # This ensures we calculate limits correctly to include all visible elements
        all_x = []
        all_y = []

        # Collect body segment points
        for part_name in [
            "left_hip",
            "right_hip",
            "left_knee",
            "right_knee",
            "left_ankle",
            "right_ankle",
            "left_shoulder",
            "right_shoulder",
            "nose",
            "left_foot_index",
            "right_foot_index",
            "left_heel",
            "right_heel",
            "left_elbow",
            "right_elbow",
            "left_wrist",
            "right_wrist",
        ]:
            pt = get_pt(row, part_name)
            if pt[0] is not None:
                all_x.append(pt[0])
                all_y.append(pt[1])

        # Also include points from separation lines (knee and ankle)
        lk = get_pt(row, "left_knee")
        rk = get_pt(row, "right_knee")
        la = get_pt(row, "left_ankle")
        ra = get_pt(row, "right_ankle")

        if lk[0] is not None and rk[0] is not None:
            all_x.extend([lk[0], rk[0]])
            all_y.extend([lk[1], rk[1]])

        if la[0] is not None and ra[0] is not None:
            all_x.extend([la[0], ra[0]])
            all_y.extend([la[1], ra[1]])

        # Calculate limits with padding BEFORE plotting
        if all_x and all_y:
            x_min, x_max = min(all_x), max(all_x)
            y_min, y_max = min(all_y), max(all_y)

            x_range = x_max - x_min
            y_range = y_max - y_min

            # Use 30% padding (increased from 20%) to ensure all elements are visible
            # Minimum padding of 15cm to ensure enough space
            x_padding = max(x_range * 0.3, 0.15)
            y_padding = max(y_range * 0.5, 0.3)

            # Set limits BEFORE plotting to ensure proper aspect ratio calculation
            ax.set_xlim(x_min - x_padding, x_max + x_padding)
            ax.set_ylim(y_min - y_padding, y_max + y_padding)

        # Plot All Segments
        for start_part, end_part in body_segments:
            p1 = get_pt(row, start_part)
            p2 = get_pt(row, end_part)

            if p1[0] is not None and p2[0] is not None:
                color = "black"  # Default
                linewidth = 2

                # Colorize sides
                if "left" in start_part or "left" in end_part:
                    color = "blue"
                elif "right" in start_part or "right" in end_part:
                    color = "red"

                ax.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, linewidth=linewidth, alpha=0.9)
                ax.plot(p1[0], p1[1], "ko", markersize=3)
                ax.plot(p2[0], p2[1], "ko", markersize=3)

        # Plot Neck/Head (Shoulder Midpoint -> Nose)
        ls = get_pt(row, "left_shoulder")
        rs = get_pt(row, "right_shoulder")
        nose = get_pt(row, "nose")

        if all(x is not None for x in [ls[0], rs[0], nose[0]]):
            mid_should_x = (ls[0] + rs[0]) / 2
            mid_should_y = (ls[1] + rs[1]) / 2
            ax.plot([mid_should_x, nose[0]], [mid_should_y, nose[1]], color="black", linewidth=2)
            ax.plot(nose[0], nose[1], "ko", markersize=4)

        # Get key points for specific annotations
        lk = get_pt(row, "left_knee")
        rk = get_pt(row, "right_knee")
        la = get_pt(row, "left_ankle")
        ra = get_pt(row, "right_ankle")

        # Knee Sep
        if lk[0] is not None and rk[0] is not None:
            ax.plot([lk[0], rk[0]], [lk[1], rk[1]], "m--", linewidth=2, label="Knee Sep")

        # Ankle Sep
        if la[0] is not None and ra[0] is not None:
            ax.plot([la[0], ra[0]], [la[1], ra[1]], "c--", linewidth=2, label="Ankle Sep")

        # Metrics Text
        metrics_text = f"Event: {title}\nTime: {frame_idx / fps:.2f}s\n\n"

        # 1. Valgus Ratio
        valgus_ratio = results.get(f"valgus_ratio_{phase_key}")
        if valgus_ratio:
            status = "HIGH RISK" if valgus_ratio < 0.8 else "Normal"
            # color = "red" if valgus_ratio < 0.8 else "green"
            metrics_text += f"Valgus Ratio: {valgus_ratio:.2f}\n({status})\n"

        # 2. KASR
        kasr = results.get(f"kasr_{phase_key}")
        if kasr:
            metrics_text += f"KASR: {kasr:.2f}\n"

        # 3. KSD
        ksd = results.get(f"ksd_{phase_key}_m")
        if ksd:
            metrics_text += f"KSD: {ksd * 100:.1f} cm\n"

        # 4. Angles
        fppa_l = results.get(f"fppa_left_{phase_key}_deg")
        fppa_r = results.get(f"fppa_right_{phase_key}_deg")
        if fppa_l:
            metrics_text += f"FPPA L: {fppa_l:.1f}°\n"
        if fppa_r:
            metrics_text += f"FPPA R: {fppa_r:.1f}°\n"

        # Add textbox
        # Add textbox outside the plot
        props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9}
        ax.text(
            1.05,
            1.0,
            metrics_text,
            transform=ax.transAxes,
            fontsize=10,
            verticalalignment="top",
            bbox=props,
        )

        # Add legend outside the plot
        ax.legend(bbox_to_anchor=(1.05, 0.45), loc="upper left", borderaxespad=0.0)

        ax.set_title(f"Valgus Analysis - {title}")
        ax.set_xlabel("Medial-Lateral (m)")
        ax.set_ylabel("Vertical (m)")

        # #region agent log
        if all_x and all_y:
            xlim_after = ax.get_xlim()
            ylim_after = ax.get_ylim()
            log_debug(
                f"vaila_and_jump.py:{1120}",
                "After set_xlim/set_ylim (before set_aspect)",
                {
                    "phase": phase_key,
                    "xlim": list(xlim_after),
                    "ylim": list(ylim_after),
                    "x_padding": float(x_padding) if all_x and all_y else 0.15,
                    "y_padding": float(y_padding) if all_x and all_y else 0.15,
                },
                "B",
            )
        # #endregion

        # Set aspect ratio to equal AFTER all elements are plotted and limits are set
        # Using adjustable='box' tells matplotlib to adjust the plot box size, not data limits
        # This ensures all plotted elements remain visible
        ax.set_aspect("equal", adjustable="box")

        # After setting aspect ratio, verify and adjust limits if needed to ensure all elements are visible
        # Get current limits after aspect ratio adjustment
        xlim_current = ax.get_xlim()
        ylim_current = ax.get_ylim()

        # Re-check all plotted points to ensure they're within bounds
        if all_x and all_y:
            x_min_data = min(all_x)
            x_max_data = max(all_x)
            y_min_data = min(all_y)
            y_max_data = max(all_y)

            # If data extends beyond current limits, expand them
            if x_min_data < xlim_current[0] or x_max_data > xlim_current[1]:
                x_padding_extra = max((x_max_data - x_min_data) * 0.1, 0.1)
                ax.set_xlim(
                    min(x_min_data - x_padding_extra, xlim_current[0]),
                    max(x_max_data + x_padding_extra, xlim_current[1]),
                )

            if y_min_data < ylim_current[0] or y_max_data > ylim_current[1]:
                y_padding_extra = max((y_max_data - y_min_data) * 0.1, 0.1)
                ax.set_ylim(
                    min(y_min_data - y_padding_extra, ylim_current[0]),
                    max(y_max_data + y_padding_extra, ylim_current[1]),
                )

            # Re-apply aspect ratio after adjusting limits
            ax.set_aspect("equal", adjustable="box")

        # #region agent log
        aspect_final = ax.get_aspect()
        xlim_final = ax.get_xlim()
        ylim_final = ax.get_ylim()
        log_debug(
            f"vaila_and_jump.py:{1143}",
            "After set_aspect (final)",
            {
                "phase": phase_key,
                "aspect": str(aspect_final),
                "xlim": list(xlim_final),
                "ylim": list(ylim_final),
            },
            "C",
        )
        # #endregion

        filename = f"{base_name}_valgus_event_{phase_key}.png"
        filepath = os.path.join(output_dir, filename)

        # Adjust layout to ensure external text is saved
        fig.tight_layout()
        plt.subplots_adjust(right=0.7)  # increased margin for text

        # #region agent log
        aspect_before_save = ax.get_aspect()
        xlim_before_save = ax.get_xlim()
        ylim_before_save = ax.get_ylim()
        log_debug(
            f"vaila_and_jump.py:{1006}",
            "Before savefig",
            {
                "phase": phase_key,
                "aspect": str(aspect_before_save),
                "xlim": list(xlim_before_save),
                "ylim": list(ylim_before_save),
            },
            "E",
        )
        # #endregion

        # Save the figure - do NOT use tight_layout as it can alter aspect ratio
        # The aspect ratio is already set correctly with adjustable='box'
        plt.savefig(filepath, dpi=100)

        # #region agent log
        aspect_after_save = ax.get_aspect()
        xlim_after_save = ax.get_xlim()
        ylim_after_save = ax.get_ylim()
        log_debug(
            f"vaila_and_jump.py:{1007}",
            "After savefig",
            {
                "phase": phase_key,
                "aspect": str(aspect_after_save),
                "xlim": list(xlim_after_save),
                "ylim": list(ylim_after_save),
            },
            "E",
        )
        # #endregion

        plt.close()
        plot_paths.append(filepath)

    # Special handling for Landing: Create multi-moment figure
    # Based on Mechanisms for Noncontact Anterior Cruciate Ligament Injury
    if landing_frame is not None and landing_frame < len(data):
        # Define moments to plot
        landing_moments = [
            ("IC (Landing)", landing_frame, "landing"),
        ]

        # Add 40ms moment if available
        frame_40ms = landing_frame + int(0.040 * fps)
        if frame_40ms < len(data) and results.get("landing_40ms_frame") is not None:
            landing_moments.append(("IC + 40ms", frame_40ms, "landing_40ms"))

        # Add 100ms moment if available
        frame_100ms = landing_frame + int(0.100 * fps)
        if frame_100ms < len(data) and results.get("landing_100ms_frame") is not None:
            landing_moments.append(("IC + 100ms", frame_100ms, "landing_100ms"))

        # Add Max Valgus moment if available
        max_valgus_frame_left = results.get("max_valgus_frame_left")
        max_valgus_frame_right = results.get("max_valgus_frame_right")
        if max_valgus_frame_left is not None or max_valgus_frame_right is not None:
            # Use the frame with maximum valgus (could be left or right)
            max_valgus_frame = (
                max_valgus_frame_left
                if max_valgus_frame_left is not None
                else max_valgus_frame_right
            )
            if max_valgus_frame_right is not None and max_valgus_frame_right > max_valgus_frame:
                max_valgus_frame = max_valgus_frame_right
            if max_valgus_frame < len(data):
                landing_moments.append(("Max Valgus", max_valgus_frame, "max_valgus"))

        # Create figure with subplots for each moment
        n_moments = len(landing_moments)
        if n_moments > 0:
            fig, axes = plt.subplots(1, n_moments, figsize=(6 * n_moments, 8))
            if n_moments == 1:
                axes = [axes]

            # Collect all points across all moments for consistent scaling
            all_x_global = []
            all_y_global = []

            for _title, frame_idx, _phase_key in landing_moments:
                if frame_idx >= len(data):
                    continue
                row = data.iloc[frame_idx]
                for part_name in [
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                    "left_shoulder",
                    "right_shoulder",
                    "nose",
                    "left_foot_index",
                    "right_foot_index",
                    "left_heel",
                    "right_heel",
                    "left_elbow",
                    "right_elbow",
                    "left_wrist",
                    "right_wrist",
                ]:
                    pt = get_pt(row, part_name)
                    if pt[0] is not None:
                        all_x_global.append(pt[0])
                        all_y_global.append(pt[1])

            # Calculate global limits
            if all_x_global and all_y_global:
                x_min_global = min(all_x_global)
                x_max_global = max(all_x_global)
                y_min_global = min(all_y_global)
                y_max_global = max(all_y_global)
                x_range_global = x_max_global - x_min_global
                y_range_global = y_max_global - y_min_global
                x_padding_global = max(x_range_global * 0.3, 0.15)
                y_padding_global = max(y_range_global * 0.5, 0.3)
            else:
                x_min_global, x_max_global = -1, 1
                y_min_global, y_max_global = -1, 1
                x_padding_global, y_padding_global = 0.3, 0.3

            # Plot each moment
            for ax_idx, (title, frame_idx, phase_key) in enumerate(landing_moments):
                if frame_idx >= len(data):
                    continue

                ax = axes[ax_idx]
                row = data.iloc[frame_idx]

                # Check validity
                check_points = [
                    "left_hip",
                    "right_hip",
                    "left_knee",
                    "right_knee",
                    "left_ankle",
                    "right_ankle",
                ]
                valid_points = True
                for pt in check_points:
                    p = get_pt(row, pt)
                    if p[0] is None:
                        valid_points = False
                        break
                if not valid_points:
                    ax.text(
                        0.5,
                        0.5,
                        f"Invalid data\n{title}",
                        ha="center",
                        va="center",
                        transform=ax.transAxes,
                    )
                    continue

                # Plot body segments
                for start_part, end_part in body_segments:
                    p1 = get_pt(row, start_part)
                    p2 = get_pt(row, end_part)

                    if p1[0] is not None and p2[0] is not None:
                        color = "black"
                        linewidth = 2

                        if "left" in start_part or "left" in end_part:
                            color = "blue"
                        elif "right" in start_part or "right" in end_part:
                            color = "red"

                        ax.plot(
                            [p1[0], p2[0]],
                            [p1[1], p2[1]],
                            color=color,
                            linewidth=linewidth,
                            alpha=0.9,
                        )
                        ax.plot(p1[0], p1[1], "ko", markersize=3)
                        ax.plot(p2[0], p2[1], "ko", markersize=3)

                # Plot Neck/Head
                ls = get_pt(row, "left_shoulder")
                rs = get_pt(row, "right_shoulder")
                nose = get_pt(row, "nose")

                if all(x is not None for x in [ls[0], rs[0], nose[0]]):
                    mid_should_x = (ls[0] + rs[0]) / 2
                    mid_should_y = (ls[1] + rs[1]) / 2
                    ax.plot(
                        [mid_should_x, nose[0]], [mid_should_y, nose[1]], color="black", linewidth=2
                    )
                    ax.plot(nose[0], nose[1], "ko", markersize=4)

                # Knee and Ankle separation lines
                lk = get_pt(row, "left_knee")
                rk = get_pt(row, "right_knee")
                la = get_pt(row, "left_ankle")
                ra = get_pt(row, "right_ankle")

                if lk[0] is not None and rk[0] is not None:
                    ax.plot([lk[0], rk[0]], [lk[1], rk[1]], "m--", linewidth=2, label="Knee Sep")

                if la[0] is not None and ra[0] is not None:
                    ax.plot([la[0], ra[0]], [la[1], ra[1]], "c--", linewidth=2, label="Ankle Sep")

                # Calculate distances for this frame
                lh = get_pt(row, "left_hip")
                rh = get_pt(row, "right_hip")
                knee_sep = (
                    dist_2d(lk[0], lk[1], rk[0], rk[1])
                    if lk[0] is not None and rk[0] is not None
                    else None
                )
                ankle_sep = (
                    dist_2d(la[0], la[1], ra[0], ra[1])
                    if la[0] is not None and ra[0] is not None
                    else None
                )
                hip_sep = (
                    dist_2d(lh[0], lh[1], rh[0], rh[1])
                    if lh[0] is not None and rh[0] is not None
                    else None
                )
                valgus_ratio = (
                    (knee_sep / hip_sep) if (knee_sep and hip_sep and hip_sep > 0) else None
                )

                # Metrics Text
                metrics_text = f"{title}\nTime: {frame_idx / fps:.3f}s\nFrame: {frame_idx}\n\n"

                # Get FPPA values based on phase_key
                if phase_key == "landing":
                    fppa_l = results.get("fppa_left_landing_deg")
                    fppa_r = results.get("fppa_right_landing_deg")
                elif phase_key == "landing_40ms":
                    fppa_l = results.get("fppa_left_landing_40ms_deg")
                    fppa_r = results.get("fppa_right_landing_40ms_deg")
                elif phase_key == "landing_100ms":
                    fppa_l = results.get("fppa_left_landing_100ms_deg")
                    fppa_r = results.get("fppa_right_landing_100ms_deg")
                elif phase_key == "max_valgus":
                    fppa_l = results.get("max_valgus_angle_left_deg")
                    fppa_r = results.get("max_valgus_angle_right_deg")
                else:
                    fppa_l = None
                    fppa_r = None

                # Add FPPA values
                if fppa_l is not None:
                    risk_l, color_l = _get_fppa_risk_classification(fppa_l)
                    metrics_text += f"FPPA L: {fppa_l:.1f}°\n({risk_l})\n"
                if fppa_r is not None:
                    risk_r, color_r = _get_fppa_risk_classification(fppa_r)
                    metrics_text += f"FPPA R: {fppa_r:.1f}°\n({risk_r})\n"

                # Add distances and ratios
                if knee_sep is not None:
                    metrics_text += f"\nKnee Sep: {knee_sep * 100:.1f} cm\n"
                if ankle_sep is not None:
                    metrics_text += f"Ankle Sep: {ankle_sep * 100:.1f} cm\n"
                if valgus_ratio is not None:
                    status = "HIGH RISK" if valgus_ratio < 0.8 else "Normal"
                    metrics_text += f"Valgus Ratio: {valgus_ratio:.2f}\n({status})"

                # Add legend for lines
                # The original legend call for ax_idx == 0 is removed as a new one is added outside the plot.

                # Add textbox outside
                props = {"boxstyle": "round", "facecolor": "white", "alpha": 0.9}
                ax.text(
                    1.05,
                    1.0,
                    metrics_text,
                    transform=ax.transAxes,
                    fontsize=9,
                    verticalalignment="top",
                    bbox=props,
                )

                # Add legend outside
                ax.legend(bbox_to_anchor=(1.05, 0.45), loc="upper left", borderaxespad=0.0)

                # Set limits and aspect
                ax.set_xlim(x_min_global - x_padding_global, x_max_global + x_padding_global)
                ax.set_ylim(y_min_global - y_padding_global, y_max_global + y_padding_global)
                ax.set_aspect("equal", adjustable="box")

                ax.set_title(f"{title}", fontsize=11, fontweight="bold")
                if ax_idx == 0:
                    ax.set_ylabel("Vertical (m)", fontsize=10)
                ax.set_xlabel("Medial-Lateral (m)", fontsize=10)
                ax.grid(True, alpha=0.3)

            # Overall title
            fig.suptitle(
                "Landing Phase Analysis - Multiple Moments", fontsize=14, fontweight="bold"
            )
            plt.tight_layout(rect=[0, 0, 1, 0.96])

            # Save figure
            # Use the same naming convention as the original landing plot
            filename = f"{base_name}_valgus_event_landing.png"
            filepath = os.path.join(output_dir, filename)
            plt.savefig(filepath, dpi=100, bbox_inches="tight")
            plt.close()
            plot_paths.append(filepath)

    return plot_paths


def generate_jump_plots(data, results, output_dir, base_name):
    plot_files = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Accessing variables from the results dictionary
    fps = results["fps"]
    takeoff_frame = results.get("takeoff_frame", None)
    max_power = results.get("max_power_W", None)
    time_max_power = results.get("time_max_power_s", None)
    power_takeoff = results.get("power_takeoff_W", None)

    # Plot the instantaneous power curve
    plt.figure(figsize=(12, 6))
    plt.plot(
        data.index / fps,
        data["power"],
        label="Instantaneous Power (W)",
        color="purple",
    )

    if takeoff_frame is not None:
        plt.axvline(takeoff_frame / fps, color="g", linestyle="--", label="Takeoff")

    if max_power is not None and time_max_power is not None:
        plt.axvline(time_max_power, color="orange", linestyle=":", label="Max. Power")
        plt.scatter(time_max_power, max_power, color="orange", zorder=5, label="Max. Power")

    if power_takeoff is not None and takeoff_frame is not None:
        plt.scatter(
            takeoff_frame / fps,
            power_takeoff,
            color="g",
            zorder=5,
            label="Power Takeoff",
        )

    plt.xlabel("Time (s)")
    plt.ylabel("Power (W)")
    plt.title("Instantaneous Power during the jump")
    plt.legend(loc="upper left")
    plt.grid(True)

    # Saving the plot
    power_plot_path = os.path.join(output_dir, f"{base_name}_power_curve_{timestamp}.png")
    plt.savefig(power_plot_path, dpi=300, bbox_inches="tight")
    plt.close()
    plot_files.append(power_plot_path)

    return plot_files


def plot_jump_phases_analysis(
    data, takeoff_frame, max_height_frame, landing_frame, fps, output_dir, base_name
):
    """
    Generate a visualization showing jump phases with colored regions.

    Args:
        data (pd.DataFrame): The jump data
        takeoff_frame (int): Frame where takeoff occurs
        max_height_frame (int): Frame where maximum height is reached
        landing_frame (int): Frame where landing occurs
        fps (int): Frames per second
        output_dir (str): Directory to save the output
        base_name (str): Base name for the output file

    Returns:
        str: Path to the saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Find squat frame (minimum CG position)
    squat_frame = data["cg_y_normalized"].idxmin()

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Convert frames to time
    time = data.index / fps

    # Plot CG path
    ax.plot(
        time,
        data["cg_y_normalized"],
        "b-",
        linewidth=3,
        label="CG Path (normalized)",
    )

    # Calculate key times
    squat_time = squat_frame / fps
    takeoff_time = takeoff_frame / fps
    landing_time = landing_frame / fps
    max_height_time = max_height_frame / fps

    # Shade the propulsion phase (from squat to takeoff)
    ax.axvspan(
        squat_time,
        takeoff_time,
        alpha=0.20,
        color="gold",
        label=f"Propulsion Phase: {takeoff_time - squat_time:.3f} s",
        zorder=0,
    )

    # Color the flight phase with a light blue background
    ax.axvspan(
        takeoff_time,
        landing_time,
        alpha=0.15,
        color="lightblue",
        label="Flight Phase",
    )

    # Color the ascent with a light green background
    ax.axvspan(
        takeoff_time,
        max_height_time,
        alpha=0.15,
        color="lightgreen",
        label="Ascent",
    )

    # Color the descent with a light red background
    ax.axvspan(
        max_height_time,
        landing_time,
        alpha=0.15,
        color="mistyrose",
        label="Descent",
    )

    # Mark the highest point
    max_height = data["cg_y_normalized"].iloc[max_height_frame]
    ax.plot(
        max_height_time,
        max_height,
        "ro",
        markersize=10,
        label=f"Max Height: {max_height:.3f}m from initial CG",
    )

    # Calculate and add flight time and velocity annotations
    flight_time = landing_time - takeoff_time
    velocity = calculate_velocity(max_height)

    # Add text annotations in boxes
    bbox_props = {"boxstyle": "round,pad=0.5", "fc": "yellow", "alpha": 0.7}
    ax.text(
        max_height_time + 0.1,
        max_height * 0.8,
        f"Flight Time: {flight_time:.3f}s",
        ha="center",
        va="center",
        bbox=bbox_props,
    )
    ax.text(
        max_height_time + 0.1,
        max_height * 0.6,
        f"Velocity: {velocity:.2f} m/s",
        ha="center",
        va="center",
        bbox=bbox_props,
    )

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position (meters from initial CG)")
    ax.set_title("Jump Phases Analysis - Normalized from Initial CG Position")
    ax.grid(True)
    ax.legend(loc="upper left")

    # Save the plot
    plot_path = os.path.join(output_dir, f"{base_name}_jump_phases_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_jump_cg_feet_analysis(
    data, takeoff_frame, max_height_frame, landing_frame, fps, output_dir, base_name
):
    """
    Generate a visualization showing CG and feet positions with phase markers.

    Args:
        data (pd.DataFrame): The jump data
        takeoff_frame (int): Frame where takeoff occurs
        max_height_frame (int): Frame where maximum height is reached
        landing_frame (int): Frame where landing occurs
        fps (int): Frames per second
        output_dir (str): Directory to save the output
        base_name (str): Base name for the output file

    Returns:
        str: Path to the saved plot
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Create figure
    fig, ax = plt.subplots(figsize=(14, 8))

    # Convert frames to time
    time = data.index / fps

    # Plot CG path
    ax.plot(
        time,
        data["cg_y_normalized"],
        "b-",
        linewidth=2,
        label="Center of Gravity (normalized)",
    )

    # Plot feet positions if available
    if "left_foot_index_y_m" in data.columns:
        normalized_left_foot = data["left_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_left_foot,
            "g-",
            linewidth=2,
            label="Left Foot (normalized)",
        )

    if "right_foot_index_y_m" in data.columns:
        normalized_right_foot = data["right_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_right_foot,
            "r-",
            linewidth=2,
            label="Right Foot (normalized)",
        )

    # Calculate key times
    takeoff_time = takeoff_frame / fps
    max_height_time = max_height_frame / fps
    landing_time = landing_frame / fps

    # Mark the phases with vertical lines
    ax.axvline(
        x=takeoff_time,
        color="green",
        linestyle="--",
        label="Takeoff",
    )
    ax.axvline(
        x=max_height_time,
        color="red",
        linestyle="--",
        label="Max Height",
    )
    ax.axvline(
        x=landing_time,
        color="black",
        linestyle="--",
        label="Landing",
    )

    # Add reference line for initial CG position
    ax.axhline(y=0, color="gray", linestyle="-", label="Initial CG Position (reference)")

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position (meters from initial CG) - Up is positive")
    ax.set_title("Jump Analysis - Normalized CG and Feet Positions")
    ax.grid(True)
    ax.legend(loc="upper left")

    # Save the plot
    plot_path = os.path.join(output_dir, f"{base_name}_cg_feet_analysis_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


def draw_fppa_overlay(
    ax,
    hip_x,
    hip_y,
    knee_x,
    knee_y,
    ankle_x,
    ankle_y,
    fppa_angle,
    side="left",
    color_segment="blue",
    color_angle="red",
):
    """
    Draw FPPA overlay on a matplotlib axes, similar to scientific figure 4.

    Draws the femur (HIP->KNEE) and tibia (KNEE->ANKLE) segments with the FPPA angle
    displayed at the knee joint.

    This function can be used for visualization in stick figure plots to show
    the FPPA angle calculation visually, similar to Figure 4 in scientific papers.

    Args:
        ax: Matplotlib axes object
        hip_x, hip_y: Hip coordinates
        knee_x, knee_y: Knee coordinates
        ankle_x, ankle_y: Ankle coordinates
        fppa_angle: FPPA angle in degrees
        side: "left" or "right" for labeling
        color_segment: Color for the segments
        color_angle: Color for the angle arc
    """
    import matplotlib.patches as patches

    # Draw femur segment (HIP -> KNEE)
    ax.plot(
        [hip_x, knee_x],
        [hip_y, knee_y],
        color=color_segment,
        linewidth=2.5,
        label=f"{side.capitalize()} Femur",
    )
    ax.plot(hip_x, hip_y, "o", color=color_segment, markersize=8, label="Hip")
    ax.plot(knee_x, knee_y, "o", color=color_segment, markersize=10, label="Knee")

    # Draw tibia segment (KNEE -> ANKLE)
    ax.plot(
        [knee_x, ankle_x],
        [knee_y, ankle_y],
        color=color_segment,
        linewidth=2.5,
        linestyle="--",
        label=f"{side.capitalize()} Tibia",
    )
    ax.plot(ankle_x, ankle_y, "o", color=color_segment, markersize=8, label="Ankle")

    # Draw angle arc at the knee
    # Calculate vectors for angle arc
    v1_x = knee_x - hip_x
    v1_y = knee_y - hip_y
    v2_x = ankle_x - knee_x
    v2_y = ankle_y - knee_y

    # Calculate angles
    angle_v1 = math.atan2(v1_y, v1_x)
    angle_v2 = math.atan2(v2_y, v2_x)

    # Draw arc (from v1 to v2)
    arc_radius = min(math.sqrt(v1_x**2 + v1_y**2), math.sqrt(v2_x**2 + v2_y**2)) * 0.3

    # Determine start and end angles for the arc
    start_angle = math.degrees(angle_v1)
    end_angle = math.degrees(angle_v2)

    # Create arc
    arc = patches.Arc(
        (knee_x, knee_y),
        arc_radius * 2,
        arc_radius * 2,
        angle=0,
        theta1=start_angle,
        theta2=end_angle,
        color=color_angle,
        linewidth=2,
    )
    ax.add_patch(arc)

    # Add angle text near the knee
    mid_angle = (start_angle + end_angle) / 2
    text_x = knee_x + arc_radius * 0.7 * math.cos(math.radians(mid_angle))
    text_y = knee_y + arc_radius * 0.7 * math.sin(math.radians(mid_angle))
    ax.text(
        text_x,
        text_y,
        f"{fppa_angle:.1f}°",
        fontsize=10,
        color=color_angle,
        fontweight="bold",
        bbox={"boxstyle": "round,pad=0.3", "facecolor": "white", "alpha": 0.8},
    )


def _get_fppa_risk_classification(fppa_angle):
    """
    Classify FPPA angle into risk categories based on scientific evidence.

    Args:
        fppa_angle (float): FPPA angle in degrees

    Returns:
        tuple: (classification_text, color_class)
    """
    if fppa_angle is None:
        return ("N/A", "black")

    abs_angle = abs(fppa_angle)
    if abs_angle < 5:
        return ("Good Alignment", "green")
    elif abs_angle <= 10:
        return ("Moderate Risk", "orange")
    else:
        # Note: High risk applies to both excessive valgus (negative) and varus (positive)
        return ("High Risk / Excessive Dynamic Valgus or Varus", "red")


def _format_fppa_with_risk(fppa_angle):
    """
    Format FPPA angle with color coding based on risk classification.

    Args:
        fppa_angle (float): FPPA angle in degrees

    Returns:
        str: HTML formatted string with color
    """
    if fppa_angle is None:
        return "N/A"

    classification, color = _get_fppa_risk_classification(fppa_angle)
    return f'<span style="color: {color}; font-weight: bold;">{fppa_angle:.1f}°</span>'


def generate_html_report(data, results, plot_files, output_dir, base_name):
    """
    Generate an HTML report with jump metrics and plots.

    Args:
        data (pd.DataFrame): DataFrame with processed data
        results (dict): Dictionary with jump metrics
        plot_files (list): List of paths to plot files
        output_dir (str): Directory to save the report
        base_name (str): Base name for the output file

    Returns:
        str: Path to the generated HTML report
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = os.path.join(output_dir, f"{base_name}_report_{timestamp}.html")

    # Try to load VAILA logo
    logo_b64 = ""
    try:
        # Assuming typical project structure: vaila/vaila/vaila_and_jump.py -> vaila/docs/images/vaila.png
        current_file = Path(__file__).resolve()
        project_root = current_file.parent.parent
        logo_path = project_root / "docs" / "images" / "vaila.png"

        if logo_path.exists():
            with open(logo_path, "rb") as img_file:
                logo_data = img_file.read()
                logo_b64 = base64.b64encode(logo_data).decode("utf-8")
    except Exception as e:
        print(f"Warning: Could not load logo: {e}")

    # Note: Valgus plots should already be generated and included in plot_files
    # before calling this function

    # Create HTML content
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>vailá - Jump Analysis Report - {base_name}</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                line-height: 1.6;
                max-width: 1200px;
                margin: 0 auto;
                padding: 20px;
                color: #333;
                background-color: #fcfcfc;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
                margin-top: 0;
            }}
            h2 {{
                color: #2980b9;
                margin-top: 40px;
                border-bottom: 1px solid #eee;
                padding-bottom: 5px;
            }}
            h3 {{
                color: #34495e;
                margin-top: 25px;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
                box-shadow: 0 1px 3px rgba(0,0,0,0.1);
                background-color: white;
            }}
            th, td {{
                border: 1px solid #e0e0e0;
                padding: 12px 15px;
                text-align: left;
            }}
            th {{
                background-color: #f8f9fa;
                font-weight: 600;
                color: #2c3e50;
            }}
            tr:nth-child(even) {{
                background-color: #f9f9f9;
            }}
            tr:hover {{
                background-color: #f1f1f1;
            }}
            .img-container {{
                text-align: center;
                margin: 30px 0;
                background: white;
                padding: 10px;
                border-radius: 4px;
                box-shadow: 0 2px 5px rgba(0,0,0,0.05);
            }}
            .img-container img {{
                max-width: 100%;
                height: auto;
                border-radius: 2px;
            }}
            .note {{
                background-color: #e8f5e9;
                border-left: 5px solid #4caf50;
                padding: 15px;
                margin: 20px 0;
                border-radius: 0 4px 4px 0;
            }}
            .references {{
                margin-top: 60px;
                background-color: #f8f9fa;
                border-left: 5px solid #3498db;
                padding: 20px;
                font-size: 0.9em;
            }}
            .references h3 {{
                margin-top: 0;
            }}
            .footer {{
                margin-top: 50px;
                border-top: 1px solid #ddd;
                padding-top: 20px;
                color: #7f8c8d;
                font-size: 0.85em;
                text-align: center;
            }}
            .warning {{
                color: #c0392b;
                font-weight: bold;
            }}
            .logo-container {{
                position: absolute;
                top: 20px;
                right: 20px;
                z-index: 1000;
            }}
            .logo-img {{
                width: 150px;
                height: auto;
                opacity: 0.9;
            }}
        </style>
    </head>
    <body>
        <div class="logo-container">
            {f'<img src="data:image/png;base64,{logo_b64}" class="logo-img" alt="VAILA Logo">' if logo_b64 else ""}
        </div>
        <h1>Jump Analysis Report</h1>
        <p><strong>Subject ID:</strong> {base_name}</p>
        <p><strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="note">
            <h3>Methodology & Coordinate System</h3>
            <p>This analysis utilizes a biomechanical coordinate system:</p>
            <ul>
                <li><strong>Origin:</strong> Bottom-left corner (0,0)</li>
                <li><strong>X-axis:</strong> Positive to the right (Medial-Lateral)</li>
                <li><strong>Y-axis:</strong> Positive upward (Vertical)</li>
                <li><strong>Z-axis:</strong> Positive forward (Anterior-Posterior)</li>
                <li><strong>Units:</strong> Meters (m)</li>
            </ul>
            <p><strong>Height Calculation:</strong> Jump height is measured relative to the <em>initial center of gravity (CG) position</em> (averaged over the first 10-20 frames). This initial height is treated as the zero reference.</p>

            <h3>Vertical Jump Power Estimation</h3>
            <p>
            Three power metrics are estimated based on Center of Mass (CG) kinematics:
            </p>

            <h4>1. Instantaneous Power</h4>
            <p>
              Calculated at each time step <em>t</em> during propulsion:<br>
              <code>P(t) = F(t) · v(t)</code><br>
              Where:
              <ul>
                <li><code>F(t) = m · (a(t) + g)</code> : Total vertical ground reaction force</li>
                <li><code>v(t)</code> : Vertical velocity of the CG</li>
              </ul>
            </p>

            <h4>2. Takeoff Power</h4>
            <p>
              Power output at the precise moment of takeoff (toes leave ground).
            </p>

            <h4>3. Average Propulsion Power</h4>
            <p>
              <code>P<sub>avg</sub> = (KE<sub>takeoff</sub> + PE<sub>max</sub>) / t<sub>propulsion</sub></code><br>
              Represents the average power output throughout the entire push-off phase.
            </p>
        </div>

        <h2>Jump Metrics</h2>
        <table>
            <tr>
                <th>Metric</th>
                <th>Value</th>
                <th>Unit</th>
            </tr>
            <tr>
                <td>Mass</td>
                <td>{results["mass_kg"]}</td>
                <td>kg</td>
            </tr>
            <tr>
                <td>Jump Height - CG Method</td>
                <td>{results.get("height_cg_method_m", "N/A")}</td>
                <td>m (from initial CG position)</td>
            </tr>
            <tr>
                <td>Jump Height - Flight Time Method</td>
                <td>{results.get("height_flight_time_method_m", "N/A")}</td>
                <td>m (calculated from flight time)</td>
            </tr>
            <tr>
                <td>Jump Height - Left Foot Method</td>
                <td>{results.get("height_left_foot_m", "N/A")}</td>
                <td>m (maximum left foot height)</td>
            </tr>
            <tr>
                <td>Jump Height - Right Foot Method</td>
                <td>{results.get("height_right_foot_m", "N/A")}</td>
                <td>m (maximum right foot height)</td>
            </tr>
            <tr>
                <td>Jump Height - Average Feet Method</td>
                <td>{results.get("height_avg_feet_m", "N/A")}</td>
                <td>m (average of left and right feet)</td>
            </tr>
            <tr>
                <td>Squat Depth</td>
                <td>{results.get("squat_depth_m", "N/A")}</td>
                <td>m (maximum downward CG displacement)</td>
            </tr>
            <tr>
                <td>Flight Time</td>
                <td>{f"{results['flight_time_s']:.3f}" if results["flight_time_s"] is not None else "N/A"}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Takeoff Velocity</td>
                <td>{results["velocity_m/s"] if results["velocity_m/s"] is not None else "N/A"}</td>
                <td>m/s</td>
            </tr>
            <tr>
                <td>Potential Energy</td>
                <td>{results["potential_energy_J"] if results["potential_energy_J"] is not None else "N/A"}</td>
                <td>J</td>
            </tr>
            <tr>
                <td>Kinetic Energy</td>
                <td>{results["kinetic_energy_J"] if results["kinetic_energy_J"] is not None else "N/A"}</td>
                <td>J</td>
            </tr>
            <tr>
                <td>Maximum Power in the Propulsion</td>
                <td>{results.get("max_power_W", "—")}</td>
                <td>W</td>
            </tr>
            <tr>
                <td>Power at Takeoff</td>
                <td>{results.get("power_takeoff_W", "—")}</td>
                <td>W</td>
            </tr>
            <tr>
                <td>Average Power in the Propulsion</td>
                <td>{results.get("power_avg_propulsion_W", "—")}</td>
                <td>W</td>
            </tr>
            <tr>
                <td>Propulsion Time</td>
                <td>{f"{results.get('propulsion_time_s', '—'):.3f}" if results.get("propulsion_time_s") is not None and results.get("propulsion_time_s") != "—" else "—"}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Ascent Time</td>
                <td>{f"{results.get('ascent_time_s', '—'):.3f}" if results.get("ascent_time_s") is not None and results.get("ascent_time_s") != "—" else "—"}</td>
                <td>s</td>
            </tr>
            <tr>
                <td>Descent Time</td>
                <td>{f"{results.get('descent_time_s', '—'):.3f}" if results.get("descent_time_s") is not None and results.get("descent_time_s") != "—" else "—"}</td>
                <td>s</td>
            </tr>
        </table>

        <h2>Bilateral Foot Analysis</h2>
        <table>
            <tr>
                <th>Parameter</th>
                <th>Left Foot</th>
                <th>Right Foot</th>
                <th>Difference</th>
            </tr>
            <tr>
                <td>Takeoff Time</td>
                <td>{f"{results.get('left_takeoff_time_s', 'N/A'):.3f}" if results.get("left_takeoff_time_s") is not None and results.get("left_takeoff_time_s") != "N/A" else "N/A"}</td>
                <td>{f"{results.get('right_takeoff_time_s', 'N/A'):.3f}" if results.get("right_takeoff_time_s") is not None and results.get("right_takeoff_time_s") != "N/A" else "N/A"}</td>
                <td>{round(abs(results.get("left_takeoff_time_s", 0) - results.get("right_takeoff_time_s", 0)), 3) if results.get("left_takeoff_time_s") and results.get("right_takeoff_time_s") else "N/A"}</td>
            </tr>
            <tr>
                <td>Landing Time</td>
                <td>{f"{results.get('left_landing_time_s', 'N/A'):.3f}" if results.get("left_landing_time_s") is not None and results.get("left_landing_time_s") != "N/A" else "N/A"}</td>
                <td>{f"{results.get('right_landing_time_s', 'N/A'):.3f}" if results.get("right_landing_time_s") is not None and results.get("right_landing_time_s") != "N/A" else "N/A"}</td>
                <td>{round(abs(results.get("left_landing_time_s", 0) - results.get("right_landing_time_s", 0)), 3) if results.get("left_landing_time_s") and results.get("right_landing_time_s") else "N/A"}</td>
            </tr>
            <tr>
                <td>Max Height</td>
                <td>{f"{results.get('height_left_foot_m', 'N/A'):.3f}" if results.get("height_left_foot_m") is not None and results.get("height_left_foot_m") != "N/A" else "N/A"}</td>
                <td>{f"{results.get('height_right_foot_m', 'N/A'):.3f}" if results.get("height_right_foot_m") is not None and results.get("height_right_foot_m") != "N/A" else "N/A"}</td>
                <td>{round(abs(results.get("height_left_foot_m", 0) - results.get("height_right_foot_m", 0)), 3) if results.get("height_left_foot_m") and results.get("height_right_foot_m") else "N/A"}</td>
            </tr>
        </table>

        <h2>Kinematics & Risk Screening</h2>

        <h3>Stability at Landing (0.4s)</h3>
        <table>
            <tr>
                   <th>Metric</th>
                   <th>Valor</th>
                   <th>Status</th>
            </tr>
            <tr>
                <td>Landing Sway (Medi-Lateral)</td>
                <td>{f"{results.get('landing_stability_sway_x', 0) * 100:.2f}" if results.get("landing_stability_unit") == "m" else f"{results.get('landing_stability_sway_x', 0):.4f}"}</td>
                <td>{"cm" if results.get("landing_stability_unit") == "m" else "normalized"}</td>
            </tr>
        </table>

        <h3>Dynamic Valgus & Alignment</h3>
        <p>Reference values: <strong>Ratio < 0.8</strong> indicates excessive knee approximation (Valgus).</p>
        <p><strong>FPPA Risk Classification:</strong> &lt; 5° = Good alignment (Green), 5°-10° = Moderate risk (Yellow), &gt; 10° = High risk / Excessive dynamic valgus or varus (Red)</p>
        <p><strong>FPPA Convention:</strong> Positive = Varus (Abduction/Lateral collapse), Negative = Valgus (Adduction/Medial collapse)</p>

        <table>
            <tr>
                <th>Phase</th>
                <th>Knee-Hip Separation Ratio</th>
                <th>FPPA Left (degrees)</th>
                <th>FPPA Right (degrees)</th>
            </tr>
            <tr>
                <td><strong>Squat (Propulsion Start)</strong></td>
                <td>
                    <span style="color: {"red" if results.get("valgus_ratio_squat") is not None and results.get("valgus_ratio_squat") < 0.8 else "black"}; font-weight: bold;">
                        {f"{results.get('valgus_ratio_squat', 0):.2f}" if results.get("valgus_ratio_squat") is not None else "N/A"}
                    </span>
                </td>
                <td>
                    {_format_fppa_with_risk(results.get("fppa_left_squat_deg"))}
                </td>
                <td>
                    {_format_fppa_with_risk(results.get("fppa_right_squat_deg"))}
                </td>
            </tr>
            <tr>
                <td><strong>Landing (Initial Contact)</strong></td>
                <td>
                    <span style="color: {"red" if results.get("valgus_ratio_landing") is not None and results.get("valgus_ratio_landing") < 0.8 else "black"}; font-weight: bold;">
                        {f"{results.get('valgus_ratio_landing', 0):.2f}" if results.get("valgus_ratio_landing") is not None else "N/A"}
                    </span>
                </td>
                <td>
                    {_format_fppa_with_risk(results.get("fppa_left_landing_deg"))}
                </td>
                <td>
                    {_format_fppa_with_risk(results.get("fppa_right_landing_deg"))}
                </td>
            </tr>
        </table>

        <h3>Landing Phase Analysis - Multiple Moments</h3>
        <p><em>Based on Mechanisms for Noncontact Anterior Cruciate Ligament Injury</em></p>
        <table>
            <tr>
                <th>Moment</th>
                <th>FPPA Left (degrees)</th>
                <th>FPPA Right (degrees)</th>
                <th>Frame</th>
                <th>Time (s)</th>
            </tr>
            <tr>
                <td><strong>IC (Initial Contact)</strong></td>
                <td>{_format_fppa_with_risk(results.get("fppa_left_landing_deg"))}</td>
                <td>{_format_fppa_with_risk(results.get("fppa_right_landing_deg"))}</td>
                <td>{results.get("landing_frame", "N/A")}</td>
                <td>{f"{results.get('landing_frame', 0) / results.get('fps', 30):.3f}" if results.get("landing_frame") is not None else "N/A"}</td>
            </tr>
            <tr>
                <td><strong>IC + 40ms</strong></td>
                <td>{_format_fppa_with_risk(results.get("fppa_left_landing_40ms_deg"))}</td>
                <td>{_format_fppa_with_risk(results.get("fppa_right_landing_40ms_deg"))}</td>
                <td>{results.get("landing_40ms_frame", "N/A")}</td>
                <td>{f"{results.get('landing_40ms_frame', 0) / results.get('fps', 30):.3f}" if results.get("landing_40ms_frame") is not None else "N/A"}</td>
            </tr>
            <tr>
                <td><strong>IC + 100ms</strong></td>
                <td>{_format_fppa_with_risk(results.get("fppa_left_landing_100ms_deg"))}</td>
                <td>{_format_fppa_with_risk(results.get("fppa_right_landing_100ms_deg"))}</td>
                <td>{results.get("landing_100ms_frame", "N/A")}</td>
                <td>{f"{results.get('landing_100ms_frame', 0) / results.get('fps', 30):.3f}" if results.get("landing_100ms_frame") is not None else "N/A"}</td>
            </tr>
            <tr>
                <td><strong>Max Valgus (0.2s window)</strong></td>
                <td>{_format_fppa_with_risk(results.get("max_valgus_angle_left_deg"))}</td>
                <td>{_format_fppa_with_risk(results.get("max_valgus_angle_right_deg"))}</td>
                <td>L: {results.get("max_valgus_frame_left", "N/A")}, R: {results.get("max_valgus_frame_right", "N/A")}</td>
                <td>L: {f"{results.get('max_valgus_frame_left', 0) / results.get('fps', 30):.3f}" if results.get("max_valgus_frame_left") is not None else "N/A"}, R: {f"{results.get('max_valgus_frame_right', 0) / results.get('fps', 30):.3f}" if results.get("max_valgus_frame_right") is not None else "N/A"}</td>
            </tr>
        </table>

        <h2>Jump Phase Frames</h2>
        <table>
            <tr>
                <th>Phase</th>
                <th>Frame</th>
                <th>Time (s)</th>
            </tr>
            <tr>
                <td>Takeoff</td>
                <td>{results["takeoff_frame"] if results["takeoff_frame"] is not None else "N/A"}</td>
                <td>{f"{results['takeoff_frame'] / results['fps']:.3f}" if results["takeoff_frame"] is not None else "N/A"}</td>
            </tr>
            <tr>
                <td>Maximum Height</td>
                <td>{results["max_height_frame"] if results["max_height_frame"] is not None else "N/A"}</td>
                <td>{f"{results['max_height_frame'] / results['fps']:.3f}" if results["max_height_frame"] is not None else "N/A"}</td>
            </tr>
            <tr>
                <td>Landing</td>
                <td>{results["landing_frame"] if results["landing_frame"] is not None else "N/A"}</td>
                <td>{f"{results['landing_frame'] / results['fps']:.3f}" if results["landing_frame"] is not None else "N/A"}</td>
            </tr>
        </table>

        <h2>Jump Analysis Visualizations</h2>
    """

    # Add images to the report
    for plot_file in plot_files:
        plot_filename = os.path.basename(plot_file)
        # Use relative paths in the HTML
        html_content += f"""
        <div class="img-container">
            <img src="{plot_filename}" alt="Jump analysis plot">
            <p><em>{plot_filename}</em></p>
        </div>
        """

    # Try to embed an animated GIF if it exists in the output directory
    try:
        maybe_gifs = [p for p in os.listdir(output_dir) if p.lower().endswith(".gif")]
        if maybe_gifs:
            gif_name = sorted(maybe_gifs)[0]
            html_content += f"""
            <div class="img-container">
                <img src="{gif_name}" alt="Jump animation (stick figure)">
                <p><em>{gif_name} — compact stick-figure animation over the jump</em></p>
            </div>
            """
    except Exception:
        pass

    # Add references section at the end
    html_content += """
        <div class="references">
            <h2>References</h2>
            <ul>
              <li>Santiago, P. R. P., Chinaglia, A. G., Flanagan, K., Bedo, B. L., Mochida, L. Y., Aceros, J., ... & Cesar, G. M. (2024). vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox. arXiv preprint arXiv:2410.07238. https://doi.org/10.48550/arXiv.2410.07238</li>
              <li>Samozino, P., Morin, J. B., Hintzy, F., & Belli, A. (2008). A simple method for measuring force, velocity and power output during squat jump. Journal of Biomechanics, 41(14), 2940-2945.</li>
              <li>Aragón-Vargas, L. F., & Gross, M. M. (1997). Kinesiological factors in vertical jump performance: differences among individuals. Journal of Applied Biomechanics, 13(1), 24-44.</li>
              <li>Harman, E. A., Rosenstein, M. T., Frykman, P. N., & Rosenstein, R. M. (1991). Estimation of human power output from vertical jump. Journal of Applied Sport Science Research, 5(3), 116-120.</li>
              <li>Sayers, S. P., Harackiewicz, D. V., Harman, E. A., Frykman, P. N., & Rosenstein, M. T. (1999). Cross-validation of three jump power equations. Medicine & Science in Sports & Exercise, 31(4), 572-577.</li>
              <li>Koga H, Nakamae A, Shima Y, Iwasa J, Myklebust G, Engebretsen L, Bahr R & Krosshaug T. (2010). Mechanisms for noncontact anterior cruciate ligament injuries: knee joint kinematics in 10 injury situations from female team handball and basketball. Am J Sports Med. 2010 Nov;38(11):2218-25. doi: 10.1177/0363546510373570</li>
            </ul>
        </div>
    """

    # Close the HTML content
    html_content += """
        <div class="footer">
            <p>Generated by vailá - Vertical Jump Analysis Tool</p>
        </div>
    </body>
    </html>
    """

    # Write the HTML file with UTF-8 encoding
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    return report_path


def process_mediapipe_data(input_file, output_dir):
    """
    Process MediaPipe data and generate visualizations and report.

    Args:
        input_file (str): Path to the input CSV file
        output_dir (str): Path to the output directory
    """
    try:
        # Generate timestamp at the beginning
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Load the .csv file
        data = pd.read_csv(input_file)
        base_name = os.path.splitext(os.path.basename(input_file))[0]

        # Invert all y coordinates (1.0 - y) to fix orientation
        for col in [c for c in data.columns if c.endswith("_y")]:
            data[col] = 1.0 - data[col]

        # Request mass/FPS/shank only once per batch via shared context
        data_folder = Path(input_file).parent
        ctx = _get_or_ask_jump_context(base_dir=data_folder)
        if ctx is None:
            print("Cancelled by user (context).")
            return
        mass = ctx["mass_kg"]
        fps = ctx["fps"]
        shank_length_real = ctx["shank_length_m"]

        # Calculate the conversion factor for normalized pixels to meters
        # Use keyword argument to avoid misplacing into 'knee'
        conversion_factor = calc_fator_convert_mediapipe(
            data,
            shank_length_real=shank_length_real,
        )
        print(f"Conversion factor: {conversion_factor:.6f} m/unit")

        # Processing to calculate the CG (already in meters)
        cg_x_m_list = []
        cg_y_m_list = []

        # Process in chunks to calculate the CG
        chunk_size = 50
        for i in range(0, len(data), chunk_size):
            chunk = data.iloc[i : i + chunk_size]
            for _, row in chunk.iterrows():
                temp_df = pd.DataFrame([row])
                # Calculate CG and store the results
                cg_x, cg_y = calculate_cg_frame(temp_df, conversion_factor)
                # Fix the issue of Series vs scalar
                cg_x_m_list.append(float(cg_x.iloc[0]) if hasattr(cg_x, "iloc") else float(cg_x))
                cg_y_m_list.append(float(cg_y.iloc[0]) if hasattr(cg_y, "iloc") else float(cg_y))

        # Add the CG columns in meters to the DataFrame
        data["cg_x_m"] = cg_x_m_list
        data["cg_y_m"] = cg_y_m_list

        # Convert all coordinates to meters
        cols_to_convert = {}

        # Convert x coordinates
        for col in [c for c in data.columns if c.endswith("_x")]:
            base_name_coord = col[:-2]
            cols_to_convert[f"{base_name_coord}_x_m"] = data[col] * conversion_factor

        # Convert y coordinates
        for col in [c for c in data.columns if c.endswith("_y")]:
            base_name_coord = col[:-2]
            cols_to_convert[f"{base_name_coord}_y_m"] = data[col] * conversion_factor

        # Convert z coordinates
        for col in [c for c in data.columns if c.endswith("_z")]:
            base_name_coord = col[:-2]
            cols_to_convert[f"{base_name_coord}_z_m"] = data[col] * conversion_factor

        # Add all converted columns at once
        conv_df = pd.DataFrame(cols_to_convert)
        data = pd.concat([data, conv_df], axis=1)

        # Calculate the reference (averages of the first 10 frames)
        n_baseline_frames_start = 10
        n_baseline_frames_end = 20

        # Calculate reference for the CG (to be used as zero)
        cg_y_ref = data["cg_y_m"].iloc[n_baseline_frames_start:n_baseline_frames_end].mean()
        cg_x_ref = data["cg_x_m"].iloc[n_baseline_frames_start:n_baseline_frames_end].mean()

        # Add this block to create relative versions of all y-coordinates
        print("Creating relative coordinates referenced to initial CG position...")
        # Collect all columns in dictionaries
        rel_cols = {}

        # Y coordinates relative to CG reference
        for col in [c for c in data.columns if c.endswith("_y_m")]:
            rel_cols[f"{col}_rel"] = data[col] - cg_y_ref
            # print(f"  Created {col}_rel column")

        # X coordinates relative to CG reference
        for col in [c for c in data.columns if c.endswith("_x_m")]:
            rel_cols[f"{col}_rel"] = data[col] - cg_x_ref
            # print(f"  Created {col}_rel column")

        # Add all column
        data = pd.concat([data, pd.DataFrame(rel_cols)], axis=1)

        # Normalize CG so that the reference is zero
        data["cg_x_normalized"] = data["cg_x_m"] - cg_x_ref
        data["cg_y_normalized"] = data["cg_y_m"] - cg_y_ref

        # Calculate the reference (averages of the first 10 frames)
        n_baseline_frames_start = 10
        n_baseline_frames_end = 20

        # Calculate reference for the CG (to be used as zero)
        cg_y_ref = data["cg_y_m"].iloc[n_baseline_frames_start:n_baseline_frames_end].mean()
        cg_x_ref = data["cg_x_m"].iloc[n_baseline_frames_start:n_baseline_frames_end].mean()

        # Calculate baseline for the feet
        has_left_foot = "left_foot_index_y_m" in data.columns
        has_right_foot = "right_foot_index_y_m" in data.columns

        if has_left_foot and has_right_foot:
            feet_y_values = (data["left_foot_index_y_m"] + data["right_foot_index_y_m"]) / 2
        elif has_left_foot:
            feet_y_values = data["left_foot_index_y_m"]
        elif has_right_foot:
            feet_y_values = data["right_foot_index_y_m"]
        else:
            # Fallback to ankles
            if "left_ankle_y_m" in data.columns and "right_ankle_y_m" in data.columns:
                feet_y_values = (data["left_ankle_y_m"] + data["right_ankle_y_m"]) / 2
            elif "left_ankle_y_m" in data.columns:
                feet_y_values = data["left_ankle_y_m"]
            elif "right_ankle_y_m" in data.columns:
                feet_y_values = data["right_ankle_y_m"]
            else:
                feet_y_values = data["cg_y_m"] * 0.8  # Estimate

        feet_baseline = feet_y_values.iloc[n_baseline_frames_start:n_baseline_frames_end].mean()

        # Add basic information
        data["mass_kg"] = mass
        data["fps"] = fps
        data["reference_cg_y"] = cg_y_ref
        data["reference_cg_x"] = cg_x_ref
        data["reference_feet_y"] = feet_baseline

        # --- Identify all jump phases using the robust function ---
        jump_phase_results = identify_jump_phases(data, feet_baseline, cg_y_ref, fps)

        # Calculate power metrics first
        dt = 1 / fps
        vel_cg = np.gradient(data["cg_y_m"], dt)
        data["cg_vy"] = vel_cg
        acc_cg = np.gradient(vel_cg, dt)
        data["cg_ay"] = acc_cg
        force_vertical = mass * (acc_cg + 9.81)
        data["force_vertical"] = force_vertical
        power = force_vertical * vel_cg
        data["power"] = power

        # Calculate power metrics during propulsion
        takeoff_frame = jump_phase_results["takeoff_frame"]
        squat_frame = jump_phase_results["propulsion_start_frame"]

        if takeoff_frame is not None and squat_frame is not None:
            propulsion_frames = range(squat_frame, takeoff_frame + 1)
            power_propulsion = power[propulsion_frames]
            max_power = np.max(power) if len(power) > 0 else 0
            # Calculate average power during propulsion phase
            avg_power_propulsion = np.mean(power_propulsion) if len(power_propulsion) > 0 else 0

            # Corrected: Find the index of the maximum value in the power series
            idx_max_power = np.argmax(
                power
            )  # Now get the index of the maximum value in the power series
            time_max_power = idx_max_power / fps  # Convert the index to time

            # Calculate takeoff power
            power_takeoff = power[takeoff_frame] if takeoff_frame < len(power) else 0
        else:
            max_power = 0
            idx_max_power = 0
            time_max_power = 0
            power_takeoff = 0

        # Calculate energies
        jump_height = jump_phase_results["height_cg_method_m"]
        velocity_takeoff = calculate_velocity(jump_height) if jump_height else 0
        potential_energy = calculate_potential_energy(mass, jump_height) if jump_height else 0
        kinetic_energy = calculate_kinetic_energy(mass, velocity_takeoff)

        # Calculate average propulsion power
        propulsion_time = jump_phase_results.get("propulsion_time_s", 0)
        power_avg_propulsion = (
            (potential_energy + kinetic_energy) / propulsion_time if propulsion_time > 0 else 0
        )

        # Prepare comprehensive results
        results = {
            "mass_kg": mass,
            "fps": fps,
            "conversion_factor": round(conversion_factor, 6),
            "flight_time_s": (
                round(jump_phase_results.get("flight_time_s", 0), 3)
                if jump_phase_results.get("flight_time_s") is not None
                else None
            ),
            "velocity_m/s": (round(velocity_takeoff, 3) if velocity_takeoff is not None else None),
            "potential_energy_J": (
                round(potential_energy, 3) if potential_energy is not None else None
            ),
            "kinetic_energy_J": (round(kinetic_energy, 3) if kinetic_energy is not None else None),
            "power_takeoff_W": (round(power_takeoff, 3) if power_takeoff is not None else None),
            "power_avg_propulsion_W": (
                round(power_avg_propulsion, 3) if power_avg_propulsion is not None else None
            ),
            "power_avg_propulsion_phase_W": (
                round(avg_power_propulsion, 3) if avg_power_propulsion is not None else None
            ),
            "max_power_W": round(max_power, 3) if max_power is not None else None,
            "frame_max_power": (int(idx_max_power) if idx_max_power is not None else None),
            "time_max_power_s": (round(time_max_power, 3) if time_max_power is not None else None),
            "takeoff_frame": int(takeoff_frame) if takeoff_frame is not None else None,
            "max_height_frame": (
                int(jump_phase_results.get("max_height_frame"))
                if jump_phase_results.get("max_height_frame") is not None
                else None
            ),
            "landing_frame": (
                int(jump_phase_results.get("landing_frame"))
                if jump_phase_results.get("landing_frame") is not None
                else None
            ),
            "propulsion_time_s": (
                round(propulsion_time, 3) if propulsion_time is not None else None
            ),
            "ascent_time_s": (
                round(jump_phase_results.get("ascent_time_s", 0), 3)
                if jump_phase_results.get("ascent_time_s") is not None
                else None
            ),
            "descent_time_s": (
                round(jump_phase_results.get("descent_time_s", 0), 3)
                if jump_phase_results.get("descent_time_s") is not None
                else None
            ),
            "squat_depth_m": (
                round(jump_phase_results.get("squat_depth_m", 0), 3)
                if jump_phase_results.get("squat_depth_m") is not None
                else None
            ),
            "height_cg_method_m": (
                round(jump_phase_results.get("height_cg_method_m", 0), 3)
                if jump_phase_results.get("height_cg_method_m") is not None
                else None
            ),
            "height_flight_time_method_m": (
                round(jump_phase_results.get("height_flight_time_method_m", 0), 3)
                if jump_phase_results.get("height_flight_time_method_m") is not None
                else None
            ),
            "height_left_foot_m": (
                round(jump_phase_results.get("height_left_foot_m", 0), 3)
                if jump_phase_results.get("height_left_foot_m") is not None
                else None
            ),
            "height_right_foot_m": (
                round(jump_phase_results.get("height_right_foot_m", 0), 3)
                if jump_phase_results.get("height_right_foot_m") is not None
                else None
            ),
            "height_avg_feet_m": (
                round(jump_phase_results.get("height_avg_feet_m", 0), 3)
                if jump_phase_results.get("height_avg_feet_m") is not None
                else None
            ),
            "left_takeoff_time_s": (
                round(jump_phase_results.get("left_takeoff_time_s", 0), 3)
                if jump_phase_results.get("left_takeoff_time_s") is not None
                else None
            ),
            "right_takeoff_time_s": (
                round(jump_phase_results.get("right_takeoff_time_s", 0), 3)
                if jump_phase_results.get("right_takeoff_time_s") is not None
                else None
            ),
            "left_landing_time_s": (
                round(jump_phase_results.get("left_landing_time_s", 0), 3)
                if jump_phase_results.get("left_landing_time_s") is not None
                else None
            ),
            "right_landing_time_s": (
                round(jump_phase_results.get("right_landing_time_s", 0), 3)
                if jump_phase_results.get("right_landing_time_s") is not None
                else None
            ),
        }

        # Collect all relevant data for the database (complete CSV)
        db_row = {
            "file_name": base_name,
            "timestamp": timestamp,
            "mass_kg": mass,
            "fps": fps,
            "conversion_factor": conversion_factor,
            "shank_length_real_m": shank_length_real,
            "reference_cg_y": cg_y_ref,
            "reference_cg_x": cg_x_ref,
            "reference_feet_y": feet_baseline,
            # Frames of the CG phases
            "squat_frame": jump_phase_results.get("propulsion_start_frame"),
            "takeoff_frame": jump_phase_results.get("takeoff_frame"),
            "max_height_frame": jump_phase_results.get("max_height_frame"),
            "landing_frame": jump_phase_results.get("landing_frame"),
            # Times of the CG phases
            "squat_time_s": (
                jump_phase_results.get("propulsion_start_frame", 0) / fps
                if jump_phase_results.get("propulsion_start_frame") is not None
                else None
            ),
            "takeoff_time_s": (
                jump_phase_results.get("takeoff_frame", 0) / fps
                if jump_phase_results.get("takeoff_frame") is not None
                else None
            ),
            "max_height_time_s": (
                jump_phase_results.get("max_height_frame", 0) / fps
                if jump_phase_results.get("max_height_frame") is not None
                else None
            ),
            "landing_time_s": (
                jump_phase_results.get("landing_frame", 0) / fps
                if jump_phase_results.get("landing_frame") is not None
                else None
            ),
            # Frames and times of the feet
            "left_takeoff_frame": jump_phase_results.get("left_takeoff_frame"),
            "right_takeoff_frame": jump_phase_results.get("right_takeoff_frame"),
            "left_landing_frame": jump_phase_results.get("left_landing_frame"),
            "right_landing_frame": jump_phase_results.get("right_landing_frame"),
            "left_takeoff_time_s": jump_phase_results.get("left_takeoff_time_s"),
            "right_takeoff_time_s": jump_phase_results.get("right_takeoff_time_s"),
            "left_landing_time_s": jump_phase_results.get("left_landing_time_s"),
            "right_landing_time_s": jump_phase_results.get("right_landing_time_s"),
            # Bilateral differences
            "bilateral_takeoff_diff_s": (
                abs(
                    jump_phase_results.get("left_takeoff_time_s", 0)
                    - jump_phase_results.get("right_takeoff_time_s", 0)
                )
                if jump_phase_results.get("left_takeoff_time_s")
                and jump_phase_results.get("right_takeoff_time_s")
                else None
            ),
            "bilateral_landing_diff_s": (
                abs(
                    jump_phase_results.get("left_landing_time_s", 0)
                    - jump_phase_results.get("right_landing_time_s", 0)
                )
                if jump_phase_results.get("left_landing_time_s")
                and jump_phase_results.get("right_landing_time_s")
                else None
            ),
            "bilateral_height_diff_m": (
                abs(
                    jump_phase_results.get("height_left_foot_m", 0)
                    - jump_phase_results.get("height_right_foot_m", 0)
                )
                if jump_phase_results.get("height_left_foot_m")
                and jump_phase_results.get("height_right_foot_m")
                else None
            ),
            # Heights by different methods
            "height_cg_method_m": jump_phase_results.get("height_cg_method_m"),
            "height_flight_time_method_m": jump_phase_results.get("height_flight_time_method_m"),
            "height_left_foot_m": jump_phase_results.get("height_left_foot_m"),
            "height_right_foot_m": jump_phase_results.get("height_right_foot_m"),
            "height_avg_feet_m": jump_phase_results.get("height_avg_feet_m"),
            "squat_depth_m": jump_phase_results.get("squat_depth_m"),
            # Times of phases
            "flight_time_s": jump_phase_results.get("flight_time_s"),
            "propulsion_time_s": jump_phase_results.get("propulsion_time_s"),
            "ascent_time_s": jump_phase_results.get("ascent_time_s"),
            "descent_time_s": jump_phase_results.get("descent_time_s"),
            # Kinematics
            "velocity_takeoff_m/s": velocity_takeoff,
            # Energies
            "potential_energy_J": potential_energy,
            "kinetic_energy_J": kinetic_energy,
            "total_energy_J": (
                (potential_energy + kinetic_energy)
                if (potential_energy is not None and kinetic_energy is not None)
                else None
            ),
            # Powers
            "power_takeoff_W": power_takeoff,
            "power_avg_propulsion_W": power_avg_propulsion,
            "max_power_W": max_power,
            "frame_max_power": idx_max_power,
            "time_max_power_s": time_max_power,
            # Specific powers (per kg)
            "power_takeoff_W_per_kg": power_takeoff / mass if power_takeoff else None,
            "power_avg_propulsion_W_per_kg": (
                power_avg_propulsion / mass if power_avg_propulsion else None
            ),
            "max_power_W_per_kg": max_power / mass if max_power else None,
        }

        # Save complete database CSV
        output_db_file = os.path.join(output_dir, f"{base_name}_jump_database_{timestamp}.csv")
        pd.DataFrame([db_row]).to_csv(output_db_file, index=False, float_format="%.3f")
        print(f"Jump database row saved: {output_db_file}")

        # Generate plots
        plot_files = []

        # 1. Generate diagnostic plot
        diagnostic_plot = generate_normalized_diagnostic_plot(
            data,
            jump_phase_results.get("takeoff_frame"),
            jump_phase_results.get("landing_frame"),
            jump_phase_results.get("max_height_frame"),
            jump_phase_results.get("propulsion_start_frame"),
            fps,
            output_dir,
            base_name,
            jump_phase_results=jump_phase_results,
        )
        plot_files.append(diagnostic_plot)

        # 2. Generate jump phases analysis plot
        phases_plot = plot_jump_phases_analysis(
            data,
            takeoff_frame,
            jump_phase_results.get("max_height_frame"),
            jump_phase_results.get("landing_frame"),
            fps,
            output_dir,
            base_name,
        )
        plot_files.append(phases_plot)

        # 3. Generate CG and feet positions analysis plot
        cg_feet_plot = plot_jump_cg_feet_analysis(
            data,
            takeoff_frame,
            jump_phase_results.get("max_height_frame"),
            jump_phase_results.get("landing_frame"),
            fps,
            output_dir,
            base_name,
        )
        plot_files.append(cg_feet_plot)

        # 4. Generate other jump plots (power curves, etc.)
        other_plot_files = generate_jump_plots(data, results, output_dir, base_name)
        plot_files.extend(other_plot_files)

        # 5. Generate compact animated GIF over the jump cycle (key frames + intermediates)
        generate_jump_animation_gif(
            data,
            jump_phase_results,
            output_dir,
            base_name,
            frames_between=3,
            figsize=(5, 5),
        )

        # Save calibrated data
        orig_cols = list(data.columns)
        rel_cols = [c for c in data.columns if c.endswith("_rel")]
        norm_cols = [c for c in data.columns if "normalized" in c]
        metadata_cols = [
            "mass_kg",
            "fps",
            "reference_cg_y",
            "reference_cg_x",
            "reference_feet_y",
        ]

        final_cols = []
        for col in orig_cols:
            if col.endswith(("_x", "_y", "_z")):
                metr = (
                    f"{col[:-2]}_x_m"
                    if col.endswith("_x")
                    else f"{col[:-2]}_y_m"
                    if col.endswith("_y")
                    else f"{col[:-2]}_z_m"
                )
                if metr in data.columns:
                    final_cols.append(metr)
            else:
                if col in data.columns:
                    final_cols.append(col)

        final_cols += rel_cols + norm_cols + metadata_cols

        cg_columns = ["cg_x_normalized", "cg_y_normalized"]
        for col in cg_columns:
            if col in data.columns and col not in final_cols:
                final_cols.append(col)

        # Add power and kinematic columns
        kinematic_cols = ["cg_vy", "cg_ay", "force_vertical", "power"]
        for col in kinematic_cols:
            if col in data.columns and col not in final_cols:
                final_cols.append(col)

        calibrated_data = data[final_cols].copy()
        output_calibrated_file = os.path.join(output_dir, f"{base_name}_calibrated_{timestamp}.csv")
        calibrated_data.to_csv(output_calibrated_file, index=False, float_format="%.6f")
        print(f"Calibrated data saved: {output_calibrated_file}")

        # Generate stick figures
        stickfig_phases_file = os.path.join(
            output_dir, f"{base_name}_stickfigures_phases_{timestamp}.png"
        )
        plot_jump_stickfigures_subplot(output_calibrated_file, stickfig_phases_file)
        plot_files.append(stickfig_phases_file)

        stickfig_output_file = os.path.join(
            output_dir, f"{base_name}_stickfigures_cg_{timestamp}.png"
        )
        plot_jump_stickfigures_with_cg(output_calibrated_file, stickfig_output_file)
        plot_files.append(stickfig_output_file)

        # Calculate Kinematics (after processing and calibration)
        # Note: data should have '_m' columns by now if calibration succeeded
        kinematic_results = calculate_kinematics(data, results)
        results.update(kinematic_results)

        # Generate valgus plots (before HTML report so they're included)
        valgus_plot = plot_valgus_ratio(data, results, output_dir, base_name)
        if valgus_plot:
            plot_files.append(valgus_plot)

        valgus_event_plots = plot_valgus_event(data, results, output_dir, base_name)
        if valgus_event_plots:
            plot_files.extend(valgus_event_plots)

        # Generate FPPA time series plot
        fppa_time_series_plot = plot_fppa_time_series(data, results, output_dir, base_name)
        if fppa_time_series_plot:
            plot_files.append(fppa_time_series_plot)

        # Generate HTML report
        report_path = generate_html_report(data, results, plot_files, output_dir, base_name)

        # Save summary metrics
        output_metrics_file = os.path.join(output_dir, f"{base_name}_jump_metrics_{timestamp}.csv")
        pd.DataFrame([results]).to_csv(output_metrics_file, index=False, float_format="%.6f")

        print(f"Jump metrics saved at: {output_metrics_file}")
        print(f"Calibrated data (in meters) saved at: {output_calibrated_file}")
        print(f"Jump analysis plots saved in: {output_dir}")
        print(f"HTML report generated: {report_path}")

        # Print diagnostic information
        print("Diagnostic info:")
        print(f"  Reference CG position: {cg_y_ref:.3f} m")
        print(f"  Jump height (CG method): {jump_phase_results.get('height_cg_method_m', 0):.3f} m")
        print(f"  Flight time: {jump_phase_results.get('flight_time_s', 0):.3f} s")
        print(f"  Max power: {max_power:.1f} W")

        return True

    except Exception as e:
        print(f"Error processing {input_file} (MediaPipe): {str(e)}")
        import traceback

        traceback.print_exc()
        return False


def process_all_mediapipe_files(target_dir):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_mediapipejump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    csv_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]

    for input_file in csv_files:
        print(f"Processing MediaPipe file: {input_file}")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        per_file_dir = os.path.join(output_dir, base_name)
        os.makedirs(per_file_dir, exist_ok=True)
        process_mediapipe_data(input_file, per_file_dir)
    print("All MediaPipe files have been processed successfully.")


def process_jump_data(input_file, output_dir, use_time_of_flight):
    """
    Process the jump data from an input file and save results to the output directory.
    """
    try:
        # Read data from the input file
        data = pd.read_csv(input_file)
        results = []

        # Auto-detect column names
        columns = list(data.columns)

        # Look for appropriate columns by name pattern
        mass_col = next((col for col in columns if "mass" in col.lower()), columns[0])

        if use_time_of_flight:
            value_col = next(
                (col for col in columns if "time" in col.lower() and "contact" not in col.lower()),
                columns[1],
            )
        else:
            value_col = next(
                (col for col in columns if "height" in col.lower() or "heigth" in col.lower()),
                columns[1],
            )

        contact_col = next((col for col in columns if "contact" in col.lower()), None)

        print(
            f"Using columns: Mass={mass_col}, {'Time' if use_time_of_flight else 'Height'}={value_col}, Contact={contact_col}"
        )

        for _index, row in data.iterrows():
            # Get values using detected column names
            mass = row[mass_col]
            second_value = row[value_col]
            contact_time = row[contact_col] if contact_col else None

            # Determine height based on whether we are using time_of_flight or height directly
            if use_time_of_flight:
                # Calculate height based on time_of_flight
                time_of_flight = second_value
                height = calculate_jump_height(time_of_flight)
            else:
                # Use height directly and calculate time_of_flight from it
                height = second_value
                time_of_flight = calculate_time_of_flight(height)

            # Perform calculations
            velocity = calculate_velocity(height)
            potential_energy = calculate_potential_energy(mass, height)
            kinetic_energy = calculate_kinetic_energy(mass, velocity)
            liftoff_force = (
                calculate_liftoff_force(mass, velocity, contact_time) if contact_time else None
            )
            average_power = (
                calculate_average_power(potential_energy, contact_time) if contact_time else None
            )
            relative_power = (average_power / mass) if average_power else None
            jump_performance_index = (
                (average_power * time_of_flight) if average_power and time_of_flight else None
            )
            total_time = time_of_flight + contact_time if time_of_flight and contact_time else None

            # Append the results for each row
            results.append(
                {
                    "height_m": round(height, 3) if height is not None else None,
                    "liftoff_force_N": (
                        round(liftoff_force, 3) if liftoff_force is not None else None
                    ),
                    "velocity_m/s": (round(velocity, 3) if velocity is not None else None),
                    "potential_energy_J": (
                        round(potential_energy, 3) if potential_energy is not None else None
                    ),
                    "kinetic_energy_J": (
                        round(kinetic_energy, 3) if kinetic_energy is not None else None
                    ),
                    "average_power_W": (
                        round(average_power, 3) if average_power is not None else None
                    ),
                    "relative_power_W/kg": (
                        round(relative_power, 3) if relative_power is not None else None
                    ),
                    "jump_performance_index": (
                        round(jump_performance_index, 3)
                        if jump_performance_index is not None
                        else None
                    ),
                    "total_time_s": (round(total_time, 3) if total_time is not None else None),
                }
            )

        # Convert results to DataFrame
        results_df = pd.DataFrame(results)

        # Generate output file name with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        output_file_path = os.path.join(output_dir, f"{base_name}_vjump_{timestamp}.csv")
        results_df.to_csv(output_file_path, index=False)

        print(f"Results saved successfully at: {output_file_path}")
    except Exception as e:
        print(f"An error occurred while processing {input_file}: {str(e)}")


def calc_fator_convert_mediapipe(df, knee="right_knee", ankle="right_ankle", shank_length_real=0.4):
    # frame 0
    rkx, rky = df[f"{knee}_x"].iloc[0], df[f"{knee}_y"].iloc[0]
    rax, ray = df[f"{ankle}_x"].iloc[0], df[f"{ankle}_y"].iloc[0]
    normalized_length = np.sqrt((rkx - rax) ** 2 + (rky - ray) ** 2)
    factor = shank_length_real / normalized_length
    return factor


def calc_fator_convert_mediapipe_simple(df, shank_length_real):
    rkx, rky = df["right_knee_x"].iloc[0], df["right_knee_y"].iloc[0]
    rax, ray = df["right_ankle_x"].iloc[0], df["right_ankle_y"].iloc[0]
    normalized_length = np.sqrt((rkx - rax) ** 2 + (rky - ray) ** 2)
    factor = shank_length_real / normalized_length
    return factor


def midpoint(df, p1, p2):
    return (df[f"{p1}_x"] + df[f"{p2}_x"]) / 2, (df[f"{p1}_y"] + df[f"{p2}_y"]) / 2


def calculate_cg_frame(df, factor):
    # Locations of CGs in proportion to the segment
    locations = {
        "head": 0.5,
        "trunk": 0.5,
        "upperarm": 0.436,
        "forearm": 0.430,
        "hand": 0.506,
        "thigh": 0.433,
        "shank": 0.433,
        "foot": 0.5,
    }
    # Relative masses
    masses = {
        "head": 0.081,
        "trunk": 0.497,
        "upperarm": 0.028,
        "forearm": 0.016,
        "hand": 0.006,
        "thigh": 0.100,
        "shank": 0.047,
        "foot": 0.014,
    }
    # Note: y coordinates have already been inverted (1.0 - y) at the beginning of processing

    # Head: midpoint between eyes and shoulders
    head_prox_x, head_prox_y = midpoint(df, "left_eye", "right_eye")
    head_dist_x, head_dist_y = midpoint(df, "left_shoulder", "right_shoulder")
    cg_head_x = head_prox_x + locations["head"] * (head_dist_x - head_prox_x)
    cg_head_y = head_prox_y + locations["head"] * (head_dist_y - head_prox_y)
    # Trunk: midpoint shoulders and hips
    trunk_prox_x, trunk_prox_y = midpoint(df, "left_shoulder", "right_shoulder")
    trunk_dist_x, trunk_dist_y = midpoint(df, "left_hip", "right_hip")
    cg_trunk_x = trunk_prox_x + locations["trunk"] * (trunk_dist_x - trunk_prox_x)
    cg_trunk_y = trunk_prox_y + locations["trunk"] * (trunk_dist_y - trunk_prox_y)
    # Upper left limb segments
    cg_upperarm_l_x = df["left_shoulder_x"] + locations["upperarm"] * (
        df["left_elbow_x"] - df["left_shoulder_x"]
    )
    cg_upperarm_l_y = df["left_shoulder_y"] + locations["upperarm"] * (
        df["left_elbow_y"] - df["left_shoulder_y"]
    )
    cg_forearm_l_x = df["left_elbow_x"] + locations["forearm"] * (
        df["left_wrist_x"] - df["left_elbow_x"]
    )
    cg_forearm_l_y = df["left_elbow_y"] + locations["forearm"] * (
        df["left_wrist_y"] - df["left_elbow_y"]
    )
    cg_hand_l_x = df["left_wrist_x"] + locations["hand"] * (
        midpoint(df, "left_pinky", "left_index")[0] - df["left_wrist_x"]
    )
    cg_hand_l_y = df["left_wrist_y"] + locations["hand"] * (
        midpoint(df, "left_pinky", "left_index")[1] - df["left_wrist_y"]
    )
    # Upper right limb segments
    cg_upperarm_r_x = df["right_shoulder_x"] + locations["upperarm"] * (
        df["right_elbow_x"] - df["right_shoulder_x"]
    )
    cg_upperarm_r_y = df["right_shoulder_y"] + locations["upperarm"] * (
        df["right_elbow_y"] - df["right_shoulder_y"]
    )
    cg_forearm_r_x = df["right_elbow_x"] + locations["forearm"] * (
        df["right_wrist_x"] - df["right_elbow_x"]
    )
    cg_forearm_r_y = df["right_elbow_y"] + locations["forearm"] * (
        df["right_wrist_y"] - df["right_elbow_y"]
    )
    cg_hand_r_x = df["right_wrist_x"] + locations["hand"] * (
        midpoint(df, "right_pinky", "right_index")[0] - df["right_wrist_x"]
    )
    cg_hand_r_y = df["right_wrist_y"] + locations["hand"] * (
        midpoint(df, "right_pinky", "right_index")[1] - df["right_wrist_y"]
    )
    # Left thigh, shank, foot
    cg_thigh_l_x = df["left_hip_x"] + locations["thigh"] * (df["left_knee_x"] - df["left_hip_x"])
    cg_thigh_l_y = df["left_hip_y"] + locations["thigh"] * (df["left_knee_y"] - df["left_hip_y"])
    cg_shank_l_x = df["left_knee_x"] + locations["shank"] * (df["left_ankle_x"] - df["left_knee_x"])
    cg_shank_l_y = df["left_knee_y"] + locations["shank"] * (df["left_ankle_y"] - df["left_knee_y"])
    cg_foot_l_x = df["left_heel_x"] + locations["foot"] * (
        df["left_foot_index_x"] - df["left_heel_x"]
    )
    cg_foot_l_y = df["left_heel_y"] + locations["foot"] * (
        df["left_foot_index_y"] - df["left_heel_y"]
    )
    # Right thigh, shank, foot
    cg_thigh_r_x = df["right_hip_x"] + locations["thigh"] * (df["right_knee_x"] - df["right_hip_x"])
    cg_thigh_r_y = df["right_hip_y"] + locations["thigh"] * (df["right_knee_y"] - df["right_hip_y"])
    cg_shank_r_x = df["right_knee_x"] + locations["shank"] * (
        df["right_ankle_x"] - df["right_knee_x"]
    )
    cg_shank_r_y = df["right_knee_y"] + locations["shank"] * (
        df["right_ankle_y"] - df["right_knee_y"]
    )
    cg_foot_r_x = df["right_heel_x"] + locations["foot"] * (
        df["right_foot_index_x"] - df["right_heel_x"]
    )
    cg_foot_r_y = df["right_heel_y"] + locations["foot"] * (
        df["right_foot_index_y"] - df["right_heel_y"]
    )
    # Vectors of the coordinates of the CGs of the segments
    cg_x = (
        masses["head"] * cg_head_x
        + masses["trunk"] * cg_trunk_x
        + masses["upperarm"] * (cg_upperarm_l_x + cg_upperarm_r_x)
        + masses["forearm"] * (cg_forearm_l_x + cg_forearm_r_x)
        + masses["hand"] * (cg_hand_l_x + cg_hand_r_x)
        + masses["thigh"] * (cg_thigh_l_x + cg_thigh_r_x)
        + masses["shank"] * (cg_shank_l_x + cg_shank_r_x)
        + masses["foot"] * (cg_foot_l_x + cg_foot_r_x)
    ) / 1.0

    cg_y = (
        masses["head"] * cg_head_y
        + masses["trunk"] * cg_trunk_y
        + masses["upperarm"] * (cg_upperarm_l_y + cg_upperarm_r_y)
        + masses["forearm"] * (cg_forearm_l_y + cg_forearm_r_y)
        + masses["hand"] * (cg_hand_l_y + cg_hand_r_y)
        + masses["thigh"] * (cg_thigh_l_y + cg_thigh_r_y)
        + masses["shank"] * (cg_shank_l_y + cg_shank_r_y)
        + masses["foot"] * (cg_foot_l_y + cg_foot_r_y)
    ) / 1.0
    # Apply conversion factor to meters
    cg_x_m = cg_x * factor
    cg_y_m = cg_y * factor
    return cg_x_m, cg_y_m


def jump_height_mediapipe(df, shank_length_real):
    factor = calc_fator_convert_mediapipe_simple(df, shank_length_real)
    cg_x, cg_y = calculate_cg_frame(df, factor)
    # Now y increases upward, so the highest value is the peak!
    jump_height = cg_y.max() - cg_y.iloc[:10].mean()
    return jump_height, cg_x, cg_y


def process_all_files_in_directory(target_dir, use_time_of_flight):
    """
    Process all .csv files in the specified directory and save the results in a new output directory.

    Args:
        target_dir (str): The path to the target directory containing .csv files.
        use_time_of_flight (bool): Whether to use time of flight data.
    """
    # Generate the output directory with the current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(target_dir, f"vaila_verticaljump_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    # List all .csv files in the target directory
    csv_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]

    # Process each .csv file
    for input_file in csv_files:
        print(f"Processing file: {input_file}")
        base_name = os.path.splitext(os.path.basename(input_file))[0]
        per_file_dir = os.path.join(output_dir, base_name)
        os.makedirs(per_file_dir, exist_ok=True)
        process_jump_data(input_file, per_file_dir, use_time_of_flight)

    print("All files have been processed successfully.")


def generate_normalized_diagnostic_plot(
    data,
    takeoff_frame,
    landing_frame,
    max_height_frame,
    squat_frame,
    fps,
    output_dir,
    base_name,
    jump_phase_results=None,
):
    """
    Generate a diagnostic plot showing the normalized CG position and jump phases.
    With corrected orientation: y increases upward, so jump appears as a peak.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    fig, ax = plt.subplots(figsize=(14, 10))

    # Convert frames to time
    time = data.index / fps

    # Plot normalized CG trajectory
    ax.plot(
        time,
        data["cg_y_normalized"],
        "b-",
        linewidth=2,
        label="CG Position (normalized m)",
    )

    # Plot feet positions
    if "left_foot_index_y_m" in data.columns:
        normalized_left_foot = data["left_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_left_foot,
            "g-",
            linewidth=1,
            alpha=0.7,
            label="Left Foot (normalized)",
        )
    if "right_foot_index_y_m" in data.columns:
        normalized_right_foot = data["right_foot_index_y_m"] - data["reference_cg_y"]
        ax.plot(
            time,
            normalized_right_foot,
            "r-",
            linewidth=1,
            alpha=0.7,
            label="Right Foot (normalized)",
        )

    # Plot reference line (zero)
    ax.axhline(y=0, color="gray", linestyle="-", label="Reference Position")

    # Mark important frames
    squat_time = squat_frame / fps
    takeoff_time = takeoff_frame / fps
    landing_time = landing_frame / fps
    max_height_time = max_height_frame / fps

    ax.axvline(x=squat_time, color="brown", linestyle="-", label=f"Squat (Frame {squat_frame})")
    ax.axvline(
        x=takeoff_time,
        color="purple",
        linestyle="-",
        label=f"Takeoff (Frame {takeoff_frame})",
    )
    ax.axvline(
        x=landing_time,
        color="orange",
        linestyle="-",
        label=f"Landing (Frame {landing_frame})",
    )
    ax.axvline(
        x=max_height_time,
        color="red",
        linestyle="-",
        label=f"Max Height (Frame {max_height_frame})",
    )

    # Shade the flight phase
    ax.axvspan(
        takeoff_time,
        landing_time,
        alpha=0.2,
        color="yellow",
        label=f"Flight Phase: {landing_time - takeoff_time:.3f} s",
    )

    # Add annotations
    jump_height = data["cg_y_normalized"].iloc[max_height_frame]
    ax.annotate(
        f"Jump Height: {jump_height:.3f} m (from initial CG position)",
        xy=(max_height_time, jump_height),
        xytext=(max_height_time, jump_height * 0.8),
        arrowprops={"facecolor": "black", "shrink": 0.05},
        ha="center",
    )

    # Add frame numbers at regular intervals
    frames_to_show = np.arange(0, len(data), 20)
    # Position frame numbers at bottom of plot
    y_pos = min(data["cg_y_normalized"]) - 0.05
    for frame in frames_to_show:
        ax.text(frame / fps, y_pos, f"{frame}", fontsize=8, ha="center")

    # Set labels and title
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Position (meters from reference) - Up is positive")
    ax.set_title("Jump Analysis - Normalized CG Position (Corrected Orientation)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")

    # Save plot
    plot_path = os.path.join(output_dir, f"{base_name}_normalized_diagnostic_{timestamp}.png")
    plt.savefig(plot_path, dpi=300, bbox_inches="tight")
    plt.close()

    return plot_path


def plot_jump_stickfigures_with_cg(
    csv_file,
    output_file,
    frames_plot=None,
    labels_plot=None,
    colors=None,
    body_segments=None,
    figsize=(10, 7),
):
    """
    Plot stick figures at key jump frames and the path of the center of gravity (CG).
    Uses normalized/relative coordinates for consistency with subplot.
    """
    df = pd.read_csv(csv_file)

    # Forçar uso de coordenadas relativas e frames iguais ao subplot
    print("STICK FIGURES: Forcing use of relative coordinates and frame logic from subplot...")

    # Checa se temos coordenadas relativas
    rel_segments_available = ("left_shoulder_x_rel" in df.columns) and (
        "left_shoulder_y_rel" in df.columns
    )

    # Prioritize relative coordinates for consistency with plots
    if rel_segments_available:
        suffix = "_rel"
        possible_cg_y = [c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns]
        possible_cg_x = [c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns]
        print("Using relative coordinates (_rel) for combined visualization")
    else:
        # If no relative coordinates, create them from meter coordinates
        print("Creating relative coordinates from meter coordinates...")
        if "reference_cg_y" in df.columns and "reference_cg_x" in df.columns:
            ref_cg_y = df["reference_cg_y"].iloc[0]
            ref_cg_x = df["reference_cg_x"].iloc[0]

            # Create relative coordinates for all landmarks
            for col in df.columns:
                if col.endswith("_y_m"):
                    base_name = col[:-4]  # Remove "_y_m"
                    df[f"{base_name}_y_rel"] = df[col] - ref_cg_y
                elif col.endswith("_x_m"):
                    base_name = col[:-4]  # Remove "_x_m"
                    df[f"{base_name}_x_rel"] = df[col] - ref_cg_x

            suffix = "_rel"
            possible_cg_y = ["cg_y_normalized"]
            possible_cg_x = ["cg_x_normalized"]
        else:
            # Fallback to meter coordinates but warn user
            suffix = "_m"
            possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
            possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]
            print("WARNING: Using meter coordinates (_m) - scale may not match other plots")

    if not possible_cg_x or not possible_cg_y:
        raise ValueError("Cannot find CG coordinates in the CSV file")

    cg_x_col = possible_cg_x[0]
    cg_y_col = possible_cg_y[0]

    # Identify key frames for jump phases if not provided
    if frames_plot is None:
        frame_initial = 0
        frame_squat = df[cg_y_col].idxmin()  # Lowest position (squat)
        frame_peak = df[cg_y_col].idxmax()  # Highest position (peak)

        # Find takeoff between squat and peak, closest to baseline (0)
        squat_to_peak_range = df.loc[frame_squat:frame_peak]
        frame_takeoff = frame_squat  # Default fallback
        baseline_tolerance = 0.02

        if not squat_to_peak_range.empty:
            baseline_candidates = squat_to_peak_range[
                abs(squat_to_peak_range[cg_y_col]) <= baseline_tolerance
            ]
            if len(baseline_candidates) > 0:
                frame_takeoff = baseline_candidates.index[0]
            elif not squat_to_peak_range[cg_y_col].empty:
                frame_takeoff = (squat_to_peak_range[cg_y_col]).abs().idxmin()

        # Find landing after peak when CG returns near baseline
        post_peak_data = df[df.index > frame_peak]
        frame_landing = len(df) - 1 if not df.empty else 0  # Default fallback

        if not post_peak_data.empty:
            landing_candidates = post_peak_data[abs(post_peak_data[cg_y_col]) <= baseline_tolerance]
            if len(landing_candidates) > 0:
                frame_landing = landing_candidates.index[0]
            elif not post_peak_data[cg_y_col].empty:
                frame_landing = (post_peak_data[cg_y_col]).abs().idxmin()

        frames_plot = [
            frame_initial,
            frame_squat,
            frame_takeoff,
            frame_peak,
            frame_landing,
        ]

    if labels_plot is None:
        labels_plot = ["Initial", "Squat", "Takeoff", "Peak", "Landing"]

    if colors is None:
        colors = ["black", "red", "blue", "green", "purple"]

    if body_segments is None:
        body_segments = [
            # Legs
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"),
            ("right_knee", "right_hip"),
            # Feet
            ("left_heel", "left_foot_index"),
            ("right_heel", "right_foot_index"),
            # Pelvis & Trunk
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "left_shoulder"),
            ("right_hip", "right_shoulder"),
            # Arms
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
        ]

    # Check which segments are available with the current suffix
    available_segments = []
    for start, end in body_segments:
        x_start = f"{start}_x{suffix}"
        y_start = f"{start}_y{suffix}"
        x_end = f"{end}_x{suffix}"
        y_end = f"{end}_y{suffix}"

        if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
            available_segments.append((start, end))

    if len(available_segments) == 0:
        print(
            "WARNING: No body segments could be matched. Creating a simple plot with CG points only."
        )
    else:
        body_segments = available_segments

    # Get common y limits to ensure consistent scaling
    all_y_values = []
    all_x_values = []

    for frame in frames_plot:
        if frame >= len(df):
            print(f"WARNING: Frame {frame} exceeds DataFrame length ({len(df)})")
            continue

        row = df.iloc[frame]

        # Add CG to the limits calculation
        if cg_x_col in row and cg_y_col in row:
            all_x_values.append(float(row[cg_x_col]))
            all_y_values.append(float(row[cg_y_col]))

        # Add body segments to the limits calculation
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]):
                    all_x_values.extend([float(row[x_start]), float(row[x_end])])
                    all_y_values.extend([float(row[y_start]), float(row[y_end])])

    # Calculate plot limits with padding
    if all_y_values and all_x_values:
        y_min, y_max = min(all_y_values), max(all_y_values)
        x_min, x_max = min(all_x_values), max(all_x_values)
        y_range = y_max - y_min
        x_range = x_max - x_min
        y_padding = y_range * 0.2
        x_padding = x_range * 0.2
        y_min -= y_padding
        y_max += y_padding
        x_min -= x_padding
        x_max += x_padding

    plt.figure(figsize=figsize)

    # Plot CG trajectory
    plt.plot(df[cg_x_col], df[cg_y_col], "--", color="gray", label="CG Path", alpha=0.5)

    # Plot stick figures at key frames
    for frame, label, color in zip(frames_plot, labels_plot, colors, strict=False):
        if frame >= len(df):
            continue

        row = df.iloc[frame]

        # Draw body segments
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]):
                    plt.plot(
                        [row[x_start], row[x_end]],
                        [row[y_start], row[y_end]],
                        color=color,
                        lw=2,
                    )

        # Add neck segment from shoulders midpoint to nose
        left_shoulder_x = f"left_shoulder_x{suffix}"
        left_shoulder_y = f"left_shoulder_y{suffix}"
        right_shoulder_x = f"right_shoulder_x{suffix}"
        right_shoulder_y = f"right_shoulder_y{suffix}"
        nose_x = f"nose_x{suffix}"
        nose_y = f"nose_y{suffix}"

        if all(
            col in row.index
            for col in [
                left_shoulder_x,
                left_shoulder_y,
                right_shoulder_x,
                right_shoulder_y,
                nose_x,
                nose_y,
            ]
        ) and not any(
            pd.isna(row[col])
            for col in [
                left_shoulder_x,
                left_shoulder_y,
                right_shoulder_x,
                right_shoulder_y,
                nose_x,
                nose_y,
            ]
        ):
            # Calculate shoulders midpoint
            mid_shoulder_x = (row[left_shoulder_x] + row[right_shoulder_x]) / 2
            mid_shoulder_y = (row[left_shoulder_y] + row[right_shoulder_y]) / 2

            # Plot line from midpoint to nose
            plt.plot(
                [mid_shoulder_x, row[nose_x]],
                [mid_shoulder_y, row[nose_y]],
                color=color,
                lw=2,
            )

        # Plot CG point
        if cg_x_col in row and cg_y_col in row:
            plt.plot(
                row[cg_x_col],
                row[cg_y_col],
                "o",
                color=color,
                label=f"{label} (frame {frame})",
                markersize=10,
            )

    # Set consistent plot limits
    if all_y_values and all_x_values:
        plt.ylim(y_min, y_max)
        plt.xlim(x_min, x_max)

    plt.xlabel("X (m) - relative to initial CG")
    plt.ylabel("Y (m) - relative to initial CG")
    plt.title("Stick Figures and CG Pathway during Counter-Movement Jump")
    plt.grid(True)
    plt.legend()
    plt.axis("equal")
    plt.tight_layout()
    plt.savefig(output_file, dpi=300)
    plt.close()

    return output_file


def generate_jump_animation_gif(
    data,
    jump_phase_results,
    output_dir,
    base_name,
    frames_between: int = 3,
    figsize=(6, 6),
):
    try:
        import imageio
    except Exception:
        print("Warning: imageio not available; skipping GIF generation")
        return None

    frame_initial = 0
    frame_squat = int(jump_phase_results.get("propulsion_start_frame", 0))
    frame_takeoff = int(jump_phase_results.get("takeoff_frame", frame_squat))
    frame_peak = int(jump_phase_results.get("max_height_frame", frame_takeoff))
    frame_landing = int(jump_phase_results.get("landing_frame", len(data) - 1))

    key_frames = [frame_initial, frame_squat, frame_takeoff, frame_peak, frame_landing]
    key_frames = [int(max(0, min(len(data) - 1, f))) for f in key_frames]

    frames = []
    for a, b in zip(key_frames[:-1], key_frames[1:], strict=False):
        if a > b:
            a, b = b, a
        frames.append(a)
        if b > a:
            step = (b - a) / (frames_between + 1)
            for i in range(1, frames_between + 1):
                frames.append(int(round(a + i * step)))
    frames.append(key_frames[-1])
    frames = sorted(set(frames))

    # Use scalar references (first row) if columns exist; fall back to 0.0
    try:
        ref_cg_x = (
            float(data["reference_cg_x"].iloc[0]) if "reference_cg_x" in data.columns else 0.0
        )
    except Exception:
        ref_cg_x = 0.0
    try:
        ref_cg_y = (
            float(data["reference_cg_y"].iloc[0]) if "reference_cg_y" in data.columns else 0.0
        )
    except Exception:
        ref_cg_y = 0.0

    body_segments = [
        # Legs
        ("left_ankle", "left_knee"),
        ("left_knee", "left_hip"),
        ("right_ankle", "right_knee"),
        ("right_knee", "right_hip"),
        # Feet
        ("left_heel", "left_foot_index"),
        ("right_heel", "right_foot_index"),
        # Pelvis & Trunk
        ("left_hip", "right_hip"),
        ("left_shoulder", "right_shoulder"),
        ("left_hip", "left_shoulder"),
        ("right_hip", "right_shoulder"),
        # Arms
        ("left_shoulder", "left_elbow"),
        ("left_elbow", "left_wrist"),
        ("right_shoulder", "right_elbow"),
        ("right_elbow", "right_wrist"),
    ]

    def get_point(row, name, suffix="_m"):
        x_col = f"{name}_x{suffix}"
        y_col = f"{name}_y{suffix}"
        if (
            x_col in row.index
            and y_col in row.index
            and not (pd.isna(row[x_col]) or pd.isna(row[y_col]))
        ):
            return float(row[x_col]) - ref_cg_x, float(row[y_col]) - ref_cg_y
        return None

    # Determine axis limits across all frames
    xs, ys = [], []
    for f in frames:
        row = data.iloc[int(f)]
        for a, b in body_segments:
            pa = get_point(row, a)
            pb = get_point(row, b)
            if pa and pb:
                xs.extend([pa[0], pb[0]])
                ys.extend([pa[1], pb[1]])
    if xs and ys:
        x_min, x_max = min(xs), max(xs)
        y_min, y_max = min(ys), max(ys)
        xr = x_max - x_min
        yr = y_max - y_min
        x_pad = xr * 0.2 if xr > 0 else 0.2
        y_pad = yr * 0.2 if yr > 0 else 0.2
        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad
    else:
        x_min, x_max, y_min, y_max = -1, 1, -1, 1

    images = []
    for f in frames:
        row = data.iloc[int(f)]
        fig, ax = plt.subplots(figsize=figsize)
        for a, b in body_segments:
            pa = get_point(row, a)
            pb = get_point(row, b)
            if pa and pb:
                ax.plot([pa[0], pb[0]], [pa[1], pb[1]], color="black", lw=2)

        # Add neck line from shoulders midpoint to nose if available
        ls = get_point(row, "left_shoulder")
        rs = get_point(row, "right_shoulder")
        nose = get_point(row, "nose")
        if ls and rs and nose:
            mid = ((ls[0] + rs[0]) / 2.0, (ls[1] + rs[1]) / 2.0)
            ax.plot([mid[0], nose[0]], [mid[1], nose[1]], color="black", lw=2)

        # Draw Center of Gravity (relative to reference)
        try:
            if "cg_x_m" in data.columns and "cg_y_m" in data.columns:
                cgx = float(row["cg_x_m"]) - ref_cg_x
                cgy = float(row["cg_y_m"]) - ref_cg_y
                ax.plot(cgx, cgy, "o", color="orange", markersize=6, markeredgecolor="black")
        except Exception:
            pass

        ax.set_xlim(x_min, x_max)
        ax.set_ylim(y_min, y_max)
        ax.set_aspect("equal")
        ax.axis("off")
        # Render using Agg canvas to reliably extract pixel buffer
        try:
            from matplotlib.backends.backend_agg import (
                FigureCanvasAgg as _FigureCanvasAgg,
            )

            canvas = _FigureCanvasAgg(fig)
            canvas.draw()
            w, h = canvas.get_width_height()
            buf = canvas.buffer_rgba()
            img = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4)
            images.append(img[:, :, :3].copy())  # drop alpha
        except Exception as e:
            print(f"GIF frame render failed at frame {f}: {e}")
        plt.close(fig)

    if not images:
        print("GIF generation skipped: no images rendered")
        return None

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    gif_path = os.path.join(output_dir, f"{base_name}_jump_anim_{timestamp}.gif")
    try:
        # loop=0 makes the GIF loop infinitely
        imageio.mimsave(gif_path, images, duration=0.08, loop=0)
        print(f"Saved GIF animation: {gif_path}")
        return gif_path
    except Exception as e:
        print(f"Failed to save GIF: {e}")
        return None


def plot_jump_stickfigures_subplot(
    csv_file,
    output_file,
    frames_plot=None,
    labels_plot=None,
    colors=None,
    body_segments=None,
    figsize=(20, 7),
):
    """
    Plot a sequence of stick figures (one per subplot) with the CG at each phase.
    Uses normalized coordinates for consistency.
    """
    df = pd.read_csv(csv_file)

    # Get FPS for time calculation
    fps = float(df["fps"].iloc[0]) if "fps" in df.columns else 30.0

    # CORRECTED: Force use of relative coordinates (_rel) or normalized coordinates for consistency
    # Check if we have _rel columns for body segments
    rel_segments_available = ("left_shoulder_x_rel" in df.columns) and (
        "left_shoulder_y_rel" in df.columns
    )

    # Prioritize relative coordinates for consistency with plots
    if rel_segments_available:
        suffix = "_rel"
        possible_cg_y = [c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns]
        possible_cg_x = [c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns]
        print("Using relative coordinates (_rel) for combined visualization")
    else:
        # If no relative coordinates, create them from meter coordinates
        print("Creating relative coordinates from meter coordinates...")
        if "reference_cg_y" in df.columns and "reference_cg_x" in df.columns:
            ref_cg_y = df["reference_cg_y"].iloc[0]
            ref_cg_x = df["reference_cg_x"].iloc[0]

            # Create relative coordinates for all landmarks
            for col in df.columns:
                if col.endswith("_y_m"):
                    base_name = col[:-4]  # Remove "_y_m"
                    df[f"{base_name}_y_rel"] = df[col] - ref_cg_y
                elif col.endswith("_x_m"):
                    base_name = col[:-4]  # Remove "_x_m"
                    df[f"{base_name}_x_rel"] = df[col] - ref_cg_x

            suffix = "_rel"
            possible_cg_y = ["cg_y_normalized"]
            possible_cg_x = ["cg_x_normalized"]
        else:
            # Fallback to meter coordinates but warn user
            suffix = "_m"
            possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
            possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]
            print("WARNING: Using meter coordinates (_m) - scale may not match other plots")

    if not possible_cg_x or not possible_cg_y:
        raise ValueError("Cannot find CG coordinates in the CSV file")

    cg_x_col = possible_cg_x[0]
    cg_y_col = possible_cg_y[0]
    print(f"Using CG columns: {cg_x_col} / {cg_y_col} for stick figure plot")
    print(f"Using body segment suffix: '{suffix}'")

    # Identify key frames for jump phases if not provided
    if frames_plot is None:
        frame_initial = 0
        frame_squat = df[cg_y_col].idxmin()  # Lowest position (squat)
        frame_peak = df[cg_y_col].idxmax()  # Highest position (peak)

        # Find takeoff: first frame after squat where CG rises above initial position
        takeoff_candidates = df.index[
            (df.index > frame_squat) & (df[cg_y_col] > df[cg_y_col].iloc[frame_initial])
        ]
        frame_takeoff = takeoff_candidates.min() if len(takeoff_candidates) > 0 else frame_squat

        # Find landing: first frame after peak where CG returns near initial position
        landing_candidates = df.index[
            (df.index > frame_peak) & (df[cg_y_col] < df[cg_y_col].iloc[frame_initial] + 0.01)
        ]
        frame_landing = landing_candidates.min() if len(landing_candidates) > 0 else df.index[-1]

        frames_plot = [
            frame_initial,
            frame_squat,
            frame_takeoff,
            frame_peak,
            frame_landing,
        ]
        print(
            f"Auto-detected frames - Initial: {frame_initial}, Squat: {frame_squat}, "
            f"Takeoff: {frame_takeoff}, Peak: {frame_peak}, Landing: {frame_landing}"
        )

    if labels_plot is None:
        labels_plot = ["Initial", "Squat", "Takeoff", "Peak", "Landing"]

    if colors is None:
        colors = ["black", "red", "blue", "green", "purple"]

    if body_segments is None:
        body_segments = [
            # Legs
            ("left_ankle", "left_knee"),
            ("left_knee", "left_hip"),
            ("right_ankle", "right_knee"),
            ("right_knee", "right_hip"),
            # Feet (new)
            ("left_heel", "left_foot_index"),
            ("right_heel", "right_foot_index"),
            # Pelvis & Trunk
            ("left_hip", "right_hip"),
            ("left_shoulder", "right_shoulder"),
            ("left_hip", "left_shoulder"),
            ("right_hip", "right_shoulder"),
            # Arms
            ("left_shoulder", "left_elbow"),
            ("left_elbow", "left_wrist"),
            ("right_shoulder", "right_elbow"),
            ("right_elbow", "right_wrist"),
            # Removing direct connections to nose
            # ("left_shoulder", "nose"),
            # ("right_shoulder", "nose"),
        ]

        # Note: We'll add special handling for shoulder midpoint to nose (neck/head)

    # Check which segments are available with the current suffix
    available_segments = []
    for start, end in body_segments:
        x_start = f"{start}_x{suffix}"
        y_start = f"{start}_y{suffix}"
        x_end = f"{end}_x{suffix}"
        y_end = f"{end}_y{suffix}"

        if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
            available_segments.append((start, end))

    print(
        f"Available segments with suffix '{suffix}': {len(available_segments)} of {len(body_segments)}"
    )

    # If no segments available with current suffix, try alternative
    if len(available_segments) == 0:
        alt_suffix = "_m" if suffix == "_rel" else "_rel"
        print(f"No segments available with '{suffix}', trying '{alt_suffix}'")

        alt_available_segments = []
        for start, end in body_segments:
            x_start = f"{start}_x{alt_suffix}"
            y_start = f"{start}_y{alt_suffix}"
            x_end = f"{end}_x{alt_suffix}"
            y_end = f"{end}_y{alt_suffix}"

            if all(col in df.columns for col in [x_start, y_start, x_end, y_end]):
                alt_available_segments.append((start, end))

        print(
            f"Available segments with '{alt_suffix}': {len(alt_available_segments)} of {len(body_segments)}"
        )

        if len(alt_available_segments) > 0:
            suffix = alt_suffix
            available_segments = alt_available_segments

            # Update CG columns to match the suffix type
            if suffix == "_rel":
                possible_cg_y = [c for c in ["cg_y_normalized", "cg_y_m_rel"] if c in df.columns]
                possible_cg_x = [c for c in ["cg_x_normalized", "cg_x_m_rel"] if c in df.columns]
            else:
                possible_cg_y = [c for c in ["cg_y_m"] if c in df.columns]
                possible_cg_x = [c for c in ["cg_x_m"] if c in df.columns]

            cg_x_col = possible_cg_x[0] if possible_cg_x else cg_x_col
            cg_y_col = possible_cg_y[0] if possible_cg_y else cg_y_col
            print(f"Updated to use CG columns: {cg_x_col} / {cg_y_col}")

    # If still no segments available, report error
    if len(available_segments) == 0:
        print(
            "WARNING: No body segments could be matched. Creating a simple plot with CG points only."
        )
    else:
        body_segments = available_segments

    # Create figure with subplots - one for each phase
    n_phases = len(frames_plot)
    fig, axes = plt.subplots(1, n_phases, figsize=figsize)

    # Ensure axes is always an array (even with a single subplot)
    if n_phases == 1:
        axes = [axes]

    # Get common y limits to ensure consistent scaling
    all_y_values = []
    all_x_values = []

    for frame in frames_plot:
        if frame >= len(df):
            print(f"WARNING: Frame {frame} exceeds DataFrame length ({len(df)})")
            continue

        row = df.iloc[frame]

        # Add CG to the limits calculation
        if cg_x_col in row and cg_y_col in row:
            all_x_values.append(float(row[cg_x_col]))
            all_y_values.append(float(row[cg_y_col]))

        # Add body segments to the limits calculation
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]):
                    all_x_values.extend([float(row[x_start]), float(row[x_end])])
                    all_y_values.extend([float(row[y_start]), float(row[y_end])])

    if all_y_values and all_x_values:
        y_min, y_max = min(all_y_values), max(all_y_values)
        x_min, x_max = min(all_x_values), max(all_x_values)
        # Add padding
        y_range = y_max - y_min
        x_range = x_max - x_min
        y_padding = y_range * 0.2  # Increased padding for better visualization
        x_padding = x_range * 0.2
        y_min -= y_padding
        y_max += y_padding
        x_min -= x_padding
        x_max += x_padding

    # Plot each phase in its own subplot
    for i, (ax, frame, label, color) in enumerate(
        zip(axes, frames_plot, labels_plot, colors, strict=False)
    ):
        if frame >= len(df):
            ax.text(
                0.5,
                0.5,
                f"Frame {frame} out of range",
                ha="center",
                va="center",
                transform=ax.transAxes,
            )
            continue

        row = df.iloc[frame]

        # Draw body segments (stick figure)
        for start, end in body_segments:
            x_start = f"{start}_x{suffix}"
            y_start = f"{start}_y{suffix}"
            x_end = f"{end}_x{suffix}"
            y_end = f"{end}_y{suffix}"

            if all(col in row.index for col in [x_start, y_start, x_end, y_end]):
                if not any(pd.isna(row[col]) for col in [x_start, y_start, x_end, y_end]):
                    ax.plot(
                        [row[x_start], row[x_end]],
                        [row[y_start], row[y_end]],
                        color=color,
                        lw=2,
                    )

        # Add neck segment from shoulders midpoint to nose
        left_shoulder_x = f"left_shoulder_x{suffix}"
        left_shoulder_y = f"left_shoulder_y{suffix}"
        right_shoulder_x = f"right_shoulder_x{suffix}"
        right_shoulder_y = f"right_shoulder_y{suffix}"
        nose_x = f"nose_x{suffix}"
        nose_y = f"nose_y{suffix}"

        if all(
            col in row.index
            for col in [
                left_shoulder_x,
                left_shoulder_y,
                right_shoulder_x,
                right_shoulder_y,
                nose_x,
                nose_y,
            ]
        ) and not any(
            pd.isna(row[col])
            for col in [
                left_shoulder_x,
                left_shoulder_y,
                right_shoulder_x,
                right_shoulder_y,
                nose_x,
                nose_y,
            ]
        ):
            # Calculate shoulders midpoint
            mid_shoulder_x = (row[left_shoulder_x] + row[right_shoulder_x]) / 2
            mid_shoulder_y = (row[left_shoulder_y] + row[right_shoulder_y]) / 2

            # Plot line from midpoint to nose
            ax.plot(
                [mid_shoulder_x, row[nose_x]],
                [mid_shoulder_y, row[nose_y]],
                color=color,
                lw=2,
            )

        # Plot the CG with a distinct marker
        if cg_x_col in row and cg_y_col in row:
            cg_x = row[cg_x_col]
            cg_y = row[cg_y_col]
            if not pd.isna(cg_x) and not pd.isna(cg_y):
                ax.plot(
                    cg_x,
                    cg_y,
                    "o",
                    color="orange",
                    markersize=12,
                    markeredgecolor="black",
                    markeredgewidth=1,
                    label="CG",
                )

        # Set common limits for consistent scale across subplots
        if all_y_values and all_x_values:
            ax.set_ylim(y_min, y_max)
            ax.set_xlim(x_min, x_max)

        # Add frame info and title
        time_s = frame / fps
        ax.set_title(f"{label}\nFrame: {frame} | Time: {time_s:.3f}s", fontsize=12)
        ax.set_xlabel("X (m) - relative to initial CG")
        ax.set_aspect("equal")  # Force equal aspect ratio

        # Only add y-label to the first subplot
        if i == 0:
            ax.set_ylabel("Y (m) - relative to initial CG")

        # Add grid for better visibility
        ax.grid(True, linestyle="--", alpha=0.7)

        # Add legend only to the first subplot to avoid redundancy
        if i == 0:
            ax.legend(loc="upper right", fontsize=10)

    # Add overall title
    # Calculate jump height relative to initial position (assuming normalized or relative coords)
    try:
        jump_height = df[cg_y_col].max()
    except:
        jump_height = 0

    fig.suptitle(
        f"Vertical Jump Phases with Center of Gravity (CG) - CG Jump Height: {jump_height:.3f} m",
        fontsize=16,
    )

    # Adjust layout
    plt.tight_layout(rect=[0, 0, 1, 0.95])  # Leave space for suptitle

    # Save the figure
    plt.savefig(output_file, dpi=300)
    plt.close(fig)

    print(f"Stick figure phases visualization saved: {output_file}")
    return output_file


def vaila_and_jump():
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent.resolve()}")
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)  # Force dialogs to be on top

    target_dir = filedialog.askdirectory(
        title="Select the target directory containing .csv files", parent=root
    )
    root.lift()  # Bring window to the front

    if not target_dir:
        messagebox.showwarning("Warning", "No target directory selected.", parent=root)
        root.destroy()
        return

    # Now with option 3!
    data_type = simpledialog.askinteger(
        "Select Data Type",
        "Select the type of data in your CSV files:\n\n"
        "1. Time of Flight Data\n"
        "2. Jump Height Data\n"
        "3. MediaPipe Shank Length Data",
        parent=root,  # Set parent window
        minvalue=1,
        maxvalue=3,
    )
    root.lift()  # Bring window to the front

    if data_type is None:
        messagebox.showwarning("Warning", "No data type selected. Exiting.", parent=root)
        root.destroy()
        return

    if data_type == 1 or data_type == 2:
        use_time_of_flight = data_type == 1
        process_all_files_in_directory(target_dir, use_time_of_flight)
    elif data_type == 3:
        process_all_mediapipe_files(target_dir)

    msg = "All CSV files have been processed and results saved.\n\n"
    if data_type == 1:
        msg += "Input data type: Time of Flight\n"
    elif data_type == 2:
        msg += "Input data type: Jump Height\n"
    else:
        msg += "Input data type: MediaPipe Shank Length\n"
    msg += f"Output directory: {os.path.join(target_dir, 'vaila_verticaljump_*')}"
    messagebox.showinfo("Success", msg, parent=root)

    root.destroy()


if __name__ == "__main__":
    vaila_and_jump()
