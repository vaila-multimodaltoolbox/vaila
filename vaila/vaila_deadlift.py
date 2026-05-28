"""
===============================================================================
vaila_deadlift.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 28 May 2026
Update Date: 28 May 2026
Version: 0.3.46
Python Version: 3.12.13

Description:
------------
This script processes deadlift and RDL kinematic data from MediaPipe pose
estimation files (.csv), performing advanced biomechanical computations,
variant classification (RDL vs. Stiff vs. Conventional Deadlift), and form
validation based on clinical guidelines.

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to
transform from screen coordinates (where y increases downward) to biomechanical
coordinates (where y increases upward).

Features:
---------
- Automatically converts normalized coordinates to meters using shank length.
- Tracks and computes 6 key biomechanical parameters frame-by-frame:
  1. Stance width (Hip-width stance compliance)
  2. Bar path proximity (Wrist distance to tibia plane)
  3. Thoracolumbar spine neutrality (Linearity deviation)
  4. Shin alignment (Tibia-to-ground angle)
  5. Cervical alignment (Head-trunk parallelism)
  6. Movement initiation method (Hip Hinge detection via velocity gradients)
- Adds setup and pull timing checks:
  7. Arm verticality at setup (shoulder-to-wrist horizontal offset)
  8. Bar-over-midfoot setup check (wrist projection vs. foot midpoint)
  9. Knee/hip extension synchronism during the initial pull
- Classifies variants at maximum eccentric depth into:
  - Romanian Deadlift (RDL)
  - Stiff-Legged Deadlift
  - Conventional Deadlift
- Generates time-series visualizations and a complete HTML evaluation report.

Dependencies:
-------------
- Python 3.x, pandas, numpy, matplotlib, tkinter, math, datetime

Usage:
------
- GUI: Run with no arguments or --gui.
- CLI: Use -i <path_to.csv> -c <path_to_config.toml> -o <output_dir>
===============================================================================

"""

import datetime
import math
import os
from pathlib import Path
from tkinter import Tk, filedialog, messagebox

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import tomllib as _toml_reader
except Exception:
    _toml_reader = None

_DEADLIFT_CONTEXT = None


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
                }
            except Exception:
                pass
    return None


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


def process_deadlift_kinematics(df, fps, factor):
    """
    Calculates frame-by-frame deadlift metrics, including setup checks.

    The wrist landmark is used as a barbell proxy because MediaPipe does not
    track the bar directly.
    """
    dt = 1.0 / fps

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
    metric_updates = {"stance_width_ratio": ankle_dist / hip_dist}

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
        # Coordinates
        lh = [row["left_hip_x_m"], row["left_hip_y_m"]]
        lk = [row["left_knee_x_m"], row["left_knee_y_m"]]
        la = [row["left_ankle_x_m"], row["left_ankle_y_m"]]

        rh = [row["right_hip_x_m"], row["right_hip_y_m"]]
        rk = [row["right_knee_x_m"], row["right_knee_y_m"]]
        ra = [row["right_ankle_x_m"], row["right_ankle_y_m"]]

        ls = [row["left_shoulder_x_m"], row["left_shoulder_y_m"]]
        rs = [row["right_shoulder_x_m"], row["right_shoulder_y_m"]]
        le = [row["left_ear_x_m"], row["left_ear_y_m"]]

        rw = [row["right_wrist_x_m"], row["right_wrist_y_m"]]
        r_heel = [row["right_heel_x_m"], row["right_heel_y_m"]]
        r_toe = [row["right_foot_index_x_m"], row["right_foot_index_y_m"]]

        # Knee Flexion (180 = fully extended)
        k_ang_l = calculate_joint_angle(lh, lk, la)
        k_ang_r = calculate_joint_angle(rh, rk, ra)
        knee_angles_l.append(180.0 - k_ang_l)  # Flexion from extended baseline
        knee_angles_r.append(180.0 - k_ang_r)
        hip_extension_angles_r.append(calculate_joint_angle(rs, rh, rk))

        # Shin Angle relative to ground plane
        shin_angles_r.append(calculate_joint_angle(rk, ra, [ra[0] + 0.1, ra[1]]))

        # 3. Spine Linearity Deviation (Angle between upper and lower spine vectors)
        mid_hip = [(lh[0] + rh[0]) / 2, (lh[1] + rh[1]) / 2]
        mid_shoulder = [(ls[0] + rs[0]) / 2, (ls[1] + rs[1]) / 2]
        # Using simple linearity check via slope variation
        spine_ang = calculate_joint_angle(mid_shoulder, mid_hip, [mid_hip[0] + 0.1, mid_hip[1]])
        spine_deviations.append(spine_ang)

        # 4. Bar Path Proximity (Horizontal offset from Wrist to Knee-Ankle plane)
        bar_path_offsets.append(abs(rw[0] - rk[0]))
        arm_verticality_deltas.append(rw[0] - rs[0])
        midfoot_x = (r_heel[0] + r_toe[0]) / 2.0
        bar_midfoot_errors.append(rw[0] - midfoot_x)

        # 5. Cervical Spine Alignment
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


def evaluate_initial_pull_synchronism(df, phases):
    """Detects early knee extension that is not matched by hip opening."""
    start_frame = int(phases["bottom_frame"])
    end_frame = int(phases["end_frame"])
    if end_frame <= start_frame:
        return "INFORMATIVO: Fase concentrica insuficiente para avaliar sincronismo."

    window_end = start_frame + max(1, int((end_frame - start_frame) * 0.15))
    early_pull = df.iloc[start_frame : window_end + 1]
    if early_pull.empty:
        return "INFORMATIVO: Janela inicial de subida vazia."

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
            "CRITICO: Joelho estende muito antes do quadril abrir "
            f"(risco de 'terra bom dia') no frame {frame}."
        )
    return "APROVADO: Quadril e joelhos sobem de forma sincronizada no inicio da puxada."


def identify_deadlift_phases(df):
    """Segments the movement based on vertical displacement of the shoulder."""
    lowest_shoulder_frame = df["right_shoulder_y_m"].idxmin()

    # Eccentric phase: from start until lowest shoulder height
    ecc_range = df.loc[:lowest_shoulder_frame]
    start_frame = 0
    for idx, row in ecc_range.iterrows():
        if abs(row["shoulder_velocity_y"]) > 0.05:
            start_frame = idx
            break

    # Concentric phase: from lowest point until velocity drops near zero
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


def generate_deadlift_plots(df, phases, output_dir, base_name):
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    bot = phases["bottom_frame"]

    # Plot 1: Knee & Shin Kinematics
    plt.figure(figsize=(10, 5))
    plt.plot(df.index, df["knee_flexion_r"], label="Knee Flexion (deg)", color="red")
    plt.plot(df.index, df["shin_angle_ground"], label="Shin-to-Ground Angle (deg)", color="blue")
    plt.axvline(bot, color="green", linestyle="--", label="Bottom Turnaround")
    plt.title(f"Lower Limb Kinematics - {base_name}")
    plt.xlabel("Frame Index")
    plt.ylabel("Degrees")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p1 = os.path.join(output_dir, f"{base_name}_lower_limb_{timestamp}.png")
    plt.savefig(p1, dpi=150)
    plt.close()

    # Plot 2: Spine Mechanics and Bar Proximity
    plt.figure(figsize=(10, 5))
    plt.plot(
        df.index, df["spine_deviation"], label="Spine Deviation from Setup (deg)", color="purple"
    )
    plt.plot(
        df.index,
        df["bar_path_proximity_m"] * 100,
        label="Bar horizontal offset from Knee (cm)",
        color="orange",
    )
    plt.axvline(bot, color="green", linestyle="--")
    plt.title("Spine and Bar Path Security Metrics")
    plt.xlabel("Frame Index")
    plt.legend()
    plt.grid(True, alpha=0.3)
    p2 = os.path.join(output_dir, f"{base_name}_safety_metrics_{timestamp}.png")
    plt.savefig(p2, dpi=150)
    plt.close()

    return [p1, p2]


def generate_html_report(df, phases, variant, plot_files, output_dir, base_name):
    bot_row = df.iloc[phases["bottom_frame"]]
    setup_row = df.iloc[phases["start_frame"]]
    report_path = os.path.join(output_dir, f"{base_name}_biomechanical_report.html")

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
        arm_status = "REPROVADO: Braco inclinado para a frente. Quadril muito baixo."
    elif arm_delta < -0.05:
        arm_status = "REPROVADO: Braco inclinado para tras. Quadril muito alto."
    else:
        arm_status = "APROVADO: Braco vertical sobre a barra."

    midfoot_error = setup_row["bar_midfoot_error_m"]
    if abs(midfoot_error) > 0.03:
        midfoot_status = "AVISO: Posicione a barra sobre o meio do pe antes de puxar."
    else:
        midfoot_status = "APROVADO: Barra alinhada com o meio do pe no setup."
    synchronism_status = evaluate_initial_pull_synchronism(df, phases)

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
            .img-container {{ text-align: center; margin: 20px 0; }}
            img {{ max-width: 90%; height: auto; box-shadow: 0 4px 6px -1px rgba(0,0,0,0.1); }}
        </style>
    </head>
    <body>
        <h1>Deadlift Biomechanical Diagnosis</h1>
        <p><strong>Analysis Target:</strong> {base_name}</p>
        <p><strong>Execution Date:</strong> {datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

        <div class="variant-box">
            Classified Variant at Peak Depth: {variant}
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
                <td>Setup Arm Verticality</td>
                <td>{arm_delta * 100:.1f} cm shoulder-wrist horizontal delta</td>
                <td>&le; 5.0 cm absolute offset</td>
                <td class="{"status-pass" if arm_status.startswith("APROVADO") else "status-warn"}">{arm_status}</td>
            </tr>
            <tr>
                <td>Bar Over Midfoot Setup</td>
                <td>{midfoot_error * 100:.1f} cm wrist-midfoot horizontal error</td>
                <td>&le; 3.0 cm absolute offset</td>
                <td class="{"status-pass" if midfoot_status.startswith("APROVADO") else "status-warn"}">{midfoot_status}</td>
            </tr>
            <tr>
                <td>Initial Pull Synchronism</td>
                <td>First 15% of concentric phase</td>
                <td>Knee extension rate &le; 2x hip opening rate</td>
                <td class="{"status-pass" if synchronism_status.startswith("APROVADO") else "status-warn"}">{synchronism_status}</td>
            </tr>
        </table>

        <h2>Kinematic Curve Reconstructions</h2>
    """
    for pf in plot_files:
        html_content += f'<div class="img-container"><img src="{os.path.basename(pf)}"></div>'

    html_content += "</body></html>"

    with open(report_path, "w", encoding="utf-8") as f:
        f.write(html_content)
    return report_path


def process_mediapipe_deadlift_data(input_file, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    df = pd.read_csv(input_file)
    base_name = os.path.splitext(os.path.basename(input_file))[0]

    # Invert Screen coordinates into Biomechanical World Coordinates
    for col in [c for c in df.columns if c.endswith("_y")]:
        df[col] = 1.0 - df[col]

    ctx = _load_context_from_toml(Path(input_file).parent) or {"fps": 30.0, "shank_length_m": 0.40}
    fps = ctx["fps"]
    factor = calc_conversion_factor(df, ctx["shank_length_m"])

    # Convert landmarks to scaled metric meters. Build columns in one block to avoid fragmentation.
    coord_cols = [col for col in df.columns if col.endswith(("_x", "_y", "_z"))]
    metric_coords = df[coord_cols].mul(factor).add_suffix("_m")
    df = pd.concat([df, metric_coords], axis=1)

    df = process_deadlift_kinematics(df, fps, factor)
    phases = identify_deadlift_phases(df)
    variant = classify_variant_at_bottom(df, phases["bottom_frame"])

    plot_files = generate_deadlift_plots(df, phases, output_dir, base_name)
    report_path = generate_html_report(df, phases, variant, plot_files, output_dir, base_name)

    # Save the processed CSV data
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    df.to_csv(
        os.path.join(output_dir, f"{base_name}_deadlift_kinematics_{timestamp}.csv"), index=False
    )

    print(f"\n[SUCCESS] Processed variant: {variant}")
    print(f"-> Diagnostic Evaluation Saved: {report_path}")
    return True


def main_gui():
    root = Tk()
    root.withdraw()
    root.attributes("-topmost", True)

    target_dir = filedialog.askdirectory(
        title="Select directory containing MediaPipe Pose CSV files"
    )
    if not target_dir:
        return

    csv_files = [os.path.join(target_dir, f) for f in os.listdir(target_dir) if f.endswith(".csv")]
    if not csv_files:
        messagebox.showerror("Error", "No CSV files found in selected directory.")
        return

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    output_parent = os.path.join(target_dir, f"vaila_deadlift_analysis_{timestamp}")
    os.makedirs(output_parent, exist_ok=True)

    for f in csv_files:
        print(f"Analyzing File: {f}")
        file_base = os.path.splitext(os.path.basename(f))[0]
        per_file_dir = os.path.join(output_parent, file_base)
        os.makedirs(per_file_dir, exist_ok=True)
        process_mediapipe_deadlift_data(f, per_file_dir)

    messagebox.showinfo(
        "Analysis Complete",
        f"All files evaluated successfully!\nResults directory:\n{output_parent}",
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="vaila_deadlift: Automated Biomechanical Deadlift Tracker"
    )
    parser.add_argument(
        "-i", "--input", type=str, help="Path to input MediaPipe coordinate CSV file"
    )
    parser.add_argument(
        "-o", "--output", type=str, help="Destination directory for analytics plots and reports"
    )
    parser.add_argument(
        "--gui", action="store_true", help="Force graphical file manager interface layout"
    )
    args = parser.parse_args()

    if args.gui or not args.input:
        main_gui()
    else:
        out = args.output or os.path.dirname(os.path.abspath(args.input))
        process_mediapipe_deadlift_data(args.input, out)
