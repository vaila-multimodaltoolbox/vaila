"""
Update: 11 January 2026
"""
import sys
from pathlib import Path

import pandas as pd

# Add vaila directory to path to import vaila_and_jump
current_dir = Path(__file__).parent.resolve()
vaila_dir = current_dir.parent.parent / "vaila"
sys.path.append(str(vaila_dir))

from vaila_and_jump import calculate_kinematics

def run_validation():
    # Load sample CSV
    csv_path = current_dir / "vaila_mediapipe" / "salto_mp_norm_savgol.csv"
    if not csv_path.exists():
        print(f"Error: CSV file not found at {csv_path}")
        return

    print(f"Loading data from {csv_path}...")
    data = pd.read_csv(csv_path)

    # Simulate basic context
    fps = 30  # Assumed for validation test
    print(f"Assumed FPS: {fps}")

    # Mock identify_jump_phases needed inputs
    # Need to calculate baseline feet/cg roughly or use dummy
    # In full script: feet_baseline, cg_y_baseline = calculate_baseline(data)
    # But calculate_baseline expects data columns which we might not have perfect names for in raw csv?
    # raw csv has: left_foot_index_y, cg_y?
    # Let's check CSV header again.
    # Header: cg_y_normalized? No.
    # Header from Step 7: "left_foot_index_y", "right_foot_index_y", "nose_y", "left_hip_y"...
    # It does NOT have "cg_y". The `vaila_and_jump.py` calculates CG.
    
    # We need to replicate CG calculation to run identify_jump_phases, OR manually specify frames.
    # Since I just want to validate kinematics Math, I can manually specify frames IF I know them,
    # OR I can mock the results dictionary.
    # Let's perform a minimal processing to get 'cg_y' and 'cg_y_normalized'
    # Reference to vaila_and_jump.py: process_all_mediapipe_files calls `calculate_center_of_gravity` (not seen in full code but implied)
    # Actually, I didn't see `calculate_center_of_gravity` in the snippets. It must be there.
    # For validation, I will just Calculate Kinematics on SPECIFIC FRAMES to assert math.
    # I don't need full phase detection if I just pick arbitrary frames to test the function.

    # Let's pick Frame 50 (arbitrary) as Squat and Frame 100 as Landing.
    # I'll create a results dict.
    results = {
        "fps": fps,
        "propulsion_start_frame": 0,  # Use frame 0 for convenience
        "landing_frame": 3,  # Use frame 3
    }

    # Calculate Kinematics
    print("\nCalculating Kinematics for Frame 0 (Squat) and Frame 3 (Landing)...")

    # Ensure data has columns we need.
    # The raw CSV has standard MP names. `calculate_kinematics` looks for `_m` or standard.
    # It handles standard.

    kinematics = calculate_kinematics(data, results)

    print("-" * 30)
    print("RESULTS:")
    print("-" * 30)
    for k, v in kinematics.items():
        if isinstance(v, list):
            print(f"{k}: [List of length {len(v)}]")
        else:
            print(f"{k}: {v}")

    # Validation checks
    print("\nValidation Analysis:")
    if "valgus_ratio_squat" in kinematics:
        print(f"Ratio Squat: {kinematics['valgus_ratio_squat']:.4f}")
        val = kinematics["valgus_ratio_squat"]
        if val > 0 and val < 2.0:
            print("  [PASS] Ratio is within reasonable physical range (0.0 - 2.0)")
        else:
            print("  [WARN] Ratio seems abnormal")

    if "knee_angle_left_squat_deg" in kinematics:

        deg = kinematics["knee_angle_left_squat_deg"]
        print(f"Knee Angle Left Squat: {deg:.2f} deg")
        if 80 < deg < 185:
            print("  [PASS] Angle is within reasonable physical range")
        else:
            print("  [WARN] Angle seems abnormal")

    if "kasr_squat" in kinematics:
        kasr = kinematics["kasr_squat"]
        print(f"KASR Squat: {kasr:.4f}")
        if 0.5 < kasr < 1.5:
            print("  [PASS] KASR is within reasonable physical range")
        else:
            print("  [WARN] KASR seems abnormal")

if __name__ == "__main__":
    run_validation()
