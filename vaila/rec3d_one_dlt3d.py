"""
================================================================================
Script: rec3d_one_dlt3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.6
Created: 02 August 2025
Last Updated: 15 February 2026

================================================================================
Description
================================================================================

Batch 3D reconstruction using the Direct Linear Transformation (DLT) method with
multiple cameras. For each camera you provide:
  - One DLT3D parameter file (11 coefficients per camera, e.g. from dlt3d.py).
  - One pixel-coordinate CSV with columns: frame, p1_x, p1_y, p2_x, p2_y, ..., pN_x, pN_y.

Frames common to all pixel files are reconstructed; output is written to a
timestamped subfolder in the chosen output directory.

Output files (same base name, in the output subfolder):
  - rec3d_YYYYMMDD_HHMMSS.csv   — 3D points (frame, p1_x, p1_y, p1_z, ...)
  - rec3d_YYYYMMDD_HHMMSS.3d    — same data, duplicate copy
  - rec3d_YYYYMMDD_HHMMSS_m.c3d — C3D in meters (POINT:UNITS=m, POINT:FRAMES set)
  - rec3d_YYYYMMDD_HHMMSS_mm.c3d — C3D in millimeters (POINT:UNITS=mm)

C3D files are compatible with viewc3d, viewc3d_pyvista, readc3d_export (inspect/
convert), and standard C3D tools. They are produced via readcsv_export.auto_create_c3d_from_csv.

================================================================================
Input file formats
================================================================================

DLT3D file:
  - CSV with one row of 11 DLT coefficients (e.g. from vaila dlt3d module).
  - One file per camera; order must match the pixel file order.

Pixel CSV:
  - Header: frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y (vailá standard).
  - One file per camera; same number of markers and matching frame sets recommended.
  - Files may be in different directories (GUI: one dialog per camera).

================================================================================
Usage
================================================================================

GUI (default):
  Run with no arguments or --gui. You are prompted for:
  1) Number of cameras
  2) One DLT3D file per camera (file dialogs; may be in different folders)
  3) One pixel CSV per camera (file dialogs; may be in different folders)
  4) Output directory
  5) Data rate in Hz (e.g. 60, 100)

CLI:
  Require --dlt3d, --pixels, --output. Optional --fps. Order of files must match.
  Example:
    python -m vaila.rec3d_one_dlt3d --dlt3d c1.dlt3d c2.dlt3d --pixels c1.csv c2.csv --fps 60 -o ./out
  Help:
    python -m vaila.rec3d_one_dlt3d --help

Documentation: vaila/help/rec3d_one_dlt3d.md and rec3d_one_dlt3d.html

Related modules:
  - dlt3d.py          — compute DLT3D coefficients from calibration data
  - readcsv_export.py — CSV to C3D (used internally); batch convert
  - readc3d_export.py — C3D to CSV; inspect C3D
  - viewc3d / viewc3d_pyvista — visualize C3D files
"""

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import ezc3d
import numpy as np
import pandas as pd
from numpy.linalg import lstsq
from rich import print


def rec3d_multicam(dlt_list, pixel_list):
    """
    Reconstructs a 3D point using multiple camera observations and their corresponding DLT3D parameters.

    Args:
        dlt_list (list of np.array): List of DLT3D parameter arrays (each of 11 elements) for each camera.
        pixel_list (list of tuple): List of observed pixel coordinates (x, y) for each camera.

    Returns:
        np.array: Reconstructed 3D point [X, Y, Z] using a least squares solution.
    """
    num_cameras = len(dlt_list)
    A_matrix = np.zeros((num_cameras * 2, 3))
    b_vector = np.zeros(num_cameras * 2)

    for i, (A_params, (x, y)) in enumerate(zip(dlt_list, pixel_list, strict=False)):
        a1, a2, a3, a4, a5, a6, a7, a8, a9, a10, a11 = A_params

        # Equations for camera i:
        # (a1 - x*a9)*X + (a2 - x*a10)*Y + (a3 - x*a11)*Z = x - a4
        # (a5 - y*a9)*X + (a6 - y*a10)*Y + (a7 - y*a11)*Z = y - a8
        row_idx = i * 2
        A_matrix[row_idx] = [a1 - x * a9, a2 - x * a10, a3 - x * a11]
        A_matrix[row_idx + 1] = [a5 - y * a9, a6 - y * a10, a7 - y * a11]
        b_vector[row_idx] = x - a4
        b_vector[row_idx + 1] = y - a8

    solution, residuals, rank, s = lstsq(A_matrix, b_vector, rcond=None)
    return solution  # [X, Y, Z]


def save_rec3d_as_c3d(rec3d_df, output_dir, default_filename, point_rate=100, conversion_factor=1):
    """
    Converts the 3D reconstruction DataFrame to a C3D file and saves it.

    Args:
        rec3d_df (pd.DataFrame): DataFrame with results (columns "frame", "p1_x", "p1_y", "p1_z", ..., "p25_x", "p25_y", "p25_z").
        output_dir (str): Directory where the file will be saved.
        default_filename (str): Default name for the C3D file.
        point_rate (int): Point sampling rate (Hz).
        conversion_factor (float): Conversion factor for coordinates (if necessary).
    """
    from tkinter import filedialog, messagebox

    num_frames = rec3d_df.shape[0]
    # Define the markers based on the actual columns
    x_columns = [col for col in rec3d_df.columns if col.endswith("_x") and col.startswith("p")]
    num_markers = len(x_columns)
    marker_labels = [f"p{i}" for i in range(1, num_markers + 1)]

    # Initialize point matrix with shape (4, num_markers, num_frames)
    points_data = np.zeros((4, num_markers, num_frames))
    for i, marker in enumerate(marker_labels):
        try:
            points_data[0, i, :] = rec3d_df[f"{marker}_x"].values * conversion_factor
            points_data[1, i, :] = rec3d_df[f"{marker}_y"].values * conversion_factor
            points_data[2, i, :] = rec3d_df[f"{marker}_z"].values * conversion_factor
        except KeyError as e:
            messagebox.showerror("Error", f"Missing data for marker {marker}: {e}")
            return
    points_data[3, :, :] = 1  # Homogeneous coordinate

    c3d = ezc3d.c3d()
    # Use existing POINT structure in ezc3d (preserves __METADATA__ for write())
    units_str = "mm" if conversion_factor == 1000 else "m"
    c3d["parameters"]["POINT"]["LABELS"]["value"] = marker_labels
    c3d["parameters"]["POINT"]["RATE"]["value"] = [point_rate]
    c3d["parameters"]["POINT"]["UNITS"]["value"] = [units_str]
    c3d["parameters"]["POINT"]["FRAMES"]["value"] = [num_frames]
    c3d["data"]["points"] = points_data

    output_c3d = filedialog.asksaveasfilename(
        title="Save C3D file",
        initialdir=output_dir,
        initialfile=default_filename,
        defaultextension=".c3d",
        filetypes=[("C3D files", "*.c3d")],
    )
    if output_c3d:
        try:
            c3d.write(output_c3d)
            messagebox.showinfo("Success", f"C3D file saved at:\n{output_c3d}")
        except Exception as e:
            messagebox.showerror("Error", f"Error saving C3D file: {e}")
    else:
        messagebox.showwarning("Warning", "C3D save operation cancelled.")


def save_rec3d_as_bvh(rec3d_df, output_dir, file_base, point_rate, gui=True, swap_yz=True):
    """
    Exports reconstructed 3D data to BVH format (Biovision Hierarchy).
    Since there is no pre-defined rigid skeleton model, each marker is
    exported as an independent ROOT node in 3D space.

    Args:
        swap_yz (bool): If True, swaps Y and Z (Y_out = Z_in, Z_out = Y_in) for Z-up systems (Blender).
    """
    import os

    import numpy as np

    bvh_filepath = os.path.join(output_dir, f"{file_base}.bvh")

    # Identifies markers from DataFrame columns
    markers = []
    for col in rec3d_df.columns:
        if col.endswith("_x") and col.startswith("p"):
            markers.append(col.replace("_x", ""))

    num_frames = len(rec3d_df)
    # Protection against division by zero, if point_rate is invalid
    frame_time = 1.0 / point_rate if point_rate > 0 else 0.01

    try:
        with open(bvh_filepath, "w", encoding="utf-8") as f:
            # ==========================================
            # SEÇÃO 1: HIERARCHY
            # ==========================================
            f.write("HIERARCHY\n")
            for marker in markers:
                f.write(f"ROOT {marker}\n")
                f.write("{\n")
                f.write("\tOFFSET 0.000000 0.000000 0.000000\n")
                f.write("\tCHANNELS 3 Xposition Yposition Zposition\n")
                f.write("\tEnd Site\n")
                f.write("\t{\n")
                f.write("\t\tOFFSET 0.000000 0.000000 0.000000\n")
                f.write("\t}\n")
                f.write("}\n")

            # ==========================================
            # SECTION 2: MOTION
            # ==========================================
            f.write("MOTION\n")
            f.write(f"Frames: {num_frames}\n")
            f.write(f"Frame Time: {frame_time:.6f}\n")

            # Format coordinates frame by frame
            for index, row in rec3d_df.iterrows():
                frame_data = []
                for marker in markers:
                    x = row.get(f"{marker}_x", 0.0)
                    y = row.get(f"{marker}_y", 0.0)
                    z = row.get(f"{marker}_z", 0.0)

                    # BVH format does not accept "NaN". Replace with 0.0
                    x = 0.0 if np.isnan(x) else x
                    y = 0.0 if np.isnan(y) else y
                    z = 0.0 if np.isnan(z) else z

                    if swap_yz:
                        # AXIS SWAP FOR BLENDER (Z-up vs Y-up)
                        # DLT Z axis goes to BVH Y column, and Y axis goes to Z.
                        # Reference: X=X, Y=Z, Z=Y (or -Y if needed, but default is Y)
                        frame_data.extend([f"{x:.6f}", f"{z:.6f}", f"{y:.6f}"])
                    else:
                        frame_data.extend([f"{x:.6f}", f"{y:.6f}", f"{z:.6f}"])

                f.write(" ".join(frame_data) + "\n")

        print(f"BVH file (mocap/Blender) created successfully (Swap Y/Z: {swap_yz})")
        return bvh_filepath

    except Exception as e:
        msg = f"Failed to save BVH file: {e}"
        print(f"Error: {msg}")
        if gui:
            from tkinter import messagebox

            messagebox.showerror("Error", msg)

        return None


def generate_blender_companion_script(output_dir, file_base, skeleton_json_path=None):
    """
    Generates a Python script to be run inside Blender.
    Creates a second Armature with STICK display whose bones connect
    the imported BVH markers via constraints (Copy Location + Stretch To).
    Native Blender approach — "bones" are visible lines that follow
    the animation automatically.
    """
    import json
    import os

    # Default connections (MediaPipe 33 keypoints) used if no JSON is provided
    default_connections = [
        ["p12", "p13"],
        ["p24", "p25"],
        ["p12", "p24"],
        ["p13", "p25"],
        ["p12", "p25"],
        ["p13", "p24"],
        ["p1", "p3"],
        ["p1", "p6"],
        ["p3", "p6"],
        ["p3", "p8"],
        ["p6", "p9"],
        ["p10", "p11"],
        ["p12", "p14"],
        ["p14", "p16"],
        ["p16", "p18"],
        ["p16", "p20"],
        ["p16", "p22"],
        ["p13", "p15"],
        ["p15", "p17"],
        ["p17", "p19"],
        ["p17", "p21"],
        ["p17", "p23"],
        ["p24", "p26"],
        ["p26", "p28"],
        ["p28", "p30"],
        ["p30", "p32"],
        ["p25", "p27"],
        ["p27", "p29"],
        ["p29", "p31"],
        ["p31", "p33"],
    ]

    connections = default_connections

    if skeleton_json_path and os.path.exists(skeleton_json_path):
        try:
            with open(skeleton_json_path, encoding="utf-8") as f:
                skeleton_data = json.load(f)
                connections = skeleton_data.get("connections", default_connections)
        except Exception as e:
            print(f"Error reading skeleton JSON: {e}. Using default connections.")

    # Don't generate if no connections (shouldn't happen with default, but good check)
    if not connections:
        return None

    script_content = f"""import bpy

# =========================================================
# Script automatically generated by vaila Toolbox
# Skeleton Visualization — Armature STICK bones
# =========================================================
# How to use:
#   1. Import the .bvh file into Blender (File > Import > BVH)
#   2. Open this script in Blender's Text Editor
#   3. Click "Run Script" (Play button)
#   4. Press Space to play the animation and see the skeleton
# =========================================================

def create_skeleton_visualization():
    print("=" * 60)
    print("vaila — Skeleton Visualization (Armature STICK)")
    print("=" * 60)

    connections = {connections}

    # ----------------------------------------------------------
    # 1. Finds the imported BVH Armature
    # ----------------------------------------------------------
    bvh_armature = None
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE':
            bvh_armature = obj
            break

    if not bvh_armature:
        print("ERROR: No Armature found in scene!")
        print("Import the .bvh file first (File > Import > BVH).")
        return

    available_bones = [b.name for b in bvh_armature.data.bones]
    print(f"Armature found: '{{bvh_armature.name}}' with {{len(available_bones)}} bones")

    # ----------------------------------------------------------
    # 2. Removes previous Vaila_Skeleton (safe re-run)
    # ----------------------------------------------------------
    old_obj = bpy.data.objects.get("Vaila_Skeleton")
    if old_obj:
        bpy.data.objects.remove(old_obj, do_unlink=True)
    old_arm = bpy.data.armatures.get("Vaila_Skeleton_Data")
    if old_arm:
        bpy.data.armatures.remove(old_arm)

    # ----------------------------------------------------------
    # 3. Creates new Armature with STICK display
    # ----------------------------------------------------------
    arm_data = bpy.data.armatures.new("Vaila_Skeleton_Data")
    arm_obj = bpy.data.objects.new("Vaila_Skeleton", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)

    # Display as thin lines (STICK) and always visible in front
    arm_data.display_type = 'STICK'
    arm_obj.show_in_front = True

    # ----------------------------------------------------------
    # 4. Enters Edit mode and creates connection bones
    # ----------------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    valid_connections = []
    for idx, (start_name, end_name) in enumerate(connections):
        if start_name not in available_bones or end_name not in available_bones:
            print(f"  Skipping: {{start_name}} -> {{end_name}} (not found in Armature)")
            continue

        bone_name = f"link_{{start_name}}_{{end_name}}"
        bone = arm_data.edit_bones.new(bone_name)
        # Temporary position — will be overwritten by constraints
        bone.head = (0.0, 0.0, idx * 0.001)
        bone.tail = (0.0, 0.1, idx * 0.001)
        bone.use_connect = False
        valid_connections.append((start_name, end_name, bone_name))

    print(f"Bones created: {{len(valid_connections)}} connections")

    # ----------------------------------------------------------
    # 5. Enters Pose mode and adds constraints
    # ----------------------------------------------------------
    bpy.ops.object.mode_set(mode='POSE')

    for start_name, end_name, bone_name in valid_connections:
        pbone = arm_obj.pose.bones[bone_name]

        # Constraint 1: copy location of start marker
        cloc = pbone.constraints.new('COPY_LOCATION')
        cloc.target = bvh_armature
        cloc.subtarget = start_name

        # Constraint 2: stretch to end marker
        stretch = pbone.constraints.new('STRETCH_TO')
        stretch.target = bvh_armature
        stretch.subtarget = end_name
        try:
            stretch.volume = 'NONE'
        except TypeError:
            stretch.volume = 'NO_VOLUME'

        # Green color (neon) for the bone — Blender 4.0+
        try:
            pbone.color.palette = 'CUSTOM'
            pbone.color.custom.normal = (0.0, 1.0, 0.3)
            pbone.color.custom.select = (1.0, 1.0, 0.0)
            pbone.color.custom.active = (1.0, 0.5, 0.0)
        except Exception:
            pass

    bpy.ops.object.mode_set(mode='OBJECT')

    # ----------------------------------------------------------
    # 6. Forces scene update
    # ----------------------------------------------------------
    bpy.context.view_layer.update()

    print("=" * 60)
    print(f"Done! {{len(valid_connections)}} skeleton connections created.")
    print("Press SPACE to play animation and see the skeleton.")
    print("=" * 60)

create_skeleton_visualization()
"""

    script_filename = f"{file_base}_blender_skeleton_viz.py"
    script_path = os.path.join(output_dir, script_filename)

    try:
        with open(script_path, "w", encoding="utf-8") as f:
            f.write(script_content)
        print(f"Blender companion script created: {script_filename}")
        return script_path
    except Exception as e:
        print(f"Failed to create Blender script: {e}")
        return None


def run_reconstruction(
    dlt_files,
    pixel_files,
    output_directory,
    point_rate,
    gui=True,
    swap_yz=True,
    skeleton_json_path=None,
):
    """
    Run 3D reconstruction from DLT3D and pixel CSV paths. Used by both GUI and CLI.

    Args:
        dlt_files: list of paths to DLT3D parameter files (one per camera)
        pixel_files: list of paths to pixel coordinate CSV files (one per camera)
        output_directory: directory where output subdir and files will be written
        point_rate: point data rate in Hz (e.g. 60, 100)
        gui: if True use messagebox for errors/success; if False use print only
        swap_yz: if True, swap Y and Z axes in BVH export (for Blender)
        skeleton_json_path: optional path to JSON file defining skeleton connections

    Returns:
        (new_dir, file_base) on success, None on failure.
    """

    def _err(msg):
        if gui:
            messagebox.showerror("Error", msg)
        else:
            print(f"Error: {msg}")
        return None

    # Load DLT3D parameters for each camera
    print("Loading DLT3D calibration parameters...")
    dlt_params_list = []
    for file in dlt_files:
        df = pd.read_csv(file)
        if df.empty:
            return _err(f"DLT3D file {os.path.basename(file)} is empty!")
        params = df.iloc[0, 1:].to_numpy().astype(float)
        dlt_params_list.append(params)

    # Load pixel coordinate data for each camera
    print("Loading pixel coordinate data...")
    pixel_dfs = []
    for file in pixel_files:
        df = pd.read_csv(file)
        required_cols = {"frame", "p1_x", "p1_y"}
        if not required_cols.issubset(df.columns):
            return _err(
                f"Pixel coordinate file {os.path.basename(file)} does not contain required columns 'frame', 'p1_x', and 'p1_y'."
            )
        if df.shape[0] == 1:
            df["frame"] = 0
        pixel_dfs.append(df)

    first_df = pixel_dfs[0]
    x_columns = [col for col in first_df.columns if col.endswith("_x") and col.startswith("p")]
    num_markers = len(x_columns)
    print(f"Detected {num_markers} markers for 3D reconstruction")

    frame_sets = [set(df["frame"]) for df in pixel_dfs]
    common_frames = set.intersection(*frame_sets)
    if not common_frames:
        return _err("No common frames found among pixel files!")
    common_frames = sorted(common_frames)
    print(f"Processing {len(common_frames)} common frames...")

    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)
    print(f"Pre-allocating array for {total_frames} frames x {num_markers} markers...")
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    reconstruction_array[:, 0] = common_frames

    progress_step = max(1, total_frames // 20)
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        frame_data = []
        valid_frame = True
        for df in pixel_dfs:
            frame_row = df[df["frame"] == frame]
            if frame_row.empty:
                valid_frame = False
                break
            frame_data.append(frame_row.iloc[0])

        if not valid_frame:
            continue

        for marker in range(1, num_markers + 1):
            pixel_obs_list = []
            valid_marker = True
            for frame_row in frame_data:
                try:
                    x_obs = float(frame_row[f"p{marker}_x"])
                    y_obs = float(frame_row[f"p{marker}_y"])
                    if np.isnan(x_obs) or np.isnan(y_obs):
                        valid_marker = False
                        break
                    pixel_obs_list.append((x_obs, y_obs))
                except Exception:
                    valid_marker = False
                    break

            col_start = 1 + (marker - 1) * 3
            if not valid_marker or len(pixel_obs_list) != len(dlt_params_list):
                pass
            else:
                point3d = rec3d_multicam(dlt_params_list, pixel_obs_list)
                reconstruction_array[frame_idx, col_start : col_start + 3] = point3d

    print("3D reconstruction completed!")

    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    rec3d_df = pd.DataFrame(reconstruction_array, columns=header)
    valid_frames_mask = ~rec3d_df.iloc[:, 1:].isna().all(axis=1)
    rec3d_df = rec3d_df[valid_frames_mask].reset_index(drop=True)

    if rec3d_df.empty:
        return _err("No valid 3D reconstruction could be performed!")

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(new_dir, exist_ok=True)

    file_base = f"rec3d_{timestamp}"
    file_3d_path = os.path.join(new_dir, f"{file_base}.3d")
    file_csv_path = os.path.join(new_dir, f"{file_base}.csv")

    print("Saving 3D reconstruction results...")
    rec3d_df.to_csv(file_3d_path, index=False, float_format="%.6f")
    rec3d_df.to_csv(file_csv_path, index=False, float_format="%.6f")

    rec3d_df_for_c3d = rec3d_df.copy()
    new_columns = []
    for col in rec3d_df_for_c3d.columns:
        if col.lower() != "frame":
            parts = col.split("_")
            if len(parts) == 2:
                new_columns.append(parts[0] + "_" + parts[1].upper())
            else:
                new_columns.append(col)
        else:
            new_columns.append(col)
    rec3d_df_for_c3d.columns = new_columns

    m_conversion = 1
    mm_conversion = 1000

    import vaila.readcsv_export as readcsv_export

    c3d_output_path_m = os.path.join(new_dir, f"{file_base}_m.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_m,
            point_rate=point_rate,
            conversion_factor=m_conversion,
        )
        if gui:
            messagebox.showinfo("Success", f"C3D file (meters) saved at:\n{c3d_output_path_m}")
        print("C3D file (meters) created successfully")
    except Exception as e:
        if gui:
            messagebox.showerror("Error", f"Failed to save C3D file (meters): {e}")
        print("Error saving C3D file (meters):", e)
        return None

    c3d_output_path_mm = os.path.join(new_dir, f"{file_base}_mm.c3d")
    try:
        readcsv_export.auto_create_c3d_from_csv(
            rec3d_df_for_c3d,
            c3d_output_path_mm,
            point_rate=point_rate,
            conversion_factor=mm_conversion,
        )
        if gui:
            messagebox.showinfo(
                "Success", f"C3D file (millimeters) saved at:\n{c3d_output_path_mm}"
            )
        print("C3D file (millimeters) created successfully")
    except Exception as e:
        if gui:
            messagebox.showerror("Error", f"Failed to save C3D file (millimeters): {e}")
        print("Error saving C3D file (millimeters):", e)
        return None

    # ---> NEW: Call to save BVH file <---
    save_rec3d_as_bvh(rec3d_df, new_dir, file_base, point_rate, gui=gui, swap_yz=swap_yz)

    print("\n=== Processing Complete ===")
    print(f"Processed {len(common_frames)} frames with {num_markers} markers")
    print(f"Output directory: {new_dir}")
    print("Files created:")
    print(f"  - {file_base}.csv (CSV format)")
    print(f"  - {file_base}.3d (3D format)")
    print(f"  - {file_base}_m.c3d (C3D in meters)")
    print(f"  - {file_base}_mm.c3d (C3D in millimeters)")
    msg_bvh = f"  - {file_base}.bvh (Mocap format for Blender"
    if swap_yz:
        msg_bvh += ", axes swapped Y<->Z)"
    else:
        msg_bvh += ")"
    print(msg_bvh)

    # ---> NEW: Companion Script for Blender <---
    # Always attempt to generate (will use default Body-33 connections if path is None)
    blender_script = generate_blender_companion_script(new_dir, file_base, skeleton_json_path)
    if blender_script:
        print(f"  - {os.path.basename(blender_script)} (Run this in Blender to visualize skeleton)")

    if gui:
        msg_bvh_gui = "• BVH file (natively opens in Blender"
        if swap_yz:
            msg_bvh_gui += ", axes Y<->Z)"
        else:
            msg_bvh_gui += ")"

        extra_msg = ""
        if blender_script:
            extra_msg = "\n• Blender visualization script generated!"

        messagebox.showinfo(
            "Processing Complete",
            f"3D reconstruction completed successfully!\n\n"
            f"Processed: {len(common_frames)} frames with {num_markers} markers\n"
            f"Output directory: {os.path.basename(new_dir)}\n\n"
            f"Files created:\n"
            f"• CSV and 3D format files\n"
            f"• C3D files (meters and millimeters)\n"
            f"{msg_bvh_gui}{extra_msg}",
        )

    return (new_dir, file_base)


def run_rec3d_one_dlt3d():
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d_one_dlt3d.py...")
    print("-" * 80)

    # --- New changes: Save files in a user-selected directory ---
    root = Tk()
    root.withdraw()

    # Step 0: Ask number of cameras (allows selecting files from different directories)
    print("Step 0: Number of cameras...")
    n_cameras = simpledialog.askinteger(
        "Number of cameras",
        "Enter number of cameras (e.g. 2):",
        minvalue=1,
        maxvalue=20,
        initialvalue=2,
    )
    if n_cameras is None:
        messagebox.showwarning("Cancelled", "Operation cancelled.")
        root.destroy()
        return

    # Step 1: Select DLT3D parameter files (one dialog per camera so each can be from a different directory)
    print("Step 1: Selecting DLT3D parameter files...")
    dlt_files = []
    for i in range(1, n_cameras + 1):
        path = filedialog.askopenfilename(
            title=f"Select DLT3D file for camera {i}",
            filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
        )
        if not path:
            messagebox.showerror("Error", f"No DLT3D file selected for camera {i}!")
            root.destroy()
            return
        dlt_files.append(path)

    # Step 2: Select pixel coordinate CSV files (one dialog per camera)
    print("Step 2: Selecting pixel coordinate CSV files...")
    pixel_files = []
    for i in range(1, n_cameras + 1):
        path = filedialog.askopenfilename(
            title=f"Select pixel coordinate CSV for camera {i}",
            filetypes=[("CSV files", "*.csv")],
        )
        if not path:
            messagebox.showerror("Error", f"No pixel coordinate file selected for camera {i}!")
            root.destroy()
            return
        pixel_files.append(path)

    if len(dlt_files) != len(pixel_files):
        messagebox.showerror(
            "Error",
            "The number of DLT3D files must match the number of pixel coordinate files!",
        )
        root.destroy()
        return

    # Step 3: Select output directory
    print("Step 3: Selecting output directory...")
    output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
    if not output_directory:
        messagebox.showerror("Error", "No output directory selected. Operation cancelled.")
        return

    # Step 4: Ask for data frequency
    print("Step 4: Setting data frequency...")
    point_rate = simpledialog.askinteger(
        "Data Frequency",
        "Enter the point data rate (Hz):",
        minvalue=1,
        initialvalue=100,
    )
    if point_rate is None:
        messagebox.showerror("Error", "Point data rate is required. Operation cancelled.")
        return

    # Step 5: Ask if user wants to swap Y and Z axes for Blender
    swap_yz = messagebox.askyesno(
        "BVH Axis Export",
        "Do you want to swap Y and Z axes for the BVH file?\n\n"
        "Select YES if you plan to open this in Blender (Z-up).\n"
        "Select NO to keep original DLT coordinates.",
    )

    # Step 6: (Optional) Select Skeleton Pose JSON
    print("Step 6: (Optional) Selecting Skeleton Pose JSON for Blender visualization...")
    skeleton_json_path = None
    use_skeleton = messagebox.askyesno(
        "Skeleton Visualization",
        "Do you have a Skeleton Pose JSON file (e.g. MediaPipe)?\n"
        "This allows generating a script to visualize connections in Blender.",
    )
    if use_skeleton:
        skeleton_json_path = filedialog.askopenfilename(
            title="Select Skeleton Pose JSON",
            filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
        )

    # Configuration summary
    print("Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Pixel files: {len(pixel_files)} cameras")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {point_rate} Hz")
    print(f"  - Swap Y/Z for BVH: {swap_yz}")
    print(
        f"  - Skeleton JSON: {os.path.basename(skeleton_json_path) if skeleton_json_path else 'None'}"
    )
    print("-" * 80)

    run_reconstruction(
        dlt_files,
        pixel_files,
        output_directory,
        point_rate,
        gui=True,
        swap_yz=swap_yz,
        skeleton_json_path=skeleton_json_path,
    )
    root.destroy()


def _cli_run():
    """CLI entry: argparse for --dlt3d, --pixels, --fps, --output."""
    parser = argparse.ArgumentParser(
        description=(
            "Batch 3D reconstruction from DLT3D parameter files and pixel CSV files "
            "(one file per camera). Output: CSV, .3d, and C3D files (meters and mm) in a "
            "timestamped subfolder under the given output directory. "
            "Without --dlt3d/--pixels/--output, or with --gui, launches the GUI."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Input:
  DLT3D files: one per camera (CSV with 11 DLT coefficients; e.g. from dlt3d module).
  Pixel files:  one per camera (CSV with header frame,p1_x,p1_y,p2_x,p2_y,...).
  Order must match: first DLT3D with first pixel file, etc.

Output:
  A new subfolder is created under DIR with name rec3d_YYYYMMDD_HHMMSS containing:
  rec3d_*.csv, rec3d_*.3d, rec3d_*_m.c3d, rec3d_*_mm.c3d.

Examples:
  %(prog)s --dlt3d cam1.dlt3d cam2.dlt3d --pixels cam1.csv cam2.csv --fps 60 -o ./out
  %(prog)s -o ./results --dlt3d a.dlt3d b.dlt3d --pixels a.csv b.csv
  %(prog)s --gui

See also: vaila/help/rec3d_one_dlt3d.md
        """,
    )
    parser.add_argument(
        "--dlt3d",
        nargs="+",
        metavar="FILE",
        help="DLT3D parameter files, one per camera (order must match --pixels)",
    )
    parser.add_argument(
        "--pixels",
        nargs="+",
        metavar="FILE",
        help="Pixel coordinate CSV files, one per camera (order must match --dlt3d)",
    )
    parser.add_argument(
        "--fps",
        type=int,
        default=100,
        metavar="HZ",
        help="Point data rate in Hz for C3D/CSV (default: 100)",
    )
    parser.add_argument(
        "-o",
        "--output",
        metavar="DIR",
        dest="output",
        help="Output directory; a timestamped subfolder will be created here",
    )
    parser.add_argument(
        "--gui",
        action="store_true",
        help="Launch GUI (file dialogs) instead of CLI",
    )
    parser.add_argument(
        "--swap-yz",
        action="store_true",
        help="Swap Y and Z axes in BVH output (optimized for Blender Z-up)",
    )
    parser.add_argument(
        "--skeleton",
        metavar="FILE",
        help="Path to Skeleton Pose JSON file (defines connections for Blender visualization)",
    )
    args = parser.parse_args()

    if args.gui or (not args.dlt3d and not args.pixels and not args.output):
        run_rec3d_one_dlt3d()
        return

    if not args.dlt3d or not args.pixels:
        # Check if user only provided --gui (already handled) but maybe they provided partial args
        if args.gui:
            run_rec3d_one_dlt3d()
            return
        print("Error: CLI mode requires --dlt3d and --pixels.", file=sys.stderr)
        sys.exit(1)
    if not args.output:
        print("Error: CLI mode requires --output.", file=sys.stderr)
        sys.exit(1)
    if len(args.dlt3d) != len(args.pixels):
        print(
            "Error: Number of --dlt3d files must match number of --pixels files.",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run_reconstruction(
        args.dlt3d,
        args.pixels,
        os.path.abspath(args.output),
        args.fps,
        gui=False,
        swap_yz=args.swap_yz,
        skeleton_json_path=args.skeleton,
    )
    if result is None:
        sys.exit(1)
    print(f"Output directory: {result[0]}")


if __name__ == "__main__":
    _cli_run()
