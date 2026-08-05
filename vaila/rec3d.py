"""
================================================================================
Script: rec3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.3.99
Created: August 03, 2025
Last Updated: 05 August 2026

Description:
    Batch 3D reconstruction using per-frame Direct Linear Transformation (DLT3D)
    parameters — i.e. a DLT "matrix" that can change frame by frame (moving or
    re-calibrated cameras), as opposed to rec3d_one_dlt3d.py which uses one
    fixed set of DLT3D parameters per camera for the whole clip.

    For each camera you provide:
      - One DLT3D parameter file with ONE ROW PER FRAME (frame, 11 coefficients).
      - One pixel-coordinate CSV, in the same directory as the other cameras'
        pixel files, with the same number of rows as the other cameras.

    All camera pixel files must be placed together in a single input directory
    (one CSV per camera) so they can be correlated frame-by-frame; the files are
    paired with --dlt-files by sorted filename order — same convention used to
    pair --dlt3d/--pixels in rec3d_one_dlt3d.py, just via a directory instead of
    an explicit file list.

    Column headers are NOT required to follow any particular naming: only the
    COLUMN ORDER matters — column 0 is the frame identifier and every pair of
    columns after that is one marker's (x, y). This makes the pixel files
    compatible with trackers that use different label conventions (vailá
    p1_x/p1_y, SAM3, YOLO, MediaPipe named joints, etc.) as long as the same
    markers appear in the same order in every camera's file.

    Output: one reconstructed 3D result (CSV + .3d + BVH + a Blender
    companion script) in a timestamped vaila_rec3d_<timestamp>/ subfolder —
    not one output per input file.

    Blender alignment (v0.3.99): --swap-yz is now the DEFAULT (use
    --no-swap-yz to keep raw DLT axes), so height ends up vertical (Z-up) in
    Blender. The generated companion script also sets the scene rate and
    frame range explicitly, because Blender's BVH importer defaults to
    update_scene_fps=False / update_scene_duration=False — without that, a
    631-frame 120 Hz capture lands in a 24 fps scene ending at frame 250 and
    plays in slow motion, truncated, while an imported C3D plays correctly.
"""

import argparse
import os
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

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


def load_pixel_csv_positional(file_path):
    """
    Load a pixel-coordinate CSV using COLUMN ORDER instead of column names.

    Column 0 (whatever it is named) is treated as the frame identifier; every
    pair of columns after that is one marker's (x, y) pixel position. A single
    header row is still expected (and discarded) so the file remains a normal
    CSV, but its label text is never inspected.

    Args:
        file_path (str): Path to the pixel CSV file.

    Returns:
        tuple[np.ndarray, np.ndarray]: (frame_values shape (n_rows,),
        xy shape (n_rows, num_markers, 2)).

    Raises:
        ValueError: if the file has fewer than 3 columns, or the number of
        coordinate columns (all columns after the first) is not even.
    """
    df = pd.read_csv(file_path)
    n_cols = df.shape[1]
    if n_cols < 3:
        raise ValueError(f"expected at least 3 columns (frame + one marker x,y), found {n_cols}")
    n_coord_cols = n_cols - 1
    if n_coord_cols % 2 != 0:
        raise ValueError(
            f"expected an even number of coordinate columns after the frame column, "
            f"found {n_coord_cols} (columns must be frame, x1, y1, x2, y2, ...)"
        )

    values = df.to_numpy(dtype=np.float64)
    frame_values = values[:, 0]
    if len(frame_values) == 1:
        # Single-row files (e.g. a single calibration/reference frame) are
        # conventionally treated as frame 0.
        frame_values = np.array([0.0])

    num_markers = n_coord_cols // 2
    xy = values[:, 1:].reshape(-1, num_markers, 2)
    return frame_values, xy


def find_common_frames(frame_arrays):
    """Sorted intersection of frame numbers present in every array (as ints)."""
    frame_sets = [{int(f) for f in arr} for arr in frame_arrays]
    common = set.intersection(*frame_sets) if frame_sets else set()
    return np.array(sorted(common), dtype=np.int64)


def _write_rec3d_output(rec_coords_df, out_path):
    """Write frame as integer, coordinates as float with 6-decimal precision."""
    df_to_save = rec_coords_df.copy()
    if "frame" in df_to_save.columns:
        df_to_save["frame"] = df_to_save["frame"].astype(int)
    df_to_save.to_csv(out_path, index=False, float_format="%.6f")


def save_rec3d_as_bvh(rec3d_df, output_dir, file_base, point_rate, gui=True, swap_yz=True):
    """
    Exports reconstructed 3D data to BVH format (Biovision Hierarchy).
    Since there is no pre-defined rigid skeleton model, each marker is
    exported as an independent ROOT node in 3D space.

    Shared by rec3d.py (per-frame-varying DLT3D) and rec3d_one_dlt3d.py
    (one fixed DLT3D per camera) — both write the same p1_x,p1_y,p1_z,...
    wide column convention, so this needs no per-caller variant.

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

    # BVH has no invalid-sample convention (unlike C3D's negative residual),
    # so an occluded marker has to be given SOME coordinate. Writing 0.0
    # teleports that joint to the world origin for the duration of the gap,
    # which reads in Blender as the skeleton violently spiking to the floor
    # centre. Linearly interpolate interior gaps and hold the nearest valid
    # sample at the head/tail instead, so occlusions degrade into a brief
    # freeze rather than a spike. Markers that are NaN for the whole trial
    # have nothing to fill from and still land on the origin.
    marker_axis_cols = [f"{m}_{axis}" for m in markers for axis in ("x", "y", "z")]
    marker_axis_cols = [c for c in marker_axis_cols if c in rec3d_df.columns]
    filled_df = rec3d_df.copy()
    if marker_axis_cols and num_frames > 1:
        n_gaps = int(filled_df[marker_axis_cols].isna().sum().sum())
        if n_gaps:
            filled_df[marker_axis_cols] = (
                filled_df[marker_axis_cols]
                .interpolate(method="linear", axis=0, limit_direction="both")
                .astype(float)
            )
            n_left = int(filled_df[marker_axis_cols].isna().sum().sum())
            print(
                f"BVH: gap-filled {n_gaps - n_left} occluded coordinate sample(s) "
                f"by interpolation"
                + (f"; {n_left} remain (marker never seen) and stay at origin" if n_left else "")
            )
    rec3d_df = filled_df

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
            # 9 decimals, not the conventional 6: at 119.88012001 Hz (NTSC
            # 120000/1001) a 6-decimal frame time of 0.008333 reads back as
            # 119.875330 Hz, so Blender's BVH importer would set a subtly
            # wrong scene rate when the file is imported on its own.
            f.write(f"Frame Time: {frame_time:.9f}\n")

            # Format coordinates frame by frame
            for _index, row in rec3d_df.iterrows():
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
                        # AXIS CONVERSION FOR BLENDER (DLT Z-up -> BVH Y-up)
                        # (x, y, z) -> (x, z, -y). Blender's BVH importer
                        # rotates the Y-up file back to Z-up as (X, Y, Z) ->
                        # (X, -Z, Y), so this round-trips to the original
                        # (x, y, z) in the scene.
                        #
                        # The negation matters: it makes this a proper
                        # ROTATION (-90 deg about X, determinant +1). Writing
                        # (x, z, y) instead is a REFLECTION, and combined with
                        # the importer's rotation it left the subject mirrored
                        # in Blender -- anatomical left and right swapped,
                        # which inverts any asymmetry read off the animation.
                        frame_data.extend([f"{x:.6f}", f"{z:.6f}", f"{-y:.6f}"])
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


def blender_scene_fps(point_rate):
    """
    Split a (possibly fractional) capture rate into Blender's integer
    ``fps`` / float ``fps_base`` pair, whose effective rate is fps / fps_base.

    Using ``fps = round(rate)`` and ``fps_base = round(rate) / rate`` makes
    the effective rate exactly ``rate`` again, and reproduces the standard
    NTSC convention for the rates that matter here:
      120            -> (120, 1.0)
      119.88012001   -> (120, 1.001)      i.e. 120000/1001
      100            -> (100, 1.0)
      29.97002997    -> (30,  1.001)
    """
    try:
        rate = float(point_rate)
    except (TypeError, ValueError):
        rate = 0.0
    if not rate > 0:
        return 30, 1.0
    fps_int = max(1, int(round(rate)))
    return fps_int, fps_int / rate


def find_unreconstructed_markers(rec3d_df):
    """Marker names (``"p42"``) that are NaN for every frame of the trial.

    These never resolved in two cameras at once — commonly finger keypoints
    on markerless input. They carry no position at all, so the BVH has to
    leave them on the world origin and any bone drawn to them would streak
    across the scene.
    """
    markers = [c[:-2] for c in rec3d_df.columns if c.endswith("_x") and c.startswith("p")]
    missing = []
    for marker in markers:
        axis_cols = [f"{marker}_{a}" for a in ("x", "y", "z")]
        axis_cols = [c for c in axis_cols if c in rec3d_df.columns]
        if axis_cols and rec3d_df[axis_cols].isna().all().all():
            missing.append(marker)
    return missing


def generate_blender_companion_script(
    output_dir,
    file_base,
    skeleton_json_path=None,
    point_rate=None,
    n_frames=None,
    mesh_dir=None,
    unreconstructed_markers=None,
):
    """
    Generates a Python script to be run inside Blender that imports this
    reconstruction **already aligned** and draws the skeleton bones.

    Why the script does the importing instead of just drawing bones: Blender's
    BVH importer defaults to ``update_scene_fps=False`` and
    ``update_scene_duration=False``, so a File > Import > BVH of a 631-frame /
    120 Hz capture lands in a scene still set to 24 fps with frame_end=250 —
    the animation plays in slow motion (26 s instead of 5.3 s) and stops a
    third of the way through, while an imported C3D (whose importer *does*
    read POINT:RATE) plays correctly. The BVH/OBJ data itself is fine; only
    the scene settings are wrong. This script sets the scene rate and frame
    range explicitly, so BVH, OBJ mesh sequence and C3D all line up.

    Args:
        output_dir: this run's timestamped output directory.
        file_base: base name shared by this run's files (no extension).
        skeleton_json_path: optional connections JSON; accepts any preset
            shipped in vaila/skeletons/ (MediaPipe, YOLO/COCO-17,
            SAM3+DINOv3 MHR70, Sapiens2 Goliath-308) or a user-authored
            equivalent with a top-level "connections": [["pA","pB"], ...].
        point_rate: capture rate in Hz, used to set the Blender scene rate
            (fractional rates such as 119.88012001 are preserved exactly via
            Blender's fps/fps_base pair).
        n_frames: number of reconstructed frames, used to set frame_end.
        mesh_dir: optional per-frame mesh directory (e.g. ``meshes_obj``) to
            import as a mesh sequence; checked for existence at run time.
        unreconstructed_markers: optional iterable of marker names (``"p42"``)
            that were never reconstructed in this trial. Connections touching
            them are dropped, so no bone is drawn to a marker parked on the
            world origin.
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

    # Drop connections that touch a marker which was never reconstructed.
    # Such a marker has no position to hold, so the BVH leaves it on the world
    # origin (BVH has no invalid-sample convention) — and a bone drawn to it
    # would streak from the body to (0, 0, 0) for the whole take. On real
    # markerless input this is common for finger keypoints, which rarely
    # resolve in two cameras at once.
    if unreconstructed_markers:
        missing = set(unreconstructed_markers)
        kept = [c for c in connections if c[0] not in missing and c[1] not in missing]
        n_dropped = len(connections) - len(kept)
        if n_dropped:
            print(
                f"Blender skeleton: dropped {n_dropped} connection(s) touching "
                f"{len(missing)} never-reconstructed marker(s) "
                f"(e.g. {', '.join(sorted(missing)[:5])})"
            )
        connections = kept

    # Don't generate if no connections (shouldn't happen with default, but good check)
    if not connections:
        return None

    fps_int, fps_base = blender_scene_fps(point_rate)
    frame_end = int(n_frames) if n_frames else 250
    bvh_name = f"{file_base}.bvh"
    bvh_path = os.path.abspath(os.path.join(output_dir, bvh_name))
    mesh_dir_name = mesh_dir or ""
    mesh_path = os.path.abspath(os.path.join(output_dir, mesh_dir)) if mesh_dir else ""
    script_name = f"{file_base}_blender_skeleton_viz.py"

    # The config header is the only f-string; the body below is a plain string
    # so Blender-side braces need no escaping.
    config = f"""import bpy
import os

# =========================================================
# Script automatically generated by vaila Toolbox
# Aligned Blender import: scene rate + frame range, BVH,
# OBJ mesh sequence, and skeleton bones.
# =========================================================
# How to use:
#   1. Open this script in Blender's Text Editor (or Scripting workspace)
#   2. Click "Run Script" (Play button)
#   3. Press Space to play the animation
#
# RUN THIS SCRIPT -- do not import the files by hand. Three things go wrong
# with a manual import, and this script fixes all of them:
#
#   * File > Import > BVH leaves update_scene_fps/update_scene_duration OFF,
#     so the scene stays at 24 fps with frame_end 250: the animation plays in
#     slow motion and stops a third of the way through.
#   * The .obj files are written in the RAW (x, y, z) DLT/world frame --
#     the same one the BVH ends up in once Blender's BVH importer applies
#     ITS OWN default axis conversion. A mesh-sequence add-on with no axis
#     controls (Stop Motion OBJ / OBJSequence) needs exactly this to line up
#     with the BVH out of the box. Blender's OWN File > Import > Wavefront
#     (.obj) dialog, by contrast, DOES convert by default and would move the
#     mesh away from the skeleton unless its axes are overridden to Forward
#     Y / Up Z. This script reads the raw floats itself with no conversion,
#     so no dialog setting can get it wrong either way.
#   * A C3D importer sets its own frame range and can reset the scene rate,
#     and (measured against this exporter's own C3D files) does not reliably
#     pick up the point rate even when the file states it correctly.
#
# Re-running is safe and is the fix for an already-broken scene: it will not
# duplicate the armature or the mesh, and it sets the rate and frame range
# LAST so nothing else gets the final word -- including after a manual C3D
# import, so importing a C3D into this same scene and then re-running this
# script is the fix for a C3D that looks like it plays at the wrong speed.
# It needs no add-ons.
# =========================================================

SCENE_FPS = {fps_int}
SCENE_FPS_BASE = {fps_base!r}
FRAME_START = 1
FRAME_END = {frame_end}
SCRIPT_NAME = {script_name!r}
BVH_NAME = {bvh_name!r}
MESH_DIR_NAME = {mesh_dir_name!r}
BVH_PATH_RECORDED = {bvh_path!r}
MESH_DIR_RECORDED = {mesh_path!r}
MESH_OBJECT_NAME = "Vaila_Mesh"
CONNECTIONS = {connections!r}
"""

    body = '''

def _script_dir():
    """Directory this script is actually being run from.

    Blender's Text Editor does not define __file__, so fall back to the text
    datablock's own filepath.
    """
    try:
        return os.path.dirname(os.path.abspath(__file__))
    except NameError:
        pass
    for text in bpy.data.texts:
        if text.filepath and os.path.basename(text.filepath) == SCRIPT_NAME:
            return os.path.dirname(os.path.abspath(bpy.path.abspath(text.filepath)))
    return ""


def _resolve(recorded, name, is_dir=False):
    """Prefer the path recorded at export time, else the one next to this script.

    The recorded path is absolute, so copying or moving the output folder --
    or generating the run somewhere temporary -- would otherwise leave this
    script importing nothing, or silently importing stale files from the old
    location while the user assumes they are looking at the new run.
    """
    exists = os.path.isdir if is_dir else os.path.isfile
    if recorded and exists(recorded):
        return recorded
    if name:
        local = os.path.join(_script_dir(), name)
        if exists(local):
            if recorded:
                print(f"Recorded path missing, using the copy next to this script: {local}")
            return local
    return recorded


BVH_PATH = _resolve(BVH_PATH_RECORDED, BVH_NAME)
MESH_DIR = _resolve(MESH_DIR_RECORDED, MESH_DIR_NAME, is_dir=True)


def setup_scene():
    """Set the scene rate and frame range to match the reconstruction.

    This is the step Blender's BVH importer does NOT do by default
    (update_scene_fps / update_scene_duration are both off), which is why an
    unmodified File > Import > BVH plays in slow motion and stops early.
    """
    scene = bpy.context.scene
    scene.render.fps = SCENE_FPS
    scene.render.fps_base = SCENE_FPS_BASE
    scene.frame_start = FRAME_START
    scene.frame_end = FRAME_END
    scene.frame_set(FRAME_START)
    effective = SCENE_FPS / SCENE_FPS_BASE if SCENE_FPS_BASE else SCENE_FPS
    print(f"Scene rate set to {effective:.6f} fps "
          f"(fps={SCENE_FPS}, fps_base={SCENE_FPS_BASE})")
    print(f"Scene frame range set to {FRAME_START}..{FRAME_END}")


def find_bvh_armature():
    """The BVH armature, if already imported (skips our own bone armature)."""
    for obj in bpy.context.scene.objects:
        if obj.type == 'ARMATURE' and obj.name != "Vaila_Skeleton":
            return obj
    return None


def import_bvh_if_needed():
    """Import the BVH with scene fps/duration updating enabled.

    Passing update_scene_fps/update_scene_duration=True is what keeps the
    markers in step with a C3D imported into the same scene.
    """
    existing = find_bvh_armature()
    if existing is not None:
        print(f"BVH armature already present: '{existing.name}' (not re-importing)")
        return existing
    if not BVH_PATH or not os.path.isfile(BVH_PATH):
        print(f"BVH not found, skipping import: {BVH_PATH}")
        return None
    bpy.ops.import_anim.bvh(
        filepath=BVH_PATH,
        update_scene_fps=True,
        update_scene_duration=True,
        use_fps_scale=False,
        frame_start=FRAME_START,
    )
    armature = find_bvh_armature()
    print(f"BVH imported: {os.path.basename(BVH_PATH)}")
    return armature


def _read_obj(path):
    """Return (flat vertex coords in Blender world space, faces) from an OBJ.

    The mesh files are written in the RAW (x, y, z) DLT/world frame -- no
    conversion needed here, because that is already where the skeleton ends
    up once Blender's BVH importer applies ITS OWN default axis conversion
    (see import_bvh_if_needed()). Reading the raw floats straight through
    rather than leaning on an importer's axis settings is what keeps the
    mesh on the skeleton regardless of which Blender OBJ importer a user
    might otherwise reach for by hand.
    """
    coords = []
    faces = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                x, y, z = (float(v) for v in line.split()[1:4])
                coords.extend((x, y, z))
            elif line.startswith("f "):
                faces.append([int(tok.split("/")[0]) - 1 for tok in line.split()[1:]])
    return coords, faces


def import_mesh_sequence():
    """Import the per-frame mesh sequence, WITHOUT needing any add-on.

    Every frame shares one topology, so instead of creating 631 objects (or
    depending on the "Stop Motion OBJ"/OBJSequence extension, which is not
    bundled with Blender and silently left the mesh un-imported when
    missing), this builds a single mesh and swaps its vertex positions on
    frame change. Only one frame's worth of geometry is ever live.
    """
    if not MESH_DIR or not os.path.isdir(MESH_DIR):
        print("No mesh sequence directory; skipping mesh import.")
        return None

    frames = sorted(f for f in os.listdir(MESH_DIR) if f.lower().endswith('.obj'))
    if not frames:
        print(f"No .obj frames in {MESH_DIR}; skipping mesh import.")
        return None

    # Re-runs must not stack meshes or handlers.
    _unregister_mesh_handler()
    old = bpy.data.objects.get(MESH_OBJECT_NAME)
    if old:
        bpy.data.objects.remove(old, do_unlink=True)

    print(f"Loading {len(frames)} mesh frames from {os.path.basename(MESH_DIR)} ...")
    coords_per_frame = []
    faces = None
    for i, name in enumerate(frames):
        coords, f = _read_obj(os.path.join(MESH_DIR, name))
        if faces is None:
            faces = f
        coords_per_frame.append(coords)
        if (i + 1) % 100 == 0 or i + 1 == len(frames):
            print(f"  {i + 1}/{len(frames)} frames")

    n_verts = len(coords_per_frame[0]) // 3
    mesh = bpy.data.meshes.new(MESH_OBJECT_NAME)
    mesh.from_pydata([(0.0, 0.0, 0.0)] * n_verts, [], faces)
    mesh.update()
    obj = bpy.data.objects.new(MESH_OBJECT_NAME, mesh)
    bpy.context.scene.collection.objects.link(obj)

    # Stash on the handler so it survives without globals leaking.
    _MESH_STATE["object_name"] = MESH_OBJECT_NAME
    _MESH_STATE["coords"] = coords_per_frame
    _register_mesh_handler()
    _apply_mesh_frame(bpy.context.scene)

    print(f"Mesh sequence ready: {len(frames)} frames, {n_verts} verts, "
          f"frame {FRAME_START} onwards (no add-on required)")
    return obj


_MESH_STATE = {"object_name": None, "coords": None}


def _apply_mesh_frame(scene):
    """Push the current frame's vertex positions into the live mesh."""
    coords = _MESH_STATE.get("coords")
    name = _MESH_STATE.get("object_name")
    if not coords or not name:
        return
    obj = bpy.data.objects.get(name)
    if obj is None:
        return
    idx = min(max(scene.frame_current - FRAME_START, 0), len(coords) - 1)
    obj.data.vertices.foreach_set("co", coords[idx])
    obj.data.update_tag()
    obj.update_tag()


def _vaila_mesh_frame_handler(scene, _depsgraph=None):
    _apply_mesh_frame(scene)


def _unregister_mesh_handler():
    for handlers in (bpy.app.handlers.frame_change_post, bpy.app.handlers.frame_change_pre):
        for h in list(handlers):
            if getattr(h, "__name__", "") == "_vaila_mesh_frame_handler":
                handlers.remove(h)


def _register_mesh_handler():
    bpy.app.handlers.frame_change_post.append(_vaila_mesh_frame_handler)


def create_skeleton_visualization(bvh_armature):
    """Second Armature with STICK bones linking the BVH markers."""
    print("=" * 60)
    print("vaila - Skeleton Visualization (Armature STICK)")
    print("=" * 60)

    if not bvh_armature:
        print("ERROR: No BVH Armature in scene; cannot build skeleton bones.")
        return

    available_bones = [b.name for b in bvh_armature.data.bones]
    print(f"Armature found: '{bvh_armature.name}' with {len(available_bones)} bones")

    # ----------------------------------------------------------
    # Removes previous Vaila_Skeleton (safe re-run)
    # ----------------------------------------------------------
    old_obj = bpy.data.objects.get("Vaila_Skeleton")
    if old_obj:
        bpy.data.objects.remove(old_obj, do_unlink=True)
    old_arm = bpy.data.armatures.get("Vaila_Skeleton_Data")
    if old_arm:
        bpy.data.armatures.remove(old_arm)

    # ----------------------------------------------------------
    # Creates new Armature with STICK display
    # ----------------------------------------------------------
    arm_data = bpy.data.armatures.new("Vaila_Skeleton_Data")
    arm_obj = bpy.data.objects.new("Vaila_Skeleton", arm_data)
    bpy.context.scene.collection.objects.link(arm_obj)

    arm_data.display_type = 'STICK'
    arm_obj.show_in_front = True

    # ----------------------------------------------------------
    # Edit mode: create one bone per connection
    # ----------------------------------------------------------
    bpy.ops.object.select_all(action='DESELECT')
    bpy.context.view_layer.objects.active = arm_obj
    arm_obj.select_set(True)
    bpy.ops.object.mode_set(mode='EDIT')

    valid_connections = []
    for idx, (start_name, end_name) in enumerate(CONNECTIONS):
        if start_name not in available_bones or end_name not in available_bones:
            print(f"  Skipping: {start_name} -> {end_name} (not found in Armature)")
            continue

        bone_name = f"link_{start_name}_{end_name}"
        bone = arm_data.edit_bones.new(bone_name)
        # Temporary position - overwritten by constraints
        bone.head = (0.0, 0.0, idx * 0.001)
        bone.tail = (0.0, 0.1, idx * 0.001)
        bone.use_connect = False
        valid_connections.append((start_name, end_name, bone_name))

    print(f"Bones created: {len(valid_connections)} connections")

    # ----------------------------------------------------------
    # Pose mode: constrain each bone between its two markers
    # ----------------------------------------------------------
    bpy.ops.object.mode_set(mode='POSE')

    for start_name, end_name, bone_name in valid_connections:
        pbone = arm_obj.pose.bones[bone_name]

        cloc = pbone.constraints.new('COPY_LOCATION')
        cloc.target = bvh_armature
        cloc.subtarget = start_name

        stretch = pbone.constraints.new('STRETCH_TO')
        stretch.target = bvh_armature
        stretch.subtarget = end_name
        try:
            stretch.volume = 'NONE'
        except TypeError:
            stretch.volume = 'NO_VOLUME'

        try:
            pbone.color.palette = 'CUSTOM'
            pbone.color.custom.normal = (0.0, 1.0, 0.3)
            pbone.color.custom.select = (1.0, 1.0, 0.0)
            pbone.color.custom.active = (1.0, 0.5, 0.0)
        except Exception:
            pass

    bpy.ops.object.mode_set(mode='OBJECT')
    bpy.context.view_layer.update()

    print(f"Done! {len(valid_connections)} skeleton connections created.")


def report():
    """Print what the scene actually ended up with.

    Worth reading: importing the BVH by hand (File > Import > BVH) leaves
    update_scene_fps/update_scene_duration off, so the scene stays at 24 fps
    with frame_end 250 — a 631-frame 120 Hz capture then plays in slow motion
    and stops a third of the way through. Re-running this script fixes an
    already-imported scene; it does not need a clean file.
    """
    scene = bpy.context.scene
    effective = scene.render.fps / scene.render.fps_base if scene.render.fps_base else scene.render.fps
    armature = find_bvh_armature()
    mesh = bpy.data.objects.get(MESH_OBJECT_NAME)
    print("=" * 60)
    print("vaila - scene report")
    print("=" * 60)
    print(f"  scene rate   : {effective:.6f} fps  (expected {SCENE_FPS / SCENE_FPS_BASE:.6f})")
    print(f"  frame range  : {scene.frame_start}..{scene.frame_end}  (expected {FRAME_START}..{FRAME_END})")
    print(f"  BVH armature : {armature.name if armature else 'MISSING'}"
          f"{f' ({len(armature.data.bones)} bones)' if armature else ''}")
    print(f"  mesh sequence: {'loaded' if mesh else 'not loaded'}")
    print(f"  bvh file     : {BVH_PATH or 'MISSING'}")
    print(f"  mesh dir     : {MESH_DIR or 'MISSING'}")
    if abs(effective - SCENE_FPS / SCENE_FPS_BASE) > 1e-3:
        print("  WARNING: scene rate is wrong - the animation will play at the wrong speed.")
    if scene.frame_end < FRAME_END:
        print("  WARNING: frame range is short - the animation will stop early.")
    # A hand-imported OBJ is the usual cause of "the mesh looks mirrored" or
    # "the mesh has Z where Y should be": the .obj files are written RAW
    # (x, y, z), matching where the skeleton ends up once Blender's BVH
    # importer applies its OWN default axis conversion. A no-conversion
    # mesh-sequence add-on (Stop Motion OBJ / OBJSequence) needs exactly
    # this to line up out of the box; Blender's OWN File > Import >
    # Wavefront (.obj) dialog DOES convert by default and needs an
    # override (Forward Y, Up Z) to cancel that back out. This script reads
    # the raw floats itself with no conversion, so Vaila_Mesh is always
    # right -- but a leftover manual import stays in the scene next to it.
    strays = [
        obj.name
        for obj in scene.objects
        if obj.type == "MESH" and obj.name != MESH_OBJECT_NAME
    ]
    if strays:
        print(f"  WARNING: other mesh objects in the scene: {', '.join(strays[:8])}"
              f"{' ...' if len(strays) > 8 else ''}")
        print("           If you imported the .obj files by hand, DELETE them -"
              f" {MESH_OBJECT_NAME} above is the correct one.")
        print("           Blender's OWN File > Import > Wavefront (.obj) dialog only"
              " lands correctly with axes overridden to Forward Y, Up Z.")
    print("=" * 60)


def main():
    armature = import_bvh_if_needed()
    import_mesh_sequence()
    create_skeleton_visualization(armature)
    # setup_scene() runs LAST on purpose: both the BVH importer and any C3D
    # importer touch the scene rate and frame range, so whatever ran before
    # must not get the final word. Setting the exact values here is what
    # stops the playback being slow and cut short.
    setup_scene()
    report()
    print("All set - press SPACE to play.")


main()
'''

    script_content = config + body

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


def _report_error(message, gui=True):
    """Surface a fatal error to whoever is driving the run.

    Always prints, and only opens a dialog in GUI mode: an unguarded
    messagebox blocks a headless/CLI run forever (nobody is there to click
    it) on a machine with a real DISPLAY, and raises TclError without one.
    """
    print(f"[red]Error: {message}[/red]")
    if gui:
        from tkinter import messagebox as _mb

        _mb.showerror("Error", message)


def process_files_in_directory(
    dlt_params_dfs,
    input_directory,
    output_directory,
    data_rate,
    swap_yz=True,
    skeleton_json_path=None,
    gui=True,
):
    """
    Reconstruct 3D coordinates from N camera pixel CSV files (one per camera,
    matching the order of dlt_params_dfs) using per-frame-varying DLT3D
    parameters for each camera.

    Args:
        dlt_params_dfs (list of pd.DataFrame): Per-camera DLT3D parameter
            tables (frame + 11 coefficients per row), same order as the pixel
            CSV files once sorted by filename.
        input_directory (str): Directory containing exactly len(dlt_params_dfs)
            pixel CSV files, one per camera.
        output_directory (str): Directory to save the reconstructed output.
        data_rate (float): Data frequency in Hz (recorded in the console summary).
        swap_yz (bool): If True, swap Y and Z axes in the BVH export (for
            Blender's Z-up convention).
        skeleton_json_path (str | None): Optional path to a skeleton
            connections JSON (see vaila/skeletons/) used to generate a
            Blender companion script that draws bones between markers.
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    num_cameras = len(dlt_params_dfs)

    # Per-camera DLT3D frames + parameters (positional: col 0 = frame, rest = 11 coeffs)
    dlt_frames_list = []
    dlt_values_list = []
    for df in dlt_params_dfs:
        arr = df.to_numpy(dtype=np.float64)
        dlt_frames_list.append(arr[:, 0])
        dlt_values_list.append(arr[:, 1:])

    csv_files = sorted([f for f in os.listdir(input_directory) if f.endswith(".csv")])

    if not csv_files:
        _report_error("No CSV files found in the selected directory!", gui)
        return
    if len(csv_files) != num_cameras:
        _report_error(
            f"Expected {num_cameras} pixel CSV file(s) in the input directory "
            f"(one per camera, matching --dlt-files), found {len(csv_files)}.",
            gui,
        )
        return

    print(
        f"Found {len(csv_files)} camera pixel file(s), matching {num_cameras} DLT3D camera set(s)"
    )

    # Load every camera's pixel file (column-order based; labels are ignored)
    pixel_frames_list = []
    pixel_xy_list = []
    for csv_file in csv_files:
        path = os.path.join(input_directory, csv_file)
        try:
            frames_arr, xy_arr = load_pixel_csv_positional(path)
        except Exception as e:
            _report_error(f"Error reading {csv_file}: {e}. Aborting.", gui)
            return
        pixel_frames_list.append(frames_arr)
        pixel_xy_list.append(xy_arr)

    num_markers = min(xy.shape[1] for xy in pixel_xy_list)
    if any(xy.shape[1] != num_markers for xy in pixel_xy_list):
        print(
            f"[yellow]Warning: camera pixel files have different marker counts; "
            f"using the smallest common count: {num_markers}[/yellow]"
        )

    common_frames = find_common_frames(pixel_frames_list)
    if common_frames.size == 0:
        _report_error("No common frames found among the camera pixel files!", gui)
        return
    print(f"Processing {len(common_frames)} common frame(s) across {num_cameras} camera(s)...")

    # Only create the output folder once every validation above has passed.
    output_dir = os.path.join(output_directory, f"vaila_rec3d_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    reconstruction_array[:, 0] = common_frames

    pixel_frame_to_row = [{int(f): i for i, f in enumerate(frames)} for frames in pixel_frames_list]
    dlt_frame_to_row = [{int(f): i for i, f in enumerate(frames)} for frames in dlt_frames_list]

    progress_step = max(1, total_frames // 20)
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        frame_int = int(frame)

        # Per-camera DLT3D parameters for THIS frame (the "DLT matrix" lookup)
        dlt_params_for_frame = []
        frame_ok = True
        for cam_idx in range(num_cameras):
            dlt_row = dlt_frame_to_row[cam_idx].get(frame_int)
            if dlt_row is None:
                frame_ok = False
                break
            A = dlt_values_list[cam_idx][dlt_row]
            if np.isnan(A).any():
                frame_ok = False
                break
            dlt_params_for_frame.append(A)
        if not frame_ok:
            continue

        pixel_row_for_cam = [pixel_frame_to_row[c][frame_int] for c in range(num_cameras)]

        for marker in range(num_markers):
            pixel_obs_list = []
            valid_marker = True
            for cam_idx in range(num_cameras):
                x_obs, y_obs = pixel_xy_list[cam_idx][pixel_row_for_cam[cam_idx], marker]
                if np.isnan(x_obs) or np.isnan(y_obs):
                    valid_marker = False
                    break
                pixel_obs_list.append((float(x_obs), float(y_obs)))
            if not valid_marker:
                continue

            point3d = rec3d_multicam(dlt_params_for_frame, pixel_obs_list)
            col_start = 1 + marker * 3
            reconstruction_array[frame_idx, col_start : col_start + 3] = point3d

    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    rec_coords_df = pd.DataFrame(reconstruction_array, columns=header)  # type: ignore
    rec_coords_df["frame"] = rec_coords_df["frame"].astype(int)

    output_file_3d = os.path.join(output_dir, f"rec3d_{timestamp}.3d")
    output_file_csv = os.path.join(output_dir, f"rec3d_{timestamp}.csv")
    _write_rec3d_output(rec_coords_df, output_file_3d)
    _write_rec3d_output(rec_coords_df, output_file_csv)

    file_base = f"rec3d_{timestamp}"
    save_rec3d_as_bvh(rec_coords_df, output_dir, file_base, data_rate, gui=gui, swap_yz=swap_yz)
    blender_script = generate_blender_companion_script(
        output_dir,
        file_base,
        skeleton_json_path,
        point_rate=data_rate,
        n_frames=total_frames,
        unreconstructed_markers=find_unreconstructed_markers(rec_coords_df),
    )

    print("\n=== Processing Complete ===")
    print(f"Cameras: {num_cameras}")
    print(f"Frames reconstructed: {total_frames}")
    print(f"Markers: {num_markers}")
    print(f"Data rate used: {data_rate} Hz")
    print(f"Output directory: {output_dir}")
    print(f"  - {file_base}.bvh (Mocap format for Blender, axes swapped Y<->Z: {swap_yz})")
    if blender_script:
        print(f"  - {os.path.basename(blender_script)} (Run this in Blender to visualize skeleton)")

    # Must stay behind `if gui:` — on the CLI path there is no Tk root and
    # nobody to click the dialog, so an unguarded showinfo() blocks forever
    # on a machine with a real DISPLAY (and raises TclError without one).
    # Matches how rec3d_one_dlt3d.py's run_reconstruction() gates its own
    # messagebox calls.
    if gui:
        messagebox.showinfo(
            "Processing Complete",
            f"3D reconstruction completed successfully!\n\n"
            f"Cameras: {num_cameras}\n"
            f"Frames: {total_frames}\n"
            f"Markers: {num_markers}\n"
            f"Data rate: {data_rate} Hz\n"
            f"Output directory: {os.path.basename(output_dir)}",
        )
    print(f"Reconstructed 3D coordinates saved to {output_dir}")


def run_rec3d(
    dlt_files=None,
    input_directory=None,
    output_directory=None,
    data_rate=None,
    swap_yz=None,
    skeleton_json_path=None,
    gui=True,
):
    """Reconstruct 3D coordinates from per-frame-varying DLT3D parameters.

    gui: True when driven from the vailá GUI (Tk dialogs and completion
        message boxes are used). Pass False from a headless/CLI caller —
        an unguarded messagebox blocks forever waiting for a click.
    """
    # Print the script version and directory
    print(f"Running script: {Path(__file__).name}")
    print(f"Script directory: {Path(__file__).parent}")
    print("Starting optimized rec3d.py...")
    print("-" * 80)

    if dlt_files is None:
        root = Tk()
        root.withdraw()

        # Step 1: Select DLT3D parameters files (multiple cameras)
        print("Step 1: Selecting DLT3D parameters files...")
        dlt_files = filedialog.askopenfilenames(
            title="Select DLT3D Parameters Files (one per camera, matching pixel file order)",
            filetypes=[("DLT3D files", "*.dlt3d"), ("CSV files", "*.csv")],
        )
        if not dlt_files:
            print("DLT file selection cancelled.")
            return

        # Step 2: Select input directory with CSV files
        print("Step 2: Selecting input directory...")
        input_directory = filedialog.askdirectory(
            title="Select Directory Containing Pixel CSV Files (one per camera)"
        )
        if not input_directory:
            print("Input directory selection cancelled.")
            return

        # Step 3: Select output directory
        print("Step 3: Selecting output directory...")
        output_directory = filedialog.askdirectory(title="Select Output Directory for Results")
        if not output_directory:
            print("Output directory selection cancelled.")
            return

        # Step 4: Ask for data frequency
        print("Step 4: Setting data frequency...")
        data_rate = simpledialog.askfloat(
            "Data Frequency",
            "Enter the data frequency (Hz), e.g. 119.88012001 for a real NTSC-derived rate:",
            minvalue=0.0001,
            initialvalue=100.0,
        )
        if data_rate is None:
            messagebox.showerror("Error", "Data frequency is required. Operation cancelled.")
            return

        # Step 5: Ask if user wants to swap Y and Z axes for Blender (default YES)
        swap_yz = messagebox.askyesno(
            "Blender Axis Export",
            "Swap Y and Z axes for the BVH output?\n\n"
            "YES (recommended, default): height becomes vertical (Z-up) in "
            "Blender.\n"
            "NO: keep the original DLT coordinates.",
            default=messagebox.YES,
        )

        # Step 6: (Optional) Select Skeleton Pose JSON
        print("Step 6: (Optional) Selecting Skeleton Pose JSON for Blender visualization...")
        skeleton_json_path = None
        use_skeleton = messagebox.askyesno(
            "Skeleton Visualization",
            "Do you have a Skeleton Pose JSON file (e.g. MediaPipe, YOLO, "
            "SAM3+DINOv3, Sapiens2 -- see vaila/skeletons/)?\n"
            "This allows generating a script to visualize connections in Blender.",
        )
        if use_skeleton:
            skeleton_json_path = filedialog.askopenfilename(
                title="Select Skeleton Pose JSON",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")],
            )

        root.destroy()
    else:
        # Headless mode
        if input_directory is None or output_directory is None or data_rate is None:
            print(
                "Error: dlt-files, input-dir, output-dir, and rate are required for headless mode."
            )
            return
        if swap_yz is None:
            swap_yz = True

    # Load and validate DLT parameters for each camera
    print("Loading DLT3D parameters...")
    dlt_params_dfs = []
    for dlt_file in dlt_files:
        df = pd.read_csv(dlt_file)
        if df.empty:
            print(f"Error: DLT3D file {os.path.basename(dlt_file)} is empty!")
            return
        dlt_params_dfs.append(df)

    print("Configuration complete:")
    print(f"  - DLT3D files: {len(dlt_files)} cameras")
    print(f"  - Input directory: {input_directory}")
    print(f"  - Output directory: {output_directory}")
    print(f"  - Data rate: {data_rate} Hz")
    print(f"  - Swap Y/Z for BVH: {swap_yz}")
    print(
        f"  - Skeleton JSON: {os.path.basename(skeleton_json_path) if skeleton_json_path else 'None'}"
    )
    for i, df in enumerate(dlt_params_dfs):
        print(f"  - Camera {i + 1}: {len(df)} frames")
    print("-" * 80)

    # Equivalent CLI command, so a GUI run can always be repeated headlessly.
    cli_parts = [
        "uv run python vaila/rec3d.py",
        f"--dlt-files {' '.join(str(f) for f in dlt_files)}",
        f"--input-dir {input_directory}",
        f"--output-dir {output_directory}",
        f"--rate {data_rate}",
    ]
    # --swap-yz is the default; only the opt-out needs to appear explicitly.
    if not swap_yz:
        cli_parts.append("--no-swap-yz")
    if skeleton_json_path:
        cli_parts.append(f"--skeleton {skeleton_json_path}")
    cli_cmd = " ".join(cli_parts)
    print(f">> {cli_cmd}")

    # Process files
    process_files_in_directory(
        dlt_params_dfs,
        input_directory,
        output_directory,
        data_rate,
        swap_yz=swap_yz,
        skeleton_json_path=skeleton_json_path,
        gui=gui,
    )

    # Repeat the equivalent CLI command LAST, after all the processing output,
    # so it is the final thing on screen and easy to copy.
    print("\n" + "=" * 80)
    print("Equivalent CLI command for this run (copy/paste to repeat headlessly):")
    print("=" * 80)
    print(f">> {cli_cmd}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=(
            "Reconstruct 3D coordinates from N camera pixel CSV files using "
            "per-frame-varying DLT3D parameters (a DLT matrix, one row per frame, "
            "per camera). --input-dir must contain exactly one pixel CSV per "
            "camera, matching --dlt-files in count (paired by sorted filename)."
        )
    )
    parser.add_argument(
        "--dlt-files", nargs="+", help="Path to DLT3D parameter files (one per camera)"
    )
    parser.add_argument(
        "--input-dir",
        help="Directory containing exactly one pixel CSV per camera (matching --dlt-files)",
    )
    parser.add_argument("--output-dir", help="Output directory for results")
    parser.add_argument(
        "--rate",
        type=float,
        help="Data frequency in Hz (accepts fractional rates, e.g. 119.88012001)",
    )
    parser.add_argument(
        "--swap-yz",
        dest="swap_yz",
        action="store_true",
        default=True,
        help=(
            "Swap Y and Z axes in the BVH output so height ends up vertical "
            "(Z-up) in Blender. This is the DEFAULT; the flag is kept for "
            "backward compatibility and as an explicit opt-in."
        ),
    )
    parser.add_argument(
        "--no-swap-yz",
        dest="swap_yz",
        action="store_false",
        help="Keep the raw DLT axes in the BVH output (no Y/Z swap)",
    )
    parser.add_argument(
        "--skeleton",
        metavar="FILE",
        help=(
            "Path to a skeleton connections JSON (defines bone connections "
            "for the Blender companion script); see vaila/skeletons/ for "
            "MediaPipe/YOLO/SAM3+DINOv3/Sapiens2 presets"
        ),
    )
    args = parser.parse_args()

    # Headless: every argument came from the command line, so no Tk dialog
    # should ever open (an unguarded one would hang the process).
    cli_mode = bool(args.dlt_files and args.input_dir and args.output_dir)
    run_rec3d(
        dlt_files=args.dlt_files,
        input_directory=args.input_dir,
        output_directory=args.output_dir,
        data_rate=args.rate,
        swap_yz=args.swap_yz,
        skeleton_json_path=args.skeleton,
        gui=not cli_mode,
    )
