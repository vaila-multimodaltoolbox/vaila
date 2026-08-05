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
Version: 0.3.99
Created: 02 August 2025
Last Updated: 05 August 2026

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
  - Column LABELS are not inspected — only column ORDER matters: column 0 is
    the frame identifier and every pair of columns after that is one marker's
    (x, y), regardless of header text (vailá p1_x/p1_y, SAM3, YOLO, MediaPipe
    named joints, etc.).
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
  - mesh_alignment.py — Umeyama fit used by the optional mesh-export feature

================================================================================
Optional: mesh-for-Blender export (--mesh-source-dir / --export-mesh)
================================================================================

If your pixel files are the MHR70-ordered (p1_x,p1_y,...,p70_x,p70_y) markers
CSVs written by sam3dinov3_visualize.py's "Visualize ID" output, you can also
export a per-frame body MESH aligned into this same DLT world space, for
Blender. For each camera pass a `--mesh-source-dir` (one per camera, same
order as --dlt3d) pointing at that visualize-ID output directory — it must
contain `<stem>_mhr70_rec3d.csv` (that camera's own monocular 3D MHR70
estimate), a `meshes_obj/` or `meshes_ply/` folder of per-frame meshes (from
sam3dinov3_visualize.py --export-mesh obj|ply), and `mesh_faces.npy`.

Simplified path: `--pixels` can be OMITTED when `--mesh-source-dir` is given
— each run directory already contains its own `*_markers.csv`, so one path
per camera (the run directory) is enough to drive both triangulation and
mesh alignment. Explicit `--pixels` is still supported (and required if you
are not using mesh export at all, or your pixel CSVs live somewhere else).

At each frame, a similarity transform (rotation + uniform scale +
translation) is fit per camera from its monocular MHR70 estimate onto the
DLT-triangulated skeleton (see mesh_alignment.umeyama_alignment); the camera
with the lowest fit residual is used as that frame's mesh source, and the
same transform is applied to its mesh vertices. This is a coordinate-frame
reconciliation, not a re-triangulation — mesh shape/proportion accuracy is
inherited entirely from the monocular estimate. Output goes to
`meshes_<fmt>/frame_NNNNNN.<fmt>` plus a `<file_base>_mesh_alignment.csv`
manifest (frame, chosen camera index, fit residual in meters) in the same
timestamped output subfolder as the other results.

================================================================================
Blender alignment (v0.3.99, mesh axis convention corrected 2026-08-05)
================================================================================

--swap-yz is the DEFAULT (use --no-swap-yz to keep raw DLT axes) and applies
to the BVH file. The exported MESH is always written in the RAW (x, y, z)
DLT/world frame — the same convention as the triangulated skeleton CSV —
REGARDLESS of --swap-yz. This is deliberate, not an oversight: Blender's
bundled BVH importer applies its own axis conversion by default, so a
swapped BVH file still lands correctly, but the "Stop Motion OBJ"/OBJSequence
family of mesh-sequence add-ons applies none at all (confirmed from its
source), so the mesh must already be in Blender's final world convention on
disk. A swapped mesh file looked right through Blender's own native OBJ
import dialog (which does convert) but wrong — "Z where Y should be" — through
that add-on, which is what most people actually use to play a mesh sequence.
See README_mesh_import.txt (written next to every meshes_<fmt>/ folder) for
the exact manual-import axis settings this implies.

The generated `<file_base>_blender_skeleton_viz.py` now imports everything
already aligned: it sets the scene rate (exactly, including fractional NTSC
rates via Blender's fps/fps_base pair) and the frame range, imports the BVH
with update_scene_fps/update_scene_duration enabled, imports the OBJ mesh
sequence starting on the same frame (reading the raw floats itself, no axis
conversion needed), then builds the skeleton bones. Without that scene setup,
Blender's BVH importer leaves a 631-frame 120 Hz capture in a 24 fps scene
ending at frame 250 — the BVH and mesh play in slow motion and stop a third
of the way through. A C3D importer is not exempt either: Blender's bundled
one (io_anim_c3d) does not reliably update the scene rate from the file even
though this exporter's C3D correctly states it (verified: POINT:RATE and the
header's frame_rate both read back as 120.0 for a 120 Hz run) — so a C3D
imported into the same scene can still look desynced from the BVH/mesh unless
this companion script is (re-)run afterward, which always sets the scene rate
last regardless of import order.

A GUI run prints the equivalent CLI command both before and after processing,
so the last thing on screen is a copy-pasteable headless re-run.
"""

import argparse
import bisect
import os
import sys
from datetime import datetime
from pathlib import Path
from tkinter import Tk, filedialog, messagebox, simpledialog

import ezc3d
import numpy as np
import pandas as pd
from rich import print

try:
    from .mesh_alignment import (
        ALIGNMENT_MARKER_INDICES,
        apply_similarity_transform,
        best_camera_alignment,
        interpolate_similarity_transform,
        read_obj_vertices,
        write_obj_mesh,
        write_ply_mesh,
    )
    from .rec3d import (
        find_common_frames,
        find_unreconstructed_markers,
        generate_blender_companion_script,
        load_pixel_csv_positional,
        rec3d_multicam,
        save_rec3d_as_bvh,
    )
except ImportError:
    from mesh_alignment import (  # ty: ignore[unresolved-import]
        ALIGNMENT_MARKER_INDICES,
        apply_similarity_transform,
        best_camera_alignment,
        interpolate_similarity_transform,
        read_obj_vertices,
        write_obj_mesh,
        write_ply_mesh,
    )
    from rec3d import (  # ty: ignore[unresolved-import]
        find_common_frames,
        find_unreconstructed_markers,
        generate_blender_companion_script,
        load_pixel_csv_positional,
        rec3d_multicam,
        save_rec3d_as_bvh,
    )


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


def _load_wide_xyz_csv(file_path):
    """
    Load a wide 3D CSV (frame, p1_x, p1_y, p1_z, p2_x, p2_y, p2_z, ...) using
    COLUMN ORDER, not labels — mirrors rec3d.load_pixel_csv_positional but for
    triples instead of pairs (used for a camera's own monocular MHR70 3D
    estimate, e.g. `<stem>_mhr70_rec3d.csv`).

    Returns:
        tuple[np.ndarray, np.ndarray]: (frame_values shape (n_rows,),
        xyz shape (n_rows, num_markers, 3)).
    """
    df = pd.read_csv(file_path)
    n_cols = df.shape[1]
    n_coord_cols = n_cols - 1
    if n_coord_cols < 3 or n_coord_cols % 3 != 0:
        raise ValueError(
            f"expected frame + a multiple-of-3 coordinate columns, found {n_coord_cols} "
            f"coordinate column(s) in {file_path}"
        )
    values = df.to_numpy(dtype=np.float64)
    frame_values = values[:, 0]
    num_markers = n_coord_cols // 3
    xyz = values[:, 1:].reshape(-1, num_markers, 3)
    return frame_values, xyz


def _find_mesh_source_bundle(mesh_source_dir, export_fmt):
    """
    Locate the files a mesh-source directory (a sam3dinov3_visualize.py
    "Visualize ID" output) must contain: the camera's own monocular MHR70 3D
    CSV, the per-frame mesh directory for `export_fmt`, and the shared face
    topology.

    Returns:
        tuple[Path, Path, Path] | None: (mhr70_rec3d_csv, mesh_frames_dir,
        mesh_faces_path), or None with a printed reason if anything is missing.
    """
    mesh_source_dir = Path(mesh_source_dir)
    matches = sorted(mesh_source_dir.glob("*_mhr70_rec3d.csv"))
    if not matches:
        print(f"[yellow]No *_mhr70_rec3d.csv found in {mesh_source_dir}[/yellow]")
        return None
    mhr70_rec3d_csv = matches[0]

    mesh_frames_dir = mesh_source_dir / f"meshes_{export_fmt}"
    if not mesh_frames_dir.is_dir():
        print(
            f"[yellow]No meshes_{export_fmt}/ folder in {mesh_source_dir} — run "
            f"sam3dinov3_visualize.py with --export-mesh {export_fmt} first.[/yellow]"
        )
        return None

    mesh_faces_path = mesh_source_dir / "mesh_faces.npy"
    if not mesh_faces_path.is_file():
        print(f"[yellow]No mesh_faces.npy found in {mesh_source_dir}[/yellow]")
        return None

    return mhr70_rec3d_csv, mesh_frames_dir, mesh_faces_path


def find_markers_csv_in_dir(mesh_source_dir):
    """
    Locate the 2D pixel markers CSV (`*_markers.csv`) inside a
    sam3dinov3_visualize.py "Visualize ID" output directory — the same
    directory already required by --mesh-source-dir. Lets --pixels be
    omitted when running with --mesh-source-dir: one directory per camera is
    then enough to drive both triangulation (this file) and mesh alignment
    (the directory's *_mhr70_rec3d.csv + meshes_<fmt>/ + mesh_faces.npy).

    Returns:
        Path | None: the markers CSV path, or None with a printed reason if
        not found (or ambiguous).
    """
    mesh_source_dir = Path(mesh_source_dir)
    matches = sorted(mesh_source_dir.glob("*_markers.csv"))
    if not matches:
        print(f"[yellow]No *_markers.csv found in {mesh_source_dir}[/yellow]")
        return None
    if len(matches) > 1:
        print(
            f"[yellow]Multiple *_markers.csv found in {mesh_source_dir}, "
            f"using {matches[0].name}[/yellow]"
        )
    return matches[0]


def _write_mesh_import_readme(mesh_dir, export_fmt, swap_yz, n_frames):
    """State the axis settings a manual Blender import needs, next to the files.

    Mesh vertices are always written RAW (x, y, z) -- the same convention as
    the reconstruction CSV and, once the BVH importer applies its own default
    axis conversion, the same convention the skeleton ends up in too. This is
    what a mesh-sequence add-on with no axis controls at all (Stop Motion OBJ
    -- confirmed by reading its source: it assigns "v x y z" straight to
    Blender's mesh, no conversion) needs to line up with the BVH out of the
    box. Blender's OWN native File > Import > Wavefront (.obj) dialog DOES
    apply a conversion by default, so it needs an override to cancel it out.
    """
    axes = "Forward Y, Up Z  (override the dialog's defaults -- see below)"
    Path(mesh_dir).joinpath("README_mesh_import.txt").write_text(
        "vaila rec3d_one_dlt3d -- aligned mesh sequence\n"
        "=============================================\n\n"
        f"{n_frames} frame(s), format .{export_fmt}, written in the RAW (x, y, z)\n"
        f"DLT/world frame -- the same convention as the reconstruction CSV.\n"
        f"Frame N of this sequence is frame N of the .c3d / .bvh in the parent folder.\n\n"
        "PREFERRED: run the *_blender_skeleton_viz.py script in the parent folder.\n"
        "It imports the BVH and this sequence and sets the scene rate and frame\n"
        "range -- none of which a manual import does.\n\n"
        "A mesh-sequence add-on with no axis settings (e.g. Stop Motion OBJ /\n"
        "OBJSequence) works with these files out of the box -- no configuration\n"
        "needed, because they never apply any axis conversion of their own.\n\n"
        "If you import these files with Blender's OWN File > Import > Wavefront\n"
        "(.obj) dialog instead, its defaults DO apply a conversion and will move\n"
        f"the mesh away from the skeleton -- override the axis settings to:\n"
        f"    {axes}\n\n"
        "A manual File > Import > BVH also leaves the scene at 24 fps with\n"
        "frame_end 250, so a long capture plays slowly and stops early.\n"
        + (
            ""
            if swap_yz
            else "\nThis run was exported WITHOUT --swap-yz: the BVH is ALSO raw (x, y, z),\n"
            "so it needs the same Forward Y, Up Z override if imported by hand.\n"
        ),
        encoding="utf-8",
    )


def reconstruct_mesh_sequence(
    rec3d_df,
    mesh_source_dirs,
    output_dir,
    file_base,
    export_fmt="obj",
    marker_indices=ALIGNMENT_MARKER_INDICES,
    swap_yz=False,
):
    """
    Align and export a per-frame body mesh from N cameras' monocular
    SAM3+DINOv3 output into this run's DLT-triangulated world space (see
    module docstring "Optional: mesh-for-Blender export").

    Args:
        rec3d_df: the already-triangulated skeleton DataFrame (frame,
            p1_x, p1_y, p1_z, ..., MHR70-ordered columns) produced earlier in
            this same run_reconstruction() call.
        mesh_source_dirs: list of per-camera sam3dinov3_visualize.py
            "Visualize ID" output directories, same order as --dlt3d/--pixels.
        output_dir: this run's timestamped output directory.
        file_base: this run's file base name (for the manifest filename).
        export_fmt: "obj" or "ply".
        marker_indices: 1-based marker indices used for the Umeyama fit
            (default: torso/hip/knee subset, see mesh_alignment.py).
        swap_yz: kept for signature symmetry with save_rec3d_as_bvh() and
            recorded in the written README, but no longer changes what gets
            written to the mesh files (see below).

    Mesh files are always written in the RAW (unswapped) DLT/world frame —
    the same (x, y, z) convention as the triangulated skeleton CSV — REGARDLESS
    of swap_yz. This looked wrong at first (the BVH is Y/Z-swapped by default
    for Blender's Z-up convention, so shouldn't the mesh match it byte for
    byte?) until measuring what actually happens with a real Blender mesh-
    sequence viewer: the bundled "Stop Motion OBJ" family of add-ons parses
    "v x y z" lines and assigns them straight to `mesh.vertices` with NO axis
    conversion at all (confirmed by reading stop_motion_obj2/core.py's
    parse_obj()/apply_to_mesh() — there is no forward/up parameter on its
    operator to override this). Blender's own BVH importer, by contrast,
    DOES apply an axis conversion by default. So a mesh file written in the
    swapped convention displayed correctly only through Blender's native
    `wm.obj_import` (which also converts) — but through the add-on most
    people actually use for a mesh sequence, its height ended up along the
    wrong Blender axis relative to the (correctly displayed) BVH skeleton:
    exactly "the mesh has Z where Y should be" reported on a real run
    (2026-08-05). Writing the mesh raw fixes the add-on path unconditionally
    and only costs one extra step for `wm.obj_import`: override Forward=Y,
    Up=Z (an identity/no-op conversion) instead of leaving its defaults —
    see the companion script and README_mesh_import.txt, both updated to
    match.

    Returns:
        dict summary (frames_written, frames_skipped, skip_reasons,
        residuals, camera_switches, manifest_path) or None if no camera's
        mesh-source bundle could be found.
    """
    if export_fmt not in ("obj", "ply"):
        raise ValueError(f"export_fmt must be 'obj' or 'ply', got {export_fmt!r}")
    writer = write_obj_mesh if export_fmt == "obj" else write_ply_mesh

    bundles = []
    for mesh_source_dir in mesh_source_dirs:
        bundles.append(_find_mesh_source_bundle(mesh_source_dir, export_fmt))
    if all(b is None for b in bundles):
        print(
            "[red]No usable mesh-source directory found for any camera; skipping mesh export.[/red]"
        )
        return None

    marker_cols_x = [f"p{i}_x" for i in marker_indices]
    marker_cols_y = [f"p{i}_y" for i in marker_indices]
    marker_cols_z = [f"p{i}_z" for i in marker_indices]
    missing_cols = [
        c for c in marker_cols_x + marker_cols_y + marker_cols_z if c not in rec3d_df.columns
    ]
    if missing_cols:
        print(
            f"[red]rec3d output is missing alignment marker columns {missing_cols[:3]}... "
            f"(need {len(marker_indices)} MHR70 markers); skipping mesh export.[/red]"
        )
        return None

    target_by_frame = {}
    for _idx, row in rec3d_df.iterrows():
        pts = np.stack(
            [
                row[marker_cols_x].to_numpy(),
                row[marker_cols_y].to_numpy(),
                row[marker_cols_z].to_numpy(),
            ],
            axis=1,
        ).astype(np.float64)
        target_by_frame[int(row["frame"])] = pts

    monocular_by_camera = []
    faces_by_camera = []
    for bundle in bundles:
        if bundle is None:
            monocular_by_camera.append(None)
            faces_by_camera.append(None)
            continue
        mhr70_rec3d_csv, _mesh_dir, mesh_faces_path = bundle
        frames_arr, xyz_arr = _load_wide_xyz_csv(mhr70_rec3d_csv)
        max_needed = max(marker_indices)
        if xyz_arr.shape[1] < max_needed:
            print(
                f"[yellow]{mhr70_rec3d_csv} has only {xyz_arr.shape[1]} markers, "
                f"need marker index {max_needed}; skipping this camera.[/yellow]"
            )
            monocular_by_camera.append(None)
            faces_by_camera.append(None)
            continue
        frame_to_row = {int(f): i for i, f in enumerate(frames_arr)}
        monocular_by_camera.append((frame_to_row, xyz_arr))
        faces_by_camera.append(np.load(mesh_faces_path))

    mesh_dirs_by_camera = [b[1] if b is not None else None for b in bundles]

    output_mesh_dir = Path(output_dir) / f"meshes_{export_fmt}"
    output_mesh_dir.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    skip_reasons = {}
    previous_camera = None
    camera_switches = 0
    frames_written = 0

    common_frames = sorted(target_by_frame.keys())

    def _mesh_frame_path(cam_idx, frame):
        """Path to camera `cam_idx`'s monocular mesh for `frame`, or None."""
        mesh_frames_dir = mesh_dirs_by_camera[cam_idx]
        if mesh_frames_dir is None or faces_by_camera[cam_idx] is None:
            return None
        # meshes_obj/meshes_ply are always named frame_NNNNNN.<fmt>
        candidate = mesh_frames_dir / f"frame_{frame:06d}.{export_fmt}"
        if candidate.is_file():
            return candidate
        candidate = mesh_frames_dir / f"frame_{frame:06d}.obj"
        return candidate if candidate.is_file() else None

    # --- Pass 1: solve the alignment on every frame where the target allows it.
    # Transforms are kept per camera because each one maps that camera's own
    # monocular space into world space; they are not interchangeable, so a gap
    # may only ever be interpolated between solved frames of the SAME camera.
    solved = {}
    solved_by_camera = {cam_idx: [] for cam_idx in range(len(bundles))}
    for frame in common_frames:
        target_pts = target_by_frame[frame]
        # No all-or-nothing NaN guard here on purpose: best_camera_alignment()
        # already drops the individual rows that are non-finite in either the
        # source or the target, per camera, before checking min_points. A
        # frame where a single alignment marker is occluded (e.g. one acromion)
        # is still perfectly solvable from the remaining ones, and rejecting it
        # outright would push it into the interpolation fallback below —
        # trading a real fit for a guessed one.
        source_points_per_camera = []
        for cam_idx in range(len(bundles)):
            entry = monocular_by_camera[cam_idx]
            if entry is None:
                source_points_per_camera.append(None)
                continue
            frame_to_row, xyz_arr = entry
            row_idx = frame_to_row.get(frame)
            if row_idx is None:
                source_points_per_camera.append(None)
                continue
            source_points_per_camera.append(xyz_arr[row_idx, [i - 1 for i in marker_indices], :])

        best_idx, best_result = best_camera_alignment(source_points_per_camera, target_pts)
        if best_idx is None or best_result is None:
            continue
        # best_camera_alignment() only returns a non-degenerate AlignmentResult here,
        # which always has R/s/t populated together (see mesh_alignment.py).
        assert best_result.R is not None and best_result.s is not None and best_result.t is not None
        solved[frame] = (best_idx, best_result)
        solved_by_camera[best_idx].append((frame, best_result.R, best_result.s, best_result.t))

    solved_frames_by_camera = {
        cam_idx: [entry[0] for entry in entries] for cam_idx, entries in solved_by_camera.items()
    }

    def _neighbours(cam_idx, frame):
        """Nearest solved (frame, R, s, t) of `cam_idx` before/after `frame`."""
        frames = solved_frames_by_camera[cam_idx]
        pos = bisect.bisect_left(frames, frame)
        before = solved_by_camera[cam_idx][pos - 1] if pos > 0 else None
        after = solved_by_camera[cam_idx][pos] if pos < len(frames) else None
        return before, after

    # --- Pass 2: write one mesh per frame, interpolating the placement of any
    # frame Pass 1 could not solve, so the exported sequence has no gaps.
    interpolated_frames = 0
    for frame in common_frames:
        interpolated = False
        if frame in solved:
            cam_idx, result = solved[frame]
            # Pass 1 only stores non-degenerate results, which always carry
            # R/s/t together (see mesh_alignment.AlignmentResult).
            assert result.R is not None and result.s is not None and result.t is not None
            R, s, t = result.R, result.s, result.t
            mesh_path = _mesh_frame_path(cam_idx, frame)
            if mesh_path is None:
                skip_reasons["missing_mesh_frame"] = skip_reasons.get("missing_mesh_frame", 0) + 1
                continue
        else:
            # Choose the camera with the closest solved frame that also has a
            # mesh for THIS frame, then interpolate within that camera alone.
            best_cam, best_gap, best_path = None, None, None
            for cam_idx in range(len(bundles)):
                if not solved_frames_by_camera[cam_idx]:
                    continue
                mesh_path = _mesh_frame_path(cam_idx, frame)
                if mesh_path is None:
                    continue
                before, after = _neighbours(cam_idx, frame)
                gaps = [abs(frame - n[0]) for n in (before, after) if n is not None]
                if not gaps:
                    continue
                gap = min(gaps)
                if best_gap is None or gap < best_gap:
                    best_cam, best_gap, best_path = cam_idx, gap, mesh_path
            if best_cam is None:
                reason = (
                    "missing_target"
                    if np.isnan(target_by_frame[frame]).any()
                    else "no_valid_camera"
                )
                skip_reasons[reason] = skip_reasons.get(reason, 0) + 1
                continue
            before, after = _neighbours(best_cam, frame)
            transform = interpolate_similarity_transform(before, after, frame)
            if transform is None:
                skip_reasons["no_valid_camera"] = skip_reasons.get("no_valid_camera", 0) + 1
                continue
            R, s, t = transform
            cam_idx, result, mesh_path = best_cam, None, best_path
            interpolated = True
            interpolated_frames += 1

        # _mesh_frame_path() only returns a path for a camera whose faces are
        # loaded, so this is never None on either branch above.
        faces = faces_by_camera[cam_idx]
        assert faces is not None and mesh_path is not None
        vertices = read_obj_vertices(mesh_path)
        transformed = apply_similarity_transform(vertices, R, s, t)
        # Always the raw (x, y, z) DLT/world frame, independent of swap_yz --
        # see the docstring above for why: the mesh-sequence add-on most
        # people actually view this with does no axis conversion of its own.
        out_faces = faces
        out_path = output_mesh_dir / f"frame_{frame:06d}.{export_fmt}"
        writer(out_path, transformed, out_faces)

        if previous_camera is not None and previous_camera != cam_idx:
            camera_switches += 1
        previous_camera = cam_idx
        frames_written += 1
        manifest_rows.append(
            {
                "frame": frame,
                "camera_index": cam_idx,
                "mean_residual_m": result.mean_residual if result is not None else np.nan,
                "rms_residual_m": result.rms_residual if result is not None else np.nan,
                "max_residual_m": result.max_residual if result is not None else np.nan,
                "n_fit_points": result.n_points if result is not None else 0,
                "interpolated": int(interpolated),
            }
        )

    manifest_path = Path(output_dir) / f"{file_base}_mesh_alignment.csv"
    manifest_df = pd.DataFrame(manifest_rows)
    manifest_df.to_csv(manifest_path, index=False, float_format="%.6f")
    _write_mesh_import_readme(output_mesh_dir, export_fmt, swap_yz, frames_written)

    # Interpolated frames carry no residual of their own (NaN in the manifest);
    # the statistics below describe the directly solved frames only.
    residuals = manifest_df["mean_residual_m"].to_numpy() if frames_written else np.array([])
    residuals = residuals[np.isfinite(residuals)]
    gap_frames = [
        int(f) for f in sorted(set(common_frames) - {int(row["frame"]) for row in manifest_rows})
    ]
    summary = {
        "frames_written": frames_written,
        "frames_interpolated": interpolated_frames,
        "frames_skipped": sum(skip_reasons.values()),
        "skip_reasons": skip_reasons,
        "camera_switches": camera_switches,
        "manifest_path": str(manifest_path),
        "output_mesh_dir": str(output_mesh_dir),
        "sequence_contiguous": not gap_frames,
        "missing_frames": gap_frames,
        "mean_residual_m": float(residuals.mean()) if residuals.size else None,
        "median_residual_m": float(np.median(residuals)) if residuals.size else None,
        "max_residual_m": float(residuals.max()) if residuals.size else None,
    }
    print("\n=== Mesh Alignment Complete ===")
    print(f"Frames written: {frames_written}")
    print(f"Frames with interpolated placement: {interpolated_frames}")
    print(f"Frames skipped: {summary['frames_skipped']} ({skip_reasons})")
    print(f"Camera switches: {camera_switches}")
    if gap_frames:
        print(
            f"[yellow]Sequence has {len(gap_frames)} gap(s) — a Blender OBJ-sequence "
            f"import will drift out of sync with the C3D/BVH.[/yellow]"
        )
    else:
        print("Sequence is gap-free (frame N of the mesh == frame N of the C3D/BVH)")
    if residuals.size:
        print(
            f"Residual (m): mean={summary['mean_residual_m']:.4f} "
            f"median={summary['median_residual_m']:.4f} max={summary['max_residual_m']:.4f}"
        )
    print(f"Mesh sequence: {output_mesh_dir}")
    print(f"Alignment manifest: {manifest_path}")
    return summary


def run_reconstruction(
    dlt_files,
    pixel_files,
    output_directory,
    point_rate,
    gui=True,
    swap_yz=True,
    skeleton_json_path=None,
    mesh_source_dirs=None,
    export_mesh="none",
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

    # Load pixel coordinate data for each camera.
    # Column labels are NOT inspected here: column 0 is the frame identifier
    # and every pair of columns after that is one marker's (x, y), regardless
    # of what the header text says (vailá p1_x/p1_y, SAM3, YOLO, MediaPipe...).
    print("Loading pixel coordinate data...")
    pixel_frames_list = []
    pixel_xy_list = []
    for file in pixel_files:
        try:
            frames_arr, xy_arr = load_pixel_csv_positional(file)
        except Exception as e:
            return _err(f"Pixel coordinate file {os.path.basename(file)}: {e}")
        pixel_frames_list.append(frames_arr)
        pixel_xy_list.append(xy_arr)

    num_markers = min(xy.shape[1] for xy in pixel_xy_list)
    if any(xy.shape[1] != num_markers for xy in pixel_xy_list):
        print(
            f"Warning: pixel files have different marker counts; "
            f"using the smallest common count: {num_markers}"
        )
    print(f"Detected {num_markers} markers for 3D reconstruction")

    common_frames = find_common_frames(pixel_frames_list)
    if common_frames.size == 0:
        return _err("No common frames found among pixel files!")
    print(f"Processing {len(common_frames)} common frames...")

    total_frames = len(common_frames)
    total_cols = 1 + (num_markers * 3)
    print(f"Pre-allocating array for {total_frames} frames x {num_markers} markers...")
    reconstruction_array = np.full((total_frames, total_cols), np.nan, dtype=np.float64)
    reconstruction_array[:, 0] = common_frames

    pixel_frame_to_row = [{int(f): i for i, f in enumerate(frames)} for frames in pixel_frames_list]

    progress_step = max(1, total_frames // 20)
    for frame_idx, frame in enumerate(common_frames):
        if frame_idx % progress_step == 0:
            progress = (frame_idx / total_frames) * 100
            print(f"Progress: {progress:.1f}% ({frame_idx}/{total_frames} frames)")

        frame_int = int(frame)
        pixel_row_for_cam = [
            pixel_frame_to_row[c].get(frame_int) for c in range(len(pixel_xy_list))
        ]
        if any(r is None for r in pixel_row_for_cam):
            continue

        for marker in range(num_markers):
            pixel_obs_list = []
            valid_marker = True
            for cam_idx, row_idx in enumerate(pixel_row_for_cam):
                x_obs, y_obs = pixel_xy_list[cam_idx][row_idx, marker]
                if np.isnan(x_obs) or np.isnan(y_obs):
                    valid_marker = False
                    break
                pixel_obs_list.append((float(x_obs), float(y_obs)))

            col_start = 1 + marker * 3
            if not valid_marker or len(pixel_obs_list) != len(dlt_params_list):
                pass
            else:
                point3d = rec3d_multicam(dlt_params_list, pixel_obs_list)
                reconstruction_array[frame_idx, col_start : col_start + 3] = point3d

    print("3D reconstruction completed!")

    header = ["frame"]
    for marker in range(1, num_markers + 1):
        header.extend([f"p{marker}_x", f"p{marker}_y", f"p{marker}_z"])

    rec3d_df = pd.DataFrame(reconstruction_array, columns=header)  # type: ignore
    valid_frames_mask = ~rec3d_df.iloc[:, 1:].isna().all(axis=1)
    rec3d_df = rec3d_df[valid_frames_mask].reset_index(drop=True)
    rec3d_df["frame"] = rec3d_df["frame"].astype(int)

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

    # ---> NEW: Optional aligned mesh-for-Blender export <---
    if mesh_source_dirs and export_mesh != "none":
        print("\nAligning and exporting mesh sequence...")
        try:
            reconstruct_mesh_sequence(
                rec3d_df,
                mesh_source_dirs,
                new_dir,
                file_base,
                export_fmt=export_mesh,
                swap_yz=swap_yz,
            )
        except Exception as e:
            print(f"[red]Mesh export failed: {e}[/red]")
            if gui:
                messagebox.showerror("Mesh Export Error", f"Mesh export failed: {e}")

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
    blender_script = generate_blender_companion_script(
        new_dir,
        file_base,
        skeleton_json_path,
        point_rate=point_rate,
        n_frames=len(rec3d_df),
        mesh_dir=(f"meshes_{export_mesh}" if mesh_source_dirs and export_mesh != "none" else None),
        unreconstructed_markers=find_unreconstructed_markers(rec3d_df),
    )
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


def _build_cli_command(
    dlt_files,
    pixel_files,
    output_directory,
    point_rate,
    swap_yz,
    skeleton_json_path,
    mesh_source_dirs,
    export_mesh,
):
    """
    Build the CLI command equivalent to a GUI run, so the terminal always
    shows a copy-pasteable way to repeat the same reconstruction headlessly.

    `pixel_files` may be None when --mesh-source-dir already supplies them
    (each Visualize-ID run directory contains its own *_markers.csv), keeping
    the printed command as short as the simplified CLI path allows.
    """
    parts = ["uv run python -m vaila.rec3d_one_dlt3d", f"--dlt3d {' '.join(dlt_files)}"]
    if pixel_files:
        parts.append(f"--pixels {' '.join(pixel_files)}")
    parts.append(f"--fps {point_rate}")
    parts.append(f"-o {output_directory}")
    # --swap-yz is the default; only the opt-out needs to appear explicitly.
    if not swap_yz:
        parts.append("--no-swap-yz")
    if skeleton_json_path:
        parts.append(f"--skeleton {skeleton_json_path}")
    if mesh_source_dirs:
        parts.append(f"--mesh-source-dir {' '.join(mesh_source_dirs)}")
        parts.append(f"--export-mesh {export_mesh}")
    return " ".join(parts)


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

    # Step 2: Select pixel coordinate CSV files, one per camera -- OR, if a
    # SAM3+DINOv3 run is being used, derive them from --mesh-source-dir below
    # instead (each Visualize-ID directory already contains its own
    # *_markers.csv, so there is no need to pick the same directory twice).
    print("Step 2: Pixel source...")
    mesh_source_dirs = None
    use_mesh_run_dirs = messagebox.askyesno(
        "Pixel Source",
        "Use per-camera SAM3+DINOv3 'Visualize ID' run directories as the "
        "pixel source?\n\n"
        "YES: pick one run directory per camera (from "
        "sam3dinov3_visualize.py) -- its own markers.csv is used "
        "automatically, and you can optionally also export the aligned "
        "mesh from the same directories.\n\n"
        "NO: pick a plain pixel-coordinate CSV per camera instead (the "
        "original flow; no mesh export available this way).",
    )
    if use_mesh_run_dirs:
        mesh_source_dirs = []
        for i in range(1, n_cameras + 1):
            path = filedialog.askdirectory(
                title=f"Select SAM3+DINOv3 'Visualize ID' run directory for camera {i}"
            )
            if not path:
                messagebox.showerror("Error", f"No run directory selected for camera {i}!")
                root.destroy()
                return
            mesh_source_dirs.append(path)

        pixel_files = []
        for mesh_dir in mesh_source_dirs:
            markers_csv = find_markers_csv_in_dir(mesh_dir)
            if markers_csv is None:
                messagebox.showerror(
                    "Error", f"No *_markers.csv found in {mesh_dir} -- cannot continue."
                )
                root.destroy()
                return
            pixel_files.append(str(markers_csv))
    else:
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
    point_rate = simpledialog.askfloat(
        "Data Frequency",
        "Enter the point data rate (Hz), e.g. 119.88012001 for a real NTSC-derived rate:",
        minvalue=0.0001,
        initialvalue=100.0,
    )
    if point_rate is None:
        messagebox.showerror("Error", "Point data rate is required. Operation cancelled.")
        return

    # Step 5: Ask if user wants to swap Y and Z axes for Blender (default YES)
    swap_yz = messagebox.askyesno(
        "Blender Axis Export",
        "Swap Y and Z axes for the BVH and mesh output?\n\n"
        "YES (recommended, default): height becomes vertical (Z-up) in "
        "Blender, and the BVH and mesh stay in the same axis convention.\n"
        "NO: keep the original DLT coordinates.",
        default=messagebox.YES,
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

    # Step 7: (Optional) Aligned mesh-for-Blender export
    print("Step 7: (Optional) Aligned mesh-for-Blender export...")
    export_mesh = "none"
    if mesh_source_dirs:
        # Step 2 already selected the run directories (simplified path) --
        # just ask whether to also export the aligned mesh from them.
        use_mesh = messagebox.askyesno(
            "Mesh Export",
            "Also export an aligned 3D body mesh sequence for Blender from "
            "the same SAM3+DINOv3 run directories?",
        )
        if use_mesh:
            export_mesh = (
                "obj" if messagebox.askyesno("Mesh Format", "Export as OBJ? (No = PLY)") else "ply"
            )
    else:
        use_mesh = messagebox.askyesno(
            "Mesh Export",
            "Export an aligned 3D body mesh sequence for Blender?\n\n"
            "Requires one sam3dinov3_visualize.py 'Visualize ID' output directory "
            "per camera (containing *_mhr70_rec3d.csv, meshes_obj/ or meshes_ply/, "
            "and mesh_faces.npy).",
        )
        if use_mesh:
            mesh_source_dirs = []
            for i in range(1, n_cameras + 1):
                path = filedialog.askdirectory(
                    title=f"Select mesh-source directory for camera {i} (Visualize ID output)"
                )
                if not path:
                    messagebox.showwarning(
                        "Mesh Export Cancelled",
                        "Mesh-source directory missing; skipping mesh export.",
                    )
                    mesh_source_dirs = None
                    break
                mesh_source_dirs.append(path)
            if mesh_source_dirs:
                export_mesh = (
                    "obj"
                    if messagebox.askyesno("Mesh Format", "Export as OBJ? (No = PLY)")
                    else "ply"
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
    print(
        f"  - Mesh export: {export_mesh} ({len(mesh_source_dirs) if mesh_source_dirs else 0} camera(s))"
    )
    print("-" * 80)

    cli_cmd = _build_cli_command(
        dlt_files,
        pixel_files if not use_mesh_run_dirs else None,
        output_directory,
        point_rate,
        swap_yz,
        skeleton_json_path,
        mesh_source_dirs,
        export_mesh,
    )
    print(f">> {cli_cmd}")

    run_reconstruction(
        dlt_files,
        pixel_files,
        output_directory,
        point_rate,
        gui=True,
        swap_yz=swap_yz,
        skeleton_json_path=skeleton_json_path,
        mesh_source_dirs=mesh_source_dirs,
        export_mesh=export_mesh,
    )

    # Repeat the equivalent CLI command LAST, after all the processing output,
    # so it is the final thing on screen and easy to copy for a headless re-run.
    print("\n" + "=" * 80)
    print("Equivalent CLI command for this run (copy/paste to repeat headlessly):")
    print("=" * 80)
    print(f">> {cli_cmd}")
    print("=" * 80)

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
        help=(
            "Pixel coordinate CSV files, one per camera (order must match --dlt3d). "
            "Optional if --mesh-source-dir is given: each mesh-source directory's own "
            "*_markers.csv is used automatically."
        ),
    )
    parser.add_argument(
        "--fps",
        type=float,
        default=100.0,
        metavar="HZ",
        help=(
            "Point data rate in Hz for C3D/CSV (default: 100). Accepts fractional "
            "rates, e.g. 119.88012001 for NTSC-derived 120000/1001 capture."
        ),
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
        dest="swap_yz",
        action="store_true",
        default=True,
        help=(
            "Swap Y and Z axes in the BVH and mesh output so height ends up "
            "vertical (Z-up) in Blender. This is the DEFAULT; the flag is kept "
            "for backward compatibility and as an explicit opt-in."
        ),
    )
    parser.add_argument(
        "--no-swap-yz",
        dest="swap_yz",
        action="store_false",
        help="Keep the raw DLT axes in the BVH and mesh output (no Y/Z swap)",
    )
    parser.add_argument(
        "--skeleton",
        metavar="FILE",
        help="Path to Skeleton Pose JSON file (defines connections for Blender visualization)",
    )
    parser.add_argument(
        "--mesh-source-dir",
        nargs="+",
        metavar="DIR",
        dest="mesh_source_dir",
        help=(
            "Per-camera sam3dinov3_visualize.py 'Visualize ID' output directory "
            "(order must match --dlt3d/--pixels); each must contain "
            "*_mhr70_rec3d.csv, meshes_obj/ or meshes_ply/, and mesh_faces.npy"
        ),
    )
    parser.add_argument(
        "--export-mesh",
        choices=["none", "obj", "ply"],
        default="none",
        help="Export an aligned per-frame mesh sequence for Blender (requires --mesh-source-dir)",
    )
    args = parser.parse_args()

    have_pixel_source = args.pixels or args.mesh_source_dir
    if args.gui or (not args.dlt3d and not have_pixel_source and not args.output):
        run_rec3d_one_dlt3d()
        return

    if not args.dlt3d or not have_pixel_source:
        # Check if user only provided --gui (already handled) but maybe they provided partial args
        if args.gui:
            run_rec3d_one_dlt3d()
            return
        print(
            "Error: CLI mode requires --dlt3d and (--pixels or --mesh-source-dir).", file=sys.stderr
        )
        sys.exit(1)
    if not args.output:
        print("Error: CLI mode requires --output.", file=sys.stderr)
        sys.exit(1)
    if args.mesh_source_dir and len(args.mesh_source_dir) != len(args.dlt3d):
        print(
            "Error: Number of --mesh-source-dir entries must match number of --dlt3d files.",
            file=sys.stderr,
        )
        sys.exit(1)
    if args.export_mesh != "none" and not args.mesh_source_dir:
        print("Error: --export-mesh requires --mesh-source-dir.", file=sys.stderr)
        sys.exit(1)

    pixel_files = args.pixels
    if not pixel_files:
        # Simplified single-argument-per-camera path: each --mesh-source-dir
        # is a sam3dinov3_visualize.py "Visualize ID" output directory, which
        # already contains its own *_markers.csv alongside the mesh data —
        # no need to also pass --pixels pointing at the same directory.
        print("No --pixels given; deriving pixel files from --mesh-source-dir...")
        pixel_files = []
        for mesh_dir in args.mesh_source_dir:
            markers_csv = find_markers_csv_in_dir(mesh_dir)
            if markers_csv is None:
                print(f"Error: could not find *_markers.csv in {mesh_dir}", file=sys.stderr)
                sys.exit(1)
            pixel_files.append(str(markers_csv))
            print(f"  {mesh_dir} -> {markers_csv.name}")

    if len(args.dlt3d) != len(pixel_files):
        print(
            "Error: Number of --dlt3d files must match number of pixel files "
            "(from --pixels, or derived from --mesh-source-dir).",
            file=sys.stderr,
        )
        sys.exit(1)

    result = run_reconstruction(
        args.dlt3d,
        pixel_files,
        os.path.abspath(args.output),
        args.fps,
        gui=False,
        swap_yz=args.swap_yz,
        skeleton_json_path=args.skeleton,
        mesh_source_dirs=args.mesh_source_dir,
        export_mesh=args.export_mesh,
    )
    if result is None:
        sys.exit(1)
    print(f"Output directory: {result[0]}")


if __name__ == "__main__":
    _cli_run()
