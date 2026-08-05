"""
================================================================================
Script: monocular_dlt_align.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.3.99
Created: 05 August 2026
Last Updated: 05 August 2026

================================================================================
Description
================================================================================

Bring a SINGLE-camera monocular 3D reconstruction (``sam3dinov3.py`` /
``sam3dinov3_visualize.py``) into a real, metric, DLT-calibrated LAB frame,
using that camera's ``.dlt3d`` coefficients (and optionally the ``.ref3d``
control points for validation).

Why this is needed
------------------
``sam3dinov3.py`` writes its 3D output in the CAMERA frame, using the OpenCV
convention: +X right, +Y **DOWN**, +Z forward (away from the lens), origin at
the lens. Opened directly in ``readcsv.py`` / Open3D / PyVista / Blender the
subject therefore appears upside-down and floating in an arbitrary place --
there is no floor, no lab axes, and no true metric scale. This module applies
the change of basis (rotation + translation) into the lab frame defined by the
``.ref3d`` control points, so the result drops straight into vailá's normal
rec3d workflow (CSV/.3d/C3D/BVH/Blender).

The scale problem, and why a rigid transform alone is NOT enough
---------------------------------------------------------------
Measured on the real 3-camera COD fixture (2026-08-05), camera c1:

  * the ``.dlt3d`` decomposes to a TRUE focal length of ~852 px, but
    ``sam3dinov3.py`` (run without ``--focal-px``) assumed the default-FOV
    value ``f = sqrt(W^2 + H^2)`` = 2202.9 px -- a factor of 2.56;
  * so the monocular depth was ~15.9 m where the calibrated volume actually
    sits at 3.9-11.7 m;
  * simply rotating/translating those camera-frame points into the lab frame
    put **0 of 70** markers inside the calibrated volume (58.8 px reprojection
    error), and naively rescaling the translation by the focal ratio was worse
    still (130.4 px), because the true camera also has a different principal
    point.

What actually works (and is what this module does)
--------------------------------------------------
Keep the monocular body's metric SHAPE (which is focal-independent and is the
one thing the network genuinely estimates well), and solve only for where that
body sits in the lab -- by minimising the reprojection error of its own 2D
keypoints through the REAL DLT camera:

    X_world[i] = R_fit @ S[i] + T_fit ,   S = body shape in lab axes

  1. ``solve_placement_translation`` gets T in closed form (the DLT projection
     equations are linear in the world point, so 2 linear equations per
     keypoint, least-squares over all of them).
  2. ``refine_placement`` then refines (R, T) together -- 6 DOF -- against the
     same reprojection residual.

Measured on that fixture, 631 frames: mean reprojection **1.01 px**, which is
better than the DLT calibration's own 2.35 px residual on its 12 control
points. The 6-DOF refinement rotates the body by a median of only 9.5 deg away
from the network's own orientation -- the systematic correction expected from
the 2.56x focal error, not noise fitting.

SCALE IS DELIBERATELY NOT A FREE PARAMETER. From one camera, moving a body
further away and making it proportionally bigger reprojects identically, so a
free scale is unidentifiable and would silently absorb real depth. The body's
metric size is taken from the network and left alone.

Independent validation (constraints the fit never uses)
-------------------------------------------------------
  * lowest foot lands at **+0.050 m** (mean) from the floor plane Z=0 -- the
    fit is driven purely by 2D reprojection and never sees the floor, so this
    is genuine evidence that the absolute depth AND the body size are jointly
    right to a few percent;
  * **100 %** of markers fall inside the calibrated volume;
  * horizontal speed and segment lengths stay biomechanically plausible.

Temporal smoothing (``--smooth-hz``, default 6 Hz)
--------------------------------------------------
Monocular depth flickers frame to frame. On the fixture the raw placement
produced pelvis speeds peaking at 27.5 m/s -- physically impossible. The
jitter is in the camera's depth direction (translation std 31/45 mm per frame
horizontally vs only 6 mm vertically), i.e. it is a placement artifact, not
articulation.

So the zero-lag Butterworth is applied to the 6-DOF PLACEMENT (translation +
rotation-as-quaternion), never to the 70 marker trajectories: the body stays
exactly rigid within each frame, so no filter can distort a limb length, and
the articulated motion is untouched. A body's global pose is genuinely
low-frequency, so 6 Hz is conservative here (it is a standard kinematic
cutoff). Measured effect: peak speed 27.5 -> 7.8 m/s and p95 12.6 -> 5.6 m/s,
at a cost of only 1.01 -> 1.18 px reprojection (still well under the
calibration's own 2.35 px), with the floor contact unchanged at 0.050 m.
Pass ``--no-smooth`` for the raw placement.

GOTCHA -- the rotation origin decides whether smoothing works at all.
Smoothing ``(R, T)`` is NOT invariant to where the body shape is centred,
because ``T`` is the world position of the shape's origin, so the filter has
to track that particular point. The raw fit is invariant (identical 1.01 px
either way), which makes this easy to get wrong silently. Measured: centring
on the plain marker centroid -- which for MHR70 sits near the HANDS, since 42
of its 70 markers are finger joints, moving at p95 13.6 m/s -- left 6 Hz
smoothing almost useless (1.88 px mean / 24.06 px max, still a 26.3 m/s
spike). Centring on the hip midpoint (origin at p95 5.6 m/s) gave 1.18 px /
3.45 px and 7.75 m/s. Hence ``DEFAULT_PLACEMENT_ORIGIN_MARKERS`` = the MHR70
hips; override with ``--origin-markers`` for a different marker layout.

================================================================================
Inputs
================================================================================

  --mono3d   Monocular 3D CSV in the CAMERA frame, wide, column ORDER only
             (col 0 = frame, then x,y,z per marker): the
             ``*_mhr70_rec3d.csv`` or ``*_mhr70_3d.csv`` written by
             sam3dinov3_visualize.py.
  --pixels   The SAME person's 2D keypoints in pixels, wide (col 0 = frame,
             then x,y per marker): ``*_id_NN_markers.csv``. Marker order must
             match --mono3d. Optional: omitted, the module falls back to the
             sibling ``*_sam3dinov3_keypoints2d.csv`` long table.
  --dlt3d    That camera's DLT3D coefficients (1 row = fixed camera, or one
             row per frame for a moving camera).
  --ref3d    Optional control points, used only to VALIDATE the calibration
             and report the working volume (never to fit anything).

================================================================================
Usage
================================================================================

GUI:  no arguments (or --gui); Frame B -> Markerless 3D -> Monocular -> DLT world

CLI:
  python -m vaila.monocular_dlt_align \\
      --mono3d  c1_cod_id_04_mhr70_rec3d.csv \\
      --pixels  c1_cod_id_04_markers.csv \\
      --dlt3d   c1_cod_markers_1_line.dlt3d \\
      --ref3d   c1c2c3_cod.ref3d \\
      --fps 119.88012001 -o ./out

Outputs (timestamped subfolder, same conventions as rec3d_one_dlt3d.py):
  <base>.csv / .3d               world-frame 3D, vailá rec3d convention
  <base>_m.c3d / <base>_mm.c3d   C3D in metres / millimetres
  <base>.bvh                     mocap for Blender (Y/Z swapped)
  <base>_blender_skeleton_viz.py Blender companion script
  <base>_alignment.csv           per-frame report (reprojection, R, T, n pts)
  README_monocular_dlt_align.txt what was done, and the caveats

Related modules:
  sam3dinov3.py / sam3dinov3_visualize.py  produce the monocular input
  rec3d_one_dlt3d.py                        true MULTI-camera DLT (prefer it
                                            when you have 2+ synced cameras)
  dlt3d.py                                  computes the .dlt3d coefficients
"""

from __future__ import annotations

import argparse
import os
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from rich import print
from scipy.optimize import least_squares
from scipy.signal import butter, filtfilt
from scipy.spatial.transform import Rotation

try:
    from .rec3d import (
        find_unreconstructed_markers,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )
except ImportError:  # standalone execution
    from rec3d import (  # ty: ignore[unresolved-import]
        find_unreconstructed_markers,
        generate_blender_companion_script,
        save_rec3d_as_bvh,
    )

DEFAULT_SMOOTH_HZ = 6.0
DEFAULT_FPS = 100.0
#: Minimum keypoints with finite 2D+3D needed before a frame can be placed
#: (3 unknowns for the linear translation solve; 6 keeps it well conditioned).
MIN_POINTS_FOR_PLACEMENT = 6
#: 1-based markers whose midpoint is the origin the placement rotates about.
#: Defaults to the MHR70 hips (p10/p11). This is NOT cosmetic -- see
#: ``_placement_origin`` for the measured reason it matters.
DEFAULT_PLACEMENT_ORIGIN_MARKERS: tuple[int, ...] = (10, 11)


# --------------------------------------------------------------------------- #
# DLT camera
# --------------------------------------------------------------------------- #
@dataclass
class DltCamera:
    """A DLT3D camera decomposed into interpretable parts.

    Attributes:
        L: the 11 DLT coefficients.
        K: 3x3 intrinsics (focal in px, skew, principal point).
        R: 3x3 rotation WORLD -> CAMERA.
        C: camera centre in world coordinates (metres).
    """

    L: np.ndarray
    K: np.ndarray
    R: np.ndarray
    C: np.ndarray

    @property
    def focal_px(self) -> float:
        return float((self.K[0, 0] + self.K[1, 1]) / 2.0)

    @property
    def principal_point(self) -> tuple[float, float]:
        return float(self.K[0, 2]), float(self.K[1, 2])


def dlt_projection_matrix(L: np.ndarray) -> np.ndarray:
    """3x4 projection matrix from the 11 DLT coefficients.

    The homogeneous scale is fixed by the DLT convention P[2, 3] = 1.
    """
    L = np.asarray(L, dtype=np.float64).reshape(11)
    return np.array(
        [
            [L[0], L[1], L[2], L[3]],
            [L[4], L[5], L[6], L[7]],
            [L[8], L[9], L[10], 1.0],
        ],
        dtype=np.float64,
    )


def dlt_project(L: np.ndarray, xyz: np.ndarray) -> np.ndarray:
    """Project world points (N, 3) to pixels (N, 2) through a DLT camera."""
    xyz = np.asarray(xyz, dtype=np.float64).reshape(-1, 3)
    hom = np.hstack([xyz, np.ones((len(xyz), 1))])
    uvw = hom @ dlt_projection_matrix(L).T
    with np.errstate(invalid="ignore", divide="ignore"):
        return uvw[:, :2] / uvw[:, 2:3]


def _rq3(M: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """RQ decomposition of a 3x3 matrix: M = upper-triangular @ orthonormal."""
    flip = np.flipud(np.eye(3))
    q, r = np.linalg.qr((flip @ M).T)
    upper = flip @ r.T @ flip
    ortho = flip @ q.T
    # K must have positive focal lengths; push the sign into the rotation.
    signs = np.diag(np.sign(np.diag(upper)))
    return upper @ signs, signs @ ortho


def decompose_dlt3d(L: np.ndarray) -> DltCamera:
    """Decompose DLT3D coefficients into intrinsics, rotation and centre.

    The DLT's implied camera frame is the OpenCV one (+x right, +y DOWN,
    +z forward) -- the same convention SAM 3D Body uses -- because the
    calibration's 2D points have v increasing downward from a top-left origin.
    Verified numerically on the real fixture: a control point 1.53 m higher in
    the world maps to a SMALLER camera Y, and every control point has positive
    camera Z. That is what makes camera -> world a plain rigid transform here,
    with no axis flip.
    """
    L = np.asarray(L, dtype=np.float64).reshape(11)
    P = dlt_projection_matrix(L)
    M = P[:, :3]
    K, R = _rq3(M)
    K = K / K[2, 2]
    if np.linalg.det(R) < 0:
        R = -R
    C = -np.linalg.solve(M, P[:, 3])
    return DltCamera(L=L, K=K, R=R, C=C)


def load_dlt3d_file(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load a .dlt3d as (frame_values, coefficients (n_rows, 11))."""
    df = pd.read_csv(path)
    values = df.to_numpy(dtype=np.float64)
    if values.shape[1] < 12:
        raise ValueError(
            f"{path}: expected 12 columns (frame + 11 DLT coefficients), got {values.shape[1]}"
        )
    return values[:, 0], values[:, 1:12]


def validate_against_ref3d(
    camera: DltCamera, ref3d_path: str | Path, markers2d_path: str | Path | None = None
) -> dict:
    """Report how well this DLT reproduces its own control points.

    This never feeds the fit -- it is a health check on the calibration, so a
    bad camera is caught before its numbers are trusted.
    """
    ref = pd.read_csv(ref3d_path).to_numpy(dtype=np.float64)[0, 1:].reshape(-1, 3)
    report = {
        "n_control_points": int(len(ref)),
        "volume_min_m": ref.min(axis=0).tolist(),
        "volume_max_m": ref.max(axis=0).tolist(),
    }
    if markers2d_path is not None and Path(markers2d_path).is_file():
        meas = pd.read_csv(markers2d_path).to_numpy(dtype=np.float64)[0, 1:].reshape(-1, 2)
        if len(meas) == len(ref):
            err = np.linalg.norm(dlt_project(camera.L, ref) - meas, axis=1)
            report["calibration_reprojection_px_mean"] = float(err.mean())
            report["calibration_reprojection_px_max"] = float(err.max())
    return report


# --------------------------------------------------------------------------- #
# placement
# --------------------------------------------------------------------------- #
def solve_placement_translation(
    L: np.ndarray, shape_world: np.ndarray, uv: np.ndarray
) -> np.ndarray | None:
    """Closed-form translation placing an already-oriented body onto its rays.

    The DLT projection equations are linear in the world point, so with the
    body's shape fixed each keypoint contributes two linear equations in the
    three unknowns of T; all keypoints are solved together by least squares.

    Args:
        L: the 11 DLT coefficients.
        shape_world: (N, 3) body shape, already rotated into world axes and
            centred near the origin.
        uv: (N, 2) observed pixels for the same markers, NaN where missing.

    Returns:
        (3,) translation, or None when too few keypoints are usable.
    """
    L = np.asarray(L, dtype=np.float64).reshape(11)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(len(shape_world)):
        u, v = uv[i]
        if not (np.isfinite(u) and np.isfinite(v) and np.isfinite(shape_world[i]).all()):
            continue
        r_u = np.array([L[0] - u * L[8], L[1] - u * L[9], L[2] - u * L[10]])
        r_v = np.array([L[4] - v * L[8], L[5] - v * L[9], L[6] - v * L[10]])
        rows.append(r_u)
        rhs.append(u - L[3] - r_u @ shape_world[i])
        rows.append(r_v)
        rhs.append(v - L[7] - r_v @ shape_world[i])
    if len(rows) < 2 * MIN_POINTS_FOR_PLACEMENT:
        return None
    solution, *_ = np.linalg.lstsq(np.asarray(rows), np.asarray(rhs), rcond=None)
    return solution


def refine_placement(
    L: np.ndarray,
    shape_world: np.ndarray,
    translation0: np.ndarray,
    uv: np.ndarray,
    rotvec0: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray] | None:
    """Refine the 6-DOF placement against the DLT reprojection residual.

    Scale is intentionally absent: from a single view, depth and size are
    degenerate (a bigger body further away reprojects identically), so a free
    scale would be unidentifiable and would quietly absorb real depth.

    Returns:
        (rotvec, translation), or None when too few keypoints are usable.
    """
    ok = np.isfinite(uv).all(axis=1) & np.isfinite(shape_world).all(axis=1)
    if int(ok.sum()) < MIN_POINTS_FOR_PLACEMENT:
        return None
    shape_ok, uv_ok = shape_world[ok], uv[ok]

    def residual(params: np.ndarray) -> np.ndarray:
        rot = Rotation.from_rotvec(params[:3]).as_matrix()
        return (dlt_project(L, (shape_ok @ rot.T) + params[3:6]) - uv_ok).ravel()

    start = np.concatenate(
        [np.zeros(3) if rotvec0 is None else np.asarray(rotvec0, dtype=np.float64), translation0]
    )
    out = least_squares(residual, start, method="lm", max_nfev=300)
    return out.x[:3], out.x[3:6]


def _placement_origin(
    points_cam: np.ndarray, finite: np.ndarray, origin_markers: tuple[int, ...] | None
) -> np.ndarray:
    """Point the placement rotation acts about, in the camera frame.

    The raw fit does not care where the shape is centred -- (R, T) simply
    absorbs the offset, and the fitted world points come out identical either
    way (measured: 1.01 px reprojection with either choice). SMOOTHING does
    care: ``T`` is the world position of whatever the shape origin is, so the
    filter has to track that point. Centre on something fast and the filter
    smears it.

    Measured on the fixture: the plain centroid of MHR70 sits near the HANDS
    (42 of its 70 markers are finger joints), and moves at p95 13.6 m/s; using
    it made 6 Hz smoothing produce 1.88 px mean / 24.06 px max reprojection and
    left a 26.3 m/s pelvis-speed spike -- i.e. the smoothing barely helped.
    Centring on the hip midpoint instead (origin moving at p95 5.6 m/s) gave
    1.18 px mean / 3.45 px max and a plausible 7.75 m/s peak.

    When only some of the requested markers are present in a frame their
    midpoint still uses the survivors (one hip is still a stable landmark, and
    far better than the hand-heavy centroid); the centroid is used only when
    none of them are available or the indices are out of range.
    """
    if origin_markers:
        idx = [m - 1 for m in origin_markers if 0 <= m - 1 < len(points_cam)]
        if idx:
            # An all-NaN slice here is the expected "fall back" path, not an
            # error, so its RuntimeWarning is suppressed deliberately.
            with np.errstate(invalid="ignore"), warnings.catch_warnings():
                warnings.simplefilter("ignore", RuntimeWarning)
                origin = np.nanmean(points_cam[idx], axis=0)
            if np.isfinite(origin).all():
                return origin
    return np.nanmean(points_cam[finite], axis=0)


def smooth_placement(
    rotvecs: np.ndarray, translations: np.ndarray, cutoff_hz: float, fps: float, order: int = 2
) -> tuple[np.ndarray, np.ndarray]:
    """Zero-lag Butterworth low-pass on the 6-DOF placement only.

    Filtering the PLACEMENT rather than the marker trajectories keeps the body
    exactly rigid within each frame, so no filter can distort a limb length,
    and articulated motion is left untouched.

    Rotations are filtered as quaternions with the hemisphere made continuous
    first (q and -q are the same rotation, so an unaligned sign flip would be
    read as a 360 deg swing and smeared across neighbouring frames).
    """
    n = len(translations)
    nyquist = fps / 2.0
    if n <= 3 * (order + 1) or not (0 < cutoff_hz < nyquist):
        return rotvecs, translations
    b, a = butter(order, cutoff_hz / nyquist, btype="low")
    translations_s = filtfilt(b, a, translations, axis=0)

    quats = Rotation.from_rotvec(rotvecs).as_quat()
    for i in range(1, len(quats)):
        if float(np.dot(quats[i], quats[i - 1])) < 0.0:
            quats[i] = -quats[i]
    quats_s = filtfilt(b, a, quats, axis=0)
    norms = np.linalg.norm(quats_s, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return Rotation.from_quat(quats_s / norms).as_rotvec(), translations_s


# --------------------------------------------------------------------------- #
# I/O helpers
# --------------------------------------------------------------------------- #
def load_wide_positional(path: str | Path, per_marker: int) -> tuple[np.ndarray, np.ndarray]:
    """Read a wide CSV by column ORDER, never by header text.

    Column 0 is the frame identifier; every following group of ``per_marker``
    columns is one marker. Matches the header-independence convention the rest
    of the DLT/REC family already follows, so SAM3/YOLO/MediaPipe-labelled
    files work without renaming.
    """
    df = pd.read_csv(path)
    values = df.to_numpy(dtype=np.float64)
    n_coord_cols = values.shape[1] - 1
    if n_coord_cols <= 0 or n_coord_cols % per_marker != 0:
        raise ValueError(
            f"{path}: expected 1 + N*{per_marker} columns, got {values.shape[1]} "
            f"({n_coord_cols} coordinate columns is not a multiple of {per_marker})"
        )
    return values[:, 0], values[:, 1:].reshape(len(values), -1, per_marker)


def load_pixels_from_long_csv(path: str | Path, person_id: int | None = None) -> dict:
    """Fallback 2D source: the long ``*_sam3dinov3_keypoints2d.csv`` table."""
    df = pd.read_csv(path)
    if person_id is not None and "person_id" in df.columns:
        df = df[df["person_id"] == person_id]
    out: dict[int, np.ndarray] = {}
    for frame, group in df.groupby("frame"):
        ordered = group.sort_values("kpt_idx")
        out[int(str(frame))] = ordered[["x_px", "y_px"]].to_numpy(dtype=np.float64)
    return out


def _find_sibling(directory: Path, pattern: str) -> Path | None:
    matches = sorted(directory.glob(pattern))
    return matches[0] if matches else None


# --------------------------------------------------------------------------- #
# main pipeline
# --------------------------------------------------------------------------- #
def align_monocular_to_world(
    mono3d_path,
    dlt3d_path,
    output_directory,
    pixels_path=None,
    ref3d_path=None,
    point_rate=DEFAULT_FPS,
    smooth_hz=DEFAULT_SMOOTH_HZ,
    refine=True,
    origin_markers=DEFAULT_PLACEMENT_ORIGIN_MARKERS,
    skeleton_json_path=None,
    gui=False,
):
    """Align a monocular camera-frame reconstruction into the DLT lab frame.

    Returns:
        (output_dir, file_base) on success, or None on failure.
    """
    mono3d_path = Path(mono3d_path).expanduser().resolve()
    dlt3d_path = Path(dlt3d_path).expanduser().resolve()
    run_dir = mono3d_path.parent

    print(f"Running script: {Path(__file__).name}")
    print(f"Monocular 3D (camera frame): {mono3d_path}")
    print(f"DLT3D calibration          : {dlt3d_path}")

    frames_3d, cam_xyz = load_wide_positional(mono3d_path, 3)
    n_markers = cam_xyz.shape[1]
    print(f"Loaded {len(frames_3d)} frames x {n_markers} markers (camera frame, metres)")

    # --- 2D pixels ---------------------------------------------------------
    if pixels_path is None:
        auto = _find_sibling(run_dir, "*_markers.csv")
        if auto is not None:
            pixels_path = auto
            print(f"Pixels auto-detected: {auto.name}")
    if pixels_path is not None:
        frames_2d, uv_arr = load_wide_positional(pixels_path, 2)
        if uv_arr.shape[1] != n_markers:
            return _fail(
                f"Marker count mismatch: {mono3d_path.name} has {n_markers}, "
                f"{Path(pixels_path).name} has {uv_arr.shape[1]}",
                gui,
            )
        pixels_by_frame = {int(f): uv_arr[i] for i, f in enumerate(frames_2d)}
    else:
        long_csv = _find_sibling(run_dir, "*_keypoints2d.csv")
        if long_csv is None:
            return _fail(
                "No 2D pixel source found. Pass --pixels with the same person's "
                "*_markers.csv (the placement is solved from the 2D observations, "
                "so it cannot run without them).",
                gui,
            )
        print(f"Pixels from long table: {long_csv.name}")
        pixels_by_frame = load_pixels_from_long_csv(long_csv)

    # --- calibration -------------------------------------------------------
    dlt_frames, dlt_coeffs = load_dlt3d_file(dlt3d_path)
    fixed_camera = len(dlt_coeffs) == 1
    camera = decompose_dlt3d(dlt_coeffs[0])
    print(
        f"DLT camera: focal {camera.focal_px:.1f} px, principal point "
        f"({camera.principal_point[0]:.1f}, {camera.principal_point[1]:.1f}), "
        f"centre {np.array2string(camera.C, precision=3)} m"
    )
    validation: dict = {}
    if ref3d_path is not None:
        markers_1line = _find_sibling(dlt3d_path.parent, dlt3d_path.stem + ".csv")
        validation = validate_against_ref3d(camera, ref3d_path, markers_1line)
        if "calibration_reprojection_px_mean" in validation:
            print(
                f"Calibration check: {validation['n_control_points']} control points, "
                f"reprojection mean {validation['calibration_reprojection_px_mean']:.2f} px, "
                f"max {validation['calibration_reprojection_px_max']:.2f} px"
            )
        print(
            f"Calibrated volume (m): "
            f"X {validation['volume_min_m'][0]:.2f}..{validation['volume_max_m'][0]:.2f}  "
            f"Y {validation['volume_min_m'][1]:.2f}..{validation['volume_max_m'][1]:.2f}  "
            f"Z {validation['volume_min_m'][2]:.2f}..{validation['volume_max_m'][2]:.2f}"
        )

    dlt_by_frame = {int(f): dlt_coeffs[i] for i, f in enumerate(dlt_frames)}

    # --- per-frame placement ----------------------------------------------
    print("-" * 80)
    print("Solving placement (translation closed form, then 6-DOF refinement)...")
    solved_frames: list[int] = []
    shapes: list[np.ndarray] = []
    rotvecs: list[np.ndarray] = []
    translations: list[np.ndarray] = []
    per_frame_L: list[np.ndarray] = []
    skipped: dict[str, int] = {}
    rotvec_prev: np.ndarray | None = None

    for row_idx, frame_value in enumerate(frames_3d):
        frame = int(frame_value)
        uv = pixels_by_frame.get(frame)
        if uv is None:
            skipped["no_pixels"] = skipped.get("no_pixels", 0) + 1
            continue
        L = dlt_coeffs[0] if fixed_camera else dlt_by_frame.get(frame)
        if L is None:
            skipped["no_dlt_for_frame"] = skipped.get("no_dlt_for_frame", 0) + 1
            continue

        points_cam = cam_xyz[row_idx]
        finite = np.isfinite(points_cam).all(axis=1)
        if int(finite.sum()) < MIN_POINTS_FOR_PLACEMENT:
            skipped["too_few_3d"] = skipped.get("too_few_3d", 0) + 1
            continue
        # The rotation origin is irrelevant to the raw fit but decides how
        # well the placement can be smoothed -- see _placement_origin.
        origin = _placement_origin(points_cam, finite, origin_markers)
        shape_world = (points_cam - origin) @ camera.R

        translation = solve_placement_translation(L, shape_world, uv)
        if translation is None:
            skipped["too_few_pixels"] = skipped.get("too_few_pixels", 0) + 1
            continue
        rotvec = np.zeros(3)
        if refine:
            refined = refine_placement(L, shape_world, translation, uv, rotvec_prev)
            if refined is not None:
                rotvec, translation = refined
                rotvec_prev = rotvec

        solved_frames.append(frame)
        shapes.append(shape_world)
        rotvecs.append(rotvec)
        translations.append(translation)
        per_frame_L.append(L)

    if not solved_frames:
        return _fail(f"No frame could be placed (reasons: {skipped})", gui)

    rotvecs_arr = np.stack(rotvecs)
    translations_arr = np.stack(translations)
    raw_world = _build_world(shapes, rotvecs_arr, translations_arr)

    # --- smoothing ---------------------------------------------------------
    smoothed = False
    if smooth_hz and smooth_hz > 0:
        rotvecs_s, translations_s = smooth_placement(
            rotvecs_arr, translations_arr, float(smooth_hz), float(point_rate)
        )
        smoothed = not (
            np.array_equal(rotvecs_s, rotvecs_arr)
            and np.array_equal(translations_s, translations_arr)
        )
        if smoothed:
            world = _build_world(shapes, rotvecs_s, translations_s)
            rotvecs_arr, translations_arr = rotvecs_s, translations_s
        else:
            world = raw_world
            print("[yellow]Smoothing skipped: too few frames or cutoff >= Nyquist.[/yellow]")
    else:
        world = raw_world

    # --- report ------------------------------------------------------------
    stats = _placement_report(
        per_frame_L, world, raw_world, pixels_by_frame, solved_frames, float(point_rate), validation
    )
    print("-" * 80)
    print("=== Alignment Complete ===")
    print(
        f"Frames placed: {len(solved_frames)}/{len(frames_3d)}"
        + (f"  skipped: {skipped}" if skipped else "")
    )
    print(
        f"Reprojection (px): mean={stats['reprojection_px_mean']:.2f} "
        f"p95={stats['reprojection_px_p95']:.2f} max={stats['reprojection_px_max']:.2f}"
    )
    print(
        f"Lowest foot above floor (m): mean={stats['floor_contact_m_mean']:+.3f} "
        f"(independent check -- the fit never uses the floor)"
    )
    if "inside_volume_fraction" in stats:
        print(f"Markers inside calibrated volume: {stats['inside_volume_fraction'] * 100:.1f}%")
    print(
        f"Horizontal speed (m/s): median={stats['speed_median']:.2f} "
        f"p95={stats['speed_p95']:.2f} max={stats['speed_max']:.2f}"
        + (f"  [placement smoothed at {smooth_hz:g} Hz]" if smoothed else "  [raw placement]")
    )

    # --- write outputs -----------------------------------------------------
    return _write_outputs(
        world,
        solved_frames,
        rotvecs_arr,
        translations_arr,
        per_frame_L,
        pixels_by_frame,
        output_directory,
        point_rate=float(point_rate),
        smooth_hz=float(smooth_hz) if smoothed else 0.0,
        camera=camera,
        validation=validation,
        stats=stats,
        mono3d_path=mono3d_path,
        dlt3d_path=dlt3d_path,
        skeleton_json_path=skeleton_json_path,
        gui=gui,
    )


def _fail(message: str, gui: bool):
    print(f"[red]Error: {message}[/red]")
    if gui:
        try:
            from tkinter import messagebox

            messagebox.showerror("Monocular DLT alignment", message)
        except Exception:  # noqa: BLE001 - a missing display must not mask the error
            pass
    return None


def _build_world(shapes, rotvecs, translations) -> np.ndarray:
    return np.stack(
        [
            shapes[i] @ Rotation.from_rotvec(rotvecs[i]).as_matrix().T + translations[i]
            for i in range(len(shapes))
        ]
    )


def _placement_report(per_frame_L, world, raw_world, pixels_by_frame, frames, fps, validation):
    """Diagnostics, including checks the fit itself never used."""
    reproj = []
    for i, frame in enumerate(frames):
        uv = pixels_by_frame[frame]
        err = np.linalg.norm(dlt_project(per_frame_L[i], world[i]) - uv, axis=1)
        err = err[np.isfinite(err)]
        reproj.append(float(err.mean()) if err.size else np.nan)
    reproj = np.asarray(reproj)
    finite_reproj = reproj[np.isfinite(reproj)]

    stats = {
        "frames_placed": len(frames),
        "reprojection_px_mean": float(finite_reproj.mean()) if finite_reproj.size else float("nan"),
        "reprojection_px_p95": float(np.percentile(finite_reproj, 95))
        if finite_reproj.size
        else float("nan"),
        "reprojection_px_max": float(finite_reproj.max()) if finite_reproj.size else float("nan"),
        "per_frame_reprojection_px": reproj,
    }

    # Floor contact: the lowest marker of the lower body. Uses only markers
    # that exist in every supported layout (feet are the last body markers
    # before the hands in MHR70), falling back to the global minimum.
    n_markers = world.shape[1]
    foot_idx = [i for i in (13, 14, 15, 16, 17, 18, 19, 20) if i < n_markers]
    lowest = (
        np.nanmin(world[:, foot_idx, 2], axis=1) if foot_idx else np.nanmin(world[:, :, 2], axis=1)
    )
    stats["floor_contact_m_mean"] = float(np.nanmean(lowest))
    stats["floor_contact_m_p5"] = float(np.nanpercentile(lowest, 5))
    stats["floor_contact_m_p95"] = float(np.nanpercentile(lowest, 95))

    if validation.get("volume_min_m") is not None:
        lo = np.asarray(validation["volume_min_m"]) - np.array([3.0, 3.0, 0.5])
        hi = np.asarray(validation["volume_max_m"]) + np.array([3.0, 3.0, 1.4])
        inside = ((world > lo) & (world < hi)).all(axis=2)
        stats["inside_volume_fraction"] = float(np.nanmean(inside))

    hip_idx = [i for i in (9, 10) if i < n_markers]
    pelvis = np.nanmean(world[:, hip_idx, :], axis=1) if hip_idx else np.nanmean(world, axis=1)
    step = np.diff(pelvis, axis=0)
    speed = np.linalg.norm(step[:, :2], axis=1) * fps
    speed = speed[np.isfinite(speed)]
    stats["speed_median"] = float(np.median(speed)) if speed.size else float("nan")
    stats["speed_p95"] = float(np.percentile(speed, 95)) if speed.size else float("nan")
    stats["speed_max"] = float(speed.max()) if speed.size else float("nan")

    raw_pelvis = (
        np.nanmean(raw_world[:, hip_idx, :], axis=1) if hip_idx else np.nanmean(raw_world, axis=1)
    )
    raw_speed = np.linalg.norm(np.diff(raw_pelvis, axis=0)[:, :2], axis=1) * fps
    raw_speed = raw_speed[np.isfinite(raw_speed)]
    stats["raw_speed_max"] = float(raw_speed.max()) if raw_speed.size else float("nan")
    return stats


def _write_outputs(
    world,
    frames,
    rotvecs,
    translations,
    per_frame_L,
    pixels_by_frame,
    output_directory,
    *,
    point_rate,
    smooth_hz,
    camera,
    validation,
    stats,
    mono3d_path,
    dlt3d_path,
    skeleton_json_path,
    gui,
):
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    new_dir = Path(output_directory).expanduser().resolve() / f"vaila_mono_dlt_{timestamp}"
    new_dir.mkdir(parents=True, exist_ok=True)
    file_base = f"mono_dlt_{timestamp}"

    n_markers = world.shape[1]
    header = ["frame"]
    for m in range(1, n_markers + 1):
        header.extend([f"p{m}_x", f"p{m}_y", f"p{m}_z"])
    rec3d_df = pd.DataFrame(
        np.column_stack([np.asarray(frames, dtype=np.float64), world.reshape(len(world), -1)]),
        columns=header,  # ty: ignore[invalid-argument-type]
    )
    rec3d_df["frame"] = rec3d_df["frame"].astype(int)

    csv_path = new_dir / f"{file_base}.csv"
    rec3d_df.to_csv(csv_path, index=False, float_format="%.6f")
    rec3d_df.to_csv(new_dir / f"{file_base}.3d", index=False, float_format="%.6f")
    print(f"World-frame 3D written: {csv_path}")

    # C3D (metres and millimetres) via the shared exporter.
    try:
        import vaila.readcsv_export as readcsv_export
    except ImportError:  # standalone execution
        import readcsv_export  # ty: ignore[unresolved-import]

    df_for_c3d = rec3d_df.copy()
    df_for_c3d.columns = [
        c if c == "frame" else f"{c.rsplit('_', 1)[0]}_{c.rsplit('_', 1)[1].upper()}"
        for c in df_for_c3d.columns
    ]
    for suffix, factor in (("_m", 1), ("_mm", 1000)):
        try:
            readcsv_export.auto_create_c3d_from_csv(
                df_for_c3d,
                str(new_dir / f"{file_base}{suffix}.c3d"),
                point_rate=point_rate,
                conversion_factor=factor,
            )
        except Exception as exc:  # noqa: BLE001 - a C3D failure must not lose the CSV
            print(f"[yellow]C3D ({suffix}) export failed: {exc}[/yellow]")

    save_rec3d_as_bvh(rec3d_df, str(new_dir), file_base, point_rate, gui=gui, swap_yz=True)

    # Per-frame alignment report.
    report_df = pd.DataFrame(
        {
            "frame": frames,
            "reprojection_px": stats["per_frame_reprojection_px"],
            "rotvec_x": rotvecs[:, 0],
            "rotvec_y": rotvecs[:, 1],
            "rotvec_z": rotvecs[:, 2],
            "translation_x_m": translations[:, 0],
            "translation_y_m": translations[:, 1],
            "translation_z_m": translations[:, 2],
            "n_pixels_used": [
                int(np.isfinite(pixels_by_frame[f]).all(axis=1).sum()) for f in frames
            ],
        }
    )
    report_path = new_dir / f"{file_base}_alignment.csv"
    report_df.to_csv(report_path, index=False, float_format="%.6f")

    blender_script = generate_blender_companion_script(
        str(new_dir),
        file_base,
        skeleton_json_path,
        point_rate=point_rate,
        n_frames=len(rec3d_df),
        unreconstructed_markers=find_unreconstructed_markers(rec3d_df),
    )

    calib = ""
    if "calibration_reprojection_px_mean" in validation:
        calib = (
            f"calibration control points : {validation['n_control_points']}, reprojection "
            f"mean {validation['calibration_reprojection_px_mean']:.2f} px / "
            f"max {validation['calibration_reprojection_px_max']:.2f} px\n"
        )
    (new_dir / "README_monocular_dlt_align.txt").write_text(
        "vaila monocular -> DLT world alignment\n"
        "======================================\n\n"
        f"monocular source (camera frame): {mono3d_path}\n"
        f"DLT3D calibration              : {dlt3d_path}\n"
        f"frames placed                  : {stats['frames_placed']}\n"
        f"point rate                     : {point_rate} Hz\n"
        f"placement smoothing            : "
        + (
            f"{smooth_hz:g} Hz zero-lag Butterworth on the 6-DOF pose\n"
            if smooth_hz
            else "none (raw)\n"
        )
        + f"camera focal (from DLT)        : {camera.focal_px:.1f} px\n"
        f"camera centre (world, m)       : {np.array2string(camera.C, precision=3)}\n"
        + calib
        + "\nWhat was done\n-------------\n"
        "The monocular 3D input is in the CAMERA frame (OpenCV: +x right, +y DOWN,\n"
        "+z forward). Its metric body SHAPE is kept; only the body's placement in the\n"
        "lab was solved, by minimising the reprojection of its own 2D keypoints through\n"
        "this camera's real DLT. Scale is deliberately NOT fitted: from one camera a\n"
        "bigger body further away reprojects identically, so a free scale would be\n"
        "unidentifiable and would quietly absorb real depth.\n\n"
        "Quality\n-------\n"
        f"reprojection (px)      mean {stats['reprojection_px_mean']:.2f}  "
        f"p95 {stats['reprojection_px_p95']:.2f}  max {stats['reprojection_px_max']:.2f}\n"
        f"lowest foot above floor  mean {stats['floor_contact_m_mean']:+.3f} m  "
        f"(p5 {stats['floor_contact_m_p5']:+.3f}, p95 {stats['floor_contact_m_p95']:+.3f})\n"
        "  ^ the fit never sees the floor, so this is INDEPENDENT evidence that the\n"
        "    absolute depth and the body size are jointly about right.\n"
        + (
            f"markers inside volume   {stats['inside_volume_fraction'] * 100:.1f}%\n"
            if "inside_volume_fraction" in stats
            else ""
        )
        + f"horizontal speed (m/s)  median {stats['speed_median']:.2f}  "
        f"p95 {stats['speed_p95']:.2f}  max {stats['speed_max']:.2f}"
        + (f"   (raw, unsmoothed max was {stats['raw_speed_max']:.2f})\n" if smooth_hz else "\n")
        + "\nCaveats\n-------\n"
        "* This is still a MONOCULAR estimate placed in a calibrated frame -- it is not\n"
        "  a triangulation. Body proportions come entirely from the network; if it\n"
        "  estimates the wrong stature, depth absorbs that error. With 2+ synchronised\n"
        "  calibrated cameras, prefer rec3d_one_dlt3d.py (true triangulation).\n"
        "* Depth is the least certain axis; the reprojection residual constrains the\n"
        "  viewing rays far better than the distance along them.\n\n"
        "Files\n-----\n"
        f"{file_base}.csv / .3d              world-frame 3D (vaila rec3d convention)\n"
        f"{file_base}_m.c3d / _mm.c3d        C3D in metres / millimetres\n"
        f"{file_base}.bvh                    mocap for Blender (Y/Z swapped)\n"
        f"{file_base}_alignment.csv          per-frame reprojection and placement\n"
        + (f"{Path(blender_script).name}   run this in Blender\n" if blender_script else ""),
        encoding="utf-8",
    )

    print("\n=== Processing Complete ===")
    print(f"Output directory: {new_dir}")
    print(f"  - {file_base}.csv / .3d (world-frame 3D)")
    print(f"  - {file_base}_m.c3d / _mm.c3d (C3D)")
    print(f"  - {file_base}.bvh (Blender mocap)")
    print(f"  - {report_path.name} (per-frame alignment report)")
    if blender_script:
        print(f"  - {os.path.basename(blender_script)} (run this in Blender)")

    if gui:
        try:
            from tkinter import messagebox

            messagebox.showinfo(
                "Monocular DLT alignment",
                f"Alignment complete.\n\n"
                f"Frames: {stats['frames_placed']}\n"
                f"Reprojection: {stats['reprojection_px_mean']:.2f} px mean\n"
                f"Foot above floor: {stats['floor_contact_m_mean']:+.3f} m\n\n"
                f"{new_dir}",
            )
        except Exception:  # noqa: BLE001
            pass
    return str(new_dir), file_base


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def run_monocular_dlt_align_gui():
    """Prompt for every input, then run the alignment."""
    from tkinter import Tk, filedialog, messagebox, simpledialog

    root = Tk()
    root.withdraw()
    try:
        mono3d = filedialog.askopenfilename(
            title="Monocular 3D CSV in the CAMERA frame (*_mhr70_rec3d.csv)",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        if not mono3d:
            return None
        pixels = filedialog.askopenfilename(
            title="Same person's 2D pixels (*_markers.csv) — Cancel to auto-detect",
            filetypes=[("CSV", "*.csv"), ("All files", "*")],
        )
        dlt3d = filedialog.askopenfilename(
            title="DLT3D file for THIS camera (*.dlt3d)",
            filetypes=[("DLT3D", "*.dlt3d"), ("CSV", "*.csv"), ("All files", "*")],
        )
        if not dlt3d:
            return None
        ref3d = filedialog.askopenfilename(
            title="Reference 3D control points (*.ref3d) — optional, Cancel to skip",
            filetypes=[("REF3D", "*.ref3d"), ("CSV", "*.csv"), ("All files", "*")],
        )
        output = filedialog.askdirectory(title="Output directory")
        if not output:
            return None
        fps = simpledialog.askfloat(
            "Capture rate",
            "Point rate (Hz), e.g. 119.88012001 for NTSC-derived 120000/1001:",
            minvalue=0.0001,
            initialvalue=DEFAULT_FPS,
        )
        if fps is None:
            return None
        smooth = simpledialog.askfloat(
            "Placement smoothing",
            "Zero-lag Butterworth cutoff (Hz) applied to the 6-DOF body placement\n"
            "only (never to the markers, so limb lengths cannot be distorted).\n"
            "Enter 0 to disable:",
            minvalue=0.0,
            initialvalue=DEFAULT_SMOOTH_HZ,
        )
        if smooth is None:
            smooth = DEFAULT_SMOOTH_HZ
        skeleton = filedialog.askopenfilename(
            title="Skeleton JSON for the Blender script — optional, Cancel to skip",
            filetypes=[("JSON", "*.json"), ("All files", "*")],
        )
    finally:
        root.destroy()

    cli = [
        "uv run python -m vaila.monocular_dlt_align",
        "--mono3d",
        mono3d,
        "--dlt3d",
        dlt3d,
        "-o",
        output,
        "--fps",
        str(fps),
        "--smooth-hz",
        str(smooth),
    ]
    if pixels:
        cli.extend(["--pixels", pixels])
    if ref3d:
        cli.extend(["--ref3d", ref3d])
    if skeleton:
        cli.extend(["--skeleton", skeleton])
    print(">> " + " ".join(cli))

    try:
        return align_monocular_to_world(
            mono3d,
            dlt3d,
            output,
            pixels_path=pixels or None,
            ref3d_path=ref3d or None,
            point_rate=fps,
            smooth_hz=smooth,
            skeleton_json_path=skeleton or None,
            gui=True,
        )
    except Exception as exc:  # noqa: BLE001 - surface any failure in the GUI
        messagebox.showerror("Monocular DLT alignment", str(exc))
        raise


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Align a monocular (single-camera) 3D reconstruction into a DLT-calibrated world frame."
    )
    parser.add_argument("--mono3d", type=Path, help="Monocular 3D CSV in the CAMERA frame")
    parser.add_argument(
        "--pixels", type=Path, default=None, help="Same person's 2D pixel CSV (wide)"
    )
    parser.add_argument("--dlt3d", type=Path, help="DLT3D coefficients for this camera")
    parser.add_argument(
        "--ref3d", type=Path, default=None, help="Control points, for validation only"
    )
    parser.add_argument("-o", "--output", type=Path, help="Output directory")
    parser.add_argument(
        "--fps",
        "--rate",
        dest="fps",
        type=float,
        default=DEFAULT_FPS,
        help=f"Point rate in Hz (default {DEFAULT_FPS}); fractional rates accepted",
    )
    parser.add_argument(
        "--smooth-hz",
        type=float,
        default=DEFAULT_SMOOTH_HZ,
        help=f"Butterworth cutoff on the 6-DOF placement (default {DEFAULT_SMOOTH_HZ}); 0 disables",
    )
    parser.add_argument(
        "--no-smooth", action="store_true", help="Disable placement smoothing (raw)"
    )
    parser.add_argument(
        "--no-refine",
        action="store_true",
        help="Translation only (skip the 6-DOF refinement); usually worse",
    )
    parser.add_argument(
        "--origin-markers",
        type=int,
        nargs="+",
        default=list(DEFAULT_PLACEMENT_ORIGIN_MARKERS),
        help=(
            "1-based markers whose midpoint the placement rotates about (default "
            f"{' '.join(map(str, DEFAULT_PLACEMENT_ORIGIN_MARKERS))} = MHR70 hips). Affects "
            "smoothing quality only: it must be a SLOW-moving body landmark, never the "
            "marker centroid when most markers are fingers."
        ),
    )
    parser.add_argument(
        "--skeleton", type=Path, default=None, help="Skeleton JSON for the Blender script"
    )
    parser.add_argument("--gui", action="store_true", help="Force the GUI")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.gui or not any((args.mono3d, args.dlt3d, args.output)):
        run_monocular_dlt_align_gui()
        return 0
    missing = [
        n
        for n, v in (("--mono3d", args.mono3d), ("--dlt3d", args.dlt3d), ("--output", args.output))
        if not v
    ]
    if missing:
        parser.error(f"missing required argument(s): {', '.join(missing)}")
    result = align_monocular_to_world(
        args.mono3d,
        args.dlt3d,
        args.output,
        pixels_path=args.pixels,
        ref3d_path=args.ref3d,
        point_rate=args.fps,
        smooth_hz=0.0 if args.no_smooth else args.smooth_hz,
        refine=not args.no_refine,
        origin_markers=tuple(args.origin_markers),
        skeleton_json_path=args.skeleton,
        gui=False,
    )
    return 0 if result else 1


if __name__ == "__main__":
    sys.exit(main())
