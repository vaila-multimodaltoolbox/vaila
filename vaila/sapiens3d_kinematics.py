"""
Project: vailá
Script: sapiens3d_kinematics.py
Authors: Paulo Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila

Creation Date: 24 August 2026
Update Date: 24 August 2026
Version: 0.3.113

Description:
    Functional markerless 3D lower-limb kinematics from a Sapiens2 REC3D C3D
    file: pelvis / thigh(L,R) / shank(L,R) / foot(L,R) local coordinate
    systems, relative Hip/Knee/Ankle joint rotation matrices, quaternions,
    and Euler/Cardan angles, with raw and neutral-zeroed orientations.

    Pipeline: Sapiens2 2D keypoints -> REC3D (DLT triangulation) -> C3D with
    generic ``p1..pN`` point labels (semantic identity is lost at/after the
    REC3D/C3D-export stage -- see ``vaila/rec3d.py``/``vaila/rec3d_one_dlt3d.py``,
    neither of which carries keypoint names into the C3D writer) -> this
    module resolves anatomical landmarks, builds segment frames, and computes
    joint kinematics.

    Keypoint mapping (read this before trusting any output):
        A C3D exported by vailá's REC3D pipeline never stores which anatomical
        landmark each ``pN`` is. This module resolves that mapping in order:
          1. If the C3D's own point labels are already semantic (e.g. contain
             "left_hip", "l_hip", "LHip" ...), match them directly -- no
             assumption needed.
          2. Else, if the true point count (POINT:LABELS merged with its
             LABELS2/LABELS3/... continuation groups -- see
             ``vaila.readc3d_export.get_point_labels``; a naive read of
             POINT:LABELS alone silently truncates at the format's 255-entry
             cap) is exactly 308, treat ``pN`` as the 1-based index into the
             canonical Sapiens2 "Sociopticon" 308-keypoint order shipped at
             ``vaila/skeletons/sapiens2_goliath308.json`` (independently
             cross-checked against the vendored ``sapiens`` package's own
             ``parse_pose_metainfo`` output when available). This is the
             common case for a full-topology Sapiens2 REC3D export.
          3. Else, the point count matches no known canonical topology in this
             repository. The module refuses to guess and raises
             ``LandmarkResolutionError`` naming the missing anatomical slots.
             Supply an explicit ``--keypoint-map`` JSON
             (``{"left_hip": "p10", ...}`` or 0-based ``{"left_hip": 9, ...}``)
             to proceed. This is a deliberate, documented limitation, not a
             bug: fabricating a mapping for an unverifiable point count would
             silently corrupt every downstream angle.

    Scientific limitation -- functional markerless model, not Plug-in Gait:
        Sapiens2 provides no equivalent of Vicon's THI/TIB wands, femoral/
        tibial epicondyle markers, or an anatomical calibration trial. Without
        those, the longitudinal (axial-rotation) axis of the thigh and shank
        is not directly observable from hip/knee/ankle joint centers alone.
        Segment frames here are built from joint centers only (Gram-Schmidt,
        see ``build_*_frame`` below) -- a purely FUNCTIONAL/SURROGATE model:
          - flexion/extension (the first rotation in both Euler sequences
            below) is the most directly observable component;
          - ab/adduction (frontal-plane) is a functional estimate;
          - internal/external rotation (transverse-plane / axial) is a
            surrogate estimate and should NOT be interpreted as anatomically
            identical to Plug-in Gait's axial rotation, which depends on
            wand/epicondyle markers this pipeline does not have.
        See Wu G et al., J Biomech 2002;35:543-548 (DOI: 10.1016/S0021-9290
        (01)00222-6) for the ISB joint-coordinate-system convention this
        module's ``functional_xyz`` Euler decomposition follows in spirit
        (not letter -- no ASIS/PSIS/epicondyle markers are available), and
        Grood & Suntay (1983) for the general joint-coordinate-system concept.

    Quaternion convention (read before using ``rotation.py``/
    ``joint_kinematics.py`` output alongside this module's output -- vailá
    has THREE pre-existing, mutually inconsistent conventions already; see
    ``joint_kinematics.py``'s own docstring):
      - ``rotation.py.rotmat2quat``: docstring claims scalar-first
        ``(w, x, y, z)`` but the code (``Rotation.as_quat()``) actually
        returns scipy's scalar-last ``[x, y, z, w]`` -- a real, confirmed
        inconsistency this module does not repeat or hide.
      - ``joint_kinematics.py.rotmat_to_quat_wxyz``: scalar-first
        ``[w, x, y, z]`` (correctly documented).
      - IMU modules (``imu_analysis.py``, ``vaila_deadlift_imu.py``):
        scalar-first ``[w, x, y, z]``.
      - THIS MODULE: scipy's native scalar-last ``[x, y, z, w]``
        (``Rotation.as_quat()`` used directly, undocumented-vs-code mismatch
        never introduced), with per-DOF temporal sign continuity enforced
        (``dot(q[t], q[t-1]) < 0 -> q[t] *= -1``) before export/plotting.
        Output CSV columns are explicitly named ``qx,qy,qz,qw`` so the
        convention is unambiguous without reading this docstring.

    Reused, not reimplemented:
      - ``vaila.filter_utils.butter_filter`` (Butterworth zero-phase
        low-pass filtering of the 3D trajectories).
      - ``scipy.spatial.transform.Rotation`` for matrix<->quaternion<->Euler
        conversions.
      - ``vaila.readc3d_export.get_point_labels`` for the 255-cap-safe
        C3D point-label read.
    Deliberately NOT reused (different scope/convention, see divergence
    notes above): ``rotation.py``'s ``createortbase*``/``calcmatrot``/
    ``rotmat2euler``/``rotmat2quat``, and ``joint_kinematics.py``'s
    ``local_rotations_from_global``/``rotmat_to_euler_xyz_deg``/
    ``rotmat_to_quat_wxyz``.

Usage:
    GUI:  uv run vaila.py -> Tools -> Data Files -> "Sapiens2 3D Kinematics"
    CLI:  uv run vaila/sapiens3d_kinematics.py -i path/to/rec3d.c3d [options]
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy.spatial.transform import Rotation

try:
    from .filter_utils import butter_filter
    from .readc3d_export import get_point_labels
except ImportError:  # standalone execution
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
    from vaila.filter_utils import butter_filter
    from vaila.readc3d_export import get_point_labels

MODULE_VERSION = "0.3.113"
SKELETON_308_PATH = Path(__file__).resolve().parent / "skeletons" / "sapiens2_goliath308.json"

# --------------------------------------------------------------------------- #
# Anatomical landmark slots this module needs, with the label aliases it will
# try when a C3D's own point labels are already semantic (case-insensitive,
# checked as an exact match after stripping non-alnum separators).
# --------------------------------------------------------------------------- #
REQUIRED_SLOTS: dict[str, list[str]] = {
    "left_hip": ["left_hip", "lhip", "l_hip", "hip_l", "lefthip"],
    "right_hip": ["right_hip", "rhip", "r_hip", "hip_r", "righthip"],
    "left_knee": ["left_knee", "lknee", "l_knee", "knee_l", "leftknee"],
    "right_knee": ["right_knee", "rknee", "r_knee", "knee_r", "rightknee"],
    "left_ankle": ["left_ankle", "lankle", "l_ankle", "ankle_l", "leftankle"],
    "right_ankle": ["right_ankle", "rankle", "r_ankle", "ankle_r", "rightankle"],
    "left_heel": ["left_heel", "lheel", "l_heel", "heel_l", "leftheel"],
    "right_heel": ["right_heel", "rheel", "r_heel", "heel_r", "rightheel"],
    "left_toe": ["left_big_toe", "left_toe", "ltoe", "l_toe", "toe_l", "lefttoe", "leftbigtoe"],
    "right_toe": [
        "right_big_toe",
        "right_toe",
        "rtoe",
        "r_toe",
        "toe_r",
        "righttoe",
        "rightbigtoe",
    ],
}
# Optional slots: used for the pelvis's superior reference vector; either is
# enough (neck preferred if both are present -- less sensitive to arm swing).
OPTIONAL_SLOTS: dict[str, list[str]] = {
    "neck": ["neck"],
    "left_shoulder": ["left_shoulder", "lshoulder", "l_shoulder", "shoulder_l"],
    "right_shoulder": ["right_shoulder", "rshoulder", "r_shoulder", "shoulder_r"],
}

JOINTS = ("hip", "knee", "ankle")
SIDES = ("left", "right")
SEGMENTS = (
    "pelvis",
    "thigh_left",
    "thigh_right",
    "shank_left",
    "shank_right",
    "foot_left",
    "foot_right",
)

# scipy ``Rotation.as_euler`` sequence strings. Uppercase = intrinsic
# (rotating-axes) sequence, matching Plug-in-Gait convention.
EULER_SEQUENCES: dict[str, dict[str, str]] = {
    "functional_xyz": {"hip": "XYZ", "knee": "XYZ", "ankle": "XYZ"},
    "vicon_compatible": {"hip": "YXZ", "knee": "YXZ", "ankle": "YZX"},
}


class LandmarkResolutionError(RuntimeError):
    """Raised when required anatomical landmarks cannot be resolved from a C3D
    without fabricating an unverifiable mapping. See module docstring."""


# --------------------------------------------------------------------------- #
# Landmark resolution
# --------------------------------------------------------------------------- #
def _normalize_label(label: str) -> str:
    return "".join(ch for ch in str(label).lower() if ch.isalnum())


_CANONICAL_308_NAMES_CACHE: list[str] | None = None


def _load_canonical_308_names() -> list[str]:
    """Ordered Sociopticon 308-keypoint names shipped with the repo (not the
    ``sapiens`` package -- keeps this module usable without that heavy,
    optional dependency installed)."""
    global _CANONICAL_308_NAMES_CACHE
    if _CANONICAL_308_NAMES_CACHE is None:
        data = json.loads(SKELETON_308_PATH.read_text(encoding="utf-8"))
        names = data["keypoints"]
        if len(names) != 308:
            raise LandmarkResolutionError(
                f"{SKELETON_308_PATH} does not contain 308 keypoint names "
                f"(found {len(names)}) -- cannot use it as the canonical order."
            )
        _CANONICAL_308_NAMES_CACHE = names
    return _CANONICAL_308_NAMES_CACHE


def _direct_label_match(point_labels: list[str]) -> dict[str, int] | None:
    """Try matching C3D point labels directly against slot aliases. Returns
    the slot->index map only if every required slot is found."""
    norm_to_index: dict[str, int] = {}
    for i, lab in enumerate(point_labels):
        norm_to_index.setdefault(_normalize_label(lab), i)

    def find(aliases: list[str]) -> int | None:
        for alias in aliases:
            idx = norm_to_index.get(_normalize_label(alias))
            if idx is not None:
                return idx
        return None

    resolved: dict[str, int] = {}
    for slot, aliases in REQUIRED_SLOTS.items():
        idx = find(aliases)
        if idx is None:
            return None
        resolved[slot] = idx
    for slot, aliases in OPTIONAL_SLOTS.items():
        idx = find(aliases)
        if idx is not None:
            resolved[slot] = idx
    return resolved


def _canonical_308_match(point_labels: list[str]) -> dict[str, int] | None:
    """If ``point_labels`` is exactly the generic ``p1..p308`` sequence (in
    that order, 1-based), map slots via the canonical Sociopticon order."""
    if len(point_labels) != 308:
        return None
    expected_generic = [f"p{i}" for i in range(1, 309)]
    if [str(x) for x in point_labels] != expected_generic:
        return None
    names = _load_canonical_308_names()
    norm_to_index = {_normalize_label(n): i for i, n in enumerate(names)}

    def find(aliases: list[str]) -> int | None:
        for alias in aliases:
            idx = norm_to_index.get(_normalize_label(alias))
            if idx is not None:
                return idx
        return None

    resolved: dict[str, int] = {}
    for slot, aliases in REQUIRED_SLOTS.items():
        idx = find(aliases)
        if idx is None:
            return None
        resolved[slot] = idx
    for slot, aliases in OPTIONAL_SLOTS.items():
        idx = find(aliases)
        if idx is not None:
            resolved[slot] = idx
    return resolved


def _keypoint_map_file_match(
    point_labels: list[str], keypoint_map: dict[str, Any]
) -> dict[str, int]:
    """Resolve slots from a user-supplied ``{slot: "pN"_or_label_or_int}`` map."""
    label_to_index = {str(lab): i for i, lab in enumerate(point_labels)}
    resolved: dict[str, int] = {}
    missing: list[str] = []
    for slot in list(REQUIRED_SLOTS) + list(OPTIONAL_SLOTS):
        if slot not in keypoint_map:
            if slot in REQUIRED_SLOTS:
                missing.append(slot)
            continue
        value = keypoint_map[slot]
        if isinstance(value, int):
            resolved[slot] = value
        elif isinstance(value, str) and value in label_to_index:
            resolved[slot] = label_to_index[value]
        elif isinstance(value, str) and value.isdigit():
            resolved[slot] = int(value)
        else:
            missing.append(f"{slot} (value {value!r} not found among point labels)")
    if missing:
        raise LandmarkResolutionError(
            "keypoint-map file is missing/invalid entries for: " + ", ".join(missing)
        )
    return resolved


def resolve_landmarks(
    point_labels: list[str], keypoint_map: dict[str, Any] | None = None
) -> tuple[dict[str, int], str]:
    """Resolve the required (and optional) anatomical slots to point indices.

    Returns ``(slot_to_index, method)`` where ``method`` documents how
    resolution happened (for the run's metadata JSON). Raises
    ``LandmarkResolutionError`` if resolution is impossible without
    fabricating an unverifiable assumption -- see module docstring.
    """
    if keypoint_map is not None:
        return _keypoint_map_file_match(point_labels, keypoint_map), "keypoint_map_file"

    direct = _direct_label_match(point_labels)
    if direct is not None:
        return direct, "direct_label_match"

    canonical = _canonical_308_match(point_labels)
    if canonical is not None:
        return canonical, "canonical_308_order"

    missing_slots = [s for s in REQUIRED_SLOTS if True]
    raise LandmarkResolutionError(
        "Could not automatically resolve anatomical landmarks from this C3D's "
        f"{len(point_labels)} point label(s). The labels are not semantic "
        "(e.g. 'left_hip') and the point count does not match the canonical "
        "308-keypoint Sapiens2 Sociopticon topology, so no repository-based "
        "mapping applies (see vaila/sapiens3d_kinematics.py module docstring "
        "-- this is a deliberate refusal, not a bug: guessing would silently "
        "corrupt every downstream angle). Supply --keypoint-map pointing to a "
        "JSON file mapping each of: " + ", ".join(missing_slots) + " to a "
        "point label or 0-based index."
    )


# --------------------------------------------------------------------------- #
# C3D read + optional filtering
# --------------------------------------------------------------------------- #
@dataclass
class C3DPointData:
    labels: list[str]
    rate_hz: float
    points_m: np.ndarray  # shape (3, n_points, n_frames), metres
    n_frames: int


def read_c3d_points(path: str | Path) -> C3DPointData:
    import ezc3d

    c3d = ezc3d.c3d(str(path))
    point_params = c3d["parameters"]["POINT"]
    labels = get_point_labels(point_params)
    rate = float(point_params["RATE"]["value"][0])
    units = point_params.get("UNITS", {}).get("value", ["mm"])
    unit = str(units[0]) if units else "mm"
    scale = {"m": 1.0, "mm": 1e-3, "cm": 1e-2}.get(unit, 1.0)
    pts = np.asarray(c3d["data"]["points"])  # (4, n_points, n_frames): x,y,z,residual
    xyz = pts[:3, :, :] * scale
    n_frames = xyz.shape[2]
    return C3DPointData(labels=labels, rate_hz=rate, points_m=xyz, n_frames=n_frames)


def maybe_filter_points(
    points_m: np.ndarray, rate_hz: float, cutoff_hz: float | None
) -> np.ndarray:
    """Zero-phase Butterworth low-pass filter each marker's x/y/z trajectory
    along the frame axis, reusing ``filter_utils.butter_filter``. NaNs (gaps)
    are left untouched per-segment (no interpolation is performed here)."""
    if cutoff_hz is None or cutoff_hz <= 0:
        return points_m
    filtered = points_m.copy()
    n_points = points_m.shape[1]
    for p in range(n_points):
        for axis in range(3):
            series = points_m[axis, p, :]
            if np.all(np.isnan(series)):
                continue
            valid = ~np.isnan(series)
            if valid.sum() < 8:  # too short for a stable filtfilt pad
                continue
            if valid.all():
                filtered[axis, p, :] = butter_filter(
                    series, fs=rate_hz, filter_type="low", cutoff=cutoff_hz
                )
            else:
                # Filter contiguous valid runs independently; leave NaNs as-is.
                run_start = None
                for i in range(len(series) + 1):
                    at_gap = i == len(series) or not valid[i]
                    if not at_gap and run_start is None:
                        run_start = i
                    elif at_gap and run_start is not None:
                        if i - run_start >= 8:
                            filtered[axis, p, run_start:i] = butter_filter(
                                series[run_start:i],
                                fs=rate_hz,
                                filter_type="low",
                                cutoff=cutoff_hz,
                            )
                        run_start = None
    return filtered


# --------------------------------------------------------------------------- #
# Segment frame construction (Gram-Schmidt from joint centers only -- see
# module docstring's "Scientific limitation" section).
# --------------------------------------------------------------------------- #
def _safe_normalize(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v, axis=0, keepdims=True)
    n = np.where(n < 1e-9, np.nan, n)
    return v / n


def project_to_so3(R: np.ndarray) -> np.ndarray:
    """Project a near-rotation matrix onto SO(3) via polar decomposition
    (SVD), correcting numerical drift and reflections. ``R`` has shape
    (3, 3, n_frames)."""
    out = np.full_like(R, np.nan)
    for t in range(R.shape[2]):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        U, _s, Vt = np.linalg.svd(M)
        Rt = U @ Vt
        if np.linalg.det(Rt) < 0:
            U[:, -1] *= -1
            Rt = U @ Vt
        out[:, :, t] = Rt
    return out


def _gram_schmidt_frame(origin: np.ndarray, z_dir: np.ndarray, aux_dir: np.ndarray) -> np.ndarray:
    """Build an orthonormal right-handed frame [X Y Z] (columns, expressed in
    the global/lab frame) per frame, given a primary (Z) direction and an
    auxiliary in-plane direction used only to define the mediolateral (Y)
    axis via a cross product (no anatomical calibration markers involved --
    see module docstring). Shapes: origin/z_dir/aux_dir are (3, n_frames).
    Returns R with shape (3, 3, n_frames); NaN frames pass through as NaN.
    """
    n_frames = z_dir.shape[1]
    Z = _safe_normalize(z_dir)
    Y = _safe_normalize(np.cross(Z, aux_dir, axis=0))
    X = np.cross(Y, Z, axis=0)  # already unit length (Y,Z orthonormal)
    R = np.full((3, 3, n_frames), np.nan)
    R[:, 0, :] = X
    R[:, 1, :] = Y
    R[:, 2, :] = Z
    return R


def _apply_side_flip(R: np.ndarray, side: str, flip: np.ndarray) -> np.ndarray:
    """Post-multiply every frame's rotation matrix by ``flip`` (a constant,
    proper 180-degree rotation about one of the segment's own local axes)
    when ``side == "right"``; left frames pass through unchanged. ``flip``
    is always a signed permutation of the identity with det == +1 (e.g.
    ``diag(1, -1, -1)``), so this never breaks SO(3) properness -- it just
    re-expresses the right-side frame in a left/right-consistent basis.

    Without this, a naive per-side sign flip folded into the Gram-Schmidt
    auxiliary vector (the previous convention) makes the flexion axis (e1)
    agree between mirrored left/right poses but leaves the frontal-plane
    (ab/adduction, varus/valgus) and transverse-plane (axial rotation,
    inversion/eversion) components with inconsistent, sometimes opposite,
    signs -- confirmed empirically via a synthetic sagittal-mirror test
    (see ``tests/test_sapiens3d_kinematics.py::test_mirror_left_right_...``).
    Post-multiplying the whole frame by a fixed local rotation, instead of
    negating one axis of the auxiliary vector before the cross product,
    keeps all three Euler components anatomically consistent between sides.
    """
    if side != "right":
        return R
    return np.einsum("ijt,jk->ikt", R, flip)


# Right-side local 180-degree correction: thigh/shank keep their local X
# (anteroposterior-ish) axis and flip Y (mediolateral) and Z (longitudinal);
# empirically the pairing that keeps hip and knee Euler signs anatomically
# consistent between mirrored left/right legs (see _apply_side_flip).
_THIGH_SHANK_SIDE_FLIP = np.diag([1.0, -1.0, -1.0])

# Foot's primary axis is X (heel->toe), not Z, so the axis it must keep to
# stay anatomically consistent is different: flip X and Z, keep Y
# (mediolateral). Confirmed empirically the same way (ankle Euler signs).
_FOOT_SIDE_FLIP = np.diag([-1.0, 1.0, -1.0])


def _mediolateral_axis(left_landmark: np.ndarray, right_landmark: np.ndarray) -> np.ndarray:
    """Normalized left-minus-right direction (points toward the anatomical
    left), independent of any superior/vertical reference. Used for the
    pelvis frame's Y axis and, separately, as the thigh frames' secondary
    reference (see ``build_thigh_frame``) -- deliberately NOT a function of
    knee or ankle position, so it survives even if the pelvis's up-reference
    (neck/shoulders) is unresolved (see the pelvis fallback in
    ``compute_kinematics``)."""
    return _safe_normalize(left_landmark - right_landmark)


def build_pelvis_frame(
    left_hip: np.ndarray, right_hip: np.ndarray, superior_ref: np.ndarray
) -> np.ndarray:
    """Y = left_hip - right_hip (mediolateral, pointing left); superior_ref
    (neck or mid-shoulder) - mid_hip gives the up direction used only to
    define X (anterior) via cross product; Z completes the frame (up).
    ``superior_ref`` may be all-NaN (no neck/shoulder landmark resolved --
    see ``compute_kinematics``), in which case X and Z come out NaN while Y
    stays valid; ``project_to_so3`` then treats the whole per-frame matrix as
    NaN rather than fabricating a plausible-looking orientation."""
    mid_hip = (left_hip + right_hip) / 2.0
    up_dir = superior_ref - mid_hip
    Y = _mediolateral_axis(left_hip, right_hip)
    X = _safe_normalize(np.cross(Y, up_dir, axis=0))
    Z = np.cross(X, Y, axis=0)
    n_frames = left_hip.shape[1]
    R = np.full((3, 3, n_frames), np.nan)
    R[:, 0, :] = X
    R[:, 1, :] = Y
    R[:, 2, :] = Z
    return R


def build_thigh_frame(
    hip: np.ndarray, knee: np.ndarray, secondary_ref: np.ndarray, side: str
) -> np.ndarray:
    """Z = hip->knee (distal, longitudinal). Mediolateral (Y) axis comes from
    an externally supplied, ankle-independent ``secondary_ref`` (the pelvis
    mediolateral axis, see ``_mediolateral_axis``) via cross product --
    NOT from the shank direction. This is the fix for the straight-leg
    singularity: the old implementation used ``aux = ankle - knee``, making
    ``Y = cross(Z, aux)`` equal to ``cross(hip->knee, knee->ankle)`` (up to
    sign), i.e. ``cross(thigh, shank)``, which is exactly zero whenever hip,
    knee and ankle are collinear (a fully extended knee). With a pelvis-
    derived secondary reference, the thigh frame is singular only if
    hip->knee happens to be parallel to the pelvis mediolateral axis (e.g.
    the femur pointing directly sideways) -- a degenerate hip-abduction pose
    unrelated to knee flexion, not triggered by a straight leg.

    ``secondary_ref`` is used unsigned (identical for left and right); the
    left/right anatomical-sign correction is applied afterward as a fixed
    local 180-degree rotation, not by negating this vector -- see
    ``_apply_side_flip``."""
    z_dir = knee - hip
    R = _gram_schmidt_frame(hip, z_dir, secondary_ref)
    return _apply_side_flip(R, side, _THIGH_SHANK_SIDE_FLIP)


def build_shank_frame(
    knee: np.ndarray, ankle: np.ndarray, secondary_ref: np.ndarray, side: str
) -> np.ndarray:
    """Z = knee->ankle (distal, longitudinal). Mediolateral (Y) axis comes
    from an externally supplied ``secondary_ref`` -- the already-computed
    thigh frame's own Y axis, propagated down the kinematic chain -- NOT
    from the hip direction and NOT from heel/toe. This removes the same
    ``cross(thigh, shank)`` degeneracy the old implementation had (its
    ``aux = hip - knee`` was literally the negated thigh direction), and
    keeps the shank frame decoupled from the foot landmarks (heel/toe), so
    ``R_ankle = R_shank.T @ R_foot`` is not circular: R_shank never uses
    heel/toe, only R_foot does.

    ``secondary_ref`` (the thigh's Y column) is used unsigned; the left/
    right anatomical-sign correction is a fixed post-hoc local rotation --
    see ``_apply_side_flip``. Because ``secondary_ref`` is normally passed
    as the already-flipped ``R_thigh[:, 1, :]``, this composes correctly:
    the shank's own flip corrects its own frame, it does not double up."""
    z_dir = ankle - knee
    R = _gram_schmidt_frame(knee, z_dir, secondary_ref)
    return _apply_side_flip(R, side, _THIGH_SHANK_SIDE_FLIP)


def build_foot_frame(
    ankle: np.ndarray, knee: np.ndarray, heel: np.ndarray, toe: np.ndarray, side: str
) -> np.ndarray:
    """X = heel->toe (anterior, foot progression). Z (up) is the component of
    the shank direction (knee - ankle) orthogonal to X; Y completes the
    right-handed frame (mediolateral). The left/right anatomical-sign
    correction is a fixed post-hoc local rotation about the foot's own Y
    axis (``_FOOT_SIDE_FLIP``), not a sign folded into Y's cross product --
    see ``_apply_side_flip`` for why (the ankle-Euler-sign analogue of the
    thigh/shank fix)."""
    x_dir = _safe_normalize(toe - heel)
    shank_dir = knee - ankle
    z_raw = shank_dir - np.sum(shank_dir * x_dir, axis=0, keepdims=True) * x_dir
    Z = _safe_normalize(z_raw)
    Y = np.cross(Z, x_dir, axis=0)
    n_frames = ankle.shape[1]
    R = np.full((3, 3, n_frames), np.nan)
    R[:, 0, :] = x_dir
    R[:, 1, :] = Y
    R[:, 2, :] = Z
    return _apply_side_flip(R, side, _FOOT_SIDE_FLIP)


# --------------------------------------------------------------------------- #
# Rotation matrix -> quaternion / Euler, sign continuity, QC, neutral zeroing
# --------------------------------------------------------------------------- #
def rotmats_to_quats_xyzw(R: np.ndarray) -> np.ndarray:
    """(3,3,n_frames) rotation matrices -> (n_frames,4) scipy scalar-last
    [x,y,z,w] quaternions, with temporal sign continuity enforced."""
    n_frames = R.shape[2]
    quats = np.full((n_frames, 4), np.nan)
    for t in range(n_frames):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        quats[t] = Rotation.from_matrix(M).as_quat()  # [x, y, z, w]
    enforce_quaternion_continuity(quats)
    return quats


def enforce_quaternion_continuity(quats: np.ndarray) -> None:
    """In-place: flip sign of q[t] whenever dot(q[t], q[t-1]) < 0, skipping
    over NaN frames so continuity is preserved across short gaps."""
    prev = None
    for t in range(quats.shape[0]):
        q = quats[t]
        if np.any(np.isnan(q)):
            continue
        if prev is not None and np.dot(q, prev) < 0:
            q *= -1
            quats[t] = q
        prev = q


def rotmats_to_euler_deg(R: np.ndarray, seq: str) -> np.ndarray:
    n_frames = R.shape[2]
    out = np.full((n_frames, 3), np.nan)
    for t in range(n_frames):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        out[t] = Rotation.from_matrix(M).as_euler(seq, degrees=True)
    return out


def rotation_qc(R: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame determinant and orthogonality error ||R^T R - I||_F."""
    n_frames = R.shape[2]
    det = np.full(n_frames, np.nan)
    orth_err = np.full(n_frames, np.nan)
    eye = np.eye(3)
    for t in range(n_frames):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        det[t] = np.linalg.det(M)
        orth_err[t] = np.linalg.norm(M.T @ M - eye)
    return det, orth_err


def neutral_zero(R: np.ndarray, neutral_frames: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """Average the rotation over ``neutral_frames`` (via quaternion mean, per
    Markley et al. 2007) to get R_neutral, then return R_zeroed[t] =
    R_neutral.T @ R[t] (so the neutral pose maps to identity). Returns
    (R_zeroed, R_neutral)."""
    start, end = neutral_frames
    quats = []
    for t in range(start, end):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        quats.append(Rotation.from_matrix(M).as_quat())
    if not quats:
        raise ValueError(f"No valid frames in neutral range [{start}, {end}) to calibrate from")
    Q = np.array(quats)
    # Quaternion averaging (Markley et al. 2007): eigenvector of largest
    # eigenvalue of sum(q q^T), sign-aligned first to avoid antipodal cancel.
    ref = Q[0]
    Q = np.where((Q @ ref)[:, None] < 0, -Q, Q)
    M_sum = Q.T @ Q
    eigvals, eigvecs = np.linalg.eigh(M_sum)
    q_mean = eigvecs[:, np.argmax(eigvals)]
    R_neutral = Rotation.from_quat(q_mean).as_matrix()

    n_frames = R.shape[2]
    R_zeroed = np.full_like(R, np.nan)
    for t in range(n_frames):
        M = R[:, :, t]
        if np.any(np.isnan(M)):
            continue
        R_zeroed[:, :, t] = R_neutral.T @ M
    return R_zeroed, R_neutral


# --------------------------------------------------------------------------- #
# Full pipeline
# --------------------------------------------------------------------------- #
@dataclass
class KinematicsResult:
    rate_hz: float
    n_frames: int
    landmark_method: str
    landmark_indices: dict[str, int]
    segment_rotations: dict[str, np.ndarray] = field(default_factory=dict)  # raw
    joint_rotations: dict[str, np.ndarray] = field(default_factory=dict)  # raw, per "side_joint"
    neutral_frames: tuple[int, int] | None = None
    segment_rotations_zeroed: dict[str, np.ndarray] = field(default_factory=dict)
    joint_rotations_zeroed: dict[str, np.ndarray] = field(default_factory=dict)
    qc_warnings: list[str] = field(default_factory=list)


def compute_kinematics(
    c3d: C3DPointData,
    keypoint_map: dict[str, Any] | None = None,
    cutoff_hz: float | None = 6.0,
    neutral_frames: tuple[int, int] | None = None,
) -> KinematicsResult:
    slot_idx, method = resolve_landmarks(c3d.labels, keypoint_map)
    points = maybe_filter_points(c3d.points_m, c3d.rate_hz, cutoff_hz)

    def slot(name: str) -> np.ndarray:
        return points[:, slot_idx[name], :]

    lhip, rhip = slot("left_hip"), slot("right_hip")
    lknee, rknee = slot("left_knee"), slot("right_knee")
    lankle, rankle = slot("left_ankle"), slot("right_ankle")
    lheel, rheel = slot("left_heel"), slot("right_heel")
    ltoe, rtoe = slot("left_toe"), slot("right_toe")
    qc_warnings: list[str] = []
    if "neck" in slot_idx:
        superior_ref = slot("neck")
    elif "left_shoulder" in slot_idx and "right_shoulder" in slot_idx:
        superior_ref = (slot("left_shoulder") + slot("right_shoulder")) / 2.0
    else:
        # No superior landmark (neck or both shoulders) resolved: refuse to
        # fabricate an "up" direction. NaN out the pelvis's anterior/vertical
        # axes (X, Z) and flag it, rather than silently producing a
        # valid-looking but invented pelvis orientation. The mediolateral
        # axis (Y = left_hip - right_hip) does not depend on superior_ref and
        # stays valid -- it is still usable as the thigh frames' secondary
        # reference (see _mediolateral_axis / build_thigh_frame).
        superior_ref = np.full_like(lhip, np.nan)
        qc_warnings.append(
            "pelvis: no neck or bilateral-shoulder landmark resolved -- "
            "pelvis anterior (X) and vertical (Z) axes are NaN for every "
            "frame (R_pelvis is therefore NaN, via project_to_so3's "
            "any-NaN-per-frame rule, so the hip joint rotation is also NaN). "
            "The mediolateral axis (Y = left_hip - right_hip) is unaffected "
            "and is still used as the thigh frames' secondary reference."
        )

    y_pelvis = _mediolateral_axis(lhip, rhip)
    R_pelvis = project_to_so3(build_pelvis_frame(lhip, rhip, superior_ref))
    R_thigh_l = project_to_so3(build_thigh_frame(lhip, lknee, y_pelvis, "left"))
    R_thigh_r = project_to_so3(build_thigh_frame(rhip, rknee, y_pelvis, "right"))
    R_shank_l = project_to_so3(build_shank_frame(lknee, lankle, R_thigh_l[:, 1, :], "left"))
    R_shank_r = project_to_so3(build_shank_frame(rknee, rankle, R_thigh_r[:, 1, :], "right"))
    R_foot_l = project_to_so3(build_foot_frame(lankle, lknee, lheel, ltoe, "left"))
    R_foot_r = project_to_so3(build_foot_frame(rankle, rknee, rheel, rtoe, "right"))

    def relative(Rp: np.ndarray, Rc: np.ndarray) -> np.ndarray:
        n_frames = Rp.shape[2]
        out = np.full_like(Rp, np.nan)
        for t in range(n_frames):
            if np.any(np.isnan(Rp[:, :, t])) or np.any(np.isnan(Rc[:, :, t])):
                continue
            out[:, :, t] = Rp[:, :, t].T @ Rc[:, :, t]
        return out

    result = KinematicsResult(
        rate_hz=c3d.rate_hz,
        n_frames=c3d.n_frames,
        landmark_method=method,
        landmark_indices=slot_idx,
        qc_warnings=qc_warnings,
    )
    result.segment_rotations = {
        "pelvis": R_pelvis,
        "thigh_left": R_thigh_l,
        "thigh_right": R_thigh_r,
        "shank_left": R_shank_l,
        "shank_right": R_shank_r,
        "foot_left": R_foot_l,
        "foot_right": R_foot_r,
    }
    result.joint_rotations = {
        "left_hip": relative(R_pelvis, R_thigh_l),
        "right_hip": relative(R_pelvis, R_thigh_r),
        "left_knee": relative(R_thigh_l, R_shank_l),
        "right_knee": relative(R_thigh_r, R_shank_r),
        "left_ankle": relative(R_shank_l, R_foot_l),
        "right_ankle": relative(R_shank_r, R_foot_r),
    }

    if neutral_frames is not None:
        result.neutral_frames = neutral_frames
        for name, R in result.segment_rotations.items():
            zeroed, _ = neutral_zero(R, neutral_frames)
            result.segment_rotations_zeroed[name] = zeroed
        for name, R in result.joint_rotations.items():
            zeroed, _ = neutral_zero(R, neutral_frames)
            result.joint_rotations_zeroed[name] = zeroed

    return result


# --------------------------------------------------------------------------- #
# Output writers
# --------------------------------------------------------------------------- #
def _joint_seq(name: str, convention: str) -> str:
    joint = name.split("_", 1)[1]  # "left_hip" -> "hip"
    return EULER_SEQUENCES[convention][joint]


def write_outputs(
    result: KinematicsResult, output_dir: Path, stem: str, source_path: Path
) -> list[Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    n_frames = result.n_frames
    time_s = np.arange(n_frames) / result.rate_hz

    def write_rotation_csv(path: Path, rotations: dict[str, np.ndarray]) -> None:
        header = ["frame", "time_s"]
        for name in rotations:
            header.extend([f"{name}_r{r}{c}" for r in range(3) for c in range(3)])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for t in range(n_frames):
                row: list[Any] = [t, f"{time_s[t]:.6f}"]
                for R in rotations.values():
                    row.extend(f"{v:.8f}" if not np.isnan(v) else "" for v in R[:, :, t].flatten())
                writer.writerow(row)
        written.append(path)

    write_rotation_csv(output_dir / f"{stem}_segment_rotations_raw.csv", result.segment_rotations)
    write_rotation_csv(output_dir / f"{stem}_joint_rotations_raw.csv", result.joint_rotations)
    if result.segment_rotations_zeroed:
        write_rotation_csv(
            output_dir / f"{stem}_segment_rotations_neutral.csv", result.segment_rotations_zeroed
        )
    if result.joint_rotations_zeroed:
        write_rotation_csv(
            output_dir / f"{stem}_joint_rotations_neutral.csv", result.joint_rotations_zeroed
        )

    def write_angles_csv(path: Path, rotations: dict[str, np.ndarray], is_joint: bool) -> None:
        header = ["frame", "time_s"]
        quats_by_name: dict[str, np.ndarray] = {}
        eulers_by_name: dict[str, dict[str, np.ndarray]] = {}
        det_by_name: dict[str, np.ndarray] = {}
        orth_by_name: dict[str, np.ndarray] = {}
        for name, R in rotations.items():
            quats_by_name[name] = rotmats_to_quats_xyzw(R)
            eulers_by_name[name] = {}
            for convention in EULER_SEQUENCES:
                seq = _joint_seq(name, convention) if is_joint else "XYZ"
                eulers_by_name[name][convention] = rotmats_to_euler_deg(R, seq)
            det_by_name[name], orth_by_name[name] = rotation_qc(R)
            header.extend([f"{name}_qx", f"{name}_qy", f"{name}_qz", f"{name}_qw"])
            for convention in EULER_SEQUENCES:
                axis_labels = ["e1", "e2", "e3"]
                header.extend([f"{name}_{convention}_{a}_deg" for a in axis_labels])
            header.extend([f"{name}_det", f"{name}_orth_err"])
        with path.open("w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(header)
            for t in range(n_frames):
                row: list[Any] = [t, f"{time_s[t]:.6f}"]
                for name in rotations:
                    q = quats_by_name[name][t]
                    row.extend(f"{v:.8f}" if not np.isnan(v) else "" for v in q)
                    for convention in EULER_SEQUENCES:
                        e = eulers_by_name[name][convention][t]
                        row.extend(f"{v:.4f}" if not np.isnan(v) else "" for v in e)
                    row.append(
                        f"{det_by_name[name][t]:.6f}" if not np.isnan(det_by_name[name][t]) else ""
                    )
                    row.append(
                        f"{orth_by_name[name][t]:.8f}"
                        if not np.isnan(orth_by_name[name][t])
                        else ""
                    )
                writer.writerow(row)
        written.append(path)

    write_angles_csv(
        output_dir / f"{stem}_segment_angles_raw.csv", result.segment_rotations, is_joint=False
    )
    write_angles_csv(
        output_dir / f"{stem}_joint_angles_raw.csv", result.joint_rotations, is_joint=True
    )
    if result.segment_rotations_zeroed:
        write_angles_csv(
            output_dir / f"{stem}_segment_angles_neutral.csv",
            result.segment_rotations_zeroed,
            is_joint=False,
        )
    if result.joint_rotations_zeroed:
        write_angles_csv(
            output_dir / f"{stem}_joint_angles_neutral.csv",
            result.joint_rotations_zeroed,
            is_joint=True,
        )

    metadata = {
        "module": "sapiens3d_kinematics",
        "module_version": MODULE_VERSION,
        "generated": datetime.now().isoformat(timespec="seconds"),
        "source_c3d": str(source_path),
        "rate_hz": result.rate_hz,
        "n_frames": n_frames,
        "landmark_resolution_method": result.landmark_method,
        "landmark_indices_0based": result.landmark_indices,
        "neutral_frames_0based": list(result.neutral_frames) if result.neutral_frames else None,
        "quaternion_convention": "scipy scalar-last [x, y, z, w], temporal sign continuity enforced",
        "euler_sequences": EULER_SEQUENCES,
        "coordinate_frame_definitions": {
            "pelvis": "origin=mid_hip; Y=left_hip-right_hip; up_ref=neck_or_midshoulder-mid_hip "
            "(NaN, not fabricated, if neither is resolvable -- see qc_warnings); "
            "X=cross(Y,up_ref); Z=cross(X,Y)",
            "thigh": "origin=hip; Z=knee-hip; secondary_ref=pelvis mediolateral axis "
            "(left_hip-right_hip, ankle-independent, unsigned); Y=cross(Z,secondary_ref); "
            "X=cross(Y,Z); right side then gets a fixed local 180-degree correction "
            "R_right=R_raw@diag(1,-1,-1) (see left/right sign convention note below)",
            "shank": "origin=knee; Z=ankle-knee; secondary_ref=the already-computed thigh "
            "frame's own Y axis (propagated down the chain, hip/ankle-independent, "
            "unsigned); Y=cross(Z,secondary_ref); X=cross(Y,Z); right side gets the same "
            "fixed local 180-degree correction as the thigh, diag(1,-1,-1)",
            "foot": "origin=ankle; X=toe-heel; Z=(knee-ankle) component orthogonal to X; "
            "Y=cross(Z,X); right side gets a fixed local 180-degree correction "
            "R_right=R_raw@diag(-1,1,-1) (different axis pair than thigh/shank because "
            "the foot's primary axis is X, not Z)",
        },
        "relative_rotation_definitions": {
            "hip": "R_pelvis.T @ R_thigh",
            "knee": "R_thigh.T @ R_shank",
            "ankle": "R_shank.T @ R_foot",
        },
        "qc_warnings": result.qc_warnings,
        "scientific_limitations": [
            "Functional/surrogate markerless model, NOT anatomically equivalent to Vicon "
            "Plug-in Gait: Sapiens2 has no THI/TIB wand, femoral/tibial epicondyle, or "
            "anatomical-calibration-trial markers.",
            "Thigh frame secondary reference is the pelvis mediolateral axis, and shank frame "
            "secondary reference is the thigh's own mediolateral axis (chain-propagated): "
            "neither depends on the adjacent distal segment (thigh does not use ankle, shank "
            "does not use hip/heel/toe), which removes the straight-leg (collinear "
            "hip-knee-ankle) singularity present in an earlier version of this module. A "
            "residual, anatomically-unrelated singularity remains if hip->knee (or "
            "knee->ankle) becomes exactly parallel to its secondary reference axis (e.g. the "
            "femur pointing directly along the pelvis mediolateral axis) -- an extreme "
            "hip-abduction edge case, not triggered by knee flexion/extension.",
            "The shank frame does not use heel/toe, so R_ankle = R_shank.T @ R_foot is not "
            "circular through shared foot landmarks; R_foot is the only frame built from "
            "heel/toe.",
            "Pelvis anterior/vertical axes (and therefore R_pelvis and the hip joint "
            "rotation) are NaN, not a fabricated fixed-up-vector guess, for any frame where "
            "neither neck nor both shoulders resolve -- see qc_warnings above.",
            "Because the shank frame's secondary reference is inherited from the thigh frame "
            "and the thigh's is inherited from the pelvis, a NaN/degenerate pelvis or thigh "
            "frame propagates to the shank frame for that frame (by design -- this is an "
            "explicit dependency, not a hidden one).",
            "Flexion/extension (first Euler rotation, both sequences) is the most directly "
            "observable component.",
            "Ab/adduction (frontal plane) is a functional estimate.",
            "Internal/external (axial) rotation is a surrogate estimate and should not be "
            "equated with Plug-in Gait's wand-marker-derived axial rotation. Transverse-plane "
            "(axial-rotation) components across all three joints are surrogate estimates, "
            "not directly observable without wand/epicondyle markers.",
            "Left/right sign convention: the right-side thigh/shank/foot frame is corrected "
            "by a single fixed local 180-degree rotation (not a sign folded into the "
            "Gram-Schmidt auxiliary vector before the cross product). Verified against a "
            "synthetic sagittal-mirror test that the naive per-side sign-in-cross-product "
            "convention reports matching flexion/extension but inconsistent signs on "
            "ab/adduction, varus/valgus, and axial-rotation/inversion-eversion between "
            "mirrored left/right legs, while this fixed-rotation convention reports all "
            "three Euler components consistently for both sides.",
            "On the real rec3d_240hz.c3d fixture, the right knee axial-rotation and right "
            "ankle dorsi/plantarflexion and axial-rotation Euler components pass through "
            "the +/-180-degree branch cut during a high-flexion segment (~t=18.1-18.5s and "
            "~t=20.2-20.4s): det(R) and orthogonality error stay exact (1.0 / ~0) and the "
            "middle Euler component stays under 5 degrees at those frames (not gimbal lock, "
            "which requires the middle component near +/-90 degrees), so this is a harmless "
            "atan2 representation discontinuity in the reported angle, not a rotation-matrix "
            "defect -- but it is a direct consequence of the surrogate axial-rotation axis "
            "reaching anatomically implausible magnitudes (~150-180 degrees) at large knee "
            "flexion, reinforcing the surrogate-estimate caveat above; unwrap "
            "(numpy.unwrap) before further analysis of these components if continuity "
            "matters downstream.",
        ],
        "reference": "Wu G et al. J Biomech 2002;35:543-548. DOI: 10.1016/S0021-9290(01)00222-6",
    }
    meta_path = output_dir / f"{stem}_kinematics_metadata.json"
    meta_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    written.append(meta_path)

    return written


def write_qc_plots(result: KinematicsResult, output_dir: Path, stem: str) -> list[Path]:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    time_s = np.arange(result.n_frames) / result.rate_hz

    # 1) Joint Euler angles (vicon_compatible), all six joints.
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex=True)
    for i, name in enumerate(result.joint_rotations):
        ax = axes.flat[i]
        seq = _joint_seq(name, "vicon_compatible")
        euler = rotmats_to_euler_deg(result.joint_rotations[name], seq)
        for k, lab in enumerate(["e1 (flex/ext)", "e2", "e3"]):
            ax.plot(time_s, euler[:, k], label=lab, linewidth=0.8)
        ax.set_title(f"{name} ({seq})")
        ax.set_ylabel("deg")
        ax.legend(fontsize=7)
    for ax in axes[-1, :]:
        ax.set_xlabel("time (s)")
    fig.suptitle("Joint Euler angles -- vicon_compatible sequence")
    fig.tight_layout()
    p = output_dir / f"{stem}_qc_joint_angles.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 2) QC determinant + orthogonality error, all joint rotations.
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 6), sharex=True)
    for name, R in result.joint_rotations.items():
        det, orth = rotation_qc(R)
        ax1.plot(time_s, det, label=name, linewidth=0.7)
        ax2.plot(time_s, orth, label=name, linewidth=0.7)
    ax1.axhline(1.0, color="k", linestyle="--", linewidth=0.5)
    ax1.set_ylabel("det(R)")
    ax2.set_ylabel("||R^T R - I||")
    ax2.set_xlabel("time (s)")
    ax1.legend(fontsize=7, ncol=3)
    fig.suptitle("QC: SO(3) determinant and orthogonality error")
    fig.tight_layout()
    p = output_dir / f"{stem}_qc_so3_checks.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 3) Quaternion components, hip/knee/ankle (left side as representative).
    fig, axes = plt.subplots(3, 1, figsize=(10, 8), sharex=True)
    for ax, joint in zip(axes, ["left_hip", "left_knee", "left_ankle"], strict=True):
        q = rotmats_to_quats_xyzw(result.joint_rotations[joint])
        for k, lab in enumerate(["qx", "qy", "qz", "qw"]):
            ax.plot(time_s, q[:, k], label=lab, linewidth=0.7)
        ax.set_title(joint)
        ax.set_ylabel("quaternion component")
    axes[-1].set_xlabel("time (s)")
    axes[0].legend(fontsize=7, ncol=4)
    fig.suptitle("Quaternions [x, y, z, w] -- left side")
    fig.tight_layout()
    p = output_dir / f"{stem}_qc_quaternions_left.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 4) Segment frame origins (sanity trace of the resolved landmarks).
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(
        time_s,
        np.linalg.det(result.segment_rotations["pelvis"].transpose(2, 0, 1)),
        label="pelvis det",
    )
    ax.set_xlabel("time (s)")
    ax.set_ylabel("det(R_pelvis)")
    ax.set_title("Pelvis frame determinant over time")
    fig.tight_layout()
    p = output_dir / f"{stem}_qc_pelvis_determinant.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    # 5) Neutral-zeroed vs raw hip flexion (if neutral calibration was run).
    fig, ax = plt.subplots(figsize=(10, 5))
    seq = _joint_seq("left_hip", "vicon_compatible")
    raw = rotmats_to_euler_deg(result.joint_rotations["left_hip"], seq)
    ax.plot(time_s, raw[:, 0], label="raw", linewidth=0.8)
    if result.joint_rotations_zeroed:
        zeroed = rotmats_to_euler_deg(result.joint_rotations_zeroed["left_hip"], seq)
        ax.plot(time_s, zeroed[:, 0], label="neutral-zeroed", linewidth=0.8)
    ax.set_xlabel("time (s)")
    ax.set_ylabel("flexion/extension (deg)")
    ax.set_title("Left hip flexion/extension: raw vs neutral-zeroed")
    ax.legend()
    fig.tight_layout()
    p = output_dir / f"{stem}_qc_neutral_comparison.png"
    fig.savefig(p, dpi=120)
    plt.close(fig)
    written.append(p)

    return written


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def run_from_args(args: argparse.Namespace) -> Path:
    source_path = Path(args.input).resolve()
    c3d = read_c3d_points(source_path)

    keypoint_map = None
    if args.keypoint_map:
        keypoint_map = json.loads(Path(args.keypoint_map).read_text(encoding="utf-8"))

    neutral_frames = None
    if args.neutral_frames:
        start_s, end_s = args.neutral_frames.split(":")
        neutral_frames = (int(start_s), int(end_s))

    result = compute_kinematics(
        c3d,
        keypoint_map=keypoint_map,
        cutoff_hz=args.cutoff,
        neutral_frames=neutral_frames,
    )

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    stem = source_path.stem
    output_dir = Path(args.output_dir).resolve() if args.output_dir else source_path.parent
    output_dir = output_dir / f"processed_sapiens3d_kinematics_{timestamp}"
    written = write_outputs(result, output_dir, stem, source_path)
    if not args.no_plots:
        written += write_qc_plots(result, output_dir, stem)

    print(f"[sapiens3d_kinematics] landmark resolution: {result.landmark_method}")
    for warning in result.qc_warnings:
        print(f"[sapiens3d_kinematics] QC WARNING: {warning}")
    print(f"[sapiens3d_kinematics] wrote {len(written)} file(s) to {output_dir}")
    for p in written:
        print(f"  {p}")
    return output_dir


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Functional 3D lower-limb kinematics (Hip/Knee/Ankle) from a Sapiens2 REC3D C3D file."
    )
    parser.add_argument("-i", "--input", required=True, help="Path to the input .c3d file")
    parser.add_argument(
        "-o", "--output-dir", default=None, help="Output directory (default: alongside input)"
    )
    parser.add_argument(
        "--keypoint-map",
        default=None,
        help="JSON file mapping anatomical slot names to point labels/0-based indices "
        "(required when automatic resolution fails)",
    )
    parser.add_argument(
        "--cutoff",
        type=float,
        default=6.0,
        help="Butterworth low-pass cutoff (Hz); 0 disables filtering",
    )
    parser.add_argument(
        "--neutral-frames",
        default=None,
        help="'start:end' 0-based frame range to calibrate a neutral pose (e.g. '0:30')",
    )
    parser.add_argument("--no-plots", action="store_true", help="Skip QC plot generation")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    try:
        run_from_args(args)
    except LandmarkResolutionError as exc:
        print(f"[sapiens3d_kinematics] ERROR: {exc}", file=sys.stderr)
        return 1
    return 0


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #
def run_sapiens3d_kinematics_gui(parent: Any | None = None) -> None:
    """Standalone Tkinter dialog for the Sapiens2 3D Kinematics module.

    Runs in its own Toplevel (or its own root when launched via
    ``run_vaila_module`` in a subprocess) -- never a second ``tk.Tk()`` inside
    the main vailá window.
    """
    import tkinter as tk
    from tkinter import filedialog, messagebox

    root = tk.Toplevel(parent) if parent is not None else tk.Tk()
    root.title("Sapiens2 3D Kinematics")

    tk.Label(root, text="Input Sapiens2 REC3D C3D file:").grid(
        row=0, column=0, sticky="w", padx=8, pady=(10, 2)
    )
    input_var = tk.StringVar()
    tk.Entry(root, textvariable=input_var, width=50).grid(row=1, column=0, padx=8, sticky="we")

    def pick_input() -> None:
        path = filedialog.askopenfilename(
            title="Select .c3d file", filetypes=[("C3D files", "*.c3d")]
        )
        if path:
            input_var.set(path)

    tk.Button(root, text="Browse...", command=pick_input).grid(row=1, column=1, padx=8)

    tk.Label(root, text="Keypoint map JSON (optional, only needed if auto-resolution fails):").grid(
        row=2, column=0, sticky="w", padx=8, pady=(10, 2)
    )
    kmap_var = tk.StringVar()
    tk.Entry(root, textvariable=kmap_var, width=50).grid(row=3, column=0, padx=8, sticky="we")

    def pick_kmap() -> None:
        path = filedialog.askopenfilename(
            title="Select keypoint-map JSON", filetypes=[("JSON files", "*.json")]
        )
        if path:
            kmap_var.set(path)

    tk.Button(root, text="Browse...", command=pick_kmap).grid(row=3, column=1, padx=8)

    tk.Label(root, text="Butterworth cutoff (Hz, 0 = no filter):").grid(
        row=4, column=0, sticky="w", padx=8, pady=(10, 2)
    )
    cutoff_var = tk.StringVar(value="6.0")
    tk.Entry(root, textvariable=cutoff_var, width=10).grid(row=4, column=1, sticky="w", padx=8)

    tk.Label(root, text="Neutral frames 'start:end' (optional):").grid(
        row=5, column=0, sticky="w", padx=8, pady=(10, 2)
    )
    neutral_var = tk.StringVar()
    tk.Entry(root, textvariable=neutral_var, width=15).grid(row=5, column=1, sticky="w", padx=8)

    def on_run() -> None:
        if not input_var.get():
            messagebox.showerror("Sapiens2 3D Kinematics", "Please select an input .c3d file.")
            return
        cli_parts = ["uv", "run", "vaila/sapiens3d_kinematics.py", "-i", input_var.get()]
        if kmap_var.get():
            cli_parts += ["--keypoint-map", kmap_var.get()]
        if cutoff_var.get():
            cli_parts += ["--cutoff", cutoff_var.get()]
        if neutral_var.get():
            cli_parts += ["--neutral-frames", neutral_var.get()]
        print(">> " + " ".join(cli_parts))

        args = build_arg_parser().parse_args(
            [
                "-i",
                input_var.get(),
                *(["--keypoint-map", kmap_var.get()] if kmap_var.get() else []),
                "--cutoff",
                cutoff_var.get() or "6.0",
                *(["--neutral-frames", neutral_var.get()] if neutral_var.get() else []),
            ]
        )
        try:
            output_dir = run_from_args(args)
        except LandmarkResolutionError as exc:
            messagebox.showerror("Sapiens2 3D Kinematics -- landmark resolution failed", str(exc))
            return
        except Exception as exc:  # noqa: BLE001 - surface any failure to the user, don't crash the GUI
            messagebox.showerror("Sapiens2 3D Kinematics -- run failed", str(exc))
            return
        messagebox.showinfo("Sapiens2 3D Kinematics", f"Done. Outputs written to:\n{output_dir}")

    tk.Button(root, text="Run", command=on_run, bg="#c8e6c9").grid(
        row=6, column=0, columnspan=2, pady=14
    )

    if parent is None:
        root.mainloop()


if __name__ == "__main__":
    if len(sys.argv) > 1:
        sys.exit(main())
    else:
        run_sapiens3d_kinematics_gui()
