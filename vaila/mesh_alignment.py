"""
================================================================================
Script: mesh_alignment.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.99
Created: 04 August 2026
Last Updated: 04 August 2026

Description:
    Umeyama similarity-transform alignment (rotation + uniform scale +
    translation — no shear/non-uniform scaling, 7 degrees of freedom) for
    reconciling a monocular 3D pose/mesh estimate (e.g. SAM3+DINOv3's SAM 3D
    Body output, which lives in that camera's own root-relative + assumed
    focal-length space) with a multi-camera DLT-triangulated skeleton in true
    metric world space.

    Used by rec3d_one_dlt3d.py's mesh-export feature: at each frame, the
    camera whose monocular skeleton best matches (lowest residual) the
    DLT-triangulated skeleton is selected as that frame's mesh source, and
    the same fitted transform is applied to its mesh vertices. This is a
    coordinate-frame reconciliation, not a re-triangulation: it introduces no
    new depth information beyond what the monocular estimate already
    encoded, it only fixes the estimate's position/scale/orientation against
    the metrically-calibrated world frame.

    Also provides minimal ASCII OBJ/PLY read/write helpers (vertices only —
    face topology is constant across frames and loaded once from a shared
    mesh_faces.npy, matching the sam3dinov3_visualize.py export convention).

Reference:
    S. Umeyama, "Least-squares estimation of transformation parameters
    between two point patterns," IEEE Transactions on Pattern Analysis and
    Machine Intelligence, 13(4), 1991.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

#: (1-based marker index in a p1..p70 MHR70-ordered wide CSV, MHR70 name).
#: Torso/hip/knee landmarks only — high real-world confidence, low
#: soft-tissue artifact, and enough vertical spread (shoulders to knees) to
#: avoid a near-planar point set. Hand/foot/finger tips and facial points are
#: deliberately excluded: they are the noisiest MHR70 keypoints and would
#: destabilize the similarity fit. Indices derived from the fixed MHR70_NAMES
#: order in vaila/sam3dinov3.py (index + 1 == marker number).
ALIGNMENT_MARKER_SPEC: tuple[tuple[int, str], ...] = (
    (6, "left-shoulder"),
    (7, "right-shoulder"),
    (10, "left-hip"),
    (11, "right-hip"),
    (12, "left-knee"),
    (13, "right-knee"),
    (68, "left-acromion"),
    (69, "right-acromion"),
    (70, "neck"),
)

#: 1-based marker indices only, in the same order as ALIGNMENT_MARKER_SPEC.
ALIGNMENT_MARKER_INDICES: tuple[int, ...] = tuple(idx for idx, _name in ALIGNMENT_MARKER_SPEC)


@dataclass
class AlignmentResult:
    """Result of fitting a similarity transform source -> target.

    ``degenerate`` is True when the source point set is too small or too
    close to planar/collinear for a numerically stable rotation estimate —
    callers must skip a degenerate result rather than trust R/s/t.
    """

    degenerate: bool
    reason: str | None = None
    R: np.ndarray | None = None
    s: float | None = None
    t: np.ndarray | None = None
    n_points: int = 0
    mean_residual: float = float("inf")
    rms_residual: float = float("inf")
    max_residual: float = float("inf")


def umeyama_alignment(
    source: np.ndarray,
    target: np.ndarray,
    *,
    min_points: int = 4,
    planarity_ratio_threshold: float = 1e-3,
) -> AlignmentResult:
    """Fit a similarity transform (R, s, t) mapping source points onto target
    points: target_i ~= s * R @ source_i + t, in the closed-form least-squares
    sense of Umeyama (1991).

    Args:
        source: (N, 3) array, N >= min_points.
        target: (N, 3) array, same N and row correspondence as source.
        min_points: minimum number of point correspondences required.
        planarity_ratio_threshold: minimum allowed ratio of the smallest to
            largest singular value of the centered source points; below this,
            the source set is treated as too close to planar/collinear for a
            numerically stable rotation and the result is flagged degenerate.

    Returns:
        AlignmentResult. Check `.degenerate` before using `.R`/`.s`/`.t`.
    """
    source = np.asarray(source, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    if source.shape != target.shape or source.ndim != 2 or source.shape[1] != 3:
        raise ValueError(
            f"source and target must both be (N, 3) arrays of the same shape, "
            f"got {source.shape} and {target.shape}"
        )
    n = source.shape[0]
    if n < min_points:
        return AlignmentResult(
            degenerate=True,
            reason=f"only {n} valid point correspondences, need >= {min_points}",
            n_points=n,
        )

    mu_src = source.mean(axis=0)
    mu_tgt = target.mean(axis=0)
    src_c = source - mu_src
    tgt_c = target - mu_tgt

    singular_values = np.linalg.svd(src_c, compute_uv=False)
    if (
        singular_values[0] <= 0
        or (singular_values[-1] / singular_values[0]) < planarity_ratio_threshold
    ):
        ratio = 0.0 if singular_values[0] <= 0 else singular_values[-1] / singular_values[0]
        return AlignmentResult(
            degenerate=True,
            reason=(
                f"source points are near-planar/collinear "
                f"(smallest/largest singular value ratio {ratio:.2e} < "
                f"{planarity_ratio_threshold:.2e})"
            ),
            n_points=n,
        )

    covariance = (tgt_c.T @ src_c) / n
    U, D, Vt = np.linalg.svd(covariance)
    S = np.eye(3)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[2, 2] = -1.0
    R = U @ S @ Vt

    var_src = float((src_c**2).sum() / n)
    if var_src <= 0:
        return AlignmentResult(
            degenerate=True, reason="source points have zero variance", n_points=n
        )
    s = float(np.trace(np.diag(D) @ S) / var_src)
    t = mu_tgt - s * (R @ mu_src)

    aligned = apply_similarity_transform(source, R, s, t)
    residuals = np.linalg.norm(aligned - target, axis=1)
    return AlignmentResult(
        degenerate=False,
        R=R,
        s=s,
        t=t,
        n_points=n,
        mean_residual=float(residuals.mean()),
        rms_residual=float(np.sqrt((residuals**2).mean())),
        max_residual=float(residuals.max()),
    )


def apply_similarity_transform(
    points: np.ndarray, R: np.ndarray, s: float, t: np.ndarray
) -> np.ndarray:
    """Apply s * R @ x + t to every row of `points` (N, 3) -> (N, 3)."""
    points = np.asarray(points, dtype=np.float64)
    return s * (points @ R.T) + np.asarray(t, dtype=np.float64)


def apply_blender_yz_swap(vertices: np.ndarray, faces: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Rotate `vertices` (N, 3) from the DLT frame into the Y-up file frame
    Blender's own importers expect: (x, y, z) -> (x, z, -y). This is the
    convention rec3d.save_rec3d_as_bvh writes under --swap-yz.

    NOT APPLIED TO THE MESH EXPORT. rec3d_one_dlt3d writes mesh vertices in
    the raw (x, y, z) DLT frame, unconditionally, because the mesh-*sequence*
    add-ons people actually play an OBJ sequence with (Stop Motion OBJ /
    OBJSequence) assign "v x y z" straight to the mesh with no axis
    conversion and expose no forward/up setting to override. A converted file
    therefore landed with the body's height on Blender's Y axis there, even
    though it was correct through the one-frame-at-a-time wm.obj_import
    dialog. Keep that in mind before wiring this helper into the mesh path
    again -- it is retained to document and test the BVH-side convention.

    The negation is what makes this a ROTATION (-90 deg about X,
    determinant +1) rather than a mirror. A bare column swap to (x, z, y)
    has determinant -1, and Blender's importers then apply their own Y-up ->
    Z-up rotation (determinant +1) on top, leaving the whole subject
    reflected: the athlete's left and right end up swapped in the scene,
    which silently inverts every asymmetry conclusion drawn from it. Faces
    keep their original winding precisely because a proper rotation already
    preserves outward-facing normals -- reversing it here would turn the
    mesh inside-out.

    Vertices and faces must be converted together, and only once for the
    final write -- never feed the rotated vertices back into
    umeyama_alignment/apply_similarity_transform, which operate in the
    original DLT frame.
    """
    vertices = np.asarray(vertices, dtype=np.float64)
    rotated_vertices = np.column_stack([vertices[:, 0], vertices[:, 2], -vertices[:, 1]])
    return rotated_vertices, np.asarray(faces)


def slerp_rotation(R0: np.ndarray, R1: np.ndarray, u: float) -> np.ndarray:
    """Spherical-linear interpolation between two rotation matrices.

    `u` is clamped to [0, 1]; u=0 returns R0 and u=1 returns R1. Rotations
    must be interpolated on SO(3), not component-wise: a linear blend of two
    rotation matrices is generally not a rotation (it shrinks toward the
    average, distorting the mesh it is applied to).
    """
    from scipy.spatial.transform import Rotation, Slerp

    u = float(np.clip(u, 0.0, 1.0))
    if u <= 0.0:
        return np.asarray(R0, dtype=np.float64)
    if u >= 1.0:
        return np.asarray(R1, dtype=np.float64)
    rotations = Rotation.from_matrix(np.stack([R0, R1]))
    return Slerp([0.0, 1.0], rotations)(u).as_matrix()


def interpolate_similarity_transform(
    before: tuple[float, np.ndarray, float, np.ndarray] | None,
    after: tuple[float, np.ndarray, float, np.ndarray] | None,
    frame: float,
) -> tuple[np.ndarray, float, np.ndarray] | None:
    """Estimate (R, s, t) at `frame` from the nearest solved neighbours.

    Each neighbour is ``(frame, R, s, t)``. With both neighbours present the
    rotation is SLERPed and scale/translation are linearly interpolated by
    frame position; with only one, that neighbour's transform is held.
    Returns None when neither neighbour exists.

    This fills alignment gaps *without* dropping frames. The mesh being
    placed is still that frame's own geometry — only its world placement is
    inferred — so the exported sequence stays frame-for-frame aligned with
    the C3D/BVH instead of developing gaps that silently desynchronise a
    Blender OBJ-sequence import.
    """
    if before is None and after is None:
        return None
    if before is None:
        _f, R, s, t = after  # ty: ignore[not-iterable]
        return np.asarray(R, dtype=np.float64), float(s), np.asarray(t, dtype=np.float64)
    if after is None:
        _f, R, s, t = before
        return np.asarray(R, dtype=np.float64), float(s), np.asarray(t, dtype=np.float64)

    f0, R0, s0, t0 = before
    f1, R1, s1, t1 = after
    span = float(f1) - float(f0)
    u = 0.0 if span == 0 else (float(frame) - float(f0)) / span
    u = float(np.clip(u, 0.0, 1.0))
    R = slerp_rotation(np.asarray(R0), np.asarray(R1), u)
    s = float(s0) + u * (float(s1) - float(s0))
    t = np.asarray(t0, dtype=np.float64) + u * (
        np.asarray(t1, dtype=np.float64) - np.asarray(t0, dtype=np.float64)
    )
    return R, s, t


def best_camera_alignment(
    source_points_per_camera: list[np.ndarray | None],
    target_points: np.ndarray,
    *,
    min_points: int = 4,
    planarity_ratio_threshold: float = 1e-3,
) -> tuple[int | None, AlignmentResult | None]:
    """Fit a similarity transform per camera and return the (index, result)
    of the camera with the lowest mean residual among non-degenerate fits.

    Each entry in `source_points_per_camera` is that camera's monocular
    marker positions for one frame (or None if unavailable), same row count
    and order as `target_points` (the DLT-triangulated marker positions for
    that frame). Rows with a non-finite value in either source or target are
    excluded from the fit for that camera before checking `min_points`.

    Returns (None, None) if every camera is unavailable or degenerate for
    this frame.
    """
    target_points = np.asarray(target_points, dtype=np.float64)
    best_idx: int | None = None
    best_result: AlignmentResult | None = None
    for idx, source_points in enumerate(source_points_per_camera):
        if source_points is None:
            continue
        source_points = np.asarray(source_points, dtype=np.float64)
        valid = np.isfinite(source_points).all(axis=1) & np.isfinite(target_points).all(axis=1)
        if valid.sum() < min_points:
            continue
        result = umeyama_alignment(
            source_points[valid],
            target_points[valid],
            min_points=min_points,
            planarity_ratio_threshold=planarity_ratio_threshold,
        )
        if result.degenerate:
            continue
        if best_result is None or result.mean_residual < best_result.mean_residual:
            best_idx = idx
            best_result = result
    return best_idx, best_result


def read_obj_vertices(path: Path) -> np.ndarray:
    """Read only the 'v x y z' vertex lines of an ASCII OBJ, in file order."""
    vertices: list[list[float]] = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("v "):
                parts = line.split()
                vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    return np.asarray(vertices, dtype=np.float64)


def write_obj_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII Wavefront OBJ (faces are 0-indexed on input)."""
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("# vailá rec3d_one_dlt3d aligned mesh frame\n")
        np.savetxt(fh, vertices, fmt="v %.6f %.6f %.6f")
        np.savetxt(fh, faces + 1, fmt="f %d %d %d")


def write_ply_mesh(path: Path, vertices: np.ndarray, faces: np.ndarray) -> None:
    """Write an ASCII PLY (faces are 0-indexed, triangles only)."""
    n_vertices = len(vertices)
    n_faces = len(faces)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write("ply\nformat ascii 1.0\n")
        fh.write(f"element vertex {n_vertices}\n")
        fh.write("property float x\nproperty float y\nproperty float z\n")
        fh.write(f"element face {n_faces}\n")
        fh.write("property list uchar int vertex_indices\nend_header\n")
        np.savetxt(fh, vertices, fmt="%.6f %.6f %.6f")
        for face in faces:
            fh.write(f"3 {int(face[0])} {int(face[1])} {int(face[2])}\n")
