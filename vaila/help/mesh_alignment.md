# mesh_alignment

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/mesh_alignment.py` |
| **Version** | 0.3.99 |
| **Author** | Paulo Santiago |
| **GUI** | No |
| **CLI** | No (library module) |

---

## Description

Shared support module for **rec3d_one_dlt3d**'s mesh-for-Blender export feature. Provides:

- `umeyama_alignment(source, target)` — closed-form similarity-transform fit (rotation + uniform scale + translation, Umeyama 1991) between two corresponding 3D point sets, with a degenerate/near-planar-input guard (rejects a numerically unstable fit rather than silently returning one).
- `apply_similarity_transform(points, R, s, t)` — apply a fitted transform to any point set (markers or mesh vertices).
- `best_camera_alignment(source_points_per_camera, target_points)` — fit per camera, return the lowest-residual non-degenerate result (used to pick which camera's mesh to use for a given frame).
- `read_obj_vertices` / `write_obj_mesh` / `write_ply_mesh` — minimal ASCII OBJ/PLY vertex I/O (faces are loaded once from a shared `mesh_faces.npy`, not re-parsed per frame).
- `apply_blender_yz_swap(vertices, faces)` — rotate vertices `(x, y, z) -> (x, z, -y)` into the Y-up file frame Blender's own importers expect (the convention `rec3d.save_rec3d_as_bvh` writes under `--swap-yz`). **Not applied to the mesh export:** mesh vertices are written in the raw `(x, y, z)` DLT frame, because mesh-*sequence* add-ons (Stop Motion OBJ / OBJSequence) assign `v x y z` straight to the mesh with no axis conversion and no forward/up setting to override. This helper is kept to document and test the BVH-side convention. The negation is what makes it a **rotation** (determinant +1) instead of a mirror: a bare column swap to `(x, z, y)` has determinant -1, and Blender's own Y-up→Z-up rotation on import leaves the subject **reflected** — anatomical left and right swapped, silently inverting every asymmetry conclusion. Face winding is preserved, because a proper rotation already keeps outward normals outward.
- `ALIGNMENT_MARKER_SPEC` / `ALIGNMENT_MARKER_INDICES` — the fixed torso/hip/knee MHR70 marker subset (1-based `p{i}` indices) used for the fit: `left-shoulder`, `right-shoulder`, `left-hip`, `right-hip`, `left-knee`, `right-knee`, `left-acromion`, `right-acromion`, `neck`. Hand/foot/finger tips and facial points are deliberately excluded — they are the noisiest MHR70 keypoints and would destabilize the fit.

This module never loads SAM3/SAM 3D Body weights and has no CUDA/GPU dependency — it is pure NumPy linear algebra plus file I/O, so it runs anywhere `rec3d_one_dlt3d.py` runs.

---

## Reference

S. Umeyama, "Least-squares estimation of transformation parameters between two point patterns," *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(4), 1991.

---

## Main functions

| Function | Description |
|----------|-------------|
| `umeyama_alignment` | Fit (R, s, t) mapping source points onto target points; returns an `AlignmentResult` with `.degenerate` set when the fit is unreliable. |
| `apply_similarity_transform` | Apply a fitted (R, s, t) to any (N, 3) point array. |
| `best_camera_alignment` | Fit per camera, return the (index, result) of the lowest-residual non-degenerate camera. |
| `read_obj_vertices` | Read only the `v x y z` lines of an ASCII OBJ, in file order. |
| `write_obj_mesh` / `write_ply_mesh` | Write an ASCII OBJ/PLY mesh from vertices + a shared 0-indexed faces array. |
| `apply_blender_yz_swap` | Rotate vertices to `(x, z, -y)` for Blender (winding preserved), matching `--swap-yz`'s BVH convention. |

---

## Related modules

| Module | Role |
|--------|------|
| **rec3d_one_dlt3d** | Consumes this module for its `--mesh-source-dir`/`--export-mesh` feature. |
| **sam3dinov3_visualize** | Produces the per-camera monocular MHR70 3D CSV + mesh sequence this module aligns. |
| **sam3dinov3** | Defines `MHR70_NAMES`, the fixed keypoint order `ALIGNMENT_MARKER_SPEC` indexes into. |

---

## Testing

- `tests/test_rec3d_mesh_alignment.py` — synthetic, CPU-only: recovers a known transform to float tolerance, flags planar/collinear input as degenerate, OBJ/PLY round-trip.
- `tests/test_rec3d_mesh_export.py` — real-data regression against a 3-camera fixture (skipped if the fixture is unavailable).

---

Part of *vailá* - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
