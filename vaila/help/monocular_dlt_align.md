# monocular_dlt_align

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing / Markerless 3D |
| **File** | `vaila/monocular_dlt_align.py` |
| **Version** | 0.3.99 |
| **Author** | Paulo Santiago |
| **GUI** | Yes — Frame B → **Markerless 3D** → **Monocular → DLT world** |
| **CLI** | Yes |
| **Runtime** | CPU only; no model weights are loaded |

---

## What problem this solves

`sam3dinov3.py` writes its 3D output in the **camera frame** (OpenCV convention: +X right, +Y **down**, +Z forward, origin at the lens). Opened directly in `readcsv.py` / Open3D / PyVista / Blender the subject is upside-down and floating somewhere arbitrary — there is no floor, no lab axes, and no true metric scale.

This module applies the **change of basis** (rotation + translation) into the lab frame defined by your `.ref3d` control points, using that camera's `.dlt3d`, so the result drops straight into vailá's normal rec3d workflow.

## Why a rigid transform alone is not enough

Measured on the real 3-camera COD fixture (camera c1, 2026-08-05):

| step | result |
|---|---|
| true focal from the `.dlt3d` decomposition | **851.7 px** |
| focal `sam3dinov3.py` actually assumed (no `--focal-px`, default FOV `√(W²+H²)`) | **2202.9 px** — a factor of **2.56** |
| monocular depth | ~15.9 m, where the calibrated volume is at **3.9–11.7 m** |
| plain rotate+translate into the lab | 58.8 px reprojection, **0 of 70** markers inside the volume |
| rescaling the translation by the focal ratio | **worse** (130.4 px) — the true camera also has a different principal point |

## What actually works

Keep the monocular body's metric **shape** (focal-independent, and the thing the network genuinely estimates), and solve only **where that body sits** by minimising the reprojection of its own 2D keypoints through the **real DLT camera**:

1. `solve_placement_translation` — closed form. The DLT projection equations are linear in the world point, so each keypoint gives two linear equations; all are solved together by least squares.
2. `refine_placement` — refines rotation **and** translation (6 DOF) against the same residual.

Result on that fixture, 631 frames: **1.18 px** mean reprojection — better than the DLT calibration's own **2.35 px** residual on its 12 control points. The 6-DOF refinement rotates the body a median of only **9.5°** from the network's own orientation, the systematic correction expected from a 2.56× focal error rather than noise fitting.

> **Scale is deliberately not a free parameter.** From one camera, a bigger body further away reprojects identically, so a free scale is unidentifiable and would quietly absorb real depth. The body's metric size comes from the network and is left alone.

## Independent validation (constraints the fit never uses)

| check | result |
|---|---|
| lowest foot above the floor plane Z=0 | **+0.050 m** mean (p5 −0.038, p95 +0.123) |
| markers inside the calibrated volume | **100 %** |
| horizontal speed | median 3.4, p95 5.6, max 7.8 m/s |

The fit is driven purely by 2D reprojection and never sees the floor, so the foot contact is genuine evidence that absolute depth **and** body size are jointly right to a few percent.

## Temporal smoothing (`--smooth-hz`, default 6 Hz)

Monocular depth flickers. Raw, the fixture produced pelvis speeds peaking at **27.5 m/s** — physically impossible. The jitter is in the camera's depth direction (translation std 31/45 mm per frame horizontally vs only 6 mm vertically), i.e. a *placement* artifact, not articulation.

The zero-lag Butterworth is therefore applied to the **6-DOF placement**, never to the 70 marker trajectories: the body stays exactly rigid within each frame, so no filter can distort a limb length, and articulated motion is untouched. Measured: peak speed 27.5 → **7.8 m/s**, p95 12.6 → 5.6 m/s, at a cost of only 1.01 → 1.18 px (still well under the calibration's own 2.35 px), floor contact unchanged. Use `--no-smooth` for the raw placement.

> ### Gotcha — the rotation origin decides whether smoothing works at all
> Smoothing `(R, T)` is **not** invariant to where the body shape is centred, because `T` is the world position of the shape's origin, so the filter must track that point. The raw fit *is* invariant (identical 1.01 px either way), which makes this easy to get wrong silently.
>
> Measured: centring on the plain marker centroid — which for MHR70 sits near the **hands**, since 42 of its 70 markers are finger joints, moving at p95 13.6 m/s — left 6 Hz smoothing nearly useless (1.88 px mean / 24.06 px max, still a 26.3 m/s spike). Centring on the hip midpoint (origin at p95 5.6 m/s) gave 1.18 px / 3.45 px and 7.75 m/s.
>
> Hence the default `--origin-markers 10 11` (MHR70 hips); override for a different marker layout.

## Inputs

| Argument | Meaning |
|---|---|
| `--mono3d` | Monocular 3D CSV in the **camera frame**, wide, read by column **order** only (col 0 = frame, then x,y,z per marker): `*_mhr70_rec3d.csv` or `*_mhr70_3d.csv`. |
| `--pixels` | The **same person's** 2D keypoints in pixels, wide (col 0 = frame, then x,y): `*_id_NN_markers.csv`. Optional — falls back to the sibling `*_sam3dinov3_keypoints2d.csv`. |
| `--dlt3d` | That camera's DLT3D coefficients (1 row = fixed camera, or one row per frame). |
| `--ref3d` | Optional control points — used only to **validate** the calibration and report the working volume, never to fit anything. |
| `--fps` | Point rate in Hz; fractional rates accepted (e.g. `119.88012001`). |
| `--smooth-hz` / `--no-smooth` | Placement smoothing cutoff (default 6 Hz) / disable it. |
| `--origin-markers` | 1-based markers whose midpoint the placement rotates about (default `10 11`). |
| `--skeleton` | Skeleton JSON for the generated Blender script. |

## Outputs

Timestamped subfolder, same conventions as `rec3d_one_dlt3d.py`:

| File | Content |
|---|---|
| `<base>.csv` / `.3d` | World-frame 3D, vailá rec3d convention (`p1_x,p1_y,p1_z,…`) |
| `<base>_m.c3d` / `<base>_mm.c3d` | C3D in metres / millimetres |
| `<base>.bvh` | Mocap for Blender (Y/Z swapped) |
| `<base>_blender_skeleton_viz.py` | Blender companion script |
| `<base>_alignment.csv` | Per-frame reprojection, rotation vector, translation, points used |
| `README_monocular_dlt_align.txt` | What was done, the quality numbers, and the caveats |

## Example

```bash
python -m vaila.monocular_dlt_align \
  --mono3d c1_cod_sam3dinov3_visualized_id_04/c1_cod_id_04_mhr70_rec3d.csv \
  --pixels c1_cod_sam3dinov3_visualized_id_04/c1_cod_id_04_markers.csv \
  --dlt3d  c1_cod_markers_1_line.dlt3d \
  --ref3d  c1c2c3_cod.ref3d \
  --skeleton vaila/skeletons/sam3dinov3_mhr70.json \
  --fps 119.88012001 -o ./out
```

## Caveats

- This is still a **monocular** estimate placed in a calibrated frame — **not a triangulation**. Body proportions come entirely from the network; if it estimates the wrong stature, depth absorbs that error.
- **With 2+ synchronised calibrated cameras, prefer [rec3d_one_dlt3d](rec3d_one_dlt3d.md)** — real triangulation.
- Depth is the least certain axis: the reprojection residual constrains the viewing rays far better than the distance along them.

## Related modules

| Module | Role |
|---|---|
| [sam3dinov3](sam3dinov3.md) / [sam3dinov3_visualize](sam3dinov3_visualize.md) | Produce the monocular camera-frame input |
| [rec3d_one_dlt3d](rec3d_one_dlt3d.md) | True multi-camera DLT triangulation — prefer it when available |
| [dlt3d](dlt3d.md) | Computes the `.dlt3d` coefficients from calibration |

## Testing

- `tests/test_monocular_dlt_align.py` — synthetic, CPU-only: recovers a known camera from its own DLT, the OpenCV `+Y down` convention on real coefficients, known placements (translation and 6-DOF), missing-pixel handling, smoothing behaviour including quaternion hemisphere continuity, and that the rotation origin does **not** change the raw fit.

---

Part of *vailá* - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
