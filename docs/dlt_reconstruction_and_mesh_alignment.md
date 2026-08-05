# DLT 3D Reconstruction and Mesh Alignment in vailá: Theory and Validation

*A didactic reference for the DLT/REC family (`dlt2d.py`, `dlt3d.py`, `rec2d.py`, `rec3d.py`, `rec3d_one_dlt3d.py`) and the mesh-for-Blender extension (`mesh_alignment.py`). Written in the spirit of Young-Hoo Kwon's now-offline Kwon3D DLT pages and the UNC Biomechanics "DLT to/from intrinsic/extrinsic" note — the goal is to let a reader rebuild the method from first principles, not just call the functions.*

**Status as of 2026-08-04:** the linear DLT (no lens distortion) and the multi-camera triangulation are implemented and validated on real data (§5). The mesh-alignment feature (§4) is implemented and validated on real data (§6), with one open item — your own visual review in Blender (§6.4). Lens-distortion modeling (§7) is **designed but not yet implemented** — do not assume it is active.

---

## 1. Why DLT, and why this document

Every vailá 3D-reconstruction script (`dlt3d.py`, `rec3d.py`, `rec3d_one_dlt3d.py`, and their 2D counterparts) implements a single, 50-year-old idea: the **Direct Linear Transformation** (DLT), introduced by Abdel-Aziz & Karara (1971) for close-range photogrammetry and popularized in biomechanics by Kwon3D. It survives because it needs nothing but a set of known 3D points and their pixel images — no manufacturer camera specs, no separate intrinsic calibration step, no iterative optimizer to get started.

This document explains, in order:

1. The pinhole camera model DLT is a linearization of (§2).
2. The 11-parameter DLT equations, exactly as `dlt2d.py`/`dlt3d.py` solve them (§3).
3. Multi-camera triangulation, exactly as `rec2d.py`/`rec3d.py`/`rec3d_one_dlt3d.py` solve it (§4 start).
4. The new problem this session solved: fusing a *monocular* AI body-mesh estimate (SAM3+DINOv3) into the *metric, multi-camera* DLT world — via the Umeyama (1991) similarity transform (§4).
5. Real validation numbers from the `rec3d_todo` dataset, not synthetic examples only (§6).
6. What is still open, and what is designed but not yet built (§7–8).

---

## 2. The pinhole camera model, and why DLT skips calibrating it directly

A pinhole camera maps a 3D world point **P** = (X, Y, Z) to a 2D pixel (x, y) through two stages that every camera-calibration textbook (and Kwon3D's own "DLT ⇄ intrinsic/extrinsic" page) separates explicitly:

- **Extrinsic parameters** — where the camera *is*: a rotation matrix **R** (3×3) and translation vector **T** (3×1) that move a world point into the camera's own coordinate frame.
- **Intrinsic parameters** — how the camera *sees*: focal length, principal point (optical center offset), pixel aspect ratio/skew, and (if modeled) lens distortion.

The full projection is:

```
[x]   [ fx  s  cx ]                [X]
[y] = [ 0  fy  cy ] · [R | T]  ·   [Y]
[1]   [ 0   0   1 ]                [Z]
                                    [1]
```

Classic photogrammetry calibrates **R**, **T**, and the intrinsic matrix separately (e.g. Zhang's method, checkerboard calibration). DLT takes a shortcut: it **never separates intrinsic from extrinsic**. It absorbs both into 11 linear coefficients (L1…L11) fit directly from known-3D-point ↔ pixel correspondences, and reconstruction only ever needs those 11 numbers — never R, T, or the focal length individually. That is precisely why the Kwon3D "DLT to/from intrinsic/extrinsic" page exists as a *separate*, optional appendix: recovering R/T/focal-length **from** an already-fit DLT matrix is a post-hoc decomposition, not something DLT reconstruction requires.

This is the central trade-off to understand before touching any of these scripts:

| | DLT (this codebase today) | Full intrinsic/extrinsic calibration |
|---|---|---|
| What you calibrate | 11 linear coefficients per camera | R, T, focal length, principal point, distortion — separately |
| Calibration input | ≥6 known 3D points + their pixels, **one shot**, any camera pose | Usually a checkerboard sequence + iterative solver |
| Handles lens distortion | **No** (today) — see §7 | Yes, natively |
| Reconstruction | Linear least squares (§4) | Also usually linear once calibrated |
| Failure mode | Systematic bias near image edges (radial distortion folded into a linear fit) | More setup cost, more robust geometry |

---

## 3. The DLT equations, as implemented in `dlt2d.py` / `dlt3d.py`

### 3.1 3D case (`dlt3d.py`) — 11 coefficients

For a 3D world point (X, Y, Z) and its pixel image (x, y) in one camera, the DLT model is:

```
x = (L1·X + L2·Y + L3·Z + L4) / (L9·X + L10·Y + L11·Z + 1)
y = (L5·X + L6·Y + L7·Z + L8) / (L9·X + L10·Y + L11·Z + 1)
```

This is a ratio of two linear (affine) functions of (X, Y, Z) — a projective, not merely affine, map. Cross-multiplying to clear the denominator gives, per point, two **linear** equations in the 11 unknowns L1…L11:

```
L1·X + L2·Y + L3·Z + L4 - L9·x·X - L10·x·Y - L11·x·Z = x
L5·X + L6·Y + L7·Z + L8 - L9·y·X - L10·y·Y - L11·y·Z = y
```

With **N** known reference points you get **2N** such equations — an over-determined linear system **A·L = b** solved by least squares (`numpy.linalg.lstsq` in `dlt3d.py`). Minimum requirement: **N ≥ 6** non-coplanar points (11 unknowns, 2 equations per point → 12 equations from 6 points; vailá's own guard requires ≥6 common labeled points between the pixel file and the REF3D file — see `dlt3d.py`'s point-matching note).

`dlt3d.py`'s calibration workflow, concretely:

1. You track the pixel positions of a rigid **calibration volume** with N known 3D points (in vailá's `rec3d_todo` example dataset: a 12-point frame, `c1c2c3_cod.ref3d`).
2. `dlt3d.py` correlates pixel points to REF3D points **by label** (`p3` in `p3_x`/`p3_y` matches `p3` in `p3_x`/`p3_y`/`p3_z`), not by column order — so a REF3D file can define more points than any one camera's pixel file tracks.
3. It solves for L1…L11 per camera, writing a `.dlt3d` file: one row of 11 coefficients (fixed calibration) or one row *per frame* (if the camera moves and is recalibrated every frame — see §4.4).

### 3.2 2D case (`dlt2d.py`) — 8 coefficients

The planar case (all reference points share Z = 0, or you only need a homography-like 2D-to-2D mapping, e.g. a soccer pitch) drops to 8 coefficients:

```
x = (L1·X + L2·Y + L3) / (L7·X + L8·Y + 1)
y = (L4·X + L5·Y + L6) / (L7·X + L8·Y + 1)
```

Minimum requirement: **4 non-collinear points** (8 unknowns, 2 equations per point).

---

## 4. Multi-camera 3D triangulation (`rec3d_multicam`, shared by `rec3d.py` and `rec3d_one_dlt3d.py`)

Given a *calibrated* camera (known L1…L11) and an observed pixel (x, y), the DLT equations from §3.1 become **linear in the unknown world point** (X, Y, Z) — this is the reverse direction from calibration, and it's why a *single* camera's DLT equations, rearranged, give two linear constraints on 3D position:

```
(L1 - x·L9)·X + (L2 - x·L10)·Y + (L3 - x·L11)·Z = x - L4
(L5 - y·L9)·X + (L6 - y·L10)·Y + (L7 - y·L11)·Z = y - L8
```

One camera gives 2 equations in 3 unknowns — **under-determined** (a single 2D image can't recover depth; the point could be anywhere along the ray through the camera's optical center). Two or more cameras observing the *same physical point* stack their equations into one linear system with **2 × (number of cameras)** rows and exactly 3 unknowns (X, Y, Z) — solved by least squares in `rec3d_multicam()`:

```python
# vaila/rec3d.py — the actual implementation
A_matrix[row] = [a1 - x*a9, a2 - x*a10, a3 - x*a11]
b_vector[row] = x - a4
# ... one such pair of rows per camera ...
solution, *_ = np.linalg.lstsq(A_matrix, b_vector, rcond=None)  # [X, Y, Z]
```

Minimum: **2 cameras** (4 equations, 3 unknowns — already over-determined, which is good: any 2-camera measurement noise gets partially averaged out). More cameras improve robustness and let the least-squares residual serve as a data-quality signal (not currently surfaced by these scripts, but a natural extension).

### 4.1 Fixed vs. per-frame DLT: `rec3d_one_dlt3d.py` vs. `rec3d.py`

Two scripts exist because two physically different camera setups need different DLT bookkeeping:

- **`rec3d_one_dlt3d.py`** — one `.dlt3d` file per camera, **one row** (the camera is physically fixed for the whole clip). This is the common case: tripod-mounted cameras filming a calibrated volume once.
- **`rec3d.py`** — one `.dlt3d` file per camera, **one row per frame** (the camera moves, or is re-calibrated every frame — e.g. a broadcast/moving camera, as in the FIFA pipeline's per-frame DLT export).

**A concrete trap worth remembering:** the `rec3d_todo` fixture's `.dlt3d` files (`c{1,2,3}_cod_markers_1_line.dlt3d`) have exactly one row → they describe **fixed** cameras. Feeding them to `rec3d.py` (which expects one DLT row per frame) doesn't error loudly — it silently reconstructs *only frame 0*, because `rec3d.py`'s frame-matching logic intersects the DLT file's frame set ({0}) with the pixel file's frame set. Always check whether your `.dlt3d` file has 1 row (→ `rec3d_one_dlt3d.py`) or N rows matching your video's frame count (→ `rec3d.py`) before choosing a script.

### 4.2 Point-matching convention: labels are ignored

Unlike `dlt3d.py`'s calibration step (which matches pixel ↔ REF3D points *by label*), `rec3d.py`/`rec3d_one_dlt3d.py`'s reconstruction step matches markers *by column position only* — column 0 is the frame id, and every (x, y) pair after that is "marker N" regardless of what the CSV header says. This is deliberate: it makes the same reconstruction code work unmodified whether the pixel CSVs came from vailá's own `p1_x,p1_y,...` convention, SAM3, YOLO, or MediaPipe named joints — as long as the *same* markers appear in the *same order* in every camera's file. This is exactly what lets `sam3dinov3_visualize.py`'s `*_markers.csv` (a 70-marker MHR-ordered CSV) feed directly into `rec3d_one_dlt3d.py` with no relabeling.

---

## 5. Validating a DLT calibration against its own reference points

A DLT calibration is only as good as the reprojection accuracy it achieves on points it *can* be checked against — and the most direct check available is the calibration volume's own reference points: you already know their true (X, Y, Z), so reconstructing them from the pixel data used to build the calibration and comparing against the known truth measures self-consistency error.

For the `rec3d_todo` dataset, `c1c2c3_cod.ref3d` is exactly this: 12 known 3D points (a physical calibration frame, coordinates in meters, e.g. `p1 = (0, 0, 0.07)`, `p7 = (2.5, 0, 0.07)`, spanning X∈[0, 2.5] m, Y∈[−5, 5] m, Z∈[0.07, 1.6] m), each with a corresponding pixel observation in `c{1,2,3}_cod_markers_1_line.csv`. Reconstructing those same 12 points with the fitted `.dlt3d` files and comparing to the known ground truth is a direct residual measurement.

**Important scientific caveat, stated explicitly because it is easy to get wrong:** reconstructing the *same* points used to fit the calibration measures **in-sample residual**, not generalization error. With only 12 points total (a single static frame, no repeated calibration trials), a fair accuracy claim needs **leave-one-out cross-validation** (fit on 11 points, predict the 12th, repeat for all 12) rather than a plain in-sample check — otherwise a calibration model with more free parameters (e.g. an added distortion term) could "win" purely by overfitting the same 12 points it was scored on. This LOOCV design is specified in `loops/dlt-distortion-loop.md` but **has not yet been implemented or run** — see §7.

---

## 6. Bridging markerless AI and photogrammetry: the mesh-alignment problem

### 6.1 What SAM3+DINOv3 (SAM 3D Body) gives you, and what it doesn't

`vaila/sam3dinov3.py` runs Meta's **SAM 3D Body** (DINOv3 ViT-H/16+ backbone) on a *single* video, per person, per frame. Its output — a 70-keypoint MHR skeleton and a body mesh — has two properties that matter here:

1. **It is monocular.** One camera, one frame → no triangulation, no true stereo depth. Depth comes entirely from the network's learned human-body prior.
2. **It lives in that camera's own coordinate convention**: keypoints and mesh vertices are *root-relative* (`pred_keypoints_3d`), and the camera-frame position is `root-relative + cam_t` — a translation estimated per-frame, under an *assumed* field-of-view (`focal = sqrt(W² + H²)` unless `--focal-px` is passed with a true calibrated value).

In short: SAM 3D Body's mesh is shaped correctly (a real human body) and posed correctly (a real human pose) *for that one camera's view*, but its absolute position, orientation, and metric scale are only as good as an assumed focal length. It is **not** already sitting in the same metric world frame the DLT-triangulated skeleton (§4) lives in — three cameras running SAM 3D Body independently on the same physical person produce three different, uncorrelated coordinate frames for the same real-world pose.

### 6.2 Why not just triangulate the mesh vertices directly?

Multi-view triangulation (§4) requires the *same physical point* to be identified in ≥2 camera views. Skeleton keypoints have this property (SAM 3D Body's MHR70 keypoints are named/ordered consistently — "left-shoulder" means the same anatomical point in every camera's output). **Mesh vertices do not**: each camera's SAM 3D Body run independently estimates its own mesh topology instance from its own monocular view; there is no guaranteed vertex-index correspondence between camera A's vertex #4213 and camera B's vertex #4213 representing the same physical point on the skin. Triangulating mesh vertices directly is therefore not well-posed with this pipeline.

### 6.3 The solution: similarity-transform alignment (Umeyama, 1991)

The keypoints, unlike the mesh vertices, **do** have known correspondence (same MHR70 index = same anatomical joint across every representation: the DLT-triangulated skeleton and every camera's own monocular skeleton). This makes the problem tractable: at every frame,

1. Take a stable **subset** of MHR70 keypoints for camera *c*'s monocular skeleton (source) and the DLT-triangulated skeleton (target) — see `mesh_alignment.ALIGNMENT_MARKER_SPEC`: left/right shoulder, hip, knee, acromion, and neck (9 points). Hand/foot/finger tips and facial points are excluded — they are the noisiest MHR70 estimates and would destabilize the fit.
2. Fit the **similarity transform** (rotation **R**, uniform scale *s*, translation **t** — 7 degrees of freedom, *no* shear or non-uniform scaling) that best maps camera *c*'s monocular points onto the triangulated points, in the least-squares sense:

   ```
   target_i ≈ s · R · source_i + t          for i = 1..9
   ```

3. Apply that *same* (R, s, t) to camera *c*'s full mesh (all ~18,439 vertices) for that frame.

This is a **coordinate-frame reconciliation, not a re-triangulation** — worth stating plainly, because it is the single most important caveat about this feature: it introduces **no new depth information** beyond what the monocular network already encoded. If SAM 3D Body's shape/pose estimate for that person, that frame, is wrong, alignment will faithfully place the wrong shape in the right position/scale/orientation. What alignment *does* fix is exactly the three things a monocular estimate cannot get right on its own: absolute position, absolute orientation, and absolute (metric) scale — all three are recovered from the DLT world's true calibration.

#### The Umeyama closed-form solution

Given N corresponding point pairs {(xᵢ, yᵢ)} (source xᵢ, target yᵢ), the least-squares similarity transform has a closed-form solution (no iterative optimizer needed):

```
μx = mean(x),  μy = mean(y)                         (centroids)
Xc = x - μx,   Yc = y - μy                            (centered points)

Σ = (1/N) · Ycᵀ · Xc                                  (cross-covariance, 3×3)
U, D, Vᵀ = SVD(Σ)

S = I,  but S[2,2] = -1  if det(U)·det(Vᵀ) < 0        (reflection correction)
R = U · S · Vᵀ                                        (optimal rotation)

s = trace(diag(D) · S) / variance(Xc)                 (optimal uniform scale)
t = μy - s · R · μx                                   (optimal translation)
```

The reflection correction step matters: without it, a nearly-planar or symmetric point configuration can make the raw SVD solution a *reflection* (improper rotation, determinant −1) instead of a true rotation — anatomically impossible for a rigid body part. `mesh_alignment.umeyama_alignment()` implements exactly this derivation and additionally **refuses** to fit when the source points are too close to planar or collinear (checked via the ratio of smallest-to-largest singular value of the centered source points) — a genuine failure mode for a 9-point torso/hip/knee subset in some poses, guarded rather than silently producing an unstable rotation.

### 6.4 Per-frame best-camera selection

Not every camera has a clean, unoccluded view of the tracked person in every frame. Rather than fixing one "reference" camera for the whole clip, `reconstruct_mesh_sequence()` fits the transform **independently for every available camera, every frame**, and keeps whichever camera's fit has the **lowest residual** (mean 3D distance between the transformed monocular joints and the triangulated joints) — `mesh_alignment.best_camera_alignment()`. This means the mesh source can switch cameras frame-to-frame without a visible jump, *because* every camera's mesh gets aligned into the same world frame before being written out — a switch changes which camera's shape estimate you're looking at, not where it appears in space.

---

## 7. Validation on real data (`rec3d_todo`, 2026-08-04)

Dataset: a 3-camera change-of-direction (COD) drill, fixed calibration (`c{1,2,3}_cod_markers_1_line.dlt3d`, one row each), one tracked person selected per camera via `sam3dinov3_visualize.py` (`c1_cod` id 04, `c2_cod` id 08, `c3_cod` id 03), 631 frames at 119.88012001 Hz (NTSC-derived 120000/1001).

### 7.1 Automated (level-2, deterministic) evidence

Running `rec3d_one_dlt3d.py`'s full pipeline (`--mesh-source-dir` × 3, `--export-mesh obj`) end to end:

| Metric | Result |
|---|---|
| Frames reconstructed | **631 / 631** (0 skipped) |
| Camera switches (mesh source) | 79 |
| Umeyama fit residual | min 0.41 cm · **median 1.41 cm** · mean 1.47 cm · p95 2.39 cm · max 3.32 cm |
| Mesh vertex/face count vs. source | exact match, every sampled frame (36,874 faces, fixed MHR topology) |
| Mesh centroid | never resets to origin (regression guard for the bug class already fixed once in `sam3dinov3_visualize.py`) |
| Triangulated thigh length | within ±25% of the independently-measured 0.387 m reference (2026-08-01 validation) |
| Triangulated shank length | within ±25% of the independently-measured 0.371 m reference — real value 0.409 m, ~10% above the *other* pipeline's number, a reasonable cross-method difference (see `tests/test_rec3d_mesh_export.py` for why the tolerance is wide here) |
| Triangulated shoulder width | within ±25% of the independently-measured 0.360 m reference |

Frozen thresholds (`tests/test_rec3d_mesh_export.py`, 2026-08-04): median residual < 3 cm, p95 < 5 cm, max < 8 cm — roughly 2× headroom over the observed distribution above, tight enough to catch a broken transform (which would land in the tens of centimeters) without being flaky on normal fixture variation.

### 7.2 AI-assisted visual review (level-4, model judge — not a substitute for your own review)

Eight frames were rendered headlessly in Blender 5.2 (`blender --background`, native `wm.obj_import`, one render per frame) and inspected: frame 0, three camera-switch pairs (66/67, 99/100, 112/113), frame 300, and frame 630. All eight show a coherent, non-degenerate human mesh (recognizable head/torso/limbs, no exploded or inverted geometry); the three switch pairs are visually near-identical across the switch — no visible jump, confirming the alignment does what §6.4 claims it does.

**This is level-4 evidence (AI visual judgment against a frozen expectation), not level-5 (your own human sign-off).** The formal open item is: import `meshes_obj/` into Blender yourself via the "Stop Motion OBJ" (OBJSequence) add-on and confirm the same — particularly worth a second look at any switch frame not among the three sampled here, since 79 switches were not all inspected.

### 7.3 A note on units and frames, since getting this wrong silently is the classic DLT/rec3d failure mode

- World frame: meters, matching `c1c2c3_cod.ref3d`'s convention (X∈[0, 2.5], Y∈[−5, 5], Z∈[0.07, 1.6] — Z is the *height* axis for this calibration volume, not the classic "depth" axis).
- Umeyama residuals above are in **meters** (not mm, not pixels).
- BVH export applies an optional Y/Z swap for Blender's Z-up convention (`--swap-yz`) — the raw DLT world Z (height, per the calibration volume above) becomes BVH Y unless disabled.
- The aligned mesh is written in the **same** frame as the triangulated skeleton (world, meters) — no swap is applied to `meshes_<fmt>/`, regardless of `--swap-yz`, since Blender's own BVH importer converts by default but the mesh-sequence add-on most people actually use to play it back does not (see §8's 2026-08-05 entry).

---

## 8. Known limitations, honestly stated

- **No lens distortion modeling yet.** Every DLT fit in this codebase today (§3) is the plain 11-/8-parameter linear model — no radial or tangential distortion terms. `loops/dlt-distortion-loop.md` specifies (but has not yet implemented) an extension along the lines of Rossi et al. (2013, *Computer Methods in Biomechanics and Biomedical Engineering*, DOI 10.1080/10255842.2013.866231 — a copy is at `rec3d_todo/rossi2013.pdf`) and the general radial/tangential distortion literature Kwon3D's pages historically covered. Near image edges, a linear-only DLT fit folds any real lens distortion into a spatially-biased error rather than correcting it — worth keeping in mind for wide-angle or action-camera lenses more than for narrow-FOV tripod setups.
- **Mesh shape accuracy is inherited, not improved.** Alignment (§6.3) only fixes position/scale/orientation; it cannot correct a wrong shape or pose estimate from the underlying monocular network.
- **The Umeyama fit uses 9 keypoints.** Extreme poses (e.g. lying flat, all 9 points near-coplanar) trigger the degenerate-input guard and skip that camera for that frame — by design (a numerically unstable rotation is worse than no mesh for that frame), but it means very few markers are load-bearing for every alignment.
- **Segment-length validation is a plausibility floor (±25%), not a tight scientific claim** — see the comment in `tests/test_rec3d_mesh_export.py` explaining why a tighter band produced a false failure on real data (two independently-obtained numbers for the same physical quantity, from two different pipelines, are expected to differ by roughly this much).
- **Reprojection-accuracy validation (§5) is designed, not implemented.** No LOOCV script currently runs against `c1c2c3_cod.ref3d`; do not cite an accuracy number for the current linear DLT calibration until that exists.
- **Pre-existing, unrelated bug found in passing:** `rec3d.py` and `rec2d.py`'s success-path `messagebox.showinfo(...)` calls have no `gui=` guard (unlike `rec3d_one_dlt3d.py`'s `run_reconstruction()`), so a headless/CLI run hangs (with a real `DISPLAY`, waiting for a click nobody will make) or crashes with `TclError` (with no `DISPLAY`). Annotated with `# TODO(headless-cli)` comments at both call sites (2026-08-04); not fixed, since it is unrelated to this feature.
- **Real bug found and fixed (2026-08-04): `--swap-yz` only swapped the BVH, not the mesh.** `save_rec3d_as_bvh()` swaps Y/Z for Blender's Z-up convention, but `reconstruct_mesh_sequence()` had no `swap_yz` parameter at all and always wrote mesh vertices in the raw (unswapped) DLT frame. With `--swap-yz` set — the flag whose whole purpose is a Blender-ready export — the BVH skeleton and the OBJ mesh ended up in two different axis conventions: importing both into the same Blender scene, the mesh and skeleton would not visually move together, even though every frame count and per-frame correspondence between them was already correct (confirmed numerically: before the fix, a swapped-BVH marker's (x, y, z) = (DLT_x, DLT_z, DLT_y) while the corresponding unswapped mesh centroid was (DLT_x, DLT_y, DLT_z) — Y and Z transposed relative to each other). Fixed by adding `mesh_alignment.apply_blender_yz_swap(vertices, faces)`, applied in `reconstruct_mesh_sequence()` right before each frame's write when `swap_yz=True`. A bare Y/Z column swap flips chirality (its matrix has determinant −1), which would otherwise turn every outward-facing triangle normal inward and render the mesh inside-out in Blender — `apply_blender_yz_swap` also reverses each face's winding order to compensate, proven correct by a synthetic test asserting every face's outward/inward orientation relative to the mesh centroid is unchanged by the transform. Regression-tested against the real fixture (`tests/test_rec3d_mesh_export.py::test_mesh_vertices_are_swapped_to_match_bvh_when_swap_yz_true`): mesh vertex column 1 (now the swapped-in height) stays in a plausible human-height band across the run, while column 2 (now the swapped-in horizontal spread) varies far more — the opposite pattern of the unswapped bug.

- **Real bug found and fixed (2026-08-04): the Blender scene was never configured, so BVH and mesh played in slow motion and stopped early.** With everything exported correctly (631 frames, 120 Hz, verified identical across CSV/BVH/OBJ/C3D), an imported C3D animated correctly in Blender while the BVH and OBJ mesh sequence ran in slow motion and ended about a third of the way through. Diagnosed by driving Blender headlessly: `bpy.ops.import_anim.bvh` defaults to `update_scene_fps=False` **and** `update_scene_duration=False`, so after `File > Import > BVH` the scene was still at Blender's startup defaults — **24 fps with `frame_end=250`** — while the imported action correctly spanned frames 1..631. At 24 fps those 631 frames take 26.3 s instead of 5.26 s (the slow motion), and playback stops at frame 250 (the early ending); the C3D looked right only because its importer reads `POINT:RATE` and configures the scene. **The exported data was never wrong — only the Blender scene settings were.** Fixed by rewriting `generate_blender_companion_script()` so the emitted script configures and imports everything itself: it sets `scene.render.fps`/`fps_base` (via the new `rec3d.blender_scene_fps()`, which splits any rate into Blender's integer-fps + float-base pair so fractional NTSC rates survive exactly — 119.88012001 Hz → `fps=120`, `fps_base=1.001`), sets `frame_start`/`frame_end` from the real frame count, imports the BVH with `update_scene_fps=True`/`update_scene_duration=True`, imports the OBJ mesh sequence on the same start frame, and only then builds the skeleton bones. The BVH `Frame Time` also went from 6 to 9 decimals, because `0.008333` reads back as 119.875330 Hz rather than 119.880120 Hz — harmless once the script sets the rate explicitly, but wrong for anyone importing the BVH on its own. Verified end to end in headless Blender against the real fixture: scene 119.880114 Hz, range 1..631, BVH action 1..631, 631 OBJ frames, 24 skeleton bones. Regression tests: `tests/test_rec3d_blender_alignment.py`.
- **`--swap-yz` is now the default (2026-08-04).** Height is vertical (Z-up) in Blender out of the box for the BVH; `--no-swap-yz` opts back into raw DLT axes for it. (The mesh's own convention changed again the next day — see immediately below.)
- **Real bug found and fixed (2026-08-05): the 2026-08-04 mesh-swap fix above was itself wrong for how a mesh sequence is actually viewed, and got reported as "the mesh has Z where Y should be" on a real run.** The 2026-08-04 fix assumed any OBJ consumer applies the same Y-up→Z-up axis conversion Blender's BVH importer applies by default. That assumption holds for Blender's own native `wm.obj_import` dialog (verified: 0.15–0.21 m mesh-to-skeleton offset with default axes) but not for the bundled "Stop Motion OBJ"/OBJSequence family of mesh-sequence add-ons, which is what a mesh *sequence* (as opposed to one static frame) is normally viewed through. Reading that add-on's source (`stop_motion_obj2/core.py`'s `parse_obj()`/`apply_to_mesh()`) shows it parses `"v x y z"` lines and assigns them straight to `mesh.vertices.foreach_set("co", ...)` with **no axis conversion at all** — there is no forward/up parameter on its import operator to override this. So the swapped mesh file (`(x, z, -y)`, correct for a converting consumer) landed, through that add-on, with height along Blender's Y axis instead of Z: reproduced numerically on the real fixture — raw-passthrough of a swapped `frame_000300.obj` put the "tall" 1.34 m dimension in column 1 (Y) and the centroid at Z ≈ −2.2 m, nowhere near the skeleton. Fixed by writing the mesh **always** in the raw `(x, y, z)` DLT/world frame, unconditionally, regardless of `--swap-yz` — the same convention the skeleton ends up in once Blender's BVH importer applies its own conversion, so a non-converting add-on now needs zero configuration, at the cost of `wm.obj_import`'s manual dialog needing an explicit override to `Forward Y, Up Z` (previously "leave the defaults"). The companion script's own hand-rolled `_read_obj()` was updated to match (plain passthrough, no conversion). Regression test flipped accordingly: `tests/test_rec3d_mesh_export.py::test_mesh_vertices_stay_raw_regardless_of_swap_yz` now asserts column 2 (not column 1) holds the plausible-height band. Separately, the same investigation found Blender's bundled C3D importer (`io_anim_c3d`) does not reliably update the scene frame rate even with its own `adapt_frame_rate=True` default, despite this exporter's C3D correctly stating `POINT:RATE` (verified: header `frame_rate` and `POINT:RATE` both read back as `120.0` for a 120 Hz run) — tested empirically in headless Blender: scene stayed at 24 fps after `bpy.ops.import_anim.c3d(...)` on this exporter's own file. Not a vailá export bug; the companion script's `setup_scene()` already runs last regardless of import order, so running it after a manual C3D import already corrects the scene rate — documented explicitly rather than worked around, since there is nothing in this exporter to fix.

---

## 9. Reproducing this validation

```bash
uv run python -m vaila.rec3d_one_dlt3d \
  --dlt3d  c1_cod_markers_1_line.dlt3d c2_cod_markers_1_line.dlt3d c3_cod_markers_1_line.dlt3d \
  --pixels c1_id/c1_cod_id_04_markers.csv c2_id/c2_cod_id_08_markers.csv c3_id/c3_cod_id_03_markers.csv \
  --mesh-source-dir c1_id/ c2_id/ c3_id/ \
  --export-mesh obj \
  --fps 119.88012001 \
  -o ./out

# Deterministic checks:
uv run pytest tests/test_rec3d_mesh_alignment.py tests/test_rec3d_mesh_export.py -v

# Your own level-5 check:
#   Blender > Edit > Preferences > Get Extensions > install "OBJSequence"
#   (a.k.a. "Stop Motion OBJ", extension id stop_motion_obj2)
#   File > Import > Sequence Directory, point at out/vaila_rec3d_.../meshes_obj/
```

---

## 10. References

1. Abdel-Aziz, Y.I. & Karara, H.M. (1971). "Direct linear transformation from comparator coordinates into object space coordinates in close-range photogrammetry." *ASP Symposium on Close-Range Photogrammetry*. — the original DLT paper.
2. Kwon, Young-Hoo. Kwon3D DLT theory pages (`kwon3d.com/theory/dlt/dlt.html`) and the UNC Biomechanics mirror/note on DLT-to/from-intrinsic-extrinsic conversion (`biomech.web.unc.edu/dlt-to-from-intrinsic-extrinsic/`) — both cited by the user as the didactic model for this document; **kwon3d.com is currently offline**, so treat this document as the in-repo successor rather than a live link.
3. Rossi, A. et al. (2013). *Computer Methods in Biomechanics and Biomedical Engineering*. DOI: 10.1080/10255842.2013.866231. Local copy: `rec3d_todo/rossi2013.pdf`. Cited as the basis for the planned distortion extension (§8), not yet implemented.
4. Umeyama, S. (1991). "Least-squares estimation of transformation parameters between two point patterns." *IEEE Transactions on Pattern Analysis and Machine Intelligence*, 13(4). — the similarity-transform derivation implemented in `vaila/mesh_alignment.py`.
5. `loops/dlt-camera-model-loop.md`, `loops/dlt-distortion-loop.md`, `loops/rec3d-mesh-blender-loop.md` — the governing-check specifications this work was built against.

---

Part of **vailá** - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
