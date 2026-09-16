# Tatame Fixed-Scene Stabilization and Multimodal AI 3D Analysis Guide

**Version:** 0.4.3 · **Updated:** 16 September 2026  
**Module Suite:** Video Stabilizer (`vaila.video_stabilizer`), Planar Geometry Tracker (`vaila.planar_geometry_tracker`), SAM3+DINOv3 3D (`vaila.sam3dinov3`), Monocular DLT World Alignment (`vaila.monocular_dlt_align`).

---

## 1. Overview: From Handheld Video to Calibrated Metric 3D Kinematics

In human movement and sports biomechanics, capturing activities (e.g., martial arts, gymnastics, vertical jumps) using mobile or handheld cameras introduces camera shake, ego-motion, and perspective variations. 

This guide provides the complete, reproducible 4-stage pipeline that transforms a handheld video into **calibrated 3D skeletal kinematics and 3D body meshes aligned with real-world metric coordinates (tatame mat)**:

```
[Raw Handheld Video + Static Scene Markers]
                     │
                     ▼ (Stage 1: Video Stabilizer)
[Stabilized Video (*.mp4) + Stabilized Markers (*.csv)]
                     │
                     ├─────────────────────────────────────────┐
                     ▼ (Stage 2: Planar Geometry Tracker)      ▼ (Stage 3: SAM3+DINOv3 3D)
[Floor DLT2D Homography + Metric Tatame Mesh (1x1m)]     [MHR70 3D Joints (m) + 36k Meshes]
                     │                                         │
                     └────────────────────┬────────────────────┘
                                          ▼ (Stage 4: Monocular DLT Alignment)
                     [Calibrated Metric 3D Kinematics in World Coordinates]
```

---

## 2. Prerequisites & Complete AI Workstation Installation

The multimodal AI pipeline requires NVIDIA CUDA GPU acceleration, SAM 3 video tracking, and the SAM 3D Body (DINOv3 backbone) stack.

### 2.1 One-Command Installation (`--full` / `-Full`)

Install everything (PyTorch CUDA, SAM 3, Sapiens2, SAM 3D Body, and runtime dependencies) with a single command:

**🐧 Linux:**
```bash
# In an existing git clone:
./install_vaila_linux.sh --full

# Or from anywhere via setup_pyproject:
bash bin/setup_pyproject.sh --target=linux-cuda --full --yes
```

**🪟 Windows (PowerShell):**
```powershell
# In an existing git clone:
.\install_vaila_win.ps1 -Full

# Or via setup_pyproject:
pwsh bin/setup_pyproject.ps1 -Target win-cuda -Full -Yes
```

### 2.2 Hugging Face Authentication (Gated Models)

SAM 3 and SAM 3D Body (DINOv3) use gated model repositories on Hugging Face. Accept the license on their model cards:
1. **SAM 3:** [https://huggingface.co/facebook/sam3](https://huggingface.co/facebook/sam3)
2. **SAM 3D Body:** [https://huggingface.co/facebook/sam-3d-body-dinov3](https://huggingface.co/facebook/sam-3d-body-dinov3)

Log in from your terminal:
```bash
uv run hf auth login
uv run hf auth whoami   # Must exit 0 and show your username
```

### 2.3 Automated Preflight Verification

Verify that all 4 AI engines are properly configured on your GPU:
```bash
uv run python -c "
import torch
print('CUDA available:', torch.cuda.is_available(), '-', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU')
import sam3; print('SAM 3: OK')
import sapiens; print('Sapiens2: OK')
from vaila.sam3dinov3 import ensure_sam3d_ready
ckpt, _ = ensure_sam3d_ready(); print('SAM 3D Body (DINOv3): OK -', ckpt.name)
"
```
Expected output:
```text
CUDA available: True - NVIDIA GeForce RTX 4090
SAM 3: OK
Sapiens2: OK
SAM 3D Body (DINOv3): OK - model.ckpt
```

---

## 3. Stage 1: Video Stabilization (`vaila.video_stabilizer`)

### 3.1 Concept & Modes

The video stabilizer tracks physically fixed scene markers (digitized in `getpixelvideo.py`) across frames to cancel camera jitter and ego-motion:
- **`--mode visual` (Recommended):** Fits a shape-preserving transform (`similarity`) across all frames. Preserves aspect ratio, eliminates shake, and keeps human body proportions exact without projective warping distortion.
- **`--mode hybrid`:** Performs visual stabilization and additionally calculates per-frame floor-plane homographies using `--geometry-config` and `--metric-markers`, exporting condition numbers and floor reprojection diagnostics.
- **`--canvas union`:** Expands the canvas dynamically to include all visible pixels across the entire clip without cropping or artificial zooming. Missing borders are filled with clean black borders.

### 3.2 Execution

Using the test fixture `tatame.mp4` (331 frames, 1080×1920, 59.94 FPS) and `tatame_markers.csv`:
- `p0` to `p7`: Floor tatame perimeter and midpoints.
- `p8` to `p10` (or `p8` to `p13`): Background anchors (wall marks, pillars).

```bash
# Visual stabilization with similarity model (shape-preserving)
uv run python -m vaila.video_stabilizer \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --mode visual \
  --model similarity \
  --canvas union \
  --debug-overlay \
  --output-dir tests/video_stabilizer/stabilizer_results/visual

# Hybrid mode (visual stabilization + metric floor diagnostics)
uv run python -m vaila.video_stabilizer \
  --video tests/video_stabilizer/tatame.mp4 \
  --markers tests/video_stabilizer/tatame_markers.csv \
  --stabilization-markers 0-10 \
  --anchor-markers 8,9,10 \
  --metric-markers 0-7 \
  --geometry-config vaila/models/planar_targets/tatame_1x1m.toml \
  --mode hybrid \
  --model similarity \
  --canvas union \
  --output-dir tests/video_stabilizer/stabilizer_results/hybrid
```

### 3.3 Measured Results & Benchmark

| Static markers | RMS before (px) | RMS after (px) | Improvement |
| --- | ---: | ---: | ---: |
| All selected markers | 77.656 | 9.484 | **87.8% reduction** |
| Background anchors p8–p10 | 90.244 | 10.109 | **88.8% reduction** |
| Floor p0–p7 (hybrid diagnostics) | 70.733 | 9.165 | **87.0% reduction** |

- Scale correction: strictly between 0.99113 and 1.14621 (consistent with focal distance).
- Rotation: −1.50° to +2.47°.
- Canvas size: 1372×2330 px (union extent preserving all camera FOV).

### 3.4 Key Output Artifacts
- `<stem>_stabilized.mp4`: Stabilized high-resolution video for downstream tracking.
- `stabilized_markers.csv`: Original marker observations mapped into the stabilized canvas frame.
- `stabilization_report.html`: Comprehensive report with RMS displacement graphs and scale metrics.
- `debug_overlay.mp4`: Side-by-side comparison video showing reference markers and residual vectors.

---

## 4. Stage 2: Planar Geometry Calibration & Target Extrapolation (`vaila.planar_geometry_tracker`)

### 4.1 Target Definition (`tatame_1x1m.toml`)

Martial arts EVA mats provide an accurate physical metric reference. A standard interlocking mat module has a 0.960 m interlocked pitch:
```toml
[target]
name = "tatame_eva_1x1m"
description = "Interlocking martial arts EVA mat (0.96m modular interlocked pitch)"
type = "polygon"

[points]
0 = { name = "corner_sw", x = 0.000, y = 0.000, z = 0.0 }
1 = { name = "mid_s",     x = 0.480, y = 0.000, z = 0.0 }
2 = { name = "corner_se", x = 0.960, y = 0.000, z = 0.0 }
3 = { name = "mid_e",     x = 0.960, y = 0.480, z = 0.0 }
4 = { name = "corner_ne", x = 0.960, y = 0.960, z = 0.0 }
5 = { name = "mid_n",     x = 0.480, y = 0.960, z = 0.0 }
6 = { name = "corner_nw", x = 0.000, y = 0.960, z = 0.0 }
7 = { name = "mid_w",     x = 0.000, y = 0.480, z = 0.0 }
```

### 4.2 Collinearity & Square Shape Preservation

When an athlete steps on or occludes a mat corner (e.g. `p0`), direct digitization fails. `vaila.planar_geometry_tracker`:
1. Fits per-frame **DLT2D** (`dlt2d.py`) on visible world↔pixel pairs.
2. Uses projective collinearity (straight lines in the world map to straight lines in perspective projection) to reconstruct missing points (such as `p0` from lines `p6-p7-p0` and `p2-p1-p0`).
3. Re-projects the metric square onto the stabilized canvas, preventing perspective distortion or skewed non-square geometries.

### 4.3 Execution

```bash
uv run python -m vaila.planar_geometry_tracker \
  --config vaila/models/planar_targets/tatame_1x1m.toml \
  --measurements-csv tests/video_stabilizer/stabilizer_results/visual/stabilized_markers.csv \
  --video-path tests/video_stabilizer/stabilizer_results/visual/tatame_stabilized.mp4 \
  --output-dir tests/video_stabilizer/geometry_results \
  --debug-viz
```

### 4.4 Key Output Artifacts
- `stabilized_markers_imputed.csv`: Complete marker coordinate series with occluded points imputed.
- `debug_projected_wireframe.mp4`: Video with the tatame metric square and grid overlaid on each frame.
- `geometry_animation.html`: Interactive browser viewer showing 2D pixel projection and world rec2d coordinates frame by frame.

---

## 5. Stage 3: Monocular Markerless 3D Mesh & Motion Capture (`vaila.sam3dinov3`)

### 5.1 Why Video Stabilization Enhances 3D Capture

SAM 3D Body uses high-capacity vision transformers (DINOv3 ViT-H backbone) to regress 3D human pose, metric root translation, and 3D body meshes from bounding boxes and silhouettes. 
Applying stabilization first:
- Eliminates camera ego-motion jitter from person trajectories.
- Improves silhouette edge stability and temporal smoothness of joint angles.
- Ensures bounding boxes adhere strictly to subject movement rather than camera bobbing.

### 5.2 Execution

Run the SAM3 + DINOv3 pipeline on the stabilized video:

```bash
# GUI Launch:
uv run python vaila/sam3dinov3.py

# CLI Direct Execution:
uv run python -m vaila.sam3dinov3 \
  -i tests/video_stabilizer/stabilizer_results/visual/tatame_stabilized.mp4 \
  -o tests/video_stabilizer/sam3d_results \
  --export-mesh \
  --device cuda
```

### 5.3 Key Output Artifacts
In the output folder `<output_dir>/<video_stem>/`:
- `*_sam3dinov3_overlay.mp4`: High-resolution video rendered with 3D skeleton reprojections, bounding boxes, and body contours.
- `*_sam3dinov3_keypoints3d.csv`: Metric 3D joint coordinates (in meters) for all 70 MHR keypoints per person per frame.
- `*_sam3dinov3_joint_angles.csv`: Anatomical joint angles in degrees (elbow, knee, hip, shoulder flexion/extension, etc.).
- `*_sam3dinov3_keypoints2d.csv`: 2D pixel coordinates of all reprojected 3D joints.
- `*_sam3dinov3_camera.csv`: Estimated focal length and metric camera translation `[tx, ty, tz]`.
- `meshes/*.npz` & `mesh_faces.npy`: Full 3D human body surface meshes (36,874 faces per person per frame) readable by PyVista, Blender, or Trimesh.
- `*_id_XX_markers.csv` & `*_rec3d.csv`: Per-person kinematics formatted for direct import into vailá tools (`rec3d.py`, DLT reconstruction).

---

## 6. Stage 4: Monocular to World Coordinate Alignment (`vaila.monocular_dlt_align`)

The camera translation `cam_t` and 3D joints from Stage 3 are initially in the camera's reference frame. Stage 4 places the 3D human motion directly onto the tatame floor plane:

1. The floor plane homography from Stage 2 defines the ground equation ($Z_{world} = 0$) and mat origin ($X=0, Y=0$).
2. `vaila.monocular_dlt_align` computes the rigid transformation ($R, t$) from camera coordinates to the world reference frame.
3. The resulting trajectory places the subject walking, jumping, or performing martial arts directly on top of the metric tatame coordinates in meters.

```bash
uv run python -m vaila.monocular_dlt_align \
  --keypoints3d tests/video_stabilizer/sam3d_results/tatame_stabilized/tatame_stabilized_sam3dinov3_keypoints3d.csv \
  --camera-csv tests/video_stabilizer/sam3d_results/tatame_stabilized/tatame_stabilized_sam3dinov3_camera.csv \
  --floor-npz tests/video_stabilizer/stabilizer_results/hybrid/floor_homographies.npz \
  --output-dir tests/video_stabilizer/world_aligned
```

---

## 7. Verification & Automated Tests

All stages are covered by automated unit and integration tests:

```bash
# Run stabilization and planar geometry tests:
uv run pytest tests/test_video_stabilizer.py tests/test_planar_geometry_tracker.py -v

# Run SAM 3 and 3D pipeline unit tests:
uv run pytest tests/test_vaila_sam.py tests/test_fifa_skeletal_pipeline.py -v

# Run all biomechanics and tracking tests:
uv run pytest tests/ -q
```

---

## 8. Troubleshooting & Common Questions

1. **`ModuleNotFoundError: No module named 'braceexpand'` or `'mhr'`:**
   - Run the full setup script: `bash bin/setup_fifa_sam3d.sh` or install missing packages with:
     `uv pip install --no-deps mhr yacs omegaconf "antlr4-python3-runtime==4.9.3" roma trimesh braceexpand pytorch-lightning torchmetrics lightning-utilities termcolor`.
2. **CUDA Out of Memory (OOM) on high-resolution video:**
   - Reduce long-edge resolution with `--max-input-long-edge 1280` or `--frame-by-frame` in `vaila_sam.py`.
   - `sam3dinov3.py` isolates video segmentation in subprocesses to ensure clean GPU VRAM between videos.
3. **Occluded Floor Markers:**
   - Use `--geometry-config vaila/models/planar_targets/tatame_1x1m.toml` in `vaila.planar_geometry_tracker` to extrapolate occluded corners while preserving straight lines and square aspect ratio.
4. **Git Conflicts after CUDA switch:**
   - The git repository tracks the portable CPU PyTorch configuration. Use `bash bin/sync_repo.sh` to update the repository across machines without merge conflicts.
