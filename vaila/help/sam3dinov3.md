# SAM3+DINOv3 3D (`sam3dinov3.py`)

## Module information

- **Category:** Markerless 3D / Meta (Facebook)
- **Version:** 0.3.108
- **Updated:** 2026-08-19
- **GUI:** Frame B → **Markerless 3D** → **SAM3+DINOv3 3D**
- **CLI:** Yes
- **Runtime:** NVIDIA CUDA required (the upstream estimator moves its batch to `cuda` unconditionally)
- **Retomada:** `--resume /caminho/processed_sam3dinov3_...` pula vídeos já concluídos e reaproveita `sam3/sam_tracks.csv`; informe também `-i` com a pasta original. Sem `--resume`, uma execução repetida com o mesmo `-i`/`-o` já retoma sozinha o `processed_sam3dinov3_*` correspondente (auto-resume); use `--fresh` para forçar uma pasta nova.

## What this pipeline does

Monocular **markerless 3D**: a single video in, a per-person 3D skeleton and body
mesh out. It is the 3D counterpart of [sam3sapiens2](sam3sapiens2.md) — same SAM3
front-end, different second stage.

1. **SAM 3 runs first** and defines each person's bounding box, silhouette, score
   and persistent `obj_id`.
2. **SAM 3D Body (DINOv3 ViT-H/16+ backbone)** receives only those SAM boxes and
   silhouettes, through
   `SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)`.
   The upstream **ViTDet detector and SAM2 segmentor are never loaded** — that
   saves VRAM and keeps SAM 3 as the single source of identity.
3. For each person the model regresses an **MHR (Momentum Human Rig)** mesh, 3D
   joints in metres, their 2D reprojection in pixels, the camera translation
   `cam_t` and the focal length.
4. `person_id == sam_obj_id`, so no second Re-ID stage can swap identities.

Because the boxes and masks come from a *video* tracker rather than per-frame
detection, identities stay stable across the whole clip — which is what makes the
per-joint 3D trajectories usable for biomechanics.

## Requirements

```bash
# SAM 3 (CUDA) stack
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam --yes
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights

# SAM 3D Body: clones facebookresearch/sam-3d-body and downloads gated weights
# (accept the license on the model card first)
bash bin/setup_fifa_sam3d.sh
```

> **Why there is no `pip install -e`.** Upstream `sam-3d-body` ships no
> `pyproject.toml`/`setup.py`, so it cannot be installed as a package. The setup
> script clones it to `sam_3d_body/` and installs only its runtime dependencies
> with `--no-deps` (protecting your CUDA torch build); this module then puts the
> **checkout root** on `sys.path` automatically. Set `VAILA_SAM3D_BODY_DIR` to
> use a checkout kept somewhere else.
>
> Two dependency traps worth knowing: `omegaconf` requires
> `antlr4-python3-runtime==4.9.3` (4.13 raises *"Could not deserialize ATN with
> version 3"*), and the MHR body model is the PyPI package **`mhr`**.

The weights land in `vaila/models/sam-3d-dinov3/`:

| File                   | Role                                     |
| ---------------------- | ---------------------------------------- |
| `model.ckpt`           | SAM 3D Body checkpoint (DINOv3 backbone) |
| `model_config.yaml`    | Backbone/decoder configuration           |
| `assets/mhr_model.pt`  | Momentum Human Rig body model            |

Both `facebook/sam3` and `facebook/sam-3d-body-dinov3` are **gated** — a 403 means
the Hugging Face account has not been granted access yet.

Model weights keep their respective **Meta licenses**, separate from vailá's AGPL
source license.

## GUI

Open **Markerless 3D → SAM3+DINOv3 3D**. The dialog collects the input video/folder,
the output parent, the SAM 3 text prompt, and the 3D options. Pressing **Run**
prints the equivalent CLI to the terminal (prefix `>>`) and then launches exactly
that command in an isolated GPU subprocess — so any GUI run is reproducible from
the shell.

## CLI

```bash
# Simplest run: SAM 3 + SAM 3D Body on one clip or a folder
uv run python -u vaila/sam3dinov3.py -i /path/to/video_or_folder -o /path/to/output -t person

# Reuse an existing SAM run instead of re-segmenting
uv run python -u vaila/sam3dinov3.py -i /path/to/videos -o /path/to/output \
    --sam-results /path/to/processed_sam_YYYYMMDD_HHMMSS

# Metric-accurate run with a known focal length, plus mesh export
uv run python -u vaila/sam3dinov3.py -i clip.mp4 -o /path/to/output \
    --focal-px 1400 --save-mesh

# Validate paths, weights and the plan without touching the GPU
uv run python -u vaila/sam3dinov3.py -i clip.mp4 -o /tmp/out --dry-run
```

### Main options

| Flag                     | Default   | Meaning                                                        |
| ------------------------ | --------- | -------------------------------------------------------------- |
| `-t, --text`             | `person`  | SAM 3 text prompt                                              |
| `--sam-results`          | –         | Reuse `processed_sam_*` (batch parent or per-video dir)         |
| `--weights-dir`          | auto      | SAM 3D Body weights dir                                        |
| `--inference-type`       | `full`    | `full` (body+hand), `body`, or `hand`                          |
| `--focal-px`             | auto      | Known focal length in px — see *Scale caveat*                   |
| `--fov-estimator`        | –         | Optional upstream FOV estimator (e.g. `moge2`); extra weights   |
| `--no-mask`              | off       | Use SAM boxes/IDs but skip mask-conditioned inference           |
| `--stride`               | `1`       | Run inference every Nth frame                                   |
| `--bbox-padding`         | `0.12`    | Context padding around the SAM contour box                      |
| `--contour-margin`       | `8`       | Dilation (px) applied to the SAM silhouette                     |
| `--max-persons`          | `32`      | Cap on people per frame                                        |
| `--save-mesh`            | off       | Write per-frame MHR vertices to `meshes/*.npz` (large)          |
| `--no-overlay`           | off       | Skip the overlay video                                          |
| `--dry-run`              | off       | Validate and print the plan, no GPU inference                   |
| `--resume RUN_DIR`       | –         | Continue an interrupted batch (explicit pin)                    |
| `--fresh`                | off       | Ignore any matching prior run under `-o`; force a new output dir |

### Automatic resume (default, no flag needed)

Re-running the same command (same `-i` and `-o`) auto-detects a matching
`processed_sam3dinov3_*` directory already under `-o` — via a
`BATCH_INPUT.json` marker written at batch start, or the completed run's own
`sam3dinov3_batch_summary.json` — and resumes it exactly like `--resume`
would, printing:

```text
Auto-resume: found matching run, reusing /path/to/processed_sam3dinov3_...
Resume: 3/8 videos already completed, 5 remaining
[SKIP] Already processed: clip_003.mp4
```

Pass `--fresh` to ignore any match and start a brand-new timestamped output
directory instead. `--fresh` and `--resume` are mutually exclusive.

## Outputs

Each video gets its own subdirectory inside `processed_sam3dinov3_<timestamp>/`.

| File                                    | Content                                                      |
| --------------------------------------- | ------------------------------------------------------------ |
| `<video>_sam3dinov3_overlay.mp4`        | SAM contour/bbox/ID + reprojected 3D skeleton, colored by joint side (left=green, right=orange, center/spine=blue) — v0.3.98, same palette as [sam3dinov3_visualize](sam3dinov3_visualize.md) |
| `<video>_sam3dinov3_keypoints3d.csv`    | Long table: root-relative **and** camera-frame metres         |
| `<video>_sam3dinov3_keypoints2d.csv`    | Long table: reprojected pixels                                |
| `<video>_sam3dinov3_camera.csv`         | Per-frame focal length, `cam_t`, bbox                         |
| `<video>_sam3dinov3_joint_angles.csv`   | Long table: local (parent-relative) joint angles for the model's own 127-joint MHR rig — Euler XYZ degrees + scalar-first quaternion, from the model's own regressed rotations — v0.3.99, see [joint_kinematics](joint_kinematics.md) |
| `<video>_id_NN_mhr70_3d.csv`            | Wide, named columns (`nose_x,nose_y,nose_z,…`)                |
| `<video>_id_NN_mhr70_rec3d.csv`         | Wide, vailá `rec3d` convention (`p1_x,p1_y,p1_z,…`)           |
| `<video>_id_NN_markers.csv`             | Wide 2D for REC2D / `getpixelvideo.py`                        |
| `<video>_sam3dinov3_predictions.json.gz`| Full provenance + per-instance predictions                    |
| `meshes/frame_NNNNNN.npz`               | Only with `--save-mesh`: vertices, `obj_ids`, `cam_t`         |
| `mesh_faces.npy`                        | Only with `--save-mesh`: shared MHR topology                  |
| `sam3dinov3_summary.json`               | Machine-readable run summary                                  |
| `README_sam3dinov3.txt`                 | Verbose per-run schema/units description                      |
| `FAILED_sam3dinov3.txt`                 | Present only when the stage fails                             |

`NN` is the SAM `obj_id`, zero-padded — the same identity you see in the overlay
and in the SAM outputs.

## Keypoints

The analysis columns use the **first 70 of the 308 MHR keypoints** (`mhr70`):
body, feet, full hands, plus anatomical extras (`left-olecranon`,
`right-cubital-fossa`, `left-acromion`, `neck`, …). Hyphens become underscores in
CSV headers (`left-shoulder` → `left_shoulder_x`).

## Coordinate systems and units

| Columns                | Meaning                                                                 |
| ---------------------- | ----------------------------------------------------------------------- |
| `x_m, y_m, z_m`        | Root-relative 3D joints, metres, camera axes (OpenCV: +x right, +y down, +z forward) |
| `xcam_m, ycam_m, zcam_m` | Camera-frame absolute joints = root-relative + `cam_t`                |
| `x_px, y_px`           | Perspective reprojection into the original full frame                    |
| `frame`                | Zero-based, matching the source video and the SAM outputs                |

Use the **`xcam_*` columns** for inter-person distances and depth; the wide
`*_mhr70_3d.csv` / `*_mhr70_rec3d.csv` files already contain those.

> ### Scale caveat
>
> Monocular depth is metric only up to the assumed camera intrinsics. Without
> `--focal-px` the model falls back to a default FOV (`f = sqrt(W² + H²)`), so
> absolute depth inherits that assumption. Supply the true focal length in pixels
> (or a FOV estimator) whenever absolute distances matter. Relative joint
> geometry within one person is far less sensitive to this than absolute depth.

## Performance notes

- One **isolated GPU subprocess per video**, so a CUDA OOM on one clip cannot
  poison the rest of the batch (`--no-isolate-batch` disables this for debugging).
- `--save-mesh` writes roughly **220 KB per person per frame** — enable it only
  when you actually need the surface, not just joints.
- `--stride N` trades temporal resolution for speed; frames between inference
  steps carry no instances.
- The upstream model prints a banner per person per frame; that chatter is
  silenced by default (`--verbose-model` restores it).

## Related modules

- [sam3sapiens2](sam3sapiens2.md) — same SAM3 front-end, 2D 308-keypoint pose
- [sapiens2_3d](sapiens2_3d.md) — this pipeline's bbox tightened by Sapiens2 keypoints before the same SAM 3D Body call
- [vaila_sam](vaila_sam.md) — SAM 3 video segmentation on its own
- [joint_kinematics](joint_kinematics.md) — shared math for `*_sam3dinov3_joint_angles.csv`
- [vaila_sapiens](vaila_sapiens.md) — Sapiens2 pose

## References

- SAM 3 — <https://ai.meta.com/research/sam3/>
- DINOv3 — <https://ai.meta.com/research/dinov3/>
- SAM 3D Body — <https://github.com/facebookresearch/sam-3d-body>
- Weights — <https://huggingface.co/facebook/sam-3d-body-dinov3>
- Rerun viewer — <https://github.com/rerun-io/sam3d-body-rerun>
