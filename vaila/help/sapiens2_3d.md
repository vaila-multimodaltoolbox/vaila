# Sapiens2 3D Pose (`sapiens2_3d.py`)

## Module information

- **Category:** Markerless 3D / Meta (Facebook)
- **Version:** 0.3.104
- **Updated:** 2026-08-11
- **GUI:** Frame B → **Markerless 3D** → **Sapiens2 3D Pose**
- **CLI:** Yes
- **Runtime:** NVIDIA CUDA required (same SAM 3D Body estimator as `sam3dinov3.py`); the optional DLT3D auto-chain step itself is CPU-only

## Scope, read this first

Sapiens2 as vendored in this repo (`vaila_sapiens.py`) is a **2D-only**
top-down pose model — x, y, score, no depth/normal/mesh head. It cannot
produce 3D on its own. The vendored SAM 3D Body estimator
(`sam_3d_body_estimator.py`'s `process_one_image()`) also accepts **no
external keypoint/pose-hint parameter at inference time** — only
`bboxes`/`masks`/`cam_int`. Its internal `keypoint_prompt_sampler` machinery
is a *training*-time self-refinement loop that reads ground-truth
`batch["keypoints_2d"]`, a field the inference-time batch builder never
populates; reusing it would mean patching unsupported internal state in
vendored upstream code.

So this module's guidance is **bbox tightening only**: Sapiens2's 308
keypoints (already computed by [sam3sapiens2](sam3sapiens2.md)) replace SAM3's
mask-derived bbox with a tighter, keypoint-derived one whenever enough
confident keypoints are available *and* they geometrically agree with the SAM3
bbox (an IoU sanity check — see below). The 3D lifter itself — SAM 3D Body
(DINOv3 backbone), the MHR mesh, all downstream outputs — is byte-identical to
[sam3dinov3](sam3dinov3.md). This module never modifies `sam3dinov3.py` or
`sam3sapiens2.py`; it only imports and calls their functions.

## What this pipeline does

1. Reuses an existing [sam3sapiens2](sam3sapiens2.md) combined run (SAM3
   bbox/contour/ID authority + Sapiens2 308-keypoint 2D pose), or runs that
   stage itself from a raw SAM3 result if only `--sam-results` is given.
2. Per frame/person, tightens the SAM3 bbox using Sapiens2 keypoints:
   - Needs at least `--min-guidance-keypoints` (default 4) keypoints scoring
     ≥ `--kpt-score-thresh` (default 0.3).
   - The resulting keypoint bbox must overlap the SAM3 bbox by at least
     `--min-sanity-iou` (default 0.05) — guards against a misassigned or
     drifted Sapiens2 detection silently sending the 3D lifter's crop
     somewhere unrelated to the tracked person.
   - Otherwise falls back to the unmodified SAM3 bbox.
3. Feeds the resulting bbox + the SAM3 contour mask into
   `SAM3DBodyEstimator.process_one_image(bboxes=..., masks=...)` — the exact
   call `sam3dinov3.py` already makes.
4. Writes the same MHR70 long/wide CSV + camera CSV + optional mesh family
   `sam3dinov3.py` writes (by calling its writer functions directly), plus one
   extra CSV recording, per frame/person, whether Sapiens2 guidance was used.
5. **Optional:** when `--dlt3d` is given, each person is automatically placed
   into the DLT-calibrated lab frame — see *Calibrated lab frame* below.

## Smart input resolution (v0.3.101)

Pointing `--sapiens2-results`/`-i` at a plausible-but-wrong directory (e.g. a
[sam3sapiens2_visualize](sam3sapiens2_visualize.md) single-ID rerender output
instead of the combined run it came from) now resolves automatically instead
of failing with a bare "not found":

- A directory holding **exactly one** `*_sam3sapiens2_predictions.json` is
  used even if its name doesn't match the video you're processing.
- A `<base>_sam3sapiens2_visualized_id_NN` (or `..._sam3dinov3_visualized_id_NN`)
  directory resolves to its sibling combined-run directory `<base>/`.
- If `-i` doesn't resolve to a usable video but `--sapiens2-results` does, the
  raw video is auto-located by walking upward from the results directory
  looking for the exact filename the predictions JSON itself remembers
  (`payload["video"]`) — this is how a batch output nested under (or next to)
  the original raw-input folder gets found without typing its path.

A genuine mismatch — the resolved predictions JSON was built from a
*different* video than the one about to be processed — still raises an error
rather than silently proceeding: mismatched guidance would misalign every
frame with no visible symptom.

## Requirements

Same as [sam3dinov3](sam3dinov3.md) **plus** the Sapiens2 stack:

```bash
# SAM 3 (CUDA) stack
bash bin/setup_pyproject.sh --target=linux-cuda --extras=gpu,sam --yes
uv run hf auth login
uv run vaila/vaila_sam.py --download-weights

# SAM 3D Body: clones facebookresearch/sam-3d-body and downloads gated weights
bash bin/setup_fifa_sam3d.sh

# Sapiens2 (CUDA) stack
uv sync --extra sapiens
bash bin/setup_sapiens2.sh
```

You also need an existing **SAM3+Sapiens2** run for the video (see
[sam3sapiens2](sam3sapiens2.md)) — or point `--sam-results` at a raw SAM3 run
and this module will run that stage for you.

## GUI

Open **Markerless 3D → Sapiens2 3D Pose**. The dialog collects the input
video/folder, the output parent, either an existing `processed_sam3sapiens2_*`
directory (preferred) or a raw SAM3 results directory, and the bbox-tightening
options. An optional **Calibrated lab frame** section takes a `.dlt3d` (and
optional `.ref3d`) to auto-chain into `monocular_dlt_align.py` after the run —
see below. Pressing **Run** prints the equivalent CLI to the terminal (prefix
`>>`) and launches exactly that command in an isolated GPU subprocess. Only
`--dlt3d`/`--ref3d`/`--export-mesh` are exposed in the dialog; the DLT-chain's
other knobs (`--smooth-hz`, `--no-refine`, `--origin-markers`, `--skeleton`)
keep their defaults unless set from the CLI.

## CLI

```bash
# Reuse an existing SAM3+Sapiens2 combined run:
uv run python -u vaila/sapiens2_3d.py \
    -i /path/to/video.mp4 -o /path/to/output \
    --sapiens2-results /path/to/processed_sam3sapiens2_YYYYMMDD_HHMMSS

# Only a raw SAM3 run exists -- the Sapiens2 stage runs first automatically:
uv run python -u vaila/sapiens2_3d.py \
    -i /path/to/video.mp4 -o /path/to/output \
    --sam-results /path/to/processed_sam_YYYYMMDD_HHMMSS

# Auto-chain into the DLT3D-calibrated lab frame (per person):
uv run python -u vaila/sapiens2_3d.py \
    -i /path/to/video.mp4 -o /path/to/output \
    --sapiens2-results /path/to/processed_sam3sapiens2_YYYYMMDD_HHMMSS \
    --dlt3d /path/to/camera.dlt3d --ref3d /path/to/control_points.ref3d \
    --save-mesh --export-mesh obj
```

### Main options

| Flag                        | Default | Meaning                                                          |
| --------------------------- | ------- | ----------------------------------------------------------------- |
| `--sapiens2-results`        | –       | Existing `sam3sapiens2.py` output dir (preferred front end)        |
| `--sam-results`              | –       | Raw SAM3 output; the Sapiens2 stage runs first if given instead    |
| `--min-guidance-keypoints`  | `4`     | Min. confident Sapiens2 keypoints required to trust its bbox       |
| `--kpt-score-thresh`         | `0.3`   | Sapiens2 keypoint confidence threshold                             |
| `--kpt-bbox-padding-frac`   | `0.08`  | Padding around the tight keypoint bounding box                     |
| `--min-sanity-iou`           | `0.05`  | Min. IoU vs the SAM3 bbox for guidance to be trusted                |
| `--weights-dir`              | auto    | SAM 3D Body weights dir                                            |
| `--inference-type`           | `full`  | `full` (body+hand), `body`, or `hand`                              |
| `--focal-px`                 | auto    | Known focal length in px — see *Scale caveat* in sam3dinov3         |
| `--no-mask`                  | off     | Use SAM boxes/IDs but skip mask-conditioned inference               |
| `--stride`                   | `1`     | Run inference every Nth frame                                       |
| `--save-mesh`                | off     | Write per-frame MHR vertices to `meshes/*.npz` (large)              |
| `--no-overlay`               | off     | Skip the overlay video                                              |

### Calibrated lab frame (DLT3D auto-chain) options

| Flag                  | Default | Meaning                                                                    |
| ---------------------- | ------- | --------------------------------------------------------------------------- |
| `--dlt3d`              | –       | This camera's DLT3D calibration. Given, triggers the auto-chain per person   |
| `--ref3d`              | –       | Control points, for validation only (same convention as `monocular_dlt_align.py`) |
| `--smooth-hz`          | `6.0`   | Butterworth cutoff on the 6-DOF placement (0 disables via `--no-smooth`)     |
| `--no-smooth`          | off     | Disable placement smoothing (raw)                                           |
| `--no-refine`          | off     | Translation only (skip the 6-DOF refinement); usually worse                 |
| `--origin-markers`     | `10 11` | 1-based MHR70 markers whose midpoint the placement rotates about (hips)      |
| `--skeleton`           | –       | Skeleton JSON for the Blender companion script                              |
| `--export-mesh`        | `obj`   | `none`/`obj`/`ply` — needs `--save-mesh` in this same run                     |

The chain calls `monocular_dlt_align.align_monocular_to_world()` **unmodified**
once per detected person, using that person's own
`*_id_NN_mhr70_rec3d.csv`/`*_id_NN_markers.csv` this run just wrote (the point
rate passed is the video's own real fps, not a guess). One person's placement
failure is logged and skipped — never fatal for the rest of the run, since
the underlying SAM3D outputs are already safely on disk regardless. See
[monocular_dlt_align](monocular_dlt_align.md) for the placement math, the
scale/scale-is-not-free caveat, and how to import the result in Blender.

## Outputs

Each video gets its own subdirectory inside `processed_sapiens2_3d_<timestamp>/`.
Identical to [sam3dinov3](sam3dinov3.md#outputs) (`*_sam3dinov3_keypoints3d.csv`,
`*_sam3dinov3_camera.csv`, `*_id_NN_mhr70_*.csv`, `*_sam3dinov3_predictions.json.gz`,
optional `meshes/*.npz`), **plus**:

| File                                  | Content                                                                 |
| -------------------------------------- | ------------------------------------------------------------------------ |
| `<video>_sapiens2_3d_overlay.mp4`      | Same overlay as `sam3dinov3`'s, from the tightened bboxes                 |
| `<video>_sapiens2_3d_guidance.csv`     | **New.** `frame,person_id,guided,num_keypoints_used` — whether Sapiens2 guidance was actually used for that frame/person, and how many keypoints cleared the score threshold |
| `dlt_world/id_NN/`                     | **Only with `--dlt3d`.** That person's calibrated-lab-frame CSV/.3d/C3D/BVH (+ mesh if `--save-mesh --export-mesh` was set) — same output family `monocular_dlt_align.py` writes standalone |
| `sapiens2_3d_summary.json`             | Machine-readable run summary, incl. guided-frame counts and `dlt_world_outputs` (per person_id, when `--dlt3d` was used) |
| `README_sapiens2_3d.txt`               | Verbose per-run schema/units description                                 |
| `FAILED_sapiens2_3d.txt`               | Present only when the stage fails                                        |

`sam3dinov3.py`'s own shared writers/schema are reused **unmodified** — this
module never changes a CSV column another pipeline depends on; the guidance
flag lives only in its own companion CSV.

## Coordinate systems and units

Identical to [sam3dinov3](sam3dinov3.md#coordinate-systems-and-units) — the 3D
lifter is unchanged. Same scale caveat: monocular depth is metric only up to
the assumed camera intrinsics.

## Known limitation (MVP scope)

- Processes videos sequentially in one process — no per-video subprocess
  isolation like `sam3dinov3.py`'s batch coordinator. `_release_gpu_memory()`
  runs between videos.
- Cannot start from a completely raw video with no existing SAM3 or
  SAM3+Sapiens2 run — see Requirements.

## Related modules

- [sam3dinov3](sam3dinov3.md) — the 3D lifter this module reuses unmodified
- [sam3sapiens2](sam3sapiens2.md) — the front end this module consumes
- [monocular_dlt_align](monocular_dlt_align.md) — the DLT-chain placement math this module calls unmodified
- [vaila_sapiens](vaila_sapiens.md) — Sapiens2 2D pose on its own

## References

- SAM 3 — <https://ai.meta.com/research/sam3/>
- Sapiens2 — <https://about.meta.com/realitylabs/codecavatars/sapiens/>
- SAM 3D Body — <https://github.com/facebookresearch/sam-3d-body>
- Weights — <https://huggingface.co/facebook/sam-3d-body-dinov3>
