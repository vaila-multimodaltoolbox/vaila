# SAM3+DINOv3 3D — Visualize selected ID

**Version:** 0.3.98
**Updated:** 2026-08-03

This CPU-only tool rerenders an existing `processed_sam3dinov3_*` (SAM3+DINOv3 3D / SAM 3D Body) result. It does not load SAM3 or SAM 3D Body weights, so it is safe to run right after a GPU inference run to isolate one person, and does not repeat GPU allocation.

It draws the selected person's SAM contour, bounding box, ID, and reprojected MHR70 skeleton on the original video, then writes a new ID-specific directory. The root includes filtered keypoint/camera/contour/predictions/overlay outputs (and filtered per-frame mesh, when the source run has `meshes/`); `source_artifacts/` preserves the original run for provenance.

## Overlay style (v0.3.96)

The selected-ID overlay matches the SAM3+DINOv3 look:

1. **SAM3 contour** — semi-transparent silhouette fill (anti-aliased edge, alpha 0.45) + outline, bbox
2. **MHR70 skeleton** — 2D reprojection (`keypoints_2d_px`) drawn with `sam3dinov3.skeleton_edges`, colored by joint side using the `left-`/`right-` name prefix: **left = green**, **right = orange**, **center/spine = blue**. An edge only takes a side color when both its endpoints share that side; mixed edges (e.g. neck→shoulder) fall back to the center color.
3. **`ID nn  z=… m`** label — depth taken from `cam_t_m[2]`

Identity is never reassigned: `person_id == sam_obj_id`, exactly as in the source run.

## Mesh export for Blender (v0.3.96)

When the *source* `sam3dinov3.py` run was executed with `--save-mesh`, this tool can also export the selected person's per-frame body mesh (MHR vertices + shared faces, with `cam_t` translation applied so the body keeps moving instead of resetting to the origin every frame) as a sequence Blender can play back as an animation:

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02 \
  --export-mesh obj   # or: --export-mesh ply (smaller, binary)
```

This writes `meshes_obj/frame_NNNNNN.obj` (or `meshes_ply/frame_NNNNNN.ply`) — one complete mesh file per frame, sharing the MHR topology recorded in `mesh_faces.npy`.

**To view it in Blender:**

1. `Edit > Preferences > Add-ons`, enable the built-in **"Mesh: Stop Motion OBJ"** add-on (ships with Blender, no download needed).
2. `File > Import > Mesh Sequence`, then point it at the `meshes_obj/` (or `meshes_ply/`) folder.
3. Blender creates one object that swaps its mesh data every frame — press play to see the body move.

> **Note:** `sam3sapiens2_visualize.py` (SAM3+Sapiens2, 2D/3D **keypoints only**) has no surface mesh — Sapiens2 does not regress a body mesh, only joints — so mesh export only applies to the SAM3+DINOv3 3D pipeline. If you only need a skeleton in Blender, drive an Empty/Armature per joint from the `*_id_NN_mhr70_rec3d.csv` (or the Sapiens2 equivalent) instead.

If the source run has no `meshes/` (the common case — mesh export is off by default in `sam3dinov3.py` because it is large, ~220 KB/person/frame), `--export-mesh` logs a warning and skips silently; rerun `sam3dinov3.py` with `--save-mesh` first.

## CLI

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02
```

`--id` selects the SAM/person identity. If you omit `--id` while supplying `--sam3d-results`, `--video`, and `--output`, the CLI prints the available IDs and **prompts interactively** until a valid ID is entered:

```bash
uv run python -u vaila/sam3dinov3_visualize.py \
  --sam3d-results /path/to/processed_sam3dinov3_.../video_stem \
  --video /path/to/video.mp4 --output /path/to/selected_id_out
# >> Available person IDs: [0, 1, 2, 4]
# >> Enter SAM/person ID to visualize: 4
```

Use `--list-ids` to discover IDs and exit without rendering. `--dry-run` validates paths and the video's frame count/dimensions against the run. `--overwrite` allows a non-empty output directory. `--export-mesh {obj,ply}` writes the Blender mesh sequence described above (default `none`).

## GUI

Frame B → **Markerless 3D** → **SAM3+DINOv3 Visualize ID** (or run with no arguments). Select the processed run directory first. The matching source video is filled automatically from `sam3dinov3_summary.json`; if the run was moved, choose the synchronized/cropped source video manually. Select an **output parent** and an ID from the combobox. Check **"Export mesh sequence (.obj, for Blender)"** to also write the OBJ sequence described above (needs `--save-mesh` in the source run). The GUI creates a new ID-specific child directory, so an existing non-empty parent is safe and repeated runs receive a numeric suffix.

Before rendering, the CLI `--dry-run` path verifies the source frame count and image dimensions against `width`/`height`/`n_frames` recorded in the gzipped predictions — this prevents accidentally applying a synchronized/cropped result to the wrong source video.

## Outputs

| File | Content |
| --- | --- |
| `<video>_sam3dinov3_id_NN_overlay.mp4` | SAM contour/bbox/ID + reprojected MHR70 skeleton, one person only |
| `<video>_sam3dinov3_keypoints3d.csv` | Long table filtered to `person_id == NN` |
| `<video>_sam3dinov3_keypoints2d.csv` | Long table filtered to `person_id == NN` |
| `<video>_sam3dinov3_camera.csv` | Per-frame focal/`cam_t`/bbox filtered to `person_id == NN` |
| `<video>_id_NN_mhr70_3d.csv` | Copied as-is (already per-ID in the source run) |
| `<video>_id_NN_mhr70_rec3d.csv` | Copied as-is (vailá `rec3d` convention) |
| `<video>_id_NN_markers.csv` | Copied as-is (wide 2D for REC2D / `getpixelvideo.py`) |
| `<video>_sam3dinov3_predictions.json.gz` | Filtered provenance, one person only |
| `sam_tracks.csv`, `sam_bbox_tracks.csv`, `sam_contours.json` | Filtered to `obj_id == NN` |
| `meshes/frame_NNNNNN.npz` | Only when the source run has `meshes/`: single-person `vertices`/`cam_t` slice |
| `mesh_faces.npy` | Copied when meshes were filtered |
| `meshes_obj/frame_NNNNNN.obj` or `meshes_ply/frame_NNNNNN.ply` | Only with `--export-mesh`: Blender-importable mesh sequence |
| `sam3dinov3_selected_id_manifest.json` | Machine-readable summary of this selection |
| `README_sam3dinov3_selected_id.txt` | Human-readable provenance note |
| `source_artifacts/` | Full original run, preserved for provenance |

`NN` is the SAM `obj_id`, zero-padded — the same identity used throughout `sam3dinov3.py`.

Frames are zero-based; 2D coordinates are full-frame pixels; 3D coordinates are metres. SAM3 remains the identity authority; SAM 3D Body never reassigns IDs.

## Related modules

- [sam3dinov3](sam3dinov3.md) — the GPU inference run this tool rerenders
- [sam3sapiens2_visualize](sam3sapiens2_visualize.md) — the 2D-pose counterpart this tool mirrors
