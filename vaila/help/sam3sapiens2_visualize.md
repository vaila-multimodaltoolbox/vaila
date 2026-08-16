# SAM3+Sapiens2 — Visualize selected ID

**Version:** 0.3.106  
**Updated:** 2026-08-16

This CPU-only tool rerenders an existing `processed_sam3sapiens2_*` result. It does not load SAM3 or Sapiens2 weights, so it is safe for visualization after an inference run and does not repeat GPU allocation.

It draws the selected SAM identity, contour, bounding box, and Sapiens2 keypoints on the original video, then writes a new ID-specific directory. The root includes filtered tracking, pose, contour, audit, and overlay-video outputs; `source_artifacts/` preserves the original run for provenance.

## Overlay style (v0.3.96)

The selected-ID overlay matches the combined SAM3+Sapiens2 look:

1. **SAM3 contour** — semi-transparent silhouette fill (anti-aliased edge, alpha 0.45) + outline, bbox, and `SAM #id` label
2. **Sapiens2 skeleton** — official `keypoints308` palette with **left = green** and **right = orange** limb colors (full 308 links when the `sapiens` extra is installed; COCO-21 left/right fallback otherwise)

> Note: this pipeline has no body mesh — Sapiens2 regresses joints only, not a surface. For a Blender-importable **3D body mesh**, use [sam3dinov3_visualize](sam3dinov3_visualize.md) `--export-mesh obj` on a SAM3+DINOv3 run made with `--save-mesh`.

## CLI

```bash
uv run python -u vaila/sam3sapiens2_visualize.py \
  --sam-results /path/to/processed_sam3sapiens2_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02
```

`--id` selects the SAM/stable identity. If you omit `--id` while supplying `--sam-results`, `--video`, and `--output`, the CLI prints the available IDs and **prompts interactively** until a valid ID is entered:

```bash
uv run python -u vaila/sam3sapiens2_visualize.py \
  --sam-results /path/to/processed_sam3sapiens2_.../video_stem \
  --video /path/to/video.mp4 --output /path/to/selected_id_out
# >> Available SAM IDs: [0, 1, 2, 4]
# >> Enter SAM/stable ID to visualize: 4
```

Use `--list-ids` to discover IDs and exit without rendering. `--dry-run` validates paths. `--overwrite` allows a non-empty output directory. `--kpt-thr` changes the keypoint confidence threshold; `--no-all-keypoints` draws only the 21 principal body points.

## GUI

Frame B → **Markerless 2D** → **SAM3+Sapiens2 Visualize ID** (or run with no arguments). Select the processed run directory first. The matching source video is filled automatically from `sam3sapiens2_summary.json`; if the run was moved, choose the synchronized/cropped source video manually. Select an **output parent** and an ID from the combobox. The GUI creates a new ID-specific child directory, so an existing non-empty parent is safe and repeated runs receive a numeric suffix.

Before rendering, the tool verifies the source frame count and image dimensions against the predictions. This prevents accidentally applying a 631-frame synchronized result to its 2,809-frame original video.

Frames are zero-based and coordinates are full-frame pixels. SAM3 remains the identity authority; Sapiens2 never reassigns IDs.
