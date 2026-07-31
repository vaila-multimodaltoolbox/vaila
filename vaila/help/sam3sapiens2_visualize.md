# SAM3+Sapiens2 — Visualize selected ID

**Version:** 0.3.86  
**Updated:** 2026-07-31

This CPU-only tool rerenders an existing `processed_sam3sapiens2_*` result. It does not load SAM3 or Sapiens2 weights, so it is safe for visualization after an inference run and does not repeat GPU allocation.

It draws the selected SAM identity, contour, bounding box, and Sapiens2 keypoints on the original video, then writes a new ID-specific directory. The root includes filtered tracking, pose, contour, audit, and overlay-video outputs; `source_artifacts/` preserves the original run for provenance.

## CLI

```bash
uv run python -u vaila/sam3sapiens2_visualize.py \
  --sam-results /path/to/processed_sam3sapiens2_.../video_stem \
  --video /path/to/video.mp4 --id 2 --output /path/to/selected_id_02
```

Use `--list-ids` to discover IDs and `--dry-run` to validate paths. `--kpt-thr` changes the keypoint confidence threshold; `--no-all-keypoints` draws only the 21 principal body points.

## GUI

Frame B → **YOLO + FB** → **SAM3+Sapiens2 Visualize ID**. Select the processed run directory, source video, output directory, and then choose an ID from the populated selection box.

Frames are zero-based and coordinates are full-frame pixels. SAM3 remains the identity authority; Sapiens2 never reassigns IDs.
