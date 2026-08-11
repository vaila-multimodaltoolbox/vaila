# geometric_reid

## Module Information

- **Category:** Processing / Re-ID
- **File:** `vaila/geometric_reid.py`
- **Version:** 0.3.104
- **Updated:** 11 August 2026
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** No (library module)
- **CLI Interface:** No (consumed by yolov26track, vaila_sam, reid_markers)

## Description

Shared geometric Re-ID helpers used by YOLO tracking, SAM chunk linking, and
marker correction to keep ID-stabilization logic consistent.

### Core components

| Function / Class | Purpose |
|------------------|---------|
| `assignment_min_cost(cost_matrix)` | Hungarian 1:1 assignment (SciPy + greedy fallback) |
| `bbox_iou_xyxy(a, b)` | IoU for `(x1, y1, x2, y2)` boxes |
| `bbox_iou_xywh(a, b)` | IoU for `(x, y, w, h)` boxes |
| `centroid_xyxy(bbox)` | Centroid from xyxy |
| `mask_iou_u8(a, b)` | Binary mask IoU (uint8 arrays) |
| `apply_homography_to_xy(points, H)` | Map Nx2 points through 3×3 homography |
| `pairwise_link_cost(...)` | Full cost computation (distance + IoU + velocity + mask) |
| `GeometricLinkerConfig` | Dataclass of all tunable parameters |
| `GeometricFrameLinker` | Stateful per-frame Hungarian linker with velocity EMA |
| `write_reid_links_csv(path, links, header)` | Audit CSV writer |

### Parameters (GeometricLinkerConfig)

| Parameter | Default | Description |
|-----------|---------|-------------|
| `max_gap` | 12 | Max frame gap before track expires |
| `max_centroid_dist_px` | 180.0 | Max centroid distance (px) |
| `min_iou` | 0.05 | Min IoU gate |
| `direction_weight` | 0.0 | Velocity-direction penalty (0=off) |
| `homography_matrix` | None | 3×3 for pitch-plane distances |
| `mask_iou_weight` | 0.0 | Binary mask IoU cost weight |
| `max_tracks` | None | **New (v0.3.102).** Bounded identity-slot pool: caps the number of *concurrently* live `stable_id` values. `None` = unlimited (unchanged default; every pre-existing call site is unaffected). A track idle longer than `max_gap` frees its numeric slot for reuse by a later, different physical subject. When the pool is genuinely exhausted (more simultaneous detections than `max_tracks` in one frame), the least-recently-updated slot is stolen rather than a detection being silently dropped — logged to `GeometricFrameLinker.forced_reassignments` as `(frame, raw_id, stolen_stable_id)` tuples. |

### `max_tracks`: a real pre-existing bug it also fixed

Before this field existed, `GeometricFrameLinker.assign_frame`'s
expired-track cleanup loop had a condition that could never actually be
true (it compared an already-gap-filtered set against the same gap
threshold), so `self.active` entries were silently kept forever instead of
being purged — harmless for the unbounded-id case (just an unused memory
leak, entries were still excluded from matching via the `active_ids`
filter each frame), but it would have defeated slot recycling entirely
once `max_tracks` needed a slot to actually free up. Fixed alongside the
new field: expired entries are now purged eagerly at the top of
`assign_frame`, before matching or spawning. Caught a real symptom on
real data: `yolov26track --max-ids 16 --stabilize-ids` (unrelated flags,
same underlying linker) produced **17** stable ids from a 16-id-capped
buffer on a real 16,693-frame video — the live linker's own unbounded
`next_stable_id` counter growing past the cap when a track's geometric
continuity broke. `reid_markers`' new `max_ids`-bounded merge (which uses
this same `max_tracks` field) correctly re-consolidated that exact
17-id output back down to 16 with zero dropped rows.

### Used by

- `vaila/yolov26track.py` — `_GeometricTrackLinker = GeometricFrameLinker`
- `vaila/vaila_sam.py` — `_stabilize_sam_track_ids`, `_build_cross_chunk_id_maps`
- `vaila/reid_markers.py` — `geometric_reid_align_markers` (imports `assignment_min_cost`); the new `merge_fragmented_ids_geometric` (Geometric ReID v2) uses `GeometricFrameLinker`/`GeometricLinkerConfig` directly, including the new `max_tracks` slot pool

---

📅 **Last Updated:** 11 August 2026 (v0.3.104)
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
