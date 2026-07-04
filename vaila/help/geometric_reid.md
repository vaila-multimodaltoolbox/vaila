# geometric_reid

## Module Information

- **Category:** Processing / Re-ID
- **File:** `vaila/geometric_reid.py`
- **Version:** 0.3.68
- **Updated:** 04 July 2026
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

### Used by

- `vaila/yolov26track.py` — `_GeometricTrackLinker = GeometricFrameLinker`
- `vaila/vaila_sam.py` — `_stabilize_sam_track_ids`, `_build_cross_chunk_id_maps`
- `vaila/reid_markers.py` — `geometric_reid_align_markers` (imports `assignment_min_cost`)

---

📅 **Last Updated:** 04 July 2026 (v0.3.68)
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
