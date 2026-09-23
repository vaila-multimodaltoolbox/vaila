SAM 3 video export
source_original=/home/preto/data/vaila/tests/markerless_2d_analysis/camera_01_cube_test_png_196_190_446_dbox_h265.mp4
session_resource=/home/preto/data/vaila/tests/markerless_2d_analysis/camera_01_cube_test_png_196_190_446_dbox_h265.mp4
subsampled_to_disk=False spatial_downscale=False max_input_long_edge_cap=1920 max_input_frames_cap=4096 session_frames=257
checkpoint=/home/preto/data/vaila/vaila/models/sam3/sam3.pt
prompt='person'
prompt_frame_requested=0 prompt_frame_used=0
frames_with_outputs=257 / 257

Output files reference (only files that were actually written exist in this dir):

  sam_tracks.csv            Long-format **bounding-box** table (also aliased as
                            ``sam_bbox_tracks.csv``). One row per
                            (frame, obj_id). Columns:
                              frame                  Frame index (0-based)
                              obj_id                 Persistent SAM object ID
                                                     (chunked runs: global ID
                                                     after cross-chunk
                                                     tracklet linking)
                              x_px,y_px,w_px,h_px    Bounding box in pixel
                                                     space (top-left + size)
                              score                  SAM confidence (0..1)
                              area_px                Mask area in pixels
                              n_polygons             # contours in the mask
                              largest_polygon_pts    Vertex count of the
                                                     largest contour
                              cx_px,cy_px            Mask centroid (pixels)
                            This is the file the vailá pixel tool
                            (getpixelvideo.py) consumes for the
                            ``Load Tracking CSV`` button.

  sam_bbox_tracks.csv       Discoverability alias (hardlink or copy) of
                            ``sam_tracks.csv`` — same contents, ``bbox`` in
                            the name so users can spot it quickly.

  sam_frames_meta.csv       Per-frame metadata with **normalised** bbox
                            (xc, yc, w, h in [0,1]) — useful for
                            resolution-independent downstream code.

  sam_points.csv            vailá pixel-marker format (wide). One row per
                            frame, columns frame, p0_x, p0_y, p1_x, p1_y, …
                            One column-pair per obj_id; ready for direct
                            loading in getpixelvideo or rec2d. Written by
                            default (``--postprocess-points all``).

  sam_id_map.csv            Maps SAM obj_id to the column slot (p{N}) used in
                            ``sam_points.csv``.

  sam_vaila_center.csv      Simple vailá-style ``frame,x1,y1,...,xN,yN`` —
                            one (x,y) per object using the **bbox center**
                            as anchor. Ready for rec2d / getpixelvideo.
  sam_vaila_bottom.csv      Same format — **bottom-center** (foot) anchor.
  sam_vaila_top.csv         Same format — **top-center** anchor.
  sam_vaila_left.csv        Same format — **left-center** anchor.
  sam_vaila_right.csv       Same format — **right-center** anchor.

  sam_masks_manifest.csv    Index of per-frame mask PNGs (when written).

  sam_contours.json[.gz]    Polygon vertices per object, per frame
                            (schema ``vaila_sam_contours_v1``). Suitable for
                            silhouette analysis / mesh fitting.

  <video>_sam_overlay.mp4   Coloured-mask overlay video (when written).

  masks/                    Per-frame, per-object binary mask PNGs (when
                            ``--save-mask-png`` was on). Named
                            ``frame_NNNNNN_obj_K.png``.

  FAILED_sam.txt            Only present if the run failed irrecoverably;
                            contains the reason (e.g. OOM exhaustion).
