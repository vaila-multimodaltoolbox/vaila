vailá SAM3+Sapiens2 run
video=/home/preto/data/vaila/tests/markerless_2d_analysis/camera_01_cube_test_png_196_190_446_dbox_h265.mp4
sam_results=/home/preto/data/vaila/tests/markerless_2d_analysis/processed_sam3sapiens2_20260923_114930/camera_01_cube_test_png_196_190_446_dbox_h265/sam3
model=1b
stride=1
identity_authority=SAM3 obj_id
person_detector=SAM3 (DETR disabled)
contour_focus=True
bbox_padding_fraction=0.12
contour_margin_px=8

Pipeline
--------
1. SAM3 segments/tracks people and defines bbox, silhouette, score and obj_id.
2. Sapiens2 receives only those SAM boxes; DETR is never loaded.
3. Each Sapiens crop is focused with the corresponding SAM silhouette.
4. Keypoints outside the dilated silhouette have confidence attenuated.
5. stable_id/person_id is exactly SAM obj_id; no second Re-ID can swap it.

Main outputs
------------
<video>_sam3sapiens2_overlay.mp4  SAM contour+bbox+ID and Sapiens2 skeleton.
<video>_sam3sapiens2_predictions.json  Full provenance and 308-keypoint instances.
<video>_sam3sapiens2_vaila.csv  Long frame/person/keypoint table.
sam3sapiens2_id_audit.csv  Per-frame proof that sam_obj_id == stable_id.
<video>_markers.csv (frame,p0_x,p0_y,...) and sapiens_vaila_*.csv  REC2D/REC3D/getpixelvideo outputs.
sapiens_points.csv, sapiens_id_map.csv, sapiens_bbox_tracks.csv  Stable SAM-ID tables.
<video>_id_NN_sapiens_pose.csv  Wide 308-keypoint file per SAM identity.
README_sam3sapiens2.txt  This file.
FAILED_sam3sapiens2.txt  Exists only when this combined stage fails.

The original SAM artifacts remain under the path recorded above (or in ./sam3
when SAM3 was run by this pipeline). Coordinate units are full-frame pixels and
frames are zero-based.
