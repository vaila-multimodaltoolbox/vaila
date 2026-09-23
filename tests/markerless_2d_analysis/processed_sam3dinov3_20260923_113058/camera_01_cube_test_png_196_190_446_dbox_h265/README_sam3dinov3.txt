vailá SAM3 + DINOv3 (SAM 3D Body) markerless 3D run
video=/home/preto/data/vaila/tests/markerless_2d_analysis/camera_01_cube_test_png_196_190_446_dbox_h265.mp4
sam_results=/home/preto/data/vaila/tests/markerless_2d_analysis/processed_sam3dinov3_20260923_113058/camera_01_cube_test_png_196_190_446_dbox_h265/sam3
sam3d_weights=/home/preto/data/vaila/vaila/models/sam-3d-dinov3
inference_type=body
stride=1
mask_conditioned=True
bbox_padding_fraction=0.12
contour_margin_px=8
focal_px=auto (default FOV)
n_keypoints=70
identity_authority=SAM3 obj_id
person_detector=SAM3 (ViTDet disabled)
segmentor=SAM3 contours (SAM2 disabled)

Pipeline
--------
1. SAM 3 segments/tracks people and defines bbox, silhouette, score and obj_id.
2. SAM 3D Body (DINOv3 ViT-H/16+ backbone) receives only those SAM boxes and
   silhouettes; the upstream ViTDet detector and SAM2 segmentor never load.
3. For each person it regresses an MHR (Momentum Human Rig) mesh, 3D joints in
   metres, their 2D reprojection in pixels, the camera translation and focal.
4. person_id is exactly the SAM obj_id, so no second Re-ID can swap identities.

Coordinate systems and units
----------------------------
x_m,y_m,z_m         Root-relative 3D joints, metres, camera axes (OpenCV: +x
                    right, +y down, +z forward/away from the camera).
xcam_m,ycam_m,zcam_m Camera-frame absolute joints = root-relative + cam_t.
                    Use these for inter-person distances and depth.
x_px,y_px           Perspective reprojection into the original full frame.
frames              Zero-based, matching the source video and the SAM outputs.

Scale caveat: monocular depth is metric only up to the assumed camera intrinsics.
Without --focal-px the model falls back to a default FOV (f = sqrt(W^2+H^2)),
so absolute depth carries that assumption. Supply the true focal length in
pixels (or a FOV estimator) whenever absolute distances matter.

Main outputs
------------
<video>_sam3dinov3_overlay.mp4          SAM contour/bbox/ID + reprojected 3D skeleton,
                                         colored by joint side (left=green, right=orange,
                                         center/spine=blue) — same palette as
                                         sam3sapiens2_visualize / sam3dinov3_visualize.
<video>_sam3dinov3_keypoints3d.csv      Long table, root-relative and camera-frame metres.
<video>_sam3dinov3_keypoints2d.csv      Long table, reprojected pixels.
<video>_sam3dinov3_camera.csv           Per-frame focal length, cam_t and bbox.
<video>_sam3dinov3_joint_angles.csv     Long table, local (parent-relative) joint angles
                                         for the model's own 127-joint MHR rig: Euler XYZ
                                         degrees + scalar-first (w,x,y,z) quaternion, from
                                         the model's own regressed rotations (not a
                                         position-only heuristic) -- see joint_kinematics.py.
<video>_id_NN_mhr70_3d.csv              Wide, named columns (nose_x, nose_y, nose_z, ...).
<video>_id_NN_mhr70_rec3d.csv           Wide, vailá rec3d convention (p0_x,p0_y,p0_z, ...).
<video>_id_NN_markers.csv               Wide 2D for REC2D / getpixelvideo.
<video>_sam3dinov3_predictions.json.gz  Full provenance and per-instance predictions.
meshes/frame_NNNNNN.npz                 Only with --save-mesh (vertices + obj_ids).
mesh_faces.npy                          Only with --save-mesh (shared MHR topology).
sam3dinov3_summary.json                 Machine-readable run summary.
README_sam3dinov3.txt                   This file.
FAILED_sam3dinov3.txt                   Exists only when this stage fails.

The original SAM artifacts remain under the path recorded above (or in ./sam3
when SAM 3 was run by this pipeline).

References
----------
SAM 3        https://ai.meta.com/research/sam3/
DINOv3       https://ai.meta.com/research/dinov3/
SAM 3D Body  https://github.com/facebookresearch/sam-3d-body
Weights      https://huggingface.co/facebook/sam-3d-body-dinov3
