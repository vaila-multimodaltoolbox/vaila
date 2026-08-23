# Skeleton templates for testing and reconstruction pipelines

This directory contains standardized skeleton template JSON definitions used across
vailá tests, markerless 2D/3D tracking pipelines, and DLT reconstructions:

- vaila/rec3d_one_dlt3d.py & vaila/rec3d.py (monocular/multiview 3D reconstruction and Blender companion visualization)
- vaila/sam3dinov3.py & vaila/sam3dinov3_visualize.py (SAM 3D Body / MHR-70)
- vaila/sam3sapiens2.py & vaila/sam3sapiens2_visualize.py & vaila/vaila_sapiens.py (Sapiens2 Goliath-308)
- vaila/fifa_skeletal_pipeline.py & vaila/fifa_to_dlt.py (FIFA 15 Keypoints)
- vaila/blender_viz.py (auto-infer skeleton preset by marker count)
- vaila/tugturn.py (Timed Up and Go turn analysis)

## Presets Table

| File | Schema | Marker Count | Edges | Tracker / Origin |
|------|--------|--------------|-------|------------------|
| fifa_body15.json | fifa_body_15 | 15 | 16 | FIFA Skeletal Tracking Light 2026 |
| yolo_coco17.json | coco_17_keypoints | 17 | 16 | YOLOv8/v11/v26 Pose / COCO-17 |
| mediapipe_hand21.json | mediapipe_hand_21 | 21 | 21 | MediaPipe Hand (Single Hand) |
| openpose_body25.json | openpose_body_25 | 25 | 24 | OpenPose Body-25 / SAM3D body25 |
| halpe26.json | halpe_26 | 26 | 29 | Halpe 26 Body+Feet (AlphaPose / YOLO-Pose) |
| soccerfield_calib29.json | soccerfield_calib_29 | 29 | 23 | Soccer Field 29 FIFA Calib Ref |
| soccerfield_pitch32.json | soccerfield_pitch_32 | 32 | 37 | Soccer Field 32 Canonical Pitch Keypoints |
| mediapipe_pose33.json | mediapipe_pose_33_pn | 33 | 32 | MediaPipe BlazePose (33 keypoints) |
| mediapipe_hands42.json | mediapipe_hands_42 | 42 | 42 | MediaPipe Both Hands (Left 21 + Right 21) |
| sam3dinov3_mhr70.json | sam3dinov3_mhr70 | 70 | 30 | SAM3+DINOv3 (SAM 3D Body MHR-70) |
| mediapipe_holistic75.json | mediapipe_holistic_75 | 75 | 76 | MediaPipe Holistic (Body 33 + Hands 42) |
| coco_wholebody133.json | coco_wholebody_133 | 133 | 71 | COCO WholeBody / Sapiens-133 |
| sapiens2_goliath308.json | sapiens2_goliath_308 | 308 | 71 | Sapiens2 Sociopticon / Goliath (308 points) |

## Compatibility Fixtures & Aliases

For seamless backward compatibility with existing tests and scripts:
- skeleton_pose_mediapipe.json -> identical to mediapipe_pose33.json
- skeleton_pose_yolo.json -> identical to yolo_coco17.json
- skeleton_pose_sam3dinov3.json -> identical to sam3dinov3_mhr70.json
- skeleton_pose_sapiens2.json -> identical to sapiens2_goliath308.json
- skeleton_pose_fifa.json -> identical to fifa_body15.json
- skeleton_pose_openpose25.json -> identical to openpose_body25.json
- skeleton_pose_hand21.json -> identical to mediapipe_hand21.json

## Convention

Every connection pair ["pA", "pB"] references the **1-based marker index** p1..pN in the wide reconstruction CSV (frame, p1_x, p1_y, p1_z, ...).
