# YOLO26 Depth 3D

- **Where:** Frame B → **Markerless 3D** → **YOLO26 Depth 3D**
- **Module:** `vaila/yolo_depth3d.py`
- **CLI:** `uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4 [--pixels file.csv] [--dlt3d cam.dlt3d] [--segment-lengths file.csv]`

Monocular 3D from one video in *vailá*: 2D keypoints (YOLO26-pose or an existing pixel CSV) plus
the metric depth map of the YOLO26 depth task give one 3D point per keypoint.

- **No calibration:** 3D in the camera frame; the focal length is a guess unless `--focal-px`,
  `--fov-deg` or `--intrinsics-toml` (Chessboard Calib) is given.
- **With a `.dlt3d`:** each point sits on its pixel ray of the calibrated camera, so the result is
  in the lab frame, and the depths of the whole trial are fitted together: constant segment
  lengths, feet on the floor (Z = 0), trunk above the feet, smooth depth. The depth map (and the
  z of a MediaPipe `*_mp_norm.csv`, found automatically) says which end of a segment is nearer;
  the segment length says by how much.
- **`--segment-lengths`:** the lengths of the same person from a previous run or from a
  two-camera reconstruction.
- **Outputs:** `yolodepth3d_<timestamp>/` with the 3D CSV (Rec3D layout), C3D, the segment
  lengths, the depth read at every point and a summary with quality numbers.

On the two-camera test clip the error on the 13 body joints against the two-camera reconstruction
was 7.1 and 7.0 cm (two views) with the default run and 6.8 and 4.9 cm with the person's segment lengths;
SAM3+DINOv3 3D placed with the same DLT gave 11.8 and 8.0 cm.
Full description, the measured table and the limits: [`vaila/help/yolo_depth3d.md`](../../vaila/help/yolo_depth3d.md).
