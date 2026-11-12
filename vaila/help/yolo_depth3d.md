# yolo_depth3d

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing / Markerless 3D |
| **File** | `vaila/yolo_depth3d.py` |
| **Version** | 0.4.17 |
| **Updated** | 12 November 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes — Frame B → **Markerless 3D** → **YOLO26 Depth 3D** |
| **CLI** | Yes |
| **Runtime** | GPU recommended (about 50 ms per frame on an RTX 4090); the depth weights download on first use |

## What it does

Monocular 3D from **one** video in *vailá*: 2D keypoints, the metric depth map of the
[YOLO26 depth task](https://docs.ultralytics.com/tasks/depth) and, when the camera is calibrated,
its DLT3D.

The 2D keypoints come from YOLO26-pose (17 COCO points), or from any *vailá* pixel CSV given
with `--pixels` (`frame,x1,y1,x2,y2,...`: MediaPipe, YOLO, Sapiens2, getpixelvideo).

| Level | You give | You get |
|---|---|---|
| 1. No calibration | the video | 3D in the camera frame (OpenCV axes: +X right, +Y down, +Z forward, metres). The focal length is `--focal-px`, `--fov-deg` or a guess (the image diagonal); the outputs say which |
| 2. Calibrated | `--dlt3d` (and optionally `--intrinsics-toml` from Chessboard Calib) | 3D in the lab frame: each keypoint sits on its exact pixel ray of the DLT camera |
| 3. Whole-trial fit | the same `--dlt3d`; on by default (`--fit auto`) for 17 or 33 markers | the depth of every point along its ray is solved for the whole trial at once |

## The whole-trial fit

Every point stays on its own pixel ray; only its distance along the ray is solved. One
least-squares problem covers all the frames, with these pieces of evidence:

- **Segment lengths.** Each segment keeps one length during the trial. The lengths are estimated
  from the trial itself, or given with `--segment-lengths`.
- **Order of the ends.** With the length known, a segment has two possible depths: one end nearer
  or further than the other. The depth map decides the side. A MediaPipe `*_mp_norm.csv` (its z)
  decides it better: on the test clip its **sign** was right for the arm segments in 80 to 100 % of
  the frames, while its amount was 2 to 4 times too large. Only the sign is used.
- **Floor.** A foot marker is never under its standing height above the plane Z = 0 of the lab,
  and is softly held on it (a lifted foot is allowed).
- **Balance.** The trunk stays above the feet (a lean or a step is allowed). On the test clip the
  centre of the trunk was 1 to 2 cm from the centre of the feet. `--no-balance` turns it off.
- **Smooth depth in time.**

The depth map is read in a square **crop** around the person (`--crop`, on by default): it puts
several times more pixels on the body. The full-frame map sets the level of that crop.

The file `<video>_segment_lengths.csv` holds the lengths that were used. It can be passed to
another run of the same person with `--segment-lengths`, and so can a 3D CSV of that person with
the same markers (for example a two-camera Rec3D).

## What the depth is, and is not

- The map is the depth of the **visible surface**, in metres along the optical axis. A joint
  centre is behind the skin, and an occluded keypoint gets the depth of what is in front of it.
- The **absolute scale** of the released weights is an estimate. Ultralytics reports 4 to 46 %
  off, depending on the camera. On the test clip it was 42 % and 48 % too short.
- The map is good at **where the person is** and poor at the **depth between joints**. The fit
  uses it for the first pass and for the order of the ends, not for the amount.
- Monocular 3D is an estimate, not a measurement. It does not replace a two-camera
  reconstruction.

## Measured on a two-camera clip

Clip: one person 3.5 to 4 m from two phone cameras (1080 × 1920, 30 fps, 183 frames), standing,
moving the arms, in loose trousers. 33 MediaPipe points per camera, DLT3D from an 8-point
calibration object (0.9 × 0.9 × 1.8 m). Reference: the two-camera Rec3D reconstruction of the same
points. Error = mean distance to that reference, over the 13 body joints (nose, shoulders, elbows,
wrists, hips, knees, ankles) and over all 33 points; model `yolo26x-depth.pt`.

| Configuration | Camera 1: 13 joints | all 33 | Camera 2: 13 joints | all 33 |
|---|---|---|---|---|
| `--dlt3d` (the default: whole-trial fit, MediaPipe z found, lengths from the trial) | **7.1 cm** | **7.5 cm** | **7.0 cm** | **8.1 cm** |
| the same + `--segment-lengths` from the two-camera reconstruction | **6.8 cm** | **7.0 cm** | **4.9 cm** | **5.7 cm** |
| `--dlt3d --pose-z none` (order from the depth map only) | 9.3 cm | 9.3 cm | 10.2 cm | 11.5 cm |
| the same + `--segment-lengths` | 8.6 cm | 8.3 cm | 7.9 cm | 8.8 cm |
| `--dlt3d --no-balance` | 9.5 cm | 10.7 cm | 5.9 cm | 7.1 cm |
| `--dlt3d --no-crop` | 8.3 cm | 8.9 cm | 11.1 cm | 14.9 cm |
| `--dlt3d --fit off --scale floor-frame --smooth-hz 3` | 14.2 cm | 13.0 cm | 10.7 cm | 12.6 cm |
| `--dlt3d --fit off --scale floor-frame` | 15.3 cm | 13.9 cm | 11.6 cm | 13.5 cm |
| `--dlt3d --fit off --scale floor` | 19.0 cm | 17.7 cm | 15.0 cm | 16.7 cm |
| `--dlt3d --fit off --scale control` (thin calibration poles) | 66.6 cm | 63.5 cm | 29.6 cm | 26.9 cm |
| `--dlt3d --fit off` (scale as it comes from the network) | 155.0 cm | 152.3 cm | 163.8 cm | 160.8 cm |
| No calibration (shape only, rigidly aligned to the reference) | 27.3 cm | 30.9 cm | 27.9 cm | 31.8 cm |

Two more runs on the same clip, on the 13 body joints:

| Method | Camera 1 | Camera 2 |
|---|---|---|
| This module with YOLO26-pose (17 points, no pixel CSV), `--dlt3d` | 9.0 cm | 9.6 cm |
| SAM3+DINOv3 3D, then Monocular → DLT world | 11.8 cm | 8.0 cm |

The YOLO26-pose row also carries the difference between the two detectors' 2D points (about
2 cm), because the reference is made of MediaPipe points.

What these numbers say:

- With a DLT3D the lateral and vertical position of each point is right by construction (the
  point is on its ray). The error is along the camera's line of sight.
- Against the best run without the fit, the error on the 13 joints falls from 14.2 to 7.1 cm
  (camera 1) and from 10.7 to 7.0 cm (camera 2), and the segments vary by 1 to 2 % instead of
  5 to 6 %.
- The MediaPipe z matters for the arms. Without it the map decides the side, and it misreads a
  fast, blurred arm in front of a wall.
- What is left is mostly the **depth of the whole person**. It comes from the standing foot, and
  the assumed height of the foot markers moves it: here 1 cm of height is 4 to 5 cm of depth,
  because the cameras are low (0.96 m). The trousers hide the ankles in this clip. Giving the
  person's segment lengths removes most of that error, because the size of the body then fixes
  the depth.
- `--scale control` failed here: the control points are on thin poles, and at two of the eight
  the map read the wall behind.

One clip is one clip: a standing person in a room. Check the quality numbers of your own run.

## Scale of the depth (`--scale`)

The fit starts from the floor-scaled map, so with the fit on `--scale none` means `floor-frame`.

| Mode | Needs | How |
|---|---|---|
| `none` | nothing | the network's own scale |
| `floor` | `--dlt3d` | one scale for the trial: the standing foot must be on the plane Z = 0 of the lab |
| `floor-frame` | `--dlt3d` | the same, per frame (moving median over a third of a second) |
| `control` | `--dlt3d`, `--ref3d`, `--control-video`, `--control-pixels` | compares the map with the true depth of the calibration control points. Use it only when those points are on surfaces the map resolves (not thin poles) |

Standing heights of the foot markers: MediaPipe ankle 0.12 m and heel 0.08 m; YOLO26-pose ankle
0.10 m (it sits 2 to 3 cm lower than MediaPipe's on the test clip). `--foot-height` replaces
them with one value. The summary reports how much the scale changes per centimetre of that
height (`scale_change_per_cm_of_foot_height`).

## Usage

```bash
# window
uv run --no-sync python -m vaila.yolo_depth3d

# no calibration: YOLO26-pose + depth, camera frame
uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4

# calibrated camera, MediaPipe pixel CSV: lab frame, whole-trial fit,
# checked against a second camera
uv run --no-sync python -m vaila.yolo_depth3d -i c1.mp4 --pixels c1_pixel_vaila.csv \
    --dlt3d c1.dlt3d --check-dlt3d c2.dlt3d --check-pixels c2_pixel_vaila.csv

# the same person's segment lengths from a two-camera reconstruction
uv run --no-sync python -m vaila.yolo_depth3d -i c1.mp4 --pixels c1_pixel_vaila.csv \
    --dlt3d c1.dlt3d --segment-lengths rec3d_two_cameras.csv

# the real lens from Chessboard Calib (removes the distortion of the pixels)
uv run --no-sync python -m vaila.yolo_depth3d -i video.mp4 \
    --intrinsics-toml cam_intrinsics.toml
```

Main options:

| Option | Meaning |
|---|---|
| `-i` | a video, or a folder of videos |
| `--pixels` | pixel CSV instead of YOLO26-pose (one video) |
| `--depth-model` | `yolo26{n,s,m,l,x}-depth.pt` (default `x`; use a smaller one on CPU) |
| `--dlt3d` | the camera's `.dlt3d`: result in the lab frame, whole-trial fit |
| `--fit` | `auto` (default), `on`, `off` |
| `--pose-z` | MediaPipe `*_mp_norm.csv`; `auto` (default) looks for it beside the pixel CSV, in its `<video>/` folder and up to two folders below the video; `none` turns it off |
| `--segment-lengths` | the `*_segment_lengths.csv` of a previous run, or a 3D CSV of the same person and markers |
| `--no-balance` | do not assume that the trunk stays above the feet |
| `--crop` / `--no-crop`, `--crop-margin` | depth read in a crop around the person (default on, margin 0.25) |
| `--intrinsics-toml` | Chessboard Calib `<camera>_intrinsics.toml` |
| `--focal-px`, `--fov-deg` | focal length with no calibration file |
| `--scale`, `--foot-height` | scale of the depth (see above) |
| `--mask` | read the depth only inside the person mask (`yolo26n-seg`) |
| `--check-dlt3d`, `--check-pixels` | a second camera, for an independent reprojection check |
| `--reference-3d` | a reference 3D CSV to compare with |
| `--preview`, `--pointcloud-every N` | depth video with the keypoints; `.ply` point clouds |
| `--all-people` | write every tracked person (YOLO26-pose), not only the longest track |

With `--fit off` the earlier pipeline runs: `--joint-depth` (part of the depth between joints
that is kept, default 0.25), `--smooth-hz` (zero-lag low-pass of the depth) and `--bone-fix`
(per-frame segment-length correction).

## Outputs

A new `yolodepth3d_<timestamp>/` folder beside the video (or under `-o`):

| File | Content |
|---|---|
| `<video>_yolodepth3d_world.csv` | lab frame (with `--dlt3d`), Rec3D layout `frame,p0_x,p0_y,p0_z,...` |
| `<video>_yolodepth3d_cam.csv` | camera frame, same layout |
| `<video>_yolodepth3d_m.c3d` | C3D in metres |
| `<video>_segment_lengths.csv` | the segment lengths of the fit: `marker_a, marker_b, length_m, source` |
| `<video>_depth_at_points.csv` | per frame and marker: pixel, confidence, depth read, depth used |
| `<video>_yolodepth3d_summary.json` | model, camera, scale, fit and the quality numbers |
| `configuration_used.toml` | the options of the run |
| `<video>_depth_preview.mp4`, `*_cloud_f*.ply` | optional |

A point with no keypoint (outside the image, low confidence) stays empty; nothing is filled in.

Quality numbers in the summary: variation of the segment lengths, height of the lowest foot,
fraction of points inside the calibrated volume (`--ref3d`), reprojection error in the second
camera and distance to the reference reconstruction.

## Limits

- The depth of the whole person comes from the foot on the floor. A jump, a seated person or a
  hidden foot breaks that; a wrong foot height shifts the person along the line of sight.
- The balance term assumes an upright person. It is soft, but a long lean (a sprint start) is
  outside what it was checked on.
- Without a pose z (YOLO26-pose, other pixel CSVs) the order of the ends comes from the depth
  map alone, which fails on fast, blurred or thin limbs.
- The fit needs 17 (COCO) or 33 (MediaPipe) markers and a DLT3D. Other marker sets and the
  uncalibrated mode use the depth read at each joint.
- Several people are tracked by YOLO26-pose without re-identification.
- The first use downloads the weights from the Ultralytics release (network needed).

## See also

- `monocular_dlt_align` (Markerless 3D → Monocular → DLT world): places a SAM 3D Body
  reconstruction in the lab frame.
- `rec3d` / `rec3d_one_dlt3d`: reconstruction with two or more cameras.
- `chessboard_calib`: lens and DLT3D from a chessboard.
