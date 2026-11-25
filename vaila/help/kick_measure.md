# kick_measure

> English · [Português](kick_measure_pt.md)

## Module information

| Field | Value |
|-------|--------|
| **Category** | Analysis |
| **File** | `vaila/kick_measure.py` |
| **Version** | 0.4.19 |
| **Updated** | 24 November 2026 |
| **Author** | Paulo Roberto Pereira Santiago |
| **GUI** | Yes (measuring happens in `getpixelvideo.py`, with the step panel of `kick_panel.py`; this module is the logic behind it) |
| **CLI** | Yes (`python -m vaila.kick_measure --run DIR`, `freekiki.py kick-measure DIR`, `getpixelvideo.py --kick-run DIR`, `python -m vaila.kick_measure --run DIR --export-scene OUT`, `python -m vaila.kick_measure --run DIR --write-cameras`, `freekiki.py kick-cameras DIR`) |

---

## What it is

This module opens a FreeKiki kick analysis folder and turns any pixel of its video into metres of the pitch, **in any frame**, not only in the frames that the pipeline could calibrate. With it, *vailá* measures on the video, live, and shows the result at once: distance, angle, speed, acceleration, height of an object or of the ball, the time between two events, the spin of the ball, and the length and radius of a curve. Everything is combined with the AI Track of getpixelvideo.

Pitch frame (the same as FreeKiki): `x` along the pitch, `y` across, `z` up, in metres, origin at the centre spot. One video frame is one time step of `1 / fps` seconds (frame index from 0).

## How to open a kick folder

| Entry | What to do |
|-------|------------|
| getpixelvideo, **Tpl:** button | Press **Tpl:** and answer **K** (Kick Measure), pick the `kick_analysis_*` folder in the browser and press **Enter**. If the video open now is not the video of the camera of the folder, it opens the right one. |
| getpixelvideo, **Kick SAM** button | With the video open, click **Kick SAM** and choose the `kick_analysis_*` folder. |
| getpixelvideo CLI | `uv run --no-sync vaila/getpixelvideo.py --kick-run DIR -f VIDEO` (`-f` is optional: without it the reference video of the folder is used). |
| FreeKiki window | **Frame B → Soccer Tools → FreeKiki**, section 5, **Open run and correct...**: choose the `kick_analysis_*` folder and the **Camera** (Reference or Other), then press **Measure kick** (next to *Correct ball* and *Correct field keypoints*), or **Measure kick: video + SAM folder**. |
| FreeKiki CLI | `uv run --no-sync vaila/freekiki.py kick-measure DIR [--camera reference\|other] [--no-wait]` |
| Python module | `uv run --no-sync python -m vaila.kick_measure --run DIR [--camera reference\|other] [--info] [--export cameras.json]` prints the cameras (calibrated, solved, tracked, approximate), the ball and the people; `--export` writes them to a JSON; `--write-cameras` writes the per-frame calibration files (see *Per-frame calibration files*). |

The **Load** button of getpixelvideo keeps its own job (FreeKiki run to *correct*, `Tpl: L`); Kick Measure is `Tpl: K`. Every GUI entry prints the equivalent command in the terminal (`>> uv run --no-sync vaila/getpixelvideo.py --kick-run ... -f ...`), so the session can be repeated without the window.

`DIR` can be the `kick_analysis_*` folder, its `*_3d_*` folder or the detect run inside `work/`.

### What loading does

Loading shows **only the calibration**; nothing else is drawn until you ask for it in the step panel (next section).

- The camera of **every** frame becomes one `camera3d` calibration of the Quick Measure session (see below), so the keys `1`-`6` already give real metres. When the per-frame files of the next section exist next to the SAM 3 outputs, they are read (fast); otherwise the cameras are built on the spot, tracking included.
- The pitch lines are re-projected on the video with that camera, coloured by the source of the camera of the frame: DLT3D green, solved yellow, tracked cyan, approximate orange.
- The ball track (`<video>_markers.csv` next to the video, else `ball.csv` of the 3D folder) is **not** written into the marker p0 any more: the panel does it only when you press **Ball → marker p0 (edit / AI Track)**. A p0 you already clicked is never overwritten, and AI Track keeps working.
- The mouse hover shows `Real: (x, y) m`, and `[approx. camera]` when the frame has no calibration of its own.
- In mode `8` the bottom of the window shows `s`, `z`, speed and acceleration of the current frame.

## Step panel: measure by parts and export to mkvis3d and Blender

Why: with 31 boxes, labels and the ball in p0 drawn at once, nothing could be measured. The panel (`vaila/kick_panel.py`) docks on the right of the video and goes in four steps; every layer is off until you switch it on. Texts follow the language of the `H` help (English or Portuguese).

**How to start.** Open the video in getpixelvideo, click **Kick SAM** (or FreeKiki → **Measure kick: video + SAM folder**, or `--kick-run DIR`) and choose the `kick_analysis_*` folder.

| Step | What it has |
|------|-------------|
| **1 Calib** | Frames counted by calibration source (DLT3D, solved, tracked, approximate), the source of **this** frame, and **Field lines** on/off. Check that the pitch lines sit on the grass, then go on. |
| **2 Layers** | Toggles **Boxes**, **Pose 2D**, **Ball**, **Labels**; role chips **all / kicker / keeper / team A / team B / referee / selected**; the list of people with a checkbox each (the wheel over the panel scrolls it). A click on a player in the video selects it. **Ball → marker p0 (edit / AI Track)** puts the ball track into p0, only when you press it. |
| **3 Metrics** | Choose the person among the selected ones, the metric and, when needed, the body point, then **+ Add metric**. Each added metric shows the value *now* / *max* in the panel and above the player in the video. Below, **Measure by hand** buttons start the Quick Measure modes: `1` distance, `3` angle, `4` velocity, `7` height base+top, `8` line plane, `9` event time, `0` spin, `,` arc, `.` height line. |
| **4 Export** | **mkvis3d C3D**, **Blender BVH + script**, **Metrics CSV** or **All**; **Open mkvis3d** and **Open Blender** launch the viewers without blocking the window. |

Collapse the panel with ▶ and reopen it from the tab at the right edge.

### Metrics

| Metric | What it is | Unit |
|--------|-----------|------|
| Speed on the pitch (`speed`) | speed of the feet point of the person on the grass | m/s |
| Distance to the ball (`distance_ball`) | feet point to the ball on the grass | m |
| Distance to another person (`distance_person`) | needs a second selected person | m |
| Height of a body point (`height_kpt`) | `z` of the point in the 3D pose | m |
| Jump height (`jump`) | rise of the pelvis over the standing baseline (median of the first 10 valid frames); the summary max is the jump | m |
| Speed of a body point (`kpt_speed`) | 3D speed of the point | m/s |

Body points: `pelvis` (mean of both hips), `head`, `neck`, `left_wrist`, `right_wrist`, `left_ankle`, `right_ankle`. Frames without data give no value (never an invented one); a metric that cannot be computed (for example a distance with no second person) says why.

### Export

The export goes into `<kick_analysis>/processed_kickmeasure_YYYYMMDD_HHMMSS/`. With **nobody selected** it exports every person that has a 3D pose.

| File | Content |
|------|---------|
| `kick_scene.c3d` | markers `P<id>_<kpt>` of the selected people (3D poses), `BALL`, and the pitch template as static `FKP_<name>`; open it with mkvis3d |
| `kick_scene_c3d.csv` | the same markers as a CSV |
| `blender/person_<id>.bvh` | one BVH per person (gaps are filled with the nearest valid position) |
| `blender/ball_path.json`, `blender/pitch_lines.json` | the ball path and the pitch lines for Blender |
| `blender/kick_scene_blender.py` | plain `bpy` script: imports the BVHs, a ball sphere keyframed from the JSON, the pitch lines as curves; the scene frame rate and range are set last |
| `metrics.csv`, `metrics_summary.json` | per-frame values (`frame,time_s,<metric>...`) and max / min / mean / frame of max of each metric |
| `README.txt` | the commands to open the files |

The terminal prints the equivalent command, so the export can be repeated without the window:

```bash
uv run --no-sync python -m vaila.kick_measure --run DIR --export-scene OUT \
    --people 21,23 --metric speed:21 --metric height_kpt:23:right_wrist \
    --metric distance_person:21:23 [--no-blender] [--no-mkvis3d]
```

`--metric KIND:PERSON[:OTHER_OR_POINT]` can be repeated: the third field is the other person for `distance_person` and the body point for `height_kpt` and `kpt_speed` (default `pelvis`); `jump` always uses the pelvis. `--people` takes SAM ids.

**mkvis3d** is a separate project. *vailá* finds it through the environment variable `VAILA_MKVIS3D` (path of `mkvis3d.py`) or a `mkvis3d/` folder next to the *vailá* checkout; without it the **Open mkvis3d** button says so and the export of the other files still works. By hand: `python mkvis3d.py kick_scene.c3d` opens the 3D viewer and `python mkvis3d.py info kick_scene.c3d` prints a summary.

Real example (Vargas kick, 169 frames at 30 fps): 31 SAM ids, 24 with a 3D pose; 97 pitch segments per frame; `mkvis3d info` of the export: 169 frames, 190 markers (2 x 70 body points, BALL, 49 FKP_*); kicker P21 speed max 8.7 m/s on the pitch (source `box_foot`, the SAM box foot seen by each frame's camera, 60 of 96 frames). A speed or distance series uses one position source, the one with the most valid frames (`ground_tracks`, the 3D-pose ground point, the ankles or `box_foot`), and `metrics_summary.json` records it. `box_foot` skips frames whose camera is approximate, because those cameras copy a neighbour and do not follow the pan (the foot jumped 9 m in one frame). So frames 36-43 and 47-74 of the kicker have no speed.

### Limits of the panel

- **One camera.** Heights from the pitch plane depend on the reference plane chosen; the ± of the manual modes still applies.
- **Jump from monocular 3D poses is noisy:** in the real example the pelvis moved ±0.18 m around the baseline, so read a jump as an order of magnitude, not as a measurement of a few centimetres.
- BVH fills gaps with the nearest valid position, so a person that leaves the frame stands still there.
- The referee files `referees/P<n>_referee*.c3d` of the 3D folder use the `player_id` of `player_map.csv`, not the SAM id; the export converts it.

## Keys (MEASURE on: `Q`)

| Key | Measure | Clicks | Result |
|-----|---------|--------|--------|
| `1` | Distance | 2 | metres on the grass of that frame |
| `2` | Area | 3 or more + Enter | m² |
| `3` | Angle | 3 (vertex in the middle) | degrees |
| `4` | Velocity | 2, on different frames | m/s (needs FPS) |
| `5` | Acceleration | 3, on distinct frames | m/s² (needs FPS) |
| `6` | Level | 2 | H+V guide and tape |
| `7` | **Height** | 2 in the same frame: base on the grass, then top | metres, with the plane used and a ± of 1 px |
| `8` | **Plane of the line** | 2 on the grass (kick, arrival), then 1 click per ball position | `s` (distance along the line) and `z` (height) of each click, plus 3D speed and acceleration of the sequence |
| `9` | **Event** | 2, each in its own frame (start, end) | duration in s (± 1 frame), mean speed between the two points, frame rate |
| `0` | **Spin** | each click marks the *same* mark of the ball in another frame | turns, rps and rpm (± 1 frame); the result updates from the 2nd click |
| `,` | **Arc / curve** | 3 or more along the curve | curve length and fitted arc (radius, angle, sagitta); updates on every click |
| `.` | **Height line** | 1 click at the wanted height, after the line of `8` | a horizontal guide at that height, drawn over the video in every frame |

`Enter` closes the open results (`0`, `,`); `Ctrl+Z` / `Ctrl+Y` undo and redo; **Clear** / Backspace wipe the drawing and keep the calibration. `,` and `.` are the free keys of getpixelvideo and exist on US and ABNT2 keyboards.

Spin: `k` clicks of the same mark are `k - 1` turns between the first and the last frame, so `rps = (k - 1) / ((last - first) / fps)` and `rpm = 60 rps`. At `fps / 2` turns per second or more the mark is seen less than twice per turn and the count aliases; the result carries a note when that happens.

## Calibration `camera3d`

- **One camera per frame.** A frame with a FreeKiki DLT3D keeps it untouched (source `dlt3d`). A frame without one is **solved**: rotation and zoom are fitted, per frame, to the raw pitch keypoints of `field_kps_raw.csv` (3 or more points with confidence at least 0.5, robust loss), starting from the neighbour frame (source `solved`, with the reprojection error in pixels).
- **Fixed optical centre.** Only rotation and zoom are free: the centre is the median of the **10 calibrated frames nearest in time**, not one median of the whole clip. A single global centre failed on a long lens (about 7000 px of focal length), where the centre of the DLT3D drifts metres along the viewing axis. The fixed centre is a broadcast-camera hypothesis (camera on a tripod).
- **Tracked frames.** A frame with too few pitch points to be solved used to copy the nearest good frame, while the broadcast camera pans and zooms tens of pixels per frame. Now the static scene (crowd, advertising boards, line markings) is followed from frame to frame (Shi-Tomasi corners and Lucas-Kanade, matches at distances of 1, 2, 4, 8 and 16 frames), with the SAM person boxes (grown 10 %), the ball and the broadcast overlays (scoreboard, channel logos) masked out, and the rotation and zoom of every such frame are fitted to those matches. The pitch is carried from the known cameras on **both sides** of the gap (up to 4 anchors per side). The frame is `tracked`; `rms_px` is its uncertainty in pixels (the disagreement of the anchors that carried it plus the misfit of the fixed-centre family, in quadrature) and `from_frame` is its nearest anchor.
- **Approximate frames.** A frame stays `approximate` (copy of the nearest good frame) only when the tracker fails: fewer than 60 matches touch it, the feature transfer error is above 4 px, its uncertainty is above 60 px, or no anchor is in reach. The tool says so (hover tag, a column in the results, a warning in the HUD of getpixelvideo) instead of inventing a calibration. Measures taken there are less reliable.
- A pixel becomes a point on the grass **of its own frame**: moving the camera between frames is already accounted for.

## Per-frame calibration files

Every kick run keeps the camera of **every** video frame in four files, written **always** next to the SAM 3 outputs of the camera: `<kick_analysis>/work/<stem>/sam3dinov3/processed_sam3dinov3_<time>/<stem>/` (where `<stem>_sam3dinov3_predictions.json.gz` and the `<stem>_id_NN_*.csv` files are). The pipeline writes them in the stage `cameras:<stem>`, right after `sam:<stem>` (also with `--stop-after sam`, so the video can be measured before the 3D stage); `freekiki.py kick-cameras DIR` and the CLI below write them for an existing run.

| File | Content |
|------|---------|
| `<stem>_frames.dlt3d` | CSV `frame,L1..L11`, one row per video frame, all finite (`L12 = 1`): the DLT3D of each frame, in metres of the pitch frame |
| `<stem>_frames.dlt2d` | CSV `frame,L1..L8`, one row per frame: the ground-plane (`z = 0`) homography of the same camera (`pixel = H [x, y, 1]`, `H[2,2] = 1`); it is what `QuickMeasureCalibration.from_dlt2d_sequence` reads |
| `<stem>_frames_cameras.csv` | `frame,source,rms_px,n_points,focal_px,from_frame`: where each camera comes from and how good it is |
| `<stem>_frames_cameras.json` | summary: video, number of frames, size, fps, counts per source, optical centre, who made it and when |

Sources of a camera (column `source`; the colour is the one of the pitch lines and of the bar of the step panel):

| Source | Colour | What it is | `rms_px` |
|--------|--------|------------|----------|
| `dlt3d` | green | the FreeKiki DLT3D of the frame, untouched | empty |
| `solved` | yellow | rotation and zoom fitted to the raw pitch keypoints of the frame (fixed optical centre) | reprojection error of those points |
| `tracked` | cyan | carried by the image motion from the known cameras on both sides of a gap | uncertainty of the carry (anchor disagreement plus misfit, quadrature) |
| `approximate` | orange | copy of the nearest good frame; the tracker failed | empty |

`rms_px` is therefore **not** one comparable number: for `solved` it is a fit residual on real points, for `tracked` it is an expected error of the carried camera. Read it as "about this many pixels on the pitch", and compare `tracked` frames only with each other.

CLI:

```bash
uv run --no-sync python -m vaila.kick_measure --run KICK_DIR --write-cameras [--cameras-dir DIR] [--no-track] [--camera reference|other]
uv run --no-sync python -m vaila.kick_measure --run KICK_DIR --validate-tracking 5,10,20,30
uv run --no-sync vaila/freekiki.py kick-cameras KICK_DIR [--camera reference|other]
```

- `--write-cameras` writes the four files into the SAM 3 folder and prints the paths and the counts per source; `--cameras-dir DIR` writes them there instead.
- `--no-track` does not track the approximate frames and ignores saved per-frame cameras (the old behaviour, for comparison).
- `--validate-tracking K,K,...` is the hold-out test (default `5,10,20,30`): a frame whose camera is known is pretended unknown, tracked from a good frame `k` frames away, and compared with the real camera.
- The FreeKiki window has the button **Calibration per frame** (section 5, **Open run and correct...**), which runs `kick-cameras` and mirrors the command in the terminal.

**Validation (hold-out, Fla x CAP clip, 169 frames).** At `k = 10` frames from a known camera the median error is **4.5 px** on the raw pitch keypoints (the reference itself, the real camera against the same noisy keypoints, is at about 3.7 px: the floor of the test) and **0.21 m** on the ground (9 image pixels against the real camera). Projected on the video, the pitch lines of frames 56-74 fall 1-4 px from the white lines. On this clip 42 frames became `tracked` and 2 stayed `approximate` (116-117). Limits of this validation: one clip, one broadcast camera; the error grows with the distance to the anchors, and frames 100-140 are the weakest of the clip, so measure there with the `rms_px` of the frame in mind; a camera cut is not detected (a cut needs its own run); a frame the tracker cannot follow (too few matches, see above) stays `approximate`.

**How getpixelvideo uses them.** Opening a video, `--kick-run`, the **Kick SAM** button and the **Load Tracking CSV** of a SAM 3 file install `<stem>_frames.dlt3d` as the calibration of the session when there is none yet (also `getpixelvideo.py -f VIDEO --calib-frames FILE`, or **CALIB** `L` with the `.dlt3d` / `.dlt2d`). The HUD says *Calibrated per frame* and warns on `approximate` frames; the kick panel and the pitch lines colour `tracked` cyan. See [getpixelvideo](getpixelvideo.md).

## What a height assumes

One camera sees the ball along a ray, so a height always depends on the plane that cuts that ray. The result names the plane:

- **`7` Height:** a vertical plane through the base point, perpendicular to the camera ray of the base. A pole or a person stands in it. The base must be on the grass. The ± is the largest change of the height when either click moves 1 px.
- **`8` Plane of the line:** the vertical plane through the line you marked on the grass. The ball is **taken in that plane**, but one camera cannot tell how far the ball is across the line. The ± of each point is the change of `z` if the plane were shifted 2 m sideways. With a typical broadcast camera that is about 0.1 m of height per metre of lateral error (2 m of lateral error changed the height by at most 0.37 m in the Vargas x Rossi clip).
- The height of a ball that is far from the plane you chose is therefore a **model** result: choose the line that follows the kick.

## Limits

- **One camera.** No triangulation: the plane chosen decides the height.
- **No camera cuts.** The module does not detect a cut of the broadcast camera; a cut moves the optical centre, so a cut needs its own run (a segment per run).
- **Frames with too few pitch points** are tracked from the image motion (accuracy in *Per-frame calibration files*); only the frames the tracker cannot follow stay approximate (copy of the nearest frame).
- Validated on a single real clip (Fla x CAP, the Vargas x Rossi kick, broadcast camera); other cameras and matches are not verified.
- The spin counts what you click; it does not detect the mark by itself.
- The `dlt3d` calibration of Quick Measure (11 parameters from one video) remains reserved; `camera3d` is the one in use here.

## Files written

Loading and measuring write nothing into the `kick_analysis_*` folder; only the **Export** step writes there (`processed_kickmeasure_<timestamp>/`, see above). **S** (save) in getpixelvideo writes next to the video:

| File | Content |
|------|---------|
| `<video>_markers.csv` | the markers of getpixelvideo, including the p0 ball track you corrected |
| `processed_quickmeasure_<timestamp>/` | one folder per session, refreshed by each save |
| `<stem>_quickmeasure_points.csv` | every click: frame, pixel and real coordinates |
| `<stem>_quickmeasure_calibration.csv` | the calibration (`camera3d`, source folder) |
| `<stem>_quickmeasure_camera3d.csv` | one row per video frame: `source` (`dlt3d` / `solved` / `tracked` / `approximate`), `approximate`, `rms_px`, `n_points`, and the 3x4 matrix `P11`...`P34` |
| `<stem>_quickmeasure_results.csv` and `..._results_<type>.csv` | one result per row; types `height`, `line_plane`, `plane_point`, `height_line`, `event`, `spin`, `arc` (plus `distance`, `angle`...), with `uncertainty`, `camera_source`, `approximate`, `plane`, `s`, `z`, `speed`, `accel`, `turns`, `rps`, `rpm`, `radius`, `angle_deg`, `sagitta` |
| `<stem>_quickmeasure_lineplane_series.csv` | mode `8`: x, y, z, s, speed, acceleration (and the components) per frame, from the first to the last ball click (needs FPS and 2 clicks on different frames) |
| `quickmeasure_report.html` | didactic report: every measure, its plane and its ± |
| `README_quickmeasure.txt` | list of the files and the command to recompute |

The four per-frame calibration files are described in *Per-frame calibration files* (they live next to the SAM 3 outputs, not here). The `--export` of the module CLI writes a JSON with the summary, the camera of each frame, the ball pixels and the number of people per frame.

## Python API (frozen in the docstrings of `vaila/kick_measure.py`)

| Name | Role |
|------|------|
| `load_kick_measure_run(path, camera="reference")` | opens the folder: `KickMeasureRun` (video, fps, cameras, ball track, people) |
| `load_people_info`, `load_pose_2d`, `pose_connections`, `field_segments_px` | people (role, team, frames), 2D poses, bones and the pitch lines projected on a frame, for the panel layers |
| `metric_series(run, kind, person, ...)` | `MetricSeries` of a metric (`METRIC_KINDS`, `BODY_POINTS`) with `at(frame)` and `summary()` |
| `export_scene(run, out_dir, people, ...)`, `find_mkvis3d`, `mkvis3d_argv` | the export of the table above |
| `write_frame_calibration`, `read_frame_calibration`, `find_frame_calibration` | write, read back and find the per-frame files (`<stem>_frames.*`) |
| `track_approximate_frames`, `track_features`, `validate_tracking` | frame-to-frame tracking of the approximate frames and its hold-out test |
| `ground_homography(P)`, `CameraTrack.homography(frame)` | the `z = 0` homography of a camera (the `.dlt2d` row) |
| `CameraTrack` | one 3x4 matrix per frame; `from_calibration`, `matrix`, `ground_point`, `height_base_top`, `counts` |
| `ground_point(P, u, v, z=0)` | pixel to the grass (or to a horizontal plane at height `z`) |
| `height_base_top(P, base_uv, top_uv)` | height of an object by the vertical plane through its base |
| `LinePlane` | vertical plane of a line on the grass: `locate`, `lateral_sensitivity`, `height_line` |
| `event_time`, `mean_speed`, `spin_rate` | time, mean speed, spin (± 1 frame) |
| `path_length`, `fit_arc`, `fit_parabola` | curve length and fits |
| `velocity_series`, `acceleration_series`, `speed_series` | kinematics of a track, optional smoothing |

## Related modules

| Module | Role |
|--------|------|
| [getpixelvideo](getpixelvideo.md) | host: loads the folder (`--kick-run`, `Tpl: K`, **Kick SAM**), docks the step panel, takes the clicks |
| [quickmeasure](quickmeasure.md) | measure modes and the session files (`camera3d` calibration) |
| [freekiki](freekiki.md) | produces the `kick_analysis_*` folder; `kick-measure`, `kick-cameras` and the **Measure kick** and **Calibration per frame** buttons |
| [soccerfield](soccerfield.md) | the 49-point pitch template that the cameras are fitted to |

---

Part of *vailá* - Multimodal Toolbox · [Português](kick_measure_pt.md) · [GitHub](https://github.com/vaila-multimodaltoolbox/vaila)
