# chessboard_calib

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing |
| **File** | `vaila/chessboard_calib.py` |
| **Button** | Frame C-A → **Chessboard Calib** (below DLT/REC 2D-3D) |
| **Version** | 0.4.15 |
| **Updated** | 08 November 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes |
| **CLI** | Yes |

---

## What it does

Calibrates several cameras from chessboard videos (one video per camera, same session) and exports what *vailá* needs for 3D reconstruction:

| Output | Used by |
|--------|---------|
| `<camera>.dlt3d` | **Rec3D 1DLT / MultiDLT** (`rec3d_one_dlt3d.py`, `rec3d.py`) |
| `<camera>_intrinsics.toml` (`fx, fy, cx, cy, k1, k2, k3, p1, p2`) | **Distort Video** (`vaila_lensdistortvideo.py`); `--intrinsics-toml-dir` of a later run |
| `<camera>_board_pixels.csv` | detected board corners, *vailá* pixel format (`frame, p1_x, p1_y, …`) |
| `cameras.json` | K, distortion, R, t, camera centre and frame offset per camera |
| `calibration_report.json`, `README_calibration.txt` | errors, offsets, world frame, validation |
| `validation_board_3d.csv` | board corners triangulated with the exported DLTs |
| `qa/<camera>_coverage.png` | corners used; the calibration is measured inside the red outline |

Results go to `<output>/processed_chessboard_calib_YYYYMMDD_HHMMSS/`.

## Pipeline

1. **Corners.** Every `--step`-th frame is searched. While the board was seen recently, only an upscaled crop around it is searched (`--zoom`, default 2), which finds small boards and is fast; otherwise the full frame is searched every `--search-every` frames. Corners are refined to sub-pixel accuracy and numbered from the top-left corner of the upright board, rows left to right, so a physical corner has the same number in every camera.

   **Frame numbers are video timestamps:** frame = round(t × fps), first frame 0, the numbering of getpixelvideo and of a constant-frame-rate copy. Network cameras sometimes drop frames (gaps in the timestamps), and a frame after a gap may also fail to decode. Those numbers are missing, and the log, `calibration_report.json` (`frame_numbering`) and the README list them per camera. Counting frames while reading would shift every later frame by the number of drops, by a different amount in each camera. When an MP4 reports a lower average rate only because of the drops, the nominal rate (`r_frame_rate`) is used.
2. **Intrinsics** per camera (`cv2.calibrateCameraExtended`). The lens model depends on how much of the image the corners cover (`--intrinsics auto`):

   | Coverage of the image | Model |
   |---|---|
   | ≥ 30 % | `standard`: f, principal point, k1, k2, p1, p2 |
   | 5–30 % | `constrained`: f, k1, k2 (principal point at the centre, no tangential terms) |
   | < 5 % | `minimal`: f, k1 |

   Views whose error is far above the median are dropped once and the camera is refitted.

   **Board tilt.** The report gives, per camera, how much the board orientation varied (`board_tilt_p95_deg`, p95 of the angle between each view's board normal and the mean normal). Below 20° the board kept almost one orientation. Then the focal length cannot be told apart from the board distance, and the log and README print a WARNING. The validation does not see this problem, because it measures the board itself. Points away from the board are off by centimetres.

   **Loading the lens.** To skip this step, give `--intrinsics-toml-dir` with a folder of `<camera>_intrinsics.toml` files, for example the `processed_chessboard_calib_*` folder of a calibration of the same cameras where the board moved. Each video takes the TOML whose camera name starts or is contained in its name. A cut copy (`<camera>_sync_188_frames_11_to_191`) or an equalized copy (`source_2_<camera>_..._equalized`) therefore uses the lens of the original camera. The log says which file each camera used and warns when none matches. You can also record separate close-up videos and pass `--intrinsics-input`.
3. **Frame offsets** (`--sync auto`, needs `--step 1`). The board's rotation between frames has the same angle in every camera, so the lag that maximizes its correlation between two cameras is their offset. Each lag is refined to a fraction of a frame (parabola through the correlation peak). Pairs with correlation below 0.8 are ignored, the offsets are solved together and rounded to whole frames. The offsets before rounding are reported (`frame_offsets_subframe`). Cameras that are not genuinely synchronized differ by fractions of a frame, which a whole-frame shift cannot remove. Manual offsets: `--frame-offsets 0,2,-1,…` (camera = file-name order). They may have decimals, for example offsets measured in a recording where the board moved (`--frame-offsets 0,-0.025,0.681`). The decimals are kept for `--subframe` and the RIFE command. Convention: *camera frame = reference frame + offset*.

   **The sub-frame offsets need a moving board.** Their precision comes from the board's motion. On the lab's frontal clip, with the board held still, the estimate was off by 0.1–0.2 frame. On a segment where the board moved, a recalibration after shifting the videos by whole frames recovered the expected remainder within 0.015 frame.

   Fractions of a frame can be used in two ways. With `undistort --subframe`, the pixel CSVs are interpolated in time. Alternatively, the videos themselves are aligned with RIFE. When a camera is more than 0.1 frame off and no camera dropped frames, the log and README print a ready **Video Event Equalizer** command with decimal events. See [video_event_equalizer](video_event_equalizer.md#sub-frame-events-decimals).
4. **Extrinsics.** Board pose per camera and frame (planar PnP), initial camera poses from the cameras that see the board at the same time, then a bundle adjustment of all camera and board poses with fixed intrinsics. Board views with error above max(3 × median, 2 px) are removed once and the fit is repeated.
5. **World frame.**
   - `--world board` (default): origin at corner 0 (top-left inner corner) of the board in the frame seen by most cameras. With `--board-axes upright`, X runs along the board rows, Z points up and Y points through the board, away from the cameras.
   - `--world control`: a CSV with `name,x,y,z` in metres (`--control-points`) and a folder (`--control-pixels`) with one *vailá* pixel CSV per camera, named with the video stem, holding the same points (first row: `frame, p1_x, p1_y, …`). The points are triangulated and a rigid transform (or similarity with `--control-scale`) aligns them; residuals are reported per point.
6. **Checks.**
   - Bundle adjustment reprojection error per camera.
   - Board corners triangulated with the exported DLTs after undistortion (`vaila.rec3d.rec3d_available_views`), compared with the square size. This uses the calibration frames (consistency check).
   - **Held-out blocks** (`--holdout-block 60`): extrinsics fitted on alternate 60-frame blocks and the board triangulated in the other blocks. Same session and board; intrinsics from all views.

## Recording the calibration videos

- Show the board to every camera, close and far, and move it over the whole image, including edges and corners. The lens distortion is only measured where the board appeared; the coverage image shows that region.
- **Tilt the board 30–45° in several directions** (left/right and towards/away from the camera). A board held still in one place gives a focal length that trades against distance. One example is the lab's frontal clip of the board standing on the force plate, with a tilt of 7–8°. There the cameras 06/07/08 got f = 1284 / 1195 / 989 px instead of 1075 / 1125 / 1055 px. Points 2–3 m from the board were then 58 mm (median) away from the 8-camera calibration, while its own validation still read 0.76 mm. If such a clip defines the world frame, load the lens of a calibration where the board moved (`--intrinsics-toml-dir`).
- Keep the board upright (rows roughly horizontal, within ±45°): the corner numbering relies on it.
- Several cameras should see the board at the same moments; the cameras must form one connected chain of shared views.
- Measure the square side over several squares with a ruler; it sets the metric scale of everything. The lab board has 98 mm squares (`--square 0.098`, the default). A wrong value scales every coordinate: 0.1 instead of 0.098 makes 1 m read 1.020 m.

## After calibration: 3D reconstruction

1. Remove the lens distortion from the pixel CSVs with this calibration. Each CSV name must start with or contain its camera video name (e.g. `169.254.43.186_01_..._markers.csv`):

   ```bash
   uv run vaila/chessboard_calib.py undistort --calib out/processed_chessboard_calib_YYYYMMDD_HHMMSS \
       -i pixels/ --apply-offsets
   ```

   In the GUI: **Chessboard Calib** → **Undistort pixels for Rec3D...**. `--apply-offsets` rewrites the `frame` column to the reference timeline (*frame − offset*). The offsets were measured in the calibration recording, so they apply only to pixel CSVs of those same recordings. Leave the option out if the marker videos are already synchronized, and measure the offsets again for a new recording.

   `--subframe` (GUI: answer *yes* to "Use the sub-frame offsets") uses the offsets before rounding (`frame_offsets_subframe`) instead. The value of each camera at reference frame *n* is interpolated linearly between its frames ⌊n + e⌋ and ⌈n + e⌉. A missing or empty neighbour gives an empty value: gaps and dropped frames are not filled. An integer offset gives exactly the result of `--apply-offsets`. A calibration made with `--sync none` or manual offsets has no fractional offsets, and `--subframe` stops with an error. Pixel CSVs of videos already aligned with the Video Event Equalizer need neither option.

   The CSV frame numbers must follow the video timestamps, as getpixelvideo and this module do. Tools that count frames while reading (Markerless 2D, Video to PNG) number every frame after a dropped frame differently. If the report lists dropped frames, run those tools on constant-frame-rate copies (`ffmpeg -i <camera>.mp4 -fps_mode cfr -r 30 -c:v libx264 -crf 12 <camera>_cfr.mp4`). In a copy, a dropped frame repeats the previous image, and the frame numbers match the timestamps.
2. Run **Rec3D 1DLT** with the `.dlt3d` files and the undistorted CSVs. Step 1 prints the exact command (`python -m vaila.rec3d_one_dlt3d --dlt3d … --pixels … --fps … -o …`).

> **Do not use Distort CSV before Rec3D with these files.** `vaila_datdistort.py` reprojects with `cv2.getOptimalNewCameraMatrix` at a fixed 1920×1080, which is a different camera matrix from the one in the `.dlt3d` files; the 3D points would be wrong.

Verified on the lab videos (8 cameras; 06 and 08 November 2026). All 8 cameras lost 2–3 frames at the same moment, about 14.6 s into the recording.

- **Offsets:** with timestamp numbering, the offsets of cameras 07 and 08 relative to camera 06 are the same before and after the drop (−3 and +5). Counting frames gave −3/−4 and +5/+4.
- **8 cameras, 98 mm squares** (`processed_chessboard_calib_20261108_150033`):
  - bundle adjustment 0.47 px;
  - board triangulation 0.76 mm median (p95 1.16 mm), held-out blocks 0.77 mm (p95 1.20 mm);
  - board tilt 54–70° in every camera.
  - The earlier run with 0.1 m had the same lens and every coordinate 2 % larger.
- **Rec3D 1DLT, cameras 06/07/08,** board corners over 352 frames:
  - all frames: 0.74 mm median with `--apply-offsets`, 0.69 mm with `--subframe`;
  - the 10 % of frames where the board moved fastest: 0.70 → 0.62 mm median, p95 0.85 → 0.72 mm.
- **Frontal clip** (cameras 06/07/08, board standing still on the force plate, 181 frames). The 8-camera calibration serves as the comparison:

  | | Lens estimated from the clip | Lens of the 8-camera run |
  |---|---|---|
  | f (px) | 1284 / 1195 / 989 | 1075 / 1125 / 1055 |
  | Camera distances 06–07 / 06–08 (m) | 3.15 / 7.23 | 2.53 / 6.62 (8 cameras: 2.50 / 6.61) |
  | Disagreement 1–1.5 m from the board | 12 mm | 2.4 mm |
  | Disagreement 2–3 m from the board | 58 mm | 6.0 mm |
  | Own validation | 0.76 mm | 0.75 mm |

- **RIFE alignment** on a segment where the board moved (frames 750–930 of cameras 06/07/08). After shifting by whole frames only, a new calibration measured −0.038 / −0.332 frame, against −0.025 / −0.319 expected. With the decimal events (RIFE), it measured −0.038 / −0.057 frame, and the bundle adjustment fell from 0.36 to 0.28 px.

## CLI

```bash
uv run vaila/chessboard_calib.py                                   # GUI
uv run vaila/chessboard_calib.py -i videos/ --pattern 6x8 --square 0.098
uv run vaila/chessboard_calib.py -i cut_videos/ --intrinsics-toml-dir out/processed_chessboard_calib_YYYYMMDD_HHMMSS
uv run vaila/chessboard_calib.py -i videos/ -o out/ --step 1 --zoom 2 --sync auto
uv run vaila/chessboard_calib.py -i videos/ --reuse-detections out/processed_chessboard_calib_YYYYMMDD_HHMMSS \
    --world control --control-points floor_points.csv --control-pixels floor_pixels/
uv run vaila/chessboard_calib.py undistort --calib out/processed_chessboard_calib_YYYYMMDD_HHMMSS -i pixels/ --apply-offsets
uv run vaila/chessboard_calib.py undistort --calib out/processed_chessboard_calib_YYYYMMDD_HHMMSS -i pixels/ --subframe
```

On NVIDIA/CUDA workstations use `uv run --no-sync`.

| Option | Default | Meaning |
|---|---|---|
| `-i/--input` | — | folder with one video per camera |
| `-o/--output` | input folder | output parent folder |
| `--pattern` | `6x8` | inner corners per row x per column (a board of 7 x 9 squares is `6x8`) |
| `--square` | `0.098` | square side in metres (the lab board); sets the scale of everything |
| `--step` | `1` | use every N-th frame |
| `--zoom` | `2.0` | upscale of the crop around the board |
| `--search-every` | `10` | full-frame search interval when the board is lost |
| `--intrinsics` | `auto` | `auto`, `full`, `standard`, `constrained`, `minimal` |
| `--intrinsics-input` | — | separate intrinsics videos (same file stems) |
| `--intrinsics-toml-dir` | — | load the lens of each camera from `<camera>_intrinsics.toml` (e.g. a `processed_chessboard_calib_*` folder), matched by camera name |
| `--sync` | `auto` | `auto` or `none` |
| `--frame-offsets` | — | manual offsets, comma-separated |
| `--max-lag` | `30` | largest offset searched |
| `--world` | `board` | `board` or `control` |
| `--board-axes` | `upright` | axes for `--world board` (`upright` or raw `board`) |
| `--control-points`, `--control-pixels`, `--control-scale` | — | control-point world frame |
| `--reuse-detections` | — | reuse `<stem>_board_pixels.csv` of a previous run |
| `--max-ba-frames` | `1000` | most frames in the bundle adjustment |
| `--holdout-block` | `60` | held-out check block size (0 disables) |
| `--workers` | CPU count | parallel detection processes |

`undistort` options: `--calib`, `-i/--input`, `-o/--output`, `--apply-offsets` (whole frames), `--subframe` (offsets before rounding, linear interpolation).

## Limits

- The lens model is measured only where the board appeared. With wide-angle lenses and a board that stays near the image centre, the distortion near the image border is extrapolated; points reconstructed there are less accurate than the validation numbers suggest.
- Both checks use the same session and board. They show that the cameras agree with each other and with the board size; they are not an independent accuracy measurement against a reference system.
- A board that keeps one orientation does not determine the focal length (see *Board tilt* above). The validation cannot detect this, because it measures the board itself.
- Offsets measured by `--sync auto` come from the board's motion. With a board that hardly moves, they are uncertain by about 0.1–0.2 frame.
- `--apply-offsets` uses whole frames. Cameras that are not genuinely synchronized keep up to half a frame between them (17 ms at 30 fps), and fast movements carry that timing error. `--subframe` and the RIFE alignment reduce it, but their in-between values are estimates, not measurements.
- The fisheye lens model is not offered; the exported TOML and the undistortion use the standard OpenCV model (k1, k2, p1, p2, k3).

## Tests

```bash
uv run pytest tests/test_chessboard_calib.py tests/test_video_timestamp_numbering.py tests/test_video_event_equalizer.py -v
```

Synthetic cameras and boards with known truth. The tests cover:

- corner order and detection of a small board with zoom;
- intrinsics and bundle adjustment;
- frame offsets, including a lag of a fraction of a frame;
- TOML format compatibility and DLT export;
- an end-to-end run on rendered videos with one camera starting 3 frames late;
- a video with dropped frames: timestamp numbers in the detection and in getpixelvideo, paused and playing;
- the lens loaded by camera name for cut and equalized copies, and the board-tilt warning;
- `--subframe`: linear motion recovered exactly, gaps left empty, integer offsets equal to `--apply-offsets`;
- the RIFE command keeps every position inside every video, and the Video Event Equalizer accepts decimal events (`tests/test_video_event_equalizer.py`).
