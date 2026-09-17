# Draw Sports Fields / Courts

Main GUI button: **Draw Sports** (Frame C, visualization column).  
Choose a surface, then **Help** in that dialog opens the HTML page in your browser.

## Available models

| Choice | Tool | Model file (`vaila/models/`) |
|---|---|---|
| Soccer | `drawsportsfields.run_drawsportsfields("soccer")` | `soccerfield_ref3d.csv` (full FIFA markings) |
| FIFA Dataset (32 KP order) | `drawsportsfields.run_drawsportsfields("fifa_dataset")` | `soccerfield_ref3d_fifa.csv` + overlay `0..31` canonical keypoints from `fifa_dataset_builder` |
| Soccer Field Kiki — 49 KP | `drawsportsfields.run_drawsportsfields("kiki")` | `soccerfield_kiki.csv` (32 pitch-line keypoints `[0..31]` + 17 3D/pitch features `[32..48]`) |
| Tennis | `drawsportsfields.run_drawsportsfields("tennis")` | `tenniscourt_ref3d.csv` |
| Basketball | `drawsportsfields.run_drawsportsfields("basketball")` | `basketballcourt_ref3d.csv` |
| Volleyball | `drawsportsfields.run_drawsportsfields("volleyball")` | `volleyball_ref3d.csv` |
| Futsal | `drawsportsfields.run_drawsportsfields("futsal")` | `futsal_ref3d.csv` |
| Handball | `drawsportsfields.run_drawsportsfields("handball")` | `handball_ref3d.csv` |

## Soccer Field Kiki (49 KP) & De-overlapped Coincident Keypoints

Use `--type kiki` (or load `vaila/models/soccerfield_kiki.csv`) to visualize all 49 reference points, indices 0..48 (32 pitch line intersections `[0..31]` plus 17 3D/pitch features `[32..48]`: corner flag tops at $z=1.5\,\text{m}$, goal post tops at $z=2.44\,\text{m}$, goal net ground points, plus the center spot point 48 at $(0,0,0)$).

Key features:
- **Center field spot (point 48)**: Explicitly marks the central point inside the center circle at $(X=0, Y=0, Z=0)$, next in numerical order.
- **Differentiated position & orientation**: For points with identical $(x, y)$ ground coordinates (e.g. corner ground points `0, 5, 24, 29` vs corner flag tops `38, 39, 46, 47`, or goal post bases vs tops), labels are rendered with spacious separation, horizontal orientation, and dotted leader lines connecting each badge to its marker.
- **Goal post vs net separation**: Goal post labels and net points are drawn backwards ("para trás", behind the goal line) avoiding crossing or encroaching into the penalty area.
- **Reference label de-overlapping**: Any custom model loaded where multiple control points share $(x, y)$ coordinates automatically benefits from radial offset and angular orientation spreading with leader lines.

## Penalty arc ("meia lua") geometry

Law 1 / the FIFA *Football Stadiums Guidelines* define the penalty arc as the part
of a **circle of radius 9.15 m centred on the penalty mark** that falls outside the
penalty area. Its two endpoints are therefore not free parameters: they are the
**intersections of that circle with the penalty-area line**.

`penalty_arc_geometry()` computes those endpoints from that definition rather than
reading them from the model, so the drawn arc always starts and ends exactly on the
penalty-area line. On a regulation 105 x 68 m pitch the penalty-area line is 5.5 m
from the penalty mark, so the half-angle is

$$\theta = \arccos\!\left(\frac{16.5 - 11}{9.15}\right) = 53.0518^\circ,
\qquad \Delta y = \sqrt{9.15^2 - 5.5^2} = 7.312489\ \text{m}.$$

The arc radius is resolved in this order:

1. the distance from the penalty mark to a stored `*_penalty_arc_*_intersection`
   point, which keeps non-uniformly scaled models (small-sided pitches) consistent
   with their own stored geometry;
2. the model's own centre-circle radius, which the norm makes equal to the arc
   radius (this is the path taken by `soccerfield_kiki.csv`, which stores no arc
   keypoints);
3. the regulation 9.15 m.

`*_penalty_arc_top` is the arc **apex**, at 11 + 9.15 = 20.15 m (and
105 - 11 - 9.15 = 84.85 m), *not* a point on the penalty-area line.

Regulation values now encoded exactly in `soccerfield_ref3d.csv`,
`soccerfield_ref3d_fifa.csv` and `soccerfield_kiki.csv`:

| Marking | Value |
|---|---|
| Goal width (posts) | 7.32 m, i.e. 34 +/- 3.66 -> `30.34` / `37.66` |
| Goal area | 5.5 m deep x 18.32 m wide -> `24.84` / `43.16` |
| Penalty area | 16.5 m deep x 40.32 m wide -> `13.84` / `54.16` |
| Penalty mark | 11 m from the goal line |
| Penalty arc / centre circle radius | 9.15 m |
| Arc endpoints on the penalty-area line | `26.687511` / `41.312489` |
| Arc apex | `20.15` / `84.85` |

`vaila/models/planar_targets/soccerfield_broadcast.toml` carries the same arc as
angles for the homography overlay: `-53.0518 .. 53.0518` (left) and
`126.9482 .. 233.0518` (right).

### Points 10, 11, 18 and 19: Kiki versus FIFA Dataset

The two modes deliberately share the index order but no longer conflate two
different purposes:

- **Soccer Field Kiki — 49 KP** is the visual/calibration model. Points **10/11**
  (left) and **18/19** (right) are the four visible meetings between the penalty
  arcs and penalty-area front lines: `(±35.95, ±7.312489)` in the centred frame.
  Their orange markers therefore sit exactly on the ends of the white arcs.
- **FIFA Dataset — 32 KP** remains compatible with the existing
  Roboflow/Supervision training schema. Its same indices are virtual subdivision
  anchors at the goal-area Y levels `(±35.95, ±9.16)`; changing those coordinates
  would invalidate existing YOLO labels and trained models.

The Kiki CSV names the four visible points explicitly as
`*_penalty_arc_*_intersection`. The numbered badge is offset slightly from its
orange marker so the badge does not cover the actual intersection.

## Interactive Calibration Keypoint Editor

Button **Calib Keypoints** (in the top toolbar):

1. **Click-to-Add (Define Label Name)**: Left-click anywhere on the pitch to capture data-space metric coordinates and immediately open the definition dialog with the Label Name pre-focused. Type the name, select or type $z$ (with quick presets: Ground 0m, Flag 1.5m, Crossbar 2.44m), and press **Enter** to place the point.
2. **Direct Coordinates / Manual Entry**: A dedicated section in the Editor window allows typing coordinates directly ($X, Y, Z$, label name, point number) without clicking, with quick coordinate presets (Center, Left Goal, Right Goal, Ground, Flag, Bar) and instant `+ Add Keypoint (From Coords)`. Selecting any existing keypoint in the table populates the form for rapid coordinate and label editing.
3. **Augment an existing model**: Start from `soccerfield_kiki.csv`, `soccerfield_ref3d.csv`, or any loaded model, add customized calibration control points, and save.
4. **Build from scratch**: Click **Create From Scratch (Clear All)** to hide existing reference points while keeping regulation pitch lines visible, then click on the pitch or type coordinates to construct a new reference model from zero.
5. **Save Model CSV**: Export the keypoint set to CSV (`point_name,point_number,flip_idx,x,y,z,x_norm,y_norm`), directly usable by DLT3D, REC3D, getpixelvideo, and `readcsv.py`.
6. **Save Model C3D**: Export calibration points directly to standard `.c3d` files (`parameters.POINT` labels, rates, units) for verification in 3D viewers (`showc3d.py`, `viewc3d_pyvista.py`).
7. **View in 3D**: Instantly launch complete 3D soccer field visualization in native Matplotlib (`plot_calibration_model_3d`). Renders full regulation 3D pitch markings, 3D goals (vertical posts [traves], horizontal crossbars [travessões], net depth frameworks and supports), 4 corner flagpoles with pennants, keypoint drop lines connecting elevated points to the pitch floor (Z=0), and box aspect ratio (`daspect`) control.

## Responsive 2-Row Toolbar

To ensure all controls remain visible without requiring the user to maximize or resize the window, `run_soccerfield` organizes buttons into two logical, responsive rows:
- **Row 1 (Field & View)**: `Load Default Field`, `Load Custom Field`, `Surface Color`, `Hide/Show Reference Points`, `Show/Hide Axis Values`, `Heatmap`, `Help`.
- **Row 2 (Tools & Calib)**: `Load Markers CSV`, `Select Markers`, `Load Scout CSV`, `Scout Filters`, `Create Manual Markers`, `Clear All Markers`, `Calib Keypoints` (teal), `Export REF3D…` (blue).

Both rows fit comfortably within standard display resolutions ($\ge 800\,\text{px}$ width) without horizontal clipping. In the Calibration Keypoint Editor, the listbox is optimized to height 6 with minimum geometry constraints (`780x640`), guaranteeing the bottom action buttons (`Save Model CSV…`, `Save Model C3D…`, `View in 3D…`, `Load Base Model…`, `Close`) remain permanently visible and accessible.

## 3D Environment Verification (`plot_calibration_model_3d`, `showc3d`, `readcsv`, `viewc3d_pyvista`)

Soccer field calibration models can be inspected in full 3D across all vailá visualization tools:
- **`plot_calibration_model_3d` (Matplotlib Native)**: Built directly into `drawsportsfields.py` for calibration preview. Features:
  - **3D Pitch Lines**: Complete perimeter, halfway line, center circle ($R=9.15\,\text{m}$), center spot, penalty boxes, goal boxes, penalty spots, penalty arcs, and corner arcs.
  - **3D Goals**: Regulation vertical posts ($Z \in [0, 2.44]\,\text{m}$), crossbar at $Z=2.44\,\text{m}$, rear ground depth bars ($2.0\,\text{m}$ behind goal line), top depth bars, rear vertical supports, and subtle net mesh lines.
  - **Corner Flags**: 4 vertical flagpoles ($Z \in [0, 1.5]\,\text{m}$) with directional pennants.
  - **Daspect / Human-Friendly Aspect Control**: Default aspect ratio ($4.5\times Z$) expands vertical features so goals and elevated calibration points are easily visible and readable instead of squashed into $2\%$ of field length. A toolbar button toggles instantly between **Human-Friendly ($4.5\times Z$)** and **True 1:1 Scale**.
  - **Camera View Presets**: One-click camera buttons for *Isometric*, *Touchline*, *Behind Goal*, and *Top-Down (Plan)* views.
  - **Toggles**: Dedicated buttons to toggle point labels and vertical drop lines ($Z \to 0$).
- **`showc3d.py` (Matplotlib)**: Automatically detects soccer field scale, renders regulation pitch boundary lines, halfway line, center circle, and 3D point labels with toggle support.
- **`viewc3d_pyvista.py` (PyVista)**: Features green turf plane, regulation field markings at $Z=0$, adaptive camera scaling, and auto-enabled 3D point labels for single-frame calibration models.
- **`readcsv.py`**: Directly recognizes calibration model CSV files (e.g. `soccerfield_kiki.csv` or custom models), parsing keypoints into 3D coordinates ready for PyVista, Open3D, or Matplotlib viewing.

## FIFA Dataset labeling reference (new)

Use `--type fifa_dataset` when your goal is to label new broadcast frames/videos for
the 32-keypoint pitch dataset used by `vaila.fifa_dataset_builder`.

This mode draws the FIFA field and overlays:

- canonical keypoint order **0..31** (exact zero-based YOLO-pose order),
- short semantic names near each point for human QA,
- a footer reminder that ordering matches dataset/getpixelvideo export.

This reference is intended to reduce keypoint index swaps while clicking in
`vaila/getpixelvideo.py`.

## Export REF3D for DLT3D

Button **Export REF3D…** (always available once a field model is loaded):

1. Multi-select control points from the loaded model (or the 32 FIFA dataset
   keypoints when `--type fifa_dataset`).
2. Choose p-index base **0** (default, and the *vailá* standard: `getpixelvideo.py`,
   `dlt3d.py` and `rec3d*.py` all number points from `p0`) or **1** (legacy only,
   for interoperability with 1-based datasets).
3. Saves:
   - `*.ref3d` — one-row world XYZ (`frame,p0_x,p0_y,p0_z,…`) for `dlt3d.py`
   - `*.ref3d_map.csv` — `p_index` ↔ source name / index
   - `*_pixel_template.csv` — empty pixel columns to fill / match in getpixelvideo

The dialog header states how many reference points the loaded model exposes and
their label range, so the count cannot be confused with the last label (the Kiki
model has **49** points labelled `0..48`, not 48 points).

Workflow: mark the same `pN` order in getpixelvideo → run `dlt3d.py` (pixel +
`.ref3d`) → use the `.dlt3d` in `rec3d_one_dlt3d.py`. Because `dlt3d.py` pairs
pixel and reference points **by numeric index**, a reference file exported with a
different base than the pixel file would pair `pN` with `pN` where `pN-1` was
meant; since 0.4.3 `dlt3d.py` detects that exact shift-by-one and refuses the run
instead of returning a plausible but wrong calibration. (`fifa_to_dlt.py` is the
alternate path from FIFA `cameras/*.npz`, not this REF3D file.)

## CSV format

Columns: `point_name`, `point_number`, `x`, `y`, `z` (metres).  
Soccer models must include all points required by `soccerfield_ref3d.csv`.
Simpler models need at least the four corners:
`bottom_left_corner`, `top_left_corner`, `bottom_right_corner`, `top_right_corner`.

## CLI usage

```bash
uv run vaila/drawsportsfields.py -t soccer
uv run vaila/drawsportsfields.py -t kiki
uv run vaila/drawsportsfields.py -t fifa_dataset
uv run vaila/drawsportsfields.py -t tennis
uv run vaila/drawsportsfields.py -t basketball
uv run vaila/drawsportsfields.py --field vaila/models/soccerfield_kiki.csv
uv run vaila/drawsportsfields.py --markers data.csv --heatmap
```

## Further reading

- [Tennis court detection (research implementation)](https://github.com/mmmmmm44/tennis_court_detection)
- [sportypy](https://github.com/sportsdataverse/sportypy) — regulation surfaces for several sports

---

**Version:** 0.4.3<br>
**Updated:** 17 September 2026
