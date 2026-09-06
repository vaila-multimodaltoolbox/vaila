# Draw Sports Fields / Courts

Main GUI button: **Draw Sports** (Frame C, visualization column).  
Choose a surface, then **Help** in that dialog opens the HTML page in your browser.

## Available models

| Choice | Tool | Model file (`vaila/models/`) |
|---|---|---|
| Soccer | `drawsportsfields.run_drawsportsfields("soccer")` | `soccerfield_ref3d.csv` (full FIFA markings) |
| FIFA Dataset (32 KP order) | `drawsportsfields.run_drawsportsfields("fifa_dataset")` | `soccerfield_ref3d_fifa.csv` + overlay `01..32` canonical keypoints from `fifa_dataset_builder` |
| FIFA 49 KP (Kiki Model) | `drawsportsfields.run_drawsportsfields("kiki")` | `soccerfield_kiki.csv` (32 pitch + 16 3D features + center spot point 48) |
| Tennis | `drawsportsfields.run_drawsportsfields("tennis")` | `tenniscourt_ref3d.csv` |
| Basketball | `drawsportsfields.run_drawsportsfields("basketball")` | `basketballcourt_ref3d.csv` |
| Volleyball | `drawsportsfields.run_drawsportsfields("volleyball")` | `volleyball_ref3d.csv` |
| Futsal | `drawsportsfields.run_drawsportsfields("futsal")` | `futsal_ref3d.csv` |
| Handball | `drawsportsfields.run_drawsportsfields("handball")` | `handball_ref3d.csv` |

## FIFA 49 KP (Kiki Model) & De-overlapped Coincident Keypoints

Use `--type kiki` (or load `vaila/models/soccerfield_kiki.csv`) to visualize all 49 keypoints (32 pitch line intersections, 16 3D features: corner flag tops at $z=1.5\,\text{m}$, goal post tops at $z=2.44\,\text{m}$, goal net ground points, plus the center spot point 48 at $(0,0,0)$).

Key features:
- **Center field spot (point 48)**: Explicitly marks the central point inside the center circle at $(X=0, Y=0, Z=0)$, next in numerical order.
- **Differentiated position & orientation**: For points with identical $(x, y)$ ground coordinates (e.g. corner ground points `0, 5, 24, 29` vs corner flag tops `38, 39, 46, 47`, or goal post bases vs tops), labels are rendered with spacious separation, horizontal orientation, and dotted leader lines connecting each badge to its marker.
- **Goal post vs net separation**: Goal post labels and net points are drawn backwards ("para trás", behind the goal line) avoiding crossing or encroaching into the penalty area.
- **Reference label de-overlapping**: Any custom model loaded where multiple control points share $(x, y)$ coordinates automatically benefits from radial offset and angular orientation spreading with leader lines.

## Interactive Calibration Keypoint Editor

Button **Calib Keypoints** (in the top toolbar):

1. **Click-to-Add (Define Label Name)**: Left-click anywhere on the pitch to capture data-space metric coordinates and immediately open the definition dialog with the Label Name pre-focused. Type the name, select or type $z$ (with quick presets: Ground 0m, Flag 1.5m, Crossbar 2.44m), and press **Enter** to place the point.
2. **Direct Coordinates / Manual Entry**: A dedicated section in the Editor window allows typing coordinates directly ($X, Y, Z$, label name, point number) without clicking, with quick coordinate presets (Center, Left Goal, Right Goal, Ground, Flag, Bar) and instant `+ Add Keypoint (From Coords)`. Selecting any existing keypoint in the table populates the form for rapid coordinate and label editing.
3. **Augment an existing model**: Start from `soccerfield_kiki.csv`, `soccerfield_ref3d.csv`, or any loaded model, add customized calibration control points, and save.
4. **Build from scratch**: Click **Create From Scratch (Clear All)** to hide existing reference points while keeping regulation pitch lines visible, then click on the pitch or type coordinates to construct a new reference model from zero.
5. **Save Model CSV**: Export the keypoint set to CSV (`point_name,point_number,flip_idx,x,y,z,x_norm,y_norm`), directly usable by DLT3D, REC3D, getpixelvideo, and `readcsv.py`.
6. **Save Model C3D**: Export calibration points directly to standard `.c3d` files (`parameters.POINT` labels, rates, units) for verification in 3D viewers (`showc3d.py`, `viewc3d_pyvista.py`).
7. **View in 3D**: Instantly launch 3D visualization of the calibration keypoints in PyVista or Matplotlib to check pitch geometry, elevations, and label placement in a real 3D environment.

## 3D Environment Verification (`showc3d`, `readcsv`, `viewc3d_pyvista`)

Soccer field calibration models can be inspected in full 3D across all vailá visualization tools:
- **`showc3d.py` (Matplotlib)**: Automatically detects soccer field scale, renders regulation pitch boundary lines, halfway line, center circle, and 3D point labels with toggle support.
- **`viewc3d_pyvista.py` (PyVista)**: Features green turf plane, regulation field markings at $Z=0$, adaptive camera scaling, and auto-enabled 3D point labels for single-frame calibration models.
- **`readcsv.py`**: Directly recognizes calibration model CSV files (e.g. `soccerfield_kiki.csv` or custom models), parsing keypoints into 3D coordinates ready for PyVista, Open3D, or Matplotlib viewing.

## FIFA Dataset labeling reference (new)

Use `--type fifa_dataset` when your goal is to label new broadcast frames/videos for
the 32-keypoint pitch dataset used by `vaila.fifa_dataset_builder`.

This mode draws the FIFA field and overlays:

- canonical keypoint order **01..32** (exact YOLO-pose order),
- short semantic names near each point for human QA,
- a footer reminder that ordering matches dataset/getpixelvideo export.

This reference is intended to reduce keypoint index swaps while clicking in
`vaila/getpixelvideo.py`.

## Export REF3D for DLT3D

Button **Export REF3D…** (always available once a field model is loaded):

1. Multi-select control points from the loaded model (or the 32 FIFA dataset
   keypoints when `--type fifa_dataset`).
2. Choose p-index base **1** (recommended for `dlt3d.py`) or **0** (FIFA
   getpixelvideo slot style).
3. Saves:
   - `*.ref3d` — one-row world XYZ (`frame,p1_x,p1_y,p1_z,…`) for `dlt3d.py`
   - `*.ref3d_map.csv` — `p_index` ↔ source name / index
   - `*_pixel_template.csv` — empty pixel columns to fill / match in getpixelvideo

Workflow: mark the same `pN` order in getpixelvideo → run `dlt3d.py` (pixel +
`.ref3d`) → use the `.dlt3d` in `rec3d_one_dlt3d.py`. (`fifa_to_dlt.py` is the
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

