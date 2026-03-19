# tennis_court

## Module Information

- **Category:** Visualization
- **File:** `vaila/tennis_court.py`
- **Version:** 0.0.2
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** Yes
- **CLI Support:** Yes (`--court`, `--markers`, `--color`, `--heatmap`)

## Description

Draws an ITF-standard tennis court and allows overlaying player position
trajectories (from 2D/3D reconstruction CSV files) and generating KDE heatmaps
of player movement patterns. Mirrors the architecture of `soccerfield.py`.

Court dimensions (ITF Standard):
  - Total length: 23.77 m
  - Doubles width: 10.97 m
  - Singles width: 8.23 m (1.37 m alley on each side)
  - Service box length: 6.40 m (net to service line)
  - Net height at center: 0.914 m (posts: 1.07 m)

## Usage

### GUI
```bash
python tennis_court.py
python -m vaila.tennis_court
```

### CLI
```bash
python tennis_court.py --court <path_to_court_csv>
python tennis_court.py --markers <path_to_markers_csv>
python tennis_court.py --markers <path_to_markers_csv> --heatmap
python tennis_court.py --color clay
```

## Main Functions

**Total functions found:** 16

- `draw_line`
- `draw_rectangle`
- `plot_court`
- `load_and_plot_markers`
- `run_tenniscourt`
- `load_court`
- `load_custom_court`
- `toggle_ref_points`
- `toggle_axis`
- `change_court_color`
- `load_markers_csv_action`
- `open_marker_selection_dialog`
- `toggle_manual_marker_mode`
- `create_marker`
- `delete_marker`
- `save_markers_csv`
- `clear_all_markers`
- `show_heatmap`
- `_draw_court_on_ax`

## Buttons

| Button | Description |
|--------|-------------|
| Load Default Court | Renders the ITF tennis court from default model |
| Load Custom Court | Load a custom court model CSV |
| Surface | Cycle court colours (Hard blue, Hard green, Clay, Grass) |
| Load Markers CSV | Overlay player trajectories from vaila-format CSV |
| Select Markers | Choose which markers to display |
| Heatmap | Generate KDE heatmap from loaded marker trajectories |
| Show/Hide Ref Points | Toggle numbered reference markers |
| Show/Hide Axis | Toggle numeric X/Y axes |
| Create Manual Markers | Click on court to place markers manually |
| Clear All Markers | Remove all overlaid data |
| Help | Show help dialog |

## Manual Markers

- Left-click: place marker at current frame
- Shift + Left-click: place next marker number
- Right-click: delete nearest marker
- Ctrl+S: save markers to CSV

---

Part of vaila - Multimodal Toolbox
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
