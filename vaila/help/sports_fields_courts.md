# Draw Sports Fields / Courts

Main GUI button: **Draw Sports** (Frame C, visualization column).  
Choose a surface, then **Help** in that dialog opens the HTML page in your browser.

## Available models

| Choice | Tool | Model file (`vaila/models/`) |
|---|---|---|
| Soccer | `drawsportsfields.run_drawsportsfields("soccer")` | `soccerfield_ref3d.csv` (full FIFA markings) |
| FIFA Dataset (32 KP order) | `drawsportsfields.run_drawsportsfields("fifa_dataset")` | `soccerfield_ref3d_fifa.csv` + overlay `01..32` canonical keypoints from `fifa_dataset_builder` |
| Tennis | `drawsportsfields.run_drawsportsfields("tennis")` | `tenniscourt_ref3d.csv` |
| Basketball | `drawsportsfields.run_drawsportsfields("basketball")` | `basketballcourt_ref3d.csv` |
| Volleyball | `drawsportsfields.run_drawsportsfields("volleyball")` | `volleyball_ref3d.csv` |
| Futsal | `drawsportsfields.run_drawsportsfields("futsal")` | `futsal_ref3d.csv` |
| Handball | `drawsportsfields.run_drawsportsfields("handball")` | `handball_ref3d.csv` |

## FIFA Dataset labeling reference (new)

Use `--type fifa_dataset` when your goal is to label new broadcast frames/videos for
the 32-keypoint pitch dataset used by `vaila.fifa_dataset_builder`.

This mode draws the FIFA field and overlays:

- canonical keypoint order **01..32** (exact YOLO-pose order),
- short semantic names near each point for human QA,
- a footer reminder that ordering matches dataset/getpixelvideo export.

This reference is intended to reduce keypoint index swaps while clicking in
`vaila/getpixelvideo.py`.

## CSV format

Columns: `point_name`, `point_number`, `x`, `y`, `z` (metres).  
Soccer models must include all points required by `soccerfield_ref3d.csv`.
Simpler models need at least the four corners:
`bottom_left_corner`, `top_left_corner`, `bottom_right_corner`, `top_right_corner`.

## CLI usage

```bash
uv run vaila/drawsportsfields.py -t soccer
uv run vaila/drawsportsfields.py -t fifa_dataset
uv run vaila/drawsportsfields.py -t tennis
uv run vaila/drawsportsfields.py -t basketball
uv run vaila/drawsportsfields.py --field path/to/custom.csv
uv run vaila/drawsportsfields.py --markers data.csv --heatmap
```

## Further reading

- [Tennis court detection (research implementation)](https://github.com/mmmmmm44/tennis_court_detection)
- [sportypy](https://github.com/sportsdataverse/sportypy) — regulation surfaces for several sports

