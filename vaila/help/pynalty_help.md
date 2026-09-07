# User Guide - Pynalty Analysis Tool

## Introduction

*Pynalty* analyses penalty kicks from video for coaches and keepers: flight
speed and placement, goalkeeper reaction and dive, reach/time-to-ball
saveability, optional MediaPipe pose in athlete crops, and self-contained HTML
reports (EN/PT) plus CSV rows for a session database.

**Version:** 0.3.129  
**Date:** 07 September 2026  
**Project:** *vailá* - Multimodal Toolbox

## Key Features

- **7-step didactic wizard** — welcome overlay, left step panel (why/how/readback), progress, zoom toward cursor.
- **DLT2D goal calibration** — pixels → metres on the goal plane; configurable goal size and penalty distance.
- **Ball path** — manual marks and/or YOLO auto-detect (`A`); interpolated gaps; pixel + modelled 3D CSVs; HTML canvas animation.
- **Athlete boxes + pose** — drag Kicker/GK boxes, MediaPipe on crop-upscale (`P`); pose CSVs and kinematics.
- **Anthropometrics** — GK stature / arm span / standing reach (`B`) drive the reach envelope and save verdict.
- **Reports** — `report.html` / `report_pt.html`, tidy `results.csv`, wide `pynalty_summary.csv`, appendable `pynalty_database.csv`.

## Workflow (wizard)

1. **Keeper starts moving** — scrub, then ENTER or click to lock the frame.
2. **Ball contact** — click BALL centre (frame locks automatically), then KEEPER centre.
3. **Ball at goal line** — click BALL, then KEEPER, then **G**=goal / **D**=save / **M**=miss / **W**=woodwork.
4. **Goal calibration** — four corners in order.
5. **Ball path (optional)** — `A` YOLO or click; skip with Step >.
6. **Pose boxes (optional)** — drag boxes + `P`; skip with Step >.
7. **Body measures (optional)** — `B`, or skip to use default GK stature **1.88 m**.

`Step >` only advances when the current **required** step is finished. Optional steps never block save.

## Launch

- **GUI:** Frame B → **Pynalty**, or `uv run vaila/pynalty.py` (file picker).
- **CLI marking:** `uv run vaila/pynalty.py -i video.mp4 -o out_dir -c data.toml`
- **Headless regenerate:** add `--report-only` (requires `-c`).
- **Extras:** `--database`, `--no-wizard`, `--auto-ball`, `--pose`, `--penalty-distance`, `--goal-width`, `--goal-height`, `--lang en|pt|both`

Every GUI/CLI run prints a copy-paste **Equivalent CLI** line (`>>` prefix).

## Controls

| Key / action | Effect |
| --- | --- |
| Left / Right | Prev / next frame |
| Space | Play / pause |
| Mouse wheel | Zoom toward cursor |
| Drag | Pan |
| ENTER | Confirm step / open body dialog on step 7 |
| A | Auto-detect ball (YOLO) |
| P | Run MediaPipe pose in boxes |
| B | Anthropometrics dialog |
| S / L / H | Save / Load TOML / Help |
| Buttons | Step navigation and the same actions |

## Outputs

Written under `<video_stem>_results/` (or `-o`):

| File | Role |
| --- | --- |
| `data.toml` | Reloadable marks + geometry + anthro |
| `results.csv` | Tidy per-variable table |
| `pynalty_summary.csv` | One wide row for the penalty |
| `pynalty_database.csv` | Session DB (`--database` or beside output); keyed by video + kick frame |
| `report.html` / `report_pt.html` | Self-contained coaching reports |
| `ball_path_pixel.csv` / `ball_path_3d.csv` | Measured path + gravity flight model |
| `pose_kicker_pixel.csv` / `pose_gk_pixel.csv` | Pose landmarks when step 6 ran |
| `snapshot_*.png` | Event stills |

## Modules

| Module | Responsibility |
| --- | --- |
| `vaila/pynalty.py` | Pygame wizard + CLI |
| `vaila/pynalty_analysis.py` | DLT, flight, zones, reach, metrics |
| `vaila/pynalty_vision.py` | YOLO ball, MediaPipe pose, overlays |
| `vaila/pynalty_report.py` | HTML + CSV writers |

## Notes

- Goal-plane DLT is exact only on the goal mouth; full-flight speeds in the report use the fitted 3D model, not raw pixel deltas.
- Reach analysis needs step 7 anthropometrics; without them the verdict stays “not assessed”.
- YOLO / MediaPipe are optional: mark by hand if those stacks are missing.

## Support

- Help index: `vaila/help/index.html`
- Button doc: `docs/vaila_buttons/pynalty.md`
- Issues: https://github.com/vaila-multimodaltoolbox/vaila/issues
