# User Guide - Pynalty Analysis Tool

## Introduction

**Pynalty** is a specialized video analysis tool designed to calculate penalty kick statistics, specifically the ball's velocity and distance traveled. Integrated into the **vailá** toolbox, it provides a user-friendly interface to mark critical moments (kick and goal) and positions to automatically compute results based on goal calibration.

**Version:** 0.1.0  
**Date:** 19 December 2025  
**Project:** vailá - Multimodal Toolbox

## Key Features

- **Video-Based Analysis:** Analyze penalty kicks directly from video files.
- **State-Based Workflow:** Guided step-by-step process (Kick Frame -> Goal Frame -> Calibration).
- **Automatic Calculation:** Computes distance, velocity (m/s and km/h) using DLT 2D reconstruction.
- **Data Export:** Save results to **TOML** (for reloading) and **CSV** (for Excel/Analysis).
- **Interactive GUI:** Zoom, pan, frame slider, and visual markers.
- **Help Overlay:** Press **H** for quick reference.

## Workflow

1.  **Launch Pynalty**: Click the "Pynalty" button in the vailá main window (Frame B, Row 5).
2.  **Load Video**: Select the video file containing the penalty kick.
3.  **Select Kick Frame**: Use arrows/slider to find the moment the kicker hits the ball. Press **ENTER**.
4.  **Mark Ball (Kick)**: Click on the center of the ball.
5.  **Mark Goalkeeper (Kick)**: Click on the center of the Goalkeeper at the kick moment.
6.  **Select GK Move Frame**: Navigate to the frame where the Goalkeeper STARTS moving (reaction/anticipation). Press **ENTER**.
7.  **Select Goal Frame**: Navigate to the moment the ball crosses the goal line (or is saved). Press **ENTER**.
8.  **Mark Ball (Goal)**: Click on the center of the ball at the goal line.
9.  **Mark Goalkeeper (Goal)**: Click on the center of the Goalkeeper at the goal moment.
10. **Calibration**: Click the 4 corners of the goal in the specified order (displayed on screen):
    1.  **Bottom-Left** (Poste Esquerdo Inferior) / (Trave Esquerda Baixo)
    2.  **Top-Left** (Poste Esquerdo Superior) / (Trave Esquerda Cima)
    3.  **Top-Right** (Poste Direito Superior) / (Trave Direita Cima)
    4.  **Bottom-Right** (Poste Direito Inferior) / (Trave Direita Baixo)
11. **Results**: View distance, velocity, and GK response time statistics.

## Controls

### Navigation
| Key | Action |
|-----|--------|
| **SPACE** | Play / Pause Video |
| **Left / Right Arrows** | Previous / Next Frame |
| **Up / Down Arrows** | +/- 10 Frames |
| **Mouse Wheel** | Zoom In / Out |
| **+ / -** | Zoom In / Out |
| **Middle Click + Drag** | Pan Image |
| **Slider (Bottom)** | Drag to seek frames |

### Actions
| Key | Action |
|-----|--------|
| **ENTER** | Confirm Frame Selection (Kick/GK/Goal) |
| **Left Click** | Mark Ball / GK / Calibration Point |
| **Right Click** | Undo Last Mark / Go Back Step |
| **S** | Save Analysis to TOML |
| **C** | Save Results to CSV |
| **L** | Load Analysis from TOML |
| **F** | Change Video FPS (Manual Override) |
| **H** | Toggle Help Overlay |

## Saving and Loading

- **TOML (S Key):** Saves marked points and state. Use this to resume work later (Load with **L**).
- **CSV (C Key):** Saves final calculated statistics (Velocity, Distance, Coordinates) and raw points to a CSV file for reporting.

## Troubleshooting

- **Wrong Velocity?** Ensure the goal calibration points are clicked in the correct order (Bottom-Left -> Top-Left -> Top-Right -> Bottom-Right). The system assumes standard goal dimensions (7.32m x 2.44m).
- **Video not loading?** Ensure `opencv-python` is installed and the video format is supported.
