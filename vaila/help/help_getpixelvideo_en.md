# User Guide - Pixel Coordinate Tool (getpixelvideo.py)

## Introduction

The Pixel Coordinate Tool (getpixelvideo.py) allows you to mark and save pixel coordinates in video frames. Developed by Prof. Dr. Paulo R. P. Santiago, this tool offers features such as zoom for precise annotations, dynamic window resizing, frame navigation, and the ability to save results in CSV format.

## Requirements

- Python 3.12 or higher
- Libraries: pygame, cv2 (OpenCV), pandas, numpy, tkinter

## Getting Started

1. Run the script using Python: `python getpixelvideo.py`
2. Select a video file when prompted
3. Choose whether to load existing points from a previously saved file

## Interface

The tool interface consists of:
- Video viewing area (upper section)
- Control panel (bottom section) with:
  - Current frame information
  - Slider for navigating between frames
  - Buttons for main functions (Load, Save, Help, 1 Line, Persistence, Sequential)

## Marking Modes

### Normal Mode (default)
- Each click selects and updates the currently selected marker
- Navigate between markers using TAB
- Each marker maintains its ID across all frames
- Activate with: default mode at startup

### 1 Line Mode (C key)
- Creates a sequence of connected points in a single frame
- Each click adds a new marker in sequential order
- Useful for tracing paths or outlines
- Activate with: C key

### Sequential Mode (S key)
- Each click creates a new marker with incremental IDs
- No need to select markers first
- Only available in Normal mode
- Activate with: S key (only in Normal mode)

## Keyboard Commands

### Video Navigation
- **Space**: Play/Pause
- **Right Arrow**: Next frame (when paused)
- **Left Arrow**: Previous frame (when paused)
- **Up Arrow**: Fast forward (when paused)
- **Down Arrow**: Rewind (when paused)

### Zoom and Pan
- **+**: Zoom in
- **-**: Zoom out
- **Middle Click + Drag**: Pan/move the view

### Markers
- **Left Click**: Add/update marker
- **Right Click**: Remove last marker
- **TAB**: Next marker in current frame
- **SHIFT+TAB**: Previous marker in current frame
- **DELETE**: Delete selected marker
- **A**: Add new empty marker to file
- **R** or **D**: Remove selected marker

### Modes
- **C**: Toggle "1 Line" mode
- **S** or **O**: Toggle Sequential mode (only in Normal mode)
- **P**: Toggle Persistence mode
- **1**: Decrease persistence frames (when persistence enabled)
- **2**: Increase persistence frames (when persistence enabled)
- **3**: Toggle between full and limited persistence

### Other
- **ESC**: Save and exit

## Persistence Mode

Persistence mode shows markers from previous frames, creating a visual "trail":
- **P**: Enables/disables persistence
- **1**: Decreases the number of displayed frames
- **2**: Increases the number of displayed frames
- **3**: Toggles between persistence modes (disabled → full → limited)

## Saving and Loading

### Saving Coordinates
- Press **ESC** to save and exit
- Click the **Save** button to save without exiting
- Files are saved as CSV in the same directory as the video
- Different modes save to different files:
  - Normal Mode: `video_name_markers.csv`
  - 1 Line Mode: `video_name_markers_1_line.csv`
  - Sequential Mode: `video_name_markers_sequential.csv`

### Loading Coordinates
- Select "Yes" when prompted at startup
- Or click the **Load** button at any time

## Tips

1. Use Sequential mode when you want to create multiple markers without worrying about selection
2. Use 1 Line mode to trace contours or paths in a single frame
3. Automatic backups are created with timestamps to prevent data loss
4. Use zoom for greater precision when marking coordinates
5. The **A** key is useful for adding empty markers that can be filled in later

## Troubleshooting

- If the window doesn't show the video: check if the video file is in a supported format
- If markers don't appear when loading: check if the CSV file is in the correct format
- If performance is slow: reduce the window size or disable persistence mode
