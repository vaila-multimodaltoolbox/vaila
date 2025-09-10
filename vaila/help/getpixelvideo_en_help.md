# User Guide - Pixel Coordinate Tool (getpixelvideo.py)

## Introduction

The Pixel Coordinate Tool (getpixelvideo.py) is a comprehensive video annotation tool that allows you to mark and save pixel coordinates in video frames. Developed by Prof. Dr. Paulo R. P. Santiago, this tool offers advanced features including zoom for precise annotations, dynamic window resizing, frame navigation, multi-format CSV support, and advanced data visualization capabilities.

**Version:** 0.0.8  
**Date:** 27 July 2025  
**Project:** vailá - Multimodal Toolbox

## Key Features

- **Multi-format Support:** Load and visualize MediaPipe, YOLO tracking, and vailá standard formats
- **Advanced Visualization:** Stick figures for MediaPipe, bounding boxes for YOLO tracking
- **Flexible Marking:** Multiple marker modes for different annotation needs
- **Zoom & Navigation:** Full zoom capabilities with frame-by-frame navigation
- **Persistence Mode:** View marker trails across multiple frames
- **Auto-detection:** Automatically detect CSV format or manual selection
- **Comprehensive Documentation:** HTML help file with detailed instructions

## Requirements

- **Python 3.12+**
- **OpenCV (cv2):** Video processing
- **Pygame:** GUI and visualization
- **Pandas:** CSV data handling
- **NumPy:** Numerical operations
- **Tkinter:** File dialogs (usually included with Python)

### Installation

```bash
pip install opencv-python pygame pandas numpy
```

## Getting Started

1. **Run the script:** `python vaila/getpixelvideo.py`
2. **Select video file:** Choose the video to process
3. **Load existing data:** Choose whether to load existing keypoints
4. **Select format:** If loading data, choose CSV format:
   - **Auto-detect (recommended):** Automatically detects the format
   - **MediaPipe format:** For landmark data with stick figure visualization
   - **YOLO tracking format:** For tracking data with bounding box visualization
   - **vailá standard format:** For standard coordinate data
5. **Navigate & annotate:** Use the interface to navigate, zoom, and edit markers
6. **Save results:** Save the annotated data in CSV format

## Interface

The tool interface consists of:
- **Video viewing area** (upper section) with zoom and pan capabilities
- **Control panel** (bottom section) with:
  - Current frame information
  - Slider for navigating between frames
  - Buttons for main functions (Load, Save, Help, 1 Line, Persistence, Sequential)
  - Format-specific visualization controls

## Supported File Formats

### MediaPipe Format
Used for pose estimation and landmark detection data.

**Format:** `frame, landmark_0_x, landmark_0_y, landmark_0_z, landmark_1_x, landmark_1_y, landmark_1_z, ...`

**Example:**
```csv
frame,landmark_0_x,landmark_0_y,landmark_0_z,landmark_1_x,landmark_1_y,landmark_1_z
0,100.5,200.3,0.0,150.2,250.1,0.0
1,105.2,205.1,0.0,155.3,255.2,0.0
```

**Visualization:** Stick figures with landmarks connected by lines

### YOLO Tracking Format
Used for object tracking and detection data.

**Format:** `Frame, Tracker ID, Label, X_min, Y_min, X_max, Y_max, Confidence, Color_R, Color_G, Color_B`

**Example:**

```csv
Frame,Tracker ID,Label,X_min,Y_min,X_max,Y_max,Confidence,Color_R,Color_G,Color_B
0,1,person,100,200,200,300,0.9,255,0,0
0,2,person,300,400,400,500,0.8,0,255,0
```

**Visualization:** Bounding boxes with labels and tracker IDs

### vailá Standard Format

Standard coordinate format used by vailá toolbox.

**Format:** `frame, p1_x, p1_y, p2_x, p2_y, p3_x, p3_y, ...`

**Example:**

```csv
frame,p1_x,p1_y,p2_x,p2_y
0,100.5,200.3,150.2,250.1
1,105.2,205.1,155.3,255.2
```

**Visualization:** Point markers with IDs

## Marking Modes

### Normal Mode (default)
- Each click selects and updates the currently selected marker
- Navigate between markers using TAB
- Each marker maintains its ID across all frames
- **Use case:** Tracking specific points across frames
- **Activation:** Default mode at startup

### 1 Line Mode (C key)
- Creates points in sequence in one frame
- Each click adds a new marker in sequential order
- **Use case:** Tracing contours, paths, or outlines
- **Activation:** Press C key to toggle
- **Behavior:** Each click creates a new sequential marker

### Sequential Mode (S key)
- Each click creates a new marker with incrementing IDs
- No need to select markers first
- Only available in Normal mode
- **Use case:** Quick annotation of multiple points
- **Activation:** Press S key to toggle
- **Behavior:** Automatic ID increment for each new marker

## Keyboard Commands

### Video Navigation
| Key | Action |
|-----|--------|
| **Space** | Play/Pause |
| **→** | Next frame (when paused) |
| **←** | Previous frame (when paused) |
| **↑** | Fast forward (when paused) |
| **↓** | Rewind (when paused) |
| **Drag Slider** | Jump to specific frame |

### Zoom & Pan
| Key | Action |
|-----|--------|
| **+** | Zoom in |
| **-** | Zoom out |
| **Mouse Wheel** | Zoom in/out |
| **Middle Click** | Enable pan/move |

### Marker Management
| Key | Action |
|-----|--------|
| **Left Click** | Add/update marker |
| **Right Click** | Remove last marker |
| **TAB** | Next marker in current frame |
| **SHIFT+TAB** | Previous marker in current frame |
| **DELETE** | Delete selected marker |
| **A** | Add new empty marker to file |
| **R** | Remove last marker from file |

### Mode Controls
| Key | Action |
|-----|--------|
| **C** | Toggle "1 Line" mode |
| **S** | Toggle Sequential mode (Normal mode only) |
| **P** | Toggle Persistence mode |
| **1** | Decrease persistence frames |
| **2** | Increase persistence frames |
| **3** | Toggle full persistence |

### File Operations
| Key | Action |
|-----|--------|
| **S** | Save current markers |
| **B** | Make backup of current data |
| **L** | Reload coordinates from file |
| **H** | Show help dialog |
| **D** | Open full documentation (in help) |

### Other
| Key | Action |
|-----|--------|
| **ESC** | Save and exit |

## Data Visualization

### MediaPipe Visualization
When loading MediaPipe data, the tool displays stick figures:
- **Landmarks:** Red dots for each detected point
- **Connections:** Green lines connecting related landmarks
- **Pose Structure:** Full body pose with head, arms, torso, and legs

### YOLO Tracking Visualization
When loading YOLO tracking data, the tool displays bounding boxes:
- **Bounding Boxes:** Colored rectangles around detected objects
- **Labels:** Object class and tracker ID displayed
- **Colors:** Unique colors for different tracker IDs
- **Confidence:** Detection confidence values shown

### vailá Standard Visualization
Standard point marker visualization:
- **Markers:** Green circles for each point
- **Numbers:** Marker IDs displayed next to points
- **Selection:** Orange highlight for selected marker

## Persistence Mode

Persistence mode shows markers from previous frames, creating a visual "trail":
- **P:** Enables/disables persistence
- **1:** Decreases the number of displayed frames
- **2:** Increases the number of displayed frames
- **3:** Toggles between persistence modes (disabled → full → limited)

**Features:**
- Fading trails show marker movement
- Configurable number of frames to display
- Visual feedback for marker trajectories

## Saving and Loading

### Saving Options

#### Standard Save (S key)
- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers.csv`
- **Location:** Same directory as video file

#### 1 Line Save
- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers_sequential.csv`
- **Use:** For path tracing and contour data

#### Sequential Save
- **Format:** `frame, p1_x, p1_y, p2_x, p2_y, ...`
- **File:** `{video_name}_markers_sequential.csv`
- **Use:** For multiple point annotations

### Loading Coordinates
- Select "Yes" when prompted at startup
- Or click the **Load** button at any time
- **Auto-detection:** Automatically detects CSV format
- **Manual selection:** Choose format manually if needed

## Advanced Features

### Auto-detection
The tool automatically detects CSV format based on column structure:
- **MediaPipe:** Detects 'landmark' in column names
- **YOLO:** Detects 'Frame', 'Tracker ID', 'X_min' columns
- **vailá:** Detects 'frame' and 'p' pattern in columns

### Backup and Recovery
Built-in backup system for data safety:

- **Backup:** Press B to create backup
- **Reload:** Press L to reload from file
- **Auto-backup:** Automatic backups before major operations

### Documentation Access
- **Quick Help:** Press H for in-app help
- **Full Documentation:** Press D in help dialog to open HTML documentation
- **HTML Documentation:** Located at `vaila/help/getpixelvideo_help.html`

## Tips and Best Practices

1. **Use Sequential mode** when you want to create multiple markers without worrying about selection
2. **Use 1 Line mode** to trace contours or paths in a single frame
3. **Automatic backups** are created with timestamps to prevent data loss
4. **Use zoom** for greater precision when marking coordinates
5. **The A key** is useful for adding empty markers that can be filled in later
6. **Persistence mode** is great for visualizing movement patterns
7. **Auto-detection** works best with properly formatted CSV files
8. **Backup regularly** using the B key to prevent data loss

## Troubleshooting

### Common Issues

#### Video Not Loading
- Check if video file is corrupted
- Ensure video format is supported (MP4, AVI, MOV, MKV)
- Verify file path doesn't contain special characters

#### CSV Format Not Detected
- Check CSV file structure matches expected format
- Use manual format selection if auto-detection fails
- Verify CSV file is not corrupted

#### Performance Issues
- Reduce video resolution for better performance
- Close other applications to free memory
- Use smaller zoom levels for large videos

#### Visualization Problems
- Ensure CSV data is in the correct format
- Check if coordinate values are within video dimensions
- Verify landmark connections for MediaPipe data

### Error Messages

| Error | Solution |
|-------|----------|
| "Error opening video file" | Check video format and file integrity |
| "No keypoint file selected" | Select a valid CSV file or start fresh |
| "Unknown format" | Use manual format selection |
| "Error loading coordinates" | Check CSV file format and structure |

## Support and Documentation

- **In-app Help:** Press H for quick help
- **Full Documentation:** Press D in help dialog for complete HTML documentation
- **HTML Documentation:** `vaila/help/getpixelvideo_help.html`
- **Project Repository:** https://github.com/paulopreto/vaila-multimodaltoolbox

## Version History

### Version 0.0.8 (27 July 2025)
- Added support for multiple CSV formats (MediaPipe, YOLO tracking, vailá standard)
- Implemented auto-detection of CSV format
- Added visualization for MediaPipe stick figures and YOLO bounding boxes
- Created comprehensive HTML documentation
- Added quick access to full documentation via 'D' key
- Improved help dialog with format information
- Enhanced error handling and user feedback

### Previous Versions
- Version 0.0.7: Basic functionality with zoom and marker modes
- Version 0.0.6: Initial implementation with video navigation
