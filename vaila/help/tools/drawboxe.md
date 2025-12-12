# drawboxe

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/drawboxe.py`
- **Lines:** 1390
- **Version:** 0.0.7
- **Author:** Paulo Roberto Pereira Santiago
- **GUI Interface:** ✅ Yes
- **Email:** paulosantiago@usp.br
- **GitHub:** [vaila-multimodaltoolbox/vaila](https://github.com/vaila-multimodaltoolbox/vaila)

## 📖 Description

**drawboxe** is a tool for drawing boxes and polygons on videos. It allows you to create rectangular, trapezoidal, and free-form polygons to process videos, filling specific areas or excluding areas from processing.

### Main features:

- Drawing rectangular, trapezoidal, and free-form polygons
- "Inside" mode (fills selected area) and "Outside" mode (fills everything except the area)
- Custom color selection via RGB sliders or color picker
- Batch processing support for multiple videos
- Frame intervals (process only specific frames)
- Save and load configurations in TOML files

## 🎯 How to Use

### 1. Starting drawboxe

Run the script from the command line or through the vailá graphical interface:

```bash
python drawboxe.py
```

Or through vailá: **Tools → Video Processing → DrawBoxe**

### 2. Selecting Video Directory

When started, drawboxe will ask you to select a directory containing the videos to be processed.

**📁 Supported formats:** MP4, AVI, MOV, MKV

### 3. Drawing Shapes

The first frame of the first video will be displayed for you to draw shapes:

#### 🔷 Shape Types:

| Shape | How to Draw | Clicks Required |
|-------|-------------|-----------------|
| **Rectangle** | Click on two opposite corners | 2 clicks |
| **Trapezoid** | Click on the 4 vertices of the trapezoid | 4 clicks |
| **Free** (Free Polygon) | Click on multiple points, then press 'z' to close | 3+ clicks + 'z' key |

#### 📋 Fill Modes:

- **Inside:** Fills only the selected area with the chosen color
- **Outside:** Fills everything except the selected area (indicated by hatching ///)

## ⌨️ Keyboard Shortcuts

| Key | Action |
|-----|-------|
| `e` | Toggle mode (Inside/Outside) |
| `t` | Toggle shape (Rectangle/Trapezoid/Free) |
| `c` | Activate color picker (select color from image) |
| `z` | Finish free polygon (close current polygon) |
| `d` | Complete current polygon (move to next) |
| `r` | Clear all shapes |
| `w` | Save configuration (TOML file) |
| `q` | Load configuration (TOML file) |
| `a` | Abort script (exit without processing) |
| `Enter` | Save and exit (process videos) |

## 🖱️ Mouse Actions

- **Left Click:** Add point (shape definition)
- **Right Click:** Undo last point or remove last shape
- **Drag (with Color Picker active):** Select area to extract color

## 🎨 Color Selection

You can choose colors in two ways:

1. **RGB Sliders:** Use the sliders on the right panel to adjust Red, Green, and Blue values (0-255)
2. **Color Picker:** Click the "Pick Color" button or press 'c', then drag on the image to select an area and extract the average color

💡 **Tip:** Color preview is displayed in the "Color Preview" panel

## 💾 TOML Configuration Files

### File Structure

drawboxe automatically saves the configuration in a `[video]_dbox.toml` file in the same directory as the video.

### TOML File Example

```toml
# ================================================================
# DrawBoxe Configuration File
# Generated automatically by drawboxe.py in vaila Multimodal Analysis Toolbox
# Created: 2025-01-15 10:30:00
# ================================================================

[description]
title = "DrawBoxe Configuration File"
version = "0.0.7"

[video_info]
basename = "my_video"
original_path = "/path/to/my_video.mp4"

# ================================================================
# FRAME INTERVALS (OPTIONAL)
# ================================================================
# Define which frame ranges to process. If not specified, all frames will be processed.
# Format: start_frame, end_frame (inclusive, 0-indexed)
# 
# Example: To process frames 100-200 and 500-600, uncomment and modify:
# [[frame_intervals]]
# id = 1
# start_frame = 100
# end_frame = 200
#
# [[frame_intervals]]
# id = 2
# start_frame = 500
# end_frame = 600
# ================================================================

# Example of frame intervals (uncomment and modify as needed):
# [[frame_intervals]]
# id = 1
# start_frame = 0
# end_frame = 100

# [[frame_intervals]]
# id = 2
# start_frame = 200
# end_frame = 300

# ================================================================
# BOX 1: RECTANGLE - INSIDE MODE
# ================================================================
[[boxes]]
id = 1
mode = "inside"
shape = "rectangle"
color_r = 1.000000
color_g = 0.000000
color_b = 0.000000
coord_1_x = 100.500000
coord_1_y = 150.300000
coord_2_x = 300.200000
coord_2_y = 150.300000
coord_3_x = 300.200000
coord_3_y = 250.800000
coord_4_x = 100.500000
coord_4_y = 250.800000
```

### 📝 How to Define Frame Intervals in TOML

**Step by step:**

1. Open the `[video]_dbox.toml` file in a text editor
2. Locate the `# FRAME INTERVALS (OPTIONAL)` section
3. Uncomment (remove the `#`) from the frame intervals example lines
4. Modify the `start_frame` and `end_frame` values as needed
5. You can add multiple intervals, each with a different `id`
6. Save the file
7. When processing, drawboxe will ask if you want to use the intervals from the TOML

⚠️ **Important:**
- Frames are indexed starting from 0 (first frame = 0)
- Intervals are inclusive (start_frame and end_frame are processed)
- If you don't specify frame intervals, all frames will be processed
- You can have multiple intervals in the same file

### Practical Example of Frame Intervals

To process only frames 0-100, 200-300, and 500-600:

```toml
[[frame_intervals]]
id = 1
start_frame = 0
end_frame = 100

[[frame_intervals]]
id = 2
start_frame = 200
end_frame = 300

[[frame_intervals]]
id = 3
start_frame = 500
end_frame = 600
```

## 📤 Video Processing

After drawing shapes and pressing `Enter`:

1. drawboxe will ask if you want to use frame intervals from a .txt file or from the TOML
2. If you already have a TOML file with frame intervals, it will be automatically detected
3. Videos will be processed and saved in a `video_2_drawbox_[timestamp]` directory
4. Processed videos will have the suffix `_dbox.mp4`

## 🔧 Main Functions

**Total functions: 20**

- `save_first_frame` - Saves the first frame of the video
- `extract_frames` - Extracts all frames from the video
- `apply_boxes_directly_to_video` - Applies boxes directly to the video
- `apply_boxes_to_frames` - Applies boxes to specific frames
- `reassemble_video` - Reassembles the video from frames
- `clean_up` - Cleans up temporary files
- `save_config_toml` - Saves configuration to TOML
- `load_config_toml` - Loads configuration from TOML
- `get_box_coordinates` - Graphical interface for drawing boxes
- `load_frame_intervals` - Loads frame intervals from .txt file
- `run_drawboxe` - Main execution function

## ⚠️ Warnings and Limitations

- Clicks outside the image are ignored
- Free polygons need at least 3 points
- Processing large videos may take time
- Make sure you have enough disk space for processed videos
- FFmpeg must be installed on the system for frame extraction

## 📚 Requirements

- Python 3.x
- OpenCV (cv2)
- Matplotlib
- NumPy
- Tkinter (usually included with Python)
- TOML (for configuration files)
- FFmpeg (for video processing)

## 📝 Version History

- **v0.0.7:** Added hatching to indicate "outside" mode
- **v0.0.6:** Support for free polygons
- **v0.0.5:** Support for trapezoidal boxes
- **v0.0.4:** Support for rectangular boxes
- **v0.0.3:** Support for frame intervals
- **v0.0.2:** Support for multiple videos
- **v0.0.1:** Initial version

---

📅 **Generated automatically on:** January 15, 2025  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)  
📧 **Contact:** paulosantiago@usp.br
