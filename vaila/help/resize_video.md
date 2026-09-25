# resize_video

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\resize_video.py`
- **Lines:** 1390+
- **Size:** updated with CLI and batch workflow
- **Version:** 0.4.5
- **Author:** -------
- **GUI Interface:** ✅ Yes

## 📖 Description


resize_video.py

Description:
-----------
This script provides tools for improving pose detection in videos:
1. Batch resize videos to higher resolutions (2x-8x)
2. Crop specific regions of interest and resize them
3. Convert MediaPipe, YOLO, and vailá pixel coordinates back to original video coordinates

Version:
--------
0.4.0
Create: 27 April 2025
update: 14 September 2026

Author:
-------
Prof. PhD. Paulo R. P. Santiago

License:
--------
This code is licensed under the GNU General Public License v3.0.

Dependencies:
-------------
- Python 3.12.9
- opencv-python
- tkinter
- pandas (for coordinates conversion)


## 🔧 Main Functions

**Total functions found:** 22+

- `get_video_info`
- `resize_with_opencv`
- `convert_coordinates`
- `convert_coordinates_by_format`
- `convert_mediapipe_coordinates`
- `validate_scale_factor`
- `iter_video_files`
- `format_resize_cli_command`
- `run_resize_cli`
- `batch_resize_videos`
- `run_resize_video`
- `select_roi`
- `select_input_dir`
- `select_output_dir`
- `revert_coordinates_gui`
- `crop_and_resize_single`
- `start_batch_processing`
- `mouse_callback`
- `set_format_and_highlight`
- `update_progress`
- `select_metadata_file`
- `select_pixel_csv_file`
- `select_output_csv_file`
- `main`

## Crop-only crash fix

The **Crop and Resize** button now keeps all Tkinter work on the GUI thread. Video
encoding runs in a worker, while progress messages are delivered through a queue
and `after()`. This prevents the Linux/X11 segmentation fault (exit code 139) that
could occur immediately after confirming an ROI. The ROI is clamped to the source
video dimensions before encoding, including when the requested rectangle reaches an
edge. Output dimensions are rounded up to even pixels for codec compatibility. Scale factor `1` is valid for crop-only output; it still writes a new cropped
video and metadata JSON.

The selected coordinates are reported in the terminal and progress window. The
metadata stores the exact clamped crop used for every frame, so coordinate reversal
continues to work.

## GUI workflow

The resizer opens as a modal child of the main vailá window, so it stays in front
without creating a second Tk root. **Full Resize** processes every video in the
selected input directory. **Crop and Resize** selects one reference video and asks
whether the same ROI should be applied to all videos. **Batch Crop (same ROI)**
always applies the selected ROI to every supported video in the input directory.
Processing runs in a worker and progress is displayed in the GUI.

## CLI workflow

The module can run without a display. A file processes one video; a directory
processes all `.mp4`, `.avi`, `.mov` and `.mkv` files. The command returns code 1
when no videos are found or any item fails.

```bash
# Resize every video in a directory
python -m vaila.resize_video --input ./videos --output ./resized --scale 2

# Apply one ROI to every video (x y width height are source pixels)
python -m vaila.resize_video --input ./videos --output ./cropped \
  --scale 1 --roi 120 40 640 480

# Include nested directories
python -m vaila.resize_video -i ./videos -o ./resized --scale 2 --recursive
```

**Version:** 0.4.0
**Updated:** 24 September 2026




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
