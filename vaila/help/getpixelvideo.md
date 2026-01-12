# getpixelvideo - Pixel Coordinate Tool

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/getpixelvideo.py`
- **Version:** 0.3.3 (Updated: 11 January 2026)
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes (Pygame based)

## 📖 Description

This tool enables marking and saving pixel coordinates in video frames, with zoom functionality for precise annotations. The window can be resized dynamically, and all UI elements adjust accordingly. Users can navigate the video frames, mark points, label images for machine learning, view YOLO tracking data, and save results in CSV format.

### New Features in Version 0.3.3:
- **GUI Pygame in Linux:** Uses Pygame for the GUI on Linux systems to avoid conflicts with Tkinter, ensuring smoother operation.

### New Features in Version 0.3.2:
1.  **Labeling Mode:** A new "Labeling" button allows for labeling images in video frames specifically for Machine Learning training.
2.  **YOLO Dataset Support:** Support for YOLO dataset directory structures.
3.  **YOLO Tracking Visualization:** Visualize tracking data from CSV files (in `all_id_detection.csv` format) with bounding box overlays directly on the video.

## 🚀 How to Use

1.  **Select Video:** Choose the video file you wish to process.
2.  **Load Keypoints (Optional):** Select a keypoint file to load existing data if available.
3.  **Mark Points:** Use the mouse to mark points on the video frames. You can zoom in for better precision.
4.  **Label Images (Optional):** Use the labeling mode to creating bounding boxes for object detection datasets.
5.  **Save Results:** Save your marked coordinates or labeled data to a CSV file.

## ⌨️ Command Line Usage

```bash
python getpixelvideo.py [options]
```

### Options:
-   `-h`, `--help`: Show this help message and exit.
-   `-v`, `--version`: Show version information and exit.
-   `-f FILE`, `--file FILE`: Specify the video file to process.
-   `-k KEYPOINT`, `--keypoint KEYPOINT`: Specify the keypoint file to load.
-   `-l`, `--labeling`: Start in labeling mode to label images for ML training.
-   `-s`, `--save`: Save the results in CSV format.
-   `-p`, `--persistence`: Show persistence mode (visualize points from previous frames).
-   `-a`, `--auto`: Show auto-marking mode.
-   `-c`, `--sequential`: Show sequential mode.

## 🔧 Main Functions

-   `get_color_for_id`: Generates consistent colors for markers.
-   `play_video_with_controls`: Main function handling video playback and user interaction.
-   `pygame_file_dialog`: Native Pygame file dialog for selecting files without Tkinter conflicts.
-   `export_video_with_annotations`: Exports the video with markers and bounding boxes burned in.
-   `load_coordinates_from_file`: Loads existing coordinate data.
-   `save_coordinates`: Saves current markings to CSV.
-   `save_labeling_project`: Saves bounding box data for ML.

## 📜 License

This program is licensed under the **GNU Affero General Public License v3.0**.
For more details, visit: [https://www.gnu.org/licenses/agpl-3.0.html](https://www.gnu.org/licenses/agpl-3.0.html)

---

📅 **Last Updated:** 11 January 2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
