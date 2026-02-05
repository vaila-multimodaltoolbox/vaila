# MP Angles - MediaPipe Angle Calculation

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila/mpangles.py`
- **Version:** 0.0.3
- **Author:** Paulo R. P. Santiago
- **Email:** paulosantiago@usp.br
- **Creation Date:** 31 March 2025
- **Update Date:** 5 February 2026
- **Python Version:** 3.12.12
- **GUI Interface:** ✅ Yes
- **License:** AGPL v3.0

## 📖 Description

The **MP Angles** module calculates absolute and relative joint angles from landmark coordinates obtained from MediaPipe pose estimation. It processes CSV files containing MediaPipe landmark data (33 landmarks) and generates angle CSVs, optional annotated video with skeleton overlay, and an HTML report.

### Key Features

1. **Absolute Angles**
   - Calculates segment angles relative to the horizontal axis.
   - **Left side in video:** -180° to +180°.
   - **Right side in video:** 0° to 360°.
   - All segments: trunk, neck, thigh, shank, foot, upper arm, forearm, hand (both sides).

2. **Relative Angles**
   - Joint angles between connected segments (shoulder, elbow, wrist, hip, knee, ankle, neck, trunk).
   - **Neck (relative)** is shown at the **top center** of the video (neck with trunk).

3. **Smoothing (optional)**
   - **Butterworth low-pass filter** to smooth landmark trajectories.
   - Section title: *Smoothing (optional)*. Checkbox: *Apply Butterworth low-pass filter (smooth joint trajectories)*.
   - Hint: *Use when data is noisy; leave unchecked for raw angles.*
   - Parameters: Cutoff (Hz), Order, FPS — always editable; filter is applied only when the checkbox is enabled at run time.

4. **Input: CSV or Directory only**
   - **CSV:** Select a landmark CSV file. When you run analysis, you are asked to select the **video** corresponding to that CSV; the module then generates skeleton + angles overlay on that video and the report.
   - **Directory:** Batch mode. The module looks for **videos** in the folder and, for each video, finds a matching landmark CSV by name (prefers `*_pixel_vaila.csv`). It runs visualization for each pair and creates a **batch_analysis_report.html** in the directory with links to each output folder and report.

5. **HTML Report**
   - **Angles:** Table preview (first rows of relative angles CSV). Download links for all CSV files.
   - **Pose sequence (stick figures):** PNG with colored stick figures by frame (blue left, red right, gray center, green joints) and angle values at joints — no embedded video in the report.
   - For **directory** runs: **batch_analysis_report.html** lists each processed video with links to its report and output video.

6. **Video overlay (when video is used)**
   - Skeleton and angle values drawn on the video. Angle panels in **corners:** top-left/right = Relative, bottom-left/right = Absolute (left -180..180, right 0..360). Angle values at joints on the figure.

## 🔧 Main Functions

- **`main()`** — Entry point (CLI or GUI).
- **`process_angles(input_csv, output_csv, filter_config=None, video_dims=None)`** — Computes all angles (Rel, Abs180, Abs360); returns DataFrames.
- **`process_video_with_visualization(video_path, csv_path=None, output_dir=None, filter_config=None)`** — Runs angle calculation, draws skeleton + angles on video, generates report and stick figure PNG; returns output_dir.
- **`process_directory_videos(directory_path, filter_config=None)`** — Batch: finds videos, matches CSVs by name, runs visualization for each.
- **`draw_skeleton_enhanced(frame, landmarks, rel_angles, abs_angles, abs_angles_360=None)`** — Draws skeleton and angle panels (left abs -180..180, right abs 0..360 when abs_angles_360 given).
- **`plot_stick_sequence_mpangles(df, output_path, num_frames=8, rel_df=None, abs_df=None)`** — Generates colored stick-figure sequence PNG for the report.
- **`generate_html_report(output_dir, video_path, csv_paths, ...)`** — Creates analysis_report.html with angles table and stick figure image.
- **`generate_batch_html_report(directory_path, results_with_dirs)`** — Creates batch_analysis_report.html in the directory.

## 📊 Supported Angles

### Relative (joint angles)

Neck, trunk, shoulder, elbow, wrist, hip, knee, ankle (left and right where applicable).

### Absolute (segment angles)

Trunk, neck, thigh, shank, foot, upper arm, forearm, hand (left and right). Output in both -180..180 and 0..360 formats.

## 📥 Input Requirements

### CSV format

- **First column:** Frame index.
- **Next 66 columns:** 33 landmarks × (x, y), e.g. `p1_x, p1_y, p2_x, p2_y, ...` (or equivalent).
- **Minimum:** 67 columns total. Files with fewer columns (e.g. angle outputs) are ignored in directory mode.

**Preferred format:** `*_pixel_vaila.csv` (vaila format: frame + p1_x..p33_y). The module prefers this when multiple CSVs match the same video name.

### Directory mode

- Folder may contain videos (e.g. `.mp4`, `.avi`, `.mov`, `.mkv`, `.webm`) and landmark CSVs.
- For each video, a CSV with matching base name (e.g. `video_pixel_vaila.csv` for `video.mp4`) is used. Angle output files (`processed_*`, `*_rel.csv`, `*_abs_*.csv`) are automatically skipped.

## 📤 Output Files

For each processed input:

1. **`*_rel.csv`** — Relative angles.
2. **`*_abs_180.csv`** — Absolute angles (-180° to +180°).
3. **`*_abs_360.csv`** — Absolute angles (0° to 360°).

When a video is used (CSV + video or directory batch):

- **`angles_{videoname}.mp4`** — Annotated video with skeleton and angles.
- **`{basename}_stick_sequence.png`** — Stick figure sequence (colored, with angle labels).
- **`analysis_report.html`** — Report with angles table and stick figure PNG (no embedded video).

For **directory** runs:

- **`batch_analysis_report.html`** — In the selected directory, with links to each video’s output folder, report, and video.

## 🚀 Usage

### GUI

1. Launch the module. Configuration window opens.
2. **Input:** Choose **CSV** or **Dir** (browse to file or folder).
3. **Smoothing (optional):** Check *Apply Butterworth low-pass filter...* if you want smoothing; set Cutoff (Hz), Order, FPS as needed (fields always editable).
4. Click **RUN ANALYSIS**.  
   - If you chose a CSV, you will be asked to select the **video** for that CSV; then overlay and report are generated.  
   - If you chose a directory, videos are processed in batch and the batch report is created in that directory.

### CLI

```bash
# Directory (batch): finds videos and matching CSVs, overlay + report for each
python -m vaila.mpangles -i /path/to/dir

# CSV only: angle CSVs only (no video, no report)
python -m vaila.mpangles -i /path/to/landmarks.csv

# CSV + video: overlay and HTML report
python -m vaila.mpangles -i /path/to/landmarks.csv -v /path/to/video.mp4

# Everything together: CSV + video + Butterworth filter
python -m vaila.mpangles -i /path/to/landmarks.csv -v /path/to/video.mp4 --filter --cutoff 6.0 --order 4 --fps 30

# Directory with filter
python -m vaila.mpangles -i /path/to/dir --filter --cutoff 6.0 --order 4 --fps 30
```

**Arguments:** `-i/--input` (CSV or directory), `-v/--video` (video path; use with CSV for overlay and report), `-o/--output` (optional; default: folder `angles_video_<timestamp>` next to the video), `--filter`, `--cutoff`, `--order`, `--fps`.

With **CSV + video** (with or without `--filter`), output is written to a new folder containing the three angle CSVs, `angles_<videoname>.mp4`, the stick-figure PNG, and `analysis_report.html`.

### Programmatic

```python
from vaila.mpangles import process_video_with_visualization, process_angles

filter_config = {'enabled': True, 'cutoff': 6.0, 'order': 4, 'fps': 30}
process_video_with_visualization("video.mp4", csv_path="video_pixel_vaila.csv", filter_config=filter_config)
```

## ⚙️ Configuration

- **Filter:** Cutoff (Hz), Order, FPS. Only used when the smoothing checkbox is enabled.
- **Angle formats:** Relative and both absolute formats are always produced. In the video overlay, left absolute uses -180..180 and right absolute uses 0..360.

## 📐 Angle Definitions

- **Relative:** Angles between adjacent segments (e.g. elbow = upper arm vs forearm).
- **Absolute:** Angle of each segment with respect to the horizontal (e.g. thigh, shank, trunk, neck, upper arm, forearm, hand).

## ⚠️ Requirements

- **Python:** 3.12.12 or compatible.
- **Libraries:** pandas, numpy, opencv-python, scipy, rich; tkinter (GUI). Optional: matplotlib (for stick figure sequence in report).

## 🐛 Troubleshooting

- **No CSV/videos in directory:** Ensure the folder contains either videos with matching `*_pixel_vaila.csv` (or similar) or valid landmark CSVs. Angle output files are skipped.
- **CSV has wrong number of columns:** Input must have at least 67 columns (frame + 66 coordinates). Angle CSVs (few columns) are rejected with a clear message.
- **File dialog doesn’t show videos:** Use “All files” or the specific video format in the dialog when selecting the video for a CSV.

## 🔗 Related Documentation

- **[MP Angles Button Documentation](../../docs/vaila_buttons/mp-angles-calculation.md)** — GUI button
- **[Markerless 2D Analysis Help](markerless_2d_analysis.md)** — Typical input source
- **[vailá Main Documentation](../../docs/index.md)** — Toolbox overview

## 📄 License

AGPL v3.0.

---

**Last Updated:** February 2026 · Part of vailá - Multimodal Toolbox
