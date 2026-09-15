# Resize Video

The **Resize Video** button (`C_B_r4_c3`) opens a modal child window of vailá, so
the controls remain in front of the main window. Choose an input and output
directory, then use **Full Resize** for every video or **Batch Crop (same ROI)**
to select one region and apply it to all supported videos. **Crop and Resize**
asks whether the selected ROI should be reused for the whole directory.

The same processing is available without a display:

```bash
python -m vaila.resize_video --input ./videos --output ./resized --scale 2
python -m vaila.resize_video --input ./videos --output ./cropped --scale 1 --roi 120 40 640 480
```

Supported extensions are `.mp4`, `.avi`, `.mov` and `.mkv`. Crop metadata is
written beside each output video so MediaPipe, YOLO and vailá coordinates can be
converted back to the source video.

**Version:** 0.4.0  
**Updated:** 14 September 2026
