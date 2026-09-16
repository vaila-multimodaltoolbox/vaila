# Crop Face — via Markerless 2D Chooser

> **v0.4.3:** this used to be its own button (`C_B_r1_c2`, text "Crop Face")
> under Video and Image. It is now under the **"Other 2D tools"** section of
> the **Markerless 2D** coringa chooser (`B1_r1_c4`, method
> `markerless_2d_analysis`) — the underlying handler and script
> (`vaila/crop_faces_atletas.py`) are unchanged. Slot `C_B_r1_c2` is now
> **Planar Geo**.

Batch-crops athlete photos into square 5×5 cm JPEGs at 300 DPI using MediaPipe
Face Detector.

## Usage

1. Click **Markerless 2D** in Frame B, then **Crop Face** in "Other 2D tools".
2. Select the input photo directory, then the output directory.
3. On first use the BlazeFace model downloads into
   `vaila/models/crop_face/face_detector.task` (or pick a `.task` / `.tflite`
   if download fails).

```bash
uv run python vaila/crop_faces_atletas.py
uv run python vaila/crop_faces_atletas.py --input /path/photos --output /path/out
uv run python vaila/crop_faces_atletas.py --download-model
```

[Full module help](../../vaila/help/crop_faces_atletas.md).
