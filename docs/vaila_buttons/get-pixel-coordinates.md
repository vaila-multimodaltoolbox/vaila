# getpixelvideo.py — Get Pixel Coordinates

**Button:** Frame C → Video and Image → **GetPixelCoord**  
**Module:** [`vaila/getpixelvideo.py`](../../vaila/getpixelvideo.py)  
**Help:** [`vaila/help/getpixelvideo.md`](../../vaila/help/getpixelvideo.md) · [HTML](../../vaila/help/getpixelvideo.html)  
**Version:** 0.4.19 · **Updated:** 16 November 2026

## Overview

Mark and save pixel coordinates on video frames or PNG sequences (zoom, pan,
templates, FIFA pitch, ML dataset export).

## Opening media (v0.3.105)

One file picker — **no** video-vs-PNG question. Type is auto-detected:

- Video file → video
- Lone PNG → single image
- Any PNG in a folder with multiple PNGs → PNG sequence
- Directory (CLI) → PNG sequence

```bash
uv run vaila/getpixelvideo.py
uv run vaila/getpixelvideo.py -f VIDEO.mp4
uv run vaila/getpixelvideo.py -d /path/to/png_frames
```

## Calibration clicks for DLT2D / DLT3D (1-line mode)

Press **C** (`Mode: 1-line`) and **Ctrl+I** (INSERT), then click the calibration
points in the order of the `.ref2d` / `.ref3d` file. Right click removes the last
point. When each position of the calibration object is a separate video, press
**F8** (**Add video**) to open another one: the points go with you and the numbering continues.
**Save** writes `<video>_markers_1_line.csv` (the pixel file for `dlt2d.py` /
`dlt3d.py`) and a `.toml` with the video and frame of every point.

```bash
uv run vaila/getpixelvideo.py -f POS2.mp4 --one-line POS1_markers_1_line.csv
```

## REF3D builder (button REF3D, Shift+R)

The measures come from `ref3d_template.toml` in the folder of the videos: in the first REF3D dialog **N** writes it (pole, box and triad sections) and opens it in the text editor, **T** (the default, Enter) loads it; options 1/2/3 still type them. CLI: `python -m vaila.ref3d_builder --new-template FILE` / `--template-file FILE -o calib.ref3d`.

Illustrated step-by-step guide with schematic drawings: `vaila/help/ref3d_guide.html` (PT: `ref3d_guide_pt.html`); in the window, **F1** with REF3D on or the **Guide** button of the REF3D panel opens it.

Builds the `.ref3d` from the measures of a box, a pole with marks, or a triad
(X side, Y forward, Z up arms), stood at the corners of a rectangle or at listed
positions. A side panel lists the points and draws the object in 3D; click them in
order. From 6 points the DLT3D is solved at every click and the panel shows the
error and fx / fy (a real camera has fx ≈ fy). Save writes the pixel CSV, the
`.ref3d`, the template `.toml` and the `.dlt3d`. Help: [`vaila/help/ref3d_builder.md`](../../vaila/help/ref3d_builder.md).

```bash
uv run --no-sync python -m vaila.ref3d_builder --template pole --marks 0,1 --positions "rect 1.38 x 5" -o calib.ref3d
```

## Help inside the window

**H** opens the keys and buttons, starting with the steps for the mode you are in
(English / Portuguese with **L**). Hovering a toolbar button shows what it does and
its key. **Open** (F8) opens another video in the same window.

See the full help page for key bindings, Load Track CSV, FIFA mode, and ML export.
