# REF3D builder (ref3d_builder.py)

**Version:** 0.4.19
**Date:** 16 November 2026
**Updated:** 18 November 2026
**Authors:** Prof. Dr. Paulo R. P. Santiago, Rafael L. M. Monteiro
**Project:** *vailá* - Multimodal Toolbox

## What it does

A DLT3D calibration (`dlt3d.py`) needs two files with the same points `p0, p1, ...`:
the 3D coordinates in metres (`.ref3d`) and their pixels in the video (a 1-line CSV).
The REF3D builder writes the `.ref3d` from the measures of the calibration object,
so you do not type every coordinate, and solves and checks the DLT3D from your clicks.

In *vailá* it is the **REF3D** button of `getpixelvideo` (Shift+R, or option 4 of
CALIB): see the getpixelvideo help, section *REF3D*. This page describes the
templates, the files and the command line.

Step by step with drawings: [illustrated REF3D calibration guide](ref3d_guide.md)
([português](ref3d_guide_pt.md)).

## Templates

Axes: **X** sideways, **Y** forward, **Z** up, right-handed (**X × Y = Z**); metres.

| Template | You give | Points, in this order |
| --- | --- | --- |
| **Box** | size `X x Y x Z`; optional marks along X, Y, Z | corners and the points of the edges at the marks; bottom layer first (Z), then Y, then X |
| **Pole** | heights of the marks (`0, 0.5, 1`); positions | for each position, the marks from the bottom up |
| **Triad** | marks of the X, Y and Z arms (distances from the origin); positions | for each position: origin, X marks, Y marks, Z marks |

Positions (pole and triad): `rect X x Y` = the 4 corners of a rectangle, in the order
`0,0 / X,0 / 0,Y / X,Y`; or a list `x,y; x,y; ...` in the order you will move the
object. Empty = only at `0,0`.

Example: a 1 m pole with marks at 0 and 1 m, stood at the corners of a walkway
1.38 m wide and 5 m long, is `marks 0, 1` and `rect 1.38 x 5`: 8 points. **N** writes a
richer blank template by default (`marks = [0, 0.25, 0.5, 0.75, 1]`, `rect = [1.4, 5]`)
with comments that explain every key; trim it to what you measured.

Optional `[videos]` table (pole and triad only): `"file.mp4" = [x, y]` names the floor
position of each calibration clip. getpixelvideo then routes clicks in that video to that
corner's points. The `[x, y]` must be one of the template positions.

All points on one plane (a pole with one mark, a triad with no Z arm and one
position) give no DLT3D: the builder says so.

## The check after the fit

From 6 points off one plane, the 11 DLT3D parameters are solved by least squares
(the same equations as `dlt3d.py`) and the camera is decomposed. A real camera has
square pixels: **fx ≈ fy**. When a measure of the object is wrong, the DLT still
reproduces the clicks to about 1 px, but fx and fy drift apart: the builder reports
**NOT A REAL CAMERA** (limits as in `monocular_dlt_align.py`: |fy/fx − 1| ≤ 0.25,
small skew, mean error ≤ 10 px). Check the measures with a tape before using the
calibration. With few points (6) there is no redundancy and errors hide in the camera;
use 8 to 12 points spread over the volume.

## Files

Saved next to the video on screen when you press Save in getpixelvideo:

| File | Content |
| --- | --- |
| `<video>_markers_1_line.csv` | pixels, one row `frame,p0_x,p0_y,...` (blank = not visible) |
| `<video>_markers_1_line.toml` | video and frame where each point was clicked |
| `<video>_markers_1_line.ref3d` | 3D points, one row `frame,p0_x,p0_y,p0_z,...` (read by `dlt3d.py`) |
| `ref3d_template.toml` (folder of the videos) | the measures filled in by hand: REF3D **N** writes it, **T** loads it |
| `<video>_markers_1_line.template.toml` | the measures used, in the same format (can be edited and loaded) |
| `<video>_markers_1_line.dlt3d` | the 11 coefficients, in `dlt3d.py`'s format; same format and equations as the file **Make DLT3D** writes, solved from the exact clicks (the pixel CSV keeps whole pixels, so Make DLT3D would differ by under 1 px); it does not need to be run again |

Files that already exist are copied to `.vaila_markers_history/` before being replaced.

## Command line

```bash
# A template TOML to fill in (pole, box and triad sections; --template picks the kind),
# then the .ref3d from it. Neither overwrites a file.
uv run --no-sync python -m vaila.ref3d_builder --new-template ref3d_template.toml
uv run --no-sync python -m vaila.ref3d_builder --template-file ref3d_template.toml \
    -o calib.ref3d

# Write a .ref3d from measures given on the command line; refuses to overwrite.
uv run --no-sync python -m vaila.ref3d_builder --template pole --marks 0,1 \
    --positions "rect 1.38 x 5" -o calib.ref3d
uv run --no-sync python -m vaila.ref3d_builder --template box --size "1.38 x 5 x 1" \
    --div-y 1,2,3,4 -o volume.ref3d
uv run --no-sync python -m vaila.ref3d_builder --template triad --arms-x 0.5 \
    --arms-y 0.5,1 --arms-z 0.5,1 --positions "rect 2 x 4" -o triad.ref3d

# Solve and check the DLT3D of a .ref3d and its pixel CSV (exit 2 = not a real camera).
uv run --no-sync python -m vaila.ref3d_builder --ref3d calib.ref3d \
    --pixels calib_markers_1_line.csv -o calib.dlt3d
```

## Limits

- The measures come from your tape; the check only shows when they disagree with the image.
- Lens distortion is not modelled by the 11-parameter DLT3D.
