# C3D Metadata Editor & Creator (`c3d_metadata.py`)

## Module information

- **Category:** Data Files
- **Version:** 0.3.131
- **Updated:** 2026-09-09
- **GUI:** Frame C → Data Files → **C3D Metadata** (`C_A_r2_c2`), also accessible from the **C3D <--> CSV** (`C_A_r1_c2`) menu dialog
- **CLI:** Yes (`uv run vaila/c3d_metadata.py` or `uv run vaila.py --metadata`)

## Purpose

C3D is the biomechanics standard binary format for motion capture marker coordinates and synchronized analog signals (force plates, EMG).
The `c3d_metadata.py` module provides a comprehensive tool to:

1. **Inspect C3D Metadata:** View manufacturer details, point frame rate (FPS), point coordinates units, number of markers, frame counts, analog channels, analog sampling rate, subframe ratio, force platforms, and events.
2. **Edit Existing C3D Files:** Modify point frequency, coordinates units (with automatic coordinate scaling between `mm` and `m`), analog sampling rates, manufacturer name, and software tag.
3. **Synchronize Analog Invariants:** Automatically keeps `ANALOG:RATE = POINT:RATE * subframe_ratio`, preventing frame count mismatch errors when saving.
4. **Sanitize Character Encoding:** Cleans Latin-1 / non-ASCII units (such as `mm/s²` or `\udcb2`) to UTF-8 before passing to C++ `ezc3d` SWIG bindings, preventing write crashes.
5. **Create New C3D Template Files:** Generate clean, valid C3D files with user-defined frames, markers, analog channels, and sample rates for testing or simulation.

Source files are never overwritten in place unless explicitly targeted.

## GUI

Click **C3D Metadata** in Frame C (`C_A_r2_c2`) or select **C3D Metadata (Edit/Create)** inside the **C3D <--> CSV** chooser:

- **Browse File / Directory:** Load any `.c3d` file to inspect its full parameter metadata and header values.
- **Inspect Mode:** View all markers, analogs, manufacturer, and timing information in an organized, scrollable text viewer.
- **Edit Parameters:** Set new FPS, change units (`mm` ↔ `m`) with optional coordinate scaling, adjust analog frequency, and set manufacturer / software tags.
- **Save As:** Write out a verified, binary-compliant `.c3d` file.
- **Create New C3D Template:** Dialog tab to generate a synthetic C3D file from scratch with custom dimensions.

## CLI

```bash
# Launch the interactive GUI
uv run vaila/c3d_metadata.py
# or through vaila.py:
uv run vaila.py --metadata

# Inspect C3D metadata headlessly
uv run vaila/c3d_metadata.py --show /path/to/trial.c3d

# Modify C3D parameters
uv run vaila/c3d_metadata.py -i input.c3d -o output.c3d --fps 120.0 --point-units m --scale-coords --manufacturer "Qualisys"

# Create a new template C3D file
uv run vaila/c3d_metadata.py --create -o template.c3d --fps 100 --frames 200 --markers 10 --analogs 8
```
