# Edit CSV/C3D (`edit_csv_c3d.py`)

## Module information

- **Category:** Data Files
- **Version:** 0.3.111
- **Updated:** 2026-08-24
- **GUI:** Frame C → Data Files → **Edit CSV/C3D** (`C_A_r1_c1`)
- **CLI:** Yes

## Purpose

Frame C button `C_A_r1_c1` (previously "Edit CSV") opens a directory that
holds `.csv` and/or `.c3d` files and applies the same column-editing tools
from `rearrange_data.py`'s `ColumnReorderGUI` to both:

- `.csv` files go straight into the editor, exactly like the old "Edit CSV"
  button.
- `.c3d` files are converted to a marker CSV first (headless, via
  `readc3d_export.c3d_markers_to_dataframe`), edited alongside the CSVs,
  then converted back to `.c3d` (via `readcsv_export.
  auto_create_c3d_from_csv`), preserving **POINT RATE**, **ANALOG RATE**,
  **POINT UNITS**, analog channels, and occlusion residual flags (a `NaN`
  sample in the CSV round-trips back to a negative C3D residual — it is
  never silently parked at the world origin).

Source files are never overwritten. Every run writes into a fresh
`processed_edit_csv_c3d_YYYYMMDD_HHMMSS/` directory.

The **C3D <--> CSV** button (`C_A_r1_c2`, the `readc3d_export.py` /
`readcsv_export.py` batch converters) is a separate tool and is not
replaced by this module — use it for one-off full C3D↔CSV exports without
the column editor.

## GUI

Click **Edit CSV/C3D**, pick a directory containing `.csv` and/or `.c3d`
files. Hidden files (dotfiles) are ignored. The same `ColumnReorderGUI`
editor used by the old "Edit CSV" button opens on the combined set —
`.c3d` files appear as their staged marker CSV (`Time, LABEL_X, LABEL_Y,
LABEL_Z, ...`). Edit and close the editor (`Esc` for Save & Exit, or
`Ctrl+S` for an intermediate save) as usual. After the editor closes:

- Files that started as `.csv` are written to `<output>/<stem>_final.csv`.
- Files that started as `.c3d` are converted back and written to
  `<output>/<stem>.c3d`.

Clicking **Run** prints the equivalent CLI command inside a highlighted
banner in the terminal — copy/paste it to repeat this run headlessly.

## CLI

```bash
# No args -> GUI (same as the button)
uv run vaila/edit_csv_c3d.py

# GUI, pre-filled with a directory
uv run vaila/edit_csv_c3d.py -i INPUT_DIR [-o OUTPUT_DIR]

# Headless: round-trip every .csv/.c3d file with no column changes
uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity

# Headless: keep/reorder columns by exact header name (CSV and C3D markers)
uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --columns Time,p1_X,p1_Y,p1_Z
```

Headless mode (`--identity` or `--columns`) never opens a Tk window — it is
safe to run in a script or CI. `--columns` mirrors `rearrange_data.
reshapedata`'s filtering: only existing headers are kept, in the order
given. For `.c3d`-derived data, `auto_create_c3d_from_csv` derives marker
labels from complete `LABEL_X/Y/Z` triples in column order, so a
`--columns` list touching C3D markers must keep whole X/Y/Z triples
together and in order, or the round-tripped C3D will be malformed.

## Scientific contract

- POINT RATE, ANALOG RATE, and POINT UNITS (m vs mm) from the source
  `.c3d` are preserved on write-back.
- Analog channels are preserved when present (round-tripped through the
  same `Time` + channel-column CSV schema used by the **C3D <--> CSV**
  exporter).
- A `NaN` sample in the edited CSV becomes a **negative residual** in the
  written C3D (occluded/untracked), never a valid sample sitting at the
  world origin — see `tests/test_c3d_invalid_points.py` and
  `tests/test_edit_csv_c3d.py`.
- Coordinate frame / axis labels are unchanged; this tool reorders/edits
  columns, it does not re-express lab axes (use `rearrange_data`'s
  existing lab-reference action on the CSV for that).

## Tests

```bash
uv run pytest tests/test_edit_csv_c3d.py tests/test_c3d_invalid_points.py tests/test_vaila_cli_menu.py -v
```

---

📅 **Added:** 24/08/2026
🔗 **Part of** <i>vailá</i> - Multimodal Toolbox
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
