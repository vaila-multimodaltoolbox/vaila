# Edit CSV/C3D (`edit_csv_c3d.py`)

## Module information

- **Category:** Data Files
- **Version:** 0.4.5
- **Updated:** 2026-09-23
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

Click **Edit CSV/C3D**, pick a directory. A **Select Files to Edit** dialog
opens:

- **Depth** field: `0` = that directory only (default), `N` = descend `N`
  levels, `-1` = unlimited (every subdirectory).
- **Scan** lists every `.csv`/`.c3d` match (hidden dotfiles, and this
  module's own prior `processed_edit_csv_c3d_*` output, are always skipped).
- **Process All** edits every listed file; **Process Selected File** edits
  only the one highlighted in the list.

The same `ColumnReorderGUI` editor used by the old "Edit CSV" button opens
on the chosen set — `.c3d` files appear as their staged marker CSV (`Time,
LABEL_X, LABEL_Y, LABEL_Z, ...`). Files from different subdirectories are
staged together (flattened, collision-safe names) but each edited file is
written back under its **original relative subdirectory** inside the output
folder. Edit and close the editor (`Esc` for Save & Exit, or `Ctrl+S` for an
intermediate save) as usual. After the editor closes:

- Files that started as `.csv` are written to
  `<output>/<rel_dir>/<stem>_final.csv`.
- Files that started as `.c3d` are converted back and written to
  `<output>/<rel_dir>/<stem>.c3d`.

Clicking **Run** prints the equivalent CLI command inside a highlighted
banner in the terminal — copy/paste it to repeat this run headlessly (the
printed command always mirrors depth `0`; pass `-r`/`-d` yourself for a
deeper headless rerun, see CLI below).

### Bulk column rename / renumber

Inside the editor, **Edit -> Rename Column(s)...** opens a rename dialog:

- Double-click a header in the list to rename just that one column.
- Bulk section: a regex with **exactly one** numeric capture group (default
  `p(\d+)_`) plus an integer **offset** (default `-1`). Example — headers
  `frame,p1_x,p1_y,...,p70_x,p70_y` with pattern `p(\d+)_` and offset `-1`
  renumber every `pN_x`/`pN_y` down by one in a single operation, previewed
  before applying:
  - `p1_x -> p0_x`, `p1_y -> p0_y`, ..., `p70_x -> p69_x`, `p70_y -> p69_y`
    (140 renames), `frame` untouched.

A rename that would create duplicate resulting headers, or that renames
only part of an `_x/_y/_z` marker triple while leaving a sibling axis
unrenamed, is rejected before it is applied (keeps `auto_create_c3d_from_csv`'s
LABEL_X/Y/Z triple grouping intact for `.c3d` round-trips).

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

# Headless: recurse into every subdirectory (-r == -d -1)
uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity -r

# Headless: recurse exactly 2 levels down
uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity -d 2
```

Headless mode (`--identity` or `--columns`) never opens a Tk window — it is
safe to run in a script or CI. `--columns` mirrors `rearrange_data.
reshapedata`'s filtering: only existing headers are kept, in the order
given. For `.c3d`-derived data, `auto_create_c3d_from_csv` derives marker
labels from complete `LABEL_X/Y/Z` triples in column order, so a
`--columns` list touching C3D markers must keep whole X/Y/Z triples
together and in order, or the round-tripped C3D will be malformed.

`-r/--recursive` and `-d/--depth N` control how far the headless scan
descends (`-1` unlimited, `0` `INPUT_DIR` only — the default, `N` levels
down); each file's output mirrors its original relative subdirectory under
`OUTPUT_DIR`. Bulk column rename/renumber has **no headless flag** — it is
GUI-only (see above).

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
uv run pytest tests/test_edit_csv_c3d.py tests/test_edit_csv_c3d_recursive_rename.py tests/test_c3d_invalid_points.py tests/test_vaila_cli_menu.py -v
```

---

📅 **Added:** 24/08/2026
🔗 **Part of** <i>vailá</i> - Multimodal Toolbox
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
