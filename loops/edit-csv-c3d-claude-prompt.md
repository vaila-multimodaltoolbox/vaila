# Implement Edit CSV/C3D (Claude Code CLI — implement now)

You are in the **vailá** repo at `/home/preto/data/vaila` (Python 3.12, Tkinter only, `uv run`).

**Do this work now.** Do not interview. Do not start an agent loop. Do not commit unless asked. Implement the feature, tests, GUI/CLI parity, and help/docs.

Repo rules: `AGENTS.md`, `CLAUDE.md`, `.cursor/rules/vaila.mdc`. Dual-import on every `vaila/` module. No second `tk.Tk()` when launched from the main GUI. Timestamped outputs. Never overwrite source files. GUI Run must print a copy-paste CLI with `print_gui_cli_mirror` from `vaila/cli_highlight.py` (`>>` prefix; absl eats `[bracketed]` stdout).

Current global version: **0.3.111**. Update Date on every edited `*.py`: **24 August 2026**. Keep Version **0.3.111**. Update `README.md` Last updated if you touch it (already 2026-08-24). Update help index “Generated on” / module Version+Updated.

---

## Goal

Frame C button `C_A_r1_c1` currently says **Edit CSV** and only runs `vaila/rearrange_data.py`.

Change it to **Edit CSV/C3D**. The button must accept `.csv` and `.c3d` and apply the **same CSV edits**. For C3D:

1. Convert C3D → CSV (markers) using the **C3D↔CSV** pipeline.
2. Edit with the existing rearrange_data tools.
3. Convert edited CSV → C3D and write timestamped output.
4. Leave original files untouched.

CSV-only dirs keep current rearrange_data behavior.

The **C3D <--> CSV** button (`C_A_r1_c2`) stays. Do not merge the two buttons.

---

## New script (required)

Create **`vaila/edit_csv_c3d.py`**. Wire the renamed button to this module. Keep `vaila/rearrange_data.py` as the CSV editor internals — do not duplicate `ColumnReorderGUI`.

Suggested public entry:

```python
def run_edit_csv_c3d() -> None:  # GUI, called from vaila.py
    ...

def main() -> None:  # argparse CLI
    ...
```

`if __name__ == "__main__": main()`

### Conversion APIs (do not call the GUI converters)

User said “use `readcsv_export.py`”. That file is **CSV → C3D only**. C3D → CSV lives in **`vaila/readc3d_export.py`**. Use **headless** functions, not dialogs:

| Direction | File | Use this | Do **not** call |
|-----------|------|----------|-----------------|
| C3D → CSV | `vaila/readc3d_export.py` | `importc3d()` + marker CSV writer (`LABEL_X/Y/Z` + `Time`, same schema as `save_to_files`) | `convert_c3d_to_csv()`, `batch_convert_c3d_to_csv()` (they create extra `Tk()` + file dialogs) |
| CSV → C3D | `vaila/readcsv_export.py` | `auto_create_c3d_from_csv(points_df, output_path, analog_df=..., point_rate=..., analog_rate=..., conversion_factor=..., point_units=...)` | `convert_csv_to_c3d()`, `batch_convert_csv_to_c3d()` |

If you need a small headless helper (e.g. `c3d_markers_to_dataframe(path) -> tuple[pd.DataFrame, meta]`), add it in `readc3d_export.py` or the new module. Prefer extracting rather than copy-pasting `save_to_files`.

`auto_create_c3d_from_csv` marker labels come from `col.rsplit("_", 1)[0]` on columns `[1::3]` (skip first time/frame column). Marker CSV from `readc3d_export` already matches (`Time`, `LABEL_X`, `LABEL_Y`, `LABEL_Z`). Keep that contract.

### C3D scientific contract

- Preserve **POINT RATE**, **ANALOG RATE**, **POINT UNITS** (m vs mm), and analog channels when present (`analog_df=`).
- Do not park occluded samples at the origin. NaN in CSV must stay invalid residuals on write-back (`tests/test_c3d_invalid_points.py`).
- Coordinate frame / axis labels stay as in the source C3D; this tool reorders/edits columns, it does not re-express lab axes unless the user uses rearrange_data’s existing lab-ref action on the CSV.
- Mixed directory: process `.csv` and `.c3d` together. Hidden files (`.name`) ignored.
- Output dir: `processed_edit_csv_c3d_YYYYMMDD_HHMMSS/` (or under user `-o`). CSV edits → `*_final.csv`. C3D edits → `*.c3d` with the same stem. Optionally keep the intermediate markers CSV next to the C3D for provenance; do not replace the input.

### Tkinter

`rearrange_data_in_directory()` currently does `root = tk.Tk(); root.withdraw()`. That is unsafe inside `vaila.py`. The new entry must:

- If `tk._default_root` exists: use a `Toplevel` / existing root; never a second `Tk()`.
- Standalone CLI/GUI: one `Tk()` is OK.
- Headless CLI (`-i` + flags, no display): no Tk, no hidden root.

---

## GUI

`vaila.py`:

- Button text `Edit CSV` → **`Edit CSV/C3D`** (~line 1300).
- ASCII map (~line 401) and any comments `C_A_r1_c1`.
- `reorder_csv_data` (~2784): lazy-import `edit_csv_c3d` and call `run_edit_csv_c3d()`. Keep the method name unless you also update `vaila_cli_menu.py` handler consistently.
- After the user confirms directory / run options, print the equivalent CLI:

```python
from vaila.cli_highlight import print_gui_cli_mirror
print_gui_cli_mirror("vaila/edit_csv_c3d", argv_list)
```

Follow `.claude/skills/yolo-fb-gui-cli/SKILL.md`. Use `print_gui_cli_mirror`, do not invent a local ANSI helper.

Also update:

- `vaila/vaila_cli_menu.py` label `Edit CSV` → `Edit CSV/C3D`
- `vaila/vaila_cli_hints.py` `reorder_csv_data` hint → `uv run vaila/edit_csv_c3d.py --help`
- `tests/test_vaila_cli_menu.py` string `Edit CSV`
- `vaila/__init__.py` only if you export the new entry (optional)

GUI file picker: directory containing `.csv` and/or `.c3d`. Then the existing rearrange GUI on the CSV view (converted C3D markers + native CSVs). After save, write C3D back for every file that started as `.c3d`.

---

## CLI (must work without GUI)

```text
uv run vaila/edit_csv_c3d.py
    # no args → GUI (same as button)

uv run vaila/edit_csv_c3d.py -i INPUT_DIR [-o OUTPUT_DIR]
    # GUI on that dir if no headless edit flags

uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --identity
    # headless: CSV copy through editor pipeline; C3D round-trip with no column change
    # used by tests

uv run vaila/edit_csv_c3d.py -i INPUT_DIR -o OUTPUT_DIR --columns COL1,COL2,...
    # headless column keep/reorder by header names (CSV and C3D markers)
```

Print `>> vaila/edit_csv_c3d: Equivalent CLI` on GUI run with the exact argv that reproduces that run. Headless must not open Tk.

Keep existing `rearrange_data.py --yolo-tracker` CLI working (do not break it).

---

## Tests (required)

Add `tests/test_edit_csv_c3d.py`. Use synthetic C3D via `auto_create_c3d_from_csv` (see `tests/test_c3d_invalid_points.py`). No 20 MiB fixtures.

Minimum cases:

1. CSV identity / column reorder on a tiny CSV (headers preserved, values match, source file unchanged).
2. C3D identity round-trip: points, labels, rate, units match within float tolerance; source `.c3d` unchanged.
3. C3D column subset/reorder via `--columns`, then C3D labels match the new order.
4. Occluded/NaN samples still get negative residuals (do not regress origin-blob).
5. Analog preserved when the source C3D has analog channels (if easy with ezc3d; otherwise document skip and still pass analog_df when the converter produced `_analogs.csv`).
6. CLI formatter / `print_gui_cli_mirror` argv contains `-i` and `-o`.
7. Headless path does not instantiate `tk.Tk` (mock or simply don’t import a GUI branch).

Do **not** treat a plot as proof. Assertions on arrays/headers/files.

---

## Docs / help (required)

Update all user-facing names and commands:

- `vaila/help/rearrange_data.md` + `.html` — note CSV+C3D via new entry, Version/Updated
- **New** `vaila/help/edit_csv_c3d.md` + `.html` (italic `<i>vailá</i>` / `*vailá*` in prose)
- `vaila/help/index.md` + `index.html` — `C_A_r1_c1 - Edit CSV/C3D`, link the new help; Generated on 24/08/2026
- `docs/vaila_buttons/edit-csv.md` and/or `reorder-csv-data.md` + `docs/vaila_buttons/README.md`
- `docs/api/modules.md` if it lists `rearrange_data`
- `README.md` Frame C map: `Edit CSV` → `Edit CSV/C3D`
- `CLAUDE.md` Frame C-A button list
- `vaila/help/README.md` if it lists rearrange_data

GUI and CLI examples in help must match the real argparse.

---

## QA (run these)

```bash
uv run ruff check vaila/edit_csv_c3d.py vaila/rearrange_data.py vaila/readc3d_export.py vaila/readcsv_export.py vaila.py tests/test_edit_csv_c3d.py --fix
uv run ruff format vaila/edit_csv_c3d.py vaila/rearrange_data.py vaila/readc3d_export.py vaila.py tests/test_edit_csv_c3d.py
uv run ty check vaila/edit_csv_c3d.py
uv run pytest tests/test_edit_csv_c3d.py tests/test_c3d_invalid_points.py tests/test_vaila_cli_menu.py -v
```

Fix failures you introduced. Do not “fix” tests by weakening them.

---

## Done when

- Button label is **Edit CSV/C3D** and launches `edit_csv_c3d.py`.
- `.csv` still editable as today.
- `.c3d` round-trips through CSV edits and writes a new C3D.
- GUI prints `>> vaila/edit_csv_c3d: ...` copy-paste CLI.
- Headless CLI works without Tk.
- Help/docs/menu/hints updated.
- Focused pytest + ruff + ty pass.

Report: files changed, CLI examples, test command + result. Do not git commit.
