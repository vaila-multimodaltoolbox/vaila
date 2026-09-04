# viewc3d_pyvista

## Module Information

- **Category:** Visualization
- **File:** `vaila/viewc3d_pyvista.py`
- **Version:** 0.3.121
- **Updated:** 04 September 2026
- **Author:** Paulo Santiago
- **GUI Interface:** Yes
- **Backend:** PyVista (VTK)

## Description

VTK-based 3D viewer for C3D and CSV marker data. Timeline, interactive marker picking (left-click to select), skeleton connections, trails, export (screenshot, PNG sequence, MP4), quality stats. Same color palette and marker visibility options as the Open3D viewer (`viewc3d.py`).

**Multi-C3D (v0.3.121):** load several `.c3d` files in one window with distinct per-file colors and a synchronized master timeline (Open3D Ctrl+L parity). Press **L** at runtime, multi-select at startup, or pass multiple paths via CLI `-i`.

### To run

```bash
uv run vaila/viewc3d_pyvista.py -i path/to/file.c3d
uv run vaila/viewc3d_pyvista.py -i a.c3d b.c3d
uv run vaila/viewc3d_pyvista.py path/to/one.c3d
python -m vaila.viewc3d_pyvista -i a.c3d b.c3d
```

Omit paths to open a multi-select file dialog. On launch the terminal prints `script=` and a `>> Equivalent CLI` mirror.

**Dependencies:** `pyvista`, `ezc3d`, `numpy` (via `uv sync`)

### Architecture

- **MokkaLikeViewer** — Single class: state, `init_gui`, `update_frame`, key handlers. Load from C3D (`load_c3d_paths`) or from arrays (`from_array` for CSV).
- **LoadedC3DPy** — Extra files (index ≥ 1): fixed palette color + PolyData actor.
- File 0 uses palette index 0 (Orange); **C** cycles file-0 color only. Extras keep load-time colors.
- Master FPS defaults to the highest source rate; on mismatch a prompt can downsample to the lowest.

### Key Features

- C3D and CSV support; automatic unit detection (mm/m)
- Multi-C3D overlay (`-i` / dialog / **L**)
- **Left-click** to select marker (name shown on screen)
- **C** cycles marker color for file 0 (Orange, Blue, Green, Red, White, Yellow, Purple, Cyan, Pink, Gray, Black)
- **M** — Dialog to show/hide markers (file 0)
- View presets (1–4), background cycle (B), grid (G), labels (X)
- Trail (T), speed ([ ]), marker size (+ −), skeleton from JSON (J) — file 0 only
- Export: K screenshot, Z PNG sequence, V MP4
- Distance mode (D): click two markers to measure
- Info (I), quality stats (A), help (H)

### Main Keyboard Shortcuts

- **Navigation:** Space Play | ← → ±1 | ↑ ↓ ±10 | PgUp/PgDn ±100 | S Start | End End
- **View:** R Reset | 1–4 Presets | B Background | G Grid | X Labels | C Colors
- **Data:** L Load more C3D | T Trail | { } Trail length | [ ] Speed | + − Size | M Markers
- **Skeleton:** J Load JSON
- **Export:** K Screenshot | Z PNG seq | V MP4
- **Info:** I Info | A Stats | D Distance | H Help | Escape Clear

### Mouse

- **Left click** — Select marker (shows name)
- Left drag — Rotate | Middle/Right drag — Pan | Wheel — Zoom

## Main Functions / Classes

- **MokkaLikeViewer** — Main viewer class (C3D and CSV)
- **MokkaLikeViewer.from_array** — Build viewer from NumPy arrays (e.g. from readcsv)
- **LoadedC3DPy** — Extra C3D container
- **merge_c3d_input_paths** / **build_viewc3d_pyvista_cli** — CLI helpers
- **AVAILABLE_COLORS** — Palette for C key / per-file colors (same as viewc3d.py)

---

Part of *[vailá](https://github.com/vaila-multimodaltoolbox/vaila)* — Multimodal Toolbox
