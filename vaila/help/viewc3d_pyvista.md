# viewc3d_pyvista

## 📋 Module Information

- **Category:** Visualization
- **File:** `vaila/viewc3d_pyvista.py`
- **Version:** 0.2.0
- **Author:** Paulo Santiago
- **GUI Interface:** ✅ Yes
- **Backend:** PyVista (VTK)

## 📖 Description

VTK-based 3D viewer for C3D and CSV marker data. Timeline, interactive marker picking (left-click to select), skeleton connections, trails, export (screenshot, PNG sequence, MP4), quality stats. Same color palette and marker visibility options as the Open3D viewer (viewc3d.py).

**To run:** `uv run viewc3d_pyvista.py` or `python -m vaila.viewc3d_pyvista [file.c3d]`

**Dependencies:** `pip install pyvista ezc3d numpy`

### Architecture

- **MokkaLikeViewer** – Single class: state, `init_gui`, `update_frame`, key handlers. Load from C3D (`load_c3d`) or from arrays (`from_array` for CSV).

### Key Features

- C3D and CSV support; automatic unit detection (mm/m)
- **Left-click** to select marker (name shown on screen)
- **C** cycles marker color (Orange, Blue, Green, Red, White, Yellow, Purple, Cyan, Pink, Gray, Black)
- **M** – Dialog to show/hide markers
- View presets (1–4), background cycle (B), grid (G), labels (X)
- Trail (T), speed ([ ]), marker size (+ −), skeleton from JSON (J)
- Export: K screenshot, Z PNG sequence, V MP4
- Distance mode (D): click two markers to measure
- Info (I), quality stats (A), help (H)

### Main Keyboard Shortcuts

- **Navigation:** Space Play | ← → ±1 | ↑ ↓ ±10 | PgUp/PgDn ±100 | S Start | End End
- **View:** R Reset | 1–4 Presets | B Background | G Grid | X Labels | C Colors
- **Data:** T Trail | { } Trail length | [ ] Speed | + − Size | M Markers
- **Skeleton:** J Load JSON
- **Export:** K Screenshot | Z PNG seq | V MP4
- **Info:** I Info | A Stats | D Distance | H Help | Escape Clear

### Mouse

- **Left click** – Select marker (shows name)
- Left drag – Rotate | Middle/Right drag – Pan | Wheel – Zoom

## 🔧 Main Functions / Classes

- **MokkaLikeViewer** – Main viewer class (C3D and CSV)
- **MokkaLikeViewer.from_array** – Build viewer from NumPy arrays (e.g. from readcsv)
- **AVAILABLE_COLORS** – Palette for C key (same as viewc3d.py)

---

🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub](https://github.com/paulopreto/vaila-multimodaltoolbox)
