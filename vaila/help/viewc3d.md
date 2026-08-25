# viewc3d

## 📋 Module Information

- **Category:** Visualization
- **File:** `vaila\viewc3d.py`
- **Version:** 0.3.111
- **Author:** Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description

Advanced 3D viewer for C3D files with adaptive visualization for different scales.
Automatically detects and converts units (millimeters/meters) with enhanced confidence scoring.
Features adaptive ground plane, grid, and camera positioning based on data scale.
Soccer field lines and penalty areas supported.
Supports loading multiple C3D files into one running viewer (`Ctrl+L`), each
with its own marker color, synced to one timeline even when source FPS differ.

**To run:** `uv run viewc3d.py`. For a VTK/PyVista-based viewer (C3D and CSV), use `viewc3d_pyvista.py` (see `help/viewc3d_pyvista.md`).

### Architecture

- **VailaModel** (data layer): C3D acquisition with `GetPointFrame`, `GetFrameNumber`, `GetPointFrequency`, `get_bounds()`, `is_z_up()`. Clean separation from visualization; enables multi-C3D loading.
- **VailaView** (Open3D layer): Scene, trails, dynamic grid, Z-up camera, marker picking.
- **LoadedC3D**: per-file viewer state (color, spheres) for files added via `Ctrl+L`.

### Key Features

- Adaptive visualization for small (lab) to large (soccer field) scales
- Automatic unit detection (mm/m) with confidence scoring
- Interactive marker selection with search and filter options
- Real-time marker labels with color coding
- Ground grid toggle and field line customization
- **Trails (ghosting):** older segments drawn darker; velocity-based coloring
- **Z-up camera:** automatic when Z range > Y range (biomechanics convention)
- **Marker picking:** `` ` `` / `` ~ `` keys to cycle and highlight one marker (yellow)
- **Multi-file loading (`Ctrl+L`):** add more C3D files at runtime, each with a
  distinct marker color; timeline syncs to the highest-FPS loaded file by
  default, with a prompt to downsample to the lowest FPS instead (nearest-index
  playback lookup — no data is interpolated or overwritten)
- Matplotlib fallback for systems without OpenGL support

### Main Keyboard Shortcuts

- **Navigation:** ← → frame; ↑ ↓ 60 frames; S/E start/end; Space play/pause
- **Markers:** +/- size; C color; X labels; `` ` `` / `` ~ `` pick (highlight)
- **View:** T background; Y ground; G field lines; M grid; R reset camera
- **Trails:** W toggle trails with ghosting
- **Multi-file:** `Ctrl+L` load additional C3D file(s)
- **Help:** H open help (browser)

## 🔧 Main Functions / Classes

- **VailaModel** – Data layer (C3D acquisition API)
- **VailaView** – Open3D visualization layer
- **LoadedC3D** – Per-file state for files added via `Ctrl+L`
- `run_viewc3d` – Main entry point
- `resolve_master_fps`, `master_frame_count`, `map_master_frame_to_local` – Multi-file timeline sync (pure helpers)
- `assign_file_color` – Per-file marker color assignment
- `load_c3d_file` – Load C3D with unit detection
- `select_markers` – Marker selection dialog
- `detect_c3d_units` – mm/m detection
- `create_ground_plane`, `create_ground_grid` – Dynamic grid from bounds
- `run_viewc3d_fallback` – Matplotlib fallback when OpenGL unavailable
- `load_field_lines_from_csv`, `create_football_field_lines` – Field overlay
- Plus: `create_coordinate_lines`, `create_x_marker`, `check_opengl_support`, etc.

---

🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
