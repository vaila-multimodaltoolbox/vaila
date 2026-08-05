# blender_viz

## Module information

| Field | Value |
|-------|--------|
| **Category** | Visualization |
| **File** | `vaila/blender_viz.py` |
| **Version** | 0.3.99 |
| **Updated** | 05 August 2026 |
| **Author** | Paulo Santiago |
| **GUI** | Yes — Frame C → Visualization → **Animation Blender** (`C_C_r4_c1`) |
| **CLI** | Yes — `uv run python -m vaila.blender_viz -i RUN_DIR` |

---

## Description

Opens a *rec3d* reconstruction in Blender with the animation already imported and the scene already configured. It replaces the manual routine of launching Blender, loading `<base>_blender_skeleton_viz.py` in the Text Editor and pressing **Run Script**.

Every *rec3d* run — from both `rec3d.py` (per-frame DLT) and `rec3d_one_dlt3d.py` (one fixed DLT per camera) — writes that companion script next to its outputs. The script imports the BVH, builds the OBJ/PLY mesh sequence without needing any add-on, draws the skeleton bones, and sets the scene rate and frame range **last**. That last step is the one Blender's own importers skip: `File > Import > BVH` leaves `update_scene_fps` and `update_scene_duration` off, so a 631-frame 120 Hz capture lands in a scene still at 24 fps with `frame_end` 250 — it plays in slow motion and stops a third of the way through.

`blender_viz` finds Blender, resolves the script, and runs `blender --python <script>`. The script calls `main()` at module bottom, so it executes on startup.

---

## What it accepts

| Selection | Behaviour |
|---|---|
| A *rec3d* run directory | Uses the newest `*_blender_skeleton_viz.py` inside it |
| A run directory with **no** script | Rebuilds one from the run's own files (see below) |
| The `*_blender_skeleton_viz.py` file itself | Uses it directly |

### Rebuilding a missing script

Everything the generator needs is recoverable from the folder, so runs produced before the companion script existed still work:

| Input | Source |
|---|---|
| Frame count and capture rate | The BVH `MOTION` header (`Frames:`, `Frame Time:`), refined by `POINT:RATE` in `<base>_m.c3d` when the two agree |
| Marker layout, never-reconstructed markers | The reconstruction CSV |
| Mesh sequence | A `meshes_obj/` or `meshes_ply/` subfolder |
| Skeleton connections | Inferred from the marker **count**: 17 → `yolo_coco17`, 33 → `mediapipe_pose33`, 70 → `sam3dinov3_mhr70`, 308 → `sapiens2_goliath308` |

The count, not the highest marker index, is what identifies the layout — `sapiens2_goliath308`'s connection list tops out at `p63` despite the layout having 308 markers.

---

## Finding Blender

Resolution order, first working hit wins. Each candidate is validated by running `--version`, so a wrong pick fails immediately with an explanation instead of opening nothing.

1. `--blender` command-line argument
2. `VAILA_BLENDER` environment variable
3. The path saved in `~/.vaila/vaila_config.toml`
4. `blender` on `PATH`
5. Usual install locations for the platform:
   - **Linux:** `/snap/bin/blender`, `/usr/bin/blender`, `/usr/local/bin/blender`, Flatpak exports
   - **macOS:** `/Applications/Blender.app/Contents/MacOS/Blender`
   - **Windows:** `C:\Program Files\Blender Foundation\Blender */blender.exe` (newest first)
6. **GUI only:** a file dialog — and the choice is saved, so the question is asked once

```toml
# ~/.vaila/vaila_config.toml
[blender]
executable = "/snap/bin/blender"
```

A saved path that stops working (Blender uninstalled or moved) is skipped and auto-detection continues.

---

## CLI

```bash
# Open a run directory
uv run python -m vaila.blender_viz -i /path/to/vaila_rec3d_20260805_093649

# Point straight at the script
uv run python -m vaila.blender_viz -i /path/to/rec3d_..._blender_skeleton_viz.py

# Force a rebuild — the fix for a script whose recorded paths went stale
uv run python -m vaila.blender_viz -i RUN_DIR --regenerate

# Use a specific Blender
uv run python -m vaila.blender_viz -i RUN_DIR --blender /snap/bin/blender

# Headless: verify the scene without opening a window (exits when done)
uv run python -m vaila.blender_viz -i RUN_DIR --background
```

Running with no `-i` opens the GUI dialogs. The GUI prints the equivalent CLI on launch with the `>>` prefix, so any run can be repeated headlessly.

---

## Notes

- The launch is **non-blocking**: the *vailá* window stays usable while Blender is open.
- Re-running is safe. The companion script does not duplicate the armature or the mesh, and re-running it is the documented fix for a scene whose rate or frame range was already broken by a manual import.
- `--background` runs Blender with `-b` and waits for it, which is what makes it usable as a check in scripts.

---

## See also

- [rec3d](rec3d.md) — multi-camera reconstruction (per-frame DLT)
- [rec3d_one_dlt3d](rec3d_one_dlt3d.md) — reconstruction with one fixed DLT per camera, plus mesh export
- [mesh_alignment](mesh_alignment.md) — the alignment behind the exported mesh sequence
