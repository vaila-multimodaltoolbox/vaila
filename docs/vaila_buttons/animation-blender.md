# Animation Blender (`C_C_r4_c1`, v0.3.117)

## Overview

Frame C → **Visualization** → **Animation Blender** opens a *rec3d* reconstruction in Blender with the animation already imported and the scene already configured. It replaces the manual routine: launch Blender, load `<base>_blender_skeleton_viz.py` in the Text Editor, press **Run Script**.

Module: [`vaila/blender_viz.py`](../../vaila/blender_viz.py) · handler: `Vaila.animation_blender` · help: [`vaila/help/blender_viz.md`](../../vaila/help/blender_viz.md)

## Saving so anyone can open it (v0.3.117)

After the companion script runs, **File > Save As** now produces a single self-contained `.blend`: the skeleton (BVH action) already saved fine, and as of v0.3.117 the per-frame body mesh does too, since it is baked into the mesh's own Shape Keys (one per frame, `CONSTANT`-interpolated) instead of a live Python `frame_change_post` handler holding coordinates in memory. A handler and a plain dict are not Blender data-blocks and never survived a save/reload — the mesh used to freeze on whatever frame it was on at save time. See `generate_blender_companion_script()` in [`vaila/rec3d.py`](../../vaila/rec3d.py) and the "Saving the mesh animation" section of [`vaila/help/rec3d.md`](../../vaila/help/rec3d.md) for the mechanism, and `docs/claude-session-notes-archive.md` for the verification evidence (fresh-process reopen, coordinates checked against the source OBJ files).

## Why the companion script and not a plain import

Every *rec3d* run — from `rec3d.py` (per-frame DLT) and `rec3d_one_dlt3d.py` (one fixed DLT per camera) — writes a `*_blender_skeleton_viz.py` next to its outputs. That script imports the BVH, builds the OBJ/PLY mesh sequence with no add-on, draws the skeleton bones, and sets the scene rate and frame range **last**.

Blender's own importers skip that last step. `File > Import > BVH` leaves `update_scene_fps` and `update_scene_duration` off, so a 631-frame 120 Hz capture lands in a scene still at 24 fps with `frame_end` 250 — it plays in slow motion and stops a third of the way through. Importing the OBJ sequence by hand has a second trap: picking `up = Z` "because the data is Z-up" leaves the body about 0.65 m tall and sliding 2.2 m across the take, which looks like the mesh sinking through the floor.

The button exists so neither mistake is possible.

## GUI → CLI mirror

The GUI prints the equivalent command on launch, with the `>>` prefix (absl logging, pulled in by mediapipe/opencv, eats `[bracketed]` stdout):

```
>> vaila/blender_viz: Equivalent CLI
>> uv run python -m vaila.blender_viz -i RUN_DIR --blender /snap/bin/blender
```

| Task | Command |
|---|---|
| Open a run directory | `uv run python -m vaila.blender_viz -i /path/to/vaila_rec3d_20260805_093649` |
| Open a script directly | `uv run python -m vaila.blender_viz -i /path/to/rec3d_..._blender_skeleton_viz.py` |
| Rebuild a stale script first | `uv run python -m vaila.blender_viz -i RUN_DIR --regenerate` |
| Force a specific Blender | `uv run python -m vaila.blender_viz -i RUN_DIR --blender /snap/bin/blender` |
| Headless scene check | `uv run python -m vaila.blender_viz -i RUN_DIR --background` |

`--background` adds Blender's `-b` and waits for the process, so it returns Blender's exit code and is usable as a check in scripts. Without it the launch is non-blocking, keeping the *vailá* Tk loop responsive while Blender is open.

## Blender discovery

First working hit wins; each candidate is validated by running `--version`, so a wrong pick fails immediately instead of opening nothing.

1. `--blender` argument
2. `VAILA_BLENDER` environment variable
3. `~/.vaila/vaila_config.toml`
4. `blender` on `PATH`
5. Per-OS install locations (Linux snap/apt/Flatpak, macOS `Blender.app/Contents/MacOS/Blender`, Windows `Program Files\Blender Foundation\Blender *`, newest first)
6. GUI only: a file dialog — and the choice is persisted, so the user is asked once

```toml
# ~/.vaila/vaila_config.toml
[blender]
executable = "/snap/bin/blender"
```

A saved path that stops working is skipped and auto-detection continues, so uninstalling or moving Blender does not dead-end the button.

## Runs with no companion script

Older runs predate the script, and a script whose recorded paths went stale is worth rebuilding. Both cases are handled by regenerating from the run folder's own files:

| Input | Recovered from |
|---|---|
| Frame count, capture rate | BVH `MOTION` header, refined by `POINT:RATE` in `<base>_m.c3d` when the two agree within 0.5% |
| Marker layout, never-reconstructed markers | The reconstruction CSV, via `rec3d.find_unreconstructed_markers` |
| Mesh sequence | A `meshes_obj/` or `meshes_ply/` subfolder |
| Skeleton connections | Marker **count**: 17 → `yolo_coco17`, 33 → `mediapipe_pose33`, 70 → `sam3dinov3_mhr70`, 308 → `sapiens2_goliath308` |

**Gotcha:** the marker *count* identifies the layout, not the highest marker index — `sapiens2_goliath308`'s connection list tops out at `p63` despite the layout having 308 markers, so an index-based rule misfiles it as MHR70. An unrecognised count is not fatal; the generator falls back to its built-in connections.

The BVH rate needs the C3D cross-check because the BVH stores a frame *time* that has to be inverted: 120 Hz comes back as 120.0000048 from the 9-decimal `Frame Time`, while the C3D stores the rate itself. The 9 decimals are still what make fractional NTSC-derived rates like 119.88012001 Hz survive the round trip at all — at 6 decimals it reads back as 119.875330 Hz.

## Tests

`tests/test_blender_viz.py` — pure Python, no `bpy` and no Blender process: BVH header round-trip at fractional rates, skeleton inference for all four layouts, script resolution (direct `.py`, newest-in-folder, regeneration, forced rebuild), executable-discovery precedence including a stale saved path, config round-trip, and argv construction with `subprocess.Popen` monkeypatched.

## Related

- [`vaila/help/blender_viz.md`](../../vaila/help/blender_viz.md) — user-facing help
- [`docs/dlt_reconstruction_and_mesh_alignment.md`](../dlt_reconstruction_and_mesh_alignment.md) — the reconstruction and mesh-alignment pipeline that produces these runs
