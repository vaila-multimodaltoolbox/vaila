"""
================================================================================
Script: blender_viz.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

Author: Paulo Santiago
Version: 0.3.104
Created: 05 August 2026
Last Updated: 11 August 2026

Description:
    Launches Blender directly from vailá with a rec3d reconstruction already
    imported and the scene correctly configured — the "Animation Blender"
    button in Frame C-C (Visualization).

    Every rec3d run (rec3d.py and rec3d_one_dlt3d.py alike) writes a
    ``<base>_blender_skeleton_viz.py`` companion script next to its outputs.
    That script imports the BVH, builds the OBJ/PLY mesh sequence without any
    add-on, draws the skeleton bones, and — crucially — sets the scene rate
    and frame range LAST, which Blender's own BVH importer does not do
    (``update_scene_fps``/``update_scene_duration`` default to False, leaving
    a 631-frame 120 Hz capture in a 24 fps scene that stops at frame 250).

    Until now the user had to open Blender, load that script in the Text
    Editor and press Run Script. This module removes that step: it finds the
    Blender executable, resolves (or regenerates) the companion script, and
    runs ``blender --python <script>``, which executes the script on startup
    because it calls ``main()`` at module bottom.

Usage:
    GUI (from vailá): Frame C -> Visualization -> Animation Blender

    CLI:
      uv run python -m vaila.blender_viz -i /path/to/vaila_rec3d_YYYYMMDD_HHMMSS
      uv run python -m vaila.blender_viz -i /path/to/rec3d_..._blender_skeleton_viz.py
      uv run python -m vaila.blender_viz -i RUN_DIR --regenerate
      uv run python -m vaila.blender_viz -i RUN_DIR --blender /snap/bin/blender
      uv run python -m vaila.blender_viz -i RUN_DIR --background   # headless check

License:
    This program is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by the
    Free Software Foundation, either version 3 of the License, or (at your
    option) any later version.
================================================================================
"""

from __future__ import annotations

import glob
import os
import platform
import re
import shutil
import subprocess
import sys
from pathlib import Path

try:
    from .cli_highlight import print_gui_cli_mirror
    from .rec3d import find_unreconstructed_markers, generate_blender_companion_script
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from rec3d import (  # ty: ignore[unresolved-import]
        find_unreconstructed_markers,
        generate_blender_companion_script,
    )

COMPANION_SUFFIX = "_blender_skeleton_viz.py"

# Marker count -> preset in vaila/skeletons/. The count is the only reliable
# discriminator: the MAX marker index does not work, because
# sapiens2_goliath308's connection list tops out at p63 even though the layout
# has 308 markers.
SKELETON_PRESET_BY_MARKER_COUNT = {
    17: "yolo_coco17.json",
    33: "mediapipe_pose33.json",
    70: "sam3dinov3_mhr70.json",
    308: "sapiens2_goliath308.json",
}


# ---------------------------------------------------------------------------
# User configuration (~/.vaila/vaila_config.toml)
# ---------------------------------------------------------------------------


def vaila_config_path() -> Path:
    """Location of the per-user vailá config file."""
    return Path.home() / ".vaila" / "vaila_config.toml"


def load_vaila_config() -> dict:
    """Read the user config, returning {} when absent or unreadable.

    Deliberately never raises: a corrupt config must not stop a user from
    opening Blender, it just falls back to auto-detection.
    """
    path = vaila_config_path()
    if not path.is_file():
        return {}
    try:
        import toml

        return toml.load(path)
    except Exception as exc:  # noqa: BLE001
        print(f">> vaila/blender_viz: ignoring unreadable config {path}: {exc}")
        return {}


def save_vaila_config(config: dict) -> bool:
    """Write the user config, returning whether it succeeded."""
    path = vaila_config_path()
    try:
        import toml

        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as fh:
            toml.dump(config, fh)
        return True
    except Exception as exc:  # noqa: BLE001
        print(f">> vaila/blender_viz: could not save config {path}: {exc}")
        return False


def remember_blender_executable(executable: str) -> bool:
    """Persist a Blender path so the user is only ever asked once."""
    config = load_vaila_config()
    config.setdefault("blender", {})["executable"] = str(executable)
    return save_vaila_config(config)


# ---------------------------------------------------------------------------
# Blender discovery
# ---------------------------------------------------------------------------


def blender_version(executable: str) -> str | None:
    """Return the reported version string, or None if this is not Blender.

    Running ``--version`` is what turns a wrong pick (a folder, a launcher
    script, the wrong binary) into an immediate, explainable failure instead
    of a window that never opens.
    """
    try:
        result = subprocess.run(
            [executable, "--version"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"Blender\s+([0-9][^\s]*)", result.stdout or "")
    return match.group(1) if match else None


def candidate_blender_paths() -> list[str]:
    """Plausible Blender locations for this OS, in preference order."""
    system = platform.system()
    candidates: list[str] = []
    which = shutil.which("blender")
    if which:
        candidates.append(which)
    if system == "Darwin":
        candidates += [
            "/Applications/Blender.app/Contents/MacOS/Blender",
            str(Path.home() / "Applications/Blender.app/Contents/MacOS/Blender"),
        ]
    elif system == "Windows":
        for root in (
            os.environ.get("PROGRAMFILES", r"C:\Program Files"),
            os.environ.get("PROGRAMFILES(X86)", r"C:\Program Files (x86)"),
        ):
            candidates += sorted(
                glob.glob(os.path.join(root, "Blender Foundation", "Blender *", "blender.exe")),
                reverse=True,  # newest version first
            )
    else:  # Linux / BSD
        candidates += [
            "/snap/bin/blender",
            "/usr/bin/blender",
            "/usr/local/bin/blender",
            "/var/lib/flatpak/exports/bin/org.blender.Blender",
            str(Path.home() / ".local/share/flatpak/exports/bin/org.blender.Blender"),
        ]
    # Preserve order while dropping duplicates.
    return list(dict.fromkeys(candidates))


def find_blender_executable(gui: bool = False, explicit: str | None = None) -> str | None:
    """Locate a working Blender executable.

    Order: ``--blender`` argument, ``VAILA_BLENDER``, the saved user config,
    ``PATH``, then this OS's usual install locations. In GUI mode a file
    dialog is the last resort, and whatever the user picks is remembered so
    the question is never asked twice.
    """
    ordered: list[str] = []
    if explicit:
        ordered.append(explicit)
    env = os.environ.get("VAILA_BLENDER")
    if env:
        ordered.append(env)
    saved = load_vaila_config().get("blender", {}).get("executable")
    if saved:
        ordered.append(str(saved))
    ordered += candidate_blender_paths()

    for candidate in dict.fromkeys(ordered):
        version = blender_version(candidate)
        if version:
            print(f">> vaila/blender_viz: using Blender {version} at {candidate}")
            return candidate

    if not gui:
        return None

    from tkinter import filedialog, messagebox

    messagebox.showwarning(
        "Blender not found",
        "vailá could not find Blender automatically.\n\n"
        "Select the Blender executable (on macOS it is inside "
        "Blender.app/Contents/MacOS/Blender).\n\n"
        "The choice is saved, so you will only be asked once.",
    )
    chosen = filedialog.askopenfilename(title="Select the Blender executable")
    if not chosen:
        return None
    version = blender_version(chosen)
    if not version:
        messagebox.showerror(
            "Not a Blender executable",
            f"{chosen}\n\ndid not respond to --version as Blender. "
            "Pick the executable itself, not a folder or a shortcut.",
        )
        return None
    remember_blender_executable(chosen)
    print(f">> vaila/blender_viz: using Blender {version} at {chosen} (saved to config)")
    return chosen


# ---------------------------------------------------------------------------
# Companion-script resolution and regeneration
# ---------------------------------------------------------------------------


def parse_bvh_header(bvh_path) -> tuple[int, float] | None:
    """Recover ``(n_frames, point_rate)`` from a BVH MOTION header.

    The exporter writes ``Frame Time`` with 9 decimals precisely so fractional
    capture rates (NTSC-derived 119.88012001 Hz, say) survive this round trip;
    6 decimals would read back as 119.875330 Hz.
    """
    n_frames = None
    frame_time = None
    try:
        with open(bvh_path, encoding="utf-8") as fh:
            for line in fh:
                stripped = line.strip()
                if stripped.startswith("Frames:"):
                    n_frames = int(stripped.split(":", 1)[1])
                elif stripped.startswith("Frame Time:"):
                    frame_time = float(stripped.split(":", 1)[1])
                    break  # Frame Time always follows Frames; motion data next
    except (OSError, ValueError) as exc:
        print(f">> vaila/blender_viz: could not read BVH header from {bvh_path}: {exc}")
        return None
    if not n_frames or not frame_time or frame_time <= 0:
        return None
    return n_frames, 1.0 / frame_time


def read_c3d_rate(c3d_path) -> float | None:
    """POINT:RATE from a C3D, or None if it cannot be read.

    Preferred over the BVH-derived rate when the two agree: the BVH stores a
    frame *time* that has to be inverted, so 120 Hz comes back as
    120.0000048, whereas the C3D stores the rate itself.
    """
    try:
        import ezc3d

        c3d = ezc3d.c3d(str(c3d_path))
        rate = float(c3d["parameters"]["POINT"]["RATE"]["value"][0])
    except Exception:  # noqa: BLE001 - optional refinement, never fatal
        return None
    return rate if rate > 0 else None


def skeletons_dir() -> Path:
    """Directory holding the shipped skeleton connection presets."""
    return Path(__file__).resolve().parent / "skeletons"


def infer_skeleton_preset(n_markers: int) -> str | None:
    """Path to the preset matching this marker count, or None if unknown.

    An unknown count is not an error: the generator falls back to its built-in
    MediaPipe connections, which is better than refusing to open Blender.
    """
    filename = SKELETON_PRESET_BY_MARKER_COUNT.get(n_markers)
    if not filename:
        return None
    path = skeletons_dir() / filename
    return str(path) if path.is_file() else None


def find_run_file_base(run_dir) -> str | None:
    """The ``rec3d_<timestamp>`` base name shared by a run's output files."""
    run_dir = Path(run_dir)
    for bvh in sorted(run_dir.glob("*.bvh")):
        if (run_dir / f"{bvh.stem}.csv").is_file():
            return bvh.stem
    csvs = [p for p in sorted(run_dir.glob("*.csv")) if not p.stem.endswith("_mesh_alignment")]
    return csvs[0].stem if csvs else None


def regenerate_companion_script(run_dir) -> str | None:
    """Rebuild the Blender companion script from a run's own output files.

    Everything the generator needs is recoverable from the folder: the frame
    count and capture rate from the BVH header, the marker layout and the
    never-reconstructed markers from the reconstruction CSV, and the mesh
    directory by name. This is what lets the button work on runs produced
    before the companion script existed, or whose script was lost.
    """
    import pandas as pd

    run_dir = Path(run_dir)
    file_base = find_run_file_base(run_dir)
    if not file_base:
        print(f">> vaila/blender_viz: no rec3d output found in {run_dir}")
        return None

    csv_path = run_dir / f"{file_base}.csv"
    if not csv_path.is_file():
        print(f">> vaila/blender_viz: missing {csv_path.name}; cannot regenerate the script")
        return None
    df = pd.read_csv(csv_path)
    markers = [c for c in df.columns if c.endswith("_x") and c.startswith("p")]

    header = parse_bvh_header(run_dir / f"{file_base}.bvh")
    if header:
        n_frames, point_rate = header
    else:
        n_frames, point_rate = len(df), 120.0
        print(
            f">> vaila/blender_viz: no readable BVH header; assuming {point_rate} Hz "
            f"over {n_frames} frames"
        )
    # Refine with the C3D's stored rate when the two describe the same take;
    # a large disagreement means the C3D belongs to something else, so keep
    # the BVH value, which is what the animation itself was written against.
    c3d_rate = read_c3d_rate(run_dir / f"{file_base}_m.c3d")
    if c3d_rate and abs(c3d_rate - point_rate) / point_rate < 0.005:
        point_rate = c3d_rate

    mesh_dir = None
    for name in ("meshes_obj", "meshes_ply"):
        if (run_dir / name).is_dir():
            mesh_dir = name
            break

    skeleton = infer_skeleton_preset(len(markers))
    print(
        f">> vaila/blender_viz: regenerating companion script for {file_base} "
        f"({len(markers)} markers, {n_frames} frames, {point_rate:.6f} Hz, "
        f"skeleton={os.path.basename(skeleton) if skeleton else 'default'}, "
        f"mesh={mesh_dir or 'none'})"
    )
    return generate_blender_companion_script(
        str(run_dir),
        file_base,
        skeleton_json_path=skeleton,
        point_rate=point_rate,
        n_frames=n_frames,
        mesh_dir=mesh_dir,
        unreconstructed_markers=find_unreconstructed_markers(df),
    )


def resolve_companion_script(target, regenerate: bool = False) -> str | None:
    """Turn a user selection into a runnable companion script.

    `target` may be the script itself or a rec3d run directory. For a
    directory the newest ``*_blender_skeleton_viz.py`` wins, and one is
    generated when none exists (or when `regenerate` forces a rebuild, which
    is the fix for a script whose recorded paths point somewhere stale).
    """
    path = Path(target)
    if path.is_file():
        if path.suffix != ".py":
            print(f">> vaila/blender_viz: {path} is not a Python script")
            return None
        return str(path)
    if not path.is_dir():
        print(f">> vaila/blender_viz: no such file or directory: {path}")
        return None

    if regenerate:
        return regenerate_companion_script(path)
    scripts = sorted(path.glob(f"*{COMPANION_SUFFIX}"), key=lambda p: p.stat().st_mtime)
    if scripts:
        return str(scripts[-1])
    return regenerate_companion_script(path)


# ---------------------------------------------------------------------------
# Launch
# ---------------------------------------------------------------------------


def build_blender_argv(script_path, blender_executable, background: bool = False) -> list[str]:
    """Argv that makes Blender run the companion script on startup.

    The companion script calls ``main()`` at module bottom, so ``--python`` is
    all it takes — no ``--python-expr`` wrapper and no add-on.
    """
    argv = [str(blender_executable)]
    if background:
        argv.append("-b")
    argv += ["--python", str(script_path)]
    return argv


def launch_blender(script_path, blender_executable, background: bool = False):
    """Start Blender without blocking the caller.

    Non-blocking on purpose: the vailá Tk main loop has to keep running while
    Blender is open, and Blender's own window is where the user works next.
    """
    argv = build_blender_argv(script_path, blender_executable, background=background)
    print(">> vaila/blender_viz: " + " ".join(argv))
    return subprocess.Popen(argv)


def format_blender_viz_cli(target, blender_executable=None, background=False, regenerate=False):
    """Copy-pasteable CLI equivalent of a GUI run.

    Uses the ``>>`` prefix convention rather than ``[brackets]``: absl logging,
    pulled in by mediapipe/opencv, silently eats bracketed stdout prefixes.
    """
    parts = ["uv run python -m vaila.blender_viz", f"-i {target}"]
    if blender_executable:
        parts.append(f"--blender {blender_executable}")
    if regenerate:
        parts.append("--regenerate")
    if background:
        parts.append("--background")
    return " ".join(parts)


# ---------------------------------------------------------------------------
# Entry points
# ---------------------------------------------------------------------------


def run_blender_viz() -> None:
    """GUI entry point — called from vaila.py (Frame C-C: Animation Blender)."""
    from tkinter import filedialog, messagebox

    print("=" * 80)
    print("vailá - Animation Blender")
    print("=" * 80)

    target = filedialog.askdirectory(
        title="Select a rec3d run directory (vaila_rec3d_YYYYMMDD_HHMMSS)"
    )
    if not target:
        # Second chance for users who would rather point at the script itself.
        target = filedialog.askopenfilename(
            title="Or select a *_blender_skeleton_viz.py script",
            filetypes=[("Blender companion script", "*.py"), ("All files", "*.*")],
        )
    if not target:
        print(">> vaila/blender_viz: cancelled")
        return

    script_path = resolve_companion_script(target)
    if not script_path:
        messagebox.showerror(
            "No visualization script",
            f"Could not find or build a Blender companion script for:\n{target}\n\n"
            "Select a rec3d output directory (the one holding the .bvh and .csv) "
            "or the *_blender_skeleton_viz.py script itself.",
        )
        return

    blender_executable = find_blender_executable(gui=True)
    if not blender_executable:
        messagebox.showerror(
            "Blender not found",
            "vailá needs Blender to show this animation.\n\n"
            "Install it from https://www.blender.org/download/ , or set the "
            "VAILA_BLENDER environment variable to its full path.",
        )
        return

    print_gui_cli_mirror("vaila/blender_viz", format_blender_viz_cli(target, blender_executable))

    try:
        launch_blender(script_path, blender_executable)
    except OSError as exc:
        messagebox.showerror("Could not start Blender", f"{blender_executable}\n\n{exc}")
        return

    messagebox.showinfo(
        "Blender launching",
        f"Blender is opening with:\n{os.path.basename(script_path)}\n\n"
        "The BVH, mesh sequence, skeleton bones, scene rate and frame range "
        "are all set up automatically — just press SPACE to play.",
    )


def main(argv=None) -> int:
    """CLI entry point."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Open a vailá rec3d reconstruction in Blender, fully set up.",
    )
    parser.add_argument(
        "-i",
        "--input",
        dest="target",
        help="rec3d run directory, or a *_blender_skeleton_viz.py script",
    )
    parser.add_argument(
        "--blender",
        dest="blender",
        default=None,
        help="Path to the Blender executable (otherwise auto-detected)",
    )
    parser.add_argument(
        "--regenerate",
        action="store_true",
        help="Rebuild the companion script from the run's own files before launching",
    )
    parser.add_argument(
        "--background",
        action="store_true",
        help="Run Blender headless (-b): useful to verify a scene without a window",
    )
    args = parser.parse_args(argv)

    if not args.target:
        run_blender_viz()
        return 0

    script_path = resolve_companion_script(args.target, regenerate=args.regenerate)
    if not script_path:
        print(">> vaila/blender_viz: no companion script could be resolved")
        return 1

    blender_executable = find_blender_executable(gui=False, explicit=args.blender)
    if not blender_executable:
        print(">> vaila/blender_viz: Blender not found. Pass --blender PATH or set VAILA_BLENDER.")
        return 1

    process = launch_blender(script_path, blender_executable, background=args.background)
    if args.background:
        return process.wait()
    return 0


if __name__ == "__main__":
    sys.exit(main())
