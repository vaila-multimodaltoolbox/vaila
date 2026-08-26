"""
Project: vailá
Script: rec3d_mesh_pipeline.py
Authors: Paulo Santiago, Sergio Barroso, Felipe Dias, Lennin Abrão
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila

Creation Date: 26 August 2026
Update Date: 26 August 2026
Version: 0.3.116

Description:
    Repeats, on any N-camera dataset, the verified 3-step chain that turns
    Sapiens2-guided per-camera 3D mesh estimation into one Blender-ready,
    DLT3D-aligned multi-camera mesh sequence:

        1. ``sapiens2_3d.py``            -- per camera, GPU (isolated subprocess).
        2. ``sam3dinov3_visualize.py``   -- per camera, extract one person's
                                             mesh bundle (in-process; CPU/cv2 only).
        3. ``rec3d_one_dlt3d.py``        -- merge N cameras' bundles into one
                                             aligned mesh sequence (subprocess).

    This module does not reimplement any of those three scripts; it only
    orchestrates them in order, validates each stage's real result (never
    trusting exit code alone -- ``sapiens2_3d.py`` can exit 0 while every
    frame silently failed if its CUDA/nvrtc environment is broken), and
    reports what happened.

    Cameras are described in a TOML manifest (arbitrary N, unlike a flat
    per-camera CLI flag):

        output_dir = "/path/to/out_parent"
        export_mesh = "obj"          # obj | ply
        overwrite = false

        [[camera]]
        video = "/path/c1.mp4"
        sapiens2_results = "/path/c1_sam3sapiens2_visualized_id_04"
        dlt3d = "/path/c1.dlt3d"
        id = 4                        # optional; auto if exactly one ID exists

        [[camera]]
        video = "/path/c2.mp4"
        sapiens2_results = "/path/c2_..."
        dlt3d = "/path/c2.dlt3d"

    ``sapiens2_results`` may instead be ``sam_results`` (a raw SAM3 run),
    matching ``sapiens2_3d.py``'s own two input modes.

Usage:
    # Headless, from a manifest:
    uv run python -u vaila/rec3d_mesh_pipeline.py --config pipeline.toml

    # GUI: pick cameras interactively (Frame B -> Markerless 3D ->
    # Multi-Camera Mesh Pipeline), which writes a manifest and re-launches
    # this same script headlessly as an isolated GPU subprocess.

License:
    This program is licensed under the GNU Affero General Public License v3.0.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import re
import subprocess
import sys
import tkinter as tk
from dataclasses import dataclass, field
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any

import toml

try:
    from .cli_highlight import print_gui_cli_mirror
    from .gpu_subprocess import run_isolated_gpu_subprocess
    from .sam3dinov3_visualize import (
        discover_ids,
        load_predictions,
        resolve_run_dir,
        visualize_selected_id,
    )
    from .sam3sapiens2 import _prepare_gui_root
except ImportError:  # standalone execution
    from cli_highlight import print_gui_cli_mirror  # ty: ignore[unresolved-import]
    from gpu_subprocess import run_isolated_gpu_subprocess  # ty: ignore[unresolved-import]
    from sam3dinov3_visualize import (  # ty: ignore[unresolved-import]
        discover_ids,
        load_predictions,
        resolve_run_dir,
        visualize_selected_id,
    )
    from sam3sapiens2 import _prepare_gui_root  # ty: ignore[unresolved-import]

MIN_CAMERAS = 2
MESH_EXPORT_FORMATS = ("obj", "ply")
# Matches only rec3d_one_dlt3d.py's primary marker csv (rec3d_<timestamp>.csv),
# not its rec3d_<timestamp>_joint_angles.csv / _mesh_alignment.csv siblings --
# both of those also match a bare "rec3d_*.csv" glob.
_REC3D_PRIMARY_CSV_RE = re.compile(r"^rec3d_\d{8}_\d{6}\.csv$")


def _log(message: str) -> None:
    print(f">> vaila/rec3d_mesh_pipeline: {message}", flush=True)


class PipelineError(RuntimeError):
    """A pipeline stage failed; the run is aborted, nothing downstream runs."""


@dataclass(slots=True)
class CameraSpec:
    video: Path
    dlt3d: Path
    sapiens2_results: Path | None = None
    sam_results: Path | None = None
    id: int | None = None
    label: str = ""

    def __post_init__(self) -> None:
        if not self.label:
            self.label = self.video.stem


@dataclass(slots=True)
class PipelineConfig:
    output_dir: Path
    cameras: list[CameraSpec]
    export_mesh: str = "obj"
    overwrite: bool = False


@dataclass(slots=True)
class CameraStageResult:
    label: str
    sapiens2_3d_log: Path
    sam3d_results_dir: Path
    selected_id: int
    mesh_bundle_dir: Path


@dataclass(slots=True)
class PipelineResult:
    output_dir: Path
    cameras: list[CameraStageResult] = field(default_factory=list)
    merge_log: Path | None = None
    merge_output_dir: Path | None = None
    report_path: Path | None = None


def load_config(config_path: Path) -> PipelineConfig:
    """Parse and validate a pipeline TOML manifest."""
    raw = toml.load(config_path)
    if not raw.get("output_dir"):
        raise PipelineError("Manifest is missing required key 'output_dir'.")
    camera_rows = raw.get("camera") or []
    if len(camera_rows) < MIN_CAMERAS:
        raise PipelineError(
            f"DLT3D triangulation needs at least {MIN_CAMERAS} cameras; "
            f"manifest has {len(camera_rows)} [[camera]] entries."
        )
    cameras: list[CameraSpec] = []
    for i, row in enumerate(camera_rows, start=1):
        video = row.get("video")
        dlt3d = row.get("dlt3d")
        sapiens2_results = row.get("sapiens2_results")
        sam_results = row.get("sam_results")
        if not video or not dlt3d:
            raise PipelineError(f"[[camera]] #{i} is missing 'video' or 'dlt3d'.")
        if not sapiens2_results and not sam_results:
            raise PipelineError(f"[[camera]] #{i} needs 'sapiens2_results' or 'sam_results'.")
        video_path = Path(video).expanduser().resolve()
        dlt3d_path = Path(dlt3d).expanduser().resolve()
        results_path = Path(sapiens2_results or sam_results).expanduser().resolve()
        if not video_path.is_file():
            raise PipelineError(f"[[camera]] #{i}: video not found: {video_path}")
        if not dlt3d_path.is_file():
            raise PipelineError(f"[[camera]] #{i}: dlt3d file not found: {dlt3d_path}")
        if not results_path.is_dir():
            raise PipelineError(f"[[camera]] #{i}: results dir not found: {results_path}")
        cameras.append(
            CameraSpec(
                video=video_path,
                dlt3d=dlt3d_path,
                sapiens2_results=Path(sapiens2_results).expanduser().resolve()
                if sapiens2_results
                else None,
                sam_results=Path(sam_results).expanduser().resolve() if sam_results else None,
                id=row.get("id"),
                label=row.get("label", ""),
            )
        )
    export_mesh = raw.get("export_mesh", "obj")
    if export_mesh not in MESH_EXPORT_FORMATS:
        raise PipelineError(
            f"'export_mesh' must be one of {MESH_EXPORT_FORMATS}, got {export_mesh!r}"
        )
    return PipelineConfig(
        output_dir=Path(raw["output_dir"]).expanduser().resolve(),
        cameras=cameras,
        export_mesh=export_mesh,
        overwrite=bool(raw.get("overwrite", False)),
    )


def _run_sapiens2_3d_stage(camera: CameraSpec, stage_dir: Path) -> tuple[Path, Path]:
    """Run sapiens2_3d.py for one camera; return (log_path, sam3d_results_dir)."""
    stage_dir.mkdir(parents=True, exist_ok=True)
    log_path = stage_dir / "sapiens2_3d.log"
    script = str(Path(__file__).resolve().parent / "sapiens2_3d.py")
    cmd = [
        sys.executable,
        "-u",
        script,
        "-i",
        str(camera.video),
        "-o",
        str(stage_dir),
        "--save-mesh",
        "--export-mesh",
        "obj",
    ]
    if camera.sapiens2_results is not None:
        cmd += ["--sapiens2-results", str(camera.sapiens2_results)]
    else:
        cmd += ["--sam-results", str(camera.sam_results)]

    _log(f"[{camera.label}] Stage 1/3 sapiens2_3d.py -> {log_path}")
    with log_path.open("w") as log_file:
        result = run_isolated_gpu_subprocess(
            cmd, device=0, stdout=log_file, stderr=subprocess.STDOUT
        )

    log_text = log_path.read_text(errors="replace")
    batch_lines = [line for line in log_text.splitlines() if "Batch done:" in line]
    if not batch_lines:
        raise PipelineError(
            f"[{camera.label}] sapiens2_3d.py produced no 'Batch done:' line "
            f"(returncode={result.returncode}). Never trust exit code alone -- "
            f"see {log_path} for the full log."
        )
    batch_line = batch_lines[-1]
    try:
        failed_count = int(batch_line.split(" ok, ")[1].split(" failed")[0])
    except (IndexError, ValueError) as exc:
        raise PipelineError(
            f"[{camera.label}] could not parse '{batch_line}' from {log_path}"
        ) from exc
    if result.returncode != 0 or failed_count > 0:
        raise PipelineError(
            f"[{camera.label}] sapiens2_3d.py failed: {batch_line} "
            f"(returncode={result.returncode}); see {log_path}"
        )
    _log(f"[{camera.label}] {batch_line}")

    candidates = sorted(stage_dir.glob("processed_sapiens2_3d_*"))
    if not candidates:
        raise PipelineError(
            f"[{camera.label}] sapiens2_3d.py reported success but no "
            f"processed_sapiens2_3d_* directory was created under {stage_dir}"
        )
    # sapiens2_3d.py nests its real output one level deeper, under the
    # video's stem (processed_sapiens2_3d_TIMESTAMP/<video_stem>/) --
    # resolve down to it now so every downstream consumer (_resolve_camera_id,
    # _run_visualize_stage) gets a dir that actually contains the
    # *_sam3dinov3_predictions.json.gz file.
    run_dir = resolve_run_dir(candidates[-1], camera.video)
    return log_path, run_dir


def _resolve_camera_id(camera: CameraSpec, run_dir: Path) -> int:
    payload = load_predictions(run_dir)
    available = discover_ids(run_dir, payload)
    if not available:
        raise PipelineError(f"[{camera.label}] no person IDs found in {run_dir}")
    if camera.id is not None:
        if camera.id not in available:
            raise PipelineError(
                f"[{camera.label}] id {camera.id} is not available; choices: {available}"
            )
        return camera.id
    if len(available) == 1:
        return available[0]
    raise PipelineError(
        f"[{camera.label}] multiple person IDs available ({available}); "
        f"set 'id' in the manifest to pick one."
    )


def _run_visualize_stage(
    camera: CameraSpec, run_dir: Path, selected_id: int, bundle_dir: Path, overwrite: bool
) -> Path:
    _log(f"[{camera.label}] Stage 2/3 sam3dinov3_visualize.py id={selected_id} -> {bundle_dir}")
    run_dir = resolve_run_dir(run_dir, camera.video)
    visualize_selected_id(
        run_dir,
        camera.video,
        selected_id,
        bundle_dir,
        overwrite=overwrite,
        export_mesh="obj",
    )
    return bundle_dir


def _run_merge_stage(
    config: PipelineConfig, camera_results: list[CameraStageResult], output_dir: Path
) -> tuple[Path, Path]:
    log_path = output_dir / "rec3d_one_dlt3d.log"
    script = str(Path(__file__).resolve().parent / "rec3d_one_dlt3d.py")
    cmd = [
        sys.executable,
        "-u",
        script,
        "--dlt3d",
        *[str(c.dlt3d) for c in config.cameras],
        "--mesh-source-dir",
        *[str(r.mesh_bundle_dir) for r in camera_results],
        "--export-mesh",
        config.export_mesh,
        "-o",
        str(output_dir),
    ]
    _log(f"Stage 3/3 rec3d_one_dlt3d.py merge ({len(config.cameras)} cameras) -> {log_path}")
    with log_path.open("w") as log_file:
        completed = subprocess.run(cmd, stdout=log_file, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0:
        raise PipelineError(
            f"rec3d_one_dlt3d.py merge failed (returncode={completed.returncode}); see {log_path}"
        )
    # rec3d_one_dlt3d.py never writes into the dir passed via -o directly --
    # it always nests its real output one level deeper, under its own
    # vaila_rec3d_<timestamp>/ subdir. Resolve down to it so the csv check
    # (and everything the caller reports as the final mesh location) points
    # at a dir that actually contains rec3d_*.csv.
    merge_dirs = sorted(output_dir.glob("vaila_rec3d_*"))
    merged = (
        sorted(p for p in merge_dirs[-1].glob("rec3d_*.csv") if _REC3D_PRIMARY_CSV_RE.match(p.name))
        if merge_dirs
        else []
    )
    if not merge_dirs or not merged:
        raise PipelineError(
            f"rec3d_one_dlt3d.py exited 0 but no vaila_rec3d_*/rec3d_*.csv was "
            f"produced in {output_dir}"
        )
    _log(f"Merge complete: {merged[-1].name}")
    return log_path, merge_dirs[-1]


def run_pipeline(config: PipelineConfig) -> PipelineResult:
    """Run the full 3-step chain for every camera in ``config``. Fail fast."""
    if len(config.cameras) < MIN_CAMERAS:
        raise PipelineError(f"Need at least {MIN_CAMERAS} cameras, got {len(config.cameras)}.")
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = config.output_dir / f"vaila_rec3d_mesh_pipeline_{timestamp}"
    out_dir.mkdir(parents=True, exist_ok=True)
    result = PipelineResult(output_dir=out_dir)

    for i, camera in enumerate(config.cameras, start=1):
        cam_tag = f"cam{i}_{camera.label}"
        stage_dir = out_dir / f"{cam_tag}_sapiens2_3d"
        log_path, run_dir = _run_sapiens2_3d_stage(camera, stage_dir)
        selected_id = _resolve_camera_id(camera, run_dir)
        bundle_dir = out_dir / "mesh_bundles" / f"{cam_tag}_id_{selected_id:02d}"
        _run_visualize_stage(camera, run_dir, selected_id, bundle_dir, config.overwrite)
        result.cameras.append(
            CameraStageResult(
                label=camera.label,
                sapiens2_3d_log=log_path,
                sam3d_results_dir=run_dir,
                selected_id=selected_id,
                mesh_bundle_dir=bundle_dir,
            )
        )

    result.merge_log, result.merge_output_dir = _run_merge_stage(config, result.cameras, out_dir)

    report = {
        "output_dir": str(out_dir),
        "cameras": [
            {
                "label": c.label,
                "sapiens2_3d_log": str(c.sapiens2_3d_log),
                "sam3d_results_dir": str(c.sam3d_results_dir),
                "selected_id": c.selected_id,
                "mesh_bundle_dir": str(c.mesh_bundle_dir),
            }
            for c in result.cameras
        ],
        "merge_log": str(result.merge_log),
        "merge_output_dir": str(result.merge_output_dir),
        "export_mesh": config.export_mesh,
        "generated_at": dt.datetime.now().isoformat(timespec="seconds"),
    }
    report_path = out_dir / "pipeline_report.json"
    report_path.write_text(json.dumps(report, indent=2))
    result.report_path = report_path
    _log(f"Pipeline done -> {out_dir}")
    return result


# --------------------------------------------------------------------------- #
# GUI
# --------------------------------------------------------------------------- #


@dataclass(slots=True)
class _CameraRowVars:
    video: tk.StringVar
    results: tk.StringVar
    dlt3d: tk.StringVar
    id_: tk.StringVar
    frame: ttk.Labelframe


class RecMeshPipelineDialog(tk.Toplevel):
    """Collect N cameras + shared settings, then relaunch headlessly."""

    def __init__(self, master: tk.Tk | tk.Toplevel) -> None:
        super().__init__(master)
        self.title("Multi-Camera Mesh Pipeline (repeat on other files)")
        self.resizable(True, True)
        self.result: Path | None = None
        self._rows: list[_CameraRowVars] = []

        outer = ttk.Frame(self, padding=10)
        outer.pack(fill="both", expand=True)

        ttk.Label(
            outer,
            text=(
                "Sapiens2 3D (per camera) -> SAM3+DINOv3 Visualize ID (per camera) "
                "-> DLT3D multi-camera merge. Needs at least 2 cameras."
            ),
            wraplength=560,
            justify="left",
        ).pack(anchor="w", pady=(0, 8))

        self.rows_container = ttk.Frame(outer)
        self.rows_container.pack(fill="both", expand=True)

        btn_row = ttk.Frame(outer)
        btn_row.pack(fill="x", pady=4)
        ttk.Button(btn_row, text="Add camera", command=self._add_row).pack(side="left")

        settings = ttk.LabelFrame(outer, text="Output settings", padding=8)
        settings.pack(fill="x", pady=8)
        ttk.Label(settings, text="Output parent dir:").grid(row=0, column=0, sticky="w")
        self.output_var = tk.StringVar()
        ttk.Entry(settings, textvariable=self.output_var, width=50).grid(
            row=0, column=1, sticky="we"
        )
        ttk.Button(settings, text="Browse", command=self._browse_output).grid(row=0, column=2)
        ttk.Label(settings, text="Export mesh:").grid(row=1, column=0, sticky="w")
        self.export_mesh_var = tk.StringVar(value="obj")
        ttk.Combobox(
            settings,
            textvariable=self.export_mesh_var,
            values=MESH_EXPORT_FORMATS,
            width=8,
            state="readonly",
        ).grid(row=1, column=1, sticky="w")
        self.overwrite_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            settings, text="Overwrite existing bundles", variable=self.overwrite_var
        ).grid(row=2, column=0, columnspan=2, sticky="w")
        settings.columnconfigure(1, weight=1)

        actions = ttk.Frame(outer)
        actions.pack(fill="x", pady=(8, 0))
        ttk.Button(actions, text="Cancel", command=self.destroy).pack(side="right")
        ttk.Button(actions, text="Run", command=self._on_run).pack(side="right", padx=6)

        self._add_row()
        self._add_row()
        self.grab_set()

    def _add_row(self) -> None:
        idx = len(self._rows) + 1
        frame = ttk.LabelFrame(self.rows_container, text=f"Camera {idx}", padding=6)
        frame.pack(fill="x", pady=4)
        video = tk.StringVar()
        results = tk.StringVar()
        dlt3d = tk.StringVar()
        id_ = tk.StringVar()

        def browse_video() -> None:
            path = filedialog.askopenfilename(title="Select video")
            if path:
                video.set(path)

        def browse_results() -> None:
            path = filedialog.askdirectory(title="Select sapiens2/sam3sapiens2 results dir")
            if path:
                results.set(path)

        def browse_dlt3d() -> None:
            path = filedialog.askopenfilename(title="Select .dlt3d file")
            if path:
                dlt3d.set(path)

        ttk.Label(frame, text="Video:").grid(row=0, column=0, sticky="w")
        ttk.Entry(frame, textvariable=video, width=45).grid(row=0, column=1, sticky="we")
        ttk.Button(frame, text="Browse", command=browse_video).grid(row=0, column=2)
        ttk.Label(frame, text="Results dir:").grid(row=1, column=0, sticky="w")
        ttk.Entry(frame, textvariable=results, width=45).grid(row=1, column=1, sticky="we")
        ttk.Button(frame, text="Browse", command=browse_results).grid(row=1, column=2)
        ttk.Label(frame, text="DLT3D file:").grid(row=2, column=0, sticky="w")
        ttk.Entry(frame, textvariable=dlt3d, width=45).grid(row=2, column=1, sticky="we")
        ttk.Button(frame, text="Browse", command=browse_dlt3d).grid(row=2, column=2)
        ttk.Label(frame, text="Person ID (blank=auto):").grid(row=3, column=0, sticky="w")
        ttk.Entry(frame, textvariable=id_, width=10).grid(row=3, column=1, sticky="w")

        def remove_row() -> None:
            if len(self._rows) <= MIN_CAMERAS:
                messagebox.showwarning(
                    "Multi-Camera Mesh Pipeline",
                    f"At least {MIN_CAMERAS} cameras are required.",
                    parent=self,
                )
                return
            frame.destroy()
            self._rows.remove(row_vars)
            for i, r in enumerate(self._rows, start=1):
                r.frame.configure(text=f"Camera {i}")

        row_vars = _CameraRowVars(video=video, results=results, dlt3d=dlt3d, id_=id_, frame=frame)
        ttk.Button(frame, text="Remove", command=remove_row).grid(row=3, column=2, sticky="e")
        frame.columnconfigure(1, weight=1)
        self._rows.append(row_vars)

    def _browse_output(self) -> None:
        path = filedialog.askdirectory(title="Select output parent directory")
        if path:
            self.output_var.set(path)

    def _on_run(self) -> None:
        try:
            output_dir = self.output_var.get().strip()
            if not output_dir:
                raise ValueError("Output parent directory is required.")
            cameras = []
            for i, row in enumerate(self._rows, start=1):
                video = row.video.get().strip()
                results = row.results.get().strip()
                dlt3d = row.dlt3d.get().strip()
                if not video or not results or not dlt3d:
                    raise ValueError(
                        f"Camera {i}: video, results dir, and DLT3D file are all required."
                    )
                cam: dict[str, Any] = {"video": video, "sapiens2_results": results, "dlt3d": dlt3d}
                id_raw = row.id_.get().strip()
                if id_raw:
                    cam["id"] = int(id_raw)
                cameras.append(cam)
            if len(cameras) < MIN_CAMERAS:
                raise ValueError(f"At least {MIN_CAMERAS} cameras are required.")
        except ValueError as exc:
            messagebox.showerror("Multi-Camera Mesh Pipeline", str(exc), parent=self)
            return

        manifest = {
            "output_dir": output_dir,
            "export_mesh": self.export_mesh_var.get(),
            "overwrite": bool(self.overwrite_var.get()),
            "camera": cameras,
        }
        manifest_dir = Path(output_dir).expanduser().resolve()
        manifest_dir.mkdir(parents=True, exist_ok=True)
        manifest_path = manifest_dir / f"rec3d_mesh_pipeline_{dt.datetime.now():%Y%m%d_%H%M%S}.toml"
        manifest_path.write_text(toml.dumps(manifest))
        self.result = manifest_path
        self.destroy()


def run_rec3d_mesh_pipeline_gui(existing_root: tk.Tk | tk.Toplevel | None = None) -> None:
    """GUI entry point: collect cameras, then relaunch headlessly (isolated)."""
    owns_root = existing_root is None
    root = existing_root if existing_root is not None else tk.Tk()
    if owns_root:
        root.withdraw()
    _prepare_gui_root(root, owns_root=owns_root)
    dialog = RecMeshPipelineDialog(root)
    root.wait_window(dialog)
    manifest_path = dialog.result
    if owns_root:
        root.destroy()
    if manifest_path is None:
        return
    cli = [
        "uv",
        "run",
        "python",
        "-u",
        "vaila/rec3d_mesh_pipeline.py",
        "--config",
        str(manifest_path),
    ]
    print_gui_cli_mirror("vaila/rec3d_mesh_pipeline", cli)
    launch = [sys.executable, "-u", str(Path(__file__).resolve()), "--config", str(manifest_path)]
    gui_result = run_isolated_gpu_subprocess(launch, device=0)
    if gui_result.returncode != 0:
        messagebox.showerror(
            "Multi-Camera Mesh Pipeline",
            f"Pipeline failed (exit {gui_result.returncode}). See per-stage logs under the output directory.",
        )
    else:
        messagebox.showinfo(
            "Multi-Camera Mesh Pipeline", "Pipeline finished. See pipeline_report.json."
        )


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Repeat the Sapiens2 3D -> SAM3+DINOv3 Visualize ID -> DLT3D "
            "multi-camera mesh pipeline on any dataset described by a TOML manifest."
        )
    )
    parser.add_argument("--config", type=Path, help="Pipeline TOML manifest")
    parser.add_argument("--open-help", action="store_true")
    return parser


def _help_path() -> Path:
    return Path(__file__).resolve().parent / "help" / "rec3d_mesh_pipeline.md"


def main() -> None:
    parser = _build_parser()
    args = parser.parse_args()
    if args.open_help:
        import webbrowser

        if _help_path().is_file():
            webbrowser.open_new_tab(_help_path().as_uri())
        return
    if args.config is None:
        run_rec3d_mesh_pipeline_gui()
        return
    config = load_config(args.config)
    try:
        run_pipeline(config)
    except PipelineError as exc:
        _log(f"FAILED: {exc}")
        sys.exit(1)


if __name__ == "__main__":
    main()
