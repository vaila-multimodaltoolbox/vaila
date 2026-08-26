# rec3d_mesh_pipeline

## Module information

| Field | Value |
|-------|--------|
| **Category** | Processing / Markerless 3D |
| **File** | `vaila/rec3d_mesh_pipeline.py` |
| **Version** | 0.3.116 |
| **Author** | Paulo Santiago |
| **GUI** | Yes — Frame B → **Markerless 3D** → **Multi-Camera Mesh Pipeline** |
| **CLI** | Yes |
| **Runtime** | GPU per camera (`sapiens2_3d.py` stage); merge stage is CPU only |

---

## What problem this solves

A real 3-camera dataset showed that a Blender-ready 3D mesh can be produced
from `sam3sapiens2.py` output by chaining three existing scripts by hand:

1. `sapiens2_3d.py` per camera — Sapiens2-guided SAM 3D Body/DINOv3 mesh.
2. `sam3dinov3_visualize.py` per camera — extract one person's mesh bundle
   (works unmodified against `sapiens2_3d.py` output, same
   `vaila_sam3dinov3_v1` schema).
3. `rec3d_one_dlt3d.py --mesh-source-dir` — triangulate/merge N cameras
   into one aligned mesh sequence (OBJ/PLY + BVH + C3D) for Blender.

This module wraps that proven chain behind one button (and CLI) so it can
be repeated on other files without manual orchestration. It does not
reimplement any of the three scripts; it only orchestrates them, validates
each stage's real result, and reports what happened.

## The NVRTC/CUDA environment bug this closes

In venvs where torch's cu13 wheels coexist with older cu12 nvidia
packages, `libnvrtc-builtins.so` can live outside the default
dynamic-loader search path (`nvidia/cu13/lib`, no RPATH) — so a
torch.compile-triggering worker (`sapiens2_3d.py`, `sam3dinov3.py`) can
fail *every frame internally while still exiting 0*. The only reliable
signal is the printed `Batch done: N ok, M failed` line, never the exit
code alone.

`vaila/gpu_subprocess.py::ensure_cuda_nvrtc_env()` patches the child's
`LD_LIBRARY_PATH` before every GPU-isolated subprocess launch (fixes the
GUI path for free), and `reexec_self_if_nvrtc_env_missing()` self-re-execs
a direct CLI invocation of `sapiens2_3d.py`/`sam3dinov3.py` with the same
fix before any heavy import — so this pipeline's button needs no manual
`export LD_LIBRARY_PATH` step.

## Input — TOML manifest

N cameras is variable, so a flat per-camera CLI flag doesn't fit; the
manifest follows vailá's existing TOML-configured-CLI convention
(`interp_smooth_split.py -c smooth_config.toml`):

```toml
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
```

`sapiens2_results` may instead be `sam_results` (a raw SAM3 run),
matching `sapiens2_3d.py`'s own two input modes. At least 2 cameras are
required — DLT3D triangulation needs 2+ views.

## Pipeline stages

1. **Validate** the manifest: `>= 2` cameras, every video/dlt3d path
   exists, every results dir exists, `export_mesh in {obj, ply}`.
2. **Per camera**, run `sapiens2_3d.py -i video -o <out>/camN_sapiens2_3d
   --save-mesh --export-mesh obj` (no `--dlt3d`/`--ref3d` — the raw
   combined per-camera output, not the DLT-aligned monocular one, is what
   `sam3dinov3_visualize.py` needs) via `run_isolated_gpu_subprocess`.
   Parses the logged `Batch done: N ok, M failed` line and aborts the
   whole run with a clear error if it's missing or `M > 0` — exit code is
   never trusted alone.
3. **Per camera**, resolve the person ID (manifest `id`, or the sole
   available ID) and call `visualize_selected_id()` in-process to extract
   that person's mesh bundle.
4. **Merge**: `rec3d_one_dlt3d.py --dlt3d <cam1.dlt3d> ... --mesh-source-dir
   <bundle1> ... --export-mesh obj -o <out>`, checked for a produced
   `rec3d_*.csv`.
5. Writes `<out>/pipeline_report.json` (per-camera logs, chosen IDs, mesh
   bundle dirs, merge log) for headless-run auditability.

## GUI

`RecMeshPipelineDialog` — one row per camera (Browse buttons for video /
results dir / `.dlt3d`, optional person ID), **Add camera**/**Remove**
(minimum 2 enforced), shared output dir / mesh format / overwrite
settings. **Run** writes the manifest and relaunches this same script
headlessly as an isolated GPU subprocess — the GUI thread never blocks on
GPU work.

## Example

```bash
uv run python -u vaila/rec3d_mesh_pipeline.py --config pipeline.toml
```

## Outputs

`<output_dir>/vaila_rec3d_mesh_pipeline_YYYYMMDD_HHMMSS/`:

| Path | Content |
|---|---|
| `camN_<label>_sapiens2_3d/` | Per-camera `sapiens2_3d.py` run + `sapiens2_3d.log` |
| `mesh_bundles/camN_<label>_id_NN/` | Per-camera extracted mesh bundle (`sam3dinov3_visualize.py` output) |
| `rec3d_*.csv`, `meshes_obj/` (or `_ply/`), `.bvh`, `.c3d` | Final merged multi-camera mesh sequence (`rec3d_one_dlt3d.py` output) |
| `rec3d_one_dlt3d.log` | Merge-stage log |
| `pipeline_report.json` | Per-camera stage results + final output dir, for auditability |

## Caveats

- Every camera stage runs sequentially and any single failure aborts the
  whole run (fail fast) rather than merging a partial set.
- The person ID must resolve unambiguously per camera (set `id` in the
  manifest when a camera's results dir has more than one person).

## Related modules

| Module | Role |
|---|---|
| [sapiens2_3d](sapiens2_3d.md) | Stage 1 — per-camera Sapiens2-guided 3D mesh |
| [sam3dinov3_visualize](sam3dinov3_visualize.md) | Stage 2 — per-camera mesh bundle extraction |
| [rec3d_one_dlt3d](rec3d_one_dlt3d.md) | Stage 3 — multi-camera DLT3D merge |
| [monocular_dlt_align](monocular_dlt_align.md) | Single-camera alternative when only 1 view is available |

## Testing

- `tests/test_rec3d_mesh_pipeline.py` — manifest validation (min-2-cameras
  gate, missing video/dlt3d/results-dir errors, `sapiens2_results`/
  `sam_results` alias, `export_mesh` validation) and
  `ensure_cuda_nvrtc_env()`'s pure path-patching logic (no GPU needed).

---

Part of *vailá* - Multimodal Toolbox  
[GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
