# Treadmill LC Integration Handoff

Last updated: 2026-06-30

This file documents the current treadmill load-cell processing integration in vailá. It is intended as handoff material for future agents and maintainers. It describes what is implemented now, how the workflow is organized, and which behavior must be preserved.

## Current State

The integration is implemented as the **Frame B -> B6_r7_c3 - Treadmill LC** button and as the module `vaila/treadmill_lc.py`.

### Primary Files

- `vaila/treadmill_lc.py`: Main module with GUI/CLI entry points and processing stages.
- `tests/test_treadmill_lc.py`: Unit tests for calibration helpers, filtering config, interpolation helpers, COP layout, figure generation, and window selection.
- `vaila/help/treadmill_lc.md` and `vaila/help/treadmill_lc.html`: User-facing documentation.
- `vaila.py`: Main GUI entry point wired to the Treadmill LC button.
- `README.md` and `vaila/help/index.*`: Main documentation entries.

## Workflow

The full pipeline order is:

1. **Adjust + Interpolate**
2. **Filter**
3. **Process Metrics**

The GUI exposes these stages as separate buttons and as one full-pipeline action. The normal GUI should not present interpolation as a separate standalone stage because adjustment and interpolation are a single user workflow.

## File Discovery

The module must distinguish trials from calibration and metadata files by filename.

- Trials: `s*_d*_t*.csv`
- Tare calibration: `s*_d*_tara.csv`
- Participant weight calibration: `s*_d*_peso.csv`
- Plate-weight calibration: `s*_d*_*kg.csv`
- Borg metadata: `borg_*.txt`

Borg TXT files are never processed as trials. When a Borg file has a `Peso` column, the weight is used automatically. When `Tent` includes the configured problem marker, such as `T02*`, that trial should be reviewed in the adjustment/interpolation stage.

## Stage Behavior

### Adjust + Interpolate

- Plot the four load cells and the summed signal.
- Let the user select the affected load cell channels.
- Let the user mark artifact START/END intervals.
- Provide undo/correction before saving.
- Allow interval treatment modes: remove, `NaN`, zero, neutral mean, and linear bridge.
- Let the user compare up to four interpolation methods.
- Let the user choose the final method after visual comparison.
- Show a preview and only save after approval.
- Save approved CSVs to `LIMPOS` as `*_LIMPO.csv`.
- Save metadata as JSON, TOML, and CSV sidecars with selected cells, intervals, treatment mode, selected interpolation methods, final method, and interpolation parameters.

### Filter

- Default filter is low-pass Butterworth SOS at 40 Hz.
- Median filtering uses `scipy.ndimage.median_filter` with configurable `edge_mode`; default is `nearest`.
- Zero-phase filtering uses `sosfiltfilt`.
- Optional mains-notch filtering supports 50 Hz and 60 Hz.
- `filter_type` supports `lowpass`, `bandpass`, `highpass`, `median`, and `none`.
- Batch GUI filtering previews one calibration file and one running trial, then applies the same settings to the remaining files without opening per-file plot windows.
- Filtered data keep their source filenames in `filtrado`; spectrum diagnostics are saved to `filter_analysis` with explicit `filter_` names.

### Process Metrics

- Group files by subject-day prefix (`sXX_dYY`).
- Discover matching calibration files and Borg metadata for each group.
- Average all calibration files over only the central 5 seconds to avoid edge transients.
- Compute calibration once per subject-day group where possible.
- Support simple calibration from `tara` and `peso`.
- Support plate-weight calibration from `*kg.csv` files as complementary calibration points.
- Manual analysis-window selection uses left click START and optional END, right click to clear marks, and Enter to finalize. One click means START to final sample. Invalid selections reopen the same selector.
- Save per-attempt step details as `*_processing_steps.csv`.
- Save one biomechanical metrics file per subject-day as `sXX_dYY_processing_metrics.csv`.
- Save processing outputs to timestamped `results_YYYYMMDD_HHMMSS` and `figures_YYYYMMDD_HHMMSS` directories.

## COP Convention

The load-cell layout is fixed:

- Cell 1: superior left
- Cell 2: inferior left
- Cell 3: superior right
- Cell 4: inferior right

Distances:

- 58 cm from left to right cells
- 113 cm from superior to inferior cells

COP is computed in centimeters with origin at the treadmill center. `cop_x` is medio-lateral and must be plotted on the horizontal axis. `cop_y` is anterior-posterior and must be plotted on the vertical axis.

Generated COP figures:

- `processing_cop_trajectory.png`: full analyzed COP trajectory, not per-step COP.
- `processing_cop_report_interactive.html`: optional lightweight Plotly report with GRF, derivative, and full COP.
- `processing_overview.png`: total GRF plus first derivative over time in seconds.

All time axes and colorbars should use seconds, based on `FS = 1000` unless the config changes the sample rate.

## Output Naming Contract

Keep stage names in derived outputs so filtering diagnostics are never confused with biomechanical processing results:

- `filtrado/s01_d01_t01.csv`: filtered data, original trial/calibration filename preserved for downstream discovery.
- `filter_analysis/s01_d01_t01_filter_Cell_1_spectrum.png`: filtering spectrum/PSD figure.
- `filter_analysis/s01_d01_t01_filter_spectrum_metrics.csv`: filtering spectrum metrics.
- `results_YYYYMMDD_HHMMSS/s01_d01_t01_processing_steps.csv`: biomechanical per-step output.
- `results_YYYYMMDD_HHMMSS/s01_d01_processing_metrics.csv`: subject-day biomechanical metrics.
- `figures_YYYYMMDD_HHMMSS/s01_d01_t01/processing_overview.png`: processing overview figure.
- `figures_YYYYMMDD_HHMMSS/s01_d01_t01/processing_cop_trajectory.png`: processing COP figure.
- `figures_YYYYMMDD_HHMMSS/s01_d01_t01/processing_cop_report_interactive.html`: processing interactive report.


## TOML Configuration

The TOML should remain readable and stage-oriented:

- `[pipeline]`: controls full pipeline execution.
- `[general]`: shared paths, file pattern, and sample rate.
- `[adjust]`: artifact marker, interval treatment, review behavior.
- `[interpolation]`: method comparison and interpolation defaults.
- `[filters]`: filter type, cutoff frequencies, median window, edge mode, and notch settings.
- `[processing]`: calibration, body weight handling, analysis window, step detection, negative-GRF clipping, and report generation.

Important defaults:

- `filter_type = "lowpass"`
- `lowpass_cutoff = 40.0`
- `edge_mode = "nearest"`
- `fs = 1000`
- `generate_interactive_report = true`
- `use_advanced_calibration = true`

## CLI Usage

Common command:

```bash
uv run python -m vaila.treadmill_lc --input-dir /path/to/csv_folder --step all
```

Common steps: `all`, `adjust`, `filter`, and `process`.

## Automated Tests

Run:

```bash
uv run pytest tests/test_treadmill_lc.py -v
```

The tests should cover:

- interval merging and interval normalization
- TOML defaults and compatibility loading
- selected-cell-only adjustment
- adjustment metadata sidecars
- interpolation helpers
- central calibration window
- COP geometry and plotting labels
- manual analysis-window normalization
- lightweight figure/report generation

## Development Notes

- Keep Tkinter as the GUI framework.
- Keep batch processing memory-conscious: close Matplotlib figures and avoid accumulating arrays or GUI windows.
- Do not delete old `results_*` or `figures_*` output folders.
- Keep per-attempt step files and per-day metrics files.
- Keep full-COP figures instead of per-strike COP images unless explicitly requested later.
- Update script metadata, README, help index, and module help when changing Python files, per `AGENTS.md`.
