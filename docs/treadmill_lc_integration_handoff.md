# Treadmill LC Integration Handoff

Last updated: 2026-08-17

**Resume (Antigravity / any agent):** read `docs/sessions/2026-08-17-treadmill-lc-english.md` and `.claude/skills/treadmill-lc-continuation/SKILL.md`. Antigravity rule: `.agent/rules/treadmill-lc-continue.md`.

Handoff doc for treadmill load-cell processing integration in *vailá*. For future agents/maintainers. Covers what implemented now, workflow org, behavior must preserve.

English UI/COP labels = baseline. Portuguese `tara` / `peso` / `LIMPO` filenames + Borg `Peso` = accepted aliases only.

## Current State

Integration = **Frame B -> B6_r7_c3 - Treadmill LC** button + module `vaila/treadmill_lc.py`.

### Primary Files

- `vaila/treadmill_lc.py`: main module, GUI/CLI entry points + processing stages.
- `tests/test_treadmill_lc.py`: unit tests — calibration helpers, filtering config, interpolation helpers, COP layout, figure gen, window selection.
- `vaila/help/treadmill_lc.md` + `vaila/help/treadmill_lc.html`: user docs.
- `vaila.py`: main GUI entry, wired to Treadmill LC button.
- `README.md` + `vaila/help/index.*`: main doc entries.

## Workflow

Full pipeline order:

1. **Adjust + Interpolate**
2. **Filter**
3. **Process Metrics**

GUI exposes stages as separate buttons + one full-pipeline action. Normal GUI shouldn't show interpolation as standalone stage — adjustment + interpolation = single user workflow.

## File Discovery

Module must distinguish trials from calibration/metadata files by filename.

- Trials: `s*_d*_t*.csv`
- Tare calibration: `s*_d*_tare.csv` (legacy `tara` still accepted)
- Participant weight calibration: `s*_d*_weight.csv` (legacy `peso` still accepted)
- Plate-weight calibration: `s*_d*_*kg.csv`
- Borg metadata: `borg_*.txt`

Borg/info TXT files never processed as trials. When metadata file has `Weight` column (legacy `Peso` still works), weight used automatically. When `Trial` includes configured problem marker, e.g. `T02*`, review that trial at adjustment/interpolation stage.

## Stage Behavior

### Adjust + Interpolate

- Plot four load cells + summed signal.
- User selects affected load cell channels.
- User marks artifact START/END intervals.
- Undo/correction before save.
- Interval treatment modes: remove, `NaN`, zero, neutral mean, linear bridge.
- User compares up to four interpolation methods.
- User picks final method after visual compare.
- Preview shown, save only after approval.
- Approved CSVs save to stable `clean` by default, or `clean_YYYYMMDD_HHMMSS`
  with timestamped output enabled, w/ original trial name.
- Metadata saves as JSON, TOML, CSV sidecars — selected cells, intervals, treatment mode, selected interpolation methods, final method, interpolation params.

### Filter

- Default filter: low-pass Butterworth SOS @ 40 Hz.
- Median filtering: `scipy.ndimage.median_filter`, configurable `edge_mode`; default `nearest`.
- Zero-phase filtering: `sosfiltfilt`.
- Optional mains-notch filter: 50 Hz + 60 Hz.
- `filter_type` supports `lowpass`, `bandpass`, `highpass`, `median`, `none`.
- Batch GUI filtering previews one calibration file + one running trial, then applies same settings to rest w/o per-file plot windows.
- Filtered data keeps source filenames in stable `filtered` by default (or
  timestamped `filtered_YYYYMMDD_HHMMSS`); spectrum diagnostics save to
  `filter_analysis` w/ explicit `filter_` names.

### Process Metrics

- Group files by subject-day prefix (`sXX_dYY`).
- Discover matching calibration files + Borg metadata per group.
- Average all calibration files over central 5s only (avoid edge transients).
- Compute calibration once per subject-day group where possible.
- Support simple calibration from `tare` + `weight`.
- Support plate-weight calibration from `*kg.csv` files as complementary calibration points.
- GUI/CLI analysis windows use recorded TOML values when present; otherwise CLI
  indices or automatic first-impact detection are used. The foot-strike selector
  remains only for explicit legacy interactive calls.
- Save per-attempt step details as `*_processing_steps.csv`.
- Save one biomechanical metrics file per subject-day: `sXX_dYY_processing_metrics.csv`.
- Save processing outputs to stable `results` + `figures` directories by default;
  `--timestamp-output`/`-T` opts into timestamped directories.

## COP Convention

Load-cell layout fixed:

- Cell 1: anterior left
- Cell 2: posterior left
- Cell 3: anterior right
- Cell 4: posterior right

Distances:

- 58 cm left-right cells (mediolateral)
- 113 cm anterior-posterior cells (anteroposterior)

COP computed in cm, origin at treadmill center. `cop_x` = mediolateral, plot on horizontal axis. `cop_y` = anteroposterior, plot on vertical axis.

Generated COP figures:

- `processing_cop_trajectory.png`: full analyzed COP trajectory, not per-step COP.
- `processing_cop_report_interactive.html`: optional lightweight Plotly report — GRF, derivative, full COP.
- `processing_overview.png`: total GRF + first derivative over time (seconds).

All time axes/colorbars use seconds, based on `FS = 1000` unless config changes sample rate.

## Output Naming Contract

Keep stage names in derived outputs — filtering diagnostics never confused w/ biomechanical processing results:

- `filtered/s01_d01_t01.csv`: filtered data, original trial/calibration filename preserved for downstream discovery.
- `filter_analysis/s01_d01_t01_filter_Cell_1_spectrum.png`: filtering spectrum/PSD figure.
- `filter_analysis/s01_d01_t01_filter_spectrum_metrics.csv`: filtering spectrum metrics.
- `results/s01_d01_t01_processing_steps.csv`: biomechanical per-step output.
- `results/s01_d01_processing_metrics.csv`: subject-day biomechanical metrics.
- `figures/s01_d01_t01/processing_overview.png`: processing overview figure.
- `figures/s01_d01_t01/processing_cop_trajectory.png`: processing COP figure.
- `figures/s01_d01_t01/processing_cop_report_interactive.html`: processing interactive report.


## TOML Configuration

TOML stays readable, stage-oriented:

- `[pipeline]`: controls full pipeline execution.
- `[general]`: shared paths, file pattern, sample rate.
- `[adjust]`: artifact marker, interval treatment, review behavior.
- `[interpolation]`: method comparison + interpolation defaults.
- `[filters]`: filter type, cutoff freqs, median window, edge mode, notch settings.
- `[processing]`: calibration, body weight handling, analysis window, step detection, negative-GRF clipping, report generation.

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
uv run vaila/treadmill_lc.py --input-dir /path/to/csv_folder --step all
```

Common steps: `all`, `adjust`, `filter`, `process`.

## Automated Tests

Run:

```bash
uv run pytest tests/test_treadmill_lc.py -v
```

Tests should cover:

- interval merging + interval normalization
- TOML defaults + compatibility loading
- selected-cell-only adjustment
- adjustment metadata sidecars
- interpolation helpers
- central calibration window
- COP geometry + plotting labels
- manual analysis-window normalization
- lightweight figure/report generation

## Development Notes

- Keep Tkinter as GUI framework.
- Keep batch processing memory-conscious: close Matplotlib figures, avoid accumulating arrays/GUI windows.
- Stable stage folders are intentionally cleared and recreated on rerun; use
  `--timestamp-output`/`-T` when historical output folders must be preserved.
- Keep per-attempt step files + per-day metrics files.
- Keep full-COP figures instead of per-strike COP images unless explicitly requested later.
- Update script metadata, README, help index, module help when changing Python files, per `AGENTS.md`.
