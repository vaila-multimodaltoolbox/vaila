# Treadmill LC - Treadmill GRF Help

## Overview

The **Treadmill LC** tool processes instrumented treadmill load-cell data in a guided workflow for running analysis. It supports artifact adjustment with interpolation review, signal filtering, calibration, body-weight normalization, center of pressure (COP), step detection, per-step metrics, and subject-day summaries.

Use it from **Multimodal Analysis -> Treadmill LC**. The full workflow is:

1. **Adjust + Interpolate**
2. **Filter**
3. **Process Metrics**

The tool is TOML-configurable so the same settings can be reused for batch processing.

## Expected Files

The input folder can contain trial files, calibration files, and Borg metadata files. The tool separates these automatically by filename.

- **Running trials**: `s*_d*_t*.csv`
- **Tare calibration**: `s*_d*_tara.csv`
- **Participant weight calibration**: `s*_d*_peso.csv`
- **Plate-weight calibration**: `s*_d*_*kg.csv`, for example `s01_d01_20kg.csv`
- **Borg metadata**: `borg_*.txt`

Borg TXT files are not processed as trials. When a matching Borg file has a `Peso` column, the value is used as participant body weight. If the `Tent` value has the configured problem marker, such as `T02*`, the corresponding trial is flagged for adjustment/interpolation review.

## Processing Stages

### 1. Adjust + Interpolate

This stage is used to correct artifacts before filtering and metric extraction.

- Plot the four load cells and the summed signal.
- Select the affected load cell channels in a single multi-selection dialog, then mark intervals only on the selected cell plots.
- Mark one or more artifact intervals with START/END click pairs. The selected limits are redrawn as vertical dashed START/END lines, matching the processing-window selection style. ENTER is accepted only when every START has a matching END.
- Use right click to undo markings when available.
- Choose the interval treatment: remove segment, set to `NaN`, set to zero, neutral mean, or linear bridge.
- Compare up to four interpolation methods visually.
- Choose the final interpolation method.
- Approve the preview before saving. If rejected, the same file is reopened for correction.

A timestamped `clean_YYYYMMDD_HHMMSS` folder is created for each adjustment run so previous runs are not overwritten. Every running trial is written there with its original name (`sXX_dYY_tZZ.csv`): adjusted/interpolated trials contain the corrected signal, and trials without marked intervals are copied unchanged. Adjustment and interpolation metadata are saved beside the CSV as:

- `*_adjust_intervals.json`
- `*_adjust_intervals.toml`
- `*_adjust_intervals.csv`

The metadata records intervals, selected cells, interval treatment, selected interpolation methods, final method, and interpolation parameters.

### 2. Filter

This stage smooths the signal while preserving treadmill force behavior.

- Default filter: low-pass Butterworth SOS at 40 Hz.
- Median filtering uses `scipy.ndimage.median_filter` with configurable edge mode; default is `nearest`.
- Zero-phase filtering uses `sosfiltfilt`.
- Optional mains-noise notch filtering supports 50 Hz and 60 Hz power grids.
- Available `filter_type` values: `lowpass`, `bandpass`, `highpass`, `median`, and `none`.

During batch filtering, the GUI previews one calibration file and one running file. After approval, the same filter settings are applied to the remaining files without opening a plot for every file.

Filtered running CSV files are saved inside `filtered_YYYYMMDD_HHMMSS` with the canonical `sXX_dYY_tZZ.csv` name, even if the input folder contains a legacy `*_LIMPO.csv` or `*_clean.csv` file. Calibration files keep their calibration names. Frequency diagnostics are saved to `filter_analysis_YYYYMMDD_HHMMSS` with explicit `filter_` names, such as `s01_d01_t01_filter_spectrum_metrics.csv` and `s01_d01_t01_filter_Cell_1_spectrum.png`.

### 3. Process Metrics

This stage calibrates the four load-cell signals and computes running metrics.

- Uses matching calibration files from the same subject-day group (`sXX_dYY`).
- Uses the central 5 seconds of each calibration recording to avoid edge transients.
- Supports simple calibration using `tara` and `peso`.
- Supports plate-weight calibration using `*kg.csv` files as complementary calibration points.
- Uses Borg `Peso` automatically when available, with TOML/manual value as fallback.
- Allows manual analysis-window selection. Click START and optional END, then press Enter. One click means START to the final sample. Invalid selections reopen the same file.
- Detects steps and saves temporal, force, impulse, loading-rate, COP, and asymmetry metrics.

All pipeline output stages use timestamped folders so previous runs are not overwritten. Processing outputs are saved to:

- `results_YYYYMMDD_HHMMSS`
- `figures_YYYYMMDD_HHMMSS`

Step details are saved per attempt as `*_processing_steps.csv`. Biomechanical metrics are consolidated once per subject-day as `sXX_dYY_processing_metrics.csv`.

## COP Convention

The load-cell layout is interpreted as:

- Cell 1: superior left
- Cell 2: inferior left
- Cell 3: superior right
- Cell 4: inferior right

Distances:

- Left-right distance: 58 cm
- Anterior-posterior distance: 113 cm

COP is calculated in centimeters with origin at the center of the treadmill. In the figures, **COP X** is medio-lateral and is shown on the horizontal axis. **COP Y** is anterior-posterior and is shown on the vertical axis. The COP trace represents the center of load/contact over the instrumented deck; it is not belt displacement and should not be interpreted as stride length along the treadmill.

## Figures

Each processed attempt can save:

- `processing_overview.png`: total GRF, detected support regions, peaks, and first derivative over time in seconds.
- `processing_strike_attributes.png`: original-inspired strike panels for representative steps, marking peak force, transient, max loading-slope point, derivative, and colored impulse regions.
- `processing_stride_map.png`: full detected support regions plus all strikes normalized to 0-100% support for visual inspection of consistency and asymmetry.
- `processing_cop_trajectory.png`: full analyzed COP contact-load trajectory in centimeters, with time shown by color, fixed 58 x 113 cm deck limits, and the four load-cell positions.
- `processing_cop_report_interactive.html`: optional tugturn-style interactive report with GRF, derivative, deck geometry, load-cell positions, and full COP trajectory.

All time axes and time colorbars use seconds, based on the configured sample rate (`fs`, default `1000 Hz`).

## Output Naming

Output names include the stage whenever the file contains derived diagnostics or metrics. Signal stages treat `sXX_dYY_tZZ.csv` as the standard running-trial name. Legacy `sXX_dYY_tZZ_LIMPO.csv` files are still accepted as input, but the current adjustment and filtering stages write corrected, unchanged, and filtered trials with the canonical original trial name so the next stage sees one homogeneous set. Sidecar CSV files such as `*_adjust_intervals.csv`, `*_filter_spectrum_metrics.csv`, and `*_processing_steps.csv` are ignored by later stages.

- Adjusted/interpolated data for the next stage: `clean_YYYYMMDD_HHMMSS/s01_d01_t01.csv`
- Filtered data for the next stage: `filtered_YYYYMMDD_HHMMSS/s01_d01_t01.csv`
- Filtering spectrum figure: `filter_analysis_YYYYMMDD_HHMMSS/s01_d01_t01_filter_Cell_1_spectrum.png`
- Filtering spectrum metrics: `filter_analysis_YYYYMMDD_HHMMSS/s01_d01_t01_filter_spectrum_metrics.csv`
- Processing step table: `results_YYYYMMDD_HHMMSS/s01_d01_t01_processing_steps.csv`
- Processing daily metrics: `results_YYYYMMDD_HHMMSS/s01_d01_processing_metrics.csv`
- Processing figures: saved directly in the figures folder with trial prefix (e.g. `figures_YYYYMMDD_HHMMSS/s01_d01_t01_processing_overview.png`, `s01_d01_t01_processing_strike_attributes.png`, `s01_d01_t01_processing_stride_map.png`, `s01_d01_t01_processing_cop_trajectory.png`, and `s01_d01_t01_processing_cop_report_interactive.html`)


## TOML Configuration

The GUI can create or load a TOML file before processing. The main sections are:

- `[pipeline]`: controls full-pipeline execution.
- `[general]`: shared paths, file patterns, and sample rate.
- `[adjust]`: artifact marking, metadata marker, and interval treatment.
- `[interpolation]`: method comparison and noninteractive interpolation defaults.
- `[filters]`: median window, filter type, cutoff frequencies, edge mode, and notch settings.
- `[processing]`: calibration, body weight, optional processing filter, analysis window, legacy valley/cut step detection, negative-GRF clipping, and report options.

For batch processing, create a TOML once, review it in the GUI editor, then reuse it for folders collected under the same conditions.

## GUI Usage

Open **Multimodal Analysis -> Treadmill LC** and choose:

- **Run Full Pipeline**: Adjust + Interpolate -> Filter -> Process Metrics.
- **Adjust + Interpolate**: Only artifact correction and interpolation review.
- **Filter Only**: Only filtering and frequency diagnostics.
- **Process Metrics Only**: Only calibration and running metrics.
- **Create TOML Template**: Save a reusable configuration file.
- **Help**: Open this documentation.

## CLI Usage

Run the tool from a terminal:

```bash
uv run python -m vaila.treadmill_lc --input-dir /path/to/csv_folder --step all
```

Common `--step` values are `all`, `adjust`, `filter`, and `process`.

---
- **Version**: 0.3.68
- **Updated**: 02 July 2026
