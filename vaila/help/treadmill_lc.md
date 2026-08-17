# Treadmill LC - Treadmill GRF Help

## Overview

The **Treadmill LC** tool processes instrumented treadmill load-cell data in a robust, standardized workflow for running biomechanics analysis. It supports artifact adjustment with interpolation review, signal filtering, calibration, body-weight normalization, automatic signal-start detection, center of pressure (COP), step detection, per-step metrics, and subject-day summaries.

Use it from **Frame B: Multimodal Analysis -> B_B_r5_c2 - Treadmill LC**.

## Repeat a GUI run from the terminal

The GUI follows the same GUI→CLI mirror convention as `sam3sapiens2.py`.
When you click **Run Full Pipeline** or a stage button, the terminal immediately
prints:

1. a Rich table with the effective paths, step, filters, interpolation, and
   processing parameters;
2. the two saved run-history TOMLs (one in the input directory and one in the
   output root);
3. a bold-yellow `>> vaila/treadmill_lc: Equivalent CLI` banner containing a
   shell-quoted, copy/paste command.

The banner is printed again when the GUI run finishes. **Use the final banner**:
by then its TOML has also recorded choices made in interactive windows,
including artifact intervals, the approved interpolation method for each trial,
and each trial's analysis window. The replay therefore does not reopen Tkinter
dialogs.

```text
====================================================================================================
>> vaila/treadmill_lc: Equivalent CLI for the completed GUI run (copy/paste):
>>   uv run vaila/treadmill_lc.py --config '/data/run/treadmill_lc_run_history.toml' --input-dir '/data/run' --output-dir '/data/run/output' --step all
====================================================================================================
```

Copy the command beginning with `uv run`. Paths containing spaces are quoted
automatically. Redirected logs and `NO_COLOR=1` keep the banner as plain text,
without ANSI escape codes.

### 5-Stage Logical Calibration & Biomechanics Pipeline

The processing pipeline strictly adheres to the following sequence:

1. **Zero Offset**: Baseline tare voltage subtraction: $V_{\text{offset\_corrected}} = V_{\text{raw}} - \bar{V}_{\text{tare}}$.
2. **Calibration Matrix**: Gain and scaling transformation to Body Weight units: $F_{\text{cells}} = (m \cdot V_{\text{cells}} + b) / W_{\text{kg}}$, $F_{\text{vertical}} = \sum F_{\text{cells}}$.
3. **Coordinate Transformation**: Conversion using deck geometry (58.0 cm Mediolateral $\times$ 113.0 cm Anteroposterior) to standard English biomechanics channels: `medial_lateral` (COP X), `anterior_posterior` (COP Y), and `vertical` (GRF total).
4. **Signal Filtering**: Zero-phase Butterworth low-pass filtering on all physical channels.
5. **Event Detection**: Support strike segmentation with a single **`sidefoot`** column:
   - `sidefoot: 0` = Right
   - `sidefoot: 1` = Left

---

## Expected Files

The input folder can contain trial files, calibration files, and Borg metadata files. The tool separates these automatically by filename:

- **Running trials**: `s*_d*_t*.csv`
- **Tare calibration**: `s*_d*_tare.csv` (legacy `s*_d*_tara.csv` still accepted)
- **Participant weight calibration**: `s*_d*_weight.csv` (legacy `s*_d*_peso.csv` still accepted)
- **Plate-weight calibration**: `s*_d*_*kg.csv`, for example `s01_d01_20kg.csv`
- **Borg metadata**: `borg_*.txt` or `info_*.txt`

Borg/info TXT files are not processed as trials. When a matching metadata file has a `Weight` column (legacy `Peso` is still accepted), the value is used as participant body weight unless `--weight` or an explicit TOML `weight` overrides it. If the `Trial` value has the configured problem marker, such as `T02*`, the corresponding trial is flagged for adjustment/interpolation review.

---

## Processing Stages

### 1. Filter (Zero-Phase + PSD)

This stage smooths the signal while preserving treadmill force behavior.

- Default filter: low-pass Butterworth SOS at 40 Hz (configurable via TOML).
- Median filtering uses `scipy.ndimage.median_filter` with configurable edge mode; default is `nearest`.
- Zero-phase filtering uses `sosfiltfilt`.
- Optional mains-noise notch filtering supports 50 Hz and 60 Hz power grids.
- Available `filter_type` values: `lowpass`, `bandpass`, `highpass`, `median`, and `none`.

Filtered running CSV files are saved inside stable `filtered` with the canonical
`sXX_dYY_tZZ.csv` name. Frequency diagnostics are saved to stable
`filter_analysis` with explicit `filter_` names, such as
`s01_d01_t01_filter_spectrum_metrics.csv` and
`s01_d01_t01_filter_Cell_1_spectrum.png`. Both become timestamped when `-T` is
enabled.

### 2. Adjust + Interpolate

This stage is used to correct artifacts after filtering and before metric extraction.

- Plots the four load cells and the summed signal.
- Interactive click-based marking of artifact intervals.
- Interval treatments: remove segment, set to `NaN`, set to zero, neutral mean, or linear bridge.
- Multi-method interpolation preview and approval.
- Sidecar metadata is stored in `.json`, `.toml`, and `.csv`.

### 3. Process Metrics

This stage executes the 5-stage calibration pipeline and extracts spatial, temporal, kinetic, and asymmetry metrics:

- Uses matching calibration files from the same subject-day group (`sXX_dYY`).
- Uses the central 5 seconds of each calibration recording to avoid edge transients.
- Computes spatial COP metrics using standard English keys: `cop_medial_lateral_mean`, `cop_anterior_posterior_mean`, `cop_medial_lateral_range`, `cop_anterior_posterior_range`, `cop_anterior_posterior_initial`, `cop_anterior_posterior_final`.
- Detects steps with a single `sidefoot` column: `0` (Right) and `1` (Left).
- Computes Asymmetry Index (`*_ASI`), right mean/std (`*_mean_R`, `*_std_R`), and left mean/std (`*_mean_L`, `*_std_L`).
- In headless mode, resolves each analysis window from CLI indices, then the
  trial's TOML window, then automatic detection. Automatic start is the first
  peak returned by `scipy.signal.find_peaks` whose height reaches 50% of the
  signal maximum; automatic end is the signal length.

Outputs:
- `results/sXX_dYY_tZZ_processing_steps.csv`
- `results/sXX_dYY_processing_metrics.csv`
- `figures/` overview, COP trajectory, strike attributes, stride map, and interactive HTML report.

Stage folders are stable by default: `clean`, `adjusted`, `filtered`,
`filter_analysis`, `figures`, and `results` are cleared and recreated on a
rerun with the same output path. Enable **Timestamped output folders** in the
GUI or pass `--timestamp-output`/`-T` to preserve every run in new timestamped
directories. When
**Output Dir**, `--output-dir`, and `paths.output_dir` are empty, both GUI and
CLI create `<input-dir>/output/`. This shared resolution is what makes the
printed GUI command write to the same place as the original run.

---

## TOML configuration and run history

A unified `.toml` configures paths, selected step, filtering, interpolation,
and processing:

```toml
weight = 72.0 # optional explicit participant weight

[paths]
input_dir = "tests/treadmill_lc"
output_dir = "tests/treadmill_lc/output"

[execution]
step = "process"
timestamp_output = false

[filters]
median_window = 5
filter_type = "lowpass"
lowpass_cutoff = 40.0
bandpass_lowcut = 0.0
bandpass_highcut = 40.0
filter_order = 4
edge_mode = "nearest"

[interpolation]
max_comparison_methods = 4
spline_order = 3
rbf_window_size = 200

[processing]
participant_weight_kg = 70.0
use_advanced_calibration = true
filter_cutoff_hz = 50.0
apply_processing_filter = false
detection_threshold_bw = 0.1
generate_figures = true
generate_interactive_report = true
```

Every GUI or CLI run overwrites a single
`treadmill_lc_run_history.toml` at the input directory and the same filename at
the output root. Stable stage folders are reused by default; timestamped stage
folders are created only when `--timestamp-output`/`-T` (or the GUI checkbox)
is enabled. Runs do not accumulate extra history copies.

After a GUI run, generated replay sections are appended to the same history:

```toml
[adjustments.s01_d01_t01]
adjustment_mode = "nan"
processed = true

[adjustments.s01_d01_t01.interpolation]
status = "adjusted_and_interpolated"
final_method = "pchip"
spline_order = 3
rbf_window_size = 200

[[adjustments.s01_d01_t01.intervals]]
start_index = 1250
end_index_exclusive = 1390
cells_0based = [0, 2]

[analysis_windows.s01_d01_t01]
start_index = 5000
end_index_exclusive = 45000
source_samples = 60000
start_source = "gui"
end_source = "gui"
```

These sections reproduce GUI-only decisions headlessly. They normally should
not be edited manually.

---

## CLI Usage & Headless Mode

CLI execution does not open GUI windows. Rich tables and milestones print the
effective parameters, paths, stage, completion status, and replay command.

```bash
# Fully automatic process run without reading TOML. The CLI weight also
# overrides a Weight/Peso value found in info/Borg metadata:
uv run vaila/treadmill_lc.py -i path/to/csvs -o path/to/results -s process -a -w 72.0

# Override a window manually while keeping all other TOML settings:
uv run vaila/treadmill_lc.py -i path/to/csvs -s process -b 526 -e 60000

# Keep a separate timestamped copy instead of reusing stable stage folders:
uv run vaila/treadmill_lc.py -i path/to/csvs -s all -T

# Best option: paste the final command printed by GUI Run:
uv run vaila/treadmill_lc.py -c path/to/treadmill_lc_run_history.toml \
  -i path/to/csvs -o path/to/results -s all

# Build a new full run manually:
uv run vaila/treadmill_lc.py -c path/to/config.toml -i path/to/raw_csvs -o path/to/results -s all

# Run a specific stage; output defaults to path/to/csvs/output/:
uv run vaila/treadmill_lc.py -i path/to/csvs -s filter

# Generated histories already contain paths and step, so this shorter form also works:
uv run vaila/treadmill_lc.py -c path/to/treadmill_lc_run_history.toml

# Open the single-window launcher GUI:
uv run vaila/treadmill_lc.py -g
```

Short aliases are also available: `-i/-o` for input/output, `-c` or `-t` for
config/TOML path, `-s` for step, `-w` for weight, `-b/-e` for start/end,
`-a` for automatic mode, `-T` for timestamped output, and `-g` for GUI. The
long forms remain accepted. Both `uv run vaila/treadmill_lc.py --help` and
`uv run vaila/treadmill_lc.py -h` show the complete current CLI.

After an input directory is selected, the GUI displays the matching subject's
`Weight`/`Peso` from `info_*.txt` or `borg_*.txt`. If no metadata is available,
it labels the existing fallback instead of silently presenting it as the
subject's weight.

The main GUI is only the launcher. Its Run buttons execute the loaded TOML in
the same non-interactive mode as CLI: recorded adjustment and analysis-window
choices are reused, no foot-strike selection figure is opened, and stages do
not pause for preview, confirmation, or success message boxes. Progress and
errors remain visible in the GUI log and terminal.

`--toml-path` defaults to `treadmill_lc_run_history.toml`, resolved inside the
input directory. For backward compatibility, the CLI next tries
`processing_configuration_used.toml`. If neither exists, it prints a warning,
uses the built-in settings, detects each start automatically, and continues.
`--config` remains the strict replay option: an explicitly requested missing
config is an error. `--force-auto` ignores all TOML settings; supply
`--input-dir` with it.

### CLI/TOML priority

| Value | Resolution order |
|---|---|
| Participant weight | `--weight` → TOML `weight` → matching info/Borg `Weight`/`Peso` → existing 70 kg fallback |
| Start index | `--start-index` → `[analysis_windows.<trial>]` → first large impact peak |
| Exclusive end index | `--end-index` → `[analysis_windows.<trial>]` → total signal samples |

CLI and automatically detected windows are written back to both stable run
histories with `start_source` and `end_source`, so the next run is reproducible.

### GUI button to CLI step mapping

| GUI action | Printed `--step` | Headless replay |
|---|---:|---|
| Run Full Pipeline | `all` | Filter → adjustment replay → interpolation compatibility pass → metrics |
| Filter Only | `filter` | Applies the recorded filter settings |
| Adjust + Interp | `adjust` | Replays recorded intervals and final interpolation method |
| Process Only | `process` | Reuses each recorded analysis window |

If a manual CLI config has no recorded adjustment or analysis-window sections,
the run safely applies no interactive artifact edits and detects the analysis
start automatically.

---
- **Version**: 0.3.107
- **Updated**: 17 August 2026
