# emg_labiocom

EMG (Electromyography) analysis toolkit — part of the **vailá** Multimodal Toolbox.

## Module Information

| Field | Value |
|---|---|
| Category | Analysis |
| File | `vaila/emg_labiocom.py` |
| Version | **0.3.50** |
| Updated | **2026-06-09** |
| Author | Prof. Dr. Paulo R. P. Santiago |
| GUI | Yes |
| CLI | Yes |
| Help (HTML) | [emg_labiocom.html](emg_labiocom.html) |

## Overview

This module analyzes surface EMG (sEMG) signals with time-domain,
frequency-domain and time-frequency-domain techniques. It produces CSV
results, PNG / SVG figures, a statistical summary, and an optional
HTML report.

The reader **auto-detects** several common CSV layouts so you don't
need to pre-format your files.

## Supported CSV input formats

The smart reader (`_read_emg_csv_smart`) tries to recognise the file
layout from the relationship between the header field count and the
first data row field count.

### 1. Standard CSV (Anglo-Saxon decimal)

```
time,EMG_volts
0.0000,0.000123
0.0005,-0.000087
...
```

- `.` decimal, `,` field separator
- Any number of columns; first matching `EMG` or `Volt`-named column
  is auto-selected. A specific column can be forced with `--channel`.

### 2. Trigno / Delsys EMGworks export (European decimal)

```
X_s,Trigno_sensor_14:_EMG_14_Volts,X_s,Trigno_sensor_14:_Acc_14.X_g,...
0,00000e+000,0,00000e+000,0,00000e+000,0,00000e+000,...
5,19231e-004,0,00000e+000,6,75000e-003,0,00000e+000,...
```

In this format the comma is **both** the decimal separator and the CSV
field separator. The header has `N` fields but every data row has `2N`
fields. The reader rebuilds the values pairwise (`"5"` + `"19231e-004"`
→ `"5.19231e-004"`) and then parses the buffer normally.

Multi-channel files (EMG + accelerometer interleaved) are supported:
the EMG channel is auto-picked by name and accelerometer / gyro
channels are ignored.

### 3. Generic multi-column CSV

Pass `--channel <name|substring|index>` to select an arbitrary column.

## Sample rate

The sample rate `fs` is **auto-estimated** from the time column as
`1 / median(Δt)`. Override with `--fs <Hz>`. Common defaults:

| System | Typical fs (Hz) |
|---|---|
| Trigno EMG | 1925.93 |
| EMGworks Delsys | 2000 |
| Biopac | 1000 – 2000 |
| Noraxon | 1500 – 2000 |

## CLI usage

```bash
# GUI (no arguments)
uv run vaila/emg_labiocom.py

# Inspect a file without running analysis
uv run vaila/emg_labiocom.py --inspect /path/to/file.csv

# Single file, full signal, headless, with HTML report
uv run vaila/emg_labiocom.py \
    -i /home/preto/Desktop/aula_fim/emg_aula_26.csv \
    -o ./results --full --no-plot --report

# Batch directory, force fs=2000 Hz, force channel by substring
uv run vaila/emg_labiocom.py -i ./emg_dir -o ./out --fs 2000 -c EMG_14 --full

# Explicit segments (sample indices)
uv run vaila/emg_labiocom.py -i file.csv -o out -s "1000,5000;10000,15000"

# Data already in microVolts (skip default Volts → µV scaling)
uv run vaila/emg_labiocom.py -i file.csv -o out --unit-scale 1.0 --full
```

### CLI flags

| Flag | Purpose |
|---|---|
| `-i`, `--input` | Input CSV/TXT file or directory of CSV/TXT files. |
| `-o`, `--output` | Output directory (a `emg_labiocom_<timestamp>/` is created inside). |
| `--fs` | Sample rate (Hz). Defaults to auto-detected from time column. |
| `-c`, `--channel` | Column name, case-insensitive substring, or 0-based index. |
| `-s`, `--selections` | Explicit segments: `"start1,end1;start2,end2;..."`. |
| `--full` | Analyse the whole signal as one segment (overrides `--selections`). |
| `--report` | Render HTML report per file + `index.html`. |
| `--no-plot` | Headless: no interactive plots (PNG / SVG still saved). |
| `--unit-scale` | Signal multiplier (default `1e6` Volts → µV; use `1.0` if already µV). |
| `--inspect FILE` | Print detected layout for `FILE` and exit. |
| `--gui` | Force GUI mode. |

## GUI usage

Launch from `vaila.py` → Frame B → **EMG Analysis**, or directly:

```bash
uv run vaila/emg_labiocom.py
```

GUI flow:

1. Select an **input directory** containing CSV/TXT EMG files.
2. Select a **reference EMG file** used to define the segments
   (segments are reused for every file in the directory).
3. The reader auto-detects the layout and pre-fills:
   - the EMG channel name (any column containing `EMG` or `Volt`),
   - the sample rate from the time column.
4. Confirm or change the **EMG channel** (substring match).
5. Confirm or change the **sample rate**.
6. Choose **selection mode**: *Interactive* (mouse) or *Manual*
   (`start,end;start,end;...` text input).
7. Choose whether to **show plots** during analysis.
8. Choose whether to **generate HTML reports**.
9. Select the **output directory**.

### Mouse controls (Interactive mode)

| Action | Effect |
|---|---|
| Left click | Set segment start |
| Shift + Left click | Set segment end (commits segment) |
| Right click | Remove last segment |
| `Esc` | Cancel current start point |
| **Clear All** / **Done** buttons | Reset / confirm and close |

## Analyses computed per segment

- Band-pass filter (Butterworth 4th-order, 10 – 450 Hz)
- Full-wave rectification + linear envelope (10 Hz low-pass)
- Root-mean-square (250 ms window, 50 % overlap) with polynomial fit
- Median frequency (Welch PSD) with polynomial fit
- Power spectral density (Welch) with peak frequency
- Spectrogram (STFT)
- Continuous wavelet transform (Morlet) — when `PyWavelets` is installed
- Time-domain features: ZCR, MAV, VAR, WL, Wilson amplitude, slope-sign changes
- Frequency-domain features: mean / peak frequency, spectral variance,
  band powers (20-50 / 50-100 / 100-150 Hz), power ratio, MMFP
- Fatigue indices: classic FI, modified FI, Dimitrov FI, MF slope
- Statistical features: skewness, kurtosis, Hjorth mobility & complexity

## Outputs

For each input file, a subdirectory `<filename>/` is created inside
the timestamped output root, containing:

```
emg_labiocom_<YYYYMMDD_HHMMSS>/
├── index.html                                  # links to all reports (when --report)
└── <filename>/
    ├── <filename>_results_emg_labiocom.csv     # per-window results
    ├── <filename>_statistical_summary.csv      # summary stats
    ├── <filename>_segment_<k>_filtered_emg.{png,svg}
    ├── <filename>_segment_<k>_rectified_emg.{png,svg}
    ├── <filename>_segment_<k>_rms.{png,svg}
    ├── <filename>_segment_<k>_median_frequency.{png,svg}
    ├── <filename>_segment_<k>_pwelch.{png,svg}
    ├── <filename>_segment_<k>_spectrogram.{png,svg}
    ├── <filename>_segment_<k>_wavelet.{png,svg}      # when PyWavelets is available
    └── <filename>_emg_report.html              # when --report
```

## Programmatic API

```python
from vaila.emg_labiocom import (
    _read_emg_csv_smart,     # raw DataFrame loader
    _select_emg_channel,     # (time, emg, t_col, e_col, fs_est)
    _load_emg_signal,        # (emg_microvolts, fs_est, e_col)
    emg_analysis,            # full pipeline on one file
    run_emg_cli,             # argparse entry point
    run_emg_gui,             # Tkinter entry point
)

df = _read_emg_csv_smart("emg_aula_26.csv")
time, emg, t_col, e_col, fs = _select_emg_channel(df)
print(t_col, e_col, fs)
```

## Troubleshooting

| Symptom | Likely cause | Fix |
|---|---|---|
| `Channel 'X' not found` | Channel name typo. | Use `--inspect FILE` to list columns. |
| Signal looks flat / too small | File is already in microVolts. | `--unit-scale 1.0`. |
| `fs` printed as `1923` instead of `2000` | Trigno true rate is ~1925.93 Hz. | OK, accept auto-detected value or pass `--fs 2000`. |
| Segment too short warnings | Segment < 2 s. | Pick longer segments or use `--full`. |
| Empty output / no figures | Channel selected has all NaNs. | Pass `--channel` explicitly. |

---

Generated 2026-06-09. Part of [vailá Multimodal Toolbox](https://github.com/vaila-multimodaltoolbox/vaila).
