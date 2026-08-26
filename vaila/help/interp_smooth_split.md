# interp_smooth_split

## Module Information

- **Category:** Processing
- **File:** `vaila/interp_smooth_split.py` (+ shared core `vaila/interp_smooth_core.py`)
- **Version:** 0.3.115
- **Author:** Paulo R. P. Santiago
- **GUI:** yes | **CLI:** yes
- **Updated:** 26 August 2026

## Description

*vailá* tool for gap filling (interpolation), smoothing/filtering, optional final
resampling, and optional dataset splitting of CSV **and C3D** time-series used in
biomechanics.

GUI and CLI call the **same numerical core** (`interp_smooth_core`). The interactive
preview/tester stays GUI-only (CSV or C3D load for preview).

### Pipeline

```text
CSV  → validate → gap handling → smooth/filter → optional resample → output CSV
C3D  → marker CSV (c3d_markers_to_dataframe) → same pipeline → C3D (auto_create_c3d_from_csv)
```

C3D write-back preserves POINT labels, RATE, UNITS, occlusion residuals (NaN → negative
residual), and analog channels when present — same bridge as **Edit CSV/C3D**.
When Butterworth `fs` is unset for a C3D input, POINT RATE is used.

### Gap filling

`linear`, `cubic`, `nearest`, `kalman`, `none`, `skip`

### Smoothing

`none`, `savgol`, `lowess`, `kalman`, `butterworth`, `splines`, `arima`, `median`, `hampel`

### Sampling-rate semantics

| Concept | Meaning |
| --- | --- |
| Butterworth `fs` | Filter sampling frequency (Hz / FPS of the signal being filtered) |
| Time Column Sample Rate Override | Rebuild `Time = frame / rate` only; does **not** replace Butterworth `fs` |
| Original / Final rate | Optional final resampling grid |

### Index columns

Columns named `Time`/`t`/`tempo` and `frame`/`frames`/`frame_index` are treated as index
axes, never as signals: they are excluded from gap filling, smoothing, and resampling
interpolation regardless of their position in the file.

After resampling, index columns are rebuilt. CSV files whose first column is `frame` keep
that column first with integer values `0 … N−1` (no synthetic `Time` column is added).
Files that already have `Time` keep it on the resampled time grid; an existing `frame`
column is reset to `0 … N−1`.

Padding adds and removes exactly `int(rows × percent / 100)` rows on each edge, so the
output always has the same row count as the input when resampling is off.

Cutoff must satisfy `0 < cutoff < fs/2`. Invalid cutoffs raise errors; they are **not**
silently clamped.

Hz means samples per second; FPS means frames per second. Both describe sampling rate
for this tool.

### Derivatives

When a valid Time axis exists: `np.gradient(y, time)` (units / s and / s²).
When only a constant rate is known: use `dt = 1/fs`.
Otherwise: per-sample derivatives (not labeled as physical velocity/acceleration).

### Resampling

Optional stage **after** smoothing. Shared target Time grid for all numeric columns.
Downsampling can apply an explicit anti-alias Butterworth (`--no-antialias` to disable).
Upsampling interpolates the processed signal; it does **not** create new measured data.

### Automatic Butterworth recommendation (GUI)

Winter-style residual-RMS heuristic over candidate cutoffs. Displayed as a
**recommended configuration** (best candidate under that criterion), not as a unique optimum.

---

## How to run

### GUI

```bash
uv run vaila/interp_smooth_split.py
uv run vaila/interp_smooth_split.py --gui
```

### CLI

Precedence: **explicit CLI flags > TOML > defaults**.

### GUI action → CLI

| GUI action | CLI |
| --- | --- |
| Gap fill method | `--interp-method linear\|cubic\|nearest\|kalman\|none\|skip` |
| Max gap (frames) | `--max-gap N` |
| Smooth none | `--smooth-method none` |
| Savitzky-Golay | `--smooth-method savgol --window-length W --polyorder P` |
| LOWESS | `--smooth-method lowess --frac F --iterations I` |
| Kalman | `--smooth-method kalman` (+ TOML/params `n_iter`, `mode`) |
| Butterworth | `--smooth-method butterworth --fs FS --cutoff FC --filter-order O` |
| Splines | `--smooth-method splines --smoothing-factor S` |
| ARIMA | `--smooth-method arima --arima-p P --arima-d D --arima-q Q` |
| Moving median | `--smooth-method median --median-kernel K` |
| Hampel | `--smooth-method hampel --hampel-window W --hampel-sigma S` |
| Padding % | `--padding PCT` |
| Split dataset | `--split` |
| Time Column Sample Rate Override | `--time-column-rate HZ` |
| Enable final resampling | `--resample --original-rate R0 --final-rate R1` |
| Disable anti-alias | `--no-antialias` |
| Interactive preview / Winter / recommend | GUI only |

```bash
# Basic
uv run vaila/interp_smooth_split.py -i ./data

# Butterworth
uv run vaila/interp_smooth_split.py -i ./data --smooth-method butterworth \
    --fs 100 --cutoff 10 --filter-order 4

# Savitzky-Golay
uv run vaila/interp_smooth_split.py -i ./data --smooth-method savgol \
    --window-length 7 --polyorder 3

# Time-column rebuild (not Butterworth fs)
uv run vaila/interp_smooth_split.py -i ./data --time-column-rate 240

# Downsample
uv run vaila/interp_smooth_split.py -i ./data --resample \
    --original-rate 100 --final-rate 50

# Upsample
uv run vaila/interp_smooth_split.py -i ./data --resample \
    --original-rate 30 --final-rate 60

# TOML + override
uv run vaila/interp_smooth_split.py -i ./data -c ./smooth_config.toml --cutoff 8
```

---

## TOML (`smooth_config.toml`)

Sections: `[interpolation]`, `[smoothing]`, `[padding]`, `[split]`, `[time_column]`,
`[resample]` (`enabled`, `original_rate`, `final_rate`, `antialias`, `antialias_cutoff`).

---

## Tests

```bash
uv run pytest tests/test_interp_smooth_split.py -v
```

---

Part of *vailá* — Multimodal Toolbox  
https://github.com/vaila-multimodaltoolbox/vaila
