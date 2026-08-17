# Treadmill LC test data

This directory provides a test case and calibration files for `vaila/treadmill_lc.py`.

## Files

- `s01_d01_t01.csv` - Data of a running trial.
- `s01_d01_tare.csv` - Simple calibration tare file.
- `s01_d01_weight.csv` - Simple calibration participant weight file.
- `s01_d01_01kg.csv` to `s01_d01_35kg.csv` - Advanced calibration files (01, 05, 10, 15, 20, 25, 30, and 35 kg).
- `info_s01_d01.txt` - Subject information file containing the real body weight (61.6 kg) and trial metadata (`Subject,Day,Trial,BORG,Speed,Weight`).
- `processing_configuration_used.toml` - Unified process-stage configuration fixture.

## CLI Usage Example

```bash
uv run vaila/treadmill_lc.py --input-dir tests/treadmill_lc/ --step all

# Ignore TOML, detect the first large impact peak, and use the CLI weight:
uv run vaila/treadmill_lc.py --input-dir tests/treadmill_lc/ --step process --force-auto --weight 61.6
```

Without `--output-dir`, outputs are generated below `tests/treadmill_lc/output/`
using stable `clean`, `adjusted`, `filtered`, `filter_analysis`, `figures`, and
`results` directories. Re-running overwrites those stage directories. Add
`-T`/`--timestamp-output` to keep separate timestamped folders.
GUI and CLI runs write an identical stable `treadmill_lc_run_history.toml` to
the input directory and output root. After a GUI run, the history also contains artifact
intervals, the final interpolation method, and analysis windows. The terminal's
final highlighted `>> vaila/treadmill_lc: Equivalent CLI` command can therefore
be pasted to repeat the run without reopening the GUI.

Without `--force-auto`, resolution is CLI index/weight first, then TOML, then
automatic start detection and the existing info/Borg weight fallback. A missing
fallback TOML prints a warning and does not abort processing.

## See also

- [treadmill_lc.html](../../vaila/help/treadmill_lc.html)
- [treadmill_lc.md](../../vaila/help/treadmill_lc.md)
