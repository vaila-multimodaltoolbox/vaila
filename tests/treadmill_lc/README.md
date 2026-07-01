# Treadmill LC test data

This directory provides a test case and calibration files for `vaila/treadmill_lc.py`.

## Files

- `s01_d01_t01.csv` - Data of a running trial.
- `s01_d01_tara.csv` - Simple calibration tare file.
- `s01_d01_peso.csv` - Simple calibration participant weight file.
- `s01_d01_01kg.csv` to `s01_d01_35kg.csv` - Advanced calibration files (01, 05, 10, 15, 20, 25, 30, and 35 kg).
- `info_s01_d01.txt` - Subject information file containing the real body weight (61.6 kg) and trial metadata.

## CLI Usage Example

```bash
uv run python -m vaila.treadmill_lc --input-dir tests/treadmill_lc/ --step all
```

Outputs will be generated in processed subdirectories inside the target directory.

## See also

- [treadmill_lc.html](file:///home/labiocom-abel/Downloads/vaila/vaila/help/treadmill_lc.html)
- [treadmill_lc.md](file:///home/labiocom-abel/Downloads/vaila/vaila/help/treadmill_lc.md)
