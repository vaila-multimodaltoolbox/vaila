# TUGTURN test data

This directory provides a compact CLI test case for `vaila/tugturn.py`.

## Files

- `s26_m1_t1.csv`
- `s26_m1_t1.toml`
- `s26_m1_t1.txt`
- `skeleton_pose_mediapipe.json`

## Quick check (without `-o`)

```bash
uv run vaila/tugturn.py -i tests/tugturn/s26_m1_t1.csv -c tests/tugturn/s26_m1_t1.toml -y 0.15
```

In CLI mode, when `-o` is omitted, output is auto-created next to the input CSV:

- `tests/tugturn/result_tugturn_s26_m1_t1/`

See also:

- `vaila/help/tugturn.html`
- `vaila/help/tugturn.md`
