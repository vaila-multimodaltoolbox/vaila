# Deadlift

Frame B button for `vaila/vaila_deadlift.py`.

The tool analyzes MediaPipe Pose CSV files for deadlift and RDL technique, including arm verticality, bar-over-midfoot setup, early pull synchronism, shin angle, spine deviation, and variant classification.

Run from GUI: **Frame B -> Deadlift**.

Run from CLI:

```bash
uv run python vaila/vaila_deadlift.py -i path/to/pose.csv -o path/to/output
```
