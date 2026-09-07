# Pynalty

Frame B button (`B_r5_c5`) for `vaila/pynalty.py`.

Interactive penalty-kick analysis: mark goalkeeper move, contact, goal-line
crossing and goal corners; optional YOLO ball path, MediaPipe pose in athlete
bboxes, and goalkeeper anthropometrics for reach/time-to-ball verdicts.

Outputs (per clip): `data.toml`, `results.csv`, `pynalty_summary.csv`, EN/PT
HTML reports with ball-path animation, snapshots, optional pose CSVs, and an
appendable `pynalty_database.csv`.

Run from GUI: **Frame B → Pynalty**.

Run from CLI:

```bash
uv run vaila/pynalty.py -i video.mp4 -o out_dir
uv run vaila/pynalty.py -i video.mp4 -o out_dir -c data.toml --report-only
uv run vaila/pynalty.py -i video.mp4 -o out_dir --database db.csv \
  --auto-ball --pose --penalty-distance 11 --lang both
```

Modules: `pynalty_analysis.py` (DLT / flight / reach), `pynalty_vision.py`
(YOLO + MediaPipe), `pynalty_report.py` (HTML + CSV).

Help: `vaila/help/pynalty_help.md`.

---

**Last Updated:** 07 September 2026  
**Part of *vailá* - Multimodal Toolbox**  
**License:** AGPLv3.0
