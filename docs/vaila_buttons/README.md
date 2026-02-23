# vailá GUI Buttons Documentation

This directory contains documentation for all buttons in the vailá GUI (`vaila.py`).

## Structure

Each button in the vailá GUI has its own documentation file:
- **Markdown format** (`.md`) - For easy editing and version control
- **HTML format** (`.html`) - For web viewing (to be generated)

## Button List

| Position | Button Text | Method | Documentation |
|----------|-------------|--------|---------------|
| A_r1_c1 | Rename | `rename_files` | [rename-files.md](rename-files.md) |
| A_r1_c2 | Import | `import_file` | [import-file.md](import-file.md) |
| A_r1_c3 | Export | `export_file` | [export-file.md](export-file.md) |
| A_r1_c4 | Copy | `copy_file` | [copy-file.md](copy-file.md) |
| A_r1_c5 | Move | `move_file` | [move-file.md](move-file.md) |
| A_r1_c6 | Remove | `remove_file` | [remove-file.md](remove-file.md) |
| A_r1_c7 | Tree | `tree_file` | [tree-file.md](tree-file.md) |
| A_r1_c8 | Find | `find_file` | [find-file.md](find-file.md) |
| A_r1_c9 | Transfer | `transfer_file` | [transfer-file.md](transfer-file.md) |
| B_r1_c1 | Select YOLO tracker version to use: | `imu_analysis` | [imu-analysis.md](imu-analysis.md) |
| B_r1_c2 | Select YOLO tracker version to use: | `cluster_analysis` | [cluster-analysis.md](cluster-analysis.md) |
| B_r1_c3 | Select YOLO tracker version to use: | `mocap_analysis` | [mocap-analysis.md](mocap-analysis.md) |
| B_r1_c4 | Select YOLO tracker version to use: | `markerless_2d_analysis` | [markerless-2d-analysis.md](markerless-2d-analysis.md) |
| B_r1_c5 | Select YOLO tracker version to use: | `markerless_3d_analysis` | [markerless-3d-analysis.md](markerless-3d-analysis.md) |
| B_r2_c1 | Select YOLO tracker version to use: | `vector_coding` | [vector-coding.md](vector-coding.md) |
| B_r2_c2 | Select YOLO tracker version to use: | `emg_analysis` | [emg-analysis.md](emg-analysis.md) |
| B_r2_c3 | Select YOLO tracker version to use: | `force_analysis` | [force-analysis.md](force-analysis.md) |
| B_r2_c4 | Select YOLO tracker version to use: | `gnss_analysis` | [gnss-analysis.md](gnss-analysis.md) |
| B_r2_c5 | Select YOLO tracker version to use: | `eeg_analysis` | [eeg-analysis.md](eeg-analysis.md) |
| B_r3_c1 | Select YOLO tracker version to use: | `hr_analysis` | [hr-analysis.md](hr-analysis.md) |
| B_r3_c2 | Select YOLO tracker version to use: | `cube2d_kinematics` | [cube2d-kinematics.md](cube2d-kinematics.md) |
| B_r3_c3 | Select YOLO tracker version to use: | `vailajump` | [vailajump.md](vailajump.md) |
| B_r3_c4 | Select YOLO tracker version to use: | `markerless2d_mpyolo` | [markerless2d-mpyolo.md](markerless2d-mpyolo.md) |
| B_r3_c5 | Select YOLO tracker version to use: | `animal_open_field` | [animal-open-field.md](animal-open-field.md) |
| B_r4_c1 | Select YOLO tracker version to use: | `tracker` | [tracker.md](tracker.md) |
| B_r4_c2 | Which conversion would you like to perform? | `ml_walkway` | [ml-walkway.md](ml-walkway.md) |
| B_r4_c3 | Which conversion would you like to perform? | `markerless_hands` | [markerless-hands.md](markerless-hands.md) |
| B_r4_c4 | Which conversion would you like to perform? | `mp_angles_calculation` | [mp-angles-calculation.md](mp-angles-calculation.md) |
| B_r4_c5 | Which conversion would you like to perform? | `markerless_live` | [markerless-live.md](markerless-live.md) |
| B_r5_c1 | Which conversion would you like to perform? | `ultrasound` | [ultrasound.md](ultrasound.md) |
| B_r5_c2 | Which conversion would you like to perform? | `brainstorm` | [brainstorm.md](brainstorm.md) |
| B_r5_c3 | Which conversion would you like to perform? | `scout` | [scout.md](scout.md) |
| B5_r6_c3 | TUG and TURN | `run_tugturn` | [../../vaila/help/tugturn.md](../../vaila/help/tugturn.md) |
| C_r1_c1 | Which conversion would you like to perform? | `reorder_csv_data` | [reorder-csv-data.md](reorder-csv-data.md) |

## Categories

### File Manager (Frame A)
- A_r1_c1 through A_r1_c9

### Multimodal Analysis (Frame B)
- B1_r1_c1 through B5_r5_c5

### Tools (Frame C)
- C_A, C_B, C_C sections

## Related Documentation

- Script-specific help: `vaila/help/` - Contains help for individual Python scripts
- Module documentation: `docs/` - Contains module-level documentation

---

**Last Updated:** November 2025
