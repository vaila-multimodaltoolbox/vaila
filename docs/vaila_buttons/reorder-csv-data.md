# Edit CSV/C3D - Button C_A_r1_c1

## Overview

**Button Position:** C_A_r1_c1  
**Method Name:** `reorder_csv_data` (`vaila.py`) → `edit_csv_c3d.run_edit_csv_c3d` (`vaila/edit_csv_c3d.py`)  
**Button Text:** Edit CSV/C3D

## Description

Runs the Edit CSV/C3D module. Lets the user pick a directory containing
`.csv` and/or `.c3d` files and edits them with `rearrange_data.py`'s
`ColumnReorderGUI`. `.c3d` files are converted to a marker CSV first
(`readc3d_export.c3d_markers_to_dataframe`), edited alongside any `.csv`
files, then converted back to `.c3d` (`readcsv_export.
auto_create_c3d_from_csv`), preserving point rate, analog rate, units,
analog channels, and occlusion (NaN ↔ negative residual). Source files are
never overwritten; every run writes into a fresh
`processed_edit_csv_c3d_YYYYMMDD_HHMMSS/` directory. Clicking **Run**
prints the equivalent CLI command. Full details:
[`vaila/help/edit_csv_c3d.md`](../../vaila/help/edit_csv_c3d.md).

## Usage

1. Click the **Edit CSV/C3D** button in the vailá GUI
2. Select the directory containing `.csv` and/or `.c3d` files
3. Edit columns in the `ColumnReorderGUI` editor, then close it (`Esc` to save & exit)
4. Review the output files in the printed output directory

## Related Scripts

This button launches one or more Python scripts from the `vaila/` directory. For detailed script documentation, see:
- `vaila/help/` - Script-specific help files

## Integration

This button integrates with other vailá modules:
- Check related buttons in the same frame/section
- Output files can be used as input for other modules

## Troubleshooting

### Common Issues

- **Module not found**: Ensure all dependencies are installed
- **File not found**: Check that input files exist in the specified directory
- **Permission errors**: Ensure write permissions for output directory

### Getting Help

- Check the script-specific help in `vaila/help/`
- Review the main documentation in `docs/`
- Open an issue on GitHub if problems persist

---

**Last Updated:** November 2025  
**Part of vailá - Multimodal Toolbox**  
**License:** AGPLv3.0
