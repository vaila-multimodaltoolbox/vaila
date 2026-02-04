# readc3d_export

## 📋 Module Information

- **Category:** Processing
- **File:** `vaila\readc3d_export.py`
- **Lines:** 1986
- **Size:** 82210 characters
- **Version:** 0.2.1
- **Author:** Prof. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description

===============================================================================
readc3d_export.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Creation Date: 25 September 2024
Update Date: 03 February 2026
Version: 0.2.1
Python Version: 3.12.12

Description:
This script processes .c3d files, extracting marker data, analog data, events, and points residuals,
and saves them into CSV files. It also allows the option to save the data in Excel format.
The script leverages Dask for efficient data handling and processing, particularly useful
when working with large datasets.

Features:
- Extracts and saves marker data with time columns.
- Extracts and saves analog data with time columns, including their units.
- Extracts and saves events with their labels and times.
- Extracts and saves points residuals with time columns.
- Supports saving the data in CSV format.
- Optionally saves the data in Excel format (can be slow for large files).
- Generates an info file containing metadata about markers, analogs, and their units.
- Generates a simplified short info file with key parameters and headers.
- Handles encoding errors to avoid crashes due to unexpected characters.
- Extracts and saves force platform data including Center of Pressure (COP) from the C3D file into CSV files.
- **Didactic C3D Inspector:** A tabbed GUI to explore C3D structure, check data health, and view parameters/events. Opened via the "Inspect C3D" button in the C3D <-> CSV dialog.
- **Export Reports:** In the Inspector, use "Save TXT Report" and "Save HTML Report" to save inspection summaries. Reports work with any C3D (including files with force platforms). Special or invalid characters in the data are handled when writing (UTF-8 with replacement of characters that cannot be encoded).

## Inspect C3D and reports

From the vailá GUI, go to **C_A_r1_c2 (C3D <--> CSV)** and click **Inspect C3D**. Choose a .c3d file to open the Didactic C3D Inspector. The Inspector has three tabs: **Overview** (file info, acquisition settings, data health), **Parameters Map** (expandable parameter groups), and **Data & Events** (timeline events and marker list). Use the buttons **Save TXT Report** and **Save HTML Report** to export a text or HTML inspection report to a location of your choice.

## 🔧 Main Functions

**Total functions found:** 15

- `save_info_file`
- `save_short_info_file`
- `save_events`
- `importc3d`
- `save_empty_file`
- `save_platform_data`
- `save_rotation_data`
- `save_meta_points_data`
- `save_header_summary`
- `save_parameter_groups`
- `save_data_statistics`
- `save_to_files`
- `convert_c3d_to_csv`
- `batch_convert_c3d_to_csv`
- `inspect_c3d_gui`
- `DidacticC3DInspector` (Class)

---

📅 **Generated automatically on:** February 2026
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
