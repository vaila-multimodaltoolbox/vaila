# mergestack

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila/mergestack.py`
- **Lines:** 137
- **Size:** 4231 characters
- **Version:** 2025-02-28 15:30:00
- **Author:** Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


mergestack.py
Version: 2025-02-28 15:30:00
Author: Paulo R. P. Santiago
Version: 0.0.3

Dependencies:
- Python 3.12.9
- pandas
- tkinter
- os

Contact:
--------
paulosantiago@usp.br

Version history:
----------------
- v0.0.1 (2025-02-28): Initial version. Merge and stack CSV files.
- v0.0.2 (2025-02-28): Added function to select file.
- v0.0.3 (2025-02-28): Added function to stack CSV files.

Usage:
------
python mergestack.py

Description:
------------
This script allows you to merge and stack CSV files.
The script provides functionality to merge and stack CSV files in different ways:

1. Merge: Combines two CSV files horizontally (column-wise), inserting one file's columns
   into another at a specified position.

2. Stack: Combines two CSV files vertically (row-wise), appending one file's rows
   to another either at the beginning or end.

The script uses a graphical user interface (GUI) for file selection and provides
feedback through message boxes and console output.

Main funct...

## 🔧 Main Functions

**Total functions found:** 3

- `select_file`
- `merge_csv_files`
- `stack_csv_files`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
