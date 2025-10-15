# native_file_dialog

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila\native_file_dialog.py`
- **Lines:** 203
- **Size:** 7099 characters
- **Version:** 0.1.0
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ❌ No

## 📖 Description


================================================================================
Native File Dialog Module - native_file_dialog.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 16 January 2025
Version: 0.1.0
Python Version: 3.12.11

Description:
------------
This module provides native file dialogs that work without conflicts with Pygame.
It uses system-native dialogs via subprocess to avoid event loop conflicts.

Usage:
------
from native_file_dialog import open_native_file_dialog, open_yes_no_dialog

# Open file dialog
file_path = open_native_file_dialog(
    title="Select File",
    file_types=[("*.csv", "CSV Files"), ("*.txt", "Text Files")]
)

# Open yes/no dialog
result = open_yes_no_dialog(
    title="Question",
    message="Do you want to continue?"
)

===================================================================...

## 🔧 Main Functions

**Total functions found:** 3

- `open_native_file_dialog`
- `open_yes_no_dialog`
- `open_save_file_dialog`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
