# dialogsuser

## 📋 Module Information

- **Category:** Utils
- **File:** `vaila\dialogsuser.py`
- **Lines:** 112
- **Size:** 4765 characters
- **Version:** 0.4.5
- **Updated:** 24 September 2026
- **Author:** Prof. Dr. Paulo R. P. Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


================================================================================
Sample Rate and File Type Input Dialog
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-29
Version: 1.0

Overview:

This Python module provides a simple graphical user interface (GUI) using Tkinter to collect user input for the sample rate and file type. It validates the inputs and ensures that the user enters a valid sample rate (float) and a supported file type (either 'csv' or 'c3d'). The inputs are returned in a dictionary for further use in data processing workflows.

Main Features:

    1. User Input Collection:
        - Prompts the user to enter a sample rate and file type (either 'csv' or 'c3d').
        - Validates that the sample rate is a valid float and that the file type is either 'csv' or 'c3d'.

    2. Error Handling and Validation:
        - Displays an errorr message if the sample rate is not a valid floa...

## 🔧 Main Functions

**Total functions found:** 5

- `get_user_inputs`
- `confirm`
- `default_output_dir` — folder of the input file/folder (list: first item)
- `ask_output_directory` — output dialog pre-selected on the input folder; one OK click keeps it
- `link_output_to_input` — form GUIs: output field follows the input field until the user picks another folder

**Output default:** across *vailá* modules the output directory defaults to the input folder to save clicks; choose any other folder to override.




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
