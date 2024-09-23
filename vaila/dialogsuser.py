"""
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
        - Displays an error message if the sample rate is not a valid float.
        - Ensures that only 'csv' or 'c3d' is accepted as a valid file type, showing an error for invalid input.

    3. GUI with Tkinter:
        - The interface is built using Tkinter with `LabelFrame` and `Entry` widgets for a clean and user-friendly layout.
        - Provides a "Confirm" button to submit the inputs and a label to display errors in case of invalid entries.

Key Functions and Their Functionality:

    get_user_inputs():
        - Launches a Tkinter window to collect the sample rate and file type from the user.
        - Validates the inputs:
            - The sample rate must be a valid float.
            - The file type must be either 'csv' or 'c3d'.
        - Returns a dictionary containing the following keys:
            - 'sample_rate': The user-specified sample rate (float).
            - 'file_type': The file type entered by the user ('csv' or 'c3d').
        - Closes the GUI after the inputs are confirmed or an error is displayed.

Usage Notes:

    - This module is useful in workflows where it is necessary to collect the sample rate and file type before processing data.
    - The sample rate is expected to be a floating-point number, and the file type is restricted to 'csv' or 'c3d' to ensure compatibility with common data formats in biomechanics and motion analysis.
    - The "Confirm" button collects and validates the inputs, closing the dialog upon successful submission.

Changelog for Version 1.0:

    - Initial version with support for sample rate and file type input collection.
    - Added validation for sample rate as a float and file type as 'csv' or 'c3d'.

License:

This script is distributed under the GPL3 License.
================================================================================
"""

import os
import tkinter as tk
from tkinter import ttk


def get_user_inputs():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    def confirm():
        try:
            sample_rate = float(sample_rate_entry.get())
            file_type = file_type_entry.get().strip().lower()
            if file_type not in ["csv", "c3d"]:
                error_label.config(text="Please enter 'csv' or 'c3d' for file type.")
                return
            user_inputs["sample_rate"] = sample_rate
            user_inputs["file_type"] = file_type
            app.quit()  # Quit the main loop
        except ValueError:
            error_label.config(text="Please enter a valid float for sample rate.")

    app = tk.Tk()
    app.title("Select File Format and Sample Rate")

    user_inputs = {}  # Dictionary to hold user inputs

    # Create a LabelFrame for entering sample rate
    sample_rate_frame = ttk.LabelFrame(app, text="Enter sample rate:")
    sample_rate_frame.pack(padx=10, pady=10)

    # Create an Entry widget for sample rate
    sample_rate_entry = ttk.Entry(sample_rate_frame)
    sample_rate_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering file type
    file_type_frame = ttk.LabelFrame(app, text="Enter file type (csv or c3d):")
    file_type_frame.pack(padx=10, pady=10)

    # Create an Entry widget for file type
    file_type_entry = ttk.Entry(file_type_frame)
    file_type_entry.pack(padx=10, pady=5)

    # Create a confirm button to collect the sample rate and file type
    confirm_button = ttk.Button(app, text="Confirm", command=confirm)
    confirm_button.pack(pady=10)

    # Create a label to display errors
    error_label = tk.Label(app, text="", fg="red")
    error_label.pack()

    app.mainloop()
    app.destroy()  # Close the GUI window after the main loop ends

    return user_inputs
