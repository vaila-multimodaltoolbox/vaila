"""
================================================================================
Cluster Configuration Input Dialog with Default Values
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-07-29
Version: 1.1

Overview:

This Python module provides a graphical user interface (GUI) using Tkinter to collect input from the user regarding configuration details for two clusters. The dialog requests the sample rate, the configuration (A/B/C/D) for each cluster, and their respective names. Default values are provided for ease of use: sample rate is set to 100, and the cluster configurations default to 'A' and 'C'. Cluster names default to 'Trunk' and 'Pelvis'. The collected information is then returned in a dictionary for further processing.

Main Features:

    1. User Input Collection:
        - Collects a sample rate (float) and validates it. Default value is `100`.
        - Accepts the configuration type (A/B/C/D) for both clusters. Defaults are `A` for Cluster 1 and `C` for Cluster 2.
        - Gathers user-defined names for each cluster. Default names are `Trunk` for Cluster 1 and `Pelvis` for Cluster 2.

    2. Validation and Error Handling:
        - Ensures that the sample rate is a valid float.
        - Displays error messages if the sample rate is invalid or if cluster names are missing.

    3. GUI with Tkinter:
        - The interface is built using Tkinter with various `LabelFrame` and `Entry` widgets for a structured input form.
        - Input fields for sample rate, cluster configurations, and cluster names are presented with default values for a streamlined experience.

Key Functions and Their Functionality:

    get_user_inputs():
        - Launches the Tkinter GUI to collect inputs from the user.
        - Populates the fields with default values: sample rate (`100`), cluster configurations (`A` and `C`), and cluster names (`Trunk` and `Pelvis`).
        - Validates the sample rate and ensures that cluster names are not empty.
        - Returns a dictionary with the following keys:
            - 'sample_rate': The user-specified or default sample rate (float).
            - 'cluster1_config': Configuration type for cluster 1 (default 'A').
            - 'cluster2_config': Configuration type for cluster 2 (default 'C').
            - 'cluster1_name': The name given to cluster 1 (default 'Trunk').
            - 'cluster2_name': The name given to cluster 2 (default 'Pelvis').
        - Closes the GUI after the inputs are confirmed.

Usage Notes:

    - This module is typically used in biomechanical analysis workflows where user input is required to configure clusters representing body segments.
    - Default values are provided for user convenience, but they can be modified as needed.
    - Pressing the "Confirm" button or hitting the "Enter" key will submit the inputs.

Changelog for Version 1.1:

    - Added default values for sample rate (`100`), cluster configurations (`A` and `C`), and cluster names (`Trunk` and `Pelvis`).
    - Improved error handling for invalid sample rates and empty cluster names.
    - Bound the "Enter" key to confirm the inputs for a smoother user experience.

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

    def confirm(event=None):
        try:
            sample_rate = float(sample_rate_entry.get())
            cluster1_config = cluster1_config_entry.get().strip().upper()
            cluster2_config = cluster2_config_entry.get().strip().upper()
            cluster1_name = cluster1_name_entry.get().strip()
            cluster2_name = cluster2_name_entry.get().strip()

            if not cluster1_name or not cluster2_name:
                error_label.config(text="Cluster names cannot be empty.")
                return

            user_inputs["sample_rate"] = sample_rate
            user_inputs["cluster1_config"] = cluster1_config
            user_inputs["cluster2_config"] = cluster2_config
            user_inputs["cluster1_name"] = cluster1_name
            user_inputs["cluster2_name"] = cluster2_name

            app.quit()  # Quit the main loop
        except ValueError:
            error_label.config(text="Please enter a valid float for sample rate.")

    app = tk.Tk()
    app.title("Enter Configuration Details")

    user_inputs = {}  # Dictionary to hold user inputs

    # Create a frame to hold the inputs and the image
    main_frame = ttk.Frame(app)
    main_frame.pack(padx=10, pady=10)

    # Create a frame for inputs
    input_frame = ttk.Frame(main_frame)
    input_frame.grid(row=0, column=0, padx=10, pady=10)

    # Create a LabelFrame for entering sample rate
    sample_rate_frame = ttk.LabelFrame(input_frame, text="Enter sample rate:")
    sample_rate_frame.pack(padx=10, pady=10)

    # Create an Entry widget for sample rate with default value 100
    sample_rate_entry = ttk.Entry(sample_rate_frame)
    sample_rate_entry.insert(0, "100")  # Default value
    sample_rate_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 1 configuration
    cluster1_config_frame = ttk.LabelFrame(
        input_frame, text="Cluster 1 Configuration (A/B/C/D):"
    )
    cluster1_config_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 1 configuration with default value 'A'
    cluster1_config_entry = ttk.Entry(cluster1_config_frame)
    cluster1_config_entry.insert(0, "A")  # Default value
    cluster1_config_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 1 name
    cluster1_name_frame = ttk.LabelFrame(input_frame, text="Cluster 1 Name:")
    cluster1_name_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 1 name with default value 'Trunk'
    cluster1_name_entry = ttk.Entry(cluster1_name_frame)
    cluster1_name_entry.insert(0, "Trunk")  # Default value
    cluster1_name_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 2 configuration
    cluster2_config_frame = ttk.LabelFrame(
        input_frame, text="Cluster 2 Configuration (A/B/C/D):"
    )
    cluster2_config_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 2 configuration with default value 'C'
    cluster2_config_entry = ttk.Entry(cluster2_config_frame)
    cluster2_config_entry.insert(0, "C")  # Default value
    cluster2_config_entry.pack(padx=10, pady=5)

    # Create a LabelFrame for entering Cluster 2 name
    cluster2_name_frame = ttk.LabelFrame(input_frame, text="Cluster 2 Name:")
    cluster2_name_frame.pack(padx=10, pady=10)

    # Create an Entry widget for Cluster 2 name with default value 'Pelvis'
    cluster2_name_entry = ttk.Entry(cluster2_name_frame)
    cluster2_name_entry.insert(0, "Pelvis")  # Default value
    cluster2_name_entry.pack(padx=10, pady=5)

    # Create a confirm button to collect the inputs
    confirm_button = ttk.Button(input_frame, text="Confirm", command=confirm)
    confirm_button.pack(pady=10)

    # Create a label to display errors
    error_label = tk.Label(input_frame, text="", fg="red")
    error_label.pack()

    # Bind the Return key (Enter) to the confirm function
    app.bind("<Return>", confirm)

    app.mainloop()
    app.destroy()  # Close the GUI window after the main loop ends

    return user_inputs
