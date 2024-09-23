"""
================================================================================
Custom 3D Rotation Processing Toolkit
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-09-21
Version: 1.1

Overview:

This Python script is designed for rotating and transforming 3D motion capture data.
It allows the application of predefined or custom rotation angles (in degrees) and
rotation orders (e.g., 'xyz', 'zyx') to sets of 3D data points (e.g., from CSV files).

The script supports different predefined rotation settings (A, B, C) and enables users 
to provide custom angles and rotation sequences to rotate datasets as needed for biomechanical 
analysis. Output files are generated after the rotations, preserving the original time column 
and saving the rotated data in CSV format.

Main Features:

    1. Predefined Rotations:
        - Options 'A', 'B', and 'C' apply standard rotation transformations to the dataset:
          'A' applies a 180° rotation around the Z-axis.
          'B' applies a 90° clockwise rotation around the Z-axis.
          'C' applies a 90° counterclockwise rotation around the Z-axis.

    2. Custom Rotation:
        - Users can input custom angles in the form of a list (e.g., [x, y, z] degrees).
        - Optionally, the rotation order (e.g., 'xyz', 'zyx') can be specified.
        - Custom rotations are applied to each set of points, allowing flexible transformations 
          for various biomechanical data configurations.

    3. Automated File Processing:
        - The script processes all CSV files in the specified input directory.
        - Outputs are saved to a subfolder 'rotated_files', with the rotation type added as a suffix.
        - The output retains the original precision of the data (number of decimal places).

Key Functions and Their Functionality:

    modify_lab_coords():
        - Applies rotation transformations to the dataset based on specified angles and rotation order.

    get_labcoord_angles():
        - Maps the predefined rotation options ('A', 'B', 'C') to specific angles.
        - Parses custom angles and rotation order provided by the user.

    process_files():
        - Processes all CSV files in the input directory.
        - Applies rotation transformations and saves the results to a new subfolder.

    run_modify_labref():
        - Main function to execute the rotation processing based on the user's input.

Usage Notes:

    - Predefined options: 'A', 'B', 'C' represent standard rotations for biomechanical datasets.
    - Custom option: Allows users to provide custom angles and rotation orders.
      Example usage: `run_modify_labref('[-45, 30, 90], zyx', 'path/to/csv_files')`
    - Ensure that the CSV files include a 'Time' column, as this is required for the rotation processing.

Changelog for Version 1.1:

    - Added support for custom rotation angles and user-defined rotation orders.
    - Improved error handling and feedback when invalid inputs are provided.
    - Automated the saving of rotated datasets with customizable precision based on input data.

License:

This script is distributed under the GPL3 License.
================================================================================
"""

import os
import numpy as np
import pandas as pd
from vaila.rotation import rotdata


def modify_lab_coords(data, labcoord_angles, ordem):
    if labcoord_angles:
        data = rotdata(
            data.T,
            xth=labcoord_angles[0],
            yth=labcoord_angles[1],
            zth=labcoord_angles[2],
            ordem=ordem,  # Apply the custom rotation order
        ).T
    return data


def get_labcoord_angles(option):
    if option == "A":
        return (0, 0, 180), "_rot180", "xyz"
    elif option == "B":
        return (0, 0, 90), "_rot90clock", "xyz"
    elif option == "C":
        return (0, 0, -90), "_rot90cclock", "xyz"
    else:
        try:
            # Split the custom input if it includes both angles and rotation order
            parts = option.split(",")
            custom_angles = eval(parts[0])

            if len(parts) == 2:
                ordem = parts[1].strip()  # Extract the rotation order if provided
            else:
                ordem = "xyz"  # Default to 'xyz' if no order is specified

            # Ensure that the custom angles are a valid list of three values
            if isinstance(custom_angles, list) and len(custom_angles) == 3:
                return custom_angles, "_custom", ordem
            else:
                raise ValueError("Custom angles must be a list of three elements.")
        except (SyntaxError, NameError, ValueError) as e:
            print(f"Invalid base orientation input: {e}. Using canonical base.")
            return (0, 0, 0), "_canonical", "xyz"


def process_files(input_dir, labcoord_angles, suffix, ordem):
    output_dir = os.path.join(input_dir, "rotated_files")
    os.makedirs(output_dir, exist_ok=True)

    file_names = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        data = pd.read_csv(file_path)

        # Verificar se a coluna "Time" está presente em qualquer variação de maiúsculas e minúsculas
        time_col_present = any(col.lower() == "time" for col in data.columns)
        if not time_col_present:
            print(
                f"Error: Column 'Time' not found in {file_name}. Please include a 'Time' column."
            )
            continue

        # Renomear a coluna de tempo para garantir que tenha um nome consistente
        data.rename(
            columns={col: "Time" for col in data.columns if col.lower() == "time"},
            inplace=True,
        )

        modified_data = data.copy()
        time_column = data["Time"]

        for i in range(1, len(data.columns), 3):
            points = data.iloc[:, i : i + 3].values
            modified_points = modify_lab_coords(points, labcoord_angles, ordem)
            modified_data.iloc[:, i : i + 3] = modified_points

        if "Time" not in modified_data.columns:
            modified_data.insert(0, "Time", time_column)

        # Obter o número de casas decimais dos dados originais
        float_format = "%.6f"  # Padrão para 6 casas decimais
        sample_value = data.iloc[0, 1]
        if isinstance(sample_value, float):
            decimal_places = len(str(sample_value).split(".")[1])
            float_format = f"%.{decimal_places}f"

        base_name, ext = os.path.splitext(file_name)
        output_file_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
        modified_data.to_csv(output_file_path, index=False, float_format=float_format)
        print(f"\n*** Processed and saved: {output_file_path} ***\n")

    print("\n" + "*" * 50)
    print("     All files have been processed and saved successfully!     ")
    print("*" * 50 + "\n")


def run_modify_labref(option, input_dir):
    labcoord_angles, suffix, ordem = get_labcoord_angles(option)
    process_files(input_dir, labcoord_angles, suffix, ordem)
