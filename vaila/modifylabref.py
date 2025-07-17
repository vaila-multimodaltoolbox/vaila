"""
Project: vailá Multimodal Toolbox
Script: modifylabref.py - Custom 3D Rotation Processing Toolkit

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 Sep 2024
Update Date: 17 Jul 2025
Version: 0.0.3

Description:
    This script is designed for rotating and transforming 3D motion capture data.
    It allows the application of predefined or custom rotation angles (in degrees) and
    rotation orders (e.g., 'xyz', 'zyx') to sets of 3D data points (e.g., from CSV files).

    Main Features:
    1. Predefined Rotations:
        - Options 'A', 'B', and 'C' apply standard rotation transformations:
          'A' applies a 180 degree rotation around the Z-axis.
          'B' applies a 90 degree clockwise rotation around the Z-axis.
          'C' applies a 90 degree counterclockwise rotation around the Z-axis.

    2. Custom Rotation:
        - Users can input custom angles in the format: [x, y, z]
        - Optionally specify rotation order: [x, y, z], xyz
        - Supports all 6 rotation orders: xyz, xzy, yxz, yzx, zxy, zyx

    3. Automated File Processing:
        - Processes all CSV files in the specified input directory
        - Outputs saved to 'rotated_files' subfolder
        - Preserves original data precision

Usage:
    Run the script from the command line:
        python modifylabref.py

Requirements:
    - Python 3.x
    - numpy
    - pandas
    - scipy

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v1.3: Fixed data missing handling and precision preservation
    - v1.2: Added custom rotation support with multiple rotation orders
    - v1.1: Added predefined rotation options (A, B, C)
    - v1.0: Initial version with basic 3D rotation functionality
"""

"""
================================================================================
Custom 3D Rotation Processing Toolkit
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-09-21
Version: 1.3

Overview:

This Python script is designed for rotating and transforming 3D motion capture data.
It allows the application of predefined or custom rotation angles (in degrees) and
rotation orders (e.g., 'xyz', 'zyx') to sets of 3D data points (e.g., from CSV files).

Main Features:

    1. Predefined Rotations:
        - Options 'A', 'B', and 'C' apply standard rotation transformations:
          'A' applies a 180 degree rotation around the Z-axis.
          'B' applies a 90 degree clockwise rotation around the Z-axis.
          'C' applies a 90 degree counterclockwise rotation around the Z-axis.

    2. Custom Rotation:
        - Users can input custom angles in the format: [x, y, z]
        - Optionally specify rotation order: [x, y, z], xyz
        - Supports all 6 rotation orders: xyz, xzy, yxz, yzx, zxy, zyx

    3. Automated File Processing:
        - Processes all CSV files in the specified input directory
        - Outputs saved to 'rotated_files' subfolder
        - Preserves original data precision

Usage Examples:

    run_modify_labref('A', 'path/to/csv_files')
    run_modify_labref('[0, -45, 0]', 'path/to/csv_files')
    run_modify_labref('[0, -45, 0], zyx', 'path/to/csv_files')

License: GPL3
================================================================================
"""

import os
import numpy as np
import pandas as pd
from scipy.spatial.transform import Rotation as R
from datetime import datetime
from rich import print


def rotdata(data, xth=0, yth=0, zth=0, ordem="xyz"):
    """
    Rotate the data based on the specified angles and order.
    
    Parameters:
    data (np.ndarray): The input data to be rotated, shape (3, n).
    xth, yth, zth (float): Rotation angles in degrees for x, y, and z axes.
    ordem (str): The order of rotation. Options are 'xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx'.
    
    Returns:
    np.ndarray: The rotated data.
    """
    # Create the rotation object using Euler angles
    rotation_object = R.from_euler(ordem, [xth, yth, zth], degrees=True)
    
    # Apply the rotation to the data
    datrot = rotation_object.apply(data.T).T
    
    return datrot


def modify_lab_coords(data, labcoord_angles, ordem):
    """Apply rotation to the coordinate data."""
    if labcoord_angles:
        data = rotdata(
            data.T,
            xth=labcoord_angles[0],
            yth=labcoord_angles[1],
            zth=labcoord_angles[2],
            ordem=ordem,
        ).T
    return data


def parse_custom_rotation_input(input_str):
    """
    Parse custom rotation input string and extract angles and rotation order.
    
    Supported formats:
    - "[x, y, z]" (uses default 'xyz' order)
    - "[x, y, z], xyz" (custom angles and order)
    
    Returns:
    tuple: (angles_list, rotation_order)
    """
    try:
        input_str = input_str.strip()
        
        # Check if input contains rotation order specification
        if ',' in input_str and any(order in input_str.lower() for order in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']):
            # Split by comma and find the rotation order
            parts = input_str.split(',')
            ordem = 'xyz'  # default
            angles_parts = []
            
            for part in parts:
                part = part.strip().lower()
                if part in ['xyz', 'xzy', 'yxz', 'yzx', 'zxy', 'zyx']:
                    ordem = part
                else:
                    angles_parts.append(part)
            
            angles_str = ','.join(angles_parts)
        else:
            ordem = 'xyz'
            angles_str = input_str
        
        # Parse angles - remove brackets if present
        angles_str = angles_str.strip()
        if angles_str.startswith('[') and angles_str.endswith(']'):
            angles_str = angles_str[1:-1]
        
        # Convert to list of floats
        angles = [float(x.strip()) for x in angles_str.split(',')]
        
        if len(angles) != 3:
            raise ValueError(f"Expected 3 angles, got {len(angles)}")
        
        return angles, ordem
        
    except Exception as e:
        raise ValueError(f"Invalid rotation format: {e}")


def get_labcoord_angles(option):
    """
    Get rotation angles and order based on the option provided.
    
    Args:
        option (str): Rotation option - can be 'A', 'B', 'C', or a rotation string
    
    Returns:
        tuple: (angles, suffix, rotation_order)
    """
    if option == "A":
        return (0, 0, 180), "_rot180", "xyz"
    elif option == "B":
        return (0, 0, 90), "_rot90clock", "xyz"
    elif option == "C":
        return (0, 0, -90), "_rot90cclock", "xyz"
    else:
        # Try to parse as custom rotation string
        try:
            angles, ordem = parse_custom_rotation_input(option)
            # Clean suffix to avoid invalid filename characters
            clean_angles = [str(int(angle)) if float(angle).is_integer() else str(angle).replace('.', 'p').replace('-', 'neg') for angle in angles]
            suffix = f"_custom_{clean_angles[0]}_{clean_angles[1]}_{clean_angles[2]}_{ordem}"
            print(f"Using custom rotation: X={angles[0]}, Y={angles[1]}, Z={angles[2]} degrees (order: {ordem.upper()})")
            return angles, suffix, ordem
            
        except ValueError as e:
            print(f"ERROR: Invalid rotation input: {e}")
            print("Using canonical base (no rotation).")
            return (0, 0, 0), "_canonical", "xyz"


def detect_column_precision(data, col_idx):
    """
    Detect the number of decimal places for a specific column.
    
    Args:
        data (pd.DataFrame): The dataframe (as string)
        col_idx (int): Column index to analyze
    
    Returns:
        int: Number of decimal places for this column
    """
    max_decimals = 0
    
    # Sample multiple rows to get a representative precision
    sample_rows = min(10, len(data))
    
    for row_idx in range(sample_rows):
        value_str = str(data.iloc[row_idx, col_idx])
        if value_str != 'nan' and '.' in value_str:
            decimal_part = value_str.split('.')[1]
            max_decimals = max(max_decimals, len(decimal_part))
    
    return max_decimals


def save_with_original_precision(data, original_data_str, output_path):
    """
    Save dataframe preserving the original precision of each column.
    
    Args:
        data (pd.DataFrame): Data to save (numeric)
        original_data_str (pd.DataFrame): Original data as strings
        output_path (str): Output file path
    """
    # Detect precision for each column
    column_formats = {}
    for i in range(len(data.columns)):
        precision = detect_column_precision(original_data_str, i)
        column_formats[i] = precision
    
    # Apply formatting to each column, but ensure minimum precision for rotated data
    formatted_data = data.copy()
    for i, col in enumerate(formatted_data.columns):
        precision = column_formats.get(i, 6)
        
        # Check if the column has decimal values (indicating that rotation was applied)
        has_decimals = (formatted_data[col] % 1 != 0).any()

        if precision == 0 and has_decimals:
            # Override precision if rotation created decimal values
            precision = 6
            
        if precision == 0 and not has_decimals:
            # Integer formatting only if truly integers
            formatted_data[col] = formatted_data[col].apply(lambda x: f"{int(x)}" if pd.notna(x) else "")
        else:
            # Float formatting with specific precision
            formatted_data[col] = formatted_data[col].apply(lambda x: f"{x:.{precision}f}" if pd.notna(x) else "")
    
    # Save without additional float formatting since we already formatted
    formatted_data.to_csv(output_path, index=False)


def process_files(input_dir, labcoord_angles, suffix, ordem):
    """Process all CSV files in the input directory."""
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Create a timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(input_dir, f"rotated_files_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    file_names = [f for f in os.listdir(input_dir) if f.endswith(".csv")]

    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        
        # Read the original data as string first to preserve precision info
        try:
            data_str = pd.read_csv(file_path, dtype=str)
            data = pd.read_csv(file_path)
        except Exception as e:
            print(f"Error reading {file_name}: {e}")
            continue

        # Verify if the file has at least one column
        if len(data.columns) == 0:
            print(f"Error: No columns found in {file_name}. Skipping file.")
            continue

        # Verify if the file has at least 4 columns (first column + at least one trio X, Y, Z)
        if len(data.columns) < 4:
            print(f"Warning: {file_name} has less than 4 columns. Expected format: [Time/Index, X1, Y1, Z1, ...]")
            continue

        # Store precision info for each column before processing
        column_precision = {}
        for i, col in enumerate(data.columns):
            precision = detect_column_precision(data_str, i)
            column_precision[i] = precision

        print(f"Column precision detected for {file_name}:")
        for i, precision in column_precision.items():
            print(f"  Column {i} ({data.columns[i]}): {precision} decimal places")

        # Use the first column as reference (time/index), preserving its original name
        first_column = data.iloc[:, 0]
        modified_data = data.copy()

        # Process groups of 3 columns (X, Y, Z) starting from the second column
        for i in range(1, len(data.columns), 3):
            # Verify if there are at least 3 columns remaining to form a complete trio
            if i + 2 < len(data.columns):
                points = data.iloc[:, i : i + 3].values
                modified_points = modify_lab_coords(points, labcoord_angles, ordem)
                modified_data.iloc[:, i : i + 3] = modified_points
            else:
                # If there are not 3 complete columns, warn and skip
                remaining_cols = len(data.columns) - i
                print(f"Warning: {file_name} has {remaining_cols} remaining columns at position {i}. Expected groups of 3 (X, Y, Z). Skipping incomplete group.")

        # Ensure that the first column is preserved
        modified_data.iloc[:, 0] = first_column

        # Save with original precision preserved
        base_name, ext = os.path.splitext(file_name)
        output_file_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
        
        try:
            save_with_original_precision(modified_data, data_str, output_file_path)
            print(f"Processed and saved: {output_file_path}")
        except Exception as e:
            print(f"Error saving {output_file_path}: {e}")
            # Fallback to simple save
            modified_data.to_csv(output_file_path, index=False)

    print("\nAll files have been processed and saved successfully!")


def run_modify_labref(option, input_dir):
    """
    Main function to execute rotation processing.
    
    Args:
        option (str): Rotation option ('A', 'B', 'C', or custom rotation string)
        input_dir (str): Path to directory containing CSV files to process
    """
    labcoord_angles, suffix, ordem = get_labcoord_angles(option)
    process_files(input_dir, labcoord_angles, suffix, ordem)


def main():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    """Interactive main function for script execution."""
    print("\nCUSTOM 3D ROTATION PROCESSING TOOLKIT")
    print("="*50)
    
    # Get input directory
    while True:
        try:
            input_dir = input("\nEnter the path to your CSV files directory: ").strip().strip('"')
            if not input_dir:
                print("Please enter a valid directory path.")
                continue
            if not os.path.exists(input_dir):
                print(f"Directory not found: {input_dir}")
                continue
            if not os.path.isdir(input_dir):
                print(f"Path is not a directory: {input_dir}")
                continue
            break
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    # Check for CSV files
    csv_files = [f for f in os.listdir(input_dir) if f.endswith('.csv')]
    if not csv_files:
        print(f"No CSV files found in directory: {input_dir}")
        return
    
    print(f"\nFound {len(csv_files)} CSV file(s) to process:")
    for i, file in enumerate(csv_files[:5], 1):
        print(f"   {i}. {file}")
    if len(csv_files) > 5:
        print(f"   ... and {len(csv_files) - 5} more files")
    
    # Get rotation option
    print("\nROTATION OPTIONS:")
    print("   A - 180 degree rotation around Z-axis")
    print("   B - 90 degree clockwise rotation around Z-axis") 
    print("   C - 90 degree counterclockwise rotation around Z-axis")
    print("   Custom - Enter format: [x, y, z] or [x, y, z], xyz")
    print("   Examples: [0, -45, 0] or [0, -45, 0], zyx")
    
    while True:
        try:
            option = input("\nEnter your choice: ").strip()
            if not option:
                print("Please enter a valid option.")
                continue
            
            if option.upper() in ['A', 'B', 'C']:
                option = option.upper()
                break
            else:
                # Try to validate custom rotation format
                try:
                    parse_custom_rotation_input(option)
                    break
                except ValueError as e:
                    print(f"Invalid rotation format: {e}")
                    print("Please try again. Use format: [x, y, z] or [x, y, z], xyz")
                    continue
                    
        except KeyboardInterrupt:
            print("\nOperation cancelled.")
            return
    
    # Confirm and execute
    print(f"\nReady to process {len(csv_files)} file(s) with rotation option: {option}")
    confirm = input("Proceed? (y/n): ").strip().lower()
    
    if confirm in ['y', 'yes', '']:
        print("\nPROCESSING...")
        print("="*30)
        
        try:
            run_modify_labref(option, input_dir)
        except Exception as e:
            print(f"Error during processing: {e}")
            return
        
        print(f"\nProcessing completed successfully!")
        print(f"Check the 'rotated_files' subfolder in: {input_dir}")
    else:
        print("Operation cancelled.")


if __name__ == "__main__":
    main()