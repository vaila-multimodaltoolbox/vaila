"""
Script: dlc2vaila.py
Author: Prof. Dr. Paulo Santiago
Version: 1.0.0
Last Updated: December 9, 2024

Description:
    This script converts DLC (DeepLabCut) CSV files into a format compatible with
    the vailá multimodal toolbox. It processes all CSV files from a specified input
    directory, adjusts their structure, and saves the converted files in a newly
    created directory with a timestamped name.

    The conversion process includes:
    - Retaining only the third header line from the original DLC file.
    - Removing the first column temporarily, processing the remaining data, and re-adding the first column.
    - Excluding every third column from the data.
    - Generating a new header with the format: 'frame, p1_x, p1_y, p2_x, p2_y, ...'.
    - Saving the processed files in a dedicated output directory with a timestamp.

Usage:
    - Run the script to select an input directory containing DLC CSV files.
    - The script will process each file and save the converted outputs in a new
      directory created within the input directory.

How to Execute:
    1. Ensure Python and required dependencies are installed:
       - Install NumPy: `pip install numpy`
       - Install pandas: `pip install pandas`
       - Tkinter is usually bundled with Python installations.
    2. Open a terminal and navigate to the directory where `dlc2vaila.py` is located.
    3. Run the script using Python:

       python dlc2vaila.py

    4. Follow the graphical interface prompts to select the input directory.
    5. The script will process the files and save them in a new directory.

New Features:
    - Batch processing of all DLC CSV files in the selected directory.
    - Creation of a dedicated output directory with a timestamped name.
    - Reduction of data noise by excluding irrelevant columns.
    - Generation of vailá-compatible headers.

Requirements:
    - Python 3.11.9 or later
    - NumPy (`pip install numpy`)
    - pandas (`pip install pandas`)
    - Tkinter (usually included with Python installations)

Output:
    For each processed DLC CSV file, the following output is generated:
    - A converted CSV file with the same name as the original, appended with a timestamp.
    - All converted files are saved in a new directory named `dlc_to_vaila_<timestamp>` within the input directory.

Example:
    1. Select a folder containing DLC CSV files.
    2. The script processes each file and saves the converted outputs in a new folder named
       `dlc_to_vaila_<timestamp>` within the input directory.
    3. Each converted file will have a filename in the format `<original_name>_<timestamp>.csv`.

License:
    This program is free software: you can redistribute it and/or modify it under the terms of
    the GNU General Public License as published by the Free Software Foundation, either version 3
    of the License, or (at your option) any later version.

    This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY;
    without even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
    See the GNU General Public License for more details.

    You should have received a copy of the GNU GPLv3 (General Public License Version 3) along with this program.
    If not, see <https://www.gnu.org/licenses/>.
"""

import os
from datetime import datetime
from tkinter import Tk, filedialog, messagebox

import numpy as np
import pandas as pd
from rich import print


def process_csv_files_with_numpy(directory, save_directory):
    """
    Process all DLC CSV files in the specified directory:
    - Retain only the third header line.
    - Remove the first column, process the remaining data, and add the first column back as integers.
    - Remove every third column starting from the second one.
    - Generate a header in the format: 'frame, p1_x, p1_y, p2_x, p2_y, ...'.
    - Save the processed files in the specified save directory with a timestamp suffix.
    """
    # List all CSV files in the directory
    csv_files = [file for file in os.listdir(directory) if file.endswith(".csv")]

    for csv_file in csv_files:
        file_path = os.path.join(directory, csv_file)

        # Read the file header manually
        with open(file_path) as f:
            lines = f.readlines()

        # Retain only the third header line
        header_line = lines[2].strip()

        # Load the data, skipping the first three lines
        data = np.genfromtxt(file_path, delimiter=",", skip_header=3)

        # Save the original first column and convert it to integers
        col0 = data[:, 0:1].astype(int)

        # Remove the first column
        data_without_col0 = data[:, 1:]

        # Select columns to keep (exclude every third column)
        columns_to_keep = [i for i in range(data_without_col0.shape[1]) if (i + 1) % 3 != 0]
        processed_data = data_without_col0[:, columns_to_keep]

        # Add the first column back to the processed data
        final_data = np.hstack((col0, processed_data))

        # Generate a header in the format: 'frame, p1_x, p1_y, p2_x, p2_y, ...'
        num_points = (final_data.shape[1] - 1) // 2  # Exclude the 'frame' column
        headers = ["frame"] + [
            f"p{i + 1}_x" if j % 2 == 0 else f"p{i + 1}_y"
            for i in range(num_points)
            for j in range(2)
        ]

        # Convert the final data into a DataFrame for saving
        df = pd.DataFrame(final_data, columns=headers)

        # Ensure the 'frame' column is saved as integers
        df["frame"] = df["frame"].astype(int)

        # Generate timestamped filename
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_file_name = f"{os.path.splitext(csv_file)[0]}_{timestamp}.csv"
        output_path = os.path.join(save_directory, output_file_name)

        # Save the processed data
        df.to_csv(output_path, index=False, float_format="%.6f")

        print(f"Processed file saved at: {output_path}")


def batch_convert_dlc(directory=None):
    """
    Batch process all DLC files in a directory using the numpy-based method.
    Create a new directory to save the converted files and provide user feedback.

    Args:
        directory (str, optional): Path to the directory containing DLC CSV files. If not provided, a file dialog will be displayed.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # If no directory is provided, open the file dialog
    if directory is None:
        root = Tk()
        root.withdraw()  # Hide the root Tkinter window
        directory = filedialog.askdirectory(title="Select Directory Containing DLC CSV Files")

    if not directory:
        messagebox.showerror("Error", "No directory selected. Operation cancelled.")
        return

    # Create a new directory for saving converted files
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_directory = os.path.join(directory, f"dlc_to_vaila_{timestamp}")
    os.makedirs(save_directory, exist_ok=True)

    print(f"Processing DLC files in directory: {directory}")
    print(f"Converted files will be saved in: {save_directory}")

    process_csv_files_with_numpy(directory, save_directory)

    # Show a success message
    messagebox.showinfo(
        "DLC to vailá Conversion",
        f"Batch conversion of DLC files to vailá format completed successfully!\n\n"
        f"All converted files have been saved in the directory:\n{save_directory}",
    )


# Main function
if __name__ == "__main__":
    batch_convert_dlc()
