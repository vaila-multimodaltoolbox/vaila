"""
================================================================================
VICON CSV Split Batch Processor
================================================================================
Author: Prof. Dr. Paulo R. P. Santiago
Date: 2024-09-20
Version: 1.1

Description:
------------
This Python script processes CSV files generated from the VICON Nexus system and splits the data into separate files by device. It processes only the CSV files in the specified directory (first level, without entering subdirectories) and exports them into a user-specified output folder. The header information is cleaned and sanitized, and the files are saved with additional timestamp information for traceability.

Main Features:
--------------
1. **Batch Processing**: The script automatically finds and processes all CSV files in the specified directory (without subdirectories).
2. **Header Merging and Cleaning**: It merges multiple header rows, replaces problematic characters, and sanitizes unit symbols for compatibility.
3. **File Creation Timestamp**: Automatically retrieves and adds the file creation timestamp to each processed CSV.
4. **Error Handling**: Detects and reports parsing errors for each device during CSV processing.
5. **User Interface**: Prompts the user for input/output directories using Tkinter's graphical file dialogs.

Usage Instructions:
-------------------
1. Run the script in a Python environment that has the required packages installed (pandas, tkinter).
2. When prompted, select the source directory containing the CSV files to process and an output directory where the split CSVs will be saved.
3. The script will process all CSV files in the source directory (first level only) and split them by device, saving them in the output directory.

Dependencies:
-------------
- Python 3.x
- Required Libraries: pandas, tkinter

Changelog:
----------
Version 1.1 - 2024-09-20:
    - Restricted processing to CSV files in the first level of the source directory.
    - Added improvements to file handling and error checking.

License:
--------
This script is licensed under the MIT License.

Disclaimer:
-----------
This script is provided "as is" without warranty of any kind. The author is not responsible for any damage or loss resulting from the use of this script.
================================================================================
"""

import pandas as pd
import os
import re
from io import StringIO
from datetime import datetime
from tkinter import Tk, filedialog


def clean_header(header):
    """Sanitize the header to handle specific units and replace problematic characters."""
    header = header.replace("mm/s²", "mm_s2")
    header = header.replace("deg/s", "deg_s")
    header = re.sub(r"[\s]+", "_", header)
    header = re.sub(r"[^\w]", "", header)
    return header


def merge_headers(header1, header2):
    """Merge two header rows into a single header, ensuring unique header names."""
    header = []
    used_names = set()
    for h1, h2 in zip(header1, header2):
        h1 = clean_header(str(h1))
        h2 = clean_header(str(h2))

        combined_name = f"{h1}_{h2}" if h1 and h2 else h1 or h2
        count = 1
        original_name = combined_name
        while combined_name in used_names:
            combined_name = f"{original_name}_{count}"
            count += 1

        used_names.add(combined_name)
        header.append(combined_name)

    return header


def get_file_creation_datetime(filepath):
    """Retrieve the file creation date and time, and format it as a string."""
    stat = os.stat(filepath)
    try:
        creation_time = stat.st_birthtime
    except AttributeError:
        creation_time = stat.st_ctime
    return datetime.fromtimestamp(creation_time).strftime("%Y-%m-%d %H:%M:%S")


def read_csv_devs(filepath, output_dir):
    """Process a CSV file and save the results split by devices."""
    file_datetime = get_file_creation_datetime(filepath)

    with open(filepath, "r", encoding="utf-8-sig") as file:
        lines = file.readlines()

    blank_line_indices = [
        index for index, line in enumerate(lines) if line.strip() == ""
    ]
    blank_line_indices = [0] + blank_line_indices + [len(lines)]

    devs = {}
    dev_count = 0
    first_dev = True

    for i in range(len(blank_line_indices) - 1):
        start_index = blank_line_indices[i] + 1
        end_index = blank_line_indices[i + 1]

        header_offset = 2 if first_dev else 3
        data_offset = 4 if first_dev else 5

        if start_index < end_index - data_offset:
            dev_lines = lines[start_index:end_index]
            header_lines = StringIO(
                "".join(dev_lines[header_offset : header_offset + 2])
            )
            data_str = StringIO("".join(dev_lines[data_offset:]))

            try:
                headers_df = pd.read_csv(header_lines, header=None)
                merged_header = merge_headers(headers_df.iloc[0], headers_df.iloc[1])
                df = pd.read_csv(data_str, names=merged_header)
                df.insert(
                    0, "Timestamp", file_datetime
                )  # Insert the timestamp column at position 0
                devs[f"dev_{dev_count}"] = df
                output_filename = os.path.join(
                    output_dir,
                    f"{os.path.splitext(os.path.basename(filepath))[0]}_dev{dev_count + 1}.csv",
                )
                df.to_csv(output_filename, index=False)
                print(f"Saved: {output_filename}")
            except pd.errors.ParserError as e:
                print(f"Error parsing dev {dev_count}: {e}")
                devs[f"dev_{dev_count}_error"] = "".join(dev_lines[:100])

            first_dev = False
            dev_count += 1

    return devs


def select_directory():
    """Prompt the user to select a source and output directory."""
    root = Tk()
    root.withdraw()  # Hide the root window
    root.tk.call("wm", "attributes", ".", "-topmost", "1")  # Keep window on top

    # Prompt the user to select the source directory
    src_directory = filedialog.askdirectory(
        title="Select the Source Directory with CSV Files"
    )

    # Prompt the user to select the output directory
    output_directory = filedialog.askdirectory(
        title="Select the Output Directory to Save Processed Files"
    )

    return src_directory, output_directory


def process_csv_files_first_level(src_directory, output_directory):
    """Process all CSV files in the first level of the selected directory."""
    if not src_directory or not output_directory:
        print("Operation canceled by the user.")
        return

    # Find all CSV files in the first level of the source directory
    csv_files = [
        os.path.join(src_directory, file)
        for file in os.listdir(src_directory)
        if file.endswith(".csv")
    ]

    if not csv_files:
        print("No CSV files found in the selected directory.")
        return

    # Create the main output directory with a timestamp
    datetime_suffix = datetime.now().strftime("%Y%m%d_%H%M%S")
    main_output_dir = os.path.join(
        output_directory, f"vicon_csv_split_{datetime_suffix}"
    )
    os.makedirs(main_output_dir, exist_ok=True)

    for csv_file in csv_files:
        print(f"Processing file: {csv_file}")
        file_base = os.path.splitext(os.path.basename(csv_file))[0]
        dev_output_dir = os.path.join(main_output_dir, f"{file_base}_splitdevice")
        os.makedirs(dev_output_dir, exist_ok=True)

        devs = read_csv_devs(csv_file, dev_output_dir)

        for key, df in devs.items():
            if isinstance(df, pd.DataFrame):
                print(f"{key}: Shape = {df.shape}")
                print(df.head())
            else:
                print(f"{key} contains an error or non-data text: {df}")


if __name__ == "__main__":
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Get the source and output directory from the user
    src_directory, output_directory = select_directory()

    # Process CSV files only in the first level of the source directory
    process_csv_files_first_level(src_directory, output_directory)
