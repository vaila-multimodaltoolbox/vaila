"""
================================================================================
join2dataset.py
================================================================================
Author: Prof. Paulo Santiago
Date: 05 June 2024
Version: 1.0

Description:
------------
This script consolidates multiple CSV files located in subdirectories of a given
base directory into a single unified dataset. It scans for files ending with
"_result.csv" in each subdirectory, extracts relevant data, and adds a "Subject"
column derived from the filename. The resulting combined dataset is saved as a
CSV file in a newly created directory, timestamped with the current date and time.

Key Functionalities:
---------------------
1. Scans all subdirectories in a specified base directory for CSV files that
   end with "_result.csv".
2. Adds a "Subject" column based on the filename, to identify data by subject.
3. Reorders columns to place "Subject" between "TimeStamp" and "Trial".
4. Concatenates all found datasets into a single CSV file.
5. Saves the resulting dataset in a timestamped directory with a unique name.

How to Use:
-----------
1. Run the script from the command line, passing the path to the base directory as an argument.
   Example:
       python join2dataset.py /path/to/base_directory
2. The script will create a new directory with the current date and time in the execution directory,
   and save the combined dataset there.

Modules and Packages Required:
------------------------------
- Python Standard Libraries: os, sys, datetime.
- External Libraries: pandas (for CSV reading, manipulation, and saving).

Example:
--------
Assume the base directory has the following structure:
    base_directory/
    ├── subject1/
    │   └── subject1_result.csv
    ├── subject2/
    │   └── subject2_result.csv

Running the script:
    python join2dataset.py /path/to/base_directory

The output dataset will be saved as a CSV file in a directory created with the format
`Dataset_results_<timestamp>_base_directory`, and will contain a consolidated dataset
with the additional "Subject" column.

License:
--------
This script is licensed under the GNU General Public License v3.0 (GPLv3).

Disclaimer:
-----------
This script is provided "as is," without any warranty, express or implied. The author is
not liable for any damage or data loss resulting from the use of this script. It is intended
solely for academic and research purposes.

Changelog:
----------
- 2024-06-05: Initial creation of the script with directory scanning and dataset concatenation functionalities.
================================================================================
"""

import os
import sys
from datetime import datetime

import pandas as pd

print(
    r"""
 _          ____  _        ____       __  __   | Biomechanics and Motor Control Laboratory
| |        |  _ \(_)      /  __|     |  \/  |  | Developed by: Paulo R. P. Santiago
| |    __ _| |_) |_  ___ |  /    ___ | \  / |  | paulosantiago@usp.br
| |   / _' |  _ <| |/ _ \| |    / _ \| |\/| |  | University of Sao Paulo
| |__' (_| | |_) | | (_) |  \__' (_) | |  | |  | https://orcid.org/0000-0002-9460-8847
|____|\__'_|____/|_|\___/ \____|\___/|_|  |_|  | Date: 05 Jun 2024
"""
)


def join_datasets(base_dir):
    # Create a list to store dataframes
    dfs = []

    # Sort and iterate over each subdirectory in the base directory
    for subdir in sorted(os.listdir(base_dir)):
        full_path = os.path.join(base_dir, subdir)
        if os.path.isdir(full_path):
            # Look for files that end with '_result.csv'
            for file in os.listdir(full_path):
                if file.endswith("_result.csv"):
                    file_path = os.path.join(full_path, file)
                    # Read the CSV and add it to the list
                    df = pd.read_csv(file_path)
                    # Create the "Subject" column by extracting the first part of the filename
                    df["Subject"] = df["FileName"].apply(lambda x: x.split("_")[0])
                    # Reorder the columns so that "Subject" is between "TimeStamp" and "Trial"
                    columns = df.columns.tolist()
                    # Remove the 'Subject' column if it already exists
                    if "Subject" in columns:
                        columns.remove("Subject")
                    # Insert the 'Subject' column at the desired position
                    index = columns.index("Trial")
                    columns.insert(index, "Subject")
                    df = df[columns]
                    dfs.append(df)

    # Concatenate all the dataframes
    full_dataset = pd.concat(dfs, ignore_index=True)

    # Create the directory and file name with the current date and time in the script's execution directory
    now = datetime.now()
    formatted_now = now.strftime("%Y%m%d_%H%M%S")
    base_dir_name = os.path.basename(os.path.normpath(base_dir))
    current_dir = os.getcwd()
    results_dir_name = f"Dataset_results_{formatted_now}_{base_dir_name}"
    results_dir = os.path.join(current_dir, results_dir_name)
    os.makedirs(results_dir, exist_ok=True)

    # Save the combined dataset
    dataset_filename = f"{results_dir_name}.csv"
    full_dataset.to_csv(os.path.join(results_dir, dataset_filename), index=False)

    return f"Dataset saved in {results_dir}"


# Use the function with a command-line argument
if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python join2dataset.py <path_to_base_directory>")
    else:
        base_directory = sys.argv[1]
        print(join_datasets(base_directory))
