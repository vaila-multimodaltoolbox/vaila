"""
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

Main functions:
- select_file(): Opens a file dialog to select CSV files
- merge_csv_files(): Merges two CSV files horizontally at a specified column position
- stack_csv_files(): Stacks two CSV files vertically (appending rows)

Example workflow:
1. User runs the script
2. GUI prompts for file selection
3. User selects files and specifies merge/stack options
4. Script processes the files and saves the result
5. Confirmation message is displayed

License:
--------
This script is licensed under the GNU General Public License v3.0. GPLv3.

Citation:
--------
If you use this script, please cite the following paper:
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo Roberto Pereira Santiago and Guilherme Manna Cesar and Ligia Yumi Mochida and Juan Aceros and others},
  year={2024},
  eprint={2410.07238},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2410.07238}
}
"""

import os
from tkinter import filedialog, messagebox

import pandas as pd


def select_file(prompt):
    return filedialog.askopenfilename(title=prompt, filetypes=[("CSV files", "*.csv")])


def merge_csv_files(base_file, merge_file, save_path, insert_position=None):
    base_file_name = os.path.basename(base_file)
    merge_file_name = os.path.basename(merge_file)
    save_file_name = os.path.basename(save_path)

    print(f"Loading base file: {base_file_name}")
    base_df = pd.read_csv(base_file)
    print(f"Loading merge file: {merge_file_name}")
    merge_df = pd.read_csv(merge_file)

    if insert_position is None or insert_position > len(base_df.columns):
        insert_position = len(base_df.columns) + 1

    insert_position -= 1  # Adjusting to 0-based index

    print(
        f"Merging {merge_file_name} into {base_file_name} at column position {insert_position + 1}."
    )

    merged_df = pd.concat(
        [
            base_df.iloc[:, :insert_position],
            merge_df,
            base_df.iloc[:, insert_position:],
        ],
        axis=1,
    )
    merged_df.to_csv(save_path, index=False)

    print(f"Merged file saved as {save_file_name}.")
    messagebox.showinfo("Success", f"Merged file saved as {save_file_name}.")


def stack_csv_files(base_file, stack_file, save_path, position="end"):
    base_file_name = os.path.basename(base_file)
    stack_file_name = os.path.basename(stack_file)
    save_file_name = os.path.basename(save_path)

    print(f"Loading base file: {base_file_name}")
    base_df = pd.read_csv(base_file)
    print(f"Loading stack file: {stack_file_name}")
    stack_df = pd.read_csv(stack_file)

    # Remove header of stack file
    stack_df.columns = base_df.columns

    if position == "start":
        print(f"Stacking {stack_file_name} at the beginning of {base_file_name}.")
        stacked_df = pd.concat([stack_df, base_df], ignore_index=True)
    else:
        print(f"Stacking {stack_file_name} at the end of {base_file_name}.")
        stacked_df = pd.concat([base_df, stack_df], ignore_index=True)

    stacked_df.to_csv(save_path, index=False)

    print(f"Stacked file saved as {save_file_name}.")
    messagebox.showinfo("Success", f"Stacked file saved as {save_file_name}.")
