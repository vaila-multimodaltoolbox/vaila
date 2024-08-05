"""
mergestack.py
Version: 2024-07-31 15:30:00
"""

import pandas as pd
from tkinter import filedialog, messagebox
import os

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

    print(f"Merging {merge_file_name} into {base_file_name} at column position {insert_position + 1}.")
    
    merged_df = pd.concat([base_df.iloc[:, :insert_position], merge_df, base_df.iloc[:, insert_position:]], axis=1)
    merged_df.to_csv(save_path, index=False)
    
    print(f"Merged file saved as {save_file_name}.")
    messagebox.showinfo("Success", f"Merged file saved as {save_file_name}.")

def stack_csv_files(base_file, stack_file, save_path, position='end'):
    base_file_name = os.path.basename(base_file)
    stack_file_name = os.path.basename(stack_file)
    save_file_name = os.path.basename(save_path)
    
    print(f"Loading base file: {base_file_name}")
    base_df = pd.read_csv(base_file)
    print(f"Loading stack file: {stack_file_name}")
    stack_df = pd.read_csv(stack_file)
    
    # Remove header of stack file
    stack_df.columns = base_df.columns
    
    if position == 'start':
        print(f"Stacking {stack_file_name} at the beginning of {base_file_name}.")
        stacked_df = pd.concat([stack_df, base_df], ignore_index=True)
    else:
        print(f"Stacking {stack_file_name} at the end of {base_file_name}.")
        stacked_df = pd.concat([base_df, stack_df], ignore_index=True)
    
    stacked_df.to_csv(save_path, index=False)
    
    print(f"Stacked file saved as {save_file_name}.")
    messagebox.showinfo("Success", f"Stacked file saved as {save_file_name}.")
