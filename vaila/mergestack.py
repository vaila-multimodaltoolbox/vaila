import pandas as pd
import numpy as np
from scipy.interpolate import interp1d
from pykalman import KalmanFilter
import os
from tkinter import filedialog, messagebox
from rich import print

# Print the directory and name of the script being executed
print(f"Running script: {os.path.basename(__file__)}")
print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

def select_file(prompt):
    return filedialog.askopenfilename(title=prompt, filetypes=[("CSV files", "*.csv")])

def merge_csv_files(base_file, merge_file, save_path, insert_position=None):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

def fill_missing_rows(csv_file, save_path):
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    # Read the CSV file
    df = pd.read_csv(csv_file)
    
    # Determine which frame indices are missing
    all_indices = set(range(df['frame_index'].min(), df['frame_index'].max() + 1))
    current_indices = set(df['frame_index'])
    missing_indices = sorted(list(all_indices - current_indices))
    
    # If there are no missing rows, return early
    if not missing_indices:
        print(f"No missing indices in {csv_file}.")
        return

    print(f"Missing indices in {csv_file}: {missing_indices}")

    # Set the frame index as the DataFrame index for easier interpolation
    df.set_index('frame_index', inplace=True)
    
    # Interpolate linearly to create values for the missing rows
    interpolated_data = []
    for column in df.columns:
        if df[column].dtype == np.float64 or df[column].dtype == np.int64:
            f = interp1d(df.index, df[column], kind='linear', fill_value='extrapolate')
            interpolated_values = f(missing_indices)
            interpolated_data.append(interpolated_values)
        else:
            # For non-numeric columns, fill with the nearest value
            f = interp1d(df.index, df[column], kind='nearest', fill_value='extrapolate')
            interpolated_values = f(missing_indices)
            interpolated_data.append(interpolated_values)
    
    # Create new DataFrame for missing rows
    new_rows = pd.DataFrame(np.array(interpolated_data).T, columns=df.columns, index=missing_indices)
    new_rows.index.name = 'frame_index'
    
    # Concatenate the existing and new DataFrames, then sort by frame index
    complete_df = pd.concat([df, new_rows]).sort_index()
    
    # Optionally, apply a Kalman filter to smooth the data further
    kf = KalmanFilter(initial_state_mean=complete_df.iloc[0], n_dim_obs=complete_df.shape[1])
    kf = kf.em(complete_df.values, n_iter=10)
    (filtered_state_means, _) = kf.filter(complete_df.values)
    complete_df.loc[:, :] = filtered_state_means
    
    # Save the complete DataFrame to a new CSV file
    complete_df.reset_index(inplace=True)
    complete_df.to_csv(save_path, index=False)
    print(f"Missing rows added and saved to {save_path}.")

def fill_missing_rows_in_gui():
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    csv_file = select_file("Select the CSV file to fill missing rows")
    if not csv_file:
        return

    save_path = filedialog.asksaveasfilename(title="Save Filled CSV File As", defaultextension=".csv", filetypes=[("CSV files", "*.csv")])
    if not save_path:
        return

    fill_missing_rows(csv_file, save_path)
    messagebox.showinfo("Success", f"Missing rows added and saved to {save_path}.")
