# ===============================================================================
# process_gait_features.py
# ===============================================================================
# Author: Abel Gonçalves Chinaglia
# Ph.D. Candidate in PPGRDF - FMRP - USP
# Date: 05 Feb. 2025
# Version: 1.0.0
# Python Version: 3.8+
#
# Description:
# ------------
# This script extracts spatial and temporal features from gait analysis data
# stored in .csv files. It computes metrics such as mean, variance, range, speed,
# and step length for each individual and trial, organizing the results in a final
# .csv file for further analysis.
#
# Key Features:
# -------------
# 1. Asks for the participant's name to save it in the results.
# 2. Requests the number of steps per trial to divide data into blocks for analysis.
# 3. Calculates spatial, temporal, and kinematic features for each trial and individual.
# 4. Automatically processes all .csv files in the selected directory.
# 5. Saves the results in an output .csv file for further analysis.
#
# Usage:
# ------
# 1. Place all gait data files in the main directory.
# 2. Run the script and select the input and output directories via a graphical interface.
#
# Output Files:
# -------------
# - `gait_features.csv`: Contains extracted features for all individuals and trials.
#
# License:
# --------
# This program is licensed under the GNU Lesser General Public License v3.0.
# For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html
# ===============================================================================

import os
import pandas as pd
import numpy as np
from rich import print
from tkinter import Tk, filedialog, simpledialog
from datetime import datetime


def calculate_features(data_block):
    """
    Calculate spatial, temporal, and kinematic features from gait data.
    """
    features = {}
    # Spatial statistics
    features["left_heel_x_mean"] = data_block["left_heel_x"].mean()
    features["left_heel_y_mean"] = data_block["left_heel_y"].mean()
    features["left_foot_index_x_mean"] = data_block["left_foot_index_x"].mean()
    features["left_foot_index_y_mean"] = data_block["left_foot_index_y"].mean()

    features["right_heel_x_mean"] = data_block["right_heel_x"].mean()
    features["right_heel_y_mean"] = data_block["right_heel_y"].mean()
    features["right_foot_index_x_mean"] = data_block["right_foot_index_x"].mean()
    features["right_foot_index_y_mean"] = data_block["right_foot_index_y"].mean()

    # Variance
    features["left_heel_x_var"] = data_block["left_heel_x"].var()
    features["left_heel_y_var"] = data_block["left_heel_y"].var()
    features["left_foot_index_x_var"] = data_block["left_foot_index_x"].var()
    features["left_foot_index_y_var"] = data_block["left_foot_index_y"].var()

    features["right_heel_x_var"] = data_block["right_heel_x"].var()
    features["right_heel_y_var"] = data_block["right_heel_y"].var()
    features["right_foot_index_x_var"] = data_block["right_foot_index_x"].var()
    features["right_foot_index_y_var"] = data_block["right_foot_index_y"].var()

    # Range of motion
    features["left_heel_x_range"] = (
        data_block["left_heel_x"].max() - data_block["left_heel_x"].min()
    )
    features["left_heel_y_range"] = (
        data_block["left_heel_y"].max() - data_block["left_heel_y"].min()
    )

    features["right_heel_x_range"] = (
        data_block["right_heel_x"].max() - data_block["right_heel_x"].min()
    )
    features["right_heel_y_range"] = (
        data_block["right_heel_y"].max() - data_block["right_heel_y"].min()
    )

    return features


def divide_into_blocks(data, num_steps):
    """
    Divide the data into blocks based on the number of steps provided.
    """
    block_size = len(data) // num_steps
    blocks = [
        data.iloc[i * block_size : (i + 1) * block_size] for i in range(num_steps)
    ]
    return blocks


def process_files_and_save(input_dir, output_dir):
    """
    Process all .csv files in the input directory and save extracted features to output.
    """

    csv_files = [
        os.path.join(input_dir, f) for f in os.listdir(input_dir) if f.endswith(".csv")
    ]
    all_features = []

    for file in csv_files:
        print(f"Processing file: {os.path.basename(file)}")
        data = pd.read_csv(file)

        num_steps = simpledialog.askinteger(
            "Number of Steps",
            f"Enter the number of steps for {os.path.basename(file)}:",
        )
        if not num_steps or num_steps <= 0:
            print("Invalid number of steps. Skipping file...")
            continue

        blocks = divide_into_blocks(data, num_steps)
        for i, block in enumerate(blocks):
            features = calculate_features(block)
            features["Participant"] = os.path.splitext(os.path.basename(file))[0]
            features["Trial"] = os.path.basename(file)
            features["Step_Block"] = i + 1
            all_features.append(features)

    if all_features:
        result_df = pd.DataFrame(all_features)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if len(csv_files) == 1:
            base_name = os.path.splitext(os.path.basename(csv_files[0]))[0]
            output_file = os.path.join(
                output_dir, f"{base_name}_gaitfeatures_{timestamp}.csv"
            )
        else:
            output_file = os.path.join(output_dir, f"gaitfeatures_{timestamp}.csv")
        result_df.to_csv(output_file, index=False)
        print(f"Feature extraction complete. Results saved to {output_file}.")
    else:
        print("No features were extracted. No output file created.")


def run_process_gait_features():
    """
    Main function to select directories and process gait data.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    input_dir = filedialog.askdirectory(title="Select Input Directory with .csv Files")
    if not input_dir:
        print("No input directory selected. Exiting...")
        return

    output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
    if not output_dir:
        print("No output directory selected. Exiting...")
        return

    process_files_and_save(input_dir, output_dir)


if __name__ == "__main__":
    run_process_gait_features()
