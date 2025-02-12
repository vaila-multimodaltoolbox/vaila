# =============================================================================
# ML_valid_models.py
# =============================================================================
# Author: Abel Gonçalves Chinaglia
# Ph.D. Candidate in PPGRDF - FMRP - USP
# Date: 05 Feb. 2025
# Version: 1.0.0
# Python Version: 3.8+

# Description:
# ------------
# This script validates previously trained machine learning models by testing
# their performance on a separate validation dataset. It loads models trained
# with cross-validation and evaluates their prediction accuracy on new data.

# Key Features:
# --------------
# - Loads and validates models for multiple gait-related features
# - Calculates comprehensive metrics including MSE, RMSE, MAE, R², and others
# - Supports validation of models trained with cross-validation
# - Handles data preprocessing with saved StandardScaler parameters
# - Interactive file selection through GUI dialogs
# - Progress tracking with progress bars
# - Completion notification through GUI dialog

# Execution:
# ----------
# - Run the script:
#   $ python valid_models.py
# - Select the feature dataset file (CSV) when prompted
# - Select the target dataset file (CSV) when prompted
# - The script will automatically process the validation data and generate metrics

# Output Structure:
# -----------------
# - Validation metrics are saved in the respective model directories
# - Each target feature has its own metrics file with detailed performance measures
# - Results are saved as CSV files containing all evaluation metrics

# License:
# --------
# This program is licensed under the GNU Lesser General Public License v3.0.
# For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================
import os
import json
import pickle
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score,
    explained_variance_score,
    max_error,
    median_absolute_error,
)
from tkinter import Tk, filedialog, messagebox
from tqdm import tqdm  # For progress bar
import argparse


# Function to calculate metrics
def calculate_metrics(y_true, y_pred):
    epsilon = 1e-10  # To avoid division by zero
    interval_tolerance = 0.1  # 10% tolerance
    within_tolerance = np.abs(y_true - y_pred) <= interval_tolerance * np.abs(
        y_true + epsilon
    )
    accuracy_within_tolerance = np.mean(within_tolerance) * 100  # Accuracy percentage
    return {
        "MSE": mean_squared_error(y_true, y_pred),
        "RMSE": np.sqrt(mean_squared_error(y_true, y_pred)),
        "MAE": mean_absolute_error(y_true, y_pred),
        "MedAE": median_absolute_error(y_true, y_pred),
        "Max_Error": max_error(y_true, y_pred),
        "RAE": np.sum(np.abs(y_true - y_pred))
        / np.sum(np.abs(y_true - np.mean(y_true))),
        "Accuracy(%)": accuracy_within_tolerance,
        "R²": r2_score(y_true, y_pred),
        "Explained_Variance": explained_variance_score(y_true, y_pred),
    }


# Function to open a file dialog and select a file
def select_file(
    title="Select a File", filetypes=(("CSV Files", "*.csv"), ("All Files", "*.*"))
):
    root = Tk()
    root.withdraw()  # Hide the main window
    file_path = filedialog.askopenfilename(title=title, filetypes=filetypes)
    root.destroy()
    return file_path


# Function to open a directory dialog and select a directory
def select_directory(title="Select a Directory"):
    root = Tk()
    root.withdraw()  # Hide the main window
    directory_path = filedialog.askdirectory(title=title)
    root.destroy()
    return directory_path

def plot_metrics(metrics_df, target_name, save_dir):
    """Plots the metrics for each model in a bar chart with different colors and similar scales, and saves them as PNG files."""

    metrics = metrics_df.columns.drop('Model')
    num_metrics = len(metrics)
    model_names = metrics_df['Model'].unique()
    num_models = len(model_names)

    # Generate a color palette with enough colors for all models
    palette = sns.color_palette("husl", num_models)  # You can change "husl" to other palettes

    for metric in metrics:
        plt.figure(figsize=(10, 6))

        # Calculate min and max values for the current metric across all models
        min_val = metrics_df[metric].min()
        max_val = metrics_df[metric].max()

        for i, model in enumerate(model_names):
            values = metrics_df[metrics_df['Model'] == model][metric].values
            plt.bar(model, values, color=palette[i], label=model)  # Use color from palette

        plt.title(f'{metric} for {target_name}')
        plt.ylabel(metric)
        plt.ylim(min_val - (max_val - min_val) * 0.1, max_val + (max_val - min_val) * 0.1)  # Set y-axis limits with some padding
        plt.xticks(rotation=45, ha='right')
        plt.legend()  # Show legend
        plt.tight_layout()

        filename = os.path.join(save_dir, f'{target_name}_{metric}.png')
        plt.savefig(filename)
        plt.close()


def run_ml_valid_models():
    """
    Main function to validate machine learning models
    """
    # Get file paths using Tkinter
    print("Please select the feature dataset file for validation...")
    features_path = select_file(title="Select Feature Dataset (CSV)")

    print("Please select the target dataset file for validation...")
    targets_path = select_file(title="Select Target Dataset (CSV)")

    if not features_path or not targets_path:
        print("File selection canceled. Exiting...")
        exit()

    # Load datasets
    valid_features = pd.read_csv(features_path)
    valid_targets = pd.read_csv(targets_path)

    # Replace values 0.0 with 0.00005 in all columns of the targets DataFrame
    valid_targets.replace(0.0, 0.00005, inplace=True)

    columns_features = [
        "left_heel_x_mean",
        "left_heel_y_mean",
        "left_foot_index_x_mean",
        "left_foot_index_y_mean",
        "right_heel_x_mean",
        "right_heel_y_mean",
        "right_foot_index_x_mean",
        "right_foot_index_y_mean",
        "left_heel_x_var",
        "left_heel_y_var",
        "left_foot_index_x_var",
        "left_foot_index_y_var",
        "right_heel_x_var",
        "right_heel_y_var",
        "right_foot_index_x_var",
        "right_foot_index_y_var",
        "left_heel_x_speed",
        "left_heel_y_speed",
        "left_foot_index_x_speed",
        "left_foot_index_y_speed",
        "right_heel_x_speed",
        "right_heel_y_speed",
        "right_foot_index_x_speed",
        "right_foot_index_y_speed",
        "left_step_length",
        "right_step_length",
        "left_heel_x_range",
        "left_heel_y_range",
        "left_foot_index_x_range",
        "left_foot_index_y_range",
        "right_heel_x_range",
        "right_heel_y_range",
        "right_foot_index_x_range",
        "right_foot_index_y_range",
    ]

    columns_targets = [
        "step_length",
        "stride_length",
        "supp_base",
        "supp_time_single",
        "supp_time_double",
        "step_width",
        "stride_width",
        "stride_velocity",
        "step_time",
        "stride_time",
    ]

    valid_features = valid_features[columns_features]
    valid_targets = valid_targets[columns_targets]

    # Modificar esta parte para permitir seleção do diretório dos modelos
    print("Please select the models directory...")
    models_path = select_directory(title="Select Models Directory")

    if not models_path:
        print("Models directory selection canceled. Exiting...")
        exit()

    # Process each target
    for target in tqdm(columns_targets, desc="Processing targets"):
        print(f"\nProcessing target: {target}")
        target_path = os.path.join(models_path, target)

        if not os.path.exists(target_path):
            print(f"No models found for target: {target}")
            continue

        target_metrics = []

        # Load scaler parameters
        scaler_path = os.path.join(target_path, "scaler_params.json")
        if os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                scaler_params = json.load(f)
                scaler = StandardScaler()
                scaler.mean_ = np.array(scaler_params["mean"])
                scaler.scale_ = np.array(scaler_params["scale"])
                valid_features_scaled = scaler.transform(valid_features)
            print(f"Loaded scaler parameters from: {scaler_path}")

        # Process each model
        model_files = [
            f for f in os.scandir(target_path) if f.name.endswith("_model.pkl")
        ]
        for model_file in tqdm(
            model_files, desc=f"Validating models for {target}", unit="model"
        ):
            model_name = model_file.name.replace("_model.pkl", "")
            print(f"\nValidating model: {model_name}")

            # Load and validate model
            with open(model_file.path, "rb") as f:
                model = pickle.load(f)
            print(f"Loaded model from: {model_file.path}")

            y_pred = model.predict(valid_features_scaled)
            metrics = calculate_metrics(valid_targets[target], y_pred)
            metrics["Model"] = model_name
            target_metrics.append(metrics)


        # Save metrics
        if target_metrics:
            metrics_df = pd.DataFrame(target_metrics)
            metrics_file = os.path.join(target_path, f"validation_metrics.csv")
            metrics_df.to_csv(metrics_file, index=False)
            print(f"Validation metrics saved to: {metrics_file}")

            # The plots will be saved in the same directory as the CSV
            plots_dir = target_metrics_path  # No need to create a subdirectory

            # Plot the metrics and save them as PNGs
            metrics_df = pd.read_csv(metrics_file)
            plot_metrics(metrics_df, target, plots_dir)


    print("\nValidation completed.")


if __name__ == "__main__":
    run_ml_valid_models()
