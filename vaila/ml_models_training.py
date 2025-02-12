# =============================================================================
# ML_models_training.py
# =============================================================================
# Author: Abel Gonçalves Chinaglia
# Ph.D. Candidate in PPGRDF - FMRP - USP
# Date: 05 Feb. 2025
# Version: 1.0.0
# Python Version: 3.8+

# Description:
# ------------
# This script processes datasets to train and evaluate multiple machine learning
# models for predicting gait-related features based on human pose data extracted
# from kinematic signals. It uses K-Fold Cross-Validation for model evaluation
# and trains final models with the complete dataset.

# Key Features:
# --------------
# - Support for multiple regression models: XGBoost, KNN, MLP, SVR, RandomForest,
#   GradientBoosting, and LinearRegression.
# - Implements K-Fold Cross-Validation (10 folds) for robust model evaluation
# - Calculates detailed metrics for model evaluation, including MSE, RMSE, MAE,
#   R², and others, for each prediction target.
# - StandardScaler normalization ensures data consistency across models.
# - Organized output structure for saving models and metrics
# - Automatically replaces zero values (0.0) in target datasets with a small
#   non-zero constant to ensure numerical stability in model training.
# - Interactive file selection through GUI dialogs
# - Progress tracking with progress bars
# - Completion notification through GUI dialog

# Execution:
# ----------
# - Run the script:
#   $ python ML_models_training.py
# - Select the feature dataset file (CSV) when prompted
# - Select the target dataset file (CSV) when prompted
# - The script will automatically create necessary directories and process the data
# - A completion message will appear when finished

# Output Structure:
# -----------------
# - Models are saved in 'models/<target>/' directories, where
#   `<target>` corresponds to the predicted feature (e.g., step_length)
# - Each model is saved as a pickle file (.pkl) with its corresponding
#   scaler parameters in JSON format
# - Cross-validation metrics are saved in 'metrics/<target>/' directories
#   as CSV files

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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.svm import SVR
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression
from xgboost import XGBRegressor
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
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns


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
        "Accuracy(%)": accuracy_within_tolerance,  # Percentage of predictions within the interval
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


# Function to select directory
def select_directory(title="Select Directory to Save Models"):
    root = Tk()
    root.withdraw()  # Hide the main window
    directory = filedialog.askdirectory(title=title)
    root.destroy()
    return directory

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


def run_ml_models_training():
    """
    Main function to train machine learning models
    """

    # Get current timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Get file paths using Tkinter
    print("Please select the feature dataset file...")
    features_path = select_file(title="Select Feature Dataset (CSV)")

    print("Please select the target dataset file...")
    targets_path = select_file(title="Select Target Dataset (CSV)")

    print("Please select where to save the models...")
    save_directory = select_directory()

    if not features_path or not targets_path or not save_directory:
        print("File/directory selection canceled. Exiting...")
        exit()

    # Create output paths with timestamp
    output_base_path = os.path.join(save_directory, f"models_{timestamp}")
    metrics_base_path = os.path.join(save_directory, f"metrics_{timestamp}")
    os.makedirs(output_base_path, exist_ok=True)
    os.makedirs(metrics_base_path, exist_ok=True)

    print(f"\nModels will be saved in: {output_base_path}")
    print(f"Metrics will be saved in: {metrics_base_path}\n")

    # Load datasets
    features_df = pd.read_csv(features_path)
    targets_df = pd.read_csv(targets_path)

    # Replace values 0.0 with 0.00005 in all columns of the targets_df DataFrame
    targets_df.replace(0.0, 0.00005, inplace=True)

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

    # Removing the separation of validation data
    train_features = features_df[columns_features]
    train_targets = targets_df[columns_targets]

    # StandardScaler for normalization
    scaler = StandardScaler()
    train_features_scaled = scaler.fit_transform(train_features)  # Fit with all data

    # Models
    models = {
        "XGBoost": XGBRegressor(objective="reg:squarederror", random_state=15),
        "KNN": KNeighborsRegressor(n_neighbors=5),
        "MLP": MLPRegressor(hidden_layer_sizes=(100,), max_iter=1000, random_state=15),
        "SVR": SVR(kernel="rbf", C=1.0, epsilon=0.1),
        "RandomForest": RandomForestRegressor(n_estimators=100, random_state=15),
        "GradientBoost": GradientBoostingRegressor(n_estimators=100, random_state=15),
        "LinearRegression": LinearRegression(),
    }

    for target in columns_targets:
        print(f"\nProcessing target: {target}")
        target_output_path = os.path.join(output_base_path, target)
        target_metrics_path = os.path.join(metrics_base_path, target)
        os.makedirs(target_output_path, exist_ok=True)
        os.makedirs(target_metrics_path, exist_ok=True)

        cv_metrics = []

        # Progress bar for model training
        for model_name, model in tqdm(
            models.items(), desc=f"Training Models for {target}", unit="model"
        ):
            # K-Fold Cross-Validation (with all data)
            kf = KFold(n_splits=10, shuffle=True, random_state=42)
            cv_results = []

            # Progress bar for K-Fold Cross-Validation
            for train_idx, test_idx in tqdm(
                kf.split(train_features_scaled),
                desc=f"K-Fold CV for {model_name}",
                total=10,
                unit="fold",
            ):
                X_fold_train, X_fold_test = (
                    train_features_scaled[train_idx],
                    train_features_scaled[test_idx],
                )
                y_fold_train, y_fold_test = (
                    train_targets[target].iloc[train_idx],
                    train_targets[target].iloc[test_idx],
                )
                model.fit(X_fold_train, y_fold_train)
                y_fold_pred = model.predict(X_fold_test)
                cv_results.append(calculate_metrics(y_fold_test, y_fold_pred))

            averaged_metrics = {
                key: np.mean([m[key] for m in cv_results]) for key in cv_results[0]
            }
            cv_metrics.append({"Model": model_name, **averaged_metrics})

            # Train final model with all data after cross-validation
            model.fit(train_features_scaled, train_targets[target])

            # Save model (pickle)
            model_path = os.path.join(target_output_path, f"{model_name}_model.pkl")
            os.makedirs(os.path.dirname(model_path), exist_ok=True)
            with open(model_path, "wb") as f:
                pickle.dump(model, f)
            print(f"Model saved to: {model_path}")

            # Save scaler (json)
            scaler_params = {
                "mean": scaler.mean_.tolist(),
                "scale": scaler.scale_.tolist(),
            }
            scaler_path = os.path.join(target_output_path, f"scaler_params.json")
            os.makedirs(os.path.dirname(scaler_path), exist_ok=True)
            with open(scaler_path, "w") as f:
                json.dump(scaler_params, f)
            print(f"Scaler parameters saved to: {scaler_path}")

        # Save metrics
        metrics_file = os.path.join(target_metrics_path, "cross_validation_metrics.csv")
        pd.DataFrame(cv_metrics).to_csv(metrics_file, index=False)
        print(f"Metrics saved to: {metrics_file}")

        # The plots will be saved in the same directory as the CSV
        plots_dir = target_metrics_path  # No need to create a subdirectory

        # Plot the metrics and save them as PNGs
        metrics_df = pd.read_csv(metrics_file)
        plot_metrics(metrics_df, target, plots_dir)  # Pass target_metrics_path as save_dir



    print("\nTraining completed.")


if __name__ == "__main__":
    run_ml_models_training()
