# =============================================================================
# walkway_ml_prediction.py
# =============================================================================
# Author: Abel Gonçalves Chinaglia
# Ph.D. Candidate in PPGRDF - FMRP - USP
# Date: 05 Feb. 2025
# Version: 1.0.0
# Python Version: 3.8+

# Description:
# ------------
# This script loads pre-trained machine learning models to predict gait-related
# features from input data. It provides a graphical user interface (GUI) for
# selecting the metrics to predict and the input data file.

# Key Features:
# --------------
# - Loads pre-trained models (joblib format) for various gait metrics.
# - Scales input features using pre-calculated scaling parameters (JSON format).
# - Handles missing scaler parameters gracefully.
# - Preserves subject name and trial number in the output CSV file.
# - Provides a user-friendly Tkinter-based GUI for metric selection and file input.
# - Automatically closes the GUI window after prediction is complete.

# Execution:
# ----------
# - Ensure that the pre-trained models and scaler parameter files are present
#   in the 'models' directory.  These should have been created by a training
#   script (e.g., ML_models_training.py).
# - Run the script:
#   $ python walkway_ml_prediction.py
# - Select the input CSV file containing the features.
# - Choose the metrics to predict using the GUI.
# - The results will be saved in a timestamped directory within the current
#   working directory.

# Input Data Format:
# ------------------
# The input CSV file should contain the following columns:
# - subject_name (string): The name or ID of the subject.
# - trial_number (integer): The trial number for the data.
# - [Other numerical features]: The features used for prediction.

# Output Structure:
# -----------------
# The results are saved in a CSV file named 'result_ml_walkway.csv' within a
# timestamped directory (e.g., walkway_ml_result_20241212_103000). The CSV file
# contains the following columns:
# - subject_name
# - trial_number
# - [Predicted metrics]: The predicted values for the selected metrics.

# License:
# --------
# This program is licensed under the GNU Lesser General Public License v3.0.
# For more details, visit: https://www.gnu.org/licenses/lgpl-3.0.html

# =============================================================================

import os
import json
import joblib
import numpy as np
import pandas as pd
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime

# Function to load models and make predictions
def predict_metrics(selected_metrics, valid_features, output_dir):
    # Permitir seleção do diretório dos modelos
    models_path = filedialog.askdirectory(title="Select Models Directory")
    if not models_path:
        messagebox.showwarning("Warning", "No models directory selected.")
        return

    results = {}

    for metric in selected_metrics:
        model_path = os.path.join(models_path, metric, f"{metric}_model.pkl")
        scaler_path = os.path.join(models_path, metric, "scaler_params.json")

        if not os.path.exists(model_path):
            print(f"No model found for {metric}. Skipping.")
            continue

        print(f"Using model {metric}_model.pkl for {metric}")
        model = joblib.load(model_path)

        valid_features_scaled = valid_features.copy()

        valid_features_numeric = valid_features_scaled.drop(["subject_name", "trial_number"], axis=1) # Seleciona apenas as colunas numéricas para escalar

        if os.path.exists(scaler_path):
            with open(scaler_path, 'r') as f:
                scaler_params = json.load(f)

            mean = np.array(scaler_params.get("mean", []))
            std = np.array(scaler_params.get("std", []))

            if len(mean) > 0 and len(std) > 0:
                valid_features_scaled = (valid_features_numeric - mean) / std # Escala apenas as features numéricas
            else:
                print(f"Invalid scaler parameters for {metric}. Using unscaled features.")
                valid_features_scaled = valid_features_numeric # Usa as features numéricas não escaladas


        if hasattr(model, 'predict'):
            y_pred = model.predict(valid_features_scaled)
            results[metric] = y_pred
        else:
            print(f"The model {metric}_model.pkl does not support the 'predict' function. Skipping.")

    if results:
        results_df = pd.DataFrame(results)

        if "subject_name" in valid_features.columns and "trial_number" in valid_features.columns:
            # Insere as colunas 'subject_name' e 'trial_number' no início
            results_df.insert(0, "trial_number", valid_features["trial_number"])
            results_df.insert(0, "subject_name", valid_features["subject_name"])

        result_file = os.path.join(output_dir, 'result_ml_walkway.csv')
        results_df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")

    else:
        print("No results were generated.")

# Function to select files and run prediction
def run_prediction():
    feature_file = filedialog.askopenfilename(title="Select the features file", filetypes=[("CSV Files", "*.csv")])
    if not feature_file:
        return

    valid_features = pd.read_csv(feature_file)

    selected_metrics = [metric for metric, var in metric_vars.items() if var.get() == 1]
    if not selected_metrics:
        messagebox.showwarning("Warning", "No metrics selected.")
        return

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"walkway_ml_result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    predict_metrics(selected_metrics, valid_features, output_dir)
    messagebox.showinfo("Success", f"Results saved in {output_dir}")
    root.destroy()  # Fecha a janela principal

# Create GUI
root = tk.Tk()
root.title("Metric Selection for Prediction")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label = ttk.Label(frame, text="Select the metrics to be evaluated:\nEnsure model directory and feature file are correctly set.")
label.grid(row=0, column=0, sticky=tk.W, pady=5)

metrics = [
    "step_length", "step_time", "step_width", "stride_length", "stride_time",
    "stride_velocity", "stride_width", "support_base", "support_time_doubled", "support_time_single"
]
metric_vars = {metric: tk.IntVar() for metric in metrics}

for i, metric in enumerate(metrics):
    ttk.Checkbutton(frame, text=metric, variable=metric_vars[metric]).grid(row=i+1, column=0, sticky=tk.W)

run_button = ttk.Button(frame, text="Run Prediction", command=run_prediction)
run_button.grid(row=len(metrics)+1, column=0, pady=10)

root.mainloop()
