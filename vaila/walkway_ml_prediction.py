# =============================================================================
# walkway_ml_prediction.py
# =============================================================================
# Author: Abel Gonçalves Chinaglia
# Ph.D. Candidate in PPGRDF - FMRP - USP
# Date: 05 Feb. 2025
# Update: 11 Feb. 2025
# Python Version: 3.8+

# Description:
# ------------
# This script loads pre-trained machine learning models to predict gait-related
# features from input data. It provides a graphical user interface (GUI) for
# selecting the metrics to predict and the input data file.

# Key Features:
# --------------
# - Loads pre-trained models (joblib format) for various gait metrics
# - Offers option to use default models or custom trained models
# - Scales input features using pre-calculated scaling parameters (JSON format)
# - Handles missing scaler parameters gracefully
# - Preserves subject name and trial number in the output CSV file
# - Provides a user-friendly Tkinter-based GUI for metric selection and file input
# - Automatically closes the GUI window after prediction is complete

# Execution:
# ----------
# - Run the script:
#   $ python walkway_ml_prediction.py
# - Select the input CSV file containing the features
# - Choose whether to use default models or custom models:
#   * Default models: Uses pre-trained models from vaila/vaila/models directory
#   * Custom models: Allows selection of a directory containing user-trained models
# - Choose the metrics to predict using the GUI
# - The results will be saved in a timestamped directory within the current
#   working directory

# Model Directory Structure:
# -------------------------
# The models directory should contain:
# /models
#   - step_length.pkl
#   - step_length_scaler_params.json
#   - step_time.pkl
#   - step_time_scaler_params.json
#   ... (and so on for other metrics)

# Input Data Format:
# ------------------
# The input CSV file should contain the following columns:
# - subject_name (string): The name or ID of the subject
# - trial_number (integer): The trial number for the data
# - [Other numerical features]: The features used for prediction

# Output Structure:
# -----------------
# The results are saved in a CSV file named 'result_ml_walkway.csv' within a
# timestamped directory (e.g., walkway_ml_result_20241212_103000). The CSV file
# contains the following columns:
# - subject_name
# - trial_number
# - [Predicted metrics]: The predicted values for the selected metrics

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
from rich import print
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from datetime import datetime


# Function to load models and make predictions
def predict_metrics(selected_metrics, valid_features, output_dir):
    # Perguntar ao usuário se deseja usar os modelos padrão
    use_default = messagebox.askyesno(
        "Model Selection",
        "Would you like to use the default pre-trained models?\n\n"
        + "Click 'Yes' to use default models from vaila/vaila/models\n"
        + "Click 'No' to select your own models directory",
    )

    if use_default:
        models_path = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), "models"))
        print(f"Using default models directory: {models_path}")
        if not os.path.exists(models_path):
            messagebox.showerror("Error", "Default models directory not found.")
            return
    else:
        # Permitir seleção do diretório dos modelos pelo usuário
        models_path = filedialog.askdirectory(title="Select Models Directory")
        if not models_path:
            messagebox.showwarning("Warning", "No models directory selected.")
            return

    # Solicitar diretório de saída
    output_dir = filedialog.askdirectory(title="Select Output Directory for Results")
    if not output_dir:
        messagebox.showwarning("Warning", "No output directory selected.")
        return

    results = {}
    
    for metric in selected_metrics:
        # Modificar o caminho para corresponder à estrutura atual
        model_path = os.path.join(models_path, f"{metric}.pkl")
        scaler_path = os.path.join(models_path, f"{metric}_scaler_params.json")

        if not os.path.exists(model_path):
            print(f"No model found for {metric}. Skipping.")
            print(f"Looked for model at: {model_path}")
            continue

        print(f"Using model {metric}.pkl for {metric}")
        model = joblib.load(model_path)

        valid_features_scaled = valid_features.copy()
        
        # Lista de colunas para ignorar
        columns_to_ignore = [
            "subject_name", "trial_number",  # colunas originais
            "Participant", "Trial", "Step_Block"  # colunas adicionais para ignorar
        ]
        
        # Remover todas as colunas não numéricas e as colunas especificadas
        numeric_columns = valid_features_scaled.select_dtypes(include=['float64', 'int64']).columns
        columns_to_keep = [col for col in numeric_columns if col not in columns_to_ignore]
        valid_features_numeric = valid_features_scaled[columns_to_keep]

        if os.path.exists(scaler_path):
            with open(scaler_path, "r") as f:
                scaler_params = json.load(f)

            mean = np.array(scaler_params.get("mean", []))
            std = np.array(scaler_params.get("std", []))

            if len(mean) > 0 and len(std) > 0:
                valid_features_scaled = (valid_features_numeric - mean) / std
            else:
                print(f"Invalid scaler parameters for {metric}. Using unscaled features.")
                valid_features_scaled = valid_features_numeric
        else:
            print(f"Scaler parameters for {metric} not found. Using unscaled features.")
            valid_features_scaled = valid_features_numeric

        if hasattr(model, "predict"):
            y_pred = model.predict(valid_features_scaled)
            results[metric] = y_pred
        else:
            print(f"The model {metric}.pkl does not support the 'predict' function. Skipping.")

    if results:
        results_df = pd.DataFrame(results)
        if "subject_name" in valid_features.columns and "trial_number" in valid_features.columns:
            results_df.insert(0, "trial_number", valid_features["trial_number"])
            results_df.insert(0, "subject_name", valid_features["subject_name"])
        elif "Participant" in valid_features.columns and "Trial" in valid_features.columns:
            results_df.insert(0, "trial_number", valid_features["Trial"])
            results_df.insert(0, "subject_name", valid_features["Participant"])

        # Criar timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Criar nome do arquivo com timestamp
        result_filename = f"result_ml_walkway_{timestamp}.csv"
        
        # Juntar o caminho completo
        result_file = os.path.join(output_dir, result_filename)
        
        # Salvar o arquivo
        results_df.to_csv(result_file, index=False)
        print(f"Results saved to {result_file}")
    else:
        print("No results were generated.")


# Function to select files and run prediction
def run_prediction(selected_metrics=None):
    """
    Main function to run prediction with pre-selected metrics.
    """
    print("Please select the features file...")
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

    feature_file = filedialog.askopenfilename(
        title="Select the features file", filetypes=[("CSV Files", "*.csv")]
    )
    if not feature_file:
        return

    valid_features = pd.read_csv(feature_file)
    print("Metrics selected for prediction:", selected_metrics)  # Print para depuração

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = os.path.join(os.getcwd(), f"walkway_ml_result_{timestamp}")
    os.makedirs(output_dir, exist_ok=True)

    predict_metrics(selected_metrics, valid_features, output_dir)
    messagebox.showinfo("Success", f"Results saved in {output_dir}")
    root.destroy()  # Fecha a janela principal


# Criação da interface gráfica (GUI)
root = tk.Tk()
root.title("Metric Selection for Prediction")

frame = ttk.Frame(root, padding=10)
frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))

label = ttk.Label(
    frame,
    text="Select the metrics to be evaluated:\nEnsure model directory and feature file are correctly set.",
)
label.grid(row=0, column=0, sticky=tk.W, pady=5)

metrics = [
    "step_length", "step_time", "step_width", "stride_length", "stride_time",
    "stride_velocity", "stride_width", "support_base", "support_time_doubled", "support_time_single"
]

# Cria o Listbox com seleção múltipla e insere todas as métricas
listbox = tk.Listbox(frame, selectmode=tk.MULTIPLE, height=len(metrics))
for metric in metrics:
    listbox.insert(tk.END, metric)
# Seleciona todos os itens por padrão
listbox.select_set(0, tk.END)
listbox.grid(row=1, column=0, sticky=tk.W, pady=5)

# Cria um frame para os botões "Select All", "Unselect All" e "Confirm"
button_frame = ttk.Frame(frame)
button_frame.grid(row=2, column=0, pady=10)

def select_all_metrics():
    listbox.select_set(0, tk.END)

def unselect_all_metrics():
    listbox.selection_clear(0, tk.END)

select_all_button = ttk.Button(button_frame, text="Select All", command=select_all_metrics)
select_all_button.grid(row=0, column=0, padx=5)

unselect_all_button = ttk.Button(button_frame, text="Unselect All", command=unselect_all_metrics)
unselect_all_button.grid(row=0, column=1, padx=5)

def confirm_and_run_prediction():
    # Obtém as métricas selecionadas a partir do Listbox
    selected_indices = listbox.curselection()
    selected_metrics = [listbox.get(i) for i in selected_indices]
    print("Selected metrics:", selected_metrics)
    if not selected_metrics:
        messagebox.showwarning("Warning", "No metrics selected.")
        return
    run_prediction(selected_metrics)

confirm_button = ttk.Button(button_frame, text="Confirm", command=confirm_and_run_prediction)
confirm_button.grid(row=0, column=2, padx=5)

root.mainloop()

if __name__ == "__main__":
    run_prediction()
