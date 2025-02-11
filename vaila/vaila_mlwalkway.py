import os
import platform
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
from vaila import process_gait_features, ml_models_training, ml_valid_models, walkway_ml_prediction


"""
This module creates a new window with buttons for:
1. Training new ML models
2. Validating trained models
3. Processing MediaPipe data for features
4. Running predictions with pre-trained models
"""

def run_process_gait_features():
    from vaila import run_process_gait_features
    run_process_gait_features()


def run_ml_models_training():
    from vaila import run_ml_models_training
    run_ml_models_training() 

def run_ml_valid_models():
    from vaila import run_ml_valid_models
    run_ml_valid_models()


def run_walkway_ml_prediction():
    from vaila import run_walkway_ml_prediction
    run_walkway_ml_prediction()


# GUI tk window
def run_vaila_mlwalkway_gui():
    root = tk.Tk()
    root.title("vaila ML Walkway")
    root.geometry("300x200")
    root.mainloop()
    # button to Process Gait Features, ML Models Training, ML Models Validation, ML Models Prediction
    # button to Process MediaPipe Data for Features

    # Create frame for buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(expand=True, padx=20, pady=20)

    # Create and pack buttons
    process_btn = ttk.Button(
        button_frame,
        text="Process Gait Features",
        command=lambda: run_process_gait_features()
    )
    process_btn.pack(fill='x', pady=5)

    train_btn = ttk.Button(
        button_frame, 
        text="Train ML Models",
        command=lambda: run_ml_models_training()
    )
    train_btn.pack(fill='x', pady=5)

    validate_btn = ttk.Button(
        button_frame,
        text="Validate ML Models", 
        command=lambda: run_ml_valid_models()
    )
    validate_btn.pack(fill='x', pady=5)

    predict_btn = ttk.Button(
        button_frame,
        text="Run ML Predictions",
        command=lambda: run_walkway_ml_prediction()
    )
    predict_btn.pack(fill='x', pady=5)

    # With the button Process Gait Features, run the process_gait_features file
    process_btn.pack(fill='x', pady=5)

    # With the button Process MediaPipe Data for Features, run the process_mediapipe_data file
    process_btn.pack(fill='x', pady=5)


    # Start the mainloop
    root.mainloop()     
 

if __name__ == "__main__":
    run_vaila_mlwalkway_gui()

