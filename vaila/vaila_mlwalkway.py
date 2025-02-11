"""
vaila_mlwalkway.py

Create by: Abel G. Chinaglia & Paulo R. P. Santiago
LaBioCoM - Laboratory of Biomechanics and Motor Control
Date: 10.Feb.2025  
Update: 11.Feb

This module provides a graphical user interface (GUI) for executing various machine learning (ML) tasks related to gait analysis using the VAILA system. The GUI includes buttons for:
1. Processing gait features from MediaPipe data (use pixel data from MediaPipe in button vailá -> Makerless 2D).
2. Training ML models (use features in extract  -> ).
3. Validating trained ML models.
4. Running ML predictions using pre-trained models.

Each button triggers the respective function that executes the corresponding ML pipeline.

"""

import os
import platform
import subprocess
import sys
import tkinter as tk
from tkinter import ttk, messagebox, Toplevel

"""
This module creates a new window with buttons for:
1. Training new ML models
2. Validating trained models
3. Processing MediaPipe data for features
4. Running predictions with pre-trained models
"""


def run_process_gait_features():
    from vaila.process_gait_features import run_process_gait_features

    run_process_gait_features()


def run_ml_models_training():
    from vaila.ml_models_training import run_ml_models_training

    run_ml_models_training()


def run_ml_valid_models():
    from vaila.ml_valid_models import run_ml_valid_models

    run_ml_valid_models()


def run_walkway_ml_prediction():
    from vaila.walkway_ml_prediction import run_prediction

    run_prediction()


# GUI tk window
def run_vaila_mlwalkway_gui():
    root = tk.Tk()
    root.title("vaila ML Walkway")
    root.geometry("300x200")

    # Create frame for buttons
    button_frame = ttk.Frame(root)
    button_frame.pack(expand=True, padx=20, pady=20)

    # Create and pack buttons
    process_btn = ttk.Button(
        button_frame,
        text="Process Gait Features",
        command=lambda: run_process_gait_features(),
    )
    process_btn.pack(fill="x", pady=5)

    train_btn = ttk.Button(
        button_frame, text="Train ML Models", command=lambda: run_ml_models_training()
    )
    train_btn.pack(fill="x", pady=5)

    validate_btn = ttk.Button(
        button_frame, text="Validate ML Models", command=lambda: run_ml_valid_models()
    )
    validate_btn.pack(fill="x", pady=5)

    predict_btn = ttk.Button(
        button_frame,
        text="Run ML Predictions",
        command=lambda: run_walkway_ml_prediction(),
    )
    predict_btn.pack(fill="x", pady=5)

    root.mainloop()


if __name__ == "__main__":
    run_vaila_mlwalkway_gui()
