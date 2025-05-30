"""
Project: vailá
Script: yolotrain.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 May 2025
Update Date: 24 May 2025
Version: 0.1.0

Description:
    This script provides a graphical user interface (GUI) for training YOLO models using the Ultralytics YOLO library.
    It allows users to select a dataset, configure training parameters, and start the training process.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
        python yolotrain.py

Requirements:
    - Python 3.x
    - Ultralytics YOLO
    - Tkinter (for GUI operations)
    - Additional dependencies as imported (numpy, csv, etc.)

License:
    This project is licensed under the terms of the MIT License (or another applicable license).

Change History:
    - First version.

Notes:
    - Ensure that all dependencies are installed.
    - Since the script uses a graphical interface (Tkinter) for model selection and configuration, a GUI-enabled environment is required.
"""

import os
import sys
import pathlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext
from ultralytics import YOLO
import torch
import datetime
import re  # For validation of the project/run name


# --- Class to redirect the console output to the Text Widget ---
class ConsoleRedirector:
    """
    Redirects the standard output (stdout) to a Tkinter text widget,
    allowing the user to view the training progress in the GUI.
    """

    def __init__(self, text_widget):
        self.text_widget = text_widget
        self.original_stdout = sys.stdout  # Saves a reference to the original stdout

    def write(self, text):
        """Writes the text to the widget and the original stdout."""
        self.text_widget.insert(tk.END, text)
        self.text_widget.see(tk.END)  # Automatically scrolls to the end
        self.text_widget.update_idletasks()  # Forces the GUI to update
        self.original_stdout.write(text)  # Also writes to the original console

    def flush(self):
        """Clears the buffer of the original stdout."""
        self.original_stdout.flush()


# --- Main Application Class ---
class YOLOTrainApp(tk.Tk):
    """
    Graphical User Interface (GUI) to configure and start the training of YOLO models.
    """

    def __init__(self):
        super().__init__()
        self.title("YOLO Training Interface - vailá toolbox")
        self.geometry("800x700")  # Initial window size
        self.configure(padx=10, pady=10)  # Global padding for the window

        # Set the window to be always on top
        self.attributes("-topmost", True)

        # Tkinter variables to store the GUI configurations
        self.dataset_path = tk.StringVar()
        self.yaml_path = tk.StringVar()
        self.base_model = tk.StringVar(
            value="yolov8m.pt"
        )  # YOLOv8 medium model as default
        self.epochs = tk.StringVar(value="100")  # Default number of epochs
        self.batch_size = tk.StringVar(value="16")  # Default batch size
        self.img_size = tk.StringVar(value="640")  # Default image size
        self.device = tk.StringVar(
            value="cuda" if torch.cuda.is_available() else "cpu"
        )  # Default device
        self.project_name = tk.StringVar(
            value="yolo_custom_run"
        )  # Default project/run name

        # Validation configurations for numeric fields
        self.validate_numeric_cmd = self.register(self._validate_numeric)

        self.create_widgets()  # Creates the elements of the interface

    def create_widgets(self):
        """Creates and organizes the widgets in the main window."""
        # Configure grid for responsive resizing
        self.grid_rowconfigure(8, weight=1)  # Makes the console expand vertically
        self.grid_columnconfigure(
            1, weight=1
        )  # Makes the input field expand horizontally

        # Labels and Input Fields for the Configurations

        # 1. Dataset Path
        tk.Label(self, text="Dataset Folder:").grid(
            row=0, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.dataset_path, width=60).grid(
            row=0, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Button(self, text="Browse", command=self.browse_dataset).grid(
            row=0, column=2, padx=5, pady=5
        )

        # 2. YAML Path
        tk.Label(self, text="Dataset YAML File:").grid(
            row=1, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.yaml_path, width=60).grid(
            row=1, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Button(self, text="Browse", command=self.browse_yaml).grid(
            row=1, column=2, padx=5, pady=5
        )

        # 3. Base Model
        tk.Label(self, text="YOLO Base Model:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        tk.OptionMenu(
            self,
            self.base_model,
            "yolov8n.pt",
            "yolov8s.pt",
            "yolov8m.pt",
            "yolov8l.pt",
            "yolov8x.pt",
        ).grid(row=2, column=1, padx=5, pady=5, sticky="w")

        # 4. Epochs
        tk.Label(self, text="Epochs:").grid(row=3, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(
            self,
            textvariable=self.epochs,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=3, column=1, padx=5, pady=5, sticky="w")

        # 5. Batch Size
        tk.Label(self, text="Batch Size:").grid(
            row=4, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(
            self,
            textvariable=self.batch_size,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=4, column=1, padx=5, pady=5, sticky="w")

        # 6. Image Size
        tk.Label(self, text="Image Size (px):").grid(
            row=5, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(
            self,
            textvariable=self.img_size,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # 7. Device (GPU/CPU)
        tk.Label(self, text="Device:").grid(row=6, column=0, sticky="e", padx=5, pady=5)
        tk.OptionMenu(self, self.device, "cpu", "cuda").grid(
            row=6, column=1, padx=5, pady=5, sticky="w"
        )

        # 8. Project Name (for the output folder in runs/train)
        tk.Label(self, text="Run Name:").grid(
            row=7, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.project_name, width=30).grid(
            row=7, column=1, padx=5, pady=5, sticky="w"
        )

        # Start Training Button
        self.start_button = tk.Button(
            self,
            text="Start Training",
            command=self.start_training_thread,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
        )
        self.start_button.grid(
            row=9, column=0, columnspan=3, pady=20, sticky="nsew"
        )  # Expand in all directions

        # Training Output Console (to display training logs)
        tk.Label(self, text="Training Output:").grid(
            row=10, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        self.console = scrolledtext.ScrolledText(
            self, height=15, wrap="word"
        )  # 'wrap="word"' avoids word breaks at the end of the line
        self.console.grid(
            row=11, column=0, columnspan=3, padx=10, pady=5, sticky="nsew"
        )  # Expand in all directions

    def browse_dataset(self):
        """Allows the user to select the root directory of the dataset."""
        self.attributes("-topmost", False)  # Remove topmost before opening dialog
        path = filedialog.askdirectory(
            title="Select the Root Directory of your YOLO Dataset"
        )
        self.attributes("-topmost", True)  # Restore topmost after dialog is closed
        if path:
            self.dataset_path.set(path)
            # Tries to fill the YAML path automatically if a data.yaml exists
            default_yaml = os.path.join(path, "data.yaml")
            if os.path.exists(default_yaml):
                self.yaml_path.set(default_yaml)
            else:
                self.yaml_path.set("")  # Clears if not found

    def browse_yaml(self):
        """Allows the user to select the dataset YAML file."""
        self.attributes("-topmost", False)  # Remove topmost before opening dialog
        file = filedialog.askopenfilename(
            title="Select your dataset.yaml file",
            filetypes=[("YAML files", "*.yaml *.yml")],
        )
        self.attributes("-topmost", True)  # Restore topmost after dialog is closed
        if file:
            self.yaml_path.set(file)

    def _validate_numeric(self, value):
        """Validates if the input value is numeric."""
        return value.isdigit() or value == ""

    def start_training_thread(self):
        """Starts the training in a separate thread to avoid blocking the GUI."""
        self.start_button.config(
            state=tk.DISABLED
        )  # Disables the button while training
        self.console.delete("1.0", tk.END)  # Clears the previous console
        self.console.insert(tk.END, "Starting training...\n")

        # Starts the training in a new thread
        training_thread = threading.Thread(target=self._run_training_logic)
        training_thread.daemon = True  # Defines as a daemon to ensure the thread terminates with the application
        training_thread.start()

    def _run_training_logic(self):
        """Contains the main logic of the YOLO training."""
        original_stdout = sys.stdout  # Saves the original stdout
        sys.stdout = ConsoleRedirector(self.console)  # Redirects stdout to the widget

        try:
            # 1. Validation of the inputs
            dataset_folder = self.dataset_path.get()
            yaml_file = self.yaml_path.get()
            epochs_val = self.epochs.get()
            batch_val = self.batch_size.get()
            imgsz_val = self.img_size.get()
            selected_device = self.device.get()
            run_name = self.project_name.get()

            if not dataset_folder or not os.path.isdir(dataset_folder):
                messagebox.showerror(
                    "Error", "Please select a valid dataset directory."
                )
                return

            if not yaml_file or not os.path.exists(yaml_file):
                messagebox.showerror(
                    "Error", f"Dataset YAML file not found: {yaml_file}"
                )
                return

            try:
                epochs = int(epochs_val)
                batch = int(batch_val)
                imgsz = int(imgsz_val)
                if epochs <= 0 or batch <= 0 or imgsz <= 0:
                    messagebox.showerror(
                        "Error",
                        "Epochs, Batch Size and Image Size must be positive numbers.",
                    )
                    return
            except ValueError:
                messagebox.showerror(
                    "Error",
                    "Please insert valid numeric values for Epochs, Batch Size and Image Size.",
                )
                return

            if selected_device == "cuda" and not torch.cuda.is_available():
                messagebox.showwarning(
                    "Warning",
                    "CUDA (GPU) selected, but not available. Training will be executed on CPU.",
                )
                selected_device = "cpu"  # Forces to CPU

            # Basic validation for run_name
            if not run_name or not re.match(r"^[a-zA-Z0-9_-]+$", run_name):
                messagebox.showerror(
                    "Error", "Invalid run name. Use only letters, numbers, '-' or '_'."
                )
                return

            # 2. Prepare the arguments for the YOLO training
            print(f"Loading base model: {self.base_model.get()}")
            model = YOLO(
                self.base_model.get()
            )  # The Ultralytics library downloads the model if it is not found

            output_project_dir = os.path.join(dataset_folder, "runs")
            # The Ultralytics library creates runs/train/run_name automatically

            self.console.insert(tk.END, "\n--- Training Configurations ---\n")
            self.console.insert(tk.END, f"  Dataset YAML: {yaml_file}\n")
            self.console.insert(tk.END, f"  Base Model: {self.base_model.get()}\n")
            self.console.insert(tk.END, f"  Epochs: {epochs}\n")
            self.console.insert(tk.END, f"  Batch Size: {batch}\n")
            self.console.insert(tk.END, f"  Image Size: {imgsz}\n")
            self.console.insert(tk.END, f"  Device: {selected_device}\n")
            self.console.insert(
                tk.END, f"  Output Folder (Project): {output_project_dir}\n"
            )
            self.console.insert(tk.END, f"  Run Name: {run_name}\n")
            self.console.insert(tk.END, "-------------------------------------\n\n")

            # 3. Start the YOLO training
            print("Starting the YOLO training... Please wait.")
            results = model.train(
                data=yaml_file,
                epochs=epochs,
                imgsz=imgsz,
                batch=batch,
                device=selected_device,
                project=output_project_dir,
                name=run_name,
                exist_ok=True,  # Allows using an existing run name, overwriting or adding '_n'
                # patience=50, # Example: Adding early stopping
                # optimizer='AdamW', # Example: Selecting optimizer
            )

            # 4. Post-training information
            best_model_path = os.path.join(
                output_project_dir, run_name, "weights", "best.pt"
            )
            print(f"\nTraining completed successfully!")
            print(f"Best model saved in: {best_model_path}")
            print(
                f"Detailed results (graphics, logs) are in: {os.path.join(output_project_dir, run_name)}"
            )

            messagebox.showinfo(
                "Training Completed",
                f"The YOLO training completed successfully!\n"
                f"Best model saved in: '{best_model_path}'\n"
                f"Detailed results (graphics, logs) are in: '{os.path.join(output_project_dir, run_name)}'",
            )

        except Exception as e:
            print(f"\nError during training: {str(e)}")
            messagebox.showerror(
                "Training Error", f"An error occurred during training:\n{str(e)}"
            )
        finally:
            sys.stdout = original_stdout  # Restores the original stdout
            self.start_button.config(state=tk.NORMAL)  # Re-enables the button


# --- Application Execution Entry Point ---
def run_yolotrain_gui():
    """
    Entry point function to run the YOLO Training GUI.
    This function handles global configurations and starts the Tkinter application.
    """
    print(f"Running script: {pathlib.Path(__file__).name}")
    print(f"Script directory: {pathlib.Path(__file__).parent.resolve()}")

    # Ensures that KMP_DUPLICATE_LIB_OK is configured to avoid library conflicts
    os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
    # Limits the number of threads to avoid conflicts, good practice for stable training
    torch.set_num_threads(1)

    app = YOLOTrainApp()
    app.mainloop()


if __name__ == "__main__":
    run_yolotrain_gui()
