"""
Project: vailá Multimodal Toolbox
Script: yolotrain.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 May 2025
Update Date: 09 June 2025
Version: 0.0.3

Description:
    This script provides a graphical user interface (GUI) for training YOLO models using the Ultralytics YOLO library.
    It allows users to select a dataset, configure training parameters, and start the training process.
    Compatible with data labeled using AnyLabeling tool.

Usage:
    Run the script from the command line:
        python yolotrain.py

Requirements:
    - Python 3.x
    - Ultralytics YOLO
    - Tkinter (for GUI operations)
    - Additional dependencies as imported (numpy, csv, etc.)

License:
    This project is licensed under the terms of GNU General Public License v3.0.

Change History:
    - v0.0.3: Added support for AnyLabeling data, improved UI with ttk.Combobox, added more YOLO versions
    - v0.0.2: Added validation and threading support
    - v0.0.1: First version

Notes:
    - Ensure that all dependencies are installed.
    - For AnyLabeling users: Export your annotations in YOLO format or convert them using provided instructions.
"""

import os
import sys
import pathlib
import threading
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from ultralytics import YOLO
import torch
import datetime
import re  # For validation of the project/run name
import webbrowser


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
        self.geometry("900x800")  # Initial window size
        self.configure(padx=10, pady=10)  # Global padding for the window

        # Set the window to be always on top
        self.attributes("-topmost", True)

        # Available YOLO models
        self.available_models = [
            "yolov8n.pt",  # nano
            "yolov8s.pt",  # small
            "yolov8m.pt",  # medium
            "yolov8l.pt",  # large
            "yolov8x.pt",  # extra large
            "yolov8n-seg.pt",  # nano segmentation
            "yolov8s-seg.pt",  # small segmentation
            "yolov8m-seg.pt",  # medium segmentation
            "yolov8l-seg.pt",  # large segmentation
            "yolov8x-seg.pt",  # extra large segmentation
            "yolov5n.pt",  # YOLOv5 nano
            "yolov5s.pt",  # YOLOv5 small
            "yolov5m.pt",  # YOLOv5 medium
            "yolov5l.pt",  # YOLOv5 large
            "yolov5x.pt",  # YOLOv5 extra large
            "yolov9c.pt",  # YOLOv9 compact
            "yolov9e.pt",  # YOLOv9 enhanced
            "yolov10n.pt",  # YOLOv10 nano
            "yolov10s.pt",  # YOLOv10 small
            "yolov10m.pt",  # YOLOv10 medium
            "yolov10l.pt",  # YOLOv10 large
            "yolov10x.pt",  # YOLOv10 extra large
            "yolo11n.pt",  # YOLO11 nano
            "yolo11s.pt",  # YOLO11 small
            "yolo11m.pt",  # YOLO11 medium
            "yolo11l.pt",  # YOLO11 large
            "yolo11x.pt",  # YOLO11 extra large
        ]

        # Tkinter variables to store the GUI configurations
        self.dataset_path = tk.StringVar()
        self.yaml_path = tk.StringVar()
        self.base_model = tk.StringVar(value="yolo11m.pt")  # YOLO11 medium model as default
        self.epochs = tk.StringVar(value="100")  # Default number of epochs
        self.batch_size = tk.StringVar(value="16")  # Default batch size
        self.img_size = tk.StringVar(value="640")  # Default image size
        self.device = tk.StringVar(value="cuda" if torch.cuda.is_available() else "cpu")  # Default device
        self.project_name = tk.StringVar(value="yolo_custom_run")  # Default project/run name

        # Validation configurations for numeric fields
        self.validate_numeric_cmd = self.register(self._validate_numeric)

        self.create_widgets()  # Creates the elements of the interface

    def create_widgets(self):
        """Creates and organizes the widgets in the main window."""
        # Configure grid for responsive resizing
        self.grid_rowconfigure(12, weight=1)  # Makes the console expand vertically
        self.grid_columnconfigure(1, weight=1)  # Makes the input field expand horizontally

        # Title and Instructions
        title_label = tk.Label(
            self, 
            text="YOLO Training Interface", 
            font=("Arial", 16, "bold")
        )
        title_label.grid(row=0, column=0, columnspan=3, pady=10)

        # Help and Instructions Buttons Frame
        buttons_frame = tk.Frame(self)
        buttons_frame.grid(row=1, column=0, columnspan=3, pady=5)

        # Complete Help Button
        help_button = tk.Button(
            buttons_frame,
            text="❓ Complete Help Guide",
            command=self.show_complete_help,
            bg="#FF9800",
            fg="white",
            font=("Arial", 10, "bold")
        )
        help_button.pack(side="left", padx=5)

        # AnyLabeling Instructions Button
        instructions_button = tk.Button(
            buttons_frame,
            text="📖 AnyLabeling Data Guide",
            command=self.show_anylabeling_instructions,
            bg="#2196F3",
            fg="white",
            font=("Arial", 10)
        )
        instructions_button.pack(side="left", padx=5)

        # Labels and Input Fields for the Configurations

        # 1. Dataset Path
        tk.Label(self, text="Dataset Folder:").grid(
            row=2, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.dataset_path, width=60).grid(
            row=2, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Button(self, text="Browse", command=self.browse_dataset).grid(
            row=2, column=2, padx=5, pady=5
        )

        # 2. YAML Path
        tk.Label(self, text="Dataset YAML File:").grid(
            row=3, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.yaml_path, width=60).grid(
            row=3, column=1, padx=5, pady=5, sticky="ew"
        )
        tk.Button(self, text="Browse", command=self.browse_yaml).grid(
            row=3, column=2, padx=5, pady=5
        )

        # 3. Base Model (using ttk.Combobox instead of OptionMenu)
        tk.Label(self, text="YOLO Base Model:").grid(
            row=4, column=0, sticky="e", padx=5, pady=5
        )
        self.model_combo = ttk.Combobox(
            self,
            textvariable=self.base_model,
            values=self.available_models,
            state="readonly",
            width=20
        )
        self.model_combo.grid(row=4, column=1, padx=5, pady=5, sticky="w")
        self.model_combo.current(self.available_models.index("yolo11m.pt"))

        # 4. Epochs
        tk.Label(self, text="Epochs:").grid(row=5, column=0, sticky="e", padx=5, pady=5)
        tk.Entry(
            self,
            textvariable=self.epochs,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=5, column=1, padx=5, pady=5, sticky="w")

        # 5. Batch Size
        tk.Label(self, text="Batch Size:").grid(
            row=6, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(
            self,
            textvariable=self.batch_size,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=6, column=1, padx=5, pady=5, sticky="w")

        # 6. Image Size
        tk.Label(self, text="Image Size (px):").grid(
            row=7, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(
            self,
            textvariable=self.img_size,
            width=10,
            validate="key",
            validatecommand=(self.validate_numeric_cmd, "%P"),
        ).grid(row=7, column=1, padx=5, pady=5, sticky="w")

        # 7. Device (GPU/CPU) - using ttk.Combobox
        tk.Label(self, text="Device:").grid(row=8, column=0, sticky="e", padx=5, pady=5)
        self.device_combo = ttk.Combobox(
            self,
            textvariable=self.device,
            values=["cpu", "cuda", "mps"],  # Added MPS for Apple Silicon
            state="readonly",
            width=10
        )
        self.device_combo.grid(row=8, column=1, padx=5, pady=5, sticky="w")
        
        # Auto-detect best device
        if torch.cuda.is_available():
            self.device_combo.current(1)  # cuda
        elif torch.backends.mps.is_available():
            self.device_combo.current(2)  # mps
        else:
            self.device_combo.current(0)  # cpu

        # 8. Project Name (for the output folder in runs/train)
        tk.Label(self, text="Run Name:").grid(
            row=9, column=0, sticky="e", padx=5, pady=5
        )
        tk.Entry(self, textvariable=self.project_name, width=30).grid(
            row=9, column=1, padx=5, pady=5, sticky="w"
        )

        # Start Training Button
        self.start_button = tk.Button(
            self,
            text="🚀 Start Training",
            command=self.start_training_thread,
            bg="#4CAF50",
            fg="white",
            font=("Arial", 12, "bold"),
            height=2
        )
        self.start_button.grid(
            row=10, column=0, columnspan=3, pady=10, sticky="ew"
        )

        # Training Output Console (to display training logs)
        tk.Label(self, text="Training Output:").grid(
            row=11, column=0, columnspan=3, sticky="w", padx=5, pady=5
        )
        self.console = scrolledtext.ScrolledText(
            self, height=15, wrap="word", font=("Consolas", 9)
        )
        self.console.grid(
            row=12, column=0, columnspan=3, padx=10, pady=5, sticky="nsew"
        )

    def show_complete_help(self):
        """Shows complete help guide for using the YOLO training interface."""
        help_window = tk.Toplevel(self)
        help_window.title("YOLO Training Interface - Complete Help Guide")
        help_window.geometry("800x700")
        help_window.attributes("-topmost", True)

        # Create scrollable text
        text_frame = tk.Frame(help_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set, font=("Arial", 11))
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        help_content = """
YOLO TRAINING INTERFACE - COMPLETE HELP GUIDE
==============================================

This interface helps you train custom YOLO models for object detection. Here's everything you need to know:

═══════════════════════════════════════════════════════════════════════════════════

1. DATASET FOLDER
-----------------
• Purpose: Root directory containing your training dataset
• Structure should include:
  - images/ (training and validation images)
  - labels/ (corresponding annotation files)
  - data.yaml (dataset configuration file)
• Example: C:/my_dataset/

═══════════════════════════════════════════════════════════════════════════════════

2. DATASET YAML FILE
--------------------
• Purpose: Configuration file that tells YOLO where to find your data
• Contains:
  - Paths to train/validation images
  - Number of classes
  - Class names list
• Example content:
  train: ./images/train
  val: ./images/val
  nc: 3
  names: ['car', 'person', 'bike']

═══════════════════════════════════════════════════════════════════════════════════

3. YOLO BASE MODEL
------------------
• Purpose: Pre-trained model to start from (transfer learning)
• Model sizes (from smallest to largest):

YOLOv5 Series:
  - yolov5n.pt: Nano (fastest, least accurate)
  - yolov5s.pt: Small (balanced speed/accuracy)
  - yolov5m.pt: Medium (good balance)
  - yolov5l.pt: Large (slower, more accurate)
  - yolov5x.pt: Extra Large (slowest, most accurate)

YOLOv8 Series:
  - yolov8n.pt: Nano (1.8M parameters)
  - yolov8s.pt: Small (11.2M parameters)
  - yolov8m.pt: Medium (25.9M parameters)
  - yolov8l.pt: Large (43.7M parameters)
  - yolov8x.pt: Extra Large (68.2M parameters)

YOLOv8 Segmentation:
  - yolov8n-seg.pt to yolov8x-seg.pt (for instance segmentation)

YOLO11 Series (Latest - Recommended):
  - yolo11n.pt: Nano (newest, most efficient)
  - yolo11s.pt to yolo11x.pt: Small to Extra Large

• Recommendation: Start with yolo11m.pt for best balance

═══════════════════════════════════════════════════════════════════════════════════

4. EPOCHS
---------
• Purpose: Number of complete passes through your dataset
• Range: 50-300 (typically)
• Guidelines:
  - Small dataset: 100-200 epochs
  - Large dataset: 50-100 epochs
  - More epochs = longer training but potentially better results
  - Early stopping will prevent overfitting

═══════════════════════════════════════════════════════════════════════════════════

5. BATCH SIZE
-------------
• Purpose: Number of images processed simultaneously
• Affects:
  - Training speed (larger = faster)
  - Memory usage (larger = more GPU memory needed)
  - Training stability (smaller = more stable)
• Guidelines:
  - GPU with 8GB: batch size 16-32
  - GPU with 4GB: batch size 8-16
  - CPU training: batch size 4-8
  - If you get "out of memory" errors, reduce batch size

═══════════════════════════════════════════════════════════════════════════════════

6. IMAGE SIZE (PX)
------------------
• Purpose: Resolution that images will be resized to during training
• Common values: 320, 416, 512, 640, 800, 1024, 1280
• Impact:
  - Larger = better detection of small objects
  - Larger = slower training and inference
  - Must be multiple of 32
• Guidelines:
  - Small objects: 640-1024px
  - Large objects: 320-640px
  - Default 640px works well for most cases
  - Higher resolution = more GPU memory needed

═══════════════════════════════════════════════════════════════════════════════════

7. DEVICE
---------
• Purpose: Hardware to use for training
• Options:
  - cpu: Use CPU (slow but always available)
  - cuda: Use NVIDIA GPU (fast, requires CUDA-compatible GPU)
  - mps: Use Apple Silicon GPU (for Mac M1/M2/M3)
• Auto-detection: Interface automatically selects best available device
• GPU training is 10-50x faster than CPU

═══════════════════════════════════════════════════════════════════════════════════

8. RUN NAME
-----------
• Purpose: Name for this training session
• Creates folder: dataset/runs/train/[run_name]/
• Contains:
  - weights/ (trained models)
  - results.png (training graphs)
  - confusion_matrix.png
  - val_batch*.jpg (validation images)
• Use descriptive names: "car_detection_v1", "yolo11m_100epochs"

═══════════════════════════════════════════════════════════════════════════════════

TRAINING PROCESS EXPLAINED
===========================

1. Data Loading: YOLO loads your images and labels
2. Data Augmentation: Random transforms to improve generalization
3. Forward Pass: Model predicts object locations
4. Loss Calculation: Compares predictions to ground truth
5. Backward Pass: Updates model weights
6. Validation: Tests on validation set
7. Repeat for all epochs

Training automatically stops early if no improvement for 50 epochs.

═══════════════════════════════════════════════════════════════════════════════════

TIPS FOR BEST RESULTS
======================

Dataset Quality:
• Minimum 100 images per class
• Balanced class distribution
• Diverse backgrounds and lighting
• High-quality annotations
• Include difficult cases

Training Tips:
• Start with pre-trained models (transfer learning)
• Use data augmentation
• Monitor training graphs
• Validate on separate test set
• Use early stopping to prevent overfitting

Hardware Recommendations:
• GPU with 8GB+ VRAM for batch size 16+
• Fast SSD for large datasets
• 16GB+ RAM for data loading

Common Issues:
• "Out of memory": Reduce batch size or image size
• "No improvement": Check your data quality and labels
• "Training too slow": Use smaller model or reduce image size
• "Poor accuracy": More data, better annotations, or larger model

═══════════════════════════════════════════════════════════════════════════════════

OUTPUT FILES
============

After training, you'll find in runs/train/[run_name]/:
• best.pt: Best model weights (use this for inference)
• last.pt: Final epoch weights
• results.png: Training/validation metrics graphs
• confusion_matrix.png: Shows classification performance
• F1_curve.png, PR_curve.png: Performance curves
• train_batch*.jpg: Training augmentation examples
• val_batch*.jpg: Validation predictions

═══════════════════════════════════════════════════════════════════════════════════

Need more help? Check:
• Ultralytics Documentation: https://docs.ultralytics.com/
• YOLO GitHub: https://github.com/ultralytics/ultralytics
• Community Forum: https://community.ultralytics.com/

"""

        text_widget.insert("1.0", help_content)
        text_widget.config(state="disabled")

        # Add close button
        close_frame = tk.Frame(help_window)
        close_frame.pack(pady=10)
        
        close_button = tk.Button(
            close_frame,
            text="Close",
            command=help_window.destroy,
            bg="#f44336",
            fg="white",
            font=("Arial", 10, "bold")
        )
        close_button.pack()

    def show_anylabeling_instructions(self):
        """Shows instructions for preparing AnyLabeling data for YOLO training."""
        instructions_window = tk.Toplevel(self)
        instructions_window.title("AnyLabeling Data Preparation Guide")
        instructions_window.geometry("700x600")
        instructions_window.attributes("-topmost", True)

        # Create scrollable text
        text_frame = tk.Frame(instructions_window)
        text_frame.pack(fill="both", expand=True, padx=10, pady=10)

        scrollbar = tk.Scrollbar(text_frame)
        scrollbar.pack(side="right", fill="y")

        text_widget = tk.Text(text_frame, wrap="word", yscrollcommand=scrollbar.set)
        text_widget.pack(side="left", fill="both", expand=True)
        scrollbar.config(command=text_widget.yview)

        instructions = """
ANYLABELING TO YOLO DATA PREPARATION GUIDE
==========================================

AnyLabeling is a powerful annotation tool that supports multiple formats. 
To use your AnyLabeling annotations with YOLO, follow these steps:

1. EXPORT FROM ANYLABELING
--------------------------
   In AnyLabeling:
   • Complete your annotations (bounding boxes for object detection)
   • Go to File → Export Labels
   • Choose "YOLO" format
   • Save to your desired location

2. ORGANIZE YOUR DATASET
------------------------
   Create the following folder structure:
   
   your_dataset/
   ├── images/
   │   ├── train/
   │   │   ├── image1.jpg
   │   │   ├── image2.jpg
   │   │   └── ...
   │   └── val/
   │       ├── image1.jpg
   │       ├── image2.jpg
   │       └── ...
   ├── labels/
   │   ├── train/
   │   │   ├── image1.txt
   │   │   ├── image2.txt
   │   │   └── ...
   │   └── val/
   │       ├── image1.txt
   │       ├── image2.txt
   │       └── ...
   └── data.yaml

3. CREATE data.yaml FILE
------------------------
   Create a YAML file with the following content:
   
   train: ../train/images
   val: ../val/images
   
   nc: 2  # number of classes
   names: ['class1', 'class2']  # class names

   Note: Replace the number of classes and class names with your actual data.

4. SPLIT YOUR DATA
------------------
   • Use 80% of images for training (train folder)
   • Use 20% of images for validation (val folder)
   • Each image must have a corresponding .txt label file

5. YOLO LABEL FORMAT
--------------------
   Each label file contains one line per object:
   
   class_index x_center y_center width height
   
   Where:
   • class_index: 0-based class index
   • x_center, y_center: center coordinates (normalized 0-1)
   • width, height: object dimensions (normalized 0-1)

6. TIPS FOR BEST RESULTS
------------------------
   • Ensure balanced class distribution
   • Use high-quality images
   • Annotate objects completely
   • Include diverse backgrounds and lighting
   • Minimum 100 images per class recommended
   • Use data augmentation for small datasets

7. COMMON ISSUES
----------------
   • Missing labels: Each image needs a .txt file
   • Wrong paths in data.yaml: Use relative paths
   • Class index mismatch: Ensure 0-based indexing
   • Empty label files: Remove images without objects

For more information:
• AnyLabeling: https://github.com/vietanhdev/anylabeling
• YOLO format: https://docs.ultralytics.com/datasets/detect/

"""

        text_widget.insert("1.0", instructions)
        text_widget.config(state="disabled")

        # Add close button
        close_button = tk.Button(
            instructions_window,
            text="Close",
            command=instructions_window.destroy,
            bg="#f44336",
            fg="white"
        )
        close_button.pack(pady=10)

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
                # Also check for dataset.yaml
                dataset_yaml = os.path.join(path, "dataset.yaml")
                if os.path.exists(dataset_yaml):
                    self.yaml_path.set(dataset_yaml)
                else:
                    self.yaml_path.set("")  # Clears if not found

    def browse_yaml(self):
        """Allows the user to select the dataset YAML file."""
        self.attributes("-topmost", False)  # Remove topmost before opening dialog
        file = filedialog.askopenfilename(
            title="Select your dataset.yaml file",
            filetypes=[("YAML files", "*.yaml *.yml"), ("All files", "*.*")],
        )
        self.attributes("-topmost", True)  # Restore topmost after dialog is closed
        if file:
            self.yaml_path.set(file)

    def _validate_numeric(self, value):
        """Validates if the input value is numeric."""
        return value.isdigit() or value == ""

    def start_training_thread(self):
        """Starts the training in a separate thread to avoid blocking the GUI."""
        self.start_button.config(state=tk.DISABLED)  # Disables the button while training
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

            # Device validation
            if selected_device == "cuda" and not torch.cuda.is_available():
                messagebox.showwarning(
                    "Warning",
                    "CUDA (GPU) selected, but not available. Training will be executed on CPU.",
                )
                selected_device = "cpu"  # Forces to CPU
            elif selected_device == "mps" and not torch.backends.mps.is_available():
                messagebox.showwarning(
                    "Warning",
                    "MPS (Apple Silicon) selected, but not available. Training will be executed on CPU.",
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
            model = YOLO(self.base_model.get())  # The Ultralytics library downloads the model if it is not found

            output_project_dir = os.path.join(dataset_folder, "runs")
            # The Ultralytics library creates runs/train/run_name automatically

            self.console.insert(tk.END, "\n--- Training Configurations ---\n")
            self.console.insert(tk.END, f"  Dataset YAML: {yaml_file}\n")
            self.console.insert(tk.END, f"  Base Model: {self.base_model.get()}\n")
            self.console.insert(tk.END, f"  Epochs: {epochs}\n")
            self.console.insert(tk.END, f"  Batch Size: {batch}\n")
            self.console.insert(tk.END, f"  Image Size: {imgsz}\n")
            self.console.insert(tk.END, f"  Device: {selected_device}\n")
            self.console.insert(tk.END, f"  Output Folder (Project): {output_project_dir}\n")
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
                patience=50,  # Early stopping patience
                save=True,  # Save train checkpoints and results
                save_period=-1,  # Save checkpoint every epoch (disabled with -1)
                cache=False,  # Cache images for faster training
                workers=8,  # Number of worker threads for data loading
                pretrained=True,  # Use pretrained weights
                optimizer='auto',  # Optimizer (auto, SGD, Adam, AdamW, etc.)
                verbose=True,  # Verbose output during training
                seed=0,  # Random seed for reproducibility
                deterministic=True,  # Deterministic training for reproducibility
                single_cls=False,  # Train as single-class
                rect=False,  # Rectangular training
                cos_lr=False,  # Use cosine learning rate scheduler
                close_mosaic=10,  # Disable mosaic augmentation for final epochs
                amp=True,  # Automatic Mixed Precision training
                fraction=1.0,  # Dataset fraction to train on
                profile=False,  # Profile ONNX and TensorRT speeds
                freeze=None,  # Freeze layers
                # lr0=0.01,  # Initial learning rate
                # lrf=0.01,  # Final learning rate factor
                # momentum=0.937,  # Momentum
                # weight_decay=0.0005,  # Weight decay
                # warmup_epochs=3.0,  # Warmup epochs
                # warmup_momentum=0.8,  # Warmup momentum
                # warmup_bias_lr=0.1,  # Warmup bias learning rate
                # box=7.5,  # Box loss gain
                # cls=0.5,  # Classification loss gain
                # dfl=1.5,  # DFL loss gain
                # label_smoothing=0.0,  # Label smoothing
                # nbs=64,  # Nominal batch size
                # overlap_mask=True,  # Overlap masks (instance segmentation)
                # mask_ratio=4,  # Mask downsample ratio (instance segmentation)
                # dropout=0.0,  # Dropout rate
                # val=True,  # Validate during training
            )

            # 4. Post-training information
            best_model_path = os.path.join(
                output_project_dir, run_name, "weights", "best.pt"
            )
            print(f"\nTraining completed successfully!")
            print(f"Best model saved in: {best_model_path}")
            print(f"Detailed results (graphics, logs) are in: {os.path.join(output_project_dir, run_name)}")

            # Show completion dialog with options
            result = messagebox.askyesno(
                "Training Completed",
                f"The YOLO training completed successfully!\n\n"
                f"Best model saved in:\n'{best_model_path}'\n\n"
                f"Detailed results are in:\n'{os.path.join(output_project_dir, run_name)}'\n\n"
                f"Would you like to open the results folder?",
            )
            
            if result:
                # Open the results folder
                results_path = os.path.join(output_project_dir, run_name)
                if sys.platform == "win32":
                    os.startfile(results_path)
                elif sys.platform == "darwin":
                    os.system(f"open '{results_path}'")
                else:
                    os.system(f"xdg-open '{results_path}'")

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
