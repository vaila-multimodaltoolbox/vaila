# *vailá* - Versatile Anarcho-Integrated Multimodal Toolbox

## 📌 Overview

*vailá* is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. The name "*vailá*" blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software.

![*vailá* Logo](images/vaila.png)

## 🚀 New Feature: *vailá* ML Walkway

### 👥 Created by: Abel G. Chinaglia & Paulo R. P. Santiago

**Lab:** LaBioCoM - Laboratory of Biomechanics and Motor Control
📅 **Date:** 10.Feb.2025
🔄 **Update:** 14.Feb.2025

### 🏃‍♂️ What is *vailá* ML Walkway?

*vailá* ML Walkway is a **graphical user interface (GUI)** designed to facilitate various **machine learning (ML) tasks** related to gait analysis. With this module, users can:

✅ **Process gait features** using pixel data from MediaPipe (**via Makerless 2D in *vailá***).
✅ **Train ML models** using extracting features e targets.
✅ **Validate ML models** after training.
✅ **Run ML predictions** using pre-trained models or new models trained by you.

Each function is accessible through buttons in the user-friendly GUI, automating complex ML workflows with ease!

---

## 📊 Available Multimodal Analysis

### 1️⃣ IMU Analysis

🔎 **Analyze Inertial Measurement Unit (IMU) data** stored in CSV or C3D format. This helps process and interpret motion data from wearable IMU sensors.

🔧 **How to use:** Select your IMU CSV or C3D files in the GUI/CLI. The system will extract and visualize motion data.

### 2️⃣ Kinematic Cluster Analysis

🔎 **Analyze cluster marker data** stored in CSV format, useful for interpreting motion data from marker-based motion capture systems.

🔧 **How to use:** Upload your cluster CSV files, set the sample rate, and define configurations for trunk/pelvis. Optionally, provide anatomical position data.

### 3️⃣ Kinematic Mocap Full Body Analysis

🔎 **Analyze full-body motion capture data** from C3D format.

🔧 **How to use:** Select your C3D files in the GUI/CLI, and the system will process the motion capture data.

## 🔧 Complete Documentation

The vailá Multimodal Toolbox now includes comprehensive documentation for all modules and GUI buttons:

### 🖱️ GUI Button Documentation

All buttons in the vailá GUI (`vaila.py`) are documented in **[docs/vaila_buttons/](vaila_buttons/README.md)**:

- **[File Manager Buttons](vaila_buttons/README.md#file-manager-frame-a)**: Rename, Import, Export, Copy, Move, Remove, Tree, Find, Transfer
- **[Multimodal Analysis Buttons](vaila_buttons/README.md#multimodal-analysis-frame-b)**: IMU, MoCap, Markerless 2D/3D, Vector Coding, EMG, Force Plate, GNSS/GPS, and more
- **[Tools Buttons](vaila_buttons/README.md#tools-frame-c)**: Data Files, Video Processing, Visualization tools

**Quick Links:**
- **[Markerless 2D Analysis](vaila_buttons/markerless-2d-button.md)** (B1_r1_c4) - Advanced pose estimation with MediaPipe and YOLOv11
- **[All Button Documentation](vaila_buttons/README.md)** - Complete list of all GUI buttons

### 📖 Module Documentation

#### **Markerless Analysis**
- **[2D Analysis](modules/markerless-analysis/markerless-2d-analysis.md)**: Advanced 2D pose estimation with MediaPipe
- **[3D Analysis](modules/markerless-analysis/markerless-3d-analysis.md)**: 3D pose reconstruction from monocular video
- **[Live Analysis](modules/markerless-analysis/markerless-live.md)**: Real-time markerless tracking

#### **Motion Capture**
- **[Full Body Analysis](modules/motion-capture/mocap-full-body.md)**: Complete motion capture data processing
- **[Cluster Analysis](modules/motion-capture/cluster-analysis.md)**: Anatomical marker cluster analysis

#### **Sensor Analysis**
- **[IMU Analysis](modules/sensors/imu-analysis.md)**: Inertial measurement unit data processing
- **[Force Plate Analysis](modules/sensors/force-plate.md)**: Ground reaction force analysis
- **[EMG Analysis](modules/sensors/emg-analysis.md)**: Electromyography signal processing
- **[GNSS/GPS Analysis](modules/sensors/gnss-gps.md)**: GPS trajectory analysis

#### **Data Processing Tools**
- **[C3D ↔ CSV Conversion](modules/tools/c3d-csv-conversion.md)**: Motion capture file format conversion
- **[Data Filtering](modules/tools/data-filtering.md)**: Advanced signal processing algorithms

#### **Video Processing Tools**
- **[Video Tools](modules/tools/video-tools.md)**: Video manipulation and processing
- **[Video Compression](modules/tools/video-compression.md)**: Multi-format video compression

#### **Visualization Tools**
- **[2D Plotting](modules/visualization/plot-2d.md)**: Comprehensive 2D data visualization
- **[View 3D](modules/visualization/view3d.md)**: 3D data visualization

#### **Machine Learning**
- **[ML Walkway](modules/ml-walkway/ml-walkway.md)**: Complete machine learning walkway analysis system
- **[Model Training](modules/ml-walkway/model-training.md)**: Train custom ML models for gait analysis
- **[Model Validation](modules/ml-walkway/model-validation.md)**: Validate trained ML models
- **[Walkway Prediction](modules/ml-walkway/walkway-prediction.md)**: Use trained models for walkway predictions
- **[YOLO Training](modules/ml-walkway/yolo-training.md)**: Train YOLO models for object detection
- **[YOLOv11 Tracking](modules/ml-walkway/yolov11track.md)**: YOLOv11 based tracking system
- **[YOLOv12 Tracking](modules/ml-walkway/yolov12track.md)**: YOLOv12 based tracking system
- **[YOLO Tracking](modules/ml-walkway/yolo-tracking.md)**: YOLO object detection and tracking

#### **Specialized Tools**
- **[Vector Coding](modules/tools/vector-coding.md)**: Joint coupling analysis
- **[Open Field Analysis](modules/tools/open-field.md)**: Animal behavior analysis
- **[Vertical Jump Analysis](modules/tools/vertical-jump.md)**: Jump performance metrics

### 📚 API Reference

- **[Complete Module List](api/modules.md)**: Comprehensive reference of all vailá modules
- **[Function Reference](api/functions.md)**: Detailed function documentation

---

## 🛠 Installation & Setup

### ⚡ Powered by *uv*

*vailá* now uses **[uv](https://github.com/astral-sh/uv)**, an extremely fast Python package installer and resolver, written in Rust. **uv is the recommended installation method for all platforms** (Windows, Linux, macOS) due to its **10-100x faster installation** and **faster execution times** compared to Conda.

**Note:** Conda installation methods are still available but are now considered legacy due to slower installation and execution times.

### 🔹 Prerequisites

✔ **uv**: Will be automatically installed by the installation scripts, or install manually from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)  
✔ **FFmpeg**: Required for video processing functionalities (installed automatically on Windows)

### 🔹 Installation Methods

**Windows (Recommended):**
```powershell
git clone https://github.com/vaila-multimodaltoolbox/vaila
cd vaila
.\install_vaila_win_uv.ps1
```

**Linux and macOS (Using uv):**
```bash
# Install uv
curl -LsSf https://astral.sh/uv/install.sh | sh

# Clone and install
git clone https://github.com/vaila-multimodaltoolbox/vaila
cd vaila
uv sync

# Run vailá
uv run vaila.py
```

**Legacy Conda Method:**
See the [main README.md](../README.md) for Conda installation instructions (slower, not recommended).

---

## 🎯 Running the *vailá* ML Walkway

🔹 **From the terminal:**

```bash
python vaila_mlwalkway.py
```

🔹 **From the applications menu:** Select *vailá* ML Walkway from the installed applications.

This will open a **graphical interface** where you can select different **machine learning tasks** for gait analysis.

### 🎯 What Does Each Step of *vailá* ML Walkway Do?

#### **Process Gait Features**

The **Process Gait Features** functionality uses a `.csv` file containing pixel data from MediaPipe (**via Markerless 2D in *vailá***) to calculate the necessary features for training new models using **Train ML Models** or for applying predictions using **Run ML Predictions**. It calculates features based on the number of steps that occurred.

**Important:**
If pixel data from MediaPipe is not used, it will not be possible to train, validate, or run predictions.

---

#### **Train ML Models**

The **Train ML Models** functionality allows training models using processed features from a `.csv` file and target variables. The selected gait variables include:

- **Step Length**
- **Stride Length**
- **Support Base**
- **Support Time Single Stand**
- **Support Time Double Stand**
- **Step Width**
- **Stride Width**
- **Stride Velocity**
- **Step Time**
- **Stride Time**

For each gait variable, algorithms like **XGBoost**, **KNN**, **MLP**, **SVR**, **Random Forest**, **Gradient Boosting** e **Linear Regression** are applied. Performance metrics (MSE, RMSE, MAE, MedAE, Max Error, RAE, Accuracy, R² e Explained Variance) are subsequently saved.

---

#### **Validate ML Models**

Select the directory with trained models and validate them using test data (processed via **Process Gait Features**) and target variables.

This step produces the same performance metrics as training, allowing you to identify the best model.

---

#### **Run ML Predictions**

This functionality applies gait variable predictions using either pre-trained models or those you have trained. You can select which metrics to predict, and the output is provided in a `.csv` file.

For additional details, refer to the main documentation.

---

© 2024 *vailá* - Multimodal Toolbox. All rights reserved.
