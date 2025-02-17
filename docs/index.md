# *vailá* - Versatile Anarcho-Integrated Multimodal Toolbox

## 📌 Overview

*vailá* is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. The name "*vailá*" blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software.

![*vailá* Logo](images/vaila.png){width=300}

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
✅ **Run ML predictions** using pre-trained models ou modelos novos treinados por você.  

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

---

## 🛠 Installation & Setup

### 🔹 Prerequisites

✔ **Conda**: Install [Anaconda](https://www.anaconda.com/download/success) and ensure it is accessible from the command line.  
✔ **FFmpeg**: Required for video processing functionalities.

### 🔹 Clone the Repository

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila
cd vaila
```

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
