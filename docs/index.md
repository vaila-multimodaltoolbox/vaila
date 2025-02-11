# *vailá* - Versatile Anarcho-Integrated Multimodal Toolbox

## 📌 Overview

*vailá* is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. The name "*vailá*" blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software.

<center>
  <img src="images/vaila.png" alt="*vailá* Logo" width="300"/>
</center>

## 🚀 New Feature: *vailá* ML Walkway

### 👥 Created by: Abel G. Chinaglia & Paulo R. P. Santiago  
**Lab:** LaBioCoM - Laboratory of Biomechanics and Motor Control  
📅 **Date:** 10.Feb.2025  
🔄 **Update:** 11.Feb.2025  

### 🏃‍♂️ What is *vailá* ML Walkway?

*vailá* ML Walkway is a **graphical user interface (GUI)** designed to facilitate various **machine learning (ML) tasks** related to gait analysis. With this module, users can:

✅ **Process gait features** using pixel data from MediaPipe (**via Makerless 2D in *vailá***).  
✅ **Train ML models** by extracting features.  
✅ **Validate ML models** after training.  
✅ **Run ML predictions** using pre-trained models.  

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

For additional details, refer to the main documentation.

---

© 2024 *vailá* - Multimodal Toolbox. All rights reserved.
