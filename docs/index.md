# vailá - Versatile Anarcho-Integrated Multimodal Toolbox

## Overview

vailá is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. "vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software.

<center>
  <img src="images/vaila.png" alt="vailá Logo" width="300"/>
</center>

[Next: vailá manifest!](#vailá-manifest)

---

## vailá manifest!

If you have new ideas or suggestions, please send them to us.
Join us in the liberation from paid software with the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

[...]

[Previous: Overview](#overview) | [Next: Available Multimodal Analysis](#available-multimodal-analysis)

---

## Available Multimodal Analysis

### 1. IMU Analysis

**Description:** Analyze Inertial Measurement Unit (IMU) data stored in CSV or C3D format. It processes and interprets motion data from wearable IMU sensors.

**Instructions:** Use the GUI or CLI to select your IMU CSV or C3D files. The analysis will process these files to extract and visualize the motion data.

[Previous: vailá manifest!](#vailá-manifest) | [Next: Kinematic Cluster Analysis](#2-kinematic-cluster-analysis)

---

### 2. Kinematic Cluster Analysis

**Description:** Analyze cluster marker data stored in CSV format. It helps in interpreting the motion data collected from marker-based motion capture systems.

**Instructions:** Use the GUI or CLI to select your cluster CSV files. You will be prompted to enter the sample rate and the configuration for the trunk and pelvis. Optionally, provide anatomical position data.

[Previous: IMU Analysis](#1-imu-analysis) | [Next: Kinematic Mocap Full Body Analysis](#3-kinematic-mocap-full-body-analysis)

---

### 3. Kinematic Mocap Full Body Analysis

**Description:** Analyze full-body motion capture data in C3D format. It processes the data captured by motion capture systems that track full-body movements.

**Instructions:** Use the GUI or CLI to select your C3D files. The analysis will convert these files and process the motion capture data.

[Previous: Kinematic Cluster Analysis](#2-kinematic-cluster-analysis) | [Next: Markerless 2D with video](#4-markerless-2d-with-video)

---

### 4. Markerless 2D with video

**Description:** Analyze 2D video data without using markers. It processes the motion data from 2D video recordings to extract relevant motion parameters.

**Instructions:** Use the GUI or CLI to select your 2D video files. The analysis will process these videos to extract motion data.

[Previous: Kinematic Mocap Full Body Analysis](#3-kinematic-mocap-full-body-analysis) | [Next: Markerless 3D with multiple videos](#5-markerless-3d-with-multiple-videos)

---

### 5. Markerless 3D with multiple videos

**Description:** Process 3D video data without markers. It analyzes 3D video recordings to extract motion data and parameters.

**Instructions:** Use the GUI or CLI to select your 3D video files. The analysis will process these videos to extract 3D motion data.

[Previous: Markerless 2D with video](#4-markerless-2d-with-video) | [Next: Available Tools](#available-tools)

---

## Available Tools

### 6. Edit CSV

**Description:** Organize and rearrange data files within a specified directory. This tool allows you to reorder columns, cut and select rows, and modify the global reference system of the data. It also allows for unit conversion and data reshaping, ensuring alignment with the desired coordinate system and consistent data formatting.

**Instructions:** Use the GUI or CLI to select your CSV files. The tool will guide you through the process of editing the files as needed, including reordering columns, cutting and selecting rows, modifying the laboratory coordinate system, and converting units.

[Previous: Markerless 3D with multiple videos](#5-markerless-3d-with-multiple-videos) | [Next: Convert C3D data to CSV](#7-convert-c3d-data-to-csv)

[...]

---

## Additional Commands

**h** - Display this help message.

**exit** - Exit the program.

To use this toolbox, simply select the desired option by typing the corresponding number and pressing Enter. You can also type 'h' to view this help message or 'exit' to quit the program.

---

[Previous: Environment Setup/Install](#environment-setupinstall)

