# vailá - Versatile Anarcho-Integrated Multimodal Toolbox

## Overview

vailá is a toolbox designed to enhance biomechanics analysis by leveraging multiple motion capture systems. "vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese "vai lá," meaning "go there and do it!" This toolbox empowers you to explore, experiment, and create without the constraints of expensive commercial software.

<center>
  <img src="images/vaila.png" alt="vailá Logo" width="300"/>
</center>

## vailá manifest!

If you have new ideas or suggestions, please send them to us.
Join us in the liberation from paid software with the "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

In front of you stands a versatile and anarcho-integrated tool, designed to challenge the boundaries of commercial systems. This software, not a mere substitute, is a symbol of innovation and freedom, now available and accessible. However, this brave visitation of an old problem is alive and determined to eliminate these venal and virulent barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility. We have left the box open with vailá to insert your ideas and processing in a liberated manner. The only verdict is versatility; a vendetta against exorbitant costs, held as a vow, not in vain, for the value and veracity of which shall one day vindicate the vigilant and the virtuous in the field of motion analysis. Surely, this torrent of technology tends to be very innovative, so let me simply add that it is a great honor to have you with us and you may call this tool vailá.

― The vailá idea!

"vailá" é uma expressão que mistura a sonoridade da palavra francesa "voilà" com o incentivo direto em português "vai lá". É uma chamada à ação, um convite à iniciativa e à liberdade de explorar, experimentar e criar sem as limitações impostas por softwares comerciais caros. "vailá" significa "vai lá e faça!", encorajando todos a aproveitar o poder das ferramentas versáteis e integradas do "vailá: Análise versátil da libertação anarquista integrada na caixa de ferramentas multimodal" para realizar análises com dados de múltiplos sistemas.

## Available Multimodal Analysis

### 1. IMU Analysis

**Description:** Analyze Inertial Measurement Unit (IMU) data stored in CSV or C3D format. It processes and interprets motion data from wearable IMU sensors.

**Instructions:** Use the GUI or CLI to select your IMU CSV or C3D files. The analysis will process these files to extract and visualize the motion data.

### 2. Kinematic Cluster Analysis

**Description:** Analyze cluster marker data stored in CSV format. It helps in interpreting the motion data collected from marker-based motion capture systems.

**Instructions:** Use the GUI or CLI to select your cluster CSV files. You will be prompted to enter the sample rate and the configuration for the trunk and pelvis. Optionally, provide anatomical position data.

### 3. Kinematic Mocap Full Body Analysis

**Description:** Analyze full-body motion capture data in C3D format. It processes the data captured by motion capture systems that track full-body movements.

**Instructions:** Use the GUI or CLI to select your C3D files. The analysis will convert these files and process the motion capture data.

### 4. Markerless 2D with video

**Description:** Analyze 2D video data without using markers. It processes the motion data from 2D video recordings to extract relevant motion parameters.

**Instructions:** Use the GUI or CLI to select your 2D video files. The analysis will process these videos to extract motion data.

### 5. Markerless 3D with multiple videos

**Description:** Process 3D video data without markers. It analyzes 3D video recordings to extract motion data and parameters.

**Instructions:** Use the GUI or CLI to select your 3D video files. The analysis will process these videos to extract 3D motion data.

## Available Tools

### 6. Edit CSV

**Description:** Organize and rearrange data files within a specified directory. This tool allows you to reorder columns, cut and select rows, and modify the global reference system of the data. It also allows for unit conversion and data reshaping, ensuring alignment with the desired coordinate system and consistent data formatting.

**Instructions:** Use the GUI or CLI to select your CSV files. The tool will guide you through the process of editing the files as needed, including reordering columns, cutting and selecting rows, modifying the laboratory coordinate system, and converting units.

### 7. Convert C3D data to CSV

**Description:** Convert motion capture data files from C3D format to CSV format. This makes it easier to analyze and manipulate the data using standard data analysis tools.

**Instructions:** Use the GUI or CLI to select your C3D files. The tool will convert these files to CSV format.

### 8. Metadata info

**Description:** Extract metadata information from video files. This is useful for synchronizing video data with other motion capture data.

**Instructions:** Use the GUI or CLI to select your video files. The tool will extract and display metadata information for each video file.

### 9. Cut videos based on list

**Description:** Cut video files based on a list of specified time intervals. This is useful for segmenting videos into relevant portions for analysis.

**Instructions:** Use the GUI or CLI to select your video files and the list of time intervals. The tool will process these files to cut the videos accordingly.

### 10. Draw a black box around videos

**Description:** Overlay a black box around video frames to highlight specific areas of interest. This can help in focusing on particular regions during analysis.

**Instructions:** Use the GUI or CLI to select your video files. The tool will overlay a black box around the specified area in the videos.

### 11. Compress videos to HEVC (H.265)

**Description:** Compress video files using the HEVC (H.265) codec. This helps in reducing the file size while maintaining video quality.

**Instructions:** Use the GUI or CLI to select your video files. The tool will compress these videos using the HEVC codec.

### 12. Compress videos to H.264

**Description:** Compress video files using the H.264 codec. This helps in reducing the file size while maintaining video quality.

**Instructions:** Use the GUI or CLI to select your video files. The tool will compress these videos using the H.264 codec.

### 13. Plot 2D

**Description:** Generate 2D plots from CSV or C3D files using Matplotlib. This tool helps in visualizing the data in a 2D graph format.

**Instructions:** Use the GUI or CLI to select your data files. The tool will create a 2D plot of the data.

## Additional Commands

**h** - Display this help message.

**exit** - Exit the program.

To use this toolbox, simply select the desired option by typing the corresponding number and pressing Enter. You can also type 'h' to view this help message or 'exit' to quit the program.

## Environment Setup/Install

* Install the complete [Anaconda](https://www.anaconda.com/download/success) virtual environment for your operating system. Remember to add the path and possible dependencies.

1. Clone the repository:

    ```bash
    git clone https://github.com/vaila-multimodaltoolbox/vaila
    cd vaila
    ```

2. Install the required environment and dependencies using the provided installation scripts:

- For **Linux**:

    ```bash
    ./install_vaila_linux.sh
    ```

- For **macOS**:

    ```bash
    ./install_vaila_mac.sh
    ```

- For **Windows**:

    Run the batch script:

    ```bat
    install_vaila_win.bat
    ```

3. Running the Application:

To run the vailá toolbox, activate the environment and start the application using the provided scripts.

- **macOS**: 

    ```bash
    ./mac_launch_vaila.sh
    ```

- **Linux**: 

    ```bash
    ./linux_launch_vaila.sh
    ```

- **Windows**: 

    Run from the Windows Terminal profile created, or activate the environment manually:

    ```bash
    conda activate vaila
    python vaila.py
    ```

4. Follow the multimodal menu instructions in the GUI or CLI.

---

© 2024 vailá - Multimodal Toolbox. All rights reserved.

