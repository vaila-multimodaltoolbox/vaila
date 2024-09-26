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

---

## Installation and Setup

### Prerequisites

- **Conda**: Ensure that [Anaconda](https://www.anaconda.com/download/success) is installed and accessible from the command line.

- **FFmpeg**: Required for video processing functionalities.

### Clone the Repository

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila
cd vaila
```

## Installation Instructions

---

## 🟠 For Linux:

1. **Make the installation script executable**:

```bash
sudo chmod +x install_vaila_linux.sh
```

2. **Run instalation script**:

```bash
./install_vaila_linux.sh
```

- The script will:

  - Set up the Conda environment using `./yaml_for_conda_env/vaila_linux.yaml`.
  - Copy program files to your home directory (~/vaila).
  - Install ffmpeg from system repositories.
  - Create a desktop entry for easy access.

3. **Notes**:

- Run the script as your regular user, not with sudo.
- Ensure that Conda is added to your PATH and accessible from the command line.

---

## ⚪ For macOS:

1. **Make the installation script executable**:

```bash
sudo chmod +x install_vaila_mac.sh
```

2. **Run the installation script**:

```bash
./install_vaila_mac.sh
```

- The script will:
  - Set up the Conda environment using `./yaml_for_conda_env/vaila_mac.yaml`.
  - Copy program files to your home directory (`~/vaila`).
  - Install ffmpeg using Homebrew.
  - Convert the .iconset folder to an .icns file for the app icon.
  - Create an application bundle (`vaila.app`) in your Applications folder.
  - Create a symbolic link in `/Applications` to the app in your home directory.

3. **Notes**:
  
- You may be prompted for your password when the script uses sudo to create the symbolic link.
- Ensure that Conda is added to your PATH and accessible from the command line.

---

## 🔵 For Windows:

1. **Download and Install Anaconda**:

- Make sure to download and install [Anaconda](https://www.anaconda.com/download/success).
- Ensure that **Conda** is installed and accessible from the command line.

2. **Download vailá**:
  You can either:

- **Clone the repository** using git:

  ```bash
  git clone https://github.com/vaila-multimodaltoolbox/vaila
  cd vaila
  ```
  
- **Or download the vailá zip file directly from your browser:**
  - **Go to** `https://github.com/vaila-multimodaltoolbox/vaila/archive/refs/heads/main.zip`
  - **Unzip the downloaded file (for example, into your Downloads folder).**

3. **Run the Installation Script as Administrator:**
  
  Now you need to install vailá `Anaconda Powershell Prompt` as an administrator:

- Press the Windows key and search for "Anaconda PowerShell Prompt".
- Right-click on Anaconda PowerShell Prompt and select "Run as administrator".
- Navigate to the directory where you git clone or downloaded unzipped the vailá files (e.g., C:\Users\YourUserName\Downloads\vaila).
- Run the installation script:

  ```Anaconda Powershell Prompt
  ./install_vaila_win.ps1
  ```

4. **Allow Script Execution (if necessary):**

If the script execution is blocked, you may need to change the execution policy. Run this command in PowerShell as Administrator:

  ```Anaconda Powershell Prompt
  Set-ExecutionPolicy RemoteSigned
  ```

5. **The Instructions Displayed by the Script:**

The script will:

- Set up the Conda environment using `vaila_win.yaml` inside directory/folder `yaml_for_conda_env`.  
- Install FFmpeg using winget or Chocolatey if necessary.
- Copy program files to C:\ProgramData\vaila.
- Configure the vaila profile in Windows Terminal (recommend you have Windows Terminal installed on your computer).
- Create a Desktop a Start Menu shortcut.

6. **If the Windows Terminal profile was not automatically added, follow the instructions below to manually add it. (if necessary)**

### Manual Addition to Windows Terminal (if necessary)

 ```json
{
    "name": "vaila",
    "commandline": "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command '& \"%ProgramData%\\Anaconda3\\shell\\condabin\\conda-hook.ps1\" ; conda activate \"vaila\" ; cd \"C:\\ProgramData\\vaila\" ; python \"vaila.py\"'",
    "startingDirectory": "C:\\ProgramData\\vaila",
    "icon": "C:\\ProgramData\\vaila\\docs\\images\\vaila_ico.png",
    "colorScheme": "Vintage",
    "guid": "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}",
    "hidden": false
}
```

Now the installation instructions are simplified, ensuring users download vailá, run the script with administrator privileges, and see the vailá icon in all the intended places (Desktop, Start Menu, and Windows Terminal).

When vailá is installed on Windows, the files are copied to the following directory:

  ```PowerShell
  C:\ProgramData\vaila
  ```

This location is used to store data and configuration files that are accessible to all users on the system. However, the `ProgramData` directory is usually hidden in Windows. To access it, follow these steps:

- Open File Explorer.
- Go to the View tab in the top menu.
- Check the Hidden items box to display the `C:\ProgramData` directory.
- Navigate to `C:\ProgramData\vaila`.

For more information about the ProgramData folder on Windows, you can refer to the official [Microsoft documentation](https://learn.microsoft.com/en-us/windows-hardware/customize/desktop/unattend/microsoft-windows-shell-setup-folderlocations-programdata)

---

## Running the Application

### Running the Application After installation, you can launch *vailá* from your applications menu or directly from the terminal, depending on your operating system.

- 🟠 Linux and ⚪ macOS: **From the Terminal bash or zsh**

1. Navigate to the `vaila` directory:
   
  ```bash
  cd ~/vaila
  ``` 

and run command:

  ```bash
  conda activate vaila
  python3 vaila.py
  ```

- 🔵 Windows

- Click on the `vaila` icon in the Applications menu or use the shortcut in desktop or Windows Terminal.

- Windows: **From the Windows Terminal (Anaconda in path) or use Anaconda PowerShell**

1. Open Anaconda Prompt or Anaconda Powershell Prompt (Anaconda Powershell is recommended) and run command:

```Anaconda Powershell
conda activate vaila
python vaila.py
```

---

## If preferred, you can also run *vailá* from the launch scripts.

### For 🟠 Linux and ⚪ macOS 

- From the Applications Menu:
  
  - Look for `vaila` in your applications menu and launch it by clicking on the icon. 

--- 

#### From the Terminal If you prefer to run *vailá* from the terminal or if you encounter issues with the applications menu, you can use the provided launch scripts.

##### 🟠Linux and ⚪ macOS

- **Make the script executable** (if you haven't already):

- 🟠 **Linux**
  
```bash
sudo chmod +x ~/vaila/linux_launch_vaila.sh
```

- **Run the script**:
  
```bash
~/vaila/linux_launch_vaila.sh 
```

- ⚪ **macOS**
  
```bash
sudo chmod +x ~/vaila/mac_launch_vaila.sh
```

- **Run the script**:
  
```bash
~/vaila/mac_launch_vaila.sh 
```

#### Notes for 🟠 Linux and ⚪ macOS 

- **Ensure Conda is in the Correct Location**:
  - The launch scripts assume that Conda is installed in `~/anaconda3`. 
  - If Conda is installed elsewhere, update the `source` command in the scripts to point to the correct location.

- **Verify Paths**:
  - Make sure that the path to `vaila.py` in the launch scripts matches where you have installed the program.
  - By default, the scripts assume that `vaila.py` is located in `~/vaila`.

- **Permissions**:
  - Ensure you have execute permissions for the launch scripts and read permissions for the program files. 

--- 

## Unistallation Instructions

## For Uninstallation on Linux

1. **Run the uninstall script**:

```bash
sudo chmod +x uninstall_vaila_linux.sh
./uninstall_vaila_linux.sh
```

- The script will:
  - Remove the `vaila` Conda environment.
  - Delete the `~/vaila` directory.
  - Remove the desktop entry.

2. **Notes**:

- Run the script `./uninstall_vaila_linux.sh` as your regular user, not with sudo.
- Ensure that Conda is added to your PATH and accessible from the command line.

## For Uninstallation on macOs

1. **Run the uninstall script**:

```bash
sudo chmod +x uninstall_vaila_mac.sh
./uninstall_vaila_mac.sh
```

- The script will:
  - Remove the `vaila` Conda environment.
  - Delete the `~/vaila` directory.
  - Remove `vaila.app` from /Applications.
  - Refresh the Launchpad to remove cached icons.

2. **Notes**:

- Run the script as your regular user, not with sudo.
- You will prompted for your password when the script uses `sudo` to remove the app from `/Applications`.

## For Uninstallation on Windows

1. **Run the uninstallation script as Administrator in Anaconda PowerShell Prompt**:

- PowerShell Script:
  - Execute `.\uninstall_vaila_win.ps1`.

1. **Follow the Instructions Displayed by the Script**:

- The script will:
  - Remove the `vaila` Conda environment.
  - Delete the `C:\ProgramData\vaila` directory.
  - Remove the Windows Terminal profile (settings.json file).
  - Delete the desktop shortcut if it exists.

1. **Manual Removal of Windows Terminal Profile (if necessary)**:

- If the Windows Terminal profile is not removed automatically (e.g., when using the `uninstall_vaila_win.ps1` script), you may need to remove it manually:

```Anaconda PowerShell Prompt
conda remove -n vaila --all
```

Remove directory `vaila` inside `C:ProgramData\`.

---

© 2024 vailá - Multimodal Toolbox. All rights reserved.
