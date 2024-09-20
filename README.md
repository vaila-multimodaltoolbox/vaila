# _vailá_ - Multimodal Toolbox

<p align="center">
  <img src="docs/images/vaila.png" alt="vailá Logo" width="300"/>
</p>

<div align="center">
  <table>
    <tr>
      <th>Build Type</th>
      <th>Linux</th>
      <th>macOS</th>
      <th>Windows</th>
    </tr>
    <tr>
      <td><strong>Build Status</strong></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="Linux Build Status"></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="macOS Build Status"></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="Windows Build Status"></td>
    </tr>
  </table>
</div>

## Development of _vailá_: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox

## Introduction

The analysis of human movement is fundamental in both health and sports biomechanics, providing valuable insights into various aspects of physical performance, rehabilitation, and injury prevention. However, existing software often restricts user control and customization, acting as a "black box." With _vailá_, users have the freedom to explore, customize, and create their own tools in a truly open-source and collaborative environment.

_vailá_ (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source multimodal toolbox that leverages data from multiple biomechanical systems to enhance human movement analysis. It integrates data from:

- **Retroreflective Motion Capture Systems** (e.g., Vicon, OptiTrack)
- **Inertial Measurement Unit (IMU) Systems** (e.g., Delsys, Noraxon)
- **Markerless Video Capture Technology** (e.g., OpenPose, MediaPipe)
- **Electromyography (EMG) Systems** (e.g., Delsys, Noraxon)
- **Force Plate Systems** (e.g., AMTI, Bertec)
- **GPS/GNSS Systems** (e.g., Garmin, Trimble)
- **MEG/EEG Systems** (for brain activity monitoring)
- **HR/ECG Systems** (for heart rate and electrical activity)

By integrating these diverse data sources, _vailá_ allows for comprehensive and accurate analysis of movement patterns, which is particularly beneficial for research and clinical applications.

## Key Features

- **Multimodal Data Analysis**: Analyze data from various sources such as IMU sensors, motion capture, markerless tracking, EMG, force plates, and GPS/GNSS systems.
- **File Management**: Tools for file operations, including rename, import, export, copy, move, remove, tree, find, and transfer.
- **Data Conversion**: Convert between C3D and CSV formats, and perform Direct Linear Transformation (DLT) methods for 2D and 3D reconstructions.
- **Video Processing**: Tools for converting videos to images, cutting videos, compressing (H.264 and HEVC H.265), synchronizing videos, and extracting pixel coordinates.
- **Data Visualization**: Display and plot 2D and 3D graphs; visualize CSV and C3D data.

## Description

This multimodal toolbox integrates data from various motion capture systems to facilitate advanced biomechanical analysis by combining multiple data sources. The primary objective is to improve understanding and evaluation of movement patterns across different contexts.

## *vailá* Manifest

### English Version

Join us in the liberation from paid software with the "vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

In front of you stands a versatile tool designed to challenge the boundaries of commercial systems. This software is a symbol of innovation and freedom, determined to eliminate barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.

With _vailá_, you are invited to explore, experiment, and create without constraints. "vailá" means "go there and do it!" — encouraging you to harness its power to perform analysis with data from multiple systems.

### Versão em Português

Junte-se a nós na libertação do software pago com o "vailá: Análise Versátil da Libertação Anarquista Integrada na Caixa de Ferramentas Multimodal".

Diante de você está uma ferramenta versátil, projetada para desafiar as fronteiras dos sistemas comerciais. Este software é um símbolo de inovação e liberdade, determinado a eliminar as barreiras que protegem o monopólio do software caro, garantindo a disseminação do conhecimento e a acessibilidade.

Com _vailá_, você é convidado a explorar, experimentar e criar sem restrições. "vailá" significa "vai lá e faça!" — encorajando você a aproveitar seu poder para realizar análises com dados de múltiplos sistemas.

## Installation and Setup

### Prerequisites

- **Conda**: Ensure that [Anaconda](https://www.anaconda.com/products/distribution) is installed and accessible from the command line.

- **FFmpeg**: Required for video processing functionalities.

### Clone the Repository

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila
cd vaila
```

## Installation Instructions

## For Linux

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

## For macOS

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

## For Windows

1. **Run the installation script as Administrator**:

- PowerShell Script:
  - Right-click `install_vaila_win.ps1` and select "Run with PowerShell" as Administrator.

  - Batch Script:
    - Right-click `install_vaila_win.bat` and select "Run as Administrator".

- Observation: If failing run as Administrator in Windows PowerShell open the PowerShell console as Administrator and go to directory of vaila.

2. **Allow Script Execution**:

- If script execution is blocked, change the execution policy:

```powershell
Set-ExecutionPolicy RemoteSigned
```

- Run this command in PowerShell as Administrator before running the installation script.

3. **Follow the Instructions Displayed by the Script**:

- The script will:
  - Set up the Conda environment using vaila_win.yaml.
  - Install ffmpeg using winget or chocolatey if necessary.
  - Copy program files to your home directory (C:\Users\YourUsername\vaila).
  - Configure the vaila profile in Windows Terminal.
  - Create a desktop shortcut if Windows Terminal is not installed.

4. **Notes**:

- Ensure that Conda is added to your PATH and accessible from the command line.
- The script must be run with administrative privileges.

5. **On Windows, after running install, a button is added to the Windows Terminal. If the automatic insertion fails, manually add the following profile to your Windows Terminal `settings.json` file**:

```json
        {
            "name": "vailá",
            "commandline": "pwsh.exe -ExecutionPolicy Bypass -NoExit -Command '& \"%ProgramData%\\Anaconda3\\shell\\condabin\\conda-hook.ps1\" ; conda activate \"vaila\" ; cd \"C:\\vaila_programs\\vaila\" ; python \"vaila.py\"'",
            "startingDirectory": "C:\\vaila_programs\\vaila",
            "icon": "C:\\vaila_programs\\vaila\\docs\\images\\vaila_ico.png",
            "colorScheme": "Vintage",
            "guid": "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}",
            "hidden": false
        }
```

## Running the Application

### Running the Application After installation, you can launch _vailá_ from your applications menu or directly from the terminal, depending on your operating system.

- Linux and macOS: **From the Terminal bash or zsh**

1. Navigate to the `vaila` directory: ```cd ~/vaila ``` and run command:

```bash
conda activate vaila
python3 vaila.py
```

- Windows: **From the Windows Terminal (Anaconda in path) or use Anaconda PowerShell**

2. Open Anaconda Prompt or Anaconda Powershell Prompt (Anaconda Powershell is recommended) and run command:

```cmd
conda activate vaila
python vaila.py
```

### If preferred, you can also run _vailá_ from the launch scripts.

### For Linux and macOS 

- From the Applications Menu:
  
  - Look for `vaila` in your applications menu and launch it by clicking on the icon. 

#### From the Terminal If you prefer to run _vailá_ from the terminal or if you encounter issues with the applications menu, you can use the provided launch scripts.

##### Linux and macOS

- **Make the script executable** (if you haven't already):

```bash
sudo chmod +x ~/vaila/linux_launch_vaila.sh
```

- **Run the script**:
  
```bash
~/vaila/linux_launch_vaila.sh 
```

#### Notes for Linux and macOS 

- **Ensure Conda is in the Correct Location**: 
  - The launch scripts assume that Conda is installed in `~/anaconda3`. 
  - If Conda is installed elsewhere, update the `source` command in the scripts to point to the correct location.

- **Verify Paths**: 
  - Make sure that the path to `vaila.py` in the launch scripts matches where you have installed the program. 
  - By default, the scripts assume that `vaila.py` is located in `~/vaila`. 

- **Permissions**: 
  - Ensure you have execute permissions for the launch scripts and read permissions for the program files. 

### For Windows 

#### From Windows Terminal 

- **Using the_vailá _Profile**: 
  - Open **Windows Terminal**. - Select the `vailá` profile that was added during installation. 
  - This profile automatically activates the Conda environment and launches *vailá*. 

#### From Desktop Shortcut 

- If a desktop shortcut was created during installation: 
- **Double-click** the `vailá` icon on your desktop to launch the application.

#### Manually Running from Command Prompt or PowerShell If you prefer to run *vailá* manually or if the profile/shortcut is not available: 

- **Activate the Conda Environment**: ```powershell conda activate vaila ```
- **Navigate to the Program Directory**: ```powershell cd C:\vaila_programs\vaila ```
- **Run the Application**: ```powershell python vaila.py ```

#### Notes for Windows 

- **Ensure Conda is in the Correct Location**:
  - The instructions assume that Conda is installed and added to your system's `PATH`.
  - If Conda is installed in a non-standard location, you may need to provide the full path to `conda.exe`.

- **Verify Paths**: 
  - Ensure that `vaila.py` is located in `C:\vaila_programs\vaila`.
  - If installed elsewhere, adjust the commands accordingly.

- **Permissions**: 
  - Make sure you have the necessary permissions to execute scripts and access the program files. 
  - Running the application does not require administrative privileges. ### Additional Notes

#### For All Platforms - **Conda Environment Activation**: 

- It's important to activate the `vaila` Conda environment before running the application to ensure all dependencies are available. 

- **Script Execution Permissions**: 
  - On Linux and macOS, you may need to make the launch scripts executable using the `chmod +x` command. 
  - On Windows, you may need to adjust the execution policy to allow PowerShell scripts to run if you're using a `.ps1` script. 
  - **Updating Scripts for Custom Installations**: 
  - If you have installed Conda or _vailá_ in custom directories, make sure to update the paths in the launch scripts accordingly. 

#### Making the Scripts Executable (Linux/macOS) If you haven't already made the launch scripts executable, you can do so using the following commands:

For Linux

```bash
sudo chmod +x ~/vaila/linux_launch_vaila.sh
```

# For macOS 

```bash
chmod +x ~/vaila/mac_launch_vaila.sh 
```

#### Running the Scripts 

- **Linux**:

```bash
~/vaila/linux_launch_vaila.sh 
``` 

- **macOS**:

```zsh ~/vaila/mac_launch_vaila.sh 
``` 

- **Windows**: - Double-click the `win_launch_vaila.bat` file or run it from the Command Prompt: 

```cmd
C:\vaila_programs\vaila\win_launch_vaila.bat 
``` 

- Alternatively, if you have a PowerShell launch script (`win_launch_vaila.ps1`), you can run it from PowerShell:
  
```powershell
C:\vaila_programs\vaila\win_launch_vaila.ps1 
``` 

- **Note**: You may need to adjust the execution policy to allow the script to run: 
 
```powershell Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser 
``` 

- Be cautious when changing execution policies. 

### Troubleshooting - **Conda Not Found**: 

- If you receive an error indicating that `conda` is not found, ensure that Conda is installed and added to your system's `PATH`. 
- On Linux/macOS, you may need to initialize Conda in your shell. 

- **Permission Denied**: 
  - If you encounter permission errors when running the scripts, check that you have the necessary execute permissions. 
  - Use `chmod +x script_name.sh` on Linux/macOS to make a script executable. 

- **Module Not Found Errors**: 
  - If the application reports missing modules, ensure that the `vaila` Conda environment is activated and that all dependencies are installed. 
  - You can try updating the environment: 

```bash
conda activate vaila conda env update -f path_to_vaila_environment_yaml_file.yaml --prune 
``` 

- **Application Does Not Start**: 
  - Verify that the paths in the launch scripts are correct. 
  - Check for error messages in the terminal that may indicate what the issue is.

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

1. **Run the uninstallation script as Administrator**:

- PowerShell Script:
  - Right-click `uninstall_vaila_win.ps1` and select "Run with PowerShell" as Administrator.

- Batch Script:
  - Right-click `uninstall_vaila_win.bat` and select "Run as Administrator".

2. **Follow the Instructions Displayed by the Script**:

- The script will:
  - Remove the `vaila` Conda environment.
  - Delete the `C:\Users\YourUsername\vaila` directory.
  - Remove the Windows Terminal profile (PowerShell Script).
  - Delete the desktop shortcut if it exists.

3. **Manual Removal of Windows Terminal Profile (if necessary)**:

- If the Windows Terminal profile is not removed automatically (e.g., when using the batch script), you may need to remove it manually:

4. **Notes**:

- The script must be run with administrative privileges.
- Ensure that Conda is added to your PATH and accessible from the command line.

5. **On Windows, if logo `vailá` still in Windows Terminal remove manually the `settings.json` file**:

```json
    {
        "colorScheme": "Vintage",
        "commandline": "pwsh.exe -ExecutionPolicy ByPass -NoExit -Command \"& 'C:\\ProgramData\\anaconda3\\shell\\condabin\\conda-hook.ps1' ; conda activate 'vaila' ; python 'vaila.py' \"",
        "guid": "{17ce5bfe-17ed-5f3a-ab15-5cd5baafed5b}",
        "hidden": false,
        "icon": "C:\\vaila_programs\\vaila\\vaila\\images\\vaila_ico.png",
        "name": "vailá",
        "startingDirectory": "C:\\vaila_programs\\vaila"
    }
```

## Project Structure

An overview of the project structure:

```bash
tree vaila

vaila
├── __init__.py
├── __pycache__
├── batchcut.py
├── cluster_analysis.py
├── cluster_analysis_cli.py
├── common_utils.py
├── compress_videos_h264.py
├── compress_videos_h265.py
├── compressvideo.py
├── cop_analysis.py
├── cop_calculate.py
├── data_processing.py
├── dialogsuser.py
├── dialogsuser_cluster.py
├── dlt2d.py
├── dlt3d.py
├── drawboxe.py
├── ellipse.py
├── emg_labiocom.py
├── extractpng.py
├── filemanager.py
├── filter_utils.py
├── filtering.py
├── fixnoise.py
├── fonts
├── force_cmj.py
├── force_cube_fig.py
├── forceplate_analysis.py
├── getpixelvideo.py
├── gnss_analysis.py
├── images
├── imu_analysis.py
├── listjointsnames.py
├── maintools.py
├── markerless_2D_analysis.py
├── markerless_3D_analysis.py
├── mergestack.py
├── mocap_analysis.py
├── modifylabref.py
├── modifylabref_cli.py
├── numberframes.py
├── plotting.py
├── readc3d_export.py
├── readcsv.py
├── readcsv_export.py
├── rearrange_data.py
├── rearrange_data_dask.py
├── rec2d.py
├── rec2d_one_dlt2d.py
├── rotation.py
├── run_vector_coding.py
├── run_vector_coding_GUI.py
├── showc3d.py
├── showc3d_nodash.py
├── spectral_features.py
├── stabilogram_analysis.py
├── standardize_header.py
├── sync_flash.py
├── syncvid.py
├── utils.py
├── vaila_manifest.py
├── vaila_upscaler.py
├── vailaplot2d.py
├── vailaplot3d.py
├── vector_coding.py
├── videoprocessor.py
└── videoprocessor2.py
```

## Authors

- **Paulo Roberto Pereira Santiago** [ORCID: 0000-0002-9460-8847](https://orcid.org/0000-0002-9460-8847)  
  Biomechanics and Motor Control Lab, School of Physical Education and Sport of Ribeirão Preto, University of São Paulo, Brazil  
  Graduate Program in Rehabilitation and Functional Performance, Ribeirão Preto Medical School, University of São Paulo, Brazil  

- **Abel Gonçalves Chinaglia** [ORCID: 0000-0002-6955-7187](https://orcid.org/0000-0002-6955-7187)  
  Graduate Program in Rehabilitation and Functional Performance, Ribeirão Preto Medical School, University of São Paulo, Brazil  

- **Kira Flanagan** [ORCID: 0000-0003-0317-6346](https://orcid.org/0000-0003-0317-6346)  
  College of Computing, Engineering and Construction, University of North Florida, USA  

- **Bruno Luiz de Souza Bedo** [ORCID: 0000-0003-3821-2327](https://orcid.org/0000-0003-3821-2327)  
  Laboratory of Technology and Sports Performance Analysis, School of Physical Education and Sport, University of São Paulo, Brazil  

- **Ligia Yumi Mochida** [ORCID: 0009-0005-7266-3799](https://orcid.org/0009-0005-7266-3799)  
  Laboratory of Applied Biomechanics and Engineering, Brooks College of Health, University of North Florida, USA  
  Department of Physical Therapy, Brooks College of Health, University of North Florida, USA  

- **Juan Aceros** [ORCID: 0000-0001-6381-7032](https://orcid.org/0000-0001-6381-7032)  
  Laboratory of Applied Biomechanics and Engineering, Brooks College of Health, University of North Florida, USA  
  College of Computing, Engineering and Construction, University of North Florida, USA

- **Aline Bononi** [ORCID: 0000-0001-8169-0864](https://orcid.org/0000-0001-8169-0864)
  Pharmacist at the Municipality of Ribeirão ~/vaila/linux_launch_vaila.sh Preto - Brazil

- **Guilherme Manna Cesar** [ORCID: 0000-0002-5596-9439](https://orcid.org/0000-0002-5596-9439)  
  Laboratory of Applied Biomechanics and Engineering, Brooks College of Health, University of North Florida, USA  
  Department of Physical Therapy, Broo~/vaila/linux_launch_vaila.sh ks College of Health, University of North Florida, USA  

## Contribution

We encourage creativity and innovation to enhance and expand the functionality of this toolbox. You can make a difference by contributing to the project! To get involved, feel free to fork the repository, experiment with new ideas, and create a branch for your changes. When you're ready, submit a pull request so we can review and potentially integrate your contributions.

Don't hesitate to learn, explore, and experiment. Be bold, and don't be afraid to make mistakes—every attempt is a step towards improvement!

## License

This project is primarily licensed under the GNU Lesser General Public License v3.0. Please cite our work if you use the code or data. Let's collaborate and push the boundaries together!
