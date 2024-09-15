# vailá - multimodaltoolbox

<p align="center">
  <img src="docs/images/vaila.png" alt="vailá Logo" width="300"/>
</p>

<div align="center">
  <table>
    <tr>
      <th>Build Type</th>
      <th>Linux</th>
      <th>MacOS</th>
      <th>Windows</th>
    </tr>
    <tr>
      <td><strong>Build Status</strong></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="Linux Build Status"></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="MacOS Build Status"></td>
      <td><img src="https://img.shields.io/badge/Build-OK-brightgreen.svg" alt="Windows Build Status"></td>
    </tr>
  </table>
</div>

## Development of vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox

## Introduction

The analysis of human movement is fundamental in both health and sports biomechanics, providing valuable insights into various aspects of physical performance, rehabilitation, and injury prevention. However, existing software often restricts user control and customization, acting as a "black box." With __vailá__, users have the freedom to explore, customize, and create their own tools in a truly open-source and collaborative environment.

__vailá__ (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source multimodal toolbox that leverages data from multiple biomechanical systems to enhance human movement analysis. It integrates data from:

- **Retroreflective Motion Capture Systems** (e.g., Vicon, OptiTrack)
- **Inertial Measurement Unit (IMU) Systems** (e.g., Delsys, Noraxon)
- **Markerless Video Capture Technology** (e.g., OpenPose, MediaPipe)
- **Electromyography (EMG) Systems** (e.g., Delsys, Noraxon)
- **Force Plate Systems** (e.g., AMTI, Bertec)
- **GPS/GNSS Systems** (e.g., Garmin, Trimble)
- **MEG/EEG Systems** (for brain activity monitoring)
- **HR/ECG Systems** (for heart rate and electrical activity)

By integrating these diverse data sources, __vailá__ allows for comprehensive and accurate analysis of movement patterns, which is particularly beneficial for research and clinical applications.

## Key Features

- **Multimodal Data Analysis**: Analyze data from various sources such as IMU sensors, motion capture, markerless tracking, EMG, force plates, and GPS/GNSS systems.
- **File Management**: Tools for file operations, including rename, import, export, copy, move, remove, tree, find, and transfer.
- **Data Conversion**: Convert between C3D and CSV formats, and perform Direct Linear Transformation (DLT) methods for 2D and 3D reconstructions.
- **Video Processing**: Tools for converting videos to images, cutting videos, compressing (H.264 and HEVC H.265), synchronizing videos, and extracting pixel coordinates.
- **Data Visualization**: Display and plot 2D and 3D graphs; visualize CSV and C3D data.

## Description

This multimodal toolbox integrates data from various motion capture systems to facilitate advanced biomechanical analysis by combining multiple data sources. The primary objective is to improve understanding and evaluation of movement patterns across different contexts.

## vailá Manifest

### English Version

Join us in the liberation from paid software with the "vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

In front of you stands a versatile tool designed to challenge the boundaries of commercial systems. This software is a symbol of innovation and freedom, determined to eliminate barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.

With __vailá__, you are invited to explore, experiment, and create without constraints. "vailá" means "go there and do it!" — encouraging you to harness its power to perform analysis with data from multiple systems.

### Versão em Português

Junte-se a nós na libertação do software pago com o "vailá: Análise versátil da libertação anarquista integrada na caixa de ferramentas multimodal".

Diante de você está uma ferramenta versátil, projetada para desafiar as fronteiras dos sistemas comerciais. Este software é um símbolo de inovação e liberdade, determinado a eliminar as barreiras que protegem o monopólio do software caro, garantindo a disseminação do conhecimento e a acessibilidade.

Com __vailá__, você é convidado a explorar, experimentar e criar sem restrições. "vailá" significa "vai lá e faça!" — encorajando você a aproveitar seu poder para realizar análises com dados de múltiplos sistemas.

## Environment Setup/Install

To set up the development environment, follow these steps:

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

### Running the Application

To run the vailá toolbox, activate the environment and start the application using the provided scripts.

#### For macOS

Use the script `mac_launch_vaila.sh`:

  ```bash
  ./mac_launch_vaila.sh
  ```

#### For Linux

Use the script `linux_launch_vaila.sh`:

  ```bash
  ./linux_launch_vaila.sh
  ```

#### For Windows

On Windows, after running `install_vaila_win.bat`, a button is added to the Windows Terminal. If the automatic insertion fails, manually add the following profile to your Windows Terminal `settings.json` file:

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

### Analyzing Data

To run the toolbox and analyze data:

- **Windows 11**:

    ```bash
    python vaila.py
    ```

- **Linux and macOS**:

    ```bash
    python3 vaila.py
    ```

Follow the multimodal menu instructions in GUI or click on the `imagination!` button to access CLI commands.

## Project Structure

An overview of the project structure:

```bash
tree vaila

vaila
├── __init__.py
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
│   └── mrrobot.ttf
├── force_cmj.py
├── force_cube_fig.py
├── forceplate_analysis.py
├── getpixelvideo.py
├── gnss_analysis.py
├── images
│   ├── cluster_config.png
│   ├── eeferp.png
│   ├── gui.png
│   ├── preto.png
│   ├── unf.png
│   ├── usp.png
│   ├── vaila.ico
│   ├── vaila_edge_w.png
│   ├── vaila_ico.png
│   ├── vaila_ico_mac.png
│   ├── vaila_ico_mac_original.png
│   ├── vaila_ico_trans.ico
│   ├── vaila_icon_win_original.ico
│   ├── vaila_logo.png
│   ├── vaila_trans_square.png
│   ├── vaila_transp.ico
│   └── vaila_white_square.png
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

- **Guilherme Manna Cesar** [ORCID: 0000-0002-5596-9439](https://orcid.org/0000-0002-5596-9439)  
  Laboratory of Applied Biomechanics and Engineering, Brooks College of Health, University of North Florida, USA  
  Department of Physical Therapy, Brooks College of Health, University of North Florida, USA  

## Contribution

We encourage creativity and innovation to enhance and expand the functionality of this toolbox. You can make a difference by contributing to the project! To get involved, feel free to fork the repository, experiment with new ideas, and create a branch for your changes. When you're ready, submit a pull request so we can review and potentially integrate your contributions.

Don't hesitate to learn, explore, and experiment. Be bold, and don't be afraid to make mistakes—every attempt is a step towards improvement!

## License

This project is primarily licensed under the GNU Lesser General Public License v3.0. Please cite our work if you use the code or data. Let's collaborate and push the boundaries together!