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

The analysis of human movement, such as trunk and pelvis coordination during sit-to-stand movements, is crucial for understanding the biomechanics of individuals with disabilities, particularly for those with conditions such as cerebral palsy. Traditional motion capture systems, while highly accurate, often require extensive setup and are limited in their ability to capture natural movements in various environments. Recent advancements in motion capture technologies, including inertial measurement units (IMUs) and markerless video-based systems, provide new opportunities for comprehensive movement analysis.

This study presents the development of the vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox. This toolbox/toolkit designed to enhance the analysis of human movement by leveraging data from multiple biomechanical systems. The toolbox incorporates data from:

- **Retroreflective Motion Capture Systems**, such as Vicon, which provide high-precision tracking of reflective markers placed on the body.
- **Inertial Measurement Unit (IMU) Systems**, such as those from Delsys, which offer portable and unobtrusive tracking of body segment accelerations and orientations.
- **Markerless Video Capture Technology**, such as OpenPose, which enables the estimation of human poses from video footage without the need for markers, facilitating more natural movement analysis.
- **Electromyography (EMG) Systems**, which measure muscle activation signals, providing insights into the neuromuscular aspects of movement.
- **Force Plate Systems**, which measure ground reaction forces, offering data on balance, gait, and other aspects of biomechanics.
- **GPS/GNSS Systems**, which provide precise location data, useful for outdoor movement analysis.

By integrating these diverse data sources, the vailá allows for a more comprehensive and accurate analysis of movement patterns. This toolbox is particularly beneficial for research and clinical applications where detailed biomechanical assessments are required to develop and evaluate interventions aimed at improving motor function in individuals with disabilities.

The development and validation of this toolbox involved collaborative efforts across multiple institutions, ensuring robust and reliable performance. The primary objective of this paper is to describe the technical implementation of the vailá in Multimodal Toolbox and demonstrate its application in the analysis of sit-to-stand movements in children with cerebral palsy. The results highlight the potential of this integrated approach to provide deeper insights into the biomechanics of movement, ultimately contributing to the advancement of rehabilitation strategies and the improvement of quality of life for individuals with movement disorders.

## Description

This multimodal toolbox integrates data from various motion capture systems, including:

- **Retroreflective Motion Capture Systems** (e.g., OptiTrack, Vicon, Qualisys)
- **Inertial Measurement Unit (IMU) Systems** (e.g., Delsys, Noraxon)
- **Markerless Video Capture Technology** (e.g., MediaPipe or OpenPose for human pose estimation)
- **Electromyography (EMG) Systems** (e.g., Delsys, Noraxon)
- **Force Plate Systems** (e.g., AMTI, Bertec, Kistler)
- **GPS/GNSS Systems** (e.g., Garmin, Trimble)

This project aims to facilitate advanced biomechanical analysis by combining multiple data sources to improve the understanding and evaluation of movement patterns.

## vailá manifest

### English Version

If you have new ideas or suggestions, please send them to us.
Join us in the liberation from paid software with the "vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox".

In front of you stands a versatile and anarcho-integrated tool, designed to challenge the boundaries of commercial systems.
This software, not a mere substitute, is a symbol of innovation and freedom, now available and accessible.
However, this brave visitation of an old problem is alive and determined to eliminate these venal and
virulent barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.
We have left the box open with vailá to insert your ideas and processing in a liberated manner.
The only verdict is versatility; a vendetta against exorbitant costs, held as a vow, not in vain, for the value and veracity of which shall one day
vindicate the vigilant and the virtuous in the field of motion analysis.
Surely, this torrent of technology tends to be very innovative, so let me simply add that it is a great honor to have you with us
and you may call this tool vailá.

― The vailá idea!

"vailá" is an expression that blends the sound of the French word "voilà" with the direct encouragement in Portuguese BR "vai lá."
It is a call to action, an invitation to initiative and freedom to explore, experiment, and create without the constraints imposed by expensive commercial software.
"vailá" means "go there and do it!", encouraging everyone to harness the power of the "vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox" to perform analysis with data from multiple systems.

### Versão em Português

Se você tiver novas ideias ou sugestões, envie-as para nós.
Junte-se a nós na libertação do software pago com o "vailá: Análise versátil da libertação anarquista integrada na caixa de ferramentas multimodal".

Diante de você está uma ferramenta versátil e integrada anarquicamente, projetada para desafiar as fronteiras dos sistemas comerciais.
Este software, não é um mero substituto, é um símbolo de inovação e liberdade, agora disponível e acessível.
No entanto, esta ousada visitação de um velho problema está viva e determinada a eliminar essas barreiras venais e
virulentas que protegem o monopólio do software caro, garantindo a disseminação do conhecimento e a acessibilidade.
Deixamos a caixa aberta com vailá para inserir suas ideias e processamento de maneira liberada.
O único veredito é versatilidade; uma vingança contra custos exorbitantes, mantida como um voto, não em vão, pelo valor e veracidade que um dia
vindicarão os vigilantes e virtuosos no campo da análise de movimento.
Certamente, este torrente de tecnologia tende a ser muito inovador, então me permita simplesmente acrescentar que é uma grande honra tê-lo conosco
e você pode chamar esta ferramenta de vailá.

― A ideia vailá!

"vailá" é uma expressão que mistura a sonoridade da palavra francesa "voilà" com o incentivo direto em português "vai lá".
É uma chamada à ação, um convite à iniciativa e à liberdade de explorar, experimentar e criar sem as limitações impostas por softwares comerciais caros.
"vailá" significa "vai lá e faça!", encorajando todos a aproveitar o poder das ferramentas versáteis e integradas do "vailá: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox" para realizar análises com dados de múltiplos sistemas.

## Authors
Guilherme Manna Cesar<sup>1</sup>, Ligia Yumi Mochida<sup>1</sup>, Bruno Luiz de Souza Bedo<sup>2</sup>, Paulo Roberto Pereira Santiago<sup>3</sup>

1 - University of North Florida, Laboratory of Applied Biomechanics and Engineering  
2 - School of Physical Education and Sport, University of São Paulo, Laboratory of Technology and Sports Performance Analysis  
3 - School of Physical Education and Sport of Ribeirão Preto, University of São Paulo, Biomechanics and Motor Control Laboratory

## Environment Setup

To set up the development environment, follow these steps:

1. Clone the repository:

```bash
  git clone https://github.com/vaila-multimodaltoolbox/vaila
  cd vaila-multimodaltoolbox/yaml_for_conda_env
```

2. Create [Anaconda](https://www.anaconda.com/download/success) virtual environment and install the dependencies:

- For Linux:

```bash
  conda env create -f vaila_linux.yaml
```

- For macOS:

```bash
  conda env create -f vaila_macos.yaml
```

- For Windows:

```bash
  conda env create -f vaila_win.yaml
```

If you need to update the environment (replace `vaila_linux.yaml` or `vaila_macos.yaml`):

```bash
  conda env update -f vaila_win.yaml
```

3. Activate the conda environment:

```bash
  conda activate vaila
```

4. Run the toolbox to analyze data:

- Windows 11
  
  ```bash
   python vaila.py
   ```

- Linux and MacOS

```bash
  python3 vaila.py
```

### Running the Application

To make it easier to launch the application, you can use the provided scripts for different operating systems.

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

Use the batch script `win_launch_vaila.bat`:

```bat
  win_launch_vaila.bat
```

5. Follow the multimodal menu instructions in GUI or CLI:

<p align="center">
  <img src="docs/images/vaila_start_gui.png" alt="vailá GUI" width="800"/>
</p>

```bash
:::::::::'##::::'##::::'###::::'####:'##::::::::::'###::::'####::::::::::
::::::::: ##:::: ##:::'## ##:::. ##:: ##:::::::::'## ##::: ####::::::::::
::::::::: ##:::: ##::'##:. ##::: ##:: ##::::::::'##:. ##::. ##:::::::::::
::::::::: ##:::: ##:'##:::. ##:: ##:: ##:::::::'##:::. ##:'##::::::::::::
:::::::::. ##:: ##:: #########:: ##:: ##::::::: #########:..:::::::::::::
::::::::::. ## ##::: ##.... ##:: ##:: ##::::::: ##.... ##::::::::::::::::
:::::::::::. ###:::: ##:::: ##:'####: ########: ##:::: ##::::::::::::::::
::::::::::::...:::::..:::::..::....::........::..:::::..:::::::::::::::::
Mocap fullbody_c3d        Markerless_3D_videos       Markerless_2D_video
                  \                |                /
                   v               v               v
            +-------------------------------------------+
IMU_csv --> |          vailá - multimodaltoolbox        | <-- Cluster_csv
            +-------------------------------------------+
                                  |
                                  v
                   +-----------------------------+
                   |           Results           |
                   +-----------------------------+
                                  |
                                  v
                       +---------------------+
                       | Visualization/Graph |
                       +---------------------+
=========================== File Manager ===============================
 Import (im)  |  Export (ex)  |  Copy (cp)  |  Move (mv)  |  Remove (rm)
========================= Available Multimodal =========================
1. IMU Analysis
2. Kinematic Cluster Analysis
3. Kinematic Motion Capture Full Body Analysis
4. Markerless 2D with video
5. Markerless 3D with multiple videos
============================= Available Tools ==========================
1. Edit CSV
2. Convert C3D data to CSV
3. Metadata info
4. Cut videos based on list
5. Draw a black box around videos
6. Compress videos to HEVC (H.265)
7. Compress videos to H.264
8. Plot 2D

Type 'h' for help or 'exit' to quit.

Choose an analysis option or file manager command:
```

## Project Structure

Here is an overview of the project structure:

```plaintext
╰─$ tree multimodal_mocap_coord_toolbox
multimodal_mocap_coord_toolbox
├── __init__.py
├── batchcut.py
├── cluster_analysis.py
├── cluster_analysis_cli.py
├── common_utils.py
├── compress_videos_h264.py
├── compress_videos_h265.py
├── compressvideo.py
├── data_processing.py
├── dialogsuser.py
├── dialogsuser_cluster.py
├── dlt2d.py
├── drawboxe.py
├── emg_labiocom.py
├── extractpng.py
├── filemanager.py
├── filtering.py
├── fonts
│   └── mrrobot.ttf
├── getpixelvideo.py
├── images
│   ├── cluster_config.png
│   ├── eeferp.png
│   ├── gui.png
│   ├── preto.png
│   ├── unf.png
│   ├── usp.png
│   ├── vaila.ico
│   ├── vaila_edge_w.png
│   ├── vaila_ico_mac.png
│   ├── vaila_ico_mac_original.png
│   ├── vaila_ico_trans.ico
│   ├── vaila_icon_win_original.ico
│   ├── vaila_logo.png
│   ├── vaila_trans_square.png
│   ├── vaila_transp.ico
│   └── vaila_white_square.png
├── imu_analysis copy.py
├── imu_analysis.py
├── listjointsnames.py
├── maintools.py
├── markerless_2D_analysis.py
├── markerless_3D_analysis.py
├── mergestack.py
├── metadatavid.sh
├── mocap_analysis copy.py
├── mocap_analysis.py
├── modifylabref.py
├── modifylabref_cli.py
├── numberframes.py
├── plotting.py
├── readc3d_export.py
├── readcsv copy.py
├── readcsv.py
├── readcsv_export.py
├── rearrange_data.py
├── rearrange_data_dask.py
├── rec2d.py
├── rec2d_one_dlt2d.py
├── rotation.py
├── run_vector_coding copy.py
├── run_vector_coding.py
├── run_vector_coding_GUI.py
├── showc3d copy.py
├── showc3d.py
├── showc3d_nodash.py
├── syncvid.py
├── utils.py
├── vaila_manifest.py
├── vailaplot2d.py
├── vailaplot3d.py
├── vector_coding.py
└── videos_synchronization.py
```

## Contribution

We welcome contributions to improve and expand the functionality of this toolbox. To contribute, please fork the repository, create a branch for your changes, and submit a pull request.

## License

This project is primarily licensed under the GNU Lesser General Public License v3.0. Note that the software is provided "as is", without warranty of any kind, express or implied. If you use the code or data, please cite us!
