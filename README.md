# _vailá_ - Multimodal Toolbox

[![GitHub release](https://img.shields.io/github/v/release/vaila-multimodaltoolbox/vaila)](https://github.com/vaila-multimodaltoolbox/vaila/releases)
[![License: AGPL v3](https://img.shields.io/badge/License-AGPL%20v3-blue.svg)](https://www.gnu.org/licenses/agpl-3.0.html)
[![Python 3.12](https://img.shields.io/badge/python-3.12%20%7C%20uv-brightgreen)](https://github.com/astral-sh/uv)

<p align="center">
  <img src="docs/images/vaila.png" alt="vailá Logo" width="300"/>
</p>

<div align="center">
  <table>
    <tr>
      <th>Operating System</th>
      <th>Installation Method</th>
      <th>Status</th>
    </tr>
    <tr>
      <td><strong>🪟 Windows</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
    <tr>
      <td><strong>🐧 Linux</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
    <tr>
      <td><strong>🍎 macOS</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
  </table>
</div>

---

## TL;DR

**_vailá_** is an open‑source Python toolbox that integrates video, motion‑capture, force‑plate, IMU, EMG/EEG and GNSS data into a reproducible, end‑to‑end multimodal biomechanics workflow. Install with a single click (binaries) or via the reproducible `uv` environment.

---

## Table of Contents

- [Abstract](#abstract)
- [Protocol Overview](#protocol-overview)
- [Key Features](#key-features)
- [Intended Audience](#intended-audience)
- [Installation](#stage-i-installation-and-environment-setup)
- [Quick‑Start Example](#quick-start-example)
- [Workflow Details (Stages II‑VI)](#protocol-workflow-details)
- [System Requirements](#system-requirements)
- [Manifest](#vailá-manifest)
- [Project Structure](#project-structure)
- [Uninstallation](#uninstallation-instructions)
- [Citation](#citing-vailá)
- [Contributing](#contribution)
- [License](#license)
- [References](#references)

---

## Abstract

Quantitative analysis of human movement increasingly relies on integrating heterogeneous data streams—video, motion capture, force plates, inertial sensors and electrophysiological recordings—into temporally aligned, three‑dimensional representations. However, many existing tools target a single modality, require extensive ad‑hoc scripting or depend on proprietary software, which limits accessibility and reproducibility. Here we present **_vailá_**, an open‑source Python toolbox that provides an end‑to‑end workflow for multimodal biomechanics. The protocol guides users through six stages: (I) installation and environment setup using the high‑performance `uv` package manager; (II) video preprocessing, including synchronization and optional distortion correction; (III) markerless 2D pose estimation via MediaPipe; (IV) camera calibration with Direct Linear Transformation (DLT); (V) 3D reconstruction from multi‑camera views; and (VI) interactive visualization and export to standard formats (C3D, CSV). In contrast to tools that focus solely on pose estimation or downstream musculoskeletal simulation, _vailá_ combines pose extraction, temporal synchronization, 2D‑3D reconstruction and human‑in‑the‑loop verification within a single, reproducible environment. The complete workflow—from raw video to 3D coordinates—can be executed in approximately 1–4 h, depending on dataset size and hardware, and requires only basic familiarity with the command line. Example datasets, configuration files and troubleshooting guides are provided. _vailá_ enables researchers in clinical rehabilitation, sports science and motor control to perform scalable, transparent and reproducible multimodal analyses without reliance on commercial software.

---

## Protocol Overview

This software implements a standardized protocol for multimodal analysis, organized into six specific stages:

1. **Stage I: Installation & Setup** – Environment creation using `uv` for reproducible Python dependency management.
2. **Stage II: Video Preprocessing** – Synchronization, cutting, and distortion correction of raw video feeds.
3. **Stage III: 2D Pose Estimation** – Markerless tracking using MediaPipe and YOLO models.
4. **Stage IV: Calibration** – Camera parameter estimation using Direct Linear Transformation (DLT).
5. **Stage V: 3D Reconstruction** – Converting 2D views into 3D metric coordinates.
6. **Stage VI: Visualization & Export** – Human‑in‑the‑loop verification and export to C3D/CSV.

---

## Key Features

- **Multimodal integration** – Video, MoCap (C3D), IMU, EMG/EEG, GNSS/GPS, force plates, HR/ECG, ultrasound.
- **End‑to‑end pipeline** – Six clearly defined stages from raw data to export.
- **Reproducible environment** – Managed by `uv` (deterministic lockfile).
- **Dual installation tracks** – Binary quick‑start for non‑technical users; source‑code protocol for reproducibility.
- **DeepLabCut integration** – Import DLC pose estimation data via `dlc2vaila.py`.
- **YOLOv11/v12 tracking** – Advanced object tracking with re‑identification.
- **Video processing suite** – Compression (H.264/H.265/H.266), merging, resizing, frame extraction.
- **Specialized biomechanical analyses** – Sit‑to‑Stand, vertical jump, gait (GRF), balance (stabilogram).
- **Animal behavior analysis** – Open field tracking and trajectory analysis.
- **Open‑source & extensible** – AGPL‑v3 license, community contributions welcome.
- **Extensive documentation** – GUI button guide, API reference, example datasets.

---

## Intended Audience

This protocol is intended for biomechanics researchers, rehabilitation scientists, and motor‑control specialists who seek an open‑source, accessible, reproducible workflow for multimodal data integration without reliance on commercial platforms. Users require only basic command‑line familiarity.

---

## Stage I: Installation and Environment Setup

To ensure reproducibility across different operating systems, _vailá_ offers two installation tracks:

### Option A: Quick Start (Binaries)

Pre‑compiled binaries are available for Windows and macOS. This is the fastest way to get started.

- **Windows**: Download `vaila-setup.exe` from the [Releases Page](https://github.com/vaila-multimodaltoolbox/vaila/releases).
- **macOS**: Download `vaila.dmg` from the same page.
- **Linux**: Please use the Source Code method (Option B).

### Option B: Protocol Implementation (Source Code)

This method ensures you have the exact environment used in the protocol, managed by `uv`. It allows code inspection and modification.

#### Prerequisite: Get the Code

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
```

#### 🪟 Windows (PowerShell)

```powershell
.\install_vaila_win_uv.ps1
```

#### 🐧 Linux (Bash)

```bash
chmod +x install_vaila_linux_uv.sh
./install_vaila_linux_uv.sh
```

#### 🍎 macOS (Bash/Zsh)

```bash
chmod +x install_vaila_mac_uv.sh
./install_vaila_mac_uv.sh
```

---

## Quick‑Start Example

The following one‑liner runs the full pipeline on a sample dataset (included in `docs/example_data`):

```bash
uv run vaila.py --input docs/example_data/video.mp4 --output results/ --stage all
```

The command automatically:

1. Sets up the `uv` environment (if not already done).
2. Performs video preprocessing, 2D pose estimation, calibration, 3D reconstruction and visualisation.
3. Saves the final 3D coordinates as `results/kinematics.c3d` and `results/kinematics.csv`.

---

## Protocol Workflow Details

The integration of _vailá_ into your research pipeline follows these processing stages:

### Stage II: Video Preprocessing

- **Goal**: Prepare video data for analysis to ensure temporal and spatial alignment.
- **Core Tools**:
  - **Synchronization**: `syncvid.py` – Multi‑camera temporal alignment with flash detection
  - **Trimming**: `cutvideo.py` – Interactive frame‑accurate cutting with batch support
  - **Lens Correction**: `vaila_distortvideo_gui.py` – Radial distortion removal
- **Additional Tools**:
  - **Compression**: `compress_videos_h264.py`, `compress_videos_h265.py`, `compress_videos_h266.py`
  - **Frame Extraction**: `extractpng.py` – Export frames as PNG sequences
  - **Video Merging**: `merge_multivideos.py` – Combine multiple video sources
  - **Resizing**: `resize_video.py` – Change resolution while preserving aspect ratio
  - **Duplicate Removal**: `rm_duplicateframes.py` – Clean repeated frames

### Stage III: Markerless 2D Pose Estimation

- **Goal**: Extract biological landmark coordinates from standard 2D video feeds.
- **Core Tools**:
  - **MediaPipe + YOLO**: `markerless2d_mpyolo.py` – Combined detection and pose estimation
  - **MediaPipe Standalone**: `markerless_2d_analysis.py` – Full‑body pose (33 landmarks)
  - **Hand Tracking**: `mphands.py` – 21 hand landmarks per hand
- **Additional Tools**:
  - **DeepLabCut Import**: `dlc2vaila.py` – Convert DLC outputs to vailá format
  - **YOLOv11/v12 Tracking**: `yolov11track.py`, `yolov12track.py` – Object tracking with re‑ID
  - **Angle Calculation**: `mpangles.py` – Joint angles from MediaPipe landmarks

### Stage IV: Camera Calibration (DLT)

- **Goal**: Establish the mathematical relationship between 2D pixel space and 3D metric space.
- **Tools**:
  - **2D Calibration**: `dlt2d.py` – 8‑parameter DLT for planar analysis
  - **3D Calibration**: `dlt3d.py` – 11‑parameter DLT for volumetric reconstruction
  - **Camera Parameters**: `getcampardistortlens.py` – Extract intrinsic parameters

### Stage V: 3D Reconstruction

- **Goal**: Triangulate 2D coordinates from multiple views into a unified 3D reconstruction.
- **Tools**:
  - **Multi‑Camera Reconstruction**: `rec3d.py` – Least‑squares triangulation
  - **Single‑DLT 2D Reconstruction**: `rec2d_one_dlt2d.py` – Planar reconstruction
  - **Multi‑DLT 3D Reconstruction**: `rec3d_one_dlt3d.py` – Per‑frame DLT parameters

### Stage VI: Visualization and Export

- **Goal**: Validate results through visualization and export standard biomechanics formats.
- **Core Tools**:
  - **3D Viewer**: `viewc3d.py` – Interactive Open3D visualization with marker selection
  - **2D Plotting**: `vailaplot2d.py` – Time series, scatter, and multi‑axis plots
  - **C3D Preview**: `showc3d.py` – Quick C3D file inspection
- **Export Formats**:
  - **C3D**: Standard motion capture format (Vicon, Qualisys compatible)
  - **CSV**: Universal tabular format for statistical analysis
  - **Excel**: Optional `.xlsx` export for spreadsheet users

---

## Supported Modalities

| Modality            | Input Formats      | Key Scripts                                 | Description                          |
| ------------------- | ------------------ | ------------------------------------------- | ------------------------------------ |
| **Motion Capture**  | C3D, CSV           | `readc3d_export.py`, `mocap_analysis.py`    | Vicon, Qualisys, OptiTrack           |
| **Markerless Pose** | Video (MP4, AVI)   | `markerless2d_mpyolo.py`                    | MediaPipe, YOLO, DeepLabCut          |
| **IMU/Inertial**    | CSV, C3D           | `imu_analysis.py`                           | Delsys, Noraxon, Xsens               |
| **EMG**             | CSV, C3D           | `emg_labiocom.py`                           | Spectral analysis, fatigue detection |
| **Force Plates**    | CSV, C3D           | `forceplate_analysis.py`, `cop_analysis.py` | AMTI, Bertec, Kistler                |
| **GNSS/GPS**        | GPX, KML, KMZ, CSV | `gnss_analysis.py`                          | Trajectory, speed, distance          |
| **Ultrasound**      | Images             | `usound_biomec1.py`                         | Muscle architecture analysis         |

---

## Specialized Analysis Types

| Analysis                  | Script                                       | Description                                            |
| ------------------------- | -------------------------------------------- | ------------------------------------------------------ |
| **Balance/Posturography** | `cop_analysis.py`, `stabilogram_analysis.py` | Center of Pressure, sway metrics, ellipse area         |
| **Gait Analysis**         | `grf_gait.py`, `numstepsmp.py`               | Ground reaction forces, step detection, spatiotemporal |
| **Vertical Jump**         | `vaila_and_jump.py`                          | Countermovement jump metrics (flight time, peak force) |
| **Sit‑to‑Stand**          | `sit2stand.py`                               | Functional mobility assessment                         |
| **Vector Coding**         | `run_vector_coding.py`                       | Intersegmental coordination patterns                   |
| **Cluster Kinematics**    | `cluster_analysis.py`                        | Euler angles from marker clusters                      |
| **Animal Open Field**     | `animal_open_field.py`                       | Rodent trajectory and behavior analysis                |
| **Soccer Field**          | `soccerfield.py`                             | Player tracking visualization                          |

---

## System Requirements

- **Operating Systems**: Windows 10 or later, macOS 12 or later, Ubuntu 20.04 or later.
- **Python**: 3.12 (installed automatically by `uv`).
- **External Dependencies**: FFmpeg (installed by the scripts), CUDA (optional for GPU‑accelerated YOLO), Git.
- **Hardware**: Minimum 8 GB RAM, GPU recommended for large video datasets.

---

## _vailá_ Manifest

### English Version

Join us in the liberation from paid software with the "_vailá_ – Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

In front of you stands a versatile tool designed to challenge the boundaries of commercial systems. This software is a symbol of innovation and freedom, determined to eliminate barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.

With _vailá_ you are invited to explore, experiment, and create without constraints. "_vailá_" means "go there and do it!" — encouraging you to harness its power to perform analysis with data from multiple systems.

### Versão em Português

Junte-se a nós na libertação do software pago com o "_vailá_: Análise Versátil da Libertação Anarquista Integrada na Caixa de Ferramentas Multimodal".

Diante de você está uma ferramenta versátil, projetada para desafiar as fronteiras dos sistemas comerciais. Este software é um símbolo de inovação e liberdade, determinado a eliminar as barreiras que protegem o monopólio do software caro, garantindo a disseminação do conhecimento e a acessibilidade.

Com _vailá_ você é convidado a explorar, experimentar e criar sem restrições. "_vailá_" significa "vai lá e faça!" — encorajando você a aproveitar seu poder para realizar análises com dados de múltiplos sistemas.

---

## Project Structure

```text
vaila/
├── vaila.py                # Main entry point
├── install_vaila_linux_uv.sh
├── install_vaila_mac_uv.sh
├── install_vaila_win_uv.ps1
├── pyproject.toml          # Dependency specification (uv)
├── docs/
│   ├── images/
│   ├── vaila_nature_protocols_v1-1.pdf
│   └── help.md
├── vaila/                  # Package source
│   ├── __init__.py
│   ├── ... (modules)
└── tests/                  # Test suite
```

---

## Uninstallation Instructions

### Linux

```bash
sudo chmod +x uninstall_vaila_linux.sh
./uninstall_vaila_linux.sh
```

### macOS

```bash
sudo chmod +x uninstall_vaila_mac.sh
./uninstall_vaila_mac.sh
```

### Windows (uv method)

Delete the installation folder (`C:\Program Files\vaila` or `C:\Users\<User>\vaila`), remove shortcuts and the Windows‑Terminal profile if created.

---

## Citing _vailá_

```bibtex
@misc{vaila2024,
  title={vailá – Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo Roberto Pereira Santiago and Guilherme Manna Cesar and Ligia Yumi Mochida and Juan Aceros and others},
  year={2024},
  eprint={2410.07238},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2410.07238}
}

@article{santiago2025vaila,
  title={vailá: an open‑source multimodal toolbox for biomechanics},
  author={Santiago, Paulo RP and Cesar, Guilherme M and Mochida, Ligia Y and others},
  journal={Nature Protocols},
  year={2025},
  doi={10.1038/s41596-025-XXXX-X}
}
```

Please cite both the pre‑print and the final Nature Protocols article.

---

## Contribution

We encourage creativity and innovation to enhance and expand the functionality of this toolbox. Fork the repository, experiment with new ideas, and create a branch for your changes. When you’re ready, submit a pull request so we can review and potentially integrate your contributions.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL‑v3). The license ensures that any use of _vailá_, including network/server usage, maintains the freedom of the software and requires source‑code availability.

---

## References

1. Santiago, P. R. P. _et al._ "vailá – Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox". _arXiv_ 2410.07238 (2024).
2. Tahara, A. K., Chinaglia, A. G., Monteiro, R. L. M., et al. "Predicting walkway spatiotemporal parameters using a markerless, pixel‑based machine learning approach". _Brazilian Journal of Motor Behavior_ 19, 1 (2025).
3. Mochida, L. Y., Santiago, P. R. P., Lamb, M., Cesar, G. M. "Multimodal Motion Capture Toolbox for Enhanced Analysis of Intersegmental Coordination in Children with Cerebral Palsy and Typically Developing". _JOVE_ 206, e69604 (2025).
4. Nature Protocols Author Guidelines: https://www.nature.com/nprot/for-authors/protocols

---

## Mermaid Diagram of the Workflow

```mermaid
flowchart LR
    A[Stage I: Installation] --> B[Stage II: Video Pre‑processing]
    B --> C[Stage III: 2D Pose Estimation]
    C --> D[Stage IV: Calibration (DLT)]
    D --> E[Stage V: 3D Reconstruction]
    E --> F[Stage VI: Visualization & Export]
```

---

_End of README_

<p align="center">
  <img src="docs/images/vaila.png" alt="vailá Logo" width="300"/>
</p>

<div align="center">
  <table>
    <tr>
      <th>Operating System</th>
      <th>Installation Method</th>
      <th>Status</th>
    </tr>
    <tr>
      <td><strong>🪟 Windows</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
    <tr>
      <td><strong>🐧 Linux</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
    <tr>
      <td><strong>🍎 macOS</strong></td>
      <td>uv (Recommended)</td>
      <td>✅ Ready</td>
    </tr>
  </table>
</div>

## Development of _vailá_: Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox

## Abstract

Quantitative analysis of human movement increasingly relies on integrating heterogeneous data streams—video, motion capture, force plates, inertial sensors and electrophysiological recordings—into temporally aligned, three-dimensional representations. However, many existing tools target a single modality, require extensive ad-hoc scripting or depend on proprietary software, which limits accessibility and reproducibility. Here we present **_vailá_**, an open-source Python toolbox that provides an end-to-end workflow for multimodal biomechanics. The protocol guides users through six stages: (I) installation and environment setup using the high-performance `uv` package manager; (II) video preprocessing, including synchronization and optional distortion correction; (III) markerless 2D pose estimation via MediaPipe; (IV) camera calibration with Direct Linear Transformation (DLT); (V) 3D reconstruction from multi-camera views; and (VI) interactive visualization and export to standard formats (C3D, CSV). In contrast to tools that focus solely on pose estimation or downstream musculoskeletal simulation, _vailá_ combines pose extraction, temporal synchronization, 2D–3D reconstruction and human-in-the-loop verification within a single, reproducible environment. The complete workflow—from raw video to 3D coordinates—can be executed in approximately 1–4 h, depending on dataset size and hardware, and requires only basic familiarity with the command line. Example datasets, configuration files and troubleshooting guides are provided. _vailá_ enables researchers in clinical rehabilitation, sports science and motor control to perform scalable, transparent and reproducible multimodal analyses without reliance on commercial software.

## Protocol Overview

This software implements a standardized protocol for multimodal analysis, organized into six specific stages:

1.  **Stage I: Installation & Setup** - Environment creation using `uv` for reproducible Python dependency management.
2.  **Stage II: Video Preprocessing** - Synchronization, cutting, and distortion correction of raw video feeds.
3.  **Stage III: 2D Pose Estimation** - Markerless tracking using MediaPipe and YOLO models.
4.  **Stage IV: Calibration** - Camera parameter estimation using DLT (Direct Linear Transformation).
5.  **Stage V: 3D Reconstruction** - Converting 2D views into 3D metric coordinates.
6.  **Stage VI: Visualization & Export** - Human-in-the-loop verification and export to C3D/CSV.

## Intended Audience

This protocol is intended for biomechanics researchers, rehabilitation scientists, and motor control specialists who seek an open-source, accessible, reproducible workflow for multimodal data integration without reliance on commercial platforms. Users require only basic command-line familiarity.

## _vailá_ Manifest

### English Version

Join us in the liberation from paid software with the "_vailá_ - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox."

In front of you stands a versatile tool designed to challenge the boundaries of commercial systems. This software is a symbol of innovation and freedom, determined to eliminate barriers that protect the monopoly of expensive software, ensuring the dissemination of knowledge and accessibility.

With _vailá_, you are invited to explore, experiment, and create without constraints. "_vailá_" means "go there and do it!" — encouraging you to harness its power to perform analysis with data from multiple systems.

### Versão em Português

Junte-se a nós na libertação do software pago com o "_vailá_: Análise Versátil da Libertação Anarquista Integrada na Caixa de Ferramentas Multimodal".

Diante de você está uma ferramenta versátil, projetada para desafiar as fronteiras dos sistemas comerciais. Este software é um símbolo de inovação e liberdade, determinado a eliminar as barreiras que protegem o monopólio do software caro, garantindo a disseminação do conhecimento e a acessibilidade.

Com _vailá_, você é convidado a explorar, experimentar e criar sem restrições. "_vailá_" significa "vai lá e faça!" — encorajando você a aproveitar seu poder para realizar análises com dados de múltiplos sistemas.

---

## Stage I: Installation and Environment Setup

To ensure reproducibility across different operating systems, _vailá_ offers two installation tracks:

1.  **Quick Start (Binaries)**: Recommended for trial, education, or non-coding users.
2.  **Protocol Implementation (Source Code)**: Recommended for the _Nature Protocols_ workflow, utilizing `uv` for reproducible environment management.

### Option A: Quick Start (Binaries)

Pre-compiled binaries are available for Windows and macOS. This is the fastest way to get started.

- **Windows**: Download `vaila-setup.exe` from the [Releases Page](https://github.com/vaila-multimodaltoolbox/vaila/releases).
- **macOS**: Download `vaila.dmg` from the [Releases Page](https://github.com/vaila-multimodaltoolbox/vaila/releases).
- **Linux**: Please use the Source Code method.

### Option B: Protocol Implementation (Source Code)

This method ensures you have the exact environment used in the protocol, managed by `uv`. It allows for code inspection and modification.

#### Prerequisite: Get the Code

Clone the repository or download the ZIP:

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
```

#### 🪟 Windows (PowerShell)

Run the automated PowerShell script. This script installs `uv`, Python 3.12, FFmpeg, and sets up the environment.

```powershell
.\install_vaila_win_uv.ps1
```

#### 🐧 Linux (Bash)

Make the script executable and run it:

```bash
chmod +x install_vaila_linux_uv.sh
./install_vaila_linux_uv.sh
```

#### 🍎 macOS (Bash/Zsh)

Make the script executable and run it:

```bash
chmod +x install_vaila_mac_uv.sh
./install_vaila_mac_uv.sh
```

### Verification (All Platforms)

To verify the installation, launch _vailá_:

```bash
uv run vaila.py
```

## Protocol Workflow Details

The integration of _vailá_ into your research pipeline follows these processing stages:

### Stage II: Video Preprocessing

- **Goal**: Prepare video data for analysis to ensure temporal and spatial alignment.
- **Tools**:
  - **Synchronization**: Use `C_B_r2_c3 - Make Sync file` to aligns varied frame rates and start times.
  - **Trimming**: Use `C_B_r4_c2 - Cut Video` to isolate the specific movement of interest.
  - **Lens Correction**: Use `C_B_r4_c1 - Distort video` to remove radial distortion if necessary.

### Stage III: Markerless 2D Pose Estimation

- **Goal**: Extract biological landmark coordinates from standard 2D video feeds.
- **Tools**:
  - **Inference**: Select `B1_r1_c4 - Markerless2D` to apply MediaPipe or YOLO models.
  - **Verification**: Review the generated overlay videos to ensure tracking fidelity.

### Stage IV: Camera Calibration (DLT)

- **Goal**: Establish the mathematical relationship between 2D pixel space and 3D metric space.
- **Tools**:
  - **Parameter Calculation**: Apply `C_A_r2_c1 - Make DLT2D` or `C_A_r3_c1 - Make DLT3D` using reference points or calibration frames.

### Stage V: 3D Reconstruction

- **Goal**: Triangulate 2D coordinates from multiple views into a unified 3D reconstruction.
- **Tools**:
  - **Reconstruction Algorithm**: Execute `C_A_r3_c2 - Rec3D 1DLT` or `C_A_r3_c3 - Rec3D MultiDLT` to generate metric 3D data.

### Stage VI: Visualization and Export

- **Goal**: Validate results through visualization and export standard biomechanics formats.
- **Tools**:
  - **Interactive Plotting**: Use `C_C_r2_c2 - Plot 3D` to explore kinematics.
  - **Export**: Data is automatically saved in widely supported `.c3d` and `.csv` formats for statistical analysis.

---

## Uninstallation Instructions

## For Uninstallation on Linux 🐧

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

## For Uninstallation on macOs 🍎

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

## For Uninstallation on Windows 🪟

### If you installed using uv (New Method)

1. **Manual Removal**:

   - Delete the installation directory:
     - If installed as Administrator: `C:\Program Files\vaila`
     - If installed as Standard User: `C:\Users\<YourUser>\vaila`
   - Remove the Desktop shortcut
   - Remove the Start Menu shortcut
   - Remove the Windows Terminal profile manually (if needed)

2. **Windows Terminal Profile Removal**:
   - Open Windows Terminal settings (JSON)
   - Remove the `vaila` profile entry from `profiles.list`

### If you installed using Conda (Legacy Method)

1. **Run the uninstallation script as Administrator in Anaconda/Miniconda PowerShell Prompt**:

- PowerShell Script:
  ```powershell
  ExecutionPolicy Bypass -File .\uninstall_vaila_win.ps1
  .\uninstall_vaila_win.ps1
  ```

2. **Follow the Instructions Displayed by the Script**:

- The script will:
  - Remove the `vaila` Conda environment.
  - Delete the `C:\Users\your_user_name_here\AppData\Local\vaila` directory.
  - Remove the Windows Terminal profile (settings.json file).
  - Delete the desktop shortcut if it exists.

3. **Manual Removal of Windows Terminal Profile (if necessary)**:

- If the Windows Terminal profile is not removed automatically (e.g., when using the `uninstall_vaila_win.ps1` script), you may need to remove it manually:

```Anaconda/Miniconda PowerShell Prompt
conda remove -n vaila --all
```

Remove directory `vaila` inside `C:\Users\your_user_name_here\AppData\Local\vaila`.

---

## Project Structure

<p align="center">
  <img src="docs/images/vaila_start_gui.png" alt="vailá Start GUI" width="600"/>
</p>

```bash
                                             o
                                _,  o |\  _,/
                          |  |_/ |  | |/ / |
                           \/  \/|_/|/|_/\/|_/
##########################################################################
Mocap fullbody_c3d           Markerless_3D       Markerless_2D_MP
                  \                |                /
                   v               v               v
   CUBE2D  --> +---------------------------------------+ <-- Vector Coding
   IMU_csv --> |       vailá - multimodal toolbox      | <-- Cluster_csv
Open Field --> +---------------------------------------+ <-- Force Plate
              ^                   |                    ^ <-- YOLOv11 and MediaPipe
        EMG__/                    v                     \__Tracker YOLOv11
                    +--------------------------+
                    | Results: Data and Figure |
                    +--------------------------+

============================ File Manager (Frame A) ========================
A_r1_c1 - Rename          A_r1_c2 - Import           A_r1_c3 - Export
A_r1_c4 - Copy            A_r1_c5 - Move             A_r1_c6 - Remove
A_r1_c7 - Tree            A_r1_c8 - Find             A_r1_c9 - Transfer

========================== Multimodal Analysis (Frame B) ===================
B1_r1_c1 - IMU            B1_r1_c2 - MoCapCluster    B1_r1_c3 - MoCapFullBody
B1_r1_c4 - Markerless2D   B1_r1_c5 - Markerless3D

B2_r2_c1 - Vector Coding  B2_r2_c2 - EMG             B2_r2_c3 - Force Plate
B2_r2_c4 - GNSS/GPS       B2_r2_c5 - MEG/EEG

B3_r3_c1 - HR/ECG         B3_r3_c2 - Markerless_MP_Yolo  B3_r3_c3 - vailá_and_jump
B3_r3_c4 - Cube2D         B3_r3_c5 - Animal Open Field
B3_r4_c1 - Tracker        B3_r4_c2 - ML Walkway       B3_r4_c3 - Markerless Hands
B3_r4_c4 - vailá          B3_r4_c5 - vailá
============================== Tools Available (Frame C) ===================
C_A: Data Files

C_A_r1_c1 - Edit CSV      C_A_r1_c2 - C3D <--> CSV   C_A_r1_c3 - Gapfill | split
C_A_r2_c1 - Make DLT2D    C_A_r2_c2 - Rec2D 1DLT     C_A_r2_c3 - Rec2D MultiDLT
C_A_r3_c1 - Make DLT3D    C_A_r3_c2 - Rec3D 1DLT     C_A_r3_c3 - Rec3D MultiDLT
C_A_r4_c1 - vailá         C_A_r4_c2 - vailá          C_A_r4_c3 - vailá

C_B: Video and Image
C_B_r1_c1 - Video<-->PNG  C_B_r1_c2 - vaiá          C_B_r1_c3 - Draw Box
C_B_r2_c1 - CompressH264  C_B_r2_c2 - CompressH265  C_B_r2_c3 - Make Sync file
C_B_r3_c1 - GetPixelCoord C_B_r3_c2 - Metadata info C_B_r3_c3 - Merge Videos
C_B_r4_c1 - Distort video C_B_r4_c2 - Cut Video     C_B_r4_c3 - vailá

C_C: Visualization
C_C_r1_c1 - Show C3D      C_C_r1_c2 - Show CSV       C_C_r2_c1 - Plot 2D
C_C_r2_c2 - Plot 3D       C_C_r3_c1 - vailá          C_C_r3_c2 - vailá
C_C_r4_c1 - vailá         C_C_r4_c2 - vailá          C_C_r4_c3 - vailá

Type 'h' for help or 'exit' to quit.

Use the button 'imagination!' to access command-line (xonsh) tools for advanced multimodal analysis!
```

An overview of the project structure:

```bash
vaila
├── vaila.py                        # Main Entry Point
├── install_vaila_linux_uv.sh       # Linux Installer (uv)
├── install_vaila_mac_uv.sh         # macOS Installer (uv)
├── install_vaila_win_uv.ps1        # Windows Installer (uv)
├── pyproject.toml                  # Project Dependencies (uv/poetry)
├── vaila                           # Package Source Directory
│   ├── __init__.py
│   ├── animal_open_field.py        # Animal Open Field analysis
│   ├── backup_markerless.py        # Backup tools for markerless data
│   ├── batchcut.py                 # Batch video cutting tools
│   ├── brainstorm.py               # Brainstorming/Notes tool
│   ├── cluster_analysis.py         # Cluster analysis for motion capture
│   ├── common_utils.py             # Common utility functions
│   ├── compress_videos_h264.py     # H.264 video compression
│   ├── compress_videos_h265.py     # H.265 (HEVC) video compression
│   ├── cop_analysis.py             # Center of Pressure (CoP) analysis
│   ├── cube2d_kinematics.py        # 2D kinematics analysis tools
│   ├── cutvideo.py                 # Video cutting tools
│   ├── dlc2vaila.py                # DeepLabCut to vailá converter
│   ├── dlt2d.py                    # 2D Direct Linear Transformation
│   ├── dlt3d.py                    # 3D Direct Linear Transformation
│   ├── drawboxe.py                 # Draw box in video frames
│   ├── emg_labiocom.py             # EMG signal analysis tools
│   ├── extractpng.py               # Extract PNG frames from videos
│   ├── filemanager.py              # File management utilities
│   ├── force_cube_fig.py           # 3D force data visualization
│   ├── forceplate_analysis.py      # Force plate analysis tools
│   ├── getpixelvideo.py            # Extract pixel coordinates from video
│   ├── gnss_analysis.py            # GNSS/GPS data analysis tools
│   ├── grf_gait.py                 # Ground Reaction Force (GRF) gait analysis
│   ├── images/                     # GUI assets and images
│   ├── imu_analysis.py             # IMU sensor data analysis
│   ├── interp_smooth_split.py      # Interpolation and smoothing tools
│   ├── markerless2d_mpyolo.py      # Markerless 2D tracking (MP-YOLO)
│   ├── markerless_live.py          # Live markerless tracking
│   ├── merge_multivideos.py        # Merge multiple videos
│   ├── ml_models_training.py       # ML models training
│   ├── mocap_analysis.py           # Motion capture full body analysis
│   ├── models/                     # Trained models (YOLO, etc.)
│   ├── modifylabref.py             # Modify laboratory references
│   ├── mpangles.py                 # MediaPipe angles calculation
│   ├── mphands.py                  # MediaPipe hands analysis
│   ├── plotting.py                 # Data plotting tools
│   ├── process_gait_features.py    # Gait feature extraction
│   ├── readc3d_export.py           # Read and export C3D files
│   ├── readcsv.py                  # Read CSV data
│   ├── rec2d.py                    # 2D Reconstruction
│   ├── rec3d.py                    # 3D Reconstruction
│   ├── reid_markers.py             # Re-identification of markers
│   ├── reid_yolotrack.py           # Re-ID with YOLO tracking
│   ├── rotation.py                 # Rotation analysis tools
│   ├── run_vector_coding.py        # Vector coding analysis
│   ├── scout_vaila.py              # Scout tool
│   ├── showc3d.py                  # Visualize C3D data
│   ├── sit2stand.py                # Sit-to-Stand analysis
│   ├── soccerfield.py              # Soccer field analysis
│   ├── spectral_features.py        # Spectral feature extraction
│   ├── stabilogram_analysis.py     # Stabilogram analysis tools
│   ├── syncvid.py                  # Synchronize video files
│   ├── utils.py                    # General utility scripts
│   ├── vaila_and_jump.py           # Vertical jump analysis tool
│   ├── vaila_manifest.py           # Manifest file for vailá
│   ├── vailaplot2d.py              # Plot 2D biomechanical data
│   ├── vailaplot3d.py              # Plot 3D biomechanical data
│   ├── vector_coding.py            # Joint vector coding analysis
│   ├── videoprocessor.py           # Video processing tools
│   ├── viewc3d.py                  # Visualize C3D files
│   ├── walkway_ml_prediction.py    # ML prediction for walkway
│   ├── yolotrain.py                # YOLO training utility
│   ├── yolov11track.py             # YOLOv11 based tracking
│   └── yolov12track.py             # YOLOv12 based tracking
```

## Documentation

### 📚 Script Help Documentation

Comprehensive documentation for all Python scripts and modules in \_ vailá:

- **[Script Help Index (HTML)](vaila/help/index.html)** - Complete documentation for all Python modules and scripts (HTML version)
- **[Script Help Index (Markdown)](vaila/help/index.md)** - Complete documentation for all Python modules and scripts (Markdown version)

The help documentation includes detailed information about:

- Module descriptions and functionality
- Configuration parameters
- Usage instructions
- Input/output formats
- Requirements and dependencies

### 📖 Additional Documentation

- **[Project Documentation](docs/index.md)** - Overview and module documentation
- **[Help Guide](docs/help.md)** - User guide and installation instructions
- **[GUI Button Documentation](docs/vaila_buttons/README.md)** - Complete documentation for all GUI buttons

---

## Citing _vailá_

If you use _vailá_ in your research or project, please consider citing our work:

```bibtex
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo Roberto Pereira Santiago and Guilherme Manna Cesar and Ligia Yumi Mochida and Juan Aceros and others},
  year={2024},
  eprint={2410.07238},
  archivePrefix={arXiv},
  primaryClass={cs.HC},
  url={https://arxiv.org/abs/2410.07238}
}

@article{tahara2025predicting,
  title={Predicting walkway spatiotemporal parameters using a markerless, pixel-based machine learning approach},
  author={Tahara, Ariany K and Chinaglia, Abel G and Monteiro, Rafael LM and Bedo, Bruno LS and Cesar, Guilherme M and Santiago, Paulo RP},
  journal={Brazilian Journal of Motor Behavior},
  volume={19},
  number={1},
  pages={e462--e462},
  year={2025}
}

@article{Mochida2025,
  author = {Mochida, Ligia Yumi and Santiago, Paulo R. P. and Lamb, Miranda and Cesar, Guilherme M.},
  title = {Multimodal Motion Capture Toolbox for Enhanced Analysis of Intersegmental Coordination in Children with Cerebral Palsy and Typically Developing},
  journal = {Journal of Visualized Experiments},
  year = {2025},
  number = {206},
  pages = {e69604},
  doi = {10.3791/69604},
  url = {https://www.jove.com/t/69604/multimodal-motion-capture-toolbox-for-enhanced-analysis}
}
```

## You can also refer to the tool's GitHub repository for more details and updates:

- [_vailá_ on arXiv](https://arxiv.org/abs/2410.07238)
- [_vailá_ GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)

## Contribution

We encourage creativity and innovation to enhance and expand the functionality of this toolbox. You can make a difference by contributing to the project! To get involved, feel free to fork the repository, experiment with new ideas, and create a branch for your changes. When you're ready, submit a pull request so we can review and potentially integrate your contributions.

Don't hesitate to learn, explore, and experiment. Be bold, and don't be afraid to make mistakes—every attempt is a step towards improvement!

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPLv3).
This license ensures that any use of vailá, including network/server usage,
maintains the freedom of the software and requires source code availability.

For more details, see the [LICENSE](LICENSE) file or visit:
https://www.gnu.org/licenses/agpl-3.0.html
