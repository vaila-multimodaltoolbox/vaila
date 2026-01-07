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

**_vailá_** is an open‑source Python toolbox that integrates video, motion‑capture, force‑plate, IMU, EMG/EEG and GNSS data into a reproducible, end‑to‑end multimodal biomechanics workflow. 

**Quick Start:**
1. Install via binaries (Windows/macOS) or source code (all platforms)
2. Launch GUI: `uv run vaila.py`
3. Access all tools through the intuitive graphical interface

Installation is automated via `uv` - no manual Python setup required!

---

## Table of Contents

- [Abstract](#abstract)
- [Protocol Overview](#protocol-overview)
- [Key Features](#key-features)
- [Intended Audience](#intended-audience)
- [Installation](#stage-i-installation-and-environment-setup)
- [Quick‑Start Example](#quick-start-example)
- [How to Use _vailá_](#how-to-use-vailá)
- [Workflow Details (Stages II‑VI)](#protocol-workflow-details)
- [Supported Modalities](#supported-modalities)
- [Specialized Analysis Types](#specialized-analysis-types)
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

Quantitative analysis of human movement increasingly relies on integrating heterogeneous data streams—video, motion capture, force plates, inertial sensors and electrophysiological recordings—into temporally aligned, three‑dimensional representations. However, many existing tools target a single modality, require extensive ad‑hoc scripting or depend on proprietary software, which limits accessibility and reproducibility. Here we present **_vailá_**, an open‑source Python toolbox that provides an end‑to‑end workflow for multimodal biomechanics. The protocol guides users through six stages: (I) installation and environment setup using the high‑performance `uv` package manager; (II) video preprocessing, including synchronization and optional distortion correction; (III) markerless 2D pose estimation via MediaPipe; (IV) camera calibration with Direct Linear Transformation (DLT); (V) 3D reconstruction from multi‑camera views; and (VI) interactive visualization and export to standard formats (C3D, CSV). In contrast to tools that focus solely on pose estimation or downstream musculoskeletal simulation, _vailá_ combines pose extraction, temporal synchronization, 2D‑3D reconstruction and human‑in‑the‑loop verification within a single, reproducible environment. The complete workflow—from raw video to 3D coordinates—can be executed in approximately 1–4 h, depending on dataset size and hardware, and requires only basic familiarity with the command line. Example datasets, configuration files and troubleshooting guides are provided. _vailá_ enables researchers in clinical rehabilitation, sports science and motor control to perform scalable, transparent and reproducible multimodal analyses without reliance on commercial software.

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
- **YOLOv11 tracking** – Advanced object tracking with re‑identification.
- **Video processing suite** – Compression (H.264/H.265/H.266), merging, resizing, frame extraction.
- **Specialized biomechanical analyses** – Sit‑to‑Stand, vertical jump, gait (GRF), balance (stabilogram).
- **Animal behavior analysis** – Open field tracking and trajectory analysis.
- **Open‑source & extensible** – AGPL‑v3 license, community contributions welcome.
- **Extensive documentation** – GUI button guide, API reference, example datasets.

---

## Intended Audience

This protocol is intended for biomechanics researchers, rehabilitation scientists, and motor‑control specialists who seek an open‑source, accessible, reproducible workflow for multimodal data integration without reliance on commercial platforms. 

**User Requirements:**
- Basic familiarity with command line (for installation)
- GUI-based workflow means minimal technical expertise needed for daily use
- Python knowledge helpful but not required (all tools accessible via GUI)

---

## Stage I: Installation and Environment Setup

To ensure reproducibility across different operating systems, _vailá_ offers two installation tracks:

### Option A: Quick Start (Binaries)

Pre‑compiled binaries are available for Windows and macOS. This is the fastest way to get started and requires no technical setup.

- **Windows**: Download `vaila-setup.exe` from the [Releases Page](https://github.com/vaila-multimodaltoolbox/vaila/releases) and run the installer.
- **macOS**: Download `vaila.dmg` from the [Releases Page](https://github.com/vaila-multimodaltoolbox/vaila/releases), mount it, and drag the application to your Applications folder.
- **Linux**: Please use the Source Code method (Option B).

After installation, launch _vailá_ from your applications menu or desktop shortcut.

### Option B: Protocol Implementation (Source Code)

This method ensures you have the exact environment used in the protocol, managed by `uv`. It allows code inspection and modification. The installation scripts automatically handle all dependencies.

#### Prerequisite: Get the Code

```bash
git clone https://github.com/vaila-multimodaltoolbox/vaila.git
cd vaila
```

#### 🪟 Windows (PowerShell)

Run the installation script (Administrator privileges recommended for full installation):

```powershell
.\install_vaila_win_uv.ps1
```

**What happens during installation:**
- `uv` package manager is automatically installed if not present
- Python 3.12.12 is installed via `uv` if needed
- All Python dependencies are installed from `pyproject.toml`
- FFmpeg is installed for video processing
- Installation location:
  - **With Administrator**: `C:\Program Files\vaila`
  - **Without Administrator**: `C:\Users\<YourUser>\vaila`

#### 🐧 Linux (Bash)

Make the script executable and run it:

```bash
chmod +x install_vaila_linux_uv.sh
./install_vaila_linux_uv.sh
```

**What happens during installation:**
- System dependencies (Python, Git, FFmpeg, etc.) are installed via `apt`
- `uv` package manager is automatically installed if not present
- Python 3.12.12 is installed via `uv` if needed
- All Python dependencies are installed from `pyproject.toml`
- Installation location: `~/vaila`

#### 🍎 macOS (Bash/Zsh)

Make the script executable and run it:

```bash
chmod +x install_vaila_mac_uv.sh
./install_vaila_mac_uv.sh
```

**What happens during installation:**
- System dependencies are verified and installed via Homebrew if needed
- `uv` package manager is automatically installed if not present
- Python 3.12.12 is installed via `uv` if needed
- All Python dependencies are installed from `pyproject.toml`
- FFmpeg is installed for video processing
- Installation location: `~/vaila`

#### Verification

After installation, verify that _vailá_ is working correctly:

```bash
uv run vaila.py
```

If the GUI launches successfully, installation is complete!

---

## Quick‑Start Example

After installation, launch the _vailá_ graphical interface:

```bash
uv run vaila.py
```

This opens the main GUI where you can access all tools through organized buttons:

- **File Management (Frame A)**: Rename, import, export, copy, move, and organize files
- **Multimodal Analysis (Frame B)**: IMU, MoCap, markerless tracking, EMG, force plates, and more
- **Tools (Frame C)**: Video processing, DLT calibration, data conversion, visualization

The GUI provides an intuitive way to access all _vailá_ functionality without command-line complexity.

### Running Individual Modules

You can also run individual modules directly from the command line:

```bash
# Run YOLOv11 tracker
uv run python -m vaila.yolov11track

# Run markerless 2D analysis
uv run python -m vaila.markerless_2d_analysis

# Run video cutting tool
uv run python -m vaila.cutvideo
```

All modules are accessible both through the GUI and directly via command line.

---

## How to Use _vailá_

_vailá_ is primarily a GUI-based application, making it accessible to users with minimal command-line experience.

### Launching the Application

**Main GUI (Recommended):**
```bash
uv run vaila.py
```

This launches the main graphical interface where all tools are organized into logical sections. Most users will interact with _vailá_ exclusively through this GUI.

### Accessing Tools

**Via GUI:**
- Click buttons in the main interface to launch specific tools
- Each tool opens its own configuration dialog
- Follow on-screen prompts to select files and configure parameters

**Via Command Line (Advanced):**
Individual modules can be run directly for scripting and automation:
```bash
uv run python -m vaila.<module_name>
```

### Documentation

- **GUI Button Guide**: See `docs/vaila_buttons/README.md` for detailed documentation on all GUI buttons
- **Module Help**: Each module includes built-in help accessible via the GUI
- **Online Documentation**: Visit the [project documentation](docs/index.md) for comprehensive guides

---

## Protocol Workflow Details

The integration of _vailá_ into your research pipeline follows these processing stages. All tools are accessible through the GUI, and can also be run directly as Python modules:

### Stage II: Video Preprocessing

- **Goal**: Prepare video data for analysis to ensure temporal and spatial alignment.
- **Access**: GUI buttons in Frame C (Video and Image tools) or run modules directly
- **Core Tools**:
  - **Synchronization**: `syncvid.py` (`C_B_r2_c3`) – Multi‑camera temporal alignment with flash detection
  - **Trimming**: `cutvideo.py` (`C_B_r4_c2`) – Interactive frame‑accurate cutting with batch support
  - **Lens Correction**: `vaila_distortvideo_gui.py` (`C_B_r4_c1`) – Radial distortion removal
- **Additional Tools**:
  - **Compression**: `compress_videos_h264.py`, `compress_videos_h265.py`, `compress_videos_h266.py` (`C_B_r2_c1`, `C_B_r2_c2`)
  - **Frame Extraction**: `extractpng.py` (`C_B_r1_c1`) – Export frames as PNG sequences
  - **Video Merging**: `merge_multivideos.py` (`C_B_r3_c3`) – Combine multiple video sources
  - **Resizing**: `resize_video.py` (`C_B_r4_c3`) – Change resolution while preserving aspect ratio
  - **Metadata**: `numberframes.py` (`C_B_r3_c2`) – Extract precise video metadata (FPS, duration, frame count)
  - **Duplicate Removal**: `rm_duplicateframes.py` – Clean repeated frames

### Stage III: Markerless 2D Pose Estimation

- **Goal**: Extract biological landmark coordinates from standard 2D video feeds.
- **Access**: GUI buttons in Frame B (Multimodal Analysis) or run modules directly
- **Core Tools**:
  - **MediaPipe + YOLO**: `markerless2d_mpyolo.py` (`B3_r3_c2`) – Combined detection and pose estimation
  - **MediaPipe Standalone**: `markerless_2d_analysis.py` (`B1_r1_c4`) – Full‑body pose (33 landmarks) with CPU/GPU options
  - **Hand Tracking**: `mphands.py` (`B3_r4_c3`) – 21 hand landmarks per hand
- **Additional Tools**:
  - **DeepLabCut Import**: `dlc2vaila.py` – Convert DLC outputs to vailá format
  - **YOLOv11 Tracking**: `yolov11track.py` (`B3_r4_c1`) – Multi-object tracking with re‑identification (BoT-SORT/ByteTrack)
  - **Angle Calculation**: `mpangles.py` (`B3_r4_c4`) – Joint angles from MediaPipe landmarks
  - **Live Tracking**: `markerless_live.py` (`B3_r4_c5`) – Real-time pose estimation from webcam

### Stage IV: Camera Calibration (DLT)

- **Goal**: Establish the mathematical relationship between 2D pixel space and 3D metric space.
- **Access**: GUI buttons in Frame C (Data Files tools) or run modules directly
- **Tools**:
  - **2D Calibration**: `dlt2d.py` (`C_A_r2_c1`) – 8‑parameter DLT for planar analysis
  - **3D Calibration**: `dlt3d.py` (`C_A_r3_c1`) – 11‑parameter DLT for volumetric reconstruction
  - **Camera Parameters**: `getcampardistortlens.py` – Extract intrinsic parameters

### Stage V: 3D Reconstruction

- **Goal**: Triangulate 2D coordinates from multiple views into a unified 3D reconstruction.
- **Access**: GUI buttons in Frame C (Data Files tools) or run modules directly
- **Tools**:
  - **Multi‑Camera Reconstruction**: `rec3d.py` (`C_A_r3_c2`) – Least‑squares triangulation
  - **Single‑DLT 2D Reconstruction**: `rec2d_one_dlt2d.py` (`C_A_r2_c2`) – Planar reconstruction
  - **Multi‑DLT 3D Reconstruction**: `rec3d_one_dlt3d.py` (`C_A_r3_c3`) – Per‑frame DLT parameters

### Stage VI: Visualization and Export

- **Goal**: Validate results through visualization and export standard biomechanics formats.
- **Access**: GUI buttons in Frame C (Visualization tools) or run modules directly
- **Core Tools**:
  - **3D Viewer**: `viewc3d.py` (`C_C_r2_c2`) – Interactive Open3D visualization with marker selection
  - **2D Plotting**: `vailaplot2d.py` (`C_C_r2_c1`) – Time series, scatter, and multi‑axis plots
  - **C3D Preview**: `showc3d.py` (`C_C_r1_c1`) – Quick C3D file inspection
  - **CSV Viewer**: `readcsv.py` (`C_C_r1_c2`) – Browse and inspect CSV data
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

- **Operating Systems**: Windows 10 or later, macOS 12 or later, Ubuntu 20.04 or later
- **Python**: 3.12.12 (automatically installed by `uv` during installation - no manual Python setup required)
- **Package Manager**: `uv` (automatically installed by installation scripts if not present)
- **External Dependencies**:
  - **FFmpeg**: Automatically installed by installation scripts (required for video processing)
  - **CUDA**: Optional, for GPU‑accelerated YOLO and MediaPipe processing (NVIDIA GPUs only)
  - **Git**: Required for cloning the repository (usually pre-installed)
- **Hardware**:
  - **Minimum**: 8 GB RAM
  - **Recommended**: 16+ GB RAM, NVIDIA GPU with CUDA support for faster processing of large video datasets

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
├── vaila.py                    # Main entry point (GUI launcher)
├── pyproject.toml              # Dependency specification (uv)
├── uv.lock                     # Locked dependency versions (uv)
│
├── install_vaila_win_uv.ps1    # Windows installation script (uv)
├── install_vaila_linux_uv.sh   # Linux installation script (uv)
├── install_vaila_mac_uv.sh     # macOS installation script (uv)
│
├── vaila/                      # Main package directory
│   ├── __init__.py
│   ├── markerless_2d_analysis.py      # MediaPipe 2D pose (CPU/GPU)
│   ├── markerless_2d_analysis_nvidia.py  # MediaPipe GPU acceleration
│   ├── markerless2d_mpyolo.py         # MediaPipe + YOLO combined
│   ├── yolov11track.py                # YOLOv11 multi-object tracking
│   ├── cutvideo.py                    # Video cutting tool
│   ├── numberframes.py               # Video metadata extraction
│   ├── syncvid.py                    # Video synchronization
│   ├── dlt2d.py, dlt3d.py            # DLT calibration
│   ├── rec2d.py, rec3d.py            # 2D/3D reconstruction
│   ├── viewc3d.py                    # 3D visualization
│   ├── imu_analysis.py               # IMU data analysis
│   ├── mocap_analysis.py             # Motion capture analysis
│   ├── forceplate_analysis.py        # Force plate analysis
│   ├── emg_labiocom.py               # EMG analysis
│   ├── models/                       # Trained models (YOLO, MediaPipe, etc.)
│   ├── help/                         # Module help documentation (HTML/MD)
│   │   ├── index.html, index.md
│   │   ├── analysis/                 # Analysis tool documentation
│   │   ├── tools/                    # Utility tool documentation
│   │   └── ml/                       # Machine learning documentation
│   └── ... (100+ additional modules)
│
├── docs/                       # Project documentation
│   ├── index.md                 # Main documentation index
│   ├── help.md                  # User help guide
│   ├── images/                  # Documentation images
│   ├── vaila_buttons/           # GUI button documentation
│   │   ├── README.md
│   │   ├── tools/                # Tool button docs
│   │   ├── ml-walkway/           # ML walkway docs
│   │   └── ... (button-specific docs)
│   └── api/                     # API reference documentation
│
└── tests/                      # Test suite with example data
    ├── markerless_2d_analysis/  # Test videos and configs
    ├── DLT3D_and_Rec3d/         # DLT test data
    ├── C3D_to_CSV_TOOLS/        # C3D conversion tests
    └── ... (additional test data)
```

**Key Directories:**
- **`vaila/`**: All Python modules and scripts
- **`vaila/help/`**: Built-in help documentation for each module (HTML and Markdown)
- **`docs/`**: Project-wide documentation, GUI button guides, and API reference
- **`tests/`**: Example datasets and test files for various modules

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

**With Administrator privileges:**
- Delete the installation folder: `C:\Program Files\vaila`
- Remove desktop shortcuts and Start Menu entries
- Remove Windows Terminal profile if created

**Without Administrator privileges:**
- Delete the installation folder: `C:\Users\<YourUser>\vaila`
- Remove desktop shortcuts if created

**Note:** The `uv` environment and Python installation remain on your system. To completely remove Python installed by `uv`, you may need to manually delete `%LOCALAPPDATA%\uv` (advanced users only).

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

We encourage creativity and innovation to enhance and expand the functionality of this toolbox. Fork the repository, experiment with new ideas, and create a branch for your changes. When you're ready, submit a pull request so we can review and potentially integrate your contributions.

---

## License

This project is licensed under the GNU Affero General Public License v3.0 (AGPL‑v3). The license ensures that any use of _vailá_, including network/server usage, maintains the freedom of the software and requires source‑code availability.

---

## References

1. Santiago, P. R. P. _et al._ "vailá – Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox". _arXiv_ 2410.07238 (2024).
2. Tahara, A. K., Chinaglia, A. G., Monteiro, R. L. M., et al. "Predicting walkway spatiotemporal parameters using a markerless, pixel‑based machine learning approach". _Brazilian Journal of Motor Behavior_ 19, 1 (2025).
3. Mochida, L. Y., Santiago, P. R. P., Lamb, M., Cesar, G. M. "Multimodal Motion Capture Toolbox for Enhanced Analysis of Intersegmental Coordination in Children with Cerebral Palsy and Typically Developing". _JOVE_ 206, e69604 (2025).
4. Nature Protocols Author Guidelines: https://www.nature.com/nprot/for-authors/protocols

---

## Mermaid Diagram of the Workflow

```mermaid
flowchart LR
    A["Stage I: Installation"] --> B["Stage II: Video Preprocessing"]
    B --> C["Stage III: 2D Pose Estimation"]
    C --> D["Stage IV: Calibration DLT"]
    D --> E["Stage V: 3D Reconstruction"]
    E --> F["Stage VI: Visualization and Export"]
```

---

_End of README_
