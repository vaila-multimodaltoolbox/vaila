# vailá - Versatile Anarcho Integrated Multimodal Toolbox Help

## Overview

vailá is an open-source multimodal toolbox for human movement analysis. It integrates data from various sources – including IMU, MoCap, markerless tracking, GNSS/GPS, EMG, and more – enabling advanced and customizable analysis.

## GUI Button Documentation

All buttons in the vailá GUI (`vaila.py`) are documented in **[vaila_buttons/](vaila_buttons/README.md)**:

- **[File Manager Buttons](vaila_buttons/README.md#file-manager-frame-a)**: Rename, Import, Export, Copy, Move, Remove, Tree, Find, Transfer
- **[Multimodal Analysis Buttons](vaila_buttons/README.md#multimodal-analysis-frame-b)**: IMU, MoCap, Markerless 2D/3D, Vector Coding, EMG, Force Plate, GNSS/GPS, and more
- **[Tools Buttons](vaila_buttons/README.md#tools-frame-c)**: Data Files, Video Processing, Visualization tools

**Quick Links:**

- **[Markerless 2D Analysis](vaila_buttons/markerless-2d-button.md)** (B1_r1_c4) - Advanced pose estimation with MediaPipe and YOLOv11
- **[All Button Documentation](vaila_buttons/README.md)** - Complete list of all GUI buttons

## Key Features

- **Data Integration:** Supports multiple data types (IMU, MoCap, markerless, GNSS, EMG)
- **Data Processing & Analysis:** Feature extraction, advanced analysis, and 2D/3D visualization
- **Machine Learning:** Modules for training, validation, and prediction using ML models
- **File Management:** Organization, renaming, copying, and file movement
- **Video Processing:** Extraction of frames, compression (H.264, H.265, H.266), and video trimming

## Installation Instructions

### ⚡ Powered by *uv* (Recommended)

*vailá* now uses **[uv](https://github.com/astral-sh/uv)**, an extremely fast Python package installer. **uv is the recommended installation method** for all platforms due to its **10-100x faster installation** and **faster execution times** compared to Conda.

### Prerequisites

- **uv**: Will be automatically installed by the installation scripts, or install manually from [https://github.com/astral-sh/uv](https://github.com/astral-sh/uv)
- **FFmpeg**: Required for video processing functionalities (installed automatically on Windows)

### Steps

1. **Clone the repository:**

   ```bash
   git clone https://github.com/vaila-multimodaltoolbox/vaila.git
   cd vaila
   ```

2. **Set up the environment:**

   **Windows (Recommended):**

   ```powershell
   .\install_vaila_win_uv.ps1
   ```

   **Linux and macOS (Using uv):**

   ```bash
   # Install uv
   curl -LsSf https://astral.sh/uv/install.sh | sh
   
   # Sync dependencies
   uv sync
   ```

   **Legacy Conda Method:**
   - **Linux:** Run `./install_vaila_linux.sh`
   - **macOS:** Run `./install_vaila_mac.sh`
   - **Windows:** Run `.\install_vaila_win.ps1` (Conda-based, slower)

3. **Run vailá:**

   **Using uv (Recommended):**

   ```bash
   uv run vaila.py
   ```

   **Using Conda (Legacy):**

   ```bash
   conda activate vaila
   python vaila.py
   ```

## Modules and Tools

The vailá toolbox comprises the following modules:

- **IMU Analysis**
- **MoCap Analysis** (Cluster, Full Body)
- **Markerless Analysis** (2D/3D)
- **Force Plate Analysis**
- **GNSS/GPS Analysis**
- **EEG/EMG Analysis**
- **ML Walkway:**
  - Model Training
  - Model Validation
  - Prediction with Pre-trained Models
- **File Management**
- **Video Processing:**
  - DrawBoxe - Draw boxes and polygons on videos
- **Visualization**

## How to Use

After setting up the environment, run vailá using:

**Using uv (Recommended):**

```bash
uv run vaila.py
```

**Using Conda (Legacy):**

```bash
conda activate vaila
python vaila.py
```

The graphical interface allows you to select the desired module. For example, when selecting "ML Walkway," a window with options for model training, validation, or prediction will open.

## Related Documentation

- **[GUI Button Documentation](vaila_buttons/README.md)** - Complete documentation for all GUI buttons
- **[Script Help Index](../vaila/help/index.html)** - Complete documentation for all Python modules and scripts
- **[Script Help Files](../vaila/help/README.md)** - Help for individual Python scripts
- **[Main Documentation](index.md)** - Overview and module documentation

### Video Processing Tools

- **[DrawBoxe](../vaila/help/tools/drawboxe.html)** - Draw boxes and polygons on videos with frame interval support

## Contributing

Contributions are welcome! If you encounter issues or have suggestions, please submit a pull request or open an issue on GitHub.

## License

This project is licensed under the **GNU Affero General Public License v3.0**. See the `LICENSE` file for more details.

## How to Cite

If vailá is useful for your research, please cite:

```bibtex
@misc{vaila2024,
  title={vailá - Versatile Anarcho Integrated Liberation Ánalysis in Multimodal Toolbox},
  author={Paulo R. P. Santiago and Abel G. Chinaglia and others},
  year={2024},
  url={https://github.com/vaila-multimodaltoolbox/vaila}
}
```

---

© 2024 vailá - Multimodal Toolbox. All rights reserved.
