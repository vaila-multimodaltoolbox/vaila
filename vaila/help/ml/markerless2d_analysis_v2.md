# markerless2d_analysis_v2

## 📋 Informações do Módulo

- **Categoria:** Ml
- **Arquivo:** `vaila\markerless2d_analysis_v2.py`
- **Linhas:** 797
- **Tamanho:** 27887 caracteres
- **Versão:** 0.2.0
- **Autor:** Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: markerless_2D_analysis_v2.py
Author: Prof. Dr. Paulo Santiago
Version: 0.2.0
Created: 01 December 2024
Updated: 09 June 2025

Description:
Version 0.2.0 of the markerless_2D_analysis.py script that corrects problems with detection
using YOLO and MediaPipe, especially for use on CPU.

Main improvements:
- Correction of MediaPipe processing within bounding boxes
- Optimization for CPU with better performance
- Better tracking of people
- More efficient and robust processing

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.

Requirements:
- Python 3.12.11
- OpenCV (`pip install opencv-python`)
- MediaPipe (`pip install mediapipe`)
- Ultralytics (`pip install ultralytics`)
- Tkinter (usually included with Python installations)


## 🔧 Funções Principais

**Total de funções encontradas:** 12

- `get_hardware_info`
- `get_pose_config`
- `download_or_load_yolo_model`
- `detect_persons_with_yolo`
- `process_frame_with_mediapipe`
- `apply_temporal_filter`
- `apply_kalman_filter`
- `apply_savgol_filter`
- `process_video`
- `process_videos_in_directory`
- `body`
- `apply`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
