# markerless_live

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\markerless_live.py`
- **Linhas:** 1486
- **Tamanho:** 56506 caracteres
- **Versão:** 0.0.2
- **Autor:** Moser José (https://moserjose.com/),  Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: markerless_live.py
Author: Moser José (https://moserjose.com/),  Prof. Dr. Paulo Santiago
Version: 0.0.2
Created: April 9, 2025
Last Updated: April 11, 2025

Description:
This script performs real-time pose estimation and angle calculation using either YOLO or
MediaPipe's Pose model. It provides a graphical interface for selecting the pose detection
engine and configuring its parameters, allowing users to analyze movement in real-time
through their webcam.

Features:
- Dual engine support:
    - YOLO (better for multiple people detection)
    - MediaPipe (faster, optimized for single person)
- Real-time angle calculations for various body joints
- Configurable parameters for each engine:
    - YOLO: confidence threshold, model selection
    - MediaPipe: model complexity, detection confidence
- Visual feedback:
    - Skeleton overlay
    - Joint angles display
    - Person detection bounding box
- Data export:
    - CSV files with angle measurements
    - Angle plots over time
...

## 🔧 Funções Principais

**Total de funções encontradas:** 12

- `list_available_cameras`
- `download_model`
- `run_markerless_live`
- `calculate_angle`
- `adapt_to_keypoint_structure`
- `process_keypoints`
- `process_frame`
- `get_model_path`
- `detect_yolo_model_type`
- `process_frame`
- `save_data`
- `run`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
