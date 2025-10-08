# markerless3d_analysis_v2

## 📋 Module Information

- **Category:** Ml
- **File:** `vaila/markerless3d_analysis_v2.py`
- **Lines:** 1020
- **Size:** 37296 characters
- **Version:** 0.0.1
- **Author:** Prof. Dr. Paulo Santiago
- **GUI Interface:** ✅ Yes

## 📖 Description


Script: markerless_3D_analysis.py
Author: Prof. Dr. Paulo Santiago
Version: 0.0.1
Create:
Last Updated: April 2025

Description:
Versão aprimorada do script markerless_3D_analysis.py que incorpora detecção prévia
com YOLO para melhorar a precisão do MediaPipe na estimativa de pose, especialmente
em casos de oclusão no plano sagital.

Melhorias:
- Detecção prévia de pessoas usando YOLO antes do MediaPipe
- Processamento em regiões de interesse (ROI) para aumentar precisão
- Rastreamento de múltiplas pessoas ao longo dos frames
- Melhoria na detecção em plano sagital e casos de oclusão parcial
- Mantém compatibilidade com o formato de saída original

Usage:
- Run the script to open a graphical interface for selecting the input directory
  containing video files (.mp4, .avi, .mov), the output directory, and for
  specifying the MediaPipe configuration parameters.
- The script processes each video, generating an output video with overlaid pose
  landmarks, and CSV files containing both no...

## 🔧 Main Functions

**Total functions found:** 18

- `get_pose_config`
- `download_or_load_yolo_model`
- `detect_persons_with_yolo`
- `process_person_with_mediapipe`
- `process_video`
- `compute_iou`
- `apply_kalman_filter`
- `apply_savgol_filter`
- `estimate_missing_landmarks`
- `apply_anatomical_constraints`
- `landmarks_to_mp_format`
- `process_videos_in_directory`
- `body`
- `apply`
- `register`
- `deregister`
- `get_best_match`
- `update`




---

📅 **Generated automatically on:** 08/10/2025 14:24:24
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
