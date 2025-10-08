# markerless_2d_analysis_nvidia

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\markerless_2d_analysis_nvidia.py`
- **Linhas:** 594
- **Tamanho:** 22338 caracteres
- **Versão:** 0.3.0
- **Autor:** Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: markerless_2d_analysis_nvidia.py
Author: Prof. Dr. Paulo Santiago
Version: 0.3.0
Last Updated: 25 April 2025

Description:
This script performs batch processing of videos for 2D pose estimation using
MediaPipe's Pose model. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation. The default settings
prioritize the highest detection accuracy and tracking precision, which may
increase computational cost.

New Features:
- Run in NVIDIA GPU in Linux
- Configure NVIDIA CUDA and cuDNN in Linux first
- Default values for MediaPipe parameters are set to maximize detection and
  tracking accuracy:
    - `min_detection_confidence=1.0`
    - `min_tracking_confidenc...

## 🔧 Funções Principais

**Total de funções encontradas:** 7

- `get_pose_config`
- `apply_temporal_filter`
- `estimate_occluded_landmarks`
- `process_video`
- `process_videos_in_directory`
- `body`
- `apply`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
