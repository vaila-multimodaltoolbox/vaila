# markerless_2d_analysis

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\markerless_2d_analysis.py`
- **Linhas:** 3171
- **Tamanho:** 126209 caracteres
- **Versão:** 0.6.0
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Project: vailá Multimodal Toolbox
Script: markerless_2D_analysis.py - Markerless 2D Analysis with Video Resize

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 10 August 2025
Version: 0.6.0

Description:
This script performs batch processing of videos for 2D pose estimation using
MediaPipe's Pose model. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files. The script also generates a
video with the landmarks overlaid on the original frames.

NEW: Integrated video resize functionality to improve pose detection for:
- Small/distant subjects in videos
- Low resolution videos
- Better landmark accuracy through upscaling

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `butter_filter`
- `savgol_smooth`
- `lowess_smooth`
- `spline_smooth`
- `kalman_smooth`
- `arima_smooth`
- `get_default_config`
- `save_config_to_toml`
- `load_config_from_toml`
- `get_config_filepath`
- `get_pose_config`
- `resize_video_opencv`
- `convert_coordinates_to_original`
- `apply_interpolation_and_smoothing`
- `apply_temporal_filter`
- `estimate_occluded_landmarks`
- `pad_signal`
- `is_linux_system`
- `get_system_memory_info`
- `should_use_batch_processing`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
