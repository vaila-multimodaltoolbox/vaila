# backup_markerless

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\backup_markerless.py`
- **Linhas:** 2299
- **Tamanho:** 94365 caracteres
- **Versão:** 0.5.0
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Project: vailá Multimodal Toolbox
Script: markerless_2D_analysis.py - Markerless 2D Analysis with Video Resize

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2025
Update Date: 29 July 2025
Version: 0.5.0

Description:
This script performs batch processing of videos for 2D pose estimation using
MediaPipe's Pose model. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

NEW: Integrated video resize functionality to improve pose detection for:
- Small/distant subjects in videos
- Low resolution videos
- Better landmark accuracy through upscaling

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation. The default set...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

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
- `process_video`
- `process_videos_in_directory`
- `pad_signal`
- `body`
- `create_default_toml_template`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:43  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
