# markerless2d_mpyolo

## 📋 Informações do Módulo

- **Categoria:** Ml
- **Arquivo:** `vaila\markerless2d_mpyolo.py`
- **Linhas:** 2300
- **Tamanho:** 84006 caracteres
- **Versão:** 0.1.0 - Enhanced
- **Autor:** Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: markerless2d_mpyolo.py
Author: Prof. Dr. Paulo Santiago
Version: 0.1.0 - Enhanced
Created: 18 February 2025
Last Updated: 20 July 2025

Description:
This script combines YOLOv12 for person detection/tracking with MediaPipe for pose estimation.

🔥 ENHANCED VERSION with key improvements:
- 4x bbox scaling for better MediaPipe accuracy
- Configurable safety margins (25% default) to prevent landmark explosion outside bbox
- Landmark validation (minimum 10/33 valid landmarks required)
- Enhanced visualization with pose quality indicators
- Real-time success rate monitoring

Key Features:
✅ YOLO person detection and tracking
✅ Enhanced MediaPipe pose estimation with scaling
✅ Safety margins to contain landmarks within person region
✅ Automatic coordinate conversion and validation
✅ Enhanced CSV output with quality metrics
✅ Improved visualization with pose confidence indicators

Usage:
python markerless2d_mpyolo.py


## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `get_color_palette`
- `get_color_for_id`
- `download_model`
- `initialize_csv`
- `save_detection_to_csv`
- `save_landmarks_to_csv`
- `get_parameters_dialog`
- `process_person_with_mediapipe_enhanced`
- `process_person_with_mediapipe`
- `save_person_data_to_csv`
- `process_yolo_tracking`
- `process_mediapipe_pose`
- `process_single_pose`
- `create_visualization_video`
- `save_pose_to_csv`
- `run_tracker_in_thread`
- `run_multithreaded_tracking`
- `save_tracking_data_to_csv`
- `process_video_enhanced`
- `save_enhanced_person_data`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
