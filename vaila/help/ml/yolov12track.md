# yolov12track

## 📋 Informações do Módulo

- **Categoria:** Ml
- **Arquivo:** `vaila\yolov12track.py`
- **Linhas:** 1383
- **Tamanho:** 50161 caracteres
- **Versão:** 0.0.3
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Project: vailá
Script: yolov12track.py

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 18 February 2025
Update Date: 02 Augst 2025
Version: 0.0.3

Description:
    This script performs object detection and tracking on video files using the YOLO model v12.
    It integrates multiple features, including:
      - Object detection and tracking using the Ultralytics YOLO library.
      - A graphical interface (Tkinter) for dynamic parameter configuration.
      - Video processing with OpenCV, including drawing bounding boxes and overlaying tracking data.
      - Generation of CSV files containing frame-by-frame tracking information per tracker ID.
      - Video conversion to more compatible formats using FFmpeg.

Usage:
    Run the script from the command line by passing the path to a video file as an argument:
            python yolov12track.py

Requirements:
    - Python 3.x
    - OpenCV
    - PyT...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `initialize_csv`
- `update_csv`
- `get_hardware_info`
- `detect_optimal_device`
- `validate_device_choice`
- `standardize_filename`
- `process_video`
- `get_color_for_id_improved`
- `get_color_palette`
- `get_color_for_id`
- `create_combined_person_csv`
- `run_yolov12track`
- `body`
- `show_help`
- `hide_help`
- `validate`
- `apply`
- `body`
- `browse_custom_model`
- `on_tab_changed`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
