# markerless3d_analysis_v2

## 📋 Informações do Módulo

- **Categoria:** Ml
- **Arquivo:** `vaila\markerless3d_analysis_v2.py`
- **Linhas:** 1020
- **Tamanho:** 37296 caracteres
- **Versão:** 0.0.1
- **Autor:** Prof. Dr. Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


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

## 🔧 Funções Principais

**Total de funções encontradas:** 18

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

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
