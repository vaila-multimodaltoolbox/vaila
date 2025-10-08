# mpangles

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\mpangles.py`
- **Linhas:** 2282
- **Tamanho:** 74359 caracteres
- **Versão:** 0.1.1
- **Autor:** Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


===============================================================================
mpangles.py
===============================================================================
Author: Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 31 March 2025
Update Date: 11 April 2025
Version: 0.1.1
Python Version: 3.12.9

Description:
------------
This script calculates absolute and relative angles from landmark coordinates
obtained from MediaPipe pose estimation. It processes CSV files containing
landmark data and generates new CSV files with computed angles.

Key Features:
-------------
1. Absolute Angles:
   - Calculates angles between segments and horizontal axis
   - Uses arctan2 for robust angle calculation

2. Relative Angles:
   - Computes angles between connected segments
   - Uses arctan2 for dot product angle calculation

3. Supported Angles:
    - Elbow angle (between upper arm and forearm)
    - Shoulder angle (betw...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `select_directory`
- `process_directory`
- `get_vector_landmark`
- `compute_midpoint`
- `compute_absolute_angle`
- `compute_relative_angle`
- `compute_knee_angle`
- `compute_hip_angle`
- `compute_ankle_angle`
- `compute_shoulder_angle`
- `compute_elbow_angle`
- `compute_neck_angle`
- `compute_wrist_angle`
- `process_absolute_angles`
- `process_angles`
- `draw_skeleton_and_angles`
- `process_video_with_visualization`
- `select_video_file`
- `select_csv_file`
- `run_mp_angles`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
