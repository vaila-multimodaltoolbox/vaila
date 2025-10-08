# mphands

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\mphands.py`
- **Linhas:** 319
- **Tamanho:** 11909 caracteres


- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


mphands.py
Created by: Flávia Pessoni Faleiros Macêdo & Paulo Roberto Pereira Santiago
date: 01/01/2025
updated: 11/02/2025

Description:
This script uses the MediaPipe Hand Landmarker in video mode to detect hand landmarks
from a user-selected video. It processes the entire video offline, saves the processed
video with drawn landmarks, and outputs the landmark data into a CSV file.

MediaPipe Hands for Vailá (Video Mode Offline Analysis)
---------------------------------------------------------

This script uses the MediaPipe Hand Landmarker in video mode to detect hand landmarks
from a user-selected video. It processes the entire video offline, saves the processed
video with drawn landmarks, and outputs the landmark data into a CSV file.

Requirements:
- Python 3.x
- OpenCV (pip install opencv-python)
- MediaPipe (pip install mediapipe)
- requests (pip install requests)
- Tkinter (usually bundled with Python)

The "hand_landmarker.task" model will be downloaded to the project's "mod...

## 🔧 Funções Principais

**Total de funções encontradas:** 5

- `download_model_if_needed`
- `get_landmark_color`
- `draw_hand_landmarks`
- `select_video_file`
- `run_mphands`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
