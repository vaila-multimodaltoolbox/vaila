# mocap_analysis

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\mocap_analysis.py`
- **Linhas:** 403
- **Tamanho:** 14857 caracteres
- **Versão:** 0.5.1
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Project: vailá Multimodal Toolbox
Script: mocap_analysis.py - Mocap Analysis

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 29 July 2024
Update Date: 27 July 2025
Version: 0.5.1

Description:
This script performs batch processing of videos for 3D pose estimation using
Mocap data. It processes videos from a specified input directory,
overlays pose landmarks on each video frame, and exports both normalized and
pixel-based landmark coordinates to CSV files.

The user can configure key MediaPipe parameters via a graphical interface,
including detection confidence, tracking confidence, model complexity, and
whether to enable segmentation and smooth segmentation.

Features:
- Added temporal filtering to smooth landmark movements.
- Added estimation of occluded landmarks based on anatomical constraints.
- Added log file with video metadata and processing information.

Usage:
- Run the script to open ...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `save_results_to_csv`
- `read_anatomical_csv`
- `analyze_mocap_fullbody_data`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
