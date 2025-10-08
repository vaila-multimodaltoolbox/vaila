# vaila_lensdistortvideo

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila/vaila_lensdistortvideo.py`
- **Linhas:** 349
- **Tamanho:** 11795 caracteres
- **Versão:** 0.0.1
- **Autor:** Prof. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


===============================================================================
vaila_lensdistortvideo.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Date: 20 December 2024
Version: 0.0.1
Python Version: 3.12.8
===============================================================================

Camera Calibration Parameters and Their Meanings
=================================================

This script processes videos by applying lens distortion correction based on
intrinsic camera parameters and distortion coefficients. It also demonstrates
how to calculate these parameters using field of view (FOV) and resolution.

Intrinsic Camera Parameters:
-----------------------------
1. fx, fy (Focal Length):
   - Represent the focal length of the lens in pixels along the x-axis (fx) and y-axis (fy).
   - Larger values indicate a narrower field of view.
   - Calculated using the formula:
     fx = (width / 2) / tan(horizonta...

## 🔧 Funções Principais

**Total de funções encontradas:** 5

- `load_distortion_parameters`
- `process_video`
- `select_directory`
- `select_file`
- `run_distortvideo`




---

📅 **Gerado automaticamente em:** 08/10/2025 14:00:12  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
