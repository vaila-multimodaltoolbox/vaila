# dlt3d

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\dlt3d.py`
- **Linhas:** 271
- **Tamanho:** 10271 caracteres
- **Versão:** 0.0.3
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Script: dlt3d.py
================================================================================
vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Roberto Pereira Santiago
Version: 0.0.3
Create: 24 February, 2025
Last Updated: 02 August, 2025

Description:
    This script calculates the Direct Linear Transformation (DLT) parameters for 3D coordinate transformations.
    It uses pixel coordinates from video calibration data and corresponding real-world 3D coordinates to compute the 11
    DLT parameters for each frame (or uses a single row of real-world coordinates for all frames).

    New Features:
      - Generates a REF3D template (with _x, _y, _z columns) from the pixel file.
      - ...

## 🔧 Funções Principais

**Total de funções encontradas:** 6

- `read_pixel_file`
- `read_ref3d_file`
- `calculate_dlt3d_params`
- `process_files`
- `save_dlt_parameters`
- `main`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
