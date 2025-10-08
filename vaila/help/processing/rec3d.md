# rec3d

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\rec3d.py`
- **Linhas:** 296
- **Tamanho:** 11090 caracteres
- **Versão:** 0.0.2
- **Autor:** Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Script: rec3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.2
Created: August 03, 2025
Last Updated: August 03, 2025

Description:
    Optimized batch processing of 3D coordinates reconstruction using corresponding DLT3D parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 3D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT3D parameters that can vary per frame (11 parameters per frame).

    Optimizations:
    - Pre-allocat...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `rec3d_multicam`
- `process_files_in_directory`
- `run_rec3d`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
