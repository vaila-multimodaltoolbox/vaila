# rec2d_one_dlt2d

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila/rec2d_one_dlt2d.py`
- **Linhas:** 211
- **Tamanho:** 7327 caracteres
- **Versão:** 0.0.3
- **Autor:** Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Script: rec2d_one_dlt2d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/paulopreto/vaila-multimodaltoolbox
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.3
Created: August 9, 2024
Last Updated: August 02, 2025

Description:
    Optimized batch processing of 2D coordinates reconstruction using corresponding DLT parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 2D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    Uses DLT2D parameters that can vary per frame.

    Optimizations:
    - Pre-allocated NumPy array...

## 🔧 Funções Principais

**Total de funções encontradas:** 4

- `read_coordinates`
- `rec2d`
- `process_files_in_directory`
- `run_rec2d_one_dlt2d`




---

📅 **Gerado automaticamente em:** 08/10/2025 14:00:12  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
