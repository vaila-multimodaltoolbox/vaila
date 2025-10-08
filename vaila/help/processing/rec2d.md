# rec2d

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\rec2d.py`
- **Linhas:** 218
- **Tamanho:** 7592 caracteres
- **Versão:** 0.0.2
- **Autor:** Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Script: rec2d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.2
Created: August 9, 2024
Last Updated: August 02, 2025

Description:
    Optimized batch processing of 2D coordinates reconstruction using corresponding DLT parameters for each frame.
    Processes multiple CSV files containing pixel coordinates and reconstructs them to 2D real-world coordinates.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
      ...
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
      ...
     ...

## 🔧 Funções Principais

**Total de funções encontradas:** 4

- `read_coordinates`
- `rec2d`
- `process_files_in_directory`
- `run_rec2d`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
