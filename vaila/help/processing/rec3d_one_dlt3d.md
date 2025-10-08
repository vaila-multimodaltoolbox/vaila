# rec3d_one_dlt3d

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\rec3d_one_dlt3d.py`
- **Linhas:** 432
- **Tamanho:** 16550 caracteres
- **Versão:** 0.0.4
- **Autor:** Paulo Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


================================================================================
Script: rec3d_one_dlt3d.py
================================================================================

vailá - Multimodal Toolbox
© Paulo Santiago, Guilherme Cesar, Ligia Mochida, Bruno Bedo
https://github.com/vaila-multimodaltoolbox/vaila
Please see AUTHORS for contributors.

================================================================================
Author: Paulo Santiago
Version: 0.0.4
Created: August 02, 2025
Last Updated: August 02, 2025

Description:
    Optimized batch 3D reconstruction using the Direct Linear Transformation (DLT) method with multiple cameras.
    Each camera has a corresponding DLT3D parameter file (one set of 11 parameters per file) and a pixel coordinate CSV file.
    The pixel files are expected to use vailá's standard header:
      frame,p1_x,p1_y,p2_x,p2_y,...,pN_x,pN_y
    For each common frame found among the pixel files, the script reconstructs 3D points for all...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `rec3d_multicam`
- `save_rec3d_as_c3d`
- `run_rec3d_one_dlt3d`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
