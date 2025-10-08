# modifylabref

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\modifylabref.py`
- **Linhas:** 491
- **Tamanho:** 17284 caracteres
- **Versão:** 0.0.3
- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ❌ Não

## 📖 Descrição


Project: vailá Multimodal Toolbox
Script: modifylabref.py - Custom 3D Rotation Processing Toolkit

Author: Paulo Roberto Pereira Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 21 Sep 2024
Update Date: 17 Jul 2025
Version: 0.0.3

Description:
    This script is designed for rotating and transforming 3D motion capture data.
    It allows the application of predefined or custom rotation angles (in degrees) and
    rotation orders (e.g., 'xyz', 'zyx') to sets of 3D data points (e.g., from CSV files).

    Main Features:
    1. Predefined Rotations:
        - Options 'A', 'B', and 'C' apply standard rotation transformations:
          'A' applies a 180 degree rotation around the Z-axis.
          'B' applies a 90 degree clockwise rotation around the Z-axis.
          'C' applies a 90 degree counterclockwise rotation around the Z-axis.

    2. Custom Rotation:
        - Users can input custom angles in the format: [x, y, z]
     ...

## 🔧 Funções Principais

**Total de funções encontradas:** 9

- `rotdata`
- `modify_lab_coords`
- `parse_custom_rotation_input`
- `get_labcoord_angles`
- `detect_column_precision`
- `save_with_original_precision`
- `process_files`
- `run_modify_labref`
- `main`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
