# showc3d

## 📋 Informações do Módulo

- **Categoria:** Visualization
- **Arquivo:** `vaila\showc3d.py`
- **Linhas:** 245
- **Tamanho:** 7994 caracteres

- **Autor:** Prof. Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


Script: showc3d.py
Author: Prof. Paulo Roberto Pereira Santiago
Date: 29/07/2024
Updated: 07/02/2025

Description:
------------
This script visualizes marker data from a C3D file using Matplotlib.
Marker positions are converted from millimeters to meters.
The user is prompted to select which markers to display.
A Matplotlib 3D scatter plot is used to animate the data,
complete with a slider to choose frames and a play/pause button.
The FPS from the C3D file is used to ensure correct playback speed.

Usage:
------
1. Ensure you have installed:
   - ezc3d (pip install ezc3d)
   - numpy
   - matplotlib (pip install matplotlib)
   - tkinter (usually included with Python)
2. Run the script and select a C3D file and markers to display.
3. Use the slider or Play/Pause button to control the animation.


## 🔧 Funções Principais

**Total de funções encontradas:** 11

- `load_c3d_file`
- `select_markers`
- `draw_cartesian_axes`
- `main`
- `show_c3d`
- `select_all`
- `unselect_all`
- `on_select`
- `update_frame`
- `timer_callback`
- `play_pause`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
