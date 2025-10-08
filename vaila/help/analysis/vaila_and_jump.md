# vaila_and_jump

## 📋 Informações do Módulo

- **Categoria:** Analysis
- **Arquivo:** `vaila\vaila_and_jump.py`
- **Linhas:** 3310
- **Tamanho:** 120960 caracteres
- **Versão:** 0.1.0
- **Autor:** Prof. Paulo R. P. Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


===============================================================================
vaila_and_jump.py
===============================================================================
Author: Prof. Paulo R. P. Santiago
Email: paulosantiago@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 24 Oct 2024
Update Date: 14 Aug 2025
Version: 0.1.0
Python Version: 3.12.11

Description:
------------
This script processes jump data from multiple .csv files in a specified directory,
performing biomechanical calculations based on either the time of flight or the
jump height. The results are saved in a new output directory with a timestamp
for each processed file.

For MediaPipe data, the script automatically inverts y-coordinates (1.0 - y) to
transform from screen coordinates (where y increases downward) to biomechanical
coordinates (where y increases upward). This allows proper visualization and
analysis of the jumping motion.

Features:
---------
- Supports two calculation...

## 🔧 Funções Principais

**Total de funções encontradas:** 20

- `calculate_force`
- `calculate_jump_height`
- `calculate_power`
- `calculate_velocity`
- `calculate_kinetic_energy`
- `calculate_potential_energy`
- `calculate_average_power`
- `calculate_liftoff_force`
- `calculate_time_of_flight`
- `calculate_baseline`
- `identify_jump_phases`
- `generate_jump_plots`
- `plot_jump_phases_analysis`
- `plot_jump_cg_feet_analysis`
- `generate_html_report`
- `process_mediapipe_data`
- `process_all_mediapipe_files`
- `process_jump_data`
- `calc_fator_convert_mediapipe`
- `calc_fator_convert_mediapipe_simple`




---

📅 **Gerado automaticamente em:** 08/10/2025 10:07:00  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
