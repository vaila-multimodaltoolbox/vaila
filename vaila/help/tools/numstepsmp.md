# numstepsmp

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila/numstepsmp.py`
- **Linhas:** 1473
- **Tamanho:** 55909 caracteres

- **Autor:** Paulo Roberto Pereira Santiago
- **Interface Gráfica:** ✅ Sim

## 📖 Descrição


numstepsmp.py

Description:
    Opens a dialog to select a CSV file of foot coordinates
    and calculates the number of steps based on foot position using
    MediaPipe data. Includes Butterworth filtering and advanced metrics using
    multiple markers (ankle, heel, toe).

Author:
    Paulo Roberto Pereira Santiago

Created:
    14 May 2025
Updated:
    16 May 2025

Usage:
    python numstepsmp.py

Dependencies:
    - pandas
    - numpy
    - scipy
    - tkinter (GUI for file selection)
    - matplotlib (optional, for visualization)


## 🔧 Funções Principais

**Total de funções encontradas:** 16

- `butterworth_filter`
- `filter_signals`
- `calculate_feet_metrics`
- `count_steps_original`
- `count_steps_basic`
- `count_steps_velocity`
- `count_steps_sliding_window`
- `count_steps_mean_y`
- `count_steps_z_depth`
- `detect_foot_strikes_heel_z`
- `detect_foot_strikes`
- `count_steps`
- `export_results`
- `run_numsteps`
- `extract_gait_features`
- `run_numsteps_gui`




---

📅 **Gerado automaticamente em:** 08/10/2025 14:00:12  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
