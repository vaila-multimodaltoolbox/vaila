# filter_utils

## 📋 Informações do Módulo

- **Categoria:** Processing
- **Arquivo:** `vaila\filter_utils.py`
- **Linhas:** 110
- **Tamanho:** 3926 caracteres
- **Versão:** 1.1
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ❌ Não

## 📖 Descrição


Module: filter_utils.py
Description: This module provides a unified and flexible Butterworth filter function for low-pass and band-pass filtering of signals. The function supports edge effect mitigation through optional signal padding and uses second-order sections (SOS) for improved numerical stability.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-09-12

Changelog:
- Version 1.1 (2024-09-12):
  - Modified `butter_filter` to handle multidimensional data.
  - Adjusted padding length dynamically based on data length.
  - Fixed issues causing errors when data length is less than padding length.

Usage Example:
- Low-pass filter:
  `filtered_data_low = butter_filter(data, fs=1000, filter_type='low', cutoff=10, order=4)`

- Band-pass filter:
  `filtered_data_band = butter_filter(data, fs=1000, filter_type='band', lowcut=5, highcut=15, order=4)`


## 🔧 Funções Principais

**Total de funções encontradas:** 1

- `butter_filter`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
