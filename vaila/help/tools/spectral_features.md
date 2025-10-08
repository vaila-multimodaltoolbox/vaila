# spectral_features

## 📋 Informações do Módulo

- **Categoria:** Tools
- **Arquivo:** `vaila\spectral_features.py`
- **Linhas:** 174
- **Tamanho:** 6014 caracteres
- **Versão:** 1.1
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ❌ Não

## 📖 Descrição


Module: spectral_features.py
Description: Provides functions to calculate spectral features from power spectral density (PSD) data,
including total power, power frequency percentiles, power mode, spectral moments, centroid frequency,
frequency dispersion, energy content in specific frequency bands, and frequency quotient.

Author: Prof. Dr. Paulo R. P. Santiago
Version: 1.1
Date: 2024-11-13

Changelog:
- Version 1.1 (2024-11-13):
  - Added robust handling for empty frequency ranges.
  - Adjusted frequency range dynamically when out of bounds.
- Version 1.0 (2024-09-12):
  - Initial release with functions to compute various spectral features from PSD data.

Usage:
- Import the module and use the functions to compute spectral features:
  from spectral_features import *
  total_power_ml = total_power(freqs_ml, psd_ml)
  power_freq_50_ml = power_frequency_50(freqs_ml, psd_ml)
  # etc.


## 🔧 Funções Principais

**Total de funções encontradas:** 13

- `adjust_frequency_range`
- `total_power`
- `power_frequency_50`
- `power_frequency_95`
- `power_mode`
- `spectral_moment`
- `centroid_frequency`
- `frequency_dispersion`
- `energy_content`
- `energy_content_below_0_5`
- `energy_content_0_5_2`
- `energy_content_above_2`
- `frequency_quotient`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:53:50  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
