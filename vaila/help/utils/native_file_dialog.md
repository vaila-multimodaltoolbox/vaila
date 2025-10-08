# native_file_dialog

## 📋 Informações do Módulo

- **Categoria:** Utils
- **Arquivo:** `vaila\native_file_dialog.py`
- **Linhas:** 203
- **Tamanho:** 7099 caracteres
- **Versão:** 0.1.0
- **Autor:** Prof. Dr. Paulo R. P. Santiago
- **Interface Gráfica:** ❌ Não

## 📖 Descrição


================================================================================
Native File Dialog Module - native_file_dialog.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 16 January 2025
Version: 0.1.0
Python Version: 3.12.11

Description:
------------
This module provides native file dialogs that work without conflicts with Pygame.
It uses system-native dialogs via subprocess to avoid event loop conflicts.

Usage:
------
from native_file_dialog import open_native_file_dialog, open_yes_no_dialog

# Open file dialog
file_path = open_native_file_dialog(
    title="Select File",
    file_types=[("*.csv", "CSV Files"), ("*.txt", "Text Files")]
)

# Open yes/no dialog
result = open_yes_no_dialog(
    title="Question",
    message="Do you want to continue?"
)

===================================================================...

## 🔧 Funções Principais

**Total de funções encontradas:** 3

- `open_native_file_dialog`
- `open_yes_no_dialog`
- `open_save_file_dialog`




---

📅 **Gerado automaticamente em:** 08/10/2025 09:18:44  
🔗 **Parte do vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
