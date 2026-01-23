# filemanager.py

## 📋 Module Information

- **Category:** Tools  
- **File:** `vaila/filemanager.py`  
- **Version:** 0.1.1  
- **Author:** Paulo Roberto Pereira Santiago  
- **GUI Interface:** ✅ Yes  
- **License:** AGPLv3  

## 📖 Description
GUI toolkit for file operations (copy, move, remove, import/export, rename, tree, find, transfer). Provides terminal feedback for each action and supports pattern/extension filters with safe guards against dangerous paths. Now includes a built-in cross-platform SSH Transfer tool for bidirectional file transfer (Upload/Download).

## 🚀 Key Updates
- Terminal feedback for every action (Copy/Move/Remove/Rename/Find/Tree/Transfer/Import menu).
- Copy: patterns empty → copy all files; extension empty → all extensions; collision-safe naming.
- Normalize names: accent/lower/underscore cleanup; topdown=False to avoid path breaks.
- Safety: forbidden patterns for destructive remove.
- **Transfer GUI**: Integrated SSH transfer tool with Upload/Download modes, supporting `rsync`, `scp`, and `paramiko`.

## 🎛️ Controls & Prompts
- Source/destination directory pickers (Tk dialogs) per operation.
- Copy/Move: ask extension (blank = all), patterns (blank = all); creates timestamped output dirs (`vaila_copy_...`, `vaila_move_...`); collision suffix `_1`, `_2`...
- Rename: text replace with extension filter.
- Tree/Find: saves results with timestamp in selected destination.
- Tree/Find: saves results with timestamp in selected destination.
- Transfer: Opens "File Transfer Configuration" window. Select Mode (Upload/Download), Local/Remote paths, and SSH credentials. Supports password (via `paramiko`) or key-based auth.

## 🛠 Main Functions
- `copy_file` / `process_copy`  
- `move_file` / `process_move`  
- `export_file`, `remove_file`, `import_file`, `rename_files`, `tree_file`, `find_file`, `transfer_file`  

## ⚠️ Safety Notes
- Remove blocks dangerous patterns/system paths.  
- Normalize uses topdown=False to rename children before parents.  

## 🔗 Project
📅 **Last Updated:** Jan 2026  
🔗 **Part of vailá - Multimodal Toolbox**  
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
