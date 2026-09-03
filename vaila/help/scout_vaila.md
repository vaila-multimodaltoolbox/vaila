# scout_vaila

## 📋 Module Information

- **Category:** Tools
- **File:** `vaila\scout_vaila.py`
- **Lines:** 2510
- **Size:** 95343 characters
- **Version:** 0.3.120
- **Author:** Paulo Roberto Pereira Santiago and Rafael Luiz Martins Montero
- **GUI Interface:** ✅ Yes

## 📖 Description


Project: vailá Multimodal Toolbox
Script: scout_vaila.py - Integrated Sports Scouting (Annotation + Analysis)

Author: Paulo Roberto Pereira Santiago and Rafael Luiz Martins Montero
Email: paulosantiago@usp.br and rafaell_mmonteiro@usp.br
GitHub: https://github.com/vaila-multimodaltoolbox/vaila
Creation Date: 12 August 2025
Update Date: 03 September 2026
Version: 0.3.120

Description:
    Integrated GUI to annotate sports events on a virtual soccer field and generate
    quick analyses (e.g., heatmaps). Inspired by manual scouting tools and designed
    to fit the vailá project style. No external field image is required; the field is
    drawn to scale using standard FIFA dimensions (105m x 68m).

Usage:
    GUI mode: click "Scout" inside the "Soccer Tools" launcher in the vailá
    main window (Frame B), or run with no flags to auto-locate/create the
    default TOML config:
        uv run vaila/scout_vaila.py
    or:
        python -m vaila.scout_vaila

    CLI mode: load a specific config non-interactively before the window
    opens (the annotation workflow itself stays interactive):
        uv run vaila/scout_vaila.py -c my_scout_config.toml


Requirements:
    - Python 3.x
    - tkinter (GUI)
    - matplotlib
    - seabo...

## 🔧 Main Functions

**Total functions found:** 20

- `read_toml_config`
- `write_toml_config`
- `write_toml_template`
- `draw_soccer_field`
- `run_scout`
- `to_row`
- `destroy`
- `start_timer`
- `pause_timer`
- `reset_timer`
- `clear_events`
- `save_csv`
- `load_csv`
- `draw_action_view`
- `draw_heatmap`
- `load_config`
- `save_config`
- `edit_config`
- `open_help`
- `open_shortcuts`




---

📅 **Generated automatically on:** 15/10/2025 08:04:44
🔗 **Part of vailá - Multimodal Toolbox**
🌐 [GitHub Repository](https://github.com/vaila-multimodaltoolbox/vaila)
