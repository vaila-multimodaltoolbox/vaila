# AGENTS.md

This file provides guidance to **AI Agents** (Antigravity, Cursor, Claude Code, Windsurf, etc.) and terminal tools (like warp.dev) when working with code in this repository.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) is an open-source Python 3.12 multimodal toolbox for biomechanical data analysis. It integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS, and other sensor data through a Tkinter-based GUI. Licensed under AGPLv3.

## Build & Run Commands

```bash
# Run the application (recommended)
uv run vaila.py

# Install dependencies
uv sync                    # CPU-only
uv sync --extra gpu        # With GPU support

# Lint and format
uv run ruff check vaila/           # Lint
uv run ruff check vaila/ --fix     # Lint with auto-fix
uv run ruff format vaila/          # Format

# Type checking
uv run ty check vaila/

# Run a single module standalone (some modules support CLI)
uv run vaila/interp_smooth_split.py -i /path/to/csv_dir -c smooth_config.toml

# Run automated tests
uv run pytest tests/                           # Run all tests
uv run pytest tests/test_vaila_and_jump.py -v   # Run jump specific tests
uv run pytest tests/test_tugturn.py -v          # Run TUG specific tests
uv run pytest tests/test_dlt_rec.py -v          # Run DLT/Rec math tests
uv run pytest tests/test_dlt_rec_integration.py -v # Run DLT/Rec pipeline tests

# Install git hooks (pre-commit blocks files >24MB)
bash install-hooks.sh
```

The project uses `pytest` for automated testing.

- `tests/test_vaila_and_jump.py` — Unit tests for biomechanical calculations.
- `tests/test_vaila_and_jump_integration.py` — Integration tests for full analysis pipelines using sample data.
- The `tests/vaila_and_jump/` directory contains the sample data (CSV, TOML) used by these tests.

**Milestone (02 March 2026):** Refactored `vaila_and_jump.py` (v0.1.3), `vaila/tugturn.py`, and the DLT/Reconstruction suite (`dlt2d.py`, `dlt3d.py`, `rec2d_one_dlt2d.py`, `rec3d.py`, `rec3d_one_dlt3d.py`). Fixed all Ruff/Ty lint and type errors, added CLI/headless support, and established a comprehensive automated test suite across `tests/`.

## Repo structure

```
vaila/                 ← root
├── vaila.py           ← Main Tkinter GUI entry point
├── vaila/             ← All analysis modules (package)
├── tests/             ← pytest test suite
├── docs/              ← Documentation
├── .claude/
│   ├── agents/        ← Specialized agent roles (biomechanics, GUI, video, tests)
│   ├── skills/        ← Step-by-step skills (new module, port MATLAB)
│   └── commands/      ← Slash-command specs (/check, /new-module)
├── .cursor/rules/     ← Cursor IDE rules
├── pyproject.toml     ← Default (CPU)
├── pyproject_*.toml   ← Platform-specific templates
└── uv.lock
```

## Platform-Specific Configuration

The project uses a **template-based pyproject.toml system** for hardware-specific dependencies. Before creating a venv, the correct template must be copied to `pyproject.toml`:

- `pyproject_win_cuda12.toml` — Windows NVIDIA CUDA 12.1
- `pyproject_linux_cuda12.toml` — Linux NVIDIA CUDA 12.8
- `pyproject_macos.toml` — macOS Metal/MPS (Apple Silicon)
- `pyproject_universal_cpu.toml` — CPU-only fallback

The install scripts (`install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`) handle this automatically. For manual setup, copy the template **before** running `uv python pin` / `uv venv`.

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` is the main entry point. It defines the `Vaila(tk.Tk)` class which builds the entire GUI. The interface is organized into three frames:

- **Frame A (File Manager):** Rename, import, export, copy, move, remove, tree, find, transfer
- **Frame B (Multimodal Analysis):** IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS, etc.
- **Frame C (Tools):** Data Files (CSV editing, C3D conversion, DLT reconstruction), Video/Image processing, Visualization

Each GUI button dispatches to a function in `vaila/` using **lazy imports** — modules are imported inside handler methods to avoid loading the entire dependency graph at startup. Two dispatch patterns are used:

1. **Direct import + call:** `from vaila import module; module.run_function()` — runs in the same process
2. **Subprocess launch via `run_vaila_module()`:** Launches `python -m vaila.module_name` in a separate process — used when Tkinter conflicts could occur (e.g., modules that create their own Tk root)

### Package Structure (`vaila/`)

The `vaila/` package contains ~100 self-contained analysis modules. Each module typically:

- Has a `run_*()` or `analyze_*()` entry function called from the GUI
- Uses Tkinter `filedialog` to prompt users for input directories/files
- Reads CSV/C3D data using `pandas`/`numpy`/`ezc3d`
- Processes data and writes results (CSV + PNG plots) to timestamped output subdirectories

Key shared modules:

- `data_processing.py` — CSV/C3D reading with auto-header detection
- `filtering.py` / `filter_utils.py` — Butterworth and FIR filter implementations
- `common_utils.py` — Header detection and data reshaping for CSV files
- `dialogsuser.py` / `dialogsuser_cluster.py` — Reusable Tkinter input dialogs for sample rate, file type
- `filemanager.py` — All file management operations (rename, copy, move, transfer via SSH)
- `hardware_manager.py` — GPU/CPU detection, TensorRT auto-export for YOLO models
- `interp_smooth_split.py` — Interpolation, smoothing, and splitting pipeline (supports both GUI and CLI modes); configured via `smooth_config.toml`

### Import Conventions

Modules use **relative imports** when imported as part of the package (`from .readcsv import ...`) with a `try/except` fallback to absolute imports for standalone execution (`from readcsv import ...`). This dual-import pattern is common throughout the codebase.

### Ruff Configuration

Ruff is configured in `pyproject.toml` with:

- Target: Python 3.12, line length 100
- Enabled rule sets: E, W, F, I, N, NPY, UP, B, C4, SIM
- Ignored: `E501` (line length handled by formatter), `N806`/`N803` (scientific code uses uppercase variable names like X, Y, F)
- `__init__.py` files ignore `F401` (unused imports are intentional re-exports)

## Conventions

- **Python version:** 3.12 (strictly `>=3.12,<3.13`)
- **GUI framework:** Tkinter (standard library) — do not introduce other GUI frameworks
- **Scientific computing:** Use `numpy` and `pandas` efficiently for data processing
- **Naming:** Scientific code may use uppercase variable names (X, Y, Z, F, etc.) — this is acceptable and suppressed in linting
- **Output pattern:** Analysis modules write results to timestamped subdirectories (e.g., `processed_linear_lowess_YYYYMMDD_HHMMSS/`)
- **Build system:** `hatchling` backend, managed via `uv`
