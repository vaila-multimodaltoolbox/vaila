# CLAUDE.md

Guidance for **AI Assistants** (Claude Code, Antigravity, Cursor, Windsurf, etc.) when working with the **vailá** repository.

> See also: [AGENTS.md](./AGENTS.md) — shared rules for all AI agents.

## Project Overview

**vailá** (Versatile Anarcho Integrated Liberation Ánalysis) — open-source Python 3.12 multimodal toolbox for biomechanical data analysis. Integrates IMU, motion capture, markerless tracking (MediaPipe, YOLO), force plates, EMG, GNSS/GPS through a Tkinter-based GUI.

- **GitHub:** https://github.com/vaila-multimodaltoolbox/vaila
- **Python:** strictly `>=3.12,<3.13`
- **License:** AGPLv3
- **Build backend:** `hatchling` managed via [`uv`](https://docs.astral.sh/uv/)

---

## Astral Toolchain

The project uses the full [Astral](https://astral.sh) Rust-based toolchain:

| Tool                                   | Purpose                                                    | Replaces                        |
| -------------------------------------- | ---------------------------------------------------------- | ------------------------------- |
| [`uv`](https://docs.astral.sh/uv/)     | Package manager, venv, Python installer                    | pip, poetry, pyenv, virtualenv  |
| [`ruff`](https://docs.astral.sh/ruff/) | Linter + formatter                                         | flake8, black, isort, pyupgrade |
| [`ty`](https://docs.astral.sh/ty/)     | Static type checker (beta, Rust, 10-100x faster than mypy) | mypy, Pyright                   |

> **Never use** bare `pip install`, `black`, `isort`, `flake8`, or `mypy` — always use the Astral equivalents via `uv run`.

---

## Commands Reference

### uv

```bash
# Run the application
uv run vaila.py

# Sync dependencies (reads uv.lock + pyproject.toml)
uv sync                        # CPU-only
uv sync --extra gpu            # With GPU support
uv sync --frozen               # CI mode: fail if lock is outdated

# Manage dependencies
uv add <package>               # Add runtime dependency
uv add --dev <package>         # Add dev dependency
uv remove <package>            # Remove dependency
uv lock                        # Regenerate uv.lock
uv lock --upgrade              # Upgrade all packages

# Python version management
uv python install 3.12         # Install Python 3.12
uv python pin 3.12             # Pin project to 3.12
uv venv --python 3.12          # Create venv with specific version

# Global tools (outside project venv)
uv tool install ruff           # Install ruff globally
uv tool install ty             # Install ty globally
uv tool upgrade ruff           # Upgrade ruff globally
uvx ruff check vaila/          # Run ruff ephemerally (no install)

# Export for legacy tooling
uv export --format requirements-txt > requirements.txt
uv export --format requirements-txt --no-hashes --frozen > requirements.txt
```

### ruff

```bash
# Linting
uv run ruff check vaila/              # Lint all files
uv run ruff check vaila/ --fix        # Lint + auto-fix safe issues
uv run ruff check vaila/ --fix-only   # Apply fixes only, no output
uv run ruff check vaila/ --diff       # Preview what --fix would change

# Formatting (replaces black)
uv run ruff format vaila/             # Format all files
uv run ruff format vaila/ --check     # CI mode: check without writing
uv run ruff format vaila/ --diff      # Preview what format would change

# Single file
uv run ruff check vaila/my_module.py --fix
uv run ruff format vaila/my_module.py
```

**Inline suppression:**

```python
x = some_var  # noqa: F841
x = some_var  # noqa: F841, E501
```

**Config in `pyproject.toml`:**

```toml
[tool.ruff]
target-version = "py312"
line-length = 100

[tool.ruff.lint]
select = ["E", "W", "F", "I", "N", "NPY", "UP", "B", "C4", "SIM"]
ignore = ["E501", "N806", "N803"]   # scientific uppercase vars are OK

[tool.ruff.lint.per-file-ignores]
"__init__.py" = ["F401"]            # intentional re-exports
```

### ty

```bash
# Type checking
uv run ty check vaila/                # Check all files
uv run ty check vaila/my_module.py    # Check single file
uv run ty check vaila/ --watch        # Watch mode: re-checks on save

# Override rule severity on CLI
uv run ty check vaila/ --error unresolved-import
uv run ty check vaila/ --warn  possibly-unbound
uv run ty check vaila/ --ignore division-by-zero
```

**Inline suppression:**

```python
x: int = "hello"  # ty: ignore[invalid-assignment]
x: int = "hello"  # ty: ignore[invalid-assignment, unresolved-import]
```

**Config in `pyproject.toml`:**

```toml
[tool.ty.rules]
unresolved-import  = "warn"   # "error" | "warn" | "ignore"
possibly-unbound   = "warn"
division-by-zero   = "error"
unused-ignore-comment = "warn"

[tool.ty.src]
include = ["vaila", "tests"]
exclude = ["vaila/_generated"]
```

> `ty` is in **beta** — not a drop-in replacement for mypy/Pyright; different design choices and defaults. Use alongside ruff, not instead of it.

---

## Full QA Pipeline (run before every commit)

```bash
uv run ruff check vaila/ --fix    # fix lint issues
uv run ruff format vaila/         # format code
uv run ty check vaila/            # type check
uv run pytest tests/ -v           # run tests
```

---

## Architecture

### Entry Point & GUI (`vaila.py`)

`vaila.py` defines `Vaila(tk.Tk)`, organized into three frames:

| Frame       | Purpose                                                                             |
| ----------- | ----------------------------------------------------------------------------------- |
| **Frame A** | File Manager — rename, import, export, copy, move, remove, tree, find, SSH transfer |
| **Frame B** | Multimodal Analysis — IMU, MoCap, Markerless 2D/3D, EMG, Force Plate, GNSS          |
| **Frame C** | Tools — CSV editing, C3D conversion, DLT reconstruction, video/image, visualization |

**Lazy imports** are used in all handler methods to avoid loading the full dependency graph at startup.

Two dispatch patterns:

1. **Direct import + call** — runs in the same process
2. **Subprocess via `run_vaila_module()`** — separate process (avoids Tkinter conflicts)

### Package Structure (`vaila/`)

~100 self-contained analysis modules. Each module:

- Has a `run_*()` or `analyze_*()` entry point called from the GUI
- Uses Tkinter `filedialog` for user input prompts
- Reads CSV/C3D via `pandas` / `numpy` / `ezc3d`
- Writes results (CSV + PNG plots) to timestamped output subdirectories

**Key shared modules:**

| Module                                      | Role                                                      |
| ------------------------------------------- | --------------------------------------------------------- |
| `data_processing.py`                        | CSV/C3D reading with auto-header detection                |
| `filtering.py` / `filter_utils.py`          | Butterworth and FIR filter implementations                |
| `common_utils.py`                           | Header detection and data reshaping                       |
| `dialogsuser.py` / `dialogsuser_cluster.py` | Reusable Tkinter input dialogs                            |
| `filemanager.py`                            | File management (rename, copy, move, SSH transfer)        |
| `hardware_manager.py`                       | GPU/CPU detection, TensorRT export — **do not duplicate** |
| `interp_smooth_split.py`                    | Interpolation, smoothing, splitting (GUI + CLI)           |

---

## Platform-Specific Configuration

Copy the correct template to `pyproject.toml` **before** running `uv python pin` / `uv venv`:

| Template                       | Target                          |
| ------------------------------ | ------------------------------- |
| `pyproject_win_cuda12.toml`    | Windows + NVIDIA CUDA 12.1      |
| `pyproject_linux_cuda12.toml`  | Linux + NVIDIA CUDA 12.8        |
| `pyproject_macos.toml`         | macOS Apple Silicon (Metal/MPS) |
| `pyproject_universal_cpu.toml` | CPU-only fallback               |

Install scripts handle this automatically: `install_vaila_linux.sh`, `install_vaila_mac.sh`, `install_vaila_win.ps1`.

---

## Coding Conventions

### Mandatory dual-import pattern

Every module must support both package import and standalone execution:

```python
try:
    from .readcsv import read_csv_file      # package import
    from .filtering import butter_filter
except ImportError:
    from readcsv import read_csv_file       # standalone fallback
    from filtering import butter_filter
```

### Rules

- **GUI framework:** Tkinter only — never introduce Qt, wx, Dear PyGui, etc.
- **Scientific variable names** (X, Y, Z, F, R, T, etc.) are valid — suppressed via ruff `N806`/`N803`
- **Output dirs:** always timestamped → `processed_<type>_YYYYMMDD_HHMMSS/`
- **No hard-coded absolute paths**
- **No files >24MB** (git hook enforced)

---

## Testing

```bash
uv run pytest tests/ -v                              # all tests
uv run pytest tests/test_vaila_and_jump.py -v        # biomechanical calculations
uv run pytest tests/test_tugturn.py -v               # TUG/Turn analysis
uv run pytest tests/test_dlt_rec.py -v               # DLT/Rec math
uv run pytest tests/test_dlt_rec_integration.py -v   # DLT/Rec pipeline
```

Sample data lives in `tests/vaila_and_jump/` (CSV + TOML).

---

## Common Task Recipes

### Add a new analysis module

1. Create `vaila/my_module.py` with `run_my_module()` as entry point
2. Apply dual-import pattern at the top
3. Use helpers from `dialogsuser.py` for user prompts
4. Write results to a timestamped output dir
5. Wire button in `vaila.py` with lazy import
6. Lint and type-check: `uv run ruff check vaila/my_module.py --fix && uv run ty check vaila/my_module.py`
7. Add unit test in `tests/`

### Fix all lint + type issues in one shot

```bash
uv run ruff check vaila/ --fix && uv run ruff format vaila/ && uv run ty check vaila/
```

### Run a module standalone via CLI

```bash
uv run vaila/interp_smooth_split.py -i /path/to/csv_dir -c smooth_config.toml
```

---

## Agents and skills

Step-by-step workflows and specialized agent roles are stored in the `.claude/` directory. This structure is intended to be used by any AI assistant (Claude Code, Antigravity, Cursor, etc.).

### Specialized Agents (`.claude/agents/`)

Role cards for domain experts. Use these when the task fits their specific domain:

- [biomechanics-analyst.md](file:///home/preto/Preto/vaila/.claude/agents/biomechanics-analyst.md)
- [gui-developer.md](file:///home/preto/Preto/vaila/.claude/agents/gui-developer.md)
- [video-processor.md](file:///home/preto/Preto/vaila/.claude/agents/video-processor.md)
- [test-writer.md](file:///home/preto/Preto/vaila/.claude/agents/test-writer.md)

### Technical Skills (`.claude/skills/`)

Reusable "how-to" guides for complex workflows:

- **vailá Core**: [create a new analysis module](file:///home/preto/Preto/vaila/.claude/skills/create-analysis-module.md), [port a MATLAB algorithm](file:///home/preto/Preto/vaila/.claude/skills/port-matlab-algorithm.md).
- **Reports**: [xlsx](file:///home/preto/Preto/vaila/.claude/skills/xlsx/SKILL.md) (Excel), [pdf](file:///home/preto/Preto/vaila/.claude/skills/pdf/SKILL.md), [pptx](file:///home/preto/Preto/vaila/.claude/skills/pptx/SKILL.md) (PowerPoint).
- **Automation**: [mcp-builder](file:///home/preto/Preto/vaila/.claude/skills/mcp-builder/SKILL.md) (Model Context Protocol), [webapp-testing](file:///home/preto/Preto/vaila/.claude/skills/webapp-testing/SKILL.md).
- **Visualization**: [web-artifacts-builder](file:///home/preto/Preto/vaila/.claude/skills/web-artifacts-builder/SKILL.md).

### Slash Commands (`.claude/commands/`)

Specs for common shortcuts like `/check` or `/new-module`.
