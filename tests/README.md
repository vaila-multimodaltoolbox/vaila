# _vailá_ Tests and Sample Data

This directory contains automated tests and sample data files used to verify the functionality of the _vailá_ toolbox.

## Directory Structure

- **`test_*.py`**: Executable test scripts using `pytest`.
  - `test_vaila_and_jump.py`: Unit tests for biomechanical calculation functions.
  - `test_vaila_and_jump_integration.py`: Integration tests for full data processing pipelines.
- **Subdirectories**: Contain sample data files (CSV, C3D, GPX, MP4, TOML) used for both automated and manual testing.
  - `vaila_and_jump/`: Sample data for vertical jump analysis (Time-of-flight, Jump-height, MediaPipe).
  - `C3D_to_CSV/`, `Force_Plate/`, etc.: Sample data for other analysis modules.

## How to Run Tests

Automated tests are managed using **uv** and **pytest**.

### Run All Tests

```bash
uv run pytest
```

### Run Specific Test Modules

```bash
# Biomechanical Formulas (Unit Tests)
uv run pytest tests/test_vaila_and_jump.py -v

# Data Pipelines (Integration Tests)
uv run pytest tests/test_vaila_and_jump_integration.py -v
```

### Test Configuration

Tests are configured to run headlessly (no GUI). Integration tests automatically locate sample data within this directory.

---

**Note:** When adding new analysis modules to _vailá_, it is recommended to add corresponding unit and integration tests here to ensure ongoing reliability.
