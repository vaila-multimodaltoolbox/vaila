# Skill: Create a New Analysis Module

Use this skill whenever adding a new biomechanical analysis tool to vailá.
Follow every step in order.

---

## Step 1 — Create the module file

Create `vaila/<module_name>.py` with this structure:

```python
"""
<module_name>.py — <One-line description>.

Analyzes <what> from <data source>.
Reference: Author et al. (Year). Journal. DOI
"""
from __future__ import annotations

import tkinter as tk
from tkinter import filedialog, messagebox
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from vaila.common_utils import get_vaila_dir


# ── Public entry point (called from vaila.py button) ──────────────────────────

def run_<module_name>() -> None:
    """GUI entry point. Called directly from the Tkinter button."""
    root = tk.Tk()
    root.withdraw()

    file_path = filedialog.askopenfilename(
        title="Select input file",
        filetypes=[("CSV files", "*.csv"), ("All files", "*.*")]
    )
    if not file_path:
        return

    try:
        result = process_<module_name>(Path(file_path))
        messagebox.showinfo("vailá", f"Done! Results saved to:\n{result}")
    except Exception as e:
        messagebox.showerror("vailá Error", str(e))


# ── Core processing function ───────────────────────────────────────────────────

def process_<module_name>(input_path: Path) -> Path:
    """
    Process <what> and save results.

    Parameters
    ----------
    input_path : Path
        Path to input CSV file.

    Returns
    -------
    Path
        Path to output directory containing results.
    """
    output_dir = input_path.parent / f"{input_path.stem}_<module_name>_results"
    output_dir.mkdir(exist_ok=True)

    # Load data
    df = pd.read_csv(input_path)

    # Process
    result = _compute_metric(df.values)

    # Save CSV
    result_df = pd.DataFrame(result, columns=["time_s", "metric"])
    result_df.to_csv(output_dir / "results.csv", index=False)

    # Save plot
    fig, ax = plt.subplots()
    ax.plot(result_df["time_s"], result_df["metric"])
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Metric")
    ax.set_title("<Module Name> Results")
    fig.savefig(output_dir / "plot.png", dpi=150)
    plt.close(fig)

    return output_dir


# ── Internal computation ───────────────────────────────────────────────────────

def _compute_metric(data: np.ndarray) -> np.ndarray:
    """
    Compute <metric> from raw data.

    Parameters
    ----------
    data : np.ndarray, shape (N, M)
        Raw input data in <units>.

    Returns
    -------
    np.ndarray, shape (N, 2)
        Columns: [time_s, metric_value]
    """
    # TODO: implement algorithm
    raise NotImplementedError
```

---

## Step 2 — Wire the button in vaila.py

1. Open `vaila.py`
2. Add import at the top (with other imports):
   ```python
   from vaila.<module_name> import run_<module_name>
   ```
3. Find the correct Frame builder function (A, B, or C)
4. Find the next free slot (`[free]` button or next row)
5. Add:
   ```python
   btn_<module_name> = ttk.Button(
       frame_b, text="<Button Label>", command=run_<module_name>
   )
   btn_<module_name>.grid(row=X, column=Y, padx=5, pady=5, sticky="ew")
   ```

---

## Step 3 — Write tests

Create `tests/test_<module_name>.py` — use the `test-writer` agent or
the `create-test-suite` skill.

Minimum required tests:
- `test_process_<module_name>_basic` — happy path with sample data
- `test_process_<module_name>_output_files` — verify CSV and PNG are created
- `test_compute_metric_known_values` — validate formula with known result

---

## Step 4 — Update documentation

1. Add entry to `docs/vaila_buttons/README.md`
2. Add entry to `vaila/help/index.md`
3. If new dependency needed: `uv add <package>` then commit updated `uv.lock`
   and update **all** `pyproject_*.toml` files

---

## Checklist Before Committing

- [ ] Module imports without errors: `uv run python -c "from vaila.<module_name> import run_<module_name>"`
- [ ] Tests pass: `uv run pytest tests/test_<module_name>.py -v`
- [ ] Button appears correctly in GUI: `uv run vaila.py`
- [ ] Output files created in correct location
- [ ] No hardcoded paths
- [ ] Works on Linux/macOS/Windows paths (use `pathlib.Path`)
