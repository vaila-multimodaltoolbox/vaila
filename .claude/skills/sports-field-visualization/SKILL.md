# Sports Field Visualization & GUI-CLI Integration

This skill documents the patterns and fixes for maintaining cross-platform compatibility between Tkinter GUIs and standalone CLI tools using Matplotlib, specifically focusing on the `drawsportsfields.py` and `markerless_3d_analysis.py` modules.

## Context

The `vailá` project uses a dual-mode execution pattern for many modules:
1. **Integrated GUI**: Modules are imported into `vaila.py` and run inside the main Tkinter notebook.
2. **Standalone CLI**: Modules are run directly (e.g., `uv run vaila/module.py --arg value`).

## Key Challenges & Solutions

### 1. Matplotlib Backend Synchronization
**Issue**: Standing-alone CLI execution often defaults to the `QtAgg` backend. If the system lacks Qt dependencies (common in minimal Linux installs or headless environments), the script crashes with `qt.qpa.plugin` errors.
**Solution**: Explicitly set the backend to `TkAgg` before importing `pyplot` if the module is designed for Tkinter integration.
```python
import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
```

### 2. GUI vs. CLI Execution Paths
**Issue**: Modules originally designed as GUIs often trigger the full `tk.Tk()` main loop even when run with CLI arguments (e.g., `--field`), leading to unnecessary window creation or crashes in non-interactive sessions.
**Solution**: Use the `if __name__ == "__main__":` block to distinguish between a full GUI launch and a simple plot visualization.
```python
if __name__ == "__main__":
    if args.field:
        # Static plot mode (CLI)
        plot_field(df)
        plt.show()
    else:
        # Interactive GUI mode
        run_gui()
```

### 3. App Class Portability
**Issue**: Defining the main application class inside `if __name__ == "__main__":` prevents it from being imported by other modules. Also, restricting `master` to `tk.Tk` prevents embedding it in `tk.Toplevel` windows.
**Solution**: 
- Move the `App` class to the module level.
- Update type hints to `tk.Tk | tk.Toplevel`.
```python
class App:
    def __init__(self, master: tk.Tk | tk.Toplevel):
        self.master = master
        # initialization logic...
```

### 4. Matplotlib Colormap Deprecations
**Issue**: Direct attribute access like `plt.cm.rainbow` is deprecated in modern Matplotlib versions and triggers linting errors.
**Solution**: Use the registry-based retrieval method.
```python
# GOOD
cmap = plt.get_cmap("rainbow")

# BAD
cmap = plt.cm.rainbow
```

## FIFA Field Geometry Integration

The "FIFA" layout is now natively supported using a coordinate system centered at `(0,0,0)`.

- **Reference File**: `vaila/models/soccerfield_ref3d_fifa.csv`
- **Registry Entry**:
```python
SPORT_REGISTRY["fifa"] = SportDef(
    label="FIFA Starter Kit",
    model_csv="soccerfield_ref3d_fifa.csv",
    title="FIFA Skeletal Tracking Pitch",
    plot_fn=plot_field,
)
```

## Related Modules
- [drawsportsfields.py](file:///home/preto/data/vaila/vaila/drawsportsfields.py)
- [markerless_3d_analysis.py](file:///home/preto/data/vaila/vaila/markerless_3d_analysis.py)
- [soccerfield_ref3d_fifa.csv](file:///home/preto/data/vaila/vaila/models/soccerfield_ref3d_fifa.csv)
