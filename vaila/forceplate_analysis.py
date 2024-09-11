"""
================================================================================
forceplate_analysis.py
================================================================================
Author: Prof. Paulo Santiago
Date: 9 September 2024
Version: 0.5

Description:
------------
This script provides a graphical user interface (GUI) for selecting and running various 
biomechanical analysis scripts as part of the VAILA toolbox. The available analyses include:
- Force Cube Analysis
- Center of Pressure (CoP) Balance Analysis
- Force CMJ Analysis
- Noise Signal Fixing

The user is prompted to choose the desired analysis via a GUI, and the corresponding analysis 
script is executed based on the selection.

Functionalities:
----------------
1. **GUI for Analysis Selection**: Presents the user with a choice of analyses to run.
2. **Dynamic Module Importing**: Only imports the necessary module when a specific analysis is selected, 
   optimizing performance and memory usage.
3. **Analysis Execution**: Executes the main function of the selected analysis script.

Modules and Packages Required:
-------------------------------
- Python Standard Libraries: `tkinter`
- External Libraries: None
- VAILA Toolbox Modules: 
  - `force_cube_fig`
  - `cop_analysis`
  - `force_cmj`
  - `fixnoise`

How to Use:
-----------
1. Run the script from the terminal using:
```python
    python -m vaila.forceplate_analysis
```
or by double-clicking the script if running in an environment where this is possible.

    A GUI window will appear, allowing you to select the desired analysis type. Choose the appropriate button for the analysis you wish to perform.

    The selected analysis will run, and the user will be guided through any required steps, such as selecting data files or inputting parameters.

License:

This script is licensed under the MIT License. See LICENSE file in the project root for more details.
Disclaimer:

This script is provided "as is," without warranty of any kind. Use at your own risk. It is intended for academic and research purposes only.
Changelog:

    2024-09-09: Initial creation of the script with functionality for dynamic analysis selection.
    2024-09-10: Added support for CoP Balance Analysis (cop_analysis.py).

================================================================================
"""

import tkinter as tk


def choose_analysis_type():
    """
    Opens a GUI to choose which analysis code to run.
    """
    choice = []

    def select_force_cube_fig():
        choice.append("force_cube_fig")
        choice_window.quit()
        choice_window.destroy()

    def select_cop_balance():
        choice.append("cop_balance")
        choice_window.quit()
        choice_window.destroy()

    def select_force_cmj():
        choice.append("force_cmj")
        choice_window.quit()
        choice_window.destroy()

    def select_fix_noise():
        choice.append("fix_noise")
        choice_window.quit()
        choice_window.destroy()

    choice_window = tk.Toplevel()
    choice_window.title("Choose Analysis Type")

    tk.Label(choice_window, text="Select which analysis to run:").pack(pady=10)

    btn_force_cube = tk.Button(
        choice_window, text="Force Cube Analysis", command=select_force_cube_fig
    )
    btn_force_cube.pack(pady=5)

    btn_cop_balance = tk.Button(
        choice_window, text="CoP Balance Analysis", command=select_cop_balance
    )
    btn_cop_balance.pack(pady=5)

    btn_force_cmj = tk.Button(
        choice_window, text="Force CMJ Analysis", command=select_force_cmj
    )
    btn_force_cmj.pack(pady=5)

    btn_fix_noise = tk.Button(
        choice_window, text="Fix Noise Signal", command=select_fix_noise
    )
    btn_fix_noise.pack(pady=5)

    choice_window.mainloop()

    return choice[0] if choice else None


def run_force_cube_analysis():
    """
    Runs the Force Cube Analysis.
    """
    try:
        from . import force_cube_fig

        force_cube_fig.main()
    except ImportError as e:
        print(f"Error importing force_cube_fig: {e}")


def run_cop_balance_analysis():
    """
    Runs the CoP Balance Analysis.
    """
    try:
        from . import cop_analysis

        cop_analysis.main()
    except ImportError as e:
        print(f"Error importing cop_analysis: {e}")


def run_force_cmj_analysis():
    """
    Runs the Force CMJ Analysis.
    """
    try:
        from . import force_cmj

        force_cmj.main()
    except ImportError as e:
        print(f"Error importing force_cmj: {e}")


def run_fix_noise():
    """
    Runs the Fix Noise Signal Analysis.
    """
    try:
        from . import fixnoise

        fixnoise.main()
    except ImportError as e:
        print(f"Error importing fixnoise: {e}")


def run_force_analysis():
    """
    Main function to execute the chosen force analysis.
    """
    root = tk.Tk()
    root.withdraw()

    analysis_type = choose_analysis_type()

    if analysis_type == "force_cube_fig":
        run_force_cube_analysis()
    elif analysis_type == "cop_balance":
        run_cop_balance_analysis()
    elif analysis_type == "force_cmj":
        run_force_cmj_analysis()
    elif analysis_type == "fix_noise":
        run_fix_noise()
    else:
        print("No analysis type selected.")


if __name__ == "__main__":
    run_force_analysis()
