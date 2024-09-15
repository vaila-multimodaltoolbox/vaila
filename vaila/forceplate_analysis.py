"""
================================================================================
forceplate_analysis.py
================================================================================
Author: Prof. Paulo Santiago
Date: 9 September 2024
Version: 0.5

Description:
------------
This script serves as the central control interface for the VAILA (Virtual Analysis 
for Interactive Learning in Biomechanics) toolbox, providing a user-friendly graphical 
user interface (GUI) for selecting and executing various biomechanical analysis scripts. 
The analyses offered in this toolbox support the study and evaluation of postural control, 
dynamic balance, and force measurements, which are critical in both clinical and research 
settings.

Key Analyses Supported:
-----------------------
1. Force Cube Analysis: Examines force data captured in a cubic arrangement, allowing for 
   multidirectional force vector analysis.
2. Center of Pressure (CoP) Balance Analysis: Evaluates postural stability by analyzing 
   the center of pressure data, providing insights into balance control and sway 
   characteristics.
3. Force CMJ (Countermovement Jump) Analysis: Analyzes the forces involved in a 
   countermovement jump to assess athletic performance, muscle power, and explosiveness.
4. Noise Signal Fixing: Identifies and corrects noise artifacts in force signals, ensuring 
   data accuracy for subsequent analyses.
5. Calculate CoP: Executes a new process to calculate the CoP data.

Functionalities:
----------------
1. GUI for Analysis Selection: Utilizes Python's Tkinter library to present a straightforward 
   interface where users can choose their desired analysis with ease.
2. Dynamic Module Importing: Efficiently loads only the necessary module for the selected 
   analysis, conserving system resources and improving performance.
3. Execution of Analysis: Automatically runs the main function of the selected analysis module, 
   guiding users through any necessary data input or parameter settings.

Modules and Packages Required:
------------------------------
- Python Standard Libraries: tkinter for GUI creation and management.
- External Libraries: None required; all functionalities are built using standard Python libraries.
- VAILA Toolbox Modules: 
  * force_cube_fig: For analyzing force cube data.
  * cop_analysis: For conducting CoP balance analysis.
  * force_cmj: For analyzing countermovement jump dynamics.
  * fixnoise: For correcting noise in force data.
  * cop_calculate: For calculating CoP data.

How to Use:
-----------
1. Run the Script: 
   Execute the script from the terminal using:
   python -m vaila.forceplate_analysis
   Alternatively, double-click the script file if your environment supports it.

2. Select Analysis:
   A GUI window will appear, prompting you to select the type of analysis to perform. Click on 
   the corresponding button for the desired analysis.

3. Follow Instructions:
   The selected analysis will start, and you will be guided through the necessary steps, such as 
   choosing data files or entering parameters. Follow the on-screen instructions to complete the 
   analysis.

License:
--------
This script is licensed under the MIT License. For more details, please refer to the LICENSE file 
located in the project root.

Disclaimer:
-----------
This script is provided "as is," without any warranty, express or implied. The authors are not 
liable for any damage or data loss resulting from the use of this script. It is intended solely for 
academic and research purposes.

Changelog:
----------
- 2024-09-09: Initial creation of the script with dynamic analysis selection functionality.
- 2024-09-10: Added support for CoP Balance Analysis (cop_analysis.py).
- 2024-09-14: Added "Calculate CoP" button and functionality (cop_calculate.py).
================================================================================
"""

import os
import tkinter as tk


def choose_analysis_type():
    """
    Opens a GUI to choose which analysis code to run.
    """
    # Print the directory and name of the script being executed
    print(f"Running script: {os.path.basename(__file__)}")
    print(f"Script directory: {os.path.dirname(os.path.abspath(__file__))}")

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

    def select_calculate_cop():
        choice.append("calculate_cop")
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

    # New button for "Calculate CoP"
    btn_calculate_cop = tk.Button(
        choice_window, text="Calculate CoP", command=select_calculate_cop
    )
    btn_calculate_cop.pack(pady=5)

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


def run_calculate_cop():
    """
    Runs the Calculate CoP process.
    """
    try:
        from . import cop_calculate  # Import the new script

        cop_calculate.main()  # Call the main function in cop_calculate.py
    except ImportError as e:
        print(f"Error importing cop_calculate: {e}")


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
    elif analysis_type == "calculate_cop":
        run_calculate_cop()  # Call the new function when "Calculate CoP" is selected
    else:
        print("No analysis type selected.")


if __name__ == "__main__":
    run_force_analysis()
