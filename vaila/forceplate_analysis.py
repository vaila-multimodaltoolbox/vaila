# forceplate_analysis.py
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

    def select_force_balance():
        choice.append("force_balance")
        choice_window.quit()
        choice_window.destroy()

    def select_force_cmj():
        choice.append("force_cmj")
        choice_window.quit()
        choice_window.destroy()

    choice_window = tk.Toplevel()
    choice_window.title("Choose Analysis Type")

    tk.Label(choice_window, text="Select which analysis to run:").pack(pady=10)

    btn_force_cube = tk.Button(
        choice_window, text="Force Cube Analysis", command=select_force_cube_fig
    )
    btn_force_cube.pack(pady=5)

    btn_force_balance = tk.Button(
        choice_window, text="Force Balance Analysis", command=select_force_balance
    )
    btn_force_balance.pack(pady=5)

    btn_force_cmj = tk.Button(
        choice_window, text="Force CMJ Analysis", command=select_force_cmj
    )
    btn_force_cmj.pack(pady=5)

    choice_window.mainloop()

    return choice[0] if choice else None


def run_force_cube_analysis():
    """
    Runs the Force Cube Analysis.
    """
    from . import (
        force_cube_fig,
    )  # Importa o módulo force_cube_fig apenas quando necessário

    force_cube_fig.main()  # Chama a função principal do script force_cube_fig


def run_force_balance_analysis():
    """
    Runs the Force Balance Analysis.
    """
    from . import (
        force_balance,
    )  # Importa o módulo force_balance apenas quando necessário

    force_balance.main()  # Chama a função principal do script force_balance


def run_force_cmj_analysis():
    """
    Runs the Force CMJ Analysis.
    """
    from . import force_cmj  # Importa o módulo force_cmj apenas quando necessário

    force_cmj.main()  # Chama a função principal do script force_cmj


def run_force_analysis():
    """
    Main function to execute the chosen force analysis.
    """
    root = tk.Tk()
    root.withdraw()  # Hide the main Tkinter window

    analysis_type = choose_analysis_type()

    if analysis_type == "force_cube_fig":
        run_force_cube_analysis()
    elif analysis_type == "force_balance":
        run_force_balance_analysis()
    elif analysis_type == "force_cmj":
        run_force_cmj_analysis()
    else:
        print("No analysis type selected.")


if __name__ == "__main__":
    run_force_analysis()
