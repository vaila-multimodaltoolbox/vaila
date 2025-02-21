import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import pandas as pd
import numpy as np
from scipy import interpolate
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from datetime import datetime


def get_coupling_angle(
    file,
    freq,
    joint1_name="Joint1",
    joint2_name="Joint2",
    save=True,
    savedir=None,
    savename="vectorcoding_result",
):
    print("-------------------------------------------------------------------------------")
    print(f"Processing: {file}")
    print(f"Joint 1: {joint1_name}")
    print(f"Joint 2: {joint2_name}")

    if not savename:
        savename = os.path.splitext(os.path.basename(file))[0]

    # Load CSV file
    print("\n Loading CSV file.")
    df = pd.read_csv(file)
    
    # Instead of using the header name, always use the 2nd and 3rd column (idx: 1 and 2)
    if df.shape[1] < 3:
        print("Error: The CSV file must contain at least three columns (frame + 2 data columns).")
        return
    array_joint1_raw = df.iloc[:, 1].values
    array_joint2_raw = df.iloc[:, 2].values

    # Time normalize the data
    array_joint1 = timenormalize_data(array_joint1_raw.reshape(-1, 1)).flatten()
    array_joint2 = timenormalize_data(array_joint2_raw.reshape(-1, 1)).flatten()

    print(f"\n Array Joint 1: {joint1_name}")
    print(array_joint1)

    print(f"\n Array Joint 2: {joint2_name}")
    print(array_joint2)

    print("\n Calculating Coupling Angles (Vector Coding).")
    group_percent, coupangle = calculate_coupling_angle(array_joint1, array_joint2)

    fig, ax = create_coupling_angle_figure(
        group_percent,
        coupangle,
        array_joint1,
        array_joint2,
        joint1_name,
        joint2_name,
        axis="angle",
        size=15,
    )

    phase = [f"{joint1_name}", "In-Phase", f"{joint2_name}", "Anti-Phase"]
    data = [array_joint1, array_joint2, coupangle, group_percent, phase]
    df_result = pd.DataFrame(data).T
    df_result.columns = [
        f"{joint1_name}",
        f"{joint2_name}",
        "coupling_angle",
        "phase_percentages",
        "phase",
    ]
    print(df_result.head(10))

    if save:
        output_path = savename if savedir is None else os.path.join(savedir, savename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        df_result.to_csv(f"{output_path}.csv", index=False)
        fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")

        print(f"\n All results files have been saved in {output_path}.")
        print("-------------------------------------------------------------------------------")
    else:
        fig.show()
        print("\n DataFrame with results:")
        print(df_result)
        return fig, df_result


def run_vector_coding():
    root = tk.Tk()
    root.withdraw()

    # Solicitar ao usuário a pasta de entrada contendo os arquivos CSV
    input_dir = filedialog.askdirectory(
        title="Selecione a pasta contendo os arquivos CSV para análise"
    )
    if not input_dir:
        return

    # Solicitar ao usuário a pasta onde os resultados serão salvos
    output_dir = filedialog.askdirectory(
        title="Selecione a pasta onde os resultados serão salvos"
    )
    if not output_dir:
        return

    # Obter a lista de arquivos CSV na pasta de entrada
    csv_files = [
        os.path.join(input_dir, file)
        for file in os.listdir(input_dir)
        if file.lower().endswith(".csv")
    ]
    if not csv_files:
        messagebox.showerror("Erro", "Nenhum arquivo CSV encontrado na pasta selecionada.")
        return

    # Solicitar o frame rate (Hz) e os nomes dos ângulos (joint angles) apenas uma vez
    freq = simpledialog.askfloat("Input", "Digite a frequência de amostragem (Hz):", initialvalue=100.0)
    if not freq:
        return

    joint1_name = simpledialog.askstring("Input", "Digite o nome para o primeiro ângulo (usado apenas para rotulagem):", initialvalue="Joint1")
    joint2_name = simpledialog.askstring("Input", "Digite o nome para o segundo ângulo (usado apenas para rotulagem):", initialvalue="Joint2")
    if not all([freq, joint1_name, joint2_name]):
        messagebox.showerror("Erro", "Por favor, forneça todas as informações solicitadas.")
        return

    # Criar a pasta de saída root (dentro do diretório escolhido) com um timestamp global
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_root = os.path.join(output_dir, f"vaila_vectorcoding_{timestamp}")
    os.makedirs(output_root, exist_ok=True)

    # Processar cada arquivo CSV em modo de batch
    for file in csv_files:
        base_filename = os.path.splitext(os.path.basename(file))[0]
        sub_output_dir = os.path.join(output_root, f"{base_filename}_{timestamp}")
        os.makedirs(sub_output_dir, exist_ok=True)
        savename = f"{base_filename}_{timestamp}"
        get_coupling_angle(
            file=file,
            freq=freq,
            joint1_name=joint1_name,
            joint2_name=joint2_name,
            save=True,
            savedir=sub_output_dir,
            savename=savename,
        )
    
    messagebox.showinfo("Processamento Concluído", f"Análise em batch completa.\nResultados salvos em:\n{output_root}")


def timenormalize_data(data, n_points=101):
    """Time normalize data to n_points."""
    x = np.linspace(0, 100, len(data))
    f = interpolate.interp1d(x, data, axis=0)
    return f(np.linspace(0, 100, n_points))


def calculate_coupling_angle(angle1, angle2):
    """
    Calculates Vector Coding for coordination between two joint angle time-series.

    Parameters:
    - joint1_array (np.ndarray): First joint angle time-series.
    - joint2_array (np.ndarray): Second joint angle time-series.

    Returns:
    - tuple: A tuple containing group phase percentages and coupled angles.


        Raises:
    - ValueError: If input arrays are not of equal length or are empty.
    """
    if len(angle1) != len(angle2) or len(angle1) == 0:
        raise ValueError("Input arrays must be of equal non-zero length.")

    # Calculate joint differences
    array_joint1 = np.diff(angle1, axis=0)
    array_joint2 = np.diff(angle2, axis=0)

    # Calculate vector magnitude and angle
    vm_ab = np.hypot(array_joint1, array_joint2)
    cosang_ab = np.divide(array_joint1, vm_ab, where=vm_ab!=0)
    sinang_ab = np.divide(array_joint2, vm_ab, where=vm_ab!=0)
    coupangle = np.degrees(np.arctan2(cosang_ab, sinang_ab))

    # Ensure angle values are within 0-360 range
    coupangle[coupangle < 0] += 360

    # Assign categorical variable based on angle ranges
    CtgVar_vc_DG = np.select(
        condlist=[(coupangle >= 0) & (coupangle < 22.5),        # Joint 1 - Phase 
                    (coupangle >= 22.5) & (coupangle < 67.5),   # In-Phase
                    (coupangle >= 67.5) & (coupangle < 112.5),  # Joint 2 - Phase
                    (coupangle >= 112.5) & (coupangle < 157.5), # Anti-Phase 
                    (coupangle >= 157.5) & (coupangle < 202.5), # Joint 1 - Phase 
                    (coupangle >= 202.5) & (coupangle < 247.5), # In-Phase 
                    (coupangle >= 247.5) & (coupangle < 292.5), # Joint 2 - Phase
                    (coupangle >= 292.5) & (coupangle < 337.5), # Anti-Phase
                    (coupangle >= 337.5) & (coupangle < 360)],  # Joint 1 - Phase
        choicelist=[1, 2, 3, 4, 1, 2, 3, 4, 1],
        default=0
    )
    # Group 1 - Joint 1 - Phase
    # Group 2 - In-Phase 
    # Group 3 - Joint 2 - Phase
    # Group 4 - Anti-Phase
    
    # Calculate the frequency for each pattern of coordination
    group_phase = [round((np.count_nonzero(CtgVar_vc_DG == i) / len(CtgVar_vc_DG)) * 100, 3) for i in range(1, 5)]

    return group_phase, coupangle


def create_coupling_angle_figure(group_percent, coupangle, array_joint1, array_joint2, 
                               joint1_name, joint2_name, axis="angle", size=15):
    """
    Create a figure with three subplots:
      1. Joint angles over time.
      2. Coupling angle (with reference lines and a secondary y-axis for phase labels).
      3. A bar plot showing the percentages of the 4 coordination patterns.

    The bar plot is annotated so that the percentage for each pattern (Anti-Phase, In-Phase,
    Joint1 Phase, and Joint2 Phase) is visible.
    """
    letter_size = size - 5
    mark_size = size / 2
    alpha_value = 0.5
    gray_colors = ["0.1", "0.6", "0.3", "0.8"]

    plt.close("all")
    fig, ax = plt.subplots(3, figsize=(size, size / 1.5))
    plt.subplots_adjust(hspace=0.35)

    # First subplot: Joint angles
    ax[0].set_title(
        f"Joint Angles | {joint1_name} - {joint2_name} | Axis: {axis}",
        size=letter_size,
        weight="bold"
    )
    ax[0].plot(
        array_joint1,
        marker="o",
        linestyle="-",
        color="b",
        markersize=mark_size,
        alpha=alpha_value,
        label=joint1_name
    )
    ax[0].plot(
        array_joint2,
        marker="o",
        linestyle="-",
        color="r",
        markersize=mark_size,
        alpha=alpha_value,
        label=joint2_name
    )
    ax[0].legend(loc="best", fontsize=letter_size, frameon=False)
    ax[0].set_ylabel("Joint Angle (°)", fontsize=letter_size)
    ax[0].set_xlim(0, 100)
    ax[0].set_xlabel("Cycle (%)", fontsize=letter_size)

    # Second subplot: Coupling angle
    ax[1].set_title(
        f"Coupling Angle | {joint1_name} - {joint2_name} | Axis: {axis}",
        size=letter_size,
        weight="bold"
    )
    ax[1].plot(
        coupangle,
        color="k",
        marker="o",
        markersize=mark_size,
        linestyle=":",
        alpha=alpha_value,
        label="Coupling Angle"
    )
    ax[1].legend(loc="best", fontsize=letter_size, frameon=False)
    ax[1].set_ylabel("Coupling Angle (°)", fontsize=letter_size)
    ax[1].set_xlim(0, 100)
    ax[1].set_xlabel("Cycle (%)", fontsize=letter_size)

    # Add reference lines for phase transitions
    for angle in [22.5, 67.5, 112.5, 157.5, 202.5, 247.5, 292.5, 337.5, 360]:
        ax[1].axhline(angle, color="#55555B", linestyle="dotted", linewidth=0.5)
    ax[1].tick_params(axis="y", labelsize=letter_size)
    ax[1].tick_params(axis="x", labelsize=letter_size)

    # Add secondary y-axis with phase labels
    ax2 = ax[1].twinx()
    ax2.set_yticks(
        [22.5 - 11.25, 67.5 - 22.5, 112.5 - 22.5, 157.5 - 22.5,
         202.5 - 22.5, 247.5 - 22.5, 292.5 - 22.5, 337.5 - 22.5, 360],
        [f"{joint1_name}", "In-Phase", f"{joint2_name}", "Anti-Phase",
         f"{joint1_name}", "In-Phase", f"{joint2_name}", "Anti-Phase",
         f"{joint1_name}"],
        weight="bold"
    )

    # Third subplot: Coordination patterns as a bar plot
    labels = [f"{joint1_name}", "In-Phase", f"{joint2_name}", "Anti-Phase"]
    ax[2].bar(labels, group_percent, color=gray_colors, alpha=0.7)
    ax[2].set_title(
        f"Categorization of Coordination Patterns | {joint1_name} - {joint2_name}",
        size=letter_size,
        weight="bold"
    )
    ax[2].set_ylabel("Percentage (%)", fontsize=letter_size)
    ax[2].set_ylim(0, 100)
    for i, val in enumerate(group_percent):
        ax[2].text(i, val + 2, f"{val:.1f}%", ha='center', va='bottom', fontsize=letter_size-2)

    return fig, ax


if __name__ == "__main__":
    run_vector_coding()
