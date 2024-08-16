import os
import tkinter as tk
from tkinter import filedialog, messagebox
from . import maintools as tools  # Alterar para um import relativo
import pandas as pd


def get_coupling_angle(
    file,
    axis="x",
    joint1_name="Joint1",
    joint2_name="Joint2",
    format="c3d",
    save=True,
    savedir=None,
    savename="vectorcoding_result",
):
    print(
        "-------------------------------------------------------------------------------"
    )
    print(f"Processing: {file}")
    print(f"Axis: {axis}")
    print(f"Joint 1: {joint1_name}")
    print(f"Joint2: {joint2_name}")

    if not savename:
        savename = os.path.splitext(os.path.basename(file))[0]

    if format != "c3d":
        raise ValueError("Currently, only C3D file format is supported.")

    print("\n Loading C3D file.")
    dict_kin = tools.get_kinematics_c3d(file)
    freq = tools.get_kinematic_framerate(file)

    df_joint1 = tools.get_joint_df(dict_kin, joint1_name)
    df_joint2 = tools.get_joint_df(dict_kin, joint2_name)
    array_joint1_raw = tools.timenormalize_data(df_joint1[[axis]]).flatten()
    array_joint2_raw = tools.timenormalize_data(df_joint2[[axis]]).flatten()

    print("\n Filtering data - Butterworth Filter Low Pass")
    array_joint1 = tools.butter_lowpass(freq, array_joint1_raw, fc=6)
    array_joint2 = tools.butter_lowpass(freq, array_joint2_raw, fc=6)

    print(f"\n Array Joint 1: {joint1_name}")
    print(array_joint1)

    print(f"\n Array Joint 2: {joint2_name}")
    print(array_joint2)

    print("\n Calculating Coupling Angles (Vector Coding).")
    group_percent, coupangle = tools.calculate_coupling_angle(
        array_joint1, array_joint2
    )

    fig, ax = tools.create_coupling_angle_figure(
        group_percent,
        coupangle,
        array_joint1,
        array_joint2,
        joint1_name,
        joint2_name,
        axis,
        size=15,
    )

    phase = ["Anti-Phase", "In-Phase", f"{joint1_name} Phase", f"{joint2_name} Phase"]
    data = [array_joint1, array_joint2, coupangle, group_percent, phase]
    df = pd.DataFrame(data).T
    df.columns = [
        f"{joint1_name}_{axis}",
        f"{joint2_name}_{axis}",
        "coupling_angle",
        "phase_percentages",
        "phase",
    ]
    print(df.head(10))

    if save:
        output_path = savename if savedir is None else os.path.join(savedir, savename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        df.to_csv(f"{output_path}.csv", index=False)
        fig.savefig(f"{output_path}.png", dpi=300, bbox_inches="tight")

        print(f"\n All results files have been saved in {output_path}.")
        print(
            "-------------------------------------------------------------------------------"
        )
    else:
        fig.show()
        print("\n DataFrame with results:")
        print(df)
        return fig, df


def browse_file():
    file_path = filedialog.askopenfilename(
        filetypes=[("C3D files", "*.c3d"), ("All files", "*.*")]
    )
    file_entry.delete(0, tk.END)
    file_entry.insert(0, file_path)


def run_analysis():
    try:
        get_coupling_angle(
            file=file_entry.get(),
            axis=axis_entry.get(),
            joint1_name=joint1_entry.get(),
            joint2_name=joint2_entry.get(),
            format=format_entry.get(),
            save=save_var.get(),
            savedir=savedir_entry.get(),
            savename=savename_entry.get(),
        )
        messagebox.showinfo("Success", "Analysis completed successfully!")
    except Exception as e:
        messagebox.showerror("Error", str(e))


root = tk.Tk()
root.title("Coupling Angle Analysis")

tk.Label(root, text="File:").grid(row=0, column=0, sticky="e")
file_entry = tk.Entry(root, width=50)
file_entry.grid(row=0, column=1)
tk.Button(root, text="Browse", command=browse_file).grid(row=0, column=2)

tk.Label(root, text="Axis:").grid(row=1, column=0, sticky="e")
axis_entry = tk.Entry(root)
axis_entry.grid(row=1, column=1)
axis_entry.insert(0, "x")

tk.Label(root, text="Joint 1 Name:").grid(row=2, column=0, sticky="e")
joint1_entry = tk.Entry(root)
joint1_entry.grid(row=2, column=1)
joint1_entry.insert(0, "Joint1")

tk.Label(root, text="Joint 2 Name:").grid(row=3, column=0, sticky="e")
joint2_entry = tk.Entry(root)
joint2_entry.grid(row=3, column=1)
joint2_entry.insert(0, "Joint2")

tk.Label(root, text="Format:").grid(row=4, column=0, sticky="e")
format_entry = tk.Entry(root)
format_entry.grid(row=4, column=1)
format_entry.insert(0, "c3d")

save_var = tk.BooleanVar(value=True)
tk.Checkbutton(root, text="Save Results", variable=save_var).grid(
    row=5, column=1, sticky="w"
)

tk.Label(root, text="Save Directory:").grid(row=6, column=0, sticky="e")
savedir_entry = tk.Entry(root, width=50)
savedir_entry.grid(row=6, column=1)

tk.Label(root, text="Save Name:").grid(row=7, column=0, sticky="e")
savename_entry = tk.Entry(root)
savename_entry.grid(row=7, column=1)
savename_entry.insert(0, "vectorcoding_result")

tk.Button(root, text="Run Analysis", command=run_analysis).grid(
    row=8, column=1, pady=20
)

root.mainloop()
