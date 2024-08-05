import os
import tkinter as tk
from tkinter import messagebox, filedialog, simpledialog
import maintools as tools
import pandas as pd

def get_coupling_angle(file, axis='x', joint1_name='Joint1', joint2_name='Joint2', format='c3d', save=True, savedir=None, savename='vectorcoding_result'):
    print('-------------------------------------------------------------------------------')
    print(f'Processing: {file}')
    print(f'Axis: {axis}')
    print(f'Joint 1: {joint1_name}')
    print(f'Joint2: {joint2_name}')

    if not savename:
        savename = os.path.splitext(os.path.basename(file))[0]

    if format != 'c3d':
        raise ValueError('Currently, only C3D file format is supported.')

    print('\n Loading C3D file.')
    dict_kin = tools.get_kinematics_c3d(file)
    freq = tools.get_kinematic_framerate(file)

    df_joint1 = tools.get_joint_df(dict_kin, joint1_name)
    df_joint2 = tools.get_joint_df(dict_kin, joint2_name)
    array_joint1_raw = tools.timenormalize_data(df_joint1[[axis]]).flatten()
    array_joint2_raw = tools.timenormalize_data(df_joint2[[axis]]).flatten()

    print('\n Filtering data - Butterworth Filter Low Pass')
    array_joint1 = tools.butter_lowpass(freq, array_joint1_raw, fc=6)
    array_joint2 = tools.butter_lowpass(freq, array_joint2_raw, fc=6)
    
    print(f'\n Array Joint 1: {joint1_name}')
    print(array_joint1)

    print(f'\n Array Joint 2: {joint2_name}')
    print(array_joint2)

    print('\n Calculating Coupling Angles (Vector Coding).')
    group_percent, coupangle = tools.calculate_coupling_angle(array_joint1, array_joint2)

    fig, ax = tools.create_coupling_angle_figure(group_percent, coupangle, array_joint1, array_joint2, joint1_name, joint2_name, axis, size=15)

    phase = ['Anti-Phase', 'In-Phase', f'{joint1_name} Phase', f'{joint2_name} Phase']
    data = [array_joint1, array_joint2, coupangle, group_percent, phase]
    df = pd.DataFrame(data).T
    df.columns = [f'{joint1_name}_{axis}', f'{joint2_name}_{axis}', 'coupling_angle', 'phase_percentages', 'phase']
    print(df.head(10))

    if save:
        output_path = savename if savedir is None else os.path.join(savedir, savename)
        if not os.path.exists(savedir):
            os.makedirs(savedir)

        df.to_csv(f'{output_path}.csv', index=False)
        fig.savefig(f'{output_path}.png', dpi=300, bbox_inches='tight')
        
        print(f'\n All results files have been saved in {output_path}.')
        print('-------------------------------------------------------------------------------')
    else: 
        fig.show()
        print('\n DataFrame with results:')
        print(df)
        return fig, df

def run_vector_coding():
    root = tk.Tk()
    root.withdraw()  # Esconde a janela principal

    file = filedialog.askopenfilename(filetypes=[("C3D files", "*.c3d")])
    if not file:
        return

    axis = simpledialog.askstring("Input", "Enter axis (default is 'x'):", initialvalue="x")
    joint1_name = simpledialog.askstring("Input", "Enter the name of the first joint (default is 'Joint1'):", initialvalue="Joint1")
    joint2_name = simpledialog.askstring("Input", "Enter the name of the second joint (default is 'Joint2'):", initialvalue="Joint2")
    format = 'c3d'
    save = messagebox.askyesno("Save", "Do you want to save the results?")
    savedir = filedialog.askdirectory() if save else None
    savename = simpledialog.askstring("Input", "Enter the name of the output file (default is 'vectorcoding_result'):", initialvalue="vectorcoding_result")

    if not all([file, axis, joint1_name, joint2_name, format]):
        messagebox.showerror("Error", "Please provide all the required inputs.")
        return

    get_coupling_angle(
        file=file,
        axis=axis,
        joint1_name=joint1_name,
        joint2_name=joint2_name,
        format=format,
        save=save,
        savedir=savedir,
        savename=savename
    )

if __name__ == '__main__':
    run_vector_coding()
