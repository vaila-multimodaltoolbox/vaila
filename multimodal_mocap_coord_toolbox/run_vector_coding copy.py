import maintools as tools
import os
import argparse
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

if __name__ == '__main__':
    print(r"""
            ···································································
            :__     __        _                ____          _ _              :
            :\ \   / ___  ___| |_ ___  _ __   / ___|___   __| (_)_ __   __ _  :
            : \ \ / / _ \/ __| __/ _ \| '__| | |   / _ \ / _` | | '_ \ / _` | :
            :  \ V |  __| (__| || (_) | |    | |__| (_) | (_| | | | | | (_| | :
            :   \_/ \___|\___|\__\___/|_|     \____\___/ \__,_|_|_| |_|\__, | :
            :                                                          |___/  :
            ···································································
            Made by:
                    Bruno L. S. Bedo, PhD 
                    Guilherme M. Cesar, PhD 
                    Paulo R. P. Santiago, PhD
            """)
    
    parser = argparse.ArgumentParser(description='Calculate and visualize coupling angle between two joints from a C3D file')

    parser.add_argument('--file', type=str, help='Path to the input file')
    parser.add_argument('--axis', type=str, default='x', help='Axis of interest for calculating the coupling angle')
    parser.add_argument('--joint1_name', type=str, default='Joint1', help='Name of the first joint')
    parser.add_argument('--joint2_name', type=str, default='Joint2', help='Name of the second joint')
    parser.add_argument('--format', type=str, default='c3d', help='File format (default is "c3d")')
    parser.add_argument('--save', type=bool, default=True, help='Flag indicating whether to save the results (default is True)')
    parser.add_argument('--savedir', type=str, default=None, help='Directory to save the results (default is current directory)')
    parser.add_argument('--savename', type=str, default='vectorcoding_result', help='Name of the output file (default is "vectorcoding_result")')

    args = parser.parse_args()

    get_coupling_angle(
        file=args.file,
        axis=args.axis,
        joint1_name=args.joint1_name,
        joint2_name=args.joint2_name,
        format=args.format,
        save=args.save,
        savedir=args.savedir,
        savename=args.savename
    )
