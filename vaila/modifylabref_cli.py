import os
import numpy as np
import pandas as pd
from datetime import datetime
from vaila.rotation import rotdata

def modify_lab_coords(data, labcoord_angles):
    if labcoord_angles:
        data = rotdata(data.T, xth=labcoord_angles[0], yth=labcoord_angles[1], zth=labcoord_angles[2], ordem='xyz').T
    return data

def get_labcoord_angles(option):
    if option == 'A':
        return (0, 0, 180), '_rot180'
    elif option == 'B':
        return (0, 0, 90), '_rot90clock'
    elif option == 'C':
        return (0, 0, -90), '_rot90cclock'
    else:
        try:
            custom_base = np.array(eval(option))
            if custom_base.shape == (3, 3):
                return None, '_custom'  # No need for angles if custom base is used
            else:
                raise ValueError("Custom base matrix must be 3x3.")
        except (SyntaxError, NameError, ValueError) as e:
            print(f"Invalid base orientation input: {e}. Using canonical base.")
            return None, '_canonical'

def process_files(input_dir, output_dir, labcoord_angles, suffix):
    os.makedirs(output_dir, exist_ok=True)
    file_names = [f for f in os.listdir(input_dir) if f.endswith('.csv')]

    for file_name in file_names:
        file_path = os.path.join(input_dir, file_name)
        data = pd.read_csv(file_path)

        # Verificar se a coluna "Time" está presente em qualquer variação de maiúsculas e minúsculas
        time_col_present = any(col.lower() == 'time' for col in data.columns)
        if not time_col_present:
            print(f"Error: Column 'Time' not found in {file_name}. Please include a 'Time' column.")
            continue

        # Renomear a coluna de tempo para garantir que tenha um nome consistente
        data.rename(columns={col: 'Time' for col in data.columns if col.lower() == 'time'}, inplace=True)

        modified_data = data.copy()
        time_column = data['Time']

        for i in range(1, len(data.columns), 3):
            points = data.iloc[:, i:i+3].values
            modified_points = modify_lab_coords(points, labcoord_angles)
            modified_data.iloc[:, i:i+3] = modified_points

        if 'Time' not in modified_data.columns:
            modified_data.insert(0, 'Time', time_column)

        # Obter o número de casas decimais dos dados originais
        float_format = '%.6f'  # Padrão para 6 casas decimais
        sample_value = data.iloc[0, 1]
        if isinstance(sample_value, float):
            decimal_places = len(str(sample_value).split('.')[1])
            float_format = f'%.{decimal_places}f'

        base_name, ext = os.path.splitext(file_name)
        output_file_path = os.path.join(output_dir, f"{base_name}{suffix}{ext}")
        modified_data.to_csv(output_file_path, index=False, float_format=float_format)
        print(f"\n*** Processed and saved: {output_file_path} ***\n")

    print("\n" + "*" * 50)
    print("     All files have been processed and saved successfully!     ")
    print("*" * 50 + "\n")

def run_modify_labref():
    input_dir = './data/csv_2_modify_labcoordsystem/'
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'./data/csv_2_modify_labcoordsystem/newlabcoordsys_{timestamp}/'

    print(
        "Laboratory Coordinate System Modifications:\n"
        "(A) Rotate 180 degrees in Z axis\n"
        "(B) Rotate 90 degrees clockwise in Z axis\n"
        "(C) Rotate 90 degrees counter clockwise in Z axis"
    )

    labcoord_option = input("Choose the lab coordinate modification option (A/B/C) or enter a custom base orientation: ").strip().upper()
    labcoord_angles, suffix = get_labcoord_angles(labcoord_option)
    
    if labcoord_angles is None and labcoord_option not in ['A', 'B', 'C']:
        print("Invalid base orientation input. Using canonical base.")
        labcoord_angles = (0, 0, 0)
        suffix = '_canonical'

    process_files(input_dir, output_dir, labcoord_angles, suffix)

if __name__ == "__main__":
    run_modify_labref()
