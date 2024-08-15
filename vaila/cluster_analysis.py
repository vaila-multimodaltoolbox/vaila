import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
from tkinter import messagebox, filedialog, Tk
from vaila.filtering import apply_filter
from vaila.rotation import createortbase, calcmatrot, rotmat2euler
from vaila.plotting import plot_orthonormal_bases
from vaila.readcsv import get_csv_headers, select_headers_gui
from vaila.dialogsuser_cluster import get_user_inputs
from PIL import Image
from rich import print
#import ipdb

def save_results_to_csv(base_dir, time, cluster1_euler_angles, cluster2_euler_angles, file_name):
    results = {
        'time': time,
        'cluster1_euler_x': cluster1_euler_angles[:, 0],
        'cluster1_euler_y': cluster1_euler_angles[:, 1],
        'cluster1_euler_z': cluster1_euler_angles[:, 2],
        'cluster2_euler_x': cluster2_euler_angles[:, 0],
        'cluster2_euler_y': cluster2_euler_angles[:, 1],
        'cluster2_euler_z': cluster2_euler_angles[:, 2]
    }
    df = pd.DataFrame(results)
    base_name = os.path.splitext(file_name)[0]
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_file_path = os.path.join(base_dir, f'{base_name}_cluster_{timestamp}.csv')
    df.to_csv(result_file_path, index=False)

def read_anatomical_csv(file_path):
    try:
        data = pd.read_csv(file_path)
        cluster1_median = data[['cluster1_euler_x', 'cluster1_euler_y', 'cluster1_euler_z']].median().values
        cluster2_median = data[['cluster2_euler_x', 'cluster2_euler_y', 'cluster2_euler_z']].median().values
        return {'cluster1': cluster1_median, 'cluster2': cluster2_median}
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return None

def analyze_cluster_data():
    print("Starting analyze_cluster_data...")
    
    root = Tk()
    root.withdraw()  # Hide the main Tkinter window

    selected_path = filedialog.askdirectory(title="Select Directory with CSV Files")
    if not selected_path:
        messagebox.showerror("No Directory Selected", "No directory selected. Exiting.")
        root.destroy()
        return

    print(f"Selected path: {selected_path}")
    
    use_anatomical = messagebox.askyesno("Use Anatomical Angles", "Do you want to analyze with anatomical angle data?")
    
    print("""
         (A)               (B)

          * P2             * P3
        / |               | \\
    P3 *  |               |  * P2
        \\ |               | /
         * P1             * P1


         (C)               (D)

    P3 *-----* P2          * P2
        \\   /            /   \\
          * P1       P3 *-----* P1

    """)
    
    
    image_path = os.path.join('vaila', 'images', 'cluster_config.png')
    image = Image.open(image_path)
    plt.figure(figsize=(8, 6))
    plt.title("Memorize the cluster configuration (A/B/C/D) of the trunk (Cluster 1) and pelvis (Cluster 2)")
    plt.imshow(image)
    plt.axis('off')
    plt.show()

    print("Calling get_user_inputs...")
    user_input = get_user_inputs()
    print(f"User input: {user_input}")

    sample_rate = user_input.get('sample_rate')
    cluster1_config = user_input.get('cluster1_config')
    cluster2_config = user_input.get('cluster2_config')
    
    print(f"Use anatomical: {use_anatomical}")
    print(f"Sample rate: {sample_rate}")
    print(f"Cluster 1 config: {cluster1_config}")
    print(f"Cluster 2 config: {cluster2_config}")

    configurations = ["A", "B", "C", "D"]
    if cluster1_config not in configurations or cluster2_config not in configurations:
        messagebox.showerror("Invalid Input", "Invalid input for configuration. Please enter 'A', 'B', 'C', or 'D'.")
        root.destroy()
        return
    
    # Request the input file to select headers from
    selected_file = filedialog.askopenfilename(title="Pick file to select headers", filetypes=[("CSV files", "*.csv")])
    if not selected_file:
        messagebox.showerror("Error", "No file selected to choose headers.")
        root.destroy()
        return
    
    print(f"Selected file for headers: {selected_file}")
    
    headers = get_csv_headers(selected_file)
    selected_headers = select_headers_gui(headers)  # Remove the num_points argument
    
    print(f"Selected headers: {selected_headers}")

    filter_method = 'butterworth'
    file_names = sorted([f for f in os.listdir(selected_path) if f.endswith('.csv')])

    save_directory = filedialog.askdirectory(title="Choose Directory to Save Results")
    if not save_directory:
        messagebox.showerror("No Directory Selected", "No directory selected for saving results. Exiting.")
        root.destroy()
        return

    current_time = datetime.now().strftime("%Y%m%d_%H%M%S")
    base_dir_figures = os.path.join(save_directory, f'Cluster_{current_time}/figures')
    base_dir_processed_data = os.path.join(save_directory, f'Cluster_{current_time}/processed_data')
    os.makedirs(base_dir_figures, exist_ok=True)
    os.makedirs(base_dir_processed_data, exist_ok=True)

    print(f"Base directories created: {base_dir_figures}, {base_dir_processed_data}")
    
    anatomical_data = {}
    if use_anatomical:
        anatomical_file_path = filedialog.askopenfilename(title="Select Anatomical Euler Angles CSV File", filetypes=[("CSV files", "*.csv")])
        if not anatomical_file_path:
            messagebox.showerror("No File Selected", "No anatomical data file selected. Exiting.")
            root.destroy()
            return
        
        anat_data = read_anatomical_csv(anatomical_file_path)
        if anat_data:
            anatomical_data = anat_data
        else:
            messagebox.showerror("Error Reading File", "Failed to read anatomical data file. Exiting.")
            root.destroy()
            return

    print(f"Anatomical data: {anatomical_data}")
    matplotlib_figs = []  # List to store matplotlib figures
    for idx, file_name in enumerate(file_names):
        print(f"Processing file: {file_name}")
        file_path = os.path.join(selected_path, file_name)
        data = pd.read_csv(file_path, usecols=selected_headers).values
        
        print(f"Data shape: {data.shape}")
        
        # Create time vector
        time = np.linspace(0, len(data) / sample_rate, len(data))
        print(f"Time vector created with length: {len(time)}")
        
        # Insert time column in the data in the first column
        data = np.insert(data, 0, time, axis=1)
        
        print(f"Data after time column adjustments: {data.shape}")
        
        # Apply filter to the data
        dataf = apply_filter(data[:, 1:], sample_rate, method=filter_method)
        
        print(f"Data after filtering: {dataf.shape}")
        
        # Extract points based on user input
        points = [dataf[:, i:i+3] for i in range(0, 18, 3)]
        cluster1_p1, cluster1_p2, cluster1_p3, cluster2_p1, cluster2_p2, cluster2_p3 = points
        
        print("Points extracted for clusters")
        
        blab = np.array([[1., 0., 0.], [0., 1., 0.], [0., 0., 1.]])

        cluster1_base, orig_cluster1 = createortbase(cluster1_p1, cluster1_p2, cluster1_p3, cluster1_config)
        cluster2_base, orig_cluster2 = createortbase(cluster2_p1, cluster2_p2, cluster2_p3, cluster2_config)
        
        print("Orthonormal bases created")

        # # Add more debug prints here
        print("cluster1:", cluster1_base)
        print("orig_cluster1:", orig_cluster1)
        print("cluster2_base:", cluster2_base)
        print("orig_cluster2:", orig_cluster2)
        print("cluster1_p1:", cluster1_p1)
        print("cluster1_p2:", cluster1_p2)
        print("cluster1_p3:", cluster1_p3)
        print("cluster2_p1:", cluster2_p1)
        print("cluster2_p2:", cluster2_p2)
        print("cluster2_p3:", cluster2_p3)

        fig_matplotlib = plot_orthonormal_bases(
            bases_list=[cluster1_base, cluster2_base],
            pm_list=[orig_cluster1, orig_cluster2],
            points_list=[[cluster1_p1, cluster1_p2, cluster1_p3], [cluster2_p1, cluster2_p2, cluster2_p3]],
            labels=['Cluster1', 'Cluster2'],
            title=f'Cluster Bases - {file_name}',
            global_coordinate_system=None,
            four_points=False
        )

        
        matplotlib_figs.append(fig_matplotlib)

        cluster1_rotmat = calcmatrot(cluster1_base, blab)
        cluster2_rotmat = calcmatrot(cluster2_base, blab)

        cluster1_euler_angles = rotmat2euler(cluster1_rotmat)
        cluster2_euler_angles = rotmat2euler(cluster2_rotmat)

        if use_anatomical and anatomical_data:
            cluster1_euler_angles_anat = anatomical_data['cluster1']
            cluster2_euler_angles_anat = anatomical_data['cluster2']
            cluster1_euler_angles -= cluster1_euler_angles_anat
            cluster2_euler_angles -= cluster2_euler_angles_anat

        max_val = np.max([cluster1_euler_angles, cluster2_euler_angles])
        min_val = np.min([cluster1_euler_angles, cluster2_euler_angles])

        fig, axes = plt.subplots(3, 1, figsize=(10, 12))

        name_cluster1 = 'trunk'
        name_cluster2 = 'pelvis'# Add cluster2 name here of dialog box
        
        axes[0].plot(time, cluster1_euler_angles[:, 0], label=f'{name_cluster1} X [Ext. backward (+) Flex. forward (-)]', color='red')
        axes[0].plot(time, cluster2_euler_angles[:, 0], label=f'{name_cluster2} X [Ext. backward (+) Flex. forward (-)]', linestyle='--', color='red')
        if use_anatomical and anatomical_data:
            axes[0].axhline(cluster1_euler_angles_anat[0], color='gray', linestyle='-', label='Anatomical Cluster1 X')
            axes[0].axhline(cluster2_euler_angles_anat[0], color='gray', linestyle='--', label='Anatomical Cluster2 X')
        axes[0].set_xlabel('Time (s)')
        axes[0].set_ylabel('X: Ext(+)/Flex(-) (degrees)')
        axes[0].legend()
        axes[0].set_title(f'Euler Angles - {file_name} (X-axis)')

        axes[1].plot(time, cluster1_euler_angles[:, 1], label=f'{name_cluster1} Y [Side Bending Right (+) Side Bending Left (-)]', color='green')
        axes[1].plot(time, cluster2_euler_angles[:, 1], label=f'{name_cluster2} Y [Side Bending Right (+) Side Bending Left (-)]', linestyle='--', color='green')
        if use_anatomical and anatomical_data:
            axes[1].axhline(cluster1_euler_angles_anat[1], color='gray', linestyle='-', label='Anatomical Cluster1 Y')
            axes[1].axhline(cluster2_euler_angles_anat[1], color='gray', linestyle='--', label='Anatomical Cluster2 Y')
        axes[1].set_xlabel('Time (s)')
        axes[1].set_ylabel('Y: Side Bending R(+)/L(-) (degrees)')
        axes[1].legend()
        axes[1].set_title(f'Euler Angles - {file_name} (Y-axis)')

        axes[2].plot(time, cluster1_euler_angles[:, 2], label=f'{name_cluster1} Z [Axial Rot. Right (+) Axial Rot. Left (-)]', color='blue')
        axes[2].plot(time, cluster2_euler_angles[:, 2], label=f'{name_cluster2} Z [Axial Rot. Right (+) Axial Rot. Left (-)]', linestyle='--', color='blue')
        if use_anatomical and anatomical_data:
            axes[2].axhline(cluster1_euler_angles_anat[2], color='gray', linestyle='-', label='Anatomical Cluster1 Z')
            axes[2].axhline(cluster2_euler_angles_anat[2], color='gray', linestyle='--', label='Anatomical Cluster2 Z')
        axes[2].set_xlabel('Time (s)')
        axes[2].set_ylabel('Z: Axial Rot. R(+)/L(-) (degrees)')
        axes[2].legend()
        axes[2].set_title(f'Euler Angles - {file_name} (Z-axis)')

        margin = 0.05 * (max_val - min_val)
        axes[0].set_ylim([min_val - margin, max_val + margin])
        axes[1].set_ylim([min_val - margin, max_val + margin])
        axes[2].set_ylim([min_val - margin, max_val + margin])

        plt.tight_layout()

        base_name = os.path.splitext(file_name)[0]
        fig_path = os.path.join(base_dir_figures, f'{base_name}_figure.png')
        plt.savefig(fig_path)

        save_results_to_csv(base_dir_processed_data, time, cluster1_euler_angles, cluster2_euler_angles, base_name)

        print(f"Finished processing file: {file_name}")

    for fig in matplotlib_figs:
        fig.show()

    root.destroy()

if __name__ == "__main__":
    analyze_cluster_data()
