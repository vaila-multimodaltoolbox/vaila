import os
import sys
from multimodal_mocap_coord_toolbox import (
    cluster_analysis,
    imu_analysis,
    markerless_2D_analysis,
    markerless_3D_analysis,
    mocap_analysis,
    convert_c3d_to_csv,
    rearrange_data_in_directory,
    modifylabref,
    batch_cut_videos,
    run_drawboxe,
    run_compress_videos,
    count_frames_in_videos,
    export_file, copy_file, move_file, remove_file, import_file,
    show_c3d
)

def display_header():
    header = r"""
************************ MultiModal-Mocap-Coord-Toolbox **************************
    __  ___   ________                                       __  _______ ____
   / / / / | / / ____/                                     / / / / ___// __  \
  / / / /  |/ / /_                                        / / / /\__ \/ /_/ /
 / /_/ / /|  / __/                                       / /_/ /___/ / ____/
 \____/_/ |_/_/                                         \____//____/_/
**********************************************************************************
Mocap_fullbody_c3d        Markerless_3D_videos       Markerless_2D_video
                  \                |                /
                   v               v               v
            +-------------------------------------------+
IMU_csv --> |             Multimodal Toolbox            | <-- Cluster_csv
            +-------------------------------------------+
                                  |
                                  v
                   +------------------------------+
                   |       Angle Calculation      |
                   +------------------------------+
                                  |
                                  v
                         +-----------------+
                         |  Vector Coding  |
                         +-----------------+
"""
    print(header)
    
def display_menu():
    print("============================ File Manager ===============================")
    print(" Import (im)  |  Export (ex)  |  Copy (cp)  |  Move (mv)  |  Remove (rm)")
    print("==================== Available Multimodal Analysis ======================")
    print("1. IMU Analysis")
    print("2. Kinematic Cluster Analysis")
    print("3. Kinematic Mocap Full Body Analysis")
    print("4. Markerless 2D with video")
    print("5. Markerless 3D with multiple videos")
    print("============================ Available Tools =============================")
    print("6. Reorder CSV data columns")
    print("7. Convert C3D data to CSV")
    print("8. Modify Laboratory Reference System")
    print("9. Count frames in videos")
    print("10. Cut videos based on list")
    print("11. Draw a black box around videos")
    print("12. Compress videos to HEVC (H.265)")
    print("13. Show C3D file data")

    print("\nType 'h' for help or 'exit' to quit.")
    
def display_help():
    help_file_path = './multimodal_mocap_coord_toolbox/help.txt'
    if os.path.exists(help_file_path):
        with open(help_file_path, 'r') as help_file:
            help_text = help_file.read()
        print(help_text)
    else:
        print("Help file not found.")

def handle_file_manager(command):
    if command in ['im', 'ex', 'cp', 'mv']:
        if command == 'im':
            import_file()
        elif command in ['ex', 'cp']:
            src = input("Enter the source path: ").strip()
            dest = input("Enter the destination path: ").strip()

            if command == 'ex':
                export_file(src, dest)
            elif command == 'cp':
                copy_file(src, dest)
        elif command == 'mv':
            move_file()  # Chama a função move_file sem argumentos
    elif command == 'rm':
        remove_file()
    else:
        print("Invalid file manager command.")

def handle_choice(choice):
    directory_mappings = {
        1: 'imu_csv',
        2: 'cluster_csv',
        3: 'mocap_fullbody',
        4: 'markerless_2D_video',
        5: 'markerless_3D_videos',
        6: 'rearrange_data',
        7: 'c3d_2_convert_in_csv',
        8: 'csv_2_modify_labcoordsystem'
    }

    if choice in directory_mappings:
        selected_directory = directory_mappings[choice]
        selected_path = os.path.join('./data', selected_directory)
        
        if choice == 1:
            imu_analysis.analyze_imu_data(selected_path)
        elif choice == 2:
            use_anatomical = input("Do you want to analyze with anatomical position data? (y/n): ").strip().lower() == 'y'
            cluster_analysis.analyze_cluster_data(selected_path, use_anatomical)
        elif choice == 3:
            use_anatomical = input("Do you want to analyze with anatomical position data? (y/n): ").strip().lower() == 'y'
            mocap_analysis.analyze_mocap_fullbody_data(selected_path, use_anatomical)
        elif choice == 4:
            markerless_2D_analysis.process_videos_in_directory(selected_path, os.path.join(selected_path, 'working'), pose_config={
                'min_detection_confidence': 0.1,
                'min_tracking_confidence': 0.1
            })
        elif choice == 5:
            markerless_3D_analysis.analyze_markerless_3D_data(selected_path)
        elif choice == 6:
            rearrange_data_in_directory(selected_path)
        elif choice == 7:
            convert_c3d_to_csv(selected_path, selected_path)
        elif choice == 8:
            modifylabref.run_modify_labref()
            print(f"Analysis for {selected_directory} completed.")
        elif choice == 9:
            count_frames_in_videos(selected_path)
        elif choice == 10:
            video_directory = './data/videos_2_edition/'
            batch_cut_videos(video_directory)
        elif choice == 11:
            run_drawboxe()
        elif choice == 12:
            run_compress_videos('./data/videos_2_edition/')
        elif choice == 13:
            show_c3d()
        else:
            print("Invalid choice. Please select a number from the menu.")

def main():
    display_header()

    while True:
        display_menu()
        choice = input("\nChoose an analysis option or file manager command: ").strip().lower()

        if choice == 'exit':
            print("Exiting the MULTIMODAL_TOOLBOX. Goodbye!\n")
            break
        elif choice == 'h':
            display_help()
        elif choice in ['im', 'ex', 'cp', 'mv', 'rm']:
            handle_file_manager(choice)
        elif choice.isdigit():
            handle_choice(int(choice))
        else:
            print("Invalid input. Please enter a valid command or number.")

if __name__ == "__main__":
    main()
