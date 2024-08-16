import maintools as tools


def display_joint_names_from_c3d(file_path):
    dict_kin = tools.get_kinematics_c3d(file_path)
    joint_names = list(dict_kin.keys())
    print("Available Joint Names in the C3D file:")
    for name in joint_names:
        print(name)


if __name__ == "__main__":
    file_path = input("Enter the path to the C3D file: ")
    display_joint_names_from_c3d(file_path)
