import os
import platform
import sys


def set_icon(target_path, icon_path):
    """
    Sets the icon of a file or folder on macOS using AppKit.
    """
    if platform.system() != "Darwin":
        print("Error: This script only works on macOS.")
        return False

    try:
        import Cocoa
    except ImportError:
        print("Error: pyobjc-framework-Cocoa is not installed.")
        print("Please install it with: uv pip install pyobjc-framework-Cocoa")
        return False

    # Expand user paths
    target_path = os.path.expanduser(target_path)
    icon_path = os.path.expanduser(icon_path)

    if not os.path.exists(target_path):
        print(f"Error: Target path does not exist: {target_path}")
        return False

    if not os.path.exists(icon_path):
        print(f"Error: Icon path does not exist: {icon_path}")
        return False

    try:
        # Load the image
        image = Cocoa.NSImage.alloc().initWithContentsOfFile_(icon_path)
        if not image:
            print(f"Error: Failed to load image from {icon_path}")
            return False

        # Set the icon
        workspace = Cocoa.NSWorkspace.sharedWorkspace()
        success = workspace.setIcon_forFile_options_(image, target_path, 0)

        if success:
            print(f"Successfully set icon for: {target_path}")
            return True
        else:
            print(f"Failed to set icon for: {target_path}")
            return False

    except Exception as e:
        print(f"An error occurred: {e}")
        return False


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Usage: python set_mac_icon.py <target_path> <icon_path>")
        sys.exit(1)

    target = sys.argv[1]
    icon = sys.argv[2]

    if set_icon(target, icon):
        sys.exit(0)
    else:
        sys.exit(1)
