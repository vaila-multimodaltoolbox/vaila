#!/usr/bin/env python3
"""
Script to set macOS application icons using AppKit.

This script must be run on macOS and requires pyobjc-framework-Cocoa.
Usage: python set_mac_icon.py <target_path> <icon_path>

Update: 11 January 2026
"""

import importlib.util
import os
import platform
import sys


def check_prerequisites():
    """
    Check if the script can run on this system.
    Returns (can_run, error_message)
    """
    if platform.system() != "Darwin":
        return False, "This script only works on macOS."

    # Check if Cocoa module is available without importing it
    if importlib.util.find_spec("Cocoa") is None:
        return False, (
            "Error: pyobjc-framework-Cocoa is not installed.\n"
            "Please install it with: uv pip install pyobjc-framework-Cocoa"
        )

    return True, None


def set_icon(target_path, icon_path):
    """
    Sets the icon of a file or folder on macOS using AppKit.

    Args:
        target_path: Path to the file/folder to set the icon for
        icon_path: Path to the .icns icon file

    Returns:
        bool: True if successful, False otherwise
    """
    # Check prerequisites
    can_run, error_msg = check_prerequisites()
    if not can_run:
        print(error_msg, file=sys.stderr)
        return False

    # Import Cocoa (already verified in check_prerequisites)
    try:
        import Cocoa
    except ImportError:
        print("Error: pyobjc-framework-Cocoa is not installed.", file=sys.stderr)
        print("Please install it with: uv pip install pyobjc-framework-Cocoa", file=sys.stderr)
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
    # Check if script is being executed (not imported)
    script_path = os.path.abspath(__file__)
    if not os.path.exists(script_path):
        print(f"Warning: Script file not found at {script_path}", file=sys.stderr)
        print("This script should be executed from the project root directory.", file=sys.stderr)

    if len(sys.argv) != 3:
        print("Usage: python set_mac_icon.py <target_path> <icon_path>", file=sys.stderr)
        print(
            f"Example: python {os.path.basename(script_path)} /path/to/app.app /path/to/icon.icns",
            file=sys.stderr,
        )
        sys.exit(1)

    target = sys.argv[1]
    icon = sys.argv[2]

    if set_icon(target, icon):
        sys.exit(0)
    else:
        sys.exit(1)
