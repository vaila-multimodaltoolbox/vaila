"""
================================================================================
Native File Dialog Module - native_file_dialog.py
================================================================================
vailá - Multimodal Toolbox
Author: Prof. Dr. Paulo R. P. Santiago
https://github.com/paulopreto/vaila-multimodaltoolbox
Date: 16 January 2025
Version: 0.1.0
Python Version: 3.12.11

Description:
------------
This module provides native file dialogs that work without conflicts with Pygame.
It uses system-native dialogs via subprocess to avoid event loop conflicts.

Usage:
------
from native_file_dialog import open_native_file_dialog, open_yes_no_dialog

# Open file dialog
file_path = open_native_file_dialog(
    title="Select File",
    file_types=[("*.csv", "CSV Files"), ("*.txt", "Text Files")]
)

# Open yes/no dialog
result = open_yes_no_dialog(
    title="Question",
    message="Do you want to continue?"
)

================================================================================
"""

import subprocess
import sys
import os


def open_native_file_dialog(title="Select File", file_types=None):
    """
    Open native file dialog without blocking Pygame.
    
    Args:
        title (str): Dialog title
        file_types (list): List of tuples like [("*.csv", "CSV Files")]
    
    Returns:
        str: Selected file path or None if cancelled
    """
    if file_types is None:
        file_types = [("*.*", "All Files")]
    
    try:
        if sys.platform == 'win32':
            # Windows - PowerShell
            filters = '|'.join([f'{desc} ({pat})|{pat}' for pat, desc in file_types])
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dlg = New-Object System.Windows.Forms.OpenFileDialog; "
                f"$dlg.Title = '{title}'; "
                f"$dlg.Filter = '{filters}'; "
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None
            
        elif sys.platform == 'darwin':
            # macOS - osascript
            extensions = ','.join([pat.replace('*.', '') for pat, _ in file_types])
            script = f'''
            set theFile to choose file with prompt "{title}" of type {{{extensions}}}
            return POSIX path of theFile
            '''
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            path = result.stdout.strip()
            return path if path else None
            
        else:
            # Linux - zenity (needs to be installed)
            filters = ' '.join([f'--file-filter="{desc}|{pat}"' for pat, desc in file_types])
            command = f'zenity --file-selection --title="{title}" {filters}'
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None
            
    except Exception as e:
        print(f"Error opening dialog: {e}")
        return None


def open_yes_no_dialog(title="Question", message="Continue?"):
    """
    Open native yes/no dialog.
    
    Args:
        title (str): Dialog title
        message (str): Dialog message
    
    Returns:
        bool: True for Yes, False for No
    """
    try:
        if sys.platform == 'win32':
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$result = [System.Windows.Forms.MessageBox]::Show("
                f"'{message}', '{title}', 'YesNo', 'Question'); "
                "if ($result -eq 'Yes') { 'yes' }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            return 'yes' in result.stdout.strip().lower()
            
        elif sys.platform == 'darwin':
            script = f'''
            display dialog "{message}" with title "{title}" buttons {{"No", "Yes"}} default button "Yes"
            if button returned of result is "Yes" then return "yes" else return "no"
            '''
            result = subprocess.run(['osascript', '-e', script], capture_output=True, text=True)
            return 'yes' in result.stdout.strip().lower()
            
        else:
            command = f'zenity --question --title="{title}" --text="{message}"'
            result = subprocess.run(command, shell=True)
            return result.returncode == 0
            
    except Exception as e:
        print(f"Error in dialog: {e}")
        return False


def open_save_file_dialog(title="Save File", file_types=None, default_name=""):
    """
    Open native save file dialog.
    
    Args:
        title (str): Dialog title
        file_types (list): List of tuples like [("*.csv", "CSV Files")]
        default_name (str): Default filename
    
    Returns:
        str: Selected file path or None if cancelled
    """
    if file_types is None:
        file_types = [("*.*", "All Files")]
    
    try:
        if sys.platform == 'win32':
            # Windows - PowerShell
            filters = '|'.join([f'{desc} ({pat})|{pat}' for pat, desc in file_types])
            command = (
                'powershell -Command "'
                "Add-Type -AssemblyName System.Windows.Forms; "
                "$dlg = New-Object System.Windows.Forms.SaveFileDialog; "
                f"$dlg.Title = '{title}'; "
                f"$dlg.Filter = '{filters}'; "
                f"$dlg.FileName = '{default_name}'; "
                "if ($dlg.ShowDialog() -eq 'OK') { $dlg.FileName }"
                '"'
            )
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None
            
        elif sys.platform == 'darwin':
            # macOS - osascript
            extensions = ','.join([pat.replace('*.', '') for pat, _ in file_types])
            script = f'''
            set theFile to choose file name with prompt "{title}" default name "{default_name}"
            return POSIX path of theFile
            '''
            result = subprocess.run(
                ['osascript', '-e', script],
                capture_output=True,
                text=True
            )
            path = result.stdout.strip()
            return path if path else None
            
        else:
            # Linux - zenity
            filters = ' '.join([f'--file-filter="{desc}|{pat}"' for pat, desc in file_types])
            command = f'zenity --file-selection --save --title="{title}" --filename="{default_name}" {filters}'
            result = subprocess.run(command, capture_output=True, text=True, shell=True)
            path = result.stdout.strip()
            return path if path else None
            
    except Exception as e:
        print(f"Error opening save dialog: {e}")
        return None
