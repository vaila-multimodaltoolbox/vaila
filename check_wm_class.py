
import tkinter
import sys

try:
    root = tkinter.Tk(className='Vaila')
    # Print the class name
    print(f"winfo_class: {root.winfo_class()}")
    print(f"winfo_name: {root.winfo_name()}")
except Exception as e:
    print(f"Error: {e}")
