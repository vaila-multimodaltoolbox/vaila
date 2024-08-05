import matplotlib.pyplot as plt
import pandas as pd
from tkinter import Tk, Button, filedialog, Toplevel, Checkbutton, BooleanVar, Canvas, Scrollbar, Frame, messagebox
from spm1d import stats

# Variáveis globais para armazenar as seleções do usuário
selected_files = []
selected_headers = []
plot_type = None


#if __name__ == "__main__":
    #plot_3d()