"""
Módulo: stabilogram_analysis.py
Descrição: Fornece funções para analisar dados de CoP e gerar plots de estabilograma, incluindo cálculos de RMS, velocidade, espectro de potência e densidade de oscilação.

Autor: Prof. Dr. Paulo R. P. Santiago
Versão: 1.0
Data: 2024-09-12

Referências:
- Repositório GitHub: Code Descriptors Postural Control. https://github.com/Jythen/code_descriptors_postural_control/blob/main/stabilogram/stato.py
- Liu, S., Zhai, P., Wang, L., Qiu, J., Liu, L., & Wang, H. (2021). A Review of Entropy-Based Methods in Postural Control Evaluation. Frontiers in Neuroscience, 15, 776326. https://www.ncbi.nlm.nih.gov/pmc/articles/PMC8623280/

Changelog:
- Versão 1.0 (2024-09-12):
  - Implementação inicial das funções de análise do estabilograma.
  - Incluídos cálculos para deslocamento RMS, velocidade, PSD e densidade de oscilação.
  - Adicionadas funções de plotagem para estabilograma e espectro de potência.

Uso:
- Importe o módulo e use as funções para realizar análises:
  from stabilogram_analysis import *
  rms_ml, rms_ap = compute_rms(cop_x, cop_y)
  speed_ml, speed_ap = compute_speed(cop_x, cop_y, fs)
  plot_stabilogram(cop_x, cop_y, output_path)
  # etc.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, welch, savgol_filter

def filter_signal(cop_x, cop_y, fs, cutoff=10, order=4):
    """
    Aplica um filtro passa-baixa de Butterworth aos sinais de CoP.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.
    - fs: float
        Frequência de amostragem em Hz.
    - cutoff: float, default=10
        Frequência de corte em Hz.
    - order: int, default=4
        Ordem do filtro.

    Retorna:
    - filtered_cop_x: array-like
        Dados filtrados de CoP na direção ML.
    - filtered_cop_y: array-like
        Dados filtrados de CoP na direção AP.
    """
    nyq = 0.5 * fs
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low', analog=False)
    filtered_cop_x = filtfilt(b, a, cop_x)
    filtered_cop_y = filtfilt(b, a, cop_y)
    return filtered_cop_x, filtered_cop_y

def compute_rms(cop_x, cop_y):
    """
    Calcula o deslocamento RMS nas direções ML e AP.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.

    Retorna:
    - rms_ml: float
        Deslocamento RMS na direção ML.
    - rms_ap: float
        Deslocamento RMS na direção AP.
    """
    rms_ml = np.sqrt(np.mean(cop_x ** 2))
    rms_ap = np.sqrt(np.mean(cop_y ** 2))
    return rms_ml, rms_ap

def compute_speed(cop_x, cop_y, fs, window_length=5, polyorder=3):
    """
    Calcula a velocidade do sinal de CoP usando o filtro de Savitzky-Golay.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.
    - fs: float
        Frequência de amostragem em Hz.
    - window_length: int, default=5
        Comprimento da janela do filtro (número de coeficientes).
    - polyorder: int, default=3
        Ordem do polinômio usado para ajustar as amostras.

    Retorna:
    - speed_ml: array-like
        Velocidade na direção ML.
    - speed_ap: array-like
        Velocidade na direção AP.
    """
    delta = 1 / fs
    speed_ml = savgol_filter(cop_x, window_length, polyorder, deriv=1, delta=delta)
    speed_ap = savgol_filter(cop_y, window_length, polyorder, deriv=1, delta=delta)
    return speed_ml, speed_ap

def compute_power_spectrum(cop_x, cop_y, fs):
    """
    Calcula a Densidade Espectral de Potência (PSD) dos sinais de CoP.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.
    - fs: float
        Frequência de amostragem em Hz.

    Retorna:
    - freqs_ml: array-like
        Frequências para PSD ML.
    - psd_ml: array-like
        Valores de PSD para a direção ML.
    - freqs_ap: array-like
        Frequências para PSD AP.
    - psd_ap: array-like
        Valores de PSD para a direção AP.
    """
    freqs_ml, psd_ml = welch(cop_x, fs=fs, nperseg=256)
    freqs_ap, psd_ap = welch(cop_y, fs=fs, nperseg=256)
    return freqs_ml, psd_ml, freqs_ap, psd_ap

def compute_sway_density(cop_x, cop_y, fs, radius=0.3):
    """
    Calcula a densidade de oscilação do sinal de CoP.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.
    - fs: float
        Frequência de amostragem em Hz.
    - radius: float, default=0.3
        Raio em cm para cálculo da densidade de oscilação.

    Retorna:
    - sway_density: array-like
        Valores de densidade de oscilação.
    """
    cop_signal = np.column_stack((cop_x, cop_y))
    n_samples = len(cop_signal)
    sway_density = np.zeros(n_samples)
    for t in range(n_samples):
        distances = np.linalg.norm(cop_signal - cop_signal[t], axis=1)
        sway_density[t] = np.sum(distances <= radius) / n_samples
    return sway_density

def plot_stabilogram(cop_x, cop_y, output_path):
    """
    Plota e salva o estabilograma.

    Parâmetros:
    - cop_x: array-like
        Dados de CoP na direção ML.
    - cop_y: array-like
        Dados de CoP na direção AP.
    - output_path: str
        Caminho para salvar o plot do estabilograma.
    """
    plt.figure(figsize=(10, 8))
    plt.plot(cop_x, cop_y, color='blue', linewidth=1)
    plt.title('Estabilograma')
    plt.xlabel('Deslocamento ML (cm)')
    plt.ylabel('Deslocamento AP (cm)')
    plt.grid(True)
    plt.axis('equal')
    plt.savefig(f"{output_path}_stabilogram.png", dpi=300)
    plt.close()

def plot_power_spectrum(freqs_ml, psd_ml, freqs_ap, psd_ap, output_path):
    """
    Plota e salva o espectro de potência dos sinais de CoP.

    Parâmetros:
    - freqs_ml: array-like
        Frequências para PSD ML.
    - psd_ml: array-like
        Valores de PSD para a direção ML.
    - freqs_ap: array-like
        Frequências para PSD AP.
    - psd_ap: array-like
        Valores de PSD para a direção AP.
    - output_path: str
        Caminho para salvar o plot do espectro de potência.
    """
    plt.figure(figsize=(10, 8))
    plt.semilogy(freqs_ml, psd_ml, label='ML')
    plt.semilogy(freqs_ap, psd_ap, label='AP')
    plt.xlabel('Frequência (Hz)')
    plt.ylabel('PSD (cm²/Hz)')
    plt.title('Densidade Espectral de Potência')
    plt.legend()
    plt.grid(True)
    plt.savefig(f"{output_path}_psd.png", dpi=300)
    plt.close()

def save_metrics_to_csv(metrics_dict, output_path):
    """
    Salva as métricas calculadas em um arquivo CSV.

    Parâmetros:
    - metrics_dict: dict
        Dicionário contendo as métricas para salvar.
    - output_path: str
        Caminho para salvar o arquivo CSV de métricas.
    """
    import pandas as pd
    df = pd.DataFrame([metrics_dict])
    df.to_csv(f"{output_path}_metrics.csv", index=False)

