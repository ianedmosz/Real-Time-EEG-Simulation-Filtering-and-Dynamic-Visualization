import numpy as np
from scipy.signal import lfilter
from colorednoise import powerlaw_psd_gaussian
import matplotlib.pyplot as plt
import json
import os

class EEG_Generator:
    def __init__(self, n_samples=5000, fs=256, seed=42):
        np.random.seed(seed)
        self.n_samples = n_samples
        self.fs = fs
        self.t = np.linspace(0, n_samples / fs, n_samples)   # segundos: 19.53125
    
    def generate_ar_model(self, order, phi, sigma):
        epsilon = np.random.normal(0, sigma, self.n_samples)
        ar_coeffs = [1] + [-coef for coef in phi]
        S_AR = lfilter([1], ar_coeffs, epsilon)
        return S_AR
    
    def generate_fractal_noise(self, beta):
        N_fractal = powerlaw_psd_gaussian(beta, self.n_samples)
        return N_fractal
    
    def generate_sinusoidal(self, frequencies, amplitudes):
        S_sinusoidal = np.zeros_like(self.t)
        for band, freq in frequencies.items():
            amplitude = amplitudes[band]
            phase = np.random.uniform(0, 2 * np.pi)
            S_sinusoidal += amplitude * np.sin(2 * np.pi * freq * self.t + phase)
        return S_sinusoidal
    
    def generate_combined_signal(self, S_AR, N_fractal, S_sinusoidal):
        return S_AR + N_fractal + S_sinusoidal
    
    def generate_full_signal(self, order, phi, sigma, beta, frequencies, amplitudes):
        S_AR = self.generate_ar_model(order, phi, sigma)
        N_fractal = self.generate_fractal_noise(beta)
        S_sinusoidal = self.generate_sinusoidal(frequencies, amplitudes)
        return self.generate_combined_signal(S_AR, N_fractal, S_sinusoidal)

    def generate_signals_for_channels(self, num_channels, orders, phis, sigmas, betas, frequencies, amplitudes):
        signals = []
        for i in range(num_channels):
            signal = self.generate_full_signal(
                order=orders[i],
                phi=phis[i],
                sigma=sigmas[i],
                beta=betas[i],
                frequencies=frequencies[i],
                amplitudes=amplitudes[i]
            )
            signals.append(signal)
        return np.array(signals)  # Cada fila es un canal


base_dir = os.path.dirname(__file__)
config_path = os.path.join(base_dir, "config.json")

# Cargar configuración desde JSON
with open(config_path, "r") as file:
    config = json.load(file)
    
# Inicializar el simulador con configuraciones del JSON
simulator = EEG_Generator(
    n_samples=config["simulator"]["n_samples"],
    fs=config["simulator"]["fs"],
    seed=config["simulator"]["seed"]
)

num_channels = config["channels"]
orders = config["orders"]
phis = config["phis"]
sigmas = config["sigmas"]
betas = config["betas"]

# Extraer frecuencias y amplitudes dinámicamente
frequencies = [
    {band: np.random.uniform(*range_values) for band, range_values in freq.items()}
    for freq in config["frequencies"]
]

amplitudes = [
    {band: np.random.uniform(*range_values) for band, range_values in amp.items()}
    for amp in config["amplitudes"]
]

# Generar señales para los canales
signals = simulator.generate_signals_for_channels(num_channels, orders, phis, sigmas, betas, frequencies, amplitudes)

# Guardar señales simuladas en un archivo CSV
def save_signals_to_csv(signals, filename, fs):
    """
    Guarda las señales simuladas en un archivo CSV.
    
    :param signals: ndarray, señales EEG simuladas (canales x muestras)
    :param filename: str, nombre del archivo CSV
    :param fs: int, frecuencia de muestreo
    """
    import csv
    time = np.arange(signals.shape[1]) / fs  # Generar vector de tiempo
    header = ['Time (s)'] + [f"Channel {i+1}" for i in range(signals.shape[0])]

    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(header)  # Escribir encabezados
        for i in range(signals.shape[1]):
            row = [time[i]] + list(signals[:, i])  # Tiempo + valores de canales
            writer.writerow(row)

save_signals_to_csv(signals, config.get("output_filename", "simulated_eeg_signals.csv"), simulator.fs)

# Visualizar señales generadas
plt.figure(figsize=(12, 8))
for i, signal in enumerate(signals):
    plt.plot(simulator.t, signal, label=f"Canal {i+1}")
plt.title("Señales EEG Simuladas para Múltiples Canales")
plt.xlabel("Tiempo (s)")
plt.ylabel("Amplitud")
plt.legend()
plt.grid()
plt.tight_layout()
plt.show()
