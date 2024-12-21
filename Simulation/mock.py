import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
import csv
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d  # Import for Gaussian filter
from RF_AR import EEG_Generator
import matplotlib.pyplot as plt

filename = input("Enter the filename: ")+".csv"
# Filter Configuration
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

class RealTimeFilter:
    def __init__(self, lowcut, highcut, fs, order=5, buffer_size=500, gaussian_sigma=5):
        self.b_band, self.a_band = butter_bandpass(lowcut, highcut, fs, order=order)
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros(buffer_size)
        self.gaussian_sigma = gaussian_sigma  # Standard deviation for Gaussian filter
        self.zi_band = np.zeros(max(len(self.a_band), len(self.b_band)) - 1)
        self.index = 0

    def update_buffer(self, new_sample):
        # Apply bandpass filter
        band_filtered_value, self.zi_band = lfilter(self.b_band, self.a_band, [new_sample], zi=self.zi_band)
        # Add the filtered value to the buffer
        self.data_buffer[self.index] = band_filtered_value[-1]
        self.index = (self.index + 1) % self.buffer_size
        # Apply Gaussian smoothing to the buffer
        smoothed_value = gaussian_filter1d(self.data_buffer, sigma=self.gaussian_sigma)
        return smoothed_value[(self.index - 1) % self.buffer_size]
class MockSerial:
    def __init__(self, num_channels, fs):
        # Crear instancia del generador de señales
        self.simulator = EEG_Generator(n_samples=fs, fs=fs)

        # Parámetros de simulación (estáticos)
        self.num_channels = num_channels
        self.orders = [4] * num_channels  # Orden fijo para todos los canales
        self.phis = [[0.9, -0.5, 0.3, -0.1] for _ in range(num_channels)]  # Coeficientes AR
        self.sigmas = [0.5] * num_channels  # Nivel de ruido blanco (desviación estándar)

        # Parámetros de simulación (aleatorios con rangos definidos)
        self.betas = np.random.uniform(0.8, 1.2, num_channels)  # Exponentes de ruido fractal
        self.frequencies = [
            {
                "delta": np.random.uniform(0.5, 4),
                "theta": np.random.uniform(4, 8),
                "alpha": np.random.uniform(8, 14),
                "beta": np.random.uniform(14, 30),
                "gamma": np.random.uniform(30, 100)
            }
            for _ in range(num_channels)
        ]
        self.amplitudes = [
            {
                "delta": np.random.uniform(20e-6, 200e-6),
                "theta": np.random.uniform(10e-6, 50e-6),
                "alpha": np.random.uniform(10e-6, 50e-6),
                "beta": np.random.uniform(5e-6, 20e-6),
                "gamma": np.random.uniform(1e-6, 10e-6)
            }
            for _ in range(num_channels)
        ]

        # Generar señales para múltiples canales
        self.data = self.simulator.generate_signals_for_channels(
            num_channels,
            self.orders,
            self.phis,
            self.sigmas,
            self.betas,
            self.frequencies,
            self.amplitudes
        )
        self.index = 0

    def readline(self):
        if self.index >= len(self.data[0]):  # Fin del buffer
            self.index = 0
        line = ",".join(map(str, self.data[:, self.index]))
        self.index += 1
        return f"{line}\n".encode('utf-8')

    def in_waiting(self):
        return True

# Replace Serial with MockSerial
num_channels = 4
fs = 265  # Sampling frequency
mock_serial = MockSerial(num_channels=num_channels, fs=fs)

# EEG channel configuration
channel_positions = ['FP1', 'FP2', 'C3', 'C4']
filters = [RealTimeFilter(1.0, 50.0, fs, order=5, gaussian_sigma=5) for _ in range(num_channels)]
y_data = np.zeros((num_channels, fs))
power_data = np.zeros((num_channels, fs))

# PyQtGraph Configuration
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title=f"EEG Beta 12-30Hz Real-Time and Power")
win.resize(1000, 800)
win.setWindowTitle('PyQtGraph Real-time EEG')

# Create plots dynamically based on the number of channels
plots = []
curves = []
for i in range(num_channels):
    plot = win.addPlot(title=f"Filtered EEG Signal - {channel_positions[i]}")
    curve = plot.plot(np.linspace(0, 1, fs), y_data[i], pen='b')
    plots.append(plot)
    curves.append(curve)
    win.nextRow()

power_curves = []
for i in range(num_channels):
    power_plot = win.addPlot(title=f"Power - {channel_positions[i]}")
    power_curve = power_plot.plot(np.linspace(0, 1, fs), power_data[i], pen='r')
    power_curves.append(power_curve)
    win.nextRow()


window_size = 50

def update():
    global y_data, power_data

    if True:  # Simulando `ser.in_waiting > 0`
        serial_line = mock_serial.readline().decode('utf-8').strip()
        try:
            # Procesar valores para múltiples canales
            channels = list(map(float, serial_line.split(',')))  # Datos simulados

            # Tamaño de la ventana para calcular la potencia
            window_size = 50  # Ejemplo: 50 muestras

            for i, channel_value in enumerate(channels):
                # Actualizar filtro
                smoothed_value = filters[i].update_buffer(channel_value)

                # Shift y actualizar buffers de datos
                y_data[i] = np.roll(y_data[i], -1)
                y_data[i, -1] = smoothed_value

                # Calcular potencia en una ventana móvil
                power = np.mean(np.square(y_data[i, -window_size:]))  # Últimas 'window_size' muestras
                power_data[i] = np.roll(power_data[i], -1)
                power_data[i, -1] = power

                # Actualizar gráficos
                curves[i].setData(y_data[i])
                power_curves[i].setData(np.linspace(0, 1, len(power_data[i])), power_data[i])

        except ValueError:
            print(f"Invalid value ignored: {serial_line}")



# Timer for real-time updates
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(2)  # Update every 2 ms


def export_signals_to_csv(signals, filename):
 
    with open(filename, "w", newline="") as csvfile:
        writer = csv.writer(csvfile)
        # Escribir encabezados de canales
        writer.writerow(["Channel" + str(i+1) for i in range(signals.shape[0])])
        # Escribir datos transpuestos (filas como tiempo, columnas como canales)
        writer.writerows(signals.T)


def plot_full_signals(filtered_data, fs):
    time = np.linspace(0, filtered_data.shape[1] / fs, filtered_data.shape[1])
    plt.figure(figsize=(12, 8))
    for i in range(filtered_data.shape[0]):
        plt.plot(time, filtered_data[i], label=f'Channel {i+1}')
    plt.title("Filtered EEG Signals")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.grid()
    plt.tight_layout()
    plt.show()

# Exportar señales filtradas y graficar
def on_close():
    global y_data
    print("Closing application and plotting full signals...")
    plot_full_signals(y_data, fs)
    export_signals_to_csv(y_data, filename)

if __name__ == '__main__':
    try:
        app.exec_()
    except Exception as e:
        print(f"Error: {e}")
    finally:
        on_close()