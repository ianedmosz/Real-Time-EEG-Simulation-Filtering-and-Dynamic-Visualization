import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
import csv
from scipy.signal import butter, lfilter
from scipy.ndimage import gaussian_filter1d  # Import for Gaussian filter

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

# Serial port configuration
arduino_port = 'COM5'  # Adjust port as needed
baud_rate = 115200
try:
    ser = serial.Serial(arduino_port, baud_rate, timeout=1)
    print(f"Connected to port {arduino_port}")
except Exception as e:
    print(f"Error connecting to Arduino: {e}")
    sys.exit(1)

# Filter Parameters
fs = 500  # Sampling frequency
lowcut = 12.0 
highcut = 30.0
gaussian_sigma = 5.0  # Standard deviation for Gaussian smoothing

# EEG channel configuration (user-defined positions)
#channel_positions = input("Enter the channel positions separated by commas (e.g., FP1, FP2, C3, C4): ").split(',')
channel_positions = ['FP1', 'FP2', 'C3', 'C4'].split(',')   
num_channels = len(channel_positions)

# Initialize filters and data buffers for all channels
filters = [RealTimeFilter(lowcut, highcut, fs, order=5, gaussian_sigma=gaussian_sigma) for _ in range(num_channels)]
y_data = np.zeros((num_channels, fs))
power_data = np.zeros((num_channels, fs))

# PyQtGraph Configuration
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title=f"EEG Beta {lowcut}-{highcut}Hz Real-Time and Power")
win.resize(1000, 800)
win.setWindowTitle('PyQtGraph Real-time EEG')

# Create plots dynamically based on the number of channels
plots = []
curves = []
for i in range(num_channels):
    plot = win.addPlot(title=f"Filtered EEG Signal - {channel_positions[i].strip()}")
    curve = plot.plot(np.linspace(0, 1, fs), y_data[i], pen='b')
    plots.append(plot)
    curves.append(curve)
    win.nextRow()

# Real-time update function
def update():
    global y_data, power_data
    if ser.in_waiting > 0:
        serial_line = ser.readline().decode('utf-8', errors='ignore').strip()
        try:
            # Parse the data for multiple channels
            channels = list(map(float, serial_line.split(',')))  # Assumes comma-separated data
            for i, channel_value in enumerate(channels):
                smoothed_value = filters[i].update_buffer(channel_value)

                # Shift and update data buffers
                y_data[i] = np.roll(y_data[i], -1)
                y_data[i, -1] = smoothed_value

                # Calculate power and shift power data
                power = np.mean(np.square(y_data[i]))
                power_data[i] = np.roll(power_data[i], -1)
                power_data[i, -1] = power

                # Update plots for each channel
                curves[i].setData(y_data[i])

        except ValueError:
            print(f"Invalid value ignored: {serial_line}")

# Timer for real-time updates
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(2)  # Update every 2 ms

# Run the application
if __name__ == '__main__':
    sys.exit(app.exec())

# Ask for the CSV filename at the end and save the power data
csv_filename = input("Enter the name of the CSV file to save the data (include '.csv'): ")
if not csv_filename.endswith('.csv'):
    csv_filename += '.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    header = ['Time (s)'] + [f"{pos.strip()} Value" for pos in channel_positions] + [f"{pos.strip()} Power" for pos in channel_positions]
    writer.writerow(header)
    for i in range(fs):
        row = [i / fs] + list(y_data[:, i]) + list(power_data[:, i])
        writer.writerow(row)
