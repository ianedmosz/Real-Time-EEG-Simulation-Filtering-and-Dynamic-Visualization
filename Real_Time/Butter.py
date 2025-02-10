import serial
import time
import numpy as np
import pyqtgraph as pg
from pyqtgraph.Qt import QtWidgets
import sys
import csv
from scipy.signal import butter, lfilter

# Filter Configuration
def butter_bandpass(lowcut, highcut, fs, order=5):
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    b, a = butter(order, [low, high], btype='bandpass')
    return b, a

# Smoothing filter
def butter_lowpass(cutoff, fs, order=4):
    nyq = 0.5 * fs  # Nyquist frequency
    normal_cutoff = cutoff / nyq
    b, a = butter(order, normal_cutoff, btype='low')
    return b, a

class RealTimeFilter:
    def __init__(self, lowcut, highcut, fs, order=5, buffer_size=500):
        self.b_band, self.a_band = butter_bandpass(lowcut, highcut, fs, order=order)
        self.b_smooth, self.a_smooth = butter_lowpass(smoothing_cutoff, fs, order=4)
        self.buffer_size = buffer_size
        self.data_buffer = np.zeros(buffer_size)
        self.zi_band = np.zeros(max(len(self.a_band), len(self.b_band)) - 1)
        self.zi_smooth = np.zeros(max(len(self.a_smooth), len(self.b_smooth)) - 1)
        self.index = 0

    def update_buffer(self, new_sample):
        band_filtered_value, self.zi_band = lfilter(self.b_band, self.a_band, [new_sample], zi=self.zi_band)
        smoothed_value, self.zi_smooth = lfilter(self.b_smooth, self.a_smooth, band_filtered_value, zi=self.zi_smooth)
        self.data_buffer[self.index] = smoothed_value[-1]
        self.index = (self.index + 1) % self.buffer_size
        return smoothed_value[-1]

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
fs = 500 
lowcut = 12.0
highcut = 30.0
smoothing_cutoff = 5.0
realtime_filter = RealTimeFilter(lowcut=lowcut, highcut=highcut, fs=fs, order=5)

# Data for real-time plotting
x_data = np.linspace(0, 1, fs)  # 1 second at 500 Hz
y_data = np.zeros(fs)
power_data = np.zeros(fs)

# PyQtGraph Configuration
app = QtWidgets.QApplication([])
win = pg.GraphicsLayoutWidget(show=True, title="EEG Beta 12-30Hz Real-Time and Power")
win.resize(1000, 800)
win.setWindowTitle('PyQtGraph Real-time EEG')

# Real-time filtered EEG signal plot
plot1 = win.addPlot(title="Filtered EEG Signal in Real Time")
curve1 = plot1.plot(x_data, y_data, pen='b')

# EEG signal power plot
win.nextRow()
plot2 = win.addPlot(title="EEG Signal Power")
curve2 = plot2.plot(x_data, power_data, pen='r')

powers = []
data_counter = 0
start_time = time.time()

# Real-time update function
def update():
    global y_data, power_data, powers, data_counter, start_time
    if ser.in_waiting > 0:
        serial_line = ser.readline().decode('utf-8', errors='ignore').strip()
        try:
            # Read and filter the value
            new_sample = float(serial_line)
            smoothed_value = realtime_filter.update_buffer(new_sample)

            # Shift data and add new value
            y_data = np.roll(y_data, -1)
            y_data[-1] = smoothed_value

            # Calculate power and shift power data
            power = np.mean(np.square(y_data))
            powers.append(power)
            power_data = np.roll(power_data, -1)
            power_data[-1] = power

            # Update data counter
            data_counter += 1

            # Check elapsed time to display data count every 2 seconds
            if time.time() - start_time >= 2:
                print(f"Data received in the last 2 seconds: {data_counter}")
                data_counter = 0  # Reset counter
                start_time = time.time()

           
            print(f"Signal power: {power:.4f} µV^2")

         
            plot1.setTitle(f"Filtered EEG Signal - Current Power: {power:.2f} µV^2")

            # Update the curves
            curve1.setData(y_data)
            curve2.setData(power_data)

        except ValueError:
            print(f"Invalid value ignored: {serial_line}")

# Timer for real-time updates
timer = pg.QtCore.QTimer()
timer.timeout.connect(update)
timer.start(2)  # Update every 2 ms

# Run the application
if __name__ == '__main__':
    sys.exit(app.exec())


csv_filename = input("Enter the name of the CSV file to save the data (include '.csv'): ")
if not csv_filename.endswith('.csv'):
    csv_filename += '.csv'
with open(csv_filename, mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(['Time (s)', 'Filtered EEG Value', 'Power (µV^2)'])
    for i in range(len(y_data)):
        writer.writerow([i / fs, y_data[i], power_data[i]])
