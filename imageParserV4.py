import serial
import time
import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt


bytesPerFrame = 176 * 144 * 2

# Find the Arduino over serial
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port, desc, hwid in ports:
        if "Device" in desc:
            return port
    return None

arduino_port = find_arduino_port()
if arduino_port is None:
    print("Arduino not found. Please make sure it's connected.")
else:
    ser = serial.Serial(arduino_port, 115200, timeout=1)

# Wait for Arduino to initialize
time.sleep(2)

# Trigger the Arduino to take a picture
ser.write(b'T')

# Read the data sent by Arduino
data = ser.read(bytesPerFrame)#.decode('utf-8').strip()

# Print the data
print("Data received from Arduino:")
print(data)

# Close the serial connection
ser.close()
