import serial.tools.list_ports
import numpy as np
import matplotlib.pyplot as plt
import time

# Step 1: Detect the appropriate COM port
def find_arduino_port():
    ports = list(serial.tools.list_ports.comports())
    for port, desc, hwid in ports:
        if "Device" in desc:
            return port
    return None

bytesPerFrame = 176 * 144 * 2
image_data = None

# Step 2: Establish a serial connection to the Arduino
arduino_port = find_arduino_port()
if arduino_port is None:
    print("Arduino not found. Please make sure it's connected.")
else:
    ser = serial.Serial(arduino_port, 115200, timeout=1)

# Step 3: Read and parse the image data received over serial
def parse_image_data(data):
    #bytes_list = data.split(", ")
    image_data = bytearray.fromhex("".join(bytes_list))
    return image_data

# Step 4: Display the image using matplotlib
def display_image(image_data):
    image_array = np.frombuffer(image_data, dtype=np.uint16).reshape((144, 176))
    plt.imshow(image_array, cmap='viridis')
    plt.show()


user_input = input("Press 'T' to trigger capture, 'Q' to quit: ")
if user_input == 'T':
    ser.write(b'T')  # Send 'T' to trigger capture
    time.sleep(2)    # Add a delay to allow the camera to respond
    image_data = ser.readall()#(bytesPerFrame).decode('utf-8').strip()    #parsed_data = parse_image_data(image_data)
    #display_image(parsed_data)
elif user_input == 'Q':
    pass

ser.close()
