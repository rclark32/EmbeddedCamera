import serial
import time
import matplotlib.pyplot as plt
import re
from serial.tools import list_ports

def find_arduino_port():
    arduino_ports = [p.device for p in list_ports.comports() if 'Serial' in p.description]
    if not arduino_ports:
        raise IOError("No Arduino found")
    if len(arduino_ports) > 1:
        print("Multiple Arduinos found - using the first one")
    return arduino_ports[0]

def send_command_to_arduino(ser, command):
    ser.write(command.encode())

def receive_image_from_arduino(ser):
    image_data = []
    while True:
        line = ser.readline().decode().strip()
        if line == "":
            break
        hex_values = re.findall(r'0x([0-9A-Fa-f]{2})', line)
        image_data.extend([int(val, 16) for val in hex_values])
    return image_data

image = []

def display_image(image_data, width, height):
    image = []
    for i in range(0, len(image_data), 2):
        pixel = (image_data[i + 1] << 8) | image_data[i]
        image.append(pixel)
    image = [image[i:i+width] for i in range(0, len(image), width)]
    plt.imshow(image)
    plt.show()

if __name__ == "__main__":
    arduino_port = find_arduino_port()
    ser = serial.Serial(arduino_port, 115200, timeout=1)

    try:
        while True:
            user_input = input("Press Enter to take a picture (Q to quit): ")
            if user_input.upper() == 'Q':
                break

            send_command_to_arduino(ser, 'T')
            time.sleep(2)  # Adjust this delay if necessary to ensure the image is fully received
            image_data = receive_image_from_arduino(ser)

            # Assuming QCIF resolution (176x144)
            display_image(image_data, 176, 144)

    finally:
        ser.close()
