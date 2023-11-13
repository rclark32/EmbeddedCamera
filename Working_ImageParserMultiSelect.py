import serial
import serial.tools.list_ports
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys
import time
import os

# Define constants for color and resolution modes
COLOR_MODES = {'R': 'RGB', 'G': 'Grayscale'}
RESOLUTION_MODES = {'C': (176, 144), 'V': (320, 240)}

current_color_mode = 'RGB'  # Default color mode
current_resolution = (320, 240)  # Default resolution

# New global variables
save_state = False
current_description = None

def find_arduino_port():
    arduino_ports = [p.device for p in serial.tools.list_ports.comports() if 'Device' in p.description]
    if not arduino_ports:
        raise Exception("Arduino not found")
    return arduino_ports[0]

def connect_to_arduino(port):
    ser = serial.Serial(port, 460800, timeout=1)
    return ser

def trigger_picture(ser):
    ser.write(b'T')

def set_color_mode(ser, mode):
    global current_color_mode
    if mode in COLOR_MODES:
        current_color_mode = COLOR_MODES[mode]
        ser.write(mode.encode())
    else:
        print(f"Invalid color mode: {mode}")

def set_resolution(ser, mode):
    global current_resolution
    if mode in RESOLUTION_MODES:
        current_resolution = RESOLUTION_MODES[mode]
        ser.write(mode.encode())
    else:
        print(f"Invalid resolution mode: {mode}")
        
def receive_image_data(ser, expected_bytes, progress_callback=None):
    received_bytes = b''
    total_bytes = 0

    while len(received_bytes) < expected_bytes:
        received_bytes += ser.read(expected_bytes - len(received_bytes))
        print(f"recv {len(received_bytes)} out of {expected_bytes}")
        if progress_callback:
            progress = (len(received_bytes) / expected_bytes) * 100
            progress_callback(progress)

    return received_bytes

def format_image_data(hex_data):
    if (current_color_mode == 'RGB'):
        raw_bytes = np.frombuffer(hex_data, dtype='>u2')
        image = np.zeros((len(raw_bytes), 3), dtype=int)
    
        for i in range(len(raw_bytes)):
            pixel = raw_bytes[i]
            r = ((pixel >> 11) & 0x1f) << 3
            g = ((pixel >> 5) & 0x3f) << 2
            b = ((pixel >> 0) & 0x1f) << 3
            image[i] = [r, g, b]
    
        image = np.reshape(image, (current_resolution[1],current_resolution[0], 3))
        return image
    elif (current_color_mode == 'Grayscale'):
        raw_bytes = np.frombuffer(hex_data, dtype='uint8')  # Use uint8 for 1 byte per pixel
        image = np.reshape(raw_bytes, (current_resolution[1], current_resolution[0]))
        return image
    else:
        raise ValueError(f"Unsupported color mode: {current_color_mode}")


def display_image(image):
    if current_color_mode == 'Grayscale':
        plt.imshow(image, cmap='gray')
    elif current_color_mode == 'RGB':
        plt.imshow(image)
        
    plt.show()

def progress_callback(progress):
    sys.stdout.write("\rProgress: [{:<50}] {:.2f}%".format('='*int(progress/10), progress))
    sys.stdout.flush()
    
def send_command(ser, command):
    ser.write(command.encode())

def main():
    ser = None
    start_time = None
    try:
        arduino_port = find_arduino_port()
        ser = connect_to_arduino(arduino_port)
        
        while True:
            user_input = input("Enter command (T/Enter: Take Picture, G: Grayscale R: RGB565, C: QCIF, V: QVGA, Q: Quit): ").strip().upper()

            if user_input == 'Q':
                break
            elif user_input == 'T' or user_input == '':
                start_time = time.time()
                send_command(ser, 'T')  # Send 'T' to trigger picture
                
                hex_data = receive_image_data(ser, current_resolution[0] * current_resolution[1] * (2 if current_color_mode == "RGB" else 1), progress_callback)
                
                image = format_image_data(hex_data)
                display_image(image)
                end_time = time.time()
                elapsed_time = end_time - start_time
                print(f"\nTime taken: {round(elapsed_time, 2)} seconds")
                
            elif user_input in COLOR_MODES:
                set_color_mode(ser, user_input)
                
            elif user_input in RESOLUTION_MODES:
                set_resolution(ser, user_input)
                
            else:
                print("Invalid command. Please try again.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if ser is not None:
            ser.close()

if __name__ == "__main__":
    main()