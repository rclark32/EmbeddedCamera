import serial
import serial.tools.list_ports
import numpy as np
import struct
import matplotlib.pyplot as plt

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

def receive_image_data(ser, expected_bytes):
    received_bytes = ser.read(expected_bytes)
    while len(received_bytes) < expected_bytes:
        received_bytes += ser.read(expected_bytes - len(received_bytes))
    return received_bytes

def format_image_data(hex_data):
    raw_bytes = np.frombuffer(hex_data, dtype='>u2')
    image = np.zeros((len(raw_bytes), 3), dtype=int)

    for i in range(len(raw_bytes)):
        pixel = raw_bytes[i]
        r = ((pixel >> 11) & 0x1f) << 3
        g = ((pixel >> 5) & 0x3f) << 2
        b = ((pixel >> 0) & 0x1f) << 3
        image[i] = [r, g, b]

    image = np.reshape(image, (240, 320, 3))
    return image

def display_image(image):
    plt.imshow(image)
    plt.show()

def main():
    try:
        arduino_port = find_arduino_port()
        ser = connect_to_arduino(arduino_port)
            
        input("Press Enter to take a picture (or type 'q' to quit)...")
        
        while True:
            
            if ser.in_waiting > 0:
                ser.flushInput()

            trigger_picture(ser)

            hex_data = receive_image_data(ser, 320 * 240 * 2)
            image = format_image_data(hex_data)

            display_image(image)

            user_input = input("Press Enter to take another picture or type 'q' to quit...").strip().lower()
            if user_input == 'q':
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        ser.close()

if __name__ == "__main__":
    main()
