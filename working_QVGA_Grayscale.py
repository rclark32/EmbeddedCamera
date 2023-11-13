import serial
import serial.tools.list_ports
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys
import time

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

def receive_image_data(ser, expected_bytes, progress_callback=None):
    received_bytes = b''
    total_bytes = 0

    while len(received_bytes) < expected_bytes:
        received_bytes += ser.read(expected_bytes - len(received_bytes))
        if progress_callback:
            total_bytes += len(received_bytes)
            progress = (total_bytes / expected_bytes) * 100
            progress_callback(progress)

    return received_bytes

def format_image_data(hex_data):
    raw_bytes = np.frombuffer(hex_data, dtype='>u2')
    image = np.zeros(len(raw_bytes), dtype=int)

    for i in range(len(raw_bytes)):
        # Discard the second byte
    
        pixel = raw_bytes[i] >> 8
        image[i] = pixel

    image = np.reshape(image, (240, 320))
    return image

def display_image(image):
    plt.imshow(image, cmap='gray')
    plt.show()

def progress_callback(progress):
    sys.stdout.write("\rProgress: [{:<50}] {:.2f}%".format('='*int(progress/8), progress/12))
    sys.stdout.flush()

def main():
    ser = None
    start_time = None
    try:
        arduino_port = find_arduino_port()
        ser = connect_to_arduino(arduino_port)
        
        input("Press Enter to take a picture...")

        while True:
            start_time = time.time()
            print("Taking picture...")
            if ser.in_waiting > 0:
                print("Ser inwaiting")
                ser.flushInput()

            trigger_picture(ser)

            hex_data = receive_image_data(ser, 320 * 240 * 2, progress_callback)
            image = format_image_data(hex_data)

            display_image(image)
            
            end_time = time.time()
            elapsed_time = end_time - start_time  # Calculate the elapsed time
            print(f"\n\n time taken: {round(elapsed_time,2)} seconds")
            
            user_input = input("Press Enter to take another picture or type 'q' to quit...").strip().lower()
            if user_input == 'q':
                break

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if ser is not None:
            ser.close()

if __name__ == "__main__":
    main()
