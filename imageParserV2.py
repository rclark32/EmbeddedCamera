import serial
import struct
import numpy as np
import matplotlib.pyplot as plt

# Find the appropriate COM port (you may need to adjust the port name)
ser = serial.Serial()
ser.baudrate = 115200

for port in range(10):
    try:
        ser.port = f'COM{port}'
        ser.open()
        break
    except serial.SerialException:
        pass
else:
    print("Failed to find a valid COM port")
    exit()

def receive_image():
    image_bytes = ser.read(176 * 144)
    return image_bytes

def parse_image(image_bytes):
    return list(image_bytes)

def display_image(pixels):
    img_array = np.array(pixels, dtype=np.uint8).reshape((144, 176))
    plt.imshow(img_array, cmap='gray', vmin=0, vmax=255)
    plt.axis('off')
    plt.show()
    
try:
    while True:
        command = input("Enter 'T' to trigger capture or 'Q' to quit: ")
        if command == 'T':
            ser.write(b'T')
            image_data = receive_image()
            pixels = parse_image(image_data)
            display_image(pixels)  # Display the image
        elif command == 'Q':
            break
except KeyboardInterrupt:
    pass
finally:
    ser.close()