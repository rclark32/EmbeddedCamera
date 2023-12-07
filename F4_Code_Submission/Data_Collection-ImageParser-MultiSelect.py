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

current_color_mode = 'Grayscale'  # Default color mode
current_resolution = (176, 144)  # Default resolution

# global variables
save_state = False
current_description = "Undefined" 

def find_arduino_port(): #Find serial device (ONLY WORKS WITH ONE THING PLUGGED IN)
    arduino_ports = [p.device for p in serial.tools.list_ports.comports() if 'Device' in p.description]
    if not arduino_ports:
        raise Exception("Arduino not found")
    return arduino_ports[0]

def connect_to_arduino(port):
    ser = serial.Serial(port, 2000000, timeout=1) #Probably limited to 115200
    return ser


def set_color_mode(ser, mode): # Set mode and update it on arduino
    global current_color_mode
    if mode in COLOR_MODES:
        current_color_mode = COLOR_MODES[mode]
        ser.write(mode.encode())
    else:
        print(f"Invalid color mode: {mode}")

def set_resolution(ser, mode): #Set and update resolution on arduino
    global current_resolution
    if mode in RESOLUTION_MODES:
        current_resolution = RESOLUTION_MODES[mode]
        ser.write(mode.encode())
    else:
        print(f"Invalid resolution mode: {mode}")
        
def receive_image_data(ser, expected_bytes, progress_callback=None): #Receive image byte by byte
    received_bytes = b''
    total_bytes = 0

    while len(received_bytes) < expected_bytes:
        received_bytes += ser.read(expected_bytes - len(received_bytes))
        if progress_callback:
            #progress = (len(received_bytes) / expected_bytes) * 100
            progress_callback(len(received_bytes),expected_bytes)

    return received_bytes

def format_image_data(hex_data): # Convert RGB 565 to 888
    if (current_color_mode == 'RGB'):
        raw_bytes = np.frombuffer(hex_data, dtype='>u2')
        image = np.zeros((len(raw_bytes), 3), dtype=np.uint8)
    
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


def display_image(image): # Plot image 
    plt.axis("off")
    if current_color_mode == 'Grayscale':
        plt.imshow(image, cmap='gray')
    elif current_color_mode == 'RGB':
        plt.imshow(image)
        
    plt.show()

def progress_callback(received,expected): 
    sys.stdout.write(f"\rReceived: {received}, expected: {expected}")
    sys.stdout.flush()
    
def send_command(ser, command): 
    ser.write(command.encode())

def get_next_image_id(): # Find next image id based on saved image ids
    if not os.path.exists('train_images'):
        os.makedirs('train_images')
        
    train_images = [f for f in os.listdir('train_images') if f.endswith('.jpg')]
    #print(train_images)
    if not train_images:
        return 1
    else:
        image_ids = [int(f.split('_')[1].split('.')[0]) for f in train_images]
        return max(image_ids) + 1

def save_image(image, description): # Save image with description
    global current_description
    global save_state

    if save_state:
        image_id = get_next_image_id()
        image_info = 'R' if 'R' in current_color_mode else 'G'
        image_info = image_info + "_" + ('QVGA' if current_resolution==RESOLUTION_MODES['V'] else 'QCIF')
        image_filename = f"./train_images/img_{image_id}_{image_info}_{description}.jpg"
        
        if current_color_mode == 'Grayscale':
            plt.imsave(image_filename,image, cmap='gray')
        elif current_color_mode == 'RGB':
            plt.imsave(image_filename,image)
        else:
            raise ValueError(f"Unsupported color mode: {current_color_mode}")
        print(f"Image saved as {image_filename}")

    current_description = description

def toggle_save_state():
    global save_state
    save_state = not save_state
    print(f"Save state toggled to {'On' if save_state else 'Off'}")

def set_description(description):
    global current_description
    current_description = description
    print(f"Description set to: {description}")


def capture_multiple_pictures(ser,num_pictures):
    for i in range(num_pictures):
        print(f"\n{num_pictures-i} pictures remaining ")
        take_picture(ser)

def take_picture(ser):
    start_time = None

    start_time = time.time()
    send_command(ser, 'T')  # Send 'T' to trigger picture
    
    hex_data = receive_image_data(ser, current_resolution[0] * current_resolution[1] * (2 if current_color_mode == "RGB" else 1), progress_callback)
                   
    image = format_image_data(hex_data)
    display_image(image)
    end_time = time.time()
    elapsed_time = end_time - start_time
    print(f"\nTime taken: {round(elapsed_time, 2)} seconds")
    
    if save_state:
        save_image(image, current_description)

def main():
    ser = None
    print("Current directory is ", os.getcwd(), " if you are saving images, make sure this is correct. Change it with os.chdir('filepath')")
    try:
        arduino_port = find_arduino_port()
        ser = connect_to_arduino(arduino_port)
        
        while True:
            user_input = input("\nEnter command (T/Enter: Take Picture, Any integer: take X pictures, G: Grayscale R: RGB565, C: QCIF, V: QVGA, S: Save Toggle, D: Set Description, Q: Quit):\n").strip().upper()
            
            if user_input == 'Q':
                break
            elif user_input == 'T' or user_input == '':
                take_picture(ser)
            
            elif user_input.isdigit(): #Take multiple pictures
                num_pictures = int(user_input)
                capture_multiple_pictures(ser, num_pictures)
                
            elif user_input in COLOR_MODES: # Picture modes
                set_color_mode(ser, user_input)
                
            elif user_input in RESOLUTION_MODES:
                set_resolution(ser, user_input)
                
            elif user_input == 'S': # Save state
                toggle_save_state()
                
            elif user_input == 'D': #Change description
                description = input("Enter new description: ")
                set_description(description)
                
            else:
                print("Invalid command. Please try again.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if ser is not None: #Close seerial port when program is ended
            ser.close()

if __name__ == "__main__":
    main()