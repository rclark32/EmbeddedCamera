import serial
import serial.tools.list_ports
import numpy as np
import struct
import matplotlib.pyplot as plt
import sys
import time
import os


# Define constants for color and resolution modes
COLOR_MODES = {'R': 'RGB', 'G': 'Grayscale'} #Unused
RESOLUTION_MODES = {'C': (176, 144), 'V': (320, 240)} #unused
 
current_color_mode = 'Grayscale'  # Default color mode
current_resolution = (40, 40)  # Default resolution

# New global variables
save_state = False
current_description = "Undefined"

def find_arduino_port():
    arduino_ports = [p.device for p in serial.tools.list_ports.comports() if 'Device' in p.description]
    if not arduino_ports:
        raise Exception("Arduino not found")
    return arduino_ports[0]

def connect_to_arduino(port): #Actually 115200 baud or lower
    ser = serial.Serial(port, 2000000, timeout=1)
    return ser

def trigger_picture(ser): #Unused
    ser.write(b'T')

def run_inferencing(ser): #Unused
    ser.write(b'I')
    
def inference_and_show(ser): #Unused
    ser.write(b'C')
        
def receive_image_data(ser, expected_bytes, progress_callback=None):
    received_bytes = b''
    total_bytes = 0
    # Receive image byte by byte
    while len(received_bytes) < expected_bytes:
        received_bytes += ser.read(expected_bytes - len(received_bytes))
        if progress_callback:
            #progress = (len(received_bytes) / expected_bytes) * 100
            progress_callback(len(received_bytes),expected_bytes)

    return received_bytes

def format_image_data(hex_data): #Convert grayscale data
    raw_bytes = np.frombuffer(hex_data, dtype='uint8')  # Use uint8 for 1 byte per pixel
    image = np.reshape(raw_bytes, (current_resolution[1], current_resolution[0]))
    return image
    

def display_image(image): # Plot image
    plt.axis("off")
    plt.imshow(image, cmap='gray')        
    plt.show()

def progress_callback(received,expected):
    sys.stdout.write(f"\rReceived: {received}, expected: {expected}")
    sys.stdout.flush()
    
def send_command(ser, command):
    ser.write(command.encode())

def get_next_image_id():  #Used for saving images
    if not os.path.exists('final_images'):
        os.makedirs('final_images')
        
    train_images = [f for f in os.listdir('final_images') if f.endswith('.jpg')]
    #print(train_images)
    if not train_images:
        return 1
    else:
        image_ids = [int(f.split('_')[1].split('.')[0]) for f in train_images]
        return max(image_ids) + 1

def save_image(image, description):
    global current_description
    global save_state

    if save_state:
        image_id = get_next_image_id()
        image_info = "40x40"
        image_filename = f"./final_images/img_{image_id}_{image_info}_{description}.jpg"
        
        plt.imsave(image_filename,image, cmap='gray')
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


def get_inference(ser): # Get inference as text only
    send_command(ser, 'I')  
    while True:
        line = ser.readline()
        if line.strip() == b"hex":
            break
        else:
           ln = line.decode('utf-8')
           print(ln) 

def get_picture_and_inference(ser): #Receive text, then picture
    send_command(ser, 'C')  
    
    classes = ["empty","full","partial"]
    while True:
        line = ser.readline()
        if line.strip() == b"hex": # Wait until receive 'hex' to read the image
            start_time = time.time()

            hex_data = receive_image_data(ser, current_resolution[0] * current_resolution[1] * 1, progress_callback)

            image = format_image_data(hex_data)
            display_image(image)
            end_time = time.time()
            elapsed_time = end_time - start_time
            print(f"\nTime taken: {round(elapsed_time, 2)} seconds")

            if save_state:
                save_image(image, current_description)
            break
        else:
            ln = line.decode('utf-8')
            print(ln)



def take_picture(ser): # Just take picture, no inference
    start_time = None

    start_time = time.time()
    send_command(ser, 'T')  
    
    hex_data = receive_image_data(ser, current_resolution[0] * current_resolution[1] * 1, progress_callback)
                   
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
            user_input = input("\nEnter command (C/Enter: Take Picture and inference, I: inference, T: Take picture, S: Save Toggle, D: Set Description, Q: Quit):\n").strip().upper()
            
            if user_input == 'Q':
                break
            elif user_input == 'C' or user_input == '':
                get_picture_and_inference(ser)
            elif user_input == 'T':
                take_picture(ser)
            elif user_input == 'I':
                get_inference(ser)
            elif user_input.isdigit():
                num_pictures = int(user_input)
                capture_multiple_pictures(ser, num_pictures)
                
            elif user_input == 'S':
                toggle_save_state()
                
            elif user_input == 'D':
                description = input("Enter new description: ")
                set_description(description)
                
            else:
                print("Invalid command. Please try again.")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        if ser is not None: #Close serial port when program is quit
            ser.close()

if __name__ == "__main__":
    main()