#include <TinyMLShield.h>


//Limited to QCIF
//byte imageQVGA[320 * 240 * 2]; // QCIF: 176x144 x 2 bytes per pixel (RGB565)
byte imageQCIF[176 * 144 * 1]; // QCIF: 176x144 x 2 bytes per pixel (RGB565)


//Define global variables
int bytesPerFrame;

char currentColorMode = 'G'; // Default color mode
char currentResolution = 'C'; // Default resolution

//Define functions
void setCameraDefaults();
void takePictureAndSend();


void setup() {
  Serial.begin(2000000); //Probably limited to 115200
  while (!Serial);

  initializeShield(); //Also initializes camera

  // Initialize the OV7675 camera
  setCameraDefaults(); //Set to grayscale qcif
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read();
    if (command == 'T') { // Trigger capture
      takePictureAndSend();
    } else if (command == 'G' || command == 'R') { //Change color mode
      currentColorMode = command;
      setCameraDefaults();
    } else if (command == 'C' || command == 'V') { //Change resolution
      currentResolution = command;
      setCameraDefaults();
    }
  }
}

void setCameraDefaults() { //Update camera instance
  if (currentResolution == 'C') {
    if (currentColorMode == 'R')
    {
      Camera.begin(QCIF, RGB565, 1, OV7675);
    }
    else if (currentColorMode == 'G')
    {
      Camera.begin(QCIF, GRAYSCALE, 1, OV7675);
    }
    
  } else {
      if (currentColorMode == 'R')
      {
        if(!Camera.begin(QVGA, RGB565, 1, OV7675)){
        Serial.println("Failed to initialize camera");
        while (1);
        }
      }
      
      else if (currentColorMode == 'G')
      {
        Camera.begin(QVGA, GRAYSCALE, 1, OV7675);
      }
  }

  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();
}

void takePictureAndSend() { //Take picture with correct settings and send over serial

  if (currentResolution == 'C')
  {
    Camera.readFrame(imageQCIF);

    for (int i = 0; i < bytesPerFrame; i++) {
      Serial.write(imageQCIF[i]);
    }
  }
  /*else if (currentResolution == 'V') //Vga disabled for testing
  {
    Camera.readFrame(imageQVGA);

    for (int i = 0; i < bytesPerFrame; i++) {
      Serial.write(imageQVGA[i]);
    }
  }*/
}
