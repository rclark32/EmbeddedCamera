#include <TinyMLShield.h>
#include "tensorflow/lite/micro/all_ops_resolver.h"
#include "tensorflow/lite/micro/micro_error_reporter.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/schema/schema_generated.h"
#include "tensorflow/lite/version.h"

#include "model_11.h" // Contains tensorflow model

byte imageQCIF[176 * 144 * 1]; // QCIF: 176x144 x 1 byte per pixel (Grayscale)

int croppedWidth = 40; // Width to crop from the right
byte croppedImage[40 * 40 * 1]; //Square image

//Global variables:
int bytesPerFrame;

const int input_height = 40;//144;
const int input_width = 40;//70;
const int input_channels = 1;

const int num_classes = 3;
float* result;
char* names[] = {"empty", "full", "partial"}; //Class names


//Create resolver and error reporter
tflite::AllOpsResolver tflOpsResolver;
tflite::MicroErrorReporter micro_error_reporter;

//Define empty models
const tflite::Model* trained_model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
TfLiteTensor* input = nullptr;
TfLiteTensor* output = nullptr;

//110kb arena: (May be smaller if needed)
constexpr int tensor_arena_size = 1024 * 110;
uint8_t tensor_arena[tensor_arena_size];


//Function definitions
void initializeModel();
void invokeModelInference();
void cropImage();
void sendInference();
void sendPicture();
void takePicture();

void setup() {
  Serial.begin(2000000); //Actual speed is probably 115200
  while (!Serial);

  initializeShield(); //Initializes the camera as well

  // Initialize the OV7675 camera
  Camera.begin(QCIF, GRAYSCALE, 1, OV7675); //Grayscale QCIF setup
  bytesPerFrame = Camera.width() * Camera.height() * Camera.bytesPerPixel();

  // Initialize the TensorFlow Lite model
  initializeModel();
}

void loop() {
  if (Serial.available() > 0) {
    char command = Serial.read(); //Get serial commands
    if (command == 'T' || command == 't') { //Just take and send a picture
      takePicture();
      sendPicture();
    }
    else if (command == 'I' || command == 'i') //take a picture and run inferencing, don't send picture
    {
      takePicture();
      invokeModelInference();
      sendInference();
      Serial.println("hex"); //Tells interpreter to stop reading serial

    }
    else if (command == 'C' || command == 'c') //Take and send picture and inference
    {
      takePicture();
      invokeModelInference();
      sendInference();
      Serial.println("hex"); //Tells interpreter to start reading image bytes
      sendPicture();
    }
  }
}

void initializeModel() {
  // Load the model
  trained_model = tflite::GetModel(model_arr); 
  if (trained_model->version() != TFLITE_SCHEMA_VERSION) {
    Serial.println("Model version mismatch!");
    return;
  }

  // Allocate memory fo interpreter
  interpreter = new tflite::MicroInterpreter(trained_model, tflOpsResolver, tensor_arena, tensor_arena_size, &micro_error_reporter);

  // Allocate memory for input and output tensors
  interpreter->AllocateTensors();
  input = interpreter->input(0);
  output = interpreter->output(0);
}



void takePicture() {
  Camera.readFrame(imageQCIF);
  cropImage(); // Crop to 40x40
}


void sendPicture() //Send picture byte by byte over serial
{ 
  for (int i = 0; i < sizeof(croppedImage); i++) {
    Serial.write(croppedImage[i]);
  }
}

void sendInference() //Print inference over serial
{
  for (int i = 0; i < num_classes; i++) {
    Serial.print(names[i]);
    Serial.print(": ");
    Serial.println(result[i]);
  }
  Serial.println();
}

void cropImage() { //Crop image to 40x40 pixels over the bin. Updates global variables
  int startY = 75; // start row
  int endY = 115; // end row
  int startX = 176 - 60; // start column
  int endX = 176 - 20; // end column
  
  for (int y = startY; y < endY; y++) {
    for (int x = startX; x < endX; x++) {
      int inputIndex = (y * Camera.width() + x) * Camera.bytesPerPixel();
      int outputIndex = ((y - startY) * (endX - startX) + (x - startX)) * Camera.bytesPerPixel();
      croppedImage[outputIndex] = imageQCIF[inputIndex];
    }
  }
}



void invokeModelInference() { //Run inferencing
  // Copy the input image data to the input tensor
  for (int i = 0; i < input_height * input_width; i++) {
    input->data.f[i] = static_cast<float>(croppedImage[i]) / 255.0f;
  }

  // Invoke the interpreter 
  interpreter->Invoke();

  // Access the output tensor
  result = output->data.f;

}
