import os
import cv2
import numpy as np
import random
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import confusion_matrix
from keras import layers, models
from keras.utils import to_categorical
import tensorflow as tf


# Function to load and preprocess images
def load_and_preprocess_data(path, crop):
    img = []
    lab = []
    for file in os.listdir(path):
        if not file.endswith(".jpg"):
            continue
        img.append(cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)[:, -crop:])
        lab.append(file.split("_")[-1].split(".")[0])
    return np.array(img), np.array(lab)


# function to
def display_predictions(m, img, truth, enc):
    pred_classes = enc.inverse_transform(np.argmax(m.predict(img), axis=1))
    idx = random.sample(range(len(img)), 3)
    display_images(img[idx], truth[idx], pred_classes[idx])


def display_random(num):
    idx = random.sample(range(len(images)), num)
    display_images(images[idx], labels[idx])


def display_images(img, true, pred=None):
    fig, axes = plt.subplots(1, len(img), figsize=(15, 5))

    for i in range(len(img)):
        axes[i].imshow(img[i], cmap="gray")
        if pred is None:
            axes[i].set_title(f"Class: {true[i]}")
        else:
            axes[i].set_title(f"True: {true[i]}\nPred:{pred[i]}")
        axes[i].axis("off")

    plt.show()


def display_confusion(pred, test, enc):
    y_pred = np.argmax(pred, axis=1)
    y_true = np.argmax(test, axis=1)
    names = enc.classes_
    cm = pd.DataFrame(confusion_matrix(y_true, y_pred), index=names, columns=names)

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False)
    plt.title("Confusion Matrix")
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def display_history(fit):
    # used plot method from class slides
    loss = fit.history['loss']
    val_loss = fit.history['val_loss']
    epochs = range(1, len(loss) + 1)
    plt.plot(epochs, loss, 'g.', label='Training Loss')
    plt.plot(epochs, val_loss, 'b', label='Validation Loss')
    plt.title(f'Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()


# Function: Convert some hex value into an array for C programming
def hex_to_c_array(hex_data, var_name):
    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nunsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'unsigned char ' + var_name + '[] = {'
    hex_array = []
    for i, val in enumerate(hex_data):

        # Construct string from hex
        hex_str = format(val, '#04x')

        # Add formatting so each line stays within 80 characters
        if (i + 1) < len(hex_data):
            hex_str += ','
        if (i + 1) % 12 == 0:
            hex_str += '\n '
        hex_array.append(hex_str)

    # Add closing brace
    c_str += '\n ' + format(' '.join(hex_array)) + '\n};\n\n'

    # Close out header guard
    c_str += '#endif //' + var_name.upper() + '_H'

    return c_str


# Write TFLite model to a C source (or header) file
def write_model(m):
    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    model_tflite = converter.convert()
    open("model.tflite", "wb").write(model_tflite)
    with open('model' + '.h', 'w') as file:
        file.write(hex_to_c_array(model_tflite, 'model.h'))


# Set the path to your dataset
data_path = "./unique_images"

# Define the width for cropping
crop_width = 70  # Adjust this value as needed

# Load and preprocess data
images, labels = load_and_preprocess_data(data_path, crop_width)

# show some of the images as a test
display_random(3)

# Encode labels
encoder = LabelEncoder()

# Convert labels to one-hot encoding
one_hot_labels = to_categorical(encoder.fit_transform(labels))

# reshape the images
images = images.reshape(-1, 144, crop_width, 1)

# Split the data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(images, one_hot_labels, test_size=0.4, random_state=598)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=598)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(144, crop_width, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Conv2D(8, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),
    layers.Dropout(0.25),
    layers.Flatten(),
    layers.Dense(3, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

# Train the model
fit = model.fit(X_train, y_train, epochs=50, validation_data=(X_val, y_val))

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test, y_test)
print(f"Test accuracy: {test_acc}")

# show loss history graph
display_history(fit)

# show 3 random example images
display_predictions(model, X_test, encoder.inverse_transform(np.argmax(y_test, axis=1)), encoder)

# show confusion matrix
display_confusion(model.predict(X_test), y_test, encoder)

# save c header file with tensorflow lite weights
write_model(model)
