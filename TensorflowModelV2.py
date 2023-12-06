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
import keras
import tensorflow_model_optimization as tfmot


# Function to load and preprocess images
def load_and_preprocess_data(path):
    img = []
    lab = []
    x = 95
    y = -40
    s = 40
    for file in os.listdir(path):
        if not file.endswith(".jpg"):
            continue
        img.append(cv2.imread(os.path.join(path, file), cv2.IMREAD_GRAYSCALE)[int(x-s/2):int(x+s/2), int(y-s/2):int(y+s/2)])
        label = file.split("_")[-1].split(".")[0]
        if label == "partial":
            label = "full"
        lab.append(label)
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
    print(f"Number of hex bytes: {len(hex_data)}")
    c_str = ''

    # Create header guard
    c_str += '#ifndef ' + var_name.upper() + '_H\n'
    c_str += '#define ' + var_name.upper() + '_H\n\n'

    # Add array length at top of file
    c_str += '\nconst unsigned int ' + var_name + '_len = ' + str(len(hex_data)) + ';\n'

    # Declare C variable
    c_str += 'const unsigned char ' + var_name + '[] = {'
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


def representative_data_gen():
    for input_value in tf.data.Dataset.from_tensor_slices(X_train).batch(1).take(100):
        # Model has only one input so each data point has one element.
        yield [input_value]


# Write TFLite model to a C source (or header) file
def write_model(m):
    converter = tf.lite.TFLiteConverter.from_keras_model(m)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    model_tflite = converter.convert()
    open("model.tflite", "wb").write(model_tflite)
    with open('model' + '.h', 'w') as file:
        file.write(hex_to_c_array(model_tflite, 'model.h'))
    return model_tflite


# Set the path to your dataset
data_path = "./train_images"

# Load and preprocess data
images, labels = load_and_preprocess_data(data_path)

# show some of the images as a test
display_random(3)

# Encode labels
encoder = LabelEncoder()

# Convert labels to one-hot encoding
one_hot_labels = to_categorical(encoder.fit_transform(labels))

# Split the data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(images, one_hot_labels, test_size=0.4, random_state=598)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=598)

# Build the CNN model
model = models.Sequential([
    layers.Conv2D(4, (3, 3), activation="relu", input_shape=(len(images[0]), len(images[0][0]), 1)),
    layers.MaxPooling2D((2, 6)),
    layers.Dropout(0.2),
    layers.Flatten(),
    layers.Dense(2, activation="softmax")
])

# Compile the model
model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])

model.summary()


# Train the model
fit = model.fit(X_train, y_train, epochs=40, validation_data=(X_val, y_val))

# show loss history graph
display_history(fit)

# show 3 random example images
display_predictions(model, X_test, encoder.inverse_transform(np.argmax(y_test, axis=1)), encoder)

# show confusion matrix
display_confusion(model.predict(X_test), y_test, encoder)





# q_aware stands for for quantization aware.
q_aware_model = tfmot.quantization.keras.quantize_model(model)

# `quantize_model` requires a recompile.
q_aware_model.compile(optimizer='adam', loss="categorical_crossentropy", metrics=['accuracy'])

q_aware_model.summary()

q_aware_model.fit(X_train[0:100], y_train[0:100], batch_size=50, epochs=1, validation_split=0.1)

_, baseline_model_accuracy = model.evaluate(X_test, y_test, verbose=0)
_, q_aware_model_accuracy = q_aware_model.evaluate(X_test, y_test, verbose=0)


# save c header file with tensorflow lite weights
lite = write_model(q_aware_model)
print('Baseline test accuracy:', baseline_model_accuracy)
print('Quant test accuracy:', q_aware_model_accuracy)