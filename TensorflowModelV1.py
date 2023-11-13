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
from tensorflow.keras import layers, models
from tensorflow.keras.utils import to_categorical

# Set the path to your dataset
data_path = "C:/Users/recla/Documents/EmbeddedCamera/train_images"

# Define the width for cropping
crop_width = 70  # Adjust this value as needed

# Function to load and preprocess images
def load_and_preprocess_data(data_path, crop_width):
    images = []
    labels = []

    for filename in os.listdir(data_path):
        if filename.endswith(".jpg"):
            img = cv2.imread(os.path.join(data_path, filename), cv2.IMREAD_GRAYSCALE)
            img = img[:, -crop_width:]  # Crop to the rightmost crop_width pixels
            label = filename.split("_")[-1].split(".")[0]  # Extract label from filename

            images.append(img)
            labels.append(label)

    return np.array(images), np.array(labels)


def display_predictions(model, images, true_labels, label_encoder):
    # Randomly select a few images for display
    num_display_images = 3
    selected_indices = random.sample(range(len(images)), num_display_images)

    # Predict labels for the selected images
    predicted_probs = model.predict(images.reshape(-1, 144, crop_width, 1))
    predicted_labels = np.argmax(predicted_probs, axis=1)
    predicted_class_names = label_encoder.inverse_transform(predicted_labels)

    # Display the images and predictions
    fig, axes = plt.subplots(1, num_display_images, figsize=(15, 5))

    for i, idx in enumerate(selected_indices):
        axes[i].imshow(images[idx], cmap="gray")
        axes[i].set_title(
            f"True: {true_labels[idx]}\nPredicted: {predicted_class_names[idx]}"
        )
        axes[i].axis("off")

    plt.show()


# Load and preprocess data
images, labels = load_and_preprocess_data(data_path, crop_width)

# Display a random preview of a few cropped images
num_preview_images = 3
fig, axes = plt.subplots(1, num_preview_images, figsize=(15, 5))

for i in range(num_preview_images):
    random_index = np.random.randint(0, len(images))
    axes[i].imshow(images[random_index], cmap="gray")
    axes[i].set_title(f"Class: {labels[random_index]}")
    axes[i].axis("off")

plt.show()

# Encode labels
label_encoder = LabelEncoder()
encoded_labels = label_encoder.fit_transform(labels)

# Convert labels to one-hot encoding
one_hot_labels = to_categorical(encoded_labels)

# Split the data into train, test, and validation sets
X_train, X_temp, y_train, y_temp = train_test_split(
    images, one_hot_labels, test_size=0.2, random_state=598
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=598
)

# Build the CNN model
model = models.Sequential()
model.add(layers.Conv2D(32, (3, 3), activation="relu", input_shape=(144, crop_width, 1)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation="relu"))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Flatten())
model.add(layers.Dense(64, activation="relu"))
model.add(layers.Dense(3, activation="softmax"))

# Compile the model
model.compile(
    optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"]
)

# Train the model
model.fit(
    X_train.reshape(-1, 144, crop_width, 1),
    y_train,
    epochs=10,
    validation_data=(X_val.reshape(-1, 144, crop_width, 1), y_val),
)

# Evaluate the model on the test set
test_loss, test_acc = model.evaluate(X_test.reshape(-1, 144, crop_width, 1), y_test)
print(f"Test accuracy: {test_acc}")


display_predictions(model, X_test, label_encoder.inverse_transform(np.argmax(y_test, axis=1)), label_encoder)


#%%
# Predict labels for the test set
y_pred = model.predict(X_test.reshape(-1, 144, crop_width, 1))
y_pred_classes = np.argmax(y_pred, axis=1)
y_true_classes = np.argmax(y_test, axis=1)

# Create confusion matrix
conf_mat = confusion_matrix(y_true_classes, y_pred_classes)

# Plot the confusion matrix using seaborn
class_names = label_encoder.classes_
df_conf_mat = pd.DataFrame(conf_mat, index=class_names, columns=class_names)

plt.figure(figsize=(8, 6))
sns.heatmap(df_conf_mat, annot=True, fmt="d", cmap="Blues", cbar=False)
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("True")
plt.show()