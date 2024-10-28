import os
import numpy as np
import cv2
from PIL import Image
from sklearn.model_selection import train_test_split
from tensorflow import keras
from keras import layers, models
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical

RESO=160
# Define path to your dataset
dataset_path = "archive/"

# Function to load images and labels
def load_images_from_folder(folder):
    images = []
    labels = []
    for label, subfolder in enumerate(["no", "yes"]):
        subfolder_path = os.path.join(folder, subfolder)
        for filename in os.listdir(subfolder_path):
            img = cv2.imread(os.path.join(subfolder_path, filename))
            img = Image.fromarray(img)
            img = img.resize((RESO,RESO))
            img = np.array(img)
            images.append(img)
            labels.append(label)
    return np.array(images), np.array(labels)

# Load images and labels
images, labels = load_images_from_folder(dataset_path)

# Normalize pixel values to [0, 1]
images = images.astype('float32') / 159.0

# Convert labels to one-hot encoding
labels = to_categorical(labels)

# Split dataset into training and validation sets
train_images, val_images, train_labels, val_labels = train_test_split(
    images, labels, test_size=0.2, random_state=42)

# Define your CNN model
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(RESO, RESO, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),  # Optional dropout for regularization
    layers.Dense(2, activation='softmax')  # Output layer for binary classification
])

# Compile the model
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Train the model
history = model.fit(train_images, train_labels, epochs=5, batch_size=32,
                    validation_data=(val_images, val_labels))

# Evaluate on test set (optional)

model.save("Brain_MRI_scanner.h5")
print("Model saved successfully as 'brain_tumor_detection_model.h5'")
# test_loss, test_acc = model.evaluate(test_images, test_labels)
# print("Test Accuracy:", test_acc)
