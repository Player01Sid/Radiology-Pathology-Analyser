import os
import cv2
import numpy as np
from keras.models import load_model

# Load the saved model
model = load_model("Brain_MRI_scan.h5")

# Folder containing images for prediction
prediction_folder = "archive/pred/"

# Function to load and preprocess an image
def load_and_preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = cv2.resize(img, (160, 160))  # Resize to (256, 256)
    img = img.astype('float32') / 159.0  # Normalize pixel values
    img = np.expand_dims(img, axis=0)  # Add batch dimension
    return img

# Make predictions on images in the prediction folder
# for filename in os.listdir(prediction_folder):
#     if filename.endswith(".jpg") or filename.endswith(".png"):  # Assuming images are .jpg or .png
#         image_path = os.path.join(prediction_folder, filename)
#         image = load_and_preprocess_image(image_path)
        
#         # Perform prediction
#         prediction = model.predict(image)
        
#         # If you have binary classification (yes/no)


img = load_and_preprocess_image(prediction_folder+"predTest.jpg")
prediction = model.predict(img)
print(prediction)

if prediction[0][0] <= 0.5:
    result = "With Tumor"
else:
    result = "No Tumor"

print(f"Image: {prediction_folder+"predTest.jpg"}, Prediction: {result}")
