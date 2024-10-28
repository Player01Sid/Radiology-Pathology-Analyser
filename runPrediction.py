import os
import cv2
from PIL import Image
import numpy as np
from keras.models import load_model

def makePrediction(model,img):

    img = cv2.imread(img)
    img = Image.fromarray(img)
    img = img.resize((160,160))
    img = np.array(img)

    input_img = np.expand_dims(img, axis=0)

    prediction= model.predict(input_img)

    if prediction[0][0] <= 0.5:
      return f"Tumor present."
    else:
      return f"Tumor not present."

