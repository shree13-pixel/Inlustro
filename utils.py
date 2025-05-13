import cv2
import numpy as np
from PIL import Image

def load_image(path):
    return cv2.imread(path)

def preprocess_image(image, size=(224, 224)):
    image = cv2.resize(image, size)
    image = image.astype(np.float32) / 255.0
    return image.transpose(2, 0, 1)
