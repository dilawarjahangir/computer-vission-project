from tensorflow.keras.applications import VGG16
from tensorflow.keras.applications.vgg16 import preprocess_input
import numpy as np
import cv2

vgg_model = VGG16(weights="imagenet", include_top=False)

def extract_cnn_features(image):
    resized_image = cv2.resize(image, (224, 224))
    preprocessed_image = preprocess_input(resized_image)
    features = vgg_model.predict(np.expand_dims(preprocessed_image, axis=0))
    return features.flatten()
