import cv2
import numpy as np

def extract_gabor_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    num_filters = 5
    gabor_features = []
    for theta in range(num_filters):
        theta = theta / float(num_filters) * np.pi
        kernel = cv2.getGaborKernel((31, 31), 4.0, theta, 10.0, 0.5, 0, ktype=cv2.CV_32F)
        filtered_image = cv2.filter2D(gray_image, cv2.CV_8UC3, kernel)
        gabor_features.append(filtered_image.flatten())
    return np.concatenate(gabor_features)
