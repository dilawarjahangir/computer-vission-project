import cv2

# Extract spatial binning features
def extract_spatial_binning(image, size=(32, 32)):
    return cv2.resize(image, size).ravel()
