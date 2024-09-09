from skimage.feature import hog
import cv2

def extract_hog_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    hog_features, hog_image = hog(gray_image, 
                                  orientations=9, 
                                  pixels_per_cell=(8, 8), 
                                  cells_per_block=(2, 2), 
                                  block_norm='L2-Hys', 
                                  visualize=True, 
                                  feature_vector=True)
    return hog_features
