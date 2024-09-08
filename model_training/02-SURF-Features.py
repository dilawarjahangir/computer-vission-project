import cv2

def extract_surf_features(image):
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    surf = cv2.xfeatures2d.SURF_create()
    keypoints, descriptors = surf.detectAndCompute(gray_image, None)
    return descriptors
