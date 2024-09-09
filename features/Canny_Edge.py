import numpy as np
import cv2

def cannyEdge(img):
    med_value = np.median(img)
    lower = int(max(0, 0.7 * med_value))
    upper = int(min(255, 1.13 * med_value))
    blured_img = cv2.GaussianBlur(img, (7, 7), 0)
    edge = cv2.Canny(image=blured_img, threshold1=lower, threshold2=upper)
    
    return edge