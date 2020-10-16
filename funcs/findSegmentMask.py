#%%
""" Loads file and subtract object by segmentation and saves the new files. 
Inspired by
https://stackoverflow.com/questions/44588279/find-and-draw-the-largest-contour-in-opencv-on-a-specific-color-python
https://www.pyimagesearch.com/2015/01/19/find-distance-camera-objectmarker-using-python-opencv/
"""
# import the necessary packages

import numpy as np
import matplotlib.pyplot as plt
import cv2
import imutils


def findSegmentMask(image):
    # convert the image to grayscale, blur it, and detect edges
    kernel = np.ones((9, 9), np.uint8)
    # gamma = 7
    # image = np.clip((image/255)**gamma*255, 0, 255).astype('uint8')
    # image = cv2.equalizeHist(image)
    # morphology approach to enhance edges
    image = cv2.dilate(image, kernel, iterations=3)
    image = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)
    # image = cv2.GaussianBlur(image, (9, 9), 0)
    edged = cv2.Canny(image, 35, 125)
    plt.imshow(edged)
    plt.show()
    # cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, 
                            # cv2.CHAIN_APPROX_SIMPLE)
    cnts = cv2.findContours(edged.copy(), cv2.RETR_LIST, 
                            cv2.CHAIN_APPROX_NONE)
    cnts = imutils.grab_contours(cnts)
    c = max(cnts, key=cv2.contourArea)
    # c = max(cnts, key=cv2.boundingRect)
    # returnt angled bounding box
    marker = cv2.minAreaRect(c)
    
    box = cv2.cv.BoxPoints(marker) if imutils.is_cv2() else cv2.boxPoints(marker)
    box = np.int0(box)

    mask = cv2.drawContours(np.zeros_like(image), [box], -1, (1), -1)
    return mask

# %%
