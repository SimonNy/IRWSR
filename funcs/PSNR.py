import cv2
import imutils
import argparse
from skimage.measure import compare_ssim
import numpy as np

def PSNR(x, y):
    """ Calculates the Peak Signal-To-Noise Between two 2D images of same size"""
    """ x: Reference image, y: test image """
    # x_vec = x.reshape(x.shape[0]*x.shape[1])
    # y_vec = y.reshape(y.shape[0]*y.shape[1])
    x_vec = x.ravel()
    y_vec = y.ravel()
    PSNR = 10*np.log10(255**2/(1/256**2*np.linalg.norm(y_vec-x_vec)**2))
    return PSNR


"""
Finds the SSIM
# Based o n: https://github.com/mostafaGwely/Structural-Similarity-Index-SSIM-
Based on:
https: //ourcodeworld.com/articles/read/991/how-to-calculate-the-structural-similarity-index-ssim-between-two-images-with-python#:~:text=Software%20Over%20SaaS-,How%20to%20calculate%20the%20Structural%20Similarity%20Index%20(SSIM,between%20two%20images%20with%20Python&text=The%20Structural%20Similarity%20Index%20(SSIM)%20is%20a%20perceptual%20metric%20that,by%20losses%20in%20data%20transmission.
"""
def SSIM(x, y):
    (score, diff) = compare_ssim(x, y, full=True)
    diff = (diff * 255).astype("uint8")
    return score
