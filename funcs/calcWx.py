""" Chooses between the 3 different types of degradation """
import numpy as np
import cv2
import matplotlib.pyplot as plt

def calcWx(HR_vec, LR_shape, HR_shape, N_frames, sigma, magFactor, 
                    translations, type, W = 0):
    """ Gathers the different approaches for W = DHM 
    type = 0 degradation by convolution and nearest neighbor interpolation on
             the image array
    type = 1 degradation by the gathering of W and the dot product with the 
             image vector
    type = 2 degradation by the creation of and image matrix with each column
             being the active area of the kernel with the image 
             i.e. H_vec.dot(X_matrix)
    HR_vec: Image to degrade
    LR_shape: Size of one low resolution frame
    N_frames: Amount of low resolution frames
    psfWidth: sigma value of the gausskernel in the H step
    magnificationFactor: The factor between the HR and the LR image size
    translations: The translationss for each frame
    W: for type 1 or 2. The matrix generated with the function stackW
    """
    M1, M2 = LR_shape
    M = M1 * M2

    if type == 0:
        LR_vec = np.zeros(M*N_frames)
        N1, N2 = HR_shape
        HR_vec = HR_vec.reshape(N1, N2)
        for i in range(N_frames):
            # Index start for given frame in vector
            idx = i*M
            # translate image
            x_trans, y_trans = -translations[:, i]
            # Affine translations matrix
            T = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
            # Affine transaltion
            img_deg = cv2.warpAffine(HR_vec, T, (N2, N1))
            # Blur image
            img_deg = cv2.GaussianBlur(img_deg, ksize=(11, 11), sigmaX=sigma)
            # downsample image 
            img_deg = cv2.resize(img_deg, (M2, M1), 
                            interpolation=cv2.INTER_NEAREST)
            # add noise to image?
            # img_deg += np.random.normal(size=(M1, M2), scale=1)

            LR_vec[idx:idx+M] = img_deg.ravel()

    elif type == 1:
        LR_vec = W.dot(HR_vec)
    return LR_vec


def calcWtransposed(LR_vec, LR_shape, HR_shape, N_frames, sigma, magFactor,
           translations, type, W=0):
    """ Calculates the different version of W transposed used in gradf
    type = 0 degradation by convolution and nearest neighbor interpolation on
             the image array - uses the fact that a zero padded convolution 
             transposed is a padded convolution see https://arxiv.org/pdf/1603.07285.pdf
    type = 1 degradation by the gathering of W and the dot product with the 
             image vector
    type = 2 degradation by the creation of and image matrix with each column
             being the active area of the kernel with the image 
             i.e. H_vec.dot(X_matrix)

    """
    M1, M2 = LR_shape
    M = M1 * M2

    if type == 0:
        """ Is this the right way to do it? """
        N1, N2 = HR_shape
        N = N1 * N2
        WT_vec = np.zeros(N)
        for i in range(N_frames):
            # Index start for given frame in vector
            idx = i*M
            LR_img = LR_vec[idx:idx+M].reshape(M1, M2)
            # upscale image
            img_deg = cv2.resize(LR_img, (N2, N1),
                                 interpolation=cv2.INTER_NEAREST)
            k = 11
            p = k - 1
            # pads image
            img_deg = cv2.copyMakeBorder(img_deg, p, p, p, p, 
                                        cv2.BORDER_CONSTANT, None, 0)
            # Blur image
            img_deg = cv2.GaussianBlur(img_deg, ksize=(k, k), sigmaX=sigma,
                                        borderType=cv2.BORDER_CONSTANT)

            # assume that the transposed of the translation is the reverse?
            x_trans, y_trans = translations[:, i]
            # Affine translations matrix
            T = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
            # Affine transaltion
            img_deg = cv2.warpAffine(img_deg, T, (N2, N1))
            # add noise to image?
            # img_deg += np.random.normal(size=(M1, M2), scale=1)

            WT_vec += img_deg.ravel()

    elif type == 1:
        WT_vec = W.T.dot(LR_vec)
    return WT_vec
