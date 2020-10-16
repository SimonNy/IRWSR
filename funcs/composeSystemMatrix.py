""" 
Compose System Matrix - find matrix W in y^k = W^k x
W = DHM^k, D: is subsampling, H is blur and M^k is translation of the given frame.
\hat{x} positive right(cols), \hat{y} positive down(rows)

inputs:
lrsize: LR image size (M1 x M2), rows, cols
s: magnification factor
psfWidth: The std of the point spread function - assumed Gaussian
motionEstimates: The motion estimates for the given frames(assume first frame has zero translation)

output:
W of size (M1 * M2) x (M1*s * M2*s)
"""
""" NEED TO DEFINE M1, M2, M, N1, N2 and N in function """
""" As for now cols > rows, need to fix """
""" Be aware of double definition difference in s """
#%%
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse import lil_matrix

def composeSystemMatrix(lrshape, magFactor, psfWidth, motionEstimate):
    # LR shape
    M1, M2 = lrshape
    M = M1 * M2
    # HR shape
    N1, N2 = int(round(M1 * magFactor)), int(round(M2 * magFactor))
    N = N1 * N2
    # Define all the pixels in LR
    uX, uY = np.arange(0, M2), np.arange(0, M1)
    # Define all the pixels in HR
    vX, vY = np.arange(0, N2), np.arange(0, N1)
    # max distance of the supported part of the psf. 
    # guarantees that it is atleast 1 pixel wide
    maxPsfRange = max(3 * psfWidth * magFactor, 1)
    # Find subpixel positions of the given LR frame in the HR
    uPrimeX, uPrimeY = (uX + motionEstimate[0])*magFactor, (uY + motionEstimate[1])*magFactor
    # initiate the matrix W 
    """ Very inefficent to fill a spare matrix """
    # W = csr_matrix((M, N))
    W = lil_matrix((M, N))

    # iterate over every pixel in LR. 
    for i_y in range(M1):
        for i_x in range(M2):
            # finds distance between all pixels in HR and u'
            distX, distY = np.abs(uPrimeX[i_x] - vX), np.abs(uPrimeY[i_y] - vY)
            # mask defining the squared support of the PSF
            maskX, maskY = distX <= maxPsfRange, distY <= maxPsfRange
            dist = np.meshgrid(distX[maskX],distY[maskY])
            # Finds all euclidian distances within the supported square
            dist = np.sqrt(dist[0]**2 + dist[1]**2)
            # mask defining the  radial support of the PSF
            mask = dist <= maxPsfRange
            # calc exponents in gaussianblur
            weights = np.exp(- dist / (2 * magFactor**2 * psfWidth**2))
            weights[mask != True] = 0

            # normalize
            weights = weights/np.sum(weights)
            # Avoid nans
            weights[np.isnan(weights)] = 0
            idx_u = i_x + i_y*M2
            # Every row in W corres
            # ponds to a u in y, put corresponding weights at the right column
            idx_v = np.meshgrid(vX[maskX], vY[maskY]*N2)
            idx_v = np.ravel(idx_v[0] + idx_v[1])
            W[idx_u, idx_v] = np.ravel(weights)
    return W

# %% test
# import cv2
# read_folder = "data/"
# write_folder = "results/"

# filename = "barbaraCrop.png"
# # filename = "ClassicTestCrop.jpg"

# img_ref = cv2.imread(read_folder + filename, 0)
# nframes = 3
# samp_fact = 0.5
# trans_fact = 2
# blur_fact = 2
# # Artificial degradation of reference image
# frames, trans = MFDegradation(img_ref, nframes, samp_fact,
#                               trans_fact, blur_fact, show_img=False)
# M1, M2, K = frames.shape
# mot_est = - trans * samp_fact
# W = composeSystemMatrix((M1,M2), 2, 0.4, mot_est[:,1])
# # # Check if it works"
# img_vec = np.ravel(img_ref)
# img_vec = (img_vec - np.min(img_vec)/(np.max(img_vec) - np.min(img_vec)))
# test = np.reshape(W.dot(img_vec), [M1, M2]).astype("uint8")
# plt.imshow(test, cmap="gray")



# %%


# %%
