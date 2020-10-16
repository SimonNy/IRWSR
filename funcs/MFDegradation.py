#%%
""" 
Simple multiframe degradation

Takes an image and translates, blurs and downsamples to multiple low resolution frames

Inputs:
img: image to degrade
nframe: Number of generated frames
samp_fact: The degree of downsampling
trans_fact: All translations in x and y drawn from random[0, trans_fact]
blur_fact: All gaussian blur std range from random[0, blur_fact] - maybe rethink? make constant?

Outputs:
img_deg: Array of degraded frames with size [rows, cols, N]
trans: All translations with reference to the first LR frame
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from funcs.createDirectories import createDirectories
from funcs.normalize import normalize

def MFDegradation(img, nframes, samp_fact, trans_fact=1, blur_fact=1, 
                    noise_fact=0.5, upscale = False, show_img=False, 
                    filename = "undefined", 
                    folder = "degradation", rand_seed = []):
    rows, cols = img.shape
    Drows, Dcols = np.floor((rows * samp_fact, cols * samp_fact)).astype(int)
    
    # creates a directory to save all the files
    createDirectories(folder, filename)


    # Sets random seed
    if rand_seed:
        np.random.seed(rand_seed)
    # Prepare image array for different states
    img_trans = np.zeros([rows, cols, nframes])
    img_blur = np.zeros([rows, cols, nframes])
    img_down = np.zeros([Drows, Dcols, nframes])
    if upscale:
        img_deg = np.zeros([rows, cols, nframes])
    else:
        img_deg = np.zeros([Drows, Dcols, nframes])
    
    trans = np.round(np.random.rand(2, nframes) * trans_fact, 2) - trans_fact / 2

    # # Slice after translation to avoid dark borders
    # regOfInt = (slice(rows-trans_fact//2, rows + trans_fact//2),
    #             slice(cols-trans_fact//2, cols + trans_fact//2))

    # trans[:, 0] = (0, 0)
    for i in range(nframes):
        # random translations
        x_trans, y_trans = trans[:, i]
        # Id for the i'th frame
        idx = (slice(None), slice(None), i)
        # Affine translation matrix
        T = np.float32([[1, 0, x_trans], [0, 1, y_trans]])
        # Affine transaltion
        img_trans[idx] = cv2.warpAffine(img, T, (cols, rows))

        """ tried to box so no dark edges """
        # img_trans[idx] = img_trans[regOfInt]
        # Random blur size
        # img_blur[idx] = cv2.GaussianBlur(img_trans[idx], ksize=(11, 11),
                                        #  sigmaX=np.random.rand(1)*blur_fact)
        # NonRandom blur size
        img_blur[idx] = cv2.GaussianBlur(img_trans[idx], ksize=(11, 11),
                                         sigmaX=blur_fact)
        img_down[idx] = cv2.resize(
        # img_blur[idx], (Dcols, Drows), interpolation=cv2.INTER_AREA)
            img_blur[idx], (Dcols, Drows), interpolation=cv2.INTER_NEAREST)
        if upscale:
            temp = cv2.resize(
                img_down[idx], (cols, rows), interpolation=cv2.INTER_AREA)
            img_deg[idx] = temp + \
                        np.random.normal(size=(rows, cols), scale=noise_fact)
        else:    
            img_deg[idx] = img_down[idx] + \
                np.random.normal(size=(Drows, Dcols), scale=noise_fact)
        # Added noise exceeds values given by 8bit image
        img_deg[img_deg[idx]>255,i] = 255
        img_deg[img_deg[idx]<0,i] = 0
        # Saves all imgaes in the corresponding folder
        
    if show_img == True:
        figT, axT = plt.subplots(ncols=nframes, figsize=(24, 60))
        figB, axB = plt.subplots(ncols=nframes, figsize=(24, 60))
        figD, axD = plt.subplots(ncols=nframes, figsize=(24, 60))
        for i in range(nframes):
            cv2.imwrite(folder + '/' + filename + f"/img{i}.png", img_deg[:,:,i])
            axT[i].imshow(img_trans[:, :, i], cmap="gray")
            axB[i].imshow(img_blur[:, :, i], cmap="gray")
            axD[i].imshow(img_deg[:, :, i], cmap="gray")
        
    print(f"{nframes} created with translations:")
    print(trans)
    return img_deg.astype("uint8"), trans


# %%

# %%
