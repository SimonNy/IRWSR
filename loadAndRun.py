#%%
""" 
Loads the data for the IRWSR and runs the script 
user-defined variables defined in the file defineParameters.py
"""
import numpy as np
import matplotlib.pyplot as plt
import cv2
import csv
import os

from funcs.loadFrames import loadFrames
from funcs.cropFrames import cropFrames
from funcs.enhanceFrames import enhanceFrames
from IRWSR_func import IRWSR_func
from funcs.normalize import normalize
from funcs.createDirectories import createDirectories

readFolder_list = ["degradation/conveyorPrintboard/13600micros/",
                    "degradation/conveyorPrintboard/16ms/",
                    "degradation/conveyorPrintboard/25ms/",
                    "degradation/conveyorPrintboard/50ms/"]
imageNumber_list = 31, 82, 75, 24
# Finds the region of interest in all the images
# # Constants for conveyor belt
# cropRows = (250, 450)
# cropCols = (20, 220)
cropRows = (370, 620)
# cropRows = (420, 620)
cropCols = (660, 860)
cropFrame = True
imageName = "transCrop"
# imageName = "transCropSub"


# Read all translations
# readFolder_list = [
#             "degradation/translationPrintboard/translation10ms/",
#             "degradation/translationPrintboard/translation50ms/",
#             "degradation/translationPrintboard/translation200ms/"
#             ]
# imageNumber_list = [1,1,2]
# # imageNumber_list = [1,2]
# # imageNumber_list = [1]
# # Constants for translation
# cropRows = (200, 400)
# cropCols = (0, 200)
# cropFrame = True
# imageName = "transCrop"
# # imageName = "transCropSub"

# Ruler


# readFolder_list = ["degradation/conveyorRuler/rulerInBetween/"]
# imageNumber_list = [36]
# # Constants for translation
# cropFrame = True
# imageName = "transCrop"
# cropRows = (200, 500)
# cropCols = (150, 400)

# # Simulated
# readFolder_list = ["degradation/simulatedBoardWithMarks/boardWithMarks1/",
#             "degradation/simulatedBoardWithMarks/boardWithMarks2/",
#             "degradation/simulatedBoardWithMarks/boardWithMarks3/"]
# imageNumber_list = [0,0,0]
# # Constants for translation
# cropFrame = False
# imageName = "img"


# imageName = "transCropSub"
# the last part of the image name used in loading the different files
# filetype = ".ppm"
filetype = ".png"
saveFig = True
displayImages = True

# the frame which references the translations if refFrame = [] appending order
N_frames = 7

# Preprocesses image with either histogram equalization,
# constrast limited adaptive histogram equalization
# or gamma correction, enhancePalameter = gamma
# None for no enhacement
# enhanceType = "equalizeHist"
enhanceType = "None"
# enhanceType = "gammaCorrection"
enhanceParameter = 2



for i in range(len(readFolder_list)):
    readFolder = readFolder_list[i]
    imageNumber = imageNumber_list[i]

    # show Intermediate images

    # Load frames
    frames = loadFrames(readFolder, imageName, imageNumber, filetype, N_frames)
    # for i in range(N_frames):
    #     frames[:,:,i] = normalize(frames[:,:,i])*255
    # Crop Frames
    if cropFrame:
        frames = cropFrames(frames, cropRows, cropCols)
    if displayImages:
        print("shows first frame cropped")
        plt.imshow(frames[:,:,0], cmap="gray", vmin=0, vmax=255)
    # Enhance Frames
    framesEnhanced = enhanceFrames(frames, enhanceType, enhanceParameter)

    # Starts by loading the motionEstimates
    # motionEstimates = np.load(readFolder+'motionEstimates.npy')
    motionEstimates = np.load(readFolder+'motionEstimatesTransCrop.npy')
    #%%
    SR, img_med, alpha_img, beta_img, lamb, totTime = IRWSR_func(framesEnhanced, motionEstimates, saveFig, displayImages)

    # SR_norm = normalize(SR.ravel()) * 255
    # SR_norm = normalize(SR.ravel()) * 255
    # SR_norm = normalize(SR.ravel()) * 255

    writeFolder = "results/"
    filename = readFolder.split('/')[-2]

    createDirectories(writeFolder, filename)

    SRHistEq = cv2.equalizeHist(SR.astype('uint8'))
    img_medHistEq = cv2.equalizeHist(img_med.astype('uint8'))
    firstHistEq = cv2.equalizeHist(framesEnhanced[:, :, 0])
    SRNorm = (normalize(SR)*255).astype('uint8')
    img_medNorm = (normalize(img_med)*255).astype('uint8')
    firstNorm = (normalize(framesEnhanced[:, :, 0])*255).astype('uint8')
    fig, ax = plt.subplots(ncols=3, nrows =3)
    ax[0][0].imshow(SR, cmap='gray', vmin=0, vmax=255)
    ax[0][0].set_title(f"SR")
    ax[1][0].imshow(SRHistEq, cmap='gray', vmin=0, vmax=255)
    ax[1][0].set_title(f"SR histogram Eq")
    ax[2][0].imshow(SRNorm, cmap='gray', vmin=0, vmax=255)
    ax[2][0].set_title(f"SRnorm ")
    ax[0][1].imshow(img_med, cmap='gray',  vmin=0, vmax=255)
    ax[0][1].set_title(f"median")
    ax[1][1].imshow(img_medHistEq, cmap='gray',vmin=0, vmax=255)
    ax[1][1].set_title(f"median histogram Eq")
    ax[2][1].imshow(img_medNorm, cmap='gray',vmin=0, vmax=255)
    ax[2][1].set_title(f"median Norm")
    ax[0][2].imshow(frames[:, :, 0], cmap='gray',  vmin=0, vmax=255)
    ax[0][2].set_title(f"FirstFrame")
    ax[1][2].imshow(firstHistEq, cmap='gray',  vmin=0, vmax=255)
    ax[1][2].set_title(f"first frame hist eq")
    ax[2][2].imshow(firstNorm, cmap='gray',  vmin=0, vmax=255)
    ax[2][2].set_title(f"first frame Norm")
    plt.show()
    if saveFig == True:
        fig.savefig(writeFolder + filename + '/'
                    f"subplots.png", dpi=600)# %%

    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(SR.ravel(), bins=255, range=(0, 255))
    ax[0].set_title(f"SR histogram")
    ax[1].hist(img_med.ravel(), bins=255, range=(0, 255))
    ax[1].set_title(f"median histogram")
    ax[2].hist(frames[:, :, 0].ravel(), bins=255, range=(0, 255))
    ax[2].set_title(f"first frame")
    if saveFig == True:
        fig.savefig(writeFolder + filename + '/' +
                    f"histograms.png", dpi=600)  # %%

        cv2.imwrite(writeFolder + filename + '/SR' + ".png", SR)
        cv2.imwrite(writeFolder + filename + '/SRHistEq' + ".png", SRHistEq)
        cv2.imwrite(writeFolder + filename + '/SRNorm' + ".png", SRNorm)
        cv2.imwrite(writeFolder + filename + '/img_med' + ".png", img_med)
        cv2.imwrite(writeFolder + filename + '/img_medHistEq' + ".png", img_medHistEq)
        cv2.imwrite(writeFolder + filename + '/img_medNorm' + ".png", img_medNorm)
        cv2.imwrite(writeFolder + filename + '/first' + ".png", frames[:,:,0])
        cv2.imwrite(writeFolder + filename + '/firstHistEq' + ".png", firstHistEq)
        cv2.imwrite(writeFolder + filename + '/firstNorm' + ".png", firstNorm)
        cv2.imwrite(writeFolder + filename + '/alpha' + ".png", alpha_img)
        cv2.imwrite(writeFolder + filename + '/beta' + ".png", beta_img)
        cv2.imwrite(writeFolder + filename + '/alpha_norm' + ".png",
            (normalize(alpha_img)*255).astype('uint8'))
        cv2.imwrite(writeFolder + filename + '/beta_norm' + ".png", 
            (normalize(beta_img)*255).astype('uint8'))

    # Saves relevant parameters in csv file
    from defineParameters import (delta_s, s, minIntensity, maxIntensity, \
                                    sigma0_noise, psfWidth, cBias, cLocal, \
                                    sigma0_prior, P, alphaBTV, cPrior, p, deltaCV, \
                                    Tcv, opts, degradationType, optimizationType, \
                                    crossValidation, lambL, lambU)

    const_names=['delta_s', 's', 'minIntensity', 'maxInt', 'sigma0_noise', 
                'psfWidth', 'cBias', 'cLocal', 'sigma0_prior', 'P', 'alphaBTV', 
                'cPrior', 'p', 'deltaCV', 'Tcv', 'opts', 'degrationType', 
                'optimizationType', 'crossValidation', 'lamb', 'lambL', 'lambU',
                'total time']
    const_vals=[delta_s, s, minIntensity, maxIntensity, \
                sigma0_noise, psfWidth, cBias, cLocal, \
                sigma0_prior, P, alphaBTV, cPrior, p, deltaCV, \
                Tcv, opts, degradationType, optimizationType, \
                crossValidation, lamb, lambL, lambU, totTime]
    if saveFig:
        with open(writeFolder+filename+'/details.csv', 'w', ) as myfile:
            wr=csv.writer(myfile)
            # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            test=zip(const_names, const_vals)
            for row in test:
                wr.writerow([row])

# %%

# %%

# %%
