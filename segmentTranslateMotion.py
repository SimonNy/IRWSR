#%%
"""
Segments an image from the background, motionEstimates and translates
saves new image and motionEstimates
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from funcs.loadFrames import loadFrames
from funcs.cropFrames import cropFrames
from funcs.enhanceFrames import enhanceFrames
from funcs.findSegmentMask import findSegmentMask
from funcs.Shift import Shift
from funcs.imageRegistration import imageRegistration

""" Start of loading translation images """
# readFolder = "degradation/translationPrintboard/translation10ms/"
# imageName = "10ms"
# # readFolder = "degradation/translationPrintboard/translation50ms/"
# # imageName = "50ms"
# readFolder = "degradation/translationPrintboard/translation200ms/"
# imageName = "200ms"
# # # the last part of the image name used in loading the different files
# imageNumber = 2
# filetype = ".ppm"
# # Constants for translation
# cropFrame = True
# cropRows = (500, 1150)
# cropCols = (1700, 2600)
# refFrame = 3
# N_frames = 7
# readFolder = "degradation/translationPrintboard/translation200Image/"
# imageName = "200ms"
# # # the last part of the image name used in loading the different files
# imageNumber = 2
# filetype = ".ppm"
# # Constants for translation
# cropFrame = True
# cropRows = (500, 1150)
# cropCols = (1700, 2600)
# refFrame = 1
# N_frames = 2

# For making images
# N_frames
# medianBackground = False
# """ Start of loading conveyor images """
# imageName = "recordedImage00"
# imageName = "Image"
# filetype = ".ppm"

# readFolder = "degradation/potatoSingle300Withheart70Test2/"
# readFolder = "degradation/potatoSingle300Withheart70Test3/"
# readFolder = "degradation/potatoSingle300Withheart70Test4/"
# readFolder = "degradation/potatoSingle300Withheart70Test5/"
readFolder = "degradation/potatoSingle300WithheartSlide/"
# readFolder = "degradation/boardWithMarksMotionBlur/"
# readFolder = "degradation/boardWithMarksHighMotionBlur/"
imageNumber = 1
# # Finds the region of interest in all the images
# cropRows = (420, 620)
# cropCols = (660, 860)
cropFrame = False
imageName = "potatoSingle300Withheart70"
filetype = ".png"
N_frames = 7
refFrame = 3

# readFolder = "degradation/conveyorPrintboard/13600micros/"
# imageNumber = 31
# N_back = 8
# N_frames = 8
# readFolder = "degradation/conveyorPrintboard/16ms/"
# imageNumber = 82
# N_back = 8
# N_frames = 8
# readFolder = "degradation/conveyorPrintboard/25ms/"
# imageNumber = 75
# N_back = 5
# N_frames = 8
# readFolder = "degradation/conveyorPrintboard/50ms/"
# imageNumber = 24
# N_back = 3
# N_frames = 8
# readFolder = "degradation/conveyorPrintboard/100ms/"
# imageNumber = 21
# N_frames = 6
# N_back = 3
# refFrame = 3
# medianBackground =  True
# cropFrame = True
# cropRows = (80, 1370)
# cropCols = (130, 1100)

""" For ruler """
# readFolder = "degradation/conveyorRuler/rulerWellLit/"
# imageNumber = 10
# readFolder = "degradation/conveyorRuler/rulerInBetween/"
# imageNumber = 36
# readFolder = "degradation/conveyorRuler/rulerUnderLit/"
# imageNumber = 30
# readFolder = "degradation/conveyorRuler/rulerWellLitNoConveyor/"
# imageNumber = 8
# N_frames = 4

# Choose between "single" background, "median" of images and "none"
# medianBackground =  "single"
medianBackground =  "none"
# cropFrame = True
# cropRows = (450, 1200)
# cropCols = (800, 1350)

# Image segmented before optical flow?
# segmentImages = False
segmentImages = False

""" User defined variables """
displayImages = True
# Preprocesses image with either histogram equalization,
# constrast limited adaptive histogram equalization
# or gamma correction, enhancePalameter = gamma
# None for no enhacement
# enhanceType = "equalizeHist"
enhanceType = "None"
# enhanceType = "gammaCorrection"
enhanceParameter = 1

# number of orb descriminator to find
nOrbs = 10000
# number of matches to use in affine estimate
nMatches = 50
# Uses opticalFlow to find the motionEstimates
opticalFlow = True
# opticalFlow = False

# return a smaller crop of the images after cropping
transReduce = True
cropRows2 = (50, 150)
cropCols2 = (50, 150)

#%%
# Load frames
frames = loadFrames(readFolder, imageName, imageNumber, filetype, N_frames)

if cropFrame:
    frames = cropFrames(frames, cropRows, cropCols)

# The background is either defined as the timeseries median of multipleframes
# Or just a singlee empty frame
if medianBackground == "median":
    background = loadFrames(readFolder, imageName, 10, filetype, N_back)
    background = np.median(background, axis=2).astype('uint8')
elif medianBackground == "single":
    medianKernel = 5
    background = cv2.imread(readFolder + imageName + 'Background' + filetype, 0)
    # Even out noise in background
    background = cv2.medianBlur(background, medianKernel)

if medianBackground != "none":
    # Define newaxis so all frames can subtract background in one line
    background = background[:,:,np.newaxis]
    if cropFrame:
        background = cropFrames(background, cropRows, cropCols)

    framesSubtracted = 255 - np.clip(frames - background, 0, 255)
else: 
    framesSubtracted = 255 - frames

if displayImages:
    print("Shows the first frame, the background and the subtracted background")
    plt.imshow(frames[:, :, 0], cmap='gray', vmin=0, vmax=255)
    plt.show()
    if medianBackground != "none":
        plt.imshow(background[:, :, 0], cmap='gray', vmin=0, vmax=255)
        plt.show()
    plt.imshow(framesSubtracted[:, :, 0], cmap="gray", vmin=0, vmax=255)
    plt.show()


# Enhance Frames
if segmentImages:
    segmented = np.zeros_like(frames)
    for i in range(N_frames):
        img_temp = np.copy(framesSubtracted[:,:,i])
        mask = findSegmentMask(img_temp)
        img_temp[mask == 0] = 255
        segmented[:,:,i] = img_temp
        if displayImages:
            print("Shows segmented images")
            plt.imshow(img_temp, cmap='gray', vmin=0, vmax=255)
            plt.show()
    framesEnhanced = enhanceFrames(segmented, enhanceType, enhanceParameter)
else:
    framesEnhanced = enhanceFrames(framesSubtracted, enhanceType, enhanceParameter)

# Finds the motion estimates
motionEstimates, transFrames = imageRegistration(framesEnhanced, nOrbs,
                                                 nMatches, refFrame,
                                                 opticalFlow, displayImages)
np.save(readFolder+"motionEstimatesOrg", motionEstimates)

if displayImages:
    for i in range(N_frames):
        print("Shows all frames and the enhanced features ")
        print(f"Used {enhanceType} as enhanchment")

        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(frames[:, :, i], cmap="gray", vmin=0, vmax=255)
        ax[1].imshow(framesEnhanced[:, :, i],
                     cmap="gray", vmin=0, vmax=255)
        plt.show()
    for i in range(N_frames):
        print(f"Showing image {i}")
        print('translated image left and reference right')
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(transFrames[:, :, i], cmap="gray", vmin=0, vmax=255)
        ax[1].imshow(transFrames[:, :, refFrame],
                     cmap="gray", vmin=0, vmax=255)
        plt.show()


#%% Saves a scaled down version of the images with and without 
# background subtraction enhancement
if transReduce:
    img_trans = Shift(frames, motionEstimates)
    img_transSub = Shift(framesSubtracted, motionEstimates)
    transFramesReduced = cropFrames(img_trans, cropRows2, cropCols2)
    transFramesSub = cropFrames(img_transSub, cropRows2, cropCols2)
    for i in range(N_frames):
        cv2.imwrite(readFolder +
                    f"/transCrop{imageNumber+i}.png",
                    transFramesReduced[:, :, i])
        cv2.imwrite(readFolder +
                    f"/transCropSub{imageNumber+i}.png",
                    transFramesSub[:, :, i])

    # finds new  motionEstimates for the cropped image
    imageName = f"transCropSub"
    filetype = ".png"

    frames = loadFrames(readFolder, imageName, imageNumber, filetype, N_frames)
    framesEnhanced = enhanceFrames(frames, enhanceType, enhanceParameter)
    # Finds the motion estimates
    motionEstimates, transFrames = imageRegistration(framesEnhanced, nOrbs,
                                                     nMatches, refFrame,
                                                     opticalFlow=True,
                                                     displayImages=displayImages)

    np.save(readFolder+f"motionEstimatesTransCrop", motionEstimates)

    if displayImages:
        for i in range(N_frames):
            print("Shows all frames and the enhanced features ")
            print(f"Used {enhanceType} as enhanchment")

            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(frames[:, :, i], cmap="gray", vmin=0, vmax=255)
            ax[1].imshow(framesEnhanced[:, :, i],
                         cmap="gray", vmin=0, vmax=255)
            plt.show()
        for i in range(N_frames):
            print(f"Showing image {i}")
            print('translated image left and reference right')
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(transFrames[:, :, i], cmap="gray", vmin=0, vmax=255)
            ax[1].imshow(transFrames[:, :, refFrame],
                         cmap="gray", vmin=0, vmax=255)
            plt.show()

# %%
