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

readFolder_list = [
    # "degradation/potatoSingle300Withheart70Test2/",
    # "degradation/potatoSingle300Withheart70Test3/",
    # "degradation/potatoSingle300Withheart70Test4/",
    "degradation/potatoSingle300WithheartSlide/"
    # "degradation/boardWithMarksNoMotionBlur/",
    # "degradation/boardWithMarksHighMotionBlur/"
                    ]
# imageNumber_list = [1, 1]
imageNumber_list = [1]
# Finds the region of interest in all the images

cropRows = (420, 620)
cropCols = (660, 860)
cropFrame = False
# imageName = "boardWithMarks_carbon_iron100x200x300"
imageName = "potatoSingle300Withheart70"


filetype = ".png"
saveFig = True
displayImages = True
# the frame used for reference in the motion estimates
refFrame = 3
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
    img_ref = cv2.imread(readFolder +
                         imageName + "ray_ccdX5" +
                         filetype, 0)
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
    motionEstimates = np.load(readFolder+'motionEstimatesOrg.npy')
    #%%
    from IRWSR_func import IRWSR_func
    from funcs.PSNR import PSNR, SSIM
    SR, img_med, alpha_img, beta_img, lamb, totTime = IRWSR_func(
        frames, motionEstimates, saveFig, displayImages)

    # the number LR frame used for comparison 
    frame_pick = refFrame

    SR_dB = PSNR(SR, img_ref)
    SR_SSIM = SSIM(SR, img_ref)
    rows, cols = SR.shape
    img_med_upscale = cv2.resize(
        img_med, (cols, rows), interpolation=cv2.INTER_AREA)
    med_dB = PSNR(img_med_upscale, img_ref)
    med_SSIM = SSIM(img_med_upscale, img_ref)
    first_frame_upscale = cv2.resize(
        frames[:, :, frame_pick], (cols, rows), interpolation=cv2.INTER_AREA)
    low_dB = PSNR(first_frame_upscale, img_ref)
    low_SSIM = SSIM(first_frame_upscale, img_ref)
    print("The super-resolved image")
    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
    ax[0][0].imshow(SR, cmap='gray', vmin=0, vmax=255)
    ax[0][0].set_title(f"SR ({SR_dB:.2f} dB, {SR_SSIM:.2f})")
    ax[0][1].imshow(img_ref, cmap='gray', vmin=0, vmax=255)
    ax[0][1].set_title("Reference")
    ax[1][0].imshow(img_med_upscale, cmap='gray', vmin=0, vmax=255)
    ax[1][0].set_title(f"median ({med_dB:.2f} dB, {med_SSIM:.2f})")
    ax[1][1].imshow(first_frame_upscale, cmap='gray', vmin=0, vmax=255)
    ax[1][1].set_title(
        f"reference low Resolution frame ({low_dB:.2f} dB, {low_SSIM:.2f})")
    plt.show()

    writeFolder = "results/"
    filename = readFolder.split('/')[-2]

    createDirectories(writeFolder, filename)
    
    if saveFig:
        fig.savefig("results/"+filename + f"/Measure.png", dpi=600)

    SRHistEq = cv2.equalizeHist(SR.astype('uint8'))
    img_medHistEq = cv2.equalizeHist(img_med.astype('uint8'))
    firstHistEq = cv2.equalizeHist(frames[:, :, frame_pick])
    SRNorm = (normalize(SR)*255).astype('uint8')
    img_medNorm = (normalize(img_med)*255).astype('uint8')
    firstNorm = (normalize(frames[:, :, frame_pick])*255).astype('uint8')
    fig, ax = plt.subplots(ncols=3, nrows=3)
    ax[0][0].imshow(SR, cmap='gray', vmin=0, vmax=255)
    ax[0][0].set_title(f"SR")
    ax[1][0].imshow(SRHistEq, cmap='gray', vmin=0, vmax=255)
    ax[1][0].set_title(f"SR histogram Eq")
    ax[2][0].imshow(SRNorm, cmap='gray', vmin=0, vmax=255)
    ax[2][0].set_title(f"SRnorm ")
    ax[0][1].imshow(img_med, cmap='gray',  vmin=0, vmax=255)
    ax[0][1].set_title(f"median")
    ax[1][1].imshow(img_medHistEq, cmap='gray', vmin=0, vmax=255)
    ax[1][1].set_title(f"median histogram Eq")
    ax[2][1].imshow(img_medNorm, cmap='gray', vmin=0, vmax=255)
    ax[2][1].set_title(f"median Norm")
    ax[0][2].imshow(frames[:, :, frame_pick], cmap='gray',  vmin=0, vmax=255)
    ax[0][2].set_title(f"FirstFrame")
    ax[1][2].imshow(firstHistEq, cmap='gray',  vmin=0, vmax=255)
    ax[1][2].set_title(f"first frame hist eq")
    ax[2][2].imshow(firstNorm, cmap='gray',  vmin=0, vmax=255)
    ax[2][2].set_title(f"first frame Norm")
    plt.show()
    if saveFig == True:
        fig.savefig(writeFolder + filename + '/'
                    f"subplots.png", dpi=600)  # %%

    fig, ax = plt.subplots(ncols=3)
    ax[0].hist(SR.ravel(), bins=255, range=(0, 255))
    ax[0].set_title(f"SR histogram")
    ax[1].hist(img_med.ravel(), bins=255, range=(0, 255))
    ax[1].set_title(f"median histogram")
    ax[2].hist(frames[:, :, frame_pick].ravel(), bins=255, range=(0, 255))
    ax[2].set_title(f"first frame")
    if saveFig == True:
        fig.savefig(writeFolder + filename + '/' +
                    f"histograms.png", dpi=600)  # %%

        cv2.imwrite(writeFolder + filename + '/SR' + ".png", SR)
        cv2.imwrite(writeFolder + filename +
                    '/SRHistEq' + ".png", SRHistEq)
        cv2.imwrite(writeFolder + filename +
                    '/SRNorm' + ".png", SRNorm)
        cv2.imwrite(writeFolder + filename +
                    '/img_med' + ".png", img_med)
        cv2.imwrite(writeFolder + filename +
                    '/img_medHistEq' + ".png", img_medHistEq)
        cv2.imwrite(writeFolder + filename +
                    '/img_medNorm' + ".png", img_medNorm)
        cv2.imwrite(writeFolder + filename +
                    '/ref' + ".png", frames[:, :, frame_pick])
        cv2.imwrite(writeFolder + filename +
                    '/refHistEq' + ".png", firstHistEq)
        cv2.imwrite(writeFolder + filename +
                    '/refNorm' + ".png", firstNorm)
        cv2.imwrite(writeFolder + filename +
                    '/alpha' + ".png", alpha_img)
        cv2.imwrite(writeFolder + filename +
                        '/beta' + ".png", beta_img)
        cv2.imwrite(writeFolder + filename + '/alpha_norm' + ".png",
                        (normalize(alpha_img)*255).astype('uint8'))
        cv2.imwrite(writeFolder + filename + '/beta_norm' + ".png",
                        (normalize(beta_img)*255).astype('uint8'))
    # Saves relevant parameters in csv file
    from defineParameters import (delta_s, s, minIntensity, maxIntensity,
                                    sigma0_noise, psfWidth, cBias, cLocal,
                                    sigma0_prior, P, alphaBTV, cPrior, p, deltaCV,
                                    Tcv, opts, degradationType, optimizationType,
                                    crossValidation, lambL, lambU)

    const_names = ['delta_s', 's', 'minIntensity', 'maxInt', 'sigma0_noise',
                    'psfWidth', 'cBias', 'cLocal', 'sigma0_prior', 'P', 'alphaBTV',
                    'cPrior', 'p', 'deltaCV', 'Tcv', 'opts', 'degrationType',
                    'optimizationType', 'crossValidation', 'lamb', 'lambL', 'lambU',
                    'total time']
    const_vals = [delta_s, s, minIntensity, maxIntensity,
                    sigma0_noise, psfWidth, cBias, cLocal,
                    sigma0_prior, P, alphaBTV, cPrior, p, deltaCV,
                    Tcv, opts, degradationType, optimizationType,
                    crossValidation, lamb, lambL, lambU, totTime]
    if saveFig:
        with open(writeFolder+filename+'/details.csv', 'w', ) as myfile:
            wr = csv.writer(myfile)
            # wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
            test = zip(const_names, const_vals)
            for row in test:
                wr.writerow([row])
# %%

# %%

# %%
