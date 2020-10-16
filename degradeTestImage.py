#%%
"""
Iterative Re-Weighted SR test.
Arcificially degrades a test image into multiple frames translated by
subpixeldistances and returns estimated SR

"""
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt

from funcs.MFDegradation import MFDegradation
from funcs.Shift import Shift
from funcs.createDirectories import createDirectories
from funcs.normalize import normalize 
import csv
import os

# import cv2
read_folder = "data/"
write_folder = "degradation/"


filename_list = [
                # "mandril_grayCrop.tif",
                # "cameraman.png"
                "barbaraCrop.png",
                # "barbara.png",
                # "butterflyCrop.bmp",
                # "classicTestCrop2.jpg",
                # "peppers_grayCrop.tif",
                # "printcardCrop.jpg",
                ]
randomSeed = 3
# Variables for degration
nframes = 6
# samplingFactor = 1/1.5
samplingFactor = 1/2
translationFactor = 5
blur_fact = 0.28
noise_fact = 3
# downscale and Upscales back to previous size
# downscaleUpscale = False
downscaleUpscale = False

saveFig = True
displayImages = False

for i in range(len(filename_list)):
        img_filename = filename_list[i]
        img_ref = cv2.imread(read_folder + img_filename, 0)

        filename = img_filename.split('.')[0]

        # Artificial degradation of reference image
        frames, trans = MFDegradation(img_ref, nframes, samplingFactor, translationFactor, 
                                blur_fact, noise_fact, downscaleUpscale,
                                show_img=True, 
                                filename=filename, folder = write_folder,
                                rand_seed=randomSeed)
        # Initial guess should be the temporal median of motion compensated low-resolution frames
        # Problem specific
        motionEstimates = - trans * samplingFactor
        # saves the motionEstimates 
        np.save(write_folder+'/'+filename+'/'+"motionEstimates", motionEstimates)

        img_trans = Shift(frames, motionEstimates)

        img_med = np.median(img_trans, axis = 2)

        fig, ax = plt.subplots(ncols = 3, figsize = (12, 20))
        ax[0].imshow(img_ref, cmap = "gray")
        ax[0].set_title("org")
        ax[1].imshow(img_med, cmap = "gray")
        ax[1].set_title("median")
        ax[2].imshow(img_trans[:,:,0], cmap = "gray")
        ax[2].set_title("first frame translated")


        from IRWSR_func import IRWSR_func
        from funcs.PSNR import PSNR, SSIM
        SR, img_med, alpha_img, beta_img, lamb, totTime = IRWSR_func(frames, motionEstimates, saveFig, displayImages)


        SR_dB = PSNR(SR, img_ref)
        SR_SSIM = SSIM(SR, img_ref)
        rows, cols = SR.shape
        img_med_upscale = cv2.resize(img_med, (cols, rows), interpolation=cv2.INTER_AREA)
        med_dB = PSNR(img_med_upscale, img_ref)
        med_SSIM = SSIM(img_med_upscale, img_ref)
        first_frame_upscale = cv2.resize(frames[:, :, 0], (cols, rows), interpolation=cv2.INTER_AREA)
        low_dB = PSNR(first_frame_upscale, img_ref)
        low_SSIM = SSIM(first_frame_upscale, img_ref)
        print("The super-resolved image")
        fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(12, 12))
        ax[0][0].imshow(SR, cmap='gray', vmin = 0, vmax=255)
        ax[0][0].set_title(f"SR ({SR_dB:.2f} dB, {SR_SSIM:.2f})")
        ax[0][1].imshow(img_ref, cmap='gray', vmin =0, vmax=255)
        ax[0][1].set_title("Reference")
        ax[1][0].imshow(img_med_upscale, cmap='gray', vmin =0, vmax =255)
        ax[1][0].set_title(f"median ({med_dB:.2f} dB, {med_SSIM:.2f})")
        ax[1][1].imshow(first_frame_upscale, cmap='gray', vmin=0, vmax =255)
        ax[1][1].set_title(
            f"1st low Resolution frame ({low_dB:.2f} dB, {low_SSIM:.2f})")
        plt.show()



        writeFolder = "results/"
        # filename = readFolder.split('/')[-2]

        createDirectories(writeFolder, filename)

        if saveFig:
                fig.savefig("results/"+filename +f"/Measure.png", dpi=600)

        SRHistEq = cv2.equalizeHist(SR.astype('uint8'))
        img_medHistEq = cv2.equalizeHist(img_med.astype('uint8'))
        firstHistEq = cv2.equalizeHist(frames[:, :, 0])
        SRNorm = (normalize(SR)*255).astype('uint8')
        img_medNorm = (normalize(img_med)*255).astype('uint8')
        firstNorm = (normalize(frames[:, :, 0])*255).astype('uint8')
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
        ax[0][2].imshow(frames[:, :, 0], cmap='gray',  vmin=0, vmax=255)
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
        ax[0].hist(SR.ravel(), bins=255, range=(0,255))
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
                cv2.imwrite(writeFolder + filename +
                        '/img_medHistEq' + ".png", img_medHistEq)
                cv2.imwrite(writeFolder + filename +
                        '/img_medNorm' + ".png", img_medNorm)
                cv2.imwrite(writeFolder + filename +
                        '/first' + ".png", frames[:, :, 0])
                cv2.imwrite(writeFolder + filename +
                        '/firstHistEq' + ".png", firstHistEq)
                cv2.imwrite(writeFolder + filename + '/firstNorm' + ".png", firstNorm)
                cv2.imwrite(writeFolder + filename + '/alpha' + ".png", alpha_img)
                cv2.imwrite(writeFolder + filename + '/beta' + ".png", beta_img)
                cv2.imwrite(writeFolder + filename + '/alpha_norm' + ".png",
                        (normalize(alpha_img)*255).astype('uint8'))
                cv2.imwrite(writeFolder + filename + '/beta_norm' + ".png", 
                        (normalize(beta_img)*255).astype('uint8'))
                for i in range(nframes):
                   cv2.imwrite(writeFolder  + filename +
                                 f"/LRimg{i}.png", frames[:, :, i])
                np.save(write_folder+filename+'/' +
                        "motionEstimates", motionEstimates)
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
