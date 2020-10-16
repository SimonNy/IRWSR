""" 
Image Registration by Feature Matching 
Using ORB descriptors and Brute-Force Matching:
https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_feature2d/py_matcher/py_matcher.html
Estimating an affine transformation between the two - giving the motion estimates
inspired by 
https://www.geeksforgeeks.org/image-registration-using-opencv-python/

Optical Flow inspired by:
https: // nanonets.com/blog/optical-flow/
https: // www.learnopencv.com/image-alignment-ecc-in-opencv-c-python/
https: // opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_video/py_lucas_kanade/py_lucas_kanade.html


The images are registered to match the refFrame in sequence. 
The flow: First registers the image after and previous the refFrame
          translates the images
          Takes the frames 2 steps previous and after those frames and 
          registers with the translated frames
          Transate them

Image enhanchement may increase the amount of ORB descriptors
 - histogramEqualization seems to work well


After the function a short snippet of the code is used crop interesting 
parts of the image after translation - to reduce the amount of data in IRWSR
"""

import numpy as np
import cv2
import matplotlib.pyplot as plt

from funcs.Shift import Shift

def findAndMatch(img1, img2, nOrbs, nMatches, displayImages = False):
    orbs = cv2.ORB_create(nOrbs)
    keypoints1, d1 = orbs.detectAndCompute(img1, None)
    keypoints2, d2 = orbs.detectAndCompute(img2, None)

    bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck = True)
    matches = bf.match(d1, d2)
    # sort the mathes
    matches.sort(key = lambda x: x.distance)

    # number of mathces
    # matches = matches[:int(len(matches)*90)]
    matches = matches[:nMatches]
    no_of_matches = len(matches)

    p1 = np.zeros((no_of_matches,2))
    p2 = np.zeros((no_of_matches,2))
    for i in range(no_of_matches):
        p1[i,:] = keypoints1[matches[i].queryIdx].pt
        p2[i,:] = keypoints2[matches[i].trainIdx].pt

    # plotting
    if displayImages:
        print(f"DisplayImages: {displayImages}")
        print("show orbs")
        img1_orbs = cv2.drawKeypoints(img1, keypoints1, np.array([]), 
                                        color=(0, 255, 0), flags=0)    
        img2_orbs = cv2.drawKeypoints(img2, keypoints2, np.array([]), 
                                        color=(0, 255, 0), flags=0)
        fig, ax = plt.subplots(ncols = 2)
        ax[0].imshow(img1_orbs)
        ax[1].imshow(img2_orbs)
        plt.show()
        print("show the matching features")
        match_img = cv2.drawMatches(img1, keypoints1, img2, keypoints2, matches,
                                    None)
        plt.imshow(match_img)
        plt.show()
    return p1, p2


def findOpticalFlow(img1, img2, displayImages):
    # Coloridx = (0,slice(None),0) to show track
    color = (0, 255, 0)
    # Implementing SHI TOMASI corner detection
    # Parameters for Shi-Tomasi corner detection
    st_params = dict(maxCorners=500,
                     qualityLevel=0.02,
                     minDistance=5,
                     blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=4,
                     criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10, 0.03))
    # Thresholding before feature extraction
    gray1 = cv2.medianBlur(img1, 5)
    gray2 = cv2.medianBlur(img2, 5)
    img_thres, gray1 = cv2.threshold(gray1, 0, 255,
                                     cv2.THRESH_BINARY+cv2.THRESH_OTSU)
    _, gray2 = cv2.threshold(gray2, img_thres, 255,
                             cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    p1 = cv2.goodFeaturesToTrack(gray1, mask=None, **st_params)
    p2, status, _ = cv2.calcOpticalFlowPyrLK(
        gray1, gray2, p1, None, **lk_params)

    good_p1 = p1[status == 1]
    good_p2 = p2[status == 1]

    if displayImages:
        mask = np.zeros((img1.shape[0], img1.shape[1], 3))
        # Draws the optical flow tracks
        for j, (new, old) in enumerate(zip(good_p2, good_p1)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color, 2).astype("uint8")
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            frame_test = cv2.cvtColor(img2, cv2.COLOR_GRAY2RGB)
        # Overlays the optical flow tracks on the original frame
        fig, ax = plt.subplots(ncols=2)
        ax[0].imshow(img1, cmap="gray", vmin=0, vmax=255)
        ax[1].imshow(img2, cmap="gray", vmin=0, vmax=255)
        plt.show()
        plt.show()
        output = cv2.add(frame_test, mask)
        plt.imshow(output)
        plt.show()

    return good_p1, good_p2

def findAffine(p1, p2, removeRot=True):
    """ 
    estimates the linear translation between points
    """
    affine, _ = cv2.estimateAffine2D(p1, p2)

    if removeRot:
        affine[0, 0] = 1
        affine[1, 1] = 1
        affine[0, 1] = 0
        affine[1, 0] = 0
    print(affine)
    # affine[:,2] = -affine[:,2]
    return affine

def findNeighbor(frames, idx, const, nOrbs, nMatches, opticalFlow = True, 
                displayImages=False):
    # Finds the translation with respect to the refFrame
    # eiter by orbs or optical flow
    if opticalFlow:
        p1, p2 = findOpticalFlow(frames[:, :, idx],
                              frames[:, :, idx + const],
                              displayImages)
    else:
        p1, p2 = findAndMatch(frames[:, :, idx],
                        frames[:, :, idx + const],
                        nOrbs, nMatches, displayImages)
    # estimate affine translation
    rows, cols, _ = frames.shape
    affine = findAffine(p1, p2, True)
    # traslate the image with respect to the found affine transformation
    transFrame = cv2.warpAffine(frames[:, :, idx],
                                affine, (cols, rows), cv2.INTER_LANCZOS4)
    return affine[:, 2], transFrame


def imageRegistration(frames, nOrbs, nMatches, refFrame, opticalFlow, 
                        displayImages=False):
    """ Alternative reference scheme 
        starts from reference frame and goes in each direction, translates
        and finds the next in line
    """
    _, _, N_frames = frames.shape
    motionEstimates = np.zeros((2, N_frames))
    # Need to copy ?
    frames = frames.astype("uint8")
    const = 1
    for i in range(1, N_frames):
        backIdx = refFrame - i
        nextIdx = refFrame + i
        if backIdx >= 0:
            motionEstimates[:, backIdx], frames[:, :, backIdx] \
                = findNeighbor(frames, backIdx, const, nOrbs, nMatches,
                               opticalFlow, displayImages)
        if nextIdx <= N_frames-1:
            motionEstimates[:, nextIdx], frames[:, :, nextIdx] \
                = findNeighbor(frames, nextIdx, -const, nOrbs, nMatches,
                               opticalFlow, displayImages)
        # Add one if with respect to referencne frame instead
        const += 1

    # saves the motionEstimates

    return motionEstimates, frames









def opticalFlowEstimation(frames, displayImages=False):
    rows, cols, N_frames = frames.shape
    # Coloridx = (0,slice(None),0) to show track
    color = (0, 255, 0)
    # Implementing SHI TOMASI corner detection
    # Parameters for Shi-Tomasi corner detection
    st_params = dict(maxCorners=500,
                     qualityLevel=0.02,
                     minDistance=5,
                     blockSize=7)
    # Parameters for Lucas-Kanade optical flow
    lk_params = dict(winSize=(30, 30),
                     maxLevel=4,
                     criteria=(
        cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT,
        10, 0.03))
    first_frame = frames[:, :, 0]
    # Thresholding before feature extraction
    prev_gray = cv2.medianBlur(first_frame, 5)
    img_thres, prev_gray = cv2.threshold(prev_gray, 0, 255,
                                         cv2.THRESH_BINARY+cv2.THRESH_OTSU)

    # plt.imshow(prev_gray, cmap='gray')

    p0 = cv2.goodFeaturesToTrack(prev_gray, mask=None, **st_params)

    # Shows the corners
    # corners = np.int0(prev)
    # img = np.copy(prev_gray)
    # for i in corners:
    #     x, y = i.ravel()
    #     cv2.circle(img, (x, y), 3, 255, -1)
    # plt.imshow(img)
    mask = np.zeros((rows, cols, 3))

    motionEstimates = np.zeros((2, N_frames))
    for i in range(N_frames):
        # for i in range(N_frames):
        # ret = a boolean return value from getting the frame, frame = the current frame being projected in the video

        frame = frames[:, :, i]
        gray = cv2.medianBlur(frame, 5)
        ret, gray = cv2.threshold(gray, img_thres, 255,
                                  cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        # https://docs.opencv.org/3.0-beta/modules/video/doc/motion_analysis_and_object_tracking.html#calcopticalflowpyrlk

        p1, status, error = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None, **lk_params)

        if p1 is None:
            break
        print(p0.shape)
        print(p1.shape)

        good_old = p0[status == 1]
        good_new = p1[status == 1]
        motionEstimates[:, i] = findAffine(good_old, good_new)[:, 2]
        # Draws the optical flow tracks
        # updates p0 in each frame 
        """ should reconsider """
        p0 = cv2.goodFeaturesToTrack(gray, mask=None, **st_params)
        for j, (new, old) in enumerate(zip(good_new, good_old)):
            # Returns a contiguous flattened array as (x, y) coordinates for new point
            a, b = new.ravel()
            # Returns a contiguous flattened array as (x, y) coordinates for old point
            c, d = old.ravel()
            # Draws line between new and old position with green color and 2 thickness
            mask = cv2.line(mask, (a, b), (c, d), color, 2).astype("uint8")
            # Draws filled circle (thickness of -1) at new position with green color and radius of 3
            backtorgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            frame_test = cv2.circle(backtorgb, (a, b), 10, color, -1)
            # frame = cv2.drawKeypoints(frame, (a, b), np.array([]),
            #   color=(0, 255, 0), flags=0)
        # Overlays the optical flow tracks on the original frame
        print(f"Sparse optical Flow iteration {i}")
        if displayImages:
            plt.imshow(frame, cmap="gray", vmin=0, vmax=255)
            plt.show()
            output = cv2.add(frame_test, mask)
            p0 = good_new.reshape(-1, 1, 2)
            plt.imshow(output)
            plt.show()

    img_trans = Shift(frames, -motionEstimates)
    if displayImages:
        for i in range(N_frames):
            fig, ax = plt.subplots(ncols=2)
            ax[0].imshow(img_trans[:, :, i], cmap="gray")
            ax[1].imshow(frames[:, :, i], cmap="gray")
            plt.show()
    return -motionEstimates, img_trans
