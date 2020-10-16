import numpy as np
import cv2

def loadFrames(readFolder, imageName, imageNumber, filetype, N_frames):
    """ 
    Load the desired images from a folder an puts in a 3D array. 
    readFolder: path to all images
    imageName: The name of the image files
    imageNumber: The numbers to iterate over i.e the first image end number
    filetype: filetype of the images
    N_frames: number of frames
    removeBackground: loads a file ending with background and subtracts it
    """
    # loads first image for specifing array
    print(f"Reading first image")
    print("Path: " + readFolder + imageName +
                   f'{imageNumber}' + filetype)
    img = cv2.imread(readFolder + 
                    imageName + 
                    f'{imageNumber}' +
                    filetype, cv2.IMREAD_ANYDEPTH)
    rows, cols = img.shape

    print(f"Each frame has a size of {rows}x{cols}")
    print(f"Reading {N_frames} frames in total")

    frames = np.zeros([rows, cols, N_frames])
    
    for i in range(N_frames):
        frames[:, :, i] = cv2.imread(readFolder+
                                    imageName +
                                    f'{imageNumber+i}' + 
                                    filetype, 0)
    return frames.astype('uint8')
