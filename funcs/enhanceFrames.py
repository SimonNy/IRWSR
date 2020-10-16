import numpy as np
import cv2

def enhanceImage(img, enhanceType, enhanceParameter):
    img = img.astype("uint8")
    if enhanceType == "equalizeHist":
        img = cv2.equalizeHist(img)
    elif enhanceType == "CLAHE":
        clahe = cv2.createCLAHE(clipLimit=10.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
    elif enhanceType == "histAndMedFilter":
        img = cv2.medianBlur(img, enhanceParameter)
        img = cv2.equalizeHist(img)
    elif enhanceType == "gammaCorrection":
        img = np.clip((img/255)**enhanceParameter*255, 0, 255)
    elif enhanceType == "None":
        img = img
    else:
        print(f"enhanceType: {enhanceType} dosen't exist")

    return img.astype('uint8')

def enhanceFrames(img, enhanceType, enhanceParameter=1):
    # Enhance 1 image or all frames in the array
    if len(img.shape) == 2 or img.shape[2] == 1:
        frames = enhanceImage(img, enhanceType, enhanceParameter)
    elif img.shape[2] > 1:
        frames = np.zeros(img.shape)
        for i in range(frames.shape[2]):
            frames[:, :, i] = enhanceImage(img[:,:,i], enhanceType,
                                            enhanceParameter)

    return frames.astype('uint8')
