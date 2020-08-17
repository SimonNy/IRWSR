"""
Translates all the image frames by the given motion estimate
and returns the translated frames
"""

import numpy as np
import cv2

def Shift(img_frames, motion_estimates):
    img_trans = np.zeros(img_frames.shape)
    rows, cols, nframes = img_frames.shape
    for i in range(nframes):
        idx = (slice(None), slice(None), i)
        x_trans, y_trans = motion_estimates[:, i]
        # Needs to translate by integer value or blur
        T = np.float32([[1, 0, int(x_trans)], [0, 1, int(y_trans)]])
        img_trans[idx] = cv2.warpAffine(img_frames[idx], T, (cols, rows),
                        cv2.INTER_LANCZOS4)
    return img_trans
