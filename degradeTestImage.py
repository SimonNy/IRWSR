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
from scipy.sparse import csr_matrix, rand, diags
from scipy import stats

from funcs.MFDegradation import MFDegradation
from funcs.Shift import Shift
from funcs.composeSystemMatrix import composeSystemMatrix
from funcs.weighted_median import weighted_median
from funcs.weightedMAD import weightedMAD
from funcs.createSStack import createSStack
from funcs.stackW import stackW
from funcs.normalize import normalize

from scipy.sparse import identity
from funcs.findHRimage import findHRimage

# import cv2
read_folder = "data/"
write_folder = "degradation/"

# filename = "barbaraCrop.png"
img_filename = "ClassicTestCrop3.jpg"
# filename = "barbara.png"
randomSeed = 21

img_ref = cv2.imread(read_folder + img_filename, 0)

filename = img_filename.split('.')[0]
# Variables for degration
nframes = 8
samplingFactor = 0.25
translationFactor = 3
blur_fact = 0.6
noise_fact = 3
# Artificial degradation of reference image
frames, trans = MFDegradation(img_ref, nframes, samplingFactor, translationFactor, 
                              blur_fact, noise_fact, show_img=True, 
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

# %%
