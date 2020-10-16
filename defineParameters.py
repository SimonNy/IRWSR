#%%
""" User-defined parameters for the system """
import numpy as np
from IRWSR_func import IRWSR_func
from funcs.PSNR import PSNR

# Step magnification in coarse-to-scale
delta_s = 0.1
# final magnification factor
s = 2

# intensity range of images
minIntensity = 0
maxIntensity = 1

""" Constants relevant image degradation, likelihood and the beta calculations """
# Constant to make sigma_noise consistent with std of gauss
sigma0_noise = 1.4826
# Assumed PSF width in degradation
# psfWidth = 0.4
psfWidth = 0.7
# Confidence value for the betaBias vector
# default is 0.02 of maxIntensity
cBias = 0.02 * maxIntensity
# cBias = 0.2 * maxIntensity
# Confidence value for the betaLocal vector
cLocal = 2

""" Constants relevant for the prior and the alpha calculations """
# constant to make sigma_prior consistent with std of laplacian
sigma0_prior = 1.0
# Defines window size for derivatives
P = 2
#  Alpha for Bileteral total variation
alphaBTV = 0.7
# Confidence value for the alpha vector
cPrior = 2
# sparsity parameter
p = 0.5

""" Parameters used in cross validation """
# fraction not used for parameter training
# crossValidation = False
crossValidation = True
deltaCV = 0.95
Tcv = 10
# lambL = -12
lambL = -5
lambU = 0
# lamb = 0.0215
lamb = 0.008
""" Choices in program """
# degradationType: 0 uses convolution and other image functions
# degradationTYpe: 1 constructs a W matrix with respect to x_img_vec
degradationType = 1
optimizationType = 0
priorType = 0

# Performs historgram equlization as a preprocessing measure
histEqualization = False

""" Optimization options """
# Options for the scipy optimization
opts = {'maxiter': 300,
        'disp': True,  # default value
        'gtol': 1e-7,  # default value
        'norm': np.inf,  # default value
        'eps': 1.4901161193847656e-08,  # default value
        }

""" Input """

saveFig = True
displayImages = False



# %%

# %%
