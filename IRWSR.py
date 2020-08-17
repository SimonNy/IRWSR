#%%
"""
Iterative Re-Weighted SR test.
Arcificially degrades a test image into multiple frames translated by
subpixeldistances and returns estimated SR
          
"""
from funcs.gradf import gradf
from funcs.f import f
from scipy import optimize
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

readFolder = "degradation/"
writeFolder = "results/"
saveFig = True

RegOfInt = (slice(550, 1100), slice(750, 1250), slice(None))

randomSeed = 21

# filename = "potatoSingle"
filename = "classicTestCrop3"
filetype = ".png"

# Starts by loading the motionEstimaets 
motionEstimates = np.load(readFolder+filename+'/'+'motionEstimates.npy')
# Data structure is the cols define the frames
# Number of frames to take into account.
N = np.shape(motionEstimates)[1]
# filename, file_type = file.split('.')
# read with open cv, 0 denotes grayscale
img = cv2.imread(readFolder +'/' + filename + '/img0' + filetype, 0)
rows, cols = img.shape


print(f"Each frame has a size of {rows}x{cols}")
print(f"Reading {N} frames in total")

frames = np.zeros([rows, cols, N])
for i in range(N):
    frames[:, :, i] = cv2.imread(readFolder+'/'+filename+f'/img{i}'+
                                 filetype, 0)
fig, ax = plt.subplots(ncols=2)
ax[0].imshow(frames[:, :, 0], cmap="gray", vmin=0, vmax=255)
ax[0].set_title("frame 0")
ax[1].imshow(frames[:, :, 1], cmap="gray", vmin=0, vmax=255)
ax[1].set_title("frame 1")
# ax[2].imshow(frames[:, :, 2], cmap="gray", vmin=0, vmax=255)
# ax[2].set_title("frame 2")
if saveFig == True:
     fig.savefig(writeFolder+"lowResFrames"+filename+".png",dpi=600)
# Initial guess should be the temporal median of motion compensated 
# low-resolution frames
# # Problem specific
# trans = np.array([[0, 0, 0], [0, 20, 40]])
# motionEstimates = - trans


img_trans = Shift(frames, motionEstimates)
img_med = np.median(img_trans, axis = 2)
cv2.imwrite(writeFolder + "imgMedTest.png", img_med)

fig, ax = plt.subplots(ncols = 2, figsize = (12, 20))
ax[0].imshow(img_med, cmap = "gray", vmin=0, vmax=255)
ax[0].set_title("median")
ax[1].imshow(img_trans[:,:,0], cmap = "gray", vmin=0, vmax=255)
ax[1].set_title("first frame translated")

for i in range(8):
     # shows all frames with corresponding translations
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[:, :, i], cmap="gray")
    ax[1].imshow(img_trans[:, :, i], cmap="gray")
    fig.show()

# %%
""" Initiate variables """
M1, M2, K = frames.shape
M = M1 * M2

#Initial magnification factor
s0 = 1
# Step magnification in coarse-to-scale
delta_s = 0.1
# final magnification factor
s = 2

# iteration number
t = 1
# Defines window size for derivatives
P = 2

# Constant to make sigma_noise consistent with std of gauss
sigma0_noise = 1.4826
# constant to make sigma_prior consistent with std of laplacian
sigma0_prior = 1.0
# Assumed PSF width
psfWidth = 0.4

print(f"{K} low resolution frames of size {M1}x{M2} = {M}")
# print(f"Magnified by {s} to SR image of size {N1}x{N2} = {N} ")

""" Why did I flip the motionEstimates? """
# motionEstimates = np.flipud(motionEstimates)
# %%
s_old = s0

N1, N2 = int(round(M1 * (s0 + delta_s))), int(round(M2 * (s0 + delta_s)))
N = N1 * N2

# Initial observation weights
alpha_old = np.ones(N)
beta_old = np.ones(K * M)
x_old = img_med
# convert frames to a vector in lexicographical order
y_vec = np.zeros(M*K)
for idx in range(K):
     y_vec[idx*M:idx*M+M] = frames[:, :, idx].ravel()
y_vec = normalize(y_vec, 0, 255)
""" Content of the loop """
t = 1
# while s_old <= s:
# stepwise scaling until s is acheived
s_new = np.min([s_old + delta_s, s])
# Find size of HR image with new scalng
N1, N2 = int(round(M1 * s_new)), int(round(M2 * s_new))
N = N1 * N2

Ns = (2*P + 1)**2 * N

# Propagate x to new size and define as vector
x_old = cv2.resize(x_old,(N2, N1), cv2.INTER_CUBIC)
x_vec = np.ravel(x_old)
x_vec = normalize(x_vec, 0, 255)

# Finds the matrix W which defines the relation between x and y_frames
W = stackW((M1, M2), K, s_new, psfWidth, motionEstimates)
# Calculates the residual between the given frames and Wx
residual = y_vec - W.dot(x_vec)

for i in range(8):
    # shows all frames with corresponding translations
    fig, ax = plt.subplots(ncols=2)
    ax[0].imshow(frames[:, :, i], cmap="gray")
    ax[1].imshow(W.dot(x_vec)[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
    fig.show()
#%%

# calculate sigma noise
sigma_noise = sigma0_noise * weightedMAD(residual, beta_old)
# calculate sigma prior
alphaBTV = 0.7
# Finds the S matrix that defines the derivatives in the different directions

S = createSStack((N1,N2), P, alphaBTV)
# Calculate Sx and convert format to fit with openCV
Sx = (S.dot(x_vec))
# Sx = (normalize(Sx)*255).astype('uint8')
SxTemp = Sx.reshape(N1*(P*2+1)**2, N2)
QSx = signal.medfilt2d(SxTemp, 3).ravel()

sigma_prior = sigma0_prior * \
     weightedMAD(QSx, alpha_old)

# betaBias finds outlier frames, sets confidence
cbias = 0.05 * 1
# betaLocal finds outlier pixels in independent frames, sets confidence
cLocal = 2

# Check if the median for each residual frame is below the confidence 
betaBias = np.zeros(K)
resFrameMedian = np.median(np.median(np.abs(\
     residual.reshape((M1, M2, K))), axis=0), axis=0)

betaBias[resFrameMedian <= cbias] = 1

# find beta depending on the size of the residuals
betaLocal = np.ones(K * M)
betaLocalMask = np.abs(residual) > cLocal * sigma_noise
betaLocal[betaLocalMask] = cLocal * sigma_noise / \
                         np.abs(residual[betaLocalMask])

# finds the new beta by combining betaBias and betaLocal
beta_new = np.broadcast_to(betaBias, (M, K)).T.ravel() * betaLocal

# compute alpha_new
cPrior = 2
# sparsity parameter
p = 0.5
alpha_new = np.ones(Ns)
alpha_newMask = np.abs(QSx) > cPrior * sigma_prior
alpha_new[alpha_newMask] = p*(cPrior * sigma_prior)**(1-p) / \
                         np.abs(QSx[alpha_newMask])**(1-p)
# %% Illustrates beta for different frames and alpha for different gradients
fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
for i in range(3):
     j = i+6
     ax[0][i].imshow(beta_new[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
     ax[1][i].imshow(alpha_new[j*N:(j+1)*N].reshape(N1, N2), cmap="gray")
     # ax[2][i].imshow(residual[i*M:(i+1)*M].reshape(M1,M2), cmap="gray")
     ax[0][i].set_title('beta')
     ax[1][i].set_title('alpha')
     # ax[2][i].set_title('residual')

# Creates a three col image where the first is the temp med
# second is beta map of first frame and last is
#  all gradients of alpha added together

# motionEstimates[]
fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
ax[0].imshow(img_med, cmap="gray", vmin=0, vmax=255)
ax[0].set_title('(a)')
ax[1].imshow(beta_new[0:M].reshape([M1, M2]), cmap="gray")
ax[1].set_title('(b)')
ax[2].imshow(np.sum(alpha_new.reshape([(2*P+1)**2,N1,N2]), axis=0), 
          cmap = "gray")
ax[2].set_title('(c)')
fig.tight_layout()
if saveFig == True:
     fig.savefig(writeFolder+"mapsIRWSR"+filename+".png", dpi=600)
plt.show()


# convert alpha and beta to sparse diagonal arrays
A = diags(alpha_new)
B = diags(beta_new)
# # %% Cross Validaion to find lambda
# # fraction used for parameter training
# deltaCV = 0.95
# lamb = 1
# Tcv = 2


# # create Idelta as a random vector of ones and zeros converted to a 
# # diagonal matrix
# # denotes training data
# Idelta = diags(np.random.choice(2, K * M, p=[1-deltaCV, deltaCV]))
# # denotes validation data
# Iflip = identity(K*M) - Idelta
# # find x(lambda)
# # Error term
# from funcs.HRTrain import findHRTrain
# def crossError(x, y, W, S, A, B, lamb, Idelta, Iflip):
#      tau = 1e-4
#      """ Should implement training data in findHRimage e.g. Idelta """
#      # Normalize first ? 
#      x = normalize(x) * 255
#      # Finds the given HR image for a lambda
#      SRimg = findHRTrain(x, y, W, S, A, B, lamb, tau, Idelta)
#      # Taking training part of SRimg
#      #
#      residual = y - W.dot(SRimg)
#      L = residual.T.dot(Iflip.dot(B).dot(residual))
#      print(L)
#      return  L, SRimg
     
# def optimalLamb(lambRange, x_vec, y_vec, W, S, A, B, Idelta, Iflip):
#      # maybe don't save SRframes
#      SRframes = np.zeros([N, len(lambRange)])
#      errorArray = np.zeros(len(lambRange))
#      for i in range(len(lambRange)):
#           lamb_temp = 10**(lambRange[i])
#           errorArray[i], SRframes[:,i] = crossError(x_vec,\
#                y_vec, W, S, A, B, lamb_temp, Idelta, Iflip)
#      lamb = 10**(lambRange[np.argmin(errorArray)])
#      return lamb

# # grid search values
# lambL = 1e-12; lambU = 0
# gridRange = 10
# # log10?
# # lambRange = np.arange(np.log10(lambL), np.log10(lambU))
# lambRange = np.linspace(-12, 1, 13)
# lamb = optimalLamb(lambRange, x_vec, y_vec, W, S, A, B, Idelta, Iflip)
# for tCV in range(Tcv):
#      lambRange = np.linspace(np.log10(lamb)-1/t, np.log10(lamb)+1/t, gridRange)
#      lamb = optimalLamb(lambRange, x_vec, y_vec, W, S, A, B, Idelta, Iflip)
# Tcv *= 0.5

#%% find SR image with SCG
lamb2 = 1e-5
tau = 1e-4
Tscg = 10
# define termination tolerance
eta = 1
def termCriterion(x_new, x_old, eta):
     max_pixel_change = np.max(np.abs(x_old - x_new))
     return max_pixel_change < eta


# x_vec = normalize(x_vec)
x_new = x_vec
x_old = x_vec
tSCG = 1

opts = {'maxiter': 500,
        'disp': True,  # default value
        'gtol': 1e-5,  # default value
        'norm': np.inf,  # default value
        'eps': 1.4901161193847656e-08}  # default value
""" finds the HR image by minimizing f through a conjugate gradient """
from scipy import optimize
from funcs.f import f
from funcs.gradf import gradf

# HRimg = optimize.minimize(f, x0,
#                          args=(y, W, S, A, B, lamb, tau),
#                          method="Nelder-Mead")
# Maybe not normalize ?!
x_new = normalize(x_new) * 255
HRimg = optimize.minimize(f, x_new, jac=gradf,
                              args=(y_vec, W, S, A, B, lamb2, tau),
                              method="CG", options = opts)
x_new = HRimg.x
# x_new = findHRimage(x_new, y_vec, W, S, A, B, lamb, tau)
plt.figure()
plt.imshow(x_new.reshape(N1, N2), cmap = 'gray')
plt.show
# Adds the new image to the old one if it was a residual
plt.figure()
plt.imshow((normalize(x_old) * 255+x_new).reshape(N1, N2), cmap='gray')
plt.show
# %% Converts vectors to individual frames get overview of residuals.
fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 12))
for i in range(3):
     ax[0][i].imshow(y_vec[i*M:(i+1)*M].reshape(M1,M2), cmap="gray")
     ax[1][i].imshow(W.dot(x_vec)[i*M:(i+1)*M].reshape(M1,M2), cmap="gray")
     ax[2][i].imshow(residual[i*M:(i+1)*M].reshape(M1,M2), cmap="gray")
     ax[0][i].set_title('org LR frame')
     ax[1][i].set_title('Wx')
     ax[2][i].set_title('residual')

# %%
