#%%
"""
Iterative Re-Weighted SR test.
Arcificially degrades a test image into multiple frames translated by
subpixeldistances and returns estimated SR
          
"""

from scipy import optimize
from scipy import signal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from scipy.sparse import csr_matrix, rand, diags
from scipy import stats
from time import time

from funcs.MFDegradation import MFDegradation
from funcs.Shift import Shift
from funcs.stackW import stackW
from funcs.weighted_median import weighted_median
from funcs.weightedMAD import weightedMAD
from funcs.createSStack import createSStack

from funcs.normalize import normalize
from funcs.findBeta import findBeta
from funcs.findAlpha import findAlpha
from funcs.calcWx import calcWx

from scipy.sparse import identity
from funcs.findHRimage import findHRimage
from funcs.f import f, gradf

def IRWSR_func(frames, motionEstimates, saveFig, displayImages):
     from defineParameters import  (delta_s, s, minIntensity, maxIntensity, \
                              sigma0_noise, psfWidth, cBias, cLocal, \
                              sigma0_prior, P, alphaBTV, cPrior, p, deltaCV, \
                              Tcv, opts, degradationType, optimizationType, \
                              crossValidation, lamb, lambL, lambU)
     

     M1, M2, N_frames = frames.shape
     print(f"Each frame has a size of {M1}x{M2}")
     print(f"Reading {N_frames} frames in total")
     # if saveFig == True:
     #      fig.savefig(writeFolder+"lowResFrames"+filename+".png",dpi=600)
     # Initial guess should be the temporal median of motion compensated 
     # low-resolution frames
     img_trans = Shift(frames, motionEstimates)
     img_med = np.median(img_trans, axis = 2)
     if displayImages == True:
          fig, ax = plt.subplots(ncols = 2, figsize = (12, 20))
          ax[0].imshow(img_med, cmap = "gray", vmin=0, vmax=255)
          ax[0].set_title("median")
          ax[1].imshow(img_trans[:,:,0], cmap = "gray", vmin=0, vmax=255)
          ax[1].set_title("first frame translated")
     
          for i in range(N_frames):
               print('Shows the frames and the translated')
               # shows all frames with corresponding translations
               fig, ax = plt.subplots(ncols=2)
               ax[0].imshow(frames[:, :, i], cmap="gray", vmin=0, vmax=255)
               ax[1].imshow(img_trans[:, :, i], cmap="gray", vmin=0, vmax=255)
               plt.show()
          fig, ax = plt.subplots(ncols=2)
          ax[0].hist(frames.ravel())
          ax[0].set_title("LR frames")
          ax[1].hist(img_med.ravel())
          ax[1].set_title("Median")
          plt.show()


     """ Initiate variables """
     M1, M2, K = frames.shape
     M = M1 * M2
     # parameters in the degradation which are constant in the process

     print(f"{K} low resolution frames of size {M1}x{M2} = {M}")


     # %% Prepares the loop
     # starts off with a scale factor of 1
     # if statement gurantees that one iteration will be made no matter desired output
     if s == 1:
          s_old = 1 - delta_s
     else:
          s_old = 1

     N1, N2 = int(round(M1 * (s_old + delta_s))), int(round(M2 * (s_old + delta_s)))
     N = N1 * N2
     # print(f"Magnified by {s} to SR image of size {N1}x{N2} = {N} ")
     # Initial observation weights
     alpha_old = np.ones(N)
     beta_old = np.ones(K * M)
     # Trying to round, maybe not essential
     x_old = np.round(img_med)
     # convert frames to a vector in lexicographical order
     y_vec = np.zeros(M*K)
     for idx in range(K):
          y_vec[idx*M:idx*M+M] = frames[:, :, idx].ravel()

     # Normalizes 8bit images to the range defined by the max intensity
     x_old = normalize(x_old, 0, 255) * maxIntensity
     y_vec = normalize(y_vec, 0, 255) * maxIntensity
     if displayImages:
          fig, ax = plt.subplots(ncols=2)
          ax[0].hist(y_vec)
          ax[0].set_title("LR frames")
          ax[1].hist(x_old.ravel())
          ax[1].set_title("Median")
          plt.show()
     #%%
     """ Content of the loop """
     t = 1
     totTime0 = time()
     while s_old < s:
     # for uhtaoesn in range(1):
          print(f"Starting {t} iteration")
          # stepwise scaling until s is acheived
          s_new = np.min([s_old + delta_s, s])
          # Find size of HR image with new scalng
          N1, N2 = int(round(M1 * s_new)), int(round(M2 * s_new))
          N = N1 * N2
          print(f"Magnified by {s_new} to SR image of size {N1}x{N2} = {N} ")
          
          Ns = (2*P + 1)**2 * N

          degradationParameters = {'LR_shape': (M1, M2),
                                   'HR_shape': (N1, N2),
                                   'N_frames': N_frames,
                                   'sigma': psfWidth,
                                   'translations': motionEstimates,
                                   'type': degradationType,
                                   'magFactor': s_new,
                                   }
          # Propagate x to new size and define as vector
          x_old = (cv2.resize(x_old,(N2, N1), cv2.INTER_CUBIC)).ravel()
          # x_vec = x_old.ravel()
          # Finds the matrix W which defines the relation between x and y_frames
          print("finds W")
          t0 = time()
          W = stackW(**degradationParameters)
          Wx = calcWx(HR_vec = x_old, W = W, **degradationParameters)
          # Calculates the residual between the given frames and Wx
          dt = time() - t0
          print(f'took {dt:02.2f} seconds\n')
          residual = y_vec - Wx
          if displayImages == True:
               for i in range(N_frames):
                    print("All the frames and Wx")
                    fig, ax = plt.subplots(ncols=2)
                    ax[0].imshow(frames[:, :, i], cmap="gray")
                    ax[1].imshow(Wx[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
                    plt.show()

               # %% Converts vectors to individual frames get overview of residuals.
               fig, ax = plt.subplots(ncols=3, nrows=3, figsize=(12, 12))
               for i in range(3):
                    print("Shows the residuals")
                    ax[0][i].imshow(y_vec[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
                    ax[1][i].imshow(Wx[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
                    ax[2][i].imshow(np.abs(residual[i*M:(i+1)*M].reshape(M1, M2)), cmap="gray")
                    ax[0][i].set_title('org LR frame')
                    ax[1][i].set_title('Wx')
                    ax[2][i].set_title('Absolute value of residual')
               plt.show()

               # %%
               fig, ax = plt.subplots(nrows= 2, ncols=2)
               ax[0][0].hist(y_vec)
               ax[0][0].set_title("LR frames")
               ax[0][1].hist(Wx)
               ax[0][1].set_title("Wx")
               ax[1][0].hist(x_old)
               ax[1][0].set_title("First Guess")
               ax[1][1].hist(residual)
               ax[1][1].set_title("residual")
               plt.show()
          
          """ Calculates S """
          print("finds Beta")
          t0 = time()
          # calculate sigma noise
          sigma_noise = sigma0_noise * weightedMAD(residual, beta_old)
          # Finds the new beta values
          beta_new = findBeta(residual, K, (M1, M2), sigma_noise, cBias, cLocal)
          dt = time() - t0
          print(f'took {dt:02.2f} seconds\n')
          # Finds the S matrix that defines the derivatives in the different directions
          print("Finds S")
          S = createSStack((N1,N2), P, alphaBTV)
          # Calculate Sx and convert format to fit with openCV
          Sx = (S.dot(x_old))
          # Sx = (normalize(Sx)*maxIntensity).astype('uint8')
          SxTemp = Sx.reshape(N1*(P*2+1)**2, N2)
          QSx = signal.medfilt2d(SxTemp, 3).ravel()
          dt = time() - t0
          print(f'took {dt:02.2f} seconds\n')
          # calculate sigma prior
          sigma_prior = sigma0_prior * weightedMAD(QSx, alpha_old)

          # compute alpha_new
          alpha_new = findAlpha(QSx, Ns, sigma_prior, p, cPrior)
          """ 
          Illustrates beta for different frames and alpha for different gradients
          """
          if displayImages == True:
               fig, ax = plt.subplots(ncols=3, nrows=2, figsize=(12, 8))
               for i in range(3):
                    j = i+6
                    ax[0][i].imshow(beta_new[i*M:(i+1)*M].reshape(M1, M2), cmap="gray")
                    ax[1][i].imshow(alpha_new[j*N:(j+1)*N].reshape(N1, N2), cmap="gray")
                    # ax[2][i].imshow(residual[i*M:(i+1)*M].reshape(M1,M2), cmap="gray")
                    ax[0][i].set_title('beta')
                    ax[1][i].set_title('alpha')
               plt.show()
                    # ax[2][i].set_title('residual')

          # Creates a three col image where the first is the temp med
          # second is beta map of first frame and last is
          #  all gradients of alpha added together
          beta_img = beta_new[0:M].reshape([M1, M2])
          alpha_img = np.sum(alpha_new.reshape([(2*P+1)**2, N1, N2]), axis=0)
          if displayImages == True:
               fig, ax = plt.subplots(nrows=1, ncols=3, figsize=(30, 10))
               ax[0].imshow(img_med, cmap="gray", vmin=0, vmax=255)
               ax[0].set_title('(a)')
               ax[1].imshow(beta_new[0:M].reshape([M1, M2]), cmap="gray")
               ax[1].set_title('(b)')
               ax[2].imshow(np.sum(alpha_new.reshape([(2*P+1)**2,N1,N2]), axis=0), 
                         cmap = "gray")
               ax[2].set_title('(c)')
               fig.tight_layout()
               # if saveFig == True:
               #      fig.savefig(writeFolder+"mapsIRWSR"+filename+".png", dpi=600)
               # plt.show()


          # convert alpha and beta to sparse diagonal arrays
          A = diags(alpha_new)
          B = diags(beta_new)
          #%% Cross Validaion to find lambda
          # defines the parameters going into the error funcction
          tau = 1e-4

          # create Idelta as a random vector of ones and zeros converted to a 
          print(f"Doing the cross-validation: {crossValidation}")
          # denotes training data
          Idelta = diags(np.random.choice(2, K * M, p=[1-deltaCV, deltaCV]))
          # denotes validation data
          Iflip = identity(K*M) - Idelta

          functionParameters = {'y': y_vec,
                              'W': W,
                              'S': S,
                              'A': A,
                              'B': B,
                              'tau': tau,
                              'Idelta': Idelta
                              }
          if crossValidation == True:
               from funcs.crossValidation import optimalLamb

               # grid search values
               if t > 1:
                   lambL = np.log10(lamb)-1/t
                   lambU = np.log10(lamb)+1/t

               lambRange = np.linspace(lambL, lambU, Tcv)
               
               if Tcv > 1:
                    lamb = optimalLamb(lambRange=lambRange, x_vec=x_old, Iflip=Iflip,
                                   degradationParameters=degradationParameters,
                                   functionParameters = functionParameters,
                                   optimizationType = optimizationType)
               # for tCV in range(Tcv):
               #      lambRange = np.linspace(np.log10(lamb)-1/t, np.log10(lamb)+1/t, 
               #                               gridRange)
               #      lamb = optimalLamb(lambRange = lambRange, x_vec = x_old, Iflip = Iflip,
               #                          degradationParameters = degradationParameters,
               #                          functionParameters = functionParameters,
               #                          optimizationType = optimizationType)
               Tcv = np.max((int(0.5*Tcv), 1))
               print(f"In the CV the optimal lambda is {lamb}")

          """ find SRimage with SCG """

          Tscg = 10
          # define termination tolerance
          eta = 1
          def termCriterion(x_new, x_old, eta):
               max_pixel_change = np.max(np.abs(x_old - x_new))
               return max_pixel_change < eta

          tSCG = 1
          """ Help from here ? https://stackoverflow.com/questions/24767191/scipy-is-not-optimizing-and-returns-desired-error-not-necessarily-achieved-due """
          print("Finds optimal image")
          t0 = time()
          f_val = f(x_old, lamb, y_vec, W, S, A, B, tau, Idelta, degradationParameters, False)
          grad_f_val = gradf(x_old, lamb, y_vec, W, S, A, B, tau, Idelta, degradationParameters, False)
          print(f"function val is {f_val}")
          print(f"gradient val is {grad_f_val}")
          # x_new = findHRimage(x0=x_old.reshape(N1, N2), lamb=lamb, opts=opts,
          x_new = findHRimage(x0=x_old, lamb=lamb, opts=opts,
                              degradationParameters = degradationParameters,
                              functionParameters = functionParameters,
                              optimizationType = optimizationType,
                              crossVal = False)
          # x_new_alt = findHRimage(x_old, y_vec, W, S, A, B, lamb_alt, tau, opts)
          dt = time() - t0
          print(f'took {dt:02.2f} seconds\n')

          # SR = (normalize(x_new)*255).reshape(N1, N2)
          SR = (normalize(x_new, 0, 1)*255).reshape(N1, N2)
          if displayImages == True:
               fig, ax = plt.subplots(ncols=2)
               ax[0].imshow(SR, cmap='gray', vmin=0, vmax=255)
               ax[0].set_title(f"SR")
               ax[1].imshow(img_med, cmap='gray', vmin=0, vmax=255)
               ax[1].set_title(f"median")
               plt.show()

          t +=1

          s_old = s_new
          x_old = x_new.reshape(N1,N2)
     totTime = time() - totTime0
     return SR, img_med, alpha_img, beta_img, lamb, totTime
# %%

