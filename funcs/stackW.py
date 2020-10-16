from scipy.sparse import vstack
from funcs.composeSystemMatrix import composeSystemMatrix
def stackW(LR_shape, N_frames, magFactor, sigma, translations, HR_shape, type):
     """ Finds W for every frame and stacks in one sparse matrix """
     if type == 1:
          M1, M2 = LR_shape
          W = []
          # Iterates over every frame
          for idx in range(N_frames):
               # Computes W for the given framess motioin estimate
               W.append(composeSystemMatrix((M1, M2),\
               magFactor, sigma, translations[:, idx]))
          W = vstack(W)
     else:
          W = 0
     return W