from scipy.sparse import vstack
from funcs.composeSystemMatrix import composeSystemMatrix
def stackW(LRsize, K, s, psfWidth, motionEstimates):
     """ Finds W for every frame and stacks in one sparse matrix """
     M1, M2 = LRsize
     W = []
     # Iterates over every frame
     for idx in range(K):
          # Computes W for the given framess motioin estimate
          W.append(composeSystemMatrix((M1, M2),\
          s, psfWidth, motionEstimates[:, idx]))
     W = vstack(W)
     return W