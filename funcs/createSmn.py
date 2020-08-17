import numpy as np
from scipy.sparse import identity

from funcs.createSv import createSv
from funcs.createSh import createSh

def createSmn(img_size, m, n, alphaBTV):
     """ creates Smn sparse matrix of size NxN
          Smn = alphaBTV^(|m|+|n|)*(I(NxN) - S^m_v S^n_h)
     inputs:
     img_size: Shape of image N1xN2(N=N1*N2)
     m: vertical movement
     n: horizontal movement
     alphaBTV: between 0 and 1, degree of derivative?
     """
     N1, N2 = img_size
     alpha = alphaBTV**(np.abs(m)+np.abs(n))
     S = createSv((N1,N2), m).dot(createSh((N1,N2), n))
     return alpha*(identity(N1*N2)-S)