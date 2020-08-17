""" The gradient of the energy function """
import numpy as np

def gradf(x, y, W, S, A, B, lamb, tau):
     # y, W, B, lamb, A, S = args
     res = y - W.dot(x)
     noise_term = -2 * (W.T.dot(B.dot(res)))
     z = A.dot(S.dot(x))
     char = z/np.sqrt(z**2+tau)
     # prior_term = lamb*A.dot(S.T)*char
     prior_term = lamb+S.T.dot(A)*char
     return noise_term + prior_term
