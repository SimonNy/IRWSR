import numpy as np

def f(x, y, W, S, A, B, lamb, tau):
     """ energy function for the HR image x """
     # y, W, B, lamb, A, S = args
     res = y - W.dot(x)
     noise_term = res.T.dot(B.dot(res))
     # convex Charbonnier
     char = np.sum(np.sqrt(S.dot(x)**2+tau))
     prior_term = lamb * char
     return noise_term + prior_term
