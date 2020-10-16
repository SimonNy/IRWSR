from scipy import optimize

import numpy as np
from funcs.calcWx import calcWx, calcWtransposed

def fTrain(x, y, W, S, A, B, lamb, tau, Idelta, degradationParameters):
     """ energy function for the HR image x """

     Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
     res = y - Wx
     noise_term = res.T.dot(Idelta.dot(B.dot(res)))
     # convex Charbonnier
     z = S.dot(x)
     char = np.sum(A.dot(np.sqrt(z**2+tau)))
     prior_term = lamb * char
     return noise_term + prior_term


def gradfTrain(x, y, W, S, A, B, lamb, tau, Idelta, degradationParameters):
     Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
     res = y - Wx

     noise_term = -2 * calcWtransposed(LR_vec=Idelta.dot(B.dot(res)), W=W, 
                                        **degradationParameters)
     z = A.dot(S.dot(x))
     char = z/np.sqrt(z**2+tau)
     prior_term = lamb*S.T.dot(A)*char
     return noise_term + prior_term




""" finds the HR image by minimizing f through a conjugate gradient """


def findHRTrain(x0, y, W, S, A, B, lamb, tau, Idelta, degradationParameters, opts):
     HRimg = optimize.minimize(fTrain, x0, jac=gradfTrain,
                               args=(y, W, S, A, B, lamb, tau,
                                     Idelta, degradationParameters),
                               method="CG", options=opts)
                               
     return HRimg.x
     # return optimize.fmin_cg(f, x0, fprime=gradf, \
         #  )
