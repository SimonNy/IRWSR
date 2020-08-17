from scipy import optimize
from funcs.gradf import gradf
from funcs.f import f
import numpy as np

def fTrain(x, y, W, S, A, B, lamb, tau, Idelta):
     """ energy function for the HR image x """
     # y, W, B, lamb, A, S = args
     res = y - W.dot(x)
     noise_term = res.T.dot(Idelta.dot(B.dot(res)))
     # convex Charbonnier
     z = A.dot(S.dot(x))
     char = np.sum(np.sqrt(z**2+tau))
     prior_term = lamb * char
     return noise_term + prior_term


def gradfTrain(x, y, W, S, A, B, lamb, tau, Idelta):
     # y, W, B, lamb, A, S = args
     res = y - W.dot(x)
     noise_term = -2 * (W.T.dot(Idelta.dot(B.dot(res))))
     z = A.dot(S.dot(x))
     char = z/np.sqrt(z**2+tau)
     # prior_term = lamb*A.dot(S.T)*char
     prior_term = lamb+S.T.dot(A)*char
     return noise_term + prior_term


opts = {'maxiter': 500,
        'disp': True,  # default value
        'gtol': 1e-5,  # default value
        'norm': np.inf,  # default value
        'eps': 1.4901161193847656e-08}  # default value

""" finds the HR image by minimizing f through a conjugate gradient """


def findHRTrain(x0, y, W, S, A, B, lamb, tau, Idelta):
     HRimg = optimize.minimize(fTrain, x0, jac=gradfTrain,
                               args=(y, W, S, A, B, lamb, tau, Idelta),
                               method="CG", options=opts)
     return HRimg.x
     # return optimize.fmin_cg(f, x0, fprime=gradf, \
         #  )
