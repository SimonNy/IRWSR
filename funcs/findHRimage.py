from scipy import optimize
from funcs.f import f
from funcs.gradf import gradf
import numpy as np

opts = {'maxiter' : 10,
        'disp': True,  # default value 
        'gtol': 1e-5,  # default value
        'norm': np.inf,  # default value
        'eps' : 1.4901161193847656e-08} # default value

""" finds the HR image by minimizing f through a conjugate gradient """
def findHRimage(x0, y, W, S, A, B, lamb, tau):
     HRimg = optimize.minimize(f, x0, 
                               args=(y, W, S, A, B, lamb, tau),
                               method="Nelder-Mead")
#      HRimg = optimize.minimize(f, x0, jac=gradf, 
#                                args=(y, W, S, A, B, lamb, tau),
#                                method="CG", options = opts)
     return HRimg.x
     # return optimize.fmin_cg(f, x0, fprime=gradf, \
          #  )
