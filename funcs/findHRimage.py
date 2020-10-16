from scipy import optimize
from funcs.f import f, gradf
from funcs.calcWx import calcWx



""" finds the HR image by minimizing f through a conjugate gradient """
def findHRimage(x0, lamb, functionParameters,  degradationParameters, opts,
                crossVal, optimizationType = 0):

     y = functionParameters['y']
     W = functionParameters['W']
     S = functionParameters['S']
     A = functionParameters['A']
     B = functionParameters['B']
     tau = functionParameters['tau']
     Idelta = functionParameters['Idelta']
     # function arguments
     # (x, lamb, y, W, S, A, B, tau, Idelta, degradationParameters, crossVal):
     if optimizationType == 0:
          HRimg = optimize.minimize(f, x0, jac=gradf, 
                                   args=(lamb, y, W, S, A, B, tau, Idelta, 
                                        degradationParameters, crossVal),
                                   method="CG", options = opts)
     elif optimizationType == 1:
          HRimg = optimize.minimize(f, x0,
                                   args=(lamb, y, W, S, A, B, tau, Idelta, 
                                        degradationParameters, crossVal),
                                   # method="Nelder-Mead", options = opts)  
                                   method="BFGS", options = opts)  
     return HRimg.x

