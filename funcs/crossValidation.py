import numpy as np
from funcs.findHRimage import findHRimage
from defineParameters import opts
from funcs.calcWx import calcWx

def crossError(x_vec,lamb, Iflip, degradationParameters, functionParameters,
                optimizationType):
    """ Should implement training data in findHRimage e.g. Idelta """
    # Normalize first ? 
    # x = normalize(x, minIntensity, maxIntensity) 
    # Finds the given HR image for a lambda
    N1, N2 = degradationParameters['HR_shape']
    SRimg = findHRimage(x0= x_vec.reshape(N1, N2), lamb = lamb, 
                        opts = opts, degradationParameters= degradationParameters,
                        functionParameters = functionParameters,
                        crossVal = True,
                        optimizationType =optimizationType)
    # Taking training part of SRimg
    Wx = calcWx(HR_vec=SRimg, W=functionParameters['W'], **degradationParameters)
    residual = functionParameters['y'] - Wx
    L = residual.T.dot(Iflip.dot(functionParameters['B']).dot(residual))
    print(f"The L value is {L} for lambda: {lamb}")
    return  L
    
def optimalLamb(lambRange, x_vec, Iflip, degradationParameters, 
                functionParameters, optimizationType):
    
    errorArray = np.zeros(len(lambRange))
    for i in range(len(lambRange)):
        lamb_temp = 10**(lambRange[i])
        errorArray[i] = crossError(x_vec, lamb_temp, Iflip,
                                    degradationParameters,
                                    functionParameters,
                                    optimizationType)
    lamb = 10**(lambRange[np.argmin(errorArray)])
    return lamb
