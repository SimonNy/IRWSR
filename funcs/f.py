import numpy as np
from funcs.calcWx import calcWx, calcWtransposed
import matplotlib.pyplot as plt

def f(x, lamb, y, W, S, A, B, tau, Idelta, degradationParameters, crossVal):
    """ energy function for the HR image x """
    Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
    res = y - Wx
    Bres = B.dot(res)
    # plt.figure()
    # plt.hist(Bres)
    # plt.show()
    if crossVal == True:
        Bres = Idelta.dot(Bres)
    noise_term = res.T.dot(Bres)
    # noise_term = np.log(noise_term)
    # print(f"noise: {noise_term}")
    # convex Charbonnier
    char = np.sum(A.dot(np.sqrt(S.dot(x)**2+tau)))
    prior_term = lamb * char
    # noise_term = np.log(prior_term)
    # print(f"prior: {prior_term}")
    return (noise_term + prior_term)/len(x)


def gradf(x, lamb, y, W, S, A, B, tau, Idelta, degradationParameters, crossVal):
    """ The gradient of the energy function """
    Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
    res = y - Wx
    Bres = B.dot(res)
    if crossVal == True:
        Bres = Idelta.dot(Bres)
    noise_term = -2 * calcWtransposed(LR_vec=Bres, W=W, **degradationParameters)

    z = A.dot(S.dot(x))
    char = z/np.sqrt(z**2+tau)
    # prior_term = lamb*A.dot(S.T)*char
    prior_term = lamb*S.T.dot(A)*char
    return (noise_term + prior_term)/len(x)
    # return (noise_term + prior_term)


# def f(x, y, W, S, A, B, lamb, tau, degradationParameters):
#     """ energy function for the HR image x """
#     # y, W, B, lamb, A, S = args
#     Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
#     res = y - Wx
#     noise_term = res.T.dot(B.dot(res))
#     # convex Charbonnier
#     char = np.sum(A.dot(np.sqrt(S.dot(x)**2+tau)))
#     prior_term = lamb * char
#     return noise_term + prior_term


# def gradf(x, y, W, S, A, B, lamb, tau, degradationParameters):
#     """ The gradient of the energy function """
#     # y, W, B, lamb, A, S = args
#     Wx = calcWx(HR_vec=x, W=W, **degradationParameters)
#     res = y - Wx
# #     noise_term = -2 * (W.T.dot(B.dot(res)))
#     noise_term = -2 * \
#         calcWtransposed(LR_vec=B.dot(res), W=W, **degradationParameters)
#     z = A.dot(S.dot(x))
#     char = z/np.sqrt(z**2+tau)
#     # prior_term = lamb*A.dot(S.T)*char
#     prior_term = lamb*S.T.dot(A)*char
#     return noise_term + prior_term
