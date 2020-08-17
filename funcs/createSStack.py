import numpy as np
from scipy.sparse import vstack

from funcs.createSmn import createSmn

def createSStack(img_size, P, alphaBTV):
    """ Creates S a linear sparsifying transform 
    S = (S^(-P,-P), S^(-P+1,-P), ... , S^(P,P)).T
    inputs:
    img_size: Shape of image N1xN2(N=N1*N2)
    P: range of the shifts
    alphaBTV: between 0 and 1, degree of derivative?
    """
    N1, N2 = img_size
    # meshgrid of all posible m, n values
    pRange = np.arange(P*2+1)-P
    pGrid = np.meshgrid(pRange, pRange)
    # create string with all the differnt Smn matrices
    SmnStr = "["
    for i in range((P*2+1)**2):
        SmnStr += f"createSmn((N1, N2), {pGrid[0].ravel()[i]},"\
            +f"{pGrid[1].ravel()[i]}, alphaBTV), "
    SmnStr += "]"
    # Creates s as a stack of sparse matrices
    S = vstack(eval(SmnStr))
    return S