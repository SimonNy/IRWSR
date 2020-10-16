import numpy as np

def findBeta(residual, frames, img_shape, sigma_noise, cBias=0.02, cLocal=2):
    """  Finds the observation weights for the given residuals

        Inputs:
        residual: The residual between LR frames and Wx in vector format
        frames:   The amount of LR frames
        img_shape:The shape of one LR frame
        sigma_noise:The current estimated noise level
        cBias:    The confidence for the beta bias i.e. the maximum allowed
                    median for the residual value given in the frame
        cLocal:   The cofidence for the betaLocal i.e the maximum allowed 
                    mediaf for the residual at a pixel level
    """ 
    M1, M2 = img_shape
    M = M1*M2
    K = frames
    # Check if the median for each residual frame is below the confidence 
    betaBias = np.zeros(K)
    resFrameMedian = np.median(np.median(np.abs(\
        residual.reshape((M1, M2, K))), axis=0), axis=0)
    # betaBias finds outlier frames, sets confidence of max intensity
    betaBias[resFrameMedian <= cBias] = 1

    # betaLocal finds outlier pixels in individual frames, sets confidence
    # find beta depending on the size of the residuals
    betaLocal = np.ones(K * M)
    betaLocalMask = np.abs(residual) > cLocal * sigma_noise
    betaLocal[betaLocalMask] = cLocal * sigma_noise / \
                            np.abs(residual[betaLocalMask])

    # finds the new beta by combining betaBias and betaLocal
    beta_new = np.broadcast_to(betaBias, (M, K)).T.ravel() * betaLocal
    return beta_new