import numpy as np

def findAlpha(QSx, image_vec_size, sigma_prior, p=0.5, cPrior = 2):
    """ Calculates the image prior weights
        inputs
        QSx: The image gradients 
        image_vec_size: The image size of the desired scale in vector format
        p: the sparsity parameter should be [0,1]
        cPrior: Confidence variable discriminating between flat regions
                and discontinuities i.e. the importance of the gradients
        sigma_prior: The current estimated scale parameter in teh prior
    """
    alpha_new = np.ones(image_vec_size)
    alpha_newMask = np.abs(QSx) > cPrior * sigma_prior
    alpha_new[alpha_newMask] = p*(cPrior * sigma_prior)**(1-p) / \
                               np.abs(QSx[alpha_newMask])**(1-p)
    return alpha_new