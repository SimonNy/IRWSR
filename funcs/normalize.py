import numpy as np
""" Normalizes data to the range of 0 to 1 """
""" maxVal and minVal defines range f """

def normalize(data, minVal=[], maxVal=[]):
    multiD = False
    if len(data.shape) > 1:
        multiD = True
        shape = data.shape
        data = data.ravel()
    if not maxVal and not minVal:
        data_normalized = (data-np.min(data))/\
                    (np.max(data) - np.min(data))

    else:
        data_normalized = (data-minVal)/\
                    (maxVal - minVal)
    
    if multiD:
        data_normalized = data_normalized.reshape(shape)

    return data_normalized
