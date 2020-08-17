import numpy as np
""" Normalizes data to the range of 0 to 1 """
""" maxVal and minVal defines range f """

def normalize(data, minVal=[], maxVal=[]):
    if not maxVal and not minVal:
        data_normalized = (data-np.min(data))/\
                    (np.max(data) - np.min(data))

    else:
        data_normalized = (data-minVal)/\
                    (maxVal - minVal)
    return data_normalized
