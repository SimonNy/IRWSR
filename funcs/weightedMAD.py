import numpy as np
from funcs.weighted_median import weighted_median

def weightedMAD(data, weights):
     """ calculates the Weighted median abslute deviation """
     tempMedian = weighted_median(data, weights)
     return weighted_median(np.abs(data - tempMedian), weights)