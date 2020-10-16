import numpy as np

def cropFrames(
    frames,
    rows,
    cols,
):
    """ Crops all frames in array in the same way 
        axis0 is rows
        axis1 is cols
        axis2 is frames
    """
    regOfInt = (slice(rows[0], rows[1]),
                slice(cols[0], cols[1]),
                slice(None))

    return frames[regOfInt]
