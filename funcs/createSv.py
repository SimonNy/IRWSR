# %%
""" vertical shift as a sparse block matrix
     inputs: 
     img_size: N1xN2 (N1: number of blocks, N2; size of one block)
     m: pixel translation (positiv down, negativ up)
     output:
     sparse block matrix of size NxN (N = N1*N2) 
"""
from scipy.sparse import eye
def createSv(img_size, m):
    N1, N2 = img_size

     # # defines one block
     # block = eye(N2, N2)
     # # defines an empty fill block
     # fill = np.zeros([N2, N2])
     # # matrix where ones are the blocks and zeros are empty
     # blockLayout = np.eye(N1,N1, -m)
     # # creates a string for the block layout, replaces with variable names
     # blockStr = str(blockLayout)
     # blockStr = blockStr.replace('0.', 'fill,')
     # blockStr = blockStr.replace('1.', 'block,')
     # blockStr = blockStr.replace(',]', '],')
     # blockStr = blockStr.replace('],]',']]')
     # blockStr = blockStr.replace('\n', '')
     # # creates block matrix
     # blockMatrix = bmat(eval(blockStr), format = "csr")

    """ is this equivalent to the above? """
    N = N1 * N2
    blockMatrix = eye(N, N, -m*N2)
    return blockMatrix

# N1 = 5
# N2 = 3
# m = 2
# test = createSv((N1, N2), m)
# print(test.todense())