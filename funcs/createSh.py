""" horisontal shift as a sparse block matrix
     inputs: 
     img_size: N1xN2 (N1: number of blocks, N2; size of one block)
     n: pixel translation (positiv right, negativ left)
     output:
     sparse block matrix of size NxN (N = N1*N2) 
"""
from scipy.sparse import eye, block_diag

def createSh(img_size, n):
    N1, N2 = img_size
    # creates one block
    # block = csr_matrix(np.eye(N2, N2, -n))
    block = eye(N2, N2, -n)
    # string with number of blocks in diagonal
    num_blocks = "block, " * N1
    # creates whole block matrix
    """ Check up on sparse formats """
    blockMatrix = block_diag((eval(num_blocks)), format = "csr")
    # bmat = block_diag((eval(num_blocks)))
    return blockMatrix

# N1 = 3; N2 = 4; n = 2
# test = createSh((N1,N2), n)
# # print(bmat.todense())