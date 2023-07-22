"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from scipy.sparse import csc_matrix



def convertDenseToCSC(A):
    """
        Convert a numpy dense matrix to a sparse matrix into scipy.csc format.
    """
    return csc_matrix(A)


def convertDenseToBlocksTriDiagStorage(A, blockSize):
    """
        Converte a numpy dense matrix to 3 dimensional numpy array of blocks. Handling
        the diagonal blocks, upper diagonal blocks and lower diagonal blocks separately.
    """
    nBlocks = int(np.ceil(A.shape[0]/blockSize))

    A_bloc_diag  = np.zeros((nBlocks, blockSize, blockSize), dtype=A.dtype)
    A_bloc_upper = np.zeros((nBlocks-1, blockSize, blockSize), dtype=A.dtype)
    A_bloc_lower = np.zeros((nBlocks-1, blockSize, blockSize), dtype=A.dtype)

    for i in range(nBlocks):
        A_bloc_diag[i, ] = A[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize]
        if i < nBlocks-1:
            A_bloc_upper[i, ] = A[i*blockSize:(i+1)*blockSize, (i+1)*blockSize:(i+2)*blockSize]
            A_bloc_lower[i, ] = A[(i+1)*blockSize:(i+2)*blockSize, i*blockSize:(i+1)*blockSize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower


def convertBlocksBandedToDense(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    """
        Convert a 3 dimensional numpy array of blocks to a numpy dense matrix.
    """
    nBlocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    A = np.zeros((nBlocks*blockSize, nBlocks*blockSize), dtype=A_bloc_diag.dtype)

    for i in range(nBlocks):
        A[i*blockSize:(i+1)*blockSize, i*blockSize:(i+1)*blockSize] = A_bloc_diag[i, ]
        if i < nBlocks-1:
            A[i*blockSize:(i+1)*blockSize, (i+1)*blockSize:(i+2)*blockSize] = A_bloc_upper[i, ]
            A[(i+1)*blockSize:(i+2)*blockSize, i*blockSize:(i+1)*blockSize] = A_bloc_lower[i, ]

    return A
