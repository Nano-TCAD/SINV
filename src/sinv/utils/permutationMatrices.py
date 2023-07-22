"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: 

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from scipy.sparse import csc_matrix
from mpi4py       import MPI



def generateSchurPermutationMatrix(matSize):
    """
        Generate a permutation matrix for the Schur decomposition procedure of the hpr method.
    """

    P = np.zeros((matSize, matSize), dtype=np.int32)

    offsetRight = matSize-1
    offsetLeft  = 0
    bascule     = True

    for i in range(matSize-1, -1, -1):
        if bascule:
            P[i, offsetRight] = 1
            offsetRight -= 1
            bascule = False
        else:
            P[i, offsetLeft] = 1
            offsetLeft  += 1
            bascule = True

    return P


def generateSchurBlockPermutationMatrix(nBlocks, blockSize):
    """
        Generate a block permutation matrix for the Schur bloc decomposition procedure of the hpr method.
    """

    P = np.zeros((nBlocks*blockSize, nBlocks*blockSize), dtype=np.int32)
    I = np.eye(blockSize, dtype=np.int32)

    offsetRight = nBlocks-1
    offsetLeft  = 0
    bascule     = True

    for i in range(nBlocks-1, -1, -1):
        if bascule:
            P[i*blockSize:(i+1)*blockSize, offsetRight*blockSize:(offsetRight+1)*blockSize] = I
            offsetRight -= 1
            bascule = False
        else:
            P[i*blockSize:(i+1)*blockSize, offsetLeft*blockSize:(offsetLeft+1)*blockSize] = I
            offsetLeft  += 1
            bascule = True

    return P



def generateCyclicReductionPermutationMatrix(matSize):
    """
        Generate a permutation matrix suited for the Cyclic Reduction algorithm
    """

    P = np.zeros((matSize, matSize), dtype=np.int32)

    for i in range(matSize):
        if i%2 == 0:
            P[i//2, i] = 1
        else:
            P[matSize//2 + i//2, i] = 1

    return P


def generateCyclicReductionBlockPermutationMatrix(nBlocks, blockSize):
    """
        Generate a block permutation matrix suited for the block Cyclic Reduction algorithm
    """

    P = np.zeros((nBlocks*blockSize, nBlocks*blockSize), dtype=np.int32)
    I = np.eye(blockSize, dtype=np.int32)

    for i in range(nBlocks):
        if i%2 == 0:
            P[(i//2)*blockSize:(i//2+1)*blockSize, i*blockSize:(i+1)*blockSize] = I
        else:
            P[(nBlocks//2 + i//2)*blockSize:(nBlocks//2 + i//2 + 1)*blockSize, i*blockSize:(i+1)*blockSize] = I

    return P
