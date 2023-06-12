"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from scipy.sparse import csc_matrix
from mpi4py       import MPI



def generatePermutationMatrix(matSize):

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


def generateBlockPermutationMatrix(nBlocks, blockSize):

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

