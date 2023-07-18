"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""


import utils.vizualisation       as vizu

import numpy as np
import math
import time

from mpi4py import MPI




def extract_factors(A, blocksize):
    """
        Partition the matrix A into K_i submatrices. B_i stores the connecting blocks 
        between the K_i submatrices.
    """

    nblocks = A.shape[0] // blocksize
    nblock_per_partition = nblocks // 2

    K = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    u = np.zeros((nblocks*blocksize, 2*blocksize), dtype=A.dtype)
    v = np.zeros((nblocks*blocksize, 2*blocksize), dtype=A.dtype)

    K[0:nblock_per_partition*blocksize, 0:nblock_per_partition*blocksize]                                 = A[0:nblock_per_partition*blocksize, 0:nblock_per_partition*blocksize]
    K[nblock_per_partition*blocksize:nblocks*blocksize, nblock_per_partition*blocksize:nblocks*blocksize] = A[nblock_per_partition*blocksize:nblocks*blocksize, nblock_per_partition*blocksize:nblocks*blocksize]

    u[(nblock_per_partition-1)*blocksize:nblock_per_partition*blocksize, 0:blocksize]           = A[(nblock_per_partition-1)*blocksize:nblock_per_partition*blocksize, nblock_per_partition*blocksize:(nblock_per_partition+1)*blocksize]
    u[nblock_per_partition*blocksize:(nblock_per_partition+1)*blocksize, blocksize:2*blocksize] = A[nblock_per_partition*blocksize:(nblock_per_partition+1)*blocksize, (nblock_per_partition-1)*blocksize:nblock_per_partition*blocksize]

    v[(nblock_per_partition-1)*blocksize:nblock_per_partition*blocksize, blocksize:2*blocksize] = np.identity(blocksize)
    v[nblock_per_partition*blocksize:(nblock_per_partition+1)*blocksize, 0:blocksize]           = np.identity(blocksize)

    return K, u, v





def smw(A, blocksize):

    nblocks = A.shape[0] // blocksize
    G = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    K, u, v = extract_factors(A, blocksize)

    vizu.vizualiseDenseMatrixFlat(K, "K")
    vizu.vizualiseDenseMatrixFlat(u, "u")
    vizu.vizualiseDenseMatrixFlat(v.T, "v.T")

    K_inv = np.linalg.inv(K)

    #vizu.vizualiseDenseMatrixFlat(K_inv, "K_inv")

    G = K_inv + K_inv @ u @ np.linalg.inv(np.identity(2*blocksize) + v.T @ K_inv @ u) @ v.T @ K_inv

    vizu.vizualiseDenseMatrixFlat(G, "G")


    return G