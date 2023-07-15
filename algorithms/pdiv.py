"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

PDIV (Parallel Divide & Conquer) algorithm:
@reference: https://doi.org/10.1063/1.2748621
@reference: https://doi.org/10.1063/1.3624612

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""


import utils.vizualisation       as vizu
import utils.transformMatrices   as transMat

import numpy as np
import math
import time

from mpi4py import MPI




def create_partitions(A, blocksize):
    """
        Create the partitions K and the connections vectors X and Y from a dense matrix A.
    """

    n_partitions = 2

    nblocks = A.shape[0] // blocksize
    nblock_per_partition = nblocks // n_partitions

    print("Creating ", n_partitions, " partitions of ", nblock_per_partition, " blocks eachs.")

    K = np.zeros((n_partitions, nblock_per_partition*blocksize, nblock_per_partition*blocksize), dtype=A.dtype)
    X = np.zeros((n_partitions-1, 2*blocksize, 2*blocksize), dtype=A.dtype)
    Y = np.zeros((n_partitions-1, 2*blocksize, 2*blocksize), dtype=A.dtype)

    for i in range(n_partitions):
        starting_block_of_partition = i * nblock_per_partition
        ending_block_of_partition   = (i+1) * nblock_per_partition

        starting_row_of_partition = starting_block_of_partition * blocksize
        ending_row_of_partition   = ending_block_of_partition * blocksize

        print("Partition: ", i, " starting_block_of_partition: ", starting_block_of_partition, " ending_block_of_partition: ", ending_block_of_partition)

        K[i] = A[starting_row_of_partition:ending_row_of_partition, starting_row_of_partition:ending_row_of_partition]

        if i < n_partitions-1:
            X[i, 0:blocksize, blocksize:2*blocksize] = -A[ending_row_of_partition-blocksize:ending_row_of_partition, ending_row_of_partition:ending_row_of_partition+blocksize].T
            X[i, blocksize:2*blocksize, 0:blocksize] = -A[ending_row_of_partition:ending_row_of_partition+blocksize, ending_row_of_partition-blocksize:ending_row_of_partition]
            
            Y[i, 0:blocksize, blocksize:2*blocksize]   = np.identity(blocksize, dtype=A.dtype)
            Y[i, blocksize:2*blocksize, 0:blocksize] = np.identity(blocksize, dtype=A.dtype)

    return K, X, Y


def invert_diag_partitions(K):
    """
        Invert each diagonal partition of K.
    """

    for i in range(K.shape[0]):
        K[i] = np.linalg.inv(K[i])

    return K



def pdiv(A, blocksize):
    """
        Implementation of the pairwise algorithm as described in the 2007 paper
        and extended in 2011.
    """

    nblocks = A.shape[0] // blocksize

    G = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    vizu.vizualiseDenseMatrixFlat(A, legend="A")


    K, X, Y = create_partitions(A, blocksize)

    vizu.vizualiseDenseMatrixFlat(K[0], legend="K[0]")
    vizu.vizualiseDenseMatrixFlat(K[1], legend="K[1]")
    vizu.vizualiseDenseMatrixFlat(X[0], legend="X[0]")
    vizu.vizualiseDenseMatrixFlat(Y[0], legend="Y[0]")

    K = invert_diag_partitions(K)


    return G

