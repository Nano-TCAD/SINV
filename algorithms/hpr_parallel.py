"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.vizualisation       as vizUtils
import utils.permutationMatrices as permMat
import utils.generateMatrices    as genMat
import utils.convertMatrices     as convMat

import numpy as np
import scipy.linalg as la
import time

from mpi4py import MPI



def hpr_ulcorner_process(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks  = np.zeros((nblocks,   blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks,   blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    # FOR DEBUG
    for i in range(nblocks):
        G_diag_blocks[i]  = np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        G_upper_blocks[i] = 0.9*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        if i < nblocks-1:
            G_lower_blocks[i] = 0.9*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)


    # Schur bloc decomposition
    L = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)
    U = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)
    S = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)

    inv_DLU = np.linalg.inv(A_bloc_diag[0, ])

    L = A_bloc_lower[0, ] @ inv_DLU
    U = inv_DLU @ A_bloc_upper[0, ]
    S = A_bloc_diag[1, ] - L @ A_bloc_upper[0, ]

    inv_S = np.linalg.inv(S)



    return G_diag_blocks, G_upper_blocks, G_lower_blocks



def hpr_central_process(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    # FOR DEBUG
    for i in range(nblocks):
        G_diag_blocks[i]  = 0.66*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        G_upper_blocks[i] = 0.56*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        G_lower_blocks[i] = 0.56*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)


    # Block Permutation phase
    P = permMat.generateSchurBlockPermutationMatrix(nblocks, blockSize)
    A = convMat.convertBlocksBandedToDense(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    PAP = P @ A @ P.T

    #vizUtils.vizualiseDenseMatrixFlat(P, "P")
    #vizUtils.vizualiseDenseMatrixFlat(PAP, "PAP") 
    #vizUtils.vizualiseDenseMatrixFlat(A, "A")



    # Schur bloc decomposition
    dim_schur = 2*blockSize
    dim_decomposition = nblocks*blockSize - dim_schur

    L = np.zeros((dim_schur,         dim_decomposition), dtype=A_bloc_diag.dtype)
    U = np.zeros((dim_decomposition, dim_schur),         dtype=A_bloc_diag.dtype)
    S = np.zeros((dim_schur,         dim_schur),         dtype=A_bloc_diag.dtype)

    inv_DLU = np.linalg.inv(PAP[:dim_decomposition, :dim_decomposition])

    L = PAP[dim_decomposition:, :dim_decomposition] @ inv_DLU
    U = inv_DLU @ PAP[:dim_decomposition, dim_decomposition:]
    S = PAP[dim_decomposition:, dim_decomposition:] - L @ PAP[:dim_decomposition, dim_decomposition:]

    inv_S = np.linalg.inv(S)

    """ vizUtils.vizualiseDenseMatrixFlat(L, "L") 
    vizUtils.vizualiseDenseMatrixFlat(U, "U") 
    vizUtils.vizualiseDenseMatrixFlat(S, "S") """


    # Example using the A matrix
    #Pcr = permMat.generateCyclicReductionBlockPermutationMatrix(nblocks, blockSize)
    #PcrAPcr = Pcr @ A @ Pcr.T
    #vizUtils.compareDenseMatrix(A, PcrAPcr, "A vs PcrAPcr")



    return G_diag_blocks, G_upper_blocks, G_lower_blocks



def hpr_lrcorner_process(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    # FOR DEBUG
    for i in range(nblocks):
        G_diag_blocks[i]  = 0.33*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        if i < nblocks-1:
            G_upper_blocks[i] = 0.23*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        G_lower_blocks[i] = 0.23*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)


    # Schur bloc decomposition
    L = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)
    U = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)
    S = np.zeros((blockSize, blockSize), dtype=A_bloc_diag.dtype)

    inv_DLU = np.linalg.inv(A_bloc_diag[1, ])

    L = A_bloc_lower[1, ] @ inv_DLU
    U = inv_DLU @ A_bloc_upper[0, ]
    S = A_bloc_diag[0, ] - L @ A_bloc_upper[0, ]

    inv_S = np.linalg.inv(S)



    return G_diag_blocks, G_upper_blocks, G_lower_blocks



def hpr_parallel(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    """
        Implementation of the parallel algorithm presented in section 5. of the paper
    """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    highest_rank = comm_size - 1

    """ print("rank: ", comm_rank)
    print("size: ", comm_size)
    print("highest_rank: ", highest_rank) """

    nblocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    cornerProcessSize = 2
    centralProcessSize = nblocks - 2*cornerProcessSize


    if comm_rank == 0:
        startingBlock = 0
        endingBlock   = cornerProcessSize

        G_diag_blocks[startingBlock:endingBlock]\
        , G_upper_blocks[startingBlock:endingBlock]\
        , G_lower_blocks[startingBlock:endingBlock-1] = hpr_ulcorner_process(A_bloc_diag[startingBlock:endingBlock], A_bloc_upper[startingBlock:endingBlock], A_bloc_lower[startingBlock:endingBlock-1])

        # Gather the results from the other processes
        # - Central process
        G_diag_blocks[endingBlock   : endingBlock+centralProcessSize]   = comm.recv(source=1, tag=0)
        G_upper_blocks[endingBlock  : endingBlock+centralProcessSize]   = comm.recv(source=1, tag=1)
        G_lower_blocks[endingBlock-1: endingBlock+centralProcessSize-1] = comm.recv(source=1, tag=2)

        # - Lower right corner process
        G_diag_blocks[endingBlock+centralProcessSize   : nblocks]   = comm.recv(source=2, tag=0)
        G_upper_blocks[endingBlock+centralProcessSize  : nblocks-1] = comm.recv(source=2, tag=1)
        G_lower_blocks[endingBlock+centralProcessSize-1: nblocks-1] = comm.recv(source=2, tag=2)

    elif comm_rank == 1:
        startingBlock = cornerProcessSize
        endingBlock   = startingBlock + centralProcessSize

        G_diag_blocks[startingBlock:endingBlock]\
        , G_upper_blocks[startingBlock:endingBlock]\
        , G_lower_blocks[startingBlock-1:endingBlock-1] = hpr_central_process(A_bloc_diag[startingBlock:endingBlock], A_bloc_upper[startingBlock:endingBlock], A_bloc_lower[startingBlock-1:endingBlock-1])

        # Send the results to the gathering process: (rank 0)
        comm.send(G_diag_blocks[startingBlock:endingBlock],      dest=0, tag=0)
        comm.send(G_upper_blocks[startingBlock:endingBlock],     dest=0, tag=1)
        comm.send(G_lower_blocks[startingBlock-1:endingBlock-1], dest=0, tag=2)

    else:
        startingBlock = cornerProcessSize + centralProcessSize
        endingBlock   = nblocks

        G_diag_blocks[startingBlock:endingBlock]\
        , G_upper_blocks[startingBlock:endingBlock-1]\
        , G_lower_blocks[startingBlock-1:endingBlock-1] = hpr_lrcorner_process(A_bloc_diag[startingBlock:endingBlock], A_bloc_upper[startingBlock:endingBlock], A_bloc_lower[startingBlock-1:endingBlock])

        # Send the results to the gathering process: (rank 0)
        comm.send(G_diag_blocks[startingBlock:endingBlock],      dest=0, tag=0)
        comm.send(G_upper_blocks[startingBlock:endingBlock-1],   dest=0, tag=1)
        comm.send(G_lower_blocks[startingBlock-1:endingBlock-1], dest=0, tag=2)



    return G_diag_blocks, G_upper_blocks, G_lower_blocks



