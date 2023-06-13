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




def schurInvert(A):
    # Compute the inverse of A using an explicite Schur decomposition
    # - Only interesting for teaching purposes

    size = A.shape[0]
    size2 = size//2

    # Handmade Schur decomposition
    Ls = np.zeros((size2, size2), dtype=A.dtype)
    Us = np.zeros((size2, size2), dtype=A.dtype)
    Sc = np.zeros((size2, size2), dtype=A.dtype)

    Ls = A[size2:, :size2] @ np.linalg.inv(A[:size2, :size2])
    Us = np.linalg.inv(A[:size2, :size2]) @ A[:size2, size2:]
    Sc = A[size2:, size2:] - A[size2:, :size2] @ np.linalg.inv(A[:size2, :size2]) @ A[:size2, size2:]

    G = np.zeros((size, size), dtype=A.dtype)

    G11 = np.zeros((size2, size2), dtype=A.dtype)
    G12 = np.zeros((size2, size2), dtype=A.dtype)
    G21 = np.zeros((size2, size2), dtype=A.dtype)
    G22 = np.zeros((size2, size2), dtype=A.dtype)

    G11 = np.linalg.inv(A[:size2, :size2]) + Us @ np.linalg.inv(Sc) @ Ls
    G12 = -Us @ np.linalg.inv(Sc)
    G21 = -np.linalg.inv(Sc) @ Ls
    G22 = np.linalg.inv(Sc)

    G[:size2, :size2] = G11
    G[:size2, size2:] = G12
    G[size2:, :size2] = G21
    G[size2:, size2:] = G22

    return G



def hpr_serial(A, blockSize):
    # Implementation of the serial algorithm presented in section 3. of the paper
    # - The algorithm is equivalent to an RGF but with explicit LU decomposition
    # - TODO: Convert to block storage version (dense for now)
    # - TODO: Move around the additional inversion of D in the BWD recurence

    size    = A.shape[0]
    nBlocks = size//blockSize

    G = np.zeros((size, size), dtype=A.dtype)
    L = np.zeros((size, size), dtype=A.dtype)
    D = np.zeros((size, size), dtype=A.dtype)
    U = np.zeros((size, size), dtype=A.dtype)


    tic = time.time() # -----------------------------
    # Initialisation of forward recurence
    D[0:blockSize, 0:blockSize] = A[0:blockSize, 0:blockSize]
    L[blockSize:2*blockSize, 0:blockSize]  = A[blockSize:2*blockSize, 0:blockSize] @ np.linalg.inv(D[0:blockSize, 0:blockSize])
    U[0:blockSize, blockSize:2*blockSize]  = np.linalg.inv(D[0:blockSize, 0:blockSize]) @ A[0:blockSize, blockSize:2*blockSize]

    # Forward recurence
    for i in range(0, nBlocks-1):
        b_i   = i*blockSize
        b_ip1 = (i+1)*blockSize
        b_ip2 = (i+2)*blockSize

        D_inv_i = np.linalg.inv(D[b_i:b_ip1, b_i:b_ip1])

        D[b_ip1:b_ip2, b_ip1:b_ip2] = A[b_ip1:b_ip2, b_ip1:b_ip2] - A[b_ip1:b_ip2, b_i:b_ip1] @ D_inv_i @ A[b_i:b_ip1, b_ip1:b_ip2]
        L[b_ip1:b_ip2, b_i:b_ip1]   = A[b_ip1:b_ip2, b_i:b_ip1] @ D_inv_i
        U[b_i:b_ip1, b_ip1:b_ip2]   = D_inv_i @ A[b_i:b_ip1, b_ip1:b_ip2]

    # Initialisation of backward recurence
    b_nm1 = (nBlocks-2)*blockSize
    b_n   = (nBlocks-1)*blockSize
    b_np1 = nBlocks*blockSize

    G[b_n:b_np1, b_n:b_np1] = np.linalg.inv(D[b_n:b_np1, b_n:b_np1])
    G[b_nm1:b_n, b_n:b_np1] = -G[b_n:b_np1, b_n:b_np1] @ L[b_n:b_np1, b_nm1:b_n]
    G[b_n:b_np1, b_nm1:b_n] = -U[b_nm1:b_n, b_n:b_np1] @ G[b_n:b_np1, b_n:b_np1]

    # Backward recurence
    for i in range(nBlocks-1, 0, -1):
        b_im1 = (i-1)*blockSize
        b_i   = i*blockSize
        b_ip1 = (i+1)*blockSize

        G[b_im1:b_i, b_im1:b_i] = np.linalg.inv(D[b_im1:b_i, b_im1:b_i]) + U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_i:b_ip1, b_im1:b_i] = -G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_im1:b_i, b_i:b_ip1] = -U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1]
    toc = time.time() # -----------------------------

    
    timing = toc-tic

    return G, timing












def hpr_ulcorner_process(A_bloc_diag, A_bloc_upper, A_bloc_lower):
    comm = MPI.COMM_WORLD

    nblocks   = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)

    # FOR DEBUG
    for i in range(nblocks):
        G_diag_blocks[i]  = np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        G_upper_blocks[i] = 0.9*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)
        if i < nblocks-1:
            G_lower_blocks[i] = 0.9*np.ones((blockSize, blockSize), dtype=A_bloc_diag.dtype)

    L = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    U = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    S = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)

    L = A_bloc_lower[0, ] @ np.linalg.inv(A_bloc_diag[0, ])
    U = np.linalg.inv(A_bloc_diag[0, ]) @ A_bloc_upper[0, ]
    S = A_bloc_diag[1, ] - L @ A_bloc_upper[0, ]


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

    P = permMat.generateBlockPermutationMatrix(nblocks, blockSize)
    A = convMat.convertBlocksBandedToDense(A_bloc_diag, A_bloc_upper, A_bloc_lower)

    PAP = P @ A @ P.T

    vizUtils.vizualiseDenseMatrixFlat(P, "P")
    vizUtils.vizualiseDenseMatrixFlat(PAP, "PAP")




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





    return G_diag_blocks, G_upper_blocks, G_lower_blocks



def hpr(A_bloc_diag, A_bloc_upper, A_bloc_lower):
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
        , G_lower_blocks[startingBlock-1:endingBlock-1] = hpr_lrcorner_process(A_bloc_diag[startingBlock:endingBlock], A_bloc_upper[startingBlock:endingBlock], A_bloc_lower[startingBlock:endingBlock])

        # Send the results to the gathering process: (rank 0)
        comm.send(G_diag_blocks[startingBlock:endingBlock],      dest=0, tag=0)
        comm.send(G_upper_blocks[startingBlock:endingBlock-1],   dest=0, tag=1)
        comm.send(G_lower_blocks[startingBlock-1:endingBlock-1], dest=0, tag=2)



    return G_diag_blocks, G_upper_blocks, G_lower_blocks









# Algorithms

def inverseBCR(A, blockSize):
    size    = A.shape[0]
    nBlocks = size//blockSize

    G = np.zeros((size, size), dtype=A.dtype)
    
    P, L, U = la.lu(A)


    # 1:
    i_bcr = np.array([i for i in range(0, nBlocks, 1)])

    # 2:
    A, L ,U = reduceBCR(A, L, U, i_bcr, blockSize)

    # 3:
    #G[(nBlocks-1)*blockSize:nBlocks*blockSize, (nBlocks-1)*blockSize:nBlocks*blockSize] = np.linalg.inv(A[(nBlocks-1)*blockSize:nBlocks*blockSize, (nBlocks-1)*blockSize:nBlocks*blockSize])

    # 4:
    #G = produceBCR(A, L, U, G, i_bcr, blockSize)


    return P @ G



# BCR reduction functions

def reduceBCR(A, L, U, i_bcr, blockSize):
    k = i_bcr.shape[0]
    h = int(np.log2(k))

    for level in range(0, h, 1):
        # i_elim takes the active rows of the current level
        i_elim = np.array([i for i in range(0, k, 2**(level+1))])
        print("i_elim", i_elim)

    return A, L, U


def reduce(A, L, U, row, level, i_elim):

    return 0



# BCR production functions

def produceBCR(A, L, U, G, i_bcr, blockSize):

    return 0


def cornerProduced(A, L, U, G, k_from, k_to, blockSize):

    return 0


def centerProduce(A, L, U, G, k_above, k_to, k_below, blockSize):

    return 0