"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.vizualisation as vizu

import algorithms.bcr      as bcr

import numpy as np
import time

from mpi4py import MPI



""" 
    Schur reductions functions
"""
def reduce_schur_topleftcorner(A, top_blockrow, bottom_blockrow, blocksize):

    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    # Corner elimination downward
    for i_blockrow in range(top_blockrow+1, bottom_blockrow):
        im1_rowindice = (i_blockrow-1)*blocksize
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize

        A_inv_im1_im1 = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice])

        L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]  = A[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
        U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]  = A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] -= L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
    
    return A, L, U


def reduce_schur_bottomrightcorner(A, top_blockrow, bottom_blockrow, blocksize):

    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    # Corner elimination upward
    for i_blockrow in range(bottom_blockrow-2, top_blockrow-1, -1):
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize
        ip2_rowindice = (i_blockrow+2)*blocksize

        A_inv_ip1_ip1 = np.linalg.inv(A[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice])

        L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] = A[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] @ A_inv_ip1_ip1
        U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] = A_inv_ip1_ip1 @ A[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]  -= L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] @ A[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]

    return A, L, U


def reduce_schur_center(A, top_blockrow, bottom_blockrow, blocksize):

    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    # Center elimination downward
    for i_blockrow in range(top_blockrow+2, bottom_blockrow):
        im1_rowindice = (i_blockrow-1)*blocksize
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize

        top_rowindice   = top_blockrow*blocksize
        topp1_rowindice = (top_blockrow+1)*blocksize

        A_inv_im1_im1 = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice])

        L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]     = A[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
        L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] = A[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
        U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]     = A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
        U[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice] = A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
        
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]         -= L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
        A[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice] -= L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
        A[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice]     -= L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
        A[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice]     -= L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]

    return A, L, U



""" 
    Schur production functions
"""
def produce_schur_topleftcorner(A, L, U, G, top_blockrow, bottom_blockrow, blocksize):

    # Corner produce upwards
    for i in range(bottom_blockrow-1, top_blockrow-1, -1):
        im1_rowindice = (i-1)*blocksize
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize

        G[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] = -G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] @ L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]
        G[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] = -U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
        G[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice] = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice]) - U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]

    return G


def produce_schur_bottomrightcorner(A, L, U, G, top_blockrow, bottom_blockrow, blocksize):

    # Corner produce downwards
    for i in range(top_blockrow, bottom_blockrow-1):
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]   = -G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] @ L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]
        G[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]   = -U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
        G[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice] = np.linalg.inv(A[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice]) - U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]

    return G


def produce_schur_center(A, L, U, G, top_blockrow, bottom_blockrow, blocksize):

    # Center produce upwards
    top_rowindice   = top_blockrow*blocksize
    topp1_rowindice = (top_blockrow+1)*blocksize
    topp2_rowindice = (top_blockrow+2)*blocksize
    topp3_rowindice = (top_blockrow+3)*blocksize

    botm1_rowindice = (bottom_blockrow-2)*blocksize
    bot_rowindice   = (bottom_blockrow-1)*blocksize
    botp1_rowindice = bottom_blockrow*blocksize

    G[bot_rowindice:botp1_rowindice, botm1_rowindice:bot_rowindice] = -G[bot_rowindice:botp1_rowindice, top_rowindice:topp1_rowindice] @ L[top_rowindice:topp1_rowindice, botm1_rowindice:bot_rowindice] - G[bot_rowindice:botp1_rowindice, bot_rowindice:botp1_rowindice] @ L[bot_rowindice:botp1_rowindice, botm1_rowindice:bot_rowindice]
    G[botm1_rowindice:bot_rowindice, bot_rowindice:botp1_rowindice] = -U[botm1_rowindice:bot_rowindice, bot_rowindice:botp1_rowindice] @ G[bot_rowindice:botp1_rowindice, bot_rowindice:botp1_rowindice] - U[botm1_rowindice:bot_rowindice, top_rowindice:topp1_rowindice] @ G[top_rowindice:topp1_rowindice, bot_rowindice:botp1_rowindice]

    for i in range(bottom_blockrow-2, top_blockrow, -1):
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] = -G[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice] @ L[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] - G[top_rowindice:topp1_rowindice, ip1_rowindice:ip2_rowindice] @ L[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]
        G[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] = -U[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] @ G[ip1_rowindice:ip2_rowindice, top_rowindice:topp1_rowindice] - U[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] @ G[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice]

    for i in range(bottom_blockrow-2, top_blockrow+1, -1):
        im1_rowindice = (i-1)*blocksize
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] = np.linalg.inv(A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]) - U[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] @ G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] - U[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] @ G[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] 
        G[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] = -U[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice] @ G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] - U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
        G[i_rowindice:i_rowindice, im1_rowindice:i_rowindice]   = -G[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] @ L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] - G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] @ L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]

    G[topp1_rowindice:topp2_rowindice, topp1_rowindice:topp2_rowindice] = np.linalg.inv(A[topp1_rowindice:topp2_rowindice, topp1_rowindice:topp2_rowindice]) - U[topp1_rowindice:topp2_rowindice, top_rowindice:topp1_rowindice] @ G[top_rowindice:topp1_rowindice, topp1_rowindice:topp2_rowindice] - U[topp1_rowindice:topp2_rowindice, topp2_rowindice:topp3_rowindice] @ G[topp2_rowindice:topp3_rowindice, topp1_rowindice:topp2_rowindice]

    return G



def inverse_hybrid(A, blocksize):
    """
        Invert a matrix using the hybrid parallel reduction algorithm
    """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nblocks   = A.shape[0]//blocksize

    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    top_blockrow     = 0
    bottom_blockrow  = 0



    # Phase 1. Schur reduction 
    if comm_rank == 0:
        # First / top process
        top_blockrow     = 0
        bottom_blockrow  = nblocks // comm_size

        A, L, U = reduce_schur_topleftcorner(A, top_blockrow, bottom_blockrow, blocksize)

    elif comm_rank == comm_size-1:
        # Last / bottom process
        top_blockrow     = comm_rank * (nblocks // comm_size)
        bottom_blockrow  = nblocks

        A, L, U = reduce_schur_bottomrightcorner(A, top_blockrow, bottom_blockrow, blocksize)
        
    else:
        # Middle process
        top_blockrow     = comm_rank * (nblocks // comm_size)
        bottom_blockrow  = (comm_rank+1) * (nblocks // comm_size)

        A, L, U = reduce_schur_center(A, top_blockrow, bottom_blockrow, blocksize)



    # Phase 2. BCR reduction
    nblocks_bcr_system = (comm_size-1) * 2

    A_bcr = np.zeros((nblocks_bcr_system*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)
    L_bcr = np.zeros((nblocks_bcr_system*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)
    U_bcr = np.zeros((nblocks_bcr_system*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)

    if comm_rank == 0:
        # Initialize first row of A_bcr with the Schur reduction of the first process, (1 row)
        bottom_rowindice   = (bottom_blockrow-1) * blocksize
        bottomp1_rowindice = bottom_blockrow * blocksize

        nblocks_process = bottom_blockrow - top_blockrow

        first_blockcol = nblocks_process - 1
        last_blockcol  = nblocks_bcr_system + first_blockcol
        first_colindice = first_blockcol * blocksize
        last_colindice  = last_blockcol * blocksize

        A_bcr[0:blocksize, :] = A[bottom_rowindice:bottomp1_rowindice, first_colindice:last_colindice]
        L_bcr[0:blocksize, :] = L[bottom_rowindice:bottomp1_rowindice, first_colindice:last_colindice]

        # Receive the Schur reduction from the central process, (2 rows)
        for i in range(1, comm_size-1):
            i_rowindice   = blocksize + (2 * (i-1)) * blocksize
            ip2_rowindice = i_rowindice + 2 * blocksize
            
            A_bcr[i_rowindice:ip2_rowindice, :] = comm.recv(source=i, tag=0)
            L_bcr[i_rowindice:ip2_rowindice, :] = comm.recv(source=i, tag=1)
            U_bcr[i_rowindice:ip2_rowindice, :] = comm.recv(source=i, tag=2)

        # Receive the Schur reduction from the last process, (1 row)
        last_rowindice   = (nblocks_bcr_system-1)*blocksize
        lastp1_rowindice = nblocks_bcr_system*blocksize

        A_bcr[last_rowindice:lastp1_rowindice, :] = comm.recv(source=comm_size-1, tag=0)
        L_bcr[last_rowindice:lastp1_rowindice, :] = comm.recv(source=comm_size-1, tag=1)
        U_bcr[last_rowindice:lastp1_rowindice, :] = comm.recv(source=comm_size-1, tag=2)


        # Compute the BCR reduction of the aggregated system
        vizu.vizualiseDenseMatrixFlat(A_bcr, "A_bcr")

        G_bcr = bcr.inverse_bcr(A_bcr, blocksize)

        vizu.vizualiseDenseMatrixFlat(G_bcr, "G_bcr")


    elif comm_rank == comm_size-1:
        # Last process send his Schur reduced rows to process 0 
        top_rowindice   = top_blockrow * blocksize
        topp1_rowindice = (top_blockrow+1) * blocksize

        nblocks_process = bottom_blockrow - top_blockrow

        last_blockcol  = nblocks - (nblocks_process-1)
        first_blockcol = last_blockcol - nblocks_bcr_system
        first_colindice = first_blockcol * blocksize
        last_colindice  = last_blockcol * blocksize

        comm.send(A[top_rowindice:topp1_rowindice, first_colindice:last_colindice], dest=0, tag=0)
        comm.send(L[top_rowindice:topp1_rowindice, first_colindice:last_colindice], dest=0, tag=1)
        comm.send(U[top_rowindice:topp1_rowindice, first_colindice:last_colindice], dest=0, tag=2)

    else:
        # Middle process send his Schur reduced rows to process 0 
        A_reduced = np.zeros((2*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)
        L_reduced = np.zeros((2*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)
        U_reduced = np.zeros((2*blocksize, nblocks_bcr_system*blocksize), dtype=A.dtype)

        # Rows strides
        top_rowindice   = top_blockrow * blocksize
        topp1_rowindice = (top_blockrow+1) * blocksize

        bottom_rowindice   = (bottom_blockrow-1) * blocksize
        bottomp1_rowindice = bottom_blockrow * blocksize

        # Columns strides
        nblocks_process = bottom_blockrow - top_blockrow

        start1_blockcol  = comm_rank * (nblocks // comm_size) - 1
        stop1_blockcol   = start1_blockcol + 2
        start1_colindice = start1_blockcol * blocksize
        stop1_colindice  = stop1_blockcol * blocksize

        start2_blockcol = stop1_blockcol + (nblocks_process-2)
        stop2_blockcol  = start2_blockcol + 2
        start2_colindice = start2_blockcol * blocksize
        stop2_colindice  = stop2_blockcol * blocksize

        # Row 0, Col 0
        A_reduced[0:blocksize, 0:2*blocksize]           = A[top_rowindice:topp1_rowindice, start1_colindice:stop1_colindice]
        L_reduced[0:blocksize, 0:2*blocksize]           = L[top_rowindice:topp1_rowindice, start1_colindice:stop1_colindice]
        U_reduced[0:blocksize, 0:2*blocksize]           = U[top_rowindice:topp1_rowindice, start1_colindice:stop1_colindice]

        # Row 1, Col 0
        A_reduced[blocksize:2*blocksize, 0:2*blocksize] = A[bottom_rowindice:bottomp1_rowindice, start1_colindice:stop1_colindice]
        L_reduced[blocksize:2*blocksize, 0:2*blocksize] = L[bottom_rowindice:bottomp1_rowindice, start1_colindice:stop1_colindice]
        U_reduced[blocksize:2*blocksize, 0:2*blocksize] = U[bottom_rowindice:bottomp1_rowindice, start1_colindice:stop1_colindice]

        # Row 0, Col 1
        A_reduced[0:blocksize, 2*blocksize:]           = A[top_rowindice:topp1_rowindice, start2_colindice:stop2_colindice]
        L_reduced[0:blocksize, 2*blocksize:]           = L[top_rowindice:topp1_rowindice, start2_colindice:stop2_colindice]
        U_reduced[0:blocksize, 2*blocksize:]           = U[top_rowindice:topp1_rowindice, start2_colindice:stop2_colindice]

        # Row 1, Col 1
        A_reduced[blocksize:2*blocksize, 2*blocksize:] = A[bottom_rowindice:bottomp1_rowindice, start2_colindice:stop2_colindice]
        L_reduced[blocksize:2*blocksize, 2*blocksize:] = L[bottom_rowindice:bottomp1_rowindice, start2_colindice:stop2_colindice]
        U_reduced[blocksize:2*blocksize, 2*blocksize:] = U[bottom_rowindice:bottomp1_rowindice, start2_colindice:stop2_colindice]

        comm.send(A_reduced, dest=0, tag=0)
        comm.send(L_reduced, dest=0, tag=1)
        comm.send(U_reduced, dest=0, tag=2)


