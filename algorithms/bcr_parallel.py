"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

@reference: https://doi.org/10.1016/j.jcp.2009.03.035
@reference: https://doi.org/10.1017/CBO9780511812583

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.vizualisation       as vizu
import utils.transformMatrices   as transMat

import numpy as np
import math
import time

from mpi4py import MPI



def inverse_bcr_parallel(A, blocksize):
    """
        Compute the tridiagonal-selected inverse of a matrix A using block cyclic reduction
    """

    # Padd the input matrices with identity blocks to make sure the number of blocks is a 
    # power of two - 1, condition required by the block cyclic reduction algorithm
    nblocks_initial = A.shape[0] // blocksize
    block_padding_distance = transMat.distance_to_power_of_two(nblocks_initial)

    A = transMat.identity_padding(A, block_padding_distance*blocksize)

    nblocks_padded = A.shape[0] // blocksize

    L = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)
    G = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)

    process_top_blockrow, process_bottom_blockrow = get_process_rows_ownership(nblocks_padded)
    

    # 1. Block cyclic reduction
    i_bcr = [i for i in range(nblocks_padded)]
    final_reduction_block = 0

    A, L, U, final_reduction_block = reduce_bcr(A, L, U, i_bcr, process_top_blockrow, process_bottom_blockrow, blocksize)


    # 2. Block cyclic production
    invert_block(A, G, final_reduction_block, process_top_blockrow, process_bottom_blockrow, blocksize)
    produce_bcr(A, L, U, G, i_bcr, process_top_blockrow, process_bottom_blockrow, blocksize)
    G = agregate_result_on_root(G, nblocks_padded, process_top_blockrow, process_bottom_blockrow, blocksize)


    # Cut the padding
    G = G[:nblocks_initial*blocksize, :nblocks_initial*blocksize]

    return G



###############################################################################
#                                                                             #
#                       BLOCK REDUCTIONS FUNCTIONS                            #
#                                                                             #
###############################################################################
def reduce(A, L, U, row, level, i_elim, top_blockrow, bottom_blockrow, blocksize):
    """

    """

    nblocks = A.shape[0] // blocksize
    offset_blockindex = int(math.pow(2, level)) 


    # Reduction from i (above row) and k (below row) to j row
    i_blockindex = i_elim[row] - offset_blockindex
    j_blockindex = i_elim[row]
    k_blockindex = i_elim[row] + offset_blockindex


    # Computing of row-based indices
    i_rowindex   = i_blockindex * blocksize
    ip1_rowindex = (i_blockindex + 1) * blocksize

    j_rowindex   = j_blockindex * blocksize
    jp1_rowindex = (j_blockindex + 1) * blocksize
    
    k_rowindex   = k_blockindex * blocksize
    kp1_rowindex = (k_blockindex + 1) * blocksize


    # If there is a row above
    if i_blockindex >= 0: 
        A_ii_inv = np.linalg.inv(A[i_rowindex:ip1_rowindex, i_rowindex:ip1_rowindex])
        U[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex] = A_ii_inv @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]
        L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] = A[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A_ii_inv
        
        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] = A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] - L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]

        # If the row above is not the top row
        if i_blockindex != i_elim[0]:
            h_rowindex = (i_blockindex - offset_blockindex) * blocksize
            hp1_rowindex = (i_blockindex - offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, h_rowindex:hp1_rowindex] = - L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A[i_rowindex:ip1_rowindex, h_rowindex:hp1_rowindex]

    # If there is a row below
    if k_blockindex <= nblocks-1:
        A_kk_inv = np.linalg.inv(A[k_rowindex:kp1_rowindex, k_rowindex:kp1_rowindex])
        U[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex] = A_kk_inv @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]
        L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] = A[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] @ A_kk_inv

        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] = A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] - L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]

        # If the row below is not the bottom row
        if k_blockindex != i_elim[-1]:
            l_rowindex   = (k_blockindex + offset_blockindex) * blocksize
            lp1_rowindex = (k_blockindex + offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, l_rowindex:lp1_rowindex] = - L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] @ A[k_rowindex:kp1_rowindex, l_rowindex:lp1_rowindex]

    return A, L, U


def reduce_bcr(A, L, U, i_bcr, top_blockrow, bottom_blockrow, blocksize):

    nblocks = len(i_bcr)
    height  = int(math.log2(nblocks))

    last_reduction_block = 0

    for level_blockindex in range(height):
        i_elim = compute_i_from(level_blockindex, nblocks)

        number_of_reduction = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                number_of_reduction += 1

        indice_process_start_reduction = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                indice_process_start_reduction = i
                break

        indice_process_stop_reduction  = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                indice_process_stop_reduction = i

        if number_of_reduction != 0:
            for row in range(indice_process_start_reduction, indice_process_stop_reduction + 1):
                A, L, U = reduce(A, L, U, row, level_blockindex, i_elim, top_blockrow, bottom_blockrow, blocksize)

        # Here each process should communicate the last row of the reduction to the next process
        if level_blockindex != height - 1:
            send_reducprod(A, L, U, i_elim, indice_process_start_reduction, indice_process_stop_reduction, blocksize)
            recv_reducprod(A, L, U, i_elim, indice_process_start_reduction, indice_process_stop_reduction, blocksize)

        if len(i_elim) > 0:
            last_reduction_block = i_elim[-1]

    return A, L, U, last_reduction_block



###############################################################################
#                                                                             #
#                         BLOCK PRODUCTION FUNCTIONS                          #
#                                                                             #
###############################################################################
def corner_produce(A, L, U, G, k_from, k_to, blocksize):
    """
        Corner process block production
    """
    
    k_from_rowindex   = k_from * blocksize
    kp1_from_rowindex = (k_from + 1) * blocksize

    k_to_rowindex     = k_to * blocksize
    kp1_to_rowindex   = (k_to + 1) * blocksize


    G[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex] = - G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex] @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]
    G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex] = - U[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex] @ G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex]
    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex]     = np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex]) - G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex] @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]

    return G


def center_produce(A, L, U, G, k_above, k_to, k_below, blocksize):
    """
        Center process block production
    """
    
    k_above_rowindex   = k_above * blocksize
    kp1_above_rowindex = (k_above + 1) * blocksize

    k_to_rowindex      = k_to * blocksize
    kp1_to_rowindex    = (k_to + 1) * blocksize

    k_below_rowindex   = k_below * blocksize
    kp1_below_rowindex = (k_below + 1) * blocksize
    

    G[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex] = - G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex] @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                                                                                - G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex] @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    G[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex] = - G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex] @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                                                                                - G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex] @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] = - U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] @ G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex]\
                                                                                - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] @ G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex]
    G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] = - U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] @ G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex]\
                                                                                - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] @ G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex]
    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex]       = np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex]) - G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                                                                                - G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    
    return G


def produce(A, L, U, G, i_bcr, i_prod, stride_blockindex, top_blockrow, bottom_blockrow, blocksize):
    """
        Call the appropriate production function for each block that needs to be produced
    """

    for i_prod_blockindex in range(len(i_prod)):
        k_to = i_bcr[i_prod[i_prod_blockindex]]

        if k_to >= top_blockrow and k_to < bottom_blockrow:

            if i_prod_blockindex == 0:
                # Production row is the first row within the stride_blockindex range
                # It only gets values from the below row 
                k_from = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                G = corner_produce(A, L, U, G, k_from, k_to, blocksize)

            if i_prod_blockindex != 0 and i_prod_blockindex == len(i_prod) - 1:
                if i_prod[-1] <= len(i_bcr) - stride_blockindex -1:
                    k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                    k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                    G = center_produce(A, L, U, G, k_above, k_to, k_below, blocksize)
                else:
                    # Production row is the last row within the stride_blockindex range
                    # It only gets values from the above row 
                    k_from = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                
                    G = corner_produce(A, L, U, G, k_from, k_to, blocksize)
            
            if i_prod_blockindex != 0 and i_prod_blockindex != len(i_prod) - 1:
                # Production row is in the middle of the stride_blockindex range
                # It gets values from the above and below rows
                k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                G = center_produce(A, L, U, G, k_above, k_to, k_below,  blocksize)


def comm_to_produce(A, L, U, G, i_from, i_prod, stride_blockindex, top_blockrow, bottom_blockrow, blocksize):
    """
        Determine the blocks-row that need to be communicated/received to/from other processes in order
        for each process to be able to locally produce the blocks-row it is responsible for
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()


    # Determine what belongs to the current process
    current_process_i_from = []
    for i in range(len(i_from)):
        if i_from[i] >= top_blockrow and i_from[i] < bottom_blockrow:
            current_process_i_from.append(i_from[i])

    current_process_i_prod = []
    for i in range(len(i_prod)):
        if i_prod[i] >= top_blockrow and i_prod[i] < bottom_blockrow:
            current_process_i_prod.append(i_prod[i])


    # Determine what needs to be sent to other processes
    n_blocks = A.shape[0] // blocksize
    for i in range(len(current_process_i_from)):
        k_from     = current_process_i_from[i]
        k_to_above = k_from-stride_blockindex
        k_to_below = k_from+stride_blockindex

        k_from_rowindex   = k_from * blocksize
        kp1_from_rowindex = k_from_rowindex + blocksize
        
        if k_to_above < top_blockrow and k_to_above >= 0:
            #k_to_above_rowindex   = k_to_above * blocksize
            #kp1_to_above_rowindex = k_to_above_rowindex + blocksize 

            comm.send(A[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank-1, tag=0)
            comm.send(L[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank-1, tag=1)
            comm.send(U[:, k_from_rowindex:kp1_from_rowindex], dest=comm_rank-1, tag=2)
            comm.send(G[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank-1, tag=3)
            
            #print("Process: ", comm_rank, " send ", k_from, " to produce: ", k_to_above)

        if k_to_below >= bottom_blockrow and k_to_below < n_blocks:
            #k_to_below_rowindex   = k_to_below * blocksize
            #kp1_to_below_rowindex = k_to_below_rowindex + blocksize 

            comm.send(A[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank+1, tag=0)
            comm.send(L[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank+1, tag=1)
            comm.send(U[:, k_from_rowindex:kp1_from_rowindex], dest=comm_rank+1, tag=2)
            comm.send(G[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank+1, tag=3)

            #print("Process: ", comm_rank, " send ", k_from, " to produce: ", k_to_below)


    # Determine what needs to be received from other processes
    for i in range(len(current_process_i_prod)):
        k_from_above = current_process_i_prod[i]-stride_blockindex
        k_from_below = current_process_i_prod[i]+stride_blockindex
        if k_from_above < top_blockrow and k_from_above >= 0:
            k_from_above_rowindex   = k_from_above * blocksize
            kp1_from_above_rowindex = k_from_above_rowindex + blocksize

            A[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(source=comm_rank-1, tag=0)
            L[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(source=comm_rank-1, tag=1)
            U[:, k_from_above_rowindex:kp1_from_above_rowindex] = comm.recv(source=comm_rank-1, tag=2)
            G[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(source=comm_rank-1, tag=3)

        if k_from_below >= bottom_blockrow and k_from_below < n_blocks:
            k_from_below_rowindex   = k_from_below * blocksize
            kp1_from_below_rowindex = k_from_below_rowindex + blocksize

            A[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(source=comm_rank+1, tag=0)
            L[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(source=comm_rank+1, tag=1)
            U[:, k_from_below_rowindex:kp1_from_below_rowindex] = comm.recv(source=comm_rank+1, tag=2)
            G[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(source=comm_rank+1, tag=3)


def comm_produced(G, i_prod, i_from, stride_blockindex, top_blockrow, bottom_blockrow, blocksize):
    """
        Communicate part of the produced row back to the original process
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    

    # Determine what belongs to the current process
    current_process_i_from = []
    for i in range(len(i_from)):
        if i_from[i] >= top_blockrow and i_from[i] < bottom_blockrow:
            current_process_i_from.append(i_from[i])

    current_process_i_prod = []
    for i in range(len(i_prod)):
        if i_prod[i] >= top_blockrow and i_prod[i] < bottom_blockrow:
            current_process_i_prod.append(i_prod[i])


    # Determine what needs to be sent to other processes
    n_blocks = G.shape[0] // blocksize
    for i in range(len(current_process_i_from)):
        k_from     = current_process_i_from[i]
        k_to_above = k_from-stride_blockindex
        k_to_below = k_from+stride_blockindex

        k_from_colindex   = k_from * blocksize
        kp1_from_colindex = k_from_colindex + blocksize
 
        if k_to_above < top_blockrow and k_to_above >= 0:
            k_to_above_rowindex   = k_to_above * blocksize
            kp1_to_above_rowindex = k_to_above_rowindex + blocksize

            comm.send(G[k_to_above_rowindex:kp1_to_above_rowindex, k_from_colindex:kp1_from_colindex], dest=comm_rank-1, tag=3)

        if k_to_below >= bottom_blockrow and k_to_below < n_blocks:
            k_to_below_rowindex   = k_to_below * blocksize
            kp1_to_below_rowindex = k_to_below_rowindex + blocksize

            comm.send(G[k_to_below_rowindex:kp1_to_below_rowindex, k_from_colindex:kp1_from_colindex], dest=comm_rank+1, tag=3)


    # Determine what needs to be received from other processes
    for i in range(len(current_process_i_prod)):
        k_to = current_process_i_prod[i]
        k_from_above = k_to-stride_blockindex
        k_from_below = k_to+stride_blockindex

        k_to_rowindex   = k_to * blocksize
        kp1_to_rowindex = k_to_rowindex + blocksize

        if k_from_above < top_blockrow and k_from_above >= 0:
            k_from_above_colindex   = k_from_above * blocksize
            kp1_from_above_colindex = k_from_above_colindex + blocksize

            G[k_to_rowindex:kp1_to_rowindex, k_from_above_colindex:kp1_from_above_colindex] = comm.recv(source=comm_rank-1, tag=3)
        
        if k_from_below >= bottom_blockrow and k_from_below < n_blocks:
            k_from_below_colindex   = k_from_below * blocksize
            kp1_from_below_colindex = k_from_below_colindex + blocksize

            G[k_to_rowindex:kp1_to_rowindex, k_from_below_colindex:kp1_from_below_colindex] = comm.recv(source=comm_rank+1, tag=3)


def produce_bcr(A, L, U, G, i_bcr, top_blockrow, bottom_blockrow, blocksize):
    """
        Produce the inverse of the A matrix inside of G
    """

    nblocks = len(i_bcr)
    height  = int(math.log2(nblocks))

    for level_blockindex in range(height-1, -1, -1):
        stride_blockindex = int(math.pow(2, level_blockindex))

        # Determine the blocks-row to be produced
        i_from = compute_i_from(level_blockindex, nblocks)
        i_prod = compute_i_prod(i_from, stride_blockindex)

        comm_to_produce(A, L, U, G, i_from, i_prod, stride_blockindex, top_blockrow, bottom_blockrow, blocksize)

        produce(A, L, U, G, i_bcr, i_prod, stride_blockindex, top_blockrow, bottom_blockrow, blocksize)

        comm_produced(G, i_from, i_prod, stride_blockindex, top_blockrow, bottom_blockrow, blocksize)



###############################################################################
#                                                                             #
#                                   UTILS                                     #
#                                                                             #
###############################################################################
def get_process_rows_ownership(n_blocks):
    """
        Determine the rows owned by each process
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    process_top_blockrow     = 0
    process_bottom_blockrow  = 0

    if comm_rank == 0:
        # First / top process
        process_top_blockrow     = 0
        process_bottom_blockrow  = n_blocks // comm_size
    elif comm_rank == comm_size-1:
        # Last / bottom process
        process_top_blockrow     = comm_rank * (n_blocks // comm_size)
        process_bottom_blockrow  = n_blocks
    else:
        # Middle process
        process_top_blockrow     = comm_rank * (n_blocks // comm_size)
        process_bottom_blockrow  = (comm_rank+1) * (n_blocks // comm_size)

    return process_top_blockrow, process_bottom_blockrow


def compute_i_from(level, nblocks):
    """
        Compute the blocks-row that will be used to produce blocks-row at the current level of the production tree
    """

    return [i for i in range(int(math.pow(2, level + 1)) - 1, nblocks, int(math.pow(2, level + 1)))]


def compute_i_prod(i_from, stride_blockindex):
    """
        Compute the blocks-row to be produced at the current level of the production tree
    """

    i_prod = []
    for i in range(len(i_from)):
        if i == 0:
            i_prod.append(i_from[i] - stride_blockindex)
            i_prod.append(i_from[i] + stride_blockindex)
        else:
            if i_prod[i] != i_from[i] - stride_blockindex:
                i_prod.append(i_from[i] - stride_blockindex)
            i_prod.append(i_from[i] + stride_blockindex)

    return i_prod    


def send_reducprod(A, L, U, i_from, indice_process_start_reduction, indice_process_stop_reduction, blocksize):

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    if comm_rank == 0:
        # Only need to send to 1 bottom process
        # Send i_from[indice_process_stop_reduction]
        i_rowindice   = i_from[indice_process_stop_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_stop_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank+1, tag=2)

    elif comm_rank == comm_size - 1:
        # Only need to send to 1 top process
        # Send i_from[indice_process_start_reduction]

        i_rowindice   = i_from[indice_process_start_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_start_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank-1, tag=2)

    else:
        # Need to send to 1 top process and 1 bottom process
        # Send i_from[indice_process_start_reduction] and i_from[indice_process_stop_reduction]

        i_rowindice   = i_from[indice_process_start_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_start_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank-1, tag=2)

        i_rowindice   = i_from[indice_process_stop_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_stop_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank+1, tag=2)


def recv_reducprod(A, L, U, i_to, indice_process_start_reduction, indice_process_stop_reduction, blocksize):
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    if comm_rank == 0:
        # Only need to recv from 1 bottom process
        # Recv: i_to[indice_process_stop_reduction+1]

        i_rowindice   = i_to[indice_process_stop_reduction+1] * blocksize
        ip1_rowindice = (i_to[indice_process_stop_reduction+1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank+1, tag=2)

    elif comm_rank == comm_size - 1:
        # Only need to recv from 1 top process
        # Recv i_to[indice_process_start_reduction-1]

        i_rowindice   = i_to[indice_process_start_reduction-1] * blocksize
        ip1_rowindice = (i_to[indice_process_start_reduction-1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank-1, tag=2)

    else:
        # Need to recv from 1 top process and 1 bottom process
        # Recv i_to[indice_process_start_reduction-1] and i_to[indice_process_stop_reduction+1]

        i_rowindice   = i_to[indice_process_start_reduction-1] * blocksize
        ip1_rowindice = (i_to[indice_process_start_reduction-1]+1) * blocksize 

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank-1, tag=2)

        i_rowindice   = i_to[indice_process_stop_reduction+1] * blocksize
        ip1_rowindice = (i_to[indice_process_stop_reduction+1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank+1, tag=2)


def agregate_result_on_root(G, nblocks_padded, top_blockrow, bottom_blockrow, blocksize):

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        # Root process that need to agregate all the results

        for i in range(1, comm_size-1):
            # Receive from all the central processes
            top_blockrow_pi     = i * (nblocks_padded // comm_size)
            bottom_blockrow_pi  = (i+1) * (nblocks_padded // comm_size)

            top_rowindice_pi    = top_blockrow_pi * blocksize
            bottom_rowindice_pi = bottom_blockrow_pi * blocksize

            G[top_rowindice_pi:bottom_rowindice_pi, :] = comm.recv(source=i, tag=0)

        # Receive from the last process
        top_blockrow_pi     = (comm_size-1) * (nblocks_padded // comm_size)
        bottom_blockrow_pi  = nblocks_padded

        top_rowindice_pi    = top_blockrow_pi * blocksize
        bottom_rowindice_pi = bottom_blockrow_pi * blocksize

        G[top_rowindice_pi:bottom_rowindice_pi, :] = comm.recv(source=comm_size-1, tag=0)

    else:
        top_rowindice    = top_blockrow * blocksize
        bottom_rowindice = bottom_blockrow * blocksize

        comm.send(G[top_rowindice:bottom_rowindice, :], dest=0, tag=0)

    return G


def invert_block(A, G, target_block, top_blockrow, bottom_blockrow, blocksize):
    """
        Invert a block of the matrix A and store it in G
    """
    if target_block >= top_blockrow and target_block < bottom_blockrow:
        target_row    = target_block * blocksize
        target_row_p1 = (target_block + 1) * blocksize

        G[target_row: target_row_p1, target_row: target_row_p1] = np.linalg.inv(A[target_row: target_row_p1, target_row: target_row_p1])


def send_to_produce(A, L, U, G, i_from, indice_process_start_reduction, indice_process_stop_reduction, blocksize):

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    if comm_rank == 0:
        # Only need to send to 1 bottom process
        # Send i_from[indice_process_stop_reduction]
        i_rowindice   = i_from[indice_process_stop_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_stop_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank+1, tag=2)
        comm.send(G[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=3)

    elif comm_rank == comm_size - 1:
        # Only need to send to 1 top process
        # Send i_from[indice_process_start_reduction]

        i_rowindice   = i_from[indice_process_start_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_start_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank-1, tag=2)
        comm.send(G[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=3)

    else:
        # Need to send to 1 top process and 1 bottom process
        # Send i_from[indice_process_start_reduction] and i_from[indice_process_stop_reduction]

        i_rowindice   = i_from[indice_process_start_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_start_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank-1, tag=2)
        comm.send(G[i_rowindice:ip1_rowindice, :], dest=comm_rank-1, tag=3)

        i_rowindice   = i_from[indice_process_stop_reduction] * blocksize
        ip1_rowindice = (i_from[indice_process_stop_reduction]+1) * blocksize

        comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=0)
        comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=1)
        comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank+1, tag=2)
        comm.send(G[i_rowindice:ip1_rowindice, :], dest=comm_rank+1, tag=3)


def rcve_to_produce(A, L, U, G, i_to, indice_process_start_reduction, indice_process_stop_reduction, blocksize):
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()


    if comm_rank == 0:
        # Only need to recv from 1 bottom process
        i_rowindice   = i_to[indice_process_stop_reduction+1] * blocksize
        ip1_rowindice = (i_to[indice_process_stop_reduction+1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank+1, tag=2)
        G[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=3)

    elif comm_rank == comm_size - 1:
        # Only need to recv from 1 top process
        i_rowindice   = i_to[indice_process_start_reduction-1] * blocksize
        ip1_rowindice = (i_to[indice_process_start_reduction-1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank-1, tag=2)
        G[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=3)

    else:
        # Need to recv from 1 top process and 1 bottom process
        i_rowindice   = i_to[indice_process_start_reduction-1] * blocksize
        ip1_rowindice = (i_to[indice_process_start_reduction-1]+1) * blocksize 

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank-1, tag=2)
        G[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank-1, tag=3)

        i_rowindice   = i_to[indice_process_stop_reduction+1] * blocksize
        ip1_rowindice = (i_to[indice_process_stop_reduction+1]+1) * blocksize

        A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=0)
        L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=1)
        U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank+1, tag=2)
        G[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank+1, tag=3)

