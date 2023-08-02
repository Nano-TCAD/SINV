"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import utils
from sinv import algorithms as alg
from sinv.algorithms.psr import psr_utils as psr_u

import numpy as np
import time

from mpi4py import MPI



def reduce_schur(A: np.ndarray, 
                 l_start_blockrow: list,
                 l_partitions_blocksizes: list,
                 blocksize: int):
    """ Proceed to the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    l_start_blockrow : list
        list of the starting blockrow of each process
    l_partitions_blocksizes : list
        list of the blocksize of each process
    blocksize : int
        size of a block
        
    Returns
    -------
    A : np.ndarray
        diagonal decomposition of A, work-in-place
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    start_blockrow = l_start_blockrow[comm_rank]
    partition_blocksize = l_partitions_blocksizes[comm_rank]
    
    if comm_rank == 0:
        # Is the first process
        A, L, U = reduce_schur_topleftcorner(A, start_blockrow, partition_blocksize, blocksize)
        return A, L, U
    elif comm_rank == comm_size - 1:
        # Is the last process
        A, L, U = reduce_schur_bottomrightcorner(A, start_blockrow, partition_blocksize, blocksize)
        return A, L, U
    else: 
        # Is one of the central processes
        A, L, U = reduce_schur_central(A, start_blockrow, partition_blocksize, blocksize)
        return A, L, U
    
    
    
def reduce_schur_topleftcorner(A: np.ndarray, 
                               start_blockrow: int,
                               partition_blocksize: int,
                               blocksize: int):
    """ Top left corner of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : list
        starting blockrow of partition owned by the process
    partition_blocksize : list
        partition blocksize owned by the process
    blocksize : int
        size of a block
        
    Returns
    -------
    A : np.ndarray
        diagonal decomposition of A, work-in-place
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    """
    
    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    
    # Corner elimination downward
    for i_blockrow in range(start_blockrow+1, start_blockrow+partition_blocksize):
        im1_rowindice = (i_blockrow-1)*blocksize
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize


        A_inv_im1_im1 = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice])

        L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] =\
            A[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
        
        U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] =\
            A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
        
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] -=\
            L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]\
            @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
    
    return A, L, U



def reduce_schur_bottomrightcorner(A: np.ndarray, 
                                   start_blockrow: int,
                                   partition_blocksize: int,
                                   blocksize: int):
    """ Bottom right corner of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : list
        starting blockrow of partition owned by the process
    partition_blocksize : list
        partition blocksize owned by the process
    blocksize : int
        size of a block
        
    Returns
    -------
    A : np.ndarray
        diagonal decomposition of A, work-in-place
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    """
    
    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    # Corner elimination upward
    for i_blockrow in range(start_blockrow+partition_blocksize-2, start_blockrow-1, -1):
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize
        ip2_rowindice = (i_blockrow+2)*blocksize


        A_inv_ip1_ip1 = np.linalg.inv(A[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice])

        L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] =\
            A[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice] @ A_inv_ip1_ip1
            
        U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] =\
            A_inv_ip1_ip1 @ A[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]
            
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] -=\
            L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]\
            @ A[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]
    
    return A, L, U



def reduce_schur_central(A: np.ndarray, 
                         start_blockrow: int,
                         partition_blocksize: int,
                         blocksize: int):
    """ Central part of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : list
        starting blockrow of partition owned by the process
    partition_blocksize : list
        partition blocksize owned by the process
    blocksize : int
        size of a block
        
    Returns
    -------
    A : np.ndarray
        diagonal decomposition of A, work-in-place
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    """
    
    nblocks = A.shape[0] // blocksize
    
    L = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    # Center elimination downward
    for i_blockrow in range(start_blockrow+2, start_blockrow+partition_blocksize):
        im1_rowindice = (i_blockrow-1)*blocksize
        i_rowindice   = i_blockrow*blocksize
        ip1_rowindice = (i_blockrow+1)*blocksize

        top_rowindice   = start_blockrow*blocksize
        topp1_rowindice = (start_blockrow+1)*blocksize


        A_inv_im1_im1 = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice])

        L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] =\
            A[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
            
        L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] =\
            A[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice] @ A_inv_im1_im1
            
        U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] =\
            A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
            
        U[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice] =\
            A_inv_im1_im1 @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
        
        A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] -=\
            L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]\
            @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]
            
        A[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice] -=\
            L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice]\
            @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
            
        A[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] -=\
            L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]\
            @ A[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]
            
        A[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] -=\
            L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice]\
            @ A[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]

    return A, L, U
    


def aggregate_reduced_system(A: np.ndarray,
                             l_start_blockrow: list,
                             l_partitions_blocksizes: list,
                             blocksize: int):
    """ Aggregate the Schur reduced system on the root process.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    l_start_blockrow : list
        list of the starting blockrow of the partitions
    l_partitions_blocksizes : list
        list of the blocksize of the partitions
        
    Returns 
    -------
    A_schur : np.ndarray
        Schur aggregated system
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    nblocks_schur_system = (comm_size-1) * 2

    A_schur = np.zeros((nblocks_schur_system*blocksize, 
                        nblocks_schur_system*blocksize), dtype=A.dtype)
    
    process_start_blockrow = l_start_blockrow[comm_rank]
    process_partition_size = l_partitions_blocksizes[comm_rank]
    
    if comm_rank == 0:
        # A_schur will first take as first row the (local) reduced row of the
        # root process.
        start_rowindice = (process_start_blockrow+process_partition_size-1) * blocksize
        stop_rowindice  = start_rowindice + blocksize
        
        start_colindice = (process_partition_size-1) * blocksize
        stop_colindice  = start_colindice + nblocks_schur_system * blocksize
        
        A_schur[0:blocksize, :] = A[start_rowindice:stop_rowindice,
                                    start_colindice:stop_colindice]     
        
        
        # Then, A_schur will aggregate the Schur complement rows of the centrals
        # processes. Each central process send 2 rows (4 disctincts blocks that 
        # have been locally aggregated by the sending process) to the root.
        for process_i in range(1, comm_size-1, 1):
            start_rowindice = blocksize + (process_i-1) * 2 * blocksize
            stop_rowindice  = start_rowindice + 2 * blocksize
            
            start_colindice = 2 * (process_i-1) * blocksize
            stop_colindice  = start_colindice + 4 * blocksize
            
            A_schur[start_rowindice:stop_rowindice,\
                    start_colindice:stop_colindice] = comm.recv(source=process_i, tag=0)
                
                
        # Finally, A_schur will aggregate the Schur complement row of the last
        # process.
        start_rowindice = (nblocks_schur_system-1)*blocksize
        stop_rowindice  = start_rowindice + blocksize

        A_schur[start_rowindice:stop_rowindice, :] =\
            comm.recv(source=comm_size-1, tag=0)

        
    elif comm_rank == comm_size - 1:
        # Last process send his Schur complement row to the root process.
        start_rowindice = process_start_blockrow * blocksize
        stop_rowindice  = start_rowindice + blocksize
        
        stop_colindice  = (process_start_blockrow+1) * blocksize
        start_colindice = stop_colindice - nblocks_schur_system * blocksize
        
        comm.send(A[start_rowindice:stop_rowindice,\
                    start_colindice:stop_colindice], dest=0, tag=0)
    
    
    else: 
        # Center processes send their Schur complement rows to the root process.
        # First create the compact representation of the local reduced system.
        A_reduced_compact = np.zeros((2*blocksize, 4*blocksize), dtype=A.dtype)
    
    
        # Row indices of the top and bottom blocks of the compact representation
        start_top_rowindice = process_start_blockrow * blocksize
        stop_top_rowindice  = start_top_rowindice + blocksize
        
        start_bottom_rowindice = (process_start_blockrow + process_partition_size - 1) * blocksize
        stop_bottom_rowindice  = start_bottom_rowindice + blocksize
        
        
        # Col indices of the left and right blocks of the compact representation
        start_left_colindice = (process_start_blockrow-1) * blocksize
        stop_left_colindice  = start_left_colindice + 2*blocksize
        
        start_right_colindice = start_left_colindice + process_partition_size * blocksize
        stop_right_colindice  = start_right_colindice + 2*blocksize
        
        
        # Fill the compact representation
        A_reduced_compact[0:blocksize, 0:2*blocksize] =\
            A[start_top_rowindice:stop_top_rowindice, start_left_colindice:stop_left_colindice]

        A_reduced_compact[blocksize:2*blocksize, 0:2*blocksize] =\
            A[start_bottom_rowindice:stop_bottom_rowindice, start_left_colindice:stop_left_colindice]

        A_reduced_compact[0:blocksize, 2*blocksize:] =\
            A[start_top_rowindice:stop_top_rowindice, start_right_colindice:stop_right_colindice]

        A_reduced_compact[blocksize:2*blocksize, 2*blocksize:] =\
            A[start_bottom_rowindice:stop_bottom_rowindice, start_right_colindice:stop_right_colindice]


        comm.send(A_reduced_compact, dest=0, tag=0)
    
    return A_schur
               
     
"""      
def inverse_reduced_system(A_schur: np.ndarray,
                           blocksize: int):
    
    return G_schur



def sendback_inverted_reduced_system(G_schur: np.ndarray,
                                     l_start_blockrow: list,
                                     l_partitions_blocksizes: list,
                                     blocksize: int):
    
    return G """



def produce_schur(A: np.ndarray, 
                  L: np.ndarray,
                  U: np.ndarray,
                  G: np.ndarray,
                  l_start_blockrow: list,
                  l_partitions_blocksizes: list,
                  blocksize: int):
    """ Proceed to the parallel Schur production of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    G : np.ndarray
        inverse of A

    Returns
    -------
    None
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    if comm_rank == 0:
        # Is the first process
        produce_schur_topleftcorner(A, L, U, G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    elif comm_rank == comm_size - 1:
        # Is the last process
        produce_schur_bottomrightcorner(A, L, U, G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    else: 
        # Is one of the central processes
        produce_schur_central(A, L, U, G, l_start_blockrow, l_partitions_blocksizes, blocksize)



def produce_schur_topleftcorner(A: np.ndarray,
                                L: np.ndarray,
                                U: np.ndarray,
                                G: np.ndarray,
                                l_start_blockrow: list,
                                l_partitions_blocksizes: list,
                                blocksize: int):
    
    pass
            
            
            
def produce_schur_bottomrightcorner(A: np.ndarray,
                                    L: np.ndarray,
                                    U: np.ndarray,
                                    G: np.ndarray,
                                    l_start_blockrow: list,
                                    l_partitions_blocksizes: list,
                                    blocksize: int):
    
    pass



def produce_schur_central(A: np.ndarray,
                          L: np.ndarray,
                          U: np.ndarray,
                          G: np.ndarray,
                          l_start_blockrow: list,
                          l_partitions_blocksizes: list,
                          blocksize: int):
    
    pass



def aggregate_results(G: np.ndarray,
                      l_start_blockrow: list,
                      l_partitions_blocksizes: list,
                      blocksize: int):
    
    pass

           
                            
def psr_seqsolve_refactored(A: np.ndarray, 
                            blocksize: int):
    """ Selected inversion algorithm using the parallel Schur reduction 
    algorithm. The algorithm work in place and will overwrite the input matrix A.

    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    blocksize : int
        block matrice_size
        
    Returns 
    -------
    G : np.ndarray
        block tridiagonal selected inverse of A.
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    psr_u.check_input(A, blocksize, comm_size)
    
    l_start_blockrow, l_partitions_blocksizes = psr_u.divide_matrix(A, comm_size, blocksize)
    
    A, L, U = reduce_schur(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
    
    """ utils.vizu.vizualiseDenseMatrixFlat(A, "A process " + str(comm_rank))
    utils.vizu.vizualiseDenseMatrixFlat(L, "L process " + str(comm_rank))
    utils.vizu.vizualiseDenseMatrixFlat(U, "U process " + str(comm_rank)) """
    
    A_schur = aggregate_reduced_system(A, l_start_blockrow, l_partitions_blocksizes, blocksize)

    utils.vizu.vizualiseDenseMatrixFlat(A_schur, "A_schur process " + str(comm_rank))

    """ 
    G_schur = inverse_reduced_system(A_schur, blocksize)
    
    G = sendback_inverted_reduced_system(G_schur, l_start_blockrow, l_partitions_blocksizes, blocksize)
    
    produce_schur(A, L, U, G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    
    aggregate_results(G, l_start_blockrow, l_partitions_blocksizes, blocksize) """
    
    #return G