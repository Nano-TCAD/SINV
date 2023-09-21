"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import utils
from sinv import algorithms as alg
from sinv.algorithms.psr import psr_utils as psr_u

import numpy as np

from mpi4py import MPI



def psr_seqsolve(
    A: np.ndarray, 
    blocksize: int
) -> np.ndarray:
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
        
    Notes
    -----
    Limitations:
    - The numbers of blocks must be greater than 3 times the number of processes.
        The central processes need at least 3 block-rows to work.
    - The minimal number of processes is 3.
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    psr_u.check_input(A, blocksize, comm_size)
    
    l_start_blockrow, l_partitions_blocksizes = psr_u.divide_matrix(A, comm_size, blocksize)
    A, L, U = reduce_schur(A, l_start_blockrow, l_partitions_blocksizes, blocksize)

    
    G = np.zeros(A.shape, dtype=A.dtype)
    
    if comm_rank == 0:
        A_schur = aggregate_reduced_system(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
        G_schur = inverse_reduced_system(A_schur)
        sendback_inverted_reduced_system(G_schur, G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    else:
        send_reduced_system(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
        receiveback_inverted_reduced_system(G, l_start_blockrow, l_partitions_blocksizes, blocksize)

    
    produce_schur(A, L, U, G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    aggregate_results(G, l_start_blockrow, l_partitions_blocksizes, blocksize)
        
    
    return G



def reduce_schur(
    A: np.ndarray, 
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
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
    
    
    
def reduce_schur_topleftcorner(
    A: np.ndarray, 
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Top left corner of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : int
        starting blockrow of partition owned by the process
    partition_blocksize : int
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
    
    L = np.zeros(A.shape, dtype=A.dtype)
    U = np.zeros(A.shape, dtype=A.dtype)
    
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



def reduce_schur_bottomrightcorner(
    A: np.ndarray, 
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Bottom right corner of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : int
        starting blockrow of partition owned by the process
    partition_blocksize : int
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
    
    L = np.zeros(A.shape, dtype=A.dtype)
    U = np.zeros(A.shape, dtype=A.dtype)

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



def reduce_schur_central(
    A: np.ndarray, 
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Central part of the parallel Schur reduction of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray
        block tridiagonal matrix
    start_blockrow : int
        starting blockrow of partition owned by the process
    partition_blocksize : int
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
    
    L = np.zeros(A.shape, dtype=A.dtype)
    U = np.zeros(A.shape, dtype=A.dtype)

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
    


def aggregate_reduced_system(
    A: np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> np.ndarray:
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

    A_schur[start_rowindice:stop_rowindice, :] = comm.recv(source=comm_size-1, tag=0)
    
    
    return A_schur
               
               

def send_reduced_system(
    A: np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> None:
    """ Send the local part of the reduced system to the root process for 
    inversion.
    
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
    None
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    nblocks_schur_system = (comm_size-1) * 2
    
    process_start_blockrow = l_start_blockrow[comm_rank]
    process_partition_size = l_partitions_blocksizes[comm_rank]
    
    if comm_rank == comm_size - 1:
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
     
     
      
def inverse_reduced_system(
    A_schur: np.ndarray,
) -> np.ndarray:
    """ Compute the inverse of the reduced system. Any selected inversion 
    solvers may be plugged here.
    
    Parameters
    ----------
    A_schur : np.ndarray
        reduced system to invert
    blocksize : int
        size of the blocks of the reduced system
        
    Returns
    -------
    G_schur : np.ndarray
        inverse of the reduced system
    """
    
    G_schur = np.linalg.inv(A_schur)
    
    return G_schur



def sendback_inverted_reduced_system(
    G_schur: np.ndarray,
    G : np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> None:
    """ Send back part of the inverted reduced system to their respective
    original processes.
    
    Parameters
    ----------
    G_schur : np.ndarray
        inverse of the reduced system
    G : np.ndarray
        local partition of the full inverse of A. Part of it will be overwritten
        by the inverted reduced system.
    l_start_blockrow : list
        list of the starting blockrow indices of each process
    l_partitions_blocksizes : list
        list of the number of blocks of each process
    blocksize : int
        size of the blocks of the reduced system
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    nblocks_schur_system = (comm_size-1) * 2
    
    process_start_blockrow = l_start_blockrow[comm_rank]
    process_partition_size = l_partitions_blocksizes[comm_rank]
    
    
    # The local root process part of G will first take the first blocksize rows
    # of the G_schur matrix.
    start_rowindice = (process_start_blockrow+process_partition_size-1) * blocksize
    stop_rowindice  = start_rowindice + blocksize
    
    start_colindice = (process_partition_size-1) * blocksize
    stop_colindice  = start_colindice + nblocks_schur_system * blocksize
    
    G[start_rowindice:stop_rowindice,\
      start_colindice:stop_colindice] = G_schur[0:blocksize, :]  
    
    
    # Then, process 0 send back to central processes their respective part of
    # the G_schur matrix.
    for process_i in range(1, comm_size-1, 1):
        start_rowindice = blocksize + (process_i-1) * 2 * blocksize
        stop_rowindice  = start_rowindice + 2 * blocksize
        
        start_colindice = 2 * (process_i-1) * blocksize
        stop_colindice  = start_colindice + 4 * blocksize
        
        comm.send(G_schur[start_rowindice:stop_rowindice,\
                          start_colindice:stop_colindice], dest=process_i, tag=0)
            
            
    # Finally, process 0 will send the last row of the G_schur matrix to the
    # last process.
    start_rowindice = (nblocks_schur_system-1)*blocksize
    stop_rowindice  = start_rowindice + blocksize
    
    comm.send(G_schur[start_rowindice:stop_rowindice, :], dest=comm_size-1, tag=0)



def receiveback_inverted_reduced_system(
    G : np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> None:
    """ Processes receive their respective part of the inverted reduced system
    and store it in the suited part of the G (full inverse) matrix.
    
    Parameters
    ----------
    G : np.ndarray
        local partition of the full inverse of A. Part of it will be overwritten
        by the inverted reduced system.
    l_start_blockrow : list
        list of the starting blockrow indices of each process
    l_partitions_blocksizes : list
        list of the number of blocks of each process
    blocksize : int
        size of the blocks of the reduced system
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    nblocks_schur_system = (comm_size-1) * 2
    
    process_start_blockrow = l_start_blockrow[comm_rank]
    process_partition_size = l_partitions_blocksizes[comm_rank]
    
    if comm_rank == comm_size-1:
        # Last process receives the last row of the G_schur matrix.
        start_rowindice = process_start_blockrow * blocksize
        stop_rowindice  = start_rowindice + blocksize
        
        stop_colindice  = (process_start_blockrow+1) * blocksize
        start_colindice = stop_colindice - nblocks_schur_system * blocksize
        
        G[start_rowindice:stop_rowindice,\
          start_colindice:stop_colindice] = comm.recv(source=0, tag=0)

    else:
        # Center processes receive their respective parts of the G_schur matrix.
        # First create the compact representation of the local G matrix for the
        # receiving process.
        G_reduced_compact = np.zeros((2*blocksize, 4*blocksize), dtype=G.dtype)
    
        G_reduced_compact = comm.recv(source=0, tag=0)
    
    
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
        
        
        # Match back the compact representation to the local part of the full 
        # inverse.
        G[start_top_rowindice:stop_top_rowindice,\
          start_left_colindice:stop_left_colindice] = G_reduced_compact[0:blocksize, 0:2*blocksize]

        G[start_bottom_rowindice:stop_bottom_rowindice,\
          start_left_colindice:stop_left_colindice] = G_reduced_compact[blocksize:2*blocksize, 0:2*blocksize]

        G[start_top_rowindice:stop_top_rowindice,\
          start_right_colindice:stop_right_colindice] = G_reduced_compact[0:blocksize, 2*blocksize:]

        G[start_bottom_rowindice:stop_bottom_rowindice, 
          start_right_colindice:stop_right_colindice] = G_reduced_compact[blocksize:2*blocksize, 2*blocksize:]
    


def produce_schur(
    A: np.ndarray, 
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> None:
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
    
    start_blockrow = l_start_blockrow[comm_rank]
    partition_blocksize = l_partitions_blocksizes[comm_rank]
    
    if comm_rank == 0:
        # Is the first process
        produce_schur_topleftcorner(A, L, U, G, start_blockrow, partition_blocksize, blocksize)
    elif comm_rank == comm_size - 1:
        # Is the last process
        produce_schur_bottomrightcorner(A, L, U, G, start_blockrow, partition_blocksize, blocksize)
    else: 
        # Is one of the central processes
        produce_schur_central(A, L, U, G, start_blockrow, partition_blocksize, blocksize)



def produce_schur_topleftcorner(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> None:
    """ Produce the upper left part of the full inverse.
    
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
    start_blockrow : int
        index of the first blockrow to be produced
    partition_blocksize : int
        number of blockrows to be produced by this process
    blocksize : int
        size of the blocks of the matrix A

    Returns
    -------
    None    
    """
    
    top_blockrow    = start_blockrow
    bottom_blockrow = start_blockrow + partition_blocksize
    
    # Upper left corner produce upwards
    for i in range(bottom_blockrow-1, top_blockrow-1, -1):
        im1_rowindice = (i-1)*blocksize
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize

        G[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] = -G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] @ L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]
        G[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] = -U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
        G[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice] = np.linalg.inv(A[im1_rowindice:i_rowindice, im1_rowindice:i_rowindice]) - U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]



def produce_schur_bottomrightcorner(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> None:
    """ Produce the lower right part of the full inverse.
    
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
    start_blockrow : int
        index of the first blockrow to be produced
    partition_blocksize : int
        number of blockrows to be produced by this process
    blocksize : int
        size of the blocks of the matrix A

    Returns
    -------
    None    
    """
    
    top_blockrow    = start_blockrow
    bottom_blockrow = start_blockrow + partition_blocksize
    
    # Lower left corner produce downwards
    for i in range(top_blockrow, bottom_blockrow-1):
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]   = -G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] @ L[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]
        G[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]   = -U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
        G[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice] = np.linalg.inv(A[ip1_rowindice:ip2_rowindice, ip1_rowindice:ip2_rowindice]) - U[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] @ G[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]



def produce_schur_central(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    start_blockrow: int,
    partition_blocksize: int,
    blocksize: int
) -> None:
    """ Produce one of the central parts of the full inverse.
    
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
    start_blockrow : int
        index of the first blockrow to be produced
    partition_blocksize : int
        number of blockrows to be produced by this process
    blocksize : int
        size of the blocks of the matrix A

    Returns
    -------
    None    
    """
    
    top_blockrow    = start_blockrow
    bottom_blockrow = start_blockrow + partition_blocksize 
    
    # Center processes produce upwards
    top_rowindice   = top_blockrow*blocksize
    topp1_rowindice = (top_blockrow+1)*blocksize
    topp2_rowindice = (top_blockrow+2)*blocksize
    topp3_rowindice = (top_blockrow+3)*blocksize

    botm1_rowindice = (bottom_blockrow-2)*blocksize
    bot_rowindice   = (bottom_blockrow-1)*blocksize
    botp1_rowindice = bottom_blockrow*blocksize


    G[bot_rowindice:botp1_rowindice, botm1_rowindice:bot_rowindice] =\
        -1 * G[bot_rowindice:botp1_rowindice, top_rowindice:topp1_rowindice]\
            @ L[top_rowindice:topp1_rowindice, botm1_rowindice:bot_rowindice]\
                - G[bot_rowindice:botp1_rowindice, bot_rowindice:botp1_rowindice]\
                    @ L[bot_rowindice:botp1_rowindice, botm1_rowindice:bot_rowindice]
    
    G[botm1_rowindice:bot_rowindice, bot_rowindice:botp1_rowindice] =\
        -1 * U[botm1_rowindice:bot_rowindice, bot_rowindice:botp1_rowindice]\
            @ G[bot_rowindice:botp1_rowindice, bot_rowindice:botp1_rowindice]\
                - U[botm1_rowindice:bot_rowindice, top_rowindice:topp1_rowindice]\
                    @ G[top_rowindice:topp1_rowindice, bot_rowindice:botp1_rowindice]


    for i in range(bottom_blockrow-2, top_blockrow, -1):
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice] =\
            -1 * G[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice]\
                @ L[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice]\
                    - G[top_rowindice:topp1_rowindice, ip1_rowindice:ip2_rowindice]\
                        @ L[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice]
                        
        G[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice] =\
            -1 * U[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]\
                @ G[ip1_rowindice:ip2_rowindice, top_rowindice:topp1_rowindice]\
                    - U[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice]\
                        @ G[top_rowindice:topp1_rowindice, top_rowindice:topp1_rowindice]


    for i in range(bottom_blockrow-2, top_blockrow+1, -1):
        im1_rowindice = (i-1)*blocksize
        i_rowindice   = i*blocksize
        ip1_rowindice = (i+1)*blocksize
        ip2_rowindice = (i+2)*blocksize

        G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice] =\
            np.linalg.inv(A[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice])\
                - U[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice]\
                    @ G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice]\
                        - U[i_rowindice:ip1_rowindice, ip1_rowindice:ip2_rowindice]\
                            @ G[ip1_rowindice:ip2_rowindice, i_rowindice:ip1_rowindice] 
                            
        G[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice] =\
            -1 * U[im1_rowindice:i_rowindice, top_rowindice:topp1_rowindice]\
                @ G[top_rowindice:topp1_rowindice, i_rowindice:ip1_rowindice]\
                    - U[im1_rowindice:i_rowindice, i_rowindice:ip1_rowindice]\
                        @ G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]
                   
        G[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice] =\
            -1 * G[i_rowindice:ip1_rowindice, top_rowindice:topp1_rowindice]\
                @ L[top_rowindice:topp1_rowindice, im1_rowindice:i_rowindice]\
                    - G[i_rowindice:ip1_rowindice, i_rowindice:ip1_rowindice]\
                        @ L[i_rowindice:ip1_rowindice, im1_rowindice:i_rowindice]


    G[topp1_rowindice:topp2_rowindice, topp1_rowindice:topp2_rowindice] =\
        np.linalg.inv(A[topp1_rowindice:topp2_rowindice, topp1_rowindice:topp2_rowindice])\
            - U[topp1_rowindice:topp2_rowindice, top_rowindice:topp1_rowindice]\
                @ G[top_rowindice:topp1_rowindice, topp1_rowindice:topp2_rowindice]\
                    - U[topp1_rowindice:topp2_rowindice, topp2_rowindice:topp3_rowindice]\
                        @ G[topp2_rowindice:topp3_rowindice, topp1_rowindice:topp2_rowindice]



def aggregate_results(
    G: np.ndarray,
    l_start_blockrow: list,
    l_partitions_blocksizes: list,
    blocksize: int
) -> None:
    """ Aggregate results from all processes into the global matrix G on rank 0.
    
    Parameters
    ----------
    G : np.ndarray
        inverse of A.
    l_start_blockrow : list
        list of start blockrows indices of each process.
    l_partitions_blocksizes : list
        list of block sizes of each process.
    blocksize : int
        size of the blocks
        
    Returns
    -------
    None
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        # Receive results from centrals and last processes
        for process_i in range(1, comm_size, 1):
            start_rowindex = l_start_blockrow[process_i] * blocksize
            stop_rowindex  = start_rowindex + l_partitions_blocksizes[process_i] * blocksize
            
            G[start_rowindex:stop_rowindex, :] = comm.recv(source=process_i, tag=0)
        
    else:
        start_rowindex = l_start_blockrow[comm_rank] * blocksize
        stop_rowindex  = start_rowindex + l_partitions_blocksizes[comm_rank] * blocksize
        
        comm.send(G[start_rowindex:stop_rowindex, :], dest=0, tag=0)
        
