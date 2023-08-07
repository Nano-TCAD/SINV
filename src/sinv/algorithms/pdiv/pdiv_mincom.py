"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

PDIV (P-Division) algorithm:
@reference: https://doi.org/10.1063/1.2748621
@reference: https://doi.org/10.1063/1.3624612

Pairwise algorithm:
@reference: https://doi.org/10.1007/978-3-319-78024-5_55

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.algorithms.pdiv import pdiv_utils as pdiv_u
from sinv import algorithms as alg
from sinv import utils

import numpy as np
import math

from mpi4py import MPI



def pdiv_mincom(A: np.ndarray, 
                blocksize: int) -> np.ndarray:
    """ Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
        
    Parameters
    ----------
    A : numpy matrix
        matrix to invert
    blocksize : int
        size of a block

    Returns
    -------
    K_local : numpy matrix
        inverted matrix aggregated on process 0

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated thourgh a series of reduction.

    This implementation tend to minimize the number of communication between the
    processes. Hence the inverted partitions are aggregated on the root process 
    before updated to the final inverse.
    """
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    pdiv_u.check_input(A, blocksize, comm_size)
    
    # Preprocessing
    n_partitions = comm_size
    
    l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, n_partitions, blocksize)
    K_local = allocate_memory_for_partitions(A, l_partitions_blocksizes, blocksize)
    Bu_i = []
    Bl_i = []
    
    # Partitioning
    if comm_rank == 0:
        K_i, Bu_i, Bl_i = pdiv_u.partition_subdomain(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
        pdiv_u.send_partitions(K_i, K_local)
    else:
        pdiv_u.recv_partitions(K_local, l_partitions_blocksizes, blocksize)
        
    # Inversion of the local partition
    pdiv_u.invert_partition(K_local, l_partitions_blocksizes, blocksize)
    
    # Aggregate the inverted partitions on process 0
    aggregate_inverted_partitions(K_local, l_start_blockrow, l_partitions_blocksizes, blocksize)
    
    # Update the partitions
    if comm_rank == 0:
        for partition_i in range(0, n_partitions-1, 1):
            U = compute_update_term(K_local, Bu_i, Bl_i, l_start_blockrow, l_partitions_blocksizes, partition_i, blocksize)
            pdiv_u.update_partition(K_local, U)
        
    return K_local



def allocate_memory_for_partitions(A: np.ndarray, 
                                   l_partitions_sizes: list, 
                                   blocksize: int) -> np.ndarray:
    """ Allocate the needed memory to store the current partition of the
    system at each steps of the assembly process.

    Parameters
    ----------
    A : numpy matrix
        matrix to partition
    l_partitions_sizes : list
        list of the size of each partition
    blocksize : int
        size of a block

    Returns
    -------
    K_local : numpy matrix
        local partition
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Compute the needed memory for the local K matrix and B bridges factors
    K_number_of_blocks_to_allocate = 0

    # Process 0 get allocated the size to contain the entire matrix.
    if comm_rank == 0:
        K_number_of_blocks_to_allocate = A.shape[0] // blocksize
    # Other processes only get allocated the size of their locale partition.
    else:
        K_number_of_blocks_to_allocate = l_partitions_sizes[comm_rank]

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, K_number_of_blocks_to_allocate*blocksize), dtype=A.dtype)

    return K_local



def aggregate_inverted_partitions(K_local: np.ndarray, 
                                  l_start_blockrow: list,
                                  l_partitions_blocksizes: list, 
                                  blocksize: int):
    """ Aggregate the inverted partitions on process 0.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_start_blockrow : list
        list of the starting blockrow of each partition
    l_partitions_blocksizes : list
        list of the size of each partition
    blocksize : int
        size of a block
        
    Returns
    -------
    None
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    if comm_rank == 0:
        # Will receive the inverted partitions from each process
        for i in range(1, comm_size, 1):
            start_blockrow = l_start_blockrow[i]
            stop_blockrow  = start_blockrow + l_partitions_blocksizes[i]
            
            start_rowindices = start_blockrow*blocksize
            stop_rowindices  = stop_blockrow*blocksize
            
            K_local[start_rowindices:stop_rowindices, start_rowindices:stop_rowindices] = comm.recv(source=i, tag=0)
    else:
        comm.send(K_local, dest=0, tag=0)
    


def compute_update_term(K_local: np.ndarray, 
                        Bu_i: list, 
                        Bl_i: list, 
                        l_start_blockrow: list,
                        l_partitions_blocksizes: list, 
                        partition_i: int, 
                        blocksize: int) -> np.ndarray:
    """ Compute the update term between the two assembled subpartitions.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    Bu_i : list
        local uppers bridges matrices
    Bl_i : list
        local lower bridges matrices
    l_start_blockrow : list
        list of processes starting block index
    l_partitions_blocksizes : list
        list of processes partition block size
    partition_i : int
        current partition to update
    blocksize : int
        size of a block

    Returns
    -------
    U : numpy matrix
        update term
    """

    phi_1_blocksize = l_start_blockrow[partition_i+1]
    phi_2_blocksize = l_partitions_blocksizes[partition_i+1]

    phi_1_size = phi_1_blocksize*blocksize
    phi_2_size = phi_2_blocksize*blocksize

    assembled_system_size = phi_1_size + phi_2_size

    U = np.zeros((assembled_system_size, assembled_system_size), dtype=K_local.dtype)

    Bu = Bu_i[partition_i]
    Bl = Bl_i[partition_i]

    J11, J12, J21, J22 = pdiv_u.compute_J(K_local, Bu, Bl, phi_1_size, blocksize)

    U[0:phi_1_size, 0:phi_1_size] = -1 * K_local[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J12 @ K_local[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[0:phi_1_size, phi_1_size:assembled_system_size] = -1 * K_local[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J11 @ K_local[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]
    U[phi_1_size:assembled_system_size, 0:phi_1_size] = -1 * K_local[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J22 @ K_local[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[phi_1_size:assembled_system_size, phi_1_size:assembled_system_size] = -1 * K_local[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J21 @ K_local[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]

    return U
    
