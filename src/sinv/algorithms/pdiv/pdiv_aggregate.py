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

import numpy as np
import math

from mpi4py import MPI



def pdiv_aggregate(A: np.ndarray, 
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

    This implementation agregates and update the inverted sub-partitions in a 
    divide and conquer manner. 
    
    Limitations:
    - The number of processes must be a power of 2.
    """

    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    pdiv_u.check_multiprocessing(comm_size)
    pdiv_u.check_input(A, blocksize, comm_size)
    
    # Preprocessing
    n_partitions      = comm_size
    n_reduction_steps = int(math.log2(n_partitions))

    l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, n_partitions, blocksize)
    K_local, Bu_local, Bl_local = allocate_memory_for_partitions(A, l_partitions_blocksizes, n_partitions, n_reduction_steps, blocksize)

    # Partitioning
    if comm_rank == 0:
        K_i, Bu_i, Bl_i = pdiv_u.partition_subdomain(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
        pdiv_u.send_partitions(K_i, K_local)
        send_bridges(Bu_i, Bu_local, n_partitions, n_reduction_steps)
        send_bridges(Bl_i, Bl_local, n_partitions, n_reduction_steps)
    else:
        pdiv_u.recv_partitions(K_local, l_partitions_blocksizes, blocksize)
        recv_bridges(Bu_local, n_partitions, n_reduction_steps)
        recv_bridges(Bl_local, n_partitions, n_reduction_steps)

    # Inversion of the local partition
    pdiv_u.invert_partition(K_local, l_partitions_blocksizes, blocksize)

    # Reduction steps
    for current_step in range(1, n_reduction_steps+1):
        for active_process in range(0, n_partitions, int(math.pow(2, current_step))):

            # Processes recv and send their subpartitions
            assemble_subpartitions(K_local, l_partitions_blocksizes, active_process, current_step, blocksize)

            # The active processes compute the update term and update their local partition
            if comm_rank == active_process:
                U = compute_update_term(K_local, Bu_local, Bl_local, l_partitions_blocksizes, active_process, current_step, blocksize)
                pdiv_u.update_partition(K_local, U)

    return K_local



def allocate_memory_for_partitions(A: np.ndarray, 
                                   l_partitions_blocksizes: list, 
                                   n_partitions: int, 
                                   n_reduction_steps: int, 
                                   blocksize: int) -> [np.ndarray, list, list]:
    """ Allocate the needed memory to store the current partition of the
    system at each steps of the assembly process.

    Parameters
    ----------
    A : numpy matrix
        matrix to partition
    l_partitions_blocksizes : list
        list of the size of each partition
    n_partitions : int
        number of partitions
    n_reduction_steps : int
        number of reduction steps
    blocksize : int
        size of a block

    Returns
    -------
    K_local : numpy matrix
        local partition
    Bu_local : list
        list of the local upper bridges matrices
    Bl_local : list
        list of the local lower bridges matrices
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Compute the needed memory for the local K matrix and B bridges factors
    K_havent_been_allocated = True
    K_number_of_blocks_to_allocate = 0
    B_number_of_blocks_to_allocate = 0

    # Look at the steps that will be performed backwards: we want to allocate
    # the biggest container size that will be needed by the process.
    for current_step in range(n_reduction_steps, -1, -1):
        
        # Size of the system that will be assembled at this step (equal to the stride)
        current_step_partition_stride = int(math.pow(2, current_step))

        # Look at each processes that will perform computations at this step
        for process_i in range(0, n_partitions, current_step_partition_stride):
            if process_i == comm_rank:
                if K_havent_been_allocated:
                    # Loop through the blocks that will be assembled at this step
                    # to compute the needed memory for the local K matrix
                    for i in range(process_i, process_i+current_step_partition_stride):
                        K_number_of_blocks_to_allocate += l_partitions_blocksizes[i]
                    K_havent_been_allocated = False

                if current_step > 0:
                    B_number_of_blocks_to_allocate += 1

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, K_number_of_blocks_to_allocate*blocksize), dtype=A.dtype)

    # Allocate memory for the local B bridges factors
    Bu_local = [np.zeros(blocksize, dtype=A.dtype) for i in range(B_number_of_blocks_to_allocate)]
    Bl_local = [np.zeros(blocksize, dtype=A.dtype) for i in range(B_number_of_blocks_to_allocate)]

    return K_local, Bu_local, Bl_local
                


def send_bridges(B_i: list, 
                 B_local: list, 
                 n_partitions: int, 
                 n_reduction_steps: int):
    """ Send the bridges to the correct process.

    Parameters
    ----------
    B_i : list
        list of the bridges
    B_local : list
        local bridge matrix
    n_partitions : int
        number of partitions
    n_reduction_steps : int
        number of reduction steps

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD

    local_bridge_index = 0

    for current_step in range(1, n_reduction_steps+1, 1):
        
        current_step_partition_stride = int(math.pow(2, current_step))

        for process_i in range(0, n_partitions, current_step_partition_stride):
            
            bridge_index = process_i+current_step_partition_stride//2-1

            if process_i == 0:
                B_local[local_bridge_index] = B_i[bridge_index]
                local_bridge_index += 1
            else:
                comm.send(B_i[bridge_index], dest=process_i, tag=1)



def recv_bridges(B_local: list, 
                 n_partitions: int, 
                 n_reduction_steps: int):
    """ Receive the bridges matrices from the master process.

    Parameters
    ----------
    B_local : list
        local bridge matrix
    n_partitions : int
        number of partitions
    n_reduction_steps : int
        number of reduction steps

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    local_bridge_index = 0

    for current_step in range(1, n_reduction_steps+1, 1):
        
        current_step_partition_stride = int(math.pow(2, current_step))

        for process_i in range(0, n_partitions, current_step_partition_stride):

            if process_i == comm_rank:
                B_local[local_bridge_index] = comm.recv(source=0, tag=1)
                local_bridge_index += 1



def assemble_subpartitions(K_local: np.ndarray, 
                           l_partitions_blocksizes: list, 
                           active_process: int, 
                           current_step: int, 
                           blocksize: int):
    """ Assemble two subpartitions in a diagonal manner.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_blocksizes : list
        list of processes partition block size
    active_process : int
        active process
    current_step : int
        current reduction step
    blocksize : int
        size of a block

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    current_step_partition_stride = int(math.pow(2, current_step))
    sending_process = active_process + current_step_partition_stride//2

    if comm_rank == active_process:
        # If the process is receiving: we need to compute the start and stop index 
        # of the local (and already allocated) container where the subpartition will be stored
        start_block = l_partitions_blocksizes[active_process]
        stop_block  = start_block + l_partitions_blocksizes[sending_process]

        start_index = start_block*blocksize
        stop_index  = stop_block*blocksize
        
        K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=sending_process, tag=2)
    
    elif comm_rank == sending_process:
        # If the process is the sending process: it send its entire partition
        comm.send(K_local, dest=active_process, tag=2)

    # Update the size of all the partitions that have been extended by
    # receiving a subpartition.
    l_partitions_blocksizes[active_process] += l_partitions_blocksizes[sending_process]



def compute_update_term(K_local: np.ndarray, 
                        Bu_local: np.ndarray, 
                        Bl_local: np.ndarray, 
                        l_partitions_blocksizes: list, 
                        active_process: int, 
                        current_step: int, 
                        blocksize: int) -> np.ndarray:
    """ Compute the update term between the two assembled subpartitions.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    Bu_local : list
        local uppers bridges matrices
    Bl_local : list
        local lower bridges matrices
    l_partitions_blocksizes : list
        list of processes partition block size
    active_process : int
        active process
    current_step : int
        current reduction step
    blocksize : int
        size of a block

    Returns
    -------
    U : numpy matrix
        update term
    """

    current_step_partition_stride = int(math.pow(2, current_step))

    phi_2_blocksize = l_partitions_blocksizes[active_process + current_step_partition_stride//2]
    phi_1_blocksize = l_partitions_blocksizes[active_process]-phi_2_blocksize

    phi_1_size = phi_1_blocksize*blocksize
    phi_2_size = phi_2_blocksize*blocksize

    Bu = Bu_local[current_step-1]
    Bl = Bl_local[current_step-1]
    
    U = pdiv_u.compute_full_update_term(K_local, Bu, Bl, phi_1_size, phi_2_size, blocksize)

    return U
