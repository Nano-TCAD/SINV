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

import numpy as np
import math

from mpi4py import MPI



def check_input(A: np.ndarray, 
                blocksize: int, 
                comm_size: int):
    """ Check the validity of the inputs parameters.

    Parameters
    ----------
    A : numpy matrix
        matrix to invert
    blocksize : int
        size of a block
    comm_size : int
        number of processes

    Returns
    -------
    None

    Raises
    ------
    ValueError
        The number of processes must be a power of 2.
    ValueError
        The matrix size must be a multiple of the blocksize.
    ValueError
        The blocksize must be smaller than the matrix size.
    ValueError
        The blocksize must be greater than 0.
    ValueError
        The number of blocks must be greater than the number of processes.
    """

    if not math.log2(comm_size).is_integer():
        raise ValueError("The number of processes must be a power of 2.")
    
    if A.shape[0] % blocksize != 0:
        raise ValueError("The matrix size must be a multiple of the blocksize.")
    
    if blocksize > A.shape[0]:
        raise ValueError("The blocksize must be smaller than the matrix size.")
    
    if blocksize < 1:
        raise ValueError("The blocksize must be greater than 0.")
    
    nblocks = A.shape[0] // blocksize
    if nblocks < comm_size:
        raise ValueError("The number of blocks must be greater than the number of processes.")
    


def divide_matrix(A: np.ndarray, 
                  n_partitions: int, 
                  blocksize: int) -> [list, list]:
    """ Compute the n_partitions segments that divide the matrix A.

    Parameters
    ----------
    A : numpy matrix            
        matrix to divide
    n_partitions : int
        number of partitions
    blocksize : int
        size of a block

    Returns
    -------
    l_start_blockrow : list
        list of processes starting block index
    l_partitions_sizes : list
        list of processes partition size
    """

    nblocks = A.shape[0] // blocksize
    partition_blocksize = nblocks // n_partitions
    blocksize_of_first_partition = nblocks - partition_blocksize * (n_partitions-1)

    # Compute the starting block row and the partition size for each process
    l_start_blockrow   = []
    l_partitions_sizes = []

    for i in range(n_partitions):
        if i == 0:
            l_start_blockrow   = [0]
            l_partitions_sizes = [blocksize_of_first_partition]
        else:
            l_start_blockrow.append(l_start_blockrow[i-1] + l_partitions_sizes[i-1])
            l_partitions_sizes.append(partition_blocksize)

    return l_start_blockrow, l_partitions_sizes



def allocate_memory_for_partitions(A: np.ndarray, 
                                   l_partitions_sizes: list, 
                                   n_partitions: int, 
                                   n_reduction_steps: int, 
                                   blocksize: int) -> [np.ndarray, list, list]:
    """ Allocate the needed memory to store the current partition of the
    system at each steps of the assembly process.

    Parameters
    ----------
    A : numpy matrix
        matrix to partition
    l_partitions_sizes : list
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
                        K_number_of_blocks_to_allocate += l_partitions_sizes[i]
                    K_havent_been_allocated = False

                if current_step > 0:
                    B_number_of_blocks_to_allocate += 1

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, K_number_of_blocks_to_allocate*blocksize), dtype=A.dtype)

    # Allocate memory for the local B bridges factors
    Bu_local = [np.zeros(blocksize, dtype=A.dtype) for i in range(B_number_of_blocks_to_allocate)]
    Bl_local = [np.zeros(blocksize, dtype=A.dtype) for i in range(B_number_of_blocks_to_allocate)]

    return K_local, Bu_local, Bl_local
                


def partition_subdomain(A: np.ndarray, 
                        l_start_blockrow: list, 
                        l_partitions_sizes: list, 
                        blocksize: int) -> [np.ndarray, list, list]:
    """ Partition the matrix A into K_i submatrices, Bu_i (upper) and Bl_i 
    (lower) bridge matrices that stores the connecting elements between 
    the submatrices.
    
    Parameters
    ----------
    A : numpy matrix
        matrix to partition
    l_start_blockrow : list
        list of processes starting block index
    l_partitions_sizes : list
        list of processes partition size
    blocksize : int
        size of a block

    Returns
    -------
    K_i : list
        list of the partitions
    Bu_i : list
        list of the upper bridges matrices
    Bl_i : list
        list of the lower bridges matrices
    """

    K_i  = []
    Bu_i = []
    Bl_i = []

    for i in range(len(l_start_blockrow)):
        start_index = l_start_blockrow[i]*blocksize
        stop_index  = start_index + l_partitions_sizes[i]*blocksize

        K_i.append(A[start_index:stop_index, start_index:stop_index])

        if i < len(l_start_blockrow)-1:
            Bu_i.append(A[stop_index-blocksize:stop_index, stop_index:stop_index+blocksize])
            Bl_i.append(A[stop_index:stop_index+blocksize, stop_index-blocksize:stop_index])

    return K_i, Bu_i, Bl_i



def send_partitions(K_i: np.ndarray, 
                    K_local: np.ndarray):
    """ Send the partitions to the correct process.

    Parameters
    ----------
    K_i : list
        list of the partitions
    K_local : numpy matrix
        local partition

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD

    for process_i in range(len(K_i)):
        if process_i == 0:
            # Localy store the first partition in the local K matrix
            partition_size = K_i[process_i].shape[0]
            K_local[0:partition_size, 0:partition_size] = K_i[process_i]
        else:
            # Send the partition to the correct process
            comm.send(K_i[process_i], dest=process_i, tag=0)



def recv_partitions(K_local: np.ndarray, 
                    l_partitions_sizes: list, 
                    blocksize: int):
    """ Receive the partitions from the master process.
    
    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_sizes : list
        list of processes partition size
    blocksize : int
        size of a block

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_sizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=0, tag=0)



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



def invert_partition(K_local: np.ndarray, 
                     l_partitions_sizes: list, 
                     blocksize: int):
    """ Invert the local partition.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_sizes : list
        list of processes partition size
    blocksize : int
        size of a block

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_sizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index]\
        = np.linalg.inv(K_local[start_index:stop_index, start_index:stop_index])



def assemble_subpartitions(K_local: np.ndarray, 
                           l_partitions_sizes: list, 
                           active_process: int, 
                           current_step: int, 
                           blocksize: int):
    """ Assemble two subpartitions in a diagonal manner.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_sizes : list
        list of processes partition size
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
        start_block = l_partitions_sizes[active_process]
        stop_block  = start_block + l_partitions_sizes[sending_process]

        start_index = start_block*blocksize
        stop_index  = stop_block*blocksize
        
        K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=sending_process, tag=2)
    
    elif comm_rank == sending_process:
        # If the process is the sending process: it send its entire partition
        comm.send(K_local, dest=active_process, tag=2)

    # Update the size of all the partitions that have been extended by
    # receiving a subpartition.
    l_partitions_sizes[active_process] += l_partitions_sizes[sending_process]



def compute_J(K_local: np.ndarray, 
              Bu: np.ndarray, 
              Bl: np.ndarray, 
              phi_1_size: int, 
              blocksize: int) -> [np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """ Compute the J matrix.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    Bu : numpy matrix
        bridge matrix
    Bl : numpy matrix
        bridge matrix
    phi_1_size : int
        size of phi_1
    blocksize : int
        size of a block

    Returns
    -------
    J11 : numpy matrix
        J11 matrix
    J12 : numpy matrix
        J12 matrix
    J21 : numpy matrix
        J21 matrix
    J22 : numpy matrix
        J22 matrix
    """

    J = np.zeros((2*blocksize, 2*blocksize), dtype=K_local.dtype)

    J[0:blocksize, 0:blocksize] = np.identity(blocksize, dtype=K_local.dtype)
    J[0:blocksize, blocksize:2*blocksize] = K_local[phi_1_size:phi_1_size+blocksize , phi_1_size:phi_1_size+blocksize] @ Bl
    J[blocksize:2*blocksize, 0:blocksize] = K_local[phi_1_size-blocksize:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu
    J[blocksize:2*blocksize, blocksize:2*blocksize] = np.identity(blocksize, dtype=K_local.dtype)

    J = np.linalg.inv(J)

    J11 = J[0:blocksize, 0:blocksize]
    J12 = J[0:blocksize, blocksize:2*blocksize]
    J21 = J[blocksize:2*blocksize, 0:blocksize]
    J22 = J[blocksize:2*blocksize, blocksize:2*blocksize]


    return J11, J12, J21, J22



def compute_update_term(K_local: np.ndarray, 
                        Bu_local: np.ndarray, 
                        Bl_local: np.ndarray, 
                        l_partitions_sizes: list, 
                        active_process: int, 
                        current_step: int, 
                        blocksize: int) -> np.ndarray:
    """ Compute the update term between the two assembled subpartitions.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    Bu_local : numpy matrix
        local uppers bridges matrices
    Bl_local : numpy matrix
        local lower bridges matrices
    l_partitions_sizes : list
        list of processes partition size
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

    phi_2_blocksize = l_partitions_sizes[active_process + current_step_partition_stride//2]
    phi_1_blocksize = l_partitions_sizes[active_process]-phi_2_blocksize

    phi_1_size = phi_1_blocksize*blocksize
    phi_2_size = phi_2_blocksize*blocksize

    assembled_system_size = phi_1_size + phi_2_size

    U = np.zeros((assembled_system_size, assembled_system_size), dtype=K_local.dtype)

    Bu = Bu_local[current_step-1]
    Bl = Bl_local[current_step-1]

    J11, J12, J21, J22 = compute_J(K_local, Bu, Bl, phi_1_size, blocksize)

    U[0:phi_1_size, 0:phi_1_size] = -1 * K_local[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J12 @ K_local[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[0:phi_1_size, phi_1_size:assembled_system_size] = -1 * K_local[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J11 @ K_local[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]
    U[phi_1_size:assembled_system_size, 0:phi_1_size] = -1 * K_local[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J22 @ K_local[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[phi_1_size:assembled_system_size, phi_1_size:assembled_system_size] = -1 * K_local[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J21 @ K_local[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]

    return U



def update_partition(K_local: np.ndarray, 
                     U: np.ndarray):
    """ Update the local partition with the update term.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    U : numpy matrix
        update term

    Returns
    -------
    None
    """

    start_index = 0
    stop_index  = U.shape[0]

    K_local[start_index:stop_index, start_index:stop_index] = K_local[start_index:stop_index, start_index:stop_index] + U



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
        local partition

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated thourgh a series of reduction.

    This implementation agregates and update the inverted sub-partitions in a 
    divide and conquer manner. 
    """

    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    check_input(A, blocksize, comm_size)
    
    # Preprocessing
    n_partitions      = comm_size
    n_reduction_steps = int(math.log2(n_partitions))

    l_start_blockrow, l_partitions_sizes = divide_matrix(A, n_partitions, blocksize)
    K_local, Bu_local, Bl_local = allocate_memory_for_partitions(A, l_partitions_sizes, n_partitions, n_reduction_steps, blocksize)

    # Partitioning
    if comm_rank == 0:
        K_i, Bu_i, Bl_i = partition_subdomain(A, l_start_blockrow, l_partitions_sizes, blocksize)
        send_partitions(K_i, K_local)
        send_bridges(Bu_i, Bu_local, n_partitions, n_reduction_steps)
        send_bridges(Bl_i, Bl_local, n_partitions, n_reduction_steps)
    else:
        recv_partitions(K_local, l_partitions_sizes, blocksize)
        recv_bridges(Bu_local, n_partitions, n_reduction_steps)
        recv_bridges(Bl_local, n_partitions, n_reduction_steps)

    # Inversion of the local partition
    invert_partition(K_local, l_partitions_sizes, blocksize)

    # Reduction steps
    for current_step in range(1, n_reduction_steps+1):
        for active_process in range(0, n_partitions, int(math.pow(2, current_step))):

            # Processes recv and send their subpartitions
            assemble_subpartitions(K_local, l_partitions_sizes, active_process, current_step, blocksize)

            # The active processes compute the update term and update their local partition
            if comm_rank == active_process:
                U = compute_update_term(K_local, Bu_local, Bl_local, l_partitions_sizes, active_process, current_step, blocksize)
                update_partition(K_local, U)

    return K_local
