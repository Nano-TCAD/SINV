"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Contains the utility functions for the PDIV algorithm.

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import math

from mpi4py import MPI



def check_multiprocessing(comm_size: int):
    """ Check if the number of processes is a power of 2.

    Parameters
    ----------
    comm_size : int
        number of processes
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        The number of processes must be a power of 2.
    """
    
    if not math.log2(comm_size).is_integer():
        raise ValueError("The number of processes must be a power of 2.")



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
        The matrix size must be a multiple of the blocksize.
    ValueError
        The blocksize must be smaller than the matrix size.
    ValueError
        The blocksize must be greater than 0.
    ValueError
        The number of blocks must be greater than the number of processes.
    """
    
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
    l_partitions_blocksizes : list
        list of processes partition size
    """

    nblocks = A.shape[0] // blocksize
    partition_blocksize = nblocks // n_partitions
    blocksize_of_first_partition = nblocks - partition_blocksize * (n_partitions-1)

    # Compute the starting block row and the partition size for each process
    l_start_blockrow        = []
    l_partitions_blocksizes = []

    for i in range(n_partitions):
        if i == 0:
            l_start_blockrow        = [0]
            l_partitions_blocksizes = [blocksize_of_first_partition]
        else:
            l_start_blockrow.append(l_start_blockrow[i-1] + l_partitions_blocksizes[i-1])
            l_partitions_blocksizes.append(partition_blocksize)

    return l_start_blockrow, l_partitions_blocksizes



def partition_subdomain(A: np.ndarray, 
                        l_start_blockrow: list, 
                        l_partitions_blocksizes: list, 
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
    l_partitions_blocksizes : list
        list of processes partition block size
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
        stop_index  = start_index + l_partitions_blocksizes[i]*blocksize

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
                    l_partitions_blocksizes: list, 
                    blocksize: int):
    """ Receive the partitions from the master process.
    
    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_blocksizes : list
        list of processes partition block size
    blocksize : int
        size of a block

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_blocksizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=0, tag=0)



def invert_partition(K_local: np.ndarray, 
                     l_partitions_blocksizes: list, 
                     blocksize: int):
    """ Invert the local partition.

    Parameters
    ----------
    K_local : numpy matrix
        local partition
    l_partitions_blocksizes : list
        list of processes partition block size
    blocksize : int
        size of a block

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_blocksizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index]\
        = np.linalg.inv(K_local[start_index:stop_index, start_index:stop_index])



def compute_full_update_term(K: np.ndarray, 
                             Bu: np.ndarray, 
                             Bl: np.ndarray, 
                             phi_1_size: int,
                             phi_2_size: int,
                             blocksize: int) -> np.ndarray:
    """ Compute the full update term between the two partitions.

    Parameters
    ----------
    K : numpy matrix
        local partition to update
    Bu : numpy matrix
        upper bridge matrix
    Bl : numpy matrix
        lower bridge matrix
    phi_1_size : int
        size of the first partition (rowindex)
    phi_2_size : int
        size of the second partition (rowindex)
    blocksize : int
        size of a block

    Returns
    -------
    U : numpy matrix
        update term
    """
    
    assembled_system_size = phi_1_size + phi_2_size
    
    U = np.zeros((assembled_system_size, assembled_system_size), dtype=K.dtype)

    J11, J12, J21, J22 = compute_J(K, Bu, Bl, phi_1_size, blocksize)

    U[0:phi_1_size, 0:phi_1_size] = -1 * K[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J12 @ K[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[0:phi_1_size, phi_1_size:assembled_system_size] = -1 * K[0:phi_1_size, phi_1_size-blocksize:phi_1_size] @ Bu @ J11 @ K[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]
    U[phi_1_size:assembled_system_size, 0:phi_1_size] = -1 * K[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J22 @ K[phi_1_size-blocksize:phi_1_size, 0:phi_1_size]
    U[phi_1_size:assembled_system_size, phi_1_size:assembled_system_size] = -1 * K[phi_1_size:assembled_system_size, phi_1_size:phi_1_size+blocksize] @ Bl @ J21 @ K[phi_1_size:phi_1_size+blocksize, phi_1_size:assembled_system_size]

    return U
    
    
    
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

