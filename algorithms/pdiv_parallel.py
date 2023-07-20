"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

PDIV (Parallel Divide & Conquer) algorithm:
@reference: https://doi.org/10.1063/1.2748621
@reference: https://doi.org/10.1063/1.3624612

Pairwise algorithm:
@reference: https://doi.org/10.1007/978-3-319-78024-5_55

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import sys
sys.path.append('../')

import utils.vizualisation       as vizu

import numpy as np
import math
import time

from mpi4py import MPI



def allocate_memory_for_partitions(A, n_partitions, n_reduction_steps, blocksize):
    """
        Allocate the needed memory to store the current partition of the
        system at each steps of the assembly process.

        @param A:            matrix to partition
        @param n_partitions: number of partitions
        @param blocksize:    size of a block

        @return: K_local, B_local, l_start_blockrow, l_partitions_sizes
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

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


    # Compute the needed memory for the local K matrix and B bridges factors
    K_havent_been_allocated = True
    K_number_of_blocks_to_allocate = 0
    B_number_of_blocks_to_allocate = 0

    # Look at the steps that will be performed backwards: we want to allocate
    # the biggest container size that will be needed by the process.
    for current_step in range(n_reduction_steps, -1, -1):
        
        # Size of the system that will be assembled at this step (equal to the stride)
        current_step_partition_blocksize = int(math.pow(2, current_step))

        # Look at each processes that will perform computations at this step
        for process in range(0, n_partitions, current_step_partition_blocksize):
            if process == comm_rank:
                if K_havent_been_allocated:
                    # Loop through the blocks that will be assembled at this step
                    # to compute the needed memory for the local K matrix
                    for i in range(process, process+current_step_partition_blocksize):
                        K_number_of_blocks_to_allocate += l_partitions_sizes[i]
                    K_havent_been_allocated = False

                if current_step > 0:
                    B_number_of_blocks_to_allocate += 1

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, K_number_of_blocks_to_allocate*blocksize), dtype=A.dtype)

    # Allocate memory for the local B bridges factors
    B_local = np.zeros((B_number_of_blocks_to_allocate, blocksize), dtype=A.dtype)


    return K_local, B_local, l_start_blockrow, l_partitions_sizes
                


def partition_subdomain(A, l_start_blockrow, l_partitions_sizes, blocksize):
    """
        Partition the matrix A into K_i submatrices and B_i bridge matrices
        that stores the connecting elements between the submatrices.

        @param A:                  matrix to partition
        @param l_start_blockrow:   list of processes starting block row
        @param l_partitions_sizes: list of processes partition size
        @param blocksize:          size of a block

        @return: K_i, B_i
    """

    pass


def send_partitions(K_i, l_start_blockrow, l_partitions_sizes, blocksize):
    """
        Send the partitions to the correct process.

        @param K_i:                list of the partitions
        @param l_start_blockrow:   list of processes starting block row
        @param l_partitions_sizes: list of processes partition size
        @param blocksize:          size of a block
    """

    pass


def send_bridges(B_i, n_reduction_steps, blocksize):
    """
        Send the bridges to the correct process.

        @param B_i:               list of the bridges
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block
    """

    pass


def recv_partitions(K_local, l_start_blockrow, l_partitions_sizes, blocksize):
    """
        Receive the partitions from the master process.

        @param K_local:            local partition
        @param l_start_blockrow:   list of processes starting block row
        @param l_partitions_sizes: list of processes partition size
        @param blocksize:          size of a block
    """

    pass


def recv_bridges(B_local, n_reduction_steps, blocksize):
    """
        Receive the bridges matrices from the master process.

        @param B_local:           local bridge matrix
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block
    """

    pass


def invert_partition(K_local, blocksize):
    """
        Invert the local partition.

        @param K_local:   local partition
        @param blocksize: size of a block
    """

    pass


def assemble_subpartitions(K_local, current_step, n_reduction_steps, blocksize):
    """
        Assemble two subpartitions in a diagonal manner.

        @param K_local:           local partition
        @param current_step:      current reduction step
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block
    """

    pass


def compute_update_term(K_local, B_local, current_step, n_reduction_steps, blocksize):
    """
        Compute the update term between the two assembled subpartitions.

        @param K_local:           local partition
        @param B_local:           local bridges matrices
        @param current_step:      current reduction step
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block

        @return: U
    """

    pass


def update_partition(K_local, U, current_step, n_reduction_steps, blocksize):
    """
        Update the local partition with the update term.

        @param K_local:           local partition
        @param U:                 update term
        @param current_step:      current reduction step
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block
    """

    pass


def pdiv(A, blocksize):
    """
        Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
        
        @param A:         matrix to invert
        @param blocksize: size of a block

        @return: K_local (that is on process 0 the inverted matrix)
    """

    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if not math.log2(comm_size).is_integer():
        raise ValueError("The number of processes must be a power of 2.")


    # Preprocessing
    n_partitions      = comm_size
    n_reduction_steps = int(math.log2(n_partitions))

    K_local, B_local, l_start_blockrow, l_partitions_sizes = allocate_memory_for_partitions(A, n_partitions, n_reduction_steps, blocksize)

    """ if comm_rank == 0:
        K_i, B_i = partition_subdomain(A, l_start_blockrow, l_partitions_sizes, blocksize)
        send_partitions(K_i, l_start_blockrow, l_partitions_sizes, blocksize)
        send_bridges(B_i, n_reduction_steps, blocksize)
    else:
        recv_partitions(K_local, l_start_blockrow, l_partitions_sizes, blocksize)
        recv_bridges(B_local, n_reduction_steps, blocksize)

    
    # Inversion of the local partition
    invert_partition(K_local, blocksize)


    # Reduction steps
    for current_step in range(0, n_reduction_steps):
        assemble_subpartitions(K_local, current_step, n_reduction_steps, blocksize)
        U = compute_update_term(K_local, B_local, current_step, n_reduction_steps, blocksize)
        update_partition(K_local, U, current_step, n_reduction_steps, blocksize) """
    

    return A


#assembly_process = [i for i in range(0, n_partitions, int(math.pow(2, current_step)))]