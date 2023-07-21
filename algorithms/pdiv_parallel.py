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

import utils.vizualisation as vizu

import numpy as np
import math
import time

from mpi4py import MPI




def divide_matrix(A, n_partitions, blocksize):
    """
        Compute the n_partitions segments that divide the matrix A.

        @param A:            matrix to divide
        @param n_partitions: number of partitions
        @param blocksize:    size of a block

        @return: l_start_blockrow, l_partitions_sizes
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



def allocate_memory_for_partitions(A, l_partitions_sizes, n_partitions, n_reduction_steps, blocksize):
    """
        Allocate the needed memory to store the current partition of the
        system at each steps of the assembly process.

        @param A:                  matrix to partition
        @param l_partitions_sizes: list of the size of each partition
        @param n_partitions:       number of partitions
        @param blocksize:          size of a block

        @return: K_local, B_local
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
    B_local = [np.zeros(blocksize, dtype=A.dtype) for i in range(B_number_of_blocks_to_allocate)]

    return K_local, B_local
                


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

    K_i = []
    B_i = []

    for i in range(len(l_start_blockrow)):
        start_index = l_start_blockrow[i]*blocksize
        stop_index  = start_index + l_partitions_sizes[i]*blocksize

        K_i.append(A[start_index:stop_index, start_index:stop_index])

        if i < len(l_start_blockrow)-1:
            B_i.append(A[stop_index-blocksize:stop_index, stop_index:stop_index+blocksize])

    return K_i, B_i



def send_partitions(K_i, K_local):
    """
        Send the partitions to the correct process.

        @param K_i:                list of the partitions
        @param K_local:            local partition
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



def recv_partitions(K_local, l_partitions_sizes, blocksize):
    """
        Receive the partitions from the master process.

        @param K_local:            local partition
        @param l_partitions_sizes: list of processes partition size
        @param blocksize:          size of a block
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_sizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=0, tag=0)



def send_bridges(B_i, B_local, n_partitions, n_reduction_steps):
    """
        Send the bridges to the correct process.

        @param B_i:               list of the bridges
        @param B_local:           local bridge matrix
        @param n_reduction_steps: number of reduction steps
        @param blocksize:         size of a block
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



def recv_bridges(B_local, n_partitions, n_reduction_steps):
    """
        Receive the bridges matrices from the master process.

        @param B_local:           local bridge matrix
        @param n_partitions:      number of partitions
        @param n_reduction_steps: number of reduction steps
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



def invert_partition(K_local, l_partitions_sizes, blocksize):
    """
        Invert the local partition.

        @param K_local:   local partition
        @param l_partitions_sizes: list of processes partition size
        @param blocksize: size of a block
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    start_index = 0
    stop_index  = l_partitions_sizes[comm_rank]*blocksize

    K_local[start_index:stop_index, start_index:stop_index] = np.linalg.inv(K_local[start_index:stop_index, start_index:stop_index])



def assemble_subpartitions(K_local, l_partitions_sizes, current_step, n_partitions, blocksize):
    """
        Assemble two subpartitions in a diagonal manner.

        @param K_local:            local partition
        @param l_partitions_sizes: list of processes partition size
        @param current_step:       current reduction step
        @param n_partitions:       number of partitions
        @param blocksize:          size of a block
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    current_step_partition_stride = int(math.pow(2, current_step))

    for process_recv in range(0, n_partitions, int(math.pow(2, current_step))):
        
        process_send = process_recv + current_step_partition_stride//2

        if comm_rank == process_recv:
            # If the process is receiving: we need to compute the start and stop index 
            # of the local (and already allocated) container where the subpartition will be stored
            start_block = l_partitions_sizes[process_recv]
            stop_block  = start_block + l_partitions_sizes[process_send]

            start_index = start_block*blocksize
            stop_index  = stop_block*blocksize
            
            K_local[start_index:stop_index, start_index:stop_index] = comm.recv(source=process_send, tag=2)
        
        elif comm_rank == process_send:
            # If the process is the sending process: it send its entire partition
            comm.send(K_local, dest=process_recv, tag=2)

        # Update the size of all the partitions that have been extended by
        # receiving a subpartition.
        l_partitions_sizes[process_recv] += l_partitions_sizes[process_send]



def compute_update_term(K_local, B_local, l_partitions_sizes, current_step, n_partitions, blocksize):
    """
        Compute the update term between the two assembled subpartitions.

        @param K_local:            local partition
        @param B_local:            local bridges matrices
        @param l_partitions_sizes: list of processes partition size
        @param current_step:       current reduction step
        @param n_partitions:       number of partitions
        @param blocksize:          size of a block

        @return: U
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    print("Process: ", comm_rank, " is computing the update term at step: ", current_step)

    current_step_partition_stride = int(math.pow(2, current_step))

    phi_1_size = l_partitions_sizes[comm_rank]
    phi_2_size = l_partitions_sizes[comm_rank+current_step_partition_stride//2]

    assembled_system_size = phi_1_size + phi_2_size

    U = np.zeros((assembled_system_size, assembled_system_size), dtype=K_local.dtype)

    #print(B_local[current_step-1])




    # After computing the update term, we update the size of the local partition
    # to match the summ of the two assembeld subpartitions
    l_partitions_sizes[comm_rank] = assembled_system_size
    print("Process: ", comm_rank, "l_partitions_sizes[comm_rank] = ", assembled_system_size)

    return U




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

    l_start_blockrow, l_partitions_sizes = divide_matrix(A, n_partitions, blocksize)
    K_local, B_local = allocate_memory_for_partitions(A, l_partitions_sizes, n_partitions, n_reduction_steps, blocksize)

    if comm_rank == 0:
        K_i, B_i = partition_subdomain(A, l_start_blockrow, l_partitions_sizes, blocksize)
        send_partitions(K_i, K_local)
        send_bridges(B_i, B_local, n_partitions, n_reduction_steps)
    else:
        recv_partitions(K_local, l_partitions_sizes, blocksize)
        recv_bridges(B_local, n_partitions, n_reduction_steps)
    
    # Inversion of the local partition
    invert_partition(K_local, l_partitions_sizes, blocksize)

    #vizu.vizualiseDenseMatrixFlat(K_local, f"Process: {comm_rank}, K_local")
    """ if comm_rank == 0:
            vizu.vizualiseDenseMatrixFlat(K_local, f"Process: {comm_rank}, K_local") """

    # Reduction steps
    for current_step in range(1, n_reduction_steps+1):
    #for current_step in range(1, 2):
        """ if comm_rank == 0:
            vizu.vizualiseDenseMatrixFlat(K_local, f"Process: {comm_rank}, K_local before assemble") """

        # Processes recv and send their subpartitions
        assemble_subpartitions(K_local, l_partitions_sizes, current_step, n_partitions, blocksize)

        # The active processes compute the update term and update their local partition
        for process_update in range(0, n_partitions, int(math.pow(2, current_step))):
            if comm_rank == process_update:
                U = compute_update_term(K_local, B_local, l_partitions_sizes, current_step, n_partitions, blocksize)
                #update_partition(K_local, U, current_step, n_reduction_steps, blocksize)

        """ if comm_rank == 0:
            vizu.vizualiseDenseMatrixFlat(K_local, f"Process: {comm_rank}, K_local after update") """
    



    return A

