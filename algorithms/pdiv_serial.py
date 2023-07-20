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


import utils.vizualisation       as vizu

import numpy as np
import math
import time

from mpi4py import MPI




def partition_domain(A, n_partitions, blocksize):
    """
        Partition the matrix A into K_i submatrices. B_i stores the connecting blocks 
        between the K_i submatrices.
    """

    nblocks = A.shape[0] // blocksize
    nblock_per_partition = nblocks // n_partitions

    K_i = np.zeros((n_partitions, nblock_per_partition*blocksize, nblock_per_partition*blocksize), dtype=A.dtype)
    B_i = np.zeros((n_partitions-1, blocksize, blocksize), dtype=A.dtype)

    for i in range(n_partitions):
        starting_block_of_partition = i * nblock_per_partition
        ending_block_of_partition   = (i+1) * nblock_per_partition

        starting_row_of_partition = starting_block_of_partition * blocksize
        ending_row_of_partition   = ending_block_of_partition * blocksize

        K_i[i] = A[starting_row_of_partition:ending_row_of_partition, starting_row_of_partition:ending_row_of_partition]

        if i < n_partitions-1:
            B_i[i] = A[ending_row_of_partition-blocksize:ending_row_of_partition, ending_row_of_partition:ending_row_of_partition+blocksize]

    return K_i, B_i



def invert_diag_partitions(phi_i):
    """
        Invert the given partition.
    """

    phi_i_inv = np.linalg.inv(phi_i)

    return phi_i_inv



def assemble_subpartitions(phi_1, phi_2):
    """
        Assemble the two given subpartitions into a single matrix.
    """

    assembled_system_size = phi_1.shape[0] + phi_2.shape[0]

    PHI_12 = np.zeros((assembled_system_size, assembled_system_size), dtype=phi_1.dtype)

    PHI_12[0:phi_1.shape[0], 0:phi_1.shape[0]] = phi_1
    PHI_12[phi_1.shape[0]:assembled_system_size, phi_1.shape[0]:assembled_system_size] = phi_2

    return PHI_12


def compute_update_term(phi_1, phi_2, B, blocksize):
    """
        Compute the update term for the given subpartitions.
    """

    subpartition_size = phi_1.shape[0]

    assembled_system_size = 2*subpartition_size

    U = np.zeros((assembled_system_size, assembled_system_size), dtype=phi_1.dtype)

    J11, J12, J21, J22 = compute_J(phi_1, phi_2, B, blocksize)

    U[0:subpartition_size, 0:subpartition_size] = -1 * phi_1[:, subpartition_size-blocksize:subpartition_size] @ B @ J12 @ phi_1[:, subpartition_size-blocksize:subpartition_size].T
    U[0:subpartition_size, subpartition_size:assembled_system_size] = -1 * phi_1[:, subpartition_size-blocksize:subpartition_size] @ B @ J11 @ phi_2[:, 0:blocksize].T
    U[subpartition_size:assembled_system_size, 0:subpartition_size] = -1 * phi_2[:, 0:blocksize] @ B.T @ J22 @ phi_1[:, subpartition_size-blocksize:subpartition_size].T
    U[subpartition_size:assembled_system_size, subpartition_size:assembled_system_size] = -1 * phi_2[:, 0:blocksize] @ B.T @ J21 @ phi_2[:, 0:blocksize].T

    return U


def compute_J(phi_1, phi_2, B, blocksize):
    """
        Compute the J factors of the update matrix.
    """

    subpartition_size = phi_1.shape[0]

    J = np.zeros((2*blocksize, 2*blocksize), dtype=phi_1.dtype)

    J[0:blocksize, 0:blocksize] = np.identity(blocksize, dtype=phi_1.dtype)
    J[0:blocksize, blocksize:2*blocksize] = phi_2[0:blocksize, 0:blocksize] @ B.T
    J[blocksize:2*blocksize, 0:blocksize] = phi_1[subpartition_size-blocksize:subpartition_size, subpartition_size-blocksize:subpartition_size] @ B
    J[blocksize:2*blocksize, blocksize:2*blocksize] = np.identity(blocksize, dtype=phi_1.dtype)

    J = np.linalg.inv(J)

    J11 = J[0:blocksize, 0:blocksize]
    J12 = J[0:blocksize, blocksize:2*blocksize]
    J21 = J[blocksize:2*blocksize, 0:blocksize]
    J22 = J[blocksize:2*blocksize, blocksize:2*blocksize]

    return J11, J12, J21, J22




def update_partition(PHI, U):
    
    return PHI + U



def pdiv(A, blocksize):
    """
        Serial reference implementation for the PDIV/Pairwise selected inverse
        algorithm.
    """

    nblocks = A.shape[0] // blocksize
    G = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A.dtype)

    n_partitions = 2


    # 1. Partition the problem
    K_i, B_i = partition_domain(A, n_partitions, blocksize)

    # 2. Invert the subpartitions
    for i in range(n_partitions):
        K_i[i] = invert_diag_partitions(K_i[i])

    # 3. Assemble the subpartitions
    PHI_12 = assemble_subpartitions(K_i[0], K_i[1])

    # 4. Compute the update term
    U = compute_update_term(K_i[0], K_i[1], B_i[0], blocksize)

    #vizu.compareDenseMatrix(PHI_12, "PHI_12", U, "U")

    # 5. Update the partition
    G = update_partition(PHI_12, U)

    return G

