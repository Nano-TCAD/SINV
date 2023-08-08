"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

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



def pdiv_localmap(A: np.ndarray, 
                  blocksize: int) -> np.ndarray:
    """ Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
        
    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated thourgh a series of reduction.

    This implementation perform local update of the distributed partition. Hence
    the inverted system is scattered across the processes.
    """
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    pdiv_u.check_input(A, blocksize, comm_size)
    
    l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)
    K_local = allocate_memory_for_partitions(A, l_partitions_blocksizes, blocksize)




    
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

    # All processes only get allocated the size of their locale partition.
    K_number_of_blocks_to_allocate = l_partitions_sizes[comm_rank]

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, A.shape[1]), dtype=A.dtype)

    return K_local
    
    
    


if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    isComplex = True
    seed = 63

    matrice_size = 128
    blocksize    = 8
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A_refsol = np.linalg.inv(A)
        A_pdiv_localmap = pdiv_localmap(A, blocksize)
        
        l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)
        
        start_localpart_rowindex = l_start_blockrow[comm_rank] * blocksize
        stop_localpart_rowindex  = start_localpart_rowindex + l_partitions_blocksizes[comm_rank] * blocksize
        
        A_local_slice_of_refsolution = A_refsol[start_localpart_rowindex:stop_localpart_rowindex, :]
        
        utils.vizu.compareDenseMatrix(A_local_slice_of_refsolution, f"A_local_slice_of_refsolution\n Process: {comm_rank} "  , A_pdiv_localmap, f"A_pdiv_localmap\n Process: {comm_rank} ")
        assert np.allclose(A_local_slice_of_refsolution, A_pdiv_localmap)