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
    A : numpy matrix
        matrix to invert
    blocksize : int
        size of a block

    Returns
    -------
    K_local : numpy matrix
        inverted local partition of the matrix

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated thourgh a series of reduction.

    This implementation perform local update of the distributed partition. Hence
    the inverted system is scattered across the processes.
    
    Limitations:
    - The number of processes must be a power of 2.
    """
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    pdiv_u.check_multiprocessing(comm_size)
    pdiv_u.check_input(A, blocksize, comm_size)
    
    l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)

    if comm_rank == 0:
        print("l_start_blockrow: ", l_start_blockrow)
        print("l_partitions_blocksizes: ", l_partitions_blocksizes)        
        
    start_blockrow = l_start_blockrow[comm_rank]
    partition_size = l_partitions_blocksizes[comm_rank]
    K_local, Bu_i, Bl_i = get_local_partition(A, start_blockrow, partition_size, blocksize)

    invert_partition(K_local)

    n_partitions      = comm_size
    n_reduction_steps = int(math.log2(n_partitions))
    
    for current_step in range(1, n_reduction_steps+1):
        processes_stride = int(math.pow(2, current_step))
        for active_process in range(0, n_partitions, processes_stride):
            starting_process = active_process
            ending_process   = starting_process + processes_stride - 1
            
            middle_process = get_middle_process(starting_process, ending_process)
                
            if comm_rank == 0:
                print("current_step: ", current_step, "starting_process: ", starting_process, " middle_process: ", middle_process, " ending_process: ", ending_process)


    return K_local



def get_local_partition(A: np.ndarray,
                        start_blockrow: int,
                        partition_size: int,
                        blocksize: int) -> np.ndarray:
    """
    
    """    
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Compute the needed memory for the local K matrix and B bridges factors
    K_number_of_blocks_to_allocate = 0

    # All processes only get allocated the size of their locale partition.
    K_number_of_blocks_to_allocate = partition_size

    # Allocate memory for the local K matrix
    K_local = np.zeros((K_number_of_blocks_to_allocate*blocksize, K_number_of_blocks_to_allocate*blocksize), dtype=A.dtype)

    start_rowindex = start_blockrow * blocksize
    stop_rowindex  = start_rowindex + partition_size * blocksize
    
    K_local = A[start_rowindex: stop_rowindex, start_rowindex: stop_rowindex]
    
    if comm_rank == 0:
        Bu_i = A[stop_rowindex-blocksize: stop_rowindex, stop_rowindex: stop_rowindex+blocksize]
        Bl_i = None
        
        return K_local, Bu_i, Bl_i
    
    elif comm_rank == comm_size - 1:
        Bu_i = None
        Bl_i = A[start_rowindex: start_rowindex+blocksize, start_rowindex-blocksize: start_rowindex]
        
        return K_local, Bu_i, Bl_i
    
    else:
        Bu_i = A[stop_rowindex-blocksize: stop_rowindex, stop_rowindex: stop_rowindex+blocksize]
        Bl_i = A[start_rowindex: start_rowindex+blocksize, start_rowindex-blocksize: start_rowindex]
        
        return K_local, Bu_i, Bl_i
    
    
    
def invert_partition(K_local: np.ndarray) -> np.ndarray:
    """
    
    """
    
    K_local = np.linalg.inv(K_local)



def get_middle_process(starting_process: int, 
                       ending_process: int) -> int:
    """
    
    """
    
    middle_process = starting_process - 1 + math.ceil((ending_process - starting_process) / 2) 
    
    return middle_process



if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    isComplex = True
    seed = 63

    matrice_size = 13
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        
        
        A_refsol = np.linalg.inv(A)
        A_pdiv_localmap = pdiv_localmap(A, blocksize)
        
        l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)
        
        start_localpart_rowindex = l_start_blockrow[comm_rank] * blocksize
        stop_localpart_rowindex  = start_localpart_rowindex + l_partitions_blocksizes[comm_rank] * blocksize
        
        A_local_slice_of_refsolution = A_refsol[start_localpart_rowindex:stop_localpart_rowindex, start_localpart_rowindex:stop_localpart_rowindex]
        
        #utils.vizu.compareDenseMatrix(A_local_slice_of_refsolution, f"A_local_slice_of_refsolution\n Process: {comm_rank} "  , A_pdiv_localmap, f"A_pdiv_localmap\n Process: {comm_rank} ")
        
        #assert np.allclose(A_local_slice_of_refsolution, A_pdiv_localmap)
        
        """ if comm_rank == 0:
            utils.vizu.vizualiseDenseMatrixFlat(A, "A") """
            
            