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



def pdiv_localmap(K_local: np.ndarray,
                  l_upperbridges: np.ndarray,
                  l_lowerbridges: np.ndarray, 
                  blocksize: int) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
        
    Parameters
    ----------
    K_local : numpy matrix
        local partition of the matrix to invert
    l_upperbridges : numpy matrix
        list of the upper bridges of the entire matrix
    l_lowerbridges : numpy matrix
        list of the lower bridges of the entire matrix
    blocksize : int
        size of a block

    Returns
    -------
    K_local : numpy matrix
        inverted local partition of the matrix
    l_upperbridges : numpy matrix
        updated lower bridges (return the entire list of bridges but only the
        local bridges are updated)
    l_lowerbridges : numpy matrix
        updated upper bridges (return the entire list of bridges but only the
        local bridges are updated)

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated in parallel using the 
    Sherman-Morrison formula.

    This implementation perform local update of the distributed partition. Hence
    the inverted system is scattered across the processes.
    
    Limitations:
    - The number of processes must be a power of 2.
    """
    
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    invert_partition(K_local)

    l_M = initialize_matrixmaps(blocksize)
    l_C = initialize_crossmaps(blocksize)

    n_reduction_steps = int(math.log2(comm_size))
    for current_step in range(1, n_reduction_steps + 1):
        update_maps(l_M, l_C, K_local, l_upperbridges, l_lowerbridges, current_step, blocksize)


    return K_local, l_upperbridges, l_lowerbridges

    
def invert_partition(K_local: np.ndarray) -> np.ndarray:
    """
    """
    
    K_local = np.linalg.inv(K_local)


def initialize_matrixmaps(blocksize: int) -> list[np.ndarray]:
    """
    """

    l_M = []

    for i in range(12):
        if i == 1 or i == 4 or i == 5 or i == 8:
            l_M.append(np.identity(blocksize))
        else:
            l_M.append(np.zeros((blocksize, blocksize)))
            
    return l_M


def initialize_crossmaps(blocksize: int) -> list[np.ndarray]:
    """
    """

    l_C = [np.zeros((blocksize, blocksize)) for i in range(12)]
    
    return l_C


def update_maps(l_M: np.ndarray,
                l_C: np.ndarray,
                K_local: np.ndarray,
                l_upperbridges: np.ndarray,
                l_lowerbridges: np.ndarray,
                current_step: int,
                blocksize: int) -> [np.ndarray, np.ndarray]:
    """
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    process_stride = int(math.pow(2, current_step))
    
    for active_process in range(0, comm_size, process_stride):
        starting_process = active_process
        ending_process   = active_process + process_stride - 1
        middle_process = get_middle_process(starting_process, ending_process)

        l_U = get_U(K_local, l_M, blocksize)
        
        """ Bu_mid = l_upperbridges[middle_process]
        Bl_mid = l_lowerbridges[middle_process]
        J   = get_J(l_U, Bu_mid, Bl_mid, blocksize) """

        if comm_rank == 0:        
            print("current_step: ", current_step, " active_process: ", active_process, " starting_process: ", starting_process, " ending_process: ", ending_process, " middle_process: ", middle_process)


def get_U(K_local: np.ndarray,
          l_M: np.ndarray,
          blocksize: int) -> list[np.ndarray]:
    """
    """
    
    UUR = np.zeros((blocksize, blocksize)) 
    ULL = np.zeros((blocksize, blocksize))
    ULR = np.zeros((blocksize, blocksize))
    DUL = np.zeros((blocksize, blocksize))
    DUR = np.zeros((blocksize, blocksize))
    DLL = np.zeros((blocksize, blocksize))
    
    
    
    
    return [UUR, ULL, ULR, DUL, DUR, DLL]
    
    

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
        
        
        # PDIV worflow
        pdiv_u.check_multiprocessing(comm_size)
        pdiv_u.check_input(A, blocksize, comm_size)
        
        l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)
        K_i, Bu_i, Bl_i = pdiv_u.partition_subdomain(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
        
        K_local = K_i[comm_rank]
        K_local, Bu_i, Bl_i = pdiv_localmap(K_local, Bu_i, Bl_i, blocksize)
        
        
        # Extract local reference solution
        start_localpart_rowindex = l_start_blockrow[comm_rank] * blocksize
        stop_localpart_rowindex  = start_localpart_rowindex + l_partitions_blocksizes[comm_rank] * blocksize
        A_local_slice_of_refsolution = A_refsol[start_localpart_rowindex:stop_localpart_rowindex, start_localpart_rowindex:stop_localpart_rowindex]
        
        
        #utils.vizu.compareDenseMatrix(A_local_slice_of_refsolution, f"A_local_slice_of_refsolution\n Process: {comm_rank} "  , A_pdiv_localmap, f"A_pdiv_localmap\n Process: {comm_rank} ")
        
        #assert np.allclose(A_local_slice_of_refsolution, A_pdiv_localmap)
        
        """ if comm_rank == 0:
            utils.vizu.vizualiseDenseMatrixFlat(A, "A") """
            
            