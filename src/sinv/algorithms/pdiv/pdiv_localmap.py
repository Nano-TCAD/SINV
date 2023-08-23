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
    
    K_local = invert_partition(K_local)

    l_M = initialize_matrixmaps(blocksize)
    l_C = initialize_crossmaps(blocksize)

    n_reduction_steps = int(math.log2(comm_size))
    for current_step in range(1, n_reduction_steps + 1):
        update_maps(l_M, l_C, K_local, l_upperbridges, l_lowerbridges, current_step, blocksize)
    #produce_partition(K_local, l_M, l_C, l_upperbridges, l_lowerbridges, blocksize)

    return K_local, l_upperbridges, l_lowerbridges

    
def invert_partition(K_local: np.ndarray) -> np.ndarray:
    """ Invert the local partition of the matrix.
    
    Parameters
    ----------
    K_local : numpy matrix
        local partition of the matrix to invert
    
    Returns
    -------
    K_local : numpy matrix
        inverted local partition of the matrix
        
    Notes
    -----
    The inversion of the partition should be a full inversion.
    """
    
    return np.linalg.inv(K_local)


def initialize_matrixmaps(blocksize: int) -> list[np.ndarray]:
    """ Initialize the matrix maps. The matrix maps are used to update the
    local partition of the matrix without having to rupdate the entire matrix
    at each step.
    
    Parameters
    ----------
    blocksize : int
        size of a block
        
    Returns
    -------
    l_M : list of numpy matrix
        list of the matrix maps
        
    Notes
    -----
    The matrix maps deals with the update of the partition.
    """

    l_M = []

    for i in range(12):
        # Matrix maps numbers: 1, 4, 5, 7 are initialize to identity
        if i == 0 or i == 3 or i == 4 or i == 7:
            l_M.append(np.identity(blocksize))
        else:
            l_M.append(np.zeros((blocksize, blocksize)))
            
    return l_M


def initialize_crossmaps(blocksize: int) -> list[np.ndarray]:
    """ Initialize the cross maps. The cross maps are used to update the
    local partition of the matrix without having to rupdate the entire matrix
    at each step.
    
    Parameters
    ----------
    blocksize : int
        size of a block
        
    Returns
    -------
    l_C : list of numpy matrix
        list of the cross maps
        
    Notes
    -----
    The cross maps deals with the update of the bridges.
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
    """ Update the matrix maps and the cross maps.
    
    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_C : list of numpy matrix
        list of the cross maps
    K_local : numpy matrix
        local inverted partition of the matrix
    l_upperbridges : numpy matrix
        list of the upper bridges of the entire matrix
    l_lowerbridges : numpy matrix
        list of the lower bridges of the entire matrix
    current_step : int
        current reduction step
    blocksize : int
        size of a block
        
    Returns
    -------
    l_M : list of numpy matrix
        list of the updated matrix maps
    l_C : list of numpy matrix
        list of the updated cross maps
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    process_stride = int(math.pow(2, current_step))
    
    for active_process in range(0, comm_size, process_stride):
        starting_process = active_process
        ending_process   = active_process + process_stride - 1
        middle_process = get_middle_process(starting_process, ending_process)

        if comm_rank >= starting_process and comm_rank <= ending_process:
            # If the process is part of the current reduction step, proceed.
            l_U = get_U(K_local, l_M, starting_process, middle_process, ending_process, blocksize)
            
            Bu_mid = l_upperbridges[middle_process]
            Bl_mid = l_lowerbridges[middle_process]
            J = get_J(l_U, Bu_mid, Bl_mid, blocksize)

            #utils.vizu.vizualiseDenseMatrixFlat(J, f"J\n Process: {comm_rank}")
            
            update_matrixmap(l_M, l_U, Bu_mid, Bl_mid, J, middle_process, blocksize)
            
            
            
            #update_crossmap(l_C, l_M, Bu_mid, Bl_mid, J, middle_process, blocksize)
            

        if comm_rank == 0:        
            print("current_step: ", current_step, " active_process: ", active_process, " starting_process: ", starting_process, " ending_process: ", ending_process, " middle_process: ", middle_process)


def get_middle_process(starting_process: int, 
                       ending_process: int) -> int:
    """ Compute the index of middle process of the current reduction step.
    
    Parameters
    ----------
    starting_process : int
        starting process of the current reduction step
    ending_process : int
        ending process of the current reduction step
        
    Returns
    -------
    middle_process : int
        index of the middle process of the current reduction step
    """
    
    middle_process = starting_process - 1 + math.ceil((ending_process - starting_process) / 2) 
    
    return middle_process


def get_U(K_local: np.ndarray,
          l_M: np.ndarray,
          starting_process: int,
          middle_process: int,
          ending_process: int,
          blocksize: int) -> list[np.ndarray]:
    """ Compute the U factors. U factors are a collection of 6 blocks of the
    current partition to be combined.
    
    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    starting_process : int
        starting process of the current reduction step
    middle_process : int
        middle process of the current reduction step
    ending_process : int
        ending process of the current reduction step
    blocksize : int
        size of a block
        
    Returns
    -------
    l_U : list of numpy matrix
        list of the U factors
        
    Notes
    -----
    UUR: Upper Right block of the Upper partition
    ULL: Lower Left block of the Upper partition
    ULR: Lower Right block of the Upper partition
    DUL: Upper Left block of the Down partition
    DUR: Upper Right block of the Down partition
    DLL: Lower Left block of the Down partition
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    
    UUR = np.zeros((blocksize, blocksize), dtype=K_local.dtype) 
    ULL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    ULR = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DUL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DUR = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DLL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    
    
    # Deal with UUR and ULL
    if comm_rank == starting_process:
        UUR, ULL = produce_UUR_ULL(K_local, l_M, blocksize)
        
        for process in range(starting_process + 1, ending_process + 1):
            comm.send(UUR, dest=process, tag=0)
            comm.send(ULL, dest=process, tag=1)
    elif comm_rank > starting_process and comm_rank <= ending_process:
        UUR = comm.recv(source=starting_process, tag=0)
        ULL = comm.recv(source=starting_process, tag=1)


    # Deal with ULR
    if comm_rank == middle_process:
        ULR = produce_ULR(K_local, l_M, blocksize)
        
        for process in range(starting_process, ending_process + 1):
            if process != middle_process:
                comm.send(ULR, dest=process, tag=2)
    elif comm_rank >= starting_process and comm_rank <= ending_process:
        ULR = comm.recv(source=middle_process, tag=2)
    
    
    # Deal with DUL
    if comm_rank == middle_process+1:
        DUL = produce_DUL(K_local, l_M, blocksize)
        
        for process in range(starting_process, ending_process + 1):
            if process != middle_process+1:
                comm.send(DUL, dest=process, tag=3)
    elif comm_rank >= starting_process and comm_rank <= ending_process:
        DUL = comm.recv(source=middle_process+1, tag=3)
    
    
    # Deal with DUR and DLL
    if comm_rank == ending_process:
        DUR, DLL = produce_DUR_DLL(K_local, l_M, blocksize)
        
        for process in range(starting_process, ending_process):
            comm.send(DUR, dest=process, tag=4)
            comm.send(DLL, dest=process, tag=5)
    elif comm_rank >= starting_process and comm_rank < ending_process:
        DUR = comm.recv(source=ending_process, tag=4)
        DLL = comm.recv(source=ending_process, tag=5)
                
    
    return [UUR, ULL, ULR, DUL, DUR, DLL]
    
    
def produce_UUR_ULL(K_local, l_M, blocksize) -> [np.ndarray, np.ndarray]:
    """ Produce the UUR and ULL blocks.
    
    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block
        
    Returns
    -------
    UUR : numpy matrix
        Upper Right block of the Upper partition
    ULL : numpy matrix
        Lower Left block of the Upper partition
    """
    
    start_lastblock = K_local.shape[0] - blocksize
    end_lastblock   = K_local.shape[0]
    
    UUR = l_M[0] @ K_local[0:blocksize, start_lastblock:end_lastblock]\
            + l_M[1] @ K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock]
    
    ULL = K_local[start_lastblock:end_lastblock, 0:blocksize] @ l_M[4]\
            +  K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock] @ l_M[5]
    

    return UUR, ULL


def produce_ULR(K_local, l_M, blocksize) -> np.ndarray:
    """ Produce the ULR block.
    
    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block
        
    Returns
    -------
    ULR : numpy matrix
        Lower Right block of the Upper partition
    """
    
    start_lastblock = K_local.shape[0] - blocksize
    end_lastblock   = K_local.shape[0]
    
    ULR = K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock]\
            + K_local[start_lastblock:end_lastblock, 0:blocksize] @ l_M[8] @ K_local[0:blocksize, start_lastblock:end_lastblock]\
            + K_local[start_lastblock:end_lastblock, 0:blocksize] @ l_M[9] @ K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock]\
            + K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock] @ l_M[10] @ K_local[0:blocksize, start_lastblock:end_lastblock]\
            + K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock] @ l_M[11] @ K_local[start_lastblock:end_lastblock, start_lastblock:end_lastblock]
    
    #utils.vizu.compareDenseMatrix(K_local, "K_local", ULR, f"ULR\n Process: {comm_rank}")
    
    
    return ULR


def produce_DUL(K_local, l_M, blocksize) -> np.ndarray:
    """ Produce the DUL block.
    
    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block
        
    Returns
    -------
    DUL : numpy matrix
        Upper Left block of the Down partition
    """
    
    start_lastblock = K_local.shape[0] - blocksize
    end_lastblock   = K_local.shape[0]
    
    DUL = K_local[0:blocksize, 0:blocksize]\
            + K_local[0:blocksize, 0:blocksize] @ l_M[8] @ K_local[0:blocksize, 0:blocksize]\
            + K_local[0:blocksize, 0:blocksize] @ l_M[9] @ K_local[start_lastblock:end_lastblock, 0:blocksize]\
            + K_local[0:blocksize, start_lastblock:end_lastblock] @ l_M[10] @ K_local[0:blocksize, 0:blocksize]\
            + K_local[0:blocksize, start_lastblock:end_lastblock] @ l_M[11] @ K_local[start_lastblock:end_lastblock, 0:blocksize]
    
    #utils.vizu.compareDenseMatrix(K_local, "K_local", DUL, f"DUL\n Process: {comm_rank}")
    
    
    return DUL


def produce_DUR_DLL(K_local, l_M, blocksize) -> [np.ndarray, np.ndarray]:
    """ Produce the DUR and DLL blocks.
    
    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block
        
    Returns
    -------
    DUR : numpy matrix
        Upper Right block of the Down partition
    DLL : numpy matrix
        Lower Left block of the Down partition
    """
    
    start_lastblock = K_local.shape[0] - blocksize
    end_lastblock   = K_local.shape[0]
    
    DUR = K_local[0:blocksize, 0:blocksize] @ l_M[2]\
            +  K_local[0:blocksize, start_lastblock:end_lastblock] @ l_M[3]
            
    DLL = l_M[6] @ K_local[0:blocksize, 0:blocksize]\
            + l_M[7] @ K_local[start_lastblock:end_lastblock, 0:blocksize]
    
    #utils.vizu.compareDenseMatrix(K_local, "K_local", DUR, f"DUR\n Process: {comm_rank}")
    #utils.vizu.compareDenseMatrix(K_local, "K_local", DLL, f"DLL\n Process: {comm_rank}")
    
    
    return DUR, DLL
    
    
def get_J(l_U: list[np.ndarray],
          Bu_mid: np.ndarray,
          Bl_mid: np.ndarray,
          blocksize: np.ndarray) -> np.ndarray:
    """ Compute the J matrix.
    
    Parameters
    ----------
    l_U : list of numpy matrix
        list of the U factors
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    blocksize : int
        size of a block
                
    Returns
    -------
    J : numpy matrix
        J matrix
    """
    
    J = np.zeros((2*blocksize, 2*blocksize), dtype=Bu_mid.dtype)
    
    J[0:blocksize, 0:blocksize] = np.identity(blocksize, dtype=Bu_mid.dtype)
    J[0:blocksize, blocksize:2*blocksize] = - l_U[3] @ Bl_mid
    J[blocksize:2*blocksize, 0:blocksize] = - l_U[2] @ Bu_mid
    J[blocksize:2*blocksize, blocksize:2*blocksize] = np.identity(blocksize, dtype=Bu_mid.dtype)
    
    J = np.linalg.inv(J)
    
    
    return J


def update_matrixmap(l_M: list[np.ndarray], 
                     l_U: list[np.ndarray], 
                     Bu_mid, 
                     Bl_mid, 
                     J, 
                     middle_process, 
                     blocksize):
    """
    """
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    
    UUR = l_U[0]
    DUR = l_U[4]
    ULL = l_U[1]
    DLL = l_U[5]
    
    J11 = J[0:blocksize, 0:blocksize]
    J12 = J[0:blocksize, blocksize:2*blocksize]
    J21 = J[blocksize:2*blocksize, 0:blocksize]
    J22 = J[blocksize:2*blocksize, blocksize:2*blocksize]
    
    if comm_rank <= middle_process:
        update_matrixmap_upper(l_M, UUR, DUR, ULL, DLL, Bu_mid, Bl_mid, J11, J12, J22, blocksize)
    else:
        update_matrixmap_lower(l_M, UUR, DUR, ULL, DLL, Bu_mid, Bl_mid, J11, J21, J22, blocksize)
    
    
def update_matrixmap_upper(l_M: list[np.ndarray], 
                           UUR: np.ndarray, 
                           DUR: np.ndarray, 
                           ULL: np.ndarray, 
                           DLL: np.ndarray, 
                           Bu_mid: np.ndarray, 
                           Bl_mid: np.ndarray, 
                           J11: np.ndarray, 
                           J12: np.ndarray, 
                           J22: np.ndarray):
    """
    """
    
    l_M[0] += UUR @ Bu_mid @ J12 @ l_M[6]
    l_M[1] += UUR @ Bu_mid @ J12 @ l_M[7]
    
    l_M[2] = l_M[2] @ Bu_mid @ J11 @ DUR
    l_M[3] = l_M[3] @ Bu_mid @ J11 @ DUR
    
    l_M[4] += l_M[2] @ Bu_mid @ J12 @ ULL 
    l_M[5] += l_M[3] @ Bu_mid @ J12 @ ULL
    
    l_M[6] = DLL @ Bl_mid @ J22 @ l_M[6]
    l_M[7] = DLL @ Bl_mid @ J22 @ l_M[7]

    l_M[8] += l_M[2] @ Bu_mid @ J12 @ l_M[6]
    l_M[9] += l_M[2] @ Bu_mid @ J12 @ l_M[7]
    l_M[10] += l_M[3] @ Bu_mid @ J12 @ l_M[6]
    l_M[11] += l_M[3] @ Bu_mid @ J12 @ l_M[7]



def update_matrixmap_lower(l_M: list[np.ndarray], 
                           UUR: np.ndarray, 
                           DUR: np.ndarray, 
                           ULL: np.ndarray, 
                           DLL: np.ndarray, 
                           Bu_mid: np.ndarray, 
                           Bl_mid: np.ndarray, 
                           J11: np.ndarray, 
                           J21: np.ndarray, 
                           J22: np.ndarray):
    """
    """
   
    pass
    # CONTINUE HERE

    
    
def update_crossmap(l_C: list[np.ndarray],
                    l_M: list[np.ndarray],
                    Bu_mid: np.ndarray,
                    Bl_mid: np.ndarray,
                    J: np.ndarray, 
                    middle_process: int, 
                    blocksize: int):
    """
    """
    pass





if __name__ == '__main__':
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    isComplex = True
    seed = 63

    matrice_size = 26
    blocksize    = 2
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
            
            