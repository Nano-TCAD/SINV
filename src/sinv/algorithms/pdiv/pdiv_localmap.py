"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

PDIV (P-Division) algorithm:
@reference: https://doi.org/10.1063/1.2748621
@reference: https://doi.org/10.1063/1.3624612

Pairwise algorithm:
@reference: https://doi.org/10.1007/978-3-319-78024-5_55
@reference: https://doi.org/10.1002/cpe.4918

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from sinv.algorithms.pdiv import pdiv_utils as pdiv_u
from sinv import algorithms as alg
from sinv import utils

import numpy as np
import math

from mpi4py import MPI


def pdiv_localmap(
    K_local: np.ndarray,
    l_upperbridges: np.ndarray,
    l_lowerbridges: np.ndarray,
    blocksize: int,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
    This implementation handle inverse of general matrices (no symmetry is assumed).

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
    X_diagblk : numpy matrix
        diagonal entries of the inverted local partition
    X_upperblk : numpy matrix
        upper diagonal entries of the inverted local partition, the last block
        contains the upper bridge of the local partition. (execpt for the last
        process)
    X_lowerblk : numpy matrix
        lower diagonal entries of the inverted local partition, the last block
        contains the lower bridge of the local partition. (execpt for the last
        process)

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
    comm_size = comm.Get_size()

    K_local = invert_partition(K_local)

    l_M = initialize_matrixmaps(K_local, blocksize)
    l_M_ip1 = initialize_crossmaps(K_local, blocksize)
    l_C = initialize_crossmaps(K_local, blocksize)

    n_reduction_steps = int(math.log2(comm_size))
    for current_step in range(1, n_reduction_steps + 1):
        l_M, l_C = update_maps(
            l_M,
            l_M_ip1,
            l_C,
            K_local,
            l_upperbridges,
            l_lowerbridges,
            current_step,
            blocksize,
        )

    X_diagblk, X_upperblk, X_lowerblk = produce_partition(K_local, l_M, l_C, blocksize)

    return X_diagblk, X_upperblk, X_lowerblk


def invert_partition(K_local: np.ndarray) -> np.ndarray:
    """Invert the local partition of the matrix.

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
    The inversion of the partition should be a full inversion that produce a
    full dense inverse.
    """

    return np.linalg.inv(K_local)


def initialize_matrixmaps(K_local: np.ndarray, blocksize: int) -> list[np.ndarray]:
    """Initialize the matrix maps. The matrix maps are used to update the
    local partition of the matrix without having to rupdate the entire matrix
    at each step.

    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
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
        # Matrix maps numbers: 1, 4, 5, 8 are initialize to identity
        if i == 0 or i == 3 or i == 4 or i == 7:
            l_M.append(np.identity((blocksize), dtype=K_local.dtype))
        else:
            l_M.append(np.zeros((blocksize, blocksize), dtype=K_local.dtype))

    return l_M


def initialize_crossmaps(K_local: np.ndarray, blocksize: int) -> list[np.ndarray]:
    """Initialize the cross maps. The cross maps are used to update the
    local partition of the matrix without having to rupdate the entire matrix
    at each step.

    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
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

    l_C = [np.zeros((blocksize, blocksize), dtype=K_local.dtype) for i in range(12)]

    return l_C


def update_maps(
    l_M: np.ndarray,
    l_M_ip1: np.ndarray,
    l_C: np.ndarray,
    K_local: np.ndarray,
    l_upperbridges: np.ndarray,
    l_lowerbridges: np.ndarray,
    current_step: int,
    blocksize: int,
) -> [np.ndarray, np.ndarray]:
    """Update the matrix maps and the cross maps.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
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
        ending_process = active_process + process_stride - 1
        middle_process = get_middle_process(starting_process, ending_process)

        if comm_rank >= starting_process and comm_rank <= ending_process:
            # If the process is part of the current reduction step, proceed.
            l_U = get_U(
                K_local,
                l_M,
                starting_process,
                middle_process,
                ending_process,
                blocksize,
            )

            Bu_mid = l_upperbridges[middle_process]
            Bl_mid = l_lowerbridges[middle_process]
            J = get_J(l_U, Bu_mid, Bl_mid, blocksize)

            l_M_ip1 = get_nextprocess_matrixmap(
                l_M, l_M_ip1, starting_process, middle_process, ending_process
            )
            l_C = update_crossmap(
                l_C,
                l_M,
                l_M_ip1,
                Bu_mid,
                Bl_mid,
                J,
                middle_process,
                ending_process,
                blocksize,
            )
            l_M = update_matrixmap(
                l_M, l_U, Bu_mid, Bl_mid, J, middle_process, blocksize
            )

    return l_M, l_C


def get_middle_process(starting_process: int, ending_process: int) -> int:
    """Compute the index of middle process of the current reduction step.

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

    middle_process = (
        starting_process - 1 + math.ceil((ending_process - starting_process) / 2)
    )

    return middle_process


def get_U(
    K_local: np.ndarray,
    l_M: np.ndarray,
    starting_process: int,
    middle_process: int,
    ending_process: int,
    blocksize: int,
) -> list[np.ndarray]:
    """Compute the U factors. U factors are a collection of 6 corner matrices
    that will be needed on every process to update their local partition.

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

    UUR = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    ULL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    ULR = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DUL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DUR = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    DLL = np.zeros((blocksize, blocksize), dtype=K_local.dtype)

    # Produce corner blocks
    first_blockindex = 0
    last_blockindex = K_local.shape[0] // blocksize - 1

    if comm_rank == middle_process:
        UUR = produce_toprow_element(last_blockindex, K_local, l_M, blocksize)
        ULL = produce_leftcol_element(last_blockindex, K_local, l_M, blocksize)
        ULR = produce_matrix_elements(
            last_blockindex, last_blockindex, K_local, l_M, blocksize
        )

    if comm_rank == middle_process + 1:
        DUL = produce_matrix_elements(
            first_blockindex, first_blockindex, K_local, l_M, blocksize
        )
        DUR = produce_rightcol_element(first_blockindex, K_local, l_M, blocksize)
        DLL = produce_botrow_element(first_blockindex, K_local, l_M, blocksize)

    # Communicate corner blocks
    if comm_rank == middle_process:
        for process in range(starting_process, ending_process + 1):
            if process != middle_process:
                comm.send(UUR, dest=process, tag=0)
                comm.send(ULL, dest=process, tag=1)
                comm.send(ULR, dest=process, tag=2)
    else:
        UUR = comm.recv(source=middle_process, tag=0)
        ULL = comm.recv(source=middle_process, tag=1)
        ULR = comm.recv(source=middle_process, tag=2)

    if comm_rank == middle_process + 1:
        for process in range(starting_process, ending_process + 1):
            if process != middle_process + 1:
                comm.send(DUL, dest=process, tag=3)
                comm.send(DUR, dest=process, tag=4)
                comm.send(DLL, dest=process, tag=5)
    else:
        DUL = comm.recv(source=middle_process + 1, tag=3)
        DUR = comm.recv(source=middle_process + 1, tag=4)
        DLL = comm.recv(source=middle_process + 1, tag=5)

    return [UUR, ULL, ULR, DUL, DUR, DLL]


def get_J(
    l_U: list[np.ndarray], Bu_mid: np.ndarray, Bl_mid: np.ndarray, blocksize: np.ndarray
) -> np.ndarray:
    """Compute the J matrix. This is were the cost of the update part resides.
    Updating the partition is done by computing the inverse of the J matrix that
    is a 2x2 blocks matrix.

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

    J = np.zeros((2 * blocksize, 2 * blocksize), dtype=Bu_mid.dtype)

    J[0:blocksize, 0:blocksize] = np.identity(blocksize, dtype=Bu_mid.dtype)
    J[0:blocksize, blocksize : 2 * blocksize] = -l_U[3] @ Bl_mid
    J[blocksize : 2 * blocksize, 0:blocksize] = -l_U[2] @ Bu_mid
    J[blocksize : 2 * blocksize, blocksize : 2 * blocksize] = np.identity(
        blocksize, dtype=Bu_mid.dtype
    )

    J = np.linalg.inv(J)

    return J


def get_nextprocess_matrixmap(
    l_M: list[np.ndarray],
    l_M_ip1: list[np.ndarray],
    starting_process: int,
    middle_process: int,
    ending_process: int,
) -> list[np.ndarray]:
    """Receive the matrix maps of the next process to prepare the computation
    of the cross maps.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    starting_process : int
        starting process of the current reduction step
    middle_process : int
        middle process of the current reduction step
    ending_process : int
        ending process of the current reduction step

    Returns
    -------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # 1st process only receive.
    if comm_rank == starting_process:
        if comm_rank == middle_process:
            # In a not so special case, 1st process is also the middle process.
            l_M_ip1 = lower_or_middle_process_recv(l_M_ip1)
        else:
            l_M_ip1 = upperprocess_recv(l_M_ip1)
    # Last process only send.
    elif comm_rank == ending_process:
        send_to_lower_or_middle_process(l_M)

    # Other processes send to the process above and receive from lthe one below.
    else:
        if comm_rank < middle_process:
            send_to_upper_process(l_M)
            l_M_ip1 = upperprocess_recv(l_M_ip1)
        elif comm_rank == middle_process:
            send_to_upper_process(l_M)
            l_M_ip1 = lower_or_middle_process_recv(l_M_ip1)
        else:
            send_to_lower_or_middle_process(l_M)
            l_M_ip1 = lower_or_middle_process_recv(l_M_ip1)

    return l_M_ip1


def upperprocess_recv(l_M_ip1: list[np.ndarray]) -> list[np.ndarray]:
    """Upper process receive the matrix maps from the next process.

    Parameters
    ----------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process

    Returns
    -------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    l_M_ip1[2] = comm.recv(source=comm_rank + 1, tag=2)
    l_M_ip1[3] = comm.recv(source=comm_rank + 1, tag=3)
    l_M_ip1[6] = comm.recv(source=comm_rank + 1, tag=6)
    l_M_ip1[7] = comm.recv(source=comm_rank + 1, tag=7)

    return l_M_ip1


def lower_or_middle_process_recv(l_M_ip1: list[np.ndarray]) -> list[np.ndarray]:
    """Lower or middle process receive the matrix maps from the next process.

    Parameters
    ----------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process

    Returns
    -------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    l_M_ip1[0] = comm.recv(source=comm_rank + 1, tag=0)
    l_M_ip1[1] = comm.recv(source=comm_rank + 1, tag=1)
    l_M_ip1[4] = comm.recv(source=comm_rank + 1, tag=4)
    l_M_ip1[5] = comm.recv(source=comm_rank + 1, tag=5)

    return l_M_ip1


def send_to_upper_process(l_M_ip1: list[np.ndarray]) -> None:
    """Middle or upper process send the matrix maps to the previous process.

    Parameters
    ----------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    comm.send(l_M_ip1[2], dest=comm_rank - 1, tag=2)
    comm.send(l_M_ip1[3], dest=comm_rank - 1, tag=3)
    comm.send(l_M_ip1[6], dest=comm_rank - 1, tag=6)
    comm.send(l_M_ip1[7], dest=comm_rank - 1, tag=7)


def send_to_lower_or_middle_process(l_M_ip1: list[np.ndarray]) -> None:
    """Lower process send the matrix maps to the previous process.

    Parameters
    ----------
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    comm.send(l_M_ip1[0], dest=comm_rank - 1, tag=0)
    comm.send(l_M_ip1[1], dest=comm_rank - 1, tag=1)
    comm.send(l_M_ip1[4], dest=comm_rank - 1, tag=4)
    comm.send(l_M_ip1[5], dest=comm_rank - 1, tag=5)


def update_crossmap(
    l_C: list[np.ndarray],
    l_M: list[np.ndarray],
    l_M_ip1: list[np.ndarray],
    Bu_mid: np.ndarray,
    Bl_mid: np.ndarray,
    J: np.ndarray,
    middle_process: int,
    ending_process: int,
    blocksize: int,
) -> list[np.ndarray]:
    """Update the cross maps.

    Parameters
    ----------
    l_C : list of numpy matrix
        list of the cross maps
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    middle_process : int
        middle process of the current reduction step
    ending_process : int
        ending process of the current reduction step
    blocksize : int
        size of a block

    Returns
    -------
    l_C : list of numpy matrix
        list of the updated cross maps
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank < middle_process:
        l_C = update_crossmap_upper(l_M, l_M_ip1, l_C, Bu_mid, J, blocksize)

    elif comm_rank == middle_process:
        l_C = update_crossmap_middle(l_M, l_M_ip1, l_C, Bu_mid, Bl_mid, J, blocksize)

    elif comm_rank < ending_process:
        # Last process doesn't need to update the crossmap since it
        # doesn't own any bridges matrices.
        l_C = update_crossmap_lower(l_M, l_M_ip1, l_C, Bl_mid, J, blocksize)

    return l_C


def update_crossmap_upper(
    l_M: list[np.ndarray],
    l_M_ip1: list[np.ndarray],
    l_C: list[np.ndarray],
    Bu_mid: np.ndarray,
    J: np.ndarray,
    blocksize: int,
) -> list[np.ndarray]:
    """Cross maps update formula for the upper processes.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    l_C : list of numpy matrix
        list of the cross maps
    Bu_mid : numpy matrix
        upper bridge of the middle process
    J : numpy matrix
        J matrix
    blocksize : int
        size of a block

    Returns
    -------
    l_C : list of numpy matrix
        list of the updated cross maps
    """

    M3 = l_M[2]
    M4 = l_M[3]
    M7 = l_M[6]
    M8 = l_M[7]

    M3_ip1 = l_M_ip1[2]
    M4_ip1 = l_M_ip1[3]
    M7_ip1 = l_M_ip1[6]
    M8_ip1 = l_M_ip1[7]

    J12 = J[0:blocksize, blocksize : 2 * blocksize]

    l_C[0] -= M3 @ Bu_mid @ J12 @ M7_ip1
    l_C[1] -= M3 @ Bu_mid @ J12 @ M8_ip1
    l_C[2] -= M4 @ Bu_mid @ J12 @ M7_ip1
    l_C[3] -= M4 @ Bu_mid @ J12 @ M8_ip1
    l_C[4] -= M3_ip1 @ Bu_mid @ J12 @ M7
    l_C[5] -= M3_ip1 @ Bu_mid @ J12 @ M8
    l_C[6] -= M4_ip1 @ Bu_mid @ J12 @ M7
    l_C[7] -= M4_ip1 @ Bu_mid @ J12 @ M8

    return l_C


def update_crossmap_middle(
    l_M: list[np.ndarray],
    l_M_ip1: list[np.ndarray],
    l_C: list[np.ndarray],
    Bu_mid: np.ndarray,
    Bl_mid: np.ndarray,
    J: np.ndarray,
    blocksize: int,
) -> list[np.ndarray]:
    """Cross maps update formula for the middle process.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    l_C : list of numpy matrix
        list of the cross maps
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    blocksize : int
        size of a block

    Returns
    -------
    l_C : list of numpy matrix
        list of the updated cross maps
    """

    M3 = l_M[2]
    M4 = l_M[3]
    M7 = l_M[6]
    M8 = l_M[7]

    M1_ip1 = l_M_ip1[0]
    M2_ip1 = l_M_ip1[1]
    M5_ip1 = l_M_ip1[4]
    M6_ip1 = l_M_ip1[5]

    J11 = J[0:blocksize, 0:blocksize]
    J22 = J[blocksize : 2 * blocksize, blocksize : 2 * blocksize]

    l_C[0] -= M3 @ Bu_mid @ J11 @ M1_ip1
    l_C[1] -= M3 @ Bu_mid @ J11 @ M2_ip1
    l_C[2] -= M4 @ Bu_mid @ J11 @ M1_ip1
    l_C[3] -= M4 @ Bu_mid @ J11 @ M2_ip1
    l_C[4] -= M5_ip1 @ Bl_mid @ J22 @ M7
    l_C[5] -= M5_ip1 @ Bl_mid @ J22 @ M8
    l_C[6] -= M6_ip1 @ Bl_mid @ J22 @ M7
    l_C[7] -= M6_ip1 @ Bl_mid @ J22 @ M8

    return l_C


def update_crossmap_lower(
    l_M: list[np.ndarray],
    l_M_ip1: list[np.ndarray],
    l_C: list[np.ndarray],
    Bl_mid: np.ndarray,
    J: np.ndarray,
    blocksize: int,
) -> list[np.ndarray]:
    """Cross maps update formula for the lower processes.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_M_ip1 : list of numpy matrix
        list of the matrix maps of the next process
    l_C : list of numpy matrix
        list of the cross maps
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    blocksize : int
        size of a block

    Returns
    -------
    l_C : list of numpy matrix
        list of the updated cross maps
    """

    M1 = l_M[0]
    M2 = l_M[1]
    M5 = l_M[4]
    M6 = l_M[5]

    M1_ip1 = l_M_ip1[0]
    M2_ip1 = l_M_ip1[1]
    M5_ip1 = l_M_ip1[4]
    M6_ip1 = l_M_ip1[5]

    J21 = J[blocksize : 2 * blocksize, 0:blocksize]

    l_C[0] -= M5 @ Bl_mid @ J21 @ M1_ip1
    l_C[1] -= M5 @ Bl_mid @ J21 @ M2_ip1
    l_C[2] -= M6 @ Bl_mid @ J21 @ M1_ip1
    l_C[3] -= M6 @ Bl_mid @ J21 @ M2_ip1
    l_C[4] -= M5_ip1 @ Bl_mid @ J21 @ M1
    l_C[5] -= M5_ip1 @ Bl_mid @ J21 @ M2
    l_C[6] -= M6_ip1 @ Bl_mid @ J21 @ M1
    l_C[7] -= M6_ip1 @ Bl_mid @ J21 @ M2

    return l_C


def update_matrixmap(
    l_M: list[np.ndarray],
    l_U: list[np.ndarray],
    Bu_mid: np.ndarray,
    Bl_mid: np.ndarray,
    J: np.ndarray,
    middle_process: int,
    blocksize: int,
) -> list[np.ndarray]:
    """Update the matrix maps.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_U : list of numpy matrix
        list of the U factors
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    middle_process : int
        index of the middle process of the current reduction step
    blocksize : int
        size of a block

    Returns
    -------
    l_M : list of numpy matrix
        list of the updated matrix maps
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    if comm_rank <= middle_process:
        l_M = update_matrixmap_upper(l_M, l_U, Bu_mid, Bl_mid, J, blocksize)
    else:
        l_M = update_matrixmap_lower(l_M, l_U, Bu_mid, Bl_mid, J, blocksize)

    return l_M


def update_matrixmap_upper(
    l_M: list[np.ndarray],
    l_U: list[np.ndarray],
    Bu_mid: np.ndarray,
    Bl_mid: np.ndarray,
    J: np.ndarray,
    blocksize: int,
) -> list[np.ndarray]:
    """Formula to update the matrix maps associated with the upper partition.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_U : list of numpy matrix
        list of the U factors
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    blocksize : int
        size of a block

    Returns
    -------
    l_M : list of numpy matrix
        list of the updated matrix maps
    """

    UUR = l_U[0]
    DUR = l_U[4]
    ULL = l_U[1]
    DLL = l_U[5]

    J11 = J[0:blocksize, 0:blocksize]
    J12 = J[0:blocksize, blocksize : 2 * blocksize]
    J22 = J[blocksize : 2 * blocksize, blocksize : 2 * blocksize]

    # Attention: order of the updates is important.
    l_M[8] += l_M[2] @ Bu_mid @ J12 @ l_M[6]
    l_M[9] += l_M[2] @ Bu_mid @ J12 @ l_M[7]
    l_M[10] += l_M[3] @ Bu_mid @ J12 @ l_M[6]
    l_M[11] += l_M[3] @ Bu_mid @ J12 @ l_M[7]

    l_M[0] += UUR @ Bu_mid @ J12 @ l_M[6]
    l_M[1] += UUR @ Bu_mid @ J12 @ l_M[7]

    l_M[4] += l_M[2] @ Bu_mid @ J12 @ ULL
    l_M[5] += l_M[3] @ Bu_mid @ J12 @ ULL

    l_M[2] = l_M[2] @ Bu_mid @ J11 @ DUR
    l_M[3] = l_M[3] @ Bu_mid @ J11 @ DUR

    l_M[6] = DLL @ Bl_mid @ J22 @ l_M[6]
    l_M[7] = DLL @ Bl_mid @ J22 @ l_M[7]

    return l_M


def update_matrixmap_lower(
    l_M: list[np.ndarray],
    l_U: list[np.ndarray],
    Bu_mid: np.ndarray,
    Bl_mid: np.ndarray,
    J: np.ndarray,
    blocksize: int,
) -> list[np.ndarray]:
    """Formula to update the matrix maps associated with the lower partition.

    Parameters
    ----------
    l_M : list of numpy matrix
        list of the matrix maps
    l_U : list of numpy matrix
        list of the U factors
    Bu_mid : numpy matrix
        upper bridge of the middle process
    Bl_mid : numpy matrix
        lower bridge of the middle process
    J : numpy matrix
        J matrix
    blocksize : int
        size of a block

    Returns
    -------
    l_M : list of numpy matrix
        list of the updated matrix maps
    """

    UUR = l_U[0]
    DUR = l_U[4]
    ULL = l_U[1]
    DLL = l_U[5]

    J11 = J[0:blocksize, 0:blocksize]
    J21 = J[blocksize : 2 * blocksize, 0:blocksize]
    J22 = J[blocksize : 2 * blocksize, blocksize : 2 * blocksize]

    # Attention: order of the updates is important.
    l_M[8] += l_M[4] @ Bl_mid @ J21 @ l_M[0]
    l_M[9] += l_M[4] @ Bl_mid @ J21 @ l_M[1]
    l_M[10] += l_M[5] @ Bl_mid @ J21 @ l_M[0]
    l_M[11] += l_M[5] @ Bl_mid @ J21 @ l_M[1]

    l_M[2] += l_M[4] @ Bl_mid @ J21 @ DUR
    l_M[3] += l_M[5] @ Bl_mid @ J21 @ DUR

    l_M[4] = l_M[4] @ Bl_mid @ J22 @ ULL
    l_M[5] = l_M[5] @ Bl_mid @ J22 @ ULL

    l_M[6] += DLL @ Bl_mid @ J21 @ l_M[0]
    l_M[7] += DLL @ Bl_mid @ J21 @ l_M[1]

    l_M[0] = UUR @ Bu_mid @ J11 @ l_M[0]
    l_M[1] = UUR @ Bu_mid @ J11 @ l_M[1]

    return l_M


def produce_partition(
    K_local: np.ndarray, l_M: list[np.ndarray], l_C: list[np.ndarray], blocksize: int
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Produce the partition of the matrix.

    Parameters
    ----------
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    l_C : list of numpy matrix
        list of the cross maps
    blocksize : int
        size of a block

    Returns
    -------
    X_diagblk : numpy matrix
        diagonal entries of the inverted local partition
    X_upperblk : numpy matrix
        upper diagonal entries of the inverted local partition, the last block
        contains the upper bridge of the local partition. (execpt for the last
        process)
    X_lowerblk : numpy matrix
        lower diagonal entries of the inverted local partition, the last block
        contains the lower bridge of the local partition. (execpt for the last
        process)
    """

    # 1. Produce the tridiag part of the partition.
    partition_blocksize = K_local.shape[0] // blocksize

    X_diagblk = [
        np.zeros((blocksize, blocksize), dtype=K_local.dtype)
        for i in range(partition_blocksize)
    ]
    X_upperblk = [
        np.zeros((blocksize, blocksize), dtype=K_local.dtype)
        for i in range(partition_blocksize)
    ]
    X_lowerblk = [
        np.zeros((blocksize, blocksize), dtype=K_local.dtype)
        for i in range(partition_blocksize)
    ]

    for idx in range(0, partition_blocksize, 1):
        X_diagblk[idx] = produce_matrix_elements(idx, idx, K_local, l_M, blocksize)
        X_lowerblk[idx] = produce_matrix_elements(idx + 1, idx, K_local, l_M, blocksize)
        X_upperblk[idx] = produce_matrix_elements(idx, idx + 1, K_local, l_M, blocksize)

    # 2. Produce the bridge part of the partition.
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    Bu_inv = np.zeros((blocksize, blocksize), dtype=K_local.dtype)
    Bl_inv = np.zeros((blocksize, blocksize), dtype=K_local.dtype)

    for process_i in range(0, comm_size, 1):
        Bu_inv, Bl_inv = produce_bridges(
            Bu_inv, Bl_inv, K_local, l_C, process_i, blocksize
        )

    X_upperblk[-1] = Bu_inv
    X_lowerblk[-1] = Bl_inv

    return X_diagblk, X_upperblk, X_lowerblk


def produce_toprow_element(
    col_blockindex: int, K_local: np.ndarray, l_M: list[np.ndarray], blocksize: int
) -> np.ndarray:
    """Produce an element of the top row of the partition.

    Parameters
    ----------
    col_blockindex : int
        column index of the block to produce
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem : numpy matrix
        produced element of the top row of the partition
    """

    start_colindex = col_blockindex * blocksize
    end_colindex = start_colindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem = (
        l_M[0] @ K_local[0:blocksize, start_colindex:end_colindex]
        + l_M[1]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
    )

    return elem


def produce_rightcol_element(
    row_blockindex: int, K_local: np.ndarray, l_M: list[np.ndarray], blocksize: int
) -> np.ndarray:
    """Produce an element of the right column of the partition.

    Parameters
    ----------
    row_blockindex : int
        row index of the block to produce
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem : numpy matrix
        produced element of the right column of the partition
    """

    start_rowindex = row_blockindex * blocksize
    end_rowindex = start_rowindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem = (
        K_local[start_rowindex:end_rowindex, 0:blocksize] @ l_M[2]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[3]
    )

    return elem


def produce_leftcol_element(
    row_blockindex: int, K_local: np.ndarray, l_M: list[np.ndarray], blocksize: int
) -> np.ndarray:
    """Produce an element of the left column of the partition.

    Parameters
    ----------
    row_blockindex : int
        row index of the block to produce
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem : numpy matrix
        produced element of the left column of the partition
    """

    start_rowindex = row_blockindex * blocksize
    end_rowindex = start_rowindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem = (
        K_local[start_rowindex:end_rowindex, 0:blocksize] @ l_M[4]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[5]
    )

    return elem


def produce_botrow_element(
    col_blockindex: int, K_local: np.ndarray, l_M: list[np.ndarray], blocksize: int
) -> np.ndarray:
    """Produce an element of the bottom row of the partition.

    Parameters
    ----------
    col_blockindex : int
        column index of the block to produce
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem : numpy matrix
        produced element of the bottom row of the partition
    """

    start_colindex = col_blockindex * blocksize
    end_colindex = start_colindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem = (
        l_M[6] @ K_local[0:blocksize, start_colindex:end_colindex]
        + l_M[7]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
    )

    return elem


def produce_matrix_elements(
    row_blockindex: int,
    col_blockindex: int,
    K_local: np.ndarray,
    l_M: list[np.ndarray],
    blocksize: int,
) -> np.ndarray:
    """Produce a selected block of the given partition.

    Parameters
    ----------
    row_blockindex : int
        row blockindex of the block
    col_blockindex : int
        column blockindex of the block
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem : numpy matrix
        selected block of the given partition
    """

    start_rowindex = row_blockindex * blocksize
    end_rowindex = start_rowindex + blocksize

    start_colindex = col_blockindex * blocksize
    end_colindex = start_colindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem = (
        K_local[start_rowindex:end_rowindex, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, 0:blocksize]
        @ l_M[8]
        @ K_local[0:blocksize, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, 0:blocksize]
        @ l_M[9]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[10]
        @ K_local[0:blocksize, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[11]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
    )

    return elem


def produce_update_matrix_elements(
    row_blockindex: int,
    col_blockindex: int,
    K_local: np.ndarray,
    l_M: list[np.ndarray],
    blocksize: int,
) -> np.ndarray:
    """Produce the update of a selected block of the given partition.

    Parameters
    ----------
    row : int
        row index of the block
    col : int
        column index of the block
    K_local : numpy matrix
        local inverted partition of the matrix
    l_M : list of numpy matrix
        list of the matrix maps
    blocksize : int
        size of a block

    Returns
    -------
    elem_update : numpy matrix
        update for the selected block of the given partition
    """

    start_rowindex = row_blockindex * blocksize
    end_rowindex = start_rowindex + blocksize

    start_colindex = col_blockindex * blocksize
    end_colindex = start_colindex + blocksize

    start_lastblockindex = K_local.shape[0] - blocksize
    end_lastblockindex = K_local.shape[0]

    elem_update = (
        K_local[start_rowindex:end_rowindex, 0:blocksize]
        @ l_M[8]
        @ K_local[0:blocksize, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, 0:blocksize]
        @ l_M[9]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[10]
        @ K_local[0:blocksize, start_colindex:end_colindex]
        + K_local[start_rowindex:end_rowindex, start_lastblockindex:end_lastblockindex]
        @ l_M[11]
        @ K_local[start_lastblockindex:end_lastblockindex, start_colindex:end_colindex]
    )

    return elem_update


def produce_bridges(
    Bu_inv: np.ndarray,
    Bl_inv: np.ndarray,
    K_local: np.ndarray,
    l_C: list[np.ndarray],
    process_i: int,
    blocksize: int,
) -> [np.ndarray, np.ndarray]:
    """Produce the upper and lower bridges.

    Parameters
    ----------
    Bu_inv : numpy matrix
        upper bridge to be produced
    Bl_inv : numpy matrix
        lower bridge to be produced
    K_local : numpy matrix
        local inverted partition of the matrix
    l_C : list of numpy matrix
        list of the cross maps
    process_i : int
        index of the process to produce the bridges
    blocksize : int
        size of a block

    Returns
    -------
    Bu_inv : numpy matrix
        upper bridge
    Bl_inv : numpy matrix
        lower bridge
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    N_rowindex = K_local.shape[0] - blocksize

    if comm_rank == process_i and comm_rank != comm_size - 1:
        phi1_1_N = K_local[0:blocksize, N_rowindex : N_rowindex + blocksize]
        phi1_N_1 = K_local[N_rowindex : N_rowindex + blocksize, 0:blocksize]
        phi1_N_N = K_local[
            N_rowindex : N_rowindex + blocksize, N_rowindex : N_rowindex + blocksize
        ]

        phi2_1_1 = comm.recv(source=comm_rank + 1, tag=0)
        phi2_N_1 = comm.recv(source=comm_rank + 1, tag=1)
        phi2_1_N = comm.recv(source=comm_rank + 1, tag=2)

        Bu_inv = produce_upper_bridge(
            phi1_N_1, phi1_N_N, phi2_1_1, phi2_N_1, l_C, blocksize
        )
        Bl_inv = produce_lower_bridge(
            phi1_1_N, phi1_N_N, phi2_1_1, phi2_1_N, l_C, blocksize
        )

    elif comm_rank == process_i + 1 and process_i + 1 != comm_size:
        comm.send(K_local[0:blocksize, 0:blocksize], dest=comm_rank - 1, tag=0)
        comm.send(
            K_local[N_rowindex : N_rowindex + blocksize, 0:blocksize],
            dest=comm_rank - 1,
            tag=1,
        )
        comm.send(
            K_local[0:blocksize, N_rowindex : N_rowindex + blocksize],
            dest=comm_rank - 1,
            tag=2,
        )

    return Bu_inv, Bl_inv


def produce_upper_bridge(
    phi1_N_1: np.ndarray,
    phi1_N_N: np.ndarray,
    phi2_1_1: np.ndarray,
    phi2_N_1: np.ndarray,
    l_C: list[np.ndarray],
    blocksize: int,
) -> np.ndarray:
    """Produce the upper bridge.

    Parameters
    ----------
    phi1_N_1 : numpy matrix
        Lower left block of the upper partition
    phi1_N_N : numpy matrix
        Lower right block of the upper partition
    phi2_1_1 : numpy matrix
        Upper left block of the lower partition
    phi2_N_1 : numpy matrix
        Lower left block of the lower partition
    l_C : list of numpy matrix
        list of the cross maps
    blocksize : int
        size of a block

    Returns
    -------
    Bu_inv : numpy matrix
        upper bridge
    """

    Bu_inv = np.zeros((blocksize, blocksize), dtype=phi1_N_1.dtype)

    Bu_inv = (
        phi1_N_1 @ l_C[0] @ phi2_1_1
        + phi1_N_1 @ l_C[1] @ phi2_N_1
        + phi1_N_N @ l_C[2] @ phi2_1_1
        + phi1_N_N @ l_C[3] @ phi2_N_1
    )

    return Bu_inv


def produce_lower_bridge(
    phi1_1_N: np.ndarray,
    phi1_N_N: np.ndarray,
    phi2_1_1: np.ndarray,
    phi2_1_N: np.ndarray,
    l_C: list[np.ndarray],
    blocksize: int,
) -> np.ndarray:
    """Produce the lower bridge.

    Parameters
    ----------
    phi1_1_N : numpy matrix
        Upper right block of the upper partition
    phi1_N_N : numpy matrix
        Lower right block of the upper partition
    phi2_1_1 : numpy matrix
        Upper left block of the lower partition
    phi2_1_N : numpy matrix
        Upper right block of the lower partition
    l_C : list of numpy matrix
        list of the cross maps
    blocksize : int
        size of a block

    Returns
    -------
    Bl_inv : numpy matrix
        lower bridge
    """

    Bl_inv = np.zeros((blocksize, blocksize), dtype=phi1_1_N.dtype)

    Bl_inv = (
        phi2_1_1 @ l_C[4] @ phi1_1_N
        + phi2_1_1 @ l_C[5] @ phi1_N_N
        + phi2_1_N @ l_C[6] @ phi1_1_N
        + phi2_1_N @ l_C[7] @ phi1_N_N
    )

    return Bl_inv
