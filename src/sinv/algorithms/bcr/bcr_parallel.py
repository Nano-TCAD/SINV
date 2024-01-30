"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

@reference: https://doi.org/10.1016/j.jcp.2009.03.035
@reference: https://doi.org/10.1017/CBO9780511812583

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from sinv import utils
from sinv.algorithms.bcr import bcr_utils as bcr_u

import numpy as np
import math

from mpi4py import MPI


def bcr_parallel(A: np.ndarray, blocksize: int) -> np.ndarray:
    """Performe the tridiagonal selected inversion using a parallel version of
    the block cyclic reduction algorithm.

    Parameters
    ----------
    A : np.ndarray
        matrix to invert
    blocksize : int
        size of a block of the matrix A

    Returns
    -------
    G : np.ndarray
        inverse of the matrix A
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    nblocks_initial = A.shape[0] // blocksize

    # First the input matrix may need to be 0-padded to a power of 2 number of blocks
    block_padding_distance = bcr_u.distance_to_power_of_two(nblocks_initial)
    A = bcr_u.identity_padding(A, block_padding_distance * blocksize)

    nblocks_padded = A.shape[0] // blocksize

    L = np.zeros(
        (nblocks_padded * blocksize, nblocks_padded * blocksize), dtype=A.dtype
    )
    U = np.zeros(
        (nblocks_padded * blocksize, nblocks_padded * blocksize), dtype=A.dtype
    )
    G = np.zeros(
        (nblocks_padded * blocksize, nblocks_padded * blocksize), dtype=A.dtype
    )

    # Partitionning
    l_start_blockrow, l_partitions_blocksizes = bcr_u.divide_matrix(
        nblocks_padded, comm_size
    )
    process_top_blockrow, process_bottom_blockrow = bcr_u.get_process_rowblock_index(
        l_start_blockrow[comm_rank], l_partitions_blocksizes[comm_rank]
    )

    # 1. Block cyclic reduction
    i_bcr = [i for i in range(nblocks_padded)]
    final_reduction_block = reduce_bcr(
        A, L, U, i_bcr, process_top_blockrow, process_bottom_blockrow, blocksize
    )

    # 2. Block cyclic production
    invert_block(
        A,
        G,
        final_reduction_block,
        process_top_blockrow,
        process_bottom_blockrow,
        blocksize,
    )
    produce_bcr(
        A, L, U, G, i_bcr, process_top_blockrow, process_bottom_blockrow, blocksize
    )

    # Formating result
    agregate_result_on_root(G, l_start_blockrow, l_partitions_blocksizes, blocksize)
    G = G[: nblocks_initial * blocksize, : nblocks_initial * blocksize]

    return G


def reduce(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    row: int,
    level: int,
    i_elim: list,
    blocksize: int,
) -> None:
    """Operate the reduction towards the row-th row of the matrix A.

    Parameters
    ----------
    A : np.ndarray
        the matrix to reduce
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    row : int
        the row to be reduce towards
    level : int
        the current level of the reduction in the reduction tree
    i_elim : list
        target row indices to be eliminated
    blocksize : int
        the size of the blocks in A

    Returns
    -------
    None
    """

    nblocks = A.shape[0] // blocksize
    offset_blockindex = int(math.pow(2, level))

    # Reduction from i (above row) and k (below row) to j row
    i_blockindex = i_elim[row] - offset_blockindex
    j_blockindex = i_elim[row]
    k_blockindex = i_elim[row] + offset_blockindex

    # Computing of row-based indices
    i_rowindex = i_blockindex * blocksize
    ip1_rowindex = (i_blockindex + 1) * blocksize

    j_rowindex = j_blockindex * blocksize
    jp1_rowindex = (j_blockindex + 1) * blocksize

    k_rowindex = k_blockindex * blocksize
    kp1_rowindex = (k_blockindex + 1) * blocksize

    # If there is a row above
    if i_blockindex >= 0:
        A_ii_inv = np.linalg.inv(A[i_rowindex:ip1_rowindex, i_rowindex:ip1_rowindex])

        U[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex] = (
            A_ii_inv @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]
        )

        L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] = (
            A[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A_ii_inv
        )

        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] = (
            A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex]
            - L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex]
            @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]
        )

        # If the row above is not the top row
        if i_blockindex != i_elim[0]:
            h_rowindex = (i_blockindex - offset_blockindex) * blocksize
            hp1_rowindex = (i_blockindex - offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, h_rowindex:hp1_rowindex] = (
                -L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex]
                @ A[i_rowindex:ip1_rowindex, h_rowindex:hp1_rowindex]
            )

    # If there is a row below
    if k_blockindex <= nblocks - 1:
        A_kk_inv = np.linalg.inv(A[k_rowindex:kp1_rowindex, k_rowindex:kp1_rowindex])

        U[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex] = (
            A_kk_inv @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]
        )

        L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] = (
            A[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] @ A_kk_inv
        )

        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] = (
            A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex]
            - L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]
            @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]
        )

        # If the row below is not the bottom row
        if k_blockindex != i_elim[-1]:
            l_rowindex = (k_blockindex + offset_blockindex) * blocksize
            lp1_rowindex = (k_blockindex + offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, l_rowindex:lp1_rowindex] = (
                -L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]
                @ A[k_rowindex:kp1_rowindex, l_rowindex:lp1_rowindex]
            )


def reduce_bcr(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    i_bcr: list,
    top_blockrow: int,
    bottom_blockrow: int,
    blocksize: int,
) -> int:
    """Performs block cyclic reduction in parallel on the matrix A. Computing
    during the process the LU decomposition of the matrix A. The matrix A is
    overwritten.

    Parameters
    ----------
    A : np.ndarray
        the matrix to reduce
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    i_bcr : list
        blockrows to perform the reduction on
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks in the matrix A

    Returns
    -------
    last_reduction_row : int
        the index of the last row that was reduced
    """

    nblocks = len(i_bcr)
    height = int(math.log2(nblocks))

    last_reduction_block = 0

    for level_blockindex in range(height):
        i_elim = bcr_u.compute_i_from(level_blockindex, nblocks)

        number_of_reduction = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                number_of_reduction += 1

        indice_process_start_reduction = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                indice_process_start_reduction = i
                break

        indice_process_stop_reduction = 0
        for i in range(len(i_elim)):
            if i_elim[i] >= top_blockrow and i_elim[i] < bottom_blockrow:
                indice_process_stop_reduction = i

        if number_of_reduction != 0:
            for row in range(
                indice_process_start_reduction, indice_process_stop_reduction + 1
            ):
                reduce(A, L, U, row, level_blockindex, i_elim, blocksize)

        # Here each process should communicate the last row of the reduction to the next process
        if level_blockindex != height - 1:
            communicate_reducprod(
                A,
                L,
                U,
                i_elim,
                indice_process_start_reduction,
                indice_process_stop_reduction,
                blocksize,
            )

        if len(i_elim) > 0:
            last_reduction_block = i_elim[-1]

    return last_reduction_block


def communicate_reducprod(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    i_from: list,
    indice_process_start_reduction: int,
    indice_process_stop_reduction: int,
    blocksize: int,
) -> None:
    """Communicate the last produced row of the current level to the surrounding
    processes. They may need this value to produce at the next level.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    i_from : list
        list of the active blockrow at the current level
    indice_process_start_reduction : int
        index of the first blockrow to reduce
    indice_process_stop_reduction : int
        index of the last blockrow to reduce
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        blockrow_idx = i_from[indice_process_stop_reduction]
        send_down(A, L, U, blockrow_idx, blocksize)

        blockrow_idx = i_from[indice_process_stop_reduction + 1]
        recv_down(A, L, U, blockrow_idx, blocksize)

    elif comm_rank == comm_size - 1:
        blockrow_idx = i_from[indice_process_start_reduction - 1]
        recv_up(A, L, U, blockrow_idx, blocksize)

        blockrow_idx = i_from[indice_process_start_reduction]
        send_up(A, L, U, blockrow_idx, blocksize)
    else:
        # Even ranks communicates first
        if comm_rank % 2 == 0:
            # Downward communication
            blockrow_idx = i_from[indice_process_stop_reduction]
            send_down(A, L, U, blockrow_idx, blocksize)

            blockrow_idx = i_from[indice_process_start_reduction - 1]
            recv_up(A, L, U, blockrow_idx, blocksize)

            # Upward communication
            blockrow_idx = i_from[indice_process_start_reduction]
            send_up(A, L, U, blockrow_idx, blocksize)

            blockrow_idx = i_from[indice_process_stop_reduction + 1]
            recv_down(A, L, U, blockrow_idx, blocksize)

        # Odd rank communicates second
        else:
            # Downward communication
            blockrow_idx = i_from[indice_process_start_reduction - 1]
            recv_up(A, L, U, blockrow_idx, blocksize)

            blockrow_idx = i_from[indice_process_stop_reduction]
            send_down(A, L, U, blockrow_idx, blocksize)

            # Upward communication
            blockrow_idx = i_from[indice_process_stop_reduction + 1]
            recv_down(A, L, U, blockrow_idx, blocksize)

            blockrow_idx = i_from[indice_process_start_reduction]
            send_up(A, L, U, blockrow_idx, blocksize)


def send_down(
    A: np.ndarray, L: np.ndarray, U: np.ndarray, blockrow_idx: int, blocksize: int
) -> None:
    """Send downward the last produced row of the current level to the next
    process.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    blockrow_idx : int
        index of the blockrow to communicate
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    i_rowindice = blockrow_idx * blocksize
    ip1_rowindice = (blockrow_idx + 1) * blocksize

    comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank + 1, tag=0)
    comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank + 1, tag=1)
    comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank + 1, tag=2)


def recv_up(
    A: np.ndarray, L: np.ndarray, U: np.ndarray, blockrow_idx: int, blocksize: int
) -> None:
    """Receive upward the last produced row of the current level from the
    previous process.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    blockrow_idx : int
        index of the blockrow to communicate
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    i_rowindice = blockrow_idx * blocksize
    ip1_rowindice = (blockrow_idx + 1) * blocksize

    A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank - 1, tag=0)
    L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank - 1, tag=1)
    U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank - 1, tag=2)


def send_up(
    A: np.ndarray, L: np.ndarray, U: np.ndarray, blockrow_idx: int, blocksize: int
) -> None:
    """Send upward the last produced row of the current level to the previous
    process.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    blockrow_idx : int
        index of the blockrow to communicate
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    i_rowindice = blockrow_idx * blocksize
    ip1_rowindice = (blockrow_idx + 1) * blocksize

    comm.send(A[i_rowindice:ip1_rowindice, :], dest=comm_rank - 1, tag=0)
    comm.send(L[i_rowindice:ip1_rowindice, :], dest=comm_rank - 1, tag=1)
    comm.send(U[:, i_rowindice:ip1_rowindice], dest=comm_rank - 1, tag=2)


def recv_down(
    A: np.ndarray, L: np.ndarray, U: np.ndarray, blockrow_idx: int, blocksize: int
) -> None:
    """Receive downward the last produced row of the current level from the
    next process.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    blockrow_idx : int
        index of the blockrow to communicate
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    i_rowindice = blockrow_idx * blocksize
    ip1_rowindice = (blockrow_idx + 1) * blocksize

    A[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank + 1, tag=0)
    L[i_rowindice:ip1_rowindice, :] = comm.recv(source=comm_rank + 1, tag=1)
    U[:, i_rowindice:ip1_rowindice] = comm.recv(source=comm_rank + 1, tag=2)


def invert_block(
    A: np.ndarray,
    G: np.ndarray,
    target_block: int,
    top_blockrow: int,
    bottom_blockrow: int,
    blocksize: int,
) -> None:
    """Produce the first block of the inverse of A after having perfomed the
    cyclic reduction.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    G : np.ndarray
        output inverse matrix
    target_block : int
        index of the block to invert
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    if target_block >= top_blockrow and target_block < bottom_blockrow:
        target_row = target_block * blocksize
        target_row_p1 = (target_block + 1) * blocksize

        G[target_row:target_row_p1, target_row:target_row_p1] = np.linalg.inv(
            A[target_row:target_row_p1, target_row:target_row_p1]
        )


def corner_produce(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    k_from: int,
    k_to: int,
    blocksize: int,
) -> None:
    """BCR production procedure associated with the corner production case.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition factors of A
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    G : np.ndarray
        output matrix to be produced
    k_from : int
        index of the block row to produce from
    k_to : int
        index of the block row to produce to
    blocksize : int
        size of the blocks in the matrix A

    Returns
    -------
    None
    """

    k_from_rowindex = k_from * blocksize
    kp1_from_rowindex = (k_from + 1) * blocksize

    k_to_rowindex = k_to * blocksize
    kp1_to_rowindex = (k_to + 1) * blocksize

    G[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex] = (
        -1
        * G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex]
        @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]
    )

    G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex] = (
        -U[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex]
        @ G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex]
    )

    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex] = (
        np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex])
        - G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex]
        @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]
    )


def center_produce(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    k_above: int,
    k_to: int,
    k_below: int,
    blocksize: int,
) -> None:
    """BCR production procedure associated with the center production case.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition factors of A
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    G : np.ndarray
        output matrix to be produced
    k_above : int
        index of the block row above to produce from
    k_to : int
        index of the block row to produce
    k_below : int
        index of the block row below to produce from
    blocksize : int
        size of the blocks in the matrix A

    Returns
    -------
    None
    """

    k_above_rowindex = k_above * blocksize
    kp1_above_rowindex = (k_above + 1) * blocksize

    k_to_rowindex = k_to * blocksize
    kp1_to_rowindex = (k_to + 1) * blocksize

    k_below_rowindex = k_below * blocksize
    kp1_below_rowindex = (k_below + 1) * blocksize

    G[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex] = (
        -G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex]
        @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]
        - G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex]
        @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    )

    G[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex] = (
        -G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex]
        @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]
        - G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex]
        @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    )

    G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] = (
        -U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]
        @ G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex]
        - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]
        @ G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex]
    )

    G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] = (
        -U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]
        @ G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex]
        - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]
        @ G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex]
    )

    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex] = (
        np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex])
        - G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]
        @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]
        - G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]
        @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
    )


def produce(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    i_bcr: list,
    i_prod: list,
    stride_blockindex,
    top_blockrow,
    bottom_blockrow,
    blocksize: int,
) -> None:
    """Call the appropriate production function for each block that needs to
    be produced.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    G : np.ndarray
        output inverse matrix
    i_bcr : list
        blockrows to perform the production on
    i_prod : list
        blockrows to produce
    stride_blockindex : int
        the stride between the blockrows to produce
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    for i_prod_blockindex in range(len(i_prod)):
        k_to = i_bcr[i_prod[i_prod_blockindex]]

        if k_to >= top_blockrow and k_to < bottom_blockrow:

            if i_prod_blockindex == 0:
                # Production row is the first row within the stride_blockindex range
                # It only gets values from the below row
                k_from = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                corner_produce(A, L, U, G, k_from, k_to, blocksize)

            if i_prod_blockindex != 0 and i_prod_blockindex == len(i_prod) - 1:
                if i_prod[-1] <= len(i_bcr) - stride_blockindex - 1:
                    k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                    k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                    center_produce(A, L, U, G, k_above, k_to, k_below, blocksize)
                else:
                    # Production row is the last row within the stride_blockindex range
                    # It only gets values from the above row
                    k_from = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]

                    corner_produce(A, L, U, G, k_from, k_to, blocksize)

            if i_prod_blockindex != 0 and i_prod_blockindex != len(i_prod) - 1:
                # Production row is in the middle of the stride_blockindex range
                # It gets values from the above and below rows
                k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                center_produce(A, L, U, G, k_above, k_to, k_below, blocksize)


def comm_to_produce(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    i_from: list,
    i_prod: list,
    stride_blockindex: int,
    top_blockrow: int,
    bottom_blockrow: int,
    blocksize: int,
) -> None:
    """Determine the blocks-row that need to be communicated/received to/from
    other processes in order for each process to be able to locally produce the
    blocks-row it is responsible for.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    G : np.ndarray
        output inverse matrix
    i_from : list
        blockrows to communicate/receive
    i_prod : list
        blockrows to produce
    stride_blockindex : int
        the stride between the blockrows to produce
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Determine what belongs to the current process
    current_process_i_from = []
    for i in range(len(i_from)):
        if i_from[i] >= top_blockrow and i_from[i] < bottom_blockrow:
            current_process_i_from.append(i_from[i])

    current_process_i_prod = []
    for i in range(len(i_prod)):
        if i_prod[i] >= top_blockrow and i_prod[i] < bottom_blockrow:
            current_process_i_prod.append(i_prod[i])

    # Determine what needs to be sent to other processes
    n_blocks = A.shape[0] // blocksize
    for i in range(len(current_process_i_from)):
        k_from = current_process_i_from[i]
        k_to_above = k_from - stride_blockindex
        k_to_below = k_from + stride_blockindex

        k_from_rowindex = k_from * blocksize
        kp1_from_rowindex = k_from_rowindex + blocksize

        if k_to_above < top_blockrow and k_to_above >= 0:

            comm.send(
                A[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank - 1, tag=0
            )
            comm.send(
                L[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank - 1, tag=1
            )
            comm.send(
                U[:, k_from_rowindex:kp1_from_rowindex], dest=comm_rank - 1, tag=2
            )
            comm.send(
                G[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank - 1, tag=3
            )

        if k_to_below >= bottom_blockrow and k_to_below < n_blocks:

            comm.send(
                A[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank + 1, tag=0
            )
            comm.send(
                L[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank + 1, tag=1
            )
            comm.send(
                U[:, k_from_rowindex:kp1_from_rowindex], dest=comm_rank + 1, tag=2
            )
            comm.send(
                G[k_from_rowindex:kp1_from_rowindex, :], dest=comm_rank + 1, tag=3
            )

    # Determine what needs to be received from other processes
    for i in range(len(current_process_i_prod)):
        k_from_above = current_process_i_prod[i] - stride_blockindex
        k_from_below = current_process_i_prod[i] + stride_blockindex
        if k_from_above < top_blockrow and k_from_above >= 0:
            k_from_above_rowindex = k_from_above * blocksize
            kp1_from_above_rowindex = k_from_above_rowindex + blocksize

            A[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(
                source=comm_rank - 1, tag=0
            )
            L[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(
                source=comm_rank - 1, tag=1
            )
            U[:, k_from_above_rowindex:kp1_from_above_rowindex] = comm.recv(
                source=comm_rank - 1, tag=2
            )
            G[k_from_above_rowindex:kp1_from_above_rowindex, :] = comm.recv(
                source=comm_rank - 1, tag=3
            )

        if k_from_below >= bottom_blockrow and k_from_below < n_blocks:
            k_from_below_rowindex = k_from_below * blocksize
            kp1_from_below_rowindex = k_from_below_rowindex + blocksize

            A[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(
                source=comm_rank + 1, tag=0
            )
            L[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(
                source=comm_rank + 1, tag=1
            )
            U[:, k_from_below_rowindex:kp1_from_below_rowindex] = comm.recv(
                source=comm_rank + 1, tag=2
            )
            G[k_from_below_rowindex:kp1_from_below_rowindex, :] = comm.recv(
                source=comm_rank + 1, tag=3
            )


def comm_produced(
    G: np.ndarray,
    i_prod: list,
    i_from: list,
    stride_blockindex: int,
    top_blockrow: int,
    bottom_blockrow: int,
    blocksize: int,
) -> None:
    """Communicate part of the produced row back to the original process.

    Parameters
    ----------
    G : np.ndarray
        output inverse matrix
    i_prod : list
        blockrows to produce
    i_from : list
        blockrows to communicate/receive
    stride_blockindex : int
        the stride between the blockrows to produce
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    # Determine what belongs to the current process
    current_process_i_from = []
    for i in range(len(i_from)):
        if i_from[i] >= top_blockrow and i_from[i] < bottom_blockrow:
            current_process_i_from.append(i_from[i])

    current_process_i_prod = []
    for i in range(len(i_prod)):
        if i_prod[i] >= top_blockrow and i_prod[i] < bottom_blockrow:
            current_process_i_prod.append(i_prod[i])

    # Determine what needs to be sent to other processes
    n_blocks = G.shape[0] // blocksize
    for i in range(len(current_process_i_from)):
        k_from = current_process_i_from[i]
        k_to_above = k_from - stride_blockindex
        k_to_below = k_from + stride_blockindex

        k_from_colindex = k_from * blocksize
        kp1_from_colindex = k_from_colindex + blocksize

        if k_to_above < top_blockrow and k_to_above >= 0:
            k_to_above_rowindex = k_to_above * blocksize
            kp1_to_above_rowindex = k_to_above_rowindex + blocksize

            comm.send(
                G[
                    k_to_above_rowindex:kp1_to_above_rowindex,
                    k_from_colindex:kp1_from_colindex,
                ],
                dest=comm_rank - 1,
                tag=3,
            )

        if k_to_below >= bottom_blockrow and k_to_below < n_blocks:
            k_to_below_rowindex = k_to_below * blocksize
            kp1_to_below_rowindex = k_to_below_rowindex + blocksize

            comm.send(
                G[
                    k_to_below_rowindex:kp1_to_below_rowindex,
                    k_from_colindex:kp1_from_colindex,
                ],
                dest=comm_rank + 1,
                tag=3,
            )

    # Determine what needs to be received from other processes
    for i in range(len(current_process_i_prod)):
        k_to = current_process_i_prod[i]
        k_from_above = k_to - stride_blockindex
        k_from_below = k_to + stride_blockindex

        k_to_rowindex = k_to * blocksize
        kp1_to_rowindex = k_to_rowindex + blocksize

        if k_from_above < top_blockrow and k_from_above >= 0:
            k_from_above_colindex = k_from_above * blocksize
            kp1_from_above_colindex = k_from_above_colindex + blocksize

            G[
                k_to_rowindex:kp1_to_rowindex,
                k_from_above_colindex:kp1_from_above_colindex,
            ] = comm.recv(source=comm_rank - 1, tag=3)

        if k_from_below >= bottom_blockrow and k_from_below < n_blocks:
            k_from_below_colindex = k_from_below * blocksize
            kp1_from_below_colindex = k_from_below_colindex + blocksize

            G[
                k_to_rowindex:kp1_to_rowindex,
                k_from_below_colindex:kp1_from_below_colindex,
            ] = comm.recv(source=comm_rank + 1, tag=3)


def produce_bcr(
    A: np.ndarray,
    L: np.ndarray,
    U: np.ndarray,
    G: np.ndarray,
    i_bcr: list,
    top_blockrow: int,
    bottom_blockrow: int,
    blocksize: int,
) -> None:
    """Performs the block cyclic production.

    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    G : np.ndarray
        output inverse matrix
    i_bcr : list
        blockrows to perform the production on
    top_blockrow : int
        the first blockrow that belong to the current process
    bottom_blockrow : int
        the last blockrow that belong to the current process
    blocksize : int
        size of the blocks

    Returns
    -------
    None
    """

    nblocks = len(i_bcr)
    height = int(math.log2(nblocks))

    for level_blockindex in range(height - 1, -1, -1):
        stride_blockindex = int(math.pow(2, level_blockindex))

        # Determine the blocks-row to be produced
        i_from = bcr_u.compute_i_from(level_blockindex, nblocks)
        i_prod = bcr_u.compute_i_prod(i_from, stride_blockindex)

        comm_to_produce(
            A,
            L,
            U,
            G,
            i_from,
            i_prod,
            stride_blockindex,
            top_blockrow,
            bottom_blockrow,
            blocksize,
        )

        produce(
            A,
            L,
            U,
            G,
            i_bcr,
            i_prod,
            stride_blockindex,
            top_blockrow,
            bottom_blockrow,
            blocksize,
        )

        comm_produced(
            G,
            i_from,
            i_prod,
            stride_blockindex,
            top_blockrow,
            bottom_blockrow,
            blocksize,
        )


def agregate_result_on_root(
    G: np.ndarray, l_start_blockrow: list, l_partitions_blocksizes: list, blocksize: int
) -> None:
    """Agregate the distributed results on the root process

    Parameters
    ----------
    G : np.ndarray
        inverted matrix to agregate
    l_start_blockrow : list
        list of the start blockrow of each process
    l_partitions_blocksizes : list
        list of the partitions blocksize of each process
    blocksize : int
        blocksize of the matrix

    Returns
    -------
    None
    """

    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    if comm_rank == 0:
        # Root process that need to agregate all the results

        for process_i in range(1, comm_size):
            # Receive from all the central processes
            top_blockrow_pi = l_start_blockrow[process_i]
            bottom_blockrow_pi = top_blockrow_pi + l_partitions_blocksizes[process_i]

            top_rowindice_pi = top_blockrow_pi * blocksize
            bottom_rowindice_pi = bottom_blockrow_pi * blocksize

            G[top_rowindice_pi:bottom_rowindice_pi, :] = comm.recv(
                source=process_i, tag=0
            )

    else:
        top_rowindice = l_start_blockrow[comm_rank] * blocksize
        bottom_rowindice = (
            top_rowindice + l_partitions_blocksizes[comm_rank] * blocksize
        )

        comm.send(G[top_rowindice:bottom_rowindice, :], dest=0, tag=0)
