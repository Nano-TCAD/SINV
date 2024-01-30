"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

Contains the utility functions for the BCR algorithm.

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

import numpy as np
import math

from mpi4py import MPI


def distance_to_power_of_two(matrice_size: int) -> int:
    """Compute the distance between the matrice_size and the closest power
    of two minus one.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.

    Returns
    -------
    distance : int
        The distance between the matrice_size and the closest power of two
        minus one.
    """

    p = math.ceil(math.log2(matrice_size + 1))

    closest_power = 2**p - 1
    distance = closest_power - matrice_size

    return distance


def identity_padding(A: np.ndarray, padding_size: int) -> np.ndarray:
    """Padd the A matrix with an identity matrix of size padding_size.

    Parameters
    ----------
    A : np.ndarray
        The matrix to padd.
    padding_size : int
        The size of the padding to add.

    Returns
    -------
    Ap : np.ndarray
        The padded matrix.
    """

    matrice_size = A.shape[0]

    I = np.eye(padding_size)
    Ap = np.zeros(
        (matrice_size + padding_size, matrice_size + padding_size), dtype=A.dtype
    )

    Ap[0:matrice_size, 0:matrice_size] = A
    Ap[
        matrice_size : matrice_size + padding_size,
        matrice_size : matrice_size + padding_size,
    ] = I

    return Ap


def compute_i_from(level: int, nblocks: int) -> list[int]:
    """Compute the blocks-row that will be used for production at the
    current level of the production tree.

    Parameters
    ----------
    level : int
        the current level of the production tree.
    nblocks : int
        the number of blocks in the matrix.

    Returns
    -------
    i_from : list
        the blocks-row that will be used for production at the current level
    """

    return [
        i
        for i in range(
            int(math.pow(2, level + 1)) - 1, nblocks, int(math.pow(2, level + 1))
        )
    ]


def compute_i_prod(i_from: list, stride_blockindex: int) -> list[int]:
    """Compute the blocks-row to be produced at the current level of the
    production tree.

    Parameters
    ----------
    i_from : list
        the blocks-row that will be used for production at the current level
    stride_blockindex : int
        the stride between the blocks-row to be produced at the current level

    Returns
    -------
    i_prod : list
        the blocks-row to be produced at the current level
    """

    i_prod = []
    for i in range(len(i_from)):
        if i == 0:
            i_prod.append(i_from[i] - stride_blockindex)
            i_prod.append(i_from[i] + stride_blockindex)
        else:
            if i_prod[i] != i_from[i] - stride_blockindex:
                i_prod.append(i_from[i] - stride_blockindex)
            i_prod.append(i_from[i] + stride_blockindex)

    return i_prod


def divide_matrix(n_blocks: int, n_partitions: int) -> [list[int], list[int]]:
    """Compute the n_partitions segments that divide the matrix A.

    Parameters
    ----------
    n_blocks : int
        number of blocks
    n_partitions : int
        number of partitions

    Returns
    -------
    l_start_blockrow : list
        list of processes starting block index
    l_partitions_blocksizes : list
        list of processes partition size
    """

    l_start_blockrow = []
    l_partitions_blocksizes = []

    block_stride = n_blocks // n_partitions

    for process_i in range(n_partitions - 1):
        start_blockrow = process_i * block_stride
        stop_blockrow = (process_i + 1) * block_stride

        l_start_blockrow.append(start_blockrow)
        l_partitions_blocksizes.append(stop_blockrow - start_blockrow)

    start_blockrow = (n_partitions - 1) * block_stride
    stop_blockrow = n_blocks

    l_start_blockrow.append(start_blockrow)
    l_partitions_blocksizes.append(stop_blockrow - start_blockrow)

    return l_start_blockrow, l_partitions_blocksizes


def get_process_rowblock_index(
    start_blockrow: int, partitions_blocksizes: int
) -> [int, int]:

    process_top_blockrow = start_blockrow
    process_bottom_blockrow = process_top_blockrow + partitions_blocksizes

    return process_top_blockrow, process_bottom_blockrow
