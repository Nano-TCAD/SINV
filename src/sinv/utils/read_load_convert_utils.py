"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich. All rights reserved.
"""

import numpy as np
import bsparse as bsp


def save_block_tridigonal_matrix(
    file_path: str,
    diagonal_blocks: np.ndarray,
    upper_diagonal_blocks: np.ndarray,
    lower_diagonal_blocks: np.ndarray,
    is_symmetric: bool = False,
) -> None:
    """
    Save a block tridiagonal matrix to a file.

    Parameters
    ----------
    file_path : str
        Path to the file where to save the matrix.
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.
    is_symmetric : bool, optional
        Whether the matrix is symmetric or not. The default is False.

    """

    if is_symmetric:
        np.savez(
            file_path,
            diagonal_blocks=diagonal_blocks,
            upper_diagonal_blocks=upper_diagonal_blocks,
        )
    else:
        np.savez(
            file_path,
            diagonal_blocks=diagonal_blocks,
            upper_diagonal_blocks=upper_diagonal_blocks,
            lower_diagonal_blocks=lower_diagonal_blocks,
        )


def read_block_tridiagonal_matrix(
    file_path: str,
    is_symmetric: bool = False,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Read a block tridiagonal matrix stored in .npz format. First stored array
    should contain the diagonal blocks, second array the upper diagonal blocks
    and third array the lower diagonal blocks.

    Parameters
    ----------
    file_path : str
        Path to the file where the matrix is stored.
    is_symmetric : bool, optional
        Whether the matrix is symmetric or not. The default is False.

    Returns
    -------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.

    """

    matrix = np.load(file_path)

    if is_symmetric:
        diagonal_blocks = matrix[matrix.files[0]]
        upper_diagonal_blocks = matrix[matrix.files[1]]
        lower_diagonal_blocks = np.zeros_like(upper_diagonal_blocks)

        n_blocks = diagonal_blocks.shape[0]
        for i in range(n_blocks - 1):
            lower_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :].T

    else:
        diagonal_blocks = matrix[matrix.files[0]]
        upper_diagonal_blocks = matrix[matrix.files[1]]
        lower_diagonal_blocks = matrix[matrix.files[2]]

    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def read_local_block_tridiagonal_partition(
    file_path: str,
    start_blockrow: int,
    partition_size: int,
    is_symmetric: bool = False,
    include_bridges_blocks: bool = False,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """Read a local partition of a block tridiagonal matrix stored in .npz format.
    First stored array should contain the diagonal blocks, second array the upper
    diagonal blocks and third array the lower diagonal blocks.

    Optionally include the bridge blocks (upper and lower) conecting the partition
    to it's neighbours.

    Parameters
    ----------
    file_path : str
        Path to the file where the matrix is stored.
    start_blockrow : int
        Index of the first blockrow of the local partition in the global matrix.
    partition_size : int
        Size of the local partition.
    is_symmetric : bool, optional
        Whether the matrix is symmetric or not. The default is False.
    include_bridges_blocks : bool, optional
        Whether to include the bridge blocks (upper and lower) conecting the
        partition to it's neighbours. The default is False.

    Returns
    -------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.
    """

    matrix = np.load(file_path)

    end_blockrow = start_blockrow + partition_size

    diagonal_blocks = matrix[matrix.files[0]][start_blockrow:end_blockrow, :, :]

    if include_bridges_blocks:
        start_upper_blockrow = max(0, start_blockrow - 1)
        start_lower_blockrow = max(0, start_blockrow - 1)

        n_blocks = matrix[matrix.files[0]].shape[0]

        stop_upper_blockrow = min(start_blockrow + partition_size, n_blocks)
        stop_lower_blockrow = min(start_blockrow + partition_size, n_blocks)

        upper_diagonal_blocks = matrix[matrix.files[1]][
            start_upper_blockrow:stop_upper_blockrow, :, :
        ]
        if is_symmetric:
            lower_diagonal_blocks = np.zeros_like(upper_diagonal_blocks)
            for i in range(upper_diagonal_blocks.shape[0]):
                lower_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :].T
        else:
            lower_diagonal_blocks = matrix[matrix.files[2]][
                start_lower_blockrow:stop_lower_blockrow, :, :
            ]

    else:
        upper_diagonal_blocks = matrix[matrix.files[1]][
            start_blockrow : end_blockrow - 1, :, :
        ]
        if is_symmetric:
            lower_diagonal_blocks = np.zeros_like(upper_diagonal_blocks)
            for i in range(upper_diagonal_blocks.shape[0]):
                lower_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :].T
        else:
            lower_diagonal_blocks = matrix[matrix.files[2]][
                start_blockrow : end_blockrow - 1, :, :
            ]

    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def slice_partition(
    A: bsp,
    start_blockrow: int,
    partition_size: int,
    include_bridges_blocks: bool = False,
) -> bsp:

    top_blockrow = start_blockrow
    bottom_blockrow = start_blockrow + partition_size

    if include_bridges_blocks:
        top_blockrow = max(0, start_blockrow - 1)
        bottom_blockrow = min(start_blockrow + partition_size + 1, A.bshape[0])

    return A[
        top_blockrow:bottom_blockrow,
        top_blockrow:bottom_blockrow,
    ]


def block_tridiagonal_to_BDIA(
    diagonal_blocks: np.ndarray,
    upper_diagonal_blocks: np.ndarray,
    lower_diagonal_blocks: np.ndarray,
    symmetry: str = None,
) -> bsp.BDIA:
    """Convert a block tridiagonal matrix to a bsparse.BDIA matrix.

    Parameters
    ----------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the block tridiagonal matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the block tridiagonal matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the block tridiagonal matrix.
    blocksize : int
        The size of each block in the block tridiagonal matrix.

    Returns
    -------
    bsparse_matrix : bdia
        The bsparse matrix representation of the block tridiagonal matrix.

    Raises
    ------
    ValueError
        If the given symmetry is not supported. It should either be 'symmetric'
        or 'hermitian'.
    """
    bsparse_matrix: bsp.BDIA

    if symmetry == None:
        offsets = [0, 1, -1]
        bsparse_matrix = bsp.BDIA(
            offsets,
            [diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks],
            symmetry=symmetry,
        )
    elif symmetry in ("symmetric", "hermitian"):
        offsets = [0, 1]
        bsparse_matrix = bsp.BDIA(
            offsets, [diagonal_blocks, upper_diagonal_blocks], symmetry=symmetry
        )
    else:
        raise ValueError("Invalid symmetry.")

    return bsparse_matrix
