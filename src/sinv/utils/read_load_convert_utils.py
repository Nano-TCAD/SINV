"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import bsparse as bsp


def read_block_tridiagonal_matrix(
    file_path: str,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Read a block tridiagonal matrix stored in .npz format. First stored array 
    should contain the diagonal blocks, second array the upper diagonal blocks 
    and third array the lower diagonal blocks.
    
    Parameters
    ----------
    file_path : str
        Path to the file where the matrix is stored.
        
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
    
    diagonal_blocks = matrix[matrix.files[0]]
    upper_diagonal_blocks = matrix[matrix.files[1]]
    lower_diagonal_blocks = matrix[matrix.files[2]]
    
    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def read_local_block_tridiagonal_partition(
    file_path: str,
    start_blockrow: int,
    partition_size: int,
    blocksize: int,
    include_bridges_blocks: bool = False,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Read a local partition of a block tridiagonal matrix stored in .npz format.
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
    blocksize : int
        Size of the blocks.
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
    
    diagonal_blocks = matrix[matrix.files[0]][start_blockrow*blocksize:(start_blockrow+partition_size)*blocksize,:]

    off_diagonal_number_of_blocks = partition_size - 1 
    
    start_upper_blockrow = start_blockrow
    start_lower_blockrow = start_blockrow
    
    stop_upper_blockrow = start_blockrow + off_diagonal_number_of_blocks
    stop_lower_blockrow = start_blockrow + off_diagonal_number_of_blocks
    
    if include_bridges_blocks:
        start_upper_blockrow = max(0, start_blockrow-1)
        start_lower_blockrow = max(0, start_blockrow-1)
        
        stop_upper_blockrow = min(start_blockrow + partition_size, matrix[matrix.files[1]].shape[0]//blocksize)
        stop_lower_blockrow = min(start_blockrow + partition_size, matrix[matrix.files[2]].shape[0]//blocksize)

    upper_diagonal_blocks = matrix[matrix.files[1]][start_upper_blockrow*blocksize:stop_upper_blockrow*blocksize,:]
    lower_diagonal_blocks = matrix[matrix.files[2]][start_lower_blockrow*blocksize:stop_lower_blockrow*blocksize,:]

    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def block_tridiagonal_to_BDIA(
    diagonal_blocks: np.ndarray,
    upper_diagonal_blocks: np.ndarray,
    lower_diagonal_blocks: np.ndarray,
    blocksize: int,
    symmetry: str = None,
) -> bsp.BDIA:
    """ Convert a block tridiagonal matrix to a bsparse.BDIA matrix.

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
        
    offsets = [0, 1, -1]    
    
    
    if symmetry != 'symmetric' or symmetry != 'hermitian':
        raise ValueError("The given symmetry is not supported.")
        
    bsparse_matrix = bsp.BDIA(offsets, [diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks], symmetry=symmetry)
    
    return bsparse_matrix
