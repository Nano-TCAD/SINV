"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np


def create_block_tridiagonal_matrix(
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
    is_symmetric: bool = False,
    is_diagonally_dominant: bool = True,
    seed: int = None,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Create a random block tridiagonal matrix.
    
    Parameters
    ----------
    n_blocks : int
        Number of blocks in the matrix.
    blocksize : int
        Size of the blocks.
    is_complex : bool
        Whether the matrix should be complex or real valued.
    is_symmetric : bool, optional
        Whether the matrix should be symmetric or not. The default is False.
    is_diagonally_dominant : bool, optional
        Whether the matrix should be diagonally dominant or not. The default is True.
    seed : int, optional
        Seed for the random number generator. The default is no seed.

    Returns
    -------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.
    """
    
    if seed is not None:
        np.random.seed(seed)
        
    dtype = np.float64
    if is_complex:
        dtype = np.complex128    
        
    diagonal_blocks = np.zeros((n_blocks, blocksize, blocksize), dtype=dtype)
    upper_diagonal_blocks = np.zeros((n_blocks-1, blocksize, blocksize), dtype=dtype)
    lower_diagonal_blocks = np.zeros((n_blocks-1, blocksize, blocksize), dtype=dtype)
    
    for i in range(n_blocks):
        diagonal_blocks[i, :, :] = np.random.rand(blocksize, blocksize)
        
        if is_complex:
            diagonal_blocks[i, :, :] = diagonal_blocks[i, :, :] + 1j * np.random.rand(blocksize, blocksize)
        
        if is_symmetric:
            diagonal_blocks[i, :, :] = diagonal_blocks[i, :, :] + diagonal_blocks[i, :, :].T
        
        if i < n_blocks-1:
            upper_diagonal_blocks[i, :, :] = np.random.rand(blocksize, blocksize)
            lower_diagonal_blocks[i, :, :] = np.random.rand(blocksize, blocksize)
            
            if is_complex:
                upper_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :] + 1j * np.random.rand(blocksize, blocksize)
                lower_diagonal_blocks[i, :, :] = lower_diagonal_blocks[i, :, :] + 1j * np.random.rand(blocksize, blocksize)
        
            if is_symmetric:
                upper_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :] + upper_diagonal_blocks[i, :, :].T
                lower_diagonal_blocks[i, :, :] = upper_diagonal_blocks[i, :, :].T
    
    if is_diagonally_dominant:
        diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = make_block_tridiagonal_matrix_diagonally_dominant(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks)

    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def make_block_tridiagonal_matrix_diagonally_dominant(
    diagonal_blocks: int, 
    upper_diagonal_blocks: int, 
    lower_diagonal_blocks: int,
    strictly_diagonally_dominant: bool = True,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """
    Make a block tridiagonal matrix diagonally dominant.

    This function modifies the diagonal blocks of a block tridiagonal matrix to make it diagonally dominant.
    Diagonal dominance ensures that the magnitude of the diagonal element of the diagonal block is greater than the sum of the magnitudes of the off-diagonal elements in the same block-row.

    Parameters
    ----------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.
    strictly_diagonally_dominant : bool, optional
        If True, the diagonal blocks will be strictly diagonally dominant, meaning the diagonal element will be greater than the sum of the magnitudes of the off-diagonal elements.
        If False, the diagonal blocks will be diagonally dominant, meaning the diagonal element will be equal to the sum of the magnitudes of the off-diagonal elements.
        Default is True.

    Returns
    -------
    np.ndarray, np.ndarray, np.ndarray
        The modified diagonal blocks, upper diagonal blocks, and lower diagonal blocks of the matrix.

    """
    n_blocks = diagonal_blocks.shape[0]
    blocksize = diagonal_blocks.shape[1]
    
    for i in range(n_blocks):
        for j in range(blocksize):
            row_sum = np.sum(np.abs(diagonal_blocks[i, j, :]))
            if i < n_blocks-1:
                row_sum += np.sum(np.abs(upper_diagonal_blocks[i, j, :]))
            if i > 0:
                row_sum += np.sum(np.abs(lower_diagonal_blocks[i-1, j, :]))
            
            if strictly_diagonally_dominant:
                diagonal_blocks[i, j, j] = row_sum
            else:
                diagonal_blocks[i, j, j] = row_sum - np.abs(diagonal_blocks[i, j, j])
                
    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks
                

def zero_out_dense_to_block_tridiagonal(
    dense_matrix: np.ndarray,
    blocksize: int,
) -> np.ndarray:
    """ Cut a dense matrix to a block tridiagonal matrix.

    Parameters
    ----------
    dense_matrix : np.ndarray
        The dense matrix to be cut.
    blocksize : int
        The size of the blocks in the resulting block tridiagonal matrix.

    Returns
    -------
    np.ndarray
        The resulting block tridiagonal matrix.

    Raises
    ------
    ValueError
        If the given matrix is not square or if the given blocksize does not divide the size of the matrix.

    """
    if dense_matrix.shape[0] != dense_matrix.shape[1]:
        raise ValueError("The given matrix is not square.")

    n_blocks = dense_matrix.shape[0] // blocksize

    if dense_matrix.shape[0] != n_blocks * blocksize:
        raise ValueError("The given blocksize does not divide the size of the matrix.")

    # Set to 0 all values outside the block tridiagonal
    for i in range(n_blocks):
        for j in range(n_blocks):
            if abs(i - j) > 1:
                dense_matrix[i * blocksize:(i + 1) * blocksize, j * blocksize:(j + 1) * blocksize] = 0

    return dense_matrix

