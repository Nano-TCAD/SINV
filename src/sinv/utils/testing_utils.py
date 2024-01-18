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
        
    diagonal_blocks = np.zeros((n_blocks, blocksize, blocksize))
    upper_diagonal_blocks = np.zeros((n_blocks-1, blocksize, blocksize))
    lower_diagonal_blocks = np.zeros((n_blocks-1, blocksize, blocksize))
    
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
    
    """ if is_diagonally_dominant:
        for i in range(n_blocks):
            if i == 0:
                diagonal_blocks[i, blocksize//2, blocksize//2] = np.sum(np.abs(diagonal_blocks[i, :, :]), axis=0) + np.sum(np.abs(upper_diagonal_blocks[i, :, :]), axis=0)
     """
    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks


def save_block_tridigonal_matrix(
    diagonal_blocks: np.ndarray,
    upper_diagonal_blocks: np.ndarray,
    lower_diagonal_blocks: np.ndarray,
    file_path: str,
) -> None:
    """ Save a block tridiagonal matrix to a file.

    Parameters
    ----------
    diagonal_blocks : np.ndarray
        The diagonal blocks of the matrix.
    upper_diagonal_blocks : np.ndarray
        The upper diagonal blocks of the matrix.
    lower_diagonal_blocks : np.ndarray
        The lower diagonal blocks of the matrix.
    file_path : str
        Path to the file where to save the matrix.
    
    """
    
    np.savez(file_path, diagonal_blocks=diagonal_blocks, upper_diagonal_blocks=upper_diagonal_blocks, lower_diagonal_blocks=lower_diagonal_blocks)


def cut_dense_to_block_tridiagonal(
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

