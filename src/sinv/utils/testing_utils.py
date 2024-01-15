"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import matplotlib.pyplot as plt

from read_load_convert_utils import read_block_tridiagonal_matrix


def create_block_tridiagonal_matrix(
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
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
        
    diagonal_blocks = np.random.rand(n_blocks*blocksize, blocksize)
    upper_diagonal_blocks = np.random.rand((n_blocks-1)*blocksize, blocksize)
    lower_diagonal_blocks = np.random.rand((n_blocks-1)*blocksize, blocksize)
    
    if is_complex:
        diagonal_blocks = diagonal_blocks + 1j * np.random.rand(n_blocks*blocksize, blocksize)
        upper_diagonal_blocks = upper_diagonal_blocks + 1j * np.random.rand((n_blocks-1)*blocksize, blocksize)
        lower_diagonal_blocks = lower_diagonal_blocks + 1j * np.random.rand((n_blocks-1)*blocksize, blocksize)
        
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

