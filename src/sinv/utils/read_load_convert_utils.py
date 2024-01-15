"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def read_block_tridiagonal_matrix(
    file_path: str,
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ Read a block tridiagonal matrix stored in .npz format.
    
    Parameters
    ----------
    file_path : str
        Path to the file where the matrix is stored.
    n_blocks : int
        Number of blocks of the matrix.
    blocksize : int
        Size of the blocks.
    is_complex : bool
        Whether the matrix is complex or real valued.
        
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
    
    diagonal_blocks = matrix["diagonal_blocks"]
    upper_diagonal_blocks = matrix["upper_diagonal_blocks"]
    lower_diagonal_blocks = matrix["lower_diagonal_blocks"]
    
    return diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks



def read_local_block_tridiagonal_partition(
    file_path: str,
    start_blockrow: int,
    partition_size: int,
    include_bridges_blocks: bool,
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    pass


def read_csr_matrix(
    file_path: str,
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    pass


def read_local_csr_partition(
    file_path: str,
    start_blockrow: int,
    partition_size: int,
    include_bridges_blocks: bool,
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
) -> [np.ndarray, np.ndarray, np.ndarray]:
    pass


""" def block_tridiagonal_to_bsparse(
    diagonal_blocks: np.ndarray,
    upper_diagonal_blocks: np.ndarray,
    lower_diagonal_blocks: np.ndarray,
    blocksize: int,
) -> bsparse:
    pass """