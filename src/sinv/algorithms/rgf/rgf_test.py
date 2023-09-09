"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import pytest

from quasi.bsparse import bdia, bsr, vbdia, vbsr
from quasi.bsparse._base import bsparse

from .rgf import rgf



def generateRandomNumpyMat(matrice_size: int, 
                           is_complex: bool = False,
                           is_symmetric: bool = False,
                           seed: int = None) -> np.ndarray:
    """ Generate a dense matrix of shape: (matrice_size x matrice_size) filled 
    with random numbers. The matrice may be complex or real valued.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    is_symmetric : bool, optional
        Whether the matrice should be symmetric or not. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    if seed is not None:
        np.random.seed(seed)
        
    A = np.zeros((matrice_size, matrice_size))

    if is_complex:
        A = np.random.rand(matrice_size, matrice_size)\
               + 1j * np.random.rand(matrice_size, matrice_size)
    else:
        A = np.random.rand(matrice_size, matrice_size)
        
    if is_symmetric:
        A = A + A.T
        
    return A



SEED = 63



""" Uniform blocksize tests cases
- Complex and real matrices
- Symmetric and non-symmetric matrices
================================================
| Test n  | Matrice size | Blocksize | nblocks | 
================================================
| Test 1  |     1x1      |     1     |    1    |
| Test 2  |     2x2      |     2     |    1    |
| Test 3  |     3x3      |     3     |    1    |
| Test 4  |     2x2      |     1     |    2    |
| Test 5  |     4x4      |     2     |    2    |
| Test 6  |     6x6      |     3     |    2    |
| Test 7  |     3x3      |     1     |    3    |
| Test 8  |     6x6      |     2     |    3    |
| Test 9  |     9x9      |     3     |    3    |
| Test 10 |   128x128    |     8     |    16   |
| Test 11 |   128x128    |     16    |    8    |
| Test 12 |   128x128    |     32    |    4    |
================================================ """
@pytest.mark.parametrize("bsparse_type", [bdia, bsr])
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize(
    "matrix_size, blocksize",
    [
        (1, 1), 
        (2, 2),
        (3, 3),
        (2, 1),
        (4, 2),
        (6, 3),
        (3, 1),
        (6, 2),
        (9, 3),
        (128, 8),
        (128, 16),
        (128, 32),
    ]
)
def test_rgf_bsparse(
    bsparse_type: bsparse, 
    is_complex: bool,
    is_symmetric: bool,
    matrix_size: int,
    blocksize: int
):
    """ Test the RGF algorithm on a bsparse matrix. """
    
    n_blocks = matrix_size // blocksize
    sub_mat_size = min(matrix_size, 2*blocksize)
    n_sub_mat = max(1, n_blocks - 1)        
    overlap = blocksize    
        
    A = bsparse_type.diag(
        [generateRandomNumpyMat(sub_mat_size, is_complex, is_symmetric, SEED) 
        for _ in range(n_sub_mat)], 
        blocksize, overlap
    )
    
    X_refsol = np.linalg.inv(A.toarray())
    X_rgf = rgf(A, is_symmetric)
    
    # Check main diagonal
    for i in range(X_rgf.blockorder):
        ii = slice(i*blocksize, (i+1)*blocksize)
        asser = np.allclose(X_rgf.blocks[i, i], X_refsol[ii, ii])
        
    # Check off-diagonals
    for i in range(X_rgf.blockorder - 1):
        ii = slice(i * blocksize, (i + 1) * blocksize)
        jj = slice((i + 1) * blocksize, (i + 2) * blocksize)
        assert np.allclose(X_rgf.blocks[i + 1, i], X_refsol[jj, ii])
        assert np.allclose(X_rgf.blocks[i, i + 1], X_refsol[ii, jj])



""" Variable blocksizes tests cases
- Complex and real matrices
- Symmetric and non-symmetric matrices
====================================================
| Test n  | Matrice size |    Blocksize    | nblocks | 
====================================================
| Test 1  |     1x1      | [1]             |    1    |
| Test 2  |     2x2      | [1, 1]          |    2    |
| Test 3  |     3x3      | [2, 1]          |    2    |
| Test 4  |     3x3      | [1, 2]          |    2    |
| Test 5  |     5x5      | [2, 1, 2]       |    3    |
| Test 5  |     4x4      | [1, 2, 1]       |    3    |
| Test 6  |    11x11     | [3, 2, 1, 2, 3] |    5    |
==================================================== """
@pytest.mark.parametrize("vbsparse_type", [vbdia, vbsr])
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize(
    "matrix_diag_blocksizes, offset",
    [
        ([1], 0), 
        ([1, 1], 1),
        ([2, 1], 1),
        ([1, 2], 1),
        ([2, 1, 2], 1),
        ([1, 2, 1], 1),
        ([3, 2, 1, 2, 3], 2),
    ]
)
def test_rgf_vbsparse(
    vbsparse_type: bsparse,
    is_complex: bool,
    is_symmetric: bool,
    matrix_diag_blocksizes: [int],
    offset: int
):
    """ Test the RGF algorithm on a vbsparse matrix. """
    
    blocksizes = [i + offset for i in matrix_diag_blocksizes]
    
    A = vbsparse_type.diag(
        [generateRandomNumpyMat(blocksizes[i], is_complex, is_symmetric, SEED) 
        for i in range(0, len(blocksizes))], offset
    )
    
    X_refsol = np.linalg.inv(A.toarray())
    X_rgf = rgf(A, is_symmetric)
    
    blockoffsets = np.concatenate([[0], X_rgf.blocksizes.cumsum()])

    # Main Diagonal.
    for i in range(X_rgf.blockorder):
        ii = slice(blockoffsets[i], blockoffsets[i + 1])
        assert np.allclose(X_rgf.blocks[i, i], X_refsol[ii, ii])

    # Off-Diagonals.
    for i in range(X_rgf.blockorder - 1):
        ii = slice(blockoffsets[i], blockoffsets[i + 1])
        jj = slice(blockoffsets[i + 1], blockoffsets[i + 2])
        assert np.allclose(X_rgf.blocks[i + 1, i], X_refsol[jj, ii])
        assert np.allclose(X_rgf.blocks[i, i + 1], X_refsol[ii, jj])
        
        