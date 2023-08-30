"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the RGF algorithm. 
- Complex and real matrices
- Symmetric and non-symmetric matrices
================================================
| Test n  | Matrice size | Blocksize | nblocks | 
================================================
| Test 1  |     1x1      |     1     |    1    |
| Test 2  |     2x2      |     2     |    1    |
| Test 3  |     3x3      |     3     |    1    |
================================================
| Test 4  |     2x2      |     1     |    2    |
| Test 5  |     4x4      |     2     |    2    |
| Test 6  |     6x6      |     3     |    2    |
================================================
| Test 7  |     3x3      |     1     |    3    |
| Test 8  |     6x6      |     2     |    3    |
| Test 9  |     9x9      |     3     |    3    |
================================================
| Test 10 |   128x128    |     8     |    16   |
| Test 11 |   128x128    |     16    |    8    |
| Test 12 |   128x128    |     32    |    4    |
================================================

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import pytest

from quasi.bsparse import bdia, bsr, vbdia, vbsr
from quasi.bsparse._base import bsparse

from .rgf import rgf

SEED = 63
np.random.seed(SEED)


@pytest.mark.parametrize("bsparse_type", [bdia, bsr])
@pytest.mark.parametrize(
    "matrix_parameters",
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
    matrix_parameters: [int, int]
):
    """ Test the RGF algorithm on a bsparse matrix. """
    matrix_size, blocksize = matrix_parameters
    
    n_blocks = matrix_size // blocksize
    sub_mat_size = min(matrix_size, 2*blocksize)
    n_sub_mat = max(1, n_blocks - 1)        
    overlap = blocksize    
        
    A = bsparse_type.diag(
        [np.random.rand(sub_mat_size, sub_mat_size) for _ in range(n_sub_mat)], 
        blocksize, overlap
    )
    
    X_refsol = np.linalg.inv(A.toarray())
    X_rgf = rgf(A)
    
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
        


@pytest.mark.parametrize("vbsparse_type", [vbdia, vbsr])
@pytest.mark.parametrize(
    "matrix_diag_blocksizes",
    [
        [1], 
        [1, 1],
        [2, 1],
        [1, 2],
        [2, 1, 2],
        [1, 2, 1],
        [3, 2, 1, 2, 3]
    ]
)
def test_rgf_vbsparse(
    vbsparse_type: bsparse,
    matrix_diag_blocksizes: [int]
):
    """ Test the RGF algorithm on a vbsparse matrix. """
    pass