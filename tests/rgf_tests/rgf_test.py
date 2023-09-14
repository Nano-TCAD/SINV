"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.algorithms import rgf
from sinv.utils import gen_mat

from quasi.bsparse import bdia, bsr
from quasi.bsparse._base import bsparse

import numpy as np
import pytest

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
@pytest.mark.mpi_skip()
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
        [gen_mat.generateRandomNumpyMat(sub_mat_size, is_complex, is_symmetric, SEED) 
        for _ in range(n_sub_mat)], 
        blocksize, overlap
    )
    
    X_refsol = np.linalg.inv(A.toarray())
    X_rgf = rgf.rgf(A, is_symmetric)
    
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

