"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.algorithms import rgf
from sinv.utils import testu
from sinv.utils import rlcu

# from quasi.bsparse import bdia, bsr
# from quasi.bsparse._base import bsparse

import bsparse as bsp


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
@pytest.mark.parametrize("is_complex", [False, True])
# @pytest.mark.parametrize("is_symmetric", [False])
@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize(
    "n_blocks, blocksize",
    [
        (2, 1),
        (2, 2),
        (2, 3),
        (3, 1),
        (3, 2),
        (3, 3),
        (16, 8),
        (8, 16),
        (4, 32),
    ],
)
def test_rgf_BDIA(is_complex: bool, is_symmetric: bool, n_blocks: int, blocksize: int):
    """Test the RGF algorithm on a bsparse.BDIA matrix."""

    (
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
    ) = testu.create_block_tridiagonal_matrix(
        n_blocks, blocksize, is_complex, is_symmetric
    )

    bsparse_matrix = rlcu.block_tridiagonal_to_BDIA(
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
        blocksize,
        is_symmetric,
    )

    X_refsol = np.linalg.inv(bsparse_matrix.toarray())
    X_refsol = testu.cut_dense_to_block_tridiagonal(X_refsol, blocksize)

    X_rgf = rgf.rgf(bsparse_matrix, is_symmetric)

    # Check main diagonal
    for i in range(X_rgf.bshape[0]):
        ii = slice(i * blocksize, (i + 1) * blocksize)
        asser = np.allclose(X_rgf[i, i], X_refsol[ii, ii])

    # Check off-diagonals
    for i in range(X_rgf.bshape[0] - 1):
        ii = slice(i * blocksize, (i + 1) * blocksize)
        jj = slice((i + 1) * blocksize, (i + 2) * blocksize)
        assert np.allclose(X_rgf[i + 1, i], X_refsol[jj, ii])
        assert np.allclose(X_rgf[i, i + 1], X_refsol[ii, jj])

    # assert np.allclose(X_rgf.toarray(), X_refsol)
