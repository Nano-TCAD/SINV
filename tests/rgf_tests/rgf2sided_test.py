"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.algorithms import rgf2s

import numpy as np
import pytest
from mpi4py import MPI

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
================================================ """
@pytest.mark.mpi(min_size=2)
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
def test_rgf2sided(
    is_complex: bool,
    is_symmetric: bool,
    matrix_size: int,
    blocksize: int
):
    """ Test the RGF 2-Sided algorithm. """
    
    
    
    pass
    


