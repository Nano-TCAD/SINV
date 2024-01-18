"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from os import environ
environ['OMP_NUM_THREADS'] = '1'

from sinv.algorithms import bcr_p
from sinv import utils

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
| Test 1  |     2x2      |     1     |    2    |
| Test 2  |     4x4      |     2     |    2    |
| Test 3  |     6x6      |     3     |    2    |
================================================
| Test 4  |     3x3      |     1     |    3    |
| Test 5  |     6x6      |     2     |    3    |
| Test 6  |     9x9      |     3     |    3    |
================================================
| Test 7  |   128x128    |     8     |   16    |
| Test 8  |   128x128    |    16     |    8    |
| Test 9  |   128x128    |    32     |    4    |
================================================ """
@pytest.mark.mpi(min_size=2)
@pytest.mark.parametrize("is_complex", [False, True])
@pytest.mark.parametrize("is_symmetric", [False, True])
@pytest.mark.parametrize(
    "matrix_size, blocksize",
    [
        (9, 1),
        (18, 2),
        (27, 3),
        (12, 1),
        (24, 2),
        (36, 3),
        (256, 8),
        (240, 10),
        (144, 12),
    ]
)
def test_bcrp(
    is_complex: bool,
    is_symmetric: bool,
    matrix_size: int,
    blocksize: int
):
    """ Test the BCR-P algorithm. """
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()

    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrix_size, bandwidth, is_complex, is_symmetric, SEED)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)
        assert np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)
        assert np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)
        
        
        