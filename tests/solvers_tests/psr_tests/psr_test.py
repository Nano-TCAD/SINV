"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from os import environ

environ["OMP_NUM_THREADS"] = "1"

from sinv.algorithms import psr_s
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
| Test 1  |     9x9      |     1     |    9    | 
| Test 2  |    18x18     |     2     |    9    |
| Test 3  |    27x27     |     3     |    9    |
================================================
| Test 4  |    12x12     |     1     |   12    | 
| Test 5  |    24x24     |     2     |   12    |
| Test 6  |    36x36     |     3     |   12    |
================================================
| Test 7  |   256x256    |     8     |   32    |
| Test 8  |   240x240    |    10     |   24    |
| Test 9  |   144x144    |    12     |   12    |
================================================ """


@pytest.mark.mpi(min_size=3)
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
    ],
)
def test_psr(is_complex: bool, is_symmetric: bool, matrix_size: int, blocksize: int):
    """Test the PSR algorithm."""
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    nblocks = matrix_size // blocksize
    bandwidth = np.ceil(blocksize / 2)

    if nblocks >= 3 * comm_size:
        A = utils.matu.generateBandedDiagonalMatrix(
            matrix_size, bandwidth, is_complex, is_symmetric, SEED
        )

        A_refsol = np.linalg.inv(A)
        (
            A_refsol_bloc_diag,
            A_refsol_bloc_upper,
            A_refsol_bloc_lower,
        ) = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

        A_psr = psr_s.psr_seqsolve(A, blocksize)
        (
            A_psr_bloc_diag,
            A_psr_bloc_upper,
            A_psr_bloc_lower,
        ) = utils.matu.convertDenseToBlkTridiag(A_psr, blocksize)

        if comm_rank == 0:
            assert (
                np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)
            )
