"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from os import environ

environ["OMP_NUM_THREADS"] = "1"

from sinv.algorithms import pdiv_lm
from sinv.algorithms import pdiv_u
from sinv import utils

import numpy as np
import math
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
        (2, 1),
        (4, 2),
        (6, 3),
        (3, 1),
        (6, 2),
        (9, 3),
        (128, 8),
        (128, 16),
        (128, 32),
    ],
)
def test_pdiv(is_complex: bool, is_symmetric: bool, matrix_size: int, blocksize: int):
    """Test the PDIV algorithm."""
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()

    nblocks = int(np.ceil(matrix_size / blocksize))
    if math.log2(comm_size).is_integer() and comm_size <= nblocks:

        bandwidth = np.ceil(blocksize / 2)
        A = utils.matu.generateBandedDiagonalMatrix(
            matrix_size, bandwidth, is_complex, is_symmetric, SEED
        )

        # PDIV worflow
        pdiv_u.check_multiprocessing(comm_size)
        pdiv_u.check_input(A, blocksize, comm_size)

        l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(
            A, comm_size, blocksize
        )
        K_i, Bu_i, Bl_i = pdiv_u.partition_subdomain(
            A, l_start_blockrow, l_partitions_blocksizes, blocksize
        )

        K_local = K_i[comm_rank]
        X_diagblk, X_upperblk, X_lowerblk = pdiv_lm.pdiv_localmap(
            K_local, Bu_i, Bl_i, blocksize
        )

        # Extract local reference solution
        A_refsol = np.linalg.inv(A)

        X_refsol_diagblk = [
            np.zeros((blocksize, blocksize), dtype=A_refsol.dtype)
            for i in range(0, l_partitions_blocksizes[comm_rank], 1)
        ]
        X_refsol_upperblk = [
            np.zeros((blocksize, blocksize), dtype=A_refsol.dtype)
            for i in range(0, l_partitions_blocksizes[comm_rank], 1)
        ]
        X_refsol_lowerblk = [
            np.zeros((blocksize, blocksize), dtype=A_refsol.dtype)
            for i in range(0, l_partitions_blocksizes[comm_rank], 1)
        ]

        start_localpart_blockindex = l_start_blockrow[comm_rank]
        for i in range(0, l_partitions_blocksizes[comm_rank], 1):
            i_part = i + start_localpart_blockindex
            X_refsol_diagblk[i] = A_refsol[
                i_part * blocksize : (i_part + 1) * blocksize,
                i_part * blocksize : (i_part + 1) * blocksize,
            ]
            if i_part < nblocks - 1:
                X_refsol_upperblk[i] = A_refsol[
                    i_part * blocksize : (i_part + 1) * blocksize,
                    (i_part + 1) * blocksize : (i_part + 2) * blocksize,
                ]
                X_refsol_lowerblk[i] = A_refsol[
                    (i_part + 1) * blocksize : (i_part + 2) * blocksize,
                    i_part * blocksize : (i_part + 1) * blocksize,
                ]
            else:
                X_refsol_upperblk[i] = np.zeros(
                    (blocksize, blocksize), dtype=A_refsol.dtype
                )
                X_refsol_lowerblk[i] = np.zeros(
                    (blocksize, blocksize), dtype=A_refsol.dtype
                )

        assert np.allclose(X_diagblk, X_refsol_diagblk)
        assert np.allclose(X_upperblk, X_refsol_upperblk)
        assert np.allclose(X_lowerblk, X_refsol_lowerblk)
