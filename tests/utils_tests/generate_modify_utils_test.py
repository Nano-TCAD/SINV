"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from sinv.utils import gmu

import numpy as np
import pytest


@pytest.mark.parametrize(
    "n_blocks",
    [
        pytest.param(2, id="2 blocks"),
        pytest.param(3, id="3 blocks"),
        pytest.param(4, id="4 blocks"),
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        pytest.param(1, id="1 blocksize"),
        pytest.param(2, id="2 blocksize"),
        pytest.param(3, id="3 blocksize"),
        pytest.param(4, id="4 blocksize"),
    ],
)
@pytest.mark.parametrize(
    "is_complex",
    [
        pytest.param(False, id="real"),
        pytest.param(True, id="complex"),
    ],
)
@pytest.mark.parametrize(
    "is_symmetric",
    [
        pytest.param(False, id="non-symmetric"),
        pytest.param(True, id="symmetric"),
    ],
)
@pytest.mark.parametrize(
    "is_diagonally_dominant",
    [
        pytest.param(False, id="non-diagonally dominant"),
        pytest.param(True, id="diagonally dominant"),
    ],
)
def test_create_block_tridiagonal_matrix(
    n_blocks: int,
    blocksize: int,
    is_complex: bool,
    is_symmetric: bool,
    is_diagonally_dominant: bool,
):
    SEED = 63

    (
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
    ) = gmu.create_block_tridiagonal_matrix(
        n_blocks, blocksize, is_complex, is_symmetric, is_diagonally_dominant, seed=SEED
    )

    if is_symmetric:
        for i in range(n_blocks):
            assert np.allclose(diagonal_blocks[i, :, :], diagonal_blocks[i, :, :].T)
            if i < n_blocks - 1:
                assert np.allclose(
                    upper_diagonal_blocks[i, :, :], lower_diagonal_blocks[i, :, :].T
                )

    if is_complex:
        assert np.iscomplexobj(diagonal_blocks)
        assert np.iscomplexobj(upper_diagonal_blocks)
        assert np.iscomplexobj(lower_diagonal_blocks)

    if is_diagonally_dominant:
        for i in range(n_blocks):
            for j in range(blocksize):
                row_sum = np.sum(np.abs(diagonal_blocks[i, j, :])) - np.abs(
                    diagonal_blocks[i, j, j]
                )
                if i < n_blocks - 1:
                    row_sum += np.sum(np.abs(upper_diagonal_blocks[i, j, :]))
                if i > 0:
                    row_sum += np.sum(np.abs(lower_diagonal_blocks[i - 1, j, :]))

            assert np.abs(diagonal_blocks[i, j, j]) > row_sum


@pytest.mark.parametrize(
    "n_blocks",
    [
        pytest.param(2, id="2 blocks"),
        pytest.param(3, id="3 blocks"),
        pytest.param(4, id="4 blocks"),
    ],
)
@pytest.mark.parametrize(
    "blocksize",
    [
        pytest.param(1, id="1 blocksize"),
        pytest.param(2, id="2 blocksize"),
        pytest.param(3, id="3 blocksize"),
        pytest.param(4, id="4 blocksize"),
    ],
)
def test_zero_out_dense_to_block_tridiagonal(
    n_blocks: int,
    blocksize: int,
):
    SEED = 63

    dense_matrix = np.random.rand(n_blocks * blocksize, n_blocks * blocksize)

    tridiag_cut_dense_matrix = gmu.zero_out_dense_to_block_tridiagonal(
        dense_matrix, blocksize
    )

    for i in range(n_blocks):
        for j in range(i + 2, n_blocks):
            assert np.any(
                tridiag_cut_dense_matrix[
                    i * blocksize : (i + 1) * blocksize,
                    j * blocksize : (j + 1) * blocksize,
                ]
                == np.zeros((blocksize, blocksize))
            )
            assert np.any(
                tridiag_cut_dense_matrix[
                    j * blocksize : (j + 1) * blocksize,
                    i * blocksize : (i + 1) * blocksize,
                ]
                == np.zeros((blocksize, blocksize))
            )
