"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.utils import gmu
from sinv.utils import rlcu

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
    "is_symmetric", 
    [
        pytest.param(False, id="non-symmetric"),
        pytest.param(True, id="symmetric"),
    ]
)
def test_create_block_tridiagonal_matrix(
    n_blocks: int,
    blocksize: int,
    is_symmetric: bool,
):
    SEED = 63 
    
    diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = gmu.create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex=False, is_symmetric=is_symmetric, is_diagonally_dominant=False, seed=SEED)
    
    rlcu.save_block_tridigonal_matrix("test_matrix.npz", diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, is_symmetric=is_symmetric)
    
    loaded_diagonal_blocks, loaded_upper_diagonal_blocks, loaded_lower_diagonal_blocks = rlcu.read_block_tridiagonal_matrix("test_matrix.npz", is_symmetric=is_symmetric)
    
    assert np.allclose(diagonal_blocks, loaded_diagonal_blocks)
    assert np.allclose(upper_diagonal_blocks, loaded_upper_diagonal_blocks)
    assert np.allclose(lower_diagonal_blocks, loaded_lower_diagonal_blocks)