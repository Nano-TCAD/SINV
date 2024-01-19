"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.utils import gmu
from sinv.utils import rlcu

import numpy as np
import pytest


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
    

if __name__ == "__main__":
    #test_create_block_tridiagonal_matrix(3, 3, False)
    test_create_block_tridiagonal_matrix(3, 3, True)