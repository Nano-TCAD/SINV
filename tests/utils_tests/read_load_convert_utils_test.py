"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.utils import gmu
from sinv.utils import rlcu
from sinv.utils import partu

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
def test_block_tridiagonal_to_BDIA(
    n_blocks: int,
    blocksize: int,
    is_symmetric: bool,
):
    SEED = 63 
    
    diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = gmu.create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex=False, is_symmetric=False, is_diagonally_dominant=False, seed=SEED)

    bdia_symmetry = None

    if is_symmetric:
        bdia_symmetry = "symmetric"

    BDIA = rlcu.block_tridiagonal_to_BDIA(diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, symmetry=bdia_symmetry)
    
    assert np.allclose(BDIA.diagonal(), diagonal_blocks)
    assert np.allclose(BDIA.diagonal(offset=1), upper_diagonal_blocks)
    if is_symmetric == False:
        assert np.allclose(BDIA.diagonal(offset=-1), lower_diagonal_blocks)
        
        
@pytest.mark.parametrize(
    "n_blocks, n_partitions",
    [
        (10, 3),
        (10, 5),
        (10, 10),
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
    ]
)
@pytest.mark.parametrize(
    "is_symmetric", 
    [
        pytest.param(False, id="non-symmetric"),
    ]
)
def test_read_local_block_tridiagonal_partition(
    n_blocks: int,
    n_partitions: int,
    blocksize: int,
    is_complex: bool,
    is_symmetric: bool,
):
    SEED = 63 
    
    diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks = gmu.create_block_tridiagonal_matrix(n_blocks, blocksize, is_complex, is_symmetric, is_diagonally_dominant=True, seed=SEED)
    
    rlcu.save_block_tridigonal_matrix("test_matrix.npz", diagonal_blocks, upper_diagonal_blocks, lower_diagonal_blocks, is_symmetric=is_symmetric)
    
    for i in range(n_partitions):
        start_blockrow, partition_size, end_blockrow = partu.get_local_partition_indices(i, n_partitions, n_blocks)
        
        local_diagonal_blocks, local_upper_diagonal_blocks, local_lower_diagonal_blocks = rlcu.read_local_block_tridiagonal_partition("test_matrix.npz", start_blockrow, partition_size, blocksize)
        
        assert np.allclose(local_diagonal_blocks, diagonal_blocks[start_blockrow:end_blockrow, :, :])
        #assert np.allclose(local_upper_diagonal_blocks, upper_diagonal_blocks[start_blockrow:end_blockrow-1, :, :])
        #assert np.allclose(local_lower_diagonal_blocks, lower_diagonal_blocks[start_blockrow:end_blockrow-1, :, :])