"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

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
    ]
)
def test_pdiv(
    is_complex: bool,
    is_symmetric: bool,
    matrix_size: int,
    blocksize: int
):
    """ Test the PDIV algorithm. """
    
    bandwidth = np.ceil(blocksize/2)
    nblocks   = int(np.ceil(matrix_size/blocksize))
    
    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()
    comm_rank = comm.Get_rank()
    
    if math.log2(comm_size).is_integer() and comm_size <= nblocks:
        A = utils.matu.generateBandedDiagonalMatrix(matrix_size, 
                                                    bandwidth, 
                                                    is_complex, 
                                                    is_symmetric, SEED)
        
        
        # PDIV worflow
        pdiv_u.check_multiprocessing(comm_size)
        pdiv_u.check_input(A, blocksize, comm_size)
        
        l_start_blockrow, l_partitions_blocksizes = pdiv_u.divide_matrix(A, comm_size, blocksize)
        K_i, Bu_i, Bl_i = pdiv_u.partition_subdomain(A, l_start_blockrow, l_partitions_blocksizes, blocksize)
    
        K_local = K_i[comm_rank]
        K_inv_local, Bu_inv, Bl_inv = pdiv_lm.pdiv_localmap(K_local, Bu_i, Bl_i, blocksize)
        
        
        # Extract local reference solution
        A_refsol = np.linalg.inv(A)
        start_localpart_rowindex = l_start_blockrow[comm_rank] * blocksize
        stop_localpart_rowindex  = start_localpart_rowindex + l_partitions_blocksizes[comm_rank] * blocksize
        A_local_slice_of_refsolution = A_refsol[start_localpart_rowindex:stop_localpart_rowindex,
                                                start_localpart_rowindex:stop_localpart_rowindex]
        if comm_rank < comm_size-1:
            Bu_refsol = A_refsol[stop_localpart_rowindex-blocksize:stop_localpart_rowindex,
                                 stop_localpart_rowindex:stop_localpart_rowindex+blocksize]
            Bl_refsol = A_refsol[stop_localpart_rowindex:stop_localpart_rowindex+blocksize,
                                 stop_localpart_rowindex-blocksize:stop_localpart_rowindex]
        
        for i in range(0, l_partitions_blocksizes[comm_rank], 1):
            for j in range(0, l_partitions_blocksizes[comm_rank], 1):
                if j < i-1 or j > i+1:
                    A_local_slice_of_refsolution[i*blocksize:(i+1)*blocksize, 
                                                 j*blocksize:(j+1)*blocksize]\
                                                    = np.zeros((blocksize, blocksize))
        
        
        # Check results
        assert np.allclose(A_local_slice_of_refsolution, K_inv_local)
          
        if comm_rank < comm_size-1:
            assert np.allclose(Bu_refsol, Bu_inv)
            assert np.allclose(Bl_refsol, Bl_inv)
            
            