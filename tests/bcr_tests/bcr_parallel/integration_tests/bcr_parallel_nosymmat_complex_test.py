"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the BCR parallel algorithm. 
- Complex un-symmetric matrices.
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
================================================

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import algorithms as alg
from sinv import utils

import numpy as np
import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

isComplex = True
seed = 63

@pytest.mark.mpi(min_size=2)
def test_bcr_parallel_nosymmat_complex_1():
    matrice_size = 2
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)
            
@pytest.mark.mpi(min_size=2)
def test_bcr_parallel_nosymmat_complex_2():
    matrice_size = 4
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)
def test_bcr_parallel_nosymmat_complex_3():
    matrice_size = 6
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)            
def test_bcr_parallel_nosymmat_complex_4():
    matrice_size = 3
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)            
def test_bcr_parallel_nosymmat_complex_5():
    matrice_size = 6
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)            
def test_bcr_parallel_nosymmat_complex_6():
    matrice_size = 9
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)
def test_bcr_parallel_nosymmat_complex_7():
    matrice_size = 128
    blocksize    = 8
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)        
def test_bcr_parallel_nosymmat_complex_8():
    matrice_size = 128
    blocksize    = 16
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

@pytest.mark.mpi(min_size=2)        
def test_bcr_parallel_nosymmat_complex_9():
    matrice_size = 128
    blocksize    = 32
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.matu.convertDenseToBlkTridiag(A_refsol, blocksize)

    G_bcr_p = alg.bcr_p.bcr_parallel(A, blocksize)
    G_bcr_p_bloc_diag, G_bcr_p_bloc_upper, G_bcr_p_bloc_lower = utils.matu.convertDenseToBlkTridiag(G_bcr_p, blocksize)
    
    if comm_rank == 0:
        assert np.allclose(A_refsol_bloc_diag, G_bcr_p_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_p_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_p_bloc_lower)

if __name__ == '__main__':
    test_bcr_parallel_nosymmat_complex_1()
    test_bcr_parallel_nosymmat_complex_2()
    test_bcr_parallel_nosymmat_complex_3()
    test_bcr_parallel_nosymmat_complex_4()
    test_bcr_parallel_nosymmat_complex_5()
    test_bcr_parallel_nosymmat_complex_6()
    test_bcr_parallel_nosymmat_complex_7()
    test_bcr_parallel_nosymmat_complex_8()
    test_bcr_parallel_nosymmat_complex_9()