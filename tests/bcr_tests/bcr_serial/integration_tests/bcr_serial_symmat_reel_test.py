"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

Basic tests cases for the BCR serial algorithm. 
- Reel symmetric matrices.
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

isComplex = False
seed = 63

@pytest.mark.mpi_skip()
def test_bcr_serial_symmat_reel_1():
    matrice_size = 1
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)
            
@pytest.mark.mpi_skip()
def test_bcr_serial_symmat_reel_2():
    matrice_size = 2
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()        
def test_bcr_serial_symmat_reel_3():
    matrice_size = 3
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_4():
    matrice_size = 2
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_5():
    matrice_size = 4
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_6():
    matrice_size = 6
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_7():
    matrice_size = 3
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_8():
    matrice_size = 6
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_9():
    matrice_size = 9
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)
    
@pytest.mark.mpi_skip()
def test_bcr_serial_symmat_reel_10():
    matrice_size = 128
    blocksize    = 8
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)
    
@pytest.mark.mpi_skip()    
def test_bcr_serial_symmat_reel_11():
    matrice_size = 128
    blocksize    = 16
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

@pytest.mark.mpi_skip()            
def test_bcr_serial_symmat_reel_12():
    matrice_size = 128
    blocksize    = 32
    bandwidth    = np.ceil(blocksize/2)
    
    A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
    A = utils.trans_mat.transformToSymmetric(A)

    G = alg.bcr_s.bcr_serial(A, blocksize)
    G_bcr_s_bloc_diag, G_bcr_s_bloc_upper, G_bcr_s_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(G, blocksize)
    
    A_refsol = np.linalg.inv(A)
    A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
    
    assert np.allclose(A_refsol_bloc_diag, G_bcr_s_bloc_diag)\
            and np.allclose(A_refsol_bloc_upper, G_bcr_s_bloc_upper)\
            and np.allclose(A_refsol_bloc_lower, G_bcr_s_bloc_lower)

if __name__ == '__main__':
    test_bcr_serial_symmat_reel_1()
    test_bcr_serial_symmat_reel_2()
    test_bcr_serial_symmat_reel_3()
    test_bcr_serial_symmat_reel_4()
    test_bcr_serial_symmat_reel_5()
    test_bcr_serial_symmat_reel_6()
    test_bcr_serial_symmat_reel_7()
    test_bcr_serial_symmat_reel_8()
    test_bcr_serial_symmat_reel_9()
    test_bcr_serial_symmat_reel_10()
    test_bcr_serial_symmat_reel_11()
    test_bcr_serial_symmat_reel_12()
    