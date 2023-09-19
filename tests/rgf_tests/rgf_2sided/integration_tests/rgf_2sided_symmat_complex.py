"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the RGF (2 Sided) algorithm. 
- Complexe symmetric matrices.
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

isComplex = True
seed = 63

@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_1():
    matrice_size = 1
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.matu.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
     
@pytest.mark.mpi(min_size=2)       
def test_rgf_2sided_symmat_complex_2():
    matrice_size = 2
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
        
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_3():
    matrice_size = 3
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
         
@pytest.mark.mpi(min_size=2)   
def test_rgf_2sided_symmat_complex_4():
    matrice_size = 2
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
          
@pytest.mark.mpi(min_size=2)  
def test_rgf_2sided_symmat_complex_5():
    matrice_size = 4
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
            
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_6():
    matrice_size = 6
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
            
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_7():
    matrice_size = 3
    blocksize    = 1
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
            
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_8():
    matrice_size = 6
    blocksize    = 2
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
            
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_9():
    matrice_size = 9
    blocksize    = 3
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
    
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_10():
    matrice_size = 128
    blocksize    = 8
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
        
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_11():
    matrice_size = 128
    blocksize    = 16
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)
            
@pytest.mark.mpi(min_size=2)
def test_rgf_2sided_symmat_complex_12():
    matrice_size = 128
    blocksize    = 32
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size == 2:
        A = utils.matu.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_bloc_diag, A_bloc_upper, A_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A, blocksize)
        G_diag_blocks, G_upper_blocks, G_lower_blocks = alg.rgfl.rgf_leftToRight(A_bloc_diag, A_bloc_upper, A_bloc_lower)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)
        
        assert np.allclose(A_refsol_bloc_diag, G_diag_blocks)\
                and np.allclose(A_refsol_bloc_upper, G_upper_blocks)\
                and np.allclose(A_refsol_bloc_lower, G_lower_blocks)  

if __name__ == '__main__':
    test_rgf_2sided_symmat_complex_1()
    test_rgf_2sided_symmat_complex_2()
    test_rgf_2sided_symmat_complex_3()
    test_rgf_2sided_symmat_complex_4()
    test_rgf_2sided_symmat_complex_5()
    test_rgf_2sided_symmat_complex_6()
    test_rgf_2sided_symmat_complex_7()
    test_rgf_2sided_symmat_complex_8()
    test_rgf_2sided_symmat_complex_9()
    test_rgf_2sided_symmat_complex_10()
    test_rgf_2sided_symmat_complex_11()
    test_rgf_2sided_symmat_complex_12()
    