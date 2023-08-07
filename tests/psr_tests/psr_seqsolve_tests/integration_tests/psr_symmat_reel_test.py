"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the PSR algorithm. 
- Reel symmetric matrices.
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

@pytest.mark.mpi(min_size=3)
def test_psr_symmat_reel_1():
    matrice_size = 9
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)
            
@pytest.mark.mpi(min_size=3)
def test_psr_symmat_reel_2():
    matrice_size = 18
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)
def test_psr_symmat_reel_3():
    matrice_size = 27
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)            
def test_psr_symmat_reel_4():
    matrice_size = 12
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)            
def test_psr_symmat_reel_5():
    matrice_size = 24
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)            
def test_psr_symmat_reel_6():
    matrice_size = 36
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)
def test_psr_symmat_reel_7():
    matrice_size = 256
    blocksize    = 8
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)        
def test_psr_symmat_reel_8():
    matrice_size = 240
    blocksize    = 10
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower)

@pytest.mark.mpi(min_size=3)            
def test_psr_symmat_reel_9():
    matrice_size = 144
    blocksize    = 12
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if nblocks >= 3*comm_size:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        
        A_refsol = np.linalg.inv(A)
        A_refsol_bloc_diag, A_refsol_bloc_upper, A_refsol_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_refsol, blocksize)

        A_psr = alg.psr_s.psr_seqsolve(A, blocksize)
        A_psr_bloc_diag, A_psr_bloc_upper, A_psr_bloc_lower = utils.trans_mat.convertDenseToBlockTridiag(A_psr, blocksize)
        
        if comm_rank == 0:
            assert np.allclose(A_refsol_bloc_diag, A_psr_bloc_diag)\
                and np.allclose(A_refsol_bloc_upper, A_psr_bloc_upper)\
                and np.allclose(A_refsol_bloc_lower, A_psr_bloc_lower) 

if __name__ == '__main__':
    test_psr_symmat_reel_1()
    test_psr_symmat_reel_2()
    test_psr_symmat_reel_3()
    test_psr_symmat_reel_4()
    test_psr_symmat_reel_5()
    test_psr_symmat_reel_6()
    test_psr_symmat_reel_7()
    test_psr_symmat_reel_8()
    test_psr_symmat_reel_9()
    