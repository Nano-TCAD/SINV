"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the PSR algorithm. 
- Reel symmetric matrices.
================================================
| Test n  | Matrice size | Blocksize | nblocks | 
================================================
| Test 1  |     6x6      |     1     |    6    |
| Test 2  |    12x12     |     2     |    6    |
| Test 3  |    18x18     |     3     |    6    |
================================================
| Test 4  |     8x8      |     1     |    8    |
| Test 5  |    16x16     |     2     |    8    |
| Test 6  |    24x24     |     3     |    8    |
================================================
| Test 7  |   128x128    |     8     |    16   |
| Test 8  |   128x128    |     16    |    8    |
| Test 9  |   128x128    |     32    |    4    |
================================================

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import algorithms as alg
from sinv import utils

import numpy as np
import math
import pytest
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

isComplex = False
seed = 63

@pytest.mark.mpi(min_size=3)
def test_psr_refactor_symmat_reel_1():
    matrice_size = 6
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)
            
@pytest.mark.mpi(min_size=3)
def test_psr_refactor_symmat_reel_2():
    matrice_size = 12
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)
def test_psr_refactor_symmat_reel_3():
    matrice_size = 18
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)            
def test_psr_refactor_symmat_reel_4():
    matrice_size = 8
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)            
def test_psr_refactor_symmat_reel_5():
    matrice_size = 16
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)            
def test_psr_refactor_symmat_reel_6():
    matrice_size = 24
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)
def test_psr_refactor_symmat_reel_7():
    matrice_size = 128
    blocksize    = 8
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        #A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        #if comm_rank == 0:
        #    assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)        
def test_psr_refactor_symmat_reel_8():
    matrice_size = 128
    blocksize    = 16
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)

@pytest.mark.mpi(min_size=3)            
def test_psr_refactor_symmat_reel_9():
    matrice_size = 128
    blocksize    = 32
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks:
        A = utils.gen_mat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.trans_mat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_psr = alg.psr_s_r.psr_seqsolve_refactored(A, blocksize)
        utils.vizu.compareDenseMatrix(A_refsol, "A_refsol", A_psr, "A_psr")
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_psr)    

if __name__ == '__main__':
    #test_psr_refactor_symmat_reel_1()
    #test_psr_refactor_symmat_reel_2()
    #test_psr_refactor_symmat_reel_3()
    #test_psr_refactor_symmat_reel_4()
    #test_psr_refactor_symmat_reel_5()
    #test_psr_refactor_symmat_reel_6()
    test_psr_refactor_symmat_reel_7()
    #test_psr_refactor_symmat_reel_8()
    #test_psr_refactor_symmat_reel_9() 
    