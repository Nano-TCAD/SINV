"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the P-Division algorithm. 
- Reel valued matrices.
================================================
| Test n  | Matrice size | Blocksize | nblocks | 
================================================
| Test 1  |   1000x1000  |    10     |   100   |
| Test 2  |   1000x1000  |    40     |   25    |
| Test 3  |   1000x1000  |    100    |   10    |
================================================
| Test 1  |   2500x2500  |    10     |   250   |
| Test 2  |   2500x2500  |    100    |   25    |
| Test 3  |   2500x2500  |    250    |   10    |
================================================

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import algorithms as alg
from sinv import utils

import numpy as np
import math
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
comm_rank = comm.Get_rank()

isComplex = False
seed = 63

def test_pdiv_aggregate_bigmat_reel_1():
    matrice_size = 1000
    blocksize    = 10
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)

def test_pdiv_aggregate_bigmat_reel_2():
    matrice_size = 1000
    blocksize    = 40
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_bigmat_reel_3():
    matrice_size = 1000
    blocksize    = 100
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_bigmat_reel_4():
    matrice_size = 2500
    blocksize    = 10
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_bigmat_reel_5():
    matrice_size = 2500
    blocksize    = 100
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_bigmat_reel_6():
    matrice_size = 2500
    blocksize    = 250
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
if __name__ == '__main__':
    test_pdiv_aggregate_bigmat_reel_1()
    test_pdiv_aggregate_bigmat_reel_2()
    test_pdiv_aggregate_bigmat_reel_3()
    test_pdiv_aggregate_bigmat_reel_4()
    test_pdiv_aggregate_bigmat_reel_5()
    test_pdiv_aggregate_bigmat_reel_6()
    