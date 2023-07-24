"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Basic tests cases for the P-Division algorithm. 
- Complex valued matrices.
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

isComplex = True
seed = 63

def test_pdiv_aggregate_basic_complex_1():
    matrice_size = 1
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_2():
    matrice_size = 2
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
        
def test_pdiv_aggregate_basic_complex_3():
    matrice_size = 3
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_4():
    matrice_size = 2
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_5():
    matrice_size = 4
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_6():
    matrice_size = 6
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_7():
    matrice_size = 3
    blocksize    = 1
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_8():
    matrice_size = 6
    blocksize    = 2
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            
def test_pdiv_aggregate_basic_complex_9():
    matrice_size = 9
    blocksize    = 3
    nblocks      = matrice_size // blocksize
    bandwidth    = np.ceil(blocksize/2)
    
    if comm_size <= nblocks and math.log2(comm_size).is_integer():
        A = utils.genMat.generateBandedDiagonalMatrix(matrice_size, bandwidth, isComplex, seed)
        A = utils.transMat.transformToSymmetric(A)
        A_refsol = np.linalg.inv(A)
        A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
        if comm_rank == 0:
            assert np.allclose(A_refsol, A_pdiv_aggregate)
            