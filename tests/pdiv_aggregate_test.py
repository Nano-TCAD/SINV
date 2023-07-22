"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Tests cases for the P-Division algorithm.

The following tests adjuste matrix size w.r.t the number of processes. Hence all
processes have a partition of the same size.
========================================================
| Test n  | Blocksize |    nblocks    |  Matrice size  | 
========================================================
| Test 1  | 1         | comm_size     | 1*comm_size    |
| Test 2  | 1         | 2*comm_size   | 2*comm_size    |
| Test 3  | 1         | 10*comm_size  | 10*comm_size   |
========================================================
| Test 4  | 2         | comm_size     | 2*comm_size    |
| Test 5  | 2         | 2*comm_size   | 4*comm_size    |
| Test 6  | 2         | 10*comm_size  | 20*comm_size   |
========================================================
| Test 7  | 3         | comm_size     | 3*comm_size    |
| Test 8  | 3         | 2*comm_size   | 6*comm_size    |
| Test 9  | 3         | 10*comm_size  | 30*comm_size   |
========================================================
| Test 10 | 100       | comm_size     | 100*comm_size  |
| Test 11 | 100       | 2*comm_size   | 200*comm_size  |
| Test 12 | 100       | 10*comm_size  | 1000*comm_size |
=========================================================

The following tests doesn't define a number of block as a multiple of the
number of processes. Hence not all processes have the same partition size.
========================================================
| Test n  | Matrice size |    nblocks    | Blocksize    |
========================================================

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import algorithms as alg

import numpy as np
from mpi4py import MPI

comm = MPI.COMM_WORLD
comm_size = comm.Get_size()
rank = comm.Get_rank()

def test_pdiv_aggregate_1():
    nblocks   = comm_size
    blocksize = 1
    size      = nblocks * blocksize
    A = np.random.rand(size, size) + 1j * np.random.rand(size, size)    
    A_refsol = np.linalg.inv(A)
    A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
    if rank == 0:
        assert np.allclose(A_refsol, A_pdiv_aggregate) 

def test_pdiv_aggregate_20():
    blocksize = 2
    size      = 20
    A = np.random.rand(size, size) + 1j * np.random.rand(size, size)    
    A_refsol = np.linalg.inv(A)
    A_pdiv_aggregate = alg.pdiv_a.pdiv_aggregate(A, blocksize)
    if rank == 0:
        assert np.allclose(A_refsol, A_pdiv_aggregate) 