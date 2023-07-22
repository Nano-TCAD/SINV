"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import sys
sys.path.append('../')

from sinv import utils
from sinv import algorithms as alg

import numpy as np
import math
import time

from mpi4py import MPI



if __name__ == '__main__':
    # MPI initialization
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()

    # Problem parameters
    size = 20
    blocksize = 2
    density = blocksize**2/size**2
    bandwidth = np.ceil(blocksize/2)

    isComplex = True
    seed = 63

    # Retarded Green's function initial matrix
    A_init = utils.genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    A_init = utils.transMat.transformToSymmetric(A_init)

    G_ref = np.linalg.inv(A_init)


    # Compute the retarded Green's function with the parallel P-Division algorithm
    G_pdiv = alg.pdiv_a.pdiv_aggregate(A_init, blocksize)


    if comm_rank == 0:
        # Check the correctness of the result
        result = np.allclose(G_ref, G_pdiv, rtol=1e-05, atol=1e-08)
        if result:
            print("The result is correct.")
        else:
            print("The result is incorrect.")

        G_diff = np.abs(G_ref - G_pdiv)
        #vizu.vizualiseDenseMatrixFlat(G_diff, "G_diff")

