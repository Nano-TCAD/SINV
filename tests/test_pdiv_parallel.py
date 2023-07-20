"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import sys
sys.path.append('../')

import utils.generateMatrices    as genMat
import utils.transformMatrices   as transMat
import utils.vizualisation       as vizu

import algorithms.pdiv_parallel as pdiv_p

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
    A_init = genMat.generateBandedDiagonalMatrix(size, bandwidth, isComplex, seed)
    A_init = transMat.transformToSymmetric(A_init)

    G_ref = np.linalg.inv(A_init)


    # Compute the retarded Green's function with the parallel P-Division algorithm
    G_pdiv = pdiv_p.pdiv(A_init, blocksize)


    if comm_rank == 0:
        # Compute the error
        error = np.linalg.norm(G_ref - G_pdiv, ord='fro') / np.linalg.norm(G_ref, ord='fro')
        print("Error: ", error)
        #vizu.compareDenseMatrix(G_ref, "G_ref", G_pdiv, "G_pdiv")