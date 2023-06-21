"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1017/CBO9780511812583

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.vizualisation       as vizu
import utils.transformMatrices   as transMat

import numpy as np
import math
import scipy.linalg as la
import time

from mpi4py import MPI



def block_cyclic_reduction(B, blocksize):

    #vizu.vizualiseDenseMatrixFlat(B, "B")

    nblocks_initial = B.shape[0] // blocksize
    print("nblocks_initial: ", nblocks_initial)
    block_padding_distance = transMat.distance_to_power_of_two(nblocks_initial)
    print("block_padding_distance: ", block_padding_distance)

    B = transMat.identity_padding(B, block_padding_distance*blocksize)

    vizu.vizualiseDenseMatrixFlat(B, "B")




    A = np.array([
        [2, 1, 0, 0, 0],
        [1, 2, 1, 0, 0],
        [0, 1, 2, 0, 0],
        [0, 0, 0, 1, 0],
        [0, 0, 0, 0, 1]
    ])

    
    # Extended system to size 6, Don't work (normal)
    """ A = np.array([
        [2, 1, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0],
        [0, 1, 2, 0, 0, 0],
        [0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 1]
    ]) """


    # Extended system to size 7, WORKS
    """ A = np.array([
        [2, 1, 0, 0, 0, 0, 0],
        [1, 2, 1, 0, 0, 0, 0],
        [0, 1, 2, 0, 0, 0, 0],
        [0, 0, 0, 1, 0, 0, 0],
        [0, 0, 0, 0, 1, 0, 0],
        [0, 0, 0, 0, 0, 1, 0],
        [0, 0, 0, 0, 0, 0, 1]
    ]) """

    

    #d = transMat.distance_to_power_of_two(A.shape[0])
    #print("Distance to power of two: ", d)

    #A = transMat.identity_padding(A, d)
    #vizu.vizualiseDenseMatrixFlat(A, "A")
    

    #size = A.shape[0]
    nblocks_padded = B.shape[0] // blocksize

    print("nblocks_padded: ", nblocks_padded)


    F = np.identity(B.shape[0], dtype=B.dtype)
    x = np.zeros((B.shape[0], B.shape[0]), dtype=B.dtype)


    # Compute reference solution before A being modified (work in place)
    npinvert = np.linalg.inv(B)



    index1, index2, offset = 0, 0, 0
    #alpha, gamma = 0.0, 0.0

    #vizu.vizualiseDenseMatrixFlat(A, "A")

    alpha = np.zeros((blocksize, blocksize), dtype=B.dtype)
    gamma = np.zeros((blocksize, blocksize), dtype=B.dtype)


    # Cycle reduction
    # TODO: Optimize inversion of A matrix by reusing the previous inversion between gamma and alpha
    nb_stages = int(math.log2(nblocks_padded + 1)) - 1 # number of stages of the cycle reduction

    for i_blockindex in range(nb_stages):
        print("i: ", i_blockindex)
        # i_blockindex is the curent stage of the cycle reduction
        for j_blockindex in range(int(math.pow(2, i_blockindex + 1)) - 1, nblocks_padded, int(math.pow(2, i_blockindex + 1))):
            # j_blockindex is the index of the row we are currently reducing towards
            # offset_blockindex is the distance between the two rows we are reducing from and the row we are reducing towards
            # index1_blockindex and index2_blockindex are the indices of the rows we are reducing from
            #   - index1_blockindex being the upper row and index2_blockindex the lower row
            offset_blockindex = int(math.pow(2, i_blockindex)) 
            index1_blockindex = j_blockindex - offset_blockindex
            index2_blockindex = j_blockindex + offset_blockindex

            print("  j_blockindex: ", j_blockindex, "offset_blockindex: ", offset_blockindex, "index1_blockindex: ", index1_blockindex, "index2_blockindex: ", index2_blockindex)

            # Pre-computing of row-based indices
            j_rowindex   = j_blockindex * blocksize
            jp1_rowindex = (j_blockindex + 1) * blocksize

            index1_rowindex   = index1_blockindex * blocksize
            index1p1_rowindex = (index1_blockindex + 1) * blocksize
            
            index2_rowindex   = index2_blockindex * blocksize
            index2p1_rowindex = (index2_blockindex + 1) * blocksize

            #vizu.vizualiseDenseMatrixFlat(B[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex], "B[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex]")


            alpha = B[j_rowindex:jp1_rowindex, index1_rowindex:index1p1_rowindex]\
                        @ np.linalg.inv(B[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex])
            
            gamma = B[j_rowindex:jp1_rowindex, index2_rowindex:index2p1_rowindex]\
                        @ np.linalg.inv(B[index2_rowindex:index2p1_rowindex, index2_rowindex:index2p1_rowindex])


            for k_blockindex in range(nblocks_padded):
                # k_blockindex is a column index 
                k_rowindex   = k_blockindex * blocksize
                kp1_rowindex = (k_blockindex + 1) * blocksize

                B[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] = B[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]\
                                                                        - (alpha @ B[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex]\
                                                                            + gamma @ B[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex])
                
                F[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] = F[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]\
                                                                        - ( alpha @ F[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex]\
                                                                            + gamma @ F[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex] )

            vizu.vizualiseDenseMatrixFlat(B, f"B j: {j_blockindex}")



    """ # Cycle reduction
    for i in range(int(math.log2(size + 1)) - 1):
        print("i: ", i)
        # range(int(math.log2(size + 1)) - 1) is the number of stages of the cycle reduction
        # i is the curent stage of the cycle reduction
        for j in range(int(math.pow(2, i + 1)) - 1, size, int(math.pow(2, i + 1))):
            # j is the index of the row we are currently reducing towards
            # offset is the distance between the two rows we are reducing from
            # index1 and index2 are the indices of the rows we are reducing from
            #   - index1 being the upper row and index2 the lower row
            offset = int(math.pow(2, i)) 
            index1 = j - offset
            index2 = j + offset

            print("  j: ", j, "offset: ", offset, "index1: ", index1, "index2: ", index2)


            alpha = A[j][index1] / A[index1][index1]
            gamma = A[j][index2] / A[index2][index2]

            for k in range(size):
                print("     k: ", k)
                A[j][k] -= alpha * A[index1][k] + gamma * A[index2][k]
                F[j][k] -= alpha * F[index1][k] + gamma * F[index2][k] """


    vizu.compareDenseMatrix(B, F, "F")

    """ # Back substitution
    index = (size - 1) // 2
    x[index] = F[index] / A[index][index]

    for i in range(int(math.log2(size + 1)) - 2, -1, -1):
        for j in range(int(math.pow(2, i + 1)) - 1, size, int(math.pow(2, i + 1))):
            offset = int(math.pow(2, i))
            index1 = j - offset
            index2 = j + offset

            x[index1] = F[index1]
            x[index2] = F[index2]

            for k in range(size):
                if k != index1:
                    x[index1] -= A[index1][k] * x[k]
                if k != index2:
                    x[index2] -= A[index2][k] * x[k]

            x[index1] /= A[index1][index1]
            x[index2] /= A[index2][index2] """


    vizu.compareDenseMatrix(npinvert, B, "A_ref VS A")


