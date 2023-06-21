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
    X = np.zeros((B.shape[0], B.shape[0]), dtype=B.dtype)


    # Compute reference solution before A being modified (work in place)
    npinvert = np.linalg.inv(B)




    alpha = np.zeros((blocksize, blocksize), dtype=B.dtype)
    gamma = np.zeros((blocksize, blocksize), dtype=B.dtype)


    # Cycle reduction
    # TODO: Optimize inversion of A matrix by reusing the previous inversion between gamma and alpha
    nb_stages = int(math.log2(nblocks_padded + 1)) - 1 # number of stages of the cycle reduction

    for i_blockindex in range(nb_stages):
        #print("i: ", i_blockindex)
        # i_blockindex is the curent stage of the cycle reduction
        for j_blockindex in range(int(math.pow(2, i_blockindex + 1)) - 1, nblocks_padded, int(math.pow(2, i_blockindex + 1))):
            # j_blockindex is the index of the row we are currently reducing towards
            # offset_blockindex is the distance between the two rows we are reducing from and the row we are reducing towards
            # index1_blockindex and index2_blockindex are the indices of the rows we are reducing from
            #   - index1_blockindex being the upper row and index2_blockindex the lower row
            offset_blockindex = int(math.pow(2, i_blockindex)) 
            index1_blockindex = j_blockindex - offset_blockindex
            index2_blockindex = j_blockindex + offset_blockindex

            #print("  j_blockindex: ", j_blockindex, "offset_blockindex: ", offset_blockindex, "index1_blockindex: ", index1_blockindex, "index2_blockindex: ", index2_blockindex)

            # Pre-computing of row-based indices
            j_rowindex   = j_blockindex * blocksize
            jp1_rowindex = (j_blockindex + 1) * blocksize

            index1_rowindex   = index1_blockindex * blocksize
            index1p1_rowindex = (index1_blockindex + 1) * blocksize
            
            index2_rowindex   = index2_blockindex * blocksize
            index2p1_rowindex = (index2_blockindex + 1) * blocksize


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

    vizu.compareDenseMatrix(B, F, "F")

    # Back substitution
    index_blockindex = (nblocks_padded - 1) // 2
    index_rowindex   = index_blockindex * blocksize
    indexp1_rowindex = (index_blockindex + 1) * blocksize
    X[index_rowindex:indexp1_rowindex, index_rowindex:indexp1_rowindex] = F[index_rowindex:indexp1_rowindex, index_rowindex:indexp1_rowindex] @ np.linalg.inv(B[index_rowindex:indexp1_rowindex, index_rowindex:indexp1_rowindex])

    for i in range(int(math.log2(nblocks_padded + 1)) - 2, -1, -1):
        for j in range(int(math.pow(2, i + 1)) - 1, nblocks_padded, int(math.pow(2, i + 1))):
            offset_blockindex = int(math.pow(2, i))
            index1_blockindex = j - offset_blockindex
            index2_blockindex = j + offset_blockindex

            # Pre-computing of row-based indices
            j_rowindex   = j_blockindex * blocksize
            jp1_rowindex = (j_blockindex + 1) * blocksize

            index1_rowindex   = index1_blockindex * blocksize
            index1p1_rowindex = (index1_blockindex + 1) * blocksize
            
            index2_rowindex   = index2_blockindex * blocksize
            index2p1_rowindex = (index2_blockindex + 1) * blocksize

            
            X[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex] = F[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex]
            X[index2_rowindex:index2p1_rowindex, index1_rowindex:index1p1_rowindex] = F[index2_rowindex:index2p1_rowindex, index1_rowindex:index1p1_rowindex]


            for k_blockindex in range(nblocks_padded):
                k_rowindex   = k_blockindex * blocksize
                kp1_rowindex = (k_blockindex + 1) * blocksize

                if k_blockindex != index1_blockindex:
                    X[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex] = X[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex]\
                                                            - (B[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex]\
                                                                @ X[k_rowindex:kp1_rowindex, k_rowindex:kp1_rowindex])
                if k_blockindex != index2_blockindex:
                    X[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex] = X[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex]\
                                                            - (B[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex] 
                                                                @ X[k_rowindex:kp1_rowindex, k_rowindex:kp1_rowindex])


                X[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex] = X[index1_rowindex:index1p1_rowindex, k_rowindex:kp1_rowindex] @ np.linalg.inv(B[index1_rowindex:index1p1_rowindex, index1_rowindex:index1p1_rowindex])
                X[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex] = X[index2_rowindex:index2p1_rowindex, k_rowindex:kp1_rowindex] @ np.linalg.inv(B[index2_rowindex:index2p1_rowindex, index2_rowindex:index2p1_rowindex])


    vizu.compareDenseMatrix(npinvert, X, "A_ref VS B")


