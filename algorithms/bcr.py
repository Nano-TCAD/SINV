"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1017/CBO9780511812583

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import utils.vizualisation       as vizu
import utils.permutationMatrices as permMat
import utils.generateMatrices    as genMat
import utils.convertMatrices     as convMat
import utils.transformMatrices   as transMat

import numpy as np
import math
import scipy.linalg as la
import time

from mpi4py import MPI






def block_cyclic_reduction():



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

    d = transMat.distance_to_power_of_two(A.shape[0])
    print("Distance to power of two: ", d)


    A = transMat.identity_padding(A, d)

    #vizu.vizualiseDenseMatrixFlat(A, "A")


    size = A.shape[0]

    F = np.identity(size)
    x = np.zeros((size, size))


    # Compute reference solution before A being modified (work in place)
    npinvert = np.linalg.inv(A)



    index1, index2, offset = 0, 0, 0
    alpha, gamma = 0.0, 0.0

    #vizu.vizualiseDenseMatrixFlat(A, "A")



    # Cycle reduction
    for i in range(int(math.log2(size + 1)) - 1):
        for j in range(int(math.pow(2, i + 1)) - 1, size, int(math.pow(2, i + 1))):
            offset = int(math.pow(2, i))
            index1 = j - offset
            index2 = j + offset

            alpha = A[j][index1] / A[index1][index1]
            gamma = A[j][index2] / A[index2][index2]

            for k in range(size):
                A[j][k] -= alpha * A[index1][k] + gamma * A[index2][k]
                F[j][k] -= alpha * F[index1][k] + gamma * F[index2][k]


    #vizu.compareDenseMatrix(A, F, "F")

    # Back substitution
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
            x[index2] /= A[index2][index2]


    vizu.compareDenseMatrix(npinvert, x, "A_ref VS A")


