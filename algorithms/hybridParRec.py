"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import scipy.linalg as la
import time

import vizuUtils as vizUtils



def schurInvert(A):
    size = A.shape[0]
    size2 = size//2
    

    # Handmade Schur decomposition
    Ls = np.zeros((size2, size2), dtype=A.dtype)
    Us = np.zeros((size2, size2), dtype=A.dtype)
    Sc = np.zeros((size2, size2), dtype=A.dtype)

    Ls = A[size2:, :size2] @ np.linalg.inv(A[:size2, :size2])
    Us = np.linalg.inv(A[:size2, :size2]) @ A[:size2, size2:]
    Sc = A[size2:, size2:] - A[size2:, :size2] @ np.linalg.inv(A[:size2, :size2]) @ A[:size2, size2:]

    G = np.zeros((size, size), dtype=A.dtype)

    G11 = np.zeros((size2, size2), dtype=A.dtype)
    G12 = np.zeros((size2, size2), dtype=A.dtype)
    G21 = np.zeros((size2, size2), dtype=A.dtype)
    G22 = np.zeros((size2, size2), dtype=A.dtype)

    G11 = np.linalg.inv(A[:size2, :size2]) + Us @ np.linalg.inv(Sc) @ Ls
    G12 = -Us @ np.linalg.inv(Sc)
    G21 = -np.linalg.inv(Sc) @ Ls
    G22 = np.linalg.inv(Sc)

    G[:size2, :size2] = G11
    G[:size2, size2:] = G12
    G[size2:, :size2] = G21
    G[size2:, size2:] = G22

    return G



def luDecompose(A):
    P, L, U = la.lu(A)

    return P, L, U


def lduDecompose(A):

    P, L, U = luDecompose(A)

    D = np.diag(np.diag(U)) 
    Unorm = U.copy()
    Unorm /= np.diag(U)[:, None]

    """ vizUtils.vizualiseDenseMatrixFlat(P, "P")
    vizUtils.vizualiseDenseMatrixFlat(L, "L")
    vizUtils.vizualiseDenseMatrixFlat(Unorm, "Unorm")
    vizUtils.vizualiseDenseMatrixFlat(D, "D") """

    #Ar = P @ L @ D @ Unorm
    #Ar = Lp @ D @ Unorm

    """ Ar = P @ L @ U

    vizUtils.vizualiseDenseMatrixFlat(Ar, "Ar")
    vizUtils.vizualiseDenseMatrixFlat(A, "A") """

    return P, L, D, Unorm


def hpr_full(A):
    size = A.shape[0]

    G = np.zeros((size, size), dtype=A.dtype)

    P, L, U = luDecompose(A)

    vizUtils.vizualiseDenseMatrixFlat(L, "L")
    vizUtils.vizualiseDenseMatrixFlat(U, "U")

    for i in range(size-1, -1, -1):
        if i >= size-3:
            print("L[i+1:, i]", L[i:, i])

    for i in range(size-1, -1, -1):
        if i >= size-3:
            print("U[i+1:, i]", U[i, i:])

        """ G[i+1:, i] = -G[i+1:, i+1:] @ P[i+1:, i] @ L[i+1:, i]
        G[i, i+1:] = -P[i+1:, i] @ U[i, i+1:] @ G[i+1:, i+1:] """

    return P @ G

