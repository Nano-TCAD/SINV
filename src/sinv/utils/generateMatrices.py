"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from scipy.sparse import csc_matrix



def generateRandomNumpyMat(size, isComplex=False, seed=None):
    """
        Generate a dense matrix of shape: (size x size) filled with random numbers.
            - Complex or real valued
    """
    if seed is not None:
        np.random.seed(seed)

    if isComplex:
        return np.random.rand(size, size) + 1j * np.random.rand(size, size)
    else:
        return np.random.rand(size, size)



def generateDenseMatrix(size, isComplex=False, seed=None):
    """
        Generate a dense matrix of shape: (size x size) filled with random numbers.
    """

    return generateRandomNumpyMat(size, isComplex, seed)



def generateSparseMatrix(size, density, isComplex=False, seed=None):
    """
        Generate a sparse matrix of shape: (size x size), density of non-zero elements: density,
        filled with random numbers.
    """

    A = generateRandomNumpyMat(size, isComplex, seed)

    A[A < (1-density)] = 0

    return A



def generateBandedDiagonalMatrix(size, bandwidth, isComplex=False, seed=None):
    """
        Generate a banded diagonal matrix of shape: (size x size), bandwidth: bandwidth,
        filled with random numbers.
    """

    A = generateRandomNumpyMat(size, isComplex, seed)
    
    for i in range(size):
        for j in range(size):
            if i - j > bandwidth or j - i > bandwidth:
                A[i, j] = 0

    return A
