"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def generateRandomNumpyMat(
    matrice_size: int, 
    is_complex: bool = False,
    is_symmetric: bool = False,
    seed: int = None
) -> np.ndarray:
    """ Generate a dense matrix of shape: (matrice_size x matrice_size) filled 
    with random numbers. The matrice may be complex or real valued.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    is_symmetric : bool, optional
        Whether the matrice should be symmetric or not. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """
    if seed is not None:
        np.random.seed(seed)
        
    A = np.zeros((matrice_size, matrice_size))

    if is_complex:
        A = np.random.rand(matrice_size, matrice_size)\
               + 1j * np.random.rand(matrice_size, matrice_size)
    else:
        A = np.random.rand(matrice_size, matrice_size)
        
    if is_symmetric:
        A = A + A.T
        
    return A



def generateBandedDiagonalMatrix(
    matrice_size: int,
    matrice_bandwidth: int, 
    is_complex: bool = False, 
    seed: int = None
) -> np.ndarray:
    """ Generate a banded diagonal matrix of shape: (matrice_size x matrice_size)
    with a bandwidth of "bandwidth", filled with random numbers.

    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
    is_complex : bool, optional
        Whether the matrice should be complex or real valued. The default is False.
    seed : int, optional
        Seed for the random number generator. The default is no seed.
        
    Returns
    -------
    A : np.ndarray
        The generated matrice.
    """

    A = generateRandomNumpyMat(matrice_size, is_complex, seed)
    
    for i in range(matrice_size):
        for j in range(matrice_size):
            if i - j > matrice_bandwidth or j - i > matrice_bandwidth:
                A[i, j] = 0

    return A



def transformToSymmetric(A: np.ndarray):
    """ Make a matrix symmetric by adding its transpose to itself.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to transform.
        
    Returns
    -------
    A : np.ndarray
        The transformed to symmetric matrix.
    """
    
    return A + A.T



def convertDenseToBlockTridiag(A: np.ndarray, 
                               blocksize: int):
    """ Converte a numpy dense matrix to 3 numpy arrays containing the diagonal,
    upper diagonal and lower diagonal blocks.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to convert.
    blocksize : int
        The size of the blocks.
        
    Returns
    -------
    A_bloc_diag : np.ndarray
        The diagonal blocks.
    A_bloc_upper : np.ndarray
        The upper diagonal blocks.
    A_bloc_lower : np.ndarray
        The lower diagonal blocks.
    """
    
    nblocks = int(np.ceil(A.shape[0]/blocksize))

    A_bloc_diag  = np.zeros((nblocks, blocksize, blocksize), dtype=A.dtype)
    A_bloc_upper = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)
    A_bloc_lower = np.zeros((nblocks-1, blocksize, blocksize), dtype=A.dtype)

    for i in range(nblocks):
        A_bloc_diag[i, ] = A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize]
        if i < nblocks-1:
            A_bloc_upper[i, ] = A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize]
            A_bloc_lower[i, ] = A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize]

    return A_bloc_diag, A_bloc_upper, A_bloc_lower

