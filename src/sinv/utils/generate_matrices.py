"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def generateRandomNumpyMat(matrice_size: int, 
                           is_complex: bool = False,
                           is_symmetric: bool = False,
                           seed: int = None) -> np.ndarray:
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



def generateSparseMatrix(matrice_size: int,
                         density: int, 
                         is_complex: bool = False, 
                         seed: int = None) -> np.ndarray:
    """ Generate a sparse matrix of shape: (matrice_size x matrice_size) with a 
    density of non-zero elements "density", filled with random numbers.
    
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

    A[A < (1-density)] = 0

    return A



def generateBandedDiagonalMatrix(matrice_size: int,
                                 matrice_bandwidth: int, 
                                 is_complex: bool = False, 
                                 seed: int = None) -> np.ndarray:
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



def generateSchurPermutationMatrix(matrice_size: int):
    """ Generate a permutation matrix for the Schur decomposition procedure as 
    used in the hpr method.
    
    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
        
    Returns
    -------
    P : np.ndarray
        The generated permutation matrix.
    """

    P = np.zeros((matrice_size, matrice_size), dtype=np.int32)

    offsetRight = matrice_size-1
    offsetLeft  = 0
    bascule     = True

    for i in range(matrice_size-1, -1, -1):
        if bascule:
            P[i, offsetRight] = 1
            offsetRight -= 1
            bascule = False
        else:
            P[i, offsetLeft] = 1
            offsetLeft  += 1
            bascule = True

    return P



def generateSchurBlockPermutationMatrix(nblocks: int, 
                                        blocksize: int):
    """ Generate a block permutation matrix for the Schur bloc decomposition 
    procedure as used in the hpr method.
    
    Parameters
    ----------
    nblocs : int
        Number of blocks in the matrix.
    blocksize : int
        Size of the blocks.
        
    Returns
    -------
    P : np.ndarray
        The generated block permutation matrix.
    """

    P = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=np.int32)
    I = np.eye(blocksize, dtype=np.int32)

    offsetRight = nblocks-1
    offsetLeft  = 0
    bascule     = True

    for i in range(nblocks-1, -1, -1):
        if bascule:
            P[i*blocksize:(i+1)*blocksize, offsetRight*blocksize:(offsetRight+1)*blocksize] = I
            offsetRight -= 1
            bascule = False
        else:
            P[i*blocksize:(i+1)*blocksize, offsetLeft*blocksize:(offsetLeft+1)*blocksize] = I
            offsetLeft  += 1
            bascule = True

    return P



def generateCyclicReductionPermutationMatrix(matrice_size: int):
    """ Generate a permutation matrix suited for the Cyclic Reduction algorithm.
    
    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
        
    Returns
    -------
    P : np.ndarray
        The generated permutation matrix.
    """

    P = np.zeros((matrice_size, matrice_size), dtype=np.int32)

    for i in range(matrice_size):
        if i%2 == 0:
            P[i//2, i] = 1
        else:
            P[matrice_size//2 + i//2, i] = 1

    return P



def generateCyclicReductionBlockPermutationMatrix(nblocks: int, 
                                                  blocksize: int):
    """ Generate a block permutation matrix suited for the block Cyclic 
    Reduction algorithm.
    
    Parameters
    ----------
    nblocs : int
        Number of blocks in the matrix.
    blocksize : int
        Size of the blocks.
        
    Returns
    -------
    P : np.ndarray
        The generated block permutation matrix.
    """

    P = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=np.int32)
    I = np.eye(blocksize, dtype=np.int32)

    for i in range(nblocks):
        if i%2 == 0:
            P[(i//2)*blocksize:(i//2+1)*blocksize, i*blocksize:(i+1)*blocksize] = I
        else:
            P[(nblocks//2 + i//2)*blocksize:(nblocks//2 + i//2 + 1)*blocksize, i*blocksize:(i+1)*blocksize] = I

    return P


