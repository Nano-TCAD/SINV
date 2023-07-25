"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import math



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



def distance_to_power_of_two(matrice_size: int):
    """ Compute the distance between n and the closest power of two minus one.
    
    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
        
    Returns
    -------
    distance : int
        The distance between n and the closest power of two minus one.
    """

    p = math.ceil(math.log2(matrice_size + 1))

    closest_power = 2**p - 1
    distance = closest_power - matrice_size

    return distance



def identity_padding(A: np.ndarray, 
                     padding_size: int):
    """ Padd the A matrix with an identity matrix of size padding_size.
    
    Parameters
    ----------
    A : np.ndarray
        The matrix to padd.
    padding_size : int
        The size of the padding to add.
    
    Returns
    -------
    Ap : np.ndarray
        The padded matrix.
    """

    matrice_size = A.shape[0]

    I = np.eye(padding_size)
    Ap = np.zeros((matrice_size+padding_size, matrice_size+padding_size), dtype=A.dtype)

    Ap[0:matrice_size, 0:matrice_size] = A
    Ap[matrice_size:matrice_size+padding_size, matrice_size:matrice_size+padding_size] = I

    return Ap



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



def convertBlockTridiagToDense(A_bloc_diag: np.ndarray, 
                               A_bloc_upper: np.ndarray, 
                               A_bloc_lower: np.ndarray):
    """ Convert a block tridiagonal matrix to a dense matrix.
    
    Parameters
    ----------
    A_bloc_diag : np.ndarray
        The diagonal blocks.
    A_bloc_upper : np.ndarray
        The upper diagonal blocks.
    A_bloc_lower : np.ndarray
        The lower diagonal blocks.
        
    Returns
    -------
    A : np.ndarray
        The dense matrix.
    """
    
    nblocks   = A_bloc_diag.shape[0]
    blocksize = A_bloc_diag.shape[1]

    A = np.zeros((nblocks*blocksize, nblocks*blocksize), dtype=A_bloc_diag.dtype)

    for i in range(nblocks):
        A[i*blocksize:(i+1)*blocksize, i*blocksize:(i+1)*blocksize] = A_bloc_diag[i, ]
        if i < nblocks-1:
            A[i*blocksize:(i+1)*blocksize, (i+1)*blocksize:(i+2)*blocksize] = A_bloc_upper[i, ]
            A[(i+1)*blocksize:(i+2)*blocksize, i*blocksize:(i+1)*blocksize] = A_bloc_lower[i, ]

    return A

