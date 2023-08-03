"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

Contains the utility functions for the BCR algorithm.

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import math

from mpi4py import MPI



def distance_to_power_of_two(matrice_size: int):
    """ Compute the distance between the matrice_size and the closest power 
    of two minus one.
    
    Parameters
    ----------
    matrice_size : int
        Size of the matrice to generate.
        
    Returns
    -------
    distance : int
        The distance between the matrice_size and the closest power of two 
        minus one.
    """

    p = math.ceil(math.log2(matrice_size + 1))

    closest_power = 2**p - 1
    distance = closest_power - matrice_size

    return distance



def identity_padding(A: np.ndarray, 
                     padding_size: int) -> np.ndarray:
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
