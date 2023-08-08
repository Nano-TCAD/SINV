"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035
@reference: Numerical Recipes in C (2nd Ed.): The Art of Scientific Computing (10.5555/148286)

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def block_tridiag_lusolve(A: np.ndarray, 
                          blocksize: int) -> np.ndarray:
    """ Block tridiagonal solver using non pivoting LU decomposition/solving.

    Parameters
    ----------
    A : np.ndarray
        Block tridiagonal matrix
    blocksize : int
        Block matrice_size
        
    Returns
    -------
    G : np.ndarray
        Inverse of A
    """

    matrice_size = A.shape[0]
    nblocks = matrice_size // blocksize

    G = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    L = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    U = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    D = np.zeros((matrice_size, matrice_size), dtype=A.dtype)
    
        
    # Initialisation of forward recurence
    D[0:blocksize, 0:blocksize] = A[0:blocksize, 0:blocksize]
    D_inv_0 = np.linalg.inv(D[0:blocksize, 0:blocksize])
    L[blocksize:2*blocksize, 0:blocksize] = A[blocksize:2*blocksize, 0:blocksize] @ D_inv_0
    U[0:blocksize, blocksize:2*blocksize] = D_inv_0 @ A[0:blocksize, blocksize:2*blocksize]
    
    # Forward recurence
    for i in range(1, nblocks):
        b_im1 = (i-1)*blocksize
        b_i   = i*blocksize
        b_ip1 = (i+1)*blocksize

        D_inv_i = np.linalg.inv(D[b_im1:b_i, b_im1:b_i])

        D[b_i:b_ip1, b_i:b_ip1] = A[b_i:b_ip1, b_i:b_ip1] - A[b_i:b_ip1, b_im1:b_i] @ D_inv_i @ A[b_im1:b_i, b_i:b_ip1]
        L[b_i:b_ip1, b_im1:b_i] = A[b_i:b_ip1, b_im1:b_i] @ D_inv_i
        U[b_im1:b_i, b_i:b_ip1] = D_inv_i @ A[b_im1:b_i, b_i:b_ip1]

    # Initialisation of backward recurence
    b_n   = (nblocks-1)*blocksize
    b_np1 = nblocks*blocksize

    G[b_n:b_np1, b_n:b_np1] = np.linalg.inv(D[b_n:b_np1, b_n:b_np1])
    
    # Backward recurence
    for i in range(nblocks-1, -1, -1):
        b_im1 = (i-1)*blocksize
        b_i   = i*blocksize
        b_ip1 = (i+1)*blocksize

        G[b_i:b_ip1, b_im1:b_i] = -G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_im1:b_i, b_i:b_ip1] = -U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1]
        G[b_im1:b_i, b_im1:b_i] = np.linalg.inv(D[b_im1:b_i, b_im1:b_i]) + U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]

    
    return G


