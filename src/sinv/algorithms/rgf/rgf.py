"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Nicolas Vetsch (nvetsch@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

import bsparse as bsp


import matplotlib.pyplot as plt



def rgf(
    A: bsp,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> bsp:
    """ RGF algorithm performing block-tridiagonal selected inversion on the 
    input matrix. Act from upper left to lower right.

    Parameters
    ----------
    A : bsparse
        Input matrix.
    sym_mat : bool, optional
        If True, the input matrix is assumed to be symmetric.
    save_off_diag : bool, optional
        If True, the off-diagonal blocks are saved in the output matrix.

    Returns
    -------
    G : bsparse
        Inverse of the input matrix.
    """

    # Storage for the full backward substitution 
    G = A.copy() * np.nan
    
    # 0. Inverse of the first block
    G[0, 0] = np.linalg.inv(A[0, 0])
    
    # 1. Forward substitution (performed left to right)
    for i in range(1, A.bshape[0], 1):
        G[i, i] = np.linalg.inv(A[i, i] - A[i, i-1] @ G[i-1, i-1] @ A[i-1, i])

    # 2. Backward substitution (performed right to left)
    for i in range(A.bshape[0]-2, -1, -1): 
        g_ii = G[i, i]
        G_lowerfactor = G[i+1, i+1] @ A[i+1, i] @ g_ii   
        
        if save_off_diag:
            G[i+1, i] = -G_lowerfactor
            if sym_mat == False:
                G[i, i+1] = -g_ii @ A[i, i+1] @ G[i+1, i+1]
            """ if sym_mat:
                G[i, i+1] = G[i+1, i].T
            else:
                G[i, i+1] = -g_ii @ A[i, i+1] @ G[i+1, i+1] """
            
        G[i, i]   =  g_ii + g_ii @ A[i, i+1] @ G_lowerfactor

    return G
        
        
