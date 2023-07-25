"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def rgf_rightToLeft(A_bloc_diag: np.ndarray, 
                    A_bloc_upper: np.ndarray, 
                    A_bloc_lower: np.ndarray) -> [np.ndarray, np.ndarray, np.ndarray]:
    """ RGF algorithm performing block-tridiagonal selected inversion on the 
    input matrix. Act from right to left.
    
    Parameters
    ----------
    A_bloc_diag : np.ndarray
        Block-diagonal elements of the matrix to invert.
    A_bloc_upper : np.ndarray
        Upper block-diagonal elements of the matrix to invert.
    A_bloc_lower : np.ndarray
        Lower block-diagonal elements of the matrix to invert.

    Returns
    -------
    G_diag_blocks : np.ndarray
        Diagonal blocks of the inverse matrix.
    G_upper_blocks : np.ndarray
        Upper diagonal blocks of the inverse matrix.
    G_lower_blocks : np.ndarray
        Lower diagonal blocks of the inverse matrix.
    """
    
    nblocks = A_bloc_diag.shape[0]
    blockSize = A_bloc_diag.shape[1]

    # Storage for the incomplete forward substitution
    g_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)

    # Storage for the full backward substitution 
    G_diag_blocks  = np.zeros((nblocks, blockSize, blockSize), dtype=A_bloc_diag.dtype)
    G_upper_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_upper.dtype)
    G_lower_blocks = np.zeros((nblocks-1, blockSize, blockSize), dtype=A_bloc_lower.dtype)


    # 1. Initialisation of g
    g_diag_blocks[-1, ] = np.linalg.inv(A_bloc_diag[-1, ])

    # From right to left
    for i in range(nblocks-2, -1, -1):
        g_diag_blocks[i, ] = np.linalg.inv(A_bloc_diag[i, ] - A_bloc_upper[i, ] @ g_diag_blocks[i+1, ] @ A_bloc_lower[i, ])

    # 3. Initialisation of last element of G
    G_diag_blocks[0, ] = g_diag_blocks[0, ]

    # From left to right
    for i in range(1, nblocks):
        G_diag_blocks[i, ]    =  g_diag_blocks[i, ] @ (np.identity(blockSize) + A_bloc_lower[i-1, ] @ G_diag_blocks[i-1, ] @ A_bloc_upper[i-1, ] @ g_diag_blocks[i, ])
        G_lower_blocks[i-1, ] = -g_diag_blocks[i, ] @ A_bloc_lower[i-1, ] @ G_diag_blocks[i-1, ]
        G_upper_blocks[i-1, ] =  G_lower_blocks[i-1, ].T


    return G_diag_blocks, G_upper_blocks, G_lower_blocks


