"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from quasi.bsparse._base import bsparse



def rgf_leftToRight(
    A: bsparse,
    save_off_diag: bool = True,
) -> bsparse:
    """ RGF algorithm performing block-tridiagonal selected inversion on the 
    input matrix. Act from upper left to lower right.

    Parameters
    ----------
    A : bsparse
        Input matrix.
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
    G.blocks[0, 0] = np.linalg.inv(A.blocks[0, 0])

    # 1. Forward substitution (performed left to right)
    for i in range(1, A.blockorder, 1):
        G.blocks[i, i] = np.linalg.inv(A.blocks[i, i] - A.blocks[i, i-1] @ G.blocks[i-1, i-1] @ A.blocks[i-1, i])

    # 2. Backward substitution (performed right to left)
    for i in range(A.blockorder-2, -1, -1): 
        G_ii = G.blocks[i, i]
        G_upperfactor = G.blocks[i+1, i+1] @ A.blocks[i+1, i] @ G_ii   
        
        if save_off_diag:
            G.blocks[i, i+1] = -G_ii @ A.blocks[i, i+1] @ G.blocks[i+1, i+1]
            G.blocks[i+1, i] = -G_upperfactor
            
        G.blocks[i, i]   =  G_ii + G_ii @ A.blocks[i, i+1] @ G_upperfactor
            

    return G



from quasi.bsparse import bdia, bsr, vbdia, vbsr

if __name__ == '__main__':
    SIZE = 4
    BLOCKSIZE = 2

    a = bsr.diag(
        [np.random.rand(SIZE, SIZE) for _ in range(10)], BLOCKSIZE, BLOCKSIZE
    )
    x_rgf = rgf_leftToRight(a)
    x_inv = np.linalg.inv(a.toarray())

    # Main Diagonal.
    for i in range(x_rgf.blockorder):
        ii = slice(i * BLOCKSIZE, (i + 1) * BLOCKSIZE)
        assert np.allclose(x_rgf.blocks[i, i], x_inv[ii, ii])
        
    import matplotlib.pyplot as plt
    
    # Create a subplot that show the two plots side by side
    #ax = plt.subplots(1, 2)
    """ x_rgf.show()
    plt.matshow(np.abs(x_inv))
    plt.show() """
    
    
    """ plt.subplot(1, 2, 1)
    plt_axe_0 = x_rgf.show()
    plt.subplot(1, 2, 2)
    plt_axe_1 = plt.matshow(np.abs(x_inv))
    plt.show() """

    # Off-Diagonals.
    for i in range(x_rgf.blockorder - 1):
        ii = slice(i * BLOCKSIZE, (i + 1) * BLOCKSIZE)
        jj = slice((i + 1) * BLOCKSIZE, (i + 2) * BLOCKSIZE)
        assert np.allclose(x_rgf.blocks[i + 1, i], x_inv[jj, ii])
        assert np.allclose(x_rgf.blocks[i, i + 1], x_inv[ii, jj])