"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@author: Nicolas Vetsch (nvetsch@iis.ee.ethz.ch)
@date: 2023-05

@reference: https://doi.org/10.1063/1.1432117

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from quasi.bsparse._base import bsparse



def rgf(
    A: bsparse,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> bsparse:
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
    G.blocks[0, 0] = np.linalg.inv(A.blocks[0, 0])

    # 1. Forward substitution (performed left to right)
    for i in range(1, A.blockorder, 1):
        G.blocks[i, i] = np.linalg.inv(A.blocks[i, i] - A.blocks[i, i-1] @ G.blocks[i-1, i-1] @ A.blocks[i-1, i])

    # 2. Backward substitution (performed right to left)
    for i in range(A.blockorder-2, -1, -1): 
        G_ii = G.blocks[i, i]
        G_lowerfactor = G.blocks[i+1, i+1] @ A.blocks[i+1, i] @ G_ii   
        
        if save_off_diag:
            G.blocks[i+1, i] = -G_lowerfactor
            if sym_mat:
                G.blocks[i, i+1] = G.blocks[i+1, i].T
            else:
                G.blocks[i, i+1] = -G_ii @ A.blocks[i, i+1] @ G.blocks[i+1, i+1]
            
        G.blocks[i, i]   =  G_ii + G_ii @ A.blocks[i, i+1] @ G_lowerfactor

    return G



from quasi.bsparse import bdia, bsr, vbdia, vbsr
import matplotlib.pyplot as plt

if __name__ == '__main__':
    """ MAT_SIZE  = 6
    BLOCKSIZE = 2
    N_BLOCKS  = MAT_SIZE // BLOCKSIZE

    SUB_MAT_SIZE = min(MAT_SIZE, 2*BLOCKSIZE)
    N_SUB_MAT    = max(1, N_BLOCKS-1)
    OVERLAP      = BLOCKSIZE
    

    a = bsr.diag(
        [np.random.rand(SUB_MAT_SIZE, SUB_MAT_SIZE) for _ in range(N_SUB_MAT)], BLOCKSIZE, OVERLAP
    ) """
    
    SIZE = 4
    BLOCKSIZE = 2
    
    BLOCKSIZES = [3, 2, 1, 2, 3]

    a = vbdia.diag([np.random.rand(BLOCKSIZES[i], BLOCKSIZES[i]) for i in range(0, len(BLOCKSIZES))], 0)
    
    a.show()
    plt.show()
    
    
    """ #x_rgf = rgf(a)
    x_inv = np.linalg.inv(a.toarray())

    # Main Diagonal.
    for i in range(x_rgf.blockorder):
        ii = slice(i * BLOCKSIZE, (i + 1) * BLOCKSIZE)
        assert np.allclose(x_rgf.blocks[i, i], x_inv[ii, ii])
        
    
    # Create a subplot that show the two plots side by side
    #ax = plt.subplots(1, 2)
    #x_rgf.show()
    #plt.matshow(np.abs(x_inv))
    #plt.show()
    
    
    # Off-Diagonals.
    for i in range(x_rgf.blockorder - 1):
        ii = slice(i * BLOCKSIZE, (i + 1) * BLOCKSIZE)
        jj = slice((i + 1) * BLOCKSIZE, (i + 2) * BLOCKSIZE)
        assert np.allclose(x_rgf.blocks[i + 1, i], x_inv[jj, ii])
        assert np.allclose(x_rgf.blocks[i, i + 1], x_inv[ii, jj]) """