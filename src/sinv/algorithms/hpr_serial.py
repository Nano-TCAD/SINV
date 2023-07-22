"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import time

from typing import Tuple



def schurInvert(A: np.ndarray) -> np.ndarray:
    # Compute the inverse of A using an explicite Schur decomposition
    # - Only interesting for teaching purposes

    size = A.shape[0]
    size2 = size//2

    # Handmade Schur decomposition
    Ls = np.zeros((size2, size2), dtype=A.dtype)
    Us = np.zeros((size2, size2), dtype=A.dtype)
    Sc = np.zeros((size2, size2), dtype=A.dtype)

    inv_A11 = np.linalg.inv(A[:size2, :size2])

    Ls = A[size2:, :size2] @ inv_A11
    Us = inv_A11 @ A[:size2, size2:]
    Sc = A[size2:, size2:] - A[size2:, :size2] @ inv_A11 @ A[:size2, size2:]

    G = np.zeros((size, size), dtype=A.dtype)

    G11 = np.zeros((size2, size2), dtype=A.dtype)
    G12 = np.zeros((size2, size2), dtype=A.dtype)
    G21 = np.zeros((size2, size2), dtype=A.dtype)
    G22 = np.zeros((size2, size2), dtype=A.dtype)

    inv_Sc = np.linalg.inv(Sc)

    G11 = inv_A11 + Us @ inv_Sc @ Ls
    G12 = -Us @ inv_Sc
    G21 = -inv_Sc @ Ls
    G22 = inv_Sc

    G[:size2, :size2] = G11
    G[:size2, size2:] = G12
    G[size2:, :size2] = G21
    G[size2:, size2:] = G22

    return G



def hpr_serial(A: np.ndarray, blockSize: int) -> Tuple[np.ndarray, float]:
    # Implementation of the serial algorithm presented in section 3. of the paper
    # - The algorithm is equivalent to an RGF but with explicit LU decomposition
    # - TODO: Convert to block storage version (dense for now)
    # - TODO: Move around the additional inversion of D in the BWD recurence

    size    = A.shape[0]
    nBlocks = size//blockSize

    G = np.zeros((size, size), dtype=A.dtype)
    L = np.zeros((size, size), dtype=A.dtype)
    D = np.zeros((size, size), dtype=A.dtype)
    U = np.zeros((size, size), dtype=A.dtype)


    tic = time.time() # -----------------------------
    # Initialisation of forward recurence
    D[0:blockSize, 0:blockSize] = A[0:blockSize, 0:blockSize]
    L[blockSize:2*blockSize, 0:blockSize]  = A[blockSize:2*blockSize, 0:blockSize] @ np.linalg.inv(D[0:blockSize, 0:blockSize])
    U[0:blockSize, blockSize:2*blockSize]  = np.linalg.inv(D[0:blockSize, 0:blockSize]) @ A[0:blockSize, blockSize:2*blockSize]

    # Forward recurence
    for i in range(0, nBlocks-1):
        b_i   = i*blockSize
        b_ip1 = (i+1)*blockSize
        b_ip2 = (i+2)*blockSize

        D_inv_i = np.linalg.inv(D[b_i:b_ip1, b_i:b_ip1])

        D[b_ip1:b_ip2, b_ip1:b_ip2] = A[b_ip1:b_ip2, b_ip1:b_ip2] - A[b_ip1:b_ip2, b_i:b_ip1] @ D_inv_i @ A[b_i:b_ip1, b_ip1:b_ip2]
        L[b_ip1:b_ip2, b_i:b_ip1]   = A[b_ip1:b_ip2, b_i:b_ip1] @ D_inv_i
        U[b_i:b_ip1, b_ip1:b_ip2]   = D_inv_i @ A[b_i:b_ip1, b_ip1:b_ip2]

    # Initialisation of backward recurence
    b_nm1 = (nBlocks-2)*blockSize
    b_n   = (nBlocks-1)*blockSize
    b_np1 = nBlocks*blockSize

    G[b_n:b_np1, b_n:b_np1] = np.linalg.inv(D[b_n:b_np1, b_n:b_np1])
    G[b_nm1:b_n, b_n:b_np1] = -G[b_n:b_np1, b_n:b_np1] @ L[b_n:b_np1, b_nm1:b_n]
    G[b_n:b_np1, b_nm1:b_n] = -U[b_nm1:b_n, b_n:b_np1] @ G[b_n:b_np1, b_n:b_np1]

    # Backward recurence
    for i in range(nBlocks-1, 0, -1):
        b_im1 = (i-1)*blockSize
        b_i   = i*blockSize
        b_ip1 = (i+1)*blockSize

        G[b_im1:b_i, b_im1:b_i] = np.linalg.inv(D[b_im1:b_i, b_im1:b_i]) + U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_i:b_ip1, b_im1:b_i] = -G[b_i:b_ip1, b_i:b_ip1] @ L[b_i:b_ip1, b_im1:b_i]
        G[b_im1:b_i, b_i:b_ip1] = -U[b_im1:b_i, b_i:b_ip1] @ G[b_i:b_ip1, b_i:b_ip1]
    toc = time.time() # -----------------------------

    
    timing = toc-tic

    return G, timing


