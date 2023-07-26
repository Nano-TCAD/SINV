"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



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

