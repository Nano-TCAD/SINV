"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Based on initial idea and work from: Anders Winka (awinka@iis.ee.ethz.ch)

@reference: https://doi.org/10.1063/1.1432117
@reference: https://doi.org/10.1007/s10825-013-0458-7

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

import bsparse as bsp

import numpy as np
from mpi4py import MPI


def rgf2sided_upper_process(
    A_local: bsp,
    symmetry: str = None,
    save_off_diag: bool = True,
) -> bsp:
    """

    Parameters
    ----------


    Returns
    -------

    """

    # Storage for the full backward substitution
    G_local = A_local.copy() * np.nan

    # 0. Inverse of the first block
    G_local[0, 0] = np.linalg.inv(A_local[0, 0])

    # 1. Forward substitution (performed left to right)
    for i in range(1, A_local.bshape[0] - 1, 1):
        G_local[i, i] = np.linalg.inv(
            A_local[i, i]
            - A_local[i, i - 1] @ G_local[i - 1, i - 1] @ A_local[i - 1, i]
        )

    return A_local


def rgf2sided_lower_process(
    A_local: bsp,
    symmetry: str = None,
    save_off_diag: bool = True,
) -> bsp:
    """

    Parameters
    ----------


    Returns
    -------

    """

    # Storage for the full backward substitution
    G_local = A_local.copy() * np.nan

    # 0. Inverse of the first block
    G_local[-1, -1] = np.linalg.inv(A_local[-1, -1])

    # 1. Forward substitution (performed right to left)
    for i in range(A_local.bshape[0] - 1, 1, -1):
        G_local[i, i] = np.linalg.inv(
            A_local[i - 1, i - 1]
            - A_local[i - 1, i] @ G_local[i + 1, i + 1] @ A_local[i, i - 1]
        )

    return A_local
