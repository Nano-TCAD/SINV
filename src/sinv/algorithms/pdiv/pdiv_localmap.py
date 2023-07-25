"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

PDIV (P-Division) algorithm:
@reference: https://doi.org/10.1063/1.2748621
@reference: https://doi.org/10.1063/1.3624612

Pairwise algorithm:
@reference: https://doi.org/10.1007/978-3-319-78024-5_55

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import math

from mpi4py import MPI


def pdiv_localmap():
    """ Parallel Divide & Conquer implementation of the PDIV/Pairwise algorithm.
        
    Parameters
    ----------
    None

    Returns
    -------
    None

    Notes
    -----
    The PDIV (Pairwise) algorithm is a divide and conquer approch to compute
    the inverse of a matrix. The matrix is divided into submatrices, distributed
    among the processes, inverted locally and updated thourgh a series of reduction.

    This implementation perform local update of the distributed partition. Hence
    the inverted system is scattered across the processes.
    """