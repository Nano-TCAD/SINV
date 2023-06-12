"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np

from scipy.sparse import csc_matrix
from mpi4py       import MPI



def transformToSymmetric(A):
    """
        Make a matrix symmetric by adding its transpose to itself.
    """
    return A + A.T
