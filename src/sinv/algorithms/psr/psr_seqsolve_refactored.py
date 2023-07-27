"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import utils
from sinv import algorithms as alg

import numpy as np
import time

from mpi4py import MPI





def psr_seqsolve(A: np.ndarray, 
                 blocksize: int):
    """ Selected inversion algorithm using the parallel Schur reduction 
    algorithm. The algorithm work in place and will overwrite the input matrix A.

    Parameters
    ----------
    A : np.ndarray
        Block tridiagonal matrix
    blocksize : int
        Block matrice_size
        
    Returns 
    -------
    G : np.ndarray
        Block tridiagonal selected inverse of A.
    """
    
    
    
        
    return G

