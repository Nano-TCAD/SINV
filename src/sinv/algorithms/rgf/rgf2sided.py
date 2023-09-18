"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Based on initial idea and work from: Anders Winka (awinka@iis.ee.ethz.ch)

@reference: https://doi.org/10.1063/1.1432117
@reference: https://doi.org/10.1007/s10825-013-0458-7

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv.algorithms.rgf import rgf_utils
from sinv.utils import mpu
from sinv.utils import gen_mat

from quasi.bsparse import bdia, bsr
from quasi.bsparse._base import bsparse

import numpy as np
from mpi4py import MPI



def rgf2sided(
    A: bsparse,
    sym_mat: bool = False,
    save_off_diag: bool = True,
) -> bsparse:
    """ Extension of the RGF algorithm that uses two processes that meet in the
    middle of the matrix. The array traversal is done from both sides to the
    middle.

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
    
    comm = MPI.COMM_WORLD
    comm_rank = comm.Get_rank()
    comm_size = comm.Get_size()
    
    rgf_utils.check_multiprocessing(comm_size)
    
    pass



if __name__ == "__main__":
    is_complex = False
    is_symmetric = False
    SEED = 63
    
    matrix_size = 14
    blocksize   = 2
    
    n_blocks = matrix_size // blocksize
    sub_mat_size = min(matrix_size, 2*blocksize)
    n_sub_mat = max(1, n_blocks - 1)        
    overlap = blocksize    
        
    A = bdia.diag(
        [gen_mat.generateRandomNumpyMat(sub_mat_size, is_complex, is_symmetric, SEED) 
        for _ in range(n_sub_mat)], 
        blocksize, overlap
    )
    
    l_A = mpu.create_row_partition_from_bsparse(A, 2)
    
    
    
    
    
    #X_refsol = np.linalg.inv(A.toarray())
    #X_rgf = rgf2sided(A, is_symmetric)
    
    
    