"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-08

Contains the utility functions for the PSR algorithm.

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np



def check_input(A: np.ndarray, 
                blocksize: int, 
                comm_size: int):
    """ Check the validity of the inputs parameters.

    Parameters
    ----------
    A : numpy matrix
        matrix to invert
    blocksize : int
        size of a block
    comm_size : int
        number of processes

    Returns
    -------
    None

    Raises
    ------
    ValueError
        The matrix size must be a multiple of the blocksize.
    ValueError
        The blocksize must be smaller than the matrix size.
    ValueError
        The blocksize must be greater than 0.
    ValueError
        The number of blocks must be greater than the number of processes.
    """
    
    if A.shape[0] % blocksize != 0:
        raise ValueError("The matrix size must be a multiple of the blocksize.")
    
    if blocksize > A.shape[0]:
        raise ValueError("The blocksize must be smaller than the matrix size.")
    
    if blocksize < 1:
        raise ValueError("The blocksize must be greater than 0.")
    
    nblocks = A.shape[0] // blocksize
    if nblocks < 3*comm_size:
        raise ValueError("The number of blocks is to low. There should be at least 3 blockrows per process")
        # Central processes need at least 3 (block) rows to work.
    
    
    
    
def divide_matrix(A: np.ndarray, 
                  n_partitions: int, 
                  blocksize: int) -> [list, 
                                      list]:
    """ Compute the n_partitions segments that divide the matrix A.

    Parameters
    ----------
    A : numpy matrix            
        matrix to divide
    n_partitions : int
        number of partitions
    blocksize : int
        size of a block

    Returns
    -------
    l_start_blockrow : list
        list of processes starting block index
    l_partitions_blocksizes : list
        list of processes partition size
    """

    nblocks = A.shape[0] // blocksize
    partition_blocksize = nblocks // n_partitions
    blocksize_of_first_partition = nblocks - partition_blocksize * (n_partitions-1)

    # Compute the starting block row and the partition size for each process
    l_start_blockrow        = []
    l_partitions_blocksizes = []
    

    for i in range(n_partitions):
        if i == 0:
            l_start_blockrow        = [0]
            l_partitions_blocksizes = [blocksize_of_first_partition]
        else:
            l_start_blockrow.append(l_start_blockrow[i-1] + l_partitions_blocksizes[i-1])
            l_partitions_blocksizes.append(partition_blocksize)


    return l_start_blockrow, l_partitions_blocksizes

