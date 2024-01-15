"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np


def get_local_partition_indices(
    partition_number: int,
    n_partitions: int,
    partitions_blocksizes: list,
) -> [int, int, int]:
    """ Get the indices of the local partition in the global matrix.
    
    Parameters
    ----------
    partition_number : int
        Number of the local partition.
    n_partitions : int
        Total number of partitions.
    partitions_blocksizes : list
        List containing the size of each partition.
        
    Returns
    -------
    start_blockrow : int
        Index of the first blockrow of the local partition in the global matrix.
    end_blockrow : int
        Index of the last blockrow of the local partition in the global matrix.
    partition_size : int
        Size of the local partition.
    
    """
    
    start_blockrow = 0
    end_blockrow = 0
    partition_size = partitions_blocksizes[partition_number]
    
    for i in range(partition_number):
        start_blockrow += partitions_blocksizes[i]
        
    end_blockrow = start_blockrow + partition_size
    
    return start_blockrow, end_blockrow, partition_size
    
    