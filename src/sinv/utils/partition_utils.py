"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2024 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np


def get_partitions_indices(
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> [[], [], []]:
    """ Create the partitions start/end indices and sizes for the entire problem.
    If the problem size doesn't match a perfect partitioning w.r.t the distribution,
    partitions will be resized starting from the first one.

    Parameters
    ----------    
    n_partitions : int
        Total number of partitions.
    total_size : int
        Total size of the global matrix. Equal to the sum of the sizes of all 
        partitions.
    partitions_distribution : list, optional
        Distribution of the partitions sizes, in percentage. The default is None
        and a uniform distribution is assumed.
        
    Returns
    -------
    start_blockrows : []
        List of the indices of the first blockrow of each partition in the 
        global matrix.
    partition_sizes : []
        List of the sizes of each partition.
    end_blockrows : []
        List of the indices of the last blockrow of each partition in the 
        global matrix.
    
    """
    
    if n_partitions > total_size:
        raise ValueError("Number of partitions cannot be greater than the total size of the matrix.")

    if partitions_distribution is not None:
        if n_partitions != len(partitions_distribution):
            raise ValueError("Number of partitions and number of entries in the distribution list do not match.")  
        if sum(partitions_distribution) != 100:
            raise ValueError("Sum of the entries in the distribution list is not equal to 100.")
    else:
        partitions_distribution = [100 / n_partitions] * n_partitions
    
    partitions_distribution = np.array(partitions_distribution)/100
    
    start_blockrows = []
    partition_sizes = []
    end_blockrows   = []
    
    for i in range(n_partitions):
        partition_sizes.append(round(partitions_distribution[i] * total_size))
    
    if sum(partition_sizes) != total_size:
        diff = total_size - sum(partition_sizes)
        for i in range(diff):
            partition_sizes[i] += 1
    
    for i in range(n_partitions):
        start_blockrows.append(sum(partition_sizes[:i]))
        end_blockrows.append(start_blockrows[i] + partition_sizes[i])
    
    return start_blockrows, partition_sizes, end_blockrows


def get_local_partition_indices(
    partition_number: int,
    n_partitions: int,
    total_size: int,
    partitions_distribution: list = None,
) -> [int, int, int]:
    """ Get the indices of the local partition in the global matrix. 
    
    Parameters
    ----------
    partition_number : int
        Number of the local partition.
    n_partitions : int
        Total number of partitions.
    total_size : int
        Total size of the global matrix. Equal to the sum of the sizes of all 
        partitions.
    partitions_distribution : list, optional
        Distribution of the partitions sizes, in percentage. The default is None
        and a uniform distribution is assumed. 
        
    Returns
    -------
    start_blockrow : int
        Index of the first blockrow of the local partition in the global matrix.
    end_blockrow : int
        Index of the last blockrow of the local partition in the global matrix.
    partition_size : int
        Size of the local partition.
    
    """
    
    start_blockrows, partition_sizes, end_blockrows = get_partitions_indices(n_partitions, total_size, partitions_distribution)
    
    return start_blockrows[partition_number], partition_sizes[partition_number], end_blockrows[partition_number]
    
    