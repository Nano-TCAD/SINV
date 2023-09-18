"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Contains utlitity functions for creating, distributing and aggregating partitions 
in multiprocessing algorithms.

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from quasi.bsparse import bdia, bsr
from quasi.bsparse._base import bsparse

import numpy as np
import matplotlib.pyplot as plt



def create_row_partition_from_bsparse(
    A: bsparse,
    number_of_partitions: int
) -> list[bsparse]:
    """ Partition a bsparse matrix into a list of bsparse matrices.
    
    Parameters
    ----------
    A : bsparse
        Input matrix.
    number_of_partitions : int
        Number of partitions.
        
    Returns
    -------
    l_A : list[bsparse]
        List of partitions.
    """
    
    
    number_of_blocks = A.blockorder
    slicing_blockindices = np.linspace(0, number_of_blocks, number_of_partitions+1, dtype=int)
    slicing_rowindices = [slicing_blockindices[i]*A.blocksize for i in range(number_of_partitions+1)]
    
    
    # Create the list of b_sparse matrices by slicing the lil representation of A
    lil_A = A.tolil()
    
    #print("lil_A:\n", lil_A)
    
    l_A : list[bsparse] = []
    
    for i in range(number_of_partitions):
        lil_partition = lil_A[slicing_rowindices[i]:slicing_rowindices[i+1], :]
        
        print("lil_partition:\n", lil_partition)
        
        bsparse_A = bdia.from_spmatrix(lil_partition, A.blockorder)
        
        print("bsparse_A:\n", bsparse_A)
        
        """ bsparse_A = bsparse.from_spmatrix(lil_partition)
        bsparse_A.show()
        plt.show()
        l_A.append(bsparse_A)  """
    
    #print("slicing_blockindices: ", slicing_blockindices)
    #print("slicing_rowindices: ", slicing_rowindices)
    
    
    """ for i in range(number_of_partitions):
        l_A[i].show()
        plt.show() """
    
    """ import matplotlib.pyplot as plt
    A.show()
    plt.show() """
    
    #A_rray = A.toarray()
    """ A_rray = A.tolil()
    print(A_rray) """
    
    
    
    
    pass



def distribute_row_partition():
    """
    """
    pass



def aggregate_row_partition():
    """
    """
    pass