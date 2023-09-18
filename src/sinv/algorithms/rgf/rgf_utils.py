"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-09

Contains the utility functions for the rgf2sided algorithm.

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""



def check_multiprocessing(comm_size: int):
    """ Check that the number of processes is exactly 2.

    Parameters
    ----------
    comm_size : int
        number of processes
    
    Returns
    -------
    None
    
    Raises
    ------
    ValueError
        The number of processes must 2.
    """
    
    if comm_size != 2:
        raise ValueError("The number of processes must be 2.")
    
    
    