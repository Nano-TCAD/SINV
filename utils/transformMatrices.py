"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-05

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

import numpy as np
import math



def transformToSymmetric(A):
    """
        Make a matrix symmetric by adding its transpose to itself.
    """
    return A + A.T



def distance_to_power_of_two(size):
    """
        Compute the distance between n and the closest power of two minus one
    """

    p = math.ceil(math.log2(size + 1))

    closest_power = 2**p - 1
    distance = closest_power - size

    return distance



def identity_padding(A, extension):
    """
        Add identity padding to a matrix A
    """

    size = A.shape[0]

    I = np.eye(extension)
    Ap = np.zeros((size+extension, size+extension), dtype=A.dtype)

    Ap[0:size, 0:size] = A
    Ap[size:size+extension, size:size+extension] = I

    return Ap