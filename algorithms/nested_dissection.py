"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-07

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

def nd_fill_in(N:int, n: int) -> int:
    """
        Compute the number of non-zero elements of the matrix after an n-level nested dissection
        - n>=1
    """
    summ = 0

    for i in range(1, n+1, 1):
        summ += pow(2, i-1) * 5 * pow((N//pow(3, i)), 2)

    summ += pow(2, n) * pow((N//pow(3, n)), 2)

    return summ