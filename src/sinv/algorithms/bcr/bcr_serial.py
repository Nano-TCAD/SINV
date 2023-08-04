"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2023-06

@reference: https://doi.org/10.1016/j.jcp.2009.03.035
@reference: https://doi.org/10.1017/CBO9780511812583

Copyright 2023 ETH Zurich and the QuaTrEx authors. All rights reserved.
"""

from sinv import utils
from sinv.algorithms.bcr import bcr_utils as bcr_u

import numpy as np
import math



def reduce(A: np.ndarray, 
           L: np.ndarray, 
           U: np.ndarray, 
           row: int, 
           level: int, 
           i_elim: np.ndarray, 
           blocksize: int) -> None:
    """ Operate the reduction towards the row-th row of the matrix A.
    
    Parameters
    ----------
    A : np.ndarray 
        the matrix to reduce
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    row : int
        the row to be reduce towards
    level : int
        the current level of the reduction in the reduction tree
    i_elim : np.ndarray
        target row indices to be eliminated
    blocksize : int
        the size of the blocks in A
        
    Returns
    -------
    None
    """

    nblocks = A.shape[0] // blocksize
    offset_blockindex = int(math.pow(2, level)) 


    # Reduction from i (above row) and k (below row) to j row
    i_blockindex = i_elim[row] - offset_blockindex
    k_blockindex = i_elim[row] + offset_blockindex


    # Computing of row-based indices
    i_rowindex   = i_blockindex * blocksize
    ip1_rowindex = (i_blockindex + 1) * blocksize

    j_rowindex   = i_elim[row] * blocksize
    jp1_rowindex = (i_elim[row] + 1) * blocksize
    
    k_rowindex   = k_blockindex * blocksize
    kp1_rowindex = (k_blockindex + 1) * blocksize


    # If there is a row above
    if i_blockindex >= 0: 
        A_ii_inv = np.linalg.inv(A[i_rowindex:ip1_rowindex, i_rowindex:ip1_rowindex])

        
        U[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex] =\
            A_ii_inv @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]
            
        L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] =\
            A[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A_ii_inv
        
        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] =\
            A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] - L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex] @ A[i_rowindex:ip1_rowindex, j_rowindex:jp1_rowindex]


        # If the row above is not the top row
        if i_blockindex != i_elim[0]:
            h_rowindex = (i_blockindex - offset_blockindex) * blocksize
            hp1_rowindex = (i_blockindex - offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, h_rowindex:hp1_rowindex] =\
                - L[j_rowindex:jp1_rowindex, i_rowindex:ip1_rowindex]\
                    @ A[i_rowindex:ip1_rowindex, h_rowindex:hp1_rowindex]
       

    # If there is a row below
    if k_blockindex <= nblocks-1:
        A_kk_inv = np.linalg.inv(A[k_rowindex:kp1_rowindex, k_rowindex:kp1_rowindex])
        
        U[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex] =\
            A_kk_inv @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]
            
        L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] =\
            A[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex] @ A_kk_inv

        A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex] =\
            A[j_rowindex:jp1_rowindex, j_rowindex:jp1_rowindex]\
                - L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]\
                    @ A[k_rowindex:kp1_rowindex, j_rowindex:jp1_rowindex]


        # If the row below is not the bottom row
        if k_blockindex != i_elim[-1]:
            l_rowindex   = (k_blockindex + offset_blockindex) * blocksize
            lp1_rowindex = (k_blockindex + offset_blockindex + 1) * blocksize

            A[j_rowindex:jp1_rowindex, l_rowindex:lp1_rowindex] =\
                - L[j_rowindex:jp1_rowindex, k_rowindex:kp1_rowindex]\
                    @ A[k_rowindex:kp1_rowindex, l_rowindex:lp1_rowindex]
                    


def reduce_bcr(A: np.ndarray, 
               L: np.ndarray, 
               U: np.ndarray, 
               i_bcr: np.ndarray, 
               blocksize: int) -> int:
    """ Performs block cyclic reduction on the matrix A. Computing during the 
    process the LU decomposition of the matrix A. The matrix A is overwritten.
    
    Parameters
    ----------
    A : np.ndarray 
        the matrix to reduce
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    i_bcr : np.ndarray
        blockrows to perform the reduction on        
    blocksize : int
        size of the blocks in the matrix A
    
    Returns
    -------
    last_reduction_row : int
        the index of the last row that was reduced
    """

    nblocks = len(i_bcr)
    height  = int(math.log2(nblocks))

    last_reduction_row = 0

    for level_blockindex in range(height):
        stride = int(math.pow(2, level_blockindex + 1))
        i_elim = [i for i in range(stride - 1, nblocks, stride)]

        for row in range(len(i_elim)):
            reduce(A, L, U, row, level_blockindex, i_elim, blocksize)

        last_reduction_row = i_elim[-1]

    return last_reduction_row



def invert_block(A: np.ndarray, 
                 G: np.ndarray, 
                 target_block: int, 
                 blocksize: int) -> None:
    """ Produce the first block of the inverse of A after having perfomed the 
    cyclic reduction.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    G : np.ndarray
        output inverse matrix
    target_block : int
        index of the block to invert
    blocksize : int
        size of the blocks
    
    Returns
    -------
    None
    """
    
    
    target_row    = target_block * blocksize
    target_row_p1 = (target_block + 1) * blocksize
    
    G[target_row: target_row_p1, target_row: target_row_p1] =\
        np.linalg.inv(A[target_row: target_row_p1, target_row: target_row_p1])
        


def corner_produce(A: np.ndarray, 
                   L: np.ndarray, 
                   U: np.ndarray, 
                   G: np.ndarray, 
                   k_from: int, 
                   k_to: int, 
                   blocksize: int) -> None:
    """ BCR production procedure associated with the corner production case.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition factors of A
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    G : np.ndarray
        output matrix to be produced
    k_from : int
        index of the block row to produce from
    k_to : int
        index of the block row to produce to
    blocksize : int
        size of the blocks in the matrix A
        
    Returns
    -------
    None
    """
    
    k_from_rowindex   = k_from * blocksize
    kp1_from_rowindex = (k_from + 1) * blocksize

    k_to_rowindex     = k_to * blocksize
    kp1_to_rowindex   = (k_to + 1) * blocksize


    G[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex] =\
        -1 * G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex]\
            @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]
            
    G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex] =\
        - U[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex]\
            @ G[k_from_rowindex:kp1_from_rowindex, k_from_rowindex:kp1_from_rowindex]
            
    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex] =\
        np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex])\
            - G[k_to_rowindex:kp1_to_rowindex, k_from_rowindex:kp1_from_rowindex]\
                @ L[k_from_rowindex:kp1_from_rowindex, k_to_rowindex:kp1_to_rowindex]
    


def center_produce(A: np.ndarray, 
                   L: np.ndarray, 
                   U: np.ndarray, 
                   G: np.ndarray,
                   k_above: int, 
                   k_to: int, 
                   k_below: int, 
                   blocksize: int) -> None:
    """ BCR production procedure associated with the center production case.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition factors of A
    L : np.ndarray
        lower decomposition factors of A
    U : np.ndarray
        upper decomposition factors of A
    G : np.ndarray
        output matrix to be produced
    k_above : int
        index of the block row above to produce from
    k_to : int
        index of the block row to produce
    k_below : int
        index of the block row below to produce from
    blocksize : int
        size of the blocks in the matrix A
        
    Returns
    -------
    None
    """
    
    k_above_rowindex   = k_above * blocksize
    kp1_above_rowindex = (k_above + 1) * blocksize

    k_to_rowindex      = k_to * blocksize
    kp1_to_rowindex    = (k_to + 1) * blocksize

    k_below_rowindex   = k_below * blocksize
    kp1_below_rowindex = (k_below + 1) * blocksize


    G[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex] =\
        - G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex]\
            @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                - G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex]\
                    @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
                    
    G[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex] =\
        - G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex]\
            @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                - G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex]\
                    @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]
                    
    G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex] =\
        - U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]\
            @ G[k_above_rowindex:kp1_above_rowindex, k_above_rowindex:kp1_above_rowindex]\
                - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]\
                    @ G[k_below_rowindex:kp1_below_rowindex, k_above_rowindex:kp1_above_rowindex]
                    
    G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex] =\
        - U[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]\
            @ G[k_above_rowindex:kp1_above_rowindex, k_below_rowindex:kp1_below_rowindex]\
                - U[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]\
                    @ G[k_below_rowindex:kp1_below_rowindex, k_below_rowindex:kp1_below_rowindex]
                    
    G[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex] =\
        np.linalg.inv(A[k_to_rowindex:kp1_to_rowindex, k_to_rowindex:kp1_to_rowindex])\
            - G[k_to_rowindex:kp1_to_rowindex, k_above_rowindex:kp1_above_rowindex]\
                @ L[k_above_rowindex:kp1_above_rowindex, k_to_rowindex:kp1_to_rowindex]\
                    - G[k_to_rowindex:kp1_to_rowindex, k_below_rowindex:kp1_below_rowindex]\
                        @ L[k_below_rowindex:kp1_below_rowindex, k_to_rowindex:kp1_to_rowindex]



def produce_bcr(A: np.ndarray, 
                L: np.ndarray, 
                U: np.ndarray, 
                G: np.ndarray, 
                i_bcr: np.ndarray, 
                blocksize: int) -> None:
    """ Performs the block cyclic production.
    
    Parameters
    ----------
    A : np.ndarray
        diagonal decomposition of A
    L : np.ndarray
        lower decomposition of A
    U : np.ndarray
        upper decomposition of A
    G : np.ndarray
        output inverse matrix
    i_bcr : np.ndarray
        blockrows to perform the production on
    blocksize : int
        size of the blocks
        
    Returns
    -------
    None
    """

    nblocks = len(i_bcr)
    height  = int(math.log2(nblocks))

    for level_blockindex in range(height-1, -1, -1):
        stride_blockindex = int(math.pow(2, level_blockindex))

        # Determine the blocks-row to be produced
        i_from = [i for i in range(int(math.pow(2, level_blockindex + 1)) - 1, nblocks, int(math.pow(2, level_blockindex + 1)))]

        i_prod = []
        for i in range(len(i_from)):
            if i == 0:
                i_prod.append(i_from[i] - stride_blockindex)
                i_prod.append(i_from[i] + stride_blockindex)
            else:
                if i_prod[i] != i_from[i] - stride_blockindex:
                    i_prod.append(i_from[i] - stride_blockindex)
                i_prod.append(i_from[i] + stride_blockindex)


        for i_prod_blockindex in range(len(i_prod)):
            k_to = i_bcr[i_prod[i_prod_blockindex]]

            if i_prod_blockindex == 0:
                # Production row is the first row within the stride_blockindex range
                # It only gets values from the below row 
                k_from = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                corner_produce(A, L, U, G, k_from, k_to, blocksize)

            if i_prod_blockindex != 0 and i_prod_blockindex == len(i_prod) - 1:
                if i_prod[-1] <= len(i_bcr) - stride_blockindex -1:
                    k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                    k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                    center_produce(A, L, U, G, k_above, k_to, k_below, blocksize)
                else:
                    # Production row is the last row within the stride_blockindex range
                    # It only gets values from the above row 
                    k_from = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                   
                    corner_produce(A, L, U, G, k_from, k_to, blocksize)
            
            if i_prod_blockindex != 0 and i_prod_blockindex != len(i_prod) - 1:
                # Production row is in the middle of the stride_blockindex range
                # It gets values from the above and below rows
                k_above = i_bcr[i_prod[i_prod_blockindex] - stride_blockindex]
                k_below = i_bcr[i_prod[i_prod_blockindex] + stride_blockindex]

                center_produce(A, L, U, G, k_above, k_to, k_below, blocksize)



def bcr_serial(A: np.ndarray, 
               blocksize: int) -> np.ndarray:
    """ Performe the tridiagonal selected inversion using a serial version of
    the block cyclic reduction algorithm.
    
    Parameters
    ----------
    A : np.ndarray
        matrix to invert
    blocksize : int
        size of a block of the matrix A
        
    Returns
    -------
    G : np.ndarray
        inverse of the matrix A
    """
    
    nblocks_initial = A.shape[0] // blocksize
    
    # First the input matrix may need to be 0-padded to a power of 2 number of blocks
    block_padding_distance = bcr_u.distance_to_power_of_two(nblocks_initial)
    A = bcr_u.identity_padding(A, block_padding_distance*blocksize)

    nblocks_padded = A.shape[0] // blocksize

    L = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)
    U = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)
    G = np.zeros((nblocks_padded*blocksize, nblocks_padded*blocksize), dtype=A.dtype)

    # 1. Block cyclic reduction
    i_bcr = [i for i in range(nblocks_padded)]
    final_reduction_block = reduce_bcr(A, L, U, i_bcr, blocksize)

    # 2. Block cyclic production
    invert_block(A, G, final_reduction_block, blocksize)
    produce_bcr(A, L, U, G, i_bcr, blocksize)

    # Cut the padding
    G = G[:nblocks_initial*blocksize, :nblocks_initial*blocksize]

    return G


