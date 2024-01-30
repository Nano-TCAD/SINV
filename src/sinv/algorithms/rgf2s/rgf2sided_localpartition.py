"""
@author: Vincent Maillou (vmaillou@iis.ee.ethz.ch)
@date: 2024-01

Copyright 2023-2024 ETH Zurich. All rights reserved.
"""

from sinv.algorithms.rgf2s import rgf2sided_base as rgf2s_base

import bsparse as bsp

import numpy as np
from mpi4py import MPI


def rgf2sided_localpartition(
    A_local: bsp,
    symmetry: str = None,
    save_off_diag: bool = True,
) -> bsp:
    """Perform selected inversion on a block-tridiagonal matrix using a 2-sided
    RGF algorithms. The local partition of the selected inverse is computed and
    returned by each participating processes.

    Parameters:
    A_local : bsp
        The local RGF matrix.
    symmetry : str, optional
        The symmetry type of the RGF matrix. Defaults to None.

    Returns:
    G_local : bsp
        The 2-sided local partition of the RGF matrix.
    """
    pass

    comm = MPI.COMM_WORLD
    comm_size = comm.Get_size()

    if comm_size == 1:
        raise ValueError("Number of processes must be greater than 1.")

    comm_rank = comm.Get_rank()

    if comm_rank == 0:
        return rgf2s_base.rgf2sided_upper_process(A_local, symmetry, save_off_diag)
    elif comm_rank == 1:
        return rgf2s_base.rgf2sided_lower_process(A_local, symmetry, save_off_diag)
    else:
        return None


# TODO: MOVE TO TESTING SECTION
from sinv.utils import gmu
from sinv.utils import rlcu
from sinv.utils import comm_utils
from sinv.utils import partition_utils as partu

import matplotlib.pyplot as plt

if __name__ == "__main__":
    n_blocks = 10
    blocksize = 2
    is_complex = False

    (
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
    ) = gmu.create_block_tridiagonal_matrix(
        n_blocks, blocksize, is_complex, is_symmetric=False
    )

    bsparse_matrix = rlcu.block_tridiagonal_to_BDIA(
        diagonal_blocks,
        upper_diagonal_blocks,
        lower_diagonal_blocks,
        symmetry=None,
    )

    start_blockrows, partition_sizes, end_blockrows = partu.get_partitions_indices(
        n_partitions=2,
        total_size=n_blocks,
    )

    bsparse_matrix_localpartition = (
        comm_utils.distribute_partitions_tridiagonals_matrix(
            bsparse_matrix,
            start_blockrows,
            partition_sizes,
        )
    )

    G_local = rgf2sided_localpartition(bsparse_matrix_localpartition)

    comm = MPI.COMM_WORLD

    if comm.Get_rank() == 0:
        plt.matshow(bsparse_matrix.toarray())
        plt.title("Initial matrix")

        ref_inv = np.linalg.inv(bsparse_matrix.toarray())
        ref_inv = gmu.zero_out_dense_to_block_tridiagonal(ref_inv, blocksize)

        plt.matshow(ref_inv)
        plt.title("reference inverse")

    plt.matshow(G_local.toarray())
    plt.title("Local partition n°" + str(comm.Get_rank()))

    G_global = comm_utils.aggregate_partitions(
        G_local,
        start_blockrows,
        partition_sizes,
    )

    if comm.Get_rank() == 0:
        plt.matshow(G_global.toarray())
        plt.title("RGF2S inverted matrix")
    plt.show()
